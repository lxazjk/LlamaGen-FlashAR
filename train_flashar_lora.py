import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import json
import argparse
import time
from tqdm import tqdm
import sys
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.ar_gpt import GPT_models as AR_GPT_models
from autoregressive.models.gpt import GPT_models as FLASHAR_GPT_models
from autoregressive.models.generate import generate as flashar_generate


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        for param in self.original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        result = self.original_linear(x)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_output


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.05, target_modules=['qkv', 'proj']):
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name)
                    
                    lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    setattr(parent, child_name, lora_layer)
                    lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                    break
    return lora_params


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


class ImageNetDataset(Dataset):
    def __init__(self, data_path, image_size=256, label_map=None):
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        self.dataset = ImageFolder(data_path, transform=self.transform)
        self.label_map = label_map
        self.synset_to_idx = {}
        if label_map:
            for i, synset in enumerate(label_map.keys()):
                self.synset_to_idx[synset] = i
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, folder_label = self.dataset[idx]
        
        if self.label_map is not None:
            class_name = self.dataset.classes[folder_label]
            if class_name in self.synset_to_idx:
                label = self.synset_to_idx[class_name]
            else:
                label = folder_label % len(self.label_map)
        else:
            label = folder_label
        
        return img, torch.tensor(label, dtype=torch.long)


def load_ar_weights_to_flashar(flashar_model, ar_ckpt_path, device='cpu'):
    ar_ckpt = torch.load(ar_ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(ar_ckpt, dict):
        if 'model' in ar_ckpt:
            ar_state_dict = ar_ckpt['model']
        elif 'state_dict' in ar_ckpt:
            ar_state_dict = ar_ckpt['state_dict']
        else:
            ar_state_dict = ar_ckpt
    else:
        ar_state_dict = ar_ckpt
    
    flashar_state_dict = flashar_model.state_dict()
    
    loaded_keys = []
    for key, value in ar_state_dict.items():
        if key in flashar_state_dict:
            if flashar_state_dict[key].shape == value.shape:
                flashar_state_dict[key] = value
                loaded_keys.append(key)
    
    flashar_model.load_state_dict(flashar_state_dict)
    return flashar_model, len(loaded_keys)


def setup_proximity_mask(block_size, cls_token_num=1):
    mask = torch.zeros(block_size, block_size, dtype=torch.bool)
    H = W = int(block_size ** 0.5)
    for c in range(H + W - 1):
        cur_token = []
        previous_token = []
        for h in range(H):
            w = c - h
            if 0 <= w < W:
                token_id = h * W + w
                cur_token.append(token_id)
                previous_token.append(token_id)
        for tid in cur_token:
            mask[tid, previous_token] = 1
    
    if cls_token_num > 0:
        full_mask = torch.ones(block_size + cls_token_num, block_size + cls_token_num, dtype=torch.bool)
        full_mask[cls_token_num:, cls_token_num:] = mask
        return full_mask
    return mask


def setup_curriculum_mask(block_size, progress, cls_token_num=1):
    causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).float()
    proximity_mask = setup_proximity_mask(block_size, cls_token_num=0).float()
    curriculum_mask = progress * proximity_mask + (1 - progress) * causal_mask
    
    if cls_token_num > 0:
        full_mask = torch.ones(block_size + cls_token_num, block_size + cls_token_num, dtype=torch.bool)
        full_mask[cls_token_num:, cls_token_num:] = curriculum_mask.bool()
        return full_mask
    return curriculum_mask.bool()


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss


@torch.no_grad()
def generate_samples_for_eval(model, vq_model, device, num_samples=5000, batch_size=100, 
                               latent_size=16, codebook_embed_dim=8, num_classes=1000,
                               cfg_scale=4.0, temperature=1.0, top_k=2000):
    model.eval()
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches), desc="Generating samples", leave=False):
        cur_batch = min(batch_size, num_samples - len(all_samples) * batch_size)
        class_labels = torch.randint(0, num_classes, (cur_batch,), device=device)
        
        index_sample = flashar_generate(
            model, class_labels, latent_size ** 2,
            cfg_scale=cfg_scale,
            temperature=temperature, top_k=top_k,
            top_p=1.0, sample_logits=True,
        )
        
        qzshape = [cur_batch, codebook_embed_dim, latent_size, latent_size]
        samples = vq_model.decode_code(index_sample, qzshape)
        samples = (samples + 1) / 2 * 255
        samples = samples.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        all_samples.append(samples)
    
    model.train()
    return np.concatenate(all_samples, axis=0)


def run_fid_evaluation(samples_npz_path, ref_path, output_file=None):
    cmd = f"python evaluations/c2i/evaluator.py {ref_path} {samples_npz_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    
    if output_file and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'FID' in line and 'sFID' not in line:
                    try:
                        fid = float(line.split(':')[1].strip())
                        return fid
                    except:
                        pass
    return None


def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'
    
    torch.manual_seed(args.seed + rank)
    
    if rank == 0:
        print("=" * 60)
        print(f"Training Mode: {args.train_mode}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"GPUs: {world_size}")
        print(f"Eval every {args.eval_every} epochs with {args.eval_samples} samples")
        print("=" * 60)
    
    if rank == 0:
        print("Loading VQ model...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(vq_ckpt["model"])
    vq_model.eval()
    for param in vq_model.parameters():
        param.requires_grad = False
    
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2
    
    teacher_model = None
    if args.train_mode in ['distill', 'curriculum']:
        if rank == 0:
            print("Loading Teacher (AR) model...")
        teacher_model = AR_GPT_models[args.gpt_model](
            vocab_size=args.codebook_size,
            block_size=block_size,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type='c2i',
        ).to(device)
        
        teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        if "model" in teacher_ckpt:
            teacher_model.load_state_dict(teacher_ckpt["model"])
        else:
            teacher_model.load_state_dict(teacher_ckpt)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        if rank == 0:
            print(f"Teacher model loaded: {args.teacher_ckpt}")
    
    if rank == 0:
        print("Loading Student (FlashAR) model...")
    student_model = FLASHAR_GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type='c2i',
    )
    
    if args.init_from_ar:
        student_model, loaded_keys = load_ar_weights_to_flashar(student_model, args.teacher_ckpt, device='cpu')
        if rank == 0:
            print(f"Student initialized from AR weights ({loaded_keys} keys loaded)")
    else:
        if rank == 0:
            print("Student initialized randomly")
    
    student_model = student_model.to(device)
    
    lora_params = apply_lora_to_model(
        student_model, 
        r=args.lora_r, 
        alpha=args.lora_alpha, 
        dropout=args.lora_dropout,
        target_modules=['qkv', 'proj']
    )
    
    for param in lora_params:
        param.data = param.data.to(device)
    
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    if rank == 0:
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
    
    student_model = DDP(student_model, device_ids=[local_rank], find_unused_parameters=True)
    student_model.train()
    
    label_map = None
    labels_file = os.path.join(args.data_path, "Labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            label_map = json.load(f)
        if rank == 0:
            print(f"Loaded label map with {len(label_map)} classes")
    
    if rank == 0:
        print("Creating dataset...")
    dataset = ImageNetDataset(
        args.data_path, 
        image_size=args.image_size, 
        label_map=label_map
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if rank == 0:
        print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches per GPU")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    proximity_mask = setup_proximity_mask(block_size, cls_token_num=args.cls_token_num).to(device)
    
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    scaler = torch.amp.GradScaler('cuda', enabled=(args.precision == 'fp16'))
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = open(os.path.join(args.output_dir, "train_log.txt"), 'w')
        results_file = open(os.path.join(args.output_dir, "results.csv"), 'w')
        results_file.write("epoch,train_loss,fid,is_score,time_minutes\n")
        results_file.flush()
    
    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs...")
    global_step = 0
    best_fid = float('inf')
    best_fid_epoch = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student_model.train()
        epoch_loss = 0
        num_batches = 0
        
        if args.train_mode == 'curriculum':
            progress = min(1.0, (epoch + 1) / args.curriculum_warmup)
            current_mask = setup_curriculum_mask(block_size, progress, cls_token_num=args.cls_token_num).to(device)
            if rank == 0:
                print(f"Epoch {epoch+1}: curriculum progress = {progress:.2f}")
        else:
            current_mask = proximity_mask
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=rank != 0)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                quant, emb_loss, info = vq_model.encode(imgs)
                tokens = info[2].reshape(imgs.shape[0], -1)
            
            with torch.amp.autocast('cuda', dtype=precision):
                if args.train_mode == 'direct':
                    student_logits, loss = student_model.module(
                        idx=tokens,
                        cond_idx=labels,
                        targets=tokens,
                        mask=current_mask
                    )
                
                elif args.train_mode in ['distill', 'curriculum']:
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(
                            idx=tokens, 
                            cond_idx=labels, 
                            targets=tokens
                        )
                    
                    student_logits, _ = student_model.module(
                        idx=tokens,
                        cond_idx=labels,
                        targets=tokens,
                        mask=current_mask
                    )
                    
                    loss = distillation_loss(
                        student_logits, teacher_logits, 
                        temperature=args.temperature
                    )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        
        dist.all_reduce(torch.tensor(avg_epoch_loss, device=device), op=dist.ReduceOp.SUM)
        avg_epoch_loss = avg_epoch_loss.item() / world_size
        
        elapsed = time.time() - start_time
        
        if rank == 0:
            log_msg = f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_epoch_loss:.4f} | Time: {elapsed/60:.1f}min | LR: {scheduler.get_last_lr()[0]:.6f}"
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()
        
        fid = None
        is_score = None
        
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            if rank == 0:
                print(f"\nRunning evaluation at epoch {epoch+1}...")
                eval_start = time.time()
            
            dist.barrier()
            
            samples_per_gpu = args.eval_samples // world_size
            if rank < args.eval_samples % world_size:
                samples_per_gpu += 1
            
            local_samples = generate_samples_for_eval(
                student_model.module, vq_model, device,
                num_samples=samples_per_gpu,
                batch_size=args.eval_batch_size,
                latent_size=latent_size,
                codebook_embed_dim=args.codebook_embed_dim,
                num_classes=args.num_classes,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            gathered_samples = [None] * world_size
            dist.all_gather_object(gathered_samples, local_samples)
            
            if rank == 0:
                all_samples = np.concatenate(gathered_samples, axis=0)[:args.eval_samples]
                
                eval_dir = os.path.join(args.output_dir, f"eval_epoch_{epoch+1}")
                os.makedirs(eval_dir, exist_ok=True)
                samples_path = os.path.join(eval_dir, "samples.npz")
                np.savez(samples_path, arr_0=all_samples)
                
                eval_cmd = f"python evaluations/c2i/evaluator.py {args.ref_path} {samples_path} -o {eval_dir}"
                eval_result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True, 
                                            cwd=ROOT)
                
                result_file_path = os.path.join(eval_dir, "samples.txt")
                if os.path.exists(result_file_path):
                    with open(result_file_path, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if line.startswith('FID'):
                                try:
                                    fid = float(line.split('\t')[1])
                                except:
                                    pass
                            elif line.startswith('Inception Score'):
                                try:
                                    is_score = float(line.split('\t')[1])
                                except:
                                    pass
                
                eval_time = time.time() - eval_start
                eval_msg = f"Evaluation | FID: {fid} | IS: {is_score} | Eval Time: {eval_time/60:.1f}min"
                print(eval_msg)
                log_file.write(eval_msg + '\n')
                log_file.flush()
                
                if fid is not None:
                    results_file.write(f"{epoch+1},{avg_epoch_loss:.4f},{fid},{is_score if is_score else 0},{elapsed/60:.1f}\n")
                    results_file.flush()
                    
                    if fid < best_fid:
                        best_fid = fid
                        best_fid_epoch = epoch + 1
                        checkpoint = {
                            "model": student_model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "loss": avg_epoch_loss,
                            "fid": fid,
                            "args": vars(args),
                        }
                        save_path = os.path.join(args.output_dir, "best_model.pt")
                        torch.save(checkpoint, save_path)
                        print(f"New best FID: {fid:.2f} at epoch {epoch+1}")
        
        if (epoch + 1) % args.save_every == 0 and rank == 0:
            checkpoint = {
                "model": student_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_epoch_loss,
                "args": vars(args),
            }
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    if rank == 0:
        final_checkpoint = {
            "model": student_model.module.state_dict(),
            "epoch": args.epochs,
            "args": vars(args),
        }
        save_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(final_checkpoint, save_path)
        
        summary = f"\n{'='*60}\nTraining Summary\n{'='*60}\n"
        summary += f"Best FID: {best_fid:.2f} at epoch {best_fid_epoch}\n"
        summary += f"Total training time: {(time.time() - start_time)/60:.1f} minutes\n"
        print(summary)
        log_file.write(summary)
        log_file.close()
        results_file.close()
        print(f"\nTraining completed. Final model saved to {save_path}")
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAR Training with LoRA and Multiple Modes")
    parser.add_argument("--train-mode", type=str, choices=['direct', 'distill', 'curriculum'], 
                        default='direct', help="Training mode: direct, distill, or curriculum")
    parser.add_argument("--data-path", type=str, default="./imagenet")
    parser.add_argument("--output-dir", type=str, default="./train_output")
    parser.add_argument("--teacher-ckpt", type=str, default="pretrained_models/c2i_L_256.pt")
    parser.add_argument("--vq-ckpt", type=str, default="pretrained_models/vq_ds16_c2i.pt")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--gpt-model", type=str, default="GPT-L")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--init-from-ar", action='store_true', help="Initialize student from AR weights")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--curriculum-warmup", type=int, default=5, help="Epochs for curriculum warmup")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate FID every N epochs")
    parser.add_argument("--eval-samples", type=int, default=5000, help="Number of samples for FID evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=100, help="Batch size for evaluation")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG scale for generation")
    parser.add_argument("--ref-path", type=str, default="pretrained_models/VIRTUAL_imagenet256_labeled.npz",
                        help="Reference npz for FID computation")
    args = parser.parse_args()
    main(args)

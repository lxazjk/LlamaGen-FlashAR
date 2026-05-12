# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import argparse
import inspect
import math
import os
import random
import time
import contextlib
from copy import deepcopy
from glob import glob
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import wids

# Enable TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models as FLASHAR_GPT_models
from autoregressive.models.ar_gpt import GPT_models as AR_GPT_models


#################################################################################
#                                   LoRA Utils                                 #
#################################################################################
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True, train_base=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        if not train_base:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    @classmethod
    def from_linear(cls, linear, r=0, lora_alpha=1.0, lora_dropout=0.0, train_base=False):
        lora = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
            train_base=train_base,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if linear.bias is not None:
                lora.bias.copy_(linear.bias)
        return lora

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.lora_dropout(x)
            lora_out = F.linear(lora_out, self.lora_A, bias=None)
            lora_out = F.linear(lora_out, self.lora_B, bias=None)
            result = result + lora_out * self.scaling
        return result


def apply_lora(model, target_substrings, r, alpha, dropout, train_base, logger):
    replaced = 0

    def _apply(module, prefix=""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_substrings):
                setattr(module, name, LoRALinear.from_linear(child, r=r, lora_alpha=alpha, lora_dropout=dropout, train_base=train_base))
                replaced += 1
            else:
                _apply(child, full_name)

    _apply(model)
    if logger is not None:
        logger.info(f"Applied LoRA to {replaced} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")


#################################################################################
#                           Checkpoint / Init Utils                            #
#################################################################################
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "module" in ckpt:
            return ckpt["module"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    return ckpt


def normalize_state_dict(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


#################################################################################
#                                Early Stopping                                 #
#################################################################################
class EarlyStopper:
    """Rank-0 early stopping helper. Lower is better."""

    def __init__(
        self,
        mode: str,
        patience_checks: int,
        min_delta: float,
        ema_decay: float,
        threshold: Optional[float],
    ):
        assert mode in ("plateau", "threshold", "both")
        self.mode = mode
        self.patience_checks = max(1, int(patience_checks))
        self.min_delta = float(min_delta)
        self.ema_decay = float(ema_decay)
        self.threshold = None if threshold is None else float(threshold)

        self.best = float("inf")
        self.best_step = -1
        self.num_bad = 0
        self.ema = None

    def update(self, value: float, step: int) -> tuple[bool, str]:
        v = float(value)
        if self.ema is None:
            self.ema = v
        else:
            d = self.ema_decay
            self.ema = d * self.ema + (1.0 - d) * v

        improved = (self.best - self.ema) > self.min_delta
        if improved:
            self.best = self.ema
            self.best_step = int(step)
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.mode in ("threshold", "both") and self.threshold is not None:
            if self.ema <= self.threshold:
                return True, f"threshold_reached(ema={self.ema:.6f} <= {self.threshold:.6f})"

        if self.mode in ("plateau", "both"):
            if self.num_bad >= self.patience_checks:
                return True, (
                    f"plateau(patience_checks={self.patience_checks}, min_delta={self.min_delta}, "
                    f"best_ema={self.best:.6f}@step{self.best_step})"
                )

        return False, ""


def load_checkpoint(path, map_location="cpu"):
    if path is None:
        return None
    return torch.load(path, map_location=map_location)


def init_student_from_teacher(student, teacher_state, logger=None):
    student_state = student.state_dict()
    new_state = {}
    loaded = 0

    for key, val in teacher_state.items():
        if key in student_state and student_state[key].shape == val.shape:
            new_state[key] = val
            loaded += 1

    # Reuse output/norm for medusa heads if present
    if "medusa_output.weight" in student_state and "output.weight" in teacher_state:
        if student_state["medusa_output.weight"].shape == teacher_state["output.weight"].shape:
            new_state["medusa_output.weight"] = teacher_state["output.weight"].clone()
    if "medusa_norm.weight" in student_state and "norm.weight" in teacher_state:
        if student_state["medusa_norm.weight"].shape == teacher_state["norm.weight"].shape:
            new_state["medusa_norm.weight"] = teacher_state["norm.weight"].clone()

    # Reuse last teacher layer for extra medusa layers
    base_layer = student.n_layer - 1
    extra_layers = len(student.layers) - student.n_layer
    if extra_layers > 0:
        base_prefix = f"layers.{base_layer}."
        for extra_id in range(student.n_layer, len(student.layers)):
            target_prefix = f"layers.{extra_id}."
            for key, val in teacher_state.items():
                if key.startswith(base_prefix):
                    target_key = target_prefix + key[len(base_prefix):]
                    if target_key in student_state and student_state[target_key].shape == val.shape:
                        new_state[target_key] = val.clone()

    missing, unexpected = student.load_state_dict(new_state, strict=False)
    if logger is not None:
        logger.info(f"Loaded {loaded} teacher keys into student. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")


#################################################################################
#                                 Mask Helpers                                 #
#################################################################################
def build_base_masks(model, t5_len, block_size, device):
    total_len = t5_len + block_size
    causal = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
    proximity = causal.clone()
    model.setup_proximity_mask(
        proximity[t5_len:, t5_len:],
        block_size,
    )
    eye = torch.eye(total_len, dtype=torch.bool, device=device)
    delta = proximity & ~causal
    delta_indices = [torch.nonzero(delta[q], as_tuple=False).view(-1) for q in range(total_len)]
    return causal, proximity, eye, delta_indices


def apply_caption_mask(base_mask, emb_mask, eye):
    mask = base_mask.clone()
    t5_len = emb_mask.shape[0]
    mask[:, :t5_len] = mask[:, :t5_len] & emb_mask.unsqueeze(0)
    mask = mask | eye
    return mask


def pick_mask(base_causal, base_proximity, schedule, step, anneal_steps, rng, delta_indices=None):
    if schedule == "static_causal":
        return base_causal, 0.0
    if schedule == "static_proximity":
        return base_proximity, 1.0
    if schedule in ("progressive", "linear"):
        if anneal_steps <= 0:
            return base_proximity, 1.0
        p = min(1.0, step / anneal_steps)
        mask = base_causal.clone()
        if delta_indices is not None:
            for q, idx in enumerate(delta_indices):
                if idx.numel() == 0:
                    continue
                k = int(math.ceil(p * idx.numel()))
                if k <= 0:
                    continue
                mask[:, q, idx[:k]] = True
        return mask, p
    if anneal_steps <= 0:
        return base_proximity, 1.0
    return base_proximity, 1.0


#################################################################################
#                                 Data Helpers                                 #
#################################################################################
def make_dataset_train(trainset_url, transform):
    def make_sample(sample):
        image = sample[".jpg"]
        label = sample[".json"]["prompt"]
        return transform(image), label

    trainset = wids.ShardListDataset(trainset_url, keep=True)
    trainset = trainset.add_transform(make_sample)
    return trainset


def build_caption_batch(captions, t5_xxl, args, device, base_mask, eye):
    caption_embs, emb_masks = t5_xxl.get_text_embeddings(captions)
    B = caption_embs.shape[0]
    t5_len = args.t5_feature_max_len
    dim = args.t5_feature_dim
    c_indices = torch.zeros((B, t5_len, dim), device=device)
    masks = []

    for i in range(B):
        t5_feat_len = int(emb_masks[i].sum().item())
        t5_feat_len = min(t5_feat_len, t5_len)
        if t5_feat_len > 0:
            valid_embs = caption_embs[i, :t5_feat_len].to(device)
            c_indices[i, -t5_feat_len:] = valid_embs
        emb_mask = torch.zeros((t5_len,), device=device, dtype=torch.bool)
        if t5_feat_len > 0:
            emb_mask[-t5_feat_len:] = True
        masks.append(apply_caption_mask(base_mask, emb_mask, eye))

    mask = torch.stack(masks, dim=0)
    return c_indices, mask


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    assert args.cls_token_num == args.t5_feature_max_len, "cls_token_num should match t5_feature_max_len for t2i"

    # Setup student model (FlashAR)
    latent_size = args.image_size // args.downsample_size
    student = FLASHAR_GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        drop_path_rate=args.drop_path_rate,
        medusa_attention_num=args.medusa_attention_num,
    ).to(device)
    logger.info(f"Student GPT Parameters: {sum(p.numel() for p in student.parameters()):,}")

    if args.ema:
        ema = deepcopy(student).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Load init / resume
    train_steps = 0
    start_epoch = 0
    resume_ckpt = load_checkpoint(args.gpt_ckpt, map_location="cpu") if args.gpt_ckpt else None

    if resume_ckpt is not None and args.use_lora:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        apply_lora(student, targets, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_train_base, logger)

    if resume_ckpt is not None:
        ckpt_state = normalize_state_dict(extract_state_dict(resume_ckpt))
        student.load_state_dict(ckpt_state, strict=False)
        if args.ema and isinstance(resume_ckpt, dict) and "ema" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema"])
        if isinstance(resume_ckpt, dict) and "steps" in resume_ckpt:
            train_steps = resume_ckpt["steps"]
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    else:
        if args.init_ckpt is None:
            args.init_ckpt = args.teacher_ckpt
        if args.init_ckpt:
            teacher_ckpt = load_checkpoint(args.init_ckpt, map_location="cpu")
            teacher_state = extract_state_dict(teacher_ckpt)
            init_student_from_teacher(student, teacher_state, logger)
            del teacher_ckpt
        if args.ema:
            update_ema(ema, student, decay=0)

        if args.use_lora:
            targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
            apply_lora(student, targets, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_train_base, logger)

    # Setup optimizer
    optimizer = create_optimizer(student, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = make_dataset_train(args.data_path, transform)
    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    try:
        logger.info(f"Dataset contains {len(dataset):,} images")
    except Exception:
        pass

    if resume_ckpt is not None and train_steps > 0:
        steps_per_epoch = int(len(dataset) / args.global_batch_size)
        start_epoch = int(train_steps / max(steps_per_epoch, 1))

    # Setup tokenizer + T5
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model.eval()
    vq_ckpt = load_checkpoint(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt)
    del vq_ckpt

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_model_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
    )

    # Setup teacher model for distillation
    teacher = None
    if args.kd_weight > 0:
        assert args.teacher_ckpt is not None, "Teacher ckpt must be provided when kd_weight > 0."
        teacher = AR_GPT_models[args.teacher_gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=args.dropout_p,
            ffn_dropout_p=args.dropout_p,
            drop_path_rate=0.0,
            token_dropout_p=0.0,
        ).to(device)
        teacher_ckpt = load_checkpoint(args.teacher_ckpt, map_location="cpu")
        teacher_state = extract_state_dict(teacher_ckpt)
        teacher.load_state_dict(teacher_state, strict=False)
        teacher.eval()
        requires_grad(teacher, False)
        del teacher_ckpt
        logger.info("Teacher model loaded for logits distillation.")

    # Compile model if requested
    if not args.no_compile:
        logger.info("compiling the student model... (may take several minutes)")
        student = torch.compile(student)

    # Setup base masks
    mask_model = student._orig_mod if not args.no_compile else student
    base_causal, base_proximity, eye, delta_indices = build_base_masks(
        mask_model,
        args.t5_feature_max_len,
        latent_size ** 2,
        device,
    )
    rng = random.Random(args.global_seed + rank)

    # Wrap with DDP
    student = DDP(student.to(device), device_ids=[args.gpu])
    student.train()
    if args.ema:
        ema.eval()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # Training loop
    log_steps = 0
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    start_time = time.time()
    accum_steps = max(1, args.gradient_accumulation_steps)
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    logger.info(f"Training for {args.epochs} epochs...")
    stop_training = False
    early_stop_triggered = False

    early_stopper = None
    if rank == 0 and args.early_stop:
        early_stopper = EarlyStopper(
            mode=args.early_stop_mode,
            patience_checks=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            ema_decay=args.early_stop_ema_decay,
            threshold=(None if args.early_stop_threshold < 0 else args.early_stop_threshold),
        )

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kd = 0.0
        for x, captions in loader:
            x = x.to(device, non_blocking=True)

            base_mask, mask_prob = pick_mask(
                base_causal,
                base_proximity,
                args.mask_schedule,
                train_steps,
                args.mask_anneal_steps,
                rng,
                delta_indices=delta_indices,
            )
            c_indices, attn_mask = build_caption_batch(captions, t5_xxl, args, device, base_mask, eye)

            with torch.no_grad():
                _, _, [_, _, indices] = vq_model.encode(x)
            z_indices = indices.reshape(x.shape[0], -1)

            sync = ((micro_step + 1) % accum_steps == 0)
            context = student.no_sync() if not sync else contextlib.nullcontext()
            with context:
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    student_logits, ce_loss = student(cond_idx=c_indices, idx=z_indices, targets=z_indices, mask=attn_mask)
                    kd_loss = torch.tensor(0.0, device=device)
                    if teacher is not None and args.kd_weight > 0:
                        with torch.no_grad():
                            seq_len = z_indices.size(1) - 1 + teacher.cls_token_num
                            input_pos = torch.arange(seq_len, device=device)
                            teacher_logits, _ = teacher(
                                cond_idx=c_indices,
                                idx=z_indices[:, :-1],
                                targets=None,
                                input_pos=input_pos,
                            )
                        t = args.kd_temperature
                        kd_loss = F.kl_div(
                            F.log_softmax(student_logits.float() / t, dim=-1),
                            F.softmax(teacher_logits.float() / t, dim=-1),
                            reduction="batchmean",
                        ) * (t * t)
                    loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss
                    loss = loss / accum_steps
                scaler.scale(loss).backward()

            accum_loss += (args.ce_weight * ce_loss.item() + args.kd_weight * kd_loss.item())
            accum_ce += ce_loss.item()
            accum_kd += kd_loss.item()
            micro_step += 1

            if sync:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if args.ema:
                    update_ema(ema, student.module._orig_mod if not args.no_compile else student.module)

                # Per-optimizer-step metrics before we reset accumulation buffers.
                step_loss_val = accum_loss / accum_steps
                step_ce_val = accum_ce / accum_steps
                step_kd_val = (accum_kd / accum_steps) if teacher is not None else 0.0

                running_loss += step_loss_val
                running_ce += step_ce_val
                running_kd += step_kd_val
                accum_loss = 0.0
                accum_ce = 0.0
                accum_kd = 0.0
                log_steps += 1
                train_steps += 1

                if args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                    stop_training = True

                if args.early_stop and (train_steps >= args.early_stop_warmup_steps) and (train_steps % args.early_stop_check_every == 0):
                    cur_loss = torch.tensor(step_loss_val, device=device)
                    cur_ce = torch.tensor(step_ce_val, device=device)
                    cur_kd = torch.tensor(step_kd_val, device=device)
                    dist.all_reduce(cur_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cur_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cur_kd, op=dist.ReduceOp.SUM)
                    cur_loss = (cur_loss.item() / dist.get_world_size())
                    cur_ce = (cur_ce.item() / dist.get_world_size())
                    cur_kd = (cur_kd.item() / dist.get_world_size())

                    if rank == 0 and early_stopper is not None:
                        metric_map = {"loss": cur_loss, "ce": cur_ce, "kd": cur_kd}
                        mval = metric_map.get(args.early_stop_metric, cur_ce)
                        should_stop, reason = early_stopper.update(mval, step=train_steps)
                        if should_stop:
                            stop_training = True
                            early_stop_triggered = True
                            logger.info(
                                f"EarlyStop triggered at step={train_steps} metric={args.early_stop_metric} "
                                f"ema_best={early_stopper.best:.6f}@step{early_stopper.best_step} reason={reason}"
                            )

                stop_flag = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int32)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop_training = bool(stop_flag.item())
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_ce = torch.tensor(running_ce / log_steps, device=device)
                    avg_kd = torch.tensor(running_kd / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_kd, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    avg_ce = avg_ce.item() / dist.get_world_size()
                    avg_kd = avg_kd.item() / dist.get_world_size()
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KD: {avg_kd:.4f}, "
                        f"MaskP: {mask_prob:.2f}, Steps/Sec: {steps_per_sec:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    log_steps = 0
                    start_time = time.time()

                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        if not args.no_compile:
                            model_weight = student.module._orig_mod.state_dict()
                        else:
                            model_weight = student.module.state_dict()
                        checkpoint = {
                            "model": model_weight,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "steps": train_steps,
                            "args": args,
                        }
                        if args.ema:
                            checkpoint["ema"] = ema.state_dict()
                        if not args.no_local_save:
                            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                    dist.barrier()

                if stop_training:
                    break

        scheduler.step()

        if stop_training:
            if rank == 0 and args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                logger.info(f"Reached --max-steps={args.max_steps}, stopping early.")
            break

    # Save a checkpoint when early-stop triggers (opt-in)
    es_flag = torch.tensor(1 if early_stop_triggered else 0, device=device, dtype=torch.int32)
    dist.all_reduce(es_flag, op=dist.ReduceOp.MAX)
    early_stop_triggered = bool(es_flag.item())
    if early_stop_triggered and args.save_on_early_stop and rank == 0:
        if not args.no_compile:
            model_weight = student.module._orig_mod.state_dict()
        else:
            model_weight = student.module.state_dict()
        checkpoint = {
            "model": model_weight,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "steps": train_steps,
            "args": args,
        }
        if args.ema:
            checkpoint["ema"] = ema.state_dict()
        if not args.no_local_save:
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved early-stop checkpoint to {checkpoint_path}")
        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved early-stop checkpoint in cloud to {cloud_checkpoint_path}")

    if rank == 0:
        if not args.no_compile:
            model_weight = student.module._orig_mod.state_dict()
        else:
            model_weight = student.module.state_dict()
        checkpoint = {"model": model_weight}
        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/last_version.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

    student.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    # model
    parser.add_argument("--gpt-model", type=str, choices=list(FLASHAR_GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="resume checkpoint")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--medusa-attention-num", type=int, default=1)
    parser.add_argument("--no-compile", action='store_true')

    # t5
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)

    # distillation
    parser.add_argument("--teacher-gpt-model", type=str, choices=list(AR_GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)

    # lora
    parser.add_argument("--use-lora", action='store_true')
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", type=str, default="layers.wqkv,layers.wo,layers.w1,layers.w2,layers.w3")
    parser.add_argument("--lora-train-base", action='store_true')

    # optimization
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    # runtime control
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (i.e., after gradient accumulation sync). -1 means no limit.",
    )

    # convergence / early stop
    parser.add_argument("--early-stop", action='store_true', help="Enable convergence-based early stopping")
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="ce",
        choices=["loss", "ce", "kd"],
        help="Which metric to monitor for convergence (lower is better).",
    )
    parser.add_argument(
        "--early-stop-mode",
        type=str,
        default="plateau",
        choices=["plateau", "threshold", "both"],
        help="Stop on plateau, threshold, or both.",
    )
    parser.add_argument(
        "--early-stop-warmup-steps",
        type=int,
        default=500,
        help="Do not consider early stop before this optimizer step.",
    )
    parser.add_argument(
        "--early-stop-check-every",
        type=int,
        default=50,
        help="Check convergence every N optimizer steps (must be the same on all ranks).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Plateau patience in number of checks (so effective patience in steps is patience*check_every).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum EMA improvement to be considered progress.",
    )
    parser.add_argument(
        "--early-stop-ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for smoothing the monitored metric. Larger = smoother.",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=-1.0,
        help="If >=0, stop when EMA(metric) <= threshold. -1 disables threshold stop.",
    )

    parser.add_argument(
        "--save-on-early-stop",
        action='store_true',
        help="If set, save a step checkpoint when early-stop triggers (in addition to last_version.pt).",
    )

    # mask schedule
    parser.add_argument("--mask-schedule", type=str, default="static_proximity", choices=["static_causal", "static_proximity", "linear", "progressive"])
    parser.add_argument("--mask-anneal-steps", type=int, default=20000)

    # tokenizer
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    args = parser.parse_args()
    main(args)

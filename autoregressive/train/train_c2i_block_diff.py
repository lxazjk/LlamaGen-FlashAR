"""Block Diffusion training script for class-conditional image generation.

Based on train_c2i_flashar.py but simplified: no KD, no proximity mask, no R/B
dual heads.  Uses absorbing-state discrete diffusion with block-causal
attention.
"""
import argparse
import inspect
import os
import time
import contextlib
from datetime import timedelta
from copy import deepcopy
from glob import glob
from functools import partial

import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_block_diff import BlockDiff_models
from autoregressive.models.generate_block_diff import generate_block_diff
from autoregressive.utils.block_diffusion import (
    MASK_TOKEN_ID,
    NOISE_SCHEDULES,
    get_block_assignments,
    build_block_causal_mask,
    apply_absorbing_noise,
)
from autoregressive.utils.fid_eval import run_fid_eval
from autoregressive.utils.train_logging import JsonlWriter
from tokenizer.tokenizer_image.vq_model import VQ_models


###############################################################################
#                        Checkpoint / Init Utils                              #
###############################################################################

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "module", "state_dict"):
            if k in ckpt:
                return ckpt[k]
    return ckpt


def _normalize_state_dict(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix) :]: v for k, v in state.items()}
    return state


def _parse_step_from_ckpt(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.isdigit():
        return int(name)
    if name == "last_version":
        return float("inf")
    return -1


def resolve_resume_ckpt(args, logger):
    """Auto-discover the latest checkpoint from local/cloud dirs."""
    if args.gpt_ckpt:
        return args.gpt_ckpt
    if not args.auto_resume:
        return None

    search_dirs = []
    if args.resume_local_dir:
        search_dirs.append(os.path.join(args.resume_local_dir, "checkpoints"))
    if args.resume_cloud_dir:
        search_dirs.append(os.path.join(args.resume_cloud_dir, "checkpoints"))

    candidates = []
    for ckpt_dir in search_dirs:
        if ckpt_dir and os.path.isdir(ckpt_dir):
            candidates.extend(glob(os.path.join(ckpt_dir, "*.pt")))

    candidates = [c for c in candidates if _parse_step_from_ckpt(c) >= 0]
    if not candidates:
        if logger is not None:
            logger.warning("auto-resume enabled but no valid checkpoint found.")
        return None

    best = max(candidates, key=_parse_step_from_ckpt)
    if logger is not None:
        logger.info(f"Auto-resume selected checkpoint: {best}")
    return best


def init_from_teacher(model, teacher_state, logger=None):
    """Load AR teacher weights into BlockDiffTransformer.

    Handles the expanded ``tok_embeddings`` (vocab_size+1 vs vocab_size) and
    skips medusa-specific keys that don't exist in the block-diff model.
    """
    model_state = model.state_dict()
    new_state = {}
    loaded, loaded_numel = 0, 0
    param_keys = set(name for name, _ in model.named_parameters())

    for key, val in teacher_state.items():
        if any(key.startswith(p) for p in ("medusa_norm.", "medusa_output.")):
            continue

        if key == "tok_embeddings.weight" and key in model_state:
            s_shape = model_state[key].shape
            if s_shape[0] == val.shape[0] + 1 and s_shape[1] == val.shape[1]:
                expanded = model_state[key].clone()
                expanded[: val.shape[0]] = val
                new_state[key] = expanded
                loaded += 1
                if key in param_keys:
                    loaded_numel += int(expanded.numel())
                continue

        if key in model_state and model_state[key].shape == val.shape:
            new_state[key] = val
            loaded += 1
            if key in param_keys:
                loaded_numel += int(val.numel())

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if logger:
        total = sum(int(p.numel()) for p in model.parameters())
        ratio = loaded_numel / total if total > 0 else 0
        logger.info(
            f"Teacher init: {loaded} keys loaded. "
            f"Missing: {len(missing)}, unexpected: {len(unexpected)}. "
            f"Coverage: {ratio * 100:.2f}%"
        )


###############################################################################
#                            Optimizer                                        #
###############################################################################

def _create_optimizer(model, weight_decay, lr, betas, logger):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    logger.info(
        f"Decay params: {sum(p.numel() for p in decay):,}, "
        f"no-decay: {sum(p.numel() for p in nodecay):,}"
    )
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        groups, lr=lr, betas=betas, **(dict(fused=True) if fused else {})
    )
    return optimizer


###############################################################################
#                            Training Loop                                    #
###############################################################################

def main(args):
    assert torch.cuda.is_available(), "GPU required"
    assert args.steps_per_epoch > 0

    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # ── experiment directory ────────────────────────────────────────────────
    experiment_dir = checkpoint_dir = cloud_checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_tag = args.gpt_model.replace("/", "-")
        if args.resume_local_dir:
            experiment_dir = args.resume_local_dir
        else:
            idx = len(glob(f"{args.results_dir}/*"))
            experiment_dir = (
                f"{args.results_dir}/{idx:03d}-{model_tag}-BD-{args.block_strategy}"
            )
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        if args.resume_cloud_dir:
            cloud_checkpoint_dir = f"{args.resume_cloud_dir}/checkpoints"
        else:
            ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cloud_checkpoint_dir = (
                f"{args.cloud_save_path}/{ts}/"
                f"{os.path.basename(experiment_dir)}/checkpoints"
            )
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
    else:
        logger = create_logger(None)

    # ── logging / wandb ─────────────────────────────────────────────────────
    loss_writer = eval_writer = None
    if rank == 0:
        if experiment_dir and not os.path.isabs(args.fid_sample_dir):
            args.fid_sample_dir = os.path.join(experiment_dir, args.fid_sample_dir)
        metrics_dir = os.path.join(experiment_dir, "metrics")
        loss_writer = JsonlWriter(os.path.join(metrics_dir, "train_loss_steps.jsonl"))
        eval_writer = JsonlWriter(os.path.join(metrics_dir, "eval_metrics.jsonl"))
        if not args.no_wandb:
            os.environ["WANDB_DIR"] = experiment_dir
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or os.path.basename(experiment_dir),
                config=vars(args),
            )

    # Gloo control group for slow barriers (FID eval / ckpt)
    control_group = None
    try:
        control_group = dist.new_group(
            backend="gloo", timeout=timedelta(hours=24)
        )
    except Exception:
        pass

    def control_barrier():
        if control_group is not None:
            dist.barrier(group=control_group)
        else:
            dist.barrier()

    def control_all_reduce_min(t: torch.Tensor):
        if control_group is not None:
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=control_group)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)

    logger.info(f"{args}")
    logger.info(f"rank={rank}, seed={seed}, world_size={dist.get_world_size()}")

    # ── model ───────────────────────────────────────────────────────────────
    latent_size = args.image_size // args.downsample_size
    dp = 0.0 if args.drop_path_rate > 0.0 else args.dropout_p
    model = BlockDiff_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dp,
        ffn_dropout_p=dp,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"BlockDiff params: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

    # ── init / resume ───────────────────────────────────────────────────────
    train_steps = 0
    start_epoch = 0
    resume_path = resolve_resume_ckpt(args, logger if rank == 0 else None)
    resume_ckpt = None
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location="cpu")

    if resume_ckpt is not None:
        state = _normalize_state_dict(_extract_state_dict(resume_ckpt))
        model.load_state_dict(state, strict=False)
        if args.ema and isinstance(resume_ckpt, dict) and "ema" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema"])
        if isinstance(resume_ckpt, dict) and "steps" in resume_ckpt:
            train_steps = resume_ckpt["steps"]
        logger.info(f"Resumed from {resume_path}, steps={train_steps}")
    elif args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        teacher_state = _normalize_state_dict(_extract_state_dict(ckpt))
        init_from_teacher(model, teacher_state, logger)
        del ckpt
        if args.ema:
            update_ema(ema, model, decay=0)

    # ── optimizer / scheduler ───────────────────────────────────────────────
    optimizer = _create_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger
    )
    milestones = [int(args.epochs * r) for r in (0.5, 2 / 3, 5 / 6)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
    del resume_ckpt

    # ── data ────────────────────────────────────────────────────────────────
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} images from {args.code_path}")

    steps_per_epoch = int(args.steps_per_epoch)
    if train_steps > 0:
        start_epoch = int(train_steps / max(steps_per_epoch, 1))

    # ── precompute block-causal attention mask ──────────────────────────────
    seq_len = latent_size**2
    grid_size = latent_size
    block_assignments = get_block_assignments(seq_len, grid_size, args.block_strategy)
    attn_mask = build_block_causal_mask(
        seq_len, block_assignments, cls_token_num=args.cls_token_num
    ).to(device)
    num_blocks = int(block_assignments.max().item()) + 1
    logger.info(
        f"Block strategy: {args.block_strategy}, blocks: {num_blocks}, "
        f"tokens/block: {seq_len // num_blocks}"
    )

    # noise schedule: linear (mask_rate = t) with 1/t ELBO weighting
    # (consistent with BD3-LM / MDLM parameterization)

    # ── compile ─────────────────────────────────────────────────────────────
    if not args.no_compile:
        logger.info("Compiling model …")
        model = torch.compile(model)

    # ── DDP ─────────────────────────────────────────────────────────────────
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()
    if args.ema:
        ema.eval()

    # ── VQ decoder for FID eval ─────────────────────────────────────────────
    vq_model = None
    if args.fid_ref:
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
        ).to(device)
        vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(
            vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt
        )
        vq_model.eval()
        del vq_ckpt
        if rank == 0:
            logger.info("VQ decoder loaded for FID evaluation.")

    ptdtype = {
        "none": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]

    # ── training ────────────────────────────────────────────────────────────
    log_steps = 0
    running_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    logger.info(f"Training for {args.epochs} epochs × {steps_per_epoch} steps …")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch} …")
        epoch_steps = 0
        data_iter = iter(loader)

        while epoch_steps < steps_per_epoch:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)

            B = z_indices.shape[0]
            t_blocks = torch.rand(B, num_blocks, device=device)
            t_blocks = t_blocks.clamp(min=1e-5)
            t_per_pos = t_blocks[:, block_assignments.to(device)]  # [B, seq_len]
            mask_rate_per_pos = t_per_pos  # linear schedule: mask_rate = t
            rand = torch.rand_like(z_indices.float())
            is_masked = rand < mask_rate_per_pos
            noisy_tokens = z_indices.clone()
            noisy_tokens[is_masked] = MASK_TOKEN_ID
            loss_weights = 1.0 / t_per_pos  # ELBO 1/t weighting

            with torch.cuda.amp.autocast(dtype=ptdtype):
                _, loss = model(
                    idx=noisy_tokens,
                    cond_idx=c_indices,
                    t=t_per_pos,
                    attn_mask=attn_mask,
                    targets=z_indices,
                    is_masked=is_masked,
                    loss_weights=loss_weights,
                )

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if args.ema:
                raw = (
                    model.module._orig_mod
                    if not args.no_compile
                    else model.module
                )
                update_ema(ema, raw)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            epoch_steps += 1

            if train_steps % args.log_every == 0:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                torch.cuda.synchronize()
                sps = log_steps / (time.time() - start_time)
                logger.info(
                    f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {sps:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                )
                if rank == 0:
                    if loss_writer:
                        loss_writer.write(
                            {
                                "step": train_steps,
                                "epoch": epoch,
                                "loss": avg_loss,
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )
                    if not args.no_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                            },
                            step=train_steps,
                        )
                running_loss = 0.0
                log_steps = 0
                start_time = time.time()

        scheduler.step()

        # ── epoch-end FID evaluation ────────────────────────────────────────
        is_last_epoch = (epoch == args.epochs - 1)
        do_fid = (epoch % args.fid_every_n_epochs == 0) or is_last_epoch
        if args.fid_ref and do_fid:
            fid_ok = torch.tensor(1, device=device, dtype=torch.int32)
            raw_model = (
                model.module._orig_mod if not args.no_compile else model.module
            )
            eval_model = ema if (args.ema and args.fid_use_ema) else raw_model
            generate_fn = partial(
                generate_block_diff,
                block_strategy=args.block_strategy,
                denoise_steps=args.denoise_steps,
            )
            try:
                sample_dir = os.path.join(args.fid_sample_dir, "latest")
                npz_path, txt_path, metrics = run_fid_eval(
                    args,
                    eval_model,
                    vq_model,
                    device,
                    train_steps,
                    logger,
                    generate_fn,
                    epoch=epoch,
                    sample_dir=sample_dir,
                    keep_last_samples=True,
                    rank=rank,
                    world_size=dist.get_world_size(),
                    barrier=control_barrier,
                )
                if rank == 0:
                    if eval_writer:
                        eval_writer.write(
                            {"epoch": epoch, "step": train_steps, **metrics}
                        )
                    if not args.no_wandb:
                        wandb.log(
                            {f"eval/{k}": v for k, v in metrics.items()},
                            step=train_steps,
                        )
            except Exception as e:
                fid_ok.fill_(0)
                logger.exception(f"FID eval failed at epoch={epoch}: {e}")

            control_all_reduce_min(fid_ok)
            if fid_ok.item() == 0:
                if rank == 0:
                    logger.info(f"FID eval had failures; --fid-fail-action={args.fid_fail_action}")
                if args.fid_fail_action == "stop":
                    break
            control_barrier()

        # ── save checkpoint (overwrite each epoch) ──────────────────────────
        if rank == 0:
            raw_model = (
                model.module._orig_mod if not args.no_compile else model.module
            )
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "steps": train_steps,
                "args": args,
            }
            if args.ema:
                ckpt["ema"] = ema.state_dict()
            if not args.no_local_save:
                path = f"{checkpoint_dir}/last_version.pt"
                torch.save(ckpt, path)
                logger.info(f"Saved to {path}")
            cloud_path = f"{cloud_checkpoint_dir}/last_version.pt"
            torch.save(ckpt, cloud_path)
            logger.info(f"Saved cloud to {cloud_path}")

    model.eval()
    if rank == 0:
        if loss_writer:
            loss_writer.close()
        if eval_writer:
            eval_writer.close()
        if not args.no_wandb:
            wandb.finish()
    logger.info("Done!")
    dist.destroy_process_group()


###############################################################################
#                             CLI                                             #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--no-local-save", action="store_true")
    parser.add_argument("--dataset", type=str, default="imagenet_code")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--train-max-samples", type=int, default=-1)
    parser.add_argument("--train-subset-seed", type=int, default=0)
    parser.add_argument("--train-subset-shuffle", action="store_true")

    # model
    parser.add_argument(
        "--gpt-model",
        type=str,
        choices=list(BlockDiff_models.keys()),
        default="GPT-L",
    )
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--gpt-type", type=str, default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--init-ckpt", type=str, default=None)

    # block diffusion
    parser.add_argument(
        "--block-strategy",
        type=str,
        choices=["mdlm", "row", "sub4x4"],
        default="mdlm",
    )
    parser.add_argument(
        "--noise-schedule",
        type=str,
        choices=list(NOISE_SCHEDULES.keys()),
        default="cosine",
    )
    parser.add_argument("--denoise-steps", type=int, default=16)

    # optimization
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-loss-every", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
    )

    # resume
    parser.add_argument("--resume-local-dir", type=str, default=None)
    parser.add_argument("--resume-cloud-dir", type=str, default=None)

    # fid evaluation
    parser.add_argument("--fid-every-n-epochs", type=int, default=1,
                        help="Run FID eval every N epochs (1=every epoch)")
    parser.add_argument("--fid-ref", type=str, default=None)
    parser.add_argument("--fid-num-samples", type=int, default=50000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-sample-dir", type=str, default="samples")
    parser.add_argument("--fid-use-ema", action="store_true")
    parser.add_argument("--fid-skip-evaluator", action="store_true")
    parser.add_argument("--fid-top-k", type=int, default=0)
    parser.add_argument("--fid-top-p", type=float, default=1.0)
    parser.add_argument("--fid-temperature", type=float, default=1.0)
    parser.add_argument("--fid-cfg-scale", type=float, default=1.5)
    parser.add_argument("--fid-cfg-interval", type=float, default=-1)
    parser.add_argument(
        "--fid-fail-action",
        type=str,
        default="skip",
        choices=["stop", "skip"],
    )
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size-eval", type=int, default=256)

    # wandb
    parser.add_argument("--wandb-project", type=str, default="c2i_block_diff")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()
    main(args)

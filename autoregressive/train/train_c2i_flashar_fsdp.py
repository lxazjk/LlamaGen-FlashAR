# Modified from train_c2i_flashar.py and train_c2i_fsdp.py.
import argparse
import contextlib
import functools
import inspect
import os
import random
import sys
import time
from datetime import timedelta
from glob import glob

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Enable TF32 for speed.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.logger import create_logger
from utils.ema import requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models as FLASHAR_GPT_models
from autoregressive.models.generate import generate
from autoregressive.models.ar_gpt import GPT_models as AR_GPT_models
from autoregressive.utils.mask import build_masks, pick_mask
from autoregressive.utils.fid_eval import clear_generation_caches, run_fid_eval
from autoregressive.utils.train_logging import JsonlWriter
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.train.train_c2i_flashar import (
    apply_random_context_mask,
    configure_vertical_branch,
    extract_state_dict,
    init_student_from_teacher,
    load_checkpoint,
    normalize_state_dict,
    resolve_resume_ckpt,
)


def kd_allowed(args, epoch: int, step: int) -> bool:
    kd_end_step = int(getattr(args, "kd_end_step", -1))
    kd_end_epoch = int(getattr(args, "kd_end_epoch", -1))
    if kd_end_step >= 0 and step >= kd_end_step:
        return False
    if kd_end_epoch >= 0 and epoch >= kd_end_epoch:
        return False
    return True


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    precision_map = {
        "none": torch.float32,
        "fp32": torch.float32,
        "tf32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    reduce_dtype = precision_map[args.grad_precision or args.mixed_precision]
    param_dtype = precision_map[args.mixed_precision]
    if dist.is_initialized() and dist.get_world_size() < 2:
        raise ValueError("train_c2i_flashar_fsdp.py is intended for multi-GPU FSDP; use at least 2 GPUs.")
    sharding_strategy = {
        "fsdp": ShardingStrategy.FULL_SHARD,
        "sdp": ShardingStrategy.SHARD_GRAD_OP,
        "hsdp": ShardingStrategy.HYBRID_SHARD,
    }[args.data_parallel]

    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module in model.get_fsdp_wrap_module_list(),
        ),
        device_id=device,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def create_optimizer(model, weight_decay, learning_rate, betas, rank, logger):
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay_params = [param for name, param in param_dict.items() if "norm" not in name]
    nodecay_params = [param for name, param in param_dict.items() if "norm" in name]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(param.numel() for param in decay_params)
    num_nodecay_params = sum(param.numel() for param in nodecay_params)
    logger.info(f"(rank {rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def canonical_param_name(name):
    for prefix in ("_fsdp_wrapped_module.", "module.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def get_phase_trainable_names(raw_model, scope):
    scope = str(scope or "full")
    if scope == "full":
        return None
    if scope == "none":
        return set()
    vertical_head_prefixes = ("medusa_norm.", "medusa_output.", "hv_gate_mlp.", "hv_mix_logit")
    extra_start = int(getattr(raw_model, "n_layer", 0))
    names = set()
    for name, _ in raw_model.named_parameters():
        is_vertical_head = name.startswith(vertical_head_prefixes)
        is_extra_layer = False
        if name.startswith("layers."):
            parts = name.split(".", 2)
            if len(parts) > 1 and parts[1].isdigit():
                is_extra_layer = int(parts[1]) >= extra_start
        if scope == "vertical_head" and is_vertical_head:
            names.add(name)
        elif scope == "vertical_extra" and (is_vertical_head or is_extra_layer):
            names.add(name)
    if scope not in {"vertical_head", "vertical_extra"}:
        raise ValueError(f"Unknown --phase1-train-scope: {scope}")
    return names


def log_trainable_scope(raw_model, trainable_names, scope, logger):
    total = 0
    trainable = 0
    tensors = 0
    for name, param in raw_model.named_parameters():
        n = int(param.numel())
        total += n
        if trainable_names is None or name in trainable_names:
            trainable += n
            tensors += 1
    logger.info(
        f"Train phase scope={scope}: updating {tensors} tensors, "
        f"{trainable:,}/{total:,} params ({100.0 * trainable / max(total, 1):.2f}%)."
    )


def mask_nontrainable_grads(model, trainable_names):
    if trainable_names is None:
        return
    for name, param in model.named_parameters():
        if canonical_param_name(name) not in trainable_names:
            param.grad = None


def fsdp_full_model_state(model):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        return model.state_dict()


def save_last_checkpoint(args, model, optimizer, scheduler, train_steps, checkpoint_dir, cloud_checkpoint_dir, logger, rank):
    clear_generation_caches(model)
    model_state = fsdp_full_model_state(model)
    if rank == 0:
        checkpoint = {
            "model": model_state,
            "steps": train_steps,
            "args": args,
        }
        if args.save_optimizer:
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["scheduler"] = scheduler.state_dict()
        if not args.no_local_save:
            path = os.path.join(checkpoint_dir, "last_version.pt")
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
        cloud_path = os.path.join(cloud_checkpoint_dir, "last_version.pt")
        torch.save(checkpoint, cloud_path)
        logger.info(f"Saved checkpoint in cloud to {cloud_path}")
    del model_state
    dist.barrier()


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.gpt_type == "c2i", "FSDP script currently supports c2i only."
    assert args.steps_per_epoch > 0, "--steps-per-epoch must be > 0."
    if args.ema:
        raise ValueError("EMA is intentionally disabled for FSDP FlashAR to avoid keeping a full XXL copy on every GPU.")
    if not args.no_compile:
        raise ValueError("torch.compile + FSDP is not enabled in this script; pass --no-compile.")
    if not (0.0 <= args.random_mask_prob <= 1.0):
        raise ValueError(f"--random-mask-prob must be in [0, 1], got {args.random_mask_prob}.")
    if args.random_mask_replace == "mask" and not (0 <= args.random_mask_token_id < args.vocab_size):
        raise ValueError(f"--random-mask-token-id must be in [0, {args.vocab_size - 1}], got {args.random_mask_token_id}.")

    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    experiment_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_index = None
        if args.resume_local_dir:
            experiment_dir = args.resume_local_dir
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment directory at {experiment_dir}")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        if args.resume_cloud_dir:
            cloud_results_dir = args.resume_cloud_dir
            cloud_checkpoint_dir = os.path.join(cloud_results_dir, "checkpoints")
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Resuming cloud experiment directory at {cloud_checkpoint_dir}")
        else:
            if experiment_index is None:
                experiment_index = len(glob(f"{args.results_dir}/*"))
            time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
            cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        cloud_checkpoint_dir = None

    obj = [experiment_dir, checkpoint_dir, cloud_checkpoint_dir]
    dist.broadcast_object_list(obj, src=0)
    experiment_dir, checkpoint_dir, cloud_checkpoint_dir = obj

    if args.fid_ref is not None and not os.path.isabs(args.fid_sample_dir):
        args.fid_sample_dir = os.path.join(experiment_dir, args.fid_sample_dir)

    loss_writer = None
    eval_writer = None
    if rank == 0:
        loss_writer = JsonlWriter(os.path.join(experiment_dir, "metrics", "train_loss_steps.jsonl"))
        eval_writer = JsonlWriter(os.path.join(experiment_dir, "metrics", "eval_metrics.jsonl")) if args.fid_ref is not None else None
        if not args.no_wandb:
            os.environ["WANDB_DIR"] = experiment_dir
            wandb.init(
                project=args.wandb_project,
                name=(args.wandb_name if args.wandb_name else os.path.basename(experiment_dir)),
                config=vars(args),
            )

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    dropout_p = 0.0 if args.drop_path_rate > 0.0 else args.dropout_p
    latent_size = args.image_size // args.downsample_size
    raw_student = FLASHAR_GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        medusa_attention_num=args.medusa_attention_num,
        vertical_start_layer=args.vertical_start_layer,
        hv_mix=getattr(args, "hv_mix", False),
        hv_mix_init=getattr(args, "hv_mix_init", 0.5),
        hv_gate=getattr(args, "hv_gate", False),
    ).to(device)
    configure_vertical_branch(raw_student, args, logger)
    if getattr(args, "split_loss", False):
        raw_student.split_loss = True
        raw_student.split_loss_lambda = args.split_loss_lambda
        raw_student.col0_boost = args.col0_boost
        logger.info(f"Split loss enabled: lambda={args.split_loss_lambda}, col0_boost={args.col0_boost}")
    logger.info(f"Student GPT Parameters: {sum(param.numel() for param in raw_student.parameters()):,}")

    train_steps = 0
    start_epoch = 0
    steps_per_epoch = int(args.steps_per_epoch)
    args.gpt_ckpt = resolve_resume_ckpt(args, logger if rank == 0 else None)
    resume_ckpt = load_checkpoint(args.gpt_ckpt, map_location="cpu") if args.gpt_ckpt else None
    if resume_ckpt is not None:
        ckpt_state = normalize_state_dict(extract_state_dict(resume_ckpt))
        incompatible = raw_student.load_state_dict(ckpt_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                f"Checkpoint loaded with missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)}. "
                f"missing_sample={incompatible.missing_keys[:8]}, unexpected_sample={incompatible.unexpected_keys[:8]}"
            )
        if isinstance(resume_ckpt, dict) and "steps" in resume_ckpt:
            train_steps = int(resume_ckpt["steps"])
            start_epoch = int(train_steps / max(steps_per_epoch, 1))
        logger.info(f"Resume model weights from checkpoint: {args.gpt_ckpt}; train_steps={train_steps}, start_epoch={start_epoch}")
    else:
        if args.init_ckpt is None:
            args.init_ckpt = args.teacher_ckpt
        if args.init_ckpt:
            teacher_ckpt = load_checkpoint(args.init_ckpt, map_location="cpu")
            teacher_state = normalize_state_dict(extract_state_dict(teacher_ckpt))
            init_student_from_teacher(raw_student, teacher_state, logger)
            del teacher_ckpt, teacher_state

    local_bs = int(args.global_batch_size // dist.get_world_size())
    mask_causal, mask_proximity, mask_union, _delta_indices, removal_indices = build_masks(
        raw_student,
        local_bs,
        device,
        seed=(args.global_seed + rank),
    )
    rng = random.Random(args.global_seed + rank)

    phase1_epochs = int(getattr(args, "phase1_epochs", 0))
    phase1_scope = str(getattr(args, "phase1_train_scope", "full"))
    phase_trainable_cache = {
        "full": None,
        "none": set(),
        "vertical_head": get_phase_trainable_names(raw_student, "vertical_head"),
        "vertical_extra": get_phase_trainable_names(raw_student, "vertical_extra"),
    }

    student = setup_fsdp_sync(raw_student, args, device)
    student.train()
    optimizer = create_optimizer(student, args.weight_decay, args.lr, (args.beta1, args.beta2), rank, logger)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 2 / 3), int(args.epochs * 5 / 6)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            logger.info("Loaded scheduler state from checkpoint.")
        elif start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"Advanced scheduler to start_epoch={start_epoch}; lr={scheduler.get_last_lr()[0]:.6g}.")
        if "optimizer" in resume_ckpt:
            logger.warning(
                "Optimizer state exists in checkpoint but is not loaded by default in FSDP script. "
                "This is weight-continuation, not exact optimizer-state resume."
            )

    teacher = None
    if args.kd_weight > 0:
        assert args.teacher_ckpt is not None, "Teacher ckpt must be provided when kd_weight > 0."
        logger.warning("KD teacher is replicated on every rank in this FSDP script; for XXL this may require substantial memory.")
        teacher = AR_GPT_models[args.teacher_gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=0.0,
            token_dropout_p=0.0,
        ).to(device)
        teacher_ckpt = load_checkpoint(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(normalize_state_dict(extract_state_dict(teacher_ckpt)), strict=False)
        teacher.eval()
        requires_grad(teacher, False)
        del teacher_ckpt
        logger.info("Teacher model loaded for logits distillation.")

    vq_model = None
    if args.fid_ref is not None:
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
        ).to(device)
        vq_ckpt = load_checkpoint(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt)
        vq_model.eval()
        del vq_ckpt
        if rank == 0:
            logger.info("VQ model loaded for FID evaluation.")

    dataset = build_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=local_bs,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    flip_info = "with" if dataset.flip else "without"
    aug_info = 10 if "ten_crop" in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    subset_info = ""
    if getattr(args, "train_max_samples", -1) and args.train_max_samples > 0:
        subset_info = f", subset={args.train_max_samples} (shuffle={getattr(args, 'train_subset_shuffle', False)})"
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) {flip_info} flip augmentation and {aug_info} crop augmentation{subset_info}")

    ptdtype = {
        "none": torch.float32,
        "fp32": torch.float32,
        "tf32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    accum_steps = int(args.gradient_accumulation_steps)
    if accum_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")

    running_loss = running_ce = running_kd = running_ctx_mask = 0.0
    running_gate_collapse = running_gate_entropy = running_gate_h = running_gate_v = 0.0
    log_steps = 0
    start_time = time.time()
    micro_step = 0
    stop_training = False
    active_phase_scope = None
    active_trainable_names = None
    need_gate_stats = bool(getattr(args, "hv_gate", False)) or float(getattr(args, "gate_collapse_weight", 0.0)) > 0

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        desired_phase_scope = phase1_scope if epoch < phase1_epochs else "full"
        if desired_phase_scope != active_phase_scope:
            active_phase_scope = desired_phase_scope
            active_trainable_names = phase_trainable_cache[active_phase_scope]
            log_trainable_scope(raw_student, active_trainable_names, active_phase_scope, logger)
            optimizer.zero_grad(set_to_none=True)

        if teacher is not None and not kd_allowed(args, epoch, train_steps):
            teacher = None
            torch.cuda.empty_cache()
            logger.info(f"KD disabled from epoch={epoch}, step={train_steps}; released teacher model.")

        accum_loss = accum_ce = accum_kd = accum_ctx_mask = 0.0
        accum_gate_collapse = accum_gate_entropy = accum_gate_h = accum_gate_v = 0.0
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
            student_indices, loss_targets, masked_positions, ctx_mask_ratio = apply_random_context_mask(
                z_indices=z_indices,
                mask_prob=float(args.random_mask_prob),
                vocab_size=int(args.vocab_size),
                replace_mode=str(args.random_mask_replace),
                mask_token_id=int(args.random_mask_token_id),
            )
            mask, mask_prob, prox_prob = pick_mask(
                mask_causal,
                mask_proximity,
                mask_union,
                args.mask_schedule,
                train_steps,
                args.mask_anneal_steps,
                rng,
                removal_indices=removal_indices,
            )

            # Optional schedule-controlled HV mixing.
            if getattr(args, "hv_mix", False):
                core = raw_student
                a0 = float(getattr(args, "hv_mix_init", 0.5))
                aT = float(getattr(args, "hv_mix_target", 0.5))
                steps = int(getattr(args, "hv_mix_anneal_steps", 0))
                target = a0 + (aT - a0) * min(1.0, train_steps / steps) if steps > 0 else aT
                blend_steps = int(getattr(args, "hv_mix_blend_steps", 0))
                blend = 1.0 - min(1.0, train_steps / blend_steps) if blend_steps > 0 else 0.0
                if hasattr(core, "hv_mix_target"):
                    core.hv_mix_target = target
                if hasattr(core, "hv_mix_blend"):
                    core.hv_mix_blend = blend

            sync = ((micro_step + 1) % accum_steps == 0)
            context = student.no_sync() if not sync else contextlib.nullcontext()
            with context:
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    if need_gate_stats:
                        student_logits, ce_loss, gate_stats = student(
                            cond_idx=c_indices,
                            idx=student_indices,
                            targets=loss_targets,
                            mask=mask,
                            return_stats=True,
                        )
                    else:
                        student_logits, ce_loss = student(cond_idx=c_indices, idx=student_indices, targets=loss_targets, mask=mask)
                        gate_stats = {}
                    kd_loss = torch.tensor(0.0, device=device)
                    gate_collapse_loss = gate_stats.get("loss_gate_collapse", torch.zeros((), device=device))
                    gate_entropy = gate_stats.get("hv_gate_entropy", torch.ones((), device=device))
                    gate_h = gate_stats.get("hv_gate_h", torch.full((), 0.5, device=device))
                    gate_v = gate_stats.get("hv_gate_v", torch.full((), 0.5, device=device))
                    do_kd = False
                    if teacher is not None and kd_allowed(args, epoch, train_steps):
                        opt_step = train_steps
                        if opt_step >= args.kd_start_step and (args.kd_every <= 1 or (opt_step % args.kd_every == 0)):
                            do_kd = args.kd_prob >= 1.0 or (args.kd_prob > 0.0 and random.Random(args.global_seed + opt_step).random() < args.kd_prob)
                    if do_kd:
                        with torch.no_grad():
                            seq_len = z_indices.size(1) - 1 + teacher.cls_token_num
                            input_pos = torch.arange(seq_len, device=device)
                            teacher_logits, _ = teacher(cond_idx=c_indices, idx=z_indices[:, :-1], targets=None, input_pos=input_pos)
                        t = args.kd_temperature
                        kd_token = F.kl_div(
                            F.log_softmax(student_logits.float() / t, dim=-1),
                            F.softmax(teacher_logits.float() / t, dim=-1),
                            reduction="none",
                        ).sum(dim=-1)
                        if masked_positions is not None:
                            kd_valid = (~masked_positions).to(dtype=kd_token.dtype)
                            kd_loss = (kd_token * kd_valid).sum() / kd_valid.sum().clamp_min(1.0)
                        else:
                            kd_loss = kd_token.mean()
                        kd_loss = kd_loss * (t * t)
                    loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss + args.gate_collapse_weight * gate_collapse_loss
                    loss = loss / accum_steps
                scaler.scale(loss).backward()

            accum_loss += args.ce_weight * ce_loss.item() + args.kd_weight * kd_loss.item() + args.gate_collapse_weight * gate_collapse_loss.item()
            accum_ce += ce_loss.item()
            accum_kd += kd_loss.item()
            accum_ctx_mask += ctx_mask_ratio
            accum_gate_collapse += gate_collapse_loss.item()
            accum_gate_entropy += gate_entropy.item()
            accum_gate_h += gate_h.item()
            accum_gate_v += gate_v.item()
            micro_step += 1

            if sync:
                scaler.unscale_(optimizer)
                mask_nontrainable_grads(student, active_trainable_names)
                if args.max_grad_norm != 0.0:
                    if hasattr(student, "clip_grad_norm_"):
                        student.clip_grad_norm_(args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step_loss_val = accum_loss / accum_steps
                step_ce_val = accum_ce / accum_steps
                step_kd_val = accum_kd / accum_steps if teacher is not None else 0.0
                step_ctx_mask_val = accum_ctx_mask / accum_steps
                step_gate_collapse_val = accum_gate_collapse / accum_steps
                step_gate_entropy_val = accum_gate_entropy / accum_steps
                step_gate_h_val = accum_gate_h / accum_steps
                step_gate_v_val = accum_gate_v / accum_steps

                running_loss += step_loss_val
                running_ce += step_ce_val
                running_kd += step_kd_val
                running_ctx_mask += step_ctx_mask_val
                running_gate_collapse += step_gate_collapse_val
                running_gate_entropy += step_gate_entropy_val
                running_gate_h += step_gate_h_val
                running_gate_v += step_gate_v_val
                accum_loss = accum_ce = accum_kd = accum_ctx_mask = 0.0
                accum_gate_collapse = accum_gate_entropy = accum_gate_h = accum_gate_v = 0.0
                log_steps += 1
                train_steps += 1
                epoch_steps += 1

                if rank == 0 and loss_writer is not None and (train_steps % args.log_loss_every == 0):
                    loss_writer.write({
                        "step": train_steps,
                        "epoch": epoch,
                        "epoch_step": epoch_steps,
                        "loss": step_loss_val,
                        "ce": step_ce_val,
                        "kd": step_kd_val,
                        "ctx_mask": step_ctx_mask_val,
                        "gate_collapse": step_gate_collapse_val,
                        "gate_entropy": step_gate_entropy_val,
                        "hv_gate_h": step_gate_h_val,
                        "hv_gate_v": step_gate_v_val,
                        "lr": scheduler.get_last_lr()[0],
                    })
                    if not args.no_wandb:
                        wandb.log({
                            "train/loss": step_loss_val,
                            "train/ce": step_ce_val,
                            "train/kd": step_kd_val,
                            "train/ctx_mask": step_ctx_mask_val,
                            "train/gate_collapse": step_gate_collapse_val,
                            "train/gate_entropy": step_gate_entropy_val,
                            "train/hv_gate_h": step_gate_h_val,
                            "train/hv_gate_v": step_gate_v_val,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/epoch_step": epoch_steps,
                        }, step=train_steps)

                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / max(end_time - start_time, 1e-6)
                    stats = [
                        torch.tensor(running_loss / log_steps, device=device),
                        torch.tensor(running_ce / log_steps, device=device),
                        torch.tensor(running_kd / log_steps, device=device),
                        torch.tensor(running_ctx_mask / log_steps, device=device),
                        torch.tensor(running_gate_collapse / log_steps, device=device),
                        torch.tensor(running_gate_entropy / log_steps, device=device),
                        torch.tensor(running_gate_h / log_steps, device=device),
                        torch.tensor(running_gate_v / log_steps, device=device),
                    ]
                    for value in stats:
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    avg_loss, avg_ce, avg_kd, avg_ctx_mask, avg_gate_collapse, avg_gate_entropy, avg_gate_h, avg_gate_v = [
                        value.item() / dist.get_world_size() for value in stats
                    ]
                    hv_info = ""
                    if getattr(args, "hv_gate", False):
                        hv_info = f", GateH: {avg_gate_h:.3f}, GateV: {avg_gate_v:.3f}, GateEnt: {avg_gate_entropy:.3f}, GateCol: {avg_gate_collapse:.4f}"
                    elif getattr(args, "hv_mix", False) and hasattr(raw_student, "get_hv_right_weight"):
                        hv_info = f", HVW: {raw_student.get_hv_right_weight():.3f}"
                    prox_info = f", ProxP: {prox_prob:.2f}" if prox_prob is not None else ""
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KD: {avg_kd:.4f}, "
                        f"MaskP: {mask_prob:.2f}, CtxMask: {avg_ctx_mask:.3f}{prox_info}{hv_info}, "
                        f"Steps/Sec: {steps_per_sec:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    running_loss = running_ce = running_kd = running_ctx_mask = 0.0
                    running_gate_collapse = running_gate_entropy = running_gate_h = running_gate_v = 0.0
                    log_steps = 0
                    start_time = time.time()

                if args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                    stop_training = True

                stop_flag = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int32)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop_training = bool(stop_flag.item())
                if stop_training:
                    break

        scheduler.step()

        fid_every = int(getattr(args, "fid_every", 0))
        should_run_fid = (
            args.fid_ref is not None
            and (
                fid_every <= 0
                or ((epoch + 1) % fid_every == 0)
                or (epoch + 1 == args.epochs)
                or (args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps)
            )
        )
        skip_checkpoint_save = False
        if should_run_fid:
            fid_ok = torch.tensor(1, device=device, dtype=torch.int32)
            npz_path = None
            txt_path = None
            metrics = {}
            was_training = student.training
            try:
                sample_dir = os.path.join(args.fid_sample_dir, "latest")
                npz_path, txt_path, metrics = run_fid_eval(
                    args,
                    student,
                    vq_model,
                    device,
                    train_steps,
                    logger,
                    generate,
                    epoch=epoch,
                    sample_dir=sample_dir,
                    keep_last_samples=True,
                    rank=rank,
                    world_size=dist.get_world_size(),
                    barrier=dist.barrier,
                )
            except Exception as e:
                fid_ok.fill_(0)
                logger.exception(f"FID eval failed at epoch={epoch}, step={train_steps}: {e}")
            finally:
                clear_generation_caches(student)
                if was_training:
                    student.train()
                else:
                    student.eval()
                torch.cuda.empty_cache()

            if rank == 0:
                if eval_writer is not None:
                    payload = {
                        "epoch": epoch,
                        "step": train_steps,
                        "npz_path": npz_path,
                        "txt_path": txt_path,
                    }
                    payload.update(metrics)
                    eval_writer.write(payload)
                if not args.no_wandb:
                    wb_payload = {"eval/epoch": epoch}
                    for key, value in metrics.items():
                        wb_payload[f"eval/{key}"] = value
                    wandb.log(wb_payload, step=train_steps)

            dist.all_reduce(fid_ok, op=dist.ReduceOp.MIN)
            if fid_ok.item() == 0:
                if rank == 0:
                    logger.info(f"FID eval failed; --fid-fail-action={args.fid_fail_action}.")
                skip_checkpoint_save = True
                if args.fid_fail_action == "stop":
                    stop_training = True
            dist.barrier()

        if skip_checkpoint_save:
            if rank == 0:
                logger.info("Skipping checkpoint save after failed FID eval to avoid saving an inconsistent FSDP state.")
        else:
            clear_generation_caches(student)
            save_last_checkpoint(args, student, optimizer, scheduler, train_steps, checkpoint_dir, cloud_checkpoint_dir, logger, rank)
        if stop_training:
            if rank == 0 and args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                logger.info(f"Reached --max-steps={args.max_steps}, stopping early.")
            break

    student.eval()
    if rank == 0:
        if loss_writer is not None:
            loss_writer.close()
        if eval_writer is not None:
            eval_writer.close()
        if not args.no_wandb:
            wandb.finish()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help="cloud disk path")
    parser.add_argument("--no-local-save", action="store_true", help="no save checkpoints to local path")
    parser.add_argument("--dataset", type=str, default="imagenet_code")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--train-max-samples", type=int, default=-1)
    parser.add_argument("--train-subset-seed", type=int, default=0)
    parser.add_argument("--train-subset-shuffle", action="store_true")

    # model
    parser.add_argument("--gpt-model", type=str, choices=list(FLASHAR_GPT_models.keys()), default="GPT-XXL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="resume/eval-style checkpoint with a model state")
    parser.add_argument("--auto-resume", action="store_true", help="auto pick latest checkpoint from resume dirs")
    parser.add_argument("--resume-local-dir", type=str, default=None)
    parser.add_argument("--resume-cloud-dir", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--medusa-attention-num", type=int, default=1)
    parser.add_argument("--vertical-start-layer", type=int, default=-1, help="Backbone layer index where the vertical branch starts; <0 uses final output.")
    parser.add_argument("--vertical-start-last-minus-depth", action="store_true", help="Set vertical_start_layer = n_layer - medusa_attention_num.")
    parser.add_argument("--no-compile", action="store_true", default=True)
    parser.add_argument("--results-dir", type=str, default="results")

    # distillation / init
    parser.add_argument("--teacher-gpt-model", type=str, choices=list(AR_GPT_models.keys()), default="GPT-XXL")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="teacher ckpt path for logits distillation")
    parser.add_argument("--init-ckpt", type=str, default=None, help="init student from this ckpt, e.g. pretrained_models/c2i_XXL_384.pt")
    parser.add_argument("--kd-weight", type=float, default=0.0, help="default 0 for FSDP XXL to avoid replicated teacher memory")
    parser.add_argument("--kd-prob", type=float, default=1.0)
    parser.add_argument("--kd-every", type=int, default=1)
    parser.add_argument("--kd-start-step", type=int, default=0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--kd-end-step", type=int, default=-1)
    parser.add_argument("--kd-end-epoch", type=int, default=-1)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--phase1-epochs", type=int, default=0)
    parser.add_argument("--phase1-train-scope", type=str, default="full", choices=["full", "none", "vertical_head", "vertical_extra"])

    # optimization
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-loss-every", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp32", "tf32", "fp16", "bf16"])
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--save-optimizer", action="store_true", help="Also save optimizer/scheduler in last_version.pt; can be very large for XXL.")

    # mask schedule
    parser.add_argument("--mask-schedule", type=str, default="static_proximity", choices=["static_proximity", "shrink", "curriculum"])
    parser.add_argument("--mask-anneal-steps", type=int, default=20000)
    parser.add_argument("--random-mask-prob", type=float, default=0.0)
    parser.add_argument("--random-mask-replace", type=str, default="mask", choices=["mask", "random"])
    parser.add_argument("--random-mask-token-id", type=int, default=0)

    # loss / heads
    parser.add_argument("--split-loss", action="store_true")
    parser.add_argument("--split-loss-lambda", type=float, default=0.5)
    parser.add_argument("--col0-boost", type=float, default=0.0)
    parser.add_argument("--hv-mix", action="store_true")
    parser.add_argument("--hv-mix-init", type=float, default=0.5)
    parser.add_argument("--hv-mix-target", type=float, default=0.5)
    parser.add_argument("--hv-mix-anneal-steps", type=int, default=0)
    parser.add_argument("--hv-mix-blend-steps", type=int, default=0)
    parser.add_argument("--hv-gate", action="store_true")
    parser.add_argument("--gate-collapse-weight", type=float, default=0.0)

    # fid evaluation
    parser.add_argument("--fid-every", type=int, default=0, help="Run FID every N epochs. 0 means every epoch.")
    parser.add_argument("--fid-ref", type=str, default=None)
    parser.add_argument("--fid-num-samples", type=int, default=50000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-sample-dir", type=str, default="samples")
    parser.add_argument("--fid-fail-action", type=str, default="stop", choices=["stop", "skip"])
    parser.add_argument("--fid-skip-evaluator", action="store_true")
    parser.add_argument("--fid-top-k", type=int, default=0)
    parser.add_argument("--fid-top-p", type=float, default=1.0)
    parser.add_argument("--fid-temperature", type=float, default=1.0)
    parser.add_argument("--fid-cfg-scale", type=float, default=2.0)
    parser.add_argument("--fid-cfg-interval", type=float, default=-1)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=384)

    # runtime / logging
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--wandb-project", type=str, default="c2i_flashar_fsdp")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()
    main(args)

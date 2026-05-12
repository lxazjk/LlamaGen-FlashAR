# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import argparse
import inspect
import os
import random
import time
import contextlib
from datetime import timedelta
from copy import deepcopy
from glob import glob

import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models as FLASHAR_GPT_models
from autoregressive.models.generate import generate
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.ar_gpt import GPT_models as AR_GPT_models
from autoregressive.utils.mask import build_masks, pick_mask
from autoregressive.utils.fid_eval import run_fid_eval
from autoregressive.utils.train_logging import JsonlWriter


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


def load_checkpoint(path, map_location="cpu"):
    if path is None:
        return None
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def init_student_from_teacher(student, teacher_state, logger=None):
    student_state = student.state_dict()
    new_state = {}
    loaded = 0
    loaded_numel = 0
    param_keys = set(name for name, _ in student.named_parameters())

    for key, val in teacher_state.items():
        if key in student_state and student_state[key].shape == val.shape:
            new_state[key] = val
            loaded += 1
            if key in param_keys:
                loaded_numel += int(val.numel())

    # Reuse output/norm for medusa heads if present
    if "medusa_output.weight" in student_state and "output.weight" in teacher_state:
        if student_state["medusa_output.weight"].shape == teacher_state["output.weight"].shape:
            new_state["medusa_output.weight"] = teacher_state["output.weight"].clone()
    if "medusa_norm.weight" in student_state and "norm.weight" in teacher_state:
        if student_state["medusa_norm.weight"].shape == teacher_state["norm.weight"].shape:
            new_state["medusa_norm.weight"] = teacher_state["norm.weight"].clone()

    extra_layers = len(student.layers) - student.n_layer
    if extra_layers > 0:
        vertical_start = int(getattr(student, "vertical_start_layer", student.n_layer))
        if vertical_start < student.n_layer:
            teacher_layer_ids = list(range(vertical_start, min(student.n_layer, vertical_start + extra_layers)))
            if len(teacher_layer_ids) < extra_layers:
                teacher_layer_ids.extend([student.n_layer - 1] * (extra_layers - len(teacher_layer_ids)))
        else:
            teacher_layer_ids = [student.n_layer - 1] * extra_layers
        for rel_idx, teacher_layer_id in enumerate(teacher_layer_ids):
            extra_id = student.n_layer + rel_idx
            source_prefix = f"layers.{teacher_layer_id}."
            target_prefix = f"layers.{extra_id}."
            for key, val in teacher_state.items():
                if key.startswith(source_prefix):
                    target_key = target_prefix + key[len(source_prefix):]
                    if target_key in student_state and student_state[target_key].shape == val.shape:
                        new_state[target_key] = val.clone()

    missing, unexpected = student.load_state_dict(new_state, strict=False)
    if logger is not None:
        total_numel = sum(int(p.numel()) for p in student.parameters())
        ratio = (loaded_numel / total_numel) if total_numel > 0 else 0.0
        logger.info(
            f"Loaded {loaded} teacher keys into student. Missing keys: {len(missing)}, unexpected: {len(unexpected)}. "
            f"Param init coverage: {ratio*100:.2f}%"
        )
        if ratio < 0.10:
            logger.warning(
                "Init coverage < 10%: this often means gpt-model/ckpt mismatch (e.g., GPT-B init with GPT-L ckpt), "
                "or different key naming. Training may behave like from-scratch."
            )




def override_optimizer_lr(optimizer, scheduler, lr, logger=None):
    for group in optimizer.param_groups:
        group["lr"] = lr
    if scheduler is not None:
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [lr for _ in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [lr for _ in optimizer.param_groups]
    if logger is not None:
        logger.info(f"Override optimizer lr on resume to {lr:g}")

def configure_vertical_branch(student, args, logger=None):
    requested = int(getattr(args, "vertical_start_layer", -1))
    if getattr(args, "vertical_start_last_minus_depth", False):
        requested = int(student.n_layer) - int(student.medusa_attention_num)
    if requested < 0:
        resolved = int(student.n_layer)
    else:
        resolved = max(0, min(int(student.n_layer), requested))
    student.vertical_start_layer = resolved
    if logger is not None and int(student.medusa_attention_num) > 0:
        if resolved < int(student.n_layer):
            logger.info(
                f"Vertical branch start layer: {resolved} "
                f"(depth={int(student.medusa_attention_num)}, backbone_layers={int(student.n_layer)})"
            )
        else:
            logger.info("Vertical branch start layer: legacy final-backbone output.")


def _parse_step_from_ckpt(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.isdigit():
        return int(name)
    return -1


def resolve_resume_ckpt(args, logger):
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


def atomic_torch_save(obj, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)




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


def apply_random_context_mask(
    z_indices: torch.Tensor,
    mask_prob: float,
    vocab_size: int,
    replace_mode: str,
    mask_token_id: int,
):
    """
    Randomly corrupt a subset of input image tokens to simulate noisy rollout context.

    Returns:
        input_indices: possibly corrupted token ids used as model input.
        targets: training targets with corrupted positions set to ignore_index=-100.
        masked_positions: bool tensor of masked positions, or None if disabled.
        mask_ratio: fraction of masked tokens in this batch.
    """
    if mask_prob <= 0.0:
        return z_indices, z_indices, None, 0.0

    masked_positions = torch.rand(z_indices.shape, device=z_indices.device) < mask_prob
    if not masked_positions.any():
        return z_indices, z_indices, None, 0.0

    # Keep at least one supervised token per sample to avoid empty-denominator edge cases.
    all_masked = masked_positions.all(dim=1)
    if all_masked.any():
        masked_positions[all_masked, 0] = False

    input_indices = z_indices.clone()
    if replace_mode == "random":
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=z_indices.shape,
            device=z_indices.device,
            dtype=z_indices.dtype,
        )
        input_indices = torch.where(masked_positions, random_tokens, input_indices)
    else:
        input_indices[masked_positions] = int(mask_token_id)

    targets = z_indices.clone()
    targets[masked_positions] = -100
    mask_ratio = float(masked_positions.float().mean().item())
    return input_indices, targets, masked_positions, mask_ratio


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.steps_per_epoch > 0, "--steps-per-epoch must be > 0."
    if not (0.0 <= args.random_mask_prob <= 1.0):
        raise ValueError(f"--random-mask-prob must be in [0, 1], got {args.random_mask_prob}.")
    if args.random_mask_replace == "mask" and not (0 <= args.random_mask_token_id < args.vocab_size):
        raise ValueError(
            f"--random-mask-token-id must be in [0, {args.vocab_size - 1}] "
            f"when --random-mask-replace=mask, got {args.random_mask_token_id}."
        )

    # Setup DDP
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup experiment folder
    experiment_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_index = None
        if args.resume_local_dir:
            experiment_dir = args.resume_local_dir
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment directory at {experiment_dir}")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        if args.resume_cloud_dir:
            cloud_results_dir = args.resume_cloud_dir
            cloud_checkpoint_dir = f"{cloud_results_dir}/checkpoints"
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

    # WandB + local metric logs (rank0 only)
    loss_writer = None
    eval_writer = None
    if rank == 0:
        if experiment_dir is not None and not os.path.isabs(args.fid_sample_dir):
            args.fid_sample_dir = os.path.join(experiment_dir, args.fid_sample_dir)
        metrics_dir = os.path.join(experiment_dir, "metrics")
        loss_writer = JsonlWriter(os.path.join(metrics_dir, "train_loss_steps.jsonl"))
        eval_writer = JsonlWriter(os.path.join(metrics_dir, "eval_metrics.jsonl"))

        if not args.no_wandb:
            os.environ["WANDB_DIR"] = experiment_dir
            wandb.init(
                project=args.wandb_project,
                name=(args.wandb_name if args.wandb_name else os.path.basename(experiment_dir)),
                config=vars(args),
            )

    # Use a dedicated Gloo process group for long-running synchronization points
    # (e.g., FID evaluation / checkpointing on rank-0). This avoids NCCL watchdog
    # timeouts on `dist.barrier()` while rank-0 is busy doing heavy CPU/GPU work.
    control_group = None
    try:
        control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))
    except Exception as e:
        if rank == 0:
            logger.info(f"Failed to create Gloo control group, falling back to NCCL barriers: {e}")

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

    def control_broadcast_object(obj, src: int = 0):
        obj_list = [obj if rank == src else None]
        if control_group is not None:
            dist.broadcast_object_list(obj_list, src=src, group=control_group)
        else:
            dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    if args.fid_ref is not None:
        args.fid_sample_dir = control_broadcast_object(args.fid_sample_dir, src=0)

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if rank == 0 and args.random_mask_prob > 0:
        logger.info(
            "Random context masking enabled: "
            f"prob={args.random_mask_prob:.3f}, mode={args.random_mask_replace}, "
            f"mask_token_id={args.random_mask_token_id}"
        )
    if rank == 0:
        if args.fid_ref is not None:
            if args.fid_every and args.fid_every > 0:
                logger.info(f"FID eval interval: every {args.fid_every} epoch(s).")
            else:
                logger.info("FID eval interval: every epoch.")
        if args.ckpt_every and args.ckpt_every > 0:
            logger.info("Note: --ckpt-every is ignored; only last_version.pt is kept (overwritten each epoch).")

    # Setup student model (FlashAR)
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    student = FLASHAR_GPT_models[args.gpt_model](
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
    configure_vertical_branch(student, args, logger)
    logger.info(f"Student GPT Parameters: {sum(p.numel() for p in student.parameters()):,}")
    if getattr(args, "gate_collapse_weight", 0.0) > 0 and not getattr(args, "hv_gate", False):
        logger.info("gate-collapse regularization requested without --hv-gate; the extra loss will stay zero.")

    if args.ema:
        ema = deepcopy(student).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Load init / resume
    train_steps = 0
    start_epoch = 0
    args.gpt_ckpt = resolve_resume_ckpt(args, logger if rank == 0 else None)
    resume_ckpt = load_checkpoint(args.gpt_ckpt, map_location="cpu") if args.gpt_ckpt else None

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
            teacher_state = normalize_state_dict(extract_state_dict(teacher_ckpt))
            init_student_from_teacher(student, teacher_state, logger)
            del teacher_ckpt
        if args.ema:
            update_ema(ema, student, decay=0)

    # Setup optimizer
    optimizer = create_optimizer(student, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 2 / 3), int(args.epochs * 5 / 6)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        if args.override_optimizer_lr:
            override_optimizer_lr(optimizer, scheduler, args.lr, logger)

    # Setup data
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
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    subset_info = ""
    if getattr(args, 'train_max_samples', -1) and args.train_max_samples > 0:
        subset_info = f", subset={args.train_max_samples} (shuffle={getattr(args, 'train_subset_shuffle', False)})"
    logger.info(
        f"Dataset contains {len(dataset):,} images ({args.code_path}) {flip_info} flip augmentation and {aug_info} crop augmentation{subset_info}"
    )
    steps_per_epoch = int(args.steps_per_epoch)
    if resume_ckpt is not None and train_steps > 0:
        start_epoch = int(train_steps / max(steps_per_epoch, 1))

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
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=0.0,
            token_dropout_p=0.0,
        ).to(device)
        teacher_ckpt = load_checkpoint(args.teacher_ckpt, map_location="cpu")
        teacher_state = normalize_state_dict(extract_state_dict(teacher_ckpt))
        teacher.load_state_dict(teacher_state, strict=False)
        teacher.eval()
        requires_grad(teacher, False)
        del teacher_ckpt
        logger.info("Teacher model loaded for logits distillation.")

    # Compile model if requested
    if not args.no_compile:
        logger.info("compiling the student model... (may take several minutes)")
        student = torch.compile(student)

    # Setup masks
    local_bs = int(args.global_batch_size // dist.get_world_size())
    mask_causal, mask_proximity, mask_union, delta_indices, removal_indices = build_masks(
        student._orig_mod if not args.no_compile else student,
        local_bs,
        device,
        seed=(args.global_seed + rank),
    )
    rng = random.Random(args.global_seed + rank)

    # Wrap with DDP
    student = DDP(student.to(device), device_ids=[args.gpu])
    student.train()
    if args.ema:
        ema.eval()

    # Set split-loss attributes on the underlying module
    if getattr(args, "split_loss", False):
        _core = student.module._orig_mod if (not args.no_compile) and hasattr(student.module, "_orig_mod") else student.module
        _core.split_loss = True
        _core.split_loss_lambda = args.split_loss_lambda
        _core.col0_boost = args.col0_boost
        logger.info(f"Split loss enabled: lambda={args.split_loss_lambda}, col0_boost={args.col0_boost}")

    # Setup FID eval (all ranks for distributed sampling)
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

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # Training loop
    log_steps = 0
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_ctx_mask = 0.0
    running_gate_collapse = 0.0
    running_gate_entropy = 0.0
    running_gate_h = 0.0
    running_gate_v = 0.0
    start_time = time.time()
    start_time_all = start_time
    accum_steps = max(1, args.gradient_accumulation_steps)
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    need_gate_stats = bool(getattr(args, "hv_gate", False) or getattr(args, "gate_collapse_weight", 0.0) > 0.0)

    logger.info(f"Training for {args.epochs} epochs...")
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kd = 0.0
        accum_ctx_mask = 0.0
        accum_gate_collapse = 0.0
        accum_gate_entropy = 0.0
        accum_gate_h = 0.0
        accum_gate_v = 0.0
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
            assert z_indices.shape[0] == c_indices.shape[0]
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

            # Optional: schedule-controlled HV mixing.
            if getattr(args, "hv_mix", False):
                core = student.module
                if (not args.no_compile) and hasattr(core, "_orig_mod"):
                    core = core._orig_mod
                # Target schedule: right_w goes from hv_mix_init -> hv_mix_target over hv_mix_anneal_steps.
                # User intent example: make vertical head dominant early => right_w small early.
                a0 = float(getattr(args, "hv_mix_init", 0.5))
                aT = float(getattr(args, "hv_mix_target", 0.5))
                steps = int(getattr(args, "hv_mix_anneal_steps", 0))
                if steps > 0:
                    p = min(1.0, train_steps / steps)
                    target = a0 + (aT - a0) * p
                else:
                    target = aT
                # Blend schedule: start fully following target, then hand over to learnable weight.
                blend_steps = int(getattr(args, "hv_mix_blend_steps", 0))
                if blend_steps > 0:
                    blend = 1.0 - min(1.0, train_steps / blend_steps)
                else:
                    blend = 0.0
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
                        student_logits, ce_loss = student(
                            cond_idx=c_indices,
                            idx=student_indices,
                            targets=loss_targets,
                            mask=mask,
                        )
                        gate_stats = {}
                    kd_loss = torch.tensor(0.0, device=device)
                    gate_collapse_loss = gate_stats.get("loss_gate_collapse", torch.zeros((), device=device))
                    gate_entropy = gate_stats.get("hv_gate_entropy", torch.ones((), device=device))
                    gate_h = gate_stats.get("hv_gate_h", torch.full((), 0.5, device=device))
                    gate_v = gate_stats.get("hv_gate_v", torch.full((), 0.5, device=device))
                    do_kd = False
                    if teacher is not None and args.kd_weight > 0:
                        opt_step = train_steps  # optimizer-step index (same across ranks)
                        if opt_step >= args.kd_start_step and (args.kd_every <= 1 or (opt_step % args.kd_every == 0)):
                            if args.kd_prob >= 1.0:
                                do_kd = True
                            elif args.kd_prob > 0.0:
                                # Deterministic across ranks to avoid stragglers in DDP.
                                do_kd = random.Random(args.global_seed + opt_step).random() < args.kd_prob

                    if do_kd:
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
                        kd_token = F.kl_div(
                            F.log_softmax(student_logits.float() / t, dim=-1),
                            F.softmax(teacher_logits.float() / t, dim=-1),
                            reduction="none",
                        ).sum(dim=-1)
                        if masked_positions is not None:
                            kd_valid = (~masked_positions).to(dtype=kd_token.dtype)
                            kd_denom = kd_valid.sum().clamp_min(1.0)
                            kd_loss = (kd_token * kd_valid).sum() / kd_denom
                        else:
                            kd_loss = kd_token.mean()
                        kd_loss = kd_loss * (t * t)
                    loss = (
                        args.ce_weight * ce_loss
                        + args.kd_weight * kd_loss
                        + args.gate_collapse_weight * gate_collapse_loss
                    )
                    loss = loss / accum_steps
                scaler.scale(loss).backward()

            accum_loss += (
                args.ce_weight * ce_loss.item()
                + args.kd_weight * kd_loss.item()
                + args.gate_collapse_weight * gate_collapse_loss.item()
            )
            accum_ce += ce_loss.item()
            accum_kd += kd_loss.item()
            accum_ctx_mask += ctx_mask_ratio
            accum_gate_collapse += gate_collapse_loss.item()
            accum_gate_entropy += gate_entropy.item()
            accum_gate_h += gate_h.item()
            accum_gate_v += gate_v.item()
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
                step_ctx_mask_val = accum_ctx_mask / accum_steps
                step_gate_collapse_val = accum_gate_collapse / accum_steps
                step_gate_entropy_val = accum_gate_entropy / accum_steps
                step_gate_h_val = accum_gate_h / accum_steps
                step_gate_v_val = accum_gate_v / accum_steps

                # Logging
                running_loss += step_loss_val
                running_ce += step_ce_val
                running_kd += step_kd_val
                running_ctx_mask += step_ctx_mask_val
                running_gate_collapse += step_gate_collapse_val
                running_gate_entropy += step_gate_entropy_val
                running_gate_h += step_gate_h_val
                running_gate_v += step_gate_v_val
                accum_loss = 0.0
                accum_ce = 0.0
                accum_kd = 0.0
                accum_ctx_mask = 0.0
                accum_gate_collapse = 0.0
                accum_gate_entropy = 0.0
                accum_gate_h = 0.0
                accum_gate_v = 0.0
                log_steps += 1
                train_steps += 1
                epoch_steps += 1

                if args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                    stop_training = True

                # Per-step logging (global average)
                step_loss_t = torch.tensor(step_loss_val, device=device)
                step_ce_t = torch.tensor(step_ce_val, device=device)
                step_kd_t = torch.tensor(step_kd_val, device=device)
                step_ctx_mask_t = torch.tensor(step_ctx_mask_val, device=device)
                step_gate_collapse_t = torch.tensor(step_gate_collapse_val, device=device)
                step_gate_entropy_t = torch.tensor(step_gate_entropy_val, device=device)
                step_gate_h_t = torch.tensor(step_gate_h_val, device=device)
                step_gate_v_t = torch.tensor(step_gate_v_val, device=device)
                dist.all_reduce(step_loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_ce_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_kd_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_ctx_mask_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_gate_collapse_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_gate_entropy_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_gate_h_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_gate_v_t, op=dist.ReduceOp.SUM)
                step_loss_avg = step_loss_t.item() / dist.get_world_size()
                step_ce_avg = step_ce_t.item() / dist.get_world_size()
                step_kd_avg = step_kd_t.item() / dist.get_world_size()
                step_ctx_mask_avg = step_ctx_mask_t.item() / dist.get_world_size()
                step_gate_collapse_avg = step_gate_collapse_t.item() / dist.get_world_size()
                step_gate_entropy_avg = step_gate_entropy_t.item() / dist.get_world_size()
                step_gate_h_avg = step_gate_h_t.item() / dist.get_world_size()
                step_gate_v_avg = step_gate_v_t.item() / dist.get_world_size()
                if rank == 0 and loss_writer is not None and (train_steps % args.log_loss_every == 0):
                    loss_writer.write(
                        {
                            "step": train_steps,
                            "epoch": epoch,
                            "epoch_step": epoch_steps,
                            "loss": step_loss_avg,
                            "ce": step_ce_avg,
                            "kd": step_kd_avg,
                            "ctx_mask": step_ctx_mask_avg,
                            "gate_collapse": step_gate_collapse_avg,
                            "gate_entropy": step_gate_entropy_avg,
                            "hv_gate_h": step_gate_h_avg,
                            "hv_gate_v": step_gate_v_avg,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    if not args.no_wandb:
                        wandb.log(
                            {
                                "train/loss": step_loss_avg,
                                "train/ce": step_ce_avg,
                                "train/kd": step_kd_avg,
                                "train/ctx_mask": step_ctx_mask_avg,
                                "train/gate_collapse": step_gate_collapse_avg,
                                "train/gate_entropy": step_gate_entropy_avg,
                                "train/hv_gate_h": step_gate_h_avg,
                                "train/hv_gate_v": step_gate_v_avg,
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                                "train/epoch_step": epoch_steps,
                            },
                            step=train_steps,
                        )

                # Make stop decision consistent across ranks to avoid hanging.
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
                    avg_ctx_mask = torch.tensor(running_ctx_mask / log_steps, device=device)
                    avg_gate_collapse = torch.tensor(running_gate_collapse / log_steps, device=device)
                    avg_gate_entropy = torch.tensor(running_gate_entropy / log_steps, device=device)
                    avg_gate_h = torch.tensor(running_gate_h / log_steps, device=device)
                    avg_gate_v = torch.tensor(running_gate_v / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_kd, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_ctx_mask, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_gate_collapse, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_gate_entropy, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_gate_h, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_gate_v, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    avg_ce = avg_ce.item() / dist.get_world_size()
                    avg_kd = avg_kd.item() / dist.get_world_size()
                    avg_ctx_mask = avg_ctx_mask.item() / dist.get_world_size()
                    avg_gate_collapse = avg_gate_collapse.item() / dist.get_world_size()
                    avg_gate_entropy = avg_gate_entropy.item() / dist.get_world_size()
                    avg_gate_h = avg_gate_h.item() / dist.get_world_size()
                    avg_gate_v = avg_gate_v.item() / dist.get_world_size()
                    hv_info = ""
                    if getattr(args, "hv_gate", False):
                        hv_info = (
                            f", GateH: {avg_gate_h:.3f}, GateV: {avg_gate_v:.3f}, "
                            f"GateEnt: {avg_gate_entropy:.3f}, GateCol: {avg_gate_collapse:.4f}"
                        )
                    elif getattr(args, "hv_mix", False):
                        core = student.module
                        if (not args.no_compile) and hasattr(core, "_orig_mod"):
                            core = core._orig_mod
                        if hasattr(core, "get_hv_right_weight"):
                            hv_info = f", HVW: {core.get_hv_right_weight():.3f}"
                    prox_info = ""
                    if prox_prob is not None:
                        prox_info = f", ProxP: {prox_prob:.2f}"
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KD: {avg_kd:.4f}, "
                        f"MaskP: {mask_prob:.2f}, CtxMask: {avg_ctx_mask:.3f}{prox_info}{hv_info}, "
                        f"Steps/Sec: {steps_per_sec:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    running_ctx_mask = 0.0
                    running_gate_collapse = 0.0
                    running_gate_entropy = 0.0
                    running_gate_h = 0.0
                    running_gate_v = 0.0
                    log_steps = 0
                    start_time = time.time()

            if stop_training:
                break
        scheduler.step()
        # Epoch-end FID evaluation (all ranks participate in distributed sampling).
        fid_every = int(getattr(args, "fid_every", 0))
        should_run_fid = (
            args.fid_ref is not None
            and (
                fid_every <= 0
                or ((epoch + 1) % fid_every == 0)
                or (epoch + 1 == args.epochs)
            )
        )
        if should_run_fid:
            fid_ok = torch.tensor(1, device=device, dtype=torch.int32)
            npz_path = None
            txt_path = None
            metrics = {}
            eval_model = ema if (args.ema and args.fid_use_ema) else (student.module._orig_mod if not args.no_compile else student.module)
            try:
                sample_dir = os.path.join(args.fid_sample_dir, "latest")
                npz_path, txt_path, metrics = run_fid_eval(
                    args,
                    eval_model,
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
                    barrier=control_barrier,
                )
            except Exception as e:
                fid_ok.fill_(0)
                logger.exception(f"FID eval failed at epoch={epoch}, step={train_steps}: {e}")

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
                    for k, v in metrics.items():
                        wb_payload[f"eval/{k}"] = v
                    wandb.log(wb_payload, step=train_steps)

            # Sync FID status across ranks to avoid one-rank crash causing a hang.
            control_all_reduce_min(fid_ok)
            if fid_ok.item() == 0:
                if rank == 0:
                    logger.info(f"FID eval failed; --fid-fail-action={args.fid_fail_action}.")
                if args.fid_fail_action == "stop":
                    stop_training = True
            control_barrier()

        # Save last checkpoint only (overwrite each epoch)
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
                checkpoint_path = f"{checkpoint_dir}/last_version.pt"
                atomic_torch_save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            cloud_checkpoint_path = f"{cloud_checkpoint_dir}/last_version.pt"
            atomic_torch_save(checkpoint, cloud_checkpoint_path)
            logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

        if stop_training:
            if rank == 0 and (args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps):
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
    parser.add_argument("--no-local-save", action='store_true', help="no save checkpoints to local path")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    # dataset subset (for fast convergence / ablation iterations)
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=-1,
        help="If >0, only use first N samples (optionally shuffled) from imagenet_code to speed up experiments.",
    )
    parser.add_argument(
        "--train-subset-seed",
        type=int,
        default=0,
        help="Seed used when --train-max-samples is set and --train-subset-shuffle is enabled.",
    )
    parser.add_argument(
        "--train-subset-shuffle",
        action='store_true',
        help="Shuffle ids before taking first --train-max-samples (deterministic by --train-subset-seed).",
    )

    # model
    parser.add_argument("--gpt-model", type=str, choices=list(FLASHAR_GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="resume checkpoint")
    parser.add_argument("--auto-resume", action='store_true', help="auto pick latest checkpoint from resume dirs")
    parser.add_argument("--resume-local-dir", type=str, default=None, help="existing local experiment dir, e.g., results/005-GPT-L")
    parser.add_argument("--resume-cloud-dir", type=str, default=None, help="existing cloud experiment dir, e.g., ./ckpt/2026-02-23-17-00-55/005-GPT-L")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--medusa-attention-num", type=int, default=1)
    parser.add_argument(
        "--vertical-start-layer",
        type=int,
        default=-1,
        help="Backbone layer index where the vertical branch starts. <0 keeps legacy final-layer branching.",
    )
    parser.add_argument(
        "--vertical-start-last-minus-depth",
        action="store_true",
        help="Set vertical_start_layer = n_layer - medusa_attention_num for vertical-branch ablations.",
    )
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")

    # distillation
    parser.add_argument("--teacher-gpt-model", type=str, choices=list(AR_GPT_models.keys()), default="GPT-L")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="teacher checkpoint path for logits distillation")
    parser.add_argument("--init-ckpt", type=str, default=None, help="init student from this ckpt, default to teacher ckpt")
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--kd-prob",
        type=float,
        default=1.0,
        help="Probability to apply KD on an optimizer step. <1 reduces teacher forward compute.",
    )
    parser.add_argument(
        "--kd-every",
        type=int,
        default=1,
        help="Apply KD every N optimizer steps (1 means every step).",
    )
    parser.add_argument(
        "--kd-start-step",
        type=int,
        default=0,
        help="Start applying KD from this optimizer step (warmup without KD).",
    )
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)

    # optimization
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--override-optimizer-lr", action="store_true",
                        help="After loading a resume optimizer/scheduler state, force all optimizer groups to --lr.")
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-loss-every", type=int, default=500, help="Log train/loss/ce/kd/lr/epoch every N steps")
    parser.add_argument("--ckpt-every", type=int, default=10000, help="ignored; only last checkpoint is kept")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    # mask schedule
    parser.add_argument(
        "--mask-schedule",
        type=str,
        default="static_proximity",
        choices=["static_proximity", "shrink", "curriculum"],
    )
    parser.add_argument("--mask-anneal-steps", type=int, default=20000)
    parser.add_argument(
        "--random-mask-prob",
        type=float,
        default=0.0,
        help="Randomly replace this fraction of image input tokens to simulate noisy rollout context.",
    )
    parser.add_argument(
        "--random-mask-replace",
        type=str,
        default="mask",
        choices=["mask", "random"],
        help="'mask': replace with --random-mask-token-id; 'random': replace with random visual token ids.",
    )
    parser.add_argument(
        "--random-mask-token-id",
        type=int,
        default=0,
        help="Replacement token id when --random-mask-replace=mask.",
    )

    # Split loss: separate CE for each head
    parser.add_argument("--split-loss", action='store_true', help="compute separate CE loss for R and B heads")
    parser.add_argument("--split-loss-lambda", type=float, default=0.5, help="weight for R head loss; B gets (1-lambda)")
    parser.add_argument("--col0-boost", type=float, default=0.0, help="extra loss weight for B head on column-0 tokens")

    # Right/Below logits mixing (learnable alpha)
    parser.add_argument("--hv-mix", action='store_true', help="enable learnable mixing between right/below logits")
    parser.add_argument("--hv-mix-init", type=float, default=0.5, help="initial right(head) weight in [0,1]")
    parser.add_argument("--hv-mix-target", type=float, default=0.5, help="target right(head) weight in [0,1]")
    parser.add_argument(
        "--hv-mix-anneal-steps",
        type=int,
        default=0,
        help="If >0, linearly anneal target from --hv-mix-init to --hv-mix-target over this many steps.",
    )
    parser.add_argument(
        "--hv-mix-blend-steps",
        type=int,
        default=0,
        help="If >0, blend target->learned weight over this many steps (starts 1.0 then decays to 0.0).",
    )
    parser.add_argument("--hv-gate", action="store_true",
                        help="Per-position gate MLP: replaces scalar hv_mix_logit with MLP(cat(h_R,h_B))->sigmoid.")
    parser.add_argument(
        "--gate-collapse-weight",
        type=float,
        default=0.0,
        help="Weight for hv-gate anti-collapse regularization (1 - mean binary entropy over interior gates).",
    )

    # runtime control
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (i.e., after gradient accumulation sync). -1 means no limit.",
    )

    # fid evaluation
    parser.add_argument("--fid-every", type=int, default=0, help="Run FID every N epochs. 0 means every epoch.")
    parser.add_argument("--fid-ref", type=str, default=None)
    parser.add_argument("--fid-num-samples", type=int, default=50000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-sample-dir", type=str, default="samples")
    parser.add_argument(
        "--fid-fail-action",
        type=str,
        default="stop",
        choices=["stop", "skip"],
        help="What to do if rank0 FID evaluation raises an exception. 'stop' stops training; 'skip' logs and continues.",
    )
    parser.add_argument("--fid-use-ema", action='store_true')
    parser.add_argument(
        "--fid-skip-evaluator",
        action='store_true',
        help="Only generate FID sample .npz during training; do not run evaluations/c2i/evaluator.py (run it manually later).",
    )
    parser.add_argument("--fid-top-k", type=int, default=0)
    parser.add_argument("--fid-top-p", type=float, default=1.0)
    parser.add_argument("--fid-temperature", type=float, default=1.0)
    parser.add_argument("--fid-cfg-scale", type=float, default=2.0)
    parser.add_argument("--fid-cfg-interval", type=float, default=-1)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)

    # wandb
    parser.add_argument("--wandb-project", type=str, default="c2i_flashar")
    parser.add_argument("--wandb-name", type=str, default="", help="Optional wandb run name override")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()
    main(args)

"""
Checkpoint utilities for training.
"""
import os
from glob import glob

import torch


def extract_state_dict(ckpt):
    """Extract state dict from checkpoint."""
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
    """Normalize state dict by removing module/model prefix."""
    if not isinstance(state, dict) or len(state) == 0:
        return state
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_checkpoint(path, map_location="cpu"):
    """Load checkpoint from path."""
    if path is None:
        return None
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def init_student_from_teacher(student, teacher_state, logger=None):
    """Initialize student model from teacher state dict."""
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


def _parse_step_from_ckpt(path):
    """Parse step number from checkpoint filename."""
    name = os.path.splitext(os.path.basename(path))[0]
    if name.isdigit():
        return int(name)
    return -1


def resolve_resume_ckpt(args, logger):
    """Resolve checkpoint path for resume training."""
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

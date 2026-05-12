import os
import math
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _parse_eval_txt(txt_path: Optional[str]) -> Dict[str, float]:
    if txt_path is None or not os.path.isfile(txt_path):
        return {}
    metrics = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def clear_generation_caches(model) -> None:
    """Drop temporary autoregressive generation caches before checkpointing/training."""
    core = getattr(model, "module", model)
    modules = list(core.modules()) if hasattr(core, "modules") else [core]
    for layer in getattr(core, "layers", []):
        attention = getattr(layer, "attention", None)
        if attention is not None:
            modules.append(attention)
    for module in modules:
        if hasattr(module, "kv_cache"):
            module.kv_cache = None
        if hasattr(module, "proximity_mask"):
            module.proximity_mask = None


def create_npz_from_samples(sample_dir: str, num: int) -> str:
    from PIL import Image
    import numpy as np

    samples = []
    for i in range(num):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    return npz_path


@torch.no_grad()
def run_fid_eval(
    args,
    model,
    vq_model,
    device,
    step: int,
    logger,
    generate_fn,
    *,
    epoch: Optional[int] = None,
    sample_dir: Optional[str] = None,
    keep_last_samples: bool = True,
    rank: int = 0,
    world_size: int = 1,
    barrier=None,
) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    if args.fid_ref is None:
        logger.info("FID eval skipped: --fid-ref not provided.")
        return None, None, {}

    if sample_dir is None:
        sample_dir = os.path.join(args.fid_sample_dir, "latest")
    os.makedirs(args.fid_sample_dir, exist_ok=True)
    if rank == 0 and keep_last_samples:
        _clean_dir(sample_dir)
    elif rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
    if barrier is not None:
        barrier()
    os.makedirs(sample_dir, exist_ok=True)
    if barrier is not None:
        barrier()

    model.eval()
    vq_model.eval()

    latent_size = args.image_size // args.downsample_size
    per_rank = (args.fid_num_samples + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, args.fid_num_samples)
    local_num = max(0, end - start)

    total = 0
    num_iters = math.ceil(local_num / args.fid_batch_size) if local_num > 0 else 0
    for _ in range(num_iters):
        n = min(args.fid_batch_size, local_num - total)
        if n <= 0:
            break
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [n, args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate_fn(
            model,
            c_indices,
            latent_size ** 2,
            cfg_scale=args.fid_cfg_scale,
            cfg_interval=args.fid_cfg_interval,
            temperature=args.fid_temperature,
            top_k=args.fid_top_k,
            top_p=args.fid_top_p,
            sample_logits=True,
        )
        samples = vq_model.decode_code(index_sample, qzshape)
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1)
        samples = samples.to("cpu", dtype=torch.uint8).numpy()

        from PIL import Image
        for i, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{sample_dir}/{start + total + i:06d}.png")
        total += n

    if barrier is not None:
        barrier()

    npz_path = None
    if rank == 0:
        npz_path = create_npz_from_samples(sample_dir, args.fid_num_samples)
        logger.info(f"Saved FID samples to {npz_path}")

    txt_path = None
    metrics = {}
    if rank == 0:
        if getattr(args, "fid_skip_evaluator", False):
            logger.info("Skipping FID evaluator (--fid-skip-evaluator enabled).")
        else:
            evaluator_path = os.path.join(_repo_root(), "evaluations", "c2i", "evaluator.py")
            cmd = ["python3", evaluator_path, args.fid_ref, npz_path]
            logger.info(f"Running FID evaluator: {' '.join(cmd)}")
            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                subprocess.run(cmd, check=False, cwd=_repo_root(), env=env)
                txt_path = npz_path.replace(".npz", ".txt")
                metrics = _parse_eval_txt(txt_path)
            except Exception as e:
                logger.info(f"FID evaluator failed: {e}")

    clear_generation_caches(model)
    model.train()

    if rank == 0 and epoch is not None:
        logger.info(f"FID eval done for epoch={epoch}, step={step}.")

    return npz_path, txt_path, metrics

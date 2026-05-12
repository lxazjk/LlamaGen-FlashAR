#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoregressive.models.gpt import GPT_models
from autoregressive.utils.mask import build_masks


def extract_model_weight(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def ns_get(ns, key, default=None):
    return getattr(ns, key, default) if ns is not None else default


def load_codes(code_path, image_size, num_samples, crop_idx, stride):
    code_dir = Path(code_path) / f"imagenet{image_size}_codes"
    label_dir = Path(code_path) / f"imagenet{image_size}_labels"
    max_count = 1281167
    if stride <= 0:
        stride = max(1, max_count // max(1, num_samples))
    indices = [(i * stride) % max_count for i in range(num_samples)]
    codes, labels, used = [], [], []
    for idx in indices:
        code_file = code_dir / f"{idx}.npy"
        label_file = label_dir / f"{idx}.npy"
        if not code_file.exists() or not label_file.exists():
            continue
        code = np.load(code_file)
        label = np.load(label_file)
        if code.ndim == 3:
            crop = crop_idx if crop_idx >= 0 else 0
            crop = min(crop, code.shape[1] - 1)
            code = code[:, crop, :]
        code = code.reshape(-1).astype(np.int64)
        codes.append(code)
        labels.append(int(label.reshape(-1)[0]))
        used.append(idx)
    if not codes:
        raise RuntimeError(f"No codes loaded from {code_dir}")
    return np.stack(codes), np.asarray(labels, dtype=np.int64), used


@torch.no_grad()
def gate_maps_for_batch(model, idx, cond_idx, mask):
    device = idx.device
    cond_embeddings = model.cls_embedding(cond_idx, train=False)[:, : model.cls_token_num]
    token_embeddings = model.tok_embeddings(idx)
    token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
    h = model.tok_dropout(token_embeddings)

    freqs_cis = model.freqs_cis[: token_embeddings.shape[1]].to(device)
    h, medusa_h = model._run_backbone_and_vertical(h, freqs_cis, None, mask[:, None].to(device))

    h = model.norm(h)
    logits_h = model.output(h).float()
    medusa_h = model.medusa_norm(medusa_h)
    logits_v = model.medusa_output(medusa_h).float()

    logits_h = logits_h[:, model.cls_token_num - 1 :]
    logits_v = logits_v[:, model.cls_token_num - 1 :]

    batch_size, _, vocab_size = logits_h.shape
    grid_size = model.grid_size
    h_corner = logits_h[:, 0, :]
    v_corner = logits_v[:, 0, :]
    corner_gate = model._hv_gate(h_corner, v_corner).reshape(batch_size)

    logits_h = logits_h[:, 1:, :].reshape(batch_size, grid_size, grid_size, vocab_size).roll(shifts=1, dims=2)
    logits_v = logits_v[:, 1:, :].reshape(batch_size, grid_size, grid_size, vocab_size).roll(shifts=1, dims=1)

    gate_map = torch.empty(batch_size, grid_size, grid_size, device=device, dtype=torch.float32)
    gate_map[:, 0, 0] = corner_gate.float()
    gate_map[:, 0, 1:] = 1.0  # top row is horizontal/right head only in the mixer
    gate_map[:, 1:, 0] = 0.0  # first column is vertical/below head only in the mixer
    if model.hv_gate_mlp is not None and grid_size > 1:
        interior_h = logits_h[:, 1:, 1:, :].reshape(batch_size, -1, vocab_size)
        interior_v = logits_v[:, 1:, 1:, :].reshape(batch_size, -1, vocab_size)
        interior_gate = model._hv_gate(interior_h, interior_v).reshape(batch_size, grid_size - 1, grid_size - 1)
        gate_map[:, 1:, 1:] = interior_gate.float()
    else:
        gate_map[:, 1:, 1:] = 0.5
    return gate_map


def save_heatmap(array, path, title):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    im = ax.imshow(array, vmin=0.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("latent column")
    ax.set_ylabel("latent row")
    ax.set_xticks(range(array.shape[1]))
    ax.set_yticks(range(array.shape[0]))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("horizontal/right gate weight; 0=vertical, 1=horizontal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sample_grid(maps, path, count=8):
    count = min(count, maps.shape[0])
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= count:
            continue
        im = ax.imshow(maps[i], vmin=0.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"sample {i}")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("horizontal/right gate weight")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--code-path", default="./imagenet_code_c2i_flip_ten_crop")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gpt-model", default="GPT-XL")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--out-dir", default="results/gate_heatmaps")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if device.type == "cpu":
        dtype = torch.float32

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    model_weight = extract_model_weight(checkpoint)
    has_gate = any("hv_gate_mlp" in key for key in model_weight.keys())
    if not has_gate:
        raise RuntimeError("Checkpoint does not contain hv_gate_mlp weights")

    latent_size = args.image_size // int(ns_get(ckpt_args, "downsample_size", 16))
    model = GPT_models[args.gpt_model](
        vocab_size=int(ns_get(ckpt_args, "vocab_size", 16384)),
        block_size=latent_size ** 2,
        num_classes=int(ns_get(ckpt_args, "num_classes", 1000)),
        cls_token_num=int(ns_get(ckpt_args, "cls_token_num", 1)),
        model_type=str(ns_get(ckpt_args, "gpt_type", "c2i")),
        hv_mix=bool(ns_get(ckpt_args, "hv_mix", False)),
        hv_gate=True,
        medusa_attention_num=int(ns_get(ckpt_args, "medusa_attention_num", 1)),
        vertical_start_layer=int(ns_get(ckpt_args, "vertical_start_layer", -1)),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    codes, labels, used_indices = load_codes(args.code_path, args.image_size, args.num_samples, args.crop_idx, args.stride)
    masks = build_masks(model, args.batch_size, device, seed=0)[1]

    all_maps = []
    for start in range(0, len(codes), args.batch_size):
        code_batch = torch.from_numpy(codes[start : start + args.batch_size]).to(device)
        label_batch = torch.from_numpy(labels[start : start + args.batch_size]).to(device)
        mask = masks[: code_batch.shape[0]]
        maps = gate_maps_for_batch(model, code_batch, label_batch, mask)
        all_maps.append(maps.cpu().numpy())
    all_maps = np.concatenate(all_maps, axis=0)
    avg_map = all_maps.mean(axis=0)
    std_map = all_maps.std(axis=0)
    interior = avg_map[1:, 1:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).parts[-4] if len(Path(args.ckpt).parts) >= 4 else Path(args.ckpt).stem
    avg_png = out_dir / f"{tag}_avg_gate_heatmap.png"
    std_png = out_dir / f"{tag}_std_gate_heatmap.png"
    grid_png = out_dir / f"{tag}_sample_gate_heatmaps.png"
    npz_path = out_dir / f"{tag}_gate_maps.npz"
    json_path = out_dir / f"{tag}_summary.json"

    save_heatmap(avg_map, avg_png, f"{tag}: avg gate map over {len(all_maps)} ImageNet-{args.image_size} codes")
    save_heatmap(std_map, std_png, f"{tag}: gate std map over {len(all_maps)} ImageNet-{args.image_size} codes")
    save_sample_grid(all_maps, grid_png)
    np.savez_compressed(npz_path, avg=avg_map, std=std_map, maps=all_maps, labels=labels, indices=np.asarray(used_indices))

    summary = {
        "ckpt": args.ckpt,
        "num_samples": int(len(all_maps)),
        "image_size": args.image_size,
        "grid_size": int(avg_map.shape[0]),
        "mean_all_positions": float(avg_map.mean()),
        "mean_interior_gate_h": float(interior.mean()),
        "mean_interior_gate_v": float(1.0 - interior.mean()),
        "min_interior_gate_h": float(interior.min()),
        "max_interior_gate_h": float(interior.max()),
        "avg_png": str(avg_png),
        "std_png": str(std_png),
        "sample_grid_png": str(grid_png),
        "npz": str(npz_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

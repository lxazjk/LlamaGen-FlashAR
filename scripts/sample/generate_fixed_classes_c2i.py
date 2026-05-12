import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from autoregressive.models.generate import generate
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models


def load_gpt_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        return checkpoint["module"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def parse_class_specs(specs):
    result = []
    for spec in specs:
        parts = spec.split(":", 1)
        if len(parts) == 1:
            class_id = int(parts[0])
            name = str(class_id)
        else:
            class_id = int(parts[0])
            name = parts[1].strip().replace(" ", "_").replace("/", "_")
        result.append((class_id, name))
    return result


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model.eval()
    vq_checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_checkpoint["model"])
    del vq_checkpoint

    gpt_checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    gpt_state = load_gpt_state(gpt_checkpoint)
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type="c2i",
    ).to(device=device, dtype=precision)
    missing, unexpected = gpt_model.load_state_dict(gpt_state, strict=False)
    print(f"loaded {args.gpt_model}: missing={len(missing)} unexpected={len(unexpected)}")
    gpt_model.eval()
    del gpt_checkpoint

    class_specs = parse_class_specs(args.class_spec)
    manifest = []
    for class_id, class_name in class_specs:
        class_dir = out_root / f"{class_id:03d}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        generated = 0
        batch_id = 0
        while generated < args.samples_per_class:
            batch_size = min(args.batch_size, args.samples_per_class - generated)
            seed = args.seed + class_id * 100000 + batch_id
            torch.manual_seed(seed)
            c_indices = torch.full((batch_size,), class_id, device=device, dtype=torch.long)
            qzshape = [batch_size, args.codebook_embed_dim, latent_size, latent_size]
            index_sample = generate(
                gpt_model,
                c_indices,
                latent_size ** 2,
                cfg_scale=args.cfg_scale,
                cfg_interval=args.cfg_interval,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
            )
            samples = vq_model.decode_code(index_sample, qzshape)
            for offset in range(batch_size):
                out_path = class_dir / f"{generated + offset:03d}.png"
                save_image(samples[offset], out_path, normalize=True, value_range=(-1, 1))
                saved.append(out_path)
                manifest.append(f"{class_id}\t{class_name}\t{out_path}")
            generated += batch_size
            batch_id += 1
            print(f"{class_id:03d} {class_name}: {generated}/{args.samples_per_class}", flush=True)

        grid_path = out_root / f"grid_{class_id:03d}_{class_name}.png"
        grid_batch = []
        for path in saved:
            image = Image.open(path).convert("RGB")
            tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
            grid_batch.append(tensor)
        save_image(torch.stack(grid_batch), grid_path, nrow=args.grid_nrow)
        print(f"saved grid: {grid_path}")

    (out_root / "manifest.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, default="GPT-XL", choices=list(GPT_models.keys()))
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-model", type=str, default="VQ-16", choices=list(VQ_models.keys()))
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=384, choices=[256, 384, 512])
    parser.add_argument("--downsample-size", type=int, default=16, choices=[8, 16])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--grid-nrow", type=int, default=4)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--class-spec", nargs="+", required=True, help="Format: class_id:name")
    main(parser.parse_args())

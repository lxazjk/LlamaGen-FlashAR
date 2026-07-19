> **🔦 FlashAR 系列** — [📄 项目主页](https://lxazjk.github.io/FlashAR/) · [论文](https://arxiv.org/abs/2605.09430) · [⚡ Emu3.5-FlashAR(主实现)](https://github.com/lxazjk/Emu3.5-FlashAR) · **LlamaGen-FlashAR(👈 你在这)** · [项目主页仓库](https://github.com/lxazjk/FlashAR)
>
> 本仓库是 FlashAR 在 **LlamaGen / ImageNet** 上的实现;主页与旗舰实现见 [Emu3.5-FlashAR](https://github.com/lxazjk/Emu3.5-FlashAR)。

<div align="center">

# FlashAR

### Efficient Diagonal Decoding for Autoregressive Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2605.09430-b31b1b.svg)](https://arxiv.org/abs/2605.09430)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://lxazjk.github.io/FlashAR/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

This repository contains the standalone **FlashAR image-generation** code extracted
from the previous image workspace. It keeps the ImageNet-style
training, sampling, tokenization, and evaluation pipeline while ignoring local
checkpoints, datasets, generated samples, logs, and upload workdirs.

It provides the code needed to:

- pre-extract ImageNet visual tokens for class-conditional training;
- train FlashAR class-conditional image models with DDP or FSDP;
- initialize/post-train from autoregressive teacher checkpoints;
- sample ImageNet images from FlashAR checkpoints;
- evaluate generated samples with FID/NPZ reference files;
- compare regular raster-scan AR behavior with FlashAR diagonal decoding.

## Overview

FlashAR accelerates a raster-scan autoregressive image generator by adding
vertical prediction capacity and using diagonal token-generation structure. For
an `H x W` image-token grid, decoding can proceed over anti-diagonals instead of
pure raster order, reducing the serial image-token path from `H * W` positions to
`H + W - 1` diagonal steps.

In this image-code implementation, the core FlashAR entry points are:

- `autoregressive/train/train_c2i_flashar.py` for C2I training;
- `autoregressive/train/train_c2i_flashar_fsdp.py` for multi-GPU FSDP training;
- `autoregressive/train/train_t2i_flashar.py` for T2I experiments;
- `scripts/eval/eval_flashar_c2i_fid.py` for ImageNet FID evaluation;
- `train_flashar_lora.py` for LoRA-style FlashAR experiments.

## Repository Layout

```text
.
├── autoregressive/                  # FlashAR model, training, sampling utilities
│   ├── models/                      # GPT/FlashAR model definitions and generation code
│   ├── sample/                      # C2I/T2I sampling entry points
│   ├── train/                       # DDP/FSDP training entry points
│   └── utils/                       # Masking, FID, checkpoint, logging helpers
├── dataset/                         # Dataset builders and ImageNet/webdataset loaders
├── evaluations/                     # C2I/T2I evaluation scripts and metadata
├── language/                        # T5 feature extraction helpers for T2I
├── scripts/                         # Launchers for extraction, training, sampling, evaluation
├── tokenizer/                       # VQ/VAE/VQGAN tokenizer implementations
├── tools/                           # Conversion, upload, visualization, and dataset tools
├── utils/                           # Distributed, logging, EMA, and general utilities
├── train_flashar_lora.py            # LoRA FlashAR training experiment
├── requirements.txt                 # Python dependency list
└── README.md
```

## Installation

Create a clean Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training scripts are designed for Linux + PyTorch distributed training. The
existing launchers assume multi-GPU machines and can be edited for your local
CUDA device count.

Runtime assumptions:

- Linux with Python >= 3.7 and PyTorch >= 2.1;
- A100-class GPUs are recommended for the provided multi-GPU launchers;
- update `nnodes`, `nproc_per_node`, `node_rank`, `master_addr`, and
  `master_port` in shell launchers when running across different machines;
- install the extra metric dependencies described in `evaluations/c2i/README.md`
  before running standalone C2I evaluation scripts.

## Data and Weights

Large artifacts are intentionally ignored by Git. Keep them under the following
local paths, or pass custom paths through script arguments:

```text
pretrained_models/
├── vq_ds16_c2i.pt
└── VIRTUAL_imagenet256_labeled.npz

imagenet_raw/
└── train/

imagenet_code_c2i_flip_ten_crop/
└── ... pre-extracted visual token files ...

cloud_ckpt*/                         # training checkpoints
samples*/                            # generated samples
results/                             # local experiment outputs
logs/                                # local logs
```

## Quick Start: ImageNet C2I

### Step 1: Pre-extract Image Codes

Use the VQ tokenizer to convert ImageNet images into discrete visual tokens:

```bash
bash scripts/autoregressive/extract_codes_c2i.sh \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --data-path ./imagenet_raw/train \
  --code-path ./imagenet_code_c2i_flip_ten_crop \
  --ten-crop \
  --crop-range 1.1 \
  --image-size 256
```

The training scripts expect `--code-path` to point to this generated token
folder.

### Step 2: Train FlashAR

Run the default C2I launcher:

```bash
bash scripts/train/train_flashar_c2i_from_scratch.sh
```

Or launch the training entry directly:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  autoregressive/train/train_c2i_flashar.py \
  --code-path ./imagenet_code_c2i_flip_ten_crop \
  --cloud-save-path ./cloud_ckpt \
  --dataset imagenet_code \
  --image-size 256 \
  --downsample-size 16 \
  --gpt-model GPT-L \
  --gpt-type c2i \
  --epochs 30 \
  --steps-per-epoch 2500 \
  --global-batch-size 512 \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
  --wandb-project c2i_flashar \
  --no-compile
```

Useful training options:

| Option | Description |
| --- | --- |
| `--gpt-model` | Model size, e.g. `GPT-B`, `GPT-L`, `GPT-XL`, `GPT-XXL`. |
| `--teacher-ckpt` | Optional autoregressive teacher checkpoint for distillation/init. |
| `--init-ckpt` | Optional student initialization checkpoint. |
| `--kd-weight` | Distillation loss weight. Use `0` for from-scratch CE-only training. |
| `--hv-gate` | Enable learnable horizontal/vertical fusion gate. |
| `--hv-mix` | Enable learnable right/below logit mixing. |
| `--fid-ref` | Reference NPZ used by periodic FID evaluation. |
| `--no-wandb` | Disable W&B logging. |

### Step 3: FSDP Training

For larger models, use the FSDP entry:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  autoregressive/train/train_c2i_flashar_fsdp.py \
  --code-path ./imagenet_code_c2i_flip_ten_crop \
  --cloud-save-path ./cloud_ckpt_fsdp \
  --dataset imagenet_code \
  --image-size 384 \
  --downsample-size 16 \
  --gpt-model GPT-XXL \
  --gpt-type c2i \
  --global-batch-size 512 \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
  --wandb-project c2i_flashar_fsdp
```

## Text-Conditional Experiments

The repository also keeps the original T2I utilities. For webdataset-style text
image data, first create the metadata JSON and then launch T2I training:

```bash
python scripts/analyze_tar.py

bash scripts/autoregressive/train_t2i.sh \
  --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
  --cloud-save-path ./cloud_ckpt_t2i \
  --no-local-save \
  --gpt-model GPT-XL \
  --image-size 256 \
  --epochs 60 \
  --no-compile \
  --data-path /path/to/data.json
```

For second-stage high-resolution training, resume from the first-stage
checkpoint and set `--stage1-ckpt` plus the target `--image-size`.

## Sampling

Generate class-conditional samples from a trained checkpoint:

```bash
bash scripts/autoregressive/sample_c2i.sh \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt ./cloud_ckpt/path/to/checkpoint.pt \
  --gpt-model GPT-L \
  --image-size 256 \
  --image-size-eval 256 \
  --cfg-scale 2.0
```

The launcher uses `autoregressive/sample/sample_c2i_ddp.py` and can be edited to
match the desired GPU count.

## Evaluation

Run FlashAR FID evaluation with a reference NPZ:

```bash
python scripts/eval/eval_flashar_c2i_fid.py \
  --gpt-ckpt ./cloud_ckpt/path/to/checkpoint.pt \
  --gpt-model GPT-L \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
  --sample-dir ./samples_flashar_eval \
  --num-samples 50000 \
  --batch-size 64 \
  --cfg-scale 2.0 \
  --precision bf16
```

The evaluator writes generated samples and metrics under `--sample-dir`.

## Git Hygiene

This repo is intended to track only source code, documentation, and lightweight
configuration/metadata files. The `.gitignore` excludes:

- ImageNet tar files and extracted datasets;
- pre-extracted token-code folders;
- pretrained weights and training checkpoints;
- generated samples, result folders, and logs;
- ModelScope upload workdirs and local scratch folders.

Before pushing, check what would be committed:

```bash
git status --short
git diff --cached --stat
```

## Citation

```bibtex
@article{zhou2026flashar,
  title={FlashAR: Efficient Post-Training Acceleration for Autoregressive Image Generation},
  author={Zhou, Junkang and He, Yefei and Chen, Feng and Wang, Weijie and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2605.09430},
  year={2026}
}
```

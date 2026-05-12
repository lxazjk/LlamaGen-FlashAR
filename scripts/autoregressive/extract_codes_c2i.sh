#!/usr/bin/env bash
set -x

DATA_PATH="./imagenet_raw/train"
SAVE_PATH="./imagenet_code_c2i_flip_ten_crop"
VQ_CKPT="./pretrained_models/vq_ds16_c2i.pt"

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12335 \
autoregressive/train/extract_codes_c2i.py \
--vq-ckpt $VQ_CKPT \
--data-path $DATA_PATH \
--code-path $SAVE_PATH \
--ten-crop \
--crop-range 1.1 \
--image-size 256 \
"$@"
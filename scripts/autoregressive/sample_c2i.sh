#!/usr/bin/env bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--gpt-ckpt ./pretrained_models/flashar_c2i.pt \
--gpt-model GPT-L \
--image-size 256 \
--image-size-eval 256 \
--cfg-scale 2.0 \
"$@"

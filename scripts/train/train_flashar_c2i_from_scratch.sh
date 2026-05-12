#!/usr/bin/env bash
set -x

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    autoregressive/train/train_c2i_flashar.py \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --cloud-save-path ./cloud_ckpt \
    --dataset imagenet_code --image-size 256 --downsample-size 16 \
    --gpt-model GPT-L --gpt-type c2i \
    --epochs 30 --steps-per-epoch 2500 \
    --global-batch-size 512 --num-workers 24 --log-every 100 --log-loss-every 500 \
    --kd-weight 0 \
    --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
    --fid-num-samples 50000 --fid-batch-size 64 \
    --fid-sample-dir ./samples_c2i \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --wandb-project c2i_flashar \
    --wandb-name train_from_scratch_30x2500_gptl_bs512 \
    --no-compile
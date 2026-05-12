#!/usr/bin/env bash
set -x


if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_NODELIST:-}" ]; then
  MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)}
  NNODES=${NNODES:-$SLURM_JOB_NUM_NODES}
  NODE_RANK=${NODE_RANK:-$SLURM_NODEID}
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12355}

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  autoregressive/train/train_t2i_webdata.py \
  "$@"

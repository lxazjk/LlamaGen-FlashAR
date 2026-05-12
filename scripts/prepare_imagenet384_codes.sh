#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TAR_PATH="${TAR_PATH:-./ILSVRC2012_img_train.tar}"
RAW_ROOT="${RAW_ROOT:-./imagenet_raw}"
TRAIN_DIR="${TRAIN_DIR:-${RAW_ROOT}/train}"
CLASS_TAR_DIR="${CLASS_TAR_DIR:-${RAW_ROOT}/train_tars}"
CODE_PATH="${CODE_PATH:-./imagenet_code_c2i_flip_ten_crop}"
VQ_CKPT="${VQ_CKPT:-./pretrained_models/vq_ds16_c2i.pt}"
NPROC="${NPROC:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-12336}"

log() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -f "${TAR_PATH}" ]]; then
  echo "Missing tar: ${TAR_PATH}" >&2
  exit 1
fi

mkdir -p "${TRAIN_DIR}" "${CLASS_TAR_DIR}" "${CODE_PATH}"

class_count=$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ "${class_count}" -lt 1000 ]]; then
  log "Extracting outer ImageNet train tar to ${CLASS_TAR_DIR}"
  tar -xf "${TAR_PATH}" -C "${CLASS_TAR_DIR}"

  log "Extracting per-class tar files to ${TRAIN_DIR}"
  shopt -s nullglob
  count=0
  for class_tar in "${CLASS_TAR_DIR}"/*.tar; do
    wnid="$(basename "${class_tar}" .tar)"
    mkdir -p "${TRAIN_DIR}/${wnid}"
    if ! find "${TRAIN_DIR}/${wnid}" -type f -name '*.JPEG' -print -quit | grep -q .; then
      tar -xf "${class_tar}" -C "${TRAIN_DIR}/${wnid}"
    fi
    rm -f "${class_tar}"
    count=$((count + 1))
    if (( count % 50 == 0 )); then
      log "Extracted ${count} class tar files"
    fi
  done
  rmdir "${CLASS_TAR_DIR}" 2>/dev/null || true
else
  log "ImageFolder train dir already has ${class_count} classes; skip raw extraction"
fi

class_count=$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
image_count=$(find "${TRAIN_DIR}" -type f -name '*.JPEG' | wc -l)
log "Raw train ready: classes=${class_count}, images=${image_count}"

if [[ "${class_count}" -ne 1000 ]]; then
  echo "Expected 1000 classes, got ${class_count}" >&2
  exit 1
fi

mkdir -p "${CODE_PATH}/imagenet384_codes" "${CODE_PATH}/imagenet384_labels"
existing_codes=$(find "${CODE_PATH}/imagenet384_codes" -maxdepth 1 -type f -name '*.npy' | wc -l)
existing_labels=$(find "${CODE_PATH}/imagenet384_labels" -maxdepth 1 -type f -name '*.npy' | wc -l)
if [[ "${existing_codes}" -ge 1281167 && "${existing_labels}" -ge 1281167 ]]; then
  if python - "${CODE_PATH}" <<'PY'
import os
import sys
root = sys.argv[1]
missing = []
for subdir in ("imagenet384_codes", "imagenet384_labels"):
    path = os.path.join(root, subdir)
    for i in range(1281167):
        if not os.path.exists(os.path.join(path, f"{i}.npy")):
            missing.append((subdir, i))
            break
if missing:
    print(f"missing first: {missing[0][0]}/{missing[0][1]}.npy")
    sys.exit(1)
PY
  then
    log "imagenet384 codes already complete (codes=${existing_codes}, labels=${existing_labels}); skip extraction"
    exit 0
  fi
fi

log "Starting 384 code extraction: existing_codes=${existing_codes}, existing_labels=${existing_labels}, batch_size=${BATCH_SIZE}"
torchrun \
  --nnodes=1 --nproc_per_node="${NPROC}" --node_rank=0 \
  --master_port="${MASTER_PORT}" \
  autoregressive/train/extract_codes_c2i.py \
  --vq-ckpt "${VQ_CKPT}" \
  --data-path "${TRAIN_DIR}" \
  --code-path "${CODE_PATH}" \
  --ten-crop \
  --crop-range 1.1 \
  --image-size 384 \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}"

log "384 code extraction finished"
log "codes=$(find "${CODE_PATH}/imagenet384_codes" -maxdepth 1 -type f -name '*.npy' | wc -l), labels=$(find "${CODE_PATH}/imagenet384_labels" -maxdepth 1 -type f -name '*.npy' | wc -l)"

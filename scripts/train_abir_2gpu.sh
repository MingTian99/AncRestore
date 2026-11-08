#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p debug_logs
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="debug_logs/train_abir_2gpu_${RUN_ID}.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun \
  --nproc_per_node=2 \
  --master_port=${MASTER_PORT:-4321} \
  -m basicsr.train \
  -opt options/ABIR_Patch.yml \
  --launcher pytorch 2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-options/ABIR_Patch.yml}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

GPU_IDS=${GPU_IDS:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-4321}

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  -m basicsr.train \
  -opt ${CONFIG} \
  --launcher pytorch

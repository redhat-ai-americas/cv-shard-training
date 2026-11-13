#!/bin/bash
# The Trainer runtime injects PET_* and leaves the container command to us.
set -euo pipefail

echo "distributed environment:"
env | sort | grep -E '^(PET_|MASTER_|RANK|WORLD_SIZE|LOCAL_RANK)=' || true

NNODES="${PET_NNODES:-1}"
NPROC="${PET_NPROC_PER_NODE:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "PET_MASTER_ADDR is empty, so torchrun has no rendezvous address" >&2
  exit 1
fi

export PYTHONPATH="/data/pydeps${PYTHONPATH:+:$PYTHONPATH}"
PHASE="${2:-train}"
METRICS_DIR="${METRICS_DIR:-/data/metrics}"
if [[ "${PHASE}" == "eval" ]]; then
  METRICS="${METRICS_DIR}/test-metrics.json"
else
  METRICS="${METRICS_DIR}/train-metrics.json"
fi
EPOCH_ARGS=()
if [[ -n "${EPOCHS:-}" ]]; then
  EPOCH_ARGS+=(--epochs "${EPOCHS}")
fi

SAMPLER_PID=""
if [[ "${LOCAL_RANK:-0}" == "0" && -f /data/code/gpu_sample.py ]]; then
  GPU_SAMPLE_PHASE="${PHASE}" python /data/code/gpu_sample.py &
  SAMPLER_PID=$!
fi

set +e
torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  /data/code/train.py \
  --manifest "${1:-/data/shards/manifest.json}" \
  --phase "${PHASE}" \
  --metrics "${METRICS}" \
  "${EPOCH_ARGS[@]}"
status=$?
set -e

if [[ -n "${SAMPLER_PID}" ]]; then
  kill "${SAMPLER_PID}" >/dev/null 2>&1 || true
  wait "${SAMPLER_PID}" >/dev/null 2>&1 || true
fi
exit "${status}"

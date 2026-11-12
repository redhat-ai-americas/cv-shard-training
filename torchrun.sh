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
if [[ "${PHASE}" == "eval" ]]; then
  METRICS="/data/metrics/test-metrics.json"
else
  METRICS="/data/metrics/train-metrics.json"
fi
exec torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  /data/code/train.py \
  --manifest "${1:-/data/shards/manifest.json}" \
  --phase "${PHASE}" \
  --metrics "${METRICS}"

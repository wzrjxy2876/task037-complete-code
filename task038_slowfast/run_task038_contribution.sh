#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TASK038_OUTPUT_DIR="${TASK038_OUTPUT_DIR:-/home/jixinye25/jxy_work1/task038_slowfast_runs/n09_prune50}"
PYTHON="${TASK038_PYTHON:-/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python}"
exec "$PYTHON" -m task038_slowfast.task038_cli --mode contribution --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"

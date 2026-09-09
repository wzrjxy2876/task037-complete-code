#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TASK038_OUTPUT_DIR="${TASK038_OUTPUT_DIR:-/data/jixinye25/work1/output/task038_slowfast_functional_coverage_migration/n09_remain50_exact}"
PYTHON="${TASK038_PYTHON:-/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
exec "$PYTHON" -m task038_slowfast.task038_cli --mode preflight --output_dir "$TASK038_OUTPUT_DIR"

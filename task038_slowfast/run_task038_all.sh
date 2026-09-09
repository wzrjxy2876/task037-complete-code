#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TASK038_BATCH_SIZE=16
export TASK038_OUTPUT_DIR="${TASK038_OUTPUT_DIR:-/data/jixinye25/work1/output/task038_slowfast_functional_coverage_migration/n09_remain50_exact}"
PYTHON="${TASK038_PYTHON:-/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python}"
"$PYTHON" -m task038_slowfast.task038_cli --mode preflight --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode baseline --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode descriptors --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode bms --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode contribution --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode numerical --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode prefix --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode selection --device "${TASK038_DEVICE:-cuda:0}" --target-remaining-ratio 0.50 --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode logical --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode preft --device "${TASK038_DEVICE:-cuda:0}" --output_dir "$TASK038_OUTPUT_DIR"
"$PYTHON" -m task038_slowfast.task038_cli --mode finetune --device "${TASK038_DEVICE:-cuda:0}" --gpu-ids ${TASK038_GPU_IDS:-0 1} --output_dir "$TASK038_OUTPUT_DIR"

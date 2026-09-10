#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python"
ROOT="/data/jixinye25/work1/output/task038_slowfast_functional_coverage_migration/cf_n_sweep_sigma0p050_remain50"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

case "${1:-}" in
  prepare)
    export CUDA_VISIBLE_DEVICES=0
    exec "${PYTHON_BIN}" -u -m task038_slowfast.task038_cf_n_sweep \
      --phase prepare --device cuda:0 --cuda-visible-devices 0 --root "${ROOT}"
    ;;
  gpu0)
    export CUDA_VISIBLE_DEVICES=0
    exec "${PYTHON_BIN}" -u -m task038_slowfast.task038_cf_n_sweep \
      --phase worker --n-values 45 18 --device cuda:0 --cuda-visible-devices 0 --root "${ROOT}"
    ;;
  gpu1)
    export CUDA_VISIBLE_DEVICES=1
    exec "${PYTHON_BIN}" -u -m task038_slowfast.task038_cf_n_sweep \
      --phase worker --n-values 36 27 9 --device cuda:0 --cuda-visible-devices 1 --root "${ROOT}"
    ;;
  aggregate)
    export CUDA_VISIBLE_DEVICES=0
    exec "${PYTHON_BIN}" -u -m task038_slowfast.task038_cf_n_sweep \
      --phase aggregate --device cuda:0 --cuda-visible-devices 0 --root "${ROOT}"
    ;;
  *)
    echo "usage: $0 {prepare|gpu0|gpu1|aggregate}" >&2
    exit 2
    ;;
esac

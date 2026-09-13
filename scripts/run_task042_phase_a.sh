#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/src/lgfr_runtime/task042_temporal_relation_collapse.py"
OUT_ROOT="${OUT_ROOT:-/data/jixinye25/work1/output/task042_temporal_relation_collapse}"
RUN_ID="${RUN_ID:-phase_a1_pilot}"
ANNOTATION="${ANNOTATION:-/data/jixinye25/UCF101_Frame/val_rgb_split1.txt}"

MAMBA_PY="/home/jixinye25/miniconda3/envs/mamba/bin/python"
MC_PY="/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python"

for executable in "${MAMBA_PY}" "${MC_PY}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Missing Python environment: ${executable}" >&2
    exit 2
  fi
done
if [[ ! -f "${ANNOTATION}" ]]; then
  echo "Missing UCF101 annotation: ${ANNOTATION}" >&2
  exit 2
fi

run_model() {
  local model="$1" python="$2" physical_gpu="$3"
  echo "=== ${model}: audit (physical GPU ${physical_gpu} only) ==="
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python}" "${PY_SCRIPT}" audit \
    --model "${model}" --annotation "${ANNOTATION}" --output-root "${OUT_ROOT}" --run-id "${RUN_ID}"
  echo "=== ${model}: evaluation (physical GPU ${physical_gpu} only) ==="
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python}" "${PY_SCRIPT}" run \
    --model "${model}" --annotation "${ANNOTATION}" --output-root "${OUT_ROOT}" --run-id "${RUN_ID}" \
    --device cuda:0 --videos-per-class 1
}

# CUDA_VISIBLE_DEVICES remaps the selected physical device to logical cuda:0.
# Only physical GPU 0 and GPU 1 are used; GPUs 2 and 3 are never exposed.
run_model mamba "${MAMBA_PY}" 0
run_model slowfast "${MC_PY}" 1
run_model swin "${MC_PY}" 0

CUDA_VISIBLE_DEVICES=0 "${MAMBA_PY}" "${PY_SCRIPT}" finalize \
  --output-root "${OUT_ROOT}" --run-id "${RUN_ID}"

echo "Task042 Phase A complete: ${OUT_ROOT}/${RUN_ID}"

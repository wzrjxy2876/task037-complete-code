#!/usr/bin/env bash
set -euo pipefail

# Future server entry point.  This script is created in Phase A and is not
# executed by the Task040 code-first/push-first phase.
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the pushed Task040 checkout}"
CHECKPOINT="${CHECKPOINT:-/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt}"
VAL_LIST="${VAL_LIST:?set VAL_LIST to the UCF101 validation split}"
FRAME_ROOT="${FRAME_ROOT:?set FRAME_ROOT to the UCF101 frame root}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for Task040 outputs}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-2}"

# Exact whole-unit hooks are deliberately kept on one GPU.  Do not wrap this
# oracle in DataParallel unless hook behavior is proven identical first.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

if [[ -f "$PROJECT_ROOT/src/lgfr_runtime/task040_htor_probe.py" ]]; then
  PROBE="$PROJECT_ROOT/src/lgfr_runtime/task040_htor_probe.py"
  PYTHONPATH="$PROJECT_ROOT/src/lgfr_runtime:$PROJECT_ROOT:${PYTHONPATH:-}"
elif [[ -f "$PROJECT_ROOT/task040_htor_probe.py" ]]; then
  PROBE="$PROJECT_ROOT/task040_htor_probe.py"
  PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
else
  echo "Task040 probe source not found below PROJECT_ROOT=$PROJECT_ROOT" >&2
  exit 2
fi
export PYTHONPATH

exec python "$PROBE" \
  --project_root "$PROJECT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --val_list "$VAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --num_classes 3 \
  --videos_per_class 3 \
  --seed 3407 \
  --layers "${LAYERS:-representative}" \
  --units_per_layer "${UNITS_PER_LAYER:-3}" \
  --intervention_batch_size "$INTERVENTION_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS"



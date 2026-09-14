#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy"
BASE="/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy"
OUT="$BASE/phase_e1"
SCRIPT="$REPO/scripts/task042_phase_e1_progressive_pilot.py"

if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != "MC" ]]; then
  echo "Task042 Phase E.1 must run inside tmux session MC." >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "task_042_post_bms_frame_relation_redundancy" ]]; then
  echo "Task042 checkout is on the wrong branch." >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite an existing Phase E.1 output directory: $OUT" >&2
  exit 2
fi
source "/home/jixinye25/miniconda3/etc/profile.d/conda.sh"
conda activate MC_Pruning
cd "$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python -m py_compile "$SCRIPT" "$REPO/tests/test_task042_phase_e1_progressive_pilot.py"
python -m unittest discover -s tests -p 'test_task042_phase_e1_progressive_pilot.py' -v
python "$SCRIPT" --phase prepare --repo "$REPO" --base "$BASE"
mkdir -p "$OUT/work/logs"

(
  export CUDA_VISIBLE_DEVICES=0
  python -u "$SCRIPT" --phase reference-worker --repo "$REPO" --base "$BASE" --physical-gpu 0 --shard 0
) >"$OUT/work/logs/reference_gpu0.log" 2>&1 &
REF0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  python -u "$SCRIPT" --phase reference-worker --repo "$REPO" --base "$BASE" --physical-gpu 1 --shard 1
) >"$OUT/work/logs/reference_gpu1.log" 2>&1 &
REF1=$!
set +e
wait "$REF0"; REF0_STATUS=$?
wait "$REF1"; REF1_STATUS=$?
set -e
if [[ "$REF0_STATUS" -ne 0 || "$REF1_STATUS" -ne 0 ]]; then
  echo "Reference workers failed: GPU0=$REF0_STATUS GPU1=$REF1_STATUS" >&2
  tail -n 80 "$OUT/work/logs/reference_gpu0.log" || true
  tail -n 80 "$OUT/work/logs/reference_gpu1.log" || true
  exit 1
fi
python "$SCRIPT" --phase finalize-reference --repo "$REPO" --base "$BASE"

(
  export CUDA_VISIBLE_DEVICES=0
  python -u "$SCRIPT" --phase calibration-worker --repo "$REPO" --base "$BASE" --physical-gpu 0
) >"$OUT/work/logs/calibration_gpu0.log" 2>&1 &
CAL0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  python -u "$SCRIPT" --phase calibration-worker --repo "$REPO" --base "$BASE" --physical-gpu 1
) >"$OUT/work/logs/calibration_gpu1.log" 2>&1 &
CAL1=$!
set +e
wait "$CAL0"; CAL0_STATUS=$?
wait "$CAL1"; CAL1_STATUS=$?
set -e
if [[ "$CAL0_STATUS" -ne 0 || "$CAL1_STATUS" -ne 0 ]]; then
  echo "Gradient calibration workers failed: GPU0=$CAL0_STATUS GPU1=$CAL1_STATUS" >&2
  tail -n 80 "$OUT/work/logs/calibration_gpu0.log" || true
  tail -n 80 "$OUT/work/logs/calibration_gpu1.log" || true
  exit 1
fi
python "$SCRIPT" --phase finalize-calibration --repo "$REPO" --base "$BASE"

# Independent arms A and C use the two permitted GPUs concurrently. After A
# finishes, B reuses GPU 0 while C continues on GPU 1.
(
  export CUDA_VISIBLE_DEVICES=0
  python -u "$SCRIPT" --phase arm-worker --arm A --repo "$REPO" --base "$BASE" --physical-gpu 0
) >"$OUT/work/logs/arm_A.log" 2>&1 &
ARM_A=$!
(
  export CUDA_VISIBLE_DEVICES=1
  python -u "$SCRIPT" --phase arm-worker --arm C --repo "$REPO" --base "$BASE" --physical-gpu 1
) >"$OUT/work/logs/arm_C.log" 2>&1 &
ARM_C=$!
set +e
wait "$ARM_A"; ARM_A_STATUS=$?
set -e
if [[ "$ARM_A_STATUS" -ne 0 ]]; then
  set +e; wait "$ARM_C"; ARM_C_STATUS=$?; set -e
  echo "Arm A failed: status=$ARM_A_STATUS; Arm C status=$ARM_C_STATUS" >&2
  tail -n 100 "$OUT/work/logs/arm_A.log" || true
  tail -n 100 "$OUT/work/logs/arm_C.log" || true
  exit 1
fi
(
  export CUDA_VISIBLE_DEVICES=0
  python -u "$SCRIPT" --phase arm-worker --arm B --repo "$REPO" --base "$BASE" --physical-gpu 0
) >"$OUT/work/logs/arm_B.log" 2>&1 &
ARM_B=$!
set +e
wait "$ARM_B"; ARM_B_STATUS=$?
wait "$ARM_C"; ARM_C_STATUS=$?
set -e
if [[ "$ARM_B_STATUS" -ne 0 || "$ARM_C_STATUS" -ne 0 ]]; then
  echo "Arm workers failed: B=$ARM_B_STATUS C=$ARM_C_STATUS" >&2
  tail -n 100 "$OUT/work/logs/arm_B.log" || true
  tail -n 100 "$OUT/work/logs/arm_C.log" || true
  exit 1
fi

python "$SCRIPT" --phase finalize --repo "$REPO" --base "$BASE"
echo "Task042 Phase E.1 complete. Outputs: $OUT"

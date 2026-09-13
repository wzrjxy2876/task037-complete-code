#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy"
OUT="/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy"
SCRIPT="$REPO/src/lgfr_runtime/task042_frame_relation_redundancy.py"

if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != "MC" ]]; then
  echo "Task042 GPU execution must run inside tmux session MC." >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "task_042_post_bms_frame_relation_redundancy" ]]; then
  echo "Task042 checkout is on the wrong branch." >&2
  exit 2
fi

source "/home/jixinye25/miniconda3/etc/profile.d/conda.sh"
conda activate MC_Pruning
cd "$REPO"
export PYTHONUNBUFFERED=1

python -m py_compile "$SCRIPT" "$REPO/tests/test_task042_frame_relation_redundancy.py"
python -m unittest discover -s tests -p 'test_task042_frame_relation_redundancy.py' -v
python "$SCRIPT" --phase prepare --repo-root "$REPO" --output-dir "$OUT"
python "$SCRIPT" --phase preflight --repo-root "$REPO" --output-dir "$OUT"

(
  export CUDA_VISIBLE_DEVICES=0
  python "$SCRIPT" --phase worker --gpu 0 --repo-root "$REPO" --output-dir "$OUT"
) >"$OUT/task042_gpu0.log" 2>&1 &
GPU0_PID=$!
(
  export CUDA_VISIBLE_DEVICES=1
  python "$SCRIPT" --phase worker --gpu 1 --repo-root "$REPO" --output-dir "$OUT"
) >"$OUT/task042_gpu1.log" 2>&1 &
GPU1_PID=$!

set +e
wait "$GPU0_PID"
GPU0_STATUS=$?
wait "$GPU1_PID"
GPU1_STATUS=$?
set -e
if [[ "$GPU0_STATUS" -ne 0 || "$GPU1_STATUS" -ne 0 ]]; then
  echo "Task042 worker failure: GPU0=$GPU0_STATUS GPU1=$GPU1_STATUS" >&2
  tail -n 60 "$OUT/task042_gpu0.log" || true
  tail -n 60 "$OUT/task042_gpu1.log" || true
  exit 1
fi

python "$SCRIPT" --phase finalize --repo-root "$REPO" --output-dir "$OUT"
echo "Task042 complete. Outputs: $OUT"

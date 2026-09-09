#!/usr/bin/env bash
# Task037 F3 logical-pruning and fine-tuning command sheet.
# Source code is resolved only from this standalone package.
# Reference roots below are read-only generated artifacts, never source paths
# and never PYTHONPATH entries.
#
# export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# PKG=/home/jixinye25/jxy_work1/swintrans_task037_complete
# RUNTIME=$PKG/src/task028_runtime
# RUN_ROOT=$PKG/runs/task037_n09
# TASK028_ROOT=${TASK028_ROOT:?set read-only Task028 artifact root}
# TASK029_ROOT=${TASK029_ROOT:?set read-only Task029 artifact root}
# TASK030_ROOT=${TASK030_ROOT:?set read-only Task030 artifact root}
# TASK031_ROOT=${TASK031_ROOT:?set read-only Task031 artifact root}
# TASK032_ROOT=${TASK032_ROOT:?set read-only Task032 artifact root}
# TASK033_ROOT=${TASK033_ROOT:?set read-only Task033 artifact root}
# TASK017_ROOT=${TASK017_ROOT:?set fresh Task017 artifact root}
# CHECKPOINT=${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to checkpoint-68.ckpt}
# OUT=$RUN_ROOT/task034_n09_reproduction
# export PYTHONPATH=$PKG/pruning:$PKG/models:$PKG/src/task028_runtime:$PKG/src/lgfr_runtime
#
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode identity \
#   --task028-root $TASK028_ROOT --task029-root $TASK029_ROOT \
#   --task030-root $TASK030_ROOT --task031-root $TASK031_ROOT \
#   --task032-root $TASK032_ROOT --task033-root $TASK033_ROOT \
#   --output-dir $OUT --repo-root $RUNTIME
#
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode benchmark-selection \
#   --task014-root $RUN_ROOT/task014_n09 --task016-root $RUN_ROOT/task016_n09 \
#   --task017-root $TASK017_ROOT --output-dir $OUT \
#   --device cuda:0 --steps 1000 --ranking-backend dual-gpu
#
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode select \
#   --task014-root $RUN_ROOT/task014_n09 --task016-root $RUN_ROOT/task016_n09 \
#   --task017-root $TASK017_ROOT --output-dir $OUT \
#   --device cuda:0 --ranking-backend dual-gpu
#
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode verify-logical \
#   --output-dir $OUT --checkpoint $CHECKPOINT --device cuda:0
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode preft-validate \
#   --output-dir $OUT --checkpoint $CHECKPOINT --device cuda:0
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode sanity \
#   --output-dir $OUT --checkpoint $CHECKPOINT --device cuda:0
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode speed-gate \
#   --output-dir $OUT --checkpoint $CHECKPOINT --device cuda:0 --steps 100 \
#   --ranking-backend dual-gpu
#
# Only after every gate is PASS:
# python $PKG/pruning/task034_mid_veto_50_logical_finetune.py --mode finetune \
#   --output-dir $OUT --checkpoint $CHECKPOINT --device cuda:0

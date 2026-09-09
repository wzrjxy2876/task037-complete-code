#!/usr/bin/env bash
# Task037 Contribution Field command sheet.
# Source code is resolved only from this standalone package.
#
# export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# PKG=/home/jixinye25/jxy_work1/swintrans_task037_complete
# RUN_ROOT=$PKG/runs/task037_n09
# PROJECT_ROOT=$PKG/src/lgfr_runtime
# CHECKPOINT=${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to checkpoint-68.ckpt}
# FRAME_ROOT=${UCF101_FRAME_ROOT:?set UCF101_FRAME_ROOT}
# VAL_LIST=${UCF101_VAL_LIST:?set UCF101_VAL_LIST}
# export PYTHONPATH=$PKG/contribution:$PKG/src/lgfr_runtime:$PKG/pruning:$PKG/models:$PKG/src/task028_runtime
#
# python $PKG/contribution/probe_cstc_contribution_tube.py \
#   --project_root $PROJECT_ROOT \
#   --checkpoint $CHECKPOINT \
#   --output_dir $RUN_ROOT/cstc_n09 \
#   --device cuda:0 \
#   --seed 3407 \
#   --num_classes 3 \
#   --videos_per_class 3 \
#   --frame_root $FRAME_ROOT \
#   --val_list $VAL_LIST
#
# Verify cstc_probe_metadata.json contains contribution_sample_count=9
# before using cstc_probe_arrays.npz in any pruning stage.

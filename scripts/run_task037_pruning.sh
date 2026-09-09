#!/usr/bin/env bash
# Task037 functional-pruning command sheet.
# Source code is resolved only from this standalone package.
#
# export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# PKG=/home/jixinye25/jxy_work1/swintrans_task037_complete
# RUN_ROOT=$PKG/runs/task037_n09
# CHECKPOINT=${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to checkpoint-68.ckpt}
# NPZ=$RUN_ROOT/cstc_n09/cstc_probe_arrays.npz
# export PYTHONPATH=$PKG/pruning:$PKG/models:$PKG/src/task028_runtime:$PKG/src/lgfr_runtime
#
# python $PKG/models/ucf101_videoswin_my.py \
#   --gpu 0,1 \
#   --selection_mode functional \
#   --selection_cost_mode decoupled \
#   --descriptor_variant dynamic3d \
#   --functional-score domain_average \
#   --sparsity 0.30 \
#   --sigma 0.1 \
#   --gamma_decay 0.5 \
#   --min_keep_ratio 0.1 \
#   --importance_alpha 0.5 \
#   --batch_size 4 \
#   --calib_batch_size 4 \
#   --calib_batches 10 \
#   --seed 3407 \
#   --contribution_npz $NPZ \
#   --checkpoint_path $CHECKPOINT \
#   --prune_only \
#   --output_root $RUN_ROOT/task014_n09
#
# python $PKG/models/ucf101_videoswin_my.py \
#   --gpu 0,1 \
#   --selection_mode functional \
#   --selection_cost_mode decoupled \
#   --descriptor_variant dynamic3d \
#   --functional-score domain_total \
#   --sparsity 0.30 \
#   --sigma 0.1 \
#   --gamma_decay 0.5 \
#   --min_keep_ratio 0.1 \
#   --importance_alpha 0.5 \
#   --batch_size 4 \
#   --calib_batch_size 4 \
#   --calib_batches 10 \
#   --seed 3407 \
#   --contribution_npz $NPZ \
#   --checkpoint_path $CHECKPOINT \
#   --prune_only \
#   --output_root $RUN_ROOT/task016_n09

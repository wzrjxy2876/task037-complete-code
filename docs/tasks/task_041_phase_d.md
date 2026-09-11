# Task041 Phase D — Full-Validation Masking Oracle

This is a new, diagnostic-only Phase D on the existing
task_041_collective_temporal_coverage_pruning_oracle branch.

The Phase D source freezes the existing Task041 N=3 MCTC scores and exact 29
unit identities. It runs the same Video Swin/UCF101 preprocessing and
checkpoint-68 model on all 3783 validation clips, applies only temporary
whole-unit masks, computes signed paired damage, and performs offline 1000-fold
bootstrap confidence intervals with seed 3407.

It does not recompute temporal profiles or interventions, change BMS assignments,
physically prune, fine-tune, run N=9 calibration, or create Task042.

Execution uses one baseline process and two independent masking shard processes:
GPU0 and GPU1 each load their own checkpoint/model. The final merge is CPU-only
and writes a new Phase D output directory, leaving the existing Task041 N=30
artifacts untouched.

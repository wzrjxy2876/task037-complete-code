#!/usr/bin/env bash
# Task037 complete pipeline command sheet.
# The commands are documented and are intentionally not auto-executed.
# Run in this order with fresh output under runs/task037_n09:
#
# 1. bash scripts/run_task037_contribution.sh
#    Generate Contribution Field with sample_count=9 and seed=3407.
#
# 2. bash scripts/run_task037_pruning.sh
#    Run functional domain_average and domain_total pruning.
#
# 3. bash scripts/run_task037_finetune.sh
#    Run F3 logical selection and every pre-training gate.
#
# 4. After all gates report PASS, run the exact 100-epoch FP32 SGD
#    command at the end of run_task037_finetune.sh.
#
# All executable source paths used by these commands are inside this package.

# Task041 Phase G — N=9 BCTR calibration stability only

This Task041-only runner validates the predeclared N=3 to N=9 calibration-stability gate. It is not a pruning or finetuning launch.

## Frozen inputs and scope

- Branch: task_041_collective_temporal_coverage_pruning_oracle.
- Model: Video Swin Transformer / UCF101, loaded through the validated Task040 adapter.
- Checkpoint: /home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt.
- Required SHA256: 4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63.
- Reuses the authoritative Task040 N=9 manifest, Phase-C fixed-cardinality intervention manifest, Phase-F signed N=3 records, frozen 29-unit identities, and Phase-D full-validation damage.
- Uses exactly spans 1, 2, 4, 8, 16 and the existing 16 two-frame swaps per span at T=32.
- Formula, fit bounds, BMS peers, and CE gate are unchanged. No Phase-F N=3 inference or Phase-D 3,783-clip rerun occurs.
- New GPU inference is FP32/AMP off, on physical GPUs 0 and 1 in two independent workers. The workers process disjoint missing-video shards, and both are visible in tmux session MC.
- Only the six missing videos are processed: 13,920 new rows. The required final N=9 signature count is 20,880.
- A failed identity, checkpoint, sample, intervention, or exact mask-restoration check stops the run. Do not manually bypass a failed gate.

## Test and compile

From the Task041 checkout:

    cd /home/jixinye25/jxy_work1/task040_htor_d39a947
    git status --short --branch
    python -m py_compile src/lgfr_runtime/task041_phase_g_n9_validation.py
    PYTHONPATH=src/lgfr_runtime python -m pytest -q tests/test_task041_phase_g_n9.py tests/test_task041_phase_f_bctr.py tests/test_task040_masking.py tests/test_task040_htor_core.py

The expected branch is task_041_collective_temporal_coverage_pruning_oracle. Do not change Task037 production code, D_abs/D_rel/D_st, BMS, or any frozen scientific definition.

## Visible MC execution

Activate the MC_Pruning environment, then use the Python executable and commands below. Start with a visible preparation/preflight window; the verified manifest is written to the requested output directory before any GPU inference.

    tmux new-window -t MC -n phase_g_preflight
    tmux send-keys -t MC:phase_g_preflight 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python src/lgfr_runtime/task041_phase_g_n9_validation.py prepare' C-m

After prepare succeeds, run CPU-only preflight in the same visible window:

    tmux send-keys -t MC:phase_g_preflight '/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python src/lgfr_runtime/task041_phase_g_n9_validation.py preflight' C-m

Only after CPU preflight reports success, launch the two isolated workers in separate visible windows:

    tmux new-window -t MC -n phase_g_gpu0
    tmux send-keys -t MC:phase_g_gpu0 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python src/lgfr_runtime/task041_phase_g_n9_validation.py worker --gpu 0' C-m
    tmux new-window -t MC -n phase_g_gpu1
    tmux send-keys -t MC:phase_g_gpu1 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python src/lgfr_runtime/task041_phase_g_n9_validation.py worker --gpu 1' C-m

Wait until both windows report COMPLETE and verify GPU/process state before the offline merge. Then run:

    tmux new-window -t MC -n phase_g_finalize
    tmux send-keys -t MC:phase_g_finalize 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python src/lgfr_runtime/task041_phase_g_n9_validation.py finalize' C-m

Do not run finalize unless both workers completed with exactly 6,960 rows and all masks restored exactly. It verifies N=3 replay against saved Phase-F BCTR before producing outputs.

## Required result files

All final files go under:

    /data/jixinye25/work1/output/task041_phase_g_bctr_n9_validation/

The final directory must contain exactly the 13 files enumerated in the Phase-G source. The preparation step writes the verified N=9 manifest there before GPU inference. Temporary worker shards and state are kept under /tmp/task041_phase_g_bctr_n9_validation_work.

## Stop condition

After Phase G, stop. Do not run pruning, 50% sparsity, physical pruning, finetuning, or create Task042. If the predeclared CE stability gate fails, report BCTR_REMAINS_WEAK_OR_UNRESOLVED; no post-hoc threshold may be added to force the rejected category.

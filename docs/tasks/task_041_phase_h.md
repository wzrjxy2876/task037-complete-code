# Task041 Phase H — Post-BMS Frame-Relation Temporal Stress Diagnosis

Phase H is a diagnostic-only continuation on `task_041_collective_temporal_coverage_pruning_oracle`.
It stops BCTR reconstruction-based selection and asks whether explicit frame-pair stress conditions
improve within-frozen-BMS-domain ranking of actual signed deletion CE damage over the original-only
N=9 deletion ranking.

## Frozen scope

- Reuse exactly the 29 Task041 units, their frozen Task040 identities/stages, the nine-video Phase-G
  manifest, the existing 3,783-clip full-validation CE oracle, and the exact T=32 / 80-intervention
  Phase-G manifest (16 two-single-frame swaps at each span 1, 2, 4, 8, 16).
- Keep `[D_abs, D_rel, D_st]`, BMS, Task037 production pruning, and the Phase-F / Phase-G decisions
  unchanged. BCTR, MCTC, PTR, HTOR, STIR, and reconstruction residual are comparison-only baselines.
- Use Video Swin / UCF101 with `checkpoint-68.ckpt` and its frozen SHA256; retain the 400-output
  classifier. Inference is FP32 with AMP disabled, using independent single-device workers on physical
  GPUs 0 and 1.
- For candidate `i`, each condition's signed damage is `CE(masked) - CE(unmasked)`. Preserve negative
  values. The unmasked original and 80 temporal CE values are cached once per video and reused for all
  candidates. Each candidate is temporarily whole-unit-masked using the validated Task040 hook;
  parameter/buffer versions and the hook table are checked after every candidate/video.
- Rank only within each frozen BMS domain. Report all five span-specific win rates, original-only
  win rate, same-type seven-domain CE comparisons, mixed-domain descriptive rows, and N=3/N=6/N=9
  stability computed from the same Phase-H records.
- No full-validation rerun, extra video sampling, persistent mask, physical pruning, finetuning,
  Task042, new selector, or new success threshold.

## Server execution

Run from the checked-out Task041 repository on the required branch. Keep all progress visible in the
existing `tmux MC` session. First prepare and perform the CPU-only identity/frame preflight:

```bash
cd /home/jixinye25/jxy_work1/task040_htor_d39a947
tmux new-window -t MC -n phase_h_prepare
tmux send-keys -t MC:phase_h_prepare 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py prepare' C-m
```

Wait until `prepare` exits successfully and the prompt returns, then start preflight in that window:

```bash
tmux send-keys -t MC:phase_h_prepare '/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py preflight' C-m
```

After the preflight passes, cache all unmasked CE once on GPU 0. Only after that cache is complete,
run the two independent candidate shards in separate visible MC windows:

```bash
tmux new-window -t MC -n phase_h_cache
tmux send-keys -t MC:phase_h_cache 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py cache_unmasked' C-m
```

```bash
tmux new-window -t MC -n phase_h_gpu0
tmux send-keys -t MC:phase_h_gpu0 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py worker --gpu 0' C-m
tmux new-window -t MC -n phase_h_gpu1
tmux send-keys -t MC:phase_h_gpu1 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py worker --gpu 1' C-m
```

Finalize only after both workers exit successfully and write their completion markers:

```bash
tmux new-window -t MC -n phase_h_finalize
tmux send-keys -t MC:phase_h_finalize 'cd /home/jixinye25/jxy_work1/task040_htor_d39a947 && /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_h_temporal_stress.py finalize' C-m
```

The script refuses to overwrite an existing work/output directory or repeat the one-time unmasked
cache. It writes the ten required result artifacts only to
`/data/jixinye25/work1/output/task041_phase_h_temporal_stress_ranking/`.

## Decision interpretation

The A gate is exactly the declared rule: positive same-type domain-balanced CE Spearman, plus a
strict improvement over original-only on at least one of Spearman, Kendall tau-b, or safest-unit
identity accuracy, while at least one of the remaining measures is not worse. No numerical cutoff
is added. Since no independent rejection criterion for C was specified, a failure to establish A is
reported conservatively as `TEMPORAL_STRESS_ADDS_NO_CLEAR_VALUE`; it is not upgraded to a rejection
claim after observing results.

The ten outputs are the temporal CE raw records, per-unit win rates, per-span win rates,
original-versus-temporal comparison, same-type oracle, mixed-domain description, subset stability,
frozen-baseline comparison, machine-readable summary, and report. Stop after Phase H; no follow-on
pruning or training is authorized by this phase.

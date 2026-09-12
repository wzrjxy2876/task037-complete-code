# Task041 Phase E — Signed Temporal Interaction Failure Diagnosis

## Scope

Phase E is an offline diagnosis on the existing Task040 Phase C and D.1 raw factorial logits and the existing Task041 full-validation damage table. It uses exactly the 29 frozen Task041 units and verifies the full identity tuple: Task037 global index, Task040 global index, layer, unit type, unit index, and domain.

For each raw record it reconstructs, in float64 and without rectification,

\[
C_i(v,s,m)=a-b-c+d
\]

from the stored original, masked, intervened, and intervened-masked true-class logits. Phase C's stored interaction is used only as an exact consistency check. Per-span statistics include signed mean, mean absolute value, RMS, population standard deviation, positive/negative/zero fractions, and sign balance. Unit summaries include the five-span signed/RMS norms and cancellation ratio.

The script is standalone offline Python with NumPy; it does not import torch, load a checkpoint, or access a GPU. It does not rerun temporal interventions, recompute BMS, alter MCTC, prune, fine-tune, or create a pruning score. Correlations are primary within-domain/domain-balanced, with pooled correlations marked secondary. Mixed domains 271/297 remain separate from same-type domains.

## Authoritative inputs

- Task040 Phase C: `n03_fixed_span/task040_factorial_records.csv`
- Task040 Phase D.1: `d1_targeted_domains/task040_phase_d1_new_raw_records.csv`
- Frozen Task041 unit identities and scores: `task041_masking_damage.csv`
- Task041 full-validation unit damage: `task041_fullval_unit_damage.csv`
- Frozen Task041 low/high roles and baselines: `task041_oracle_candidates.csv`, `task041_baseline_comparison.csv`

The script gates that Phase C and D.1 form a disjoint, exact partition of the 29 candidate units, each with 3 videos × 5 spans × 16 interventions, for 6,960 selected records. It refuses to overwrite any Phase E artifact.

## Execution

Run in the existing visible tmux session `MC`, using the MC_Pruning Python. The command is CPU/offline; no CUDA device is selected:

```bash
tmux new-window -t MC -n phase_e -c /home/jixinye25/jxy_work1/task040_htor_d39a947
tmux send-keys -t MC:phase_e.0 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u src/lgfr_runtime/task041_phase_e_signed_interaction.py --phase_c_raw /data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis/n03_fixed_span/task040_factorial_records.csv --phase_d1_raw /data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis/d1_targeted_domains/task040_phase_d1_new_raw_records.csv --task041_output_dir /data/jixinye25/work1/output/task041_collective_temporal_coverage_pruning_oracle --phase_d_output_dir /data/jixinye25/work1/output/task041_phase_d_fullval_resolution_audit --output_dir /data/jixinye25/work1/output/task041_phase_e_signed_interaction_diagnosis' C-m
```

## Outputs

The new Phase E directory receives exactly the requested CSV/JSON/Markdown outputs:

- `task041_signed_span_statistics.csv`
- `task041_signed_unit_summary.csv`
- `task041_signed_vs_fullval_damage.csv`
- `task041_signed_same_type_domains.csv`
- `task041_mctc_failure_cases.csv`
- `task041_baseline_flip_comparison.csv`
- `task041_phase_e_report.md`
- `task041_phase_e_summary.json`

No existing Task040/Task041 results are modified. Task041 stops after Phase E; no Task042, N=9, pruning, or fine-tuning is launched.

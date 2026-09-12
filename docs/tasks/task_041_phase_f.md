# Task041 Phase F — BMS-conditioned frame-relation audit

This is an offline, CPU-only audit on the existing 29 Task041 units. It leaves the three descriptors and BMS assignments unchanged. It does not run inference, temporal interventions, pruning, or fine-tuning.

## Inputs

- Task040 Phase C raw factorial records.
- Task040 Phase D.1 raw factorial records.
- Task041 task041_masking_damage.csv for frozen domain/unit identity including stage.
- Task041 Phase D task041_fullval_unit_damage.csv for the 3,783-sample full-validation damage oracle.
- Task041 task041_baseline_comparison.csv for the existing 18 low/high baseline cohort.

The runner verifies that Phase C and D.1 form a disjoint partition of the frozen 29 identities, and that the full-validation damage joins exactly. Phase C's reconstructed a-b-c+d is checked against stored C_interaction.

## Method

For each unit/span, it constructs a fixed-order 48-D float64 signature (three canonical videos, then the 16 exact pair-intervention identities within each video). Every unit and span must have the identical key order.

The primary bounded fit solves min ||c_i - X alpha||_2^2 with SciPy BVLS and 0 <= alpha_j <= 1; it uses no ridge/lambda or sum-to-one constraint. Coefficients, status, optimality, and unclipped normalized residuals are recorded. R_BCTR is the RMS over the five span residual ratios and is interpreted only within the frozen BMS domain.

The runner also records a best-single-substitute bounded pairwise comparison, diagnostic-only absolute and elementwise-squared representations, mixed-domain type contributions, and leave-one-video-out ranking stability. Baseline comparisons use the same existing 18 candidates and the full-validation damage table. Same-type results across domains 269, 400, 415, 102, 103, 113, and 76 are primary; mixed 271/297 and all-domain summaries are separate/descriptive.

The predeclared gate for “RETAINED FOR N=9 VALIDATION” requires both (1) at least four of seven same-type low/high pairs in the expected damage direction for logit or CE and (2) positive domain-balanced Spearman for that same damage metric. Failure to pass this gate is reported as WEAK / UNRESOLVED. Passing the gate would only justify a separately authorized N=9 validation; it does not approve pruning.

## Run on the server

Run from the Task041 repository in the existing MC tmux session with the MC_Pruning Python environment. Set CUDA_VISIBLE_DEVICES= so no GPU is visible.

    python -m py_compile src/lgfr_runtime/task041_phase_f_bctr_audit.py tests/test_task041_phase_f_bctr.py

    PYTHONPATH=src/lgfr_runtime python -m unittest tests.test_task041_phase_f_bctr -v

Then call task041_phase_f_bctr_audit.py with:
- --phase_c_raw
- --phase_d1_raw
- --frozen_units_csv
- --fullval_damage_csv
- --baseline_scores_csv
- --output_dir

The output directory must be new; the script refuses to reuse an existing Phase F output directory. It writes the eleven Task041 Phase F artifacts listed in OUT_NAMES and records the input SHA256 hashes in the summary.

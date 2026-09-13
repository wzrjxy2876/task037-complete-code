# Task042 — Post-BMS Frame-to-Frame Relation Redundancy Diagnosis

## Base and frozen context

This branch is based on Task041 commit `dc90da61a15ecb76231cf7e468681b59e501627a`. The completed Task041 Phase-I result is retained in `task_041_phase_i_result.md` and its machine-readable summary. Phase I found that 9-class diversity did not rescue temporal-stress selection; Task042 therefore changes the question from unit importance to within-domain temporal-relation redundancy.

The 51-unit cohort is the exact Task040 Phase-D.1 `task037_global_index` set. Its layer, unit type, unit index, stage, and frozen BMS `domain_id` are joined to Task037's descriptor and unit-mapping tables and checked before any GPU worker starts. The video cohort is the exact Task041 10-class by 3-video manifest; no resampling is permitted.

## Question and scope

For units already assigned to the same frozen BMS functional group, do their internal activations change in similar ways under explicit swaps of sampled video frames? This task is diagnostic only. It does not produce a pruning selector, importance score, threshold, mask, or pruning action.

The frozen descriptor coordinates, BMS assignments and sigma, model architecture, and checkpoint semantics remain unchanged. HTOR, STIR, PTR, MCTC, BCTR, and Phase-H win-rate are not metrics for this task. No CE-damage, top-1, logit-drop, or prediction-flip signal is optimized.

## Interventions and activation semantics

The 32 sampled temporal positions use Task040's fixed-cardinality enumerator at spans `{1, 2, 4, 8, 16}`. At each span there are 16 deterministic pairs, and each condition swaps exactly two positions. Every video has one baseline forward and 80 intervened forwards.

For attention head `i`, `H_i(X)` is that head's `attention_weights @ V` output before head concatenation and output projection. The read-only capture uses the attention-dropout output as the weights; in `model.eval()` dropout is the identity. For FFN neuron `i`, `H_i(X)` is the module activation output after `fc1` and GELU and before `fc2`.

For video `v` and condition `q`:

\[
e_i(v,q)=\frac{\|H_i(S_qX_v)-H_i(X_v)\|_F}{\|H_i(X_v)\|_F+10^{-12}}.
\]

The full 80-value signature is retained with exact video, span, pair, and sampled-frame indices. Within each unit-video only, the signature is standardized using its population mean and standard deviation. A standard deviation at or below `1e-12` is marked degenerate and is not repaired with epsilon.

For same-video units `i,j`, the temporal distance is `(1 - Pearson(z_i,z_j))/2`; the main pair distance is its mean over eligible videos. Span-specific distances restrict the already standardized signatures to that span's 16 conditions. No token interpolation or remapping is allowed.

## Analyses

Primary comparisons use every unordered pair inside each frozen BMS domain with at least two tested units. The analysis reports pair identities, stages, unit types, full and span-specific distances, activation norms, domain summaries, and nearest-temporal-neighbor identity with global-index tie-breaking.

Attention and FFN raw sensitivity distributions are compared before and after the within-unit-video normalization. Matched controls use the same Task037 unit type and same Swin stage across different BMS domains; Mann-Whitney U and Cliff's delta are exploratory because pair observations are non-independent.

Calibration stability is computed offline from the 30-video manifest: positions 1, 2, or 3 within each class (A/B/C), the three position-pair subsets (AB/AC/BC), and the full 10x3 set. No further inference is needed. Pair-distance rankings and nearest-neighbor identities are compared with the full set.

Descriptor complementarity uses the frozen source columns `D_abs`, `D_rel`, and `D_third` and records the source name for the third coordinate (`D_dyn`). Task041 documents call the frozen third axis `D_st`; the source column is preserved verbatim so the mapping remains auditable. No descriptor is recomputed or changed.

## Decision rule

Choose exactly one outcome: `FRAME_RELATION_REDUNDANCY_PROMISING`, `FRAME_RELATION_REDUNDANCY_WEAK_OR_UNRESOLVED`, or `FRAME_RELATION_REDUNDANCY_REJECTED`. A promising outcome requires nontrivial within-domain distance variation, median A/B/C nearest-neighbor identity stability of at least 0.75, and no near-perfect rank redundancy with the frozen raw 3D descriptor distance. Otherwise, measurable but incomplete evidence is weak/unresolved; no measurable within-domain distance variation is rejected. This is a diagnostic gate only and grants no pruning authorization.

## Required outputs

The run produces the Task042 video and unit manifests, activation-shape audit, full frame-pair sensitivity records, unit-video signature summaries, same-domain pair distances, domain structure, matched cross-domain controls, calibration stability, span-specific distances, descriptor complementarity, a JSON summary, and a Markdown report. GPU workers are isolated to physical devices 0 and 1, use disjoint deterministic video shards, run FP32 with AMP disabled, and capture all 51 target units in each forward.

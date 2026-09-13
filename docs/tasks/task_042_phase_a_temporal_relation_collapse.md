# Task042 Phase A.1 — Temporal-relation collapse screening

## Question

Does structured pruning preserve relationships among multiple video frames, or
can two models retain similar clip-level predictions while their learned
frame-to-frame trajectories diverge?

This is a screening experiment. It uses the three supplied dense checkpoints
and the existing pruning selections on the server. The saved selection rates
are not matched across architectures, so this phase cannot establish a
cross-model ranking or a matched-FLOPs result.

## Fixed inputs and controls

- Dataset: UCF101 split-1 validation RGB-frame list at
  `/data/jixinye25/UCF101_Frame/val_rgb_split1.txt`.
- Pilot set: first lexicographically ordered validation clip per action class
  (101 clips by default), using the same 32-frame `LoopPadding` and spatial
  test transform as the existing model code.
- Dense checkpoints: VideoMamba `videomamba_small-89%.pth`, SlowFast
  `slowfast-teacher-ucf101.ckpt`, and Video Swin `checkpoint-68.ckpt`.
- No fine-tuning, gradient computation, checkpoint modification, or physical
  model export.
- Conditions: clean order; disjoint adjacent-frame swaps; four temporal blocks
  reordered as `[B3, B1, B4, B2]`; and full temporal reversal.
- Per model, compare Dense, an L2-magnitude control, and the existing saved
  pruning selection. The magnitude control retains the same unit count in each
  prunable module as that model's existing selection.

## Multi-frame relation metric

Let the spatially pooled final-stage frame features be
`Z = (z_1, ..., z_T)`. Normalize each frame feature over channels:

\[
\tilde z_t = \operatorname{LayerNorm}(z_t).
\]

For each three-frame trajectory, compute first and second differences:

\[
d_t = \tilde z_{t+1}-\tilde z_t,\qquad
a_t = \tilde z_{t+2}-2\tilde z_{t+1}+\tilde z_t,
\]

\[
q_t=[d_t;a_t],\qquad R_{ij}=\cos(q_i,q_j).
\]

For Dense and Pruned relation matrices, define retention as cosine similarity
over their upper-triangular entries:

\[
\mathrm{TRR}=\cos(\operatorname{vec}_{i<j}R^{D},
                   \operatorname{vec}_{i<j}R^{P}),\qquad
\mathrm{TRC}=1-\mathrm{TRR}.
\]

Prediction sensitivity is Jensen–Shannon divergence in bits:

\[
S(X,\pi)=\mathrm{JS}_2(p(X),p(\pi(X))).
\]

The per-video CSV also includes top-1 accuracy against the validation labels,
agreement with Dense predictions, pruned-vs-dense JS, and clean-vs-perturbed
JS. SlowFast's slow and fast pathway features are spatially pooled, the slow
pathway is interpolated to the fast pathway's temporal grid, and both are
concatenated before computing TRR.

## Pruning selections used

- VideoMamba: existing saved InteractionPruner masks from the before-finetune
  20% target checkpoint.
- SlowFast: existing saved interaction masks from the approximately 50%
  checkpoint; only its masks are applied to the supplied dense teacher weights.
- Video Swin: existing BMS keep-index selection from the before-finetune 50%
  checkpoint; only its kept-unit registry is applied to the supplied dense
  checkpoint.

These are architecture-specific historical artifacts. Their target rates,
selection sources, and logical execution paths differ. Report the observed mask
retention metadata with all results; do not call it a common 30% FLOPs setting.
The L2 control is a per-module selection baseline, not a matched global-FLOPs
control. This pilot makes no latency or hardware speedup claim.

## Output and next gate

The runner writes immutable per-model CSV/JSON outputs and a cross-model report
under `/data/jixinye25/work1/output/task042_temporal_relation_collapse/`.
The discovery gate is qualitative: look for relation-matrix degradation that
is larger than the magnitude control while clip-level prediction agreement
remains relatively high. A positive screening signal must be followed by a
controlled run that regenerates all pruning masks at a predeclared, measured
FLOPs target and evaluates the full validation split before proposing a new
selector.

If the pruning and magnitude variants both lose most of their clean accuracy
or agreement with Dense, treat lower TRR as indeterminate: it does not isolate
temporal-relation damage from broad representation/prediction collapse. This
pilot applies saved selections without fine-tuning and is not a substitute for
an accuracy-matched comparison.

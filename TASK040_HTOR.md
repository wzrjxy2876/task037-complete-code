# Task040 — Hierarchical Temporal-Order Responsibility (HTOR)

Task040 is an independent Video Swin Transformer diagnostic for UCF101.  It
is developed on the existing Task037 model-loading and pruning-unit semantics.
It is not a replacement for the production pruning pipeline and it does not
run physical pruning, 50% sparsity, optimizer training, or fine-tuning.

## Scientific question

Static functional similarity does not imply temporal-order replaceability.
HTOR asks whether masking one structural pruning unit changes the model's
sensitivity to ordered relations among the actual sampled input frames.

The temporal object is the sampled input clip returned by the existing UCF101
loader, whose model input layout is `[B, C, T, H, W]`.  `T` is read at runtime;
it is not hardcoded.  Task040 requires `T >= 2` and a positive power of two.
It neither crops nor pads a nonconforming clip.

## Hierarchical intervention

At level `l`, block size is `b_l = 2**l`.  Each disjoint adjacent pair of
blocks is swapped while the order inside each block is preserved.  Indices are
zero based and manifest end indices are exclusive.  For a power-of-two `T`,
the levels contain `T/2 + T/4 + ... + 1 = T-1` interventions.

For example, at block size 2:

```text
[0, 1, 2, 3, 4, 5, 6, 7]
        -> [2, 3, 0, 1, 4, 5, 6, 7]
```

Every intervention is an explicit deterministic permutation.  The manifest
and the core tests verify that no frame is lost, duplicated, or permuted
randomly.

## Structural intervention and score

For ground-truth class `y`, `z_y` is the raw true-class logit.  For pruning
unit `i`:

```text
d_i(X)   = z_y(X)   - z_y(X^{-i})
d_i(SX)  = z_y(SX)  - z_y(SX^{-i})

tau_(i,m) = |d_i(X) - d_i(S_m X)|
            / (|d_i(X)| + |d_i(S_m X)| + eps)
```

Attention masking zeros one complete head at the existing projection input;
FFN masking zeros one complete neuron at the existing `fc2` input.  These are
temporary logical hooks and are removed in a `finally` block after every call.
No tensor is physically shrunk.

The scalar calculation is performed in float64 even though model inference is
FP32.  Both raw deletion effects and both raw logits are stored in
`task040_raw_records.csv`.

For level `l` with `N` videos and `M_l` interventions:

```text
H_i^(l) = sqrt( 1/(N*M_l) * sum_v sum_m tau_(i,v,m)^2 )

HTOR_i = sqrt( 1/L * sum_l (H_i^(l))^2 )
```

The RMS is not replaced by a simple average, and every hierarchy level has
equal final weight regardless of its intervention count.

## Scope boundaries

HTOR contains no Contribution Field, no `x * grad`, no `p_average`, no F3,
no BMS, no lambda, no cost coefficient, and no all-pair frame swapping.  The
initial run is a small exact-mask oracle using a balanced 3-class × 3-video
subset (seed 3407) and deterministic representative attention heads and FFN
neurons.  It does not claim scientific success from synthetic tests.

The exact mask oracle stays on one GPU because hook behavior under
`DataParallel` is not silently assumed to be identical.  Temporal batches are
created with GPU tensor indexing and reused for the selected units.

## Outputs of a future real run

The probe writes:

- `task040_checkpoint_identity.json`
- `task040_intervention_manifest.json`
- `task040_video_manifest.csv`
- `task040_raw_records.csv`
- `task040_unit_level_summary.csv`
- `task040_summary.json`

The identity report records checkpoint SHA256, compatible/missing/unexpected
keys, classifier-head status, model parameter counts, and discovered head/FFN
unit counts before the HTOR records are produced.

Stage D is reported as `COMPLETED_UNJUDGED`; reaching the end of the probe is
not a scientific pass. Its interpretation requires inspecting the real HTOR
distributions after the diagnostic run.


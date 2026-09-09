# Task038 SlowFast functional-coverage migration

This package migrates the frozen Task037 Dynamic3D + global BMS + dynamic F3
functional-coverage method to the historical SlowFast ResNet101 topology.

The archived files under `legacy_sources/` are byte-identical architecture and
runtime references. They are not imported as pruning algorithms. Production
code uses the archived factory only for topology and applies the Task038
residual, BN, lateral, descriptor, signed-field, fixed-domain, and logical-mask
semantics.

Pipeline:

1. preflight identity, architecture inventory, data/config and GPU safety;
2. baseline full UCF101 validation;
3. 10-batch CUDA Dynamic3D calibration;
4. one global BMS over all candidate Conv3d output channels;
5. fresh balanced N=9 signed fields, pooled independently to 16x7x7;
6. numerical gates, deterministic prefix replay, and dynamic one-at-a-time F3;
7. fresh-model logical registry, pre-finetune validation, then protocol-locked FT.

Candidate order is Fast res2-res5 (block/conv/channel), Lateral p1/res2-res4,
then Slow res2-res5. Conv3 output units mask both residual branches; downsample
is dependency-tied. No physical shrinking or speed claim is made.

The frozen F3 rule is:
`B=max(p_total,domain_damage)`, `V=max(p_average-B,0)`,
`R_F3=B+V/2`, with ascending ordinal percentile ranks and tie-break
`[R_F3,p_total,p_average,global_index]`. Parameter cost is used only for the
50% stopping budget.

Runtime output is outside Git at `$TASK038_OUTPUT_DIR`.

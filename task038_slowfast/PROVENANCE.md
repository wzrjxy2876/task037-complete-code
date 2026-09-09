# Task038 provenance

Task037 is authoritative for Dynamic3D signed temporal semantics, global robust
D_abs normalization (q=.01/.99), Schur-complement D_rel, global standardized
BMS, signed functional similarity, active-demand coverage, float32
Delta_average/Delta_total arithmetic, float64 ordinal percentile ranks, and
dynamic F3 ordering.

The exact N=9 sampler uses random.Random(3407), sorted eligible class IDs,
three sampled classes, three sampled videos per class, and sorted selected rows
within each class. The complete identity and canonical SHA256 are written to
contribution/n09_sample_identity.json.

The legacy SlowFast files are authoritative only for the
slowfast_16x8_resnet101_kinetics400 topology, stage names, fusion wiring,
checkpoint naming, and existing dataset/training mechanism. The server
authoritative paths and hashes are in
source_identity/authoritative_slowfast_references.json; the Git copies under
legacy_sources/ are immutable archive copies. Legacy InteractionPruner
science is archive-only: no old weighted amplitude formula, cross-layer impact,
per-layer clustering, manifold score, heuristic split, threshold admission, or
physical shrinking is used.

All Task038 candidate units are Conv3d output channels. Analytical costs include
the output filter, affine BN gamma/beta, and a tied downsample output filter/BN
when present. Consumer input weights and BN running statistics are excluded.

Formal fine-tuning is the historical SlowFast recovery recipe with one explicit
user-controlled difference: batch size is 16, overriding the old
myslowfast.py parser default of 4. The matching loader semantics remain
shuffle=True, drop_last=True, 9 workers, and pinned memory; the optimizer is
SGD on requires_grad=True parameters with LR*0.1, momentum=.9, configured
weight decay, CE only, FP32, no AMP, no scheduler, and seed 3407. It is
launched only with physical GPUs 0 and 1 in tmux session MC.

All runtime outputs, caches, logs, and model checkpoints are kept under
/data/jixinye25/work1/output/..., outside this Git worktree.
The authoritative formal training configuration is resolved from
`CONFIG_PATHS['slowfast_resnet101']` in
`/home/jixinye25/jxy_work1/Code/utils.py`, which resolves to
`/home/jixinye25/jxy_work1/Code/config/slowfast_16x8_resnet101_kinetics400.yaml`
(SHA256 `11561e904abee5d6be8f2d2d47dfe5daecf24b15212184a75ccb818dd20d1199`).
Its verified values are `CONFIG.TRAIN.LR=0.005`, effective fine-tune LR
`0.0005`, `CONFIG.TRAIN.W_DECAY=1e-5`, and `CONFIG.TRAIN.EPOCH_NUM=100`;
the same identity is written into `preflight.json` and
`finetune/finetune_config.json`.

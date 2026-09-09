# Task038 SlowFast functional-coverage migration

This package migrates the frozen Task037 Dynamic3D + global BMS + dynamic F3
functional-coverage method to the historical SlowFast ResNet101 topology.

The archived files under legacy_sources/ are byte-identical architecture and
runtime references. The authoritative server copies are recorded by absolute
path and SHA256 in source_identity/authoritative_slowfast_references.json.
They are not imported as pruning algorithms. Production code uses the archived
factory only for topology and applies the Task038 residual, BN, lateral,
descriptor, signed-field, fixed-domain, and logical-mask semantics.

Pipeline:

1. preflight identity, architecture inventory, data/config and GPU safety;
2. baseline full UCF101 validation;
3. 10-batch CUDA Dynamic3D calibration;
4. one global BMS over all candidate Conv3d output channels;
5. fresh balanced N=9 signed fields, pooled independently to 16x7x7;
6. numerical gates, direct-oracle exact prefix replay, and dynamic one-at-a-time F3;
7. fresh-model logical registry, pre-finetune validation, then protocol-locked
   F3 fine-tuning.

Candidate order is Fast res2-res5 (block/conv/channel), Lateral p1/res2-res4,
then Slow res2-res5. Conv3 output units mask both residual branches; downsample
is dependency-tied. No physical shrinking or speed claim is made.

The frozen F3 rule is:
B=max(p_total,domain_damage), V=max(p_average-B,0),
R_F3=B+V/2, with float64 ordinal ranks and exact ascending tie-break
[R_F3,p_total,p_average,global_index]. Raw Delta_average and Delta_total remain
float32. F3 is used only to define the pruning order; parameter accounting is
not a ranking term.

The formal compression target is 50% remaining structural-equivalent trainable
parameters. Define `r_remain = P_structural_remaining / P_original`; the formal
condition is `r_remain ~= 0.50`. The selector reruns the frozen F3 trajectory
from step 0 and the structural shape simulator chooses only between the current
and next adjacent prefix at the target crossing. This is not 50% removed
channels, units, or local output-filter costs. Task038 remains logical pruning:
`actual_state_dict_parameter_ratio` stays 1.0 and physical tensor shrinking is
not executed.

Contribution fields are signed pooled X*d z_y/dX values saved in float32 per
video/unit. They are concatenated in deterministic N=9 sample order and receive
one global per-unit L2 normalization only when loaded for functional similarity.
D_abs robust quantiles are global over all candidate units (q=.01/.99), while
D_rel uses same-layer Schur covariance.

Formal fine-tuning follows the historical myslowfast.py data/optimizer
semantics: shuffle=True, drop_last=True, 9 workers, SGD over trainable
parameters, learning rate cfg.LR*0.1, momentum=.9, weight decay, CE only,
FP32, no AMP, no scheduler, and live tqdm progress. Its legacy default batch
size was 4; the authoritative Task038 formal run explicitly overrides this to
batch size 16 at the user's instruction. Tests and finetune_config.json
record both facts. Only physical GPUs 0 and 1 are used via
CUDA_VISIBLE_DEVICES=0,1 and --gpu-ids 0 1.

Runtime output is outside Git at
/data/jixinye25/work1/output/task038_slowfast_functional_coverage_migration.
The formal LR, weight decay, and epoch count are resolved (not duplicated)
from `CONFIG_PATHS['slowfast_resnet101']` in
`/home/jixinye25/jxy_work1/Code/utils.py` ->
`/home/jixinye25/jxy_work1/Code/config/slowfast_16x8_resnet101_kinetics400.yaml`
(SHA256 `11561e904abee5d6be8f2d2d47dfe5daecf24b15212184a75ccb818dd20d1199`):
LR `0.005`, effective FT LR `0.0005`, weight decay `1e-5`, epochs `100`.

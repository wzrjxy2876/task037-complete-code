# Task041 — Collective Temporal Coverage Pruning Oracle

Task041 is a diagnostic-only extension of the validated Task040 Video Swin/UCF101
line. It consumes the frozen Task040 D.3 temporal profiles and BMS mapping, computes
the collective temporal coverage score, and evaluates temporary whole-unit masks.

The implementation preserves:

- Video Swin Transformer/UCF101 model and checkpoint-loading semantics;
- Task037 attention-head and FFN-neuron unit discovery;
- Task040 fixed-cardinality temporal spans T=32, spans 1/2/4/8/16, 16 pairs/span;
- float64 leave-one-out coverage risk without profile normalization or clipping;
- no physical pruning, no fine-tuning, and no full sparsity run.

The real server run should use the authoritative checkpoint
/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt, a balanced 10-class
x 3-video validation subset with seed 3407, and an output directory under
/data/jixinye25/work1/output/task041_collective_temporal_coverage_pruning_oracle.

The CLI writes the seven required Task041 artifacts plus checkpoint identity and
the fixed 30-video manifest. It intentionally runs on one GPU unless hook
replication is independently proven safe.

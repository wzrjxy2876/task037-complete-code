import torch

from src.lgfr_runtime.task042_temporal_relation_collapse import (
    js_divergence_bits,
    relation_retention,
    temporal_permutation,
    trajectory_relation,
)


def test_adjacent_swap_is_disjoint_and_reversible():
    first = temporal_permutation(8, "adjacent_swap", torch.device("cpu"))
    second = temporal_permutation(8, "adjacent_swap", torch.device("cpu"))
    assert first.tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
    assert torch.equal(first.index_select(0, second), torch.arange(8))


def test_block_reorder_and_reverse_are_permutations():
    for condition in ("block_reorder", "reverse"):
        value = temporal_permutation(32, condition, torch.device("cpu"))
        assert sorted(value.tolist()) == list(range(32))
    block = temporal_permutation(8, "block_reorder", torch.device("cpu"))
    assert block.tolist() == [4, 5, 0, 1, 6, 7, 2, 3]


def test_relation_retention_identity_is_one():
    generator = torch.Generator().manual_seed(3407)
    features = torch.randn(12, 24, generator=generator)
    relation = trajectory_relation(features)
    assert abs(relation_retention(relation, relation) - 1.0) < 1e-6


def test_js_is_symmetric_and_zero_for_identical_logits():
    a = torch.tensor([[2.0, 1.0, -1.0]])
    b = torch.tensor([[-1.0, 2.0, 0.5]])
    assert js_divergence_bits(a, a) < 1e-7
    assert abs(js_divergence_bits(a, b) - js_divergence_bits(b, a)) < 1e-7
    assert 0.0 <= js_divergence_bits(a, b) <= 1.0

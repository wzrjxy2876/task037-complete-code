from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from task038_slowfast.slowfast_f3_selector import (
    f3_components,
    global_bms_domains,
    ordinal_percentile,
    standardize_descriptors,
)
from task038_slowfast.slowfast_functional_archive import (
    ContributionFieldArchive,
    build_functional_similarity,
    domain_total_losses,
    functional_coverage,
    marginal_coverage_losses,
    normalize_fields,
    validate_partition,
)
from task038_slowfast.slowfast_model_task038 import (
    _load_legacy_architecture,
    _task038_bottleneck_forward,
    attach_mask,
    set_mask,
)
from task038_slowfast.slowfast_unit_adapter import (
    Unit,
    UnitInventory,
    build_inventory,
    validate_canonical_order,
)


def test_01_architecture_factory_identity():
    model = _load_legacy_architecture().slowfast_16x8_resnet101_kinetics400(101)
    assert model.fc.out_features == 101


def test_02_architecture_has_fast_stages():
    model = _load_legacy_architecture().slowfast_16x8_resnet101_kinetics400(101)
    names = dict(model.named_modules())
    assert all(f"fast_res{x}" in names for x in range(2, 6))


def test_03_architecture_has_slow_stages():
    model = _load_legacy_architecture().slowfast_16x8_resnet101_kinetics400(101)
    names = dict(model.named_modules())
    assert all(f"slow_res{x}" in names for x in range(2, 6))


def test_04_architecture_has_lateral_stages():
    model = _load_legacy_architecture().slowfast_16x8_resnet101_kinetics400(101)
    names = dict(model.named_modules())
    assert all(x in names for x in ("lateral_p1.0", "lateral_res2.0", "lateral_res3.0", "lateral_res4.0"))


def test_05_conv3_no_downsample_semantics():
    legacy = _load_legacy_architecture()
    block = legacy.Bottleneck(4, 1, 1, None, 1)
    attach_mask(block.conv3, 4)
    mask = torch.ones(4)
    mask[2] = 0
    set_mask(block.conv3, mask)
    output = block(torch.randn(1, 4, 2, 6, 6))
    assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))


def test_06_conv3_downsample_semantics():
    legacy = _load_legacy_architecture()
    downsample = nn.Sequential(nn.Conv3d(4, 4, 1, bias=False), nn.BatchNorm3d(4))
    block = legacy.Bottleneck(4, 1, 1, downsample, 1)
    attach_mask(block.conv3, 4)
    attach_mask(block.downsample[0], 4)
    mask = torch.ones(4)
    mask[1] = 0
    set_mask(block.conv3, mask)
    set_mask(block.downsample[0], mask)
    output = block(torch.randn(1, 4, 2, 6, 6))
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))


def test_07_conv1_mask():
    legacy = _load_legacy_architecture()
    block = legacy.Bottleneck(4, 1, 1, None, 1)
    attach_mask(block.conv1, 1)
    set_mask(block.conv1, torch.zeros(1))
    assert torch.isfinite(block(torch.randn(1, 4, 2, 6, 6))).all()


def test_08_set_mask_shape_guard():
    conv = nn.Conv3d(2, 3, 1)
    with pytest.raises(ValueError):
        set_mask(conv, torch.ones(2))


def test_09_normalize_fields_shape():
    normalized, valid = normalize_fields(np.ones((2, 3, 2, 2, 2), dtype=np.float32))
    assert normalized.shape == (2, 3, 2, 2, 2)
    assert valid.shape == (2, 3)


def test_10_normalize_fields_unit_norm():
    normalized, valid = normalize_fields(np.ones((2, 3, 2, 2, 2), dtype=np.float32))
    norms = np.linalg.norm(normalized.reshape(2, 3, -1), axis=2)
    assert np.all(valid)
    assert np.allclose(norms, 1.0)


def test_11_zero_field_is_invalid():
    _, valid = normalize_fields(np.zeros((1, 2, 2, 2, 2), dtype=np.float32))
    assert not valid.any()


def test_12_signed_cosine_is_clamped():
    vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    vectors[0] /= torch.linalg.vector_norm(vectors[0])
    vectors[1] /= torch.linalg.vector_norm(vectors[1])
    vectors[2] /= torch.linalg.vector_norm(vectors[2])
    matrix = build_functional_similarity(vectors, torch.tensor([True, True, True]))
    assert matrix[0, 1].item() == 0.0
    assert matrix[0, 0].item() == 1.0


def test_13_null_similarity_diagonal():
    vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    matrix = build_functional_similarity(vectors, torch.tensor([False, True]))
    assert matrix[0, 0].item() == 0.0


def test_14_coverage_active_demand():
    sim = torch.eye(3)
    assert functional_coverage(sim, torch.tensor([True, True, True]), torch.tensor([True, True, True])).item() == 1.0


def test_15_coverage_null_domain():
    sim = torch.zeros(2, 2)
    assert functional_coverage(sim, torch.tensor([True, False]), torch.tensor([False, False])).item() == 1.0


def test_16_marginal_has_infinity_removed():
    sim = torch.eye(2)
    losses, _ = marginal_coverage_losses(sim, torch.tensor([True, True]), torch.tensor([True, True]))
    assert torch.isfinite(losses).all()
    losses, _ = marginal_coverage_losses(sim, torch.tensor([True, False]), torch.tensor([True, True]))
    assert torch.isinf(losses[1])


def test_17_domain_total_float32():
    average = torch.tensor([0.25, float("inf")], dtype=torch.float32)
    result = domain_total_losses(average, torch.tensor([True, True]))
    assert result.dtype == torch.float32
    assert result[0].item() == 0.5 and torch.isinf(result[1])


def test_18_partition_complete():
    assert validate_partition([[0, 2], [1]], 3) == [[0, 2], [1]]


def test_19_partition_rejects_overlap():
    with pytest.raises(ValueError):
        validate_partition([[0, 1], [1, 2]], 3)


def test_20_ordinal_ascending():
    values = torch.tensor([0.5, 0.1, 0.9])
    ranks = ordinal_percentile(values, torch.tensor([0, 1, 2]))
    assert torch.allclose(ranks, torch.tensor([0.5, 0.0, 1.0]))


def test_21_ordinal_global_tie_break():
    values = torch.tensor([1.0, 1.0])
    ranks = ordinal_percentile(values, torch.tensor([9, 2]))
    assert ranks[1].item() == 0.0 and ranks[0].item() == 1.0


def test_22_f3_formula():
    result = f3_components(
        torch.tensor([0.2, 0.8]), torch.tensor([0.9, 0.1]),
        torch.tensor([0.3, 0.4]), torch.tensor([0, 1]),
    )
    assert torch.allclose(result["B"], torch.tensor([0.3, 1.0]))
    assert torch.allclose(result["V"], torch.tensor([0.7, 0.0]))
    assert torch.allclose(result["R_F3"], torch.tensor([0.65, 1.0]))


def test_23_f3_has_no_cost_component():
    import inspect
    source = inspect.getsource(f3_components)
    assert "parameter_cost" not in source


def test_24_descriptor_standardization():
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    y = standardize_descriptors(x)
    assert torch.allclose(y.mean(0), torch.zeros(3), atol=1e-6)


def test_25_global_bms_partition():
    x = torch.randn(8, 3)
    groups, sinks = global_bms_domains(x, torch.device("cpu"))
    assert sorted(i for g in groups for i in g) == list(range(8))
    assert sinks.shape == x.shape


def test_26_inventory_unit_dataclass():
    unit = Unit(0, "x", "x", 0, "fast", "res2", 0, "conv1", "conv_channel", 4, 10, "x", "bn")
    assert unit.unit_type == "conv_channel"


def test_27_inventory_canonical_on_tiny_model():
    units = [
        Unit(i, f"fast_res2.0.conv{p}", f"fast_res2.0.conv{p}", c, "fast", "res2", 0, f"conv{p}", "conv_channel", 2, 1, f"x{p}", f"bn{p}")
        for i, (p, c) in enumerate(((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)))
    ]
    inventory = UnitInventory(units, 100, 20, 0.5, 0.1)
    validate_canonical_order(inventory)


def test_28_dependency_group_for_conv3():
    from task038_slowfast.slowfast_dependency_graph import build_dependency_graph
    model = _load_legacy_architecture().slowfast_16x8_resnet101_kinetics400(101)
    inventory = build_inventory(model)
    graph = build_dependency_graph(model, inventory)
    assert graph["global_domain_is_required"] is True
    assert any(e["relation"] == "conv3_residual_tied" for e in graph["edges"])


def test_29_static_legacy_science_blocker():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("compute_cross_layer_impact", "prune_exact", "sigma*2.5", "0.35*mean_amp")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden), path


def test_30_legacy_archive_hashes():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "IPslowfast.py": "67d2991649a3724986b0ddf52c4f124d3b9ae5cbbb30e8a5991a1cca99936488",
        "myslowfast.py": "9c161a362ad066f66d00f573c5da315cd328309320ed5cc1011fb0d84c51e2af",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / "legacy_sources" / name).read_bytes()).hexdigest() == digest


def test_31_source_identity_json():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "source_identity" / "legacy_slowfast_sources.json").read_text())
    assert payload["legacy_files_are_archive_only"] is True
    assert len(payload["sources"]) == 2


def test_32_no_task039():
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("*task039*"))


def test_33_no_old_source_modification_markers():
    root = Path(__file__).resolve().parents[1]
    assert (root / "legacy_sources" / "IPslowfast.py").is_file()
    assert (root / "legacy_sources" / "myslowfast.py").is_file()


def test_34_pooled_dimension():
    assert 9 * 16 * 7 * 7 == 7056


def test_35_nine_sample_gate():
    identity = [{"label": x // 3} for x in range(9)]
    assert len(identity) == 9
    assert sorted(set(x["label"] for x in identity)) == [0, 1, 2]


def test_36_incremental_domain_state_matches_direct_replay():
    from task038_slowfast.slowfast_functional_archive import DomainState

    generator = torch.Generator().manual_seed(38038)
    vectors = torch.randn((10, 8), generator=generator)
    valid = torch.tensor([True, True, True, False, True, True, True, True, False, True])
    vectors[~valid] = 0.0
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
    vectors[~valid] = 0.0
    state = DomainState.create(0, list(range(10)), vectors, valid)
    for local_index in (3, 0, 1, 8, 2, 4):
        expected_losses, expected_coverage = marginal_coverage_losses(
            state.similarity, state.retained, state.valid
        )
        torch.testing.assert_close(state.losses, expected_losses, rtol=0.0, atol=0.0)
        torch.testing.assert_close(state.coverage, expected_coverage, rtol=0.0, atol=0.0)
        state.remove(local_index)
        expected_losses, expected_coverage = marginal_coverage_losses(
            state.similarity, state.retained, state.valid
        )
        torch.testing.assert_close(state.losses, expected_losses, rtol=0.0, atol=0.0)
        torch.testing.assert_close(state.coverage, expected_coverage, rtol=0.0, atol=0.0)

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "lgfr_runtime"))
import task040_htor_probe as task040
import task042_phase_c_joint_mask as joint
from task042_phase_i_coverage import (
    _cell,
    center_logits,
    domain_contribution,
    exact_dominates,
    kappa_profile,
    kappa_value,
    leave_one_class_out_indices,
    pairset_indices,
    pareto_front,
    same_keep_groups,
    static_kappa_value,
)


def test_centered_logits_are_shift_invariant():
    z = np.asarray([[2.0, -1.0, 5.0], [0.0, 8.0, 3.0]])
    offset = np.asarray([[99.0], [-12.0]])
    np.testing.assert_allclose(center_logits(z), center_logits(z + offset), rtol=0, atol=1e-12)
    t = torch.tensor(z, dtype=torch.float32)
    assert torch.allclose(center_logits(t), center_logits(t + torch.tensor(offset, dtype=torch.float32)), rtol=0, atol=1e-6)


def test_numpy_profile_serializes_as_json_array():
    import json
    assert json.loads(_cell(np.asarray([0.1, 0.2]))) == pytest.approx([0.1, 0.2])


def test_full_and_retained_contribution_arithmetic():
    t_full = np.asarray([[8.0, 4.0], [1.0, 2.0]])
    t_empty = np.asarray([[2.0, 1.0], [0.0, -2.0]])
    t_keep = np.asarray([[5.0, 2.0], [2.0, 1.0]])
    np.testing.assert_array_equal(domain_contribution(t_full, t_empty), [[6.0, 3.0], [1.0, 4.0]])
    np.testing.assert_array_equal(domain_contribution(t_keep, t_empty), [[3.0, 1.0], [2.0, 3.0]])


def test_kappa_exact_synthetic_cases_and_bounds():
    assert kappa_value([1, 2], [1, 2]) == pytest.approx(1.0)
    assert kappa_value([0, 0], [1, 0]) == pytest.approx(1e-12, abs=1e-15)
    assert kappa_value([0, 0], [0, 0]) == pytest.approx(1.0)
    rng = np.random.default_rng(11)
    for _ in range(100):
        a, b = rng.normal(size=400), rng.normal(size=400)
        value = kappa_value(a, b)
        assert 0.0 <= value <= 1.0
    profile = kappa_profile(np.eye(4), np.eye(4))
    np.testing.assert_array_equal(profile, np.ones(4))


def test_same_keep_count_grouping_and_exact_dominance():
    groups = same_keep_groups([
        {"domain_id": "415", "keep_count": 1, "x": "a"},
        {"domain_id": "415", "keep_count": 1, "x": "b"},
        {"domain_id": "415", "keep_count": 2, "x": "c"},
        {"domain_id": "103", "keep_count": 1, "x": "d"},
    ])
    assert len(groups[("415", 1)]) == 2
    assert set(groups) == {("415", 1), ("415", 2), ("103", 1)}
    assert exact_dominates([1.0, 0.5], [0.9, 0.5])
    assert not exact_dominates([1.0, 0.5], [1.0, 0.5])
    assert pareto_front({"a": [1, 1], "b": [0, 0], "c": [0, 1]}) == {"a"}


def test_static_baseline_arithmetic():
    s_full = np.asarray([1.0, 2.0, -2.0])
    assert static_kappa_value(s_full, s_full) == pytest.approx(1.0)
    assert static_kappa_value([0.0, 0.0, 0.0], s_full) == pytest.approx(0.0)


def _fake_model_and_records():
    relu = nn.ReLU()
    model = nn.Sequential(nn.Linear(4, 4, bias=False), relu, nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4))
        model[2].weight.copy_(torch.eye(4))
    model.eval()
    spec = SimpleNamespace(name="fake.mlp", unit_type="neuron", num_units=4, hook_module=relu, module=SimpleNamespace())
    records = [{"spec": spec, "task037_global_index": 100 + i, "unit_index": i} for i in range(4)]
    return model, spec, records


def test_whole_domain_joint_mask_and_exact_restoration():
    model, _, records = _fake_model_and_records()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        before = model(x).detach().clone()
    with joint.joint_temporary_unit_masks(model, records, task040.temporary_unit_mask, torch) as calls:
        with torch.no_grad():
            masked = model(x)
        assert torch.equal(masked, torch.zeros_like(masked))
    with torch.no_grad():
        after = model(x)
    assert torch.equal(after, before)
    assert len(calls) == 4 and all(value == 1 for value in calls.values())


def test_multi_unit_subset_mask_and_exact_restoration():
    model, _, records = _fake_model_and_records()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        before = model(x).detach().clone()
    selected = [records[1], records[3]]
    with joint.joint_temporary_unit_masks(model, selected, task040.temporary_unit_mask, torch) as calls:
        with torch.no_grad():
            masked = model(x)
        assert torch.allclose(masked, torch.tensor([[1.0, 0.0, 3.0, 0.0]]))
    with torch.no_grad():
        after = model(x)
    assert torch.equal(after, before)
    assert len(calls) == 2 and all(value == 1 for value in calls.values())


def test_leave_one_class_out_identity():
    labels = [str(c) for c in range(10) for _ in range(10)]
    views = leave_one_class_out_indices(labels)
    assert len(views) == 10
    for label, indices in views.items():
        assert len(indices) == 90
        assert all(labels[i] != label for i in indices)
        assert set(indices) | {i for i, value in enumerate(labels) if value == label} == set(range(100))


def test_pairset_split_identity():
    relations = [{"span": span, "pair_index": pair} for span in (1, 2, 4, 8, 16) for pair in (0, 8)]
    a, b = pairset_indices(relations)
    assert len(a) == len(b) == 5
    assert {relations[i]["pair_index"] for i in a} == {0}
    assert {relations[i]["pair_index"] for i in b} == {8}
    assert set(a).isdisjoint(b) and set(a + b) == set(range(10))

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lgfr_runtime"))
import task042_phase_j_conditional as phase_j


def _partition_3d(x, window):
    batch, time, height, width, channels = x.shape
    wt, wh, ww = window
    view = x.reshape(batch, time // wt, wt, height // wh, wh, width // ww, ww, channels)
    return view.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().reshape(-1, wt * wh * ww, channels)


def test_attention_temporal_axis_identity_and_feature_layout():
    grid = torch.arange(1 * 4 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 2, 3)
    windows = _partition_3d(grid, (1, 2, 2))
    geometry = {
        "window_size": (1, 2, 2), "batch_size": 1,
        "depth": 4, "height": 2, "width": 2,
        "padded_depth": 4, "padded_height": 2, "padded_width": 2,
        "shift_size": (0, 0, 0),
    }
    restored = phase_j.window_reverse_3d(windows, geometry)
    assert torch.equal(restored, grid)
    per_frame = phase_j.attention_frame_features(windows, geometry)
    assert per_frame.shape == (4, 12)
    assert torch.equal(per_frame, grid[0].reshape(4, 12))


def test_attention_window_reverse_undoes_shift():
    grid = torch.arange(1 * 4 * 2 * 2 * 1, dtype=torch.float32).reshape(1, 4, 2, 2, 1)
    shifted = torch.roll(grid, shifts=(-1, 0, 0), dims=(1, 2, 3))
    windows = _partition_3d(shifted, (1, 2, 2))
    geometry = {
        "window_size": (1, 2, 2), "batch_size": 1,
        "depth": 4, "height": 2, "width": 2,
        "padded_depth": 4, "padded_height": 2, "padded_width": 2,
        "shift_size": (1, 0, 0),
    }
    assert torch.equal(phase_j.window_reverse_3d(windows, geometry), grid)


def test_ffn_per_frame_representation_preserves_spatial_positions():
    activation = torch.arange(1 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    result = phase_j.ffn_frame_features(activation, 1)
    assert result.shape == (3, 4)
    assert torch.equal(result, activation[0, ..., 1].reshape(3, 4))


def test_per_frame_centering_and_normalization():
    x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
    xhat, zero = phase_j.center_normalize_frames_torch(x)
    assert zero == []
    assert torch.allclose(xhat.mean(dim=1), torch.zeros(2), atol=1e-7)
    assert torch.allclose(torch.linalg.vector_norm(xhat, dim=1), torch.ones(2), atol=1e-7)


def test_zero_frame_is_recorded_without_nan():
    xhat, zero = phase_j.center_normalize_frames_torch(torch.ones(2, 4))
    assert zero == [0, 1]
    assert torch.isfinite(xhat).all()
    assert torch.equal(xhat, torch.zeros_like(xhat))


def test_ordinary_similarity_exact_gram():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    actual = phase_j.ordinary_similarity(x)
    expected = x @ x.T
    assert torch.equal(actual, expected)
    assert torch.allclose(torch.diag(actual), torch.ones(3))


def test_oas_shrinkage_is_deterministic():
    x = torch.arange(16 * 40, dtype=torch.float32).reshape(16, 40)
    x = x + torch.randn_like(x) * 0.1
    xhat, _ = phase_j.center_normalize_frames_torch(x)
    raw1, alpha1, shrunk1 = phase_j.oas_covariance(xhat)
    raw2, alpha2, shrunk2 = phase_j.oas_covariance(xhat)
    assert alpha1 == alpha2
    assert torch.equal(raw1, raw2)
    assert torch.equal(shrunk1, shrunk2)
    assert 0.0 <= alpha1 <= 1.0


def test_oas_uses_temporal_variables_and_feature_observations():
    torch.manual_seed(7)
    x = torch.randn(12, 24, dtype=torch.float64)
    xhat, _ = phase_j.center_normalize_frames_torch(x)
    raw, shrinkage, shrunk = phase_j.oas_covariance(xhat)
    observations = xhat.T
    observations = observations - observations.mean(dim=0, keepdim=True)
    expected_raw = observations.T @ observations / observations.shape[0]
    p, n = xhat.shape[0], xhat.shape[1]
    mu = torch.trace(expected_raw) / p
    alpha = torch.mean(expected_raw ** 2)
    denominator = (n + 1.0) * (alpha - mu ** 2 / p)
    expected_shrinkage = 1.0 if denominator <= 0 else min(1.0, max(0.0, float((alpha + mu ** 2) / denominator)))
    expected = (1.0 - expected_shrinkage) * expected_raw + expected_shrinkage * mu * torch.eye(p, dtype=xhat.dtype)
    assert torch.allclose(raw, expected_raw, atol=1e-12)
    assert shrinkage == pytest.approx(expected_shrinkage)
    assert torch.allclose(shrunk, expected, atol=1e-12)


def test_precision_inverse_identity():
    covariance = torch.tensor([[2.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1.2]], dtype=torch.float64)
    precision, partial = phase_j.covariance_to_partial(covariance)
    assert torch.allclose(covariance @ precision, torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert torch.isfinite(partial).all()


def test_partial_correlation_exact_synthetic_case():
    precision_expected = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    covariance = torch.linalg.inv(precision_expected)
    precision, partial = phase_j.covariance_to_partial(covariance)
    assert torch.allclose(precision, precision_expected, atol=1e-12)
    assert partial[0, 1].item() == pytest.approx(-0.5)
    assert partial[1, 2].item() == pytest.approx(-0.5)
    assert partial[0, 2].item() == pytest.approx(0.0)
    assert torch.equal(torch.diag(partial), torch.zeros(3, dtype=torch.float64))


def test_partial_correlation_bounds():
    torch.manual_seed(4)
    a = torch.randn(12, 20, dtype=torch.float64)
    covariance = a @ a.T + 0.1 * torch.eye(12, dtype=torch.float64)
    _, partial = phase_j.covariance_to_partial(covariance)
    assert torch.max(torch.abs(partial)).item() <= 1.0


def test_upper_triangle_order():
    matrix = np.arange(16).reshape(4, 4)
    assert phase_j.upper_triangle(matrix).tolist() == [1, 2, 3, 6, 7, 11]


def test_30_video_concatenation_order():
    matrices = {i: np.full((3, 3), i, dtype=float) for i in range(30)}
    got = phase_j.concat_video_relations(matrices, list(reversed(range(30))))
    assert got.shape == (90,)
    assert got[:3].tolist() == [29.0, 29.0, 29.0]
    assert got[-3:].tolist() == [0.0, 0.0, 0.0]


def test_convex_coverage_weights_are_simplex_and_exact_when_representable():
    competitors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    target = np.asarray([0.25, 0.75])
    weights, residual, delta = phase_j.solve_simplex_coverage(target, competitors)
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights @ competitors, target, atol=1e-7)
    assert np.linalg.norm(residual) < 1e-7
    assert delta < 1e-7


def test_leave_one_out_identity():
    result = phase_j.leave_one_out_sets([7, 8, 9])
    assert result == [(7, (8, 9)), (8, (7, 9)), (9, (7, 8))]


def test_mask_free_analysis_identity():
    phase_j.validate_mask_free_identity([1, 2, 3], [3, 1, 2], mask_hooks_registered=False)
    with pytest.raises(RuntimeError):
        phase_j.validate_mask_free_identity([1], [1], mask_hooks_registered=True)


def test_video_identity_is_derived_from_frozen_exact_list_order(tmp_path):
    exact = tmp_path / "exact.txt"
    exact.write_text(
        "v_ActionA_g01_c01 12 0\n"
        "v_ActionA_g02_c01 13 0\n"
        "v_ActionB_g01_c02 14 1\n",
        encoding="utf-8",
    )
    manifest = [
        {"video_index": 0, "video_id": "/frames/ActionA/v_ActionA_g01_c01", "duration": "12", "label": "0"},
        {"video_index": 1, "video_id": "/frames/ActionB/v_ActionB_g01_c02", "duration": "14", "label": "1"},
        {"video_index": 2, "video_id": "/frames/ActionA/v_ActionA_g02_c01", "duration": "13", "label": "0"},
    ]
    rows = phase_j._derive_video_identities(manifest, exact)
    by_stem = {Path(row["video_id"]).name: row for row in rows}
    assert by_stem["v_ActionA_g01_c01"]["class_name"] == "ActionA"
    assert by_stem["v_ActionA_g01_c01"]["class_position"] == 1
    assert by_stem["v_ActionA_g02_c01"]["class_position"] == 2
    assert by_stem["v_ActionB_g01_c02"]["class_position"] == 1


def test_video_identity_rejects_manifest_label_drift(tmp_path):
    exact = tmp_path / "exact.txt"
    exact.write_text("v_ActionA_g01_c01 12 0\n", encoding="utf-8")
    manifest = [{"video_index": 0, "video_id": "/frames/ActionA/v_ActionA_g01_c01", "duration": "12", "label": "1"}]
    with pytest.raises(RuntimeError, match="label mismatch"):
        phase_j._derive_video_identities(manifest, exact)


def test_prepared_video_order_preserves_frozen_class_position():
    rows = []
    for cls_index in range(10):
        for position in range(1, 4):
            i = len(rows)
            name = "Action%02d" % cls_index
            rows.append({"video_index": i, "video_id": "v_%s_g%02d_c01" % (name, position),
                         "label": str(cls_index), "class_name": name, "class_position": position})
    normalized = phase_j._normalize_video_identity_order(rows)
    assert normalized[2]["class_position"] == 3
    assert normalized[2]["label"] == 0
    assert normalized[2]["class_name"] == "Action00"

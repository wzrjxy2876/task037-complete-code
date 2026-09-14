#!/usr/bin/env python3
"""Task042 Phase J: conditional frame-relation representation diagnosis.

No unit masks are installed. For one selected unit/video, frame features have
shape [T, D_i], where T is the model-stage temporal axis and D_i flattens the
stage's spatial positions and the selected unit's feature channels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BRANCH = "task_042_post_bms_frame_relation_redundancy"
DOMAINS = ("415", "103", "76", "271")
EXPECTED_DOMAIN_UNITS = {
    "415": (779, 1549, 1553),
    "103": (254, 2334, 16627),
    "76": (133, 33306, 36376),
    "271": (328, 1611, 7845, 30209),
}
CHECKPOINT_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DEFAULT_REPO = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
DEFAULT_BASE = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
EPS = 1e-12


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 Phase-J gate failed: " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in fields})


def _cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center_normalize_frames_torch(x: Any) -> tuple[Any, list[int]]:
    """Center each [D_i] frame row and normalize over D_i; input/output [T,D_i]."""
    import torch
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError("frame features must be a torch [T,D_i] tensor")
    centered = x - x.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    near_zero = torch.nonzero(norms[:, 0] <= EPS, as_tuple=False).flatten().tolist()
    normalized = centered / norms.clamp_min(EPS)
    return normalized, [int(i) for i in near_zero]


def window_reverse_3d(windows: Any, geometry: Mapping[str, Any]) -> Any:
    """Restore [B*nW,N,D_i] windows to [B,T,H,W,D_i], undoing shift and padding."""
    import torch
    if windows.ndim != 3:
        raise ValueError("attention windows must have shape [B*nW,N,D_i]")
    wd, wh, ww = [int(x) for x in geometry["window_size"]]
    batch = int(geometry["batch_size"])
    dp, hp, wp = (int(geometry[k]) for k in ("padded_depth", "padded_height", "padded_width"))
    channels = int(windows.shape[-1])
    if int(windows.shape[1]) != wd * wh * ww:
        raise ValueError("attention token axis does not equal the recorded 3D window volume")
    x = windows.reshape(batch, dp // wd, hp // wh, wp // ww, wd, wh, ww, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().reshape(batch, dp, hp, wp, channels)
    shift = tuple(int(v) for v in geometry["shift_size"])
    if any(shift):
        x = torch.roll(x, shifts=shift, dims=(1, 2, 3))
    depth, height, width = (int(geometry[k]) for k in ("depth", "height", "width"))
    return x[:, :depth, :height, :width, :].contiguous()


def attention_frame_features(per_head_windows: Any, geometry: Mapping[str, Any]) -> Any:
    """Map one head's [B*nW,N,head_dim] output to [T,H*W*head_dim]."""
    restored = window_reverse_3d(per_head_windows, geometry)
    if restored.shape[0] != 1:
        raise ValueError("Phase J uses one video per forward")
    return restored[0].reshape(restored.shape[1], -1)


def ffn_frame_features(activation: Any, neuron_index: int) -> Any:
    """Select post-GELU/pre-fc2 neuron from [B,T,H,W,M] as [T,H*W]."""
    if activation.ndim != 5 or activation.shape[0] != 1:
        raise ValueError("FFN activation must be [1,T,H,W,M]")
    if not 0 <= int(neuron_index) < int(activation.shape[-1]):
        raise IndexError("FFN neuron index is outside the activation width")
    return activation[0, ..., int(neuron_index)].reshape(activation.shape[1], -1)


def ordinary_similarity(xhat: Any) -> Any:
    """Cosine/Gram matrix from unit-normalized frame rows [T,D_i]."""
    return xhat @ xhat.transpose(0, 1)


def oas_covariance(xhat: Any) -> tuple[Any, float, Any]:
    """OAS covariance over T frame variables using D_i repeated feature observations.

    xhat is [T,D_i]. The observation matrix is [D_i,T], so each flattened
    within-frame feature is one observation and each sampled time is a variable.
    """
    import torch
    if xhat.ndim != 2:
        raise ValueError("OAS input must be [T,D_i]")
    t_count, feature_dim = map(int, xhat.shape)
    if t_count < 2 or feature_dim < 2:
        raise ValueError("OAS needs at least two frame variables and two observations")
    observations = xhat.transpose(0, 1)
    observations = observations - observations.mean(dim=0, keepdim=True)
    raw = observations.transpose(0, 1) @ observations / float(feature_dim)
    raw = (raw + raw.transpose(0, 1)) * 0.5
    p = float(t_count)
    trace = torch.trace(raw)
    mu = trace / p
    alpha = torch.mean(raw * raw)
    numerator = alpha + mu * mu
    denominator = (feature_dim + 1.0) * (alpha - (mu * mu / p))
    if not bool(torch.isfinite(denominator)) or float(denominator.detach().cpu()) <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = min(1.0, max(0.0, float((numerator / denominator).detach().cpu())))
    shrunk = (1.0 - shrinkage) * raw + shrinkage * mu * torch.eye(t_count, device=xhat.device, dtype=xhat.dtype)
    shrunk = (shrunk + shrunk.transpose(0, 1)) * 0.5
    return raw, float(shrinkage), shrunk


def covariance_to_partial(covariance: Any) -> tuple[Any, Any]:
    """Return precision and signed partial-correlation matrix for [T,T] covariance."""
    import torch
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("conditional covariance must be square [T,T]")
    precision = torch.linalg.inv(covariance)
    diagonal = torch.diagonal(precision)
    denom = torch.sqrt(torch.clamp(diagonal[:, None] * diagonal[None, :], min=0.0))
    partial = -precision / denom
    partial.fill_diagonal_(0.0)
    if bool(torch.any(torch.abs(partial) > 1.0 + 1e-5)):
        raise RuntimeError("partial-correlation magnitude exceeded one beyond numerical tolerance")
    return precision, torch.clamp(partial, min=-1.0, max=1.0)


def upper_triangle(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("upper-triangle input must be square")
    return array[np.triu_indices(array.shape[0], k=1)]


def concat_video_relations(matrices: Mapping[int, np.ndarray], video_order: Sequence[int]) -> np.ndarray:
    """Concatenate upper triangles in the exact given video-index order."""
    if not video_order or any(int(v) not in matrices for v in video_order):
        raise ValueError("relation matrices do not cover the requested ordered video list")
    return np.concatenate([upper_triangle(matrices[int(v)]).astype(np.float64, copy=False) for v in video_order])


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= EPS:
        return 1.0 if np.linalg.norm(a - b) <= EPS else 0.0
    return float(np.dot(a, b) / den)


def _corr(left: Sequence[float], right: Sequence[float], method: str) -> float | None:
    from scipy import stats
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return None
    with np.errstate(all="ignore"):
        if method == "pearson":
            value = stats.pearsonr(a, b)[0]
        elif method == "spearman":
            value = stats.spearmanr(a, b)[0]
        elif method == "kendall":
            value = stats.kendalltau(a, b, variant="b")[0]
        else:
            raise ValueError("unknown correlation method")
    return None if not np.isfinite(value) else float(value)


def _rank_fraction(values: Sequence[float]) -> np.ndarray:
    from scipy.stats import rankdata
    return (rankdata(np.asarray(values, dtype=np.float64), method="average") - 1.0) / max(len(values) - 1, 1)


def _top_indices(values: Sequence[float], k: int) -> list[int]:
    vals = np.asarray(values, dtype=np.float64)
    return sorted(range(vals.size), key=lambda i: (-float(vals[i]), i))[: max(0, min(int(k), vals.size))]


def partial_matrix_stats(raw: Any, shrunk: Any, precision: Any, partial: Any, shrinkage: float) -> dict[str, Any]:
    import torch
    raw_eigs = torch.linalg.eigvalsh(raw)
    shrunk_eigs = torch.linalg.eigvalsh(shrunk)

    def condition(eigs: Any) -> float:
        low, high = float(eigs[0].detach().cpu()), float(eigs[-1].detach().cpu())
        return math.inf if low <= 0.0 else high / low

    inverse_finite = bool(torch.isfinite(precision).all())
    part = partial.detach().cpu().numpy()
    return {
        "raw_min_eigenvalue": float(raw_eigs[0].detach().cpu()),
        "raw_condition_number": condition(raw_eigs),
        "shrunk_min_eigenvalue": float(shrunk_eigs[0].detach().cpu()),
        "shrunk_condition_number": condition(shrunk_eigs),
        "finite_inverse": inverse_finite,
        "max_abs_partial": float(np.max(np.abs(part))),
        "nan_count": int(np.isnan(part).sum()),
        "inf_count": int(np.isinf(part).sum()),
        "oas_shrinkage": float(shrinkage),
    }


def solve_simplex_coverage(target: Sequence[float], competitors: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Convex least-squares approximation with nonnegative simplex weights."""
    from scipy.optimize import minimize
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows = np.asarray(competitors, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != y.size or rows.shape[0] < 1:
        raise ValueError("competitor relation vectors must be [M,feature_count]")
    gram, cross = rows @ rows.T, rows @ y
    start = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=np.float64)

    def objective(alpha: np.ndarray) -> float:
        return float(alpha @ gram @ alpha - 2.0 * alpha @ cross + y @ y)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return 2.0 * (gram @ alpha - cross)

    result = minimize(
        objective, start, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * rows.shape[0],
        constraints=[{"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0),
                      "jac": lambda a: np.ones_like(a)}],
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success and np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-6:
        raise RuntimeError("simplex coverage optimizer failed: " + str(result.message))
    alpha = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    alpha /= alpha.sum()
    reconstructed = alpha @ rows
    residual = y - reconstructed
    delta = float(np.linalg.norm(residual) / (np.linalg.norm(y) + EPS))
    return alpha, residual, delta


def leave_one_out_sets(unit_ids: Sequence[int]) -> list[tuple[int, tuple[int, ...]]]:
    ids = tuple(int(x) for x in unit_ids)
    return [(uid, tuple(other for other in ids if other != uid)) for uid in ids]


def validate_mask_free_identity(expected_ids: Sequence[int], observed_ids: Sequence[int], mask_hooks_registered: bool) -> None:
    if mask_hooks_registered:
        raise RuntimeError("Phase J must be mask-free")
    if set(map(int, expected_ids)) != set(map(int, observed_ids)) or len(observed_ids) != len(expected_ids):
        raise RuntimeError("mask-free capture did not preserve the exact frozen unit identity set")


class PhaseJCapture:
    """Read-only target-unit hooks; it never changes model outputs or installs masks."""
    def __init__(self, model: Any, rows: Sequence[Mapping[str, Any]], specs: Sequence[Any], torch: Any):
        self.model, self.rows, self.torch = model, list(rows), torch
        self.specs = {(str(s.name), str(s.unit_type)): s for s in specs}
        self.by_layer: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: {"head": [], "neuron": []})
        for row in rows:
            self.by_layer[str(row["layer"])][str(row["capture_kind"])].append(row)
        self.handles: list[Any] = []
        self.features: dict[int, Any] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        self.grids: dict[str, tuple[int, ...]] = {}
        self.qkv_v: dict[str, Any] = {}
        modules = dict(model.named_modules())
        for layer, kinds in self.by_layer.items():
            spec = self.specs[(layer, "head" if kinds["head"] else "neuron")]
            block_name = layer.rsplit(".", 1)[0]
            require(block_name in modules, "unit parent Swin block is missing: " + block_name)
            self.handles.append(modules[block_name].register_forward_pre_hook(self._grid_hook(block_name)))
            if kinds["head"]:
                module = spec.module
                self.handles.append(module.qkv.register_forward_hook(self._qkv_hook(layer, module)))
                self.handles.append(module.attn_drop.register_forward_hook(self._attn_hook(layer, module, kinds["head"])))
            if kinds["neuron"]:
                self.handles.append(spec.module.act.register_forward_hook(self._ffn_hook(layer, kinds["neuron"])))

    def _grid_hook(self, block_name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            x = inputs[0]
            if x.ndim != 5:
                raise RuntimeError("Swin block input must be [B,T,H,W,C]")
            self.grids[block_name] = tuple(int(d) for d in x.shape)
        return hook

    def _qkv_hook(self, layer: str, module: Any):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            heads, head_dim = int(module.num_heads), int(module.head_dim)
            if output.ndim != 3 or int(output.shape[-1]) != 3 * heads * head_dim:
                raise RuntimeError("unexpected QKV tensor shape at " + layer)
            self.qkv_v[layer] = output.reshape(output.shape[0], output.shape[1], 3, heads, head_dim).permute(2, 0, 3, 1, 4)[2]
        return hook

    def _attn_hook(self, layer: str, module: Any, target_rows: Sequence[Mapping[str, Any]]):
        def hook(_module: Any, _inputs: Any, weights: Any) -> None:
            if layer not in self.qkv_v:
                raise RuntimeError("attention value tensor is missing at " + layer)
            per_head = self.torch.matmul(weights, self.qkv_v[layer])
            geometry = getattr(module, "_pruning_geometry", None)
            if geometry is None:
                raise RuntimeError("recorded Swin window geometry is missing at " + layer)
            b, t_count, height, width, channels = self.grids[layer.rsplit(".", 1)[0]]
            for row in target_rows:
                uid, head = int(row["task037_global_index"]), int(row["unit_index"])
                head_windows = per_head[:, head, :, :]
                self.metadata[uid] = {
                    "activation_tensor_shape": list(head_windows.shape),
                    "activation_layout": "[window_batch,window_tokens,head_dim]",
                    "T_i": t_count, "H_i": height, "W_i": width,
                    "stage_feature_channels": channels,
                    "feature_dimension": height * width * int(module.head_dim),
                    "head_dim": int(module.head_dim),
                    "grid_shape": [b, t_count, height, width, channels],
                }
                if self.collect_features:
                    restored = window_reverse_3d(head_windows, geometry)
                    if int(restored.shape[0]) != 1:
                        raise RuntimeError("Phase J expects batch-size-one video inference")
                    self.features[uid] = restored[0].reshape(t_count, -1)
        return hook

    def _ffn_hook(self, layer: str, target_rows: Sequence[Mapping[str, Any]]):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            b, t_count, height, width, channels = self.grids[layer.rsplit(".", 1)[0]]
            if output.ndim != 5 or tuple(output.shape[:4]) != (b, t_count, height, width):
                raise RuntimeError("post-GELU FFN activation must be [B,T,H,W,M]")
            for row in target_rows:
                uid, neuron = int(row["task037_global_index"]), int(row["unit_index"])
                feature = ffn_frame_features(output, neuron)
                self.metadata[uid] = {
                    "activation_tensor_shape": list(output[0, ..., neuron].shape),
                    "activation_layout": "[T,H,W] selected post-GELU neuron",
                    "full_layer_activation_shape": list(output.shape),
                    "T_i": t_count, "H_i": height, "W_i": width,
                    "stage_feature_channels": channels, "feature_dimension": height * width,
                    "head_dim": "", "grid_shape": [b, t_count, height, width, channels],
                }
                if self.collect_features:
                    self.features[uid] = feature
        return hook

    def begin(self, collect_features: bool) -> None:
        self.features.clear()
        self.metadata.clear()
        self.grids.clear()
        self.qkv_v.clear()
        self.collect_features = bool(collect_features)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _torch_relation(x: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[int]]:
    import torch
    xhat, near_zero = center_normalize_frames_torch(x)
    g = ordinary_similarity(xhat)
    raw, shrinkage, shrunk = oas_covariance(xhat)
    precision, conditional = covariance_to_partial(shrunk)
    stats = partial_matrix_stats(raw, shrunk, precision, conditional, shrinkage)
    matrices = (g.detach().to(dtype=torch.float64).cpu().numpy(),
                conditional.detach().to(dtype=torch.float64).cpu().numpy())
    require(np.isfinite(matrices[0]).all() and np.isfinite(matrices[1]).all(),
            "ordinary/conditional relation matrix contains NaN or Inf")
    require(stats["finite_inverse"] and stats["nan_count"] == 0 and stats["inf_count"] == 0,
            "precision inversion produced a nonfinite conditional matrix")
    return matrices[0], matrices[1], stats, near_zero


def _manifest_order(videos: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(row["video_index"]) for row in sorted(videos, key=lambda r: int(r["video_index"]))]


def _relation_rows(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]], units: Sequence[Mapping[str, Any]],
                   videos_by_index: Mapping[int, Mapping[str, Any]], t_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordinary_rows, conditional_rows, comparison_rows = [], [], []
    tri = np.triu_indices(t_count, k=1)
    for unit in units:
        uid = int(unit["task037_global_index"])
        for vi in sorted(matrices[uid]):
            g, c = matrices[uid][vi]["G"], matrices[uid][vi]["C"]
            video = videos_by_index[vi]
            gv, cv = g[tri], c[tri]
            n_top = max(1, int(math.ceil(.10 * gv.size)))
            top_g, top_c = set(_top_indices(np.abs(gv), n_top)), set(_top_indices(np.abs(cv), n_top))
            metadata = {
                "task037_global_index": uid, "domain_id": unit["domain_id"],
                "video_index": vi, "video_id": video["video_id"],
                "class_name": video.get("class_name", ""), "label": video["label"],
                "class_position": video["class_position"],
            }
            comparison_rows.append({
                **metadata, "pearson_G_vs_C": _corr(gv, cv, "pearson"),
                "spearman_G_vs_C": _corr(gv, cv, "spearman"),
                "kendall_G_vs_C": _corr(gv, cv, "kendall"),
                "top_abs_relation_overlap_count": len(top_g & top_c),
                "top_abs_relation_overlap_fraction": len(top_g & top_c) / float(n_top),
                "top_k": n_top, "edge_count": int(gv.size),
            })
            for i in range(t_count):
                for j in range(t_count):
                    base = {**metadata, "frame_t": i, "frame_t_prime": j}
                    ordinary_rows.append({**base, "ordinary_G": float(g[i, j])})
                    conditional_rows.append({**base, "conditional_C": float(c[i, j])})
    return ordinary_rows, conditional_rows, comparison_rows


def _indirect_cases(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]], units: Sequence[Mapping[str, Any]],
                    videos_by_index: Mapping[int, Mapping[str, Any]], t_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tri = np.triu_indices(t_count, k=1)
    edge_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*tri))}
    for unit in units:
        uid = int(unit["task037_global_index"])
        for vi, pair in matrices[uid].items():
            g, c = pair["G"], pair["C"]
            rg, rc = _rank_fraction(np.abs(g[tri])), _rank_fraction(np.abs(c[tri]))
            for a in range(t_count):
                for b in range(a + 1, t_count):
                    endpoint = edge_index[(a, b)]
                    paths = []
                    for mid in range(t_count):
                        if mid in (a, b):
                            continue
                        i1, i2 = edge_index[(min(a, mid), max(a, mid))], edge_index[(min(mid, b), max(mid, b))]
                        paths.append((min(float(rg[i1]), float(rg[i2])), -mid, mid, i1, i2))
                    strongest = max(paths)
                    video = videos_by_index[vi]
                    rows.append({
                        "task037_global_index": uid, "domain_id": unit["domain_id"],
                        "video_index": vi, "video_id": video["video_id"],
                        "class_name": video.get("class_name", ""),
                        "frame_t1": a, "frame_t2": strongest[2], "frame_t3": b,
                        "G_rank_t1_t2": float(rg[strongest[3]]), "G_rank_t2_t3": float(rg[strongest[4]]),
                        "G_rank_t1_t3": float(rg[endpoint]), "absC_rank_t1_t3": float(rc[endpoint]),
                        "endpoint_rank_drop_G_minus_absC": float(rg[endpoint] - rc[endpoint]),
                        "indirect_path_weakest_G_rank": float(strongest[0]),
                        "interpretation": "exhaustively ranked candidate; no hard threshold",
                    })
    rows.sort(key=lambda r: (-r["endpoint_rank_drop_G_minus_absC"],
                             -r["indirect_path_weakest_G_rank"], r["task037_global_index"],
                             r["video_index"], r["frame_t1"], r["frame_t3"]))
    return rows[: min(100, len(rows))]


def _relation_contrast_cases(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]],
                             units: Sequence[Mapping[str, Any]],
                             videos_by_index: Mapping[int, Mapping[str, Any]],
                             t_count: int, per_direction: int = 10) -> list[dict[str, Any]]:
    """Rank frame pairs whose relative strength changes most from |G| to |C|."""
    rows: list[dict[str, Any]] = []
    tri = np.triu_indices(t_count, k=1)
    for unit in units:
        uid = int(unit["task037_global_index"])
        for vi, pair in matrices[uid].items():
            g, c = pair["G"], pair["C"]
            abs_g, abs_c = np.abs(g[tri]), np.abs(c[tri])
            rank_g, rank_c = _rank_fraction(abs_g), _rank_fraction(abs_c)
            video = videos_by_index[vi]
            for category, ordering in (
                ("strong_G_weak_C", sorted(range(len(rank_g)), key=lambda k: (-(rank_g[k] - rank_c[k]), k))),
                ("weak_or_moderate_G_strong_C", sorted(range(len(rank_g)), key=lambda k: (-(rank_c[k] - rank_g[k]), k))),
            ):
                for edge in ordering[: min(per_direction, len(ordering))]:
                    t1, t2 = int(tri[0][edge]), int(tri[1][edge])
                    rows.append({
                        "task037_global_index": uid, "domain_id": unit["domain_id"],
                        "video_index": vi, "video_id": video["video_id"], "class_name": video.get("class_name", ""),
                        "category": category, "frame_t": t1, "frame_t_prime": t2,
                        "G": float(g[t1, t2]), "C": float(c[t1, t2]),
                        "abs_G_rank_fraction": float(rank_g[edge]),
                        "abs_C_rank_fraction": float(rank_c[edge]),
                        "rank_contrast_G_minus_C": float(rank_g[edge] - rank_c[edge]),
                        "selection": "top rank contrast within unit/video; descriptive, no threshold",
                    })
    return rows


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"min": float(a.min()), "q25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "q75": float(np.quantile(a, .75)), "max": float(a.max())}


def _video_stability(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]],
                     units: Sequence[Mapping[str, Any]], videos: Sequence[Mapping[str, Any]],
                     video_order: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_by = {int(v["video_index"]): v for v in videos}
    classes = sorted({str(v["class_name"]) for v in videos})
    for unit in units:
        uid = int(unit["task037_global_index"])
        vectors = {vi: upper_triangle(matrices[uid][vi]["C"]) for vi in video_order}
        for comparison in ("same_class", "different_class"):
            pair_metrics: dict[str, list[float]] = {m: [] for m in ("pearson", "spearman", "cosine")}
            pair_count = 0
            for pos, a in enumerate(video_order):
                for b in video_order[pos + 1:]:
                    same = str(video_by[a]["label"]) == str(video_by[b]["label"])
                    if same != (comparison == "same_class"):
                        continue
                    pair_count += 1
                    for metric in pair_metrics:
                        value = cosine_similarity(vectors[a], vectors[b]) if metric == "cosine" else _corr(vectors[a], vectors[b], metric)
                        if value is not None:
                            pair_metrics[metric].append(value)
            row: dict[str, Any] = {"row_type": "video_pair_summary", "task037_global_index": uid,
                                   "domain_id": unit["domain_id"], "comparison": comparison,
                                   "pair_count": pair_count}
            for metric, values in pair_metrics.items():
                summary = _metric_summary(values) if values else {}
                for key, value in summary.items():
                    row[metric + "_" + key] = value
            rows.append(row)
        # Position prototypes concatenate one video per class in fixed class order.
        prototypes: dict[int, np.ndarray] = {}
        for position in (1, 2, 3):
            selected = [next(vi for vi in video_order if str(video_by[vi]["class_name"]) == cls
                             and int(video_by[vi]["class_position"]) == position) for cls in classes]
            prototypes[position] = np.concatenate([vectors[vi] for vi in selected])
        for p1, p2 in ((1, 2), (1, 3), (2, 3)):
            a, b = prototypes[p1], prototypes[p2]
            rows.append({"row_type": "class_balanced_prototype", "task037_global_index": uid,
                         "domain_id": unit["domain_id"], "comparison": "P%d_vs_P%d" % (p1, p2),
                         "pair_count": len(classes), "pearson": _corr(a, b, "pearson"),
                         "spearman": _corr(a, b, "spearman"), "cosine": cosine_similarity(a, b)})
        for p1, p2, held in ((1, 2, 3), (1, 3, 2), (2, 3, 1)):
            averaged = []
            for cls in classes:
                vi1 = next(vi for vi in video_order if str(video_by[vi]["class_name"]) == cls
                           and int(video_by[vi]["class_position"]) == p1)
                vi2 = next(vi for vi in video_order if str(video_by[vi]["class_name"]) == cls
                           and int(video_by[vi]["class_position"]) == p2)
                averaged.append((vectors[vi1] + vectors[vi2]) * .5)
            pair_proto = np.concatenate(averaged)
            other = prototypes[held]
            rows.append({"row_type": "class_balanced_pair_subset", "task037_global_index": uid,
                         "domain_id": unit["domain_id"], "comparison": "P%d%d_vs_P%d" % (p1, p2, held),
                         "pair_count": len(classes), "pearson": _corr(pair_proto, other, "pearson"),
                         "spearman": _corr(pair_proto, other, "spearman"),
                         "cosine": cosine_similarity(pair_proto, other)})
    return rows


def _distance_geometry(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]],
                       units: Sequence[Mapping[str, Any]], video_order: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    vectors: dict[int, dict[str, np.ndarray]] = {}
    for unit in units:
        uid = int(unit["task037_global_index"])
        vectors[uid] = {
            "cond": concat_video_relations({vi: matrices[uid][vi]["C"] for vi in video_order}, video_order),
            "cos": concat_video_relations({vi: matrices[uid][vi]["G"] for vi in video_order}, video_order),
        }
    geometry_rows, details, summaries = [], [], {}
    unit_by_id = {int(u["task037_global_index"]): u for u in units}
    for domain in DOMAINS:
        ids = [int(u["task037_global_index"]) for u in units if str(u["domain_id"]) == domain]
        dcond, dcos = {}, {}
        for left in ids:
            for right in ids:
                dcond[left, right] = 1.0 - cosine_similarity(vectors[left]["cond"], vectors[right]["cond"])
                dcos[left, right] = 1.0 - cosine_similarity(vectors[left]["cos"], vectors[right]["cos"])
                geometry_rows.append({"domain_id": domain, "unit_a": left, "unit_b": right,
                                      "d_cond": dcond[left, right], "d_cos": dcos[left, right]})
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        if not pairs:
            continue
        vc, vg = [dcond[p] for p in pairs], [dcos[p] for p in pairs]
        rc, rg = _rank_fraction(vc), _rank_fraction(vg)
        mat_c = np.asarray([[dcond[a, b] for b in ids] for a in ids])
        mat_g = np.asarray([[dcos[a, b] for b in ids] for a in ids])
        norm_c, norm_g = np.linalg.norm(mat_c), np.linalg.norm(mat_g)
        neighbors = {}
        changes = 0
        for uid in ids:
            cn = min((x for x in ids if x != uid), key=lambda x: (dcond[uid, x], x))
            gn = min((x for x in ids if x != uid), key=lambda x: (dcos[uid, x], x))
            changed = cn != gn
            changes += int(changed)
            neighbors[str(uid)] = {"conditional": cn, "ordinary_cosine": gn, "changed": changed}
        summaries[domain] = {
            "within_domain_pair_spearman_d_cos_vs_d_cond": _corr(vg, vc, "spearman"),
            "within_domain_pair_pearson_d_cos_vs_d_cond": _corr(vg, vc, "pearson"),
            "distance_matrix_normalized_frobenius_difference": float(np.linalg.norm(mat_c / max(norm_c, EPS) - mat_g / max(norm_g, EPS))),
            "nearest_neighbor_changes": changes, "nearest_neighbors": neighbors, "unit_count": len(ids),
        }
        for k, (a, b) in enumerate(pairs):
            details.append({"row_type": "domain_pair_rank", "domain_id": domain, "unit_a": a, "unit_b": b,
                            "d_cos": dcos[a, b], "d_cond": dcond[a, b],
                            "rank_d_cos": float(rg[k]), "rank_d_cond": float(rc[k]),
                            "rank_difference_cond_minus_cos": float(rc[k] - rg[k]),
                            "nearest_neighbor_changed_for_a": neighbors[str(a)]["changed"],
                            "nearest_neighbor_changed_for_b": neighbors[str(b)]["changed"]})
    mixed = [u for u in units if str(u["domain_id"]) == "271"]
    mixed_pair_values: dict[str, list[tuple[float, float]]] = {
        "Attention-Attention": [], "FFN-FFN": [], "Attention-FFN": []}
    for i, left in enumerate(mixed):
        for right in mixed[i + 1:]:
            lt, rt = str(left["unit_type"]), str(right["unit_type"])
            pair_type = "Attention-Attention" if lt == rt == "attention_head" else (
                "FFN-FFN" if lt == rt == "ffn_neuron" else "Attention-FFN")
            a, b = int(left["task037_global_index"]), int(right["task037_global_index"])
            mixed_pair_values[pair_type].append((dcond[a, b], dcos[a, b]))
            details.append({"row_type": "domain_271_type_pair", "domain_id": "271", "unit_a": a, "unit_b": b,
                            "unit_type_a": unit_by_id[a]["unit_type"], "unit_type_b": unit_by_id[b]["unit_type"],
                            "pair_type": pair_type, "d_cond": dcond[a, b], "d_cos": dcos[a, b]})
    for pair_type, values in mixed_pair_values.items():
        details.append({
            "row_type": "domain_271_pair_type_summary", "domain_id": "271",
            "pair_type": pair_type, "pair_count": len(values),
            "mean_d_cond": float(np.mean([x[0] for x in values])) if values else "",
            "median_d_cond": float(np.median([x[0] for x in values])) if values else "",
            "mean_d_cos": float(np.mean([x[1] for x in values])) if values else "",
            "median_d_cos": float(np.median([x[1] for x in values])) if values else "",
        })
    return geometry_rows, details, summaries


def _coverability(matrices: Mapping[int, Mapping[int, Mapping[str, Any]]], units: Sequence[Mapping[str, Any]],
                  video_order: Sequence[int], t_count: int, key: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows, residual_maps = [], []
    tri = np.triu_indices(t_count, k=1)
    for domain in DOMAINS:
        members = [u for u in units if str(u["domain_id"]) == domain]
        ids = [int(u["task037_global_index"]) for u in members]
        for unit in members:
            uid = int(unit["task037_global_index"])
            target = concat_video_relations({vi: matrices[uid][vi][key] for vi in video_order}, video_order)
            others = [x for x in ids if x != uid]
            competitors = np.stack([
                concat_video_relations({vi: matrices[x][vi][key] for vi in video_order}, video_order)
                for x in others])
            alpha, residual, delta = solve_simplex_coverage(target, competitors)
            residual_by_video = residual.reshape(len(video_order), len(tri[0]))
            maps = np.zeros((len(video_order), t_count, t_count), dtype=np.float64)
            for idx, vec in enumerate(residual_by_video):
                maps[idx][tri] = vec
                maps[idx] += maps[idx].T
            rows.append({"task037_global_index": uid, "domain_id": domain, "unit_type": unit["unit_type"],
                         "competitor_ids": others, "simplex_weights": alpha.tolist(),
                         "weights_nonnegative": bool(np.all(alpha >= -1e-12)),
                         "weights_sum": float(alpha.sum()), "residual_norm": float(np.linalg.norm(residual)),
                         "target_norm": float(np.linalg.norm(target)), "delta": delta})
            residual_maps.append(maps)
    return rows, np.stack(residual_maps, axis=0)


def _descriptor_complementarity(units: Sequence[Mapping[str, Any]],
                                geometry_rows: Sequence[Mapping[str, Any]],
                                cover_cond: Sequence[Mapping[str, Any]],
                                cover_cos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    geo = {(str(r["domain_id"]), int(r["unit_a"]), int(r["unit_b"])): r for r in geometry_rows}
    dc = {int(r["task037_global_index"]): float(r["delta"]) for r in cover_cond}
    dg = {int(r["task037_global_index"]): float(r["delta"]) for r in cover_cos}
    out = []
    for domain in DOMAINS:
        members = [u for u in units if str(u["domain_id"]) == domain]
        pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1:]]
        for name in ("D_abs", "D_rel", "D_st"):
            gaps = [abs(float(a[name]) - float(b[name])) for a, b in pairs]
            d_cond = [float(geo[domain, int(a["task037_global_index"]), int(b["task037_global_index"])]["d_cond"]) for a, b in pairs]
            d_cos = [float(geo[domain, int(a["task037_global_index"]), int(b["task037_global_index"])]["d_cos"]) for a, b in pairs]
            vals = [float(u[name]) for u in members]
            out.append({"domain_id": domain, "descriptor_dimension": name, "analysis_type": "pairwise_geometry",
                        "pearson_d_cond_vs_descriptor_gap": _corr(d_cond, gaps, "pearson"),
                        "spearman_d_cond_vs_descriptor_gap": _corr(d_cond, gaps, "spearman"),
                        "pearson_d_cos_vs_descriptor_gap": _corr(d_cos, gaps, "pearson"),
                        "spearman_d_cos_vs_descriptor_gap": _corr(d_cos, gaps, "spearman")})
            unit_c = [dc[int(u["task037_global_index"])] for u in members]
            unit_g = [dg[int(u["task037_global_index"])] for u in members]
            out.append({"domain_id": domain, "descriptor_dimension": name, "analysis_type": "leave_one_out_coverability",
                        "pearson_d_cond_vs_descriptor_gap": _corr(unit_c, vals, "pearson"),
                        "spearman_d_cond_vs_descriptor_gap": _corr(unit_c, vals, "spearman"),
                        "pearson_d_cos_vs_descriptor_gap": _corr(unit_g, vals, "pearson"),
                        "spearman_d_cos_vs_descriptor_gap": _corr(unit_g, vals, "spearman")})
    return out


def _coverability_comparison(cond_rows: Sequence[Mapping[str, Any]],
                             cos_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from scipy.stats import rankdata
    c = {int(r["task037_global_index"]): float(r["delta"]) for r in cond_rows}
    g = {int(r["task037_global_index"]): float(r["delta"]) for r in cos_rows}
    ids = sorted(c)
    cr, gr = rankdata([c[i] for i in ids]), rankdata([g[i] for i in ids])
    c_order, g_order = sorted(ids, key=lambda i: (c[i], i)), sorted(ids, key=lambda i: (g[i], i))
    rows = [{"row_type": "unit", "task037_global_index": uid, "delta_cond": c[uid], "delta_cos": g[uid],
             "rank_cond": int(cr[k]), "rank_cos": int(gr[k]),
             "rank_difference_cond_minus_cos": int(cr[k] - gr[k]),
             "extreme_change": (c_order[0] == uid) != (g_order[0] == uid) or
                               (c_order[-1] == uid) != (g_order[-1] == uid)} for k, uid in enumerate(ids)]
    rows.append({"row_type": "overall", "spearman_delta_cond_vs_delta_cos": _corr([c[i] for i in ids], [g[i] for i in ids], "spearman"),
                 "kendall_delta_cond_vs_delta_cos": _corr([c[i] for i in ids], [g[i] for i in ids], "kendall"),
                 "conditional_min_uid": c_order[0], "cosine_min_uid": g_order[0],
                 "conditional_max_uid": c_order[-1], "cosine_max_uid": g_order[-1]})
    return rows


def _task042_runtime(repo: Path) -> tuple[Any, Any, Any]:
    if str(repo / "src" / "lgfr_runtime") not in sys.path:
        sys.path.insert(0, str(repo / "src" / "lgfr_runtime"))
    import task042_frame_relation_redundancy as base
    import task042_phase_i_coverage as phase_i
    return base, phase_i, base._runtime_modules(repo)[2]


def _derive_video_identities(video_rows: Sequence[Mapping[str, Any]], exact_video_list: Path) -> list[dict[str, Any]]:
    """Join class/position identity from the frozen authoritative video list.

    The Task042 video manifest intentionally stores only video path, duration,
    and numeric label. The exact list is the canonical ordering and contains
    the UCF class in each video stem; derive the within-class position from
    that frozen order instead of inventing fields in the manifest.
    """
    exact_by_stem: dict[str, dict[str, Any]] = {}
    class_counts: defaultdict[str, int] = defaultdict(int)
    class_labels: dict[str, str] = {}
    lines = [line.strip() for line in Path(exact_video_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        require(len(fields) == 3, "malformed exact Task042 video-list row %d" % line_number)
        stem, duration, label = fields
        match = re.fullmatch(r"v_(.+)_g\d+_c\d+", stem)
        require(match is not None, "cannot derive UCF class from exact video id: " + stem)
        class_name = match.group(1)
        require(stem not in exact_by_stem, "duplicate video in exact Task042 video list: " + stem)
        require(class_name not in class_labels or class_labels[class_name] == label,
                "numeric label is inconsistent within exact-list class " + class_name)
        class_labels[class_name] = label
        class_counts[class_name] += 1
        exact_by_stem[stem] = {
            "duration": duration,
            "label": label,
            "class_name": class_name,
            "class_position": class_counts[class_name],
        }

    require(len(video_rows) == len(exact_by_stem), "video manifest and exact Task042 list have different counts")
    manifest_by_stem = {Path(str(row["video_id"])).name: row for row in video_rows}
    require(len(manifest_by_stem) == len(video_rows), "duplicate basename in Task042 video manifest")
    require(set(manifest_by_stem) == set(exact_by_stem), "video manifest differs from the exact Task042 video list")

    joined: list[dict[str, Any]] = []
    for row in video_rows:
        stem = Path(str(row["video_id"])).name
        exact = exact_by_stem[stem]
        require(str(row["duration"]) == str(exact["duration"]), "duration mismatch for frozen video " + stem)
        require(str(row["label"]) == str(exact["label"]), "label mismatch for frozen video " + stem)
        require(Path(str(row["video_id"])).parent.name == exact["class_name"],
                "video path class differs from exact-list identity for " + stem)
        joined.append({**dict(row), "class_name": exact["class_name"],
                       "class_position": exact["class_position"]})
    return joined


def _prepare(repo: Path, base_root: Path, phase_root: Path) -> dict[str, Any]:
    require(not phase_root.exists(), "Phase-J output already exists; refusing overwrite")
    _, _, _ = _task042_runtime(repo)
    cfg_path = base_root / "task042_run_config.json"
    require(cfg_path.is_file(), "frozen Task042 run config is missing")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    require(cfg.get("required_branch") == BRANCH, "wrong frozen branch")
    require(Path(cfg["repo_root"]).resolve() == repo.resolve(), "run config repo path mismatch")
    require(sha256_file(Path(cfg["checkpoint_path"])) == CHECKPOINT_SHA, "checkpoint SHA mismatch")
    for key, path_key in (("profile", "profile_path"), ("descriptors", "descriptor_path"),
                          ("unit_mapping", "unit_mapping_path"), ("val_list", "val_list")):
        expected_sha = cfg.get("input_sha256", {}).get(key)
        require(expected_sha and sha256_file(Path(cfg[path_key])) == expected_sha,
                "frozen Task042 input hash mismatch: " + key)
    all_units = read_csv(Path(cfg["unit_manifest"]))
    selected = [row for row in all_units if str(row["domain_id"]) in DOMAINS]
    selected.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    ids = [int(row["task037_global_index"]) for row in selected]
    expected = {uid for domain in DOMAINS for uid in EXPECTED_DOMAIN_UNITS[domain]}
    require(len(ids) == len(set(ids)) and set(ids) == expected, "four-domain Task037 unit set differs from frozen identities")
    task037_mapping = read_csv(Path(cfg["unit_mapping_path"]))
    source_by_id = {int(row["global_index"]): row for row in task037_mapping}
    require(expected.issubset(source_by_id), "Task037 contribution mapping lacks selected global indices")
    for row in selected:
        for key in ("task037_global_index", "layer", "stage", "unit_type", "unit_index", "domain_id", "D_abs", "D_rel", "D_st"):
            require(key in row and row[key] != "", "identity/descriptor field missing: " + key)
        require(str(row["unit_type"]) in ("attention_head", "ffn_neuron"), "unexpected pruning-unit type")
        uid = int(row["task037_global_index"])
        source = source_by_id[uid]
        require(str(row["layer"]) == str(source["layer"]) and str(row["unit_type"]) == str(source["unit_type"])
                and int(row["unit_index"]) == int(source["unit_index"]),
                "Task042 unit identity differs from the exact Task037 mapping for global index " + str(uid))
        match = re.search(r"layers\.(\d+)\.", str(source["layer"]))
        require(match is not None and int(row["stage"]) == int(match.group(1)),
                "Task037 layer/stage identity mismatch for global index " + str(uid))
        require(uid in EXPECTED_DOMAIN_UNITS[str(row["domain_id"])],
                "Task037 unit domain assignment differs from frozen selected-domain identity")
    videos = _derive_video_identities(read_csv(Path(cfg["video_manifest"])), Path(cfg["exact_video_list"]))
    classes = {str(v["class_name"]) for v in videos}
    require(len(videos) == 30 and len(classes) == 10, "authoritative Task042 cohort must be 10 classes by 3 videos")
    require(all(sum(str(v["class_name"]) == cls for v in videos) == 3 for cls in classes), "cohort is not balanced 3 videos per class")
    require(all(str(v.get("class_position", "")) in ("1", "2", "3") for v in videos), "within-class position identity missing")
    video_indices = [int(v["video_index"]) for v in videos]
    require(sorted(video_indices) == list(range(30)), "video manifest indices are not exactly 0..29")
    phase_root.mkdir(parents=True)
    run_cfg = {
        "task": "TASK042 PHASE J — CONDITIONAL FRAME-RELATION DIAGNOSTIC ONLY",
        "required_branch": BRANCH, "repo_root": str(repo.resolve()), "base_output": str(base_root.resolve()),
        "phase_root": str(phase_root.resolve()), "checkpoint_path": cfg["checkpoint_path"],
        "checkpoint_sha256": CHECKPOINT_SHA, "project_root": cfg["project_root"],
        "unit_manifest": cfg["unit_manifest"], "video_manifest": cfg["video_manifest"],
        "exact_video_list": cfg["exact_video_list"], "frame_root": cfg["frame_root"], "val_list": cfg["val_list"],
        "unit_count": len(selected), "video_count": len(videos),
        "domains": {d: [int(u["task037_global_index"]) for u in selected if str(u["domain_id"]) == d] for d in DOMAINS},
        "unit_identities": [{k: r[k] for k in ("task037_global_index", "layer", "stage", "unit_type", "unit_index", "domain_id")} for r in selected],
        "video_identity_order": [{k: v[k] for k in ("video_index", "video_id", "label", "class_name", "class_position")}
                                 for v in sorted(videos, key=lambda x: int(x["video_index"]))],
        "input_sha256": {"unit_manifest": sha256_file(Path(cfg["unit_manifest"])),
                         "video_manifest": sha256_file(Path(cfg["video_manifest"])),
                         "profile": cfg["input_sha256"]["profile"],
                         "descriptors": cfg["input_sha256"]["descriptors"],
                         "unit_mapping": cfg["input_sha256"]["unit_mapping"],
                         "val_list": cfg["input_sha256"]["val_list"],
                         "exact_video_list": sha256_file(Path(cfg["exact_video_list"])),
                         "checkpoint": CHECKPOINT_SHA},
        "dtype": "float32", "amp": False, "mask_free": True,
        "oas": "analytic Oracle Approximating Shrinkage; observations=flattened feature components; variables=temporal positions",
        "temporal_axis_gate_precedes_relation_construction": True,
    }
    write_json(phase_root / "task042_phase_j_run_config.json", run_cfg)
    return run_cfg


def _load_model_and_specs(config: Mapping[str, Any], gpu: int) -> tuple[Any, Any, Any, Any, Any, list[Any], dict[str, Any]]:
    import torch
    require(gpu in (0, 1), "only physical GPUs 0 and 1 are allowed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "process must be isolated to authorized physical GPU")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker must see exactly one isolated GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3407 + gpu)
    torch.cuda.manual_seed_all(3407 + gpu)
    device = torch.device("cuda:0")
    _, phase_i, _ = _task042_runtime(Path(config["repo_root"]))
    base_cfg = json.loads((Path(config["base_output"]) / "task042_run_config.json").read_text(encoding="utf-8"))
    core, task040, ctfrs, model, specs, identity = phase_i._model(base_cfg, device)
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA and
            identity.get("classifier_head", {}).get("status") == "loaded", "checkpoint/classifier identity mismatch")
    require(not identity.get("missing_keys") and not identity.get("unexpected_keys") and not identity.get("shape_mismatches"),
            "checkpoint did not load exactly")
    model.eval()
    model.requires_grad_(False)
    return torch, core, task040, ctfrs, model, specs, identity


def _load_all_videos(config: Mapping[str, Any], workers: int) -> tuple[Any, dict[int, Any], dict[int, Mapping[str, Any]]]:
    import task042_frame_relation_redundancy as base
    _, _, ctfrs = _task042_runtime(Path(config["repo_root"]))
    base_cfg = json.loads((Path(config["base_output"]) / "task042_run_config.json").read_text(encoding="utf-8"))
    loader = base._get_loader(ctfrs, base_cfg, workers=workers)
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    by_name = {Path(str(v["video_id"])).name: v for v in config["video_identity_order"]}
    clips, rows = {}, {}
    for batch in loader:
        require(int(batch[0].shape[0]) == 1 and int(batch[0].shape[2]) == 32, "loader did not produce one 32-frame clip")
        local_index = int(batch[2][0].item())
        name = Path(str(dataset.clips[local_index][0])).name
        require(name in by_name, "loader returned a video outside the frozen manifest: " + name)
        row = by_name[name]
        vi = int(row["video_index"])
        require(vi not in clips and int(batch[1][0].item()) == int(row["label"]), "duplicate video or label mismatch")
        clips[vi], rows[vi] = batch[0][0], row
    expected = {int(v["video_index"]) for v in config["video_identity_order"]}
    require(set(clips) == expected, "loader did not reproduce all exact 30 videos")
    return dataset, clips, rows


def _capture_specs(units: Sequence[Mapping[str, Any]], specs: Sequence[Any]) -> list[dict[str, Any]]:
    by_key = {(str(s.name), str(s.unit_type)): s for s in specs}
    rows = []
    for unit in units:
        kind = "head" if unit["unit_type"] == "attention_head" else "neuron"
        require((str(unit["layer"]), kind) in by_key, "unit layer missing from authoritative capture")
        row = dict(unit)
        row["capture_kind"] = kind
        rows.append(row)
    return rows


def _shape_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], bool]:
    import torch
    torchmod, _, _, _, model, specs, _ = _load_model_and_specs(config, int(args.gpu))
    units = [u for u in read_csv(Path(config["unit_manifest"])) if str(u["domain_id"]) in DOMAINS]
    cap_rows = _capture_specs(units, specs)
    capture = PhaseJCapture(model, cap_rows, specs, torchmod)
    _, clips, video_rows = _load_all_videos(config, workers=2)
    audit, stable_per_uid = [], {}
    expected_ids = [int(r["task037_global_index"]) for r in cap_rows]
    for vi in sorted(clips):
        clip = clips[vi].to(device="cuda:0", dtype=torch.float32, non_blocking=True)
        capture.begin(collect_features=False)
        with torchmod.inference_mode():
            _ = model(clip.unsqueeze(0))
        validate_mask_free_identity(expected_ids, list(capture.metadata), mask_hooks_registered=False)
        for unit in cap_rows:
            uid = int(unit["task037_global_index"])
            meta = dict(capture.metadata[uid])
            stable_keys = ("T_i", "H_i", "W_i", "feature_dimension", "activation_layout", "activation_tensor_shape")
            if uid in stable_per_uid:
                require(all(stable_per_uid[uid][k] == meta[k] for k in stable_keys), "unit shape varied across videos")
            else:
                stable_per_uid[uid] = meta
            video = video_rows[vi]
            audit.append({"task037_global_index": uid, "layer": unit["layer"], "stage": unit["stage"],
                          "unit_type": unit["unit_type"], "unit_index": unit["unit_index"], "domain_id": unit["domain_id"],
                          "video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                          "class_position": video["class_position"], "activation_tensor_shape": meta["activation_tensor_shape"],
                          "activation_layout": meta["activation_layout"], "grid_shape_B_T_H_W_C": meta["grid_shape"],
                          "T_i": meta["T_i"], "H_i": meta["H_i"], "W_i": meta["W_i"],
                          "feature_dimension": meta["feature_dimension"], "head_dim": meta["head_dim"],
                          "resolution_consistent": True, "physical_gpu": int(args.gpu)})
        print("PHASE_J_RESOLUTION_AUDIT video=%d/%d" % (len({r["video_index"] for r in audit}), len(clips)), flush=True)
    capture.close()
    global_same = len({int(r["T_i"]) for r in audit}) == 1
    for row in audit:
        row["resolution_consistent"] = global_same
    return audit, stable_per_uid, global_same


def _relation_extract(args: argparse.Namespace, config: Mapping[str, Any],
                      stable_meta: Mapping[int, Mapping[str, Any]]) -> tuple[dict[int, dict[int, dict[str, Any]]], list[dict[str, Any]], int]:
    import torch
    torchmod, _, _, _, model, specs, _ = _load_model_and_specs(config, int(args.gpu))
    units = [u for u in read_csv(Path(config["unit_manifest"])) if str(u["domain_id"]) in DOMAINS]
    cap_rows = _capture_specs(units, specs)
    capture = PhaseJCapture(model, cap_rows, specs, torchmod)
    _, clips, video_rows = _load_all_videos(config, workers=2)
    relations, numerical = defaultdict(dict), []
    t_count = int(next(iter(stable_meta.values()))["T_i"])
    for video_idx in sorted(clips):
        clip = clips[video_idx].to(device="cuda:0", dtype=torch.float32, non_blocking=True)
        capture.begin(collect_features=True)
        with torchmod.inference_mode():
            _ = model(clip.unsqueeze(0))
        expected_ids = [int(r["task037_global_index"]) for r in cap_rows]
        validate_mask_free_identity(expected_ids, list(capture.features), mask_hooks_registered=False)
        video = video_rows[video_idx]
        for unit in cap_rows:
            uid = int(unit["task037_global_index"])
            feature, meta = capture.features[uid], capture.metadata[uid]
            require(tuple(feature.shape) == (t_count, int(stable_meta[uid]["feature_dimension"])) and int(meta["T_i"]) == t_count,
                    "per-frame feature disagrees with temporal resolution audit")
            g, c, stats, near_zero = _torch_relation(feature)
            relations[uid][video_idx] = {"G": g, "C": c}
            numerical.append({"task037_global_index": uid, "domain_id": unit["domain_id"],
                              "layer": unit["layer"], "stage": unit["stage"], "unit_type": unit["unit_type"],
                              "unit_index": unit["unit_index"], "video_index": video_idx,
                              "video_id": video["video_id"], "class_name": video["class_name"],
                              "class_position": video["class_position"], **stats,
                              "near_zero_frame_count": len(near_zero), "near_zero_frame_indices": near_zero,
                              "T_i": t_count, "feature_dimension": int(feature.shape[1]),
                              "physical_gpu": int(args.gpu)})
        print("PHASE_J_RELATIONS video=%d/%d" % (video_idx + 1, len(clips)), flush=True)
    capture.close()
    for unit in units:
        require(len(relations[int(unit["task037_global_index"])]) == 30, "unit is missing video relation matrices")
    return dict(relations), numerical, t_count


def _write_report(phase_root: Path, summary: Mapping[str, Any]) -> None:
    labels = {"A": "CONDITIONAL_FRAME_RELATION_PROMISING",
              "B": "CONDITIONAL_FRAME_RELATION_WEAK_OR_UNRESOLVED",
              "C": "CONDITIONAL_FRAME_RELATION_REJECTED"}
    lines = [
        "# Task042 Phase J — Conditional Frame-Relation Diagnostic", "",
        "Representation diagnosis only: no pruning masks, physical pruning, finetuning, validation-performance oracle, or causal interpretation.",
        "", "## Frozen run identity", "",
        "- Branch: " + str(summary.get("branch", "")),
        "- Git head: " + str(summary.get("git_head", "")),
        "- Checkpoint SHA256: " + str(summary.get("checkpoint_sha256", "")),
        "- Videos: %s; units: %s; domains: %s." % (summary.get("video_count"), summary.get("unit_count"), ", ".join(DOMAINS)),
        "- FP32, AMP=False; temporal resolution: %s." % str(summary.get("temporal_resolution", "not established")),
        "", "## Temporal-resolution gate", "", str(summary.get("temporal_resolution_status", "")), "",
    ]
    if summary.get("analysis_status") == "stopped_temporal_resolution_mismatch":
        differing = summary.get("differing_units", [])
        seen = set()
        for row in differing:
            uid = int(row["task037_global_index"])
            if uid in seen:
                continue
            seen.add(uid)
            lines.append("- unit %s, domain %s, layer %s, stage %s: T_i=%s (H×W=%s×%s)." % (
                row["task037_global_index"], row["domain_id"], row["layer"], row["stage"],
                row["T_i"], row["H_i"], row["W_i"]))
        lines.extend(["", "Relation comparison stopped before matrix construction; no temporal interpolation was applied.",
                      "", "## Decision", "", "B — " + labels["B"], "", str(summary["decision_rationale"]), ""])
    else:
        lines.extend([
            "## Covariance and conditional relation definition", "",
            "For X_i,v∈R^(T×D_i), each frame is centered over D_i and normalized by its L2 norm. OAS uses X_i,vᵀ∈R^(D_i×T): the temporal positions are variables and flattened within-frame feature components are repeated observations. The analytic OAS coefficient shrinks the empirical covariance toward μI; the same estimator is used for Attention and FFN. The signed conditional relation is −Ω_tt′/sqrt(Ω_ttΩ_t′t′), with diagonal zero. No manual shrinkage or epsilon regularizer is tuned.",
            "", "## Numerical audit", "", json.dumps(summary["numerical_summary"], ensure_ascii=False, indent=2),
            "", "## Required questions", "",
        ])
        for key in "ABCDEFGHI":
            lines.append("- **%s.** %s" % (key, summary["required_questions"][key]))
        lines.extend(["", "## Decision", "", labels[str(summary["decision"])], "",
                      str(summary["decision_rationale"]), "",
                      "Residual maps are stored with axes [unit,video,T,T] in task042_phase_j_coverability_residual_maps.npz.", ""])
    (phase_root / "task042_phase_j_report.md").write_text("\n".join(lines), encoding="utf-8")


def _questions(summary: Mapping[str, Any]) -> dict[str, str]:
    n = summary["numerical_summary"]
    p = summary["pairwise_comparison_summary"]
    s = summary["video_stability_summary"]
    cov = summary["coverability_by_domain"]
    geom = summary["domain_geometry"]
    mixed = summary["domain_271_pair_summary"]
    desc = summary["descriptor_complementarity_summary"]
    indirect = summary["indirect_relation_examples"]
    contrast = summary["relation_contrast_examples"]
    return {
        "A": "Finite inverses for all unit/video matrices: %s; minimum shrunk eigenvalue %s; maximum shrunk condition number %s." %
             (n["all_inverse_finite"], n["minimum_shrunk_eigenvalue"], n["maximum_shrunk_condition_number"]),
        "B": "Across unit/video matrices, median Pearson/Spearman/Kendall between signed upper-triangle G and C is %s/%s/%s; median top-|G|/top-|C| overlap is %s. Largest strong-G/weak-C contrast: %s; largest weak/moderate-G/strong-C contrast: %s." %
             (p["median_pearson_G_vs_C"], p["median_spearman_G_vs_C"], p["median_kendall_G_vs_C"],
              p["median_top_abs_overlap_fraction"], contrast["strong_G_weak_C"], contrast["weak_or_moderate_G_strong_C"]),
        "C": "Among the exhaustively ranked edge/path search, the largest observed endpoint rank drop is %s (unit %s, video %s, frames %s↔%s via %s; weakest-link G-path rank %s). The file records the top-ranked candidates and no hard cutoff." %
             (indirect.get("largest_rank_drop"), indirect.get("unit_id"), indirect.get("video_id"),
              indirect.get("frame_t1"), indirect.get("frame_t3"), indirect.get("frame_t2"),
              indirect.get("path_rank")),
        "D": "Median same-class versus different-class conditional-pattern similarity (Pearson/Spearman/cosine) is %s/%s/%s versus %s/%s/%s. Prototype medians for P1-vs-P2/P1-vs-P3/P2-vs-P3 (cosine) are %s; position-pair holdout medians are %s." %
             (s["same_class"]["pearson_median"], s["same_class"]["spearman_median"], s["same_class"]["cosine_median"],
              s["different_class"]["pearson_median"], s["different_class"]["spearman_median"], s["different_class"]["cosine_median"],
              s["prototype_cosine_medians"], s["pair_subset_cosine_medians"]),
        "E": "Within-domain conditional-vs-ordinary distance Spearman values are %s; nearest-neighbor changes per domain are %s; normalized distance-matrix Frobenius differences are %s." %
             ({d: geom[d]["within_domain_pair_spearman_d_cos_vs_d_cond"] for d in DOMAINS},
              {d: geom[d]["nearest_neighbor_changes"] for d in DOMAINS},
              {d: geom[d]["distance_matrix_normalized_frobenius_difference"] for d in DOMAINS}),
        "F": "Domain 271 pair counts and median conditional/ordinary distances are %s. Attention and FFN use identical centering, normalization, OAS, and T×T partial-correlation calculations." % mixed,
        "G": "Leave-one-unit-out conditional coverage delta distributions by domain are %s; pooled distribution is %s. These values describe coverability and are not importance." %
             (cov, summary["coverability_conditional"]),
        "H": "The same simplex fit gives pooled conditional/cosine deltas %s/%s, global Spearman/Kendall %s/%s, and per-unit ranking changes in the comparison table." %
             (summary["coverability_conditional"], summary["coverability_cosine"],
              summary["coverability_comparison_spearman"], summary["coverability_comparison_kendall"]),
        "I": "Across within-domain pairs, descriptor-vs-conditional-geometry Spearman summaries for D_abs/D_rel/D_st are %s; coverability correlations are %s. No descriptors are combined." %
             (desc["geometry_spearman_by_dimension"], desc["coverability_spearman_by_dimension"]),
    }


def _write_outputs(phase_root: Path, config: Mapping[str, Any], audit: Sequence[Mapping[str, Any]],
                   units: Sequence[Mapping[str, Any]], videos: Sequence[Mapping[str, Any]],
                   matrices: Mapping[int, Mapping[int, Mapping[str, Any]]],
                   numerical: Sequence[Mapping[str, Any]], t_count: int) -> dict[str, Any]:
    video_order = [int(v["video_index"]) for v in sorted(videos, key=lambda r: int(r["video_index"]))]
    video_by = {int(v["video_index"]): v for v in videos}
    ordinary, conditional, pair_cmp = _relation_rows(matrices, units, video_by, t_count)
    write_csv(phase_root / "task042_phase_j_pairwise_cosine.csv", ordinary)
    write_csv(phase_root / "task042_phase_j_conditional_relation.csv", conditional)
    write_csv(phase_root / "task042_phase_j_numerical_stability.csv", numerical)
    write_csv(phase_root / "task042_phase_j_cosine_vs_conditional.csv", pair_cmp)
    indirect = _indirect_cases(matrices, units, video_by, t_count)
    write_csv(phase_root / "task042_phase_j_indirect_relation_cases.csv", indirect)
    contrast = _relation_contrast_cases(matrices, units, video_by, t_count)
    write_csv(phase_root / "task042_phase_j_relation_contrast_cases.csv", contrast)
    video_stability = _video_stability(matrices, units, videos, video_order)
    write_csv(phase_root / "task042_phase_j_video_stability.csv", video_stability)
    geometry, geometry_details, domain_summary = _distance_geometry(matrices, units, video_order)
    write_csv(phase_root / "task042_phase_j_domain_geometry.csv", geometry)
    write_csv(phase_root / "task042_phase_j_mixed_domain.csv", geometry_details)
    cover_cond, maps_cond = _coverability(matrices, units, video_order, t_count, "C")
    cover_cos, maps_cos = _coverability(matrices, units, video_order, t_count, "G")
    write_csv(phase_root / "task042_phase_j_conditional_coverability.csv", cover_cond)
    write_csv(phase_root / "task042_phase_j_cosine_coverability.csv", cover_cos)
    np.savez_compressed(phase_root / "task042_phase_j_coverability_residual_maps.npz",
                        task037_global_indices=np.asarray([int(u["task037_global_index"]) for u in units], dtype=np.int64),
                        video_indices=np.asarray(video_order, dtype=np.int64),
                        conditional_residual_maps=maps_cond, ordinary_cosine_residual_maps=maps_cos)
    descriptor = _descriptor_complementarity(units, geometry, cover_cond, cover_cos)
    write_csv(phase_root / "task042_phase_j_descriptor_complementarity.csv", descriptor)
    cover_comparison = _coverability_comparison(cover_cond, cover_cos)
    write_csv(phase_root / "task042_phase_j_coverability_comparison.csv", cover_comparison)
    finite = [r for r in numerical if bool(r["finite_inverse"]) and int(r["nan_count"]) == 0 and int(r["inf_count"]) == 0]
    require(len(finite) == len(numerical), "numerical audit found nonfinite conditional relations")
    def cover_by_domain(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {d: _metric_summary([float(r["delta"]) for r in rows if str(r["domain_id"]) == d]) for d in DOMAINS}

    def video_metric_summary(comparison: str, metric: str) -> float | None:
        values = [float(r[metric + "_median"]) for r in video_stability
                  if r["row_type"] == "video_pair_summary" and r["comparison"] == comparison
                  and r.get(metric + "_median") not in ("", None)]
        return None if not values else float(np.median(values))

    prototype_cos = {name: float(np.median([float(r["cosine"]) for r in video_stability
                                             if r["row_type"] == "class_balanced_prototype" and r["comparison"] == name]))
                     for name in ("P1_vs_P2", "P1_vs_P3", "P2_vs_P3")}
    pair_subset_cos = {name: float(np.median([float(r["cosine"]) for r in video_stability
                                               if r["row_type"] == "class_balanced_pair_subset" and r["comparison"] == name]))
                       for name in ("P12_vs_P3", "P13_vs_P2", "P23_vs_P1")}
    pair_type_summary = {r["pair_type"]: {"pair_count": int(r["pair_count"]),
                                          "median_d_cond": r["median_d_cond"],
                                          "median_d_cos": r["median_d_cos"]}
                         for r in geometry_details if r.get("row_type") == "domain_271_pair_type_summary"}
    descriptor_summary = {
        "geometry_spearman_by_dimension": {
            name: {d: next((r["spearman_d_cond_vs_descriptor_gap"] for r in descriptor
                            if r["domain_id"] == d and r["descriptor_dimension"] == name and r["analysis_type"] == "pairwise_geometry"), None)
                  for d in DOMAINS} for name in ("D_abs", "D_rel", "D_st")},
        "coverability_spearman_by_dimension": {
            name: {d: next((r["spearman_d_cond_vs_descriptor_gap"] for r in descriptor
                            if r["domain_id"] == d and r["descriptor_dimension"] == name and r["analysis_type"] == "leave_one_out_coverability"), None)
                  for d in DOMAINS} for name in ("D_abs", "D_rel", "D_st")},
    }
    top_indirect = indirect[0] if indirect else {}
    contrast_examples = {}
    for category in ("strong_G_weak_C", "weak_or_moderate_G_strong_C"):
        candidates = [r for r in contrast if r["category"] == category]
        if category == "strong_G_weak_C":
            chosen = max(candidates, key=lambda r: r["rank_contrast_G_minus_C"]) if candidates else {}
        else:
            chosen = min(candidates, key=lambda r: r["rank_contrast_G_minus_C"]) if candidates else {}
        contrast_examples[category] = {
            "count": sum(r["category"] == category for r in contrast),
            "unit_id": chosen.get("task037_global_index"), "video_id": chosen.get("video_id"),
            "frames": [chosen.get("frame_t"), chosen.get("frame_t_prime")],
            "G": chosen.get("G"), "C": chosen.get("C"),
            "abs_G_rank_fraction": chosen.get("abs_G_rank_fraction"),
            "abs_C_rank_fraction": chosen.get("abs_C_rank_fraction"),
        }
    summary = {
        "task": "TASK042 PHASE J — CONDITIONAL FRAME-RELATION DIAGNOSTIC ONLY",
        "analysis_status": "completed", "branch": BRANCH,
        "git_head": subprocess.check_output(["git", "-C", config["repo_root"], "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": CHECKPOINT_SHA, "unit_count": len(units), "video_count": len(videos),
        "domain_unit_ids": {d: [int(u["task037_global_index"]) for u in units if str(u["domain_id"]) == d] for d in DOMAINS},
        "temporal_resolution": t_count,
        "temporal_resolution_status": "All %d selected units share audited T_i=%d; comparison used no interpolation." % (len(units), t_count),
        "same_temporal_resolution": True,
        "feature_normalization": "per-frame mean subtraction over D_i followed by L2 normalization with 1e-12 denominator offset",
        "covariance_orientation": "X_i,v [T,D_i]; observations X_i,v^T [D_i,T]; temporal positions are variables; flattened within-frame components are observations; divisor D_i",
        "oas": "deterministic analytic OAS; coefficient logged, not tuned",
        "conditional_relation": "signed partial correlation -Omega[t,t']/sqrt(Omega[t,t]*Omega[t',t']); diagonal zero",
        "mask_free": True, "pruning": False, "finetuning": False, "performance_oracle": False,
        "numerical_summary": {
            "unit_video_matrices": len(numerical), "all_inverse_finite": len(finite) == len(numerical),
            "minimum_shrunk_eigenvalue": min(float(r["shrunk_min_eigenvalue"]) for r in numerical),
            "maximum_shrunk_condition_number": max(float(r["shrunk_condition_number"]) for r in numerical),
            "median_oas_coefficient": float(np.median([float(r["oas_shrinkage"]) for r in numerical])),
            "near_zero_unit_video_count": sum(int(r["near_zero_frame_count"]) > 0 for r in numerical),
            "max_abs_partial": max(float(r["max_abs_partial"]) for r in numerical),
        },
        "pairwise_comparison_summary": {
            "median_pearson_G_vs_C": float(np.median([r["pearson_G_vs_C"] for r in pair_cmp if r["pearson_G_vs_C"] is not None])),
            "median_spearman_G_vs_C": float(np.median([r["spearman_G_vs_C"] for r in pair_cmp if r["spearman_G_vs_C"] is not None])),
            "median_kendall_G_vs_C": float(np.median([r["kendall_G_vs_C"] for r in pair_cmp if r["kendall_G_vs_C"] is not None])),
            "median_top_abs_overlap_fraction": float(np.median([r["top_abs_relation_overlap_fraction"] for r in pair_cmp])),
        },
        "domain_geometry": domain_summary,
        "coverability_conditional": _metric_summary([float(r["delta"]) for r in cover_cond]),
        "coverability_cosine": _metric_summary([float(r["delta"]) for r in cover_cos]),
        "coverability_by_domain": cover_by_domain(cover_cond),
        "coverability_comparison_spearman": cover_comparison[-1]["spearman_delta_cond_vs_delta_cos"],
        "coverability_comparison_kendall": cover_comparison[-1]["kendall_delta_cond_vs_delta_cos"],
        "video_stability_summary": {
            "same_class": {m + "_median": video_metric_summary("same_class", m) for m in ("pearson", "spearman", "cosine")},
            "different_class": {m + "_median": video_metric_summary("different_class", m) for m in ("pearson", "spearman", "cosine")},
            "prototype_cosine_medians": prototype_cos, "pair_subset_cosine_medians": pair_subset_cos,
        },
        "domain_271_pair_summary": pair_type_summary,
        "descriptor_complementarity_summary": descriptor_summary,
        "indirect_relation_examples": {
            "candidate_count_examined": len(units) * len(videos) * (t_count * (t_count - 1) // 2),
            "largest_rank_drop": top_indirect.get("endpoint_rank_drop_G_minus_absC"),
            "unit_id": top_indirect.get("task037_global_index"), "video_id": top_indirect.get("video_id"),
            "frame_t1": top_indirect.get("frame_t1"), "frame_t2": top_indirect.get("frame_t2"),
            "frame_t3": top_indirect.get("frame_t3"), "path_rank": top_indirect.get("indirect_path_weakest_G_rank"),
        },
        "relation_contrast_examples": contrast_examples,
        "required_questions": {},
        "decision": "B",
        "decision_rationale": "Evidence tables are complete; choose A, B, or C after reviewing stability, domain geometry, coverability, and descriptor-complementarity evidence.",
    }
    summary["required_questions"] = _questions(summary)
    write_json(phase_root / "task042_phase_j_summary.json", summary)
    _write_report(phase_root, summary)
    return summary


def finalize_outputs(phase_root: Path, decision: str, rationale: str) -> dict[str, Any]:
    if decision not in ("A", "B", "C"):
        raise ValueError("decision must be exactly A, B, or C")
    path = phase_root / "task042_phase_j_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["decision"], summary["decision_rationale"] = decision, rationale
    write_json(path, summary)
    _write_report(phase_root, summary)
    return summary


def run(args: argparse.Namespace) -> None:
    repo, base_root, phase_root = Path(args.repo_root), Path(args.base_output), Path(args.phase_root)
    require(subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip() == BRANCH,
            "checkout is not the frozen Task042 branch")
    require(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip() == "",
            "Phase-J support must be committed and checkout clean before server execution")
    config = _prepare(repo, base_root, phase_root)
    audit, stable_meta, same_t = _shape_audit(args, config)
    write_csv(phase_root / "task042_phase_j_temporal_resolution_audit.csv", audit)
    distinct_t = sorted({int(row["T_i"]) for row in audit})
    if not same_t:
        differing = []
        by_uid = defaultdict(set)
        for row in audit:
            by_uid[int(row["task037_global_index"])].add(int(row["T_i"]))
        for row in audit:
            if len(by_uid[int(row["task037_global_index"])]) > 1 or int(row["T_i"]) != distinct_t[0]:
                differing.append({k: row[k] for k in ("task037_global_index", "domain_id", "layer", "stage", "T_i", "H_i", "W_i")})
        summary = {
            "task": "TASK042 PHASE J — CONDITIONAL FRAME-RELATION DIAGNOSTIC ONLY",
            "analysis_status": "stopped_temporal_resolution_mismatch", "branch": BRANCH,
            "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint_sha256": CHECKPOINT_SHA, "unit_count": config["unit_count"], "video_count": config["video_count"],
            "temporal_resolution": distinct_t, "same_temporal_resolution": False,
            "temporal_resolution_status": "Different stage temporal resolutions were observed; relation comparison stopped before any relation matrix was constructed.",
            "differing_units": differing, "required_questions": {}, "decision": "B",
            "decision_rationale": "The common T×T relation space is not defined without alignment; Phase J explicitly forbids interpolation at this stage.",
            "mask_free": True, "pruning": False, "finetuning": False, "performance_oracle": False,
        }
        write_json(phase_root / "task042_phase_j_summary.json", summary)
        _write_report(phase_root, summary)
        return
    relations, numerical, t_count = _relation_extract(args, config, stable_meta)
    units = [dict(r, capture_kind="head" if r["unit_type"] == "attention_head" else "neuron")
             for r in read_csv(Path(config["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    units.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    videos = [dict(r, video_index=int(r["video_index"]), label=int(r["label"]), class_position=int(r["class_position"]))
              for r in read_csv(Path(config["video_manifest"]))]
    _write_outputs(phase_root, config, audit, units, videos, relations, numerical, t_count)
    finalize_outputs(phase_root, args.decision, args.decision_rationale)
    print("PHASE_J_COMPLETE units=%d videos=%d T=%d decision=%s" % (len(units), len(videos), t_count, args.decision), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO))
    parser.add_argument("--base-output", default=str(DEFAULT_BASE))
    parser.add_argument("--phase-root", default=str(DEFAULT_BASE / "phase_j"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--decision", choices=("A", "B", "C"), default="B")
    parser.add_argument("--decision-rationale", default="Evidence is unresolved pending diagnostic review.")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_only:
        finalize_outputs(Path(args.phase_root), args.decision, args.decision_rationale)
    else:
        run(args)


if __name__ == "__main__":
    main()

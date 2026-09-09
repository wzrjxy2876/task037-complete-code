#!/usr/bin/env python3
"""Analyze Task 009 third-descriptor candidates against Task 007 ablations.

The four descriptor geometries are evaluated independently for all units,
Attention Heads, FFN neurons, and Video Swin stages 0--3.  Pair sampling and
nearest-neighbor searches are memory safe.  When CUDA is available, descriptor
pair distances and the exact existing BMS implementation run on the GPU;
Spearman ranking, low-dimensional cKDTree queries, and reporting remain on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task009_video_descriptor_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from analyze_bms_competition_domains import (
    EPSILON,
    evaluate_groups,
    mean_pairwise_absolute_difference,
    random_baseline,
    validate_group_membership,
)
from analyze_descriptor_ablation_consistency import select_pair_indices


DESCRIPTOR_COLUMNS = (
    "D_abs",
    "D_rel",
    "D_st_old",
    "D_temporal_variation",
    "D_action_dynamic",
)
THIRD_COLUMNS = OrderedDict(
    [
        ("Old", "D_st_old"),
        ("Variation", "D_temporal_variation"),
        ("ADC", "D_action_dynamic"),
    ]
)
VARIANT_COLUMNS = OrderedDict(
    [
        ("Abs+Rel", ("D_abs", "D_rel")),
        ("Old3D", ("D_abs", "D_rel", "D_st_old")),
        (
            "Variation3D",
            ("D_abs", "D_rel", "D_temporal_variation"),
        ),
        ("ADC3D", ("D_abs", "D_rel", "D_action_dynamic")),
    ]
)
REQUIRED_SCOPES = (
    "all_units",
    "attention_only",
    "mlp_only",
    "stage0",
    "stage1",
    "stage2",
    "stage3",
)
CANDIDATE_FIELDS = {
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_st_old",
    "D_temporal_variation",
    "D_action_dynamic",
    "ablation_logit_deviation",
}
SANITY_VALUE_COLUMNS = (
    "D_temporal_variation_normal",
    "D_temporal_variation_frozen",
    "D_temporal_variation_shuffle",
    "D_action_dynamic_normal",
    "D_action_dynamic_frozen",
    "D_action_dynamic_shuffle",
)

BmsRunner = Callable[[np.ndarray], tuple[list[list[int]], dict[str, object]]]


def _stage_from_layer(layer: str) -> int:
    match = re.search(r"(?:^|\.)layers\.([0-3])(?:\.|$)", layer)
    if match is None:
        raise ValueError(f"cannot derive stage0-stage3 from {layer!r}")
    return int(match.group(1))


def _parse_finite(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number} has invalid {column!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number} column {column!r} is not finite")
    return value


def read_candidate_rows(path: Path) -> list[dict]:
    """Read the complete unit table as rows corresponding to ``V [N,5]``."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = CANDIDATE_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"candidate CSV is empty: {path}")
    rows = []
    for row_number, source in enumerate(source_rows, start=2):
        try:
            global_index = int(source["global_index"])
            unit_index = int(source["unit_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number} has invalid unit indices") from exc
        layer = str(source["layer"]).strip()
        unit_type = str(source["unit_type"]).strip()
        if unit_type not in {"attention_head", "ffn_neuron"}:
            raise ValueError(f"row {row_number} has unsupported unit type")
        row = {
            "global_index": global_index,
            "layer": layer,
            "unit_type": unit_type,
            "unit_index": unit_index,
            "stage": _stage_from_layer(layer),
        }
        for column in (*DESCRIPTOR_COLUMNS, "ablation_logit_deviation"):
            row[column] = _parse_finite(source, column, row_number)
        for column in ("D_temporal_variation", "D_action_dynamic"):
            if not -1e-6 <= float(row[column]) <= 1.0 + 1e-6:
                raise ValueError(f"row {row_number} {column} is outside [0,1]")
        rows.append(row)
    rows.sort(key=lambda row: row["global_index"])
    if [row["global_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("global_index must be unique and contiguous from zero")
    keys = [
        (row["layer"], row["unit_type"], row["unit_index"]) for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("unit identity must be unique")
    return rows


def read_sanity_rows(path: Path, candidates: Sequence[dict]) -> list[dict]:
    required = {
        "global_index", "layer", "unit_type", "unit_index", *SANITY_VALUE_COLUMNS
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        source_rows = list(reader)
    source_rows.sort(key=lambda row: int(row["global_index"]))
    if len(source_rows) != len(candidates):
        raise ValueError("sanity CSV does not contain one row per candidate unit")
    rows = []
    for row_number, (source, candidate) in enumerate(
        zip(source_rows, candidates), start=2
    ):
        key = (
            int(source["global_index"]),
            str(source["layer"]),
            str(source["unit_type"]),
            int(source["unit_index"]),
        )
        expected = (
            candidate["global_index"],
            candidate["layer"],
            candidate["unit_type"],
            candidate["unit_index"],
        )
        if key != expected:
            raise ValueError("sanity CSV unit ordering differs from candidate CSV")
        row = {
            "global_index": key[0],
            "layer": key[1],
            "unit_type": key[2],
            "unit_index": key[3],
            "stage": candidate["stage"],
        }
        for column in SANITY_VALUE_COLUMNS:
            value = _parse_finite(source, column, row_number)
            if not -1e-6 <= value <= 1.0 + 1e-6:
                raise ValueError(f"row {row_number} {column} is outside [0,1]")
            row[column] = value
        rows.append(row)
    return rows


def validate_candidate_npz(path: Path, rows: Sequence[dict]) -> None:
    required = {
        "D_abs", "D_rel", "D_old", "D_var", "D_adc",
        "ablation_effect", "global_index",
    }
    with np.load(path, allow_pickle=False) as values:
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")
        expected = {
            "D_abs": np.asarray([row["D_abs"] for row in rows]),
            "D_rel": np.asarray([row["D_rel"] for row in rows]),
            "D_old": np.asarray([row["D_st_old"] for row in rows]),
            "D_var": np.asarray([row["D_temporal_variation"] for row in rows]),
            "D_adc": np.asarray([row["D_action_dynamic"] for row in rows]),
            "ablation_effect": np.asarray(
                [row["ablation_logit_deviation"] for row in rows]
            ),
            "global_index": np.arange(len(rows), dtype=np.int64),
        }
        for name, expected_array in expected.items():
            actual = np.asarray(values[name])
            if actual.shape != expected_array.shape or not np.allclose(
                actual, expected_array, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(f"NPZ array {name} differs from candidate CSV")


def build_scopes(rows: Sequence[dict]) -> OrderedDict[str, np.ndarray]:
    """Return seven local-index vectors ``I_s [N_s]`` for required scopes."""
    scopes: OrderedDict[str, np.ndarray] = OrderedDict()
    scopes["all_units"] = np.arange(len(rows), dtype=np.int64)
    scopes["attention_only"] = np.asarray(
        [i for i, row in enumerate(rows) if row["unit_type"] == "attention_head"],
        dtype=np.int64,
    )
    scopes["mlp_only"] = np.asarray(
        [i for i, row in enumerate(rows) if row["unit_type"] == "ffn_neuron"],
        dtype=np.int64,
    )
    for stage in range(4):
        scopes[f"stage{stage}"] = np.asarray(
            [i for i, row in enumerate(rows) if row["stage"] == stage],
            dtype=np.int64,
        )
    for name in REQUIRED_SCOPES:
        if scopes[name].size < 2:
            raise ValueError(f"scope {name} has fewer than two units")
    return scopes


def standardize_descriptor(
    descriptor: np.ndarray, eps: float = EPSILON
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task 007 per-axis standardization for ``V [N,D]`` (population std)."""
    values = np.asarray(descriptor, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("descriptor must be a non-empty finite matrix [N,D]")
    mean = values.mean(axis=0)  # [D]
    std = values.std(axis=0)  # [D], Task 007 NumPy population std
    safe = np.where(std > eps, std, 1.0)
    normalized = (values - mean[None, :]) / safe[None, :]  # [N,D]
    normalized[:, std <= eps] = 0.0
    return normalized, mean, std


def build_variants(
    descriptor: np.ndarray,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, dict[str, list[float]]]]:
    """Build four independently standardized matrices ``V_hat [N,d]``."""
    values = np.asarray(descriptor, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DESCRIPTOR_COLUMNS):
        raise ValueError(f"expected descriptor [N,5], got {values.shape}")
    column_index = {name: index for index, name in enumerate(DESCRIPTOR_COLUMNS)}
    variants: OrderedDict[str, np.ndarray] = OrderedDict()
    metadata: dict[str, dict[str, list[float]]] = {}
    for name, columns in VARIANT_COLUMNS.items():
        selected = values[:, [column_index[column] for column in columns]]
        normalized, mean, std = standardize_descriptor(selected)
        variants[name] = normalized
        metadata[name] = {
            "columns": list(columns),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return variants, metadata


def resolve_analysis_device(requested: str) -> tuple[str, object | None]:
    """Resolve ``auto`` to CUDA when possible and otherwise to CPU."""
    try:
        import torch
    except ImportError:
        if requested.startswith("cuda"):
            raise RuntimeError("--device cuda requested but PyTorch is unavailable")
        return "cpu", None
    if requested == "auto":
        return ("cuda:0", torch) if torch.cuda.is_available() else ("cpu", torch)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested, torch


def pairwise_descriptor_distances(
    features: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    chunk_size: int,
    device: str,
    torch_module: object | None,
) -> np.ndarray:
    """Compute sampled distances ``d_V [P]`` in bounded CPU/GPU chunks."""
    values = np.asarray(features, dtype=np.float64)
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("pair index vectors must have equal shape [P]")
    distances = np.empty(first.size, dtype=np.float64)  # [P]
    if device.startswith("cuda"):
        if torch_module is None:
            raise RuntimeError("CUDA distance calculation requires PyTorch")
        torch = torch_module
        feature_tensor = torch.as_tensor(
            values, dtype=torch.float64, device=device
        )  # [N,D]
        for start in range(0, first.size, chunk_size):
            end = min(start + chunk_size, first.size)
            first_tensor = torch.as_tensor(first[start:end], device=device)
            second_tensor = torch.as_tensor(second[start:end], device=device)
            block = torch.linalg.vector_norm(
                feature_tensor.index_select(0, first_tensor)
                - feature_tensor.index_select(0, second_tensor),
                dim=1,
            )  # [p]
            distances[start:end] = block.detach().double().cpu().numpy()
        del feature_tensor
    else:
        for start in range(0, first.size, chunk_size):
            end = min(start + chunk_size, first.size)
            difference = values[first[start:end]] - values[second[start:end]]
            distances[start:end] = np.linalg.norm(difference, axis=1)
    return distances


def distance_spearman(
    features: np.ndarray,
    effect: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    chunk_size: int,
    device: str,
    torch_module: object | None,
) -> tuple[float, float]:
    descriptor_distance = pairwise_descriptor_distances(
        features, first, second, chunk_size, device, torch_module
    )
    effect_distance = np.abs(effect[first] - effect[second])  # [P]
    statistic = spearmanr(descriptor_distance, effect_distance)
    rho = float(statistic.statistic)
    pvalue = float(statistic.pvalue)
    return (
        rho if math.isfinite(rho) else 0.0,
        pvalue if math.isfinite(pvalue) else 1.0,
    )


def exact_nearest_neighbors(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact low-dimensional cKDTree neighbor index/distance vectors ``[N]``."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("nearest-neighbor features must have shape [N,D], N>=2")
    tree = cKDTree(values)
    candidate_count = min(values.shape[0], 8)
    candidate_distance, candidate_index = tree.query(
        values, k=candidate_count, workers=1
    )
    if candidate_count == 1:
        candidate_distance = candidate_distance[:, None]
        candidate_index = candidate_index[:, None]
    neighbor = np.empty(values.shape[0], dtype=np.int64)  # [N]
    distance = np.empty(values.shape[0], dtype=np.float64)  # [N]
    for row_index in range(values.shape[0]):
        valid = candidate_index[row_index] != row_index
        if not np.any(valid):
            raise RuntimeError(f"no non-self neighbor for row {row_index}")
        valid_distance = candidate_distance[row_index][valid]
        valid_index = candidate_index[row_index][valid]
        best_distance = float(valid_distance.min())
        tied = valid_index[
            np.isclose(valid_distance, best_distance, rtol=1e-12, atol=1e-12)
        ]
        neighbor[row_index] = int(tied.min())
        distance[row_index] = best_distance
    return neighbor, distance


def _load_exact_bms():
    from analyze_bms_competition_domains import _load_current_bms

    return _load_current_bms()


def run_exact_bms_on_device(
    features: np.ndarray, device: str
) -> tuple[list[list[int]], dict[str, object]]:
    """Run production BMS unchanged on ``V [N,d]``, preferring CUDA."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("BMS features must be finite [N,d] with N>=2")
    torch, pruner_class, sigma, source_sha = _load_exact_bms()
    descriptor = torch.as_tensor(
        values, dtype=torch.float32, device=device
    )  # [N,d]
    with torch.no_grad():
        mean = descriptor.mean(dim=0, keepdim=True)  # [1,d]
        std = descriptor.std(dim=0, keepdim=True)  # [1,d], correction=1
        normalized = (descriptor - mean) / (std + EPSILON)  # [N,d]
        if not torch.isfinite(normalized).all():
            raise ValueError("standardized BMS input contains NaN/Inf")
        pruner = pruner_class.__new__(pruner_class)
        pruner.sigma = sigma
        groups, trajectories, endpoints = pruner_class.mean_shift_clustering(
            pruner, normalized
        )
    partition = validate_group_membership(groups, values.shape[0])
    return partition, {
        "sigma": sigma,
        "device": device,
        "dtype": "torch.float32",
        "standardization": (
            "torch mean/std with default sample correction and eps=1e-8"
        ),
        "bms_source_sha256": source_sha,
        "trajectory_shape": list(trajectories.shape),
        "endpoint_shape": list(endpoints.shape),
    }


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    statistic = spearmanr(first, second)
    rho = float(statistic.statistic)
    pvalue = float(statistic.pvalue)
    return (
        rho if math.isfinite(rho) else 0.0,
        pvalue if math.isfinite(pvalue) else 1.0,
    )


def compute_descriptor_correlation(
    descriptor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Spearman statistic/p-value matrices, both shaped ``[5,5]``."""
    values = np.asarray(descriptor, dtype=np.float64)
    statistic = spearmanr(values, axis=0)
    matrix = np.asarray(statistic.statistic, dtype=np.float64)
    pvalue = np.asarray(statistic.pvalue, dtype=np.float64)
    if matrix.shape != (5, 5) or pvalue.shape != (5, 5):
        raise RuntimeError("unexpected descriptor correlation matrix shape")
    return np.nan_to_num(matrix), np.nan_to_num(pvalue, nan=1.0)


def compute_sanity_summary(
    sanity_rows: Sequence[dict], scopes: OrderedDict[str, np.ndarray]
) -> list[dict]:
    """Compute freeze/shuffle diagnostics from unit vectors ``[N_s]``."""
    rows = []
    candidate_prefix = OrderedDict(
        [("Variation", "D_temporal_variation"), ("ADC", "D_action_dynamic")]
    )
    for scope, indices in scopes.items():
        for candidate, prefix in candidate_prefix.items():
            normal = np.asarray(
                [sanity_rows[int(i)][f"{prefix}_normal"] for i in indices],
                dtype=np.float64,
            )  # [N_s]
            frozen = np.asarray(
                [sanity_rows[int(i)][f"{prefix}_frozen"] for i in indices],
                dtype=np.float64,
            )  # [N_s]
            shuffled = np.asarray(
                [sanity_rows[int(i)][f"{prefix}_shuffle"] for i in indices],
                dtype=np.float64,
            )  # [N_s]
            freeze_ratio = frozen / (normal + EPSILON)  # [N_s]
            shuffle_sensitivity = np.abs(shuffled - normal) / (
                normal + EPSILON
            )  # [N_s]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "num_units": len(indices),
                    "normal_mean": float(normal.mean()),
                    "normal_median": float(np.median(normal)),
                    "frozen_mean": float(frozen.mean()),
                    "frozen_median": float(np.median(frozen)),
                    "shuffle_mean": float(shuffled.mean()),
                    "shuffle_median": float(np.median(shuffled)),
                    "mean_freeze_ratio": float(freeze_ratio.mean()),
                    "median_freeze_ratio": float(np.median(freeze_ratio)),
                    "fraction_freeze_ratio_below_0.5": float(
                        np.mean(freeze_ratio < 0.5)
                    ),
                    "mean_shuffle_sensitivity": float(
                        shuffle_sensitivity.mean()
                    ),
                    "median_shuffle_sensitivity": float(
                        np.median(shuffle_sensitivity)
                    ),
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_correlation(path: Path, matrix: np.ndarray) -> None:
    labels = ["D_abs", "D_rel", "Old", "Variation", "ADC"]
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(5), labels, rotation=25, ha="right")
    axis.set_yticks(range(5), labels)
    for row in range(5):
        for column in range(5):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center", fontsize=9,
            )
    axis.set_title("Third-descriptor Spearman correlation")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_grouped_metric(
    path: Path,
    rows: Sequence[dict],
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    scopes = ("all_units", "attention_only", "mlp_only")
    x = np.arange(len(VARIANT_COLUMNS), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(10.4, 5.8))
    for offset, scope in enumerate(scopes):
        values = []
        for variant in VARIANT_COLUMNS:
            row = next(
                item for item in rows
                if item["scope"] == scope and item["variant"] == variant
            )
            values.append(float(row[value_key]))
        axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=scope,
        )
    axis.set_xticks(x, list(VARIANT_COLUMNS), rotation=15)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_attention_comparison(
    path: Path, nn_rows: Sequence[dict], bms_rows: Sequence[dict]
) -> None:
    variants = list(VARIANT_COLUMNS)
    nn = [
        float(next(row for row in nn_rows if row["scope"] == "attention_only" and row["variant"] == name)["nn_error"])
        for name in variants
    ]
    pair = [
        float(next(row for row in bms_rows if row["scope"] == "attention_only" and row["variant"] == name)["q_pair"])
        for name in variants
    ]
    variance = [
        float(next(row for row in bms_rows if row["scope"] == "attention_only" and row["variant"] == name)["q_var"])
        for name in variants
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for axis, values, title in zip(
        axes,
        (nn, pair, variance),
        ("NNError ↓", "BMS pair difference ↓", "BMS variance ↓"),
    ):
        axis.bar(variants, values, color=("#5470C6", "#91CC75", "#FAC858", "#EE6666"))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Attention-Head descriptor comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_video_sanity(path: Path, rows: Sequence[dict]) -> None:
    global_rows = {
        row["candidate"]: row for row in rows if row["scope"] == "all_units"
    }
    candidates = ("Variation", "ADC")
    conditions = ("normal_mean", "frozen_mean", "shuffle_mean")
    labels = ("Normal", "Frozen", "Shuffle")
    x = np.arange(len(candidates), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    for offset, (column, label) in enumerate(zip(conditions, labels)):
        values = [float(global_rows[name][column]) for name in candidates]
        axis.bar(x + (offset - 1) * width, values, width, label=label)
    axis.set_xticks(x, candidates)
    axis.set_ylabel("Mean descriptor value")
    axis.set_title("Video-specific sanity check")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _find(rows: Sequence[dict], scope: str, variant: str) -> dict:
    return next(
        row for row in rows
        if row["scope"] == scope and row["variant"] == variant
    )


def _relative_lower_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline if baseline != 0.0 else float("nan")


def _best_third_descriptor(
    nn_rows: Sequence[dict], bms_rows: Sequence[dict]
) -> str:
    candidates = ("Old3D", "Variation3D", "ADC3D")
    metric_vectors = {}
    for candidate in candidates:
        metric_vectors[candidate] = np.asarray(
            [
                float(_find(nn_rows, "all_units", candidate)["nn_error"]),
                float(_find(bms_rows, "all_units", candidate)["q_pair"]),
                float(_find(bms_rows, "all_units", candidate)["q_var"]),
                float(_find(nn_rows, "attention_only", candidate)["nn_error"]),
                float(_find(bms_rows, "attention_only", candidate)["q_pair"]),
                float(_find(bms_rows, "attention_only", candidate)["q_var"]),
            ],
            dtype=np.float64,
        )
    winners = []
    for candidate, values in metric_vectors.items():
        others = [value for name, value in metric_vectors.items() if name != candidate]
        if all(np.all(values <= other) for other in others) and any(
            np.any(values < other) for other in others
        ):
            winners.append(candidate)
    return winners[0] if len(winners) == 1 else "No unique Pareto winner"


def write_validation_summary(
    path: Path,
    correlation: np.ndarray,
    nn_rows: Sequence[dict],
    bms_rows: Sequence[dict],
    sanity_rows: Sequence[dict],
) -> None:
    index = {name: i for i, name in enumerate(DESCRIPTOR_COLUMNS)}
    rho_abs_adc = float(correlation[index["D_abs"], index["D_action_dynamic"]])
    rho_rel_adc = float(correlation[index["D_rel"], index["D_action_dynamic"]])
    nn_2d = float(_find(nn_rows, "all_units", "Abs+Rel")["nn_error"])
    nn_old = float(_find(nn_rows, "all_units", "Old3D")["nn_error"])
    nn_adc = float(_find(nn_rows, "all_units", "ADC3D")["nn_error"])
    pair_2d = float(_find(bms_rows, "all_units", "Abs+Rel")["q_pair"])
    pair_old = float(_find(bms_rows, "all_units", "Old3D")["q_pair"])
    pair_adc = float(_find(bms_rows, "all_units", "ADC3D")["q_pair"])
    var_old = float(_find(bms_rows, "all_units", "Old3D")["q_var"])
    var_adc = float(_find(bms_rows, "all_units", "ADC3D")["q_var"])
    att_nn_old = float(_find(nn_rows, "attention_only", "Old3D")["nn_error"])
    att_nn_adc = float(_find(nn_rows, "attention_only", "ADC3D")["nn_error"])
    att_pair_old = float(_find(bms_rows, "attention_only", "Old3D")["q_pair"])
    att_pair_adc = float(_find(bms_rows, "attention_only", "ADC3D")["q_pair"])
    att_var_old = float(_find(bms_rows, "attention_only", "Old3D")["q_var"])
    att_var_adc = float(_find(bms_rows, "attention_only", "ADC3D")["q_var"])
    adc_sanity = next(
        row for row in sanity_rows
        if row["scope"] == "all_units" and row["candidate"] == "ADC"
    )

    complementary = (
        (nn_adc < nn_2d or pair_adc < pair_2d)
        and abs(rho_abs_adc) < 1.0 - 1e-12
        and abs(rho_rel_adc) < 1.0 - 1e-12
    )
    nn_improves = nn_adc < nn_old
    bms_improves = pair_adc < pair_old and var_adc < var_old
    attention_improves = (
        att_nn_adc < att_nn_old
        and att_pair_adc < att_pair_old
        and att_var_adc < att_var_old
    )
    frozen_decreases = float(adc_sanity["median_freeze_ratio"]) < 1.0
    replacement_supported = (
        complementary
        and nn_improves
        and bms_improves
        and attention_improves
        and frozen_decreases
    )
    best = _best_third_descriptor(nn_rows, bms_rows)
    lines = [
        "# Video-Specific Third Descriptor Validation",
        "",
        "## 1. Complementary information",
        "",
        f"ADC rho with D_abs: {rho_abs_adc:.6f}; with D_rel: {rho_rel_adc:.6f}. ",
        (
            "ADC provides observed complementary geometric information under "
            "the directional Task 009 checks."
            if complementary
            else "The current evidence does not establish complementary benefit for ADC."
        ),
        "",
        "## 2. Local nearest-neighbor behavior",
        "",
        f"All-unit NNError: Abs+Rel={nn_2d:.6e}, Old3D={nn_old:.6e}, "
        f"ADC3D={nn_adc:.6e}. ADC versus Old3D relative improvement: "
        f"{_relative_lower_improvement(nn_adc, nn_old):.3%}. "
        + ("ADC improves NN behavior." if nn_improves else "ADC does not improve NN behavior."),
        "",
        "## 3. BMS competition-domain homogeneity",
        "",
        f"All-unit Q_pair: Abs+Rel={pair_2d:.6e}, Old3D={pair_old:.6e}, "
        f"ADC3D={pair_adc:.6e}; Old3D Q_var={var_old:.6e}, "
        f"ADC3D Q_var={var_adc:.6e}. "
        + ("ADC improves both BMS metrics." if bms_improves else "ADC does not improve both BMS metrics."),
        "",
        "## 4. Attention-Head behavior",
        "",
        f"Attention NNError Old3D/ADC3D={att_nn_old:.6e}/{att_nn_adc:.6e}; "
        f"Q_pair={att_pair_old:.6e}/{att_pair_adc:.6e}; "
        f"Q_var={att_var_old:.6e}/{att_var_adc:.6e}. "
        + ("ADC improves all three Attention metrics." if attention_improves else "ADC does not improve all three Attention metrics."),
        "",
        "## 5. Frozen-video diagnostic",
        "",
        f"ADC median freeze ratio={float(adc_sanity['median_freeze_ratio']):.6f}, "
        f"mean freeze ratio={float(adc_sanity['mean_freeze_ratio']):.6f}, "
        f"fraction below 0.5={float(adc_sanity['fraction_freeze_ratio_below_0.5']):.3%}. "
        + ("Frozen input decreases the median ADC value." if frozen_decreases else "Frozen input does not decrease the median ADC value."),
        "",
        "## 6. Replacement decision",
        "",
        (
            "Within the directional Task 009 criteria, the evidence supports "
            "considering ADC as a replacement for Old3D. Production replacement "
            "still requires a separate authorized task."
            if replacement_supported
            else "The evidence is not strong enough to replace Old3D; keep the current descriptor."
        ),
        "",
        f"Pareto result across all-unit and Attention primary metrics: {best}.",
        "",
        "ADC is interpreted only as the proportion of a pruning unit's "
        "class-conditioned contribution associated with temporally varying responses.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(
    rows: Sequence[dict],
    correlation: np.ndarray,
    distance_rows: Sequence[dict],
    nn_rows: Sequence[dict],
    bms_rows: Sequence[dict],
    sanity_rows: Sequence[dict],
) -> None:
    print("=" * 60)
    print("Video-Specific Third Descriptor Validation")
    print("-" * 60)
    print(f"Units: {len(rows)}")
    print()
    print(f"{'':24s} {'Dist-Spearman↑':>15s} {'NNError↓':>12s} {'BMS PairDiff↓':>15s}")
    print("-" * 68)
    for variant in VARIANT_COLUMNS:
        distance = _find(distance_rows, "all_units", variant)
        nn = _find(nn_rows, "all_units", variant)
        bms = _find(bms_rows, "all_units", variant)
        print(
            f"{variant:24s} {float(distance['spearman_distance_rho']):15.6f} "
            f"{float(nn['nn_error']):12.6e} {float(bms['q_pair']):15.6e}"
        )
    for scope, title in (("attention_only", "Attention only"), ("mlp_only", "MLP only")):
        print()
        print(f"{title}:")
        for variant in ("Old3D", "ADC3D"):
            distance = _find(distance_rows, scope, variant)
            nn = _find(nn_rows, scope, variant)
            bms = _find(bms_rows, scope, variant)
            print(
                f"{variant:24s} {float(distance['spearman_distance_rho']):15.6f} "
                f"{float(nn['nn_error']):12.6e} {float(bms['q_pair']):15.6e}"
            )
    index = {name: i for i, name in enumerate(DESCRIPTOR_COLUMNS)}
    print()
    print("Descriptor correlation:")
    for base in ("D_abs", "D_rel"):
        for label, column in THIRD_COLUMNS.items():
            print(
                f"rho({base}, D_{label.lower()})"
                f" {correlation[index[base], index[column]]: .6f}"
            )
    print()
    print("Video sanity (all units; means):")
    print(f"{'':24s} {'Normal':>10s} {'Frozen':>10s} {'Shuffle':>10s}")
    for candidate in ("Variation", "ADC"):
        row = next(
            item for item in sanity_rows
            if item["scope"] == "all_units" and item["candidate"] == candidate
        )
        print(
            f"{candidate:24s} {float(row['normal_mean']):10.6f} "
            f"{float(row['frozen_mean']):10.6f} {float(row['shuffle_mean']):10.6f}"
        )
    print()
    print(f"Best third descriptor: {_best_third_descriptor(nn_rows, bms_rows)}")
    print("=" * 60)


def run_analysis(
    args: argparse.Namespace,
    bms_runner: BmsRunner | None = None,
) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    candidate_path = input_dir / "third_descriptor_candidates.csv"
    sanity_path = input_dir / "third_descriptor_video_sanity.csv"
    npz_path = input_dir / "third_descriptor_candidates.npz"
    for required in (candidate_path, sanity_path, npz_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_candidate_rows(candidate_path)
    sanity = read_sanity_rows(sanity_path, rows)
    validate_candidate_npz(npz_path, rows)
    scopes = build_scopes(rows)
    descriptor = np.asarray(
        [[float(row[column]) for column in DESCRIPTOR_COLUMNS] for row in rows],
        dtype=np.float64,
    )  # [N,5]
    effect = np.asarray(
        [float(row["ablation_logit_deviation"]) for row in rows],
        dtype=np.float64,
    )  # [N]

    device, torch_module = resolve_analysis_device(args.device)
    if bms_runner is None:
        bms_runner = lambda features: run_exact_bms_on_device(features, device)
    print(f"Analysis device: {device}")
    if device.startswith("cuda") and torch_module is not None:
        print(f"GPU: {torch_module.cuda.get_device_name(0)}")

    correlation, correlation_pvalue = compute_descriptor_correlation(descriptor)
    correlation_rows = []
    for row_index, row_name in enumerate(DESCRIPTOR_COLUMNS):
        correlation_rows.append(
            {
                "dimension": row_name,
                **{
                    name: float(correlation[row_index, column_index])
                    for column_index, name in enumerate(DESCRIPTOR_COLUMNS)
                },
            }
        )
    _write_csv(
        output_dir / "third_descriptor_correlation.csv",
        ["dimension", *DESCRIPTOR_COLUMNS],
        correlation_rows,
    )

    ablation_correlation_rows = []
    for scope, indices in scopes.items():
        for candidate, column in THIRD_COLUMNS.items():
            rho, pvalue = _safe_spearman(descriptor[indices, DESCRIPTOR_COLUMNS.index(column)], effect[indices])
            ablation_correlation_rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "spearman_rho_with_ablation": rho,
                    "spearman_pvalue": pvalue,
                    "num_units": len(indices),
                }
            )
    _write_csv(
        output_dir / "third_descriptor_ablation_correlation.csv",
        ("scope", "candidate", "spearman_rho_with_ablation", "spearman_pvalue", "num_units"),
        ablation_correlation_rows,
    )

    distance_rows: list[dict] = []
    nn_rows: list[dict] = []
    bms_rows: list[dict] = []
    bms_metadata: dict[str, dict] = {}
    normalization: dict[str, dict] = {}
    membership_path = output_dir / "third_descriptor_bms_membership.csv"
    with membership_path.open("w", encoding="utf-8", newline="") as handle:
        membership_writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope", "variant", "global_index", "group_id", "group_size"
            ),
        )
        membership_writer.writeheader()
        for scope, indices in scopes.items():
            scope_descriptor = descriptor[indices]  # [N_s,5]
            scope_effect = effect[indices]  # [N_s]
            scope_types = [rows[int(index)]["unit_type"] for index in indices]
            variants, norm = build_variants(scope_descriptor)
            normalization[scope] = norm
            first, second, pair_mode = select_pair_indices(
                len(indices),
                sample_size=args.pair_sample_size,
                seed=args.seed,
                exact_pair_limit=args.exact_pair_limit,
            )
            global_pair = mean_pairwise_absolute_difference(scope_effect)
            for variant, features in variants.items():
                rho, pvalue = distance_spearman(
                    features,
                    scope_effect,
                    first,
                    second,
                    args.pair_chunk_size,
                    device,
                    torch_module,
                )
                distance_rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "spearman_distance_rho": rho,
                        "spearman_pvalue": pvalue,
                        "num_units": len(indices),
                        "num_pairs": len(first),
                        "pair_mode": pair_mode,
                    }
                )
                neighbor, neighbor_distance = exact_nearest_neighbors(features)
                neighbor_effect_difference = np.abs(
                    scope_effect - scope_effect[neighbor]
                )  # [N_s]
                nn_rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "nn_error": float(neighbor_effect_difference.mean()),
                        "mean_descriptor_nn_distance": float(
                            neighbor_distance.mean()
                        ),
                        "num_units": len(indices),
                    }
                )

                groups, metadata = bms_runner(
                    scope_descriptor[
                        :,
                        [
                            DESCRIPTOR_COLUMNS.index(column)
                            for column in VARIANT_COLUMNS[variant]
                        ],
                    ]
                )
                groups = validate_group_membership(groups, len(indices))
                metrics, _ = evaluate_groups(groups, scope_effect, scope_types)
                random_metrics = random_baseline(
                    scope_effect,
                    [len(group) for group in groups],
                    args.seed,
                    args.random_repeats,
                    float(metrics["weighted_intra_ablation_variance"]),
                    float(metrics["pairwise_intra_ablation_difference"]),
                )
                q_pair = float(metrics["pairwise_intra_ablation_difference"])
                domain_gain = (
                    1.0 - q_pair / global_pair
                    if math.isfinite(q_pair)
                    and math.isfinite(global_pair)
                    and abs(global_pair) > EPSILON
                    else float("nan")
                )
                bms_rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "num_units": len(indices),
                        "num_groups": int(metrics["num_groups"]),
                        "singleton_groups": int(metrics["singleton_groups"]),
                        "singleton_ratio": float(metrics["singleton_ratio"]),
                        "q_var": float(
                            metrics["weighted_intra_ablation_variance"]
                        ),
                        "q_pair": q_pair,
                        "q_global": global_pair,
                        "r_domain": domain_gain,
                        **random_metrics,
                        "bms_sigma": metadata.get("sigma", float("nan")),
                        "bms_device": metadata.get("device", "unknown"),
                        "bms_source_sha256": metadata.get(
                            "bms_source_sha256", ""
                        ),
                    }
                )
                bms_metadata[f"{scope}/{variant}"] = metadata
                for group_id, members in enumerate(groups):
                    size = len(members)
                    for local_index in members:
                        membership_writer.writerow(
                            {
                                "scope": scope,
                                "variant": variant,
                                "global_index": int(
                                    rows[int(indices[local_index])]["global_index"]
                                ),
                                "group_id": group_id,
                                "group_size": size,
                            }
                        )
                if device.startswith("cuda") and torch_module is not None:
                    torch_module.cuda.empty_cache()

    _write_csv(
        output_dir / "third_descriptor_variant_summary.csv",
        (
            "scope", "variant", "spearman_distance_rho", "spearman_pvalue",
            "num_units", "num_pairs", "pair_mode",
        ),
        distance_rows,
    )
    _write_csv(
        output_dir / "third_descriptor_nn_summary.csv",
        (
            "scope", "variant", "nn_error", "mean_descriptor_nn_distance",
            "num_units",
        ),
        nn_rows,
    )
    bms_fields = (
        "scope", "variant", "num_units", "num_groups", "singleton_groups",
        "singleton_ratio", "q_var", "q_pair", "q_global", "r_domain",
        "random_variance_mean", "random_variance_std", "random_pairwise_mean",
        "random_pairwise_std", "relative_variance_improvement_vs_random",
        "relative_improvement_vs_random", "empirical_pvalue", "bms_sigma",
        "bms_device", "bms_source_sha256",
    )
    _write_csv(
        output_dir / "third_descriptor_bms_summary.csv", bms_fields, bms_rows
    )
    sanity_summary = compute_sanity_summary(sanity, scopes)
    _write_csv(
        output_dir / "third_descriptor_video_sanity_summary.csv",
        (
            "scope", "candidate", "num_units", "normal_mean", "normal_median",
            "frozen_mean", "frozen_median", "shuffle_mean", "shuffle_median",
            "mean_freeze_ratio", "median_freeze_ratio",
            "fraction_freeze_ratio_below_0.5", "mean_shuffle_sensitivity",
            "median_shuffle_sensitivity",
        ),
        sanity_summary,
    )

    _plot_correlation(output_dir / "third_descriptor_correlation.png", correlation)
    _plot_grouped_metric(
        output_dir / "third_descriptor_distance_spearman.png",
        distance_rows,
        "spearman_distance_rho",
        "Distance--ablation Spearman rho",
        "Descriptor distance consistency (higher is better)",
    )
    _plot_grouped_metric(
        output_dir / "third_descriptor_nnerror.png",
        nn_rows,
        "nn_error",
        "NNError",
        "Nearest-neighbor ablation consistency (lower is better)",
    )
    _plot_grouped_metric(
        output_dir / "third_descriptor_bms_pairdiff.png",
        bms_rows,
        "q_pair",
        "BMS Q_pair",
        "BMS domain pair difference (lower is better)",
    )
    _plot_grouped_metric(
        output_dir / "third_descriptor_bms_domain_gain.png",
        bms_rows,
        "r_domain",
        "BMS R_domain",
        "BMS domain gain (higher is better)",
    )
    _plot_attention_comparison(
        output_dir / "third_descriptor_attention_comparison.png",
        nn_rows,
        bms_rows,
    )
    _plot_video_sanity(
        output_dir / "third_descriptor_video_sanity.png", sanity_summary
    )
    write_validation_summary(
        output_dir / "validation_summary.md",
        correlation,
        nn_rows,
        bms_rows,
        sanity_summary,
    )
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "analysis_device": device,
        "gpu_first": device.startswith("cuda"),
        "gpu_names": (
            [
                torch_module.cuda.get_device_name(index)
                for index in range(torch_module.cuda.device_count())
            ]
            if device.startswith("cuda") and torch_module is not None
            else []
        ),
        "pair_sample_size": args.pair_sample_size,
        "exact_pair_limit": args.exact_pair_limit,
        "pair_chunk_size": args.pair_chunk_size,
        "pair_distance_dtype": "float64",
        "pair_distance_device": device,
        "random_repeats": args.random_repeats,
        "normalization": normalization,
        "correlation_pvalue_matrix": correlation_pvalue.tolist(),
        "bms": bms_metadata,
        "nearest_neighbor_backend": (
            "CPU scipy.spatial.cKDTree (exact low-dimensional search)"
        ),
        "command_line": [sys.executable, *sys.argv],
        "input_candidate_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_console_summary(
        rows,
        correlation,
        distance_rows,
        nn_rows,
        bms_rows,
        sanity_summary,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Task 009 video-specific descriptor candidates"
    )
    parser.add_argument("--input_dir", default="video_descriptor_validation")
    parser.add_argument("--output_dir", default="video_descriptor_validation")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA preferred), cuda:0, or cpu",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pair_sample_size", type=int, default=1_000_000)
    parser.add_argument("--exact_pair_limit", type=int, default=5_000_000)
    parser.add_argument("--pair_chunk_size", type=int, default=200_000)
    parser.add_argument("--random_repeats", type=int, default=100)
    args = parser.parse_args(argv)
    positive = (
        "pair_sample_size", "exact_pair_limit", "pair_chunk_size", "random_repeats"
    )
    invalid = [name for name in positive if getattr(args, name) <= 0]
    if invalid:
        parser.error(f"arguments must be positive: {invalid}")
    return args


if __name__ == "__main__":
    run_analysis(parse_args())

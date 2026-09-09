#!/usr/bin/env python3
"""Offline validation of BMS competition domains against real ablations.

The input is the complete Task 007 ``unit_ablation_effect.csv``.  This script
does not load a model, run inference, calibrate descriptors, ablate units, or
change pruning decisions.  It reruns the repository's existing BMS method on
seven descriptor variants and evaluates the resulting partitions with the
already measured ablation effect ``E in R^[N]``.

All exact pairwise absolute-difference statistics use a sorted-value formula;
the analysis never materializes a global ``[N,N]`` matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bms_competition_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = frozenset(
    {
        "global_index",
        "layer",
        "unit_type",
        "unit_index",
        "D_abs",
        "D_rel",
        "D_st",
        "ablation_logit_deviation",
    }
)
DESCRIPTOR_COLUMNS = ("D_abs", "D_rel", "D_st")
DESCRIPTOR_VARIANTS: OrderedDict[str, tuple[int, ...]] = OrderedDict(
    [
        ("D_abs", (0,)),
        ("D_rel", (1,)),
        ("D_st", (2,)),
        ("D_abs+D_rel", (0, 1)),
        ("D_abs+D_st", (0, 2)),
        ("D_rel+D_st", (1, 2)),
        ("Full3D", (0, 1, 2)),
    ]
)
FULL_VARIANT = "Full3D"
ABS_REL_VARIANT = "D_abs+D_rel"
EXPECTED_UNIT_TYPES = frozenset({"attention_head", "ffn_neuron"})
REQUIRED_SCOPES = (
    "all_units",
    "attention_only",
    "mlp_only",
    "stage0",
    "stage1",
    "stage2",
    "stage3",
)
EPSILON = 1e-8

SUMMARY_FIELDS = [
    "scope",
    "variant",
    "dimensions",
    "num_units",
    "num_groups",
    "singleton_groups",
    "singleton_ratio",
    "mean_group_size",
    "median_group_size",
    "min_group_size",
    "max_group_size",
    "multiunit_group_count",
    "multiunit_unit_count",
    "multiunit_unit_ratio",
    "weighted_intra_ablation_variance",
    "unweighted_intra_ablation_variance",
    "pairwise_intra_ablation_difference",
    "unweighted_group_pairwise_ablation_difference",
    "multiunit_pairwise_ablation_difference",
    "global_pairwise_ablation_difference",
    "domain_improvement_ratio",
    "random_variance_mean",
    "random_variance_std",
    "random_pairwise_mean",
    "random_pairwise_std",
    "relative_variance_improvement_vs_random",
    "relative_improvement_vs_random",
    "empirical_pvalue",
    "mean_multiunit_group_cv",
    "bms_sigma",
    "bms_source_sha256",
]

GROUP_DETAIL_FIELDS = [
    "scope",
    "variant",
    "group_id",
    "group_size",
    "attention_count",
    "mlp_count",
    "mean_ablation_effect",
    "std_ablation_effect",
    "variance_ablation_effect",
    "pairwise_ablation_difference",
    "min_ablation_effect",
    "max_ablation_effect",
    "ablation_range",
    "coefficient_of_variation",
    "mixed_type",
]

MEMBERSHIP_FIELDS = [
    "scope",
    "variant",
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_st",
    "ablation_logit_deviation",
    "group_id",
    "group_size",
]


@dataclass(frozen=True)
class UnitRecord:
    """One Task 007 row and its derived Video Swin stage."""

    global_index: int
    layer: str
    unit_type: str
    unit_index: int
    D_abs: float
    D_rel: float
    D_st: float
    ablation_logit_deviation: float
    stage: int

    @property
    def descriptor(self) -> tuple[float, float, float]:
        """Return ``v_i in R^3`` ordered as abs, rel, spatio-temporal."""

        return (self.D_abs, self.D_rel, self.D_st)


def _require_finite_matrix(
    values: np.ndarray, columns: int | None = None
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {matrix.shape}")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"expected {columns} columns, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("descriptor matrix must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("descriptor matrix contains NaN or infinity")
    return matrix


def _parse_finite_float(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number} has invalid numeric value for {column!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number} column {column!r} is not finite")
    return value


def stage_from_layer(layer: str) -> int:
    """Extract Video Swin stage ``s in {0,1,2,3}`` from a module name."""

    match = re.search(r"(?:^|\.)layers\.([0-3])(?:\.|$)", layer)
    if match is None:
        raise ValueError(f"cannot derive stage0-stage3 from layer name {layer!r}")
    return int(match.group(1))


def read_unit_records(path: Path) -> list[UnitRecord]:
    """Load and strictly validate the complete Task 007 CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path} is missing required columns {sorted(missing)}")
        source_rows = list(reader)

    if not source_rows:
        raise ValueError(f"{path} contains no pruning units")

    records: list[UnitRecord] = []
    for row_number, row in enumerate(source_rows, start=2):
        try:
            global_index = int(row["global_index"])
            unit_index = int(row["unit_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"row {row_number} has an invalid global_index or unit_index"
            ) from exc
        if global_index < 0 or unit_index < 0:
            raise ValueError(f"row {row_number} contains a negative unit index")

        layer = str(row["layer"]).strip()
        unit_type = str(row["unit_type"]).strip()
        if not layer:
            raise ValueError(f"row {row_number} has an empty layer")
        if unit_type not in EXPECTED_UNIT_TYPES:
            raise ValueError(
                f"row {row_number} has unsupported unit_type {unit_type!r}"
            )

        records.append(
            UnitRecord(
                global_index=global_index,
                layer=layer,
                unit_type=unit_type,
                unit_index=unit_index,
                D_abs=_parse_finite_float(row, "D_abs", row_number),
                D_rel=_parse_finite_float(row, "D_rel", row_number),
                D_st=_parse_finite_float(row, "D_st", row_number),
                ablation_logit_deviation=_parse_finite_float(
                    row, "ablation_logit_deviation", row_number
                ),
                stage=stage_from_layer(layer),
            )
        )

    records.sort(key=lambda record: record.global_index)
    actual_indices = [record.global_index for record in records]
    if actual_indices != list(range(len(records))):
        raise ValueError("global_index must be unique and contiguous from zero")

    unit_keys = [
        (record.layer, record.unit_type, record.unit_index) for record in records
    ]
    if len(set(unit_keys)) != len(unit_keys):
        raise ValueError("(layer, unit_type, unit_index) must identify each row uniquely")
    return records


def build_scopes(records: Sequence[UnitRecord]) -> OrderedDict[str, np.ndarray]:
    """Return global, type-specific, and stage-specific row-index vectors."""

    scopes: OrderedDict[str, np.ndarray] = OrderedDict()
    scopes["all_units"] = np.arange(len(records), dtype=np.int64)
    scopes["attention_only"] = np.asarray(
        [i for i, record in enumerate(records) if record.unit_type == "attention_head"],
        dtype=np.int64,
    )
    scopes["mlp_only"] = np.asarray(
        [i for i, record in enumerate(records) if record.unit_type == "ffn_neuron"],
        dtype=np.int64,
    )
    for stage in range(4):
        scopes[f"stage{stage}"] = np.asarray(
            [i for i, record in enumerate(records) if record.stage == stage],
            dtype=np.int64,
        )

    for scope_name in REQUIRED_SCOPES:
        count = int(scopes[scope_name].size)
        if count < 2:
            raise ValueError(
                f"scope {scope_name!r} has {count} units; independent BMS needs >=2"
            )
    return scopes


def build_descriptor_variants(
    descriptor: np.ndarray,
) -> OrderedDict[str, np.ndarray]:
    """Select the exact seven raw descriptor variants before BMS scaling."""

    values = _require_finite_matrix(descriptor, columns=3)
    return OrderedDict(
        (name, values[:, dimensions].copy())
        for name, dimensions in DESCRIPTOR_VARIANTS.items()
    )


@lru_cache(maxsize=1)
def _load_current_bms() -> tuple[object, type, float, str]:
    """Lazily load the repository's exact production BMS implementation."""

    try:
        import torch
        from MC import InteractionPruner
    except ImportError as exc:  # pragma: no cover - depends on server runtime.
        raise RuntimeError(
            "The existing BMS requires the repository PyTorch environment. "
            "Install/activate MC_Pruning with torch, timm, and einops; CUDA is "
            "not required because Task 008 deliberately runs BMS on CPU."
        ) from exc

    sigma_parameter = inspect.signature(InteractionPruner.__init__).parameters.get(
        "sigma"
    )
    if sigma_parameter is None or sigma_parameter.default is inspect.Parameter.empty:
        raise RuntimeError("InteractionPruner.__init__ no longer exposes default sigma")
    sigma = float(sigma_parameter.default)
    source = inspect.getsource(InteractionPruner.mean_shift_clustering)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return torch, InteractionPruner, sigma, source_sha256


def current_bms_runtime_available() -> bool:
    """Return whether the exact repository BMS and its dependencies import."""

    try:
        _load_current_bms()
    except RuntimeError:
        return False
    return True


def validate_group_membership(
    groups: Sequence[Sequence[int]], num_units: int
) -> list[list[int]]:
    """Require a non-overlapping partition covering local indices ``0..N-1``."""

    if num_units <= 0:
        raise ValueError("num_units must be positive")
    normalized: list[list[int]] = []
    flattened: list[int] = []
    for group_id, group in enumerate(groups):
        members = sorted(int(index) for index in group)
        if not members:
            raise ValueError(f"group {group_id} is empty")
        if members[0] < 0 or members[-1] >= num_units:
            raise ValueError(f"group {group_id} contains an out-of-range index")
        normalized.append(members)
        flattened.extend(members)
    if sorted(flattened) != list(range(num_units)):
        raise ValueError("BMS group membership must cover every unit exactly once")
    return normalized


def canonical_partition(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Canonical group ordering used only by the BMS equivalence test."""

    return tuple(sorted(tuple(sorted(int(i) for i in group)) for group in groups))


def run_current_bms(features: np.ndarray) -> tuple[list[list[int]], dict[str, object]]:
    """Standardize and call ``InteractionPruner.mean_shift_clustering`` exactly.

    ``features`` has shape ``[N,d]``, where ``d in {1,2,3}``; rows are pruning
    units and columns are the selected descriptor axes.  The CPU float32 mean
    and default PyTorch sample standard deviation reproduce ``group_pruning``.
    No alternative clustering implementation or sigma override is provided.
    """

    values = _require_finite_matrix(features)
    if values.shape[0] < 2:
        raise ValueError("BMS requires at least two pruning units")
    torch, pruner_class, sigma, source_sha256 = _load_current_bms()

    descriptor = torch.as_tensor(values, dtype=torch.float32, device="cpu")  # [N,d]
    with torch.no_grad():
        mean = descriptor.mean(dim=0, keepdim=True)  # [1,d]
        std = descriptor.std(dim=0, keepdim=True)  # [1,d], default correction=1
        normalized = (descriptor - mean) / (std + EPSILON)  # [N,d]
        if not torch.isfinite(normalized).all():
            raise ValueError("standardized BMS descriptor contains NaN or infinity")

        pruner = pruner_class.__new__(pruner_class)
        pruner.sigma = sigma
        groups, trajectories, endpoints = pruner_class.mean_shift_clustering(
            pruner, normalized
        )

    partition = validate_group_membership(groups, values.shape[0])
    metadata = {
        "sigma": sigma,
        "max_iters": 60,
        "tolerance": 1e-4,
        "sink_rounding_decimals": 2,
        "device": "cpu",
        "dtype": "torch.float32",
        "standardization": "torch mean/std, default sample correction, eps=1e-8",
        "bms_source_sha256": source_sha256,
        "trajectory_shape": list(trajectories.shape),
        "endpoint_shape": list(endpoints.shape),
    }
    return partition, metadata


def pairwise_absolute_sum(values: np.ndarray) -> float:
    """Return exact ``sum_(i<j) |x_j-x_i|`` with ``O(N)`` extra memory."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("pairwise values must be a finite one-dimensional vector")
    size = vector.size
    if size < 2:
        return 0.0
    ordered = np.sort(vector)  # [N], never [N,N]
    coefficients = 2.0 * np.arange(size, dtype=np.float64) - size + 1.0  # [N]
    total = float(np.dot(coefficients, ordered))
    if total < 0.0 and abs(total) <= 1e-12:
        total = 0.0
    if total < 0.0:
        raise RuntimeError("sorted pairwise absolute-difference sum became negative")
    return total


def mean_pairwise_absolute_difference(values: np.ndarray) -> float:
    """Return exact mean ``|x_i-x_j|`` over unique pairs, or NaN for N<2."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("pairwise values must be one-dimensional")
    pair_count = vector.size * (vector.size - 1) // 2
    if pair_count == 0:
        return float("nan")
    return pairwise_absolute_sum(vector) / pair_count


def random_groups_same_sizes(
    num_units: int, group_sizes: Sequence[int], rng: np.random.Generator
) -> list[np.ndarray]:
    """Randomly partition all units while preserving the ordered size multiset."""

    sizes = [int(size) for size in group_sizes]
    if any(size <= 0 for size in sizes) or sum(sizes) != num_units:
        raise ValueError("positive group sizes must sum to num_units")
    permutation = rng.permutation(num_units)
    groups: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        groups.append(permutation[offset : offset + size])
        offset += size
    return groups


def _mixed_type_label(attention_count: int, mlp_count: int) -> str:
    if attention_count and mlp_count:
        return "mixed_attention_mlp"
    if attention_count:
        return "attention_only"
    return "mlp_only"


def evaluate_groups(
    groups: Sequence[Sequence[int]],
    effects: np.ndarray,
    unit_types: Sequence[str],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Compute exact competition-domain statistics for one BMS partition."""

    effect = np.asarray(effects, dtype=np.float64)
    if effect.ndim != 1 or effect.size == 0 or not np.isfinite(effect).all():
        raise ValueError("effects must be a non-empty finite vector E in R^[N]")
    if len(unit_types) != effect.size:
        raise ValueError("unit_types length differs from ablation-effect length")
    partition = validate_group_membership(groups, effect.size)

    details: list[dict[str, float | int | str]] = []
    total_squared_deviation = 0.0
    total_pairwise_difference = 0.0
    total_pair_count = 0
    group_variances: list[float] = []
    group_pair_means: list[float] = []
    multiunit_cvs: list[float] = []
    sizes: list[int] = []

    for group_id, members in enumerate(partition):
        indices = np.asarray(members, dtype=np.int64)
        group_effect = effect[indices]  # [G]
        size = int(indices.size)
        mean = float(group_effect.mean())
        variance = float(np.mean((group_effect - mean) ** 2))
        std = math.sqrt(max(variance, 0.0))
        pair_count = size * (size - 1) // 2
        pair_sum = pairwise_absolute_sum(group_effect)
        pair_mean = pair_sum / pair_count if pair_count else float("nan")
        coefficient_of_variation = std / (abs(mean) + EPSILON)
        attention_count = sum(unit_types[index] == "attention_head" for index in members)
        mlp_count = size - attention_count

        total_squared_deviation += variance * size
        total_pairwise_difference += pair_sum
        total_pair_count += pair_count
        group_variances.append(variance)
        sizes.append(size)
        if pair_count:
            group_pair_means.append(pair_mean)
            multiunit_cvs.append(coefficient_of_variation)

        details.append(
            {
                "group_id": group_id,
                "group_size": size,
                "attention_count": attention_count,
                "mlp_count": mlp_count,
                "mean_ablation_effect": mean,
                "std_ablation_effect": std,
                "variance_ablation_effect": variance,
                "pairwise_ablation_difference": pair_mean,
                "min_ablation_effect": float(group_effect.min()),
                "max_ablation_effect": float(group_effect.max()),
                "ablation_range": float(group_effect.max() - group_effect.min()),
                "coefficient_of_variation": coefficient_of_variation,
                "mixed_type": _mixed_type_label(attention_count, mlp_count),
            }
        )

    size_array = np.asarray(sizes, dtype=np.int64)
    singleton_groups = int(np.count_nonzero(size_array == 1))
    multiunit_group_count = len(partition) - singleton_groups
    multiunit_unit_count = int(size_array[size_array >= 2].sum())
    pairwise = (
        total_pairwise_difference / total_pair_count
        if total_pair_count
        else float("nan")
    )
    metrics: dict[str, float | int] = {
        "num_units": int(effect.size),
        "num_groups": len(partition),
        "singleton_groups": singleton_groups,
        "singleton_ratio": singleton_groups / len(partition),
        "mean_group_size": float(size_array.mean()),
        "median_group_size": float(np.median(size_array)),
        "min_group_size": int(size_array.min()),
        "max_group_size": int(size_array.max()),
        "multiunit_group_count": multiunit_group_count,
        "multiunit_unit_count": multiunit_unit_count,
        "multiunit_unit_ratio": multiunit_unit_count / effect.size,
        "weighted_intra_ablation_variance": total_squared_deviation / effect.size,
        "unweighted_intra_ablation_variance": float(np.mean(group_variances)),
        "pairwise_intra_ablation_difference": pairwise,
        "unweighted_group_pairwise_ablation_difference": (
            float(np.mean(group_pair_means)) if group_pair_means else float("nan")
        ),
        "multiunit_pairwise_ablation_difference": pairwise,
        "mean_multiunit_group_cv": (
            float(np.mean(multiunit_cvs)) if multiunit_cvs else float("nan")
        ),
    }
    return metrics, details


def primary_partition_metrics(
    groups: Sequence[Sequence[int]],
    effects: np.ndarray,
    *,
    validate: bool = True,
) -> tuple[float, float]:
    """Return only ``Q_var`` and ``Q_pair`` for repeated random partitions.

    This lightweight path avoids constructing group-detail dictionaries during
    the random baseline.  ``effects`` has shape ``[N]`` and every temporary
    group index vector has shape ``[|G_k|]``; no square matrix is created.
    """

    effect = np.asarray(effects, dtype=np.float64)
    if effect.ndim != 1 or effect.size == 0 or not np.isfinite(effect).all():
        raise ValueError("effects must be a non-empty finite vector")
    partition: Sequence[Sequence[int]] = (
        validate_group_membership(groups, effect.size) if validate else groups
    )
    total_squared_deviation = 0.0
    total_pairwise_difference = 0.0
    total_pair_count = 0
    for members in partition:
        group_effect = effect[np.asarray(members, dtype=np.int64)]  # [G]
        mean = float(group_effect.mean())
        total_squared_deviation += float(np.sum((group_effect - mean) ** 2))
        pair_count = len(members) * (len(members) - 1) // 2
        if pair_count:
            total_pairwise_difference += pairwise_absolute_sum(group_effect)
            total_pair_count += pair_count
    variance = total_squared_deviation / effect.size
    pairwise = (
        total_pairwise_difference / total_pair_count
        if total_pair_count
        else float("nan")
    )
    return variance, pairwise


def random_baseline(
    effects: np.ndarray,
    group_sizes: Sequence[int],
    seed: int,
    repeats: int,
    bms_variance: float,
    bms_pairwise: float,
) -> dict[str, float]:
    """Evaluate 100-by-default same-size random partitions."""

    if repeats <= 0:
        raise ValueError("random repetition count must be positive")
    rng = np.random.default_rng(seed)
    random_variance = np.empty(repeats, dtype=np.float64)  # [R]
    random_pairwise = np.empty(repeats, dtype=np.float64)  # [R]
    for repeat in range(repeats):
        groups = random_groups_same_sizes(len(effects), group_sizes, rng)
        variance, pairwise = primary_partition_metrics(
            groups, effects, validate=False
        )
        random_variance[repeat] = variance
        random_pairwise[repeat] = pairwise

    variance_mean = float(random_variance.mean())
    std_ddof = 1 if repeats > 1 else 0
    valid_pairwise = random_pairwise[np.isfinite(random_pairwise)]
    pairwise_mean = (
        float(valid_pairwise.mean()) if valid_pairwise.size else float("nan")
    )
    pairwise_std = (
        float(valid_pairwise.std(ddof=std_ddof))
        if valid_pairwise.size > std_ddof
        else float("nan")
    )
    empirical_pvalue = (
        float(
            (1 + np.count_nonzero(valid_pairwise <= bms_pairwise))
            / (valid_pairwise.size + 1)
        )
        if valid_pairwise.size and math.isfinite(bms_pairwise)
        else float("nan")
    )
    return {
        "random_variance_mean": variance_mean,
        "random_variance_std": float(random_variance.std(ddof=std_ddof)),
        "random_pairwise_mean": pairwise_mean,
        "random_pairwise_std": pairwise_std,
        "relative_variance_improvement_vs_random": (
            (variance_mean - bms_variance) / variance_mean
            if variance_mean > 0.0
            else float("nan")
        ),
        "relative_improvement_vs_random": (
            (pairwise_mean - bms_pairwise) / pairwise_mean
            if math.isfinite(pairwise_mean)
            and math.isfinite(bms_pairwise)
            and pairwise_mean > 0.0
            else float("nan")
        ),
        "empirical_pvalue": empirical_pvalue,
    }


def cross_type_statistics(
    groups: Sequence[Sequence[int]], unit_types: Sequence[str]
) -> dict[str, int | float]:
    """Summarize actual mixed Attention/MLP Full3D multi-unit domains."""

    counts = Counter()
    mixed_unit_count = 0
    mixed_attention_count = 0
    mixed_mlp_count = 0
    multiunit_count = 0
    for members in groups:
        if len(members) < 2:
            continue
        multiunit_count += 1
        attention_count = sum(unit_types[index] == "attention_head" for index in members)
        mlp_count = len(members) - attention_count
        label = _mixed_type_label(attention_count, mlp_count)
        counts[label] += 1
        if label == "mixed_attention_mlp":
            mixed_unit_count += len(members)
            mixed_attention_count += attention_count
            mixed_mlp_count += mlp_count

    mixed_groups = counts["mixed_attention_mlp"]
    return {
        "multiunit_group_count": multiunit_count,
        "attention_only_groups": counts["attention_only"],
        "mlp_only_groups": counts["mlp_only"],
        "mixed_attention_mlp_groups": mixed_groups,
        "mixed_group_ratio": mixed_groups / multiunit_count if multiunit_count else 0.0,
        "mixed_unit_count": mixed_unit_count,
        "mixed_unit_ratio": mixed_unit_count / len(unit_types) if unit_types else 0.0,
        "attention_units_in_mixed_groups": mixed_attention_count,
        "mlp_units_in_mixed_groups": mixed_mlp_count,
    }


def _group_size_bucket(size: int) -> str:
    if size == 2:
        return "2"
    if size <= 4:
        return "3-4"
    if size <= 8:
        return "5-8"
    if size <= 16:
        return "9-16"
    if size <= 32:
        return "17-32"
    return ">32"


def group_size_consistency_rows(
    group_details: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate Full3D multi-unit behavior by the required size buckets."""

    bucket_order = ("2", "3-4", "5-8", "9-16", "17-32", ">32")
    buckets: dict[str, list[dict[str, float | int | str]]] = {
        label: [] for label in bucket_order
    }
    for detail in group_details:
        size = int(detail["group_size"])
        if size >= 2:
            buckets[_group_size_bucket(size)].append(detail)

    rows: list[dict[str, float | int | str]] = []
    for label in bucket_order:
        details = buckets[label]
        rows.append(
            {
                "group_size_bucket": label,
                "number_of_groups": len(details),
                "number_of_units": sum(int(row["group_size"]) for row in details),
                "mean_intra_group_variance": (
                    float(
                        np.mean(
                            [float(row["variance_ablation_effect"]) for row in details]
                        )
                    )
                    if details
                    else float("nan")
                ),
                "mean_pairwise_ablation_difference": (
                    float(
                        np.mean(
                            [
                                float(row["pairwise_ablation_difference"])
                                for row in details
                            ]
                        )
                    )
                    if details
                    else float("nan")
                ),
            }
        )
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= EPSILON:
        return float("nan")
    return numerator / denominator


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_units_rows(summary_rows: Sequence[dict]) -> list[dict]:
    by_variant = {
        row["variant"]: row for row in summary_rows if row["scope"] == "all_units"
    }
    missing = set(DESCRIPTOR_VARIANTS) - set(by_variant)
    if missing:
        raise RuntimeError(f"all_units summary missing variants {sorted(missing)}")
    return [by_variant[name] for name in DESCRIPTOR_VARIANTS]


def _plot_variant_metric(
    path: Path,
    rows: Sequence[dict],
    metric: str,
    ylabel: str,
    lower_is_better: bool,
) -> None:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = ["#4C78A8"] * len(rows)
    if np.isfinite(values).any():
        best = int(np.nanargmin(values) if lower_is_better else np.nanargmax(values))
        colors[best] = "#E45756"
    axis.bar(range(len(rows)), values, color=colors)
    axis.set_xticks(range(len(rows)), [row["variant"] for row in rows], rotation=25)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_bms_vs_random(path: Path, rows: Sequence[dict]) -> None:
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.38
    bms = np.asarray(
        [float(row["pairwise_intra_ablation_difference"]) for row in rows]
    )
    random_mean = np.asarray([float(row["random_pairwise_mean"]) for row in rows])
    random_std = np.asarray([float(row["random_pairwise_std"]) for row in rows])
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.bar(x - width / 2, bms, width, label="BMS", color="#4C78A8")
    axis.bar(
        x + width / 2,
        random_mean,
        width,
        yerr=random_std,
        capsize=3,
        label="Random same-size grouping",
        color="#F2CF5B",
    )
    axis.set_xticks(x, [row["variant"] for row in rows], rotation=25)
    axis.set_ylabel("Mean intra-domain pairwise ablation difference")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_group_size_distribution(path: Path, groups: Sequence[Sequence[int]]) -> None:
    counts = Counter(len(group) for group in groups)
    sizes = np.asarray(sorted(counts), dtype=np.int64)
    frequencies = np.asarray([counts[int(size)] for size in sizes], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(sizes, frequencies, color="#59A14F", width=0.8)
    axis.set_xlabel("Full3D BMS group size")
    axis.set_ylabel("Group count")
    positive = frequencies[frequencies > 0]
    if positive.size and positive.max() / positive.min() >= 1000:
        axis.set_yscale("log")
        axis.set_ylabel("Group count (log scale)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_group_behavior(
    path: Path, group_details: Sequence[dict[str, float | int | str]]
) -> None:
    multi = [row for row in group_details if int(row["group_size"]) >= 2]
    sizes = np.asarray([int(row["group_size"]) for row in multi], dtype=np.int64)
    differences = np.asarray(
        [float(row["pairwise_ablation_difference"]) for row in multi],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(sizes, differences, s=20, alpha=0.55, color="#B279A2")
    axis.set_xlabel("Full3D BMS group size")
    axis.set_ylabel("Intra-group pairwise ablation difference")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _fmt(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NaN"
    return f"{number:.6g}"


def _find_summary(
    summary_rows: Sequence[dict], scope: str, variant: str
) -> dict:
    for row in summary_rows:
        if row["scope"] == scope and row["variant"] == variant:
            return row
    raise KeyError(f"missing summary row for {scope}/{variant}")


def _finite_metric(row: dict, key: str, fallback: float) -> float:
    value = float(row[key])
    return value if math.isfinite(value) else fallback


def write_validation_summary(
    path: Path,
    summary_rows: Sequence[dict],
    cross_type: dict[str, int | float],
) -> None:
    """Write a neutral, number-driven answer to the five scientific questions."""

    all_rows = _all_units_rows(summary_rows)
    full = _find_summary(summary_rows, "all_units", FULL_VARIANT)
    abs_rel = _find_summary(summary_rows, "all_units", ABS_REL_VARIANT)

    random_wins = sum(
        float(row["pairwise_intra_ablation_difference"])
        < float(row["random_pairwise_mean"])
        for row in all_rows
    )
    pair_ranking = sorted(
        all_rows,
        key=lambda row: _finite_metric(
            row, "pairwise_intra_ablation_difference", math.inf
        ),
    )
    variance_ranking = sorted(
        all_rows,
        key=lambda row: _finite_metric(
            row, "weighted_intra_ablation_variance", math.inf
        ),
    )
    full_pair_rank: int | None = (
        1
        + next(
            index
            for index, row in enumerate(pair_ranking)
            if row["variant"] == FULL_VARIANT
        )
        if math.isfinite(float(full["pairwise_intra_ablation_difference"]))
        else None
    )
    full_variance_rank = 1 + next(
        index
        for index, row in enumerate(variance_ranking)
        if row["variant"] == FULL_VARIANT
    )

    full_pair = float(full["pairwise_intra_ablation_difference"])
    abs_rel_pair = float(abs_rel["pairwise_intra_ablation_difference"])
    delta_pair = full_pair - abs_rel_pair
    relative_pair_change = _safe_ratio(abs_rel_pair - full_pair, abs_rel_pair)
    delta_variance = float(full["weighted_intra_ablation_variance"]) - float(
        abs_rel["weighted_intra_ablation_variance"]
    )
    delta_gain = float(full["domain_improvement_ratio"]) - float(
        abs_rel["domain_improvement_ratio"]
    )

    scope_comparisons: list[str] = []
    improved_scopes: list[str] = []
    random_consistent_scopes: list[str] = []
    for scope in REQUIRED_SCOPES:
        scope_full = _find_summary(summary_rows, scope, FULL_VARIANT)
        scope_abs_rel = _find_summary(summary_rows, scope, ABS_REL_VARIANT)
        change = float(scope_full["pairwise_intra_ablation_difference"]) - float(
            scope_abs_rel["pairwise_intra_ablation_difference"]
        )
        scope_comparisons.append(f"- `{scope}`: Delta Q_pair = {_fmt(change)}")
        if change < 0:
            improved_scopes.append(scope)
        if float(scope_full["relative_improvement_vs_random"]) > 0:
            random_consistent_scopes.append(scope)

    conditional_statement = (
        "Observed inequalities permit the narrower interpretation that adding "
        "the spatio-temporal descriptor improves local behavioral homogeneity "
        "for the all-unit BMS domains."
        if delta_pair < 0 or delta_variance < 0
        else (
            "The required inequalities do not hold, so the experiment does not "
            "support the claim that adding D_st improves all-unit BMS-domain "
            "behavioral homogeneity."
        )
    )

    lines = [
        "# BMS Competition-Domain Validation Summary",
        "",
        "This report uses only the measured Task 007 unit-ablation effects.  "
        "Lower `Q_var` and `Q_pair` are better; higher `R_domain` is better.",
        "",
        "## Question 1 — BMS versus random same-size groups",
        "",
        (
            f"For `all_units`, BMS has lower pairwise difference than the "
            f"same-size random mean for {random_wins}/7 descriptor variants. "
            f"Full3D: BMS={_fmt(full_pair)}, random={_fmt(full['random_pairwise_mean'])} "
            f"± {_fmt(full['random_pairwise_std'])}, relative improvement="
            f"{_fmt(full['relative_improvement_vs_random'])}, empirical p="
            f"{_fmt(full['empirical_pvalue'])}."
        ),
        "",
        "This is a permutation-based empirical comparison; the permutations "
        "reuse the same units and are not described as an independent formal test.",
        "",
        "## Question 2 — Full3D versus 1D/2D variants",
        "",
        (
            f"Full3D ranks "
            f"{str(full_pair_rank) + '/7' if full_pair_rank is not None else 'unavailable'} "
            f"by `Q_pair` and "
            f"{full_variance_rank}/7 by `Q_var`. The best `Q_pair` variant is "
            f"`{pair_ranking[0]['variant'] if math.isfinite(float(pair_ranking[0]['pairwise_intra_ablation_difference'])) else 'unavailable'}` "
            f"and the best `Q_var` variant is "
            f"`{variance_ranking[0]['variant']}`. Full3D has singleton ratio "
            f"{_fmt(full['singleton_ratio'])} and multi-unit coverage "
            f"{_fmt(full['multiunit_unit_ratio'])}."
        ),
        (
            "Full3D is best on both primary within-domain metrics."
            if full_pair_rank == 1 and full_variance_rank == 1
            else "The stronger claim that Full3D outperforms every lower-dimensional variant is not supported on both primary metrics."
        ),
        "",
        "## Question 3 — Adding D_st to D_abs+D_rel",
        "",
        f"- `Delta Q_pair = Q_pair(Full3D) - Q_pair(Abs+Rel) = {_fmt(delta_pair)}`",
        f"- relative `Q_pair` reduction = {_fmt(relative_pair_change)}",
        f"- `Delta Q_var = {_fmt(delta_variance)}`",
        f"- `Delta R_domain = {_fmt(delta_gain)}`",
        "",
        conditional_statement,
        "",
        "## Question 4 — Unit-type and stage consistency",
        "",
        (
            f"Full3D improves `Q_pair` over Abs+Rel in "
            f"{len(improved_scopes)}/{len(REQUIRED_SCOPES)} scopes: "
            f"{', '.join(improved_scopes) if improved_scopes else 'none'}."
        ),
        (
            f"Full3D outperforms its same-size random mean in "
            f"{len(random_consistent_scopes)}/{len(REQUIRED_SCOPES)} scopes: "
            f"{', '.join(random_consistent_scopes) if random_consistent_scopes else 'none'}."
        ),
        *scope_comparisons,
        "",
        "## Question 5 — Mixed Attention/MLP Full3D domains",
        "",
        (
            f"Among {_fmt(cross_type['multiunit_group_count'])} Full3D multi-unit "
            f"domains, {_fmt(cross_type['mixed_attention_mlp_groups'])} are mixed "
            f"Attention/MLP domains (`r_mixed={_fmt(cross_type['mixed_group_ratio'])}`). "
            f"They contain {_fmt(cross_type['mixed_unit_count'])} units "
            f"({_fmt(cross_type['mixed_unit_ratio'])} of all units), including "
            f"{_fmt(cross_type['attention_units_in_mixed_groups'])} Attention Heads "
            f"and {_fmt(cross_type['mlp_units_in_mixed_groups'])} FFN Neurons."
        ),
        "",
        "No mixed domain was forced; these are the observed BMS assignments.",
        "",
        "## Scientific boundary",
        "",
        "This experiment tests whether the existing descriptor induces BMS local "
        "domains that are behaviorally coherent under measured unit ablation.  It "
        "does not claim that the three axes completely describe a pruning unit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(
    summary_rows: Sequence[dict],
    num_units: int,
    cross_type: dict[str, int | float],
) -> None:
    all_rows = _all_units_rows(summary_rows)
    finite_pair_rows = [
        row
        for row in all_rows
        if math.isfinite(float(row["pairwise_intra_ablation_difference"]))
    ]
    finite_gain_rows = [
        row
        for row in all_rows
        if math.isfinite(float(row["domain_improvement_ratio"]))
    ]
    best_pair = (
        min(
            finite_pair_rows,
            key=lambda row: float(row["pairwise_intra_ablation_difference"]),
        )
        if finite_pair_rows
        else None
    )
    best_gain = (
        max(
            finite_gain_rows,
            key=lambda row: float(row["domain_improvement_ratio"]),
        )
        if finite_gain_rows
        else None
    )
    full = _find_summary(summary_rows, "all_units", FULL_VARIANT)
    print("=" * 76)
    print("BMS Competition-Domain Validation")
    print("-" * 76)
    print(f"Units: {num_units}")
    print("\nScope: all_units\n")
    print(f"{'Variant':<18}{'Groups':>9}{'Singleton%':>13}{'PairDiff':>14}{'DomainGain':>14}")
    print("-" * 76)
    for row in all_rows:
        print(
            f"{row['variant']:<18}{int(row['num_groups']):>9}"
            f"{100.0 * float(row['singleton_ratio']):>12.2f}%"
            f"{float(row['pairwise_intra_ablation_difference']):>14.6g}"
            f"{float(row['domain_improvement_ratio']):>14.6g}"
        )
    print(
        "\nBest PairDiff variant: "
        f"{best_pair['variant'] if best_pair is not None else 'unavailable'}"
    )
    print(
        "Best DomainGain variant: "
        f"{best_gain['variant'] if best_gain is not None else 'unavailable'}"
    )
    print("\nFull3D vs random:")
    print(f"PairDiff BMS: {_fmt(full['pairwise_intra_ablation_difference'])}")
    print(
        "Random mean ± std: "
        f"{_fmt(full['random_pairwise_mean'])} ± {_fmt(full['random_pairwise_std'])}"
    )
    print(f"Relative improvement: {_fmt(full['relative_improvement_vs_random'])}")
    print(f"Empirical p-value: {_fmt(full['empirical_pvalue'])}")
    print("\nMixed Attention/MLP Full3D domains:")
    print(
        f"{cross_type['mixed_attention_mlp_groups']}/"
        f"{cross_type['multiunit_group_count']} multi-unit groups; "
        f"{cross_type['mixed_unit_count']} units"
    )
    print("=" * 76)


BmsRunner = Callable[[np.ndarray], tuple[list[list[int]], dict[str, object]]]


def run_analysis(args: argparse.Namespace, bms_runner: BmsRunner = run_current_bms) -> None:
    """Execute every required ``scope x descriptor variant`` comparison."""

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if output_dir.resolve() == input_csv.parent.resolve():
        raise ValueError(
            "output_dir must differ from descriptor_ablation_validation; "
            "Task 008 must not overwrite Task 007 results"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_unit_records(input_csv)
    descriptors = np.asarray([record.descriptor for record in records], dtype=np.float64)
    effects = np.asarray(
        [record.ablation_logit_deviation for record in records], dtype=np.float64
    )
    scopes = build_scopes(records)
    attention_count = sum(record.unit_type == "attention_head" for record in records)
    mlp_count = len(records) - attention_count
    print(f"Loaded units: {len(records)}")
    print(f"Attention Heads: {attention_count}")
    print(f"FFN Neurons: {mlp_count}")

    summary_rows: list[dict] = []
    bms_metadata: dict[str, dict[str, object]] = {}
    full_groups: list[list[int]] | None = None
    full_group_details: list[dict[str, float | int | str]] | None = None
    full_unit_types: list[str] | None = None

    details_path = output_dir / "bms_group_details.csv"
    membership_path = output_dir / "bms_unit_membership.csv"
    with details_path.open("w", encoding="utf-8", newline="") as details_handle, \
        membership_path.open("w", encoding="utf-8", newline="") as membership_handle:
        details_writer = csv.DictWriter(details_handle, fieldnames=GROUP_DETAIL_FIELDS)
        membership_writer = csv.DictWriter(
            membership_handle, fieldnames=MEMBERSHIP_FIELDS
        )
        details_writer.writeheader()
        membership_writer.writeheader()

        for scope_name, scope_indices in scopes.items():
            scope_records = [records[int(index)] for index in scope_indices]
            scope_descriptor = descriptors[scope_indices]  # [Ns,3]
            scope_effect = effects[scope_indices]  # [Ns]
            scope_unit_types = [record.unit_type for record in scope_records]
            variants = build_descriptor_variants(scope_descriptor)
            global_pairwise = mean_pairwise_absolute_difference(scope_effect)
            print(f"\nScope: {scope_name} ({len(scope_records)} units)")

            for variant_name, features in variants.items():
                print(
                    f"\nRunning existing BMS: scope={scope_name}, "
                    f"variant={variant_name}, shape={tuple(features.shape)}"
                )
                groups, metadata = bms_runner(features)
                groups = validate_group_membership(groups, len(scope_records))
                metrics, group_details = evaluate_groups(
                    groups, scope_effect, scope_unit_types
                )
                random_metrics = random_baseline(
                    scope_effect,
                    [len(group) for group in groups],
                    seed=args.seed,
                    repeats=args.random_repeats,
                    bms_variance=float(
                        metrics["weighted_intra_ablation_variance"]
                    ),
                    bms_pairwise=float(
                        metrics["pairwise_intra_ablation_difference"]
                    ),
                )
                domain_gain = 1.0 - _safe_ratio(
                    float(metrics["pairwise_intra_ablation_difference"]),
                    global_pairwise,
                )
                summary = {
                    "scope": scope_name,
                    "variant": variant_name,
                    "dimensions": "|".join(
                        DESCRIPTOR_COLUMNS[index]
                        for index in DESCRIPTOR_VARIANTS[variant_name]
                    ),
                    **metrics,
                    "global_pairwise_ablation_difference": global_pairwise,
                    "domain_improvement_ratio": domain_gain,
                    **random_metrics,
                    "bms_sigma": metadata.get("sigma", float("nan")),
                    "bms_source_sha256": metadata.get("bms_source_sha256", ""),
                }
                summary_rows.append(summary)
                bms_metadata[f"{scope_name}/{variant_name}"] = metadata

                group_sizes = {
                    int(row["group_id"]): int(row["group_size"])
                    for row in group_details
                }
                group_ids = np.empty(len(scope_records), dtype=np.int64)  # [Ns]
                for group_id, members in enumerate(groups):
                    group_ids[np.asarray(members, dtype=np.int64)] = group_id

                for detail in group_details:
                    row = {
                        "scope": scope_name,
                        "variant": variant_name,
                        **detail,
                    }
                    if not (
                        scope_name == "all_units" and variant_name == FULL_VARIANT
                    ):
                        row["mixed_type"] = ""
                    details_writer.writerow(row)

                for local_index, record in enumerate(scope_records):
                    group_id = int(group_ids[local_index])
                    membership_writer.writerow(
                        {
                            "scope": scope_name,
                            "variant": variant_name,
                            "global_index": record.global_index,
                            "layer": record.layer,
                            "unit_type": record.unit_type,
                            "unit_index": record.unit_index,
                            "D_abs": record.D_abs,
                            "D_rel": record.D_rel,
                            "D_st": record.D_st,
                            "ablation_logit_deviation": record.ablation_logit_deviation,
                            "group_id": group_id,
                            "group_size": group_sizes[group_id],
                        }
                    )

                if scope_name == "all_units" and variant_name == FULL_VARIANT:
                    full_groups = groups
                    full_group_details = group_details
                    full_unit_types = scope_unit_types

    if full_groups is None or full_group_details is None or full_unit_types is None:
        raise RuntimeError("global Full3D groups were not generated")

    _write_csv(
        output_dir / "descriptor_variant_bms_summary.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    cross_type = cross_type_statistics(full_groups, full_unit_types)
    _write_csv(
        output_dir / "full3d_cross_type_summary.csv",
        list(cross_type),
        [cross_type],
    )
    size_rows = group_size_consistency_rows(full_group_details)
    _write_csv(
        output_dir / "full3d_group_size_consistency.csv",
        [
            "group_size_bucket",
            "number_of_groups",
            "number_of_units",
            "mean_intra_group_variance",
            "mean_pairwise_ablation_difference",
        ],
        size_rows,
    )

    all_rows = _all_units_rows(summary_rows)
    _plot_variant_metric(
        output_dir / "bms_variant_intra_variance.png",
        all_rows,
        "weighted_intra_ablation_variance",
        "Unit-weighted intra-domain ablation variance (lower is better)",
        lower_is_better=True,
    )
    _plot_variant_metric(
        output_dir / "bms_variant_pairwise_difference.png",
        all_rows,
        "pairwise_intra_ablation_difference",
        "Intra-domain pairwise ablation difference (lower is better)",
        lower_is_better=True,
    )
    _plot_variant_metric(
        output_dir / "bms_variant_domain_improvement.png",
        all_rows,
        "domain_improvement_ratio",
        "Domain-to-global improvement ratio (higher is better)",
        lower_is_better=False,
    )
    _plot_bms_vs_random(output_dir / "bms_vs_random_pairwise.png", all_rows)
    _plot_group_size_distribution(
        output_dir / "bms_group_size_distribution_full3d.png", full_groups
    )
    _plot_group_behavior(
        output_dir / "bms_group_behavior_consistency_full3d.png",
        full_group_details,
    )
    write_validation_summary(
        output_dir / "validation_summary.md", summary_rows, cross_type
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "input_sha256": _sha256_file(input_csv),
                "num_units": len(records),
                "attention_heads": attention_count,
                "ffn_neurons": mlp_count,
                "seed": args.seed,
                "random_repeats": args.random_repeats,
                "descriptor_variants": list(DESCRIPTOR_VARIANTS),
                "scopes": {name: int(indices.size) for name, indices in scopes.items()},
                "bms_runs": bms_metadata,
                "note": (
                    "Every scope and descriptor variant reruns the repository's "
                    "existing BMS independently on CPU. No model forward or Task "
                    "007 output modification occurs."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print_console_summary(summary_rows, len(records), cross_type)
    print(f"Output directory: {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate existing BMS competition domains using Task 007 ablation "
            "effects only; no model execution is performed."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="descriptor_ablation_validation/unit_ablation_effect.csv",
    )
    parser.add_argument("--output_dir", default="bms_competition_validation")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--random_repeats", type=int, default=100)
    args = parser.parse_args(argv)
    if args.random_repeats <= 0:
        parser.error("--random_repeats must be positive")
    return args


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()

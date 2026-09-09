#!/usr/bin/env python3
"""Offline analysis for full descriptor--ablation consistency validation.

The input contains one measured ablation effect for every pruning unit.  This
script never constructs an ``[N, N]`` distance matrix.  Pairwise Spearman
statistics use either all unique pairs for small inputs or a deterministic
sample of at least one million unique pairs for the full Video Swin model.
Nearest neighbours are found with chunked exact cKDTree queries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "descriptor_ablation_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - the repository runtime provides tqdm.
    def tqdm(iterable=None, **kwargs):
        del kwargs
        return iterable


DESCRIPTOR_COLUMNS = ("D_abs", "D_rel", "D_st")
DESCRIPTOR_VARIANTS = OrderedDict(
    [
        ("D_abs", (0,)),
        ("D_rel", (1,)),
        ("D_st", (2,)),
        ("D_abs+D_rel", (0, 1)),
        ("D_abs+D_st", (0, 2)),
        ("D_rel+D_st", (1, 2)),
        ("D_abs+D_rel+D_st", (0, 1, 2)),
    ]
)
DISPLAY_VARIANTS = (
    "D_abs",
    "D_rel",
    "D_st",
    "D_abs+D_rel",
    "D_abs+D_st",
    "D_rel+D_st",
    "Full 3D",
)
FULL_VARIANT = "D_abs+D_rel+D_st"


def _require_matrix(values: np.ndarray, columns: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {matrix.shape}")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(
            f"expected {columns} columns, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be non-empty and finite")
    return matrix


def standardize_descriptor(
    descriptor: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize ``V [N,D]`` independently along each descriptor axis."""
    values = _require_matrix(descriptor)
    mean = values.mean(axis=0)  # [D]
    std = values.std(axis=0)  # [D]
    safe_std = np.where(std > eps, std, 1.0)
    normalized = (values - mean[None, :]) / safe_std[None, :]  # [N,D]
    normalized[:, std <= eps] = 0.0
    return normalized, mean, std


def build_descriptor_variants(
    descriptor: np.ndarray,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, list[float]]]:
    """Return the seven standardized 1-D/2-D/3-D descriptor variants."""
    values = _require_matrix(descriptor, columns=3)
    normalized, mean, std = standardize_descriptor(values)
    variants = OrderedDict(
        (name, normalized[:, dimensions].copy())
        for name, dimensions in DESCRIPTOR_VARIANTS.items()
    )
    metadata = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "columns": list(DESCRIPTOR_COLUMNS),
    }
    return variants, metadata


def normalized_euclidean_distance(
    descriptor: np.ndarray, first: int, second: int
) -> float:
    """Euclidean distance after per-axis standardization of ``V [N,D]``."""
    normalized, _, _ = standardize_descriptor(descriptor)
    return float(np.linalg.norm(normalized[first] - normalized[second]))


def _all_unique_pairs(num_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialize two 1-D pair-index vectors, never an ``[N,N]`` matrix."""
    if num_units < 2:
        raise ValueError("at least two units are required")
    first, second = np.triu_indices(num_units, k=1)
    return first.astype(np.int64), second.astype(np.int64)


def _pair_rank_to_indices(
    ranks: np.ndarray, num_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map upper-triangle linear ranks to ``i < j`` without an ``[N,N]`` map."""
    ranks = np.asarray(ranks, dtype=np.int64)
    diagonal = float(2 * num_units - 1)
    first = np.floor(
        (diagonal - np.sqrt(diagonal * diagonal - 8.0 * ranks)) / 2.0
    ).astype(np.int64)

    start = first * (2 * num_units - first - 1) // 2
    too_large = start > ranks
    while np.any(too_large):
        first[too_large] -= 1
        start = first * (2 * num_units - first - 1) // 2
        too_large = start > ranks

    next_start = (first + 1) * (2 * num_units - first - 2) // 2
    too_small = ranks >= next_start
    while np.any(too_small):
        first[too_small] += 1
        start = first * (2 * num_units - first - 1) // 2
        next_start = (first + 1) * (2 * num_units - first - 2) // 2
        too_small = ranks >= next_start

    second = first + 1 + (ranks - start)
    if not np.all((first >= 0) & (first < second) & (second < num_units)):
        raise RuntimeError("pair-rank conversion produced invalid indices")
    return first, second


def select_pair_indices(
    num_units: int,
    sample_size: int = 1_000_000,
    seed: int = 3407,
    exact_pair_limit: int = 5_000_000,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Choose exact pairs when practical, otherwise deterministic unique pairs."""
    if num_units < 2:
        raise ValueError("at least two units are required")
    total_pairs = num_units * (num_units - 1) // 2
    if total_pairs <= exact_pair_limit:
        first, second = _all_unique_pairs(num_units)
        return first, second, "exact_all_pairs"

    if sample_size < 1_000_000:
        raise ValueError(
            "sampled pair analysis for a large input requires at least 1,000,000 pairs"
        )
    sample_size = min(int(sample_size), total_pairs)
    rng = np.random.default_rng(seed)
    ranks = rng.choice(total_pairs, size=sample_size, replace=False, shuffle=False)
    ranks.sort()
    first, second = _pair_rank_to_indices(ranks, num_units)
    return first, second, "sampled_unique_pairs"


def pairwise_descriptor_distances(
    features: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Compute distances for 1-D pair-index vectors in bounded chunks."""
    features = _require_matrix(features)
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("pair indices must be equal-length 1-D arrays")
    distances = np.empty(first.size, dtype=np.float64)  # [P]
    for start in range(0, first.size, chunk_size):
        end = min(start + chunk_size, first.size)
        difference = features[first[start:end]] - features[second[start:end]]
        distances[start:end] = np.linalg.norm(difference, axis=1)
    return distances


def spearman_distance_consistency(
    features: np.ndarray,
    ablation_effect: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    chunk_size: int = 200_000,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Spearman correlation between descriptor and ablation pair distances."""
    effect = np.asarray(ablation_effect, dtype=np.float64)
    if effect.ndim != 1 or not np.isfinite(effect).all():
        raise ValueError("ablation_effect must be a finite vector [N]")
    descriptor_distance = pairwise_descriptor_distances(
        features, first, second, chunk_size=chunk_size
    )  # [P]
    effect_distance = np.abs(effect[first] - effect[second])  # [P]
    statistic = spearmanr(descriptor_distance, effect_distance)
    rho = float(statistic.statistic)
    pvalue = float(statistic.pvalue)
    if math.isnan(rho):
        rho = 0.0
    if math.isnan(pvalue):
        pvalue = 1.0
    return rho, pvalue, descriptor_distance, effect_distance


def chunked_nearest_neighbors(
    features: np.ndarray, query_chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact non-self neighbours with chunked cKDTree queries.

    ``features`` is ``[N,D]``.  Returned indices and distances are ``[N]``.
    The tree and query batches are linear in ``N``; no ``[N,N]`` allocation is
    required.
    """
    features = _require_matrix(features)
    num_units = features.shape[0]
    if num_units < 2:
        raise ValueError("at least two units are required")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    tree = cKDTree(features)
    candidate_count = min(num_units, 8)
    neighbors = np.empty(num_units, dtype=np.int64)  # [N]
    distances = np.empty(num_units, dtype=np.float64)  # [N]

    progress = tqdm(
        range(0, num_units, query_chunk_size),
        desc="Nearest-neighbor analysis",
        leave=False,
    )
    for start in progress:
        end = min(start + query_chunk_size, num_units)
        block_distance, block_index = tree.query(
            features[start:end], k=candidate_count, workers=1
        )
        if candidate_count == 1:
            block_distance = block_distance[:, None]
            block_index = block_index[:, None]
        for local_index, global_index in enumerate(range(start, end)):
            valid = block_index[local_index] != global_index
            if not np.any(valid):
                raise RuntimeError(f"no non-self neighbor for unit {global_index}")
            valid_distance = block_distance[local_index][valid]
            valid_index = block_index[local_index][valid]
            best_distance = float(valid_distance.min())
            tied = valid_index[np.isclose(
                valid_distance, best_distance, rtol=1e-12, atol=1e-12
            )]
            neighbors[global_index] = int(tied.min())
            distances[global_index] = best_distance
    return neighbors, distances


def compute_dimension_spearman(descriptor: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix ``[3,3]`` for the raw descriptor axes."""
    values = _require_matrix(descriptor, columns=3)
    result = spearmanr(values, axis=0).statistic
    matrix = np.asarray(result, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise RuntimeError(f"unexpected dimension correlation shape {matrix.shape}")
    return np.nan_to_num(matrix, nan=0.0)


def _read_unit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "global_index", "layer", "unit_type", "unit_index",
        "D_abs", "D_rel", "D_st", "ablation_logit_deviation",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is empty or missing columns {sorted(required)}")
    rows.sort(key=lambda row: int(row["global_index"]))
    indices = [int(row["global_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("global_index must be unique and contiguous from zero")
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scope_indices(rows: list[dict[str, str]]) -> OrderedDict[str, np.ndarray]:
    scopes: OrderedDict[str, np.ndarray] = OrderedDict()
    all_indices = np.arange(len(rows), dtype=np.int64)
    scopes["all_units_global"] = all_indices
    scopes["attention_only"] = np.asarray(
        [i for i, row in enumerate(rows) if row["unit_type"] == "attention_head"],
        dtype=np.int64,
    )
    scopes["mlp_only"] = np.asarray(
        [i for i, row in enumerate(rows) if row["unit_type"] == "ffn_neuron"],
        dtype=np.int64,
    )
    for stage in range(4):
        stage_indices = np.asarray(
            [i for i, row in enumerate(rows) if row.get("stage", "") == str(stage)],
            dtype=np.int64,
        )
        if stage_indices.size >= 2:
            scopes[f"stage_{stage}"] = stage_indices
    return OrderedDict(
        (name, indices) for name, indices in scopes.items() if indices.size >= 2
    )


def _plot_variant_bars(
    path: Path,
    rows: list[dict],
    value_key: str,
    ylabel: str,
    lower_is_better: bool,
) -> None:
    global_rows = {
        row["variant"]: row for row in rows if row["scope"] == "all_units_global"
    }
    values = [float(global_rows[name][value_key]) for name in DESCRIPTOR_VARIANTS]
    labels = list(DISPLAY_VARIANTS)
    colors = ["#5B8FF9"] * 6 + ["#F6BD16"]
    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.bar(np.arange(len(values)), values, color=colors, edgecolor="#333333")
    axis.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(
        "Descriptor variant comparison"
        + (" (smaller is better)" if lower_is_better else "")
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_dimension_correlation(path: Path, matrix: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(3), DESCRIPTOR_COLUMNS)
    axis.set_yticks(range(3), DESCRIPTOR_COLUMNS)
    for row in range(3):
        for column in range(3):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center", color="black",
            )
    axis.set_title("Descriptor-dimension Spearman correlation")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_full_distance(
    path: Path, descriptor_distance: np.ndarray, effect_distance: np.ndarray
) -> None:
    max_points = 200_000
    if descriptor_distance.size > max_points:
        indices = np.linspace(
            0, descriptor_distance.size - 1, max_points, dtype=np.int64
        )
        descriptor_distance = descriptor_distance[indices]
        effect_distance = effect_distance[indices]
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    plot = axis.hexbin(
        descriptor_distance, effect_distance, gridsize=80,
        bins="log", mincnt=1, cmap="viridis",
    )
    axis.set_xlabel("Standardized full-3D descriptor distance")
    axis.set_ylabel("Absolute ablation-effect difference")
    axis.set_title("Descriptor distance vs. ablation-behavior distance")
    fig.colorbar(plot, ax=axis, label="log10(pair count)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_validation_summary(
    path: Path,
    rows: list[dict],
    num_units: int,
    attention_units: int,
    mlp_units: int,
    dimension_correlation: np.ndarray,
) -> None:
    global_rows = [row for row in rows if row["scope"] == "all_units_global"]
    best_spearman = max(global_rows, key=lambda row: float(row["spearman_rho"]))
    best_nn = min(global_rows, key=lambda row: float(row["nn_error"]))
    full = next(row for row in global_rows if row["variant"] == FULL_VARIANT)
    full_spearman_best = full["variant"] == best_spearman["variant"]
    full_nn_best = full["variant"] == best_nn["variant"]

    if full_spearman_best and full_nn_best:
        conclusion = (
            "在当前消融定义与探针样本下，完整三维描述符同时取得最高的距离"
            "Spearman一致性和最低的最近邻消融误差，结果支持其局部几何与真实"
            "单元消融行为一致的假设。"
        )
    elif full_spearman_best or full_nn_best:
        conclusion = (
            "完整三维描述符只在两项一致性指标中的一项取得最优，当前结果提供"
            "部分支持，但不足以断言三维表示在所有行为一致性指标上均最优。"
        )
    else:
        conclusion = (
            "完整三维描述符未在距离Spearman一致性或最近邻消融误差上取得最优，"
            "当前实验不支持其为最具行为一致性的表示。"
        )

    lines = [
        "# Descriptor--Ablation Consistency Validation",
        "",
        "## Evaluation coverage",
        "",
        f"- Units evaluated: {num_units}",
        f"- Attention heads: {attention_units}",
        f"- FFN neurons: {mlp_units}",
        "- Unit ablation coverage: full-unit (not sampled)",
        "",
        "## Global descriptor variants",
        "",
        "| Variant | Spearman rho | p-value | NNError | Pair mode | Pairs |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['variant']} | {float(row['spearman_rho']):.6f} | "
            f"{float(row['spearman_pvalue']):.3e} | "
            f"{float(row['nn_error']):.6e} | {row['pair_mode']} | "
            f"{int(row['num_pairs'])} |"
        )
    lines.extend(
        [
            "",
            "## Descriptor-dimension redundancy",
            "",
            "The observed Spearman matrix is:",
            "",
            "```text",
            np.array2string(dimension_correlation, precision=4),
            "```",
            "",
            "## Interpretation",
            "",
            f"Best Spearman variant: {best_spearman['variant']}.",
            f"Best NNError variant: {best_nn['variant']}.",
            "",
            conclusion,
            "",
            "本结论只检验描述符邻近关系与消融行为邻近关系，不声称三维描述符"
            "完整刻画剪枝单元的全部属性。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_console_summary(rows: list[dict], num_units: int, counts: dict[str, int]) -> None:
    global_rows = [row for row in rows if row["scope"] == "all_units_global"]
    best_spearman = max(global_rows, key=lambda row: float(row["spearman_rho"]))
    best_nn = min(global_rows, key=lambda row: float(row["nn_error"]))
    print("=" * 79)
    print("Descriptor--Ablation Consistency Validation")
    print("-" * 79)
    print(f"Units evaluated: {num_units}")
    print(f"Attention heads: {counts['attention_head']}")
    print(f"FFN neurons: {counts['ffn_neuron']}")
    print()
    print(f"{'Variant':32s} {'Spearman rho':>16s} {'NN Error':>18s}")
    print("-" * 79)
    for row in global_rows:
        print(
            f"{row['variant']:32s} {float(row['spearman_rho']):16.6f} "
            f"{float(row['nn_error']):18.6e}"
        )
    print("-" * 79)
    print(f"Best Spearman variant: {best_spearman['variant']}")
    print(f"Best NN variant: {best_nn['variant']}")
    print("=" * 79)


def analyze(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    input_csv = input_dir / "unit_ablation_effect.csv"
    rows = _read_unit_rows(input_csv)
    descriptor = np.asarray(
        [[float(row[column]) for column in DESCRIPTOR_COLUMNS] for row in rows],
        dtype=np.float64,
    )  # [N,3]
    ablation_effect = np.asarray(
        [float(row["ablation_logit_deviation"]) for row in rows],
        dtype=np.float64,
    )  # [N]
    if not np.isfinite(descriptor).all() or not np.isfinite(ablation_effect).all():
        raise ValueError("descriptor or ablation effect contains NaN/Inf")

    scopes = _scope_indices(rows)
    summary_rows: list[dict] = []
    nearest_neighbor_rows: list[dict] = []
    normalization_metadata: dict[str, dict] = {}
    full_descriptor_distance = None
    full_effect_distance = None

    for scope_name, scope_indices in tqdm(
        scopes.items(), desc="Analysis scopes", total=len(scopes)
    ):
        scope_descriptor = descriptor[scope_indices]  # [Ns,3]
        scope_effect = ablation_effect[scope_indices]  # [Ns]
        variants, norm_metadata = build_descriptor_variants(scope_descriptor)
        normalization_metadata[scope_name] = norm_metadata
        pair_first, pair_second, pair_mode = select_pair_indices(
            len(scope_indices),
            sample_size=args.pair_sample_size,
            seed=args.seed,
            exact_pair_limit=args.exact_pair_limit,
        )
        mean_pair_effect_difference = float(
            np.abs(scope_effect[pair_first] - scope_effect[pair_second]).mean()
        )

        variant_progress = tqdm(
            variants.items(),
            total=len(variants),
            desc=f"Pairwise descriptor analysis [{scope_name}]",
            leave=False,
        )
        for variant_name, features in variant_progress:
            rho, pvalue, descriptor_distance, effect_distance = (
                spearman_distance_consistency(
                    features,
                    scope_effect,
                    pair_first,
                    pair_second,
                    chunk_size=args.pair_chunk_size,
                )
            )
            neighbor_local, neighbor_distance = chunked_nearest_neighbors(
                features, query_chunk_size=args.nn_query_chunk_size
            )
            neighbor_difference = np.abs(
                scope_effect - scope_effect[neighbor_local]
            )  # [Ns]
            nn_error = float(neighbor_difference.mean())
            nn_error_norm = nn_error / (mean_pair_effect_difference + 1e-8)
            summary_rows.append(
                {
                    "scope": scope_name,
                    "variant": variant_name,
                    "dimensions": "|".join(
                        DESCRIPTOR_COLUMNS[index]
                        for index in DESCRIPTOR_VARIANTS[variant_name]
                    ),
                    "num_units": len(scope_indices),
                    "num_pairs": len(pair_first),
                    "pair_mode": pair_mode,
                    "spearman_rho": rho,
                    "spearman_pvalue": pvalue,
                    "nn_error": nn_error,
                    "nn_error_norm": nn_error_norm,
                }
            )

            if scope_name == "all_units_global":
                for local_index, global_row_index in enumerate(scope_indices):
                    neighbor_row_index = scope_indices[neighbor_local[local_index]]
                    nearest_neighbor_rows.append(
                        {
                            "scope": scope_name,
                            "variant": variant_name,
                            "global_index": int(rows[global_row_index]["global_index"]),
                            "neighbor_global_index": int(
                                rows[neighbor_row_index]["global_index"]
                            ),
                            "descriptor_distance": float(
                                neighbor_distance[local_index]
                            ),
                            "ablation_effect": float(scope_effect[local_index]),
                            "neighbor_ablation_effect": float(
                                scope_effect[neighbor_local[local_index]]
                            ),
                            "ablation_difference": float(
                                neighbor_difference[local_index]
                            ),
                        }
                    )
                if variant_name == FULL_VARIANT:
                    full_descriptor_distance = descriptor_distance
                    full_effect_distance = effect_distance

    summary_fields = [
        "scope", "variant", "dimensions", "num_units", "num_pairs",
        "pair_mode", "spearman_rho", "spearman_pvalue", "nn_error",
        "nn_error_norm",
    ]
    _write_csv(
        input_dir / "descriptor_variant_summary.csv", summary_fields, summary_rows
    )
    _write_csv(
        input_dir / "nearest_neighbor_validation.csv",
        [
            "scope", "variant", "global_index", "neighbor_global_index",
            "descriptor_distance", "ablation_effect",
            "neighbor_ablation_effect", "ablation_difference",
        ],
        nearest_neighbor_rows,
    )

    dimension_correlation = compute_dimension_spearman(descriptor)
    correlation_rows = []
    for row_index, row_name in enumerate(DESCRIPTOR_COLUMNS):
        correlation_rows.append(
            {
                "dimension": row_name,
                **{
                    column_name: float(dimension_correlation[row_index, column_index])
                    for column_index, column_name in enumerate(DESCRIPTOR_COLUMNS)
                },
            }
        )
    _write_csv(
        input_dir / "descriptor_dimension_correlation.csv",
        ["dimension", *DESCRIPTOR_COLUMNS],
        correlation_rows,
    )
    (input_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "pair_sample_size": args.pair_sample_size,
                "exact_pair_limit": args.exact_pair_limit,
                "normalization": normalization_metadata,
                "note": (
                    "Standardization is analysis-only; raw descriptor values in "
                    "unit_ablation_effect.csv are unchanged."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _plot_variant_bars(
        input_dir / "descriptor_variant_spearman.png",
        summary_rows,
        "spearman_rho",
        "Spearman rho",
        lower_is_better=False,
    )
    _plot_variant_bars(
        input_dir / "descriptor_variant_nnerror.png",
        summary_rows,
        "nn_error",
        "Nearest-neighbor ablation error",
        lower_is_better=True,
    )
    _plot_dimension_correlation(
        input_dir / "descriptor_dimension_correlation.png", dimension_correlation
    )
    if full_descriptor_distance is None or full_effect_distance is None:
        raise RuntimeError("full 3-D distance data was not computed")
    _plot_full_distance(
        input_dir / "descriptor_vs_ablation_distance_3d.png",
        full_descriptor_distance,
        full_effect_distance,
    )

    counts = {
        unit_type: sum(row["unit_type"] == unit_type for row in rows)
        for unit_type in ("attention_head", "ffn_neuron")
    }
    _write_validation_summary(
        input_dir / "validation_summary.md",
        summary_rows,
        len(rows),
        counts["attention_head"],
        counts["ffn_neuron"],
        dimension_correlation,
    )
    _print_console_summary(summary_rows, len(rows), counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze descriptor--ablation consistency without [N,N] matrices."
    )
    parser.add_argument(
        "--input_dir", default="descriptor_ablation_validation",
        help="directory containing unit_ablation_effect.csv",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--pair_sample_size", type=int, default=1_000_000,
        help="unique pair count used only when exact all-pair Spearman is impractical",
    )
    parser.add_argument(
        "--exact_pair_limit", type=int, default=5_000_000,
        help="maximum pair count retained for exact all-pair Spearman",
    )
    parser.add_argument("--pair_chunk_size", type=int, default=200_000)
    parser.add_argument("--nn_query_chunk_size", type=int, default=4096)
    args = parser.parse_args()
    if args.pair_chunk_size <= 0 or args.nn_query_chunk_size <= 0:
        parser.error("analysis chunk sizes must be positive")
    if args.exact_pair_limit <= 0:
        parser.error("exact_pair_limit must be positive")
    return args


if __name__ == "__main__":
    analyze(parse_args())

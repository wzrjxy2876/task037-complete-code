#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline probe for task-conditioned spatio-temporal contribution-pattern affinity.

Inputs
------
1. cstc_probe_arrays.npz
2. unit_cstc_metrics.csv
3. optional whole_unit_masking_validation.csv

The probe does not load a model and does not perform forward/backward inference.

Main questions
--------------
1. Is contribution-pattern affinity different from scalar descriptor affinity?
2. Do functional nearest neighbours have more similar contribution patterns than
   scalar nearest neighbours and random neighbours?
3. Does the functional affinity collapse to a narrow range?
4. Are the conclusions stable within each layer and unit type?
5. Can the existing single-unit masking file support a redundancy claim?
   (Usually no: pairwise joint masking is required.)

The functional affinity for units i and j is the average, over calibration videos,
of cosine similarity between their positive, per-video normalized T×H×W
contribution patterns.

Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline contribution-pattern affinity probe"
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--whole_masking", default="")
    parser.add_argument("--output_dir", default="./contribution_pattern_affinity")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--pair_sample", type=int, default=200000)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_full_matrices", action="store_true")
    parser.add_argument("--visualize_units", type=int, default=4)
    return parser.parse_args()


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    if np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    cols = ["stage", "block", "unit_type", "layer"]
    available = [c for c in cols if c in metrics.columns]
    table = metrics[available].drop_duplicates()
    sort_cols = [c for c in ("stage", "block", "unit_type", "layer") if c in table]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable")
    return table["layer"].tolist()


def normalize_patterns(volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[V,U,T,H,W] -> L2-normalized positive patterns [V,U,P], valid [V,U]."""
    positive = np.maximum(np.asarray(volumes, dtype=np.float32), 0.0)
    flat = positive.reshape(positive.shape[0], positive.shape[1], -1)
    l1 = flat.sum(axis=-1, keepdims=True)
    valid = l1[..., 0] > EPS
    probability = np.zeros_like(flat, dtype=np.float32)
    probability[valid] = flat[valid] / np.maximum(l1[valid], EPS)
    l2 = np.linalg.norm(probability, axis=-1, keepdims=True)
    normalized = np.zeros_like(probability, dtype=np.float32)
    nonzero = l2[..., 0] > EPS
    normalized[nonzero] = probability[nonzero] / np.maximum(l2[nonzero], EPS)
    return normalized, valid


def pack_patterns(patterns: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack [V,U,P] into [U,V*P]; pairwise dot is summed per-video cosine."""
    packed = patterns.transpose(1, 0, 2).reshape(patterns.shape[1], -1)
    unit_valid = valid.T.astype(np.float32, copy=False)  # [U,V]
    return np.ascontiguousarray(packed), np.ascontiguousarray(unit_valid)


def functional_affinity_block(
    packed: np.ndarray,
    unit_valid: np.ndarray,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """Exact mean per-video cosine using two matrix multiplications."""
    numerator = packed[row_start:row_end] @ packed.T
    denominator = unit_valid[row_start:row_end] @ unit_valid.T
    output = np.zeros_like(numerator, dtype=np.float32)
    mask = denominator > 0
    output[mask] = numerator[mask] / denominator[mask]
    return np.clip(output, 0.0, 1.0)


def standardize_scalar(metrics: pd.DataFrame) -> np.ndarray:
    values = metrics[["d_abs", "d_rel"]].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, EPS)).astype(np.float32)


def scalar_similarity_block(
    scalar: np.ndarray,
    row_start: int,
    row_end: int,
    scale: float,
) -> np.ndarray:
    a = scalar[row_start:row_end]
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(scalar * scalar, axis=1, keepdims=True).T
    d2 = np.maximum(aa + bb - 2.0 * (a @ scalar.T), 0.0)
    return np.exp(-d2 / (2.0 * scale * scale + EPS)).astype(np.float32)


def estimate_scalar_scale(
    scalar: np.ndarray, pair_sample: int, rng: np.random.RandomState
) -> float:
    units = len(scalar)
    count = min(pair_sample, max(units * (units - 1) // 2, 1))
    first = rng.randint(0, units, size=count)
    second = rng.randint(0, units, size=count)
    keep = first != second
    distances = np.linalg.norm(scalar[first[keep]] - scalar[second[keep]], axis=1)
    positive = distances[distances > EPS]
    return float(np.median(positive)) if len(positive) else 1.0


def sampled_pair_correlation(
    packed: np.ndarray,
    unit_valid: np.ndarray,
    scalar: np.ndarray,
    scalar_scale: float,
    pair_sample: int,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    units = scalar.shape[0]
    first = rng.randint(0, units, size=pair_sample)
    second = rng.randint(0, units, size=pair_sample)
    keep = first != second
    first, second = first[keep], second[keep]

    numerator = np.sum(packed[first] * packed[second], axis=1)
    denominator = np.sum(unit_valid[first] * unit_valid[second], axis=1)
    func = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator > 0
    func[mask] = numerator[mask] / denominator[mask]

    d2 = np.sum((scalar[first] - scalar[second]) ** 2, axis=1)
    scalar_sim = np.exp(-d2 / (2.0 * scalar_scale**2 + EPS))
    return {
        "rho_scalar_function": safe_spearman(scalar_sim, func),
        "functional_mean": float(np.mean(func)),
        "functional_std": float(np.std(func)),
        "functional_p05": float(np.quantile(func, 0.05)),
        "functional_p50": float(np.quantile(func, 0.50)),
        "functional_p95": float(np.quantile(func, 0.95)),
        "scalar_mean": float(np.mean(scalar_sim)),
        "sampled_pairs": int(len(func)),
    }


def nearest_neighbour_analysis(
    layer_name: str,
    layer_metrics: pd.DataFrame,
    packed: np.ndarray,
    unit_valid: np.ndarray,
    scalar: np.ndarray,
    scalar_scale: float,
    top_k: int,
    chunk_size: int,
    rng: np.random.RandomState,
    save_full: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    units = len(layer_metrics)
    rows: List[Dict[str, object]] = []
    func_matrix = np.zeros((units, units), dtype=np.float32) if save_full else None
    scalar_matrix = np.zeros((units, units), dtype=np.float32) if save_full else None

    for start in range(0, units, chunk_size):
        end = min(start + chunk_size, units)
        func = functional_affinity_block(packed, unit_valid, start, end)
        scalar_sim = scalar_similarity_block(scalar, start, end, scalar_scale)
        local_rows = np.arange(end - start)
        global_rows = np.arange(start, end)
        func[local_rows, global_rows] = -np.inf
        scalar_sim[local_rows, global_rows] = -np.inf

        if save_full:
            func_matrix[start:end] = func
            scalar_matrix[start:end] = scalar_sim

        for local, unit_index in enumerate(global_rows):
            functional_nn = np.argpartition(
                -func[local], kth=min(top_k, units - 1) - 1
            )[:top_k]
            functional_nn = functional_nn[np.argsort(-func[local, functional_nn])]
            scalar_nn = np.argpartition(
                -scalar_sim[local], kth=min(top_k, units - 1) - 1
            )[:top_k]
            scalar_nn = scalar_nn[np.argsort(-scalar_sim[local, scalar_nn])]

            forbidden = set(functional_nn.tolist() + scalar_nn.tolist() + [unit_index])
            candidates = np.array([j for j in range(units) if j not in forbidden])
            random_nn = (
                rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
                if len(candidates)
                else np.array([], dtype=int)
            )

            scalar_func_values = func[local, scalar_nn]
            random_func_values = (
                func[local, random_nn]
                if len(random_nn)
                else np.array([], dtype=np.float32)
            )

            rows.append(
                {
                    "layer": layer_name,
                    "unit_index": int(unit_index),
                    "functional_neighbors": ";".join(map(str, functional_nn.tolist())),
                    "functional_neighbor_affinity_mean": float(
                        np.mean(func[local, functional_nn])
                    ),
                    "scalar_neighbors": ";".join(map(str, scalar_nn.tolist())),
                    "scalar_neighbor_function_affinity_mean": float(
                        np.mean(scalar_func_values)
                    ),
                    "random_neighbors": ";".join(map(str, random_nn.tolist())),
                    "random_neighbor_function_affinity_mean": float(
                        np.mean(random_func_values)
                    ) if len(random_func_values) else float("nan"),
                    "neighbor_overlap_count": int(
                        len(set(functional_nn.tolist()) & set(scalar_nn.tolist()))
                    ),
                    "neighbor_overlap_ratio": float(
                        len(set(functional_nn.tolist()) & set(scalar_nn.tolist()))
                        / max(top_k, 1)
                    ),
                }
            )

    if save_full:
        safe = layer_name.replace(".", "_")
        np.save(output_dir / f"{safe}_functional_affinity.npy", func_matrix)
        np.save(output_dir / f"{safe}_scalar_affinity.npy", scalar_matrix)

    frame = pd.DataFrame(rows)
    summary = {
        "functional_nn_affinity_mean": float(
            frame["functional_neighbor_affinity_mean"].mean()
        ),
        "scalar_nn_function_affinity_mean": float(
            frame["scalar_neighbor_function_affinity_mean"].mean()
        ),
        "random_function_affinity_mean": float(
            frame["random_neighbor_function_affinity_mean"].mean()
        ),
        "topk_neighbor_overlap_mean": float(frame["neighbor_overlap_ratio"].mean()),
    }
    return frame, summary


def visualize_neighbours(
    layer_name: str,
    volumes: np.ndarray,
    nn_frame: pd.DataFrame,
    count: int,
    output_dir: Path,
) -> None:
    if count <= 0:
        return
    vis_dir = output_dir / "nearest_neighbor_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mean_volume = np.maximum(volumes, 0.0).mean(axis=0)  # [U,T,H,W]
    selected = np.linspace(0, len(mean_volume) - 1, min(count, len(mean_volume))).round().astype(int)
    safe = layer_name.replace(".", "_")

    for unit_index in selected:
        row = nn_frame[nn_frame["unit_index"] == unit_index].iloc[0]
        fnn = int(str(row["functional_neighbors"]).split(";")[0])
        snn = int(str(row["scalar_neighbors"]).split(";")[0])
        triplet = [("query", unit_index), ("functional_nn", fnn), ("scalar_nn", snn)]
        fig, axes = plt.subplots(3, mean_volume.shape[1], figsize=(1.5 * mean_volume.shape[1], 4.8))
        for r, (label, idx) in enumerate(triplet):
            volume = mean_volume[idx]
            vmax = max(float(volume.max()), EPS)
            for t in range(volume.shape[0]):
                axes[r, t].imshow(volume[t], vmin=0.0, vmax=vmax)
                axes[r, t].axis("off")
                if t == 0:
                    axes[r, t].set_ylabel(f"{label}\nu={idx}")
        fig.suptitle(f"{layer_name}: query and nearest neighbours")
        fig.tight_layout()
        fig.savefig(vis_dir / f"{safe}_query_{unit_index}.png", dpi=170)
        plt.close(fig)


def masking_diagnostics(
    whole_path: str,
    nn_all: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Dict[str, object]:
    if not whole_path or not Path(whole_path).exists():
        return {
            "available": False,
            "reason": "whole_unit_masking_validation.csv was not supplied",
        }
    whole = pd.read_csv(whole_path)
    if len(whole) == 0:
        return {"available": False, "reason": "whole masking file is empty"}

    merged = whole.merge(
        metrics[["layer", "unit_index", "d_abs", "d_rel"]],
        on=["layer", "unit_index"],
        how="left",
        suffixes=("", "_metrics"),
    )
    return {
        "available": True,
        "num_single_unit_tests": int(len(merged)),
        "rho_d_abs_true_drop": safe_spearman(
            merged["d_abs"], merged["true_mean_logit_drop"]
        ),
        "rho_d_rel_true_drop": safe_spearman(
            merged["d_rel"], merged["true_mean_logit_drop"]
        ),
        "redundancy_validation_supported": False,
        "reason": (
            "Single-unit deletion losses cannot validate pairwise redundancy. "
            "Joint deletion Δ_ij is required to evaluate R_ij = Δ_i + Δ_j - Δ_ij."
        ),
    }


def plot_layer_distributions(
    layer_name: str,
    nn_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    safe = layer_name.replace(".", "_")
    values = [
        nn_frame["functional_neighbor_affinity_mean"].to_numpy(),
        nn_frame["scalar_neighbor_function_affinity_mean"].to_numpy(),
        nn_frame["random_neighbor_function_affinity_mean"].to_numpy(),
    ]
    plt.figure(figsize=(6.4, 4.5))
    plt.boxplot(values, labels=["functional NN", "scalar NN", "random"])
    plt.ylabel("Functional pattern affinity")
    plt.title(layer_name)
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe}_neighbor_affinity_boxplot.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    arrays = np.load(args.npz, allow_pickle=True)
    metrics = pd.read_csv(args.unit_metrics)
    layers = ordered_layers(metrics)
    volume_keys = sorted(k for k in arrays.files if k.endswith("_contribution_volumes"))

    if len(layers) != len(volume_keys):
        raise ValueError(
            f"Layer count mismatch: metrics has {len(layers)}, NPZ has {len(volume_keys)} volumes"
        )

    all_nn = []
    layer_summaries: Dict[str, object] = {}

    for layer_index, (layer_name, volume_key) in enumerate(zip(layers, volume_keys)):
        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        volumes = arrays[volume_key]
        if volumes.shape[1] != len(layer_metrics):
            raise ValueError(
                f"{layer_name}: NPZ units={volumes.shape[1]}, metrics rows={len(layer_metrics)}"
            )

        patterns, valid = normalize_patterns(volumes)
        packed, unit_valid = pack_patterns(patterns, valid)
        scalar = standardize_scalar(layer_metrics)
        scalar_scale = estimate_scalar_scale(scalar, args.pair_sample, rng)
        pair_stats = sampled_pair_correlation(
            packed, unit_valid, scalar, scalar_scale, args.pair_sample, rng
        )
        nn_frame, nn_stats = nearest_neighbour_analysis(
            layer_name=layer_name,
            layer_metrics=layer_metrics,
            packed=packed,
            unit_valid=unit_valid,
            scalar=scalar,
            scalar_scale=scalar_scale,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            rng=rng,
            save_full=args.save_full_matrices,
            output_dir=output_dir,
        )
        all_nn.append(nn_frame)
        plot_layer_distributions(layer_name, nn_frame, output_dir)
        visualize_neighbours(
            layer_name, volumes, nn_frame, args.visualize_units, output_dir
        )

        layer_summaries[layer_name] = {
            "unit_type": str(layer_metrics["unit_type"].iloc[0]),
            "num_units": int(len(layer_metrics)),
            "num_videos": int(volumes.shape[0]),
            "pattern_shape": list(volumes.shape[2:]),
            "valid_pattern_rate": float(valid.mean()),
            "scalar_kernel_scale_data_adaptive": scalar_scale,
            **pair_stats,
            **nn_stats,
        }

    nn_all = pd.concat(all_nn, ignore_index=True)
    nn_all.to_csv(output_dir / "nearest_neighbor_analysis.csv", index=False)

    within_layer = pd.DataFrame(
        [{"layer": layer, **values} for layer, values in layer_summaries.items()]
    )
    within_layer.to_csv(output_dir / "within_layer_function_similarity.csv", index=False)

    masking = masking_diagnostics(args.whole_masking, nn_all, metrics)
    correlation_values = [
        values["rho_scalar_function"]
        for values in layer_summaries.values()
        if np.isfinite(values["rho_scalar_function"])
    ]
    summary = {
        "definition": (
            "Functional affinity is the mean per-video cosine similarity between "
            "positive, per-video normalized T×H×W contribution patterns."
        ),
        "num_layers": len(layer_summaries),
        "top_k": args.top_k,
        "layer_results": layer_summaries,
        "global": {
            "mean_layerwise_rho_scalar_function": float(np.mean(correlation_values))
            if correlation_values else float("nan"),
            "mean_functional_nn_affinity": float(
                nn_all["functional_neighbor_affinity_mean"].mean()
            ),
            "mean_scalar_nn_function_affinity": float(
                nn_all["scalar_neighbor_function_affinity_mean"].mean()
            ),
            "mean_random_function_affinity": float(
                nn_all["random_neighbor_function_affinity_mean"].mean()
            ),
            "mean_topk_neighbor_overlap": float(
                nn_all["neighbor_overlap_ratio"].mean()
            ),
        },
        "masking_diagnostics": masking,
        "limitations": [
            "The current probe validates pattern similarity, not pairwise pruning redundancy.",
            "Joint masking for unit pairs is required before using functional affinity in BMS.",
            "Only four layers and nine videos are available in the current NPZ.",
            "Cross-layer functional comparison is not evaluated because layer semantics and grids differ.",
        ],
        "run_config": vars(args),
    }

    with open(output_dir / "affinity_correlation.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report = [
        "Contribution Pattern Affinity Probe",
        "=" * 84,
        "",
        "This is an offline analysis. No model inference or pruning is performed.",
        "",
        "[Global]",
    ]
    for key, value in summary["global"].items():
        report.append(f"{key}: {value}")
    report += ["", "[Per layer]"]
    for layer, values in layer_summaries.items():
        report.append(f"\n{layer}")
        for key, value in values.items():
            report.append(f"  {key}: {value}")
    report += [
        "",
        "[Interpretation rules]",
        "1. Low scalar-function correlation means the pattern affinity adds information.",
        "2. Functional NN affinity should exceed scalar-NN and random affinity.",
        "3. Very high random affinity or very small affinity standard deviation indicates collapse.",
        "4. Low top-k overlap means scalar and functional neighbourhoods are structurally different.",
        "5. These results alone do not prove pruning redundancy.",
        "6. Before modifying BMS, validate high- and low-affinity pairs with joint masking.",
        "",
        "[Masking diagnostic]",
        json.dumps(masking, ensure_ascii=False, indent=2),
    ]
    (output_dir / "PATTERN_AFFINITY_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Offline pattern-affinity probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

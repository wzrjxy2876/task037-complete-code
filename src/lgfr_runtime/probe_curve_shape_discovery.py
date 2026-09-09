#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover temporal contribution-curve shapes from CTFRS NPZ outputs."
    )
    parser.add_argument("--npz", required=True, help="Path to ctfrs_curve_arrays.npz")
    parser.add_argument("--output_dir", default="./curve_shape_discovery")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--repeat_seeds", type=int, default=10)
    return parser.parse_args()


def load_unit_prototypes(npz_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    result: Dict[str, Dict[str, np.ndarray]] = {
        "head": {"curves": [], "layer_ids": [], "unit_ids": []},
        "neuron": {"curves": [], "layer_ids": [], "unit_ids": []},
    }

    role_keys = sorted(
        key for key in data.files if key.endswith("_role_distribution")
    )
    if not role_keys:
        raise ValueError("No '*_role_distribution' arrays were found in the NPZ.")

    for key in role_keys:
        layer_id = int(key.split("_")[1])
        curves = np.asarray(data[key], dtype=np.float64)
        if curves.ndim != 3:
            raise ValueError(f"{key} must have shape [videos, units, time], got {curves.shape}")

        prototype = curves.mean(axis=0)
        unit_type = "head" if prototype.shape[0] <= 128 else "neuron"

        result[unit_type]["curves"].append(prototype)
        result[unit_type]["layer_ids"].append(
            np.full(prototype.shape[0], layer_id, dtype=np.int64)
        )
        result[unit_type]["unit_ids"].append(
            np.arange(prototype.shape[0], dtype=np.int64)
        )

    for unit_type in result:
        if result[unit_type]["curves"]:
            result[unit_type] = {
                name: np.concatenate(values, axis=0)
                for name, values in result[unit_type].items()
            }
        else:
            result[unit_type] = {
                "curves": np.empty((0, 0)),
                "layer_ids": np.empty((0,), dtype=np.int64),
                "unit_ids": np.empty((0,), dtype=np.int64),
            }
    return result


def normalize_curves(curves: np.ndarray) -> np.ndarray:
    curves = np.clip(np.asarray(curves, dtype=np.float64), 0.0, None)
    sums = curves.sum(axis=1, keepdims=True)
    valid = sums[:, 0] > 1e-12
    normalized = np.full_like(curves, 1.0 / curves.shape[1])
    normalized[valid] = curves[valid] / sums[valid]
    return normalized


def extract_curve_features(curves: np.ndarray) -> pd.DataFrame:
    curves = normalize_curves(curves)
    time_bins = curves.shape[1]
    time_axis = np.linspace(0.0, 1.0, time_bins)
    rows: List[Dict[str, float]] = []

    for curve in curves:
        entropy = -float(np.sum(curve * np.log(curve + 1e-12))) / np.log(time_bins)
        concentration = 1.0 - entropy
        centroid = float(np.sum(curve * time_axis))
        first_cut = time_bins // 3
        second_cut = 2 * time_bins // 3
        early_mass = float(curve[:first_cut].sum())
        middle_mass = float(curve[first_cut:second_cut].sum())
        late_mass = float(curve[second_cut:].sum())
        slope = float(np.polyfit(time_axis, curve, deg=1)[0])
        prominence = max(1e-6, 0.05 * float(curve.max() - curve.min()))
        peaks, _ = find_peaks(curve, prominence=prominence)
        ordered = np.sort(curve)[::-1]
        second_peak_ratio = float(ordered[1] / (ordered[0] + 1e-12))
        total_variation = float(np.abs(np.diff(curve)).sum())

        rows.append(
            {
                "entropy": entropy,
                "concentration": concentration,
                "centroid": centroid,
                "early_mass": early_mass,
                "middle_mass": middle_mass,
                "late_mass": late_mass,
                "slope": slope,
                "peak_count": int(len(peaks)),
                "second_peak_ratio": second_peak_ratio,
                "total_variation": total_variation,
                "max_bin": float(curve.max()),
                "min_bin": float(curve.min()),
                "peak_bin": int(np.argmax(curve)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_k(curves: np.ndarray, k_min: int, k_max: int, seed: int) -> pd.DataFrame:
    rows = []
    upper = min(k_max, len(curves) - 1)
    for k in range(k_min, upper + 1):
        model = KMeans(n_clusters=k, n_init=30, random_state=seed)
        labels = model.fit_predict(curves)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(curves, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(curves, labels)),
                "davies_bouldin": float(davies_bouldin_score(curves, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows)


def repeated_seed_stability(
    curves: np.ndarray, k: int, repeats: int
) -> Dict[str, float]:
    labels_list = [
        KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(curves)
        for seed in range(repeats)
    ]
    aris = [
        adjusted_rand_score(first, second)
        for first, second in itertools.combinations(labels_list, 2)
    ]
    return {
        "mean_pairwise_ari": float(np.mean(aris)),
        "min_pairwise_ari": float(np.min(aris)),
        "max_pairwise_ari": float(np.max(aris)),
    }


def infer_cluster_name(row: pd.Series) -> str:
    if row["concentration"] < 0.02:
        return "near-uniform persistent"
    if row["second_peak_ratio"] >= 0.85 and row["peak_count"] >= 2:
        return "multi-peak distributed"
    if row["early_mass"] > row["late_mass"] + 0.05:
        return "early-biased"
    if row["late_mass"] > row["early_mass"] + 0.05:
        return "late-biased"
    if row["total_variation"] >= 0.30:
        return "high-variation multi-stage"
    return "broad middle-distributed"


def plot_k_scores(scores: pd.DataFrame, unit_type: str, output_dir: Path) -> None:
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(scores["k"], scores["silhouette"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title(f"{unit_type.capitalize()} curve-shape model selection")
    plt.tight_layout()
    plt.savefig(output_dir / f"{unit_type}_silhouette_by_k.png", dpi=220)
    plt.close()


def plot_cluster_centroids(
    centroids: np.ndarray,
    cluster_names: Dict[int, str],
    unit_type: str,
    output_dir: Path,
) -> None:
    time_axis = np.linspace(0.0, 1.0, centroids.shape[1])
    plt.figure(figsize=(7.2, 4.8))
    for cluster_id, centroid in enumerate(centroids):
        plt.plot(
            time_axis,
            centroid,
            marker="o",
            label=f"C{cluster_id}: {cluster_names[cluster_id]}",
        )
    plt.xlabel("Normalized action progress")
    plt.ylabel("Mean contribution probability")
    plt.title(f"{unit_type.capitalize()} data-driven curve-shape centroids")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{unit_type}_cluster_centroids.png", dpi=220)
    plt.close()


def run_group(
    unit_type: str,
    curves: np.ndarray,
    layer_ids: np.ndarray,
    unit_ids: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    curves = normalize_curves(curves)
    scores = evaluate_k(curves, args.k_min, args.k_max, args.seed)
    best_row = scores.loc[scores["silhouette"].idxmax()]
    best_k = int(best_row["k"])

    model = KMeans(n_clusters=best_k, n_init=50, random_state=args.seed)
    labels = model.fit_predict(curves)
    features = extract_curve_features(curves)
    features.insert(0, "unit_id", unit_ids)
    features.insert(0, "layer_id", layer_ids)
    features.insert(0, "unit_type", unit_type)
    features["cluster_id"] = labels

    cluster_stats = (
        features.groupby("cluster_id")
        .agg(
            size=("cluster_id", "size"),
            entropy=("entropy", "mean"),
            concentration=("concentration", "mean"),
            centroid=("centroid", "mean"),
            early_mass=("early_mass", "mean"),
            middle_mass=("middle_mass", "mean"),
            late_mass=("late_mass", "mean"),
            slope=("slope", "mean"),
            peak_count=("peak_count", "mean"),
            second_peak_ratio=("second_peak_ratio", "mean"),
            total_variation=("total_variation", "mean"),
        )
        .reset_index()
    )
    cluster_stats["inferred_shape_name"] = cluster_stats.apply(
        infer_cluster_name, axis=1
    )
    cluster_names = dict(
        zip(cluster_stats["cluster_id"], cluster_stats["inferred_shape_name"])
    )
    features["inferred_shape_name"] = features["cluster_id"].map(cluster_names)

    stability = repeated_seed_stability(curves, best_k, args.repeat_seeds)

    scores.to_csv(output_dir / f"{unit_type}_k_selection.csv", index=False)
    features.to_csv(output_dir / f"{unit_type}_unit_shape_assignments.csv", index=False)
    cluster_stats.to_csv(output_dir / f"{unit_type}_cluster_statistics.csv", index=False)
    np.save(output_dir / f"{unit_type}_cluster_centroids.npy", model.cluster_centers_)

    plot_k_scores(scores, unit_type, output_dir)
    plot_cluster_centroids(model.cluster_centers_, cluster_names, unit_type, output_dir)

    return {
        "num_units": int(len(curves)),
        "num_time_bins": int(curves.shape[1]),
        "selected_k": best_k,
        "best_silhouette": float(best_row["silhouette"]),
        "calinski_harabasz": float(best_row["calinski_harabasz"]),
        "davies_bouldin": float(best_row["davies_bouldin"]),
        **stability,
        "cluster_sizes": {
            str(int(row.cluster_id)): int(row.size)
            for row in cluster_stats.itertuples()
        },
        "cluster_names": {
            str(int(row.cluster_id)): str(row.inferred_shape_name)
            for row in cluster_stats.itertuples()
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_unit_prototypes(Path(args.npz))
    summary: Dict[str, object] = {
        "input_npz": str(Path(args.npz).resolve()),
        "method": (
            "Per-unit prototype curves are averaged over videos, normalized on the "
            "time axis, and clustered separately for attention heads and FFN neurons. "
            "k is selected by maximum silhouette score. Repeated-seed ARI measures "
            "whether the discovered partition is reproducible."
        ),
        "groups": {},
    }

    for unit_type, payload in groups.items():
        curves = payload["curves"]
        if len(curves) < 3:
            continue
        summary["groups"][unit_type] = run_group(
            unit_type=unit_type,
            curves=curves,
            layer_ids=payload["layer_ids"],
            unit_ids=payload["unit_ids"],
            args=args,
            output_dir=output_dir,
        )

    with open(output_dir / "curve_shape_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_lines = [
        "Curve Shape Discovery Report",
        "=" * 80,
        "",
        "This experiment tests whether the saved CTFRS contribution curves form",
        "reproducible data-driven temporal shape groups without using early/late labels.",
        "",
    ]
    for unit_type, result in summary["groups"].items():
        report_lines.extend(
            [
                f"[{unit_type}]",
                f"Units: {result['num_units']}",
                f"Selected k: {result['selected_k']}",
                f"Silhouette: {result['best_silhouette']:.6f}",
                f"Repeated-seed mean ARI: {result['mean_pairwise_ari']:.6f}",
                f"Repeated-seed minimum ARI: {result['min_pairwise_ari']:.6f}",
                f"Cluster sizes: {result['cluster_sizes']}",
                f"Data-derived names: {result['cluster_names']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "Interpretation rule:",
            "- Silhouette < 0.20: weak geometric separation; do not claim distinct states.",
            "- Mean ARI near 1: partition is reproducible under initialization changes.",
            "- A stable but low-silhouette partition means a dominant continuum can be",
            "  split consistently, but it is not evidence of discrete semantic states.",
        ]
    )
    (output_dir / "CURVE_SHAPE_DISCOVERY_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

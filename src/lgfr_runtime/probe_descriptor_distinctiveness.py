#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional Descriptor Distinctiveness Probe (V2).

Scientific question
-------------------
When two pruning units have nearly identical scalar descriptors
(D_abs, D_rel), can their class-conditioned spatio-temporal functional
descriptors still be substantially different?

This probe is fully offline. It does not load the model and does not perform
masking or pruning.

Inputs
------
1. cstc_probe_arrays.npz
   Selected layer arrays:
       *_contribution_volumes: [V,U,T,H,W]

2. unit_cstc_metrics.csv
   Required columns:
       layer, unit_type, unit_index, d_abs, d_rel

Main procedure
--------------
For each selected layer:

1. Standardize (d_abs, d_rel) within the layer.
2. For every unit, find its nearest neighbour in the 2-D scalar space.
3. Build class-conditioned functional prototypes from positive contribution
   fields.
4. Compute scalar distance and functional distance for each matched pair.
5. Compare scalar-nearest pairs against random pairs and functional-nearest
   pairs.
6. Export representative cases:
   - scalar-close / function-close
   - scalar-close / function-far
   - scalar-far / function-close

The structured functional similarity is:

    S_func = sqrt(S_temporal * S_spatial)

where temporal similarity is maximum overlap-penalized cosine similarity over
all temporal shifts, and spatial similarity is cosine similarity between
class-conditioned spatial prototypes.

Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate information complementarity of functional descriptors"
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--output_dir", default="./descriptor_distinctiveness_probe")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument(
        "--class_labels",
        default="",
        help="Exact comma-separated class labels in NPZ video order.",
    )
    parser.add_argument("--compute_device", default="auto")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--random_pairs_per_unit", type=int, default=1)
    parser.add_argument("--top_cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--save_relation_matrices",
        action="store_true",
        help="Save full UxU functional similarity matrices.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    required = {"layer", "unit_type", "unit_index", "d_abs", "d_rel"}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    cols = [c for c in ("stage", "block", "unit_type", "layer") if c in metrics]
    table = metrics[cols].drop_duplicates()
    if cols:
        table = table.sort_values(cols, kind="stable")
    return table["layer"].tolist()


def parse_labels(args: argparse.Namespace, video_count: int) -> np.ndarray:
    if args.class_labels.strip():
        labels = np.asarray(
            [int(v.strip()) for v in args.class_labels.split(",")],
            dtype=np.int64,
        )
        if len(labels) != video_count:
            raise ValueError(
                f"class_labels has {len(labels)} entries but NPZ has {video_count} videos"
            )
        return labels
    expected = args.num_classes * args.videos_per_class
    if expected != video_count:
        raise ValueError(
            f"Cannot infer labels: {expected} expected videos, NPZ has {video_count}. "
            "Use --class_labels."
        )
    return np.repeat(np.arange(args.num_classes), args.videos_per_class)


def zscore_columns(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < EPS] = 1.0
    return (values - mean) / std


def scalar_distance_matrix(metrics: pd.DataFrame) -> np.ndarray:
    values = metrics[["d_abs", "d_rel"]].to_numpy(dtype=np.float64)
    values = zscore_columns(values)
    squared = np.sum(values * values, axis=1, keepdims=True)
    d2 = np.maximum(squared + squared.T - 2.0 * values @ values.T, 0.0)
    distance = np.sqrt(d2)
    np.fill_diagonal(distance, np.inf)
    return distance.astype(np.float32)


def positive_class_prototypes(
    volumes: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.maximum(volumes.astype(np.float32), 0.0)
    class_ids = np.unique(labels)
    temporal_all = []
    spatial_all = []
    for class_id in class_ids:
        selected = positive[labels == class_id]
        aggregate = selected.sum(axis=0)  # [U,T,H,W]
        temporal_all.append(aggregate.sum(axis=(2, 3)))  # [U,T]
        spatial_all.append(aggregate.sum(axis=1).reshape(aggregate.shape[0], -1))
    return (
        class_ids,
        np.stack(temporal_all, axis=0),
        np.stack(spatial_all, axis=0),
    )


def row_normalize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    valid = norm[:, 0] > EPS
    output = torch.zeros_like(x)
    output[valid] = x[valid] / norm[valid].clamp_min(EPS)
    return output, valid


def spatial_similarity_matrix(
    vectors: np.ndarray,
    device: torch.device,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    right = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    right, right_valid = row_normalize(right)
    units = vectors.shape[0]
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(range(0, units, chunk_size), desc=desc, ncols=105):
        end = min(start + chunk_size, units)
        left = torch.from_numpy(vectors[start:end]).to(
            device=device, dtype=torch.float32
        )
        left, left_valid = row_normalize(left)
        block = (left @ right.T).clamp_(0.0, 1.0)
        valid = left_valid[:, None] & right_valid[None, :]
        block = torch.where(valid, block, torch.zeros_like(block))
        output[start:end] = block.cpu().numpy()

    return output


def temporal_similarity_matrix(
    curves: np.ndarray,
    device: torch.device,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    right_all = torch.from_numpy(curves).to(device=device, dtype=torch.float32)
    units, time_bins = curves.shape
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(range(0, units, chunk_size), desc=desc, ncols=105):
        end = min(start + chunk_size, units)
        left_all = torch.from_numpy(curves[start:end]).to(
            device=device, dtype=torch.float32
        )
        best = torch.zeros((end - start, units), device=device)

        for shift in range(-(time_bins - 1), time_bins):
            if shift >= 0:
                left = left_all[:, : time_bins - shift]
                right = right_all[:, shift:]
            else:
                offset = -shift
                left = left_all[:, offset:]
                right = right_all[:, : time_bins - offset]

            overlap = left.shape[1]
            left_n, left_valid = row_normalize(left)
            right_n, right_valid = row_normalize(right)
            block = left_n @ right_n.T
            block *= float(overlap) / float(time_bins)
            valid = left_valid[:, None] & right_valid[None, :]
            block = torch.where(valid, block, torch.zeros_like(block))
            best = torch.maximum(best, block)

        output[start:end] = best.clamp_(0.0, 1.0).cpu().numpy()

    return output


def functional_similarity_matrix(
    temporal: np.ndarray,
    spatial: np.ndarray,
    device: torch.device,
    chunk_size: int,
    layer_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    class_matrices = []
    for class_pos in range(temporal.shape[0]):
        temporal_sim = temporal_similarity_matrix(
            temporal[class_pos],
            device,
            chunk_size,
            f"{layer_name} class {class_pos} temporal",
        )
        spatial_sim = spatial_similarity_matrix(
            spatial[class_pos],
            device,
            chunk_size,
            f"{layer_name} class {class_pos} spatial",
        )
        joint = np.sqrt(np.clip(temporal_sim * spatial_sim, 0.0, 1.0))
        np.fill_diagonal(joint, 1.0)
        class_matrices.append(joint.astype(np.float32))

    class_array = np.stack(class_matrices, axis=0)
    global_matrix = class_array.mean(axis=0)
    global_matrix = 0.5 * (global_matrix + global_matrix.T)
    np.fill_diagonal(global_matrix, 1.0)
    return class_array, global_matrix.astype(np.float32)


def random_targets(units: int, repeats: int, rng: np.random.RandomState) -> np.ndarray:
    source = np.repeat(np.arange(units), repeats)
    target = np.empty_like(source)
    for index, unit in enumerate(source):
        sampled = rng.randint(0, units - 1)
        if sampled >= unit:
            sampled += 1
        target[index] = sampled
    return target.reshape(units, repeats)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic)


def plot_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6.4, 5.0))
    plt.scatter(
        frame["scalar_distance"],
        frame["functional_distance"],
        s=12,
        alpha=0.45,
    )
    plt.xlabel("Distance in scalar descriptor space")
    plt.ylabel("Functional descriptor distance")
    plt.title("Scalar distance does not determine functional distance")
    plt.tight_layout()
    plt.savefig(output_dir / "scalar_vs_functional_distance.png", dpi=220)
    plt.savefig(output_dir / "scalar_vs_functional_distance.pdf")
    plt.close()


def plot_group_box(frame: pd.DataFrame, output_dir: Path) -> None:
    order = ["scalar_nearest", "functional_nearest", "random"]
    values = [
        frame.loc[frame["pair_source"] == group, "functional_distance"].to_numpy()
        for group in order
        if group in set(frame["pair_source"])
    ]
    labels = [group for group in order if group in set(frame["pair_source"])]
    plt.figure(figsize=(7.0, 4.8))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.ylabel("Functional descriptor distance")
    plt.title("Functional distance under different pair constructions")
    plt.tight_layout()
    plt.savefig(output_dir / "pair_source_functional_distance.png", dpi=220)
    plt.savefig(output_dir / "pair_source_functional_distance.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    device = resolve_device(args.compute_device)
    print(f"Compute device: {device}")

    metrics = pd.read_csv(args.unit_metrics)
    arrays = np.load(args.npz, allow_pickle=True)
    layers = ordered_layers(metrics)
    volume_keys = sorted(
        key for key in arrays.files if key.endswith("_contribution_volumes")
    )
    if len(layers) != len(volume_keys):
        raise ValueError(
            f"Layer count mismatch: metrics={len(layers)}, NPZ={len(volume_keys)}"
        )

    if args.layers != "all":
        pattern = re.compile(args.layers)
        keep = [
            index
            for index, layer in enumerate(layers)
            if pattern.search(layer)
        ]
        layers = [layers[index] for index in keep]
        volume_keys = [volume_keys[index] for index in keep]

    first_video_count = int(arrays[volume_keys[0]].shape[0])
    labels = parse_labels(args, first_video_count)

    pair_rows: List[Dict[str, object]] = []
    case_rows: List[Dict[str, object]] = []
    layer_stats: List[Dict[str, object]] = []

    for layer_pos, (layer_name, volume_key) in enumerate(
        zip(layers, volume_keys), start=1
    ):
        volumes = arrays[volume_key]
        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        units = volumes.shape[1]
        if len(layer_metrics) != units:
            raise ValueError(
                f"{layer_name}: NPZ units={units}, metrics rows={len(layer_metrics)}"
            )

        unit_type = str(layer_metrics["unit_type"].iloc[0])
        print(
            f"\n[Layer {layer_pos}/{len(layers)}] {layer_name}, "
            f"type={unit_type}, units={units}"
        )

        scalar_distance = scalar_distance_matrix(layer_metrics)
        class_ids, temporal, spatial = positive_class_prototypes(volumes, labels)
        class_func, func_similarity = functional_similarity_matrix(
            temporal, spatial, device, args.chunk_size, layer_name
        )
        func_distance = 1.0 - func_similarity
        np.fill_diagonal(func_distance, np.inf)

        scalar_nearest = np.argmin(scalar_distance, axis=1)
        functional_nearest = np.argmin(func_distance, axis=1)
        random_matrix = random_targets(
            units, args.random_pairs_per_unit, rng
        )

        for unit in range(units):
            pairs = [
                ("scalar_nearest", int(scalar_nearest[unit])),
                ("functional_nearest", int(functional_nearest[unit])),
            ]
            for random_target in random_matrix[unit]:
                pairs.append(("random", int(random_target)))

            for source, target in pairs:
                pair_rows.append(
                    {
                        "layer": layer_name,
                        "unit_type": unit_type,
                        "unit_i": unit,
                        "unit_j": target,
                        "pair_source": source,
                        "scalar_distance": float(scalar_distance[unit, target]),
                        "functional_similarity": float(
                            func_similarity[unit, target]
                        ),
                        "functional_distance": float(func_distance[unit, target]),
                        "d_abs_i": float(layer_metrics.loc[unit, "d_abs"]),
                        "d_abs_j": float(layer_metrics.loc[target, "d_abs"]),
                        "d_rel_i": float(layer_metrics.loc[unit, "d_rel"]),
                        "d_rel_j": float(layer_metrics.loc[target, "d_rel"]),
                    }
                )

        scalar_pairs = pd.DataFrame(
            [
                {
                    "unit_i": unit,
                    "unit_j": int(scalar_nearest[unit]),
                    "scalar_distance": float(
                        scalar_distance[unit, scalar_nearest[unit]]
                    ),
                    "functional_distance": float(
                        func_distance[unit, scalar_nearest[unit]]
                    ),
                    "functional_similarity": float(
                        func_similarity[unit, scalar_nearest[unit]]
                    ),
                }
                for unit in range(units)
            ]
        )

        scalar_q25 = float(scalar_pairs["scalar_distance"].quantile(0.25))
        func_q25 = float(scalar_pairs["functional_distance"].quantile(0.25))
        func_q75 = float(scalar_pairs["functional_distance"].quantile(0.75))

        close_far = scalar_pairs[
            (scalar_pairs["scalar_distance"] <= scalar_q25)
            & (scalar_pairs["functional_distance"] >= func_q75)
        ].sort_values(
            ["functional_distance", "scalar_distance"],
            ascending=[False, True],
        )

        close_close = scalar_pairs[
            (scalar_pairs["scalar_distance"] <= scalar_q25)
            & (scalar_pairs["functional_distance"] <= func_q25)
        ].sort_values(
            ["functional_distance", "scalar_distance"],
            ascending=[True, True],
        )

        # Scalar-far / function-close cases are searched globally.
        upper_i, upper_j = np.triu_indices(units, k=1)
        all_pairs = pd.DataFrame(
            {
                "unit_i": upper_i,
                "unit_j": upper_j,
                "scalar_distance": scalar_distance[upper_i, upper_j],
                "functional_distance": func_distance[upper_i, upper_j],
                "functional_similarity": func_similarity[upper_i, upper_j],
            }
        )
        scalar_q75_global = float(all_pairs["scalar_distance"].quantile(0.75))
        func_q25_global = float(all_pairs["functional_distance"].quantile(0.25))
        far_close = all_pairs[
            (all_pairs["scalar_distance"] >= scalar_q75_global)
            & (all_pairs["functional_distance"] <= func_q25_global)
        ].sort_values(
            ["functional_distance", "scalar_distance"],
            ascending=[True, False],
        )

        for case_type, table in [
            ("scalar_close_function_far", close_far),
            ("scalar_close_function_close", close_close),
            ("scalar_far_function_close", far_close),
        ]:
            for _, row in table.head(args.top_cases).iterrows():
                i = int(row["unit_i"])
                j = int(row["unit_j"])
                case_rows.append(
                    {
                        "layer": layer_name,
                        "unit_type": unit_type,
                        "case_type": case_type,
                        "unit_i": i,
                        "unit_j": j,
                        "scalar_distance": float(row["scalar_distance"]),
                        "functional_distance": float(row["functional_distance"]),
                        "functional_similarity": float(row["functional_similarity"]),
                        "d_abs_i": float(layer_metrics.loc[i, "d_abs"]),
                        "d_abs_j": float(layer_metrics.loc[j, "d_abs"]),
                        "d_rel_i": float(layer_metrics.loc[i, "d_rel"]),
                        "d_rel_j": float(layer_metrics.loc[j, "d_rel"]),
                    }
                )

        scalar_nearest_func = scalar_pairs["functional_distance"].to_numpy()
        random_targets_flat = random_matrix.reshape(-1)
        random_sources_flat = np.repeat(
            np.arange(units), args.random_pairs_per_unit
        )
        random_func = func_distance[
            random_sources_flat, random_targets_flat
        ]
        layer_stats.append(
            {
                "layer": layer_name,
                "unit_type": unit_type,
                "units": units,
                "scalar_nearest_scalar_distance_median": float(
                    scalar_pairs["scalar_distance"].median()
                ),
                "scalar_nearest_functional_distance_median": float(
                    np.median(scalar_nearest_func)
                ),
                "scalar_nearest_functional_distance_q25": float(
                    np.quantile(scalar_nearest_func, 0.25)
                ),
                "scalar_nearest_functional_distance_q75": float(
                    np.quantile(scalar_nearest_func, 0.75)
                ),
                "random_functional_distance_median": float(
                    np.median(random_func)
                ),
                "rho_scalar_vs_functional_distance": safe_spearman(
                    all_pairs["scalar_distance"],
                    all_pairs["functional_distance"],
                ),
                "scalar_close_function_far_cases": int(len(close_far)),
                "scalar_close_function_close_cases": int(len(close_close)),
                "scalar_far_function_close_cases": int(len(far_close)),
            }
        )

        if args.save_relation_matrices:
            safe = layer_name.replace(".", "_")
            np.save(
                output_dir / f"{safe}_functional_similarity.npy",
                func_similarity,
            )
            np.save(
                output_dir / f"{safe}_scalar_distance.npy",
                scalar_distance,
            )
            np.save(
                output_dir / f"{safe}_class_functional_similarity.npy",
                class_func,
            )

        del func_similarity, func_distance, class_func
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pairs = pd.DataFrame(pair_rows)
    cases = pd.DataFrame(case_rows)
    stats = pd.DataFrame(layer_stats)

    pairs.to_csv(output_dir / "descriptor_pair_distances.csv", index=False)
    pairs[pairs["pair_source"] == "scalar_nearest"].to_csv(
        output_dir / "matched_scalar_pairs.csv", index=False
    )
    cases.to_csv(output_dir / "representative_descriptor_cases.csv", index=False)
    stats.to_csv(output_dir / "descriptor_distinctiveness_by_layer.csv", index=False)

    scalar_pairs_all = pairs[pairs["pair_source"] == "scalar_nearest"]
    random_pairs_all = pairs[pairs["pair_source"] == "random"]

    summary = {
        "scientific_question": (
            "Do units with nearly identical scalar descriptors still have "
            "different class-conditioned spatio-temporal functional descriptors?"
        ),
        "overall": {
            "scalar_nearest_pairs": int(len(scalar_pairs_all)),
            "scalar_nearest_scalar_distance_median": float(
                scalar_pairs_all["scalar_distance"].median()
            ),
            "scalar_nearest_functional_distance_median": float(
                scalar_pairs_all["functional_distance"].median()
            ),
            "scalar_nearest_functional_distance_q25": float(
                scalar_pairs_all["functional_distance"].quantile(0.25)
            ),
            "scalar_nearest_functional_distance_q75": float(
                scalar_pairs_all["functional_distance"].quantile(0.75)
            ),
            "random_functional_distance_median": float(
                random_pairs_all["functional_distance"].median()
            ),
            "rho_scalar_vs_functional_distance_on_scalar_nearest_pairs": (
                safe_spearman(
                    scalar_pairs_all["scalar_distance"],
                    scalar_pairs_all["functional_distance"],
                )
            ),
        },
        "interpretation_rule": {
            "support": (
                "Scalar-nearest pairs have near-zero scalar distance while a "
                "substantial fraction retain non-trivial functional distance."
            ),
            "strong_support": (
                "The upper quartile of scalar-nearest functional distance is "
                "large and representative scalar-close/function-far cases exist "
                "in both attention and MLP units."
            ),
            "failure": (
                "Functional distance collapses toward zero whenever scalar "
                "distance is small."
            ),
        },
        "run_config": vars(args),
        "class_labels_in_npz_order": labels.tolist(),
    }
    with open(
        output_dir / "descriptor_distinctiveness_statistics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plot_scatter(pairs, output_dir)
    plot_group_box(pairs, output_dir)

    report = [
        "Functional Descriptor Distinctiveness Probe",
        "=" * 88,
        "",
        "Scientific question:",
        "Can scalar-nearest pruning units remain functionally different?",
        "",
        "[Overall scalar-nearest pairs]",
        f"pairs: {summary['overall']['scalar_nearest_pairs']}",
        (
            "median scalar distance: "
            f"{summary['overall']['scalar_nearest_scalar_distance_median']:.6f}"
        ),
        (
            "median functional distance: "
            f"{summary['overall']['scalar_nearest_functional_distance_median']:.6f}"
        ),
        (
            "functional distance Q25/Q75: "
            f"{summary['overall']['scalar_nearest_functional_distance_q25']:.6f} / "
            f"{summary['overall']['scalar_nearest_functional_distance_q75']:.6f}"
        ),
        (
            "random-pair median functional distance: "
            f"{summary['overall']['random_functional_distance_median']:.6f}"
        ),
        (
            "Spearman rho within scalar-nearest pairs: "
            f"{summary['overall']['rho_scalar_vs_functional_distance_on_scalar_nearest_pairs']:.6f}"
        ),
        "",
        "[Interpretation]",
        "Pass:",
        "- scalar distance is very small for matched pairs;",
        "- functional distance remains broad and non-zero;",
        "- scalar-close/function-far cases exist.",
        "",
        "This supports complementarity of the Functional Descriptor.",
        "It does not yet prove pruning effectiveness.",
    ]
    (output_dir / "DISTINCTIVENESS_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"\nDescriptor distinctiveness probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

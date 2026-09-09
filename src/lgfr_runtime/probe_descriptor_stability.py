#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional Descriptor Stability Probe (V1).

Scientific question
-------------------
For the same pruning unit and the same action class, is its class-conditioned
spatio-temporal functional descriptor more consistent across disjoint video
subsets than descriptors belonging to different units?

Input
-----
1. cstc_probe_arrays.npz
   Each selected layer must contain:
       layer_xxx_contribution_volumes: [V, U, T, H, W]

2. unit_cstc_metrics.csv
   Must contain:
       layer, unit_type, unit_index, d_abs, d_rel

The script is fully offline. It does not load the model or perform masking.

Descriptor comparison
---------------------
For every class and repeated disjoint split A/B of its videos:

1. Positive contribution prototype:
       C_A = sum_{n in A} max(C_n, 0)
       C_B = sum_{n in B} max(C_n, 0)

2. Temporal trajectory:
       q(t) = sum_{x,y} C(t,x,y)

3. Spatial prototype:
       s(x,y) = sum_t C(t,x,y)

4. Temporal similarity:
   maximum overlap-penalized cosine similarity over every valid temporal shift.

5. Spatial similarity:
   cosine similarity between spatial prototypes.

6. Structured descriptor similarity:
       S = sqrt(S_temporal * S_spatial)

For unit i:
    self similarity  = S(F_i^A, F_i^B)
    cross similarity = S(F_i^A, F_j^B), j != i

The cross unit is sampled deterministically for each repeat. The primary test is
paired self-vs-cross at the unit/class/repeat level.

Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cross-video stability of functional descriptors"
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--output_dir", default="./descriptor_stability_probe")
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' or a regular expression over layer names",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Used only when --class_labels is not supplied",
    )
    parser.add_argument(
        "--videos_per_class",
        type=int,
        default=3,
        help="Used only when --class_labels is not supplied",
    )
    parser.add_argument(
        "--class_labels",
        default="",
        help=(
            "Comma-separated label for every NPZ video in exact saved order, "
            "for example: 39,39,39,53,53,53,72,72,72. "
            "If omitted, contiguous class blocks are assumed."
        ),
    )
    parser.add_argument(
        "--split_repeats",
        type=int,
        default=20,
        help="Repeated disjoint class-wise splits",
    )
    parser.add_argument(
        "--cross_samples_per_unit",
        type=int,
        default=1,
        help="Number of different-unit controls per self comparison",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--compute_device",
        default="auto",
        help="'auto', 'cpu', or e.g. 'cuda:0'",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--save_similarity_matrices",
        action="store_true",
        help=(
            "Save class/repeat cross-split UxU matrices. This can consume "
            "substantial disk space for FFN layers."
        ),
    )
    parser.add_argument(
        "--max_matrix_repeats",
        type=int,
        default=1,
        help="Maximum repeats per class for which full matrices are saved",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    required = {"layer", "unit_index", "unit_type"}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"unit_metrics.csv missing columns: {sorted(missing)}")

    columns = [
        column
        for column in ("stage", "block", "unit_type", "layer")
        if column in metrics.columns
    ]
    table = metrics[columns].drop_duplicates()
    sort_columns = [
        column
        for column in ("stage", "block", "unit_type", "layer")
        if column in table.columns
    ]
    if sort_columns:
        table = table.sort_values(sort_columns, kind="stable")
    return table["layer"].tolist()


def parse_labels(args: argparse.Namespace, video_count: int) -> np.ndarray:
    if args.class_labels.strip():
        labels = np.asarray(
            [int(value.strip()) for value in args.class_labels.split(",")],
            dtype=np.int64,
        )
        if len(labels) != video_count:
            raise ValueError(
                f"--class_labels contains {len(labels)} entries, "
                f"but NPZ contains {video_count} videos."
            )
        return labels

    expected = args.num_classes * args.videos_per_class
    if expected != video_count:
        raise ValueError(
            "Cannot infer labels: num_classes * videos_per_class "
            f"= {expected}, but NPZ contains {video_count} videos. "
            "Supply --class_labels in exact NPZ order."
        )
    return np.repeat(
        np.arange(args.num_classes, dtype=np.int64),
        args.videos_per_class,
    )


def build_split_plan(
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Return (class_id, repeat, subset_a, subset_b)."""
    rng = np.random.RandomState(seed)
    plan: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise ValueError(
                f"Class {class_id} has only {len(indices)} video(s); "
                "at least two are required."
            )

        seen = set()
        attempts = 0
        target = repeats
        while len(seen) < target and attempts < max(1000, repeats * 50):
            attempts += 1
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            split = max(1, len(shuffled) // 2)
            subset_a = np.sort(shuffled[:split])
            subset_b = np.sort(shuffled[split:])
            if len(subset_b) == 0:
                continue

            # A/B orientation is meaningful for cross controls, so preserve it.
            key = (tuple(subset_a.tolist()), tuple(subset_b.tolist()))
            if key in seen and len(indices) <= 4:
                # For very small class sample counts, all unique splits are soon
                # exhausted. Repeated deterministic resampling remains useful for
                # different cross-unit controls.
                pass
            else:
                seen.add(key)

            repeat_id = len(plan)
            plan.append((int(class_id), repeat_id, subset_a, subset_b))
            if sum(1 for item in plan if item[0] == int(class_id)) >= target:
                break

        class_items = [item for item in plan if item[0] == int(class_id)]
        if len(class_items) < target:
            # Complete the requested count by cycling through available splits.
            base = class_items.copy()
            if not base:
                raise RuntimeError(f"Failed to build a split for class {class_id}")
            while len(class_items) < target:
                source = base[len(class_items) % len(base)]
                item = (
                    int(class_id),
                    len(plan),
                    source[2].copy(),
                    source[3].copy(),
                )
                plan.append(item)
                class_items.append(item)

    # Replace global repeat IDs with class-local repeat IDs.
    normalized = []
    counters: Dict[int, int] = {}
    for class_id, _, subset_a, subset_b in plan:
        local = counters.get(class_id, 0)
        counters[class_id] = local + 1
        normalized.append((class_id, local, subset_a, subset_b))
    return normalized


def aggregate_positive_prototype(
    volumes: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Aggregate [V,U,T,H,W] into positive [U,T,H,W]."""
    selected = np.maximum(volumes[indices], 0.0)
    return selected.sum(axis=0, dtype=np.float32)


def temporal_and_spatial(
    prototype: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return temporal [U,T], spatial [U,HW], validity [U]."""
    temporal = prototype.sum(axis=(2, 3), dtype=np.float32)
    spatial = prototype.sum(axis=1, dtype=np.float32).reshape(
        prototype.shape[0], -1
    )
    mass = prototype.sum(axis=(1, 2, 3), dtype=np.float64)
    valid = mass > EPS
    return temporal, spatial, valid


def row_l2_normalize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    valid = norm[:, 0] > EPS
    output = torch.zeros_like(tensor)
    output[valid] = tensor[valid] / norm[valid].clamp_min(EPS)
    return output, valid


def paired_spatial_cosine(
    spatial_a: np.ndarray,
    spatial_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-aligned cosine for self comparisons."""
    units = spatial_a.shape[0]
    output = np.zeros(units, dtype=np.float32)
    valid_output = np.zeros(units, dtype=bool)

    for start in range(0, units, chunk_size):
        end = min(start + chunk_size, units)
        left = torch.from_numpy(spatial_a[start:end]).to(
            device=device, dtype=torch.float32
        )
        right = torch.from_numpy(spatial_b[start:end]).to(
            device=device, dtype=torch.float32
        )
        left_n, left_valid = row_l2_normalize(left)
        right_n, right_valid = row_l2_normalize(right)
        values = torch.sum(left_n * right_n, dim=1).clamp_(0.0, 1.0)
        output[start:end] = values.cpu().numpy()
        valid_output[start:end] = (
            left_valid & right_valid
        ).cpu().numpy()

    return output, valid_output


def paired_temporal_shift_similarity(
    temporal_a: np.ndarray,
    temporal_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-aligned maximum overlap-penalized temporal cosine."""
    units, time_bins = temporal_a.shape
    best = np.zeros(units, dtype=np.float32)
    best_shift = np.zeros(units, dtype=np.int32)
    valid_output = np.zeros(units, dtype=bool)

    for start in range(0, units, chunk_size):
        end = min(start + chunk_size, units)
        left_all = torch.from_numpy(temporal_a[start:end]).to(
            device=device, dtype=torch.float32
        )
        right_all = torch.from_numpy(temporal_b[start:end]).to(
            device=device, dtype=torch.float32
        )
        local_best = torch.zeros(end - start, device=device)
        local_shift = torch.zeros(
            end - start, device=device, dtype=torch.int64
        )
        any_valid = torch.zeros(
            end - start, device=device, dtype=torch.bool
        )

        for shift in range(-(time_bins - 1), time_bins):
            if shift >= 0:
                left = left_all[:, : time_bins - shift]
                right = right_all[:, shift:]
            else:
                offset = -shift
                left = left_all[:, offset:]
                right = right_all[:, : time_bins - offset]

            overlap = left.shape[1]
            if overlap <= 0:
                continue
            left_n, left_valid = row_l2_normalize(left)
            right_n, right_valid = row_l2_normalize(right)
            valid = left_valid & right_valid
            similarity = torch.sum(left_n * right_n, dim=1)
            similarity *= float(overlap) / float(time_bins)
            similarity = torch.where(
                valid, similarity, torch.zeros_like(similarity)
            )
            improve = similarity > local_best
            local_best = torch.where(improve, similarity, local_best)
            local_shift = torch.where(
                improve,
                torch.full_like(local_shift, shift),
                local_shift,
            )
            any_valid |= valid

        best[start:end] = local_best.clamp_(0.0, 1.0).cpu().numpy()
        best_shift[start:end] = local_shift.cpu().numpy()
        valid_output[start:end] = any_valid.cpu().numpy()

    return best, best_shift, valid_output


def paired_descriptor_similarity(
    temporal_a: np.ndarray,
    spatial_a: np.ndarray,
    temporal_b: np.ndarray,
    spatial_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temporal_sim, best_shift, valid_t = paired_temporal_shift_similarity(
        temporal_a, temporal_b, device, chunk_size
    )
    spatial_sim, valid_s = paired_spatial_cosine(
        spatial_a, spatial_b, device, chunk_size
    )
    valid = valid_t & valid_s
    structured = np.zeros_like(temporal_sim)
    structured[valid] = np.sqrt(
        np.clip(temporal_sim[valid] * spatial_sim[valid], 0.0, 1.0)
    )
    return structured, temporal_sim, spatial_sim, best_shift


def cross_similarity_for_pairs(
    temporal_a: np.ndarray,
    spatial_a: np.ndarray,
    temporal_b: np.ndarray,
    spatial_b: np.ndarray,
    source_units: np.ndarray,
    target_units: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return paired_descriptor_similarity(
        temporal_a[source_units],
        spatial_a[source_units],
        temporal_b[target_units],
        spatial_b[target_units],
        device,
        chunk_size,
    )


def full_spatial_cross_matrix(
    spatial_a: np.ndarray,
    spatial_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    right = torch.from_numpy(spatial_b).to(device=device, dtype=torch.float32)
    right, right_valid = row_l2_normalize(right)
    units = spatial_a.shape[0]
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(range(0, units, chunk_size), desc=desc, ncols=105):
        end = min(start + chunk_size, units)
        left = torch.from_numpy(spatial_a[start:end]).to(
            device=device, dtype=torch.float32
        )
        left, left_valid = row_l2_normalize(left)
        block = (left @ right.T).clamp_(0.0, 1.0)
        valid = left_valid[:, None] & right_valid[None, :]
        block = torch.where(valid, block, torch.zeros_like(block))
        output[start:end] = block.cpu().numpy()

    del right
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def full_temporal_cross_matrix(
    temporal_a: np.ndarray,
    temporal_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    right_all = torch.from_numpy(temporal_b).to(
        device=device, dtype=torch.float32
    )
    units, time_bins = temporal_a.shape
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(range(0, units, chunk_size), desc=desc, ncols=105):
        end = min(start + chunk_size, units)
        left_all = torch.from_numpy(temporal_a[start:end]).to(
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
            left_n, left_valid = row_l2_normalize(left)
            right_n, right_valid = row_l2_normalize(right)
            block = left_n @ right_n.T
            block *= float(overlap) / float(time_bins)
            valid = left_valid[:, None] & right_valid[None, :]
            block = torch.where(valid, block, torch.zeros_like(block))
            best = torch.maximum(best, block)

        output[start:end] = best.clamp_(0.0, 1.0).cpu().numpy()

    del right_all
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def full_descriptor_cross_matrix(
    temporal_a: np.ndarray,
    spatial_a: np.ndarray,
    temporal_b: np.ndarray,
    spatial_b: np.ndarray,
    device: torch.device,
    chunk_size: int,
    desc_prefix: str,
) -> np.ndarray:
    temporal = full_temporal_cross_matrix(
        temporal_a,
        temporal_b,
        device,
        chunk_size,
        f"{desc_prefix} temporal matrix",
    )
    spatial = full_spatial_cross_matrix(
        spatial_a,
        spatial_b,
        device,
        chunk_size,
        f"{desc_prefix} spatial matrix",
    )
    return np.sqrt(np.clip(temporal * spatial, 0.0, 1.0)).astype(np.float32)


def paired_rank_biserial(self_values: np.ndarray, cross_values: np.ndarray) -> float:
    differences = np.asarray(self_values) - np.asarray(cross_values)
    nonzero = differences[np.abs(differences) > EPS]
    if len(nonzero) == 0:
        return 0.0
    positive = float(np.sum(nonzero > 0))
    negative = float(np.sum(nonzero < 0))
    return (positive - negative) / len(nonzero)


def bootstrap_median_difference(
    self_values: np.ndarray,
    cross_values: np.ndarray,
    seed: int,
    iterations: int = 2000,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    differences = np.asarray(self_values) - np.asarray(cross_values)
    if len(differences) == 0:
        return float("nan"), float("nan")
    values = []
    for _ in range(iterations):
        sample = rng.choice(differences, size=len(differences), replace=True)
        values.append(float(np.median(sample)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def plot_distributions(frame: pd.DataFrame, output_dir: Path) -> None:
    self_values = frame["self_similarity"].dropna().to_numpy()
    cross_values = frame["cross_similarity"].dropna().to_numpy()

    plt.figure(figsize=(6.8, 4.8))
    bins = np.linspace(0.0, 1.0, 31)
    plt.hist(self_values, bins=bins, alpha=0.65, label="Self")
    plt.hist(cross_values, bins=bins, alpha=0.65, label="Cross-unit")
    plt.xlabel("Structured descriptor similarity")
    plt.ylabel("Count")
    plt.title("Descriptor stability: self vs cross-unit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "descriptor_hist.png", dpi=220)
    plt.savefig(output_dir / "descriptor_hist.pdf")
    plt.close()

    groups = [
        frame.loc[frame["unit_type"] == unit_type, "self_similarity"].dropna()
        for unit_type in sorted(frame["unit_type"].unique())
    ]
    cross_groups = [
        frame.loc[frame["unit_type"] == unit_type, "cross_similarity"].dropna()
        for unit_type in sorted(frame["unit_type"].unique())
    ]
    labels = []
    values = []
    for unit_type, self_group, cross_group in zip(
        sorted(frame["unit_type"].unique()), groups, cross_groups
    ):
        labels.extend([f"{unit_type}\nself", f"{unit_type}\ncross"])
        values.extend([self_group.to_numpy(), cross_group.to_numpy()])

    plt.figure(figsize=(max(7.0, 1.4 * len(values)), 4.8))
    plt.violinplot(values, showmedians=True, showextrema=False)
    plt.xticks(np.arange(1, len(values) + 1), labels)
    plt.ylabel("Structured descriptor similarity")
    plt.title("Descriptor stability by pruning-unit type")
    plt.tight_layout()
    plt.savefig(output_dir / "descriptor_violin.png", dpi=220)
    plt.savefig(output_dir / "descriptor_violin.pdf")
    plt.close()


def summarize_group(table: pd.DataFrame, seed: int) -> Dict[str, float]:
    valid = table.dropna(subset=["self_similarity", "cross_similarity"])
    self_values = valid["self_similarity"].to_numpy(dtype=np.float64)
    cross_values = valid["cross_similarity"].to_numpy(dtype=np.float64)
    if len(valid) == 0:
        return {
            "comparisons": 0,
            "self_median": float("nan"),
            "cross_median": float("nan"),
            "median_difference": float("nan"),
            "self_greater_rate": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "rank_biserial": float("nan"),
            "median_difference_ci_low": float("nan"),
            "median_difference_ci_high": float("nan"),
        }

    try:
        test = wilcoxon(
            self_values,
            cross_values,
            alternative="greater",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    except ValueError:
        statistic = float("nan")
        p_value = float("nan")

    ci_low, ci_high = bootstrap_median_difference(
        self_values, cross_values, seed
    )
    return {
        "comparisons": int(len(valid)),
        "self_median": float(np.median(self_values)),
        "cross_median": float(np.median(cross_values)),
        "median_difference": float(np.median(self_values - cross_values)),
        "self_greater_rate": float(np.mean(self_values > cross_values)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "rank_biserial": float(
            paired_rank_biserial(self_values, cross_values)
        ),
        "median_difference_ci_low": ci_low,
        "median_difference_ci_high": ci_high,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.compute_device)
    print(f"Descriptor comparison device: {device}")

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
            for index, layer_name in enumerate(layers)
            if pattern.search(layer_name)
        ]
        layers = [layers[index] for index in keep]
        volume_keys = [volume_keys[index] for index in keep]

    if not layers:
        raise ValueError("No layer matched --layers")

    first_video_count = int(arrays[volume_keys[0]].shape[0])
    labels = parse_labels(args, first_video_count)
    split_plan = build_split_plan(labels, args.split_repeats, args.seed)
    print(f"Class labels in NPZ order: {labels.tolist()}")
    print(f"Disjoint split comparisons per class: {args.split_repeats}")

    rng = np.random.RandomState(args.seed)
    rows: List[Dict[str, object]] = []
    matrix_manifest: List[Dict[str, object]] = []
    saved_matrices: List[np.ndarray] = []
    saved_matrix_names: List[str] = []

    for layer_position, (layer_name, volume_key) in enumerate(
        zip(layers, volume_keys), start=1
    ):
        volumes = arrays[volume_key]
        if volumes.shape[0] != len(labels):
            raise ValueError(
                f"{layer_name}: video count {volumes.shape[0]} differs from "
                f"label count {len(labels)}"
            )

        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        units = int(volumes.shape[1])
        if units != len(layer_metrics):
            raise ValueError(
                f"{layer_name}: NPZ units={units}, metric rows={len(layer_metrics)}"
            )
        expected_indices = np.arange(units)
        actual_indices = layer_metrics["unit_index"].to_numpy(dtype=int)
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(
                f"{layer_name}: unit_index must be contiguous 0..{units - 1}"
            )

        unit_type = str(layer_metrics["unit_type"].iloc[0])
        print(
            f"\n[Layer {layer_position}/{len(layers)}] {layer_name} "
            f"({unit_type}, units={units}, volume={tuple(volumes.shape)})"
        )

        class_repeat_saved: Dict[int, int] = {}
        layer_rows_before = len(rows)

        for class_id, repeat_id, subset_a, subset_b in tqdm(
            split_plan,
            desc=f"{layer_name} split stability",
            ncols=105,
        ):
            prototype_a = aggregate_positive_prototype(volumes, subset_a)
            prototype_b = aggregate_positive_prototype(volumes, subset_b)
            temporal_a, spatial_a, valid_a = temporal_and_spatial(prototype_a)
            temporal_b, spatial_b, valid_b = temporal_and_spatial(prototype_b)

            (
                self_similarity,
                self_temporal,
                self_spatial,
                self_shift,
            ) = paired_descriptor_similarity(
                temporal_a,
                spatial_a,
                temporal_b,
                spatial_b,
                device,
                args.chunk_size,
            )

            source_units = np.repeat(
                np.arange(units, dtype=np.int64),
                args.cross_samples_per_unit,
            )
            target_units = np.empty_like(source_units)
            for position, source in enumerate(source_units):
                target = rng.randint(0, units - 1)
                if target >= source:
                    target += 1
                target_units[position] = target

            (
                cross_similarity,
                cross_temporal,
                cross_spatial,
                cross_shift,
            ) = cross_similarity_for_pairs(
                temporal_a,
                spatial_a,
                temporal_b,
                spatial_b,
                source_units,
                target_units,
                device,
                args.chunk_size,
            )

            # Average multiple cross controls per source unit.
            cross_similarity = cross_similarity.reshape(
                units, args.cross_samples_per_unit
            )
            cross_temporal = cross_temporal.reshape(
                units, args.cross_samples_per_unit
            )
            cross_spatial = cross_spatial.reshape(
                units, args.cross_samples_per_unit
            )
            cross_shift = cross_shift.reshape(
                units, args.cross_samples_per_unit
            )

            for unit in range(units):
                if not (valid_a[unit] and valid_b[unit]):
                    continue
                for cross_index in range(args.cross_samples_per_unit):
                    target = int(
                        target_units.reshape(
                            units, args.cross_samples_per_unit
                        )[unit, cross_index]
                    )
                    rows.append(
                        {
                            "layer": layer_name,
                            "unit_type": unit_type,
                            "unit_index": unit,
                            "class_id": int(class_id),
                            "repeat": int(repeat_id),
                            "subset_a_indices": "|".join(
                                str(v) for v in subset_a.tolist()
                            ),
                            "subset_b_indices": "|".join(
                                str(v) for v in subset_b.tolist()
                            ),
                            "cross_unit_index": target,
                            "self_similarity": float(self_similarity[unit]),
                            "self_temporal_similarity": float(
                                self_temporal[unit]
                            ),
                            "self_spatial_similarity": float(
                                self_spatial[unit]
                            ),
                            "self_best_shift": int(self_shift[unit]),
                            "cross_similarity": float(
                                cross_similarity[unit, cross_index]
                            ),
                            "cross_temporal_similarity": float(
                                cross_temporal[unit, cross_index]
                            ),
                            "cross_spatial_similarity": float(
                                cross_spatial[unit, cross_index]
                            ),
                            "cross_best_shift": int(
                                cross_shift[unit, cross_index]
                            ),
                            "difference": float(
                                self_similarity[unit]
                                - cross_similarity[unit, cross_index]
                            ),
                            "d_abs": float(layer_metrics.loc[unit, "d_abs"]),
                            "d_rel": float(layer_metrics.loc[unit, "d_rel"]),
                        }
                    )

            saved_for_class = class_repeat_saved.get(int(class_id), 0)
            if (
                args.save_similarity_matrices
                and saved_for_class < args.max_matrix_repeats
            ):
                matrix = full_descriptor_cross_matrix(
                    temporal_a,
                    spatial_a,
                    temporal_b,
                    spatial_b,
                    device,
                    args.chunk_size,
                    f"{layer_name} class {class_id} repeat {repeat_id}",
                )
                safe_layer = layer_name.replace(".", "_")
                matrix_name = (
                    f"{safe_layer}__class_{class_id}__repeat_{repeat_id}"
                )
                saved_matrices.append(matrix)
                saved_matrix_names.append(matrix_name)
                matrix_manifest.append(
                    {
                        "matrix_name": matrix_name,
                        "layer": layer_name,
                        "class_id": int(class_id),
                        "repeat": int(repeat_id),
                        "shape": list(matrix.shape),
                        "subset_a_indices": subset_a.tolist(),
                        "subset_b_indices": subset_b.tolist(),
                    }
                )
                class_repeat_saved[int(class_id)] = saved_for_class + 1

            del prototype_a, prototype_b
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(
            f"  valid paired comparisons: {len(rows) - layer_rows_before}"
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "No valid descriptor comparisons were produced. "
            "Check whether contribution fields contain positive mass."
        )

    frame.to_csv(output_dir / "descriptor_self_cross.csv", index=False)

    unit_stability = (
        frame.groupby(
            ["layer", "unit_type", "unit_index", "class_id"],
            as_index=False,
        )
        .agg(
            comparisons=("self_similarity", "size"),
            self_similarity_mean=("self_similarity", "mean"),
            self_similarity_median=("self_similarity", "median"),
            self_temporal_mean=("self_temporal_similarity", "mean"),
            self_spatial_mean=("self_spatial_similarity", "mean"),
            cross_similarity_mean=("cross_similarity", "mean"),
            cross_similarity_median=("cross_similarity", "median"),
            stability_margin_mean=("difference", "mean"),
            stability_margin_median=("difference", "median"),
            self_greater_rate=("difference", lambda x: float((x > 0).mean())),
        )
    )
    unit_stability.to_csv(
        output_dir / "descriptor_stability.csv", index=False
    )

    overall = summarize_group(frame, args.seed)
    by_type = {
        unit_type: summarize_group(
            frame[frame["unit_type"] == unit_type],
            args.seed + index + 1,
        )
        for index, unit_type in enumerate(sorted(frame["unit_type"].unique()))
    }
    by_layer = {
        layer_name: summarize_group(
            frame[frame["layer"] == layer_name],
            args.seed + index + 101,
        )
        for index, layer_name in enumerate(sorted(frame["layer"].unique()))
    }
    by_class = {
        str(class_id): summarize_group(
            frame[frame["class_id"] == class_id],
            args.seed + index + 201,
        )
        for index, class_id in enumerate(sorted(frame["class_id"].unique()))
    }

    # Unpaired test is supplementary; paired Wilcoxon is primary.
    unpaired = mannwhitneyu(
        frame["self_similarity"],
        frame["cross_similarity"],
        alternative="greater",
    )

    statistics = {
        "overall": overall,
        "by_unit_type": by_type,
        "by_layer": by_layer,
        "by_class": by_class,
        "supplementary_mann_whitney": {
            "alternative": "self > cross",
            "statistic": float(unpaired.statistic),
            "p_value": float(unpaired.pvalue),
        },
        "decision_rule": {
            "primary": (
                "self median > cross median, paired Wilcoxon p < 0.05, "
                "and positive median difference confidence interval"
            ),
            "strong_support": (
                "self_greater_rate >= 0.70 and rank_biserial >= 0.40"
            ),
        },
        "run_config": vars(args),
        "class_labels_in_npz_order": labels.tolist(),
    }
    with open(
        output_dir / "descriptor_statistics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=2)

    if saved_matrices:
        payload = {
            name: matrix
            for name, matrix in zip(saved_matrix_names, saved_matrices)
        }
        np.savez_compressed(
            output_dir / "descriptor_similarity_matrix.npz",
            **payload,
        )
        with open(
            output_dir / "descriptor_similarity_matrix_manifest.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(matrix_manifest, handle, ensure_ascii=False, indent=2)
    else:
        # Preserve a machine-readable similarity artifact without allocating UxU.
        np.savez_compressed(
            output_dir / "descriptor_similarity_samples.npz",
            self_similarity=frame["self_similarity"].to_numpy(np.float32),
            cross_similarity=frame["cross_similarity"].to_numpy(np.float32),
            temporal_self=frame["self_temporal_similarity"].to_numpy(np.float32),
            temporal_cross=frame["cross_temporal_similarity"].to_numpy(np.float32),
            spatial_self=frame["self_spatial_similarity"].to_numpy(np.float32),
            spatial_cross=frame["cross_spatial_similarity"].to_numpy(np.float32),
        )

    plot_distributions(frame, output_dir)

    report_lines = [
        "Functional Descriptor Stability Probe",
        "=" * 88,
        "",
        "Scientific question:",
        "Is the same pruning unit more functionally consistent across disjoint",
        "same-class video subsets than different pruning units?",
        "",
        "[Overall]",
        f"paired comparisons: {overall['comparisons']}",
        f"self median: {overall['self_median']:.6f}",
        f"cross median: {overall['cross_median']:.6f}",
        f"median self-cross difference: {overall['median_difference']:.6f}",
        f"95% bootstrap CI: [{overall['median_difference_ci_low']:.6f}, "
        f"{overall['median_difference_ci_high']:.6f}]",
        f"self > cross rate: {overall['self_greater_rate']:.6f}",
        f"paired Wilcoxon p-value: {overall['wilcoxon_p_value']:.6g}",
        f"paired rank-biserial effect: {overall['rank_biserial']:.6f}",
        "",
        "[By unit type]",
    ]
    for unit_type, result in by_type.items():
        report_lines.extend(
            [
                f"{unit_type}:",
                f"  comparisons: {result['comparisons']}",
                f"  self median: {result['self_median']:.6f}",
                f"  cross median: {result['cross_median']:.6f}",
                f"  median difference: {result['median_difference']:.6f}",
                f"  self > cross rate: {result['self_greater_rate']:.6f}",
                f"  Wilcoxon p: {result['wilcoxon_p_value']:.6g}",
                f"  rank-biserial: {result['rank_biserial']:.6f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "[Interpretation]",
            "Pass:",
            "- Same-unit descriptors are consistently more similar than cross-unit",
            "  descriptors across disjoint same-class video subsets.",
            "- This supports descriptor stability, not pruning redundancy.",
            "",
            "Fail:",
            "- Self and cross distributions overlap without a meaningful positive",
            "  paired effect. In that case, increase calibration videos or revise",
            "  descriptor construction before proceeding to distinctiveness.",
            "",
            "This probe does not establish replaceability, redundancy, or pruning gain.",
        ]
    )
    (output_dir / "STABILITY_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"\nDescriptor stability probe complete: {output_dir.resolve()}")
    print(
        f"Overall self median={overall['self_median']:.6f}, "
        f"cross median={overall['cross_median']:.6f}, "
        f"Wilcoxon p={overall['wilcoxon_p_value']:.6g}"
    )


if __name__ == "__main__":
    main()

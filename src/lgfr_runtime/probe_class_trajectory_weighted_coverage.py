#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class-aware trajectory-weighted contribution-pattern coverage probe.

This is an upgraded version of the previous coverage probe. It keeps the same
two-stage pruning logic:

    scalar risk -> low-risk candidate pool -> contribution-pattern coverage

but upgrades the pattern from a flattened T×H×W cosine to a class-conditioned
spatio-temporal trajectory affinity.

For each class and unit:
1. Temporal trajectory:
       q_i^c(t) = sum_{video,x,y} [C_i]^+
2. Spatial prototype:
       s_i^c(x,y) = sum_{video,t} [C_i]^+
3. Shift-tolerant temporal similarity:
       maximum normalized cross-correlation over all valid temporal shifts,
       multiplied by the overlap fraction.
4. Spatial similarity:
       cosine similarity between class-conditioned spatial prototypes.
5. Joint trajectory affinity:
       sqrt(temporal_similarity * spatial_similarity)

Coverage is class balanced. The recommended strategy additionally weights each
represented unit by its class-conditioned positive contribution mass, so that
high-contribution functions receive more protection than negligible patterns.

No DTW, optical flow, learned alignment module, or tunable temporal/spatial
fusion coefficient is introduced.

Compared strategies
-------------------
- scalar_only
- flat_hybrid
- class_trajectory_pattern_only
- class_trajectory_hybrid_unweighted
- class_trajectory_hybrid_weighted  (recommended candidate)

The script masks selected units only; it does not physically prune or fine-tune.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from probe_ctfrs_dynamic_function import (
        EPS,
        build_balanced_loader,
        discover_unit_layers,
        ensure_project_importable,
        load_model,
        set_seed,
    )
    from probe_contribution_pattern_coverage_fast import (
        LayerData,
        collect_cache,
        evaluate_strategy,
        full_functional_affinity,
        infer_unit_cost,
        normalize_patterns,
        ordered_layers,
        scalar_risk_values,
        strategy_coverage,
    )
except Exception as exc:
    raise ImportError(
        "Place this script beside probe_ctfrs_dynamic_function.py and "
        "probe_contribution_pattern_coverage_fast.py."
    ) from exc


@dataclass
class ClassTrajectoryData:
    class_ids: np.ndarray                    # [C]
    affinities: np.ndarray                   # [C,U,U]
    importance_weights: np.ndarray            # [C,U], each class sums to one
    valid_units: np.ndarray                   # [C,U]
    temporal_similarity: np.ndarray           # [C,U,U]
    spatial_similarity: np.ndarray            # [C,U,U]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-aware trajectory-weighted pattern-coverage probe"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--output_dir", default="./class_trajectory_coverage_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--affinity_device", default="auto")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", choices=("true", "pred"), default="true")
    parser.add_argument("--seed", type=int, default=3407)

    # Probe budget controls only.
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--candidate_multiplier", type=float, default=2.0)
    parser.add_argument("--ablation_videos", type=int, default=9)
    parser.add_argument(
        "--layers",
        default=r"layers\.3\.blocks\.(0|1)\.(attn|mlp)",
    )
    parser.add_argument("--affinity_chunk_size", type=int, default=256)
    parser.add_argument(
        "--scalar_risk",
        choices=("d_rel", "d_abs", "product", "sum"),
        default="d_rel",
    )
    parser.add_argument(
        "--strategies",
        default=(
            "scalar_only,flat_hybrid,class_trajectory_pattern_only,"
            "class_trajectory_hybrid_unweighted,"
            "class_trajectory_hybrid_weighted"
        ),
    )
    return parser.parse_args()


def resolve_device(requested: str, model_device: str) -> torch.device:
    if requested == "auto":
        requested = model_device if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def collect_calibration_order(
    loader: torch.utils.data.DataLoader,
    cache_limit: int,
) -> Tuple[np.ndarray, List[Tuple[torch.Tensor, torch.Tensor, int]]]:
    labels: List[int] = []
    cache: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for videos, targets, indices in loader:
        label = int(targets[0])
        video_id = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        labels.append(label)
        if len(cache) < cache_limit:
            cache.append((videos.cpu(), targets.cpu(), video_id))
    return np.asarray(labels, dtype=np.int64), cache


def positive_class_prototypes(
    volumes: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return class IDs, temporal curves, spatial prototypes, contribution masses.

    Input:
        volumes [V,U,T,H,W]
    Output:
        temporal [C,U,T]
        spatial  [C,U,H*W]
        masses   [C,U]
    """
    positive = np.maximum(np.asarray(volumes, dtype=np.float32), 0.0)
    class_ids = np.unique(labels)
    temporal_all = []
    spatial_all = []
    mass_all = []

    for class_id in class_ids:
        selected = positive[labels == class_id]
        if len(selected) == 0:
            continue
        aggregate = selected.sum(axis=0)          # [U,T,H,W]
        temporal = aggregate.sum(axis=(2, 3))     # [U,T]
        spatial = aggregate.sum(axis=1).reshape(aggregate.shape[0], -1)  # [U,HW]
        mass = aggregate.sum(axis=(1, 2, 3))      # [U]
        temporal_all.append(temporal)
        spatial_all.append(spatial)
        mass_all.append(mass)

    return (
        class_ids,
        np.stack(temporal_all, axis=0),
        np.stack(spatial_all, axis=0),
        np.stack(mass_all, axis=0),
    )


def row_l2_normalize(array: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(array, dim=1, keepdim=True)
    output = torch.zeros_like(array)
    valid = norm[:, 0] > EPS
    output[valid] = array[valid] / norm[valid].clamp_min(EPS)
    return output


def shift_tolerant_temporal_similarity(
    curves: np.ndarray,
    chunk_size: int,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    """Maximum overlap-penalized cosine over all non-empty temporal shifts.

    curves: [U,T], non-negative.
    A shift with overlap L is multiplied by L/T. Therefore a one-bin accidental
    match cannot receive a high score.
    """
    curves_t = torch.from_numpy(curves).to(device=device, dtype=torch.float32)
    units, time_bins = curves_t.shape
    result = torch.zeros((units, units), device=device, dtype=torch.float32)

    for start in tqdm(
        range(0, units, chunk_size), desc=desc, ncols=105
    ):
        end = min(start + chunk_size, units)
        block_best = torch.zeros((end - start, units), device=device)

        for shift in range(-(time_bins - 1), time_bins):
            if shift >= 0:
                left = curves_t[start:end, : time_bins - shift]
                right = curves_t[:, shift:]
            else:
                offset = -shift
                left = curves_t[start:end, offset:]
                right = curves_t[:, : time_bins - offset]

            overlap = left.shape[1]
            if overlap <= 0:
                continue
            left_norm = row_l2_normalize(left)
            right_norm = row_l2_normalize(right)
            similarity = left_norm @ right_norm.T
            similarity.mul_(float(overlap) / float(time_bins))
            block_best = torch.maximum(block_best, similarity)

        result[start:end] = block_best.clamp_(0.0, 1.0)

    result = 0.5 * (result + result.T)
    result.fill_diagonal_(1.0)
    output = result.cpu().numpy()
    del curves_t, result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def cosine_similarity_matrix(
    vectors: np.ndarray,
    chunk_size: int,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    vectors_t = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    vectors_t = row_l2_normalize(vectors_t)
    units = vectors_t.shape[0]
    result = torch.empty((units, units), device=device, dtype=torch.float32)
    for start in tqdm(range(0, units, chunk_size), desc=desc, ncols=105):
        end = min(start + chunk_size, units)
        result[start:end] = (vectors_t[start:end] @ vectors_t.T).clamp_(0.0, 1.0)
    result = 0.5 * (result + result.T)
    result.fill_diagonal_(1.0)
    output = result.cpu().numpy()
    del vectors_t, result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def build_class_trajectory_data(
    volumes: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
    device: torch.device,
    layer_name: str,
) -> ClassTrajectoryData:
    class_ids, temporal, spatial, masses = positive_class_prototypes(volumes, labels)
    class_affinities = []
    temporal_matrices = []
    spatial_matrices = []
    valid_units = []
    weights = []

    for class_pos, class_id in enumerate(class_ids):
        temporal_sim = shift_tolerant_temporal_similarity(
            temporal[class_pos],
            chunk_size,
            device,
            f"{layer_name} class {class_id} temporal",
        )
        spatial_sim = cosine_similarity_matrix(
            spatial[class_pos],
            chunk_size,
            device,
            f"{layer_name} class {class_id} spatial",
        )
        joint = np.sqrt(np.clip(temporal_sim * spatial_sim, 0.0, 1.0))
        np.fill_diagonal(joint, 1.0)

        mass = masses[class_pos].astype(np.float64)
        valid = mass > EPS
        weight = np.zeros_like(mass)
        if np.any(valid):
            weight[valid] = mass[valid] / mass[valid].sum()

        class_affinities.append(joint.astype(np.float32))
        temporal_matrices.append(temporal_sim.astype(np.float32))
        spatial_matrices.append(spatial_sim.astype(np.float32))
        valid_units.append(valid)
        weights.append(weight)

    return ClassTrajectoryData(
        class_ids=np.asarray(class_ids, dtype=np.int64),
        affinities=np.stack(class_affinities, axis=0),
        importance_weights=np.stack(weights, axis=0),
        valid_units=np.stack(valid_units, axis=0),
        temporal_similarity=np.stack(temporal_matrices, axis=0),
        spatial_similarity=np.stack(spatial_matrices, axis=0),
    )


def class_balanced_coverage(
    affinities: np.ndarray,
    kept: np.ndarray,
    weights: Optional[np.ndarray],
    valid_units: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    class_scores: Dict[int, float] = {}
    for class_pos in range(affinities.shape[0]):
        represented = np.flatnonzero(valid_units[class_pos])
        if len(represented) == 0 or len(kept) == 0:
            score = 0.0
        else:
            best = np.max(
                affinities[class_pos][np.ix_(represented, kept)],
                axis=1,
            )
            if weights is None:
                score = float(np.mean(best))
            else:
                local_weights = weights[class_pos, represented]
                denominator = float(local_weights.sum())
                score = (
                    float(np.sum(local_weights * best) / max(denominator, EPS))
                    if denominator > 0
                    else 0.0
                )
        class_scores[class_pos] = score
    return float(np.mean(list(class_scores.values()))), class_scores


def greedy_class_coverage_removal(
    affinities: np.ndarray,
    valid_units: np.ndarray,
    remove_count: int,
    candidates: Sequence[int],
    unit_cost: np.ndarray,
    desc: str,
    weights: Optional[np.ndarray],
) -> Tuple[List[int], pd.DataFrame]:
    """Exact greedy removal with class-balanced weighted coverage loss.

    The implementation recomputes top-two representatives only for rows affected
    by the removed unit. C is small (3-5 in the probe), so memory remains modest.
    """
    classes, units, _ = affinities.shape
    kept_mask = np.ones(units, dtype=bool)
    candidate_mask = np.zeros(units, dtype=bool)
    candidate_mask[np.asarray(list(candidates), dtype=int)] = True
    costs = np.maximum(np.asarray(unit_cost, dtype=np.float64), EPS)

    best_values = np.zeros((classes, units), dtype=np.float32)
    second_values = np.zeros((classes, units), dtype=np.float32)
    best_units = np.full((classes, units), -1, dtype=np.int32)
    second_units = np.full((classes, units), -1, dtype=np.int32)

    def refresh(class_pos: int, rows: np.ndarray) -> None:
        kept = np.flatnonzero(kept_mask)
        if len(rows) == 0:
            return
        if len(kept) == 0:
            best_values[class_pos, rows] = 0.0
            second_values[class_pos, rows] = 0.0
            best_units[class_pos, rows] = -1
            second_units[class_pos, rows] = -1
            return
        values = affinities[class_pos][np.ix_(rows, kept)]
        if len(kept) == 1:
            best_values[class_pos, rows] = values[:, 0]
            second_values[class_pos, rows] = 0.0
            best_units[class_pos, rows] = kept[0]
            second_units[class_pos, rows] = -1
            return
        order = np.argpartition(-values, kth=1, axis=1)[:, :2]
        v0 = values[np.arange(len(rows)), order[:, 0]]
        v1 = values[np.arange(len(rows)), order[:, 1]]
        swap = v1 > v0
        first = np.where(swap, order[:, 1], order[:, 0])
        second = np.where(swap, order[:, 0], order[:, 1])
        best_values[class_pos, rows] = values[np.arange(len(rows)), first]
        second_values[class_pos, rows] = values[np.arange(len(rows)), second]
        best_units[class_pos, rows] = kept[first]
        second_units[class_pos, rows] = kept[second]

    all_rows = np.arange(units)
    for class_pos in range(classes):
        refresh(class_pos, all_rows)

    removed: List[int] = []
    trace: List[Dict[str, float]] = []

    for step in tqdm(range(remove_count), desc=desc, ncols=105):
        available = kept_mask & candidate_mask
        if not np.any(available):
            break

        losses = np.zeros(units, dtype=np.float64)
        for class_pos in range(classes):
            gaps = np.maximum(
                best_values[class_pos] - second_values[class_pos], 0.0
            ).astype(np.float64)
            represented = valid_units[class_pos]
            if weights is None:
                row_weight = np.zeros(units, dtype=np.float64)
                count = int(represented.sum())
                if count > 0:
                    row_weight[represented] = 1.0 / count
            else:
                row_weight = weights[class_pos].astype(np.float64)
            weighted_gaps = gaps * row_weight
            valid_best = best_units[class_pos] >= 0
            losses += np.bincount(
                best_units[class_pos, valid_best],
                weights=weighted_gaps[valid_best],
                minlength=units,
            ) / max(classes, 1)

        loss_per_cost = losses / costs
        loss_per_cost[~available] = np.inf
        unit = int(np.argmin(loss_per_cost))
        before, _ = class_balanced_coverage(
            affinities,
            np.flatnonzero(kept_mask),
            weights,
            valid_units,
        )

        kept_mask[unit] = False
        removed.append(unit)

        for class_pos in range(classes):
            affected = np.flatnonzero(
                (best_units[class_pos] == unit)
                | (second_units[class_pos] == unit)
            )
            refresh(class_pos, affected)

        after, _ = class_balanced_coverage(
            affinities,
            np.flatnonzero(kept_mask),
            weights,
            valid_units,
        )
        trace.append(
            {
                "step": step + 1,
                "removed_unit": unit,
                "coverage_before": before,
                "coverage_after": after,
                "coverage_loss": before - after,
                "unit_cost": float(costs[unit]),
                "loss_per_cost": float(loss_per_cost[unit]),
            }
        )

    return removed, pd.DataFrame(trace)


def scalar_selection(
    metrics: pd.DataFrame,
    remove_count: int,
    scalar_mode: str,
) -> Tuple[List[int], np.ndarray]:
    risk = scalar_risk_values(metrics, scalar_mode)
    order = np.argsort(risk)
    return order[:remove_count].astype(int).tolist(), order


def select_layer_strategies(
    layer_name: str,
    metrics: pd.DataFrame,
    unit_cost: np.ndarray,
    flat_affinity: np.ndarray,
    trajectory: ClassTrajectoryData,
    remove_count: int,
    candidate_multiplier: float,
    scalar_mode: str,
) -> Dict[str, Tuple[List[int], pd.DataFrame]]:
    scalar_removed, scalar_order = scalar_selection(
        metrics, remove_count, scalar_mode
    )
    candidate_count = min(
        len(metrics),
        max(remove_count, int(math.ceil(candidate_multiplier * remove_count))),
    )
    candidates = scalar_order[:candidate_count].astype(int).tolist()

    # Old flattened-pattern hybrid retained as a direct baseline.
    flat_class = flat_affinity[None, :, :]
    flat_valid = np.ones((1, len(metrics)), dtype=bool)
    flat_weights = np.ones((1, len(metrics)), dtype=np.float64)
    flat_weights /= flat_weights.sum(axis=1, keepdims=True)
    flat_removed, flat_trace = greedy_class_coverage_removal(
        flat_class,
        flat_valid,
        remove_count,
        candidates,
        unit_cost,
        f"{layer_name} flat hybrid",
        flat_weights,
    )

    all_units = list(range(len(metrics)))
    pattern_removed, pattern_trace = greedy_class_coverage_removal(
        trajectory.affinities,
        trajectory.valid_units,
        remove_count,
        all_units,
        unit_cost,
        f"{layer_name} trajectory pattern-only",
        trajectory.importance_weights,
    )
    unweighted_removed, unweighted_trace = greedy_class_coverage_removal(
        trajectory.affinities,
        trajectory.valid_units,
        remove_count,
        candidates,
        unit_cost,
        f"{layer_name} trajectory hybrid unweighted",
        None,
    )
    weighted_removed, weighted_trace = greedy_class_coverage_removal(
        trajectory.affinities,
        trajectory.valid_units,
        remove_count,
        candidates,
        unit_cost,
        f"{layer_name} trajectory hybrid weighted",
        trajectory.importance_weights,
    )

    scalar_trace = pd.DataFrame(
        {
            "step": np.arange(1, len(scalar_removed) + 1),
            "removed_unit": scalar_removed,
            "scalar_risk": scalar_risk_values(metrics, scalar_mode)[scalar_removed],
        }
    )
    for frame in (flat_trace, unweighted_trace, weighted_trace):
        frame["candidate_pool_size"] = candidate_count

    return {
        "scalar_only": (scalar_removed, scalar_trace),
        "flat_hybrid": (flat_removed, flat_trace),
        "class_trajectory_pattern_only": (pattern_removed, pattern_trace),
        "class_trajectory_hybrid_unweighted": (
            unweighted_removed,
            unweighted_trace,
        ),
        "class_trajectory_hybrid_weighted": (
            weighted_removed,
            weighted_trace,
        ),
    }


def trajectory_strategy_coverage(
    layer_trajectory: Mapping[str, ClassTrajectoryData],
    selections: Mapping[str, Sequence[int]],
    weighted: bool,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    layer_scores = []
    layer_detail: Dict[str, Dict[str, float]] = {}
    for layer_name, data in layer_trajectory.items():
        units = data.affinities.shape[1]
        removed = set(int(v) for v in selections.get(layer_name, []))
        kept = np.asarray([i for i in range(units) if i not in removed], dtype=int)
        weights = data.importance_weights if weighted else None
        score, by_class_pos = class_balanced_coverage(
            data.affinities, kept, weights, data.valid_units
        )
        layer_scores.append(score)
        layer_detail[layer_name] = {
            str(int(data.class_ids[pos])): float(value)
            for pos, value in by_class_pos.items()
        }
    return float(np.mean(layer_scores)), layer_detail


def per_class_masking_summary(per_video: pd.DataFrame) -> pd.DataFrame:
    return (
        per_video.groupby(["strategy", "target_index"], as_index=False)
        .agg(
            videos=("video_id", "nunique"),
            mean_target_logit_drop=("target_logit_drop", "mean"),
            mean_margin_drop=("margin_drop", "mean"),
            prediction_change_rate=("prediction_changed", "mean"),
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    requested_strategies = [
        value.strip() for value in args.strategies.split(",") if value.strip()
    ]
    allowed = {
        "scalar_only",
        "flat_hybrid",
        "class_trajectory_pattern_only",
        "class_trajectory_hybrid_unweighted",
        "class_trajectory_hybrid_weighted",
    }
    unknown = set(requested_strategies) - allowed
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    # Build the exact calibration loader first. Its iteration order must match
    # the order used when cstc_probe_arrays.npz was generated.
    ensure_project_importable(Path(args.project_root))
    loader, selected_indices, chosen_classes = build_balanced_loader(
        project_root=Path(args.project_root),
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    labels, cache = collect_calibration_order(loader, args.ablation_videos)

    arrays = np.load(args.npz, allow_pickle=True)
    metrics = pd.read_csv(args.unit_metrics)
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
        keep = [idx for idx, layer in enumerate(layers) if pattern.search(layer)]
        layers = [layers[idx] for idx in keep]
        volume_keys = [volume_keys[idx] for idx in keep]

    affinity_device = resolve_device(args.affinity_device, args.device)
    print(f"Affinity computation device: {affinity_device}")
    print(f"Calibration labels in NPZ order: {labels.tolist()}")

    selection_records: List[Dict[str, object]] = []
    trace_frames: List[pd.DataFrame] = []
    trajectory_by_layer: Dict[str, ClassTrajectoryData] = {}
    flat_layer_data: Dict[str, LayerData] = {}

    for layer_pos, (layer_name, volume_key) in enumerate(
        zip(layers, volume_keys), start=1
    ):
        volumes = arrays[volume_key]
        if volumes.shape[0] != len(labels):
            raise ValueError(
                f"{layer_name}: NPZ has {volumes.shape[0]} videos, "
                f"but loader produced {len(labels)}. Use the same seed/classes/videos "
                "as the CSTC extraction run."
            )
        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        if volumes.shape[1] != len(layer_metrics):
            raise ValueError(
                f"{layer_name}: volume units={volumes.shape[1]}, "
                f"metric rows={len(layer_metrics)}"
            )

        print(f"\n[Upgrade selection {layer_pos}/{len(layers)}] {layer_name}")
        print(f"  contribution volume: {volumes.shape}")

        flat_patterns, flat_valid = normalize_patterns(volumes)
        flat_affinity = full_functional_affinity(
            flat_patterns,
            flat_valid,
            args.affinity_chunk_size,
            compute_device=str(affinity_device),
            desc=f"{layer_name} flat affinity",
        )
        trajectory = build_class_trajectory_data(
            volumes,
            labels,
            args.affinity_chunk_size,
            affinity_device,
            layer_name,
        )
        trajectory_by_layer[layer_name] = trajectory

        unit_cost = infer_unit_cost(layer_metrics)
        flat_layer_data[layer_name] = LayerData(
            layer=layer_name,
            unit_type=str(layer_metrics["unit_type"].iloc[0]),
            metrics=layer_metrics,
            affinity=flat_affinity,
            unit_cost=unit_cost,
        )

        remove_count = max(
            1,
            min(
                len(layer_metrics) - 1,
                int(round(args.mask_ratio * len(layer_metrics))),
            ),
        )
        selected = select_layer_strategies(
            layer_name,
            layer_metrics,
            unit_cost,
            flat_affinity,
            trajectory,
            remove_count,
            args.candidate_multiplier,
            args.scalar_risk,
        )

        for strategy, (units, trace) in selected.items():
            if strategy not in requested_strategies:
                continue
            for unit in units:
                selection_records.append(
                    {
                        "strategy": strategy,
                        "layer": layer_name,
                        "unit_type": str(layer_metrics["unit_type"].iloc[0]),
                        "unit_index": int(unit),
                        "d_abs": float(layer_metrics.loc[unit, "d_abs"]),
                        "d_rel": float(layer_metrics.loc[unit, "d_rel"]),
                        "unit_cost": float(unit_cost[unit]),
                    }
                )
            trace = trace.copy()
            trace.insert(0, "layer", layer_name)
            trace.insert(0, "strategy", strategy)
            trace_frames.append(trace)

        # Save compact per-layer diagnostics.
        diagnostic_rows = []
        for class_pos, class_id in enumerate(trajectory.class_ids):
            upper = np.triu_indices(len(layer_metrics), k=1)
            diagnostic_rows.append(
                {
                    "layer": layer_name,
                    "class_id": int(class_id),
                    "temporal_similarity_mean": float(
                        trajectory.temporal_similarity[class_pos][upper].mean()
                    ),
                    "spatial_similarity_mean": float(
                        trajectory.spatial_similarity[class_pos][upper].mean()
                    ),
                    "joint_affinity_mean": float(
                        trajectory.affinities[class_pos][upper].mean()
                    ),
                    "valid_unit_rate": float(
                        trajectory.valid_units[class_pos].mean()
                    ),
                }
            )
        pd.DataFrame(diagnostic_rows).to_csv(
            output_dir / f"{layer_name.replace('.', '_')}_trajectory_diagnostics.csv",
            index=False,
        )

    selections_df = pd.DataFrame(selection_records)
    selections_df.to_csv(output_dir / "selected_units.csv", index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(
            output_dir / "greedy_selection_trace.csv", index=False
        )

    selections: Dict[str, Dict[str, List[int]]] = {}
    for strategy in requested_strategies:
        selections[strategy] = {}
        table = selections_df[selections_df["strategy"] == strategy]
        for layer_name, layer_table in table.groupby("layer"):
            selections[strategy][layer_name] = (
                layer_table["unit_index"].astype(int).tolist()
            )

    print("\nOffline upgraded selections complete. Loading model...")
    device = torch.device(args.device)
    model, model_metadata = load_model(args.adapter, args.checkpoint, device)
    model_specs = {spec.name: spec for spec in discover_unit_layers(model)}
    missing = sorted(set(layers) - set(model_specs))
    if missing:
        raise KeyError(f"Layers missing from model: {missing}")

    per_video_frames = []
    summary_rows = []

    for strategy in requested_strategies:
        frame = evaluate_strategy(
            model=model,
            layer_specs=model_specs,
            selections=selections[strategy],
            cache=cache,
            device=device,
            target_mode=args.target_mode,
        )
        frame.insert(0, "strategy", strategy)
        per_video_frames.append(frame)

        weighted_coverage, class_coverage = trajectory_strategy_coverage(
            trajectory_by_layer,
            selections[strategy],
            weighted=True,
        )
        unweighted_coverage, _ = trajectory_strategy_coverage(
            trajectory_by_layer,
            selections[strategy],
            weighted=False,
        )
        flat_coverage, _ = strategy_coverage(
            flat_layer_data, selections[strategy]
        )
        selected_table = selections_df[
            selections_df["strategy"] == strategy
        ]
        summary_rows.append(
            {
                "strategy": strategy,
                "masked_units": int(len(selected_table)),
                "masked_cost": float(selected_table["unit_cost"].sum()),
                "mean_target_logit_drop": float(
                    frame["target_logit_drop"].mean()
                ),
                "median_target_logit_drop": float(
                    frame["target_logit_drop"].median()
                ),
                "mean_margin_drop": float(frame["margin_drop"].mean()),
                "prediction_change_rate": float(
                    frame["prediction_changed"].mean()
                ),
                "flat_pattern_coverage": float(flat_coverage),
                "class_trajectory_coverage_unweighted": float(
                    unweighted_coverage
                ),
                "class_trajectory_coverage_weighted": float(
                    weighted_coverage
                ),
                "per_class_weighted_coverage_json": json.dumps(
                    class_coverage, ensure_ascii=False
                ),
            }
        )

    per_video = pd.concat(per_video_frames, ignore_index=True)
    per_video.to_csv(output_dir / "strategy_masking_per_video.csv", index=False)
    class_summary = per_class_masking_summary(per_video)
    class_summary.to_csv(output_dir / "strategy_per_class_comparison.csv", index=False)

    strategy_summary = pd.DataFrame(summary_rows)
    strategy_summary.to_csv(output_dir / "strategy_comparison.csv", index=False)

    plt.figure(figsize=(9.0, 4.8))
    plt.bar(
        strategy_summary["strategy"],
        strategy_summary["mean_target_logit_drop"],
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean target-logit drop")
    plt.title("Damage under equal masking budgets")
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_logit_drop.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.0, 4.8))
    plt.bar(
        strategy_summary["strategy"],
        strategy_summary["class_trajectory_coverage_weighted"],
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Class-balanced weighted trajectory coverage")
    plt.title("Retained class-conditioned functional coverage")
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_trajectory_coverage.png", dpi=220)
    plt.close()

    summary = {
        "model_metadata": model_metadata,
        "chosen_classes": chosen_classes,
        "calibration_labels_in_npz_order": labels.tolist(),
        "selected_clips": selected_indices,
        "definition": {
            "temporal": (
                "maximum overlap-penalized normalized cross-correlation "
                "over all temporal shifts"
            ),
            "spatial": "cosine similarity of class-conditioned spatial prototypes",
            "joint": "sqrt(temporal_similarity * spatial_similarity)",
            "coverage": (
                "class-balanced facility-location coverage; recommended version "
                "weights represented units by class-conditioned contribution mass"
            ),
        },
        "strategy_results": summary_rows,
        "decision_rule": [
            "The weighted trajectory hybrid should improve class-balanced weighted coverage over scalar-only.",
            "Its target-logit and margin drops should not exceed scalar-only consistently.",
            "Per-class damage and per-class coverage must be checked; a good global average cannot hide one damaged class.",
            "Pattern-only remains an ablation and is not recommended for direct pruning.",
        ],
        "limitations": [
            "The probe uses saved contribution fields from the calibration set.",
            "Temporal alignment is shift tolerant but does not model non-linear speed changes.",
            "Spatial prototypes aggregate positions within each class and do not perform object tracking.",
            "The probe masks units and does not measure post-finetuning accuracy.",
        ],
        "run_config": vars(args),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report = [
        "Class-Aware Trajectory-Weighted Coverage Probe",
        "=" * 92,
        "",
        "Upgrades over the previous flat coverage probe:",
        "1. temporal and spatial patterns are separated;",
        "2. temporal trajectories allow ordered shifts with overlap penalty;",
        "3. coverage is computed separately for each class;",
        "4. important contribution patterns receive larger coverage weights;",
        "5. no temporal/spatial fusion coefficient is introduced.",
        "",
    ]
    for row in summary_rows:
        report += [
            f"[{row['strategy']}]",
            f"mean_target_logit_drop: {row['mean_target_logit_drop']}",
            f"mean_margin_drop: {row['mean_margin_drop']}",
            f"prediction_change_rate: {row['prediction_change_rate']}",
            f"flat_pattern_coverage: {row['flat_pattern_coverage']}",
            (
                "class_trajectory_coverage_unweighted: "
                f"{row['class_trajectory_coverage_unweighted']}"
            ),
            (
                "class_trajectory_coverage_weighted: "
                f"{row['class_trajectory_coverage_weighted']}"
            ),
            "",
        ]
    report += [
        "Primary comparison:",
        "- scalar_only vs flat_hybrid: old coverage rule.",
        "- flat_hybrid vs class_trajectory_hybrid_unweighted: effect of class-aware trajectory representation.",
        "- unweighted vs weighted trajectory hybrid: effect of contribution-mass protection.",
        "- inspect strategy_per_class_comparison.csv before accepting any global improvement.",
    ]
    (output_dir / "TRAJECTORY_COVERAGE_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"\nUpgraded trajectory coverage probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

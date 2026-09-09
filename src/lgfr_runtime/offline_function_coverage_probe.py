#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Functional Coverage Probe.

Purpose
-------
This probe tests a narrower hypothesis than CFSP exchange:

    Starting from the scalar pruning result, can a small number of
    equal-budget local swaps improve the representation of the original
    Functional Contribution Fields?

The probe is fully offline. It does not load the model, mask units, physically
prune tensors, or train.

Scientific boundary
-------------------
The probe does NOT claim that functional similarity is causal redundancy.
Similarity is used only to measure how well the retained set represents the
observed Contribution Fields.

Coverage objective
------------------
For a layer with functional similarity matrix R in [0,1]^(U x U) and retained
mask K, define

    coverage(K) = mean_i max_{j in K} R[i,j]

and

    reconstruction_error(K) = 1 - coverage(K).

A coverage-repair swap:

1. restores one scalar-removed unit;
2. removes one scalar-boundary retained unit;
3. preserves the exact removal count;
4. chooses the swap that minimizes functional reconstruction error.

To keep the comparison controlled, replacement candidates are restricted to
the lowest-risk retained units near the scalar pruning boundary. The size of
that diagnostic boundary set is reported explicitly and is not presented as a
final method hyperparameter.

Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

EPS = 1e-12


@dataclass
class LayerField:
    layer: str
    unit_type: str
    metrics: pd.DataFrame
    class_ids: np.ndarray
    temporal: np.ndarray   # [C,U,T]
    spatial: np.ndarray    # [C,U,HW]
    valid: np.ndarray      # [C,U]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline equal-budget Functional Coverage probe"
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument(
        "--class_labels",
        required=True,
        help="Comma-separated labels in the exact NPZ video order",
    )
    parser.add_argument("--output_dir", default="./offline_coverage_probe")
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--scalar_risk",
        choices=("d_rel", "d_abs", "product", "sum"),
        default="d_rel",
    )
    parser.add_argument("--remove_ratio", type=float, default=0.10)
    parser.add_argument(
        "--max_swaps",
        type=int,
        default=10,
        help="Maximum local repair steps used to draw the diagnostic curve",
    )
    parser.add_argument(
        "--boundary_count",
        type=int,
        default=32,
        help=(
            "Number of lowest-risk retained units considered as replacement "
            "candidates at each swap. This is a probe control, not a frozen "
            "method hyperparameter."
        ),
    )
    parser.add_argument("--compute_device", default="auto")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_similarity_matrices", action="store_true")
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
        raise KeyError(f"unit metrics missing columns: {sorted(missing)}")

    columns = [
        column
        for column in ("stage", "block", "unit_type", "layer")
        if column in metrics.columns
    ]
    table = metrics[columns].drop_duplicates()
    if columns:
        table = table.sort_values(columns, kind="stable")
    return table["layer"].tolist()


def parse_labels(text: str, count: int) -> np.ndarray:
    labels = np.asarray(
        [int(value.strip()) for value in text.split(",")],
        dtype=np.int64,
    )
    if len(labels) != count:
        raise ValueError(
            f"class_labels has {len(labels)} values, but NPZ has {count} videos"
        )
    return labels


def scalar_risk(metrics: pd.DataFrame, mode: str) -> np.ndarray:
    d_abs = metrics["d_abs"].to_numpy(dtype=np.float64)
    d_rel = metrics["d_rel"].to_numpy(dtype=np.float64)
    if mode == "d_abs":
        return d_abs
    if mode == "d_rel":
        return d_rel
    if mode == "product":
        return d_abs * d_rel
    if mode == "sum":
        return d_abs + d_rel
    raise ValueError(mode)


def build_class_fields(
    volumes: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive = np.maximum(volumes.astype(np.float32), 0.0)
    class_ids = np.unique(labels)
    temporal_all = []
    spatial_all = []
    valid_all = []

    for class_id in class_ids:
        aggregate = positive[labels == class_id].sum(axis=0)  # [U,T,H,W]
        temporal_all.append(aggregate.sum(axis=(2, 3)))
        spatial_all.append(
            aggregate.sum(axis=1).reshape(aggregate.shape[0], -1)
        )
        valid_all.append(aggregate.sum(axis=(1, 2, 3)) > EPS)

    return (
        class_ids,
        np.stack(temporal_all, axis=0),
        np.stack(spatial_all, axis=0),
        np.stack(valid_all, axis=0),
    )


def row_normalize(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    valid = norm[:, 0] > EPS
    normalized = torch.zeros_like(tensor)
    normalized[valid] = tensor[valid] / norm[valid].clamp_min(EPS)
    return normalized, valid


def spatial_similarity(
    maps: np.ndarray,
    device: torch.device,
    chunk_size: int,
    description: str,
) -> np.ndarray:
    right = torch.from_numpy(maps).to(device=device, dtype=torch.float32)
    right, right_valid = row_normalize(right)
    units = maps.shape[0]
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(
        range(0, units, chunk_size),
        desc=description,
        ncols=105,
    ):
        end = min(start + chunk_size, units)
        left = torch.from_numpy(maps[start:end]).to(
            device=device,
            dtype=torch.float32,
        )
        left, left_valid = row_normalize(left)
        block = (left @ right.T).clamp_(0.0, 1.0)
        valid = left_valid[:, None] & right_valid[None, :]
        block = torch.where(valid, block, torch.zeros_like(block))
        output[start:end] = block.cpu().numpy()

    return output


def temporal_similarity(
    curves: np.ndarray,
    device: torch.device,
    chunk_size: int,
    description: str,
) -> np.ndarray:
    right_all = torch.from_numpy(curves).to(
        device=device,
        dtype=torch.float32,
    )
    units, time_bins = curves.shape
    output = np.zeros((units, units), dtype=np.float32)

    for start in tqdm(
        range(0, units, chunk_size),
        desc=description,
        ncols=105,
    ):
        end = min(start + chunk_size, units)
        left_all = torch.from_numpy(curves[start:end]).to(
            device=device,
            dtype=torch.float32,
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
            left_norm, left_valid = row_normalize(left)
            right_norm, right_valid = row_normalize(right)
            block = left_norm @ right_norm.T
            block *= float(overlap) / float(time_bins)
            valid = left_valid[:, None] & right_valid[None, :]
            block = torch.where(valid, block, torch.zeros_like(block))
            best = torch.maximum(best, block)

        output[start:end] = best.clamp_(0.0, 1.0).cpu().numpy()

    return output


def functional_similarity_matrix(
    field: LayerField,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    class_matrices = []

    for class_position, class_id in enumerate(field.class_ids):
        temporal = temporal_similarity(
            field.temporal[class_position],
            device,
            chunk_size,
            f"{field.layer} class {class_id} temporal",
        )
        spatial = spatial_similarity(
            field.spatial[class_position],
            device,
            chunk_size,
            f"{field.layer} class {class_id} spatial",
        )
        matrix = np.sqrt(np.clip(temporal * spatial, 0.0, 1.0))
        valid = field.valid[class_position]
        matrix[~valid, :] = 0.0
        matrix[:, ~valid] = 0.0
        np.fill_diagonal(matrix, valid.astype(np.float32))
        class_matrices.append(matrix.astype(np.float32))

    similarity = np.mean(np.stack(class_matrices, axis=0), axis=0)
    similarity = 0.5 * (similarity + similarity.T)
    return np.clip(similarity, 0.0, 1.0).astype(np.float32)


def coverage_values(
    similarity: np.ndarray,
    retained_mask: np.ndarray,
) -> np.ndarray:
    retained = np.flatnonzero(retained_mask)
    if len(retained) == 0:
        return np.zeros(similarity.shape[0], dtype=np.float64)
    return np.max(similarity[:, retained], axis=1).astype(np.float64)


def coverage_score(
    similarity: np.ndarray,
    retained_mask: np.ndarray,
) -> float:
    return float(np.mean(coverage_values(similarity, retained_mask)))


def classwise_coverage_score(
    class_similarity: Sequence[np.ndarray],
    retained_mask: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [coverage_score(matrix, retained_mask) for matrix in class_similarity],
        dtype=np.float64,
    )


def scalar_baseline_mask(
    risk: np.ndarray,
    remove_count: int,
) -> np.ndarray:
    order = np.argsort(risk, kind="stable")
    removed = np.zeros(len(risk), dtype=bool)
    removed[order[:remove_count]] = True
    return removed


def candidate_boundary(
    risk: np.ndarray,
    removed_mask: np.ndarray,
    boundary_count: int,
) -> np.ndarray:
    retained = np.flatnonzero(~removed_mask)
    order = retained[np.argsort(risk[retained], kind="stable")]
    return order[: min(boundary_count, len(order))]


def best_coverage_swap(
    similarity: np.ndarray,
    risk: np.ndarray,
    removed_mask: np.ndarray,
    boundary_count: int,
) -> Dict[str, float]:
    removed_units = np.flatnonzero(removed_mask)
    boundary_units = candidate_boundary(
        risk,
        removed_mask,
        boundary_count,
    )

    if len(removed_units) == 0 or len(boundary_units) == 0:
        raise RuntimeError("No valid swap candidates")

    baseline_keep = ~removed_mask
    baseline_coverage = coverage_score(similarity, baseline_keep)

    best = None

    # This is intentionally exhaustive over the small diagnostic boundary set.
    for restored in removed_units:
        keep_after_restore = baseline_keep.copy()
        keep_after_restore[restored] = True

        for replacement in boundary_units:
            if replacement == restored:
                continue
            final_keep = keep_after_restore.copy()
            final_keep[replacement] = False
            score = coverage_score(similarity, final_keep)
            risk_gap = float(risk[replacement] - risk[restored])

            candidate = {
                "restored_unit": int(restored),
                "replacement_removed_unit": int(replacement),
                "coverage_before": baseline_coverage,
                "coverage_after": score,
                "coverage_gain": score - baseline_coverage,
                "restored_scalar_risk": float(risk[restored]),
                "replacement_scalar_risk": float(risk[replacement]),
                "scalar_risk_gap": risk_gap,
            }

            # Lexicographic decision:
            # 1. maximize coverage;
            # 2. minimize scalar-risk increase;
            # 3. deterministic unit-index tie break.
            key = (
                candidate["coverage_after"],
                -candidate["scalar_risk_gap"],
                -candidate["restored_unit"],
                -candidate["replacement_removed_unit"],
            )
            if best is None or key > best["_key"]:
                candidate["_key"] = key
                best = candidate

    if best is None:
        raise RuntimeError("Failed to find a coverage swap")
    best.pop("_key")
    return best


def run_layer_probe(
    field: LayerField,
    similarity: np.ndarray,
    risk_mode: str,
    remove_ratio: float,
    max_swaps: int,
    boundary_count: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
]:
    risk = scalar_risk(field.metrics, risk_mode)
    units = len(risk)
    remove_count = max(
        1,
        min(units - 1, int(round(remove_ratio * units))),
    )

    scalar_removed = scalar_baseline_mask(risk, remove_count)
    repaired_removed = scalar_removed.copy()

    curve_rows = []
    swap_rows = []

    for step in range(max_swaps + 1):
        retained = ~repaired_removed
        score = coverage_score(similarity, retained)
        curve_rows.append(
            {
                "layer": field.layer,
                "unit_type": field.unit_type,
                "swap_step": step,
                "coverage": score,
                "functional_reconstruction_error": 1.0 - score,
                "removed_scalar_risk_sum": float(
                    risk[repaired_removed].sum()
                ),
                "selection_jaccard_vs_scalar": float(
                    np.logical_and(
                        scalar_removed,
                        repaired_removed,
                    ).sum()
                    /
                    max(
                        1,
                        np.logical_or(
                            scalar_removed,
                            repaired_removed,
                        ).sum(),
                    )
                ),
            }
        )

        if step == max_swaps:
            break

        proposal = best_coverage_swap(
            similarity,
            risk,
            repaired_removed,
            boundary_count,
        )

        # Stop if the best equal-budget boundary swap does not improve coverage.
        if proposal["coverage_gain"] <= EPS:
            break

        restored = int(proposal["restored_unit"])
        replacement = int(proposal["replacement_removed_unit"])
        repaired_removed[restored] = False
        repaired_removed[replacement] = True

        proposal.update(
            {
                "layer": field.layer,
                "unit_type": field.unit_type,
                "swap_step": step + 1,
                "boundary_count": int(boundary_count),
            }
        )
        swap_rows.append(proposal)

    scalar_keep = ~scalar_removed
    repaired_keep = ~repaired_removed

    summary = {
        "layer": field.layer,
        "unit_type": field.unit_type,
        "units": units,
        "remove_count": remove_count,
        "requested_max_swaps": max_swaps,
        "actual_swaps": len(swap_rows),
        "boundary_count": boundary_count,
        "scalar_coverage": coverage_score(similarity, scalar_keep),
        "repaired_coverage": coverage_score(similarity, repaired_keep),
        "coverage_gain": (
            coverage_score(similarity, repaired_keep)
            -
            coverage_score(similarity, scalar_keep)
        ),
        "scalar_reconstruction_error": (
            1.0 - coverage_score(similarity, scalar_keep)
        ),
        "repaired_reconstruction_error": (
            1.0 - coverage_score(similarity, repaired_keep)
        ),
        "scalar_removed_risk_sum": float(risk[scalar_removed].sum()),
        "repaired_removed_risk_sum": float(
            risk[repaired_removed].sum()
        ),
        "risk_sum_increase": float(
            risk[repaired_removed].sum()
            -
            risk[scalar_removed].sum()
        ),
        "selection_jaccard": float(
            np.logical_and(
                scalar_removed,
                repaired_removed,
            ).sum()
            /
            max(
                1,
                np.logical_or(
                    scalar_removed,
                    repaired_removed,
                ).sum(),
            )
        ),
    }

    return (
        scalar_removed,
        repaired_removed,
        pd.DataFrame(curve_rows),
        pd.DataFrame(swap_rows),
        summary,
    )


def selection_rows(
    method: str,
    layer: str,
    unit_type: str,
    removed_mask: np.ndarray,
    risk: np.ndarray,
) -> List[Dict[str, object]]:
    rows = []
    for unit in np.flatnonzero(removed_mask):
        rows.append(
            {
                "method": method,
                "layer": layer,
                "unit_type": unit_type,
                "unit_index": int(unit),
                "scalar_risk": float(risk[unit]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    if not 0.0 < args.remove_ratio < 1.0:
        raise ValueError("--remove_ratio must be in (0,1)")
    if args.max_swaps < 0:
        raise ValueError("--max_swaps must be nonnegative")
    if args.boundary_count <= 0:
        raise ValueError("--boundary_count must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.compute_device)
    print(f"Functional similarity device: {device}")

    metrics = pd.read_csv(args.unit_metrics)
    arrays = np.load(args.npz, allow_pickle=True)

    layers = ordered_layers(metrics)
    volume_keys = sorted(
        key
        for key in arrays.files
        if key.endswith("_contribution_volumes")
    )

    if len(layers) != len(volume_keys):
        raise ValueError(
            f"layer/key mismatch: {len(layers)} metric layers, "
            f"{len(volume_keys)} contribution arrays"
        )

    if args.layers != "all":
        pattern = re.compile(args.layers)
        selected = [
            index
            for index, layer in enumerate(layers)
            if pattern.search(layer)
        ]
        layers = [layers[index] for index in selected]
        volume_keys = [volume_keys[index] for index in selected]

    if not layers:
        raise ValueError("No layer matched --layers")

    labels = parse_labels(
        args.class_labels,
        arrays[volume_keys[0]].shape[0],
    )

    scalar_rows = []
    repaired_rows = []
    all_curve_frames = []
    all_swap_frames = []
    summaries = []
    manifest = []

    for position, (layer, key) in enumerate(
        zip(layers, volume_keys),
        start=1,
    ):
        volumes = arrays[key]
        layer_metrics = (
            metrics[metrics["layer"] == layer]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )

        if volumes.shape[0] != len(labels):
            raise ValueError(f"{layer}: video count mismatch")
        if volumes.shape[1] != len(layer_metrics):
            raise ValueError(
                f"{layer}: NPZ units={volumes.shape[1]}, "
                f"metrics rows={len(layer_metrics)}"
            )

        expected = np.arange(len(layer_metrics))
        actual = layer_metrics["unit_index"].to_numpy(dtype=int)
        if not np.array_equal(expected, actual):
            raise ValueError(
                f"{layer}: unit_index must be contiguous 0..U-1"
            )

        unit_type = str(layer_metrics["unit_type"].iloc[0])
        print(
            f"\n[{position}/{len(layers)}] {layer} "
            f"({unit_type}), volume={tuple(volumes.shape)}"
        )

        class_ids, temporal, spatial, valid = build_class_fields(
            volumes,
            labels,
        )
        field = LayerField(
            layer=layer,
            unit_type=unit_type,
            metrics=layer_metrics,
            class_ids=class_ids,
            temporal=temporal,
            spatial=spatial,
            valid=valid,
        )

        similarity = functional_similarity_matrix(
            field,
            device,
            args.chunk_size,
        )

        (
            scalar_removed,
            repaired_removed,
            curve,
            swaps,
            summary,
        ) = run_layer_probe(
            field=field,
            similarity=similarity,
            risk_mode=args.scalar_risk,
            remove_ratio=args.remove_ratio,
            max_swaps=args.max_swaps,
            boundary_count=args.boundary_count,
        )

        risk = scalar_risk(layer_metrics, args.scalar_risk)
        scalar_rows.extend(
            selection_rows(
                "scalar",
                layer,
                unit_type,
                scalar_removed,
                risk,
            )
        )
        repaired_rows.extend(
            selection_rows(
                "coverage_repair",
                layer,
                unit_type,
                repaired_removed,
                risk,
            )
        )
        all_curve_frames.append(curve)
        if not swaps.empty:
            all_swap_frames.append(swaps)
        summaries.append(summary)

        if args.save_similarity_matrices:
            safe = layer.replace(".", "_")
            np.save(
                output_dir / f"{safe}_functional_similarity.npy",
                similarity,
            )

        manifest.append(
            {
                "layer": layer,
                "npz_key": key,
                "unit_type": unit_type,
                "video_count": int(volumes.shape[0]),
                "unit_count": int(volumes.shape[1]),
                "field_shape": list(volumes.shape[2:]),
                "class_ids": class_ids.tolist(),
                "invalid_class_unit_count": int((~valid).sum()),
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    scalar_df = pd.DataFrame(scalar_rows)
    repaired_df = pd.DataFrame(repaired_rows)
    curve_df = pd.concat(all_curve_frames, ignore_index=True)
    swaps_df = (
        pd.concat(all_swap_frames, ignore_index=True)
        if all_swap_frames
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(summaries)

    scalar_df.to_csv(
        output_dir / "scalar_remove_units.csv",
        index=False,
    )
    repaired_df.to_csv(
        output_dir / "coverage_repair_remove_units.csv",
        index=False,
    )
    curve_df.to_csv(
        output_dir / "coverage_curve.csv",
        index=False,
    )
    swaps_df.to_csv(
        output_dir / "coverage_swap_log.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "coverage_layer_summary.csv",
        index=False,
    )

    weighted_scalar_error = float(
        np.average(
            summary_df["scalar_reconstruction_error"],
            weights=summary_df["units"],
        )
    )
    weighted_repaired_error = float(
        np.average(
            summary_df["repaired_reconstruction_error"],
            weights=summary_df["units"],
        )
    )

    global_summary = {
        "scientific_question": (
            "Can equal-budget local swaps improve Functional Contribution "
            "Field coverage relative to scalar pruning?"
        ),
        "interpretation_boundary": (
            "Coverage is observational field representation, not causal "
            "replaceability."
        ),
        "run_config": vars(args),
        "class_labels_in_npz_order": labels.tolist(),
        "layer_manifest": manifest,
        "scalar_removed_units": int(len(scalar_df)),
        "coverage_repair_removed_units": int(len(repaired_df)),
        "total_actual_swaps": int(
            summary_df["actual_swaps"].sum()
        ),
        "weighted_scalar_reconstruction_error": weighted_scalar_error,
        "weighted_repaired_reconstruction_error": weighted_repaired_error,
        "weighted_reconstruction_error_reduction": (
            weighted_scalar_error - weighted_repaired_error
        ),
        "total_scalar_removed_risk_sum": float(
            summary_df["scalar_removed_risk_sum"].sum()
        ),
        "total_repaired_removed_risk_sum": float(
            summary_df["repaired_removed_risk_sum"].sum()
        ),
        "decision_rule": {
            "continue_to_masking": (
                "Every changed layer has positive coverage gain, the weighted "
                "reconstruction error decreases, and the scalar-risk increase "
                "is small enough to support an equal-risk comparison."
            ),
            "stop": (
                "No positive equal-budget coverage swap is found, or coverage "
                "improvement requires a large scalar-risk increase."
            ),
        },
    }

    with open(
        output_dir / "coverage_probe_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(global_summary, handle, ensure_ascii=False, indent=2)

    report = [
        "Offline Functional Coverage Probe",
        "=" * 92,
        "",
        f"layers: {len(summary_df)}",
        f"scalar removed units: {len(scalar_df)}",
        f"coverage-repair removed units: {len(repaired_df)}",
        f"actual swaps: {global_summary['total_actual_swaps']}",
        "",
        (
            "weighted scalar reconstruction error: "
            f"{weighted_scalar_error:.8f}"
        ),
        (
            "weighted repaired reconstruction error: "
            f"{weighted_repaired_error:.8f}"
        ),
        (
            "error reduction: "
            f"{weighted_scalar_error - weighted_repaired_error:.8f}"
        ),
        (
            "scalar removed-risk sum: "
            f"{global_summary['total_scalar_removed_risk_sum']:.8f}"
        ),
        (
            "repaired removed-risk sum: "
            f"{global_summary['total_repaired_removed_risk_sum']:.8f}"
        ),
        "",
        "Interpretation:",
        "- Positive offline coverage gain only proves that the retained set",
        "  represents the observed Contribution Fields better.",
        "- It does not prove lower masking damage or better pruning accuracy.",
        "- Proceed next to equal-budget masking only when coverage improves",
        "  without a large scalar-risk penalty.",
    ]

    (
        output_dir / "OFFLINE_COVERAGE_REPORT.txt"
    ).write_text("\n".join(report), encoding="utf-8")

    if len(scalar_df) != len(repaired_df):
        raise AssertionError("Equal-budget constraint was violated")

    print(f"\nCoverage probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

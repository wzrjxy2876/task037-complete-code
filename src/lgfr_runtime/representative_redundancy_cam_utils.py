"""Numerical helpers for function-representative CAM validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EPS = 1e-8


def parse_unit_id(unit_id: str) -> tuple[str, str, int]:
    parts = str(unit_id).rsplit("::", 2)
    if len(parts) != 3 or parts[1] not in {"head", "neuron"}:
        raise ValueError(f"Invalid stable unit id: {unit_id!r}")
    return parts[0], parts[1], int(parts[2])


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def normalize_unit_maps(maps: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Min-max normalize each response map in ``[U,T,H,W]`` independently."""
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("unit response maps must have shape [U,T,H,W]")
    flat = values.reshape(values.shape[0], -1)
    low = flat.min(axis=1, keepdims=True)
    span = flat.max(axis=1, keepdims=True) - low
    normalized = np.divide(
        flat - low,
        span,
        out=np.zeros_like(flat),
        where=span > eps,
    )
    return normalized.reshape(values.shape).astype(np.float32)


def cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
    eps: float = EPS,
) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= eps:
        return 1.0 if np.allclose(left, right, atol=eps) else 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def soft_iou(left: np.ndarray, right: np.ndarray, eps: float = EPS) -> float:
    left = np.maximum(np.asarray(left, dtype=np.float64).reshape(-1), 0.0)
    right = np.maximum(np.asarray(right, dtype=np.float64).reshape(-1), 0.0)
    union = float(np.maximum(left, right).sum())
    if union <= eps:
        return 1.0
    return float(np.minimum(left, right).sum() / union)


def hotspot_centroid(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if values.ndim != 3:
        raise ValueError("a response map must have shape [T,H,W]")
    total = float(values.sum())
    if total <= eps:
        return np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
    axes = [
        np.linspace(0.0, 1.0, size, dtype=np.float64)
        for size in values.shape
    ]
    return np.asarray(
        [
            float((values * axes[0][:, None, None]).sum() / total),
            float((values * axes[1][None, :, None]).sum() / total),
            float((values * axes[2][None, None, :]).sum() / total),
        ]
    )


def representative_cluster_metrics(
    representative_map: np.ndarray,
    member_maps: np.ndarray,
) -> dict[str, object]:
    """Compare one representative ``[T,H,W]`` with members ``[K,T,H,W]``."""
    representative = np.asarray(representative_map, dtype=np.float32)
    members = np.asarray(member_maps, dtype=np.float32)
    if representative.ndim != 3 or members.ndim != 4:
        raise ValueError("expected representative [T,H,W] and members [K,T,H,W]")
    if members.shape[0] < 1 or members.shape[1:] != representative.shape:
        raise ValueError("member maps must be nonempty and shape-aligned")
    prototype = members.mean(axis=0, dtype=np.float64).astype(np.float32)
    member_cosines = [
        cosine_similarity(representative, member) for member in members
    ]
    member_ious = [soft_iou(representative, member) for member in members]
    centroid_distance = float(
        np.linalg.norm(
            hotspot_centroid(representative) - hotspot_centroid(prototype)
        )
    )
    return {
        "member_prototype": prototype,
        "representative_to_prototype_cosine": cosine_similarity(
            representative, prototype
        ),
        "representative_to_prototype_soft_iou": soft_iou(
            representative, prototype
        ),
        "representative_to_member_cosines": member_cosines,
        "representative_to_member_soft_ious": member_ious,
        "mean_member_cosine": float(np.mean(member_cosines)),
        "mean_member_soft_iou": float(np.mean(member_ious)),
        "hotspot_centroid_distance": centroid_distance,
    }


def select_representative_clusters(
    cluster_rows: Sequence[Mapping[str, object]],
    num_clusters: int,
    manual_cluster_ids: Sequence[int] = (),
) -> list[dict[str, object]]:
    """Select non-singleton descriptor clusters for representative validation."""
    rows = [dict(row) for row in cluster_rows]
    by_id = {int(float(row["cluster_id"])): row for row in rows}
    if manual_cluster_ids:
        missing = [cluster_id for cluster_id in manual_cluster_ids if cluster_id not in by_id]
        if missing:
            raise KeyError(f"cluster IDs are missing from export: {missing}")
        selected = [by_id[int(cluster_id)] for cluster_id in manual_cluster_ids]
    else:
        eligible = [
            row for row in rows if int(float(row.get("size", 0))) >= 2
        ]
        selected = sorted(
            eligible,
            key=lambda row: (
                -int(float(row.get("size", 0))),
                -float(row.get("mean_uniqueness", 0.0)),
                int(float(row["cluster_id"])),
            ),
        )[: int(num_clusters)]
    if not selected:
        raise ValueError("no descriptor cluster has members to compare")
    return selected


def select_cluster_members(
    unit_rows: Sequence[Mapping[str, object]],
    cluster_id: int,
    max_members: int,
) -> dict[str, object]:
    """Select a representative, descriptor-cluster members, and removed units."""
    rows = [
        dict(row)
        for row in unit_rows
        if int(float(row["cluster_id"])) == int(cluster_id)
    ]
    protected = [
        row for row in rows if str(row.get("protected", "")).lower() == "true"
    ]
    if len(protected) != 1:
        raise ValueError(
            f"cluster {cluster_id} must contain exactly one protected unit"
        )
    candidates = sorted(
        [row for row in rows if row not in protected],
        key=lambda row: int(float(row.get("unit_index", 0))),
    )
    selected = candidates[: min(int(max_members), len(candidates))]
    if not selected:
        raise ValueError(f"cluster {cluster_id} has no member to compare")
    removed = [
        row
        for row in candidates
        if str(row.get("kept", "")).strip().lower() in {"false", "0"}
    ][: int(max_members)]
    return {
        "cluster_id": int(cluster_id),
        "representative": protected[0],
        "cluster_members": selected,
        "removed_members": removed,
        "cluster_member_count": len(candidates),
    }

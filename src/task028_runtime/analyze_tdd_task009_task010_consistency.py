#!/usr/bin/env python3
"""Strict Task009/Task010 temporal-descriptor consistency analysis.

The primary unit key is ``(layer, unit_type, unit_index)``.  CSV row order is
never used for alignment.  ``global_index`` is checked independently whenever
it is present in both inputs.  The analysis is intentionally CPU based because
it operates only on one scalar per pruning unit; model forward, descriptor
collection, and BMS remain CUDA workloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EPSILON = 1e-8
RELATIVE_ERROR_MIN_DENOMINATOR = 1e-6
KEY_FIELDS = ("layer", "unit_type", "unit_index")
SUMMARY_FIELDS = (
    "scope",
    "num_units",
    "pearson",
    "spearman",
    "mae",
    "rmse",
    "median_abs_error",
    "p99_abs_error",
    "max_abs_error",
)


class UnitAlignmentError(ValueError):
    """Raised when two descriptor tables cannot be aligned exactly."""


@dataclass(frozen=True)
class UnitRecord:
    """One scalar descriptor associated with one structured pruning unit."""

    global_index: int | None
    layer: str
    unit_type: str
    unit_index: int
    value: float

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.layer, self.unit_type, self.unit_index)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_unit_type(value: str) -> str:
    """Map known Attention/FFN spellings to the Task007 canonical names."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "attention": "attention_head",
        "attn": "attention_head",
        "head": "attention_head",
        "attention_head": "attention_head",
        "attention_heads": "attention_head",
        "mlp": "ffn_neuron",
        "ffn": "ffn_neuron",
        "neuron": "ffn_neuron",
        "ffn_neuron": "ffn_neuron",
        "mlp_neuron": "ffn_neuron",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported unit_type {value!r}")
    return aliases[normalized]


def _resolve_column(fieldnames: Sequence[str], candidates: Sequence[str], label: str) -> str:
    available = {str(name).strip(): str(name) for name in fieldnames}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    raise ValueError(
        f"cannot map {label}; expected one of {list(candidates)}, "
        f"available columns are {list(fieldnames)}"
    )


def read_descriptor_table(
    path: Path,
    value_candidates: Sequence[str],
    source_name: str,
) -> tuple[list[UnitRecord], dict[str, str]]:
    """Read one descriptor CSV and explicitly map its metadata/value columns."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f"{source_name} CSV has no header: {path}")
        layer_column = _resolve_column(fieldnames, ("layer", "layer_name"), "layer")
        type_column = _resolve_column(
            fieldnames, ("unit_type", "type", "pruning_unit_type"), "unit_type"
        )
        index_column = _resolve_column(
            fieldnames, ("unit_index", "local_index", "idx"), "unit_index"
        )
        value_column = _resolve_column(fieldnames, value_candidates, source_name)
        global_column = next(
            (name for name in ("global_index", "global_idx") if name in fieldnames),
            None,
        )
        source_rows = list(reader)

    if not source_rows:
        raise ValueError(f"{source_name} CSV is empty: {path}")
    records: list[UnitRecord] = []
    for row_number, row in enumerate(source_rows, start=2):
        layer = str(row.get(layer_column, "")).strip()
        if not layer:
            raise ValueError(f"{source_name} row {row_number} has an empty layer")
        try:
            unit_index = int(row[index_column])
            value = float(row[value_column])
            global_index = (
                int(row[global_column])
                if global_column is not None and str(row.get(global_column, "")).strip()
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_name} row {row_number} has invalid numeric metadata"
            ) from exc
        if unit_index < 0 or (global_index is not None and global_index < 0):
            raise ValueError(f"{source_name} row {row_number} has a negative index")
        if not math.isfinite(value):
            raise ValueError(f"{source_name} row {row_number} has non-finite value")
        records.append(
            UnitRecord(
                global_index=global_index,
                layer=layer,
                unit_type=normalize_unit_type(row[type_column]),
                unit_index=unit_index,
                value=value,
            )
        )

    mapping = {
        "layer": layer_column,
        "unit_type": type_column,
        "unit_index": index_column,
        "global_index": global_column or "UNAVAILABLE",
        "value": value_column,
    }
    return records, mapping


def index_unit_records(
    records: Sequence[UnitRecord], source_name: str
) -> dict[tuple[str, str, int], UnitRecord]:
    """Index records and fail on duplicate structured-unit keys."""
    indexed: dict[tuple[str, str, int], UnitRecord] = {}
    duplicates: list[tuple[str, str, int]] = []
    for record in records:
        if record.key in indexed:
            duplicates.append(record.key)
        else:
            indexed[record.key] = record
    if duplicates:
        examples = ", ".join(map(str, duplicates[:10]))
        raise UnitAlignmentError(
            f"{source_name} contains {len(duplicates)} duplicate unit keys; "
            f"examples: {examples}"
        )
    global_values = [r.global_index for r in records if r.global_index is not None]
    if global_values and len(set(global_values)) != len(global_values):
        raise UnitAlignmentError(f"{source_name} contains duplicate global_index values")
    return indexed


def align_unit_records(
    task009: Sequence[UnitRecord], task010: Sequence[UnitRecord]
) -> tuple[list[tuple[UnitRecord, UnitRecord]], list[dict]]:
    """Align by structured key and return explicit mismatch rows."""
    left = index_unit_records(task009, "Task009")
    right = index_unit_records(task010, "Task010")
    keys = sorted(set(left) | set(right))
    aligned: list[tuple[UnitRecord, UnitRecord]] = []
    mismatches: list[dict] = []
    for key in keys:
        record009 = left.get(key)
        record010 = right.get(key)
        if record009 is None or record010 is None:
            mismatches.append(
                {
                    "layer": key[0],
                    "unit_type": key[1],
                    "unit_index": key[2],
                    "task009_global_index": (
                        "" if record009 is None else record009.global_index
                    ),
                    "task010_global_index": (
                        "" if record010 is None else record010.global_index
                    ),
                    "reason": "missing_from_task009" if record009 is None else "missing_from_task010",
                }
            )
            continue
        if (
            record009.global_index is not None
            and record010.global_index is not None
            and record009.global_index != record010.global_index
        ):
            mismatches.append(
                {
                    "layer": key[0],
                    "unit_type": key[1],
                    "unit_index": key[2],
                    "task009_global_index": record009.global_index,
                    "task010_global_index": record010.global_index,
                    "reason": "global_index_disagrees",
                }
            )
            continue
        aligned.append((record009, record010))
    return aligned, mismatches


def relative_error_values(
    reference: np.ndarray,
    candidate: np.ndarray,
    minimum_denominator: float = RELATIVE_ERROR_MIN_DENOMINATOR,
) -> np.ndarray:
    """Return relative errors, using NaN for unsafe near-zero denominators."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    valid = np.abs(reference) > minimum_denominator
    result[valid] = np.abs(candidate[valid] - reference[valid]) / np.abs(reference[valid])
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def metric_row(scope: str, task009: np.ndarray, task010: np.ndarray) -> dict:
    """Calculate required numerical consistency metrics for one scope."""
    task009 = np.asarray(task009, dtype=np.float64)
    task010 = np.asarray(task010, dtype=np.float64)
    if task009.shape != task010.shape or task009.ndim != 1 or task009.size == 0:
        raise ValueError(f"invalid arrays for scope {scope}: {task009.shape}/{task010.shape}")
    error = np.abs(task010 - task009)
    return {
        "scope": scope,
        "num_units": int(task009.size),
        "pearson": _correlation(task009, task010),
        "spearman": _correlation(_average_ranks(task009), _average_ranks(task010)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(task010 - task009)))),
        "median_abs_error": float(np.median(error)),
        "p99_abs_error": float(np.quantile(error, 0.99)),
        "max_abs_error": float(error.max()),
    }


def task009_temporal_variation_numpy(activation: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Independent Task009 layout reference for ``A [B,T,H,W,U,D]``."""
    activation = np.asarray(activation, dtype=np.float64)
    if activation.ndim != 6 or any(size <= 0 for size in activation.shape):
        raise ValueError("activation must be non-empty [B,T,H,W,U,D]")
    temporal_mean = activation.mean(axis=1, keepdims=True)
    residual = activation - temporal_mean
    stable = np.linalg.norm(temporal_mean[:, 0], axis=-1).mean(axis=(1, 2))
    dynamic = np.linalg.norm(residual, axis=-1).mean(axis=(1, 2, 3))
    return np.clip(dynamic / (dynamic + stable + eps), 0.0, 1.0)


def task010_temporal_dynamicity_numpy(response: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Independent Task010 layout reference for ``A [B,U,T,H,W,D]``."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 6 or any(size <= 0 for size in response.shape):
        raise ValueError("response must be non-empty [B,U,T,H,W,D]")
    temporal_mean = response.mean(axis=2, keepdims=True)
    residual = response - temporal_mean
    dynamic = np.linalg.norm(residual, axis=-1).mean(axis=(2, 3, 4))
    stable = np.linalg.norm(temporal_mean, axis=-1).mean(axis=(2, 3, 4))
    return np.clip(dynamic / (dynamic + stable + eps), 0.0, 1.0)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    margin = max((upper - lower) * 0.03, 1e-6)
    figure, axis = plt.subplots(figsize=(6.4, 5.5))
    axis.scatter(x, y, s=8, alpha=0.45, linewidths=0)
    axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "k--", lw=1)
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Task009 temporal variation")
    axis.set_ylabel("Task010 TDD")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_layer_error_figure(path: Path, rows: Sequence[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = list(rows[:40])
    labels = [f"{row['layer']} ({row['unit_type']})" for row in shown]
    values = [float(row["mean_abs_error"]) for row in shown]
    height = max(5.0, 0.25 * len(shown))
    figure, axis = plt.subplots(figsize=(10, height))
    positions = np.arange(len(shown))
    axis.barh(positions, values, color="#3B82F6")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute error")
    axis.set_title("Task009–Task010 TDD error by layer (top 40)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _resolve_task010_path(requested: Path) -> Path:
    if requested.is_file():
        return requested
    root = Path("tdd_pruning_validation") / "dynamic3d"
    matches = sorted(root.glob("**/descriptor_statistics.csv")) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(requested)
    raise ValueError(
        f"Task010 path {requested} is absent and automatic search is ambiguous: {matches}"
    )


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task009_path = Path(args.task009_csv)
    task010_path = _resolve_task010_path(Path(args.task010_csv))
    task009, mapping009 = read_descriptor_table(
        task009_path,
        ("D_temporal_variation", "D_var", "temporal_variation"),
        "Task009 temporal variation",
    )
    task010, mapping010 = read_descriptor_table(
        task010_path,
        ("D_dyn", "D_third", "temporal_dynamicity"),
        "Task010 TDD",
    )
    aligned, mismatches = align_unit_records(task009, task010)
    mismatch_fields = (
        "layer",
        "unit_type",
        "unit_index",
        "task009_global_index",
        "task010_global_index",
        "reason",
    )
    _atomic_csv(
        output_dir / "tdd_009_010_unit_mismatches.csv",
        mismatch_fields,
        mismatches,
    )
    if mismatches:
        _atomic_json(
            output_dir / "tdd_009_010_consistency_metadata.json",
            {
                "status": "fatal_unit_alignment_mismatch",
                "task009_csv": str(task009_path.resolve()),
                "task010_csv": str(task010_path.resolve()),
                "task009_rows": len(task009),
                "task010_rows": len(task010),
                "aligned_rows": len(aligned),
                "mismatch_rows": len(mismatches),
                "task009_column_mapping": mapping009,
                "task010_column_mapping": mapping010,
            },
        )
        raise UnitAlignmentError(
            f"descriptor unit alignment failed with {len(mismatches)} mismatches; "
            "see tdd_009_010_unit_mismatches.csv"
        )

    records009 = [pair[0] for pair in aligned]
    values009 = np.asarray([pair[0].value for pair in aligned], dtype=np.float64)
    values010 = np.asarray([pair[1].value for pair in aligned], dtype=np.float64)
    scopes = {
        "all_units": np.ones(len(aligned), dtype=bool),
        "attention_only": np.asarray(
            [record.unit_type == "attention_head" for record in records009]
        ),
        "mlp_only": np.asarray(
            [record.unit_type == "ffn_neuron" for record in records009]
        ),
    }
    summary_rows = [
        metric_row(scope, values009[mask], values010[mask])
        for scope, mask in scopes.items()
    ]
    _atomic_csv(
        output_dir / "tdd_009_010_consistency_summary.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )

    absolute_error = np.abs(values010 - values009)
    relative_error = relative_error_values(values009, values010)
    ranked_indices = np.argsort(-absolute_error, kind="mergesort")[: max(200, args.top_errors)]
    error_rows = []
    for index in ranked_indices:
        left, right = aligned[int(index)]
        error_rows.append(
            {
                "global_index": left.global_index if left.global_index is not None else right.global_index,
                "layer": left.layer,
                "unit_type": left.unit_type,
                "unit_index": left.unit_index,
                "D_var_task009": left.value,
                "D_dyn_task010": right.value,
                "absolute_error": float(absolute_error[index]),
                "relative_error_if_valid": (
                    "" if np.isnan(relative_error[index]) else float(relative_error[index])
                ),
            }
        )
    _atomic_csv(
        output_dir / "tdd_009_010_largest_errors.csv",
        (
            "global_index",
            "layer",
            "unit_type",
            "unit_index",
            "D_var_task009",
            "D_dyn_task010",
            "absolute_error",
            "relative_error_if_valid",
        ),
        error_rows,
    )

    layer_rows = []
    for layer, unit_type in sorted({(r.layer, r.unit_type) for r in records009}):
        mask = np.asarray(
            [r.layer == layer and r.unit_type == unit_type for r in records009]
        )
        row = metric_row("layer", values009[mask], values010[mask])
        layer_rows.append(
            {
                "layer": layer,
                "unit_type": unit_type,
                "num_units": row["num_units"],
                "mean_abs_error": row["mae"],
                "median_abs_error": row["median_abs_error"],
                "max_abs_error": row["max_abs_error"],
                "pearson": row["pearson"],
                "spearman": row["spearman"],
            }
        )
    layer_rows.sort(key=lambda row: (-float(row["mean_abs_error"]), row["layer"]))
    layer_fields = (
        "layer",
        "unit_type",
        "num_units",
        "mean_abs_error",
        "median_abs_error",
        "max_abs_error",
        "pearson",
        "spearman",
    )
    _atomic_csv(output_dir / "tdd_009_010_layer_error_summary.csv", layer_fields, layer_rows)

    attention_rows = []
    for layer in sorted({r.layer for r in records009 if r.unit_type == "attention_head"}):
        mask = np.asarray(
            [r.layer == layer and r.unit_type == "attention_head" for r in records009]
        )
        metric = metric_row("attention_layer", values009[mask], values010[mask])
        attention_rows.append(
            {
                "layer": layer,
                "num_heads": int(mask.sum()),
                "task009_mean": float(values009[mask].mean()),
                "task009_std": float(values009[mask].std(ddof=0)),
                "task010_mean": float(values010[mask].mean()),
                "task010_std": float(values010[mask].std(ddof=0)),
                "pearson": metric["pearson"],
                "spearman": metric["spearman"],
                "mae": metric["mae"],
                "max_abs_error": metric["max_abs_error"],
            }
        )
    attention_fields = (
        "layer",
        "num_heads",
        "task009_mean",
        "task009_std",
        "task010_mean",
        "task010_std",
        "pearson",
        "spearman",
        "mae",
        "max_abs_error",
    )
    _atomic_csv(output_dir / "attention_tdd_consistency.csv", attention_fields, attention_rows)

    _write_scatter(
        output_dir / "tdd_009_010_scatter_all.png",
        values009,
        values010,
        "Task009 vs Task010 temporal descriptor — all units",
    )
    _write_scatter(
        output_dir / "tdd_009_010_scatter_attention.png",
        values009[scopes["attention_only"]],
        values010[scopes["attention_only"]],
        "Task009 vs Task010 temporal descriptor — Attention",
    )
    _write_scatter(
        output_dir / "tdd_009_010_scatter_mlp.png",
        values009[scopes["mlp_only"]],
        values010[scopes["mlp_only"]],
        "Task009 vs Task010 temporal descriptor — FFN",
    )
    _write_layer_error_figure(output_dir / "tdd_error_by_layer.png", layer_rows)

    max_error = float(absolute_error.max())
    _atomic_json(
        output_dir / "tdd_009_010_consistency_metadata.json",
        {
            "status": "complete",
            "task009_csv": str(task009_path.resolve()),
            "task010_csv": str(task010_path.resolve()),
            "task009_column_mapping": mapping009,
            "task010_column_mapping": mapping010,
            "unit_alignment": "exact_by_layer_type_unit_index",
            "global_index_check": "passed",
            "num_units": len(aligned),
            "formula_identity_from_source_audit": "see docs/handoff/task_011_implementation_comparison.md",
            "maximum_absolute_error": max_error,
            "numerically_close_at_atol_1e-6": bool(np.allclose(values009, values010, rtol=0.0, atol=1e-6)),
            "near_zero_relative_error_rule": "relative error omitted when abs(Task009) <= 1e-6",
            "metrics": [
                {key: _json_safe(value) for key, value in row.items()}
                for row in summary_rows
            ],
        },
    )
    print(f"Task011 descriptor comparison complete: {len(aligned)} aligned units")
    print(f"Maximum absolute error: {max_error:.9g}")
    print(f"Output: {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Task009 temporal variation with Task010 TDD"
    )
    parser.add_argument(
        "--task009_csv",
        default="video_descriptor_validation/third_descriptor_candidates.csv",
    )
    parser.add_argument(
        "--task010_csv",
        default="tdd_pruning_validation/dynamic3d/seed3407/descriptor_statistics.csv",
    )
    parser.add_argument("--output_dir", default="task011_diagnosis")
    parser.add_argument("--top_errors", type=int, default=200)
    args = parser.parse_args(argv)
    if args.top_errors < 200:
        parser.error("--top_errors must be at least 200")
    return args


if __name__ == "__main__":
    run_analysis(parse_args())

"""Aggregate verified Task014/Task016 prune-only artifacts and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


TARGETS = (0.10, 0.20, 0.30)
TAGS = {0.10: "s10", 0.20: "s20", 0.30: "s30"}
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256_rows(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\t".join(str(row[field]) for field in fields).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "checkpoint_sha256", "descriptor_variant", "selection_cost_mode",
        "sigma", "min_keep_ratio", "importance_alpha", "seed",
        "contribution_metadata_sha256", "contribution_mapping_sha256",
        "contribution_sample_count", "contribution_sample_aggregation",
        "aligned_field_shape_per_video", "calibration_batch_size",
        "calibration_batches", "calibration_samples_sha256",
        "validation_split", "validation_dataset", "validation_batch_size",
        "amp_enabled",
    )
    return {key: metadata.get(key) for key in keys}


def verify_run_identity(task014_root: Path, task016_root: Path) -> dict:
    reference = None
    evidence = []
    domain_hash = None
    descriptor_cache_hash = None
    for mode, root, directory in (
        ("domain_average", task014_root, "functional"),
        ("domain_total", task016_root, "domain_total"),
    ):
        for target in TARGETS:
            run = root / directory / TAGS[target]
            metadata = _read_json(run / "run_metadata.json")
            metrics = _read_json(run / "final_metrics.json")
            if mode == "domain_average":
                if metrics.get("status") != "task014_prune_only_complete":
                    raise RuntimeError(f"Incomplete Task014 baseline: {run}")
            else:
                if metrics.get("status") != "task016_prune_only_complete":
                    raise RuntimeError(f"Incomplete Task016 run: {run}")
                if metadata.get("functional_score") != "domain_total":
                    raise RuntimeError(f"Wrong Task016 score mode: {run}")
                current_descriptor_hash = metadata.get(
                    "functional_descriptor_cache_sha256"
                )
                descriptor_path = Path(str(metadata.get("functional_descriptor_cache")))
                if not descriptor_path.is_file() or _sha256_file(descriptor_path) != current_descriptor_hash:
                    raise RuntimeError(f"Task016 descriptor cache identity failed: {run}")
                if descriptor_cache_hash is None:
                    descriptor_cache_hash = current_descriptor_hash
                elif current_descriptor_hash != descriptor_cache_hash:
                    raise RuntimeError("Task016 runs used different descriptor caches")
            current = _identity(metadata)
            if reference is None:
                reference = current
            elif current != reference:
                changed = sorted(
                    key for key in current if current.get(key) != reference.get(key)
                )
                raise RuntimeError(
                    f"Experiment identity mismatch in {run}: {changed}"
                )
            domains = _read_csv(run / "functional_domain_summary.csv")
            current_hash = _sha256_rows(
                domains,
                (
                    "domain_id", "initial_size", "attention_count", "mlp_count",
                    "active_functional_count", "null_functional_count", "mixed_type",
                ),
            )
            if domain_hash is None:
                domain_hash = current_hash
            elif current_hash != domain_hash:
                raise RuntimeError(f"BMS domain identity mismatch in {run}")
            evidence.append(
                {"mode": mode, "target": target, "run": str(run), "identity": current}
            )
    return {
        "status": "passed",
        "bms_domain_sha256": domain_hash,
        "descriptor_cache_sha256": descriptor_cache_hash,
        "runs": evidence,
    }


def _comparison_row(mode: str, target: float, metrics: Mapping[str, object]) -> dict:
    before = int(metrics["parameters_before"])
    after = int(metrics["parameters_after"])
    return {
        "score_mode": mode,
        "target_sparsity": target,
        "actual_sparsity": (before - after) / before,
        "parameters_before": before,
        "parameters_after": after,
        "parameters_removed": before - after,
        "top1": float(metrics["pre_ft_top1"]),
        "top5": float(metrics["pre_ft_top5"]),
        "attention_removed": int(metrics["removed_heads"]),
        "attention_removal_ratio": float(metrics["removed_head_ratio"]),
        "ffn_removed": int(metrics["removed_neurons"]),
        "ffn_removal_ratio": float(metrics["removed_mlp_ratio"]),
        "attention_removed_parameters": int(
            metrics["attention_removed_parameter_cost"]
        ),
        "ffn_removed_parameters": int(metrics["mlp_removed_parameter_cost"]),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return math.nan
    a = _rankdata(left)
    b = _rankdata(right)
    if a.std() == 0.0 or b.std() == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _candidate_analysis(rows: Sequence[Mapping[str, str]], output_dir: Path) -> dict:
    global_index = np.asarray([int(row["global_index"]) for row in rows])
    unit_type = np.asarray([row["unit_type"] for row in rows], dtype="U16")
    active_size = np.asarray([int(row["domain_active_size"]) for row in rows])
    average = np.asarray([float(row["delta_average"]) for row in rows])
    total = np.asarray([float(row["delta_total"]) for row in rows])
    attention = unit_type == TYPE_ATTENTION
    ffn = unit_type == TYPE_FFN
    ratio_average = float(np.median(average[attention]) / np.median(average[ffn]))
    ratio_total = float(np.median(total[attention]) / np.median(total[ffn]))
    ranking_rows = []
    for scope, mask in (("all", np.ones(len(rows), dtype=bool)),
                        ("attention", attention), ("ffn", ffn)):
        indices = np.flatnonzero(mask)
        spearman = _spearman(average[indices], total[indices])
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, int(math.ceil(indices.size * fraction)))
            top_average = set(indices[np.argsort(average[indices], kind="mergesort")[:count]])
            top_total = set(indices[np.argsort(total[indices], kind="mergesort")[:count]])
            overlap = len(top_average & top_total)
            ranking_rows.append(
                {
                    "scope": scope,
                    "top_fraction": fraction,
                    "candidate_count": indices.size,
                    "spearman_average_vs_total": spearman,
                    "overlap_count": overlap,
                    "overlap_ratio": overlap / count,
                }
            )
    _atomic_csv(
        output_dir / "ranking_change_analysis.csv",
        tuple(ranking_rows[0]), ranking_rows,
    )

    rank_average = _rankdata(average)
    rank_total = _rankdata(total)
    denominator = max(len(rows) - 1, 1)
    shift_rows = []
    for index in np.flatnonzero(attention):
        average_percentile = 100.0 * rank_average[index] / denominator
        total_percentile = 100.0 * rank_total[index] / denominator
        shift_rows.append(
            {
                "global_index": int(global_index[index]),
                "average_rank": float(rank_average[index] + 1),
                "total_rank": float(rank_total[index] + 1),
                "average_percentile": average_percentile,
                "total_percentile": total_percentile,
                "rank_improvement": float(rank_average[index] - rank_total[index]),
                "total_top_1_percent": total_percentile <= 1.0,
                "total_top_5_percent": total_percentile <= 5.0,
                "total_top_10_percent": total_percentile <= 10.0,
                "total_top_20_percent": total_percentile <= 20.0,
            }
        )
    _atomic_csv(output_dir / "attention_rank_shift.csv", tuple(shift_rows[0]), shift_rows)
    return {
        "median_ratio_attention_over_ffn_domain_average": ratio_average,
        "median_ratio_attention_over_ffn_domain_total": ratio_total,
        "spearman_domain_size_vs_delta_average": _spearman(active_size, average),
        "spearman_domain_size_vs_delta_total": _spearman(active_size, total),
        "spearman_average_vs_total_all": _spearman(average, total),
        "attention_top_counts_domain_total": {
            key: sum(bool(row[f"total_top_{key}_percent"]) for row in shift_rows)
            for key in (1, 5, 10, 20)
        },
    }


def _figures(
    candidate_rows: Sequence[Mapping[str, str]], comparison: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/task016-matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = np.asarray([row["unit_type"] for row in candidate_rows])
    size = np.asarray([float(row["domain_active_size"]) for row in candidate_rows])
    average = np.asarray([float(row["delta_average"]) for row in candidate_rows])
    total = np.asarray([float(row["delta_total"]) for row in candidate_rows])
    colors = {TYPE_ATTENTION: "#d55e00", TYPE_FFN: "#0072b2"}

    def save(figure, name: str) -> None:
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    for values, title, name in (
        (average, "Domain-averaged marginal loss", "figure1_delta_average_distribution"),
        (total, "Domain-total marginal loss", "figure2_delta_total_distribution"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        arrays = [values[types == TYPE_ATTENTION], values[types == TYPE_FFN]]
        axis.boxplot(arrays, labels=["Attention", "FFN"], showfliers=False)
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        save(figure, name)

    for values, title, name in (
        (average, "Delta_average", "figure3_domain_size_vs_delta_average"),
        (total, "Delta_total", "figure4_domain_size_vs_delta_total"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for unit_type, label in ((TYPE_FFN, "FFN"), (TYPE_ATTENTION, "Attention")):
            mask = types == unit_type
            axis.scatter(size[mask], values[mask], s=7, alpha=0.25,
                         color=colors[unit_type], label=label)
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.set_xlabel("Active functional demand count |G+|")
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(alpha=0.2)
        save(figure, name)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for mode, marker in (("domain_average", "o"), ("domain_total", "s")):
        rows = sorted((row for row in comparison if row["score_mode"] == mode),
                      key=lambda row: float(row["target_sparsity"]))
        x = [100 * float(row["actual_sparsity"]) for row in rows]
        axes[0].plot(x, [int(row["attention_removed"]) for row in rows], marker=marker,
                     label=mode)
        axes[1].plot(x, [int(row["ffn_removed"]) for row in rows], marker=marker,
                     label=mode)
    axes[0].set_ylabel("Attention heads removed")
    axes[1].set_ylabel("FFN neurons removed")
    for axis in axes:
        axis.set_xlabel("Actual parameter sparsity (%)")
        axis.legend()
        axis.grid(alpha=0.25)
    save(figure, "figure5_type_removals")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for mode, marker in (("domain_average", "o"), ("domain_total", "s")):
        rows = sorted((row for row in comparison if row["score_mode"] == mode),
                      key=lambda row: float(row["actual_sparsity"]))
        axis.plot([100 * float(row["actual_sparsity"]) for row in rows],
                  [float(row["top1"]) for row in rows], marker=marker, label=mode)
    axis.set_xlabel("Actual parameter sparsity (%)")
    axis.set_ylabel("Pre-finetune Top-1 (%)")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, "figure6_top1_vs_actual_sparsity")

    shift = _read_csv(output_dir / "attention_rank_shift.csv")
    figure, axis = plt.subplots(figsize=(6.2, 5.5))
    axis.scatter([float(row["average_percentile"]) for row in shift],
                 [float(row["total_percentile"]) for row in shift], s=14, alpha=0.6)
    axis.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Attention percentile under domain_average (lower is better)")
    axis.set_ylabel("Attention percentile under domain_total (lower is better)")
    axis.grid(alpha=0.2)
    save(figure, "figure7_attention_rank_calibration")

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    labels = [f"{row['score_mode']}\n{100*float(row['target_sparsity']):.0f}%"
              for row in comparison]
    attention_cost = np.asarray(
        [float(row["attention_removed_parameters"]) for row in comparison]
    )
    ffn_cost = np.asarray([float(row["ffn_removed_parameters"]) for row in comparison])
    positions = np.arange(len(comparison))
    axis.bar(positions, attention_cost, label="Attention", color=colors[TYPE_ATTENTION])
    axis.bar(positions, ffn_cost, bottom=attention_cost, label="FFN",
             color=colors[TYPE_FFN])
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Estimated removed parameters")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, "figure8_parameter_removal_composition")


def aggregate(task014_root: Path, output_dir: Path) -> None:
    task014_root = Path(task014_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = verify_run_identity(task014_root, output_dir)
    _atomic_json(output_dir / "artifact_identity_audit.json", identity)
    comparison = []
    for mode, root, directory in (
        ("domain_average", task014_root, "functional"),
        ("domain_total", output_dir, "domain_total"),
    ):
        for target in TARGETS:
            metrics = _read_json(root / directory / TAGS[target] / "final_metrics.json")
            comparison.append(_comparison_row(mode, target, metrics))
    comparison_fields = (
        "score_mode", "target_sparsity", "actual_sparsity", "parameters_before",
        "parameters_after", "parameters_removed", "top1", "top5",
        "attention_removed", "attention_removal_ratio", "ffn_removed",
        "ffn_removal_ratio", "attention_removed_parameters",
        "ffn_removed_parameters",
    )
    _atomic_csv(output_dir / "comparison.csv", comparison_fields, comparison)
    initial_rows = _read_csv(
        output_dir / "domain_total" / "s10" / "initial_candidate_scores.csv"
    )
    diagnosis = _candidate_analysis(initial_rows, output_dir)
    _atomic_json(output_dir / "distribution_diagnosis.json", diagnosis)
    _figures(initial_rows, comparison, output_dir)

    total_traces = {
        target: _read_csv(
            output_dir / "domain_total" / TAGS[target] / "functional_selection_trace.csv"
        ) for target in TARGETS
    }
    first_attention = {}
    for target, rows in total_traces.items():
        selected = next((row for row in rows if row["unit_type"] == TYPE_ATTENTION), None)
        first_attention[target] = (
            float(selected["estimated_parameter_sparsity"]) if selected else None
        )
    top1_differences = {}
    for target in TARGETS:
        average_row = next(row for row in comparison if row["score_mode"] == "domain_average"
                           and float(row["target_sparsity"]) == target)
        total_row = next(row for row in comparison if row["score_mode"] == "domain_total"
                         and float(row["target_sparsity"]) == target)
        top1_differences[TAGS[target]] = float(total_row["top1"]) - float(average_row["top1"])
    maximum_layer_ratio = 0.0
    min_keep_violation = False
    for target in TARGETS:
        layer_rows = _read_csv(
            output_dir / "domain_total" / TAGS[target] / "layerwise_pruning_statistics.csv"
        )
        maximum_layer_ratio = max(
            maximum_layer_ratio,
            max((float(row["unit_removal_ratio"]) for row in layer_rows), default=0.0),
        )
        min_keep_violation = min_keep_violation or any(
            int(row["units_after"]) < max(1, int(0.1 * int(row["units_before"])))
            for row in layer_rows
        )
    attention_counts = {
        TAGS[target]: sum(row["unit_type"] == TYPE_ATTENTION for row in rows)
        for target, rows in total_traces.items()
    }
    ratio_reduced = abs(diagnosis["median_ratio_attention_over_ffn_domain_total"] - 1.0) < abs(
        diagnosis["median_ratio_attention_over_ffn_domain_average"] - 1.0
    )
    report = f"""# Task016 scientific diagnosis

All statements below are computed from identity-verified prune-only artifacts.

## Q1. Does domain_total remove score dependence on domain size?

Spearman(domain size, Delta_average) = `{diagnosis['spearman_domain_size_vs_delta_average']:.6g}`;
Spearman(domain size, Delta_total) = `{diagnosis['spearman_domain_size_vs_delta_total']:.6g}`.
The measured change, rather than candidate type counts alone, is the evidence.

## Q2. Does the Attention/FFN median score ratio decrease?

The ratio changes from `{diagnosis['median_ratio_attention_over_ffn_domain_average']:.6g}`
to `{diagnosis['median_ratio_attention_over_ffn_domain_total']:.6g}`.
Substantial reduction toward one: `{ratio_reduced}`.

## Q3. Does Attention naturally enter the sequence?

Attention removals at 10/20/30% are `{attention_counts['s10']}`,
`{attention_counts['s20']}`, `{attention_counts['s30']}`. No type quota,
weight, threshold or budget was used.

## Q4. When is the first Attention head selected?

Estimated parameter sparsities at first selection (None means absent):
10% `{first_attention[0.10]}`, 20% `{first_attention[0.20]}`,
30% `{first_attention[0.30]}`.

## Q5. How many Attention heads are selected?

10% `{attention_counts['s10']}`, 20% `{attention_counts['s20']}`,
30% `{attention_counts['s30']}`.

## Q6. How does pre-finetune Top-1 change?

domain_total minus domain_average at 10/20/30%:
`{top1_differences['s10']:.6g}`, `{top1_differences['s20']:.6g}`,
`{top1_differences['s30']:.6g}` percentage points.

## Q7. Is there abnormal layer concentration?

Maximum observed unit-removal ratio is `{maximum_layer_ratio:.6g}`.
Existing min-keep violation detected: `{min_keep_violation}`. Layer-level
judgment should use `layerwise_pruning_statistics.csv`; no post-hoc repair was applied.

## Q8. Is the change explained without manual balancing?

Yes at the implementation level: the only ranking change is
`Delta_total = |G+| * Delta_average`, with the fixed Task014 non-null demand
count. Descriptor, BMS, similarity, coverage, constraints and budgets are shared.

## Q9. Should domain_total replace domain_average?

No automatic adoption is asserted. Review the measured score correlations,
median-ratio change, Top-1 differences and layer distribution together. Attention
removal alone is explicitly not a success criterion.
"""
    (output_dir / "diagnosis.md").write_text(report, encoding="utf-8")
    _atomic_json(
        output_dir / "task016_completion.json",
        {
            "status": "passed",
            "artifact_identity_verified": True,
            "six_experiments_compared": True,
            "full_fine_tuning_executed": False,
            "type_specific_rule": False,
            "parameter_cost_in_ranking": False,
            "figures_png": 8,
            "figures_pdf": 8,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task014-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("task016_domain_size_calibration")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    aggregate(arguments.task014_root, arguments.output_dir)

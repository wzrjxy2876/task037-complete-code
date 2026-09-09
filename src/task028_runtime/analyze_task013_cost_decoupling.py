#!/usr/bin/env python3
"""Offline aggregation for Task013 cost-decoupled selection experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/task013-matplotlib-cache")

COST_MODES = ("coupled", "decoupled")
SPARSITIES = (0.10, 0.20, 0.30)
SUMMARY_FIELDS = (
    "selection_cost_mode", "target_sparsity", "baseline_top1", "baseline_top5",
    "pre_ft_top1", "pre_ft_top5", "top1_drop", "top5_drop",
    "estimated_budget_sparsity", "physical_numel_sparsity",
    "removed_heads", "remaining_heads", "removed_neurons", "remaining_neurons",
    "attention_removed_parameter_cost",
    "mlp_removed_parameter_cost", "attention_budget_fraction",
    "mlp_budget_fraction", "num_groups", "singleton_ratio",
)
TRACE_FIELDS = (
    "cost_mode", "target_sparsity", "budget_progress", "selection_rank",
    "layer", "unit_type", "group_id", "raw_pruning_score", "parameter_cost",
    "effective_selection_score", "selected",
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _same_float(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def collect_runs(root: Path) -> list[dict]:
    runs = []
    seen = set()
    for metrics_path in sorted((root / "runs").glob("*/final_metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        if metrics.get("status") != "task013_prune_only_complete":
            continue
        metadata = _read_json(run_dir / "run_metadata.json")
        key = (
            str(metrics["selection_cost_mode"]),
            round(float(metrics["target_sparsity"]), 12),
        )
        if key in seen:
            raise ValueError(f"duplicate Task013 run: {key}")
        seen.add(key)
        runs.append(
            {
                "run_dir": run_dir,
                "metrics": metrics,
                "metadata": metadata,
                "layers": _read_csv(run_dir / "layer_pruning.csv"),
                "trace": _read_csv(run_dir / "selection_trace.csv"),
                "candidates": _read_csv(run_dir / "selection_candidates.csv"),
            }
        )
    if not runs:
        raise FileNotFoundError(f"no completed Task013 runs under {root / 'runs'}")
    return runs


def _select(runs: Sequence[dict], mode: str, sparsity: float) -> dict:
    matches = [
        run for run in runs
        if run["metrics"]["selection_cost_mode"] == mode
        and _same_float(run["metrics"]["target_sparsity"], sparsity)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Task013 run for {mode}/s={sparsity:.2f}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_run_matrix(runs: Sequence[dict]) -> list[dict]:
    ordered = [_select(runs, mode, sparsity) for sparsity in SPARSITIES for mode in COST_MODES]
    reference = ordered[0]["metadata"]
    fixed_keys = (
        "checkpoint_sha256", "descriptor_variant", "selection_mode", "seed",
        "sigma", "gamma_decay", "min_keep_ratio", "importance_alpha",
        "calibration_batch_size", "calibration_batches",
        "calibration_samples_sha256", "validation_split", "validation_dataset",
        "validation_batch_size", "amp_enabled", "cuda_visible_devices",
        "visible_gpu_count", "gpu_names",
    )
    for run in ordered:
        metadata = run["metadata"]
        metrics = run["metrics"]
        if metadata.get("status") != "task013_prune_only_complete":
            raise ValueError(f"incomplete metadata: {run['run_dir']}")
        for key in fixed_keys:
            if metadata.get(key) != reference.get(key):
                raise ValueError(f"controlled setting {key!r} differs: {run['run_dir']}")
        if bool(metrics.get("full_fine_tuning_executed")):
            raise ValueError(f"fine tuning is forbidden: {run['run_dir']}")
        if metadata.get("selection_cost_mode") != metrics.get("selection_cost_mode"):
            raise ValueError(f"cost-mode mismatch: {run['run_dir']}")
        if not _same_float(metadata["target_sparsity"], metrics["target_sparsity"]):
            raise ValueError(f"sparsity mismatch: {run['run_dir']}")
        before = int(metrics["parameters_before"])
        after = int(metrics["parameters_after"])
        estimated = int(metrics["estimated_removed_parameters"])
        if not _same_float(metrics["physical_numel_sparsity"], (before - after) / before):
            raise ValueError(f"physical sparsity mismatch: {run['run_dir']}")
        if not _same_float(metrics["estimated_budget_sparsity"], estimated / before):
            raise ValueError(f"estimated sparsity mismatch: {run['run_dir']}")
        for layer in run["layers"]:
            remaining = int(layer["remaining_units"])
            at_min = str(layer["at_min_keep"]).lower() in {"true", "1", "yes"}
            original = int(layer["original_units"])
            minimum = max(1, int(original * float(metadata["min_keep_ratio"])))
            if remaining < minimum or at_min != (remaining == minimum):
                raise ValueError(f"layer minimum mismatch: {run['run_dir']}")
    if reference.get("descriptor_variant") != "dynamic3d":
        raise ValueError("Task013 primary matrix requires Dynamic3D")
    if int(reference.get("visible_gpu_count", 0)) < 2:
        raise ValueError("Task013 requires two visible GPUs")
    if not _same_float(reference["sigma"], 0.1):
        raise ValueError("Task013 requires sigma=0.1")
    if int(reference["seed"]) != 3407:
        raise ValueError("Task013 requires seed=3407")
    baselines = {
        (float(run["metrics"]["baseline_top1"]), float(run["metrics"]["baseline_top5"]))
        for run in ordered
    }
    if len(baselines) != 1:
        raise ValueError("baseline differs across controlled Task013 runs")
    return ordered


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _summary_rows(ordered: Sequence[dict]) -> list[dict]:
    return [
        {field: run["metrics"].get(field, "") for field in SUMMARY_FIELDS}
        for run in ordered
    ]


def _write_layer_outputs(root: Path, ordered: Sequence[dict]) -> None:
    layer_dir = root / "layer_pruning"
    for run in ordered:
        mode = run["metrics"]["selection_cost_mode"]
        tag = int(round(float(run["metrics"]["target_sparsity"]) * 100))
        rows = run["layers"]
        _atomic_csv(layer_dir / f"{mode}_s{tag}.csv", rows[0].keys(), rows)

    for sparsity in SPARSITIES:
        left = _select(ordered, "coupled", sparsity)["layers"]
        right = _select(ordered, "decoupled", sparsity)["layers"]
        coupled = {(row["layer"], row["unit_type"]): row for row in left}
        decoupled = {(row["layer"], row["unit_type"]): row for row in right}
        if set(coupled) != set(decoupled):
            raise ValueError(f"layer sets differ at sparsity {sparsity}")
        rows = []
        for key in coupled:
            c_row, d_row = coupled[key], decoupled[key]
            rows.append(
                {
                    "layer": key[0],
                    "unit_type": key[1],
                    "coupled_removed": int(c_row["removed_units"]),
                    "decoupled_removed": int(d_row["removed_units"]),
                    "difference": int(d_row["removed_units"]) - int(c_row["removed_units"]),
                    "coupled_removed_ratio": float(c_row["removed_ratio"]),
                    "decoupled_removed_ratio": float(d_row["removed_ratio"]),
                }
            )
        rows.sort(key=lambda row: abs(int(row["difference"])), reverse=True)
        fields = tuple(rows[0]) if rows else ()
        _atomic_csv(root / f"cost_mode_layer_diff_s{int(sparsity * 100)}.csv", fields, rows)


def _distribution_rows(ordered: Sequence[dict]) -> list[dict]:
    rows = []
    for run in ordered:
        for unit_type in ("attention_head", "ffn_neuron"):
            values = np.asarray(
                [
                    float(row["raw_pruning_score"])
                    for row in run["candidates"] if row["unit_type"] == unit_type
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            rows.append(
                {
                    "cost_mode": run["metrics"]["selection_cost_mode"],
                    "target_sparsity": run["metrics"]["target_sparsity"],
                    "unit_type": unit_type,
                    "count": values.size,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "max": float(np.max(values)),
                }
            )
    return rows


def _diagnosis(root: Path, ordered: Sequence[dict], distribution_rows: Sequence[dict]) -> None:
    def metric(mode, sparsity, name):
        return float(_select(ordered, mode, sparsity)["metrics"][name])

    accuracy_delta = {
        sparsity: metric("decoupled", sparsity, "pre_ft_top1")
        - metric("coupled", sparsity, "pre_ft_top1")
        for sparsity in SPARSITIES
    }
    budget_delta = {
        sparsity: metric("decoupled", sparsity, "attention_budget_fraction")
        - metric("coupled", sparsity, "attention_budget_fraction")
        for sparsity in SPARSITIES
    }
    head_delta = {
        sparsity: metric("decoupled", sparsity, "removed_head_ratio")
        - metric("coupled", sparsity, "removed_head_ratio")
        for sparsity in SPARSITIES
    }
    min_delta = {
        sparsity: int(metric("decoupled", sparsity, "attention_layers_at_min_keep"))
        - int(metric("coupled", sparsity, "attention_layers_at_min_keep"))
        for sparsity in SPARSITIES
    }
    dynamic20 = [
        row for row in distribution_rows
        if row["cost_mode"] == "coupled"
        and _same_float(row["target_sparsity"], 0.20)
    ]
    medians = {row["unit_type"]: float(row["median"]) for row in dynamic20}
    raw_attention_lower = (
        medians.get("attention_head", float("nan"))
        < medians.get("ffn_neuron", float("nan"))
    )
    lines = [
        "# Task013 cost-decoupling diagnosis",
        "",
        "All statements below are generated from the six completed prune-only runs. ",
        "No significance threshold or fine-tuned result is assumed.",
        "",
    ]
    for index, sparsity in enumerate(SPARSITIES, start=1):
        lines.append(
            f"{index}. **Pre-FT Top-1 at {sparsity:.0%}:** coupled "
            f"{metric('coupled', sparsity, 'pre_ft_top1'):.6g}, decoupled "
            f"{metric('decoupled', sparsity, 'pre_ft_top1'):.6g}, difference "
            f"{accuracy_delta[sparsity]:+.6g} percentage points."
        )
    lines.extend(
        [
            "4. **Attention budget fraction:** decoupled-minus-coupled differences "
            + ", ".join(f"{s:.0%}: {budget_delta[s]:+.6g}" for s in SPARSITIES)
            + ".",
            "5. **Removed Attention-head fraction:** decoupled-minus-coupled differences "
            + ", ".join(f"{s:.0%}: {head_delta[s]:+.6g}" for s in SPARSITIES)
            + ".",
            "6. **Attention layers at minimum retention:** decoupled-minus-coupled "
            + ", ".join(f"{s:.0%}: {min_delta[s]:+d}" for s in SPARSITIES)
            + ".",
            "7. **Raw score before cost:** at 20%, median raw scores are Attention "
            f"{medians.get('attention_head', float('nan')):.6g} and MLP "
            f"{medians.get('ffn_neuron', float('nan')):.6g}; lower-score-first therefore "
            + ("already favors Attention." if raw_attention_lower else "does not favor Attention by this median comparison."),
            "8. **Main cause:** compare the raw-score medians above with the exact allocation "
            "and accuracy differences in Questions 1–6; the data do not justify assigning "
            "causality to cost coupling alone.",
            "9. **Budget preservation:** target/achieved estimated sparsity pairs are "
            + ", ".join(
                f"{run['metrics']['selection_cost_mode']} {float(run['metrics']['target_sparsity']):.0%}/"
                f"{float(run['metrics']['estimated_budget_sparsity']):.6g}"
                for run in ordered
            )
            + ".",
            "10. **Permanent redesign:** no automatic yes/no threshold was specified. "
            "Use the measured accuracy, allocation and budget values above before making "
            "a permanent method decision.",
            "",
            "## Outcome classification",
            "",
        ]
    )
    outcomes = []
    if any(accuracy_delta[s] > 0 and budget_delta[s] < 0 for s in SPARSITIES):
        outcomes.append("Outcome A is partially supported descriptively where both accuracy improves and Attention budget share falls.")
    if any(budget_delta[s] < 0 for s in SPARSITIES) and not any(accuracy_delta[s] > 0 for s in SPARSITIES):
        outcomes.append("Outcome B is supported descriptively: allocation shifts without a positive accuracy difference.")
    if all(math.isclose(budget_delta[s], 0.0, rel_tol=0.0, abs_tol=1e-12) for s in SPARSITIES):
        outcomes.append("Outcome C is supported: Attention allocation is numerically unchanged.")
    if all(accuracy_delta[s] < 0 for s in SPARSITIES):
        outcomes.append("Outcome D is supported descriptively: decoupling is worse at all three targets.")
    if accuracy_delta[0.10] > 0 and accuracy_delta[0.20] > 0 and accuracy_delta[0.30] <= 0:
        outcomes.append("Outcome E is supported descriptively: 10%/20% improve but 30% does not.")
    if not outcomes:
        outcomes.append("No single predefined Outcome A–E pattern exactly matches all observed signs.")
    lines.extend(f"- {outcome}" for outcome in outcomes)
    lines.extend(["", "Full fine-tuning executed: **NO**", ""])
    _atomic_text(root / "diagnosis_summary.md", "\n".join(lines))


def _figures(root: Path, ordered: Sequence[dict]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(SPARSITIES) * 100.0
    plots = (
        ("pre_ft_top1", "Pre-FT Top-1 (%)", "preft_top1_cost_mode.png"),
        ("attention_budget_fraction", "Attention budget fraction", "attention_budget_fraction_cost_mode.png"),
        ("removed_head_ratio", "Removed Attention-head fraction", "removed_head_ratio_cost_mode.png"),
    )
    for metric_name, ylabel, filename in plots:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        for mode, marker in (("coupled", "o"), ("decoupled", "s")):
            values = [float(_select(ordered, mode, s)["metrics"][metric_name]) for s in SPARSITIES]
            axis.plot(x, values, marker=marker, label=mode.capitalize())
        if metric_name == "pre_ft_top1":
            baseline = float(ordered[0]["metrics"]["baseline_top1"])
            axis.axhline(baseline, color="black", linestyle="--", linewidth=1, label="Baseline")
        axis.set_xlabel("Target sparsity (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    colors = {"attention_head": "tab:blue", "ffn_neuron": "tab:orange"}
    markers = {"coupled": "o", "decoupled": "x"}
    for mode in COST_MODES:
        rows = _select(ordered, mode, 0.20)["candidates"]
        for unit_type in colors:
            selected = [row for row in rows if row["unit_type"] == unit_type]
            stride = max(1, math.ceil(len(selected) / 800))
            subset = selected[::stride]
            axis.scatter(
                [int(row["parameter_cost"]) for row in subset],
                [float(row["effective_selection_score"]) for row in subset],
                s=12, alpha=0.45, color=colors[unit_type], marker=markers[mode],
                label=f"{mode} {unit_type}",
            )
    axis.set_xlabel("Parameter cost per unit")
    axis.set_ylabel("Effective selection score (lower first)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "selection_score_vs_parameter_cost.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    coupled20 = _select(ordered, "coupled", 0.20)["candidates"]
    values = [
        [float(row["raw_pruning_score"]) for row in coupled20 if row["unit_type"] == unit_type]
        for unit_type in ("attention_head", "ffn_neuron")
    ]
    axis.boxplot(values, tick_labels=["Attention heads", "FFN neurons"], showfliers=False)
    axis.set_ylabel("Raw pruning score (lower first)")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "raw_score_by_type.png", dpi=180)
    plt.close(fig)


def analyze(root: Path) -> None:
    runs = collect_runs(root)
    ordered = validate_run_matrix(runs)
    summary = _summary_rows(ordered)
    _atomic_csv(root / "summary.csv", SUMMARY_FIELDS, summary)
    attention_rows = []
    for sparsity in SPARSITIES:
        coupled = _select(ordered, "coupled", sparsity)["metrics"]
        decoupled = _select(ordered, "decoupled", sparsity)["metrics"]
        attention_rows.append(
            {
                "target_sparsity": sparsity,
                "coupled_attention_budget_fraction": coupled["attention_budget_fraction"],
                "decoupled_attention_budget_fraction": decoupled["attention_budget_fraction"],
                "difference": float(decoupled["attention_budget_fraction"]) - float(coupled["attention_budget_fraction"]),
                "coupled_removed_heads": coupled["removed_heads"],
                "decoupled_removed_heads": decoupled["removed_heads"],
            }
        )
    _atomic_csv(
        root / "attention_budget_comparison.csv", tuple(attention_rows[0]), attention_rows
    )
    _atomic_csv(
        root / "type_count_ratios.csv",
        ("selection_cost_mode", "target_sparsity", "original_heads", "removed_heads", "removed_head_ratio", "original_neurons", "removed_neurons", "removed_neuron_ratio"),
        [
            {key: run["metrics"][key] for key in (
                "selection_cost_mode", "target_sparsity", "original_heads", "removed_heads",
                "removed_head_ratio", "original_neurons", "removed_neurons", "removed_neuron_ratio",
            )}
            for run in ordered
        ],
    )
    _write_layer_outputs(root, ordered)

    extreme_rows = []
    for run in ordered:
        attention_layers = [
            row for row in run["layers"] if row["unit_type"] == "attention_head"
        ]
        count = sum(
            str(row["at_min_keep"]).lower() in {"true", "1", "yes"}
            for row in attention_layers
        )
        extreme_rows.append(
            {
                "target_sparsity": run["metrics"]["target_sparsity"],
                "cost_mode": run["metrics"]["selection_cost_mode"],
                "num_extreme_attention_layers": count,
                "fraction_extreme_attention_layers": (
                    count / len(attention_layers) if attention_layers else 0.0
                ),
            }
        )
    _atomic_csv(
        root / "extreme_attention_comparison.csv",
        tuple(extreme_rows[0]),
        extreme_rows,
    )

    trace_rows = []
    for run in ordered:
        trace_rows.extend({field: row.get(field, "") for field in TRACE_FIELDS} for row in run["trace"])
    _atomic_csv(root / "selection_trace.csv", TRACE_FIELDS, trace_rows)

    correlation_rows = []
    for run in ordered:
        for unit_type in ("all", "attention_head", "ffn_neuron"):
            rows = run["candidates"] if unit_type == "all" else [
                row for row in run["candidates"] if row["unit_type"] == unit_type
            ]
            correlation_rows.append(
                {
                    "cost_mode": run["metrics"]["selection_cost_mode"],
                    "target_sparsity": run["metrics"]["target_sparsity"],
                    "unit_type": unit_type,
                    "count": len(rows),
                    "spearman_parameter_cost_vs_selection_rank": spearman(
                        [float(row["parameter_cost"]) for row in rows],
                        [float(row["selection_rank"]) for row in rows],
                    ),
                }
            )
    _atomic_csv(root / "cost_rank_spearman.csv", tuple(correlation_rows[0]), correlation_rows)

    distribution_rows = _distribution_rows(ordered)
    _atomic_csv(root / "raw_score_by_type.csv", tuple(distribution_rows[0]), distribution_rows)
    _diagnosis(root, ordered, distribution_rows)
    _figures(root, ordered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="task013_cost_decoupling")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analyze(Path(args.output_dir))
    print(f"Task013 analysis complete: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

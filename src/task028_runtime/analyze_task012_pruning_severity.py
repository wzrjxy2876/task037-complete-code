#!/usr/bin/env python3
"""Offline aggregation for Task012 pruning-severity diagnostics.

The script reads completed per-run JSON/CSV artifacts only.  It never imports
the model, loads a checkpoint, or reruns calibration/validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from task012_diagnostics import consecutive_accuracy_drops


os.environ.setdefault("MPLCONFIGDIR", "/tmp/task012-matplotlib-cache")


VARIANTS = ("old3d", "dynamic3d")
SPARSITIES = (0.10, 0.20, 0.30, 0.40, 0.50)
PROTECTIONS = ("original", "min2", "min3")
SWEEP_FIELDS = (
    "variant", "target_sparsity", "baseline_top1", "baseline_top5",
    "pre_ft_top1", "pre_ft_top5", "top1_drop", "top5_drop",
    "estimated_budget_sparsity", "physical_numel_sparsity",
    "removed_heads", "remaining_heads", "removed_neurons",
    "remaining_neurons", "num_groups", "singleton_ratio",
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
    """Load completed run triplets and reject duplicate configurations."""
    runs = []
    seen = set()
    for metrics_path in sorted((root / "runs").glob("*/final_metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        metadata = _read_json(run_dir / "run_metadata.json")
        layers = _read_csv(run_dir / "layer_pruning.csv")
        if metrics.get("status") != "task012_prune_only_complete":
            continue
        key = (
            str(metrics.get("variant")),
            round(float(metrics.get("target_sparsity")), 12),
            str(metrics.get("attention_protection")),
        )
        if key in seen:
            raise ValueError(f"duplicate Task012 run configuration: {key}")
        seen.add(key)
        runs.append(
            {
                "run_dir": run_dir,
                "metrics": metrics,
                "metadata": metadata,
                "layers": layers,
            }
        )
    if not runs:
        raise FileNotFoundError(f"no completed Task012 runs under {root / 'runs'}")
    return runs


def _select(
    runs: Sequence[dict], variant: str, sparsity: float, protection: str
) -> dict:
    matches = [
        run for run in runs
        if run["metrics"]["variant"] == variant
        and _same_float(run["metrics"]["target_sparsity"], sparsity)
        and run["metrics"]["attention_protection"] == protection
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one run for {variant}/s={sparsity:.2f}/{protection}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_run_matrix(runs: Sequence[dict]) -> None:
    """Enforce the 10-run sweep, six-row protection set, and fixed controls."""
    required = []
    for variant in VARIANTS:
        for sparsity in SPARSITIES:
            required.append(_select(runs, variant, sparsity, "original"))
        for protection in ("min2", "min3"):
            required.append(_select(runs, variant, 0.50, protection))

    reference = required[0]["metadata"]
    fixed_keys = (
        "checkpoint_sha256", "seed", "sigma", "min_keep_ratio",
        "importance_alpha", "gamma_decay", "selection_mode",
        "calibration_batch_size", "calibration_batches",
        "calibration_samples_sha256", "validation_split",
        "validation_dataset", "validation_batch_size", "amp_enabled",
        "cuda_visible_devices", "visible_gpu_count", "gpu_names",
    )
    for run in required:
        metadata = run["metadata"]
        metrics = run["metrics"]
        if metadata.get("status") != "task012_prune_only_complete":
            raise ValueError(f"incomplete metadata: {run['run_dir']}")
        for key in fixed_keys:
            if metadata.get(key) != reference.get(key):
                raise ValueError(
                    f"controlled setting {key!r} differs in {run['run_dir']}"
                )
        if bool(run["metrics"].get("full_fine_tuning_executed")):
            raise ValueError(f"fine-tuned result is forbidden: {run['run_dir']}")
        if metadata.get("variant") != metrics.get("variant") or not _same_float(
            metadata.get("target_sparsity"), metrics.get("target_sparsity")
        ):
            raise ValueError(f"metadata/metrics identity mismatch: {run['run_dir']}")
        expected_diagnostic = {"original": 0, "min2": 2, "min3": 3}[
            metrics["attention_protection"]
        ]
        if int(metadata.get("diagnostic_min_attention_heads", -1)) != expected_diagnostic:
            raise ValueError(f"protection metadata mismatch: {run['run_dir']}")
        before = int(metrics["parameters_before"])
        after = int(metrics["parameters_after"])
        estimated = int(metrics["estimated_removed_parameters"])
        if not _same_float(metrics["physical_numel_sparsity"], (before - after) / before):
            raise ValueError(f"physical sparsity audit mismatch: {run['run_dir']}")
        if not _same_float(metrics["estimated_budget_sparsity"], estimated / before):
            raise ValueError(f"estimated sparsity audit mismatch: {run['run_dir']}")
        for layer in run["layers"]:
            remaining = int(layer["remaining_units"])
            minimum = int(layer["min_keep_units"])
            saved_at_minimum = str(layer["at_min_keep"]).strip().lower() in {
                "true", "1", "yes"
            }
            if remaining < minimum or saved_at_minimum != (remaining == minimum):
                raise ValueError(f"layer minimum audit mismatch: {run['run_dir']}")
    if not _same_float(reference["sigma"], 0.1):
        raise ValueError("Task012 requires sigma=0.1")
    if not _same_float(reference["min_keep_ratio"], 0.1):
        raise ValueError("Task012 requires min_keep_ratio=0.1")
    if int(reference["seed"]) != 3407:
        raise ValueError("Task012 requires seed=3407")
    if int(reference["visible_gpu_count"]) < 2:
        raise ValueError("Task012 requires two visible GPUs")
    baseline_pairs = {
        (float(run["metrics"]["baseline_top1"]), float(run["metrics"]["baseline_top5"]))
        for run in required
    }
    if len(baseline_pairs) != 1:
        raise ValueError("baseline values differ across controlled Task012 runs")


def layer_difference(old_rows: Sequence[dict], dynamic_rows: Sequence[dict]) -> list[dict]:
    def keyed(rows):
        result = {(row["layer"], row["unit_type"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate layer/unit_type in layer audit")
        return result

    old = keyed(old_rows)
    dynamic = keyed(dynamic_rows)
    if set(old) != set(dynamic):
        raise ValueError("Old3D and Dynamic3D layer sets differ")
    rows = []
    for key in old:
        left = old[key]
        right = dynamic[key]
        old_removed = int(left["removed_units"])
        dynamic_removed = int(right["removed_units"])
        rows.append(
            {
                "layer": key[0],
                "unit_type": key[1],
                "old_removed": old_removed,
                "dynamic_removed": dynamic_removed,
                "difference": dynamic_removed - old_removed,
                "old_removed_ratio": float(left["removed_ratio"]),
                "dynamic_removed_ratio": float(right["removed_ratio"]),
            }
        )
    rows.sort(key=lambda row: (-abs(int(row["difference"])), row["layer"], row["unit_type"]))
    return rows


def extreme_attention_record(run: dict) -> dict:
    attention = [
        row for row in run["layers"] if row["unit_type"] == "attention_head"
    ]
    if not attention:
        raise ValueError(f"no Attention layers in {run['run_dir']}")
    extreme = [
        row for row in attention
        if str(row["at_min_keep"]).strip().lower() in {"true", "1", "yes"}
    ]
    return {
        "variant": run["metrics"]["variant"],
        "target_sparsity": float(run["metrics"]["target_sparsity"]),
        "num_attention_layers": len(attention),
        "num_extreme_attention_layers": len(extreme),
        "fraction_extreme_attention_layers": len(extreme) / len(attention),
        "total_heads_removed": sum(int(row["removed_heads"]) for row in attention),
        "min_remaining_heads": min(int(row["remaining_heads"]) for row in attention),
        "extreme_layer_names": [row["layer"] for row in extreme],
    }


def _rank_average(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], rank: bool = False):
    x = _rank_average(left) if rank else np.asarray(left, dtype=float)
    y = _rank_average(right) if rank else np.asarray(right, dtype=float)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _format_correlation(value) -> str:
    return "undefined (one series is constant)" if value is None else f"{value:.6f}"


def _plot_accuracy(path: Path, sweep: Mapping[str, Sequence[dict]], metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    labels = {"old3d": "Old3D", "dynamic3d": "Dynamic3D"}
    for variant in VARIANTS:
        rows = sweep[variant]
        axis.plot(
            [float(row["metrics"]["target_sparsity"]) for row in rows],
            [float(row["metrics"][metric]) for row in rows],
            marker="o",
            linewidth=2,
            label=labels[variant],
        )
    baseline_key = "baseline_top1" if metric == "pre_ft_top1" else "baseline_top5"
    baseline = float(sweep["old3d"][0]["metrics"][baseline_key])
    axis.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
                 label="Unpruned baseline")
    axis.set_xlabel("Target sparsity")
    axis.set_ylabel("Pre-finetune accuracy (%)")
    axis.set_title("Pre-finetune Top-1 vs sparsity" if metric == "pre_ft_top1"
                   else "Pre-finetune Top-5 vs sparsity")
    axis.set_xlim(0.08, 0.52)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_series(
    path: Path,
    sweep: Mapping[str, Sequence[dict]],
    getter,
    ylabel: str,
    title: str,
    ylim=None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for variant, label in (("old3d", "Old3D"), ("dynamic3d", "Dynamic3D")):
        rows = sweep[variant]
        axis.plot(
            [float(row["metrics"]["target_sparsity"]) for row in rows],
            [getter(row) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.set_xlabel("Target sparsity")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_heads(path: Path, sweep: Mapping[str, Sequence[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(SPARSITIES))
    width = 0.36
    old = [int(run["metrics"]["removed_heads"]) for run in sweep["old3d"]]
    dynamic = [int(run["metrics"]["removed_heads"]) for run in sweep["dynamic3d"]]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.bar(x - width / 2, old, width, label="Old3D")
    axis.bar(x + width / 2, dynamic, width, label="Dynamic3D")
    axis.set_xticks(x, [f"{value:.0%}" for value in SPARSITIES])
    axis.set_xlabel("Target sparsity")
    axis.set_ylabel("Attention heads removed")
    axis.set_title("Old3D vs Dynamic3D Attention removal")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_protection(path: Path, protection_runs: Mapping[str, Sequence[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(PROTECTIONS))
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for variant, label in (("old3d", "Old3D"), ("dynamic3d", "Dynamic3D")):
        axis.plot(
            x,
            [float(run["metrics"]["pre_ft_top1"]) for run in protection_runs[variant]],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.set_xticks(x, PROTECTIONS)
    axis.set_ylabel("Pre-finetune Top-1 (%)")
    axis.set_title("50% sparsity: diagnostic Attention protection")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _robustness_sentence(sweep: Mapping[str, Sequence[dict]]) -> str:
    old = [float(run["metrics"]["pre_ft_top1"]) for run in sweep["old3d"]]
    dynamic = [float(run["metrics"]["pre_ft_top1"]) for run in sweep["dynamic3d"]]
    wins = sum(right > left for left, right in zip(old, dynamic))
    losses = sum(right < left for left, right in zip(old, dynamic))
    mean_difference = float(np.mean(np.asarray(dynamic) - np.asarray(old)))
    if wins > losses:
        label = "Dynamic3D preserves higher Top-1 more often"
    elif losses > wins:
        label = "Old3D preserves higher Top-1 more often"
    else:
        label = "neither descriptor wins more sparsity points"
    return (
        f"{label}: Dynamic3D wins {wins}/5, Old3D wins {losses}/5, and the "
        f"mean Dynamic3D-minus-Old3D difference is {mean_difference:+.6f} pp."
    )


def _diagnosis(
    sweep: Mapping[str, Sequence[dict]],
    protection_runs: Mapping[str, Sequence[dict]],
    extremes: Mapping[tuple, dict],
    step_rows: Sequence[dict],
) -> str:
    largest = {}
    for variant in VARIANTS:
        rows = [row for row in step_rows if row["variant"] == variant]
        largest[variant] = max(
            rows,
            key=lambda row: (float(row["incremental_top1_drop"]),
                             -float(row["to_sparsity"])),
        )
    correlations = {}
    for variant in VARIANTS:
        ratios = [
            extremes[(variant, sparsity)]["fraction_extreme_attention_layers"]
            for sparsity in SPARSITIES
        ]
        top1 = [float(run["metrics"]["pre_ft_top1"]) for run in sweep[variant]]
        correlations[variant] = (
            _correlation(ratios, top1), _correlation(ratios, top1, rank=True)
        )

    protection_evidence = {}
    for variant in VARIANTS:
        values = {
            run["metrics"]["attention_protection"]: run["metrics"]
            for run in protection_runs[variant]
        }
        original = values["original"]
        min2 = values["min2"]
        min3 = values["min3"]
        protection_evidence[variant] = {
            "min2_recovery": float(min2["pre_ft_top1"]) - float(original["pre_ft_top1"]),
            "min3_vs_min2": float(min3["pre_ft_top1"]) - float(min2["pre_ft_top1"]),
            "min3_recovery": float(min3["pre_ft_top1"]) - float(original["pre_ft_top1"]),
            "loss": float(original["baseline_top1"]) - float(original["pre_ft_top1"]),
            "budget_values": [
                float(values[name]["estimated_actual_sparsity"]) for name in PROTECTIONS
            ],
        }

    extreme_lines = []
    for variant in VARIANTS:
        counts = ", ".join(
            f"{sparsity:.0%}: {extremes[(variant, sparsity)]['num_extreme_attention_layers']}"
            for sparsity in SPARSITIES
        )
        extreme_lines.append(f"{variant}: {counts}")

    budget_lines = []
    for variant in VARIANTS:
        fractions = [
            float(run["metrics"]["attention_budget_fraction"])
            for run in sweep[variant]
        ]
        budget_lines.append(
            f"{variant} Attention budget fractions: "
            + ", ".join(f"{value:.6f}" for value in fractions)
        )

    outcome_lines = []
    largest_targets = [float(largest[variant]["to_sparsity"]) for variant in VARIANTS]
    if all(value >= 0.40 for value in largest_targets):
        outcome_lines.append("- **Outcome A:** assigned; both largest losses end at 40% or 50% sparsity.")
    elif any(value <= 0.20 for value in largest_targets):
        outcome_lines.append("- **Outcome B:** assigned; at least one largest loss occurs by 20% sparsity.")
    else:
        outcome_lines.append("- **Outcome A/B:** neither assigned; the observed largest-loss location is intermediate.")

    recovery_fractions = []
    max_recoveries = []
    for evidence in protection_evidence.values():
        recovery = max(evidence["min2_recovery"], evidence["min3_recovery"])
        max_recoveries.append(recovery)
        recovery_fractions.append(recovery / evidence["loss"] if evidence["loss"] > 0 else 0.0)
    if any(value >= 0.5 for value in recovery_fractions):
        outcome_lines.append(
            "- **Outcome C:** assigned by the documented operational rule: protection recovers at least half of the observed 50% Top-1 loss for a variant."
        )
    elif all(value <= 0.1 for value in recovery_fractions):
        outcome_lines.append(
            "- **Outcome D:** assigned by the documented operational rule: protection recovers at most 10% of the observed 50% Top-1 loss for both variants."
        )
    else:
        outcome_lines.append(
            "- **Outcome C/D:** inconclusive; recovery lies between the 10% 'little' and 50% 'strong' bookkeeping rules."
        )
    dynamic_wins = sum(
        float(right["metrics"]["pre_ft_top1"]) > float(left["metrics"]["pre_ft_top1"])
        for left, right in zip(sweep["old3d"], sweep["dynamic3d"])
    )
    if dynamic_wins >= 3:
        outcome_lines.append(
            f"- **Outcome E:** assigned; Dynamic3D has higher Pre-FT Top-1 at {dynamic_wins}/5 sparsities."
        )
    else:
        outcome_lines.append(
            f"- **Outcome E:** not assigned; Dynamic3D is higher at only {dynamic_wins}/5 sparsities."
        )
    outcome_lines.append(
        "- **Outcome F:** not assigned from Task012 alone because a separate BMS-domain quality metric is not part of this sweep."
    )

    def collapse_line(variant):
        row = largest[variant]
        return (
            f"{float(row['from_sparsity']):.0%} -> "
            f"{float(row['to_sparsity']):.0%}, loss "
            f"{float(row['incremental_top1_drop']):.6f} pp"
        )

    lines = [
        "# Task012 pruning-severity diagnosis",
        "",
        "> Diagnostic evidence only. Pearson/Spearman use five sweep points per variant; no significance claim is made.",
        "",
        "## Answers",
        "",
        f"1. **Old3D largest incremental collapse:** {collapse_line('old3d')}.",
        f"2. **Dynamic3D largest incremental collapse:** {collapse_line('dynamic3d')}.",
        f"3. **Relative robustness:** {_robustness_sentence(sweep)}",
        "4. **Attention layers at the minimum boundary:** " + "; ".join(extreme_lines) + ".",
        "5. **Coincidence with accuracy:** descriptive correlations of extreme-layer fraction with Pre-FT Top-1 are "
        + "; ".join(
            f"{variant} Pearson={_format_correlation(correlations[variant][0])}, "
            f"Spearman={_format_correlation(correlations[variant][1])}"
            for variant in VARIANTS
        ) + ". Inspect the step table alongside these values; correlation is not causation.",
        "6. **min2 recovery at 50%:** " + "; ".join(
            f"{variant} {protection_evidence[variant]['min2_recovery']:+.6f} pp"
            for variant in VARIANTS
        ) + ".",
        "7. **Additional min3 change over min2:** " + "; ".join(
            f"{variant} {protection_evidence[variant]['min3_vs_min2']:+.6f} pp"
            for variant in VARIANTS
        ) + ".",
        "8. **Budget after protection:** " + "; ".join(
            f"{variant} achieved " + ", ".join(
                f"{name}={value:.6%}" for name, value in zip(
                    PROTECTIONS, protection_evidence[variant]["budget_values"]
                )
            )
            for variant in VARIANTS
        ) + ". A one-percentage-point distance from 50% is used only as an operational budget check, not a scientific threshold.",
        "9. **Budget allocation difference:** " + "; ".join(budget_lines) + ". Exact per-type costs are in `budget_allocation_summary.csv`.",
        "10. **Can extreme Attention explain most of the 50% loss?** Best protection recovery as a fraction of the original 50% loss is "
        + "; ".join(
            f"{variant} {fraction:.6%}"
            for variant, fraction in zip(VARIANTS, recovery_fractions)
        ) + ". This controlled intervention is the causal diagnostic; the correlation alone is not sufficient.",
        "11. **Next diagnostic direction:** " + (
            "Attention protection merits further controlled study because at least one setting recovers a positive Top-1 amount; do not adopt it as the method without follow-up."
            if any(value > 0 for value in max_recoveries)
            else "The tested Attention protections do not recover Top-1; investigate selection/application and non-Attention allocation instead."
        ),
        "",
        "## Diagnostic outcomes",
        "",
        *outcome_lines,
        "",
        "The 10%/50% recovery fractions above are explicit operational labels for outcomes C/D, not inferred scientific phase-transition thresholds.",
        "",
        "## Parameter accounting",
        "",
        "Logical estimated pruning cost, physical tensor numel reduction, and before/after parameter counts are kept separate in every budget table.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(output_dir: Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(root)
    validate_run_matrix(runs)
    sweep = {
        variant: [_select(runs, variant, sparsity, "original") for sparsity in SPARSITIES]
        for variant in VARIANTS
    }
    protection_runs = {
        variant: [_select(runs, variant, 0.50, protection) for protection in PROTECTIONS]
        for variant in VARIANTS
    }

    summary_rows = []
    for variant in VARIANTS:
        for run in sweep[variant]:
            summary_rows.append({field: run["metrics"][field] for field in SWEEP_FIELDS})
    summary_rows.sort(
        key=lambda row: (str(row["variant"]), float(row["target_sparsity"]))
    )
    _atomic_csv(root / "sparsity_sweep_summary.csv", SWEEP_FIELDS, summary_rows)

    layer_root = root / "layer_pruning"
    layer_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for run in sweep[variant]:
            sparsity = float(run["metrics"]["target_sparsity"])
            destination = layer_root / f"{variant}_s{sparsity:.2f}.csv"
            temporary = destination.with_suffix(".csv.tmp")
            shutil.copyfile(run["run_dir"] / "layer_pruning.csv", temporary)
            temporary.replace(destination)

    step_rows = []
    for variant in VARIANTS:
        first = sweep[variant][0]["metrics"]
        points = [(0.0, float(first["baseline_top1"]))] + [
            (float(run["metrics"]["target_sparsity"]), float(run["metrics"]["pre_ft_top1"]))
            for run in sweep[variant]
        ]
        for row in consecutive_accuracy_drops(points):
            step_rows.append({"variant": variant, **row})
    _atomic_csv(
        root / "accuracy_drop_by_step.csv",
        ("variant", "from_sparsity", "to_sparsity", "top1_before",
         "top1_after", "incremental_top1_drop"),
        step_rows,
    )

    extremes = {}
    extreme_rows = []
    markdown = ["# Extreme Attention layers", "",
                "Extreme means `remaining_heads == min_keep_heads`; no new percentage threshold is used.", ""]
    for variant in VARIANTS:
        markdown.extend([f"## {variant}", ""])
        for run in sweep[variant]:
            record = extreme_attention_record(run)
            key = (variant, float(record["target_sparsity"]))
            extremes[key] = record
            extreme_rows.append({key: value for key, value in record.items()
                                 if key != "extreme_layer_names"})
            names = record["extreme_layer_names"]
            markdown.append(
                f"- **{float(record['target_sparsity']):.0%}:** "
                + (", ".join(f"`{name}`" for name in names) if names else "none")
            )
        markdown.append("")
    _atomic_csv(
        root / "extreme_attention_summary.csv",
        ("variant", "target_sparsity", "num_attention_layers",
         "num_extreme_attention_layers", "fraction_extreme_attention_layers",
         "total_heads_removed", "min_remaining_heads"),
        extreme_rows,
    )
    _atomic_text(root / "extreme_attention_layers.md", "\n".join(markdown) + "\n")

    difference_fields = (
        "layer", "unit_type", "old_removed", "dynamic_removed", "difference",
        "old_removed_ratio", "dynamic_removed_ratio",
    )
    for sparsity in SPARSITIES:
        difference = layer_difference(
            _select(runs, "old3d", sparsity, "original")["layers"],
            _select(runs, "dynamic3d", sparsity, "original")["layers"],
        )
        _atomic_csv(
            root / f"old_vs_dynamic_layer_difference_s{int(sparsity * 100):02d}.csv",
            difference_fields,
            difference,
        )

    budget_fields = (
        "variant", "attention_protection", "target_sparsity",
        "parameters_before", "parameters_after", "estimated_removed_parameters",
        "estimated_budget_sparsity", "physical_numel_sparsity",
        "attention_removed_parameter_cost", "mlp_removed_parameter_cost",
        "attention_budget_fraction", "mlp_budget_fraction",
    )
    budget_runs = [run for variant in VARIANTS for run in sweep[variant]] + [
        run for variant in VARIANTS for run in protection_runs[variant]
        if run["metrics"]["attention_protection"] != "original"
    ]
    budget_rows = [
        {field: run["metrics"][field] for field in budget_fields}
        for run in budget_runs
    ]
    budget_rows.sort(key=lambda row: (
        row["variant"], float(row["target_sparsity"]),
        PROTECTIONS.index(row["attention_protection"])
    ))
    _atomic_csv(root / "budget_allocation_summary.csv", budget_fields, budget_rows)

    protection_fields = (
        "variant", "attention_protection", "target_sparsity",
        "estimated_actual_sparsity", "physical_numel_sparsity",
        "parameters_before", "parameters_after", "estimated_removed_parameters",
        "pre_ft_top1", "pre_ft_top5", "removed_heads", "removed_neurons",
        "num_extreme_attention_layers",
    )
    protection_rows = []
    for variant in VARIANTS:
        for run in protection_runs[variant]:
            metrics = run["metrics"]
            record = extreme_attention_record(run)
            protection_rows.append({
                **{field: metrics[field] for field in protection_fields
                   if field != "num_extreme_attention_layers"},
                "num_extreme_attention_layers": record["num_extreme_attention_layers"],
            })
    _atomic_csv(root / "attention_protection_50.csv", protection_fields, protection_rows)

    _plot_accuracy(root / "preft_top1_vs_sparsity.png", sweep, "pre_ft_top1")
    _plot_accuracy(root / "preft_top5_vs_sparsity.png", sweep, "pre_ft_top5")
    _plot_series(
        root / "extreme_attention_vs_sparsity.png", sweep,
        lambda run: extremes[(run["metrics"]["variant"],
                              float(run["metrics"]["target_sparsity"]))][
                                  "fraction_extreme_attention_layers"],
        "Fraction of Attention layers at minimum", "Extreme Attention vs sparsity",
        (0, 1),
    )
    _plot_series(
        root / "attention_budget_fraction_vs_sparsity.png", sweep,
        lambda run: float(run["metrics"]["attention_budget_fraction"]),
        "Attention fraction of estimated removed cost",
        "Attention budget allocation vs sparsity", (0, 1),
    )
    _plot_heads(root / "old_vs_dynamic_heads_removed.png", sweep)
    _plot_protection(root / "attention_protection_50.png", protection_runs)
    _atomic_text(
        root / "diagnosis_summary.md",
        _diagnosis(sweep, protection_runs, extremes, step_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="task012_pruning_severity")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(Path(parse_args().output_dir))

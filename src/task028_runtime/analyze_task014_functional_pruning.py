"""Offline aggregation for the six Task014 prune-only runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence


SPARSITIES = (0.10, 0.20, 0.30)
MODES = ("bms", "functional")
SUMMARY_FIELDS = (
    "selection_mode",
    "target_sparsity",
    "baseline_top1",
    "baseline_top5",
    "pre_ft_top1",
    "pre_ft_top5",
    "top1_drop",
    "top5_drop",
    "estimated_budget_sparsity",
    "physical_numel_sparsity",
    "removed_heads",
    "removed_neurons",
    "remaining_heads",
    "remaining_neurons",
    "attention_removed_parameter_cost",
    "mlp_removed_parameter_cost",
    "attention_budget_fraction",
    "mlp_budget_fraction",
    "num_bms_domains",
    "num_removed_units",
)


def _tag(sparsity: float) -> str:
    return f"s{int(round(sparsity * 100)):02d}"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _as_float(row: Mapping[str, object], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def _load_runs(output_dir: Path):
    metrics = {}
    layers = {}
    traces = {}
    domains = {}
    dependencies = {}
    for mode in MODES:
        for sparsity in SPARSITIES:
            run_dir = output_dir / mode / _tag(sparsity)
            row = _read_json(run_dir / "final_metrics.json")
            if row.get("status") != "task014_prune_only_complete":
                raise RuntimeError(f"Incomplete Task014 run: {run_dir}")
            metrics[(mode, sparsity)] = row
            layers[(mode, sparsity)] = _read_csv(run_dir / "layer_pruning.csv")
            if mode == "functional":
                traces[sparsity] = _read_csv(
                    run_dir / "functional_selection_trace.csv"
                )
                domains[sparsity] = _read_csv(
                    run_dir / "functional_domain_summary.csv"
                )
                dependencies[sparsity] = _read_csv(
                    run_dir / "set_dependency_examples.csv"
                )
    return metrics, layers, traces, domains, dependencies


def _plot(output_dir, metrics, domains, traces, dependencies) -> None:
    import matplotlib.pyplot as plt

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    x = [int(sparsity * 100) for sparsity in SPARSITIES]

    def line_plot(filename, ylabel, key):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for mode, label in (("bms", "BMS score"), ("functional", "Functional coverage")):
            axis.plot(
                x,
                [_as_float(metrics[(mode, sparsity)], key) for sparsity in SPARSITIES],
                marker="o",
                label=label,
            )
        axis.set_xlabel("Target sparsity (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(analysis_dir / filename, dpi=220)
        plt.close(figure)

    line_plot("preft_top1_functional_vs_bms.png", "Pre-finetune Top-1 (%)", "pre_ft_top1")
    line_plot(
        "attention_budget_fraction_functional_vs_bms.png",
        "Attention budget fraction",
        "attention_budget_fraction",
    )
    line_plot(
        "removed_head_ratio_functional_vs_bms.png",
        "Removed head ratio",
        "removed_head_ratio",
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        x,
        [
            mean(_as_float(row, "final_coverage") for row in domains[sparsity])
            for sparsity in SPARSITIES
        ],
        marker="o",
        label="Mean",
    )
    axis.plot(
        x,
        [
            median(_as_float(row, "final_coverage") for row in domains[sparsity])
            for sparsity in SPARSITIES
        ],
        marker="s",
        label="Median",
    )
    axis.set_xlabel("Target sparsity (%)")
    axis.set_ylabel("Final domain coverage")
    axis.set_xticks(x)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(analysis_dir / "final_domain_coverage_vs_sparsity.png", dpi=220)
    plt.close(figure)

    attention = []
    mlp = []
    for sparsity in SPARSITIES:
        for row in traces[sparsity]:
            value = _as_float(row, "marginal_functional_loss")
            (attention if row["unit_type"] == "attention_head" else mlp).append(value)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.boxplot(
        [attention, mlp],
        tick_labels=["Attention", "MLP"],
        showfliers=False,
    )
    axis.set_ylabel("Selected marginal functional loss")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(analysis_dir / "marginal_loss_distribution_by_type.png", dpi=220)
    plt.close(figure)

    examples = []
    seen_domains = set()
    for sparsity in SPARSITIES:
        eligible_domains = {
            int(row["domain_id"])
            for row in domains[sparsity]
            if int(row["removed_count"]) >= 2 and int(row["initial_size"]) > 1
        }
        for row in dependencies[sparsity]:
            key = (sparsity, int(row["domain_id"]))
            if key[1] not in eligible_domains:
                continue
            if key in seen_domains:
                continue
            seen_domains.add(key)
            examples.append((key, _as_float(row, "change")))
            if len(examples) >= 20:
                break
        if len(examples) >= 20:
            break
    if len(examples) < 20:
        raise RuntimeError(
            "Task014 requires at least 20 non-singleton, multiply-pruned "
            f"set-dependence domains; found {len(examples)}"
        )
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(
        range(len(examples)),
        [change for _, change in examples],
    )
    axis.set_xlabel("Non-singleton domain example")
    axis.set_ylabel("Marginal-loss increase")
    axis.set_xticks(range(len(examples)))
    axis.set_xticklabels(
        [f"{int(s * 100)}%-D{domain}" for (s, domain), _ in examples],
        rotation=70,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(analysis_dir / "set_dependent_marginal_loss_examples.png", dpi=220)
    plt.close(figure)


def analyze(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    metrics, layers, traces, domains, dependencies = _load_runs(output_dir)

    summary_rows = [
        {field: metrics[(mode, sparsity)][field] for field in SUMMARY_FIELDS}
        for sparsity in SPARSITIES
        for mode in MODES
    ]
    _write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)

    comparison_fields = (
        "target_sparsity", "bms", "functional", "functional_minus_bms"
    )
    attention_budget_rows = []
    for sparsity in SPARSITIES:
        bms = _as_float(metrics[("bms", sparsity)], "attention_budget_fraction")
        functional_value = _as_float(
            metrics[("functional", sparsity)], "attention_budget_fraction"
        )
        attention_budget_rows.append(
            {
                "target_sparsity": sparsity,
                "bms": bms,
                "functional": functional_value,
                "functional_minus_bms": functional_value - bms,
            }
        )
    _write_csv(
        output_dir / "attention_budget_comparison.csv",
        comparison_fields,
        attention_budget_rows,
    )

    removed_ratio_rows = []
    for sparsity in SPARSITIES:
        bms_head = _as_float(metrics[("bms", sparsity)], "removed_head_ratio")
        functional_head = _as_float(
            metrics[("functional", sparsity)], "removed_head_ratio"
        )
        bms_mlp = _as_float(metrics[("bms", sparsity)], "removed_mlp_ratio")
        functional_mlp = _as_float(
            metrics[("functional", sparsity)], "removed_mlp_ratio"
        )
        removed_ratio_rows.append(
            {
                "target_sparsity": sparsity,
                "bms_removed_head_ratio": bms_head,
                "functional_removed_head_ratio": functional_head,
                "head_ratio_difference": functional_head - bms_head,
                "bms_removed_mlp_ratio": bms_mlp,
                "functional_removed_mlp_ratio": functional_mlp,
                "mlp_ratio_difference": functional_mlp - bms_mlp,
            }
        )
    _write_csv(
        output_dir / "removed_unit_ratio_comparison.csv",
        (
            "target_sparsity",
            "bms_removed_head_ratio",
            "functional_removed_head_ratio",
            "head_ratio_difference",
            "bms_removed_mlp_ratio",
            "functional_removed_mlp_ratio",
            "mlp_ratio_difference",
        ),
        removed_ratio_rows,
    )

    coverage_rows = []
    for sparsity in SPARSITIES:
        values = [_as_float(row, "final_coverage") for row in domains[sparsity]]
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"Invalid domain coverage at sparsity {sparsity}")
        coverage_rows.append(
            {
                "target_sparsity": sparsity,
                "mean_final_domain_coverage": mean(values),
                "median_final_domain_coverage": median(values),
                "minimum_final_domain_coverage": min(values),
            }
        )
    _write_csv(
        output_dir / "coverage_summary.csv",
        (
            "target_sparsity",
            "mean_final_domain_coverage",
            "median_final_domain_coverage",
            "minimum_final_domain_coverage",
        ),
        coverage_rows,
    )

    dependency_rows = []
    for sparsity in SPARSITIES:
        changes = [_as_float(row, "change") for row in dependencies[sparsity]]
        dependency_rows.append(
            {
                "target_sparsity": sparsity,
                "examples": len(changes),
                "positive_examples": sum(value > 0.0 for value in changes),
                "mean_change": mean(changes) if changes else 0.0,
                "maximum_change": max(changes) if changes else 0.0,
            }
        )
    _write_csv(
        output_dir / "set_dependency_summary.csv",
        (
            "target_sparsity", "examples", "positive_examples",
            "mean_change", "maximum_change",
        ),
        dependency_rows,
    )

    aggregate_trace = []
    aggregate_domains = []
    aggregate_dependencies = []
    for sparsity in SPARSITIES:
        for collection, target in (
            (traces[sparsity], aggregate_trace),
            (domains[sparsity], aggregate_domains),
            (dependencies[sparsity], aggregate_dependencies),
        ):
            for row in collection:
                target.append({"target_sparsity": sparsity, **row})
    if aggregate_trace:
        _write_csv(
            output_dir / "functional_selection_trace.csv",
            tuple(aggregate_trace[0]),
            aggregate_trace,
        )
    if aggregate_domains:
        _write_csv(
            output_dir / "functional_domain_summary.csv",
            tuple(aggregate_domains[0]),
            aggregate_domains,
        )
    if aggregate_dependencies:
        _write_csv(
            output_dir / "set_dependency_examples.csv",
            tuple(aggregate_dependencies[0]),
            aggregate_dependencies,
        )

    for sparsity in SPARSITIES:
        bms_rows = {row["layer"]: row for row in layers[("bms", sparsity)]}
        functional_rows = {
            row["layer"]: row for row in layers[("functional", sparsity)]
        }
        if set(bms_rows) != set(functional_rows):
            raise ValueError(f"Layer set differs at sparsity {sparsity}")
        rows = []
        for layer in sorted(bms_rows):
            left = bms_rows[layer]
            right = functional_rows[layer]
            rows.append(
                {
                    "layer": layer,
                    "unit_type": left["unit_type"],
                    "bms_removed_units": int(left["removed_units"]),
                    "functional_removed_units": int(right["removed_units"]),
                    "functional_minus_bms": (
                        int(right["removed_units"]) - int(left["removed_units"])
                    ),
                }
            )
        _write_csv(
            output_dir / f"layer_pruning_diff_{_tag(sparsity)}.csv",
            (
                "layer", "unit_type", "bms_removed_units",
                "functional_removed_units", "functional_minus_bms",
            ),
            rows,
        )

    _plot(output_dir, metrics, domains, traces, dependencies)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="task014_functional_pruning")
    return parser


if __name__ == "__main__":
    analyze(Path(build_parser().parse_args().output_dir))

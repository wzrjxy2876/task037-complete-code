"""Analyze Task019 dynamic-ranking and Attention causal ablations."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import task019_dynamic_ranking_causal_ablation as task019
from task017_high_sparsity_diagnosis import TYPE_ATTENTION, TYPE_FFN
from task018_high_sparsity_transition import _endpoint_metric, _load_snapshot


def quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan, "Q25": math.nan, "Q75": math.nan, "Q90": math.nan}
    q25, median, q75, q90 = np.quantile(array, (0.25, 0.5, 0.75, 0.9))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(median),
        "Q25": float(q25),
        "Q75": float(q75),
        "Q90": float(q90),
    }


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _metric(path: Path) -> dict:
    value = task019.read_json(path)
    if value.get("status") != "PASS":
        raise RuntimeError(f"Incomplete validation: {path}")
    return value


def _dynamic_metric(task016_root: Path, task018_root: Path, target: float) -> dict:
    if math.isclose(target, 0.26):
        return _metric(Path(task018_root) / "validation/s26/metrics.json")
    value = task019.read_json(task019._task016_dir(Path(task016_root), 0.30) / "final_metrics.json")
    if value.get("status") != "task016_prune_only_complete":
        raise RuntimeError("Task016 dynamic 30% endpoint is incomplete")
    return {
        **value,
        "top1": _endpoint_metric(value, "top1"),
        "top5": _endpoint_metric(value, "top5"),
        "physical_numel_sparsity": 1.0 - int(value["parameters_after"]) / int(value["parameters_before"]),
    }


def _domain_map(rows: Sequence[Mapping[str, object]], target: float) -> dict[int, dict]:
    selected = {
        int(row["domain_id"]): dict(row)
        for row in rows
        if math.isclose(float(row["snapshot_target"]), target, abs_tol=1e-12)
    }
    if len(selected) != task019.EXPECTED_DOMAINS:
        raise RuntimeError(f"Task017 domain snapshot {target:.0%} is incomplete")
    return selected


def _dynamic_trace_row(row: Mapping[str, object], variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "step": int(row["step"]),
        "incremental_step": "",
        "estimated_sparsity_before": float(row["selection_sparsity_before"]),
        "estimated_sparsity_after": float(row["selection_sparsity_after"]),
        "global_index": int(row["global_index"]),
        "unit_type": row["unit_type"],
        "layer": row["layer"],
        "unit_index": "",
        "domain_id": int(row["domain_id"]),
        "functional_score": float(row["delta_total"]),
        "Delta_average": float(row["delta_average"]),
        "Delta_total": float(row["delta_total"]),
        "rank_at_selection": 1,
        "domain_retained_ratio_before": float(row["domain_retained_ratio_before"]),
        "domain_coverage_before": float(row["domain_coverage_before"]),
        "domain_coverage_after": float(row["domain_coverage_after"]),
        "best_substitute_similarity": float(row["best_remaining_similarity"]),
        "parameter_cost": int(row["parameter_cost"]),
        "cumulative_removed_parameters": "",
    }


def _summarize_variant(
    name: str,
    start: float,
    target: float,
    trace: Sequence[Mapping[str, object]],
    metric: Mapping[str, object],
    final_domains: Sequence[Mapping[str, object]],
    total_parameters: int,
    dynamic_top1: float,
    characteristics: Mapping[int, Mapping[str, float]],
) -> dict[str, object]:
    attention = [row for row in trace if row["unit_type"] == TYPE_ATTENTION]
    ffn = [row for row in trace if row["unit_type"] == TYPE_FFN]
    attention_cost = sum(int(row["parameter_cost"]) for row in attention)
    cost = sum(int(row["parameter_cost"]) for row in trace)
    indices = [int(row["global_index"]) for row in trace]
    mean_coverage = float(np.mean([float(row["coverage"]) for row in final_domains]))
    median_best = float(
        np.median([float(row["median_best_remaining_similarity"]) for row in final_domains])
    )
    estimated_sparsity = metric.get(
        "estimated_sparsity",
        metric.get("reported_estimated_sparsity", metric.get("estimated_budget_sparsity")),
    )
    if estimated_sparsity is None:
        raise KeyError("Validation metrics do not report estimated parameter sparsity")
    return {
        "row": name,
        "start_sparsity": start,
        "target_sparsity": target,
        "top1": float(metric["top1"]),
        "top5": float(metric["top5"]),
        "top1_vs_corresponding_dynamic": float(metric["top1"]) - dynamic_top1,
        "incremental_units": len(trace),
        "incremental_attention": len(attention),
        "incremental_ffn": len(ffn),
        "attention_budget_share": attention_cost / cost if cost else 0.0,
        "incremental_domains_touched": len({int(row["domain_id"]) for row in trace}),
        "incremental_layers_touched": len({str(row["layer"]) for row in trace}),
        "unique_domains_touched_incremental": len({int(row["domain_id"]) for row in trace}),
        "unique_layers_touched_incremental": len({str(row["layer"]) for row in trace}),
        "median_D_abs_increment": float(np.median([characteristics[index]["D_abs"] for index in indices])),
        "median_functional_energy_increment": float(
            np.median([characteristics[index]["functional_energy"] for index in indices])
        ),
        "mean_domain_coverage_final": mean_coverage,
        "median_best_substitute_similarity_final": median_best,
        "estimated_sparsity": float(estimated_sparsity),
        "physical_numel_sparsity": float(metric["physical_numel_sparsity"]),
        "incremental_parameter_cost": cost,
        "estimated_removed_parameters": int(round(float(estimated_sparsity) * total_parameters)),
    }


def _save_figures(
    output_dir: Path,
    summary: Sequence[Mapping[str, object]],
    differences: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    by_name = {str(row["row"]): row for row in summary}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(2)
    ax.bar(x - 0.18, [by_name["dynamic_26"]["top1"], by_name["dynamic_30"]["top1"]], 0.36, label="Dynamic")
    ax.bar(x + 0.18, [by_name["frozen_26"]["top1"], by_name["frozen_30"]["top1"]], 0.36, label="Frozen")
    ax.set_xticks(x, ["26%", "30%"]); ax.set_ylabel("Top-1 (%)"); ax.legend()
    save(fig, "figure1_dynamic_vs_frozen_top1")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    overlap = [row for row in differences if row["cohort"] == "common"]
    only_dynamic = [row for row in differences if row["cohort"] == "dynamic_only"]
    only_frozen = [row for row in differences if row["cohort"] == "frozen_only"]
    labels = sorted({str(row["interval"]) for row in differences})
    for rows, label, bottom_rows in ((overlap, "Common", None), (only_dynamic, "Dynamic only", overlap), (only_frozen, "Frozen only", None)):
        values = [next(float(row["unit_count"]) for row in rows if row["interval"] == key) for key in labels]
        if label == "Frozen only":
            ax.bar(np.arange(len(labels)) + 0.2, values, 0.35, label=label)
        else:
            bottom = None if bottom_rows is None else [next(float(row["unit_count"]) for row in bottom_rows if row["interval"] == key) for key in labels]
            ax.bar(np.arange(len(labels)) - 0.2, values, 0.35, bottom=bottom, label=label)
    ax.set_xticks(np.arange(len(labels)), labels); ax.set_ylabel("Incremental units"); ax.legend()
    save(fig, "figure2_incremental_set_overlap")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["dynamic_26", "frozen_26", "dynamic_30", "frozen_30"]
    attention = [float(by_name[name]["attention_budget_share"]) for name in names]
    ax.bar(names, attention, label="Attention"); ax.bar(names, 1 - np.asarray(attention), bottom=attention, label="FFN")
    ax.set_ylabel("Incremental parameter share"); ax.tick_params(axis="x", rotation=15); ax.legend()
    save(fig, "figure3_unit_type_parameter_composition")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = ["dynamic_30", "no_new_attention_dynamic_30"]
    ax.bar(["Original dynamic", "No new Attention"], [float(by_name[name]["top1"]) for name in names])
    ax.set_ylabel("Top-1 (%)")
    save(fig, "figure4_attention_counterfactual_top1")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    att = [float(by_name[name]["attention_budget_share"]) for name in names]
    ax.bar(["Original dynamic", "No new Attention"], att, label="Attention")
    ax.bar(["Original dynamic", "No new Attention"], 1 - np.asarray(att), bottom=att, label="FFN")
    ax.set_ylabel("Incremental parameter share"); ax.legend()
    save(fig, "figure5_attention_counterfactual_composition")

    for figure_number, field, ylabel, name in (
        (6, "domain_coverage_after", "Domain coverage", "domain_coverage_trajectory"),
        (7, "best_substitute_similarity", "Best substitute similarity", "best_substitute_trajectory"),
        (8, "rank_at_selection", "Rank at selection", "rank_trajectory"),
    ):
        del figure_number
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for variant in ("dynamic_24_26", "frozen_26", "dynamic_28_30", "frozen_30", "no_new_attention_dynamic_30"):
            rows = [row for row in traces if row["variant"] == variant]
            if rows:
                ax.plot(range(1, len(rows) + 1), [float(row[field]) for row in rows], label=variant, alpha=0.85)
        ax.set_xlabel("Incremental selection step"); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
        save(fig, f"figure{6 if field == 'domain_coverage_after' else 7 if field == 'best_substitute_similarity' else 8}_{name}")


def _write_diagnosis(output_dir: Path, summary: Sequence[Mapping[str, object]], jaccards: Mapping[str, float]) -> None:
    rows = {str(row["row"]): row for row in summary}
    gain26 = float(rows["frozen_26"]["top1"]) - float(rows["dynamic_26"]["top1"])
    gain30 = float(rows["frozen_30"]["top1"]) - float(rows["dynamic_30"]["top1"])
    attention_gain = float(rows["no_new_attention_dynamic_30"]["top1"]) - float(rows["dynamic_30"]["top1"])

    def direction(value: float) -> str:
        if value > 0:
            return "harmful (frozen is empirically better)"
        if value < 0:
            return "protective (dynamic is empirically better)"
        return "neutral at the observed precision"

    if attention_gain > 0:
        attention_label = "a contributor; its magnitude relative to the frozen gains determines whether it is primary or secondary"
    elif attention_gain < 0:
        attention_label = "not protective to preserve; the FFN replacement is empirically more harmful"
    else:
        attention_label = "a weak contributor at the observed precision"
    stronger = "dynamic ranking amplification" if max(gain26, gain30) > attention_gain else "additional Attention removal"
    text = f"""# Task019 causal ablation diagnosis

All values below are prune-only measurements under identical identity-checked artifacts. The Attention comparison is an empirical matched-budget counterfactual estimate, not an unqualified formal causal effect.

## Q1. Does freezing the 24% ranking improve the 26% model?

Observed Top-1 difference `frozen_26 - dynamic_26 = {gain26:.6g}` percentage points.

## Q2. Does freezing the 28% ranking improve the 30% model?

Observed Top-1 difference `frozen_30 - dynamic_30 = {gain30:.6g}` percentage points.

## Q3. Is dynamic re-ranking beneficial, neutral, or harmful?

For 24->26 it is **{direction(gain26)}**. For 28->30 it is **{direction(gain30)}**.

## Q4. How different are the selected sets?

Increment Jaccard is `{jaccards['24-26']:.6g}` for 24->26 and `{jaccards['28-30']:.6g}` for 28->30. Exact type/domain counts are in `dynamic_frozen_set_difference.csv`.

## Q5. Does dynamic ranking change functional domains/types?

Yes when the corresponding dynamic-only and frozen-only type/domain counts differ; the complete evidence is reported without assigning benefit from composition alone.

## Q6. Accuracy recovered without new Attention removal

`Top1(no_new_attention_dynamic_30) - Top1(original_dynamic_30) = {attention_gain:.6g}` percentage points at matched parameter budget.

## Q7. Is Attention primary, secondary, or weak?

The observed result classifies Attention as **{attention_label}**.

## Q8. Which FFN units/domains absorb the budget?

See `attention_replacement_trace.csv`; no one-to-one correspondence is imposed.

## Q9. Accuracy versus functional coverage

Compare Top-1 with `mean_domain_coverage_final` and `counterfactual_domain_state.csv`; preserving Attention can redirect loss into other FFN domains.

## Q10. More strongly supported hypothesis

By the observed Top-1 gains, **{stronger}** has the larger measured counterfactual difference. This is descriptive, not forced.

## Q11. Interaction

The interventions alter selected sets and touched domains, so interaction is plausible; Task019 does not identify a formal interaction coefficient.

## Q12. Next mechanism

Investigate the domains and FFN replacements responsible for the larger measured loss, using these traces. No new mechanism is implemented in Task019.
"""
    (Path(output_dir) / "diagnosis.md").write_text(text, encoding="utf-8")


def analyze(
    *,
    task014_root: Path,
    task015_root: Path,
    task016_root: Path,
    task017_root: Path,
    task018_root: Path,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    identity = task019.read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task019 artifact identity gate did not pass")
    if task019.production_source_sha256(Path(__file__).resolve().parent) != identity["production_source_sha256"]:
        raise RuntimeError("Protected production source changed before Task019 analysis")

    trace30, prefixes, _summaries, _extra = task019._prefix_identity(
        Path(task016_root), Path(task017_root), Path(task018_root)
    )
    replay_rows = task019.ordered_selected_rows(
        task019.read_csv(Path(task017_root) / "replay/domain_total/incremental_selection_risk_full.csv")
    )
    replay_by_index = {int(row["global_index"]): row for row in replay_rows}
    task017_domains = task019.read_csv(Path(task017_root) / "replay/domain_total/domain_snapshots.csv")
    dynamic_domain_26 = list(_domain_map(task017_domains, 0.26).values())
    dynamic_domain_30 = list(_domain_map(task017_domains, 0.30).values())

    from task015_attention_ffn_diagnosis import (
        _descriptor_path,
        _load_descriptors,
        _load_units,
        _mapping_paths,
        infer_parameter_costs,
    )

    unit_path, layer_path, _ = _mapping_paths(Path(task014_root))
    units = _load_units(unit_path)
    costs, _ = infer_parameter_costs(units, task019.read_csv(layer_path))
    descriptor_path = _descriptor_path(Path(task014_root), task019._task016_dir(Path(task016_root), 0.30))
    descriptors = _load_descriptors(descriptor_path, units)
    energy = np.load(Path(task015_root) / "functional_energy.npy", mmap_mode="r", allow_pickle=False)
    if energy.shape != (task019.EXPECTED_UNITS,):
        raise ValueError("Task015 functional energy cache has wrong shape")
    snapshot_path = Path(task017_root) / "replay/domain_total/candidate_snapshots.npz"
    p00 = _load_snapshot(snapshot_path, 0.0)
    domain_by_global = np.full(task019.EXPECTED_UNITS, -1, dtype=np.int64)
    domain_by_global[p00["global_index"]] = p00["domain_id"]
    domain_sizes = Counter(domain_by_global.tolist())
    characteristics = {
        index: {
            "D_abs": float(descriptors[index, 0]),
            "D_rel": float(descriptors[index, 1]),
            "D_dyn": float(descriptors[index, 2]),
            "functional_energy": float(energy[index]),
            "domain_size": float(domain_sizes[int(domain_by_global[index])]),
            "parameter_cost": float(costs[index]),
        }
        for index in range(task019.EXPECTED_UNITS)
    }

    dynamic_metrics = {
        0.26: _dynamic_metric(Path(task016_root), Path(task018_root), 0.26),
        0.30: _dynamic_metric(Path(task016_root), Path(task018_root), 0.30),
    }
    variant_metrics = {
        name: _metric(output_dir / "validation" / name / "metrics.json")
        for name in task019.VARIANTS
    }
    variant_traces = {
        name: task019.read_csv(output_dir / "variants" / name / "causal_selection_trace.csv")
        for name in task019.VARIANTS
    }
    variant_domains = {
        name: task019.read_csv(output_dir / "variants" / name / "domain_state.csv")
        for name in task019.VARIANTS
    }

    interval_specs = ((0.24, 0.26, "24-26"), (0.28, 0.30, "28-30"))
    dynamic_traces: dict[str, list[dict]] = {}
    for start, target, label in interval_specs:
        start_len, target_len = len(prefixes[start]), len(prefixes[target])
        rows = [
            _dynamic_trace_row(replay_by_index[int(row["global_index"])], f"dynamic_{label.replace('-', '_')}")
            for row in trace30[start_len:target_len]
        ]
        for number, row in enumerate(rows, start=1):
            row["incremental_step"] = number
        dynamic_traces[label] = rows

    summary_rows = []
    for target, start, label, final_domains in (
        (0.26, 0.24, "24-26", dynamic_domain_26),
        (0.30, 0.28, "28-30", dynamic_domain_30),
    ):
        summary_rows.append(
            _summarize_variant(
                f"dynamic_{int(target*100)}", start, target, dynamic_traces[label],
                dynamic_metrics[target], final_domains, int(identity["parameters_before"]),
                float(dynamic_metrics[target]["top1"]), characteristics,
            )
        )
        frozen_name = f"frozen_{int(target*100)}"
        summary_rows.append(
            _summarize_variant(
                frozen_name, start, target, variant_traces[frozen_name],
                variant_metrics[frozen_name], variant_domains[frozen_name],
                int(identity["parameters_before"]), float(dynamic_metrics[target]["top1"]),
                characteristics,
            )
        )
    summary_rows.append(
        _summarize_variant(
            "no_new_attention_dynamic_30", 0.28, 0.30,
            variant_traces["no_new_attention_dynamic_30"],
            variant_metrics["no_new_attention_dynamic_30"],
            variant_domains["no_new_attention_dynamic_30"],
            int(identity["parameters_before"]), float(dynamic_metrics[0.30]["top1"]),
            characteristics,
        )
    )
    task019.atomic_csv(output_dir / "task019_causal_summary.csv", tuple(summary_rows[0]), summary_rows)
    summary_by_name = {row["row"]: row for row in summary_rows}

    dynamic_vs_frozen = [row for row in summary_rows if row["row"] in {"dynamic_26", "frozen_26", "dynamic_30", "frozen_30"}]
    for row in dynamic_vs_frozen:
        row["mode"] = "dynamic" if str(row["row"]).startswith("dynamic") else "frozen"
        target = float(row["target_sparsity"])
        if row["mode"] == "dynamic":
            row["prefix_steps_total"] = len(prefixes[target])
        else:
            construction = task019.read_json(
                output_dir / "variants" / str(row["row"]) / "construction.json"
            )
            row["prefix_steps_total"] = int(construction["prefix_steps_total"])
        row["top1_difference_vs_dynamic"] = row["top1_vs_corresponding_dynamic"]
        row["set_jaccard_vs_dynamic_increment"] = 1.0
        row["sequence_sha256"] = task019.sequence_sha256(
            int(value["global_index"])
            for value in (
                dynamic_traces["24-26"] if row["row"] == "dynamic_26" else
                dynamic_traces["28-30"] if row["row"] == "dynamic_30" else
                variant_traces[str(row["row"])]
            )
        )

    difference_rows = []
    statistics_rows = []
    jaccards: dict[str, float] = {}
    for label, frozen_name in (("24-26", "frozen_26"), ("28-30", "frozen_30")):
        dynamic = dynamic_traces[label]
        frozen = variant_traces[frozen_name]
        dynamic_indices = {int(row["global_index"]) for row in dynamic}
        frozen_indices = {int(row["global_index"]) for row in frozen}
        jaccards[label] = jaccard(dynamic_indices, frozen_indices)
        for row in dynamic_vs_frozen:
            if str(row["row"]).endswith(label[-2:]):
                row["set_jaccard_vs_dynamic_increment"] = jaccards[label]
        row_lookup = {
            int(row["global_index"]): row for row in list(dynamic) + list(frozen)
        }
        for cohort, indices in (
            ("common", dynamic_indices & frozen_indices),
            ("dynamic_only", dynamic_indices - frozen_indices),
            ("frozen_only", frozen_indices - dynamic_indices),
        ):
            rows = [row_lookup[index] for index in indices]
            difference_rows.append(
                {
                    "interval": label,
                    "cohort": cohort,
                    "unit_count": len(rows),
                    "attention_count": sum(row["unit_type"] == TYPE_ATTENTION for row in rows),
                    "ffn_count": sum(row["unit_type"] == TYPE_FFN for row in rows),
                    "parameter_cost": sum(int(row["parameter_cost"]) for row in rows),
                    "domain_count": len({int(row["domain_id"]) for row in rows}),
                    "layer_count": len({str(row["layer"]) for row in rows}),
                    "set_jaccard": jaccards[label],
                }
            )
            if cohort in {"dynamic_only", "frozen_only"}:
                start = 0.24 if label == "24-26" else 0.28
                snap = _load_snapshot(snapshot_path, start)
                snap_lookup = {
                    int(index): offset for offset, index in enumerate(snap["global_index"])
                }
                for quantity in (
                    "D_abs", "D_rel", "D_dyn", "functional_energy", "domain_size",
                    "best_substitute_similarity_at_start", "coverage_at_start", "Delta_total_at_start", "parameter_cost",
                ):
                    values = []
                    for index in indices:
                        if quantity in characteristics[index]:
                            values.append(characteristics[index][quantity])
                        elif index in snap_lookup:
                            offset = snap_lookup[index]
                            source = {
                                "best_substitute_similarity_at_start": "best_similarity",
                                "coverage_at_start": "coverage",
                                "Delta_total_at_start": "delta_total",
                            }[quantity]
                            values.append(float(snap[source][offset]))
                    statistics_rows.append(
                        {"interval": label, "cohort": cohort, "quantity": quantity, **quantiles(values)}
                    )
    task019.atomic_csv(output_dir / "dynamic_vs_frozen.csv", tuple(dynamic_vs_frozen[0]), dynamic_vs_frozen)
    task019.atomic_csv(output_dir / "dynamic_frozen_set_difference.csv", tuple(difference_rows[0]), difference_rows)
    task019.atomic_csv(output_dir / "dynamic_frozen_unit_statistics.csv", tuple(statistics_rows[0]), statistics_rows)

    attention_rows = [summary_by_name["dynamic_30"], summary_by_name["no_new_attention_dynamic_30"]]
    for row in attention_rows:
        row["mode"] = "original_dynamic_30" if row["row"] == "dynamic_30" else "no_new_attention_dynamic_30"
        row["estimated_parameter_sparsity"] = row["estimated_sparsity"]
        row["incremental_attention_parameter_cost"] = round(float(row["attention_budget_share"]) * int(row["incremental_parameter_cost"]))
        row["incremental_ffn_parameter_cost"] = int(row["incremental_parameter_cost"]) - int(row["incremental_attention_parameter_cost"])
        row["budget_overshoot"] = float(row["estimated_sparsity"]) * int(identity["parameters_before"]) - 0.30 * int(identity["parameters_before"])
    task019.atomic_csv(output_dir / "attention_counterfactual.csv", tuple(attention_rows[0]), attention_rows)

    dynamic28_30 = dynamic_traces["28-30"]
    no_attention = variant_traces["no_new_attention_dynamic_30"]
    dynamic_indices = {int(row["global_index"]): int(row["step"]) for row in dynamic28_30}
    replacement = [row for row in no_attention if int(row["global_index"]) not in dynamic_indices]
    cumulative = 0
    replacement_rows = []
    for step, row in enumerate(replacement, start=1):
        cumulative += int(row["parameter_cost"])
        replacement_rows.append(
            {
                "counterfactual_step": step,
                "selected_ffn_global_index": int(row["global_index"]),
                "layer": row["layer"],
                "domain_id": int(row["domain_id"]),
                "unit_index": int(row["unit_index"]),
                "delta_total": float(row["Delta_total"]),
                "parameter_cost": int(row["parameter_cost"]),
                "cumulative_counterfactual_cost": cumulative,
                "corresponding_original_dynamic_step_if_any": dynamic_indices.get(int(row["global_index"]), ""),
            }
        )
    fields = tuple(replacement_rows[0]) if replacement_rows else (
        "counterfactual_step", "selected_ffn_global_index", "layer", "domain_id", "unit_index", "delta_total", "parameter_cost", "cumulative_counterfactual_cost", "corresponding_original_dynamic_step_if_any"
    )
    task019.atomic_csv(output_dir / "attention_replacement_trace.csv", fields, replacement_rows)

    domain_state_rows = []
    domains_before_28 = {int(row["domain_id"]) for row in prefixes[0.28]}
    for mode, rows, selection_rows in (
        ("original_dynamic_30", dynamic_domain_30, dynamic28_30),
        (
            "no_new_attention_dynamic_30",
            variant_domains["no_new_attention_dynamic_30"],
            no_attention,
        ),
    ):
        counts = Counter(int(row["domain_id"]) for row in selection_rows)
        total = sum(counts.values())
        hhi = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
        domain_state_rows.append(
            {
                "mode": mode,
                "row_type": "summary",
                "domain_id": "",
                "new_domains_touched": len(set(counts) - domains_before_28),
                "domain_removal_concentration_hhi": hhi,
                "mean_coverage_drop": float(
                    np.mean(
                        [
                            float(row.get("coverage_drop", row.get("coverage_drop_from_initial")))
                            for row in rows
                        ]
                    )
                ),
                "median_retained_ratio": float(
                    np.median([float(row["retained_ratio"]) for row in rows])
                ),
                "median_best_substitute_similarity": float(
                    np.median([float(row["median_best_remaining_similarity"]) for row in rows])
                ),
                "retained_ratio": "",
                "coverage": "",
                "coverage_drop": "",
            }
        )
        for row in rows:
            domain_state_rows.append(
                {
                    "mode": mode,
                    "row_type": "domain",
                    "domain_id": int(row["domain_id"]),
                    "new_domains_touched": "",
                    "domain_removal_concentration_hhi": "",
                    "mean_coverage_drop": "",
                    "median_retained_ratio": "",
                    "retained_ratio": float(row["retained_ratio"]),
                    "coverage": float(row["coverage"]),
                    "coverage_drop": float(row.get("coverage_drop", row.get("coverage_drop_from_initial"))),
                    "median_best_substitute_similarity": float(row["median_best_remaining_similarity"]),
                }
            )
    task019.atomic_csv(output_dir / "counterfactual_domain_state.csv", tuple(domain_state_rows[0]), domain_state_rows)

    all_traces = dynamic_traces["24-26"] + dynamic_traces["28-30"]
    all_traces += [dict(row) for rows in variant_traces.values() for row in rows]
    task019.atomic_csv(output_dir / "causal_selection_trace.csv", task019.TRACE_FIELDS, all_traces)
    _save_figures(output_dir, summary_rows, difference_rows, all_traces)
    _write_diagnosis(output_dir, summary_rows, jaccards)

    production_unchanged = task019.production_source_sha256(Path(__file__).resolve().parent) == identity["production_source_sha256"]
    completion = {
        "artifact_identity_pass": True,
        "dynamic_24_26_reproduction_pass": bool(identity["dynamic_24_26_reproduction_pass"]),
        "dynamic_28_30_reproduction_pass": bool(identity["dynamic_28_30_reproduction_pass"]),
        "frozen_26_validation_complete": variant_metrics["frozen_26"]["status"] == "PASS",
        "frozen_30_validation_complete": variant_metrics["frozen_30"]["status"] == "PASS",
        "no_new_attention_30_validation_complete": variant_metrics["no_new_attention_dynamic_30"]["status"] == "PASS",
        "no_new_attention_added_attention_heads": int(summary_by_name["no_new_attention_dynamic_30"]["incremental_attention"]),
        "budget_semantics_unchanged": True,
        "production_pruning_code_modified": not production_unchanged,
        "analysis_complete": True,
        "fine_tuning_executed": False,
        "status": "PASS" if production_unchanged and int(summary_by_name["no_new_attention_dynamic_30"]["incremental_attention"]) == 0 else "FAIL",
    }
    if completion["status"] != "PASS":
        raise RuntimeError(f"Task019 completion gate failed: {completion}")
    task019.atomic_json(output_dir / "task019_completion.json", completion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task014-root", type=Path, required=True)
    parser.add_argument("--task015-root", type=Path, required=True)
    parser.add_argument("--task016-root", type=Path, required=True)
    parser.add_argument("--task017-root", type=Path, required=True)
    parser.add_argument("--task018-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

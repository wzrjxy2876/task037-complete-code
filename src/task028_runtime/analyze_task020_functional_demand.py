"""Analyze Task020 fixed-28%-state task-importance ablations."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import task020_functional_demand_task_importance as task020

GROUPS = (task020.GROUP_ATTENTION, task020.GROUP_REPLACEMENT,
          task020.GROUP_ORDINARY, task020.GROUP_CONTROL)
GROUP_LABELS = ("Attention", "Replacement FFN", "Ordinary deleted FFN", "Matched surviving FFN")
TASK_FIELDS = ("mean_true_logit_drop", "mean_margin_drop", "mean_ce_increase", "mean_kl")


def finite(values) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values) -> dict[str, float | int]:
    array = finite(values)
    if not array.size:
        return {"count": 0, **{name: math.nan for name in ("mean", "median", "q25", "q75", "q90")}}
    return {"count": int(array.size), "mean": float(array.mean()),
            "median": float(np.median(array)), "q25": float(np.quantile(array, .25)),
            "q75": float(np.quantile(array, .75)), "q90": float(np.quantile(array, .90))}


def ratio(left: float, right: float) -> float:
    return float(left / right) if math.isfinite(right) and abs(right) > 1e-15 else math.nan


def percentile(values: Sequence[float]) -> np.ndarray:
    ranks = task020._rankdata(values)
    return ranks / max(1, len(ranks) - 1)


def authoritative_joint_indices(cohort_rows, cohort: str) -> list[int]:
    """Return a cohort's order-sensitive sequence from the CSV file order.

    Worker rows are sorted later for analysis tables, but joint cache identity
    must follow the authoritative order in ``task020_cohorts.csv``.
    """
    return [int(row["global_index"]) for row in cohort_rows if row["cohort"] == cohort]


def write_figures(output_dir: Path, rows, domains, joint_rows, full_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = output_dir / "figures"; root.mkdir(parents=True, exist_ok=True)
    colors = ("#D55E00", "#0072B2", "#E69F00", "#009E73")
    grouped = {group: [row for row in rows if row["group"] == group] for group in GROUPS}

    def save(fig, name):
        fig.tight_layout(); fig.savefig(root / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

    for number, field, ylabel in ((1, "mean_ce_increase", "Mean CE increase"),
                                   (2, "mean_margin_drop", "Mean margin drop"),
                                   (3, "correct_to_wrong_flip_rate", "Correct-to-wrong flip rate")):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        box = ax.boxplot([[float(row[field]) for row in grouped[group]] for group in GROUPS],
                         labels=GROUP_LABELS, patch_artist=True, showfliers=True)
        for body, color in zip(box["boxes"], colors): body.set_facecolor(color); body.set_alpha(.65)
        ax.set_ylabel(ylabel); save(fig, f"figure{number}_{field}_groups")

    for number, xfield, xlabel in ((4, "delta_total_at_28", "Delta_total at 28%"),
                                    (5, "best_substitute_similarity_at_28", "Best substitute similarity at 28%"),
                                    (6, "D_abs", "D_abs"), (7, "functional_energy", "Functional energy")):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for group, label, color in zip(GROUPS, GROUP_LABELS, colors):
            subset = grouped[group]
            ax.scatter([float(row[xfield]) for row in subset], [float(row["mean_ce_increase"]) for row in subset],
                       s=22, alpha=.7, label=label, color=color)
        ax.set_xlabel(xlabel); ax.set_ylabel("Mean CE increase"); ax.legend(fontsize=8)
        save(fig, f"figure{number}_{xfield}_vs_task_sensitivity")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for group, label, color in zip(GROUPS, GROUP_LABELS, colors):
        subset = grouped[group]
        ax.scatter([float(row["functional_risk_percentile"]) for row in subset],
                   [float(row["task_risk_percentile"]) for row in subset], s=22, alpha=.7, label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1); ax.set_xlabel("Functional-risk percentile")
    ax.set_ylabel("Task-risk percentile"); ax.legend(fontsize=8); save(fig, "figure8_functional_vs_task_risk_percentile")

    attention = grouped[task020.GROUP_ATTENTION]
    fig, ax = plt.subplots(figsize=(6.8, 4.5)); x = np.arange(len(attention)); width = .36
    ax.bar(x - width / 2, [float(row["mean_ce_increase"]) for row in attention], width, label="CE increase")
    ax.bar(x + width / 2, [float(row["mean_margin_drop"]) for row in attention], width, label="Margin drop")
    ax.set_xticks(x, [str(row["global_index"]) for row in attention]); ax.set_xlabel("Attention global index")
    ax.legend(); save(fig, "figure9_attention_case_studies")

    selected = sorted(domains, key=lambda row: abs(float(row["mean_task_sensitivity"])), reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(range(len(selected)), [float(row["mean_task_sensitivity"]) for row in selected], color="#56B4E9")
    ax.set_xticks(range(len(selected)), [str(row["domain_id"]) for row in selected], rotation=70)
    ax.set_xlabel("BMS domain"); ax.set_ylabel("Mean task sensitivity"); save(fig, "figure10_domain_task_importance")

    labels = ["3 Attention", "Replacement FFN"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(labels, [float(row["top1_drop"]) for row in joint_rows], color=(colors[0], colors[1]))
    ax.set_ylabel("Matched-budget Top-1 drop"); save(fig, "figureA_matched_budget_joint_top1")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(labels, [float(row["mean_ce_increase"]) for row in joint_rows], color=(colors[0], colors[1]))
    ax.set_ylabel("Matched-budget mean CE increase"); save(fig, "figureB_matched_budget_joint_ce")
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    box = ax.boxplot([[float(row["ce_increase_per_parameter"]) for row in grouped[group]] for group in GROUPS],
                     labels=GROUP_LABELS, patch_artist=True)
    for body, color in zip(box["boxes"], colors): body.set_facecolor(color); body.set_alpha(.65)
    ax.set_ylabel("CE increase per parameter"); ax.tick_params(axis="x", rotation=15)
    save(fig, "figureC_single_unit_ce_per_parameter")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ffn_groups = (task020.GROUP_ORDINARY, task020.GROUP_CONTROL)
    ax.boxplot([[float(row["mean_ce_increase"]) for row in grouped[group]] for group in ffn_groups],
               labels=("Ordinary deleted", "Matched surviving"), patch_artist=True)
    ax.set_ylabel("Mean CE increase"); save(fig, "figureD_deleted_vs_surviving_ffn")
    individual_full = [row for row in full_rows if str(row["variant"]).startswith("attention_")
                       and row["variant"] != "attention_joint"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar([str(row["variant"]).replace("attention_", "") for row in individual_full],
           [float(row["top1_drop_from_28"]) for row in individual_full], color=colors[0])
    ax.set_xlabel("Attention global index"); ax.set_ylabel("Full-validation Top-1 drop")
    save(fig, "figureE_attention_full_validation_top1_drop")


def analyze(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    identity = task020.read_json(output_dir / "artifact_identity.json")
    cohorts = task020.read_json(output_dir / "cohort_reconstruction.json")
    baseline = task020.read_json(output_dir / "baseline_28_cache.json")
    if identity.get("code_version") != task020.CODE_VERSION or cohorts.get("code_version") != task020.CODE_VERSION:
        raise RuntimeError("Task020 artifacts were produced by a stale code version")
    if identity.get("artifact_identity_pass") is not True or cohorts.get("cohorts_reconstructed") is not True:
        raise RuntimeError("Task020 identity/cohort gate is incomplete")
    if baseline.get("baseline_28_cached") is not True or not task020.baseline_cache_valid(output_dir):
        raise RuntimeError("Task020 exact-28% baseline is missing or stale")
    source_commit_identity = task020.production_source_git_identity(Path(__file__).resolve().parent)
    recorded_identity = identity.get("production_source_git_blob_sha")
    if recorded_identity is not None and source_commit_identity != recorded_identity:
        raise RuntimeError("Protected production Git identity changed during Task020")

    rows, samples = [], []
    for worker in (0, 1):
        gate = task020.read_json(output_dir / "workers" / f"worker{worker}_completion.json")
        if not task020.result_cache_valid(output_dir / "workers" / f"worker{worker}_completion.json",
                                          output_dir, worker=worker, workers=2):
            raise RuntimeError(f"Task020 worker {worker} incomplete or stale")
        rows += task020.read_csv(output_dir / "workers" / f"worker{worker}_unit_task_importance.csv")
        samples += task020.read_csv(output_dir / "workers" / f"worker{worker}_attention_task_effect_per_sample.csv")
    expected = task020.read_csv(output_dir / "task020_cohorts.csv")
    observed_indices = [int(row["global_index"]) for row in rows]
    expected_indices = [int(row["global_index"]) for row in expected]
    if (len(observed_indices) != len(set(observed_indices)) or
            set(observed_indices) != set(expected_indices)):
        raise RuntimeError("Worker unit set differs from reconstructed cohorts")
    attention_authoritative_indices = authoritative_joint_indices(expected, task020.GROUP_ATTENTION)
    replacement_authoritative_indices = authoritative_joint_indices(expected, task020.GROUP_REPLACEMENT)
    sample_count = int(task020.read_json(output_dir / "baseline_28_cache.json")["sample_count"])
    task020.validate_attention_sample_counts(samples, attention_authoritative_indices, sample_count)
    rows.sort(key=lambda row: (GROUPS.index(str(row["cohort"])), int(row["global_index"])))
    for row in rows: row["group"] = row.pop("cohort")
    fp = percentile([float(row["delta_total_at_28"]) for row in rows])
    tp = percentile([float(row["mean_ce_increase"]) for row in rows])
    for offset, row in enumerate(rows):
        row["functional_risk_percentile"] = float(fp[offset]); row["task_risk_percentile"] = float(tp[offset])
        row["task_sensitivity"] = float(row["mean_ce_increase"])
    task020.atomic_csv(output_dir / "unit_task_importance.csv", tuple(rows[0]), rows)
    if samples:
        samples.sort(key=lambda row: (int(row["global_index"]), int(row["video_id"])))
        task020.atomic_csv(output_dir / "attention_task_effect_per_sample.csv", tuple(samples[0]), samples)

    group_stats = []
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        for field in TASK_FIELDS + ("prediction_flip_rate", "correct_to_wrong_flip_rate",
                                    "true_logit_drop_per_parameter", "margin_drop_per_parameter",
                                    "ce_increase_per_parameter", "kl_per_parameter"):
            group_stats.append({"group": group, "metric": field, **stats(float(row[field]) for row in subset)})
    task020.atomic_csv(output_dir / "group_task_importance_statistics.csv", tuple(group_stats[0]), group_stats)
    normalized_rows = [{
        "global_index": row["global_index"], "group": row["group"], "unit_type": row["unit_type"],
        "parameter_cost": row["parameter_cost"],
        **{name: row[name] for name in ("true_logit_drop_per_parameter", "margin_drop_per_parameter",
                                       "ce_increase_per_parameter", "kl_per_parameter")},
    } for row in rows]
    task020.atomic_csv(output_dir / "parameter_normalized_task_importance.csv",
                       tuple(normalized_rows[0]), normalized_rows)

    correlation_rows = []
    populations = (("all_investigated_units", rows, "diagnostic"),
                   ("investigated_ffn_only", [row for row in rows if row["unit_type"] == task020.TYPE_FFN], "diagnostic"),
                   ("three_attention_heads_descriptive_only", [row for row in rows if row["unit_type"] == task020.TYPE_ATTENTION], "descriptive_only"))
    for population, subset, interpretation in populations:
        for feature in ("delta_total_at_28", "best_substitute_similarity_at_28", "functional_energy"):
            for metric in ("mean_ce_increase", "mean_margin_drop"):
                correlation_rows.append({"population": population, "functional_metric": feature,
                    "task_metric": metric, "num_units": len(subset), "interpretation": interpretation,
                    "spearman_rho": task020.spearman([float(row[feature]) for row in subset],
                                                      [float(row[metric]) for row in subset])})
    task020.atomic_csv(output_dir / "functional_vs_task_correlation.csv", tuple(correlation_rows[0]), correlation_rows)

    descriptor_rows = []
    for descriptor in ("D_abs", "D_rel", "D_dyn"):
        for metric in ("mean_ce_increase", "mean_margin_drop", "mean_true_logit_drop"):
            descriptor_rows.append({"descriptor": descriptor, "task_metric": metric, "num_units": len(rows),
                "spearman_rho": task020.spearman([float(row[descriptor]) for row in rows],
                                                  [float(row[metric]) for row in rows])})
    task020.atomic_csv(output_dir / "descriptor_task_correlation.csv", tuple(descriptor_rows[0]), descriptor_rows)

    mismatch = []
    for row in rows:
        f, t = float(row["functional_risk_percentile"]), float(row["task_risk_percentile"]); category = "middle"
        if f <= .25 and t >= .75: category = "low_functional_high_task"
        elif f >= .75 and t <= .25: category = "high_functional_low_task"
        mismatch.append({"global_index": row["global_index"], "group": row["group"],
                         "functional_risk_percentile": f, "task_risk_percentile": t, "descriptive_bin": category})
    task020.atomic_csv(output_dir / "functional_task_mismatch.csv", tuple(mismatch[0]), mismatch)

    attention = [row for row in rows if row["group"] == task020.GROUP_ATTENTION]
    replacements = [row for row in rows if row["group"] == task020.GROUP_REPLACEMENT]
    matched_rows, case_rows = [], []
    for head in attention:
        closest = min(replacements, key=lambda row: (abs(float(row["delta_total_at_28"]) - float(head["delta_total_at_28"])),
            abs(float(row["best_substitute_similarity_at_28"]) - float(head["best_substitute_similarity_at_28"])), int(row["global_index"])))
        matched_rows.append({"attention_global_index": head["global_index"], "ffn_global_index": closest["global_index"],
            "attention_delta_total": head["delta_total_at_28"], "ffn_delta_total": closest["delta_total_at_28"],
            "attention_best_similarity": head["best_substitute_similarity_at_28"], "ffn_best_similarity": closest["best_substitute_similarity_at_28"],
            "attention_mean_ce_increase": head["mean_ce_increase"], "ffn_mean_ce_increase": closest["mean_ce_increase"],
            "attention_mean_margin_drop": head["mean_margin_drop"], "ffn_mean_margin_drop": closest["mean_margin_drop"]})
        case_rows.append({"global_index": head["global_index"], "layer": head["layer"], "domain_id": head["domain_id"],
            "delta_total_at_28": head["delta_total_at_28"], "best_substitute_similarity_at_28": head["best_substitute_similarity_at_28"],
            "functional_rank_at_28": head["functional_rank_at_28"], "mean_ce_increase": head["mean_ce_increase"],
            "mean_margin_drop": head["mean_margin_drop"], "prediction_flip_rate": head["prediction_flip_rate"],
            "matched_ffn_global_index": closest["global_index"]})
    task020.atomic_csv(output_dir / "matched_functional_risk_comparison.csv", tuple(matched_rows[0]), matched_rows)
    task020.atomic_csv(output_dir / "attention_case_studies.csv", tuple(case_rows[0]), case_rows)

    class_groups = defaultdict(list)
    for row in samples: class_groups[(int(row["global_index"]), int(row["label"]))].append(row)
    class_rows = []
    for (index, class_id), values in sorted(class_groups.items()):
        class_rows.append({"global_index": index, "class_id": class_id, "num_samples": len(values),
            "mean_ce_increase": float(np.mean([float(row["ce_increase"]) for row in values])),
            "mean_margin_drop": float(np.mean([float(row["margin_drop"]) for row in values])),
            "correct_to_wrong_flip_rate": float(np.mean([str(row["baseline_correct"]).lower() == "true" and
                str(row["ablated_correct"]).lower() != "true" for row in values]))})
    task020.atomic_csv(output_dir / "attention_class_sensitivity.csv", tuple(class_rows[0]), class_rows)

    domain_groups = defaultdict(list)
    for row in rows: domain_groups[int(row["domain_id"])].append(row)
    domain_rows = []
    for domain_id, values in sorted(domain_groups.items()):
        domain_rows.append({"domain_id": domain_id, "num_investigated_units": len(values),
            "mean_task_sensitivity": float(np.mean([float(row["mean_ce_increase"]) for row in values])),
            "median_task_sensitivity": float(np.median([float(row["mean_ce_increase"]) for row in values])),
            "mean_delta_total": float(np.mean([float(row["delta_total_at_28"]) for row in values])),
            "coverage_at_28": float(np.mean([float(row["domain_coverage_at_28"]) for row in values])),
            "best_substitute_similarity": float(np.mean([float(row["best_substitute_similarity_at_28"]) for row in values]))})
    task020.atomic_csv(output_dir / "investigated_domain_task_sensitivity.csv", tuple(domain_rows[0]), domain_rows)
    task020.atomic_csv(output_dir / "domain_task_importance.csv", tuple(domain_rows[0]), domain_rows)

    deleted_surviving = []
    for group in (task020.GROUP_ORDINARY, task020.GROUP_CONTROL):
        values = [row for row in rows if row["group"] == group]
        deleted_surviving.append({"group": group, "num_units": len(values),
            **{f"median_{field}": float(np.median([float(row[field]) for row in values])) for field in (
                "delta_total_at_28", "best_substitute_similarity_at_28", "D_abs", "D_rel", "D_dyn",
                "functional_energy", "mean_ce_increase", "mean_margin_drop", "prediction_flip_rate",
                "ce_increase_per_parameter")}})
    task020.atomic_csv(output_dir / "deleted_vs_surviving_ffn_comparison.csv",
                       tuple(deleted_surviving[0]), deleted_surviving)

    ja = task020.read_json(output_dir / "joint" / f"{task020.GROUP_ATTENTION}.json")
    jf = task020.read_json(output_dir / "joint" / f"{task020.GROUP_REPLACEMENT}.json")
    if (not task020.result_cache_valid(output_dir / "joint" / f"{task020.GROUP_ATTENTION}.json", output_dir,
            cohort=task020.GROUP_ATTENTION,
            cohort_sequence_sha256=task020.sequence_sha256(attention_authoritative_indices)) or
        not task020.result_cache_valid(output_dir / "joint" / f"{task020.GROUP_REPLACEMENT}.json", output_dir,
            cohort=task020.GROUP_REPLACEMENT,
            cohort_sequence_sha256=task020.sequence_sha256(replacement_authoritative_indices))):
        raise RuntimeError("Required joint ablation is incomplete or stale")
    budget_gate = task020.read_json(output_dir / "joint_budget_identity.json")
    if budget_gate.get("joint_parameter_budget_verified") is not True or budget_gate.get("code_version") != task020.CODE_VERSION:
        raise RuntimeError("Task020 exact joint parameter-budget identity failed")
    attention_joint = {**ja,
        "sum_individual_mean_ce_increase": float(sum(float(row["mean_ce_increase"]) for row in attention)),
        "ce_interaction_residual": float(ja["mean_ce_increase"]) - sum(float(row["mean_ce_increase"]) for row in attention),
        "sum_individual_mean_margin_drop": float(sum(float(row["mean_margin_drop"]) for row in attention)),
        "margin_interaction_residual": float(ja["mean_margin_drop"]) - sum(float(row["mean_margin_drop"]) for row in attention)}
    task020.atomic_csv(output_dir / "attention_joint_ablation.csv", tuple(attention_joint), [attention_joint])
    task020.atomic_csv(output_dir / "replacement_ffn_joint_ablation.csv", tuple(jf), [jf])

    budget_rows = {row["group"]: row for row in task020.read_csv(output_dir / "joint_budget_identity.csv")}
    joint_rows = []
    for name, result, budget in (("three_attention_heads", ja, budget_rows["three_attention_heads"]),
                                 ("replacement_ffn_set", jf, budget_rows["replacement_ffn_set"])):
        joint_rows.append({"group": name, "num_units": result["unit_count"],
            "parameter_cost": budget["parameter_cost"],
            **{field: result[field] for field in ("mean_true_logit_drop", "mean_margin_drop", "mean_ce_increase",
                "mean_kl", "prediction_flip_rate", "correct_to_wrong_flip_rate", "baseline_top1", "ablated_top1",
                "baseline_top5", "ablated_top5")},
            "top1_drop": float(result["baseline_top1"]) - float(result["ablated_top1"]),
            "top5_drop": float(result["baseline_top5"]) - float(result["ablated_top5"])})
    task020.atomic_csv(output_dir / "matched_budget_joint_comparison.csv", tuple(joint_rows[0]), joint_rows)
    joint_effect = {
        "attention_minus_replacement_top1_drop": float(joint_rows[0]["top1_drop"]) - float(joint_rows[1]["top1_drop"]),
        "attention_minus_replacement_ce_increase": float(joint_rows[0]["mean_ce_increase"]) - float(joint_rows[1]["mean_ce_increase"]),
        "attention_to_replacement_ce_ratio": ratio(float(joint_rows[0]["mean_ce_increase"]),
                                                     float(joint_rows[1]["mean_ce_increase"])),
        "joint_parameter_budget_verified": True,
    }
    task020.atomic_json(output_dir / "matched_budget_joint_effect.json", joint_effect)

    full_variants = ["baseline_28"] + [f"attention_{row['global_index']}" for row in attention] + [
        "attention_joint", "replacement_ffn_joint"]
    full_payloads = []
    for variant in full_variants:
        if not task020.full_validation_cache_valid(output_dir, variant):
            raise RuntimeError(f"Task020 full validation is incomplete or stale: {variant}")
        full_payloads.append(task020.read_json(output_dir / "full_validation" / f"{variant}.json"))
    full_baseline = full_payloads[0]
    full_rows = []
    for value in full_payloads:
        full_rows.append({**value,
            "top1_drop_from_28": float(full_baseline["top1"]) - float(value["top1"]),
            "top5_drop_from_28": float(full_baseline["top5"]) - float(value["top5"])})
    fields = ("variant", "num_additional_units", "attention_units", "ffn_units", "additional_parameter_cost",
              "top1", "top5", "top1_drop_from_28", "top5_drop_from_28", "checkpoint_sha256",
              "s28_prefix_sha256", "validation_samples", "validation_split", "amp_enabled")
    task020.atomic_csv(output_dir / "full_validation_task_importance.csv", fields, full_rows)
    task020.atomic_json(output_dir / "full_validation_completion.json", {
        "status": "PASS", "code_version": task020.CODE_VERSION,
        "full_validation_baseline_complete": True,
        "full_validation_individual_attention_complete": len(attention) == 3,
        "full_validation_joint_attention_complete": True,
        "full_validation_joint_replacement_complete": True,
        "variants": full_variants,
    })

    summary_rows = []
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        summary_rows.append({"group": group, "num_units": len(subset),
            "median_parameter_cost": float(np.median([float(row["parameter_cost"]) for row in subset])),
            "median_delta_total": float(np.median([float(row["delta_total_at_28"]) for row in subset])),
            "median_best_substitute_similarity": float(np.median([float(row["best_substitute_similarity_at_28"]) for row in subset])),
            "median_functional_energy": float(np.median([float(row["functional_energy"]) for row in subset])),
            "median_true_logit_drop": float(np.median([float(row["mean_true_logit_drop"]) for row in subset])),
            "median_margin_drop": float(np.median([float(row["mean_margin_drop"]) for row in subset])),
            "median_ce_increase": float(np.median([float(row["mean_ce_increase"]) for row in subset])),
            "median_kl": float(np.median([float(row["mean_kl"]) for row in subset])),
            "median_ce_increase_per_parameter": float(np.median([float(row["ce_increase_per_parameter"]) for row in subset])),
            "median_margin_drop_per_parameter": float(np.median([float(row["margin_drop_per_parameter"]) for row in subset])),
            "prediction_flip_rate": float(np.mean([float(row["prediction_flip_rate"]) for row in subset])),
            "correct_to_wrong_flip_rate": float(np.mean([float(row["correct_to_wrong_flip_rate"]) for row in subset]))})
    task020.atomic_csv(output_dir / "task020_summary.csv", tuple(summary_rows[0]), summary_rows)
    write_figures(output_dir, rows, domain_rows, joint_rows, full_rows)

    by_group = {row["group"]: row for row in summary_rows}; a = by_group[task020.GROUP_ATTENTION]; b = by_group[task020.GROUP_REPLACEMENT]
    rho = next(row["spearman_rho"] for row in correlation_rows if row["population"] == "all_investigated_units" and
               row["functional_metric"] == "delta_total_at_28" and row["task_metric"] == "mean_ce_increase")
    diagnosis = f"""# Task020 diagnosis

All interventions use the exact identity-gated 28% state. Task020 is diagnosis only.

## Evidence hierarchy

1. **Primary:** exact Task019 matched-parameter-budget joint intervention: three Attention heads versus the exact replacement-FFN set. Attention joint Top-1 drop is `{float(joint_rows[0]['top1_drop']):.8g}` and replacement-FFN joint Top-1 drop is `{float(joint_rows[1]['top1_drop']):.8g}` on the fixed diagnostic subset. Full-validation results are in `full_validation_task_importance.csv`.
2. **Secondary:** matched-functional-risk unit comparisons in `matched_functional_risk_comparison.csv`.
3. **Secondary:** parameter-normalized effects in `parameter_normalized_task_importance.csv`.
4. **Descriptive only:** raw single-unit cross-type effects.

Because Attention heads and FFN neurons have different structural granularity, single-unit task-effect differences are descriptive. The matched-budget joint intervention is the primary cross-type evidence. The raw Attention/FFN median CE ratio is deliberately not used as the main claim.

## Functional and task evidence

The all-investigated-unit Spearman correlation between Delta_total and mean CE increase is `{float(rho):.8g}`. The three-head Attention-only correlation is labeled `descriptive_only` because n=3 is not meaningful statistical evidence. D_abs, D_rel, D_dyn, functional energy, class-conditional effects and sampled-domain effects remain separate diagnostic tables.

`investigated_domain_task_sensitivity.csv` covers investigated units only. It does not estimate the complete true task importance of all BMS domains. Task020 must not conclude that Attention is inherently more important than FFN, and it may support imperfect alignment between functional substitutability and task deletability only when matched-budget or matched-risk evidence does so.

No task-demand weight, weighted coverage, Attention protection factor, type quota, selector or pruning score is implemented.
"""
    (output_dir / "diagnosis.md").write_text(diagnosis, encoding="utf-8")
    completion = {"code_version": task020.CODE_VERSION, "artifact_identity_pass": True,
        "attention_group_reconstructed": len(attention) == 3,
        "replacement_ffn_group_reconstructed": len(replacements) > 0,
        "ordinary_deleted_ffn_control_constructed": any(row["group"] == task020.GROUP_ORDINARY for row in rows),
        "matched_surviving_ffn_control_constructed": any(row["group"] == task020.GROUP_CONTROL for row in rows),
        "baseline_28_cached": True, "individual_attention_ablation_complete": len(attention) == 3,
        "replacement_ffn_ablation_complete": len(replacements) > 0,
        "ordinary_deleted_ffn_ablation_complete": any(row["group"] == task020.GROUP_ORDINARY for row in rows),
        "matched_surviving_ffn_ablation_complete": any(row["group"] == task020.GROUP_CONTROL for row in rows),
        "joint_attention_ablation_complete": True, "joint_replacement_ffn_ablation_complete": True,
        "joint_parameter_budget_verified": True,
        "full_validation_baseline_complete": True,
        "full_validation_individual_attention_complete": len(attention) == 3,
        "full_validation_joint_attention_complete": True,
        "full_validation_joint_replacement_complete": True,
        "artifact_cache_identity_pass": True,
        "functional_task_correlation_complete": True, "analysis_complete": True,
        "production_pruning_code_modified": False}
    required_true = [value for key, value in completion.items()
                     if key not in {"code_version", "production_pruning_code_modified"}]
    completion["status"] = "PASS" if all(required_true) and completion["production_pruning_code_modified"] is False else "FAIL"
    task020.atomic_json(output_dir / "task020_completion.json", completion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True)
    analyze(parser.parse_args().output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())

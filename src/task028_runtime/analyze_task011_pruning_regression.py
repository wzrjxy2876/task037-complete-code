#!/usr/bin/env python3
"""Aggregate Task011 controlled-run evidence without changing the method."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def _read_json(path: Path, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _scope(rows: Sequence[dict], name: str) -> dict:
    matches = [row for row in rows if row.get("scope") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one consistency row for {name}")
    return matches[0]


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _layer_key(row: dict) -> tuple[str, str]:
    return (str(row["layer"]), str(row["unit_type"]))


def _min_keep_units(row: dict) -> int:
    match = str(row["min_keep_constraint"]).split("units=")
    if len(match) != 2:
        raise ValueError(f"invalid min_keep_constraint: {row['min_keep_constraint']}")
    return int(match[1])


def compare_layer_pruning(old_rows: Sequence[dict], dynamic_rows: Sequence[dict]) -> list[dict]:
    old = {_layer_key(row): row for row in old_rows}
    dynamic = {_layer_key(row): row for row in dynamic_rows}
    if len(old) != len(old_rows) or len(dynamic) != len(dynamic_rows):
        raise ValueError("duplicate layer/type row in controlled pruning audit")
    if set(old) != set(dynamic):
        missing_old = sorted(set(dynamic) - set(old))
        missing_dynamic = sorted(set(old) - set(dynamic))
        raise ValueError(
            f"Old3D/Dynamic3D layer sets differ; old_missing={missing_old}, "
            f"dynamic_missing={missing_dynamic}"
        )
    rows = []
    for key in sorted(old):
        left = old[key]
        right = dynamic[key]
        if int(left["original_units"]) != int(right["original_units"]):
            raise ValueError(f"original unit count differs for {key}")
        old_ratio = float(left["removed_ratio"])
        dynamic_ratio = float(right["removed_ratio"])
        rows.append(
            {
                "layer": key[0],
                "unit_type": key[1],
                "original_units": int(left["original_units"]),
                "old_removed_units": int(left["removed_units"]),
                "dynamic_removed_units": int(right["removed_units"]),
                "old_remaining_units": int(left["remaining_units"]),
                "dynamic_remaining_units": int(right["remaining_units"]),
                "old_removed_ratio": old_ratio,
                "dynamic_removed_ratio": dynamic_ratio,
                "removed_ratio_difference_dynamic_minus_old": dynamic_ratio - old_ratio,
                "absolute_removed_ratio_difference": abs(dynamic_ratio - old_ratio),
                "old_estimated_removed_parameters": int(left["estimated_removed_parameters"]),
                "dynamic_estimated_removed_parameters": int(right["estimated_removed_parameters"]),
            }
        )
    return rows


def write_extreme_layers(
    path: Path,
    old_rows: Sequence[dict],
    dynamic_rows: Sequence[dict],
    differences: Sequence[dict],
) -> None:
    lines = ["# Extreme and divergent pruning layers", ""]
    for variant, rows in (("Old3D", old_rows), ("Dynamic3D", dynamic_rows)):
        lines.extend([f"## {variant}", ""])
        flags = []
        for row in rows:
            remaining = int(row["remaining_units"])
            minimum = _min_keep_units(row)
            reasons = []
            if row["unit_type"] == "attention_head" and remaining == 1:
                reasons.append("only one Attention head remains")
            if remaining == minimum:
                reasons.append("remaining units equal the existing minimum-keep constraint")
            if reasons:
                flags.append(
                    f"- `{row['layer']}` ({row['unit_type']}): "
                    f"{row['original_units']} -> {remaining}; " + "; ".join(reasons)
                )
        lines.extend(flags or ["- No layer exactly reached the existing minimum-keep boundary."])
        lines.append("")

    changed = [
        row for row in differences
        if float(row["absolute_removed_ratio_difference"]) > 0.0
    ]
    changed.sort(
        key=lambda row: (
            -float(row["absolute_removed_ratio_difference"]), row["layer"], row["unit_type"]
        )
    )
    lines.extend(
        [
            "## Largest observed Old3D/Dynamic3D differences",
            "",
            "This is a rank list, not a new pruning threshold.",
            "",
        ]
    )
    for row in changed[:20]:
        lines.append(
            f"- `{row['layer']}` ({row['unit_type']}): old removed "
            f"{float(row['old_removed_ratio']):.3%}, dynamic removed "
            f"{float(row['dynamic_removed_ratio']):.3%}, absolute difference "
            f"{float(row['absolute_removed_ratio_difference']):.3%}."
        )
    if not changed:
        lines.append("- The two controlled runs selected identical layer-wise counts.")
    lines.append("")
    _atomic_text(path, "\n".join(lines))


def _baseline_figure(path: Path, old: dict, dynamic: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Old3D", "Dynamic3D"]
    baseline = [
        float(old["metrics"]["baseline_top1"]),
        float(dynamic["metrics"]["baseline_top1"]),
    ]
    preft = [
        float(old["metrics"]["pre_ft_top1"]),
        float(dynamic["metrics"]["pre_ft_top1"]),
    ]
    x = np.arange(2)
    width = 0.34
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.bar(x - width / 2, baseline, width, label="Unpruned baseline")
    axis.bar(x + width / 2, preft, width, label="Pre-finetune pruned")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("Unpruned baseline vs pre-finetune pruning")
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _layer_figure(path: Path, rows: Sequence[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{row['layer'].replace('layers.', 'L')}:{'A' if row['unit_type'] == 'attention_head' else 'F'}"
        for row in rows
    ]
    old = [float(row["old_removed_ratio"]) for row in rows]
    dynamic = [float(row["dynamic_removed_ratio"]) for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(max(12, len(rows) * 0.28), 5.5))
    axis.plot(x, old, marker="o", markersize=2.5, linewidth=1, label="Old3D")
    axis.plot(x, dynamic, marker="o", markersize=2.5, linewidth=1, label="Dynamic3D")
    axis.set_xticks(x, labels, rotation=90, fontsize=7)
    axis.set_ylabel("Removed-unit ratio")
    axis.set_title("Layer-wise controlled pruning: Old3D vs Dynamic3D")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _metric_comparison(dynamic_value: float, old_value: float) -> str:
    if dynamic_value > old_value:
        return f"Dynamic3D is higher by {dynamic_value - old_value:.6g} percentage points"
    if dynamic_value < old_value:
        return f"Dynamic3D is lower by {old_value - dynamic_value:.6g} percentage points"
    return "the values are equal at saved precision"


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    old = _read_json(root / "controlled_runs" / "old3d.json")
    dynamic = _read_json(root / "controlled_runs" / "dynamic3d.json")
    consistency_metadata = _read_json(root / "tdd_009_010_consistency_metadata.json")
    consistency_rows = _read_csv(root / "tdd_009_010_consistency_summary.csv")
    samples = _read_json(root / "sample_set_comparison.json")
    checkpoints = _read_json(root / "checkpoint_comparison.json")
    real_trace = _read_json(root / "real_activation_trace.json", required=False)
    old_layers = _read_csv(root / "old3d_layer_pruning.csv")
    dynamic_layers = _read_csv(root / "dynamic3d_layer_pruning.csv")

    if old["configuration"] != dynamic["configuration"]:
        raise ValueError(
            "Old3D and Dynamic3D controlled configurations differ beyond descriptor variant"
        )
    old_sample_hash = old["calibration_samples"]["sample_list_sha256"]
    dynamic_sample_hash = dynamic["calibration_samples"]["sample_list_sha256"]
    if old_sample_hash != dynamic_sample_hash:
        raise ValueError("Old3D and Dynamic3D calibration sample order differs")
    old_checkpoint = old["checkpoint"]["sha256"]
    dynamic_checkpoint = dynamic["checkpoint"]["sha256"]
    if old_checkpoint != dynamic_checkpoint:
        raise ValueError("Old3D and Dynamic3D controlled checkpoints differ")

    differences = compare_layer_pruning(old_layers, dynamic_layers)
    diff_fields = (
        "layer",
        "unit_type",
        "original_units",
        "old_removed_units",
        "dynamic_removed_units",
        "old_remaining_units",
        "dynamic_remaining_units",
        "old_removed_ratio",
        "dynamic_removed_ratio",
        "removed_ratio_difference_dynamic_minus_old",
        "absolute_removed_ratio_difference",
        "old_estimated_removed_parameters",
        "dynamic_estimated_removed_parameters",
    )
    _atomic_csv(root / "old_vs_dynamic_layer_pruning_diff.csv", diff_fields, differences)
    write_extreme_layers(
        root / "extreme_pruning_layers.md", old_layers, dynamic_layers, differences
    )
    _baseline_figure(root / "baseline_vs_preft_top1.png", old, dynamic)
    _layer_figure(root / "old_vs_dynamic_layer_pruning.png", differences)

    budget_rows = []
    for result in (old, dynamic):
        budget = result["parameter_budget"]
        budget_rows.append({"variant": result["variant"], **budget})
    budget_fields = tuple(budget_rows[0].keys())
    _atomic_csv(root / "parameter_budget_audit.csv", budget_fields, budget_rows)

    all_metrics = _scope(consistency_rows, "all_units")
    attention_metrics = _scope(consistency_rows, "attention_only")
    mlp_metrics = _scope(consistency_rows, "mlp_only")
    formula_difference = (
        real_trace.get("same_forward_same_tensor_max_abs_difference")
        if real_trace else None
    )
    formula_same = (
        "YES"
        if formula_difference is not None and float(formula_difference) <= 1e-6
        else "UNKNOWN (real-activation trace unavailable or inconsistent)"
    )
    attention_trace_difference = (
        real_trace.get("attention", {}).get("max_abs_difference")
        if real_trace else None
    )
    mlp_trace_difference = (
        real_trace.get("mlp", {}).get("max_abs_difference")
        if real_trace else None
    )
    numerical_same = (
        "YES" if consistency_metadata.get("numerically_close_at_atol_1e-6") else "NO"
    )
    attention_mae = _float(attention_metrics, "mae")
    mlp_mae = _float(mlp_metrics, "mae")
    concentration = (
        "Attention"
        if attention_mae > mlp_mae
        else "MLP" if mlp_mae > attention_mae else "neither; MAE is equal"
    )
    sample_same = samples.get("same_ordered_calibration_samples", "UNKNOWN")
    checkpoint_same = checkpoints.get(
        "task009_vs_task010_same_checkpoint", "UNKNOWN"
    )
    old_metrics = old["metrics"]
    dynamic_metrics = dynamic["metrics"]
    old_budget = old["parameter_budget"]
    dynamic_budget = dynamic["parameter_budget"]
    validation = old.get("validation_dataset", {})
    checkpoint_audit = old.get("checkpoint", {})
    model_class_count = checkpoint_audit.get("model_classifier_output_dimension")
    checkpoint_class_count = checkpoint_audit.get(
        "checkpoint_classifier_output_dimension"
    )
    dataset_class_count = validation.get("num_unique_labels")
    label_minimum = validation.get("label_min")
    label_maximum = validation.get("label_max")
    dataset_required_outputs = (
        int(label_maximum) + 1
        if label_minimum is not None
        and label_maximum is not None
        and int(label_minimum) == 0
        else dataset_class_count
    )
    classifier_dataset_mismatch = (
        model_class_count is not None
        and dataset_required_outputs is not None
        and int(model_class_count) != int(dataset_required_outputs)
    )
    historical_recovered = "NOT FULLY DETERMINABLE"
    strongest_cause = (
        "The historical configuration is not fully recovered; the controlled run "
        "separates checkpoint/evaluation from pruning, but no unverified field is "
        "promoted to a causal explanation."
    )
    if classifier_dataset_mismatch:
        strongest_cause = (
            f"The evaluated model has {int(model_class_count)} classifier outputs, "
            f"but the validation label range requires {int(dataset_required_outputs)} "
            f"outputs ({int(dataset_class_count)} unique labels observed). This is a "
            "concrete checkpoint/model/data compatibility problem "
            "that must be resolved before attributing low accuracy to pruning or TDD."
        )
    elif checkpoint_same == "NO":
        strongest_cause = "Task009 and Task010 used different checkpoint hashes."
    elif sample_same == "NO" and numerical_same == "NO":
        strongest_cause = (
            "The saved Task009 and Task010 descriptor values used different ordered "
            "calibration videos; this explains why formula identity does not imply "
            "saved-value identity, but does not by itself explain the historical "
            "Old3D accuracy regression."
        )
    ready = "NO"
    if formula_same == "YES" and old_sample_hash == dynamic_sample_hash:
        ready_reason = (
            "Formula and matched controls are valid, but historical Old3D behavior "
            "has not been tied to a fully recovered configuration."
        )
    else:
        ready_reason = "Formula or matched-control prerequisites remain unresolved."
    if classifier_dataset_mismatch:
        ready_reason = (
            "Classifier output dimension and validation class count differ; resolve "
            "that checkpoint/model/data mismatch before any fine-tuning comparison."
        )

    # Historical reference values are context, not numerical pass thresholds.
    # Without recovered baseline/configuration evidence, cases A-D remain
    # explicitly indeterminate.  Case E is directly testable from the same-
    # forward implementation trace.
    diagnostic_cases = {
        "case_A_checkpoint_data_evaluation_problem": (
            "NOT FULLY DETERMINABLE: historical unpruned baseline evidence unavailable"
        ),
        "case_B_pruning_pipeline_or_configuration_regression": (
            "NOT FULLY DETERMINABLE: historical configuration/run unavailable"
        ),
        "case_C_tdd_adversely_changes_pruning": (
            "NOT FULLY DETERMINABLE: historical Old3D behavior not recovered"
        ),
        "case_D_ready_to_consider_finetuning": "NO",
        "case_E_attention_tdd_implementation_mismatch": "UNKNOWN",
    }
    if classifier_dataset_mismatch:
        diagnostic_cases["case_A_checkpoint_data_evaluation_problem"] = (
            "YES: classifier output dimension differs from validation class count"
        )
        diagnostic_cases["case_B_pruning_pipeline_or_configuration_regression"] = (
            "NOT EVALUATED AS PRIMARY: resolve CASE A first"
        )
    if attention_trace_difference is not None and mlp_trace_difference is not None:
        attention_mismatch = float(attention_trace_difference) > 1e-6
        mlp_mismatch = float(mlp_trace_difference) > 1e-6
        diagnostic_cases["case_E_attention_tdd_implementation_mismatch"] = (
            "YES" if attention_mismatch and not mlp_mismatch else "NO"
        )

    summary = f"""# Task011 diagnosis summary

This report separates mathematical correctness, saved-experiment consistency,
and pruning effectiveness. It does not infer fine-tuned accuracy.

1. **Are Task009 D_var and Task010 D_dyn mathematically identical?**
   **{formula_same}.** Same-forward maximum formula/restoration difference: {formula_difference if formula_difference is not None else 'unavailable'}.

2. **Are they numerically identical on the saved experiments?**
   **{numerical_same}.** All-unit Pearson={all_metrics['pearson']}, Spearman={all_metrics['spearman']}, MAE={all_metrics['mae']}, max error={all_metrics['max_abs_error']}.

3. **If not, why not?**
   Ordered calibration sample equality is {sample_same}; checkpoint equality is {checkpoint_same}. Different samples invalidate exact saved-value equality without constituting an implementation bug.

4. **Is the discrepancy concentrated in Attention or MLP?**
   The larger MAE is in **{concentration}**: Attention MAE={attention_metrics['mae']}, MLP MAE={mlp_metrics['mae']}.

5. **Are Task009 and Task010 using the same calibration samples?**
   **{sample_same}.** See `sample_set_comparison.json`.

6. **Are they using the same checkpoint?**
   **{checkpoint_same}.** See `checkpoint_comparison.json`.

7. **Does the current unpruned model reproduce the expected baseline?**
   The controlled baseline is Top-1={float(old_metrics['baseline_top1']):.6g}% and Top-5={float(old_metrics['baseline_top5']):.6g}%. Model classifier outputs={model_class_count if model_class_count is not None else 'UNKNOWN'}, checkpoint classifier outputs={checkpoint_class_count if checkpoint_class_count is not None else 'UNKNOWN'}, validation unique labels={dataset_class_count if dataset_class_count is not None else 'UNKNOWN'}, label range={validation.get('label_min', 'UNKNOWN')}..{validation.get('label_max', 'UNKNOWN')}. An exact historical unpruned reference was not recovered, so numerical reproduction is not fully determinable; a classifier/data dimension mismatch is nevertheless a concrete CASE A failure when present.

8. **Does current Old3D reproduce historical pre-finetune behavior?**
   Old3D controlled pre-FT is Top-1={float(old_metrics['pre_ft_top1']):.6g}% and Top-5={float(old_metrics['pre_ft_top5']):.6g}%. Historical reproduction: **{historical_recovered}**.

9. **Strongest identified cause of the regression?**
   {strongest_cause}

10. **At the same controlled configuration, is Dynamic3D better or worse before fine-tuning?**
    Old3D Top-1={float(old_metrics['pre_ft_top1']):.6g}%; Dynamic3D Top-1={float(dynamic_metrics['pre_ft_top1']):.6g}%; {_metric_comparison(float(dynamic_metrics['pre_ft_top1']), float(old_metrics['pre_ft_top1']))}.

11. **Is 50% sparsity estimated budget sparsity or physical numel sparsity?**
    It is an **estimated budget sparsity**. Old3D estimated={float(old_budget['estimated_budget_sparsity']):.9g}, physical numel={float(old_budget['physical_numel_sparsity']):.9g}; Dynamic3D estimated={float(dynamic_budget['estimated_budget_sparsity']):.9g}, physical numel={float(dynamic_budget['physical_numel_sparsity']):.9g}.

12. **Is the repository ready for final fine-tuning comparison?**
    **{ready}.** {ready_reason}

## Diagnostic cases

- CASE A: {diagnostic_cases['case_A_checkpoint_data_evaluation_problem']}.
- CASE B: {diagnostic_cases['case_B_pruning_pipeline_or_configuration_regression']}.
- CASE C: {diagnostic_cases['case_C_tdd_adversely_changes_pruning']}.
- CASE D: {diagnostic_cases['case_D_ready_to_consider_finetuning']}.
- CASE E: {diagnostic_cases['case_E_attention_tdd_implementation_mismatch']}.

- Unpruned evaluation is now measured before every controlled pruning run.
- If baseline is correct but Old3D remains far from the documented historical run, the regression is independent of TDD; exact attribution still requires the historical configuration evidence.
- Attention-only implementation mismatch is {'YES' if float(attention_metrics['max_abs_error']) > 1e-6 and float(mlp_metrics['max_abs_error']) <= 1e-6 else 'NO or not isolated to Attention'} based on saved-value error; same-forward trace is the implementation-level check.
- No 100-epoch fine-tuning was run or recommended by this diagnostic.
"""
    _atomic_text(root / "diagnosis_summary.md", summary)
    _atomic_json(
        root / "diagnosis_metadata.json",
        {
            "status": "complete",
            "formula_same": formula_same,
            "saved_values_numerically_same": numerical_same,
            "task009_task010_same_samples": sample_same,
            "task009_task010_same_checkpoint": checkpoint_same,
            "old_dynamic_control_configuration_identical": True,
            "old_dynamic_calibration_sample_hash_identical": True,
            "old_dynamic_checkpoint_hash_identical": True,
            "historical_old3d_reproduced": historical_recovered,
            "ready_for_full_finetuning": False,
            "model_classifier_output_dimension": model_class_count,
            "checkpoint_classifier_output_dimension": checkpoint_class_count,
            "validation_unique_label_count": dataset_class_count,
            "validation_required_classifier_outputs": dataset_required_outputs,
            "classifier_dataset_dimension_mismatch": classifier_dataset_mismatch,
            "diagnostic_cases": diagnostic_cases,
        },
    )
    print(f"Task011 regression diagnosis written to {root.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Task011 pruning regression evidence")
    parser.add_argument("--output_dir", default="task011_diagnosis")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

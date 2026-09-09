"""Offline Task022 diagnosis of Delta_average/Delta_total complementarity.

This module consumes the immutable PASS artifacts from Task020 and Task021.
It does not import a model, run validation, recompute descriptors or
Contribution Fields, run BMS, or alter a selector.  All results are
descriptive diagnostics over the exact 417 Task020-labelled units.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import task020_functional_demand_task_importance as task020
import task021_importance_information_retention as task021


CODE_VERSION = "task022_average_total_dual_risk_v1"
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
EXPECTED_COHORT_UNITS = 417
EXPECTED_GROUP_COUNTS = {
    task020.GROUP_ATTENTION: 3,
    task020.GROUP_REPLACEMENT: 138,
    task020.GROUP_ORDINARY: 138,
    task020.GROUP_CONTROL: 138,
}
TASK_METRICS = (
    "mean_ce_increase",
    "mean_margin_drop",
    "mean_true_logit_drop",
    "mean_kl",
    "correct_to_wrong_flip_rate",
)
REQUIRED_TASK020_FILES = (
    "task020_completion.json",
    "artifact_identity.json",
    "task020_cohorts.csv",
    "unit_task_importance.csv",
    "functional_task_mismatch.csv",
    "attention_case_studies.csv",
    "matched_functional_risk_comparison.csv",
    "full_validation_task_importance.csv",
)
REQUIRED_TASK021_FILES = (
    "task021_completion.json",
    "artifact_identity.json",
    "information_retention_ladder.csv",
    "information_rank_percentiles.csv",
    "rank_information_loss.csv",
    "stagewise_task_correlation.csv",
    "information_retention_summary.csv",
    "matched_delta_task_mismatch.csv",
    "task021_summary.csv",
    "information_loss_evidence.csv",
)


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _float(row: Mapping[str, object], *names: str, default: float | None = None) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"Non-finite numeric field {name}")
            return result
    if default is not None:
        return float(default)
    raise KeyError(names[0])


def _int(row: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return int(float(value))
    raise KeyError(names[0])


def _require_true(payload: Mapping[str, object], key: str, label: str) -> None:
    if payload.get(key) is not True:
        raise RuntimeError(f"{label} field {key!r} is not true")


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("rankdata expects a non-empty finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and array[order[end]] == array[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + 1 + end) / 2.0
        begin = end
    return ranks


def importance_percentile(values: Sequence[float], *, higher_is_important: bool = True) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float64)
    if not higher_is_important:
        transformed = -transformed
    ranks = rankdata(transformed)
    return (ranks - 1.0) / max(1, len(ranks) - 1)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        return math.nan
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def rank_gap(delta_average_percentile: float, delta_total_percentile: float) -> float:
    """Average-risk percentile minus total-risk percentile."""
    return float(delta_average_percentile - delta_total_percentile)


def assign_quadrant(delta_average: float, delta_total: float,
                    average_median: float, total_median: float) -> str:
    """Median-split labels; equality belongs to the high side."""
    average_high = float(delta_average) >= float(average_median)
    total_high = float(delta_total) >= float(total_median)
    if average_high and total_high:
        return "high_average_high_total"
    if average_high:
        return "high_average_low_total"
    if total_high:
        return "low_average_high_total"
    return "low_average_low_total"


def assign_percentile_region(average_percentile: float, total_percentile: float,
                             *, lower: float = .25, upper: float = .75) -> str:
    """Fixed percentile region used only for descriptive robustness checks."""
    average_high = float(average_percentile) >= upper
    average_low = float(average_percentile) <= lower
    total_high = float(total_percentile) >= upper
    total_low = float(total_percentile) <= lower
    if average_high and total_high:
        return "top_quartile_both"
    if average_high and total_low:
        return "high_average_low_total"
    if average_low and total_high:
        return "low_average_high_total"
    if average_low and total_low:
        return "bottom_quartile_both"
    return "middle_or_mixed"


def quantile_bin_ids(values: Sequence[float], bins: int = 4) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or bins < 1:
        raise ValueError("quantile_bin_ids expects a non-empty finite vector")
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.int64)
    for position, index in enumerate(order):
        result[index] = min(bins - 1, (position * bins) // len(array))
    return result


def top_risk_retrieval(signal: Sequence[float], task_risk: Sequence[float], *,
                       signal_higher_is_important: bool = True,
                       fraction: float = .10) -> dict[str, float | int]:
    values = np.asarray(signal, dtype=np.float64)
    risk = np.asarray(task_risk, dtype=np.float64)
    if values.shape != risk.shape or values.ndim != 1 or not len(values):
        raise ValueError("signal and task risk must be aligned non-empty vectors")
    count = max(1, int(math.ceil(len(values) * fraction)))
    signal_order = np.argsort(-values if signal_higher_is_important else values,
                              kind="mergesort")
    risk_order = np.argsort(-risk, kind="mergesort")
    selected = set(signal_order[:count].tolist())
    target = set(risk_order[:count].tolist())
    true_positive = len(selected & target)
    return {
        "selected_count": count,
        "task_risk_count": count,
        "true_positive": true_positive,
        "recall": true_positive / float(len(target)),
        "precision": true_positive / float(len(selected)),
    }


def pareto_front_ranks(delta_average: Sequence[float], delta_total: Sequence[float]) -> np.ndarray:
    """Return deterministic non-dominated front ranks, with first front = 1."""
    average = np.asarray(delta_average, dtype=np.float64)
    total = np.asarray(delta_total, dtype=np.float64)
    if average.shape != total.shape or average.ndim != 1 or not len(average):
        raise ValueError("Pareto vectors must be aligned non-empty one-dimensional arrays")
    if not np.isfinite(average).all() or not np.isfinite(total).all():
        raise ValueError("Pareto vectors must be finite")
    remaining = list(range(len(average)))
    ranks = np.zeros(len(average), dtype=np.int64)
    front = 1
    while remaining:
        nondominated = []
        for candidate in remaining:
            dominated = False
            for other in remaining:
                if other == candidate:
                    continue
                if (average[other] >= average[candidate]
                        and total[other] >= total[candidate]
                        and (average[other] > average[candidate]
                             or total[other] > total[candidate])):
                    dominated = True
                    break
            if not dominated:
                nondominated.append(candidate)
        # Keep original row order for deterministic output.
        for index in nondominated:
            ranks[index] = front
        removed = set(nondominated)
        remaining = [index for index in remaining if index not in removed]
        front += 1
    return ranks


def verify_average_total_identity(rows: Sequence[Mapping[str, object]], *,
                                  rtol: float = 1e-6, atol: float = 1e-9) -> list[dict[str, object]]:
    """Verify saved total risk against N_valid times average risk."""
    output = []
    failed = []
    for row in rows:
        index = _int(row, "global_index")
        domain_id = _int(row, "domain_id")
        count = _int(row, "domain_valid_demand_count", "domain_size")
        if count <= 0:
            raise RuntimeError(f"Non-positive domain demand count for {index}: {count}")
        average = _float(row, "delta_average", "delta_average_at_28")
        total = _float(row, "delta_total", "delta_total_at_28")
        reconstructed = float(count * average)
        error = abs(total - reconstructed)
        scale = max(abs(total), abs(reconstructed), 1.0)
        passed = bool(error <= atol + rtol * scale)
        item = {
            "global_index": index,
            "domain_id": domain_id,
            "domain_valid_demand_count": count,
            "delta_average": average,
            "delta_total": total,
            "reconstructed_delta_total": reconstructed,
            "absolute_error": error,
            "relative_error": error / max(abs(total), atol),
            "identity_pass": passed,
        }
        output.append(item)
        if not passed:
            failed.append(item)
    if failed:
        first = failed[0]
        raise RuntimeError(
            "Delta_total = N_valid * Delta_average mismatch; "
            f"first global_index={first['global_index']} error={first['absolute_error']}"
        )
    return output


def _shared_identity_values(left: Mapping[str, object], right: Mapping[str, object]) -> list[tuple[str, object, object]]:
    aliases = (
        "checkpoint_sha256", "checkpoint", "checkpoint_path", "checkpoint_identity",
        "reference_prefix_sha256", "prefix_sha256", "prefix_identity",
        "keep_prefix_sha256", "reference_state_sha256", "validation_checkpoint",
    )
    differences = []
    for key in aliases:
        if key in left and key in right and left[key] != right[key]:
            differences.append((key, left[key], right[key]))
    return differences


def _validate_cohort_rows(cohorts: Sequence[Mapping[str, object]]) -> tuple[list[int], list[int], set[int]]:
    if len(cohorts) != EXPECTED_COHORT_UNITS:
        raise RuntimeError(f"Task020 cohort has {len(cohorts)} rows, expected {EXPECTED_COHORT_UNITS}")
    indices = [_int(row, "global_index") for row in cohorts]
    if len(set(indices)) != EXPECTED_COHORT_UNITS:
        raise RuntimeError("Task020 cohort contains duplicate global_index values")
    counts = {group: sum(str(row.get("cohort", row.get("group", ""))) == group for row in cohorts)
              for group in EXPECTED_GROUP_COUNTS}
    if counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Task020 cohort group counts differ: {counts}")
    attention = [index for index, row in zip(indices, cohorts)
                 if str(row.get("cohort", row.get("group", ""))) == task020.GROUP_ATTENTION]
    replacement = [index for index, row in zip(indices, cohorts)
                   if str(row.get("cohort", row.get("group", ""))) == task020.GROUP_REPLACEMENT]
    return attention, replacement, set(indices)


def verify_task022_identity(task020_root: Path, task021_root: Path,
                            *, repo_root: Path | None = None) -> dict[str, object]:
    """Strictly gate the two PASS artifact trees before any analysis."""
    t20, t21 = Path(task020_root), Path(task021_root)
    missing20 = [name for name in REQUIRED_TASK020_FILES if not (t20 / name).is_file()]
    missing21 = [name for name in REQUIRED_TASK021_FILES if not (t21 / name).is_file()]
    if missing20 or missing21:
        raise FileNotFoundError(f"Missing Task020 files={missing20}; Task021 files={missing21}")

    # Reuse the strict Task020 Git-aware gate.  It checks exact caches, cohorts,
    # order-sensitive joint identity, and production source identity.
    task020_identity = task021.verify_task020_identity(t20, repo_root=repo_root)
    completion20 = read_json(t20 / "task020_completion.json")
    identity20 = read_json(t20 / "artifact_identity.json")
    completion21 = read_json(t21 / "task021_completion.json")
    identity21 = read_json(t21 / "artifact_identity.json")
    if completion20.get("status") != "PASS":
        raise RuntimeError("Task020 is not PASS")
    if completion21.get("status") != "PASS" or completion21.get("code_version") != task021.CODE_VERSION:
        raise RuntimeError("Task021 is not PASS at its final code version")
    for key in (
        "task020_identity_pass", "cohort_identity_pass", "information_ladder_complete",
        "stagewise_correlation_complete", "rank_loss_complete",
        "within_functional_risk_analysis_complete", "matched_risk_analysis_complete",
        "type_confound_control_complete", "diagnostic_proxy_analysis_complete", "analysis_complete",
    ):
        _require_true(completion21, key, "Task021 completion")
    if completion21.get("production_pruning_code_modified") is not False:
        raise RuntimeError("Task021 reports modified production pruning code")
    if completion21.get("gpu_pruning_or_validation_executed") is not False:
        raise RuntimeError("Task021 reports GPU pruning or validation")
    _require_true(identity21, "task020_identity_pass", "Task021 artifact identity")
    _require_true(identity21, "cohort_identity_pass", "Task021 artifact identity")
    if int(identity21.get("descriptor_units", -1)) != EXPECTED_UNITS:
        raise RuntimeError("Task021 descriptor unit count is not 36,378")
    if int(identity21.get("bms_domains", -1)) != EXPECTED_DOMAINS:
        raise RuntimeError("Task021 BMS domain count is not 423")

    cohorts = read_csv(t20 / "task020_cohorts.csv")
    attention, replacement, cohort_set = _validate_cohort_rows(cohorts)
    if identity21.get("task020_identity_sha256") != sha256_file(t20 / "artifact_identity.json"):
        raise RuntimeError("Task021 does not refer to the exact Task020 artifact identity")
    if identity21.get("task020_code_version") not in (None, task020.CODE_VERSION):
        raise RuntimeError("Task021 refers to an unexpected Task020 code version")
    if _shared_identity_values(identity20, identity21):
        raise RuntimeError(f"Task020/Task021 checkpoint or prefix identity differs: {_shared_identity_values(identity20, identity21)}")
    if identity21.get("attention_global_indices") not in (None, attention):
        raise RuntimeError("Task021 Attention identities differ from Task020 cohort order")
    if identity21.get("replacement_ffn_global_indices") not in (None, replacement):
        raise RuntimeError("Task021 replacement identities differ from Task020 cohort")

    unit_rows = read_csv(t20 / "unit_task_importance.csv")
    unit_set = {_int(row, "global_index") for row in unit_rows}
    if len(unit_rows) != EXPECTED_COHORT_UNITS or unit_set != cohort_set:
        raise RuntimeError("Task020 unit_task_importance is not the exact 417-unit cohort")
    ladder = read_csv(t21 / "information_retention_ladder.csv")
    ladder_set = {_int(row, "global_index") for row in ladder}
    if len(ladder) != EXPECTED_COHORT_UNITS or ladder_set != cohort_set:
        raise RuntimeError("Task021 ladder does not preserve the exact 417-unit cohort")
    source_ids = task020_identity.get("production_source_git_blob_sha", {})
    if identity21.get("production_source_git_blob_sha") not in (None, source_ids):
        raise RuntimeError("Task021 production source identity differs from Task020")
    return {
        "task020_root": str(t20.resolve()),
        "task021_root": str(t21.resolve()),
        "task020_identity_sha256": sha256_file(t20 / "artifact_identity.json"),
        "task021_identity_sha256": sha256_file(t21 / "artifact_identity.json"),
        "task020_completion_sha256": sha256_file(t20 / "task020_completion.json"),
        "task021_completion_sha256": sha256_file(t21 / "task021_completion.json"),
        "attention_global_indices": attention,
        "replacement_ffn_global_indices": replacement,
        "cohort_set": cohort_set,
        "descriptor_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS,
        "production_source_git_blob_sha": source_ids,
    }


def _normalize_rows(task020_root: Path, task021_root: Path, cohort_set: set[int]) -> list[dict[str, object]]:
    task_rows = read_csv(Path(task020_root) / "unit_task_importance.csv")
    ladder_by_index = {_int(row, "global_index"): row for row in read_csv(Path(task021_root) / "information_retention_ladder.csv")}
    result = []
    for source in task_rows:
        index = _int(source, "global_index")
        if index not in cohort_set:
            raise RuntimeError(f"Unexpected Task020 unit index {index}")
        ladder = ladder_by_index[index]
        row: dict[str, object] = dict(source)
        row.update({key: value for key, value in ladder.items() if key not in row or str(row.get(key, "")) == ""})
        row["global_index"] = index
        row["group"] = str(source.get("group", source.get("cohort", ladder.get("group", ""))))
        row["unit_type"] = str(source.get("unit_type", ladder.get("unit_type", "")))
        row["layer"] = str(source.get("layer", ladder.get("layer", "")))
        row["stage"] = str(source.get("stage", ladder.get("stage", "")))
        row["domain_id"] = _int(source, "domain_id")
        row["domain_valid_demand_count"] = _int(source, "domain_valid_demand_count", "domain_size")
        for name in (
            "delta_average_at_28", "delta_total_at_28", "D_abs", "D_rel", "D_dyn",
            "functional_energy", "best_substitute_similarity_at_28", "domain_coverage_at_28",
            *TASK_METRICS,
        ):
            # Ladder uses the same Task020 names; the aliases are only for
            # compatibility with hand-authored fixtures.
            aliases = (name, name.removesuffix("_at_28")) if name.endswith("_at_28") else (name,)
            row[name] = _float(row, *aliases)
        result.append(row)
    result.sort(key=_global_index_key)
    if len(result) != EXPECTED_COHORT_UNITS or {int(row["global_index"]) for row in result} != cohort_set:
        raise RuntimeError("Normalized Task022 population is not exactly 417 units")
    return result


def _attach_dual_risk(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    average = [_float(row, "delta_average_at_28") for row in output]
    total = [_float(row, "delta_total_at_28") for row in output]
    task = [_float(row, "mean_ce_increase") for row in output]
    avg_percentile = importance_percentile(average)
    total_percentile = importance_percentile(total)
    task_percentile = importance_percentile(task)
    avg_median, total_median = float(np.median(average)), float(np.median(total))
    for row, av, to, ce, ap, tp, cep in zip(output, average, total, task, avg_percentile, total_percentile, task_percentile):
        row["delta_average_percentile"] = float(ap)
        row["delta_total_percentile"] = float(tp)
        row["task_ce_percentile"] = float(cep)
        row["rank_gap"] = rank_gap(ap, tp)
        row["quadrant"] = assign_quadrant(av, to, avg_median, total_median)
        row["percentile_region"] = assign_percentile_region(ap, tp)
        row["task_ce"] = ce
    return output


def _full_validation_map(task020_root: Path) -> dict[int, float]:
    values = {}
    for row in read_csv(Path(task020_root) / "full_validation_task_importance.csv"):
        variant = str(row.get("variant", ""))
        if variant.startswith("attention_") and variant != "attention_joint":
            try:
                index = int(variant.split("_", 1)[1])
            except ValueError:
                continue
            values[index] = _float(row, "top1_drop_from_28", "top1_drop", "top1", default=math.nan)
    return values


def build_average_total_risk_plane(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = (
        "global_index", "group", "unit_type", "layer", "stage", "domain_id",
        "domain_valid_demand_count", "delta_average", "delta_total",
        "delta_average_percentile", "delta_total_percentile", "task_ce",
        "task_ce_percentile", "D_abs", "D_rel", "D_dyn", "functional_energy",
        "quadrant", "percentile_region",
    )
    return [{
        **{field: row.get(field, "") for field in fields if field not in ("delta_average", "delta_total")},
        "delta_average": _float(row, "delta_average_at_28"),
        "delta_total": _float(row, "delta_total_at_28"),
    } for row in rows]


def quadrant_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    labels = ("high_average_high_total", "high_average_low_total",
              "low_average_high_total", "low_average_low_total")
    for label in labels:
        subset = [row for row in rows if row["quadrant"] == label]
        task = [_float(row, "mean_ce_increase") for row in subset]
        margin = [_float(row, "mean_margin_drop") for row in subset]
        ce_p = [_float(row, "task_ce_percentile") for row in subset]
        output.append({
            "quadrant": label,
            "num_units": len(subset),
            "attention_count": sum(row["unit_type"] == task020.TYPE_ATTENTION for row in subset),
            "ffn_count": sum(row["unit_type"] == task020.TYPE_FFN for row in subset),
            "median_domain_size": float(np.median([_int(row, "domain_valid_demand_count") for row in subset])) if subset else math.nan,
            "median_delta_average": float(np.median([_float(row, "delta_average_at_28") for row in subset])) if subset else math.nan,
            "median_delta_total": float(np.median([_float(row, "delta_total_at_28") for row in subset])) if subset else math.nan,
            "median_ce_increase": float(np.median(task)) if task else math.nan,
            "mean_ce_increase": float(np.mean(task)) if task else math.nan,
            "median_margin_drop": float(np.median(margin)) if margin else math.nan,
            "correct_to_wrong_flip_rate": float(np.mean([_float(row, "correct_to_wrong_flip_rate") for row in subset])) if subset else math.nan,
            "high_task_risk_top10_rate": float(np.mean(np.asarray(ce_p) >= .90)) if subset else math.nan,
        })
    return output


def _region_rows(rows: Sequence[Mapping[str, object]], region: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["percentile_region"] == region]
    fields = (
        "global_index", "unit_type", "layer", "domain_id", "domain_valid_demand_count",
        "delta_average_at_28", "delta_total_at_28", "task_ce", "task_ce_percentile",
        "D_abs", "D_rel", "functional_energy",
    )
    return [{field: row.get(field, "") for field in fields} for row in selected]


def attention_trajectory(rows: Sequence[Mapping[str, object]], attention_indices: Sequence[int],
                        validation: Mapping[int, float]) -> list[dict[str, object]]:
    by_index = {int(row["global_index"]): row for row in rows}
    output = []
    for index in attention_indices:
        if int(index) not in by_index:
            raise RuntimeError(f"Attention identity {index} is absent from Task022 rows")
        row = by_index[int(index)]
        output.append({
            "global_index": int(index),
            "domain_valid_demand_count": _int(row, "domain_valid_demand_count"),
            "delta_average": _float(row, "delta_average_at_28"),
            "delta_average_percentile": float(row["delta_average_percentile"]),
            "delta_total": _float(row, "delta_total_at_28"),
            "delta_total_percentile": float(row["delta_total_percentile"]),
            "task_ce_percentile": float(row["task_ce_percentile"]),
            "full_validation_top1_drop": validation.get(int(index), math.nan),
        })
    return output


def _population_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    return {
        "all_investigated_units": list(rows),
        "investigated_ffn_only": [row for row in rows if row["unit_type"] == task020.TYPE_FFN],
        "attention_descriptive_only": [row for row in rows if row["unit_type"] == task020.TYPE_ATTENTION],
    }


def domain_size_correlations(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for population, subset in _population_rows(rows).items():
        size = [_int(row, "domain_valid_demand_count") for row in subset]
        output.extend([
            {"population": population, "signal": "domain_valid_demand_count", "target": "delta_average", "spearman": spearman(size, [_float(row, "delta_average_at_28") for row in subset]), "num_units": len(subset)},
            {"population": population, "signal": "domain_valid_demand_count", "target": "delta_total", "spearman": spearman(size, [_float(row, "delta_total_at_28") for row in subset]), "num_units": len(subset)},
            {"population": population, "signal": "domain_valid_demand_count", "target": "task_ce", "spearman": spearman(size, [_float(row, "mean_ce_increase") for row in subset]), "num_units": len(subset)},
        ])
    return output


def conditional_domain_size(rows: Sequence[Mapping[str, object]], bins: int = 4) -> list[dict[str, object]]:
    sizes = np.asarray([_int(row, "domain_valid_demand_count") for row in rows], dtype=np.float64)
    identifiers = quantile_bin_ids(sizes, bins=bins)
    output = []
    for bin_id in range(bins):
        mask = identifiers == bin_id
        subset = [row for row, keep in zip(rows, mask) if keep]
        output.append({
            "domain_size_bin": bin_id,
            "num_units": len(subset),
            "domain_size_min": float(sizes[mask].min()) if mask.any() else math.nan,
            "domain_size_max": float(sizes[mask].max()) if mask.any() else math.nan,
            "spearman_delta_average_task_ce": spearman([_float(row, "delta_average_at_28") for row in subset], [_float(row, "mean_ce_increase") for row in subset]),
            "spearman_delta_total_task_ce": spearman([_float(row, "delta_total_at_28") for row in subset], [_float(row, "mean_ce_increase") for row in subset]),
        })
    return output


def _global_index_key(row: Mapping[str, object]) -> int:
    return int(row["global_index"])


def matched_domain_size(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ffn = [row for row in rows if row["unit_type"] == task020.TYPE_FFN]
    output = []
    for row in ffn:
        candidates = [other for other in ffn if int(other["global_index"]) != int(row["global_index"])]
        if not candidates:
            continue
        target_size = _int(row, "domain_valid_demand_count")
        def candidate_key(other: Mapping[str, object]) -> tuple[int, int]:
            return (abs(_int(other, "domain_valid_demand_count") - target_size), _int(other, "global_index"))
        match = min(candidates, key=candidate_key)
        output.append({
            "global_index": int(row["global_index"]),
            "matched_global_index": int(match["global_index"]),
            "domain_size": _int(row, "domain_valid_demand_count"),
            "matched_domain_size": _int(match, "domain_valid_demand_count"),
            "absolute_domain_size_difference": abs(_int(row, "domain_valid_demand_count") - _int(match, "domain_valid_demand_count")),
            "delta_average": _float(row, "delta_average_at_28"),
            "matched_delta_average": _float(match, "delta_average_at_28"),
            "delta_total": _float(row, "delta_total_at_28"),
            "matched_delta_total": _float(match, "delta_total_at_28"),
            "task_ce": _float(row, "mean_ce_increase"),
            "matched_task_ce": _float(match, "mean_ce_increase"),
        })
    return output


def _rank_residual(values: Sequence[float], control: Sequence[float]) -> np.ndarray:
    y, x = importance_percentile(values), importance_percentile(control)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def domain_size_controlled(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    task = [_float(row, "mean_ce_increase") for row in rows]
    average = [_float(row, "delta_average_at_28") for row in rows]
    total = [_float(row, "delta_total_at_28") for row in rows]
    size = [math.log10(_int(row, "domain_valid_demand_count")) for row in rows]
    total_residual = _rank_residual(total, size)
    average_residual = _rank_residual(average, size)
    task_residual = _rank_residual(task, size)
    return [
        {"analysis": "delta_total_rank_residual_vs_task_ce", "spearman": spearman(total_residual, task), "num_units": len(rows), "control": "rank_residualized_by_log10_domain_size"},
        {"analysis": "delta_average_rank_residual_vs_task_ce", "spearman": spearman(average_residual, task), "num_units": len(rows), "control": "rank_residualized_by_log10_domain_size"},
        {"analysis": "delta_average_vs_task_ce_both_residualized", "spearman": spearman(average_residual, task_residual), "num_units": len(rows), "control": "rank_residualized_by_log10_domain_size"},
        {"analysis": "delta_total_vs_task_ce_both_residualized", "spearman": spearman(total_residual, task_residual), "num_units": len(rows), "control": "rank_residualized_by_log10_domain_size"},
    ]


def rank_disagreement(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{
        "global_index": int(row["global_index"]),
        "group": row["group"],
        "unit_type": row["unit_type"],
        "delta_average_percentile": float(row["delta_average_percentile"]),
        "delta_total_percentile": float(row["delta_total_percentile"]),
        "rank_gap": float(row["rank_gap"]),
        "task_ce": _float(row, "mean_ce_increase"),
        "task_ce_percentile": float(row["task_ce_percentile"]),
    } for row in rows]


def disagreement_enrichment(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    values = np.asarray([float(row["rank_gap"]) for row in rows])
    task = np.asarray([float(row["task_ce_percentile"]) for row in rows])
    order = np.argsort(-values, kind="mergesort")
    count = max(1, int(math.ceil(len(rows) * .10)))
    labels = np.full(len(rows), "middle_80", dtype=object)
    labels[order[:count]] = "top_positive_10"
    labels[order[-count:]] = "bottom_negative_10"
    output = []
    for label in ("top_positive_10", "bottom_negative_10", "middle_80"):
        mask = labels == label
        output.append({
            "region": label,
            "num_units": int(mask.sum()),
            "spearman_rank_gap_task_ce_percentile": spearman(values[mask], task[mask]),
            "high_task_risk_top10_fraction": float(np.mean(task[mask] >= .90)) if mask.any() else math.nan,
            "mean_rank_gap": float(values[mask].mean()) if mask.any() else math.nan,
        })
    return output


def pareto_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ranks = pareto_front_ranks([_float(row, "delta_average_at_28") for row in rows], [_float(row, "delta_total_at_28") for row in rows])
    return [{
        "global_index": int(row["global_index"]), "pareto_front_rank": int(rank),
        "is_first_front": bool(rank == 1), "delta_average": _float(row, "delta_average_at_28"),
        "delta_total": _float(row, "delta_total_at_28"), "task_ce": _float(row, "mean_ce_increase"),
        "task_ce_percentile": float(row["task_ce_percentile"]), "group": row["group"],
        "unit_type": row["unit_type"],
    } for row, rank in zip(rows, ranks)]


def pareto_enrichment(rows: Sequence[Mapping[str, object]], pareto: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for label, selected in (
        ("front_1", [row for row in pareto if int(row["pareto_front_rank"]) == 1]),
        ("front_2", [row for row in pareto if int(row["pareto_front_rank"]) == 2]),
        ("front_3", [row for row in pareto if int(row["pareto_front_rank"]) == 3]),
        ("remaining", [row for row in pareto if int(row["pareto_front_rank"]) > 3]),
    ):
        task = [_float(row, "task_ce") for row in selected]
        output.append({
            "region": label, "num_units": len(selected),
            "median_task_ce": float(np.median(task)) if task else math.nan,
            "high_task_risk_top10_fraction": float(np.mean([_float(row, "task_ce_percentile") >= .90 for row in selected])) if selected else math.nan,
            "attention_count": sum(row["unit_type"] == task020.TYPE_ATTENTION for row in selected),
            "ffn_count": sum(row["unit_type"] == task020.TYPE_FFN for row in selected),
            "median_domain_size": float(np.median([_int(next(source for source in rows if int(source["global_index"]) == int(row["global_index"])), "domain_valid_demand_count") for row in selected])) if selected else math.nan,
        })
    return output


def dual_risk_retrieval(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    task = [_float(row, "mean_ce_increase") for row in rows]
    average = np.asarray([float(row["delta_average_percentile"]) for row in rows])
    total = np.asarray([float(row["delta_total_percentile"]) for row in rows])
    pareto = -pareto_front_ranks([_float(row, "delta_average_at_28") for row in rows], [_float(row, "delta_total_at_28") for row in rows]).astype(np.float64)
    signals = {
        "delta_average": average,
        "delta_total": total,
        "max_percentile_dual_risk": np.maximum(average, total),
        "min_percentile_dual_risk": np.minimum(average, total),
        "pareto_front_rank": pareto,
    }
    output = []
    populations = {"all": list(range(len(rows))), "ffn": [i for i, row in enumerate(rows) if row["unit_type"] == task020.TYPE_FFN]}
    for population, positions in populations.items():
        for signal, values in signals.items():
            for fraction, label in ((.10, "10"), (.20, "20"), (.25, "25")):
                result = top_risk_retrieval(values[positions], np.asarray(task)[positions], fraction=fraction)
                output.append({"population": population, "measure": signal, "task_risk_fraction": label, **result})
    return output


def ffn_dual_analysis(rows: Sequence[Mapping[str, object]], pareto: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ffn = [row for row in rows if row["unit_type"] == task020.TYPE_FFN]
    if not ffn:
        return []
    average = [_float(row, "delta_average_at_28") for row in ffn]
    total = [_float(row, "delta_total_at_28") for row in ffn]
    task = [_float(row, "mean_ce_increase") for row in ffn]
    gaps = [float(row["rank_gap"]) for row in ffn]
    output = [
        {"analysis": "ffn_spearman_delta_average_vs_ce", "value": spearman(average, task), "num_units": len(ffn)},
        {"analysis": "ffn_spearman_delta_total_vs_ce", "value": spearman(total, task), "num_units": len(ffn)},
        {"analysis": "ffn_spearman_rank_gap_vs_ce", "value": spearman(gaps, task), "num_units": len(ffn)},
    ]
    ffn_indices = {int(row["global_index"]) for row in ffn}
    ffn_pareto = [row for row in pareto if int(row["global_index"]) in ffn_indices]
    for region, selected in (("front_1", [row for row in ffn_pareto if int(row["pareto_front_rank"]) == 1]),
                             ("front_2", [row for row in ffn_pareto if int(row["pareto_front_rank"]) == 2]),
                             ("front_3", [row for row in ffn_pareto if int(row["pareto_front_rank"]) == 3])):
        output.append({"analysis": f"ffn_pareto_{region}_top10_fraction", "value": float(np.mean([_float(row, "task_ce_percentile") >= .90 for row in selected])) if selected else math.nan, "num_units": len(selected)})
    ffn_positions = [index for index, row in enumerate(rows) if row["unit_type"] == task020.TYPE_FFN]
    retrieval = dual_risk_retrieval(rows)
    for measure in ("delta_average", "delta_total", "max_percentile_dual_risk", "min_percentile_dual_risk", "pareto_front_rank"):
        for fraction in ("10", "20", "25"):
            match = next(row for row in retrieval if row["population"] == "ffn" and row["measure"] == measure and row["task_risk_fraction"] == fraction)
            output.append({"analysis": f"ffn_retrieval_{measure}_top{fraction}", "value": match["recall"], "num_units": len(ffn_positions)})
    output.extend({"analysis": f"ffn_quadrant_{label}_median_ce", "value": stat["median_ce_increase"], "num_units": stat["num_units"]}
                   for label, stat in ((row["quadrant"], row) for row in quadrant_statistics(ffn)))
    return output


def type_confound(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    def average_value(row: Mapping[str, object]) -> float:
        return _float(row, "delta_average_at_28")
    def total_value(row: Mapping[str, object]) -> float:
        return _float(row, "delta_total_at_28")
    def gap_value(row: Mapping[str, object]) -> float:
        return float(row["rank_gap"])
    measures = (("delta_average", average_value), ("delta_total", total_value), ("rank_gap", gap_value))
    for measure, getter in measures:
        for population, subset in _population_rows(rows).items():
            output.append({
                "measure": measure, "population": population, "num_units": len(subset),
                "spearman_task_ce": spearman([getter(row) for row in subset], [_float(row, "mean_ce_increase") for row in subset]),
                "attention_median": float(np.median([getter(row) for row in subset if row["unit_type"] == task020.TYPE_ATTENTION])) if any(row["unit_type"] == task020.TYPE_ATTENTION for row in subset) else math.nan,
                "ffn_median": float(np.median([getter(row) for row in subset if row["unit_type"] == task020.TYPE_FFN])) if any(row["unit_type"] == task020.TYPE_FFN for row in subset) else math.nan,
                "interpretation": "attention_descriptive_only" if population == "attention_descriptive_only" else "diagnostic",
            })
    return output


def domain_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_int(row, "domain_id")].append(row)
    output = []
    for domain_id, subset in sorted(grouped.items()):
        average = [_float(row, "delta_average_at_28") for row in subset]
        total = [_float(row, "delta_total_at_28") for row in subset]
        task = [_float(row, "mean_ce_increase") for row in subset]
        output.append({
            "domain_id": domain_id,
            "domain_valid_demand_count": _int(subset[0], "domain_valid_demand_count"),
            "num_investigated_units": len(subset),
            "mean_delta_average": float(np.mean(average)), "max_delta_average": float(np.max(average)),
            "mean_delta_total": float(np.mean(total)), "sum_delta_total": float(np.sum(total)),
            "max_delta_total": float(np.max(total)), "mean_task_ce": float(np.mean(task)),
            "max_task_ce": float(np.max(task)),
        })
    for key in ("mean_delta_average", "mean_delta_total", "mean_task_ce"):
        rank = importance_percentile([row[key] for row in output])
        for row, value in zip(output, rank):
            row[f"{key}_rank"] = float(value)
    return output


def domain_size_sort_key(item: tuple[int, Sequence[Mapping[str, object]]]) -> tuple[int, int]:
    domain_id, subset = item
    return (_int(subset[0], "domain_valid_demand_count"), int(domain_id))


def domain_size_cases(rows: Sequence[Mapping[str, object]], attention_indices: Sequence[int]) -> list[dict[str, object]]:
    by_domain: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[_int(row, "domain_id")].append(row)
    ordered_pairs = sorted(by_domain.items(), key=domain_size_sort_key)
    ordered = [domain for domain, _ in ordered_pairs]
    selected = set(ordered[:3] + ordered[-3:])
    attention_set = set(int(index) for index in attention_indices)
    attention_domains = {
        _int(row, "domain_id") for row in rows
        if _int(row, "global_index") in attention_set
    }
    selected.update(attention_domains)
    output = []
    for domain_id in sorted(selected):
        subset = by_domain[domain_id]
        for row in subset:
            output.append({
                "case_group": "attention_domain" if domain_id in attention_domains else ("smallest_domains" if domain_id in set(ordered[:3]) else "largest_domains"),
                "domain_id": domain_id,
                "domain_valid_demand_count": _int(row, "domain_valid_demand_count"),
                "global_index": _int(row, "global_index"),
                "unit_type": row["unit_type"],
                "delta_average": _float(row, "delta_average_at_28"),
                "delta_total": _float(row, "delta_total_at_28"),
                "delta_average_percentile": float(row["delta_average_percentile"]),
                "delta_total_percentile": float(row["delta_total_percentile"]),
                "task_ce": _float(row, "mean_ce_increase"),
                "task_ce_percentile": float(row["task_ce_percentile"]),
            })
    return output


def _rescue(rows: Sequence[Mapping[str, object]], *, low_signal: str, split_signal: str) -> list[dict[str, object]]:
    low = [row for row in rows if float(row[f"{low_signal}_percentile"]) <= .25]
    if not low:
        return [{"subset": "empty", "num_units": 0, "median_task_ce": math.nan, "mean_task_ce": math.nan, "attention_count": 0, "ffn_count": 0}]
    split_key = {
        "delta_average": "delta_average_at_28",
        "delta_total": "delta_total_at_28",
    }.get(split_signal, split_signal)
    median = float(np.median([_float(row, split_key) for row in low]))
    groups = {
        "low_signal_low_split": [row for row in low if _float(row, split_key) < median],
        "low_signal_high_split": [row for row in low if _float(row, split_key) >= median],
    }
    output = []
    for label, subset in groups.items():
        task = [_float(row, "mean_ce_increase") for row in subset]
        output.append({
            "subset": label, "low_signal": low_signal, "split_signal": split_signal,
            "num_units": len(subset), "median_task_ce": float(np.median(task)) if task else math.nan,
            "mean_task_ce": float(np.mean(task)) if task else math.nan,
            "attention_count": sum(row["unit_type"] == task020.TYPE_ATTENTION for row in subset),
            "ffn_count": sum(row["unit_type"] == task020.TYPE_FFN for row in subset),
            "high_task_risk_top10_fraction": float(np.mean([float(row["task_ce_percentile"]) >= .90 for row in subset])) if subset else math.nan,
        })
    return output


def complementarity_evidence(rows: Sequence[Mapping[str, object]], retrieval: Sequence[Mapping[str, object]],
                             enrichment: Sequence[Mapping[str, object]], pareto_stats: Sequence[Mapping[str, object]],
                             ffn_stats: Sequence[Mapping[str, object]], size_corr: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    average_rescue = _rescue(rows, low_signal="delta_total", split_signal="delta_average")
    total_rescue = _rescue(rows, low_signal="delta_average", split_signal="delta_total")
    avg_high = next((row for row in average_rescue if row["subset"] == "low_signal_high_split"), {})
    avg_low = next((row for row in average_rescue if row["subset"] == "low_signal_low_split"), {})
    total_high = next((row for row in total_rescue if row["subset"] == "low_signal_high_split"), {})
    total_low = next((row for row in total_rescue if row["subset"] == "low_signal_low_split"), {})
    positive = next((row for row in enrichment if row["region"] == "top_positive_10"), {})
    negative = next((row for row in enrichment if row["region"] == "bottom_negative_10"), {})
    ffn_gap = next((row for row in ffn_stats if row["analysis"] == "ffn_spearman_rank_gap_vs_ce"), {})
    size_task = next((row for row in size_corr if row["population"] == "all_investigated_units" and row["target"] == "task_ce"), {})
    return [
        {"hypothesis": "average_rescues_total", "supported": bool(float(avg_high.get("mean_task_ce", math.nan)) > float(avg_low.get("mean_task_ce", math.nan))), "metric": "low-total split mean CE", "value": float(avg_high.get("mean_task_ce", math.nan)) - float(avg_low.get("mean_task_ce", math.nan)), "interpretation": "High Delta_average among low-Delta_total units is more task-dangerous."},
        {"hypothesis": "total_rescues_average", "supported": bool(float(total_high.get("mean_task_ce", math.nan)) > float(total_low.get("mean_task_ce", math.nan))), "metric": "low-average split mean CE", "value": float(total_high.get("mean_task_ce", math.nan)) - float(total_low.get("mean_task_ce", math.nan)), "interpretation": "High Delta_total among low-Delta_average units is more task-dangerous."},
        {"hypothesis": "positive_rank_gap_task_enrichment", "supported": bool(float(positive.get("high_task_risk_top10_fraction", math.nan)) > float(negative.get("high_task_risk_top10_fraction", math.nan))), "metric": "top-positive vs bottom-negative top10 fraction", "value": float(positive.get("high_task_risk_top10_fraction", math.nan)) - float(negative.get("high_task_risk_top10_fraction", math.nan)), "interpretation": "Positive rank-gap tail is enriched relative to negative tail."},
        {"hypothesis": "pareto_task_enrichment", "supported": bool(float(next((row["high_task_risk_top10_fraction"] for row in pareto_stats if row["region"] == "front_1"), math.nan)) > float(next((row["high_task_risk_top10_fraction"] for row in pareto_stats if row["region"] == "remaining"), math.nan))), "metric": "Pareto front-1 vs remaining top10 fraction", "value": float(next((row["high_task_risk_top10_fraction"] for row in pareto_stats if row["region"] == "front_1"), math.nan)) - float(next((row["high_task_risk_top10_fraction"] for row in pareto_stats if row["region"] == "remaining"), math.nan)), "interpretation": "Jointly high average/total risk enriches task risk only if this comparison supports it."},
        {"hypothesis": "ffn_only_complementarity", "supported": bool(abs(float(ffn_gap.get("value", math.nan))) >= .10), "metric": "FFN rank-gap/CE Spearman", "value": ffn_gap.get("value", math.nan), "interpretation": "A positive or negative FFN-only association indicates whether complementarity survives type control."},
        {"hypothesis": "domain_size_confounding", "supported": bool(abs(float(size_task.get("spearman", math.nan))) >= .10), "metric": "domain size vs task CE Spearman", "value": size_task.get("spearman", math.nan), "interpretation": "Domain-size association is a scale confound, not proof that average is semantic noise."},
        {"hypothesis": "attention_type_confound", "supported": bool(sum(row["unit_type"] == task020.TYPE_ATTENTION for row in rows) == 3), "metric": "Attention count", "value": 3, "interpretation": "Attention values are descriptive and never sufficient for a general claim."},
    ]


def _write_figures(output_dir: Path, rows: Sequence[Mapping[str, object]], quadrants: Sequence[Mapping[str, object]],
                   pareto: Sequence[Mapping[str, object]], rescue_average: Sequence[Mapping[str, object]],
                   rescue_total: Sequence[Mapping[str, object]], conditional: Sequence[Mapping[str, object]],
                   attention_indices: Sequence[int], domain_rows: Sequence[Mapping[str, object]]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional environment dependency
        return f"NOT_RUN: {type(exc).__name__}: {exc}"
    root = Path(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    average = np.asarray([_float(row, "delta_average_at_28") for row in rows])
    total = np.asarray([_float(row, "delta_total_at_28") for row in rows])
    ce = np.asarray([_float(row, "mean_ce_increase") for row in rows])
    size = np.asarray([_int(row, "domain_valid_demand_count") for row in rows])
    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    eps = max(float(min(np.min(average[average > 0]) if np.any(average > 0) else 1.0, np.min(total[total > 0]) if np.any(total > 0) else 1.0)) * 1e-6, 1e-12)
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); scatter = ax.scatter(np.log10(average + eps), np.log10(total + eps), c=importance_percentile(ce), s=13, cmap="viridis"); ax.set(xlabel="log10(Delta_average + eps)", ylabel="log10(Delta_total + eps)"); fig.colorbar(scatter, ax=ax, label="Task CE percentile"); save(fig, "figure01_average_total_task_risk")
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); scatter = ax.scatter(np.log10(average + eps), np.log10(total + eps), c=size, s=13, cmap="plasma"); ax.set(xlabel="log10(Delta_average + eps)", ylabel="log10(Delta_total + eps)"); fig.colorbar(scatter, ax=ax, label="N_valid"); save(fig, "figure02_average_total_domain_size")
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); colors = ["tab:orange" if row["unit_type"] == task020.TYPE_ATTENTION else "tab:blue" for row in rows]; ax.scatter(np.log10(average + eps), np.log10(total + eps), c=colors, s=13); ax.set(xlabel="log10(Delta_average + eps)", ylabel="log10(Delta_total + eps)"); save(fig, "figure03_average_total_unit_type")
    attention_set = set(int(index) for index in attention_indices); fig, ax = plt.subplots(figsize=(5.8, 4.3)); ax.scatter(np.log10(average + eps), np.log10(total + eps), c="lightgray", s=10); highlighted = [row for row in rows if int(row["global_index"]) in attention_set]; ax.scatter([math.log10(_float(row, "delta_average_at_28") + eps) for row in highlighted], [math.log10(_float(row, "delta_total_at_28") + eps) for row in highlighted], c="red", s=38); [ax.annotate(str(row["global_index"]), (math.log10(_float(row, "delta_average_at_28") + eps), math.log10(_float(row, "delta_total_at_28") + eps))) for row in highlighted]; ax.set(xlabel="log10(Delta_average + eps)", ylabel="log10(Delta_total + eps)"); save(fig, "figure04_attention_heads")
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); ax.scatter([float(row["rank_gap"]) for row in rows], [float(row["task_ce_percentile"]) for row in rows], s=12); ax.axvline(0, color="k", linewidth=.8); ax.set(xlabel="Delta_average percentile - Delta_total percentile", ylabel="Task CE percentile"); save(fig, "figure05_rank_gap_task_risk")
    fig, ax = plt.subplots(figsize=(7.0, 4.0)); ax.bar([str(row["quadrant"]).replace("_", "\n") for row in quadrants], [float(row["median_ce_increase"]) if math.isfinite(float(row["median_ce_increase"])) else 0.0 for row in quadrants]); ax.set_ylabel("Median CE increase"); save(fig, "figure06_quadrant_task_risk")
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); labels = [str(row["subset"]) for row in rescue_average]; ax.bar(labels, [float(row["mean_task_ce"]) if math.isfinite(float(row["mean_task_ce"])) else 0.0 for row in rescue_average]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Mean CE increase"); save(fig, "figure07_average_rescue_low_total")
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); labels = [str(row["subset"]) for row in rescue_total]; ax.bar(labels, [float(row["mean_task_ce"]) if math.isfinite(float(row["mean_task_ce"])) else 0.0 for row in rescue_total]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Mean CE increase"); save(fig, "figure08_total_rescue_low_average")
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); rank = np.asarray([int(row["pareto_front_rank"]) for row in pareto]); scatter = ax.scatter(np.log10(average + eps), np.log10(total + eps), c=rank, s=15, cmap="turbo"); ax.set(xlabel="log10(Delta_average + eps)", ylabel="log10(Delta_total + eps)"); fig.colorbar(scatter, ax=ax, label="Pareto front rank"); save(fig, "figure09_pareto_task_risk")
    ffn = [row for row in rows if row["unit_type"] == task020.TYPE_FFN]; fig, ax = plt.subplots(figsize=(5.8, 4.3)); ax.scatter([math.log10(_float(row, "delta_average_at_28") + eps) for row in ffn], [math.log10(_float(row, "delta_total_at_28") + eps) for row in ffn], c=[_float(row, "mean_ce_increase") for row in ffn], s=13, cmap="viridis"); ax.set(xlabel="FFN log10(Delta_average + eps)", ylabel="FFN log10(Delta_total + eps)"); save(fig, "figure10_ffn_dual_risk_plane")
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); labels = [str(row["domain_size_bin"]) for row in conditional]; ax.plot(labels, [float(row["spearman_delta_average_task_ce"]) for row in conditional], marker="o", label="Delta_average"); ax.plot(labels, [float(row["spearman_delta_total_task_ce"]) for row in conditional], marker="o", label="Delta_total"); ax.axhline(0, color="k", linewidth=.8); ax.set(xlabel="Domain-size quartile", ylabel="Spearman with task CE"); ax.legend(); save(fig, "figure11_domain_size_conditional")
    return "COMPLETE"


def _diagnosis(rows: Sequence[Mapping[str, object]], evidence: Sequence[Mapping[str, object]],
               identity: Mapping[str, object], eps: float) -> str:
    by_hypothesis = {str(row["hypothesis"]): row for row in evidence}
    def val(name: str) -> object:
        return by_hypothesis.get(name, {}).get("value", "unavailable")
    attention = [row for row in rows if row["unit_type"] == task020.TYPE_ATTENTION]
    return f"""# Task022: Average–Total Dual-Risk Complementarity Diagnosis

This is an offline, descriptive analysis over exactly `{len(rows)}` Task020-labelled units. Task020 and Task021 artifacts are read-only. No model, validation, descriptor extraction, BMS, Contribution Field extraction, pruning, or GPU experiment was executed.

The log-plane numerical constant was `eps={eps:.12g}`. It is used only for visualization, never as a pruning score. The exact identity tested for every row was `Delta_total = N_valid * Delta_average`.

## Answers

1. **Redundancy or complementarity?** Evidence value: `{val('average_rescues_total')}` for average rescuing low-total units and `{val('total_rescues_average')}` for the converse. No conclusion is forced beyond the saved evidence table.
2. **Attention high-average/low-total?** The exact Attention identities and percentile regions are in `attention_average_total_trajectory.csv` and `high_average_low_total_units.csv`; there are `{len(attention)}` descriptive Attention units.
3. **Explained by small domains?** Domain-size correlations and matched comparisons are in `domain_size_dual_risk_correlation.csv` and `matched_domain_size_risk_comparison.csv`; scaling association is not treated as semantic proof.
4. **Average rescue?** See `average_rescue_within_low_total.csv` and the `average_rescues_total` row in `complementarity_evidence.csv`.
5. **Total rescue?** See `total_rescue_within_low_average.csv` and the `total_rescues_average` row.
6. **Within FFNs?** The FFN-only result is in `ffn_dual_risk_analysis.csv`; the three Attention units are excluded from that control.
7. **How much is domain-size scaling?** Rank residual and quartile-controlled results are in `domain_size_controlled_task_correlation.csv` and `conditional_domain_size_correlation.csv`.
8. **Which signal after control?** Compare the controlled correlations directly; no learned predictor or score is introduced.
9. **Pareto enrichment?** `pareto_task_risk_enrichment.csv` and `dual_risk_pareto_frontier.csv` report the descriptive result.
10. **Does max-percentile help?** `dual_risk_retrieval.csv` reports max/min and scalar probes; max-percentile is not used for pruning.
11. **Two semantic axes or a power-law interpolation?** This task does not fit or propose a new mixing coefficient, exponent, or pruning method. The evidence only distinguishes scale association from residual task association.
12. **Next mechanism?** No mechanism is implemented. Any next experiment must be separately approved after reviewing `complementarity_evidence.csv`.

## Identity and safeguards

- Task020 identity: PASS; Task021 identity: PASS; exact 417-unit cohort: PASS.
- Average–total mathematical identity: PASS for every investigated unit.
- Production pruning sources and Task020/Task021 artifacts: unchanged and read-only.
- GPU pruning/validation: not executed.
- No type quota, Attention protection, energy weighting, selector change, or new hyperparameter was introduced.
"""


def run_analysis(task020_root: Path, task021_root: Path, output_dir: Path,
                 *, repo_root: Path | None = None) -> dict[str, object]:
    destination = Path(output_dir)
    identity = verify_task022_identity(task020_root, task021_root, repo_root=repo_root)
    rows = _normalize_rows(task020_root, task021_root, identity["cohort_set"])
    identity_rows = verify_average_total_identity(rows)
    atomic_csv(destination / "average_total_identity.csv", tuple(identity_rows[0]), identity_rows)
    rows = _attach_dual_risk(rows)
    plane = build_average_total_risk_plane(rows)
    atomic_csv(destination / "average_total_risk_plane.csv", tuple(plane[0]), plane)
    quadrants = quadrant_statistics(rows)
    atomic_csv(destination / "dual_risk_quadrant_statistics.csv", tuple(quadrants[0]), quadrants)
    high_low = _region_rows(rows, "high_average_low_total")
    low_high = _region_rows(rows, "low_average_high_total")
    atomic_csv(destination / "high_average_low_total_units.csv", tuple(high_low[0]) if high_low else ("status",), high_low or [{"status": "EMPTY"}])
    atomic_csv(destination / "low_average_high_total_units.csv", tuple(low_high[0]) if low_high else ("status",), low_high or [{"status": "EMPTY"}])
    validation = _full_validation_map(task020_root)
    trajectory = attention_trajectory(rows, identity["attention_global_indices"], validation)
    atomic_csv(destination / "attention_average_total_trajectory.csv", tuple(trajectory[0]), trajectory)
    size_corr = domain_size_correlations(rows)
    atomic_csv(destination / "domain_size_dual_risk_correlation.csv", tuple(size_corr[0]), size_corr)
    conditional = conditional_domain_size(rows)
    atomic_csv(destination / "conditional_domain_size_correlation.csv", tuple(conditional[0]), conditional)
    matched = matched_domain_size(rows)
    atomic_csv(destination / "matched_domain_size_risk_comparison.csv", tuple(matched[0]) if matched else ("status",), matched or [{"status": "EMPTY"}])
    controlled = domain_size_controlled(rows)
    atomic_csv(destination / "domain_size_controlled_task_correlation.csv", tuple(controlled[0]), controlled)
    disagreement = rank_disagreement(rows)
    atomic_csv(destination / "average_total_rank_disagreement.csv", tuple(disagreement[0]), disagreement)
    enrichment = disagreement_enrichment(rows)
    atomic_csv(destination / "rank_disagreement_task_enrichment.csv", tuple(enrichment[0]), enrichment)
    pareto = pareto_rows(rows)
    atomic_csv(destination / "dual_risk_pareto_frontier.csv", tuple(pareto[0]), pareto)
    pareto_stats = pareto_enrichment(rows, pareto)
    atomic_csv(destination / "pareto_task_risk_enrichment.csv", tuple(pareto_stats[0]), pareto_stats)
    retrieval = dual_risk_retrieval(rows)
    atomic_csv(destination / "dual_risk_retrieval.csv", tuple(retrieval[0]), retrieval)
    ffn_stats = ffn_dual_analysis(rows, pareto)
    atomic_csv(destination / "ffn_dual_risk_analysis.csv", tuple(ffn_stats[0]), ffn_stats)
    confound = type_confound(rows)
    atomic_csv(destination / "dual_risk_type_confound.csv", tuple(confound[0]), confound)
    domains = domain_summary(rows)
    atomic_csv(destination / "domain_dual_risk_summary.csv", tuple(domains[0]), domains)
    cases = domain_size_cases(rows, identity["attention_global_indices"])
    atomic_csv(destination / "domain_size_case_studies.csv", tuple(cases[0]) if cases else ("status",), cases or [{"status": "EMPTY"}])
    rescue_average = _rescue(rows, low_signal="delta_total", split_signal="delta_average")
    rescue_total = _rescue(rows, low_signal="delta_average", split_signal="delta_total")
    atomic_csv(destination / "average_rescue_within_low_total.csv", tuple(rescue_average[0]), rescue_average)
    atomic_csv(destination / "total_rescue_within_low_average.csv", tuple(rescue_total[0]), rescue_total)
    evidence = complementarity_evidence(rows, retrieval, enrichment, pareto_stats, ffn_stats, size_corr)
    atomic_csv(destination / "complementarity_evidence.csv", tuple(evidence[0]), evidence)
    average = np.asarray([_float(row, "delta_average_at_28") for row in rows])
    total = np.asarray([_float(row, "delta_total_at_28") for row in rows])
    eps = max(float(min(np.min(average[average > 0]) if np.any(average > 0) else 1.0, np.min(total[total > 0]) if np.any(total > 0) else 1.0)) * 1e-6, 1e-12)
    figure_status = _write_figures(destination, rows, quadrants, pareto, rescue_average, rescue_total, conditional, identity["attention_global_indices"], domains)
    atomic_json(destination / "artifact_identity.json", {
        "code_version": CODE_VERSION,
        "task020_identity_pass": True, "task021_identity_pass": True,
        "cohort_identity_pass": True, "average_total_identity_pass": True,
        "task020_identity_sha256": identity["task020_identity_sha256"],
        "task021_identity_sha256": identity["task021_identity_sha256"],
        "task020_root": identity["task020_root"], "task021_root": identity["task021_root"],
        "descriptor_units": EXPECTED_UNITS, "bms_domains": EXPECTED_DOMAINS,
        "cohort_units": EXPECTED_COHORT_UNITS,
        "attention_global_indices": identity["attention_global_indices"],
        "replacement_ffn_global_indices": identity["replacement_ffn_global_indices"],
        "production_source_git_blob_sha": identity["production_source_git_blob_sha"],
        "task020_artifacts_modified": False, "task021_artifacts_modified": False,
        "production_pruning_code_modified": False, "gpu_pruning_or_validation_executed": False,
        "log10_eps": eps,
    })
    summary = _summary(rows, retrieval, pareto, identity["attention_global_indices"])
    atomic_csv(destination / "task022_summary.csv", tuple(summary[0]), summary)
    completion = {
        "status": "PASS", "code_version": CODE_VERSION,
        "task020_identity_pass": True, "task021_identity_pass": True,
        "cohort_identity_pass": True, "average_total_identity_pass": True,
        "dual_risk_plane_complete": True, "quadrant_analysis_complete": True,
        "rank_disagreement_complete": True, "average_rescue_analysis_complete": True,
        "total_rescue_analysis_complete": True, "pareto_analysis_complete": True,
        "ffn_only_analysis_complete": True, "type_confound_control_complete": True,
        "analysis_complete": True,
        "production_pruning_code_modified": False,
        "gpu_pruning_or_validation_executed": False,
        "task020_artifacts_modified": False, "task021_artifacts_modified": False,
        "cohort_units": EXPECTED_COHORT_UNITS, "log10_eps": eps,
        "figure_status": figure_status,
        "task020_identity_sha256": identity["task020_identity_sha256"],
        "task021_identity_sha256": identity["task021_identity_sha256"],
    }
    atomic_json(destination / "task022_completion.json", completion)
    (destination / "diagnosis.md").write_text(_diagnosis(rows, evidence, identity, eps), encoding="utf-8")
    return {"output_dir": str(destination.resolve()), "rows": len(rows), "figure_status": figure_status, "identity": identity}


def _summary(rows: Sequence[Mapping[str, object]], retrieval: Sequence[Mapping[str, object]],
             pareto: Sequence[Mapping[str, object]], attention_indices: Sequence[int]) -> list[dict[str, object]]:
    average = np.asarray([float(row["delta_average_percentile"]) for row in rows])
    total = np.asarray([float(row["delta_total_percentile"]) for row in rows])
    pareto_score = -np.asarray([int(row["pareto_front_rank"]) for row in pareto], dtype=np.float64)
    task = np.asarray([_float(row, "mean_ce_increase") for row in rows])
    output = []
    signals = {
        "delta_average": average, "delta_total": total,
        "max_percentile_dual_risk": np.maximum(average, total),
        "min_percentile_dual_risk": np.minimum(average, total),
        "pareto_rank": pareto_score,
    }
    ffn_positions = [i for i, row in enumerate(rows) if row["unit_type"] == task020.TYPE_FFN]
    for name, values in signals.items():
        all_retrieval = {label: top_risk_retrieval(values, task, fraction=fraction)["recall"] for fraction, label in ((.10, "10"), (.20, "20"), (.25, "25"))}
        ffn_retrieval = {label: top_risk_retrieval(values[ffn_positions], task[ffn_positions], fraction=fraction)["recall"] for fraction, label in ((.10, "10"), (.20, "20"), (.25, "25"))}
        output.append({
            "signal": name, "spearman_ce_all": spearman(values, task),
            "spearman_ce_ffn": spearman(values[ffn_positions], task[ffn_positions]),
            "recall_taskrisk_top10_all": all_retrieval["10"], "recall_taskrisk_top20_all": all_retrieval["20"], "recall_taskrisk_top25_all": all_retrieval["25"],
            "recall_taskrisk_top10_ffn": ffn_retrieval["10"], "recall_taskrisk_top20_ffn": ffn_retrieval["20"],
            "attention1549_risk_percentile": _attention_value(rows, attention_indices[0], values),
            "attention20908_risk_percentile": _attention_value(rows, attention_indices[1], values),
            "attention24002_risk_percentile": _attention_value(rows, attention_indices[2], values),
        })
    return output


def _attention_value(rows: Sequence[Mapping[str, object]], index: int, values: Sequence[float]) -> float | str:
    for position, row in enumerate(rows):
        if int(row["global_index"]) == index:
            return float(values[position])
    return ""


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task020-dir", type=Path, required=True)
    parser.add_argument("--task021-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_analysis(args.task020_dir, args.task021_dir, args.output_dir, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

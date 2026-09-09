"""Offline Task021 diagnosis of information retained through Task020 stages.

Task021 consumes the PASS artifacts produced by Task020.  It never loads a
model, recomputes activations, runs validation, changes a pruning score, or
modifies a Task020 artifact.  All arrays used by the main analysis are
one-dimensional vectors indexed by ``global_index``; CSV rows are the exact
Task020 investigated population and are never silently expanded with task
labels for uninvestigated units.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import task020_functional_demand_task_importance as task020


CODE_VERSION = "task021_information_retention_diagnosis_v1"
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
ATTENTION_IDENTITIES = (1549, 24002, 20908)
GROUPS = (
    task020.GROUP_ATTENTION,
    task020.GROUP_REPLACEMENT,
    task020.GROUP_ORDINARY,
    task020.GROUP_CONTROL,
)
TASK_METRICS = (
    "mean_ce_increase",
    "mean_margin_drop",
    "mean_true_logit_drop",
    "correct_to_wrong_flip_rate",
)
LADDER_TASK_METRICS = TASK_METRICS + ("mean_kl",)
SIGNALS = (
    "D_abs",
    "D_rel",
    "D_dyn",
    "functional_energy",
    "best_substitute_similarity_at_28",
    "delta_average_at_28",
    "delta_total_at_28",
)
HIGHER_IS_IMPORTANT = {
    "D_abs": True,
    "D_rel": True,
    "D_dyn": True,
    "functional_energy": True,
    "best_substitute_similarity_at_28": False,
    "delta_average_at_28": True,
    "delta_total_at_28": True,
}
REQUIRED_TASK020_FILES = (
    "task020_completion.json",
    "artifact_identity.json",
    "cohort_reconstruction.json",
    "task020_cohorts.csv",
    "unit_task_importance.csv",
    "group_task_importance_statistics.csv",
    "functional_vs_task_correlation.csv",
    "descriptor_task_correlation.csv",
    "functional_task_mismatch.csv",
    "matched_functional_risk_comparison.csv",
    "attention_case_studies.csv",
    "attention_joint_ablation.csv",
    "replacement_ffn_joint_ablation.csv",
    "matched_budget_joint_comparison.csv",
    "full_validation_task_importance.csv",
)


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(row: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Missing numeric field {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric field {key}")
    return result


def _int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing integer field {key}")
    return int(value)


def _require_true(payload: Mapping[str, object], key: str, label: str) -> None:
    if payload.get(key) is not True:
        raise RuntimeError(f"{label} identity field {key!r} is not true")


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average-tie ranks for a finite one-dimensional vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("rankdata expects a finite one-dimensional vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and array[order[end]] == array[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def importance_percentile(values: Sequence[float], *, higher_is_important: bool = True) -> np.ndarray:
    """Return [0, 1] percentiles where larger always means more important."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("importance_percentile expects finite one-dimensional values")
    transformed = array if higher_is_important else -array
    ranks = rankdata(transformed)
    return (ranks - 1.0) / max(1, len(array) - 1)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-aware Spearman correlation without a scipy dependency."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall tau-b for modest diagnostic populations.

    Task020 labels cover a small investigated population, so the explicit
    pair count is deterministic and avoids adding scipy as a dependency.
    """
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return math.nan
    concordant = discordant = ties_x = ties_y = ties_both = 0
    for i in range(len(x) - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        xzero, yzero = dx == 0, dy == 0
        ties_x += int(xzero.sum())
        ties_y += int(yzero.sum())
        ties_both += int((xzero & yzero).sum())
        concordant += int(((dx > 0) & (dy > 0) | (dx < 0) & (dy < 0)).sum())
        discordant += int(((dx > 0) & (dy < 0) | (dx < 0) & (dy > 0)).sum())
    denominator = math.sqrt((concordant + discordant + ties_x - ties_both) *
                            (concordant + discordant + ties_y - ties_both))
    return float((concordant - discordant) / denominator) if denominator else math.nan


def rank_loss(source_a_percentile: float, source_b_percentile: float) -> float:
    """Diagnostic rank displacement; it is not an information-theoretic metric."""
    return float(source_a_percentile - source_b_percentile)


def quantile_bin_ids(values: Sequence[float], bins: int = 10) -> np.ndarray:
    """Deterministic equal-frequency bin IDs, preserving input row alignment."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or bins < 1:
        raise ValueError("quantile_bin_ids expects a non-empty finite vector")
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.int64)
    for rank, index in enumerate(order):
        result[index] = min(bins - 1, (rank * bins) // len(array))
    return result


def top_risk_retrieval(signal: Sequence[float], task_risk: Sequence[float], *,
                       signal_higher_is_important: bool, fraction: float) -> dict[str, float | int]:
    """Descriptive precision/recall at a fixed signal and task-risk fraction."""
    values = np.asarray(signal, dtype=np.float64)
    risk = np.asarray(task_risk, dtype=np.float64)
    if values.shape != risk.shape or values.ndim != 1 or not len(values):
        raise ValueError("signal and task risk must be aligned non-empty vectors")
    count = max(1, int(math.ceil(len(values) * fraction)))
    signal_order = np.argsort(-values if signal_higher_is_important else values, kind="mergesort")
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


def nearest_delta_match(rows: Sequence[Mapping[str, object]], target_index: int,
                        *, delta_key: str = "delta_total_at_28") -> Mapping[str, object]:
    """Return the nearest different row in Delta_total with a stable tie-break."""
    target = next(row for row in rows if _int(row, "global_index") == int(target_index))
    candidates = [row for row in rows if _int(row, "global_index") != int(target_index)]
    if not candidates:
        raise ValueError("At least two rows are required for nearest matching")
    target_delta = _float(target, delta_key)
    return min(candidates, key=lambda row: (abs(_float(row, delta_key) - target_delta),
                                             _int(row, "global_index")))


def raw_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Signed cosine; no absolute value or positive clamp is applied."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("raw_cosine expects equally shaped vectors")
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else math.nan


def _git_identity(repo_root: Path) -> dict[str, dict[str, str]]:
    """Use Task020's strict Git-aware source gate without importing models."""
    return task020.production_source_git_identity(Path(repo_root))


def verify_task020_identity(task020_root: Path, *, repo_root: Path | None = None) -> dict[str, object]:
    """Validate Task020 PASS artifacts and return immutable identity metadata."""
    root = Path(task020_root)
    missing = [name for name in REQUIRED_TASK020_FILES if not (root / name).is_file()]
    missing += [name for name in ("baseline_28_cache.json", "baseline_28_outputs.npz",
                                  "diagnostic_samples.csv") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Task020 PASS artifacts: {missing}")
    completion = read_json(root / "task020_completion.json")
    identity = read_json(root / "artifact_identity.json")
    cohorts_gate = read_json(root / "cohort_reconstruction.json")
    if completion.get("status") != "PASS" or completion.get("code_version") != task020.CODE_VERSION:
        raise RuntimeError("Task020 completion is not PASS at the final code version")
    for key in ("artifact_identity_pass", "artifact_cache_identity_pass", "analysis_complete"):
        _require_true(completion, key, "Task020 completion")
    if completion.get("production_pruning_code_modified") is not False:
        raise RuntimeError("Task020 reports modified production pruning code")
    if identity.get("code_version") != task020.CODE_VERSION:
        raise RuntimeError("Task020 artifact identity uses an unexpected code version")
    _require_true(identity, "artifact_identity_pass", "Task020 artifact")
    _require_true(cohorts_gate, "cohorts_reconstructed", "Task020 cohort")
    if int(identity.get("descriptor_units", -1)) != EXPECTED_UNITS:
        raise RuntimeError("Task020 descriptor unit count is not 36,378")
    if int(identity.get("bms_domains", -1)) != EXPECTED_DOMAINS:
        raise RuntimeError("Task020 BMS domain count is not 423")
    if not task020.baseline_cache_valid(root):
        raise RuntimeError("Task020 baseline cache identity is stale")
    for worker in (0, 1):
        path = root / "workers" / f"worker{worker}_completion.json"
        if not task020.result_cache_valid(path, root, worker=worker, workers=2):
            raise RuntimeError(f"Task020 worker{worker} cache identity is stale")
    expected_cohorts = read_csv(root / "task020_cohorts.csv")
    attention = [int(row["global_index"]) for row in expected_cohorts
                 if row.get("cohort") == task020.GROUP_ATTENTION]
    replacement = [int(row["global_index"]) for row in expected_cohorts
                   if row.get("cohort") == task020.GROUP_REPLACEMENT]
    if attention != list(ATTENTION_IDENTITIES):
        raise RuntimeError(f"Task020 Attention identities/order changed: {attention}")
    if list(identity.get("attention_global_indices", [])) != attention:
        raise RuntimeError("Task020 Attention identity does not match cohort CSV order")
    if list(identity.get("replacement_ffn_global_indices", [])) != replacement:
        raise RuntimeError("Task020 replacement FFN identity does not match cohort CSV order")
    if len({int(row["global_index"]) for row in expected_cohorts}) != len(expected_cohorts):
        raise RuntimeError("Task020 cohort rows contain duplicate global indices")
    unit_rows = read_csv(root / "unit_task_importance.csv")
    cohort_set = {int(row["global_index"]) for row in expected_cohorts}
    unit_set = {int(row["global_index"]) for row in unit_rows}
    if len(unit_rows) != len(unit_set) or unit_set != cohort_set:
        raise RuntimeError("Task020 unit_task_importance row set differs from cohorts")
    for cohort, sequence in ((task020.GROUP_ATTENTION, attention),
                             (task020.GROUP_REPLACEMENT, replacement)):
        path = root / "joint" / f"{cohort}.json"
        if not task020.result_cache_valid(path, root, cohort=cohort,
                                          cohort_sequence_sha256=task020.sequence_sha256(sequence)):
            raise RuntimeError(f"Task020 {cohort} joint cache identity is stale")
    for variant in ("baseline_28", *(f"attention_{index}" for index in attention),
                    "attention_joint", "replacement_ffn_joint"):
        if not task020.full_validation_cache_valid(root, variant):
            raise RuntimeError(f"Task020 full-validation cache identity is stale: {variant}")
    source_ids = _git_identity(repo_root or Path(__file__).resolve().parent)
    return {
        "task020_root": str(root.resolve()),
        "code_version": completion["code_version"],
        "descriptor_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS,
        "attention_global_indices": attention,
        "replacement_ffn_global_indices": replacement,
        "cohort_sha256": sha256_file(root / "task020_cohorts.csv"),
        "unit_task_importance_sha256": sha256_file(root / "unit_task_importance.csv"),
        "production_source_git_blob_sha": source_ids,
        "task020_identity_sha256": sha256_file(root / "artifact_identity.json"),
        "task020_completion_sha256": sha256_file(root / "task020_completion.json"),
    }


def _descriptor_file(identity: Mapping[str, object]) -> Path | None:
    task014 = Path(str(identity.get("task014_root", "")))
    task016 = Path(str(identity.get("task016_root", "")))
    candidates = (
        task016 / "domain_total" / "s30" / "model_output" / "dynamic3d" / "seed3407" / "descriptor_statistics.csv",
        task016 / "domain_total" / "s30" / "dynamic3d" / "seed3407" / "descriptor_statistics.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if task014.is_dir():
        matches = [candidate for candidate in sorted(task014.rglob("descriptor_statistics.csv"))
                   if candidate.is_file()]
        preferred = [candidate for candidate in matches
                     if "s30" in str(candidate) or "domain_total" in str(candidate)]
        if len(preferred) == 1:
            return preferred[0]
        if len(matches) == 1:
            return matches[0]
    return None


def load_global_context(identity: Mapping[str, object]) -> dict[str, object]:
    """Load saved 36,378-unit arrays when available, never recomputing them."""
    context: dict[str, object] = {"scope": "investigated", "size": 0, "arrays": {}}
    arrays: dict[str, np.ndarray] = {}
    energy_path = Path(str(identity.get("task015_root", ""))) / "functional_energy.npy"
    if energy_path.is_file():
        energy = np.load(energy_path, mmap_mode="r", allow_pickle=False)
        if energy.shape == (EXPECTED_UNITS,) and np.isfinite(energy).all():
            arrays["functional_energy"] = np.asarray(energy, dtype=np.float64)
    descriptor_path = _descriptor_file(identity)
    if descriptor_path is not None:
        descriptor_rows = read_csv(descriptor_path)
        if len(descriptor_rows) == EXPECTED_UNITS:
            values = {name: np.full(EXPECTED_UNITS, np.nan, dtype=np.float64)
                      for name in ("D_abs", "D_rel", "D_dyn")}
            valid = True
            for row in descriptor_rows:
                index = _int(row, "global_index")
                if not 0 <= index < EXPECTED_UNITS:
                    valid = False; break
                for name in values:
                    values[name][index] = _float(row, name)
            if valid and all(np.isfinite(value).all() for value in values.values()):
                arrays.update(values)
    snapshot_path = Path(str(identity.get("task017_root", ""))) / "replay" / "domain_total" / "candidate_snapshots.npz"
    if snapshot_path.is_file():
        with np.load(snapshot_path, allow_pickle=False) as payload:
            prefix = "p28_"
            required = ("global_index", "delta_average", "delta_total", "best_similarity")
            if all(prefix + key in payload for key in required):
                indices = payload[prefix + "global_index"].astype(np.int64, copy=False)
                if len(indices) == EXPECTED_UNITS and len(set(indices.tolist())) == EXPECTED_UNITS:
                    for key, output_name in (("delta_average", "delta_average_at_28"),
                                              ("delta_total", "delta_total_at_28"),
                                              ("best_similarity", "best_substitute_similarity_at_28")):
                        values = np.full(EXPECTED_UNITS, np.nan, dtype=np.float64)
                        values[indices] = payload[prefix + key].astype(np.float64, copy=False)
                        if not np.isfinite(values).all():
                            break
                        arrays[output_name] = values
    if len(arrays) >= 7 and all(value.shape == (EXPECTED_UNITS,) for value in arrays.values()):
        context.update({"scope": "all_36378", "size": EXPECTED_UNITS, "arrays": arrays})
    else:
        context.update({"scope": "investigated", "size": 0, "arrays": arrays})
    return context


def _stage_from_layer(layer: str) -> str:
    match = re.search(r"layers\.(\d+)", str(layer))
    return f"stage_{match.group(1)}" if match else "unknown"


def load_investigated_rows(task020_root: Path) -> list[dict[str, object]]:
    rows = read_csv(Path(task020_root) / "unit_task_importance.csv")
    if not rows:
        raise RuntimeError("Task020 unit_task_importance.csv is empty")
    result: list[dict[str, object]] = []
    for source in rows:
        row: dict[str, object] = dict(source)
        row["global_index"] = _int(source, "global_index")
        row["group"] = str(source.get("group", source.get("cohort", "")))
        row["unit_type"] = str(source.get("unit_type", ""))
        row["layer"] = str(source.get("layer", ""))
        row["stage"] = str(source.get("stage", "")) or _stage_from_layer(str(source.get("layer", "")))
        row["domain_id"] = _int(source, "domain_id")
        for name in SIGNALS + LADDER_TASK_METRICS:
            row[name] = _float(source, name)
        result.append(row)
    return result


def _global_percentiles(rows: Sequence[Mapping[str, object]], context: Mapping[str, object],
                        signal: str) -> np.ndarray:
    values = np.asarray([_float(row, signal) for row in rows], dtype=np.float64)
    arrays = context.get("arrays", {})
    if context.get("scope") == "all_36378" and signal in arrays:
        global_values = np.asarray(arrays[signal], dtype=np.float64)
        global_percentiles = importance_percentile(global_values,
                                                   higher_is_important=HIGHER_IS_IMPORTANT[signal])
        return global_percentiles[np.asarray([_int(row, "global_index") for row in rows], dtype=np.int64)]
    return importance_percentile(values, higher_is_important=HIGHER_IS_IMPORTANT[signal])


def attach_percentiles(rows: Sequence[Mapping[str, object]], context: Mapping[str, object]) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    for signal in SIGNALS:
        values = _global_percentiles(output, context, signal)
        name = f"{signal}_percentile"
        for row, value in zip(output, values):
            row[name] = float(value)
    ce = importance_percentile([_float(row, "mean_ce_increase") for row in output])
    for row, value in zip(output, ce):
        row["task_ce_risk_percentile"] = float(value)
    return output


def build_ladder(rows: Sequence[Mapping[str, object]], full_validation: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    full_by_variant = {str(row.get("variant")): row for row in full_validation}
    output = []
    for row in rows:
        index = _int(row, "global_index")
        full = full_by_variant.get(f"attention_{index}", {})
        output.append({
            "global_index": index,
            "group": row["group"],
            "unit_type": row["unit_type"],
            "layer": row["layer"],
            "stage": row["stage"],
            "domain_id": row["domain_id"],
            **{name: row[name] for name in LADDER_TASK_METRICS},
            **{name: row[name] for name in SIGNALS},
            "functional_rank_at_28": row.get("functional_rank_at_28", ""),
            "domain_coverage_at_28": row.get("domain_coverage_at_28", ""),
            "full_validation_top1_drop": _float(full, "top1_drop_from_28", default=math.nan)
            if full else "",
        })
    return output


def _rows_by_group(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    return {group: [row for row in rows if row.get("group") == group] for group in GROUPS}


def _correlation_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    populations = {
        "all_investigated_units": list(rows),
        "investigated_ffn_only": [row for row in rows if row["unit_type"] == task020.TYPE_FFN],
        "replacement_ffn_only": [row for row in rows if row["group"] == task020.GROUP_REPLACEMENT],
        "ordinary_deleted_ffn": [row for row in rows if row["group"] == task020.GROUP_ORDINARY],
        "matched_surviving_ffn": [row for row in rows if row["group"] == task020.GROUP_CONTROL],
        "attention_descriptive_only": [row for row in rows if row["group"] == task020.GROUP_ATTENTION],
    }
    output = []
    for population, subset in populations.items():
        for signal in SIGNALS:
            for metric in TASK_METRICS:
                output.append({
                    "population": population,
                    "signal": signal,
                    "task_metric": metric,
                    "num_units": len(subset),
                    "interpretation": "descriptive_only" if population == "attention_descriptive_only" else "diagnostic",
                    "spearman": spearman([_float(row, signal) for row in subset],
                                          [_float(row, metric) for row in subset]),
                    "kendall_tau": kendall_tau([_float(row, signal) for row in subset],
                                                [_float(row, metric) for row in subset]),
                })
    return output


def _retrieval_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    risk = [_float(row, "mean_ce_increase") for row in rows]
    output = []
    for signal in SIGNALS:
        for fraction, label in ((.10, "10"), (.20, "20"), (.25, "25")):
            result = top_risk_retrieval([_float(row, signal) for row in rows], risk,
                                        signal_higher_is_important=HIGHER_IS_IMPORTANT[signal], fraction=fraction)
            output.append({"signal": signal, "task_risk_fraction": label, **result})
    return output


def _rank_transition(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for left in SIGNALS + ("mean_ce_increase",):
        for right in SIGNALS + ("mean_ce_increase",):
            x = [(_float(row, left) if left in SIGNALS else _float(row, "mean_ce_increase")) for row in rows]
            y = [(_float(row, right) if right in SIGNALS else _float(row, "mean_ce_increase")) for row in rows]
            output.append({"signal_a": left, "signal_b": right, "num_units": len(rows),
                           "spearman": spearman(x, y), "kendall_tau": kendall_tau(x, y)})
    return output


def _energy_bins(rows: Sequence[Mapping[str, object]], bins: int = 10) -> list[dict[str, object]]:
    delta = np.asarray([_float(row, "delta_total_at_28") for row in rows], dtype=np.float64)
    energy = np.asarray([_float(row, "functional_energy") for row in rows], dtype=np.float64)
    risk = np.asarray([_float(row, "mean_ce_increase") for row in rows], dtype=np.float64)
    bin_ids = quantile_bin_ids(delta, bins)
    output = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        values = energy[mask]
        if not len(values):
            output.append({
                "delta_total_bin": bin_id, "num_units": 0,
                "delta_total_min": "", "delta_total_max": "",
                "spearman_energy_task_ce": math.nan,
                "lower_energy_half_mean_task_ce": "",
                "upper_energy_half_mean_task_ce": "",
                "upper_minus_lower_task_ce": "",
            })
            continue
        order = np.argsort(values, kind="mergesort")
        half = max(1, len(values) // 2)
        output.append({
            "delta_total_bin": bin_id,
            "num_units": int(mask.sum()),
            "delta_total_min": float(delta[mask].min()),
            "delta_total_max": float(delta[mask].max()),
            "spearman_energy_task_ce": spearman(values, risk[mask]),
            "lower_energy_half_mean_task_ce": float(risk[mask][order[:half]].mean()),
            "upper_energy_half_mean_task_ce": float(risk[mask][order[-half:]].mean()),
            "upper_minus_lower_task_ce": float(risk[mask][order[-half:]].mean() - risk[mask][order[:half]].mean()),
        })
    return output


def _matched_delta(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    risk = np.asarray([_float(row, "mean_ce_increase") for row in rows])
    count = max(1, int(math.ceil(len(rows) * .10)))
    selected_positions = set(np.argsort(-risk, kind="mergesort")[:count].tolist())
    selected_positions.update(
        position for position, row in enumerate(rows)
        if int(row["global_index"]) in ATTENTION_IDENTITIES
    )
    output = []
    for position in sorted(selected_positions, key=lambda item: (-risk[item], int(rows[item]["global_index"]))):
        row = rows[int(position)]
        match = nearest_delta_match(rows, _int(row, "global_index"))
        output.append({
            "high_task_risk_global_index": _int(row, "global_index"),
            "matched_global_index": _int(match, "global_index"),
            "high_task_risk_group": row["group"],
            "matched_group": match["group"],
            "absolute_delta_total_difference": abs(_float(row, "delta_total_at_28") - _float(match, "delta_total_at_28")),
            **{f"high_{name}": row[name] for name in ("functional_energy", "D_abs", "D_rel", "D_dyn", "mean_ce_increase")},
            **{f"matched_{name}": match[name] for name in ("functional_energy", "D_abs", "D_rel", "D_dyn", "mean_ce_increase")},
        })
    return output


def _domain_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["domain_id"])].append(row)
    output = []
    for domain_id, values in sorted(grouped.items()):
        task = np.asarray([_float(row, "mean_ce_increase") for row in values])
        energy = np.asarray([_float(row, "functional_energy") for row in values])
        delta = np.asarray([_float(row, "delta_total_at_28") for row in values])
        output.append({
            "domain_id": domain_id,
            "num_investigated_units": len(values),
            "mean_functional_energy": float(energy.mean()),
            "median_functional_energy": float(np.median(energy)),
            "mean_delta_total": float(delta.mean()),
            "median_delta_total": float(np.median(delta)),
            "mean_task_sensitivity": float(task.mean()),
            "median_task_sensitivity": float(np.median(task)),
        })
    for key in ("mean_functional_energy", "mean_delta_total", "mean_task_sensitivity"):
        percentiles = importance_percentile([row[key] for row in output]) if output else np.array([])
        for row, value in zip(output, percentiles):
            row[f"{key}_rank"] = float(value)
    return output


def _proxy_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    task = [_float(row, "mean_ce_increase") for row in rows]
    proxies = {
        "functional_energy": [_float(row, "functional_energy") for row in rows],
        "D_abs": [_float(row, "D_abs") for row in rows],
        "D_rel": [_float(row, "D_rel") for row in rows],
        "functional_energy_times_delta": [_float(row, "functional_energy") * _float(row, "delta_total_at_28") for row in rows],
        "D_abs_times_delta": [_float(row, "D_abs") * _float(row, "delta_total_at_28") for row in rows],
        "D_rel_times_delta": [_float(row, "D_rel") * _float(row, "delta_total_at_28") for row in rows],
    }
    output = []
    for name, values in proxies.items():
        retrieval = top_risk_retrieval(values, task, signal_higher_is_important=True, fraction=.10)
        attention = [row for row in rows if row["group"] == task020.GROUP_ATTENTION]
        ranks = importance_percentile(values)
        index_to_rank = {int(row["global_index"]): float(ranks[position]) for position, row in enumerate(rows)}
        output.append({"proxy": name, "spearman_ce": spearman(values, task),
                       "recall_taskrisk_top10": retrieval["recall"],
                       "precision_taskrisk_top10": retrieval["precision"],
                       "attention1549_percentile": index_to_rank.get(1549, ""),
                       "attention20908_percentile": index_to_rank.get(20908, ""),
                       "attention24002_percentile": index_to_rank.get(24002, ""),
                       "replacement_ffn_median_percentile": float(np.median([index_to_rank[int(row["global_index"])] for row in rows if row["group"] == task020.GROUP_REPLACEMENT])) if any(row["group"] == task020.GROUP_REPLACEMENT for row in rows) else "",
                       "attention_count": len(attention)})
    return output


def _magnitude_direction_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    task = [_float(row, "mean_ce_increase") for row in rows]
    output = []
    for signal in ("functional_energy", "D_abs", "D_rel", "D_dyn",
                   "best_substitute_similarity_at_28", "delta_total_at_28"):
        values = [_float(row, signal) for row in rows]
        retrieval = top_risk_retrieval(
            values, task, signal_higher_is_important=HIGHER_IS_IMPORTANT[signal], fraction=.10
        )
        output.append({
            "signal": signal,
            "signal_direction": "higher_is_important" if HIGHER_IS_IMPORTANT[signal] else "lower_is_important",
            "spearman_task_ce": spearman(values, task),
            "recall_taskrisk_top10": retrieval["recall"],
            "precision_taskrisk_top10": retrieval["precision"],
            "attention1549_rank": "",
            "attention20908_rank": "",
            "attention24002_rank": "",
        })
    return output


def _retention_summary(rows: Sequence[Mapping[str, object]], retrieval: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    task = [_float(row, "mean_ce_increase") for row in rows]
    task_p = importance_percentile(task)
    output = []
    names = (
        ("descriptor_abs", "D_abs"),
        ("descriptor_rel", "D_rel"),
        ("descriptor_dyn", "D_dyn"),
        ("raw_field_energy", "functional_energy"),
        ("direction_similarity", "best_substitute_similarity_at_28"),
        ("functional_marginal_loss", "delta_total_at_28"),
    )
    for stage, signal in names:
        values = [_float(row, signal) for row in rows]
        percentile = importance_percentile(values, higher_is_important=HIGHER_IS_IMPORTANT[signal])
        displacement = percentile - importance_percentile(
            [_float(row, "delta_total_at_28") for row in rows]
        )
        result = {
            "stage": stage,
            "signal": signal,
            "spearman_vs_ce": spearman(values, task),
            "spearman_vs_margin": spearman(values, [_float(row, "mean_margin_drop") for row in rows]),
            "spearman_vs_true_logit_drop": spearman(values, [_float(row, "mean_true_logit_drop") for row in rows]),
            "recall_taskrisk_top10": next(row["recall"] for row in retrieval if row["signal"] == signal and row["task_risk_fraction"] == "10"),
            "recall_taskrisk_top20": next(row["recall"] for row in retrieval if row["signal"] == signal and row["task_risk_fraction"] == "20"),
            "recall_taskrisk_top25": next(row["recall"] for row in retrieval if row["signal"] == signal and row["task_risk_fraction"] == "25"),
            "mean_rank_loss_to_delta_total": float(np.mean(displacement)),
            "q90_rank_loss_to_delta_total": float(np.quantile(displacement, .90)),
            "positive_rank_loss_fraction": float(np.mean(displacement > 0)),
            "num_units": len(rows),
        }
        output.append(result)
    return output


def _attention_decomposition(rows: Sequence[Mapping[str, object]], task020_root: Path) -> list[dict[str, object]]:
    matches = read_csv(task020_root / "matched_functional_risk_comparison.csv")
    by_index = {int(row["global_index"]): row for row in rows}
    output = []
    for match in matches:
        head = int(match["attention_global_index"])
        substitute = int(match["ffn_global_index"])
        if head not in by_index or substitute not in by_index:
            continue
        h, s = by_index[head], by_index[substitute]
        hnorm, snorm = _float(h, "functional_energy"), _float(s, "functional_energy")
        output.append({
            "head_global_index": head,
            "substitute_global_index": substitute,
            "head_norm": hnorm,
            "substitute_norm": snorm,
            "norm_ratio_head_over_substitute": hnorm / snorm if abs(snorm) > 1e-15 else math.nan,
            "signed_cosine": "",
            "current_clamped_similarity": _float(h, "best_substitute_similarity_at_28"),
            "head_task_sensitivity": _float(h, "mean_ce_increase"),
            "substitute_task_sensitivity": _float(s, "mean_ce_increase"),
            "raw_field_status": "NOT_RUN_raw_field_not_loaded",
        })
    return output


def _loss_evidence(rows: Sequence[Mapping[str, object]], bins: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    energy = [_float(row, "functional_energy") for row in rows]
    dabs = [_float(row, "D_abs") for row in rows]
    drel = [_float(row, "D_rel") for row in rows]
    ddyn = [_float(row, "D_dyn") for row in rows]
    delta = [_float(row, "delta_total_at_28") for row in rows]
    task = [_float(row, "mean_ce_increase") for row in rows]
    conditional = [row["spearman_energy_task_ce"] for row in bins if math.isfinite(float(row["spearman_energy_task_ce"]))]
    conditional_mean = float(np.mean(conditional)) if conditional else math.nan
    corr = {name: spearman(values, task) for name, values in (("D_abs", dabs), ("D_rel", drel),
                                                               ("D_dyn", ddyn), ("functional_energy", energy),
                                                               ("delta_total", delta))}
    delta_corr = abs(corr["delta_total"]) if math.isfinite(corr["delta_total"]) else 0.0
    energy_corr = abs(corr["functional_energy"]) if math.isfinite(corr["functional_energy"]) else 0.0
    rows_out = [
        ("descriptor_information_insufficient", max(abs(corr["D_abs"]), abs(corr["D_rel"]), abs(corr["D_dyn"])),
         "descriptor/task Spearman", "Descriptor sources are insufficient only if none retain task ordering."),
        ("field_magnitude_lost_by_normalization", conditional_mean,
         "within-Delta_total-bin energy/task Spearman", "Positive conditional association supports discarded magnitude information."),
        ("direction_only_similarity_insufficient", delta_corr,
         "Delta_total/task CE absolute Spearman", "Low final functional-risk association indicates directional aggregation loss."),
        ("max_coverage_aggregation_loss", delta_corr,
         "Delta_total/task CE absolute Spearman", "This is descriptive; no causal stage separation is asserted."),
        ("marginal_loss_aggregation_loss", delta_corr,
         "Delta_total/task CE absolute Spearman", "This is descriptive; no causal stage separation is asserted."),
    ]
    output = []
    for hypothesis, value, metric, interpretation in rows_out:
        threshold = .10
        output.append({"hypothesis": hypothesis, "supported": bool(math.isfinite(float(value)) and abs(float(value)) >= threshold),
                       "evidence_metric": metric, "value": value, "interpretation": interpretation})
    return output


def _write_figures(output_dir: Path, rows: Sequence[Mapping[str, object]], trajectories: Sequence[Mapping[str, object]],
                   retrieval: Sequence[Mapping[str, object]], bins: Sequence[Mapping[str, object]], domains: Sequence[Mapping[str, object]]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent optional dependency
        return f"NOT_RUN: {type(exc).__name__}: {exc}"
    root = Path(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    task = np.asarray([_float(row, "mean_ce_increase") for row in rows])
    delta = np.asarray([_float(row, "delta_total_at_28") for row in rows])
    energy = np.asarray([_float(row, "functional_energy") for row in rows])
    task_p = importance_percentile(task)
    delta_p = importance_percentile(delta)
    energy_p = importance_percentile(energy)

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2)); ax.scatter(delta_p, task_p, s=10); ax.set(xlabel="Delta_total percentile", ylabel="Task CE-risk percentile"); save(fig, "figure01_task_vs_delta_percentile")
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); ax.scatter(energy_p, task_p, s=10); ax.set(xlabel="Functional-energy percentile", ylabel="Task CE-risk percentile"); save(fig, "figure02_task_vs_energy_percentile")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(row["global_index"]) for row in trajectories]
    fields = ("D_abs_percentile", "D_rel_percentile", "D_dyn_percentile", "functional_energy_percentile", "delta_total_at_28_percentile", "task_ce_risk_percentile")
    for field in fields:
        ax.plot(labels, [float(row[field]) for row in trajectories], marker="o", label=field.replace("_percentile", ""))
    ax.set(xlabel="Attention global index", ylabel="Importance percentile"); ax.legend(fontsize=7); save(fig, "figure03_attention_information_trajectory")
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); labels = ["D_abs", "energy", "Delta_total", "task risk"]; values = [float(np.mean([_float(row, "D_abs_percentile") for row in rows])), float(np.mean([_float(row, "functional_energy_percentile") for row in rows])), float(np.mean([_float(row, "delta_total_at_28_percentile") for row in rows])), float(np.mean([_float(row, "task_ce_risk_percentile") for row in rows]))]; ax.bar(labels, values); ax.set_ylim(0, 1); ax.set_ylabel("Mean importance percentile"); save(fig, "figure04_rank_loss_waterfall")
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); scatter = ax.scatter(delta, task, c=energy, s=12); ax.set(xlabel="Delta_total", ylabel="Mean CE increase"); fig.colorbar(scatter, ax=ax, label="Functional energy"); save(fig, "figure05_delta_vs_ce_energy")
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); colors = ["tab:orange" if row["unit_type"] == task020.TYPE_ATTENTION else "tab:blue" for row in rows]; ax.scatter(energy, task, c=colors, s=12); ax.set(xlabel="Functional energy", ylabel="Mean CE increase"); save(fig, "figure06_energy_vs_ce_type")
    populated_bins = [row for row in bins if row["num_units"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.plot([int(row["delta_total_bin"]) for row in populated_bins], [float(row["upper_minus_lower_task_ce"]) for row in populated_bins], marker="o"); ax.axhline(0, color="k", linewidth=.8); ax.set(xlabel="Delta_total quantile bin", ylabel="Upper-minus-lower energy CE"); save(fig, "figure07_within_delta_energy")
    selected = [row for row in retrieval if row["task_risk_fraction"] == "10"]; fig, ax = plt.subplots(figsize=(7.0, 4.0)); ax.bar([str(row["signal"]) for row in selected], [float(row["recall"]) for row in selected]); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=60); ax.set_ylabel("Recall@TaskRisk10%"); save(fig, "figure08_high_task_risk_retrieval")
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); ax.axis("off"); ax.text(.02, .9, "Attention / best-substitute decomposition", fontsize=12); y=.76; 
    for row in trajectories:
        ax.text(.03, y, f"{row['global_index']}: energy={float(row['functional_energy']):.4g}, similarity={float(row['best_substitute_similarity_at_28']):.4g}"); y -= .1
    save(fig, "figure09_attention_magnitude_direction")
    ffn = [row for row in retrieval if row["signal"] in ("D_abs", "D_rel", "D_dyn", "functional_energy", "delta_total_at_28") and row["task_risk_fraction"] == "10"]; fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.bar([str(row["signal"]) for row in ffn], [float(row["recall"]) for row in ffn]); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Recall (diagnostic) "); save(fig, "figure10_ffn_signal_comparison")
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.scatter([float(row.get("mean_functional_energy_rank", 0.0)) for row in domains], [float(row.get("mean_task_sensitivity_rank", 0.0)) for row in domains], s=14); ax.set(xlabel="Domain energy rank", ylabel="Domain task rank"); save(fig, "figure11_domain_rank_mismatch")
    return "COMPLETE"


def run_analysis(task020_root: Path, output_dir: Path, *, repo_root: Path | None = None) -> dict[str, object]:
    """Run all Task021 offline diagnostics and write the required artifacts."""
    root, destination = Path(task020_root), Path(output_dir)
    identity = verify_task020_identity(root, repo_root=repo_root)
    rows = load_investigated_rows(root)
    context = load_global_context(read_json(root / "artifact_identity.json"))
    rows = attach_percentiles(rows, context)
    full_validation = read_csv(root / "full_validation_task_importance.csv")
    ladder = build_ladder(rows, full_validation)
    atomic_csv(destination / "information_retention_ladder.csv", tuple(ladder[0]), ladder)
    percentile_fields = ("global_index", "group", "D_abs_percentile", "D_rel_percentile", "D_dyn_percentile",
                         "functional_energy_percentile", "best_substitute_similarity_at_28_percentile",
                         "delta_average_at_28_percentile", "delta_total_at_28_percentile", "task_ce_risk_percentile")
    percentile_rows = [{field: row[field] for field in percentile_fields} for row in rows]
    atomic_csv(destination / "information_rank_percentiles.csv", percentile_fields, percentile_rows)
    attention_rows = [row for row in rows if row["group"] == task020.GROUP_ATTENTION]
    if [int(row["global_index"]) for row in attention_rows] != list(ATTENTION_IDENTITIES):
        attention_rows = sorted(attention_rows, key=lambda row: list(ATTENTION_IDENTITIES).index(int(row["global_index"])))
    trajectory_fields = ("global_index", "layer", "D_abs", "D_rel", "D_dyn", "functional_energy",
                         "best_substitute_similarity_at_28", "delta_total_at_28", "D_abs_percentile", "D_rel_percentile",
                         "D_dyn_percentile", "functional_energy_percentile", "delta_total_at_28_percentile",
                         "task_ce_risk_percentile", "mean_ce_increase", "full_validation_top1_drop")
    trajectory = [{field: row.get(field, "") for field in trajectory_fields} for row in attention_rows]
    atomic_csv(destination / "attention_information_trajectory.csv", trajectory_fields, trajectory)
    rank_loss_rows = []
    for left, right in (("D_abs_percentile", "delta_total_at_28_percentile"),
                        ("D_rel_percentile", "delta_total_at_28_percentile"),
                        ("functional_energy_percentile", "delta_total_at_28_percentile"),
                        ("delta_total_at_28_percentile", "task_ce_risk_percentile")):
        for row in rows:
            rank_loss_rows.append({"global_index": row["global_index"], "source_a": left, "source_b": right,
                                   "rank_loss": rank_loss(float(row[left]), float(row[right]))})
    atomic_csv(destination / "rank_information_loss.csv", ("global_index", "source_a", "source_b", "rank_loss"), rank_loss_rows)
    correlations = _correlation_rows(rows)
    atomic_csv(destination / "stagewise_task_correlation.csv", tuple(correlations[0]), correlations)
    retrieval = _retrieval_rows(rows)
    atomic_csv(destination / "task_risk_retrieval.csv", tuple(retrieval[0]), retrieval)
    transition = _rank_transition(rows)
    atomic_csv(destination / "rank_transition_matrix.csv", tuple(transition[0]), transition)
    bins = _energy_bins(rows)
    atomic_csv(destination / "energy_within_functional_risk_bins.csv", tuple(bins[0]), bins)
    matched = _matched_delta(rows)
    atomic_csv(destination / "matched_delta_task_mismatch.csv", tuple(matched[0]) if matched else ("status",), matched or [{"status": "NO_MATCH"}])
    decomposition = _attention_decomposition(rows, root)
    if decomposition:
        atomic_csv(destination / "attention_best_substitute_decomposition.csv", tuple(decomposition[0]), decomposition)
    else:
        atomic_csv(destination / "attention_best_substitute_decomposition.csv", ("status",), [{"status": "NO_MATCHES"}])
    magnitude_direction = _magnitude_direction_rows(rows)
    atomic_csv(destination / "magnitude_direction_comparison.csv", tuple(magnitude_direction[0]), magnitude_direction)
    proxies = _proxy_rows(rows)
    atomic_csv(destination / "diagnostic_proxy_comparison.csv", tuple(proxies[0]), proxies)
    groups = _rows_by_group(rows)
    confound = []
    for signal in SIGNALS:
        attention = groups[task020.GROUP_ATTENTION]; ffn = [row for row in rows if row["unit_type"] == task020.TYPE_FFN]
        confound.append({"signal": signal,
                         "attention_median": float(np.median([_float(row, signal) for row in attention])) if attention else math.nan,
                         "ffn_median": float(np.median([_float(row, signal) for row in ffn])) if ffn else math.nan,
                         "ffn_only_spearman_task_ce": spearman([_float(row, signal) for row in ffn], [_float(row, "mean_ce_increase") for row in ffn])})
    atomic_csv(destination / "type_confound_control.csv", tuple(confound[0]), confound)
    domains = _domain_rows(rows)
    for row in domains:
        row["mean_functional_energy_rank"] = row.pop("mean_functional_energy_rank", row.get("mean_functional_energy_rank", ""))
        row["mean_task_sensitivity_rank"] = row.pop("mean_task_sensitivity_rank", row.get("mean_task_sensitivity_rank", ""))
    atomic_csv(destination / "domain_information_retention.csv", tuple(domains[0]), domains)
    loss = _loss_evidence(rows, bins)
    atomic_csv(destination / "information_loss_evidence.csv", tuple(loss[0]), loss)
    raw_path = Path("/home/jixinye25/jxy_work1/swintrans_LGFR_protection_v3_20260731/cstc_probe_full/cstc_probe_arrays.npz")
    raw_status = "NOT_RUN: raw Contribution Field arrays were not loaded or recomputed"
    # The raw-field pair decomposition is optional and deliberately disabled by
    # default.  Leave a machine-readable CSV marker so downstream consumers can
    # distinguish "not run" from a missing deliverable without touching the
    # Task020 source artifact.
    atomic_csv(
        destination / "attention_raw_field_pair_decomposition.csv",
        ("status", "reason", "path_exists"),
        [{"status": "NOT_RUN", "reason": raw_status, "path_exists": raw_path.is_file()}],
    )
    atomic_json(destination / "raw_field_pair_status.json", {"status": "NOT_RUN", "reason": raw_status, "path_exists": raw_path.is_file()})
    summary_signals = (
        ("D_abs", lambda row: _float(row, "D_abs")),
        ("D_rel", lambda row: _float(row, "D_rel")),
        ("D_dyn", lambda row: _float(row, "D_dyn")),
        ("functional_energy", lambda row: _float(row, "functional_energy")),
        ("best_substitute_similarity", lambda row: _float(row, "best_substitute_similarity_at_28")),
        ("delta_total", lambda row: _float(row, "delta_total_at_28")),
        ("energy_times_delta", lambda row: _float(row, "functional_energy") * _float(row, "delta_total_at_28")),
        ("Dabs_times_delta", lambda row: _float(row, "D_abs") * _float(row, "delta_total_at_28")),
        ("Drel_times_delta", lambda row: _float(row, "D_rel") * _float(row, "delta_total_at_28")),
    )
    summary = []
    for name, getter in summary_signals:
        values = [getter(row) for row in rows]
        risk = importance_percentile(values, higher_is_important=(name != "best_substitute_similarity"))
        index_to_p = {int(row["global_index"]): float(risk[pos]) for pos, row in enumerate(rows)}
        retrieval_signal = {"best_substitute_similarity": "best_substitute_similarity_at_28",
                            "delta_total": "delta_total_at_28"}.get(name, name)
        rec = {
            f"recall_taskrisk_top{label}": next(
                float(item["recall"])
                for item in retrieval
                if item["signal"] == retrieval_signal and item["task_risk_fraction"] == label
            ) if retrieval_signal in SIGNALS else top_risk_retrieval(
                values, [_float(row, "mean_ce_increase") for row in rows],
                signal_higher_is_important=True, fraction=float(label) / 100
            )["recall"]
            for label in ("10", "20", "25")
        }
        replacement = [index_to_p[int(row["global_index"])] for row in rows if row["group"] == task020.GROUP_REPLACEMENT]
        ordinary = [index_to_p[int(row["global_index"])] for row in rows if row["group"] == task020.GROUP_ORDINARY]
        surviving = [index_to_p[int(row["global_index"])] for row in rows if row["group"] == task020.GROUP_CONTROL]
        summary.append({"signal": name,
                        "spearman_ce_all": spearman(values, [_float(row, "mean_ce_increase") for row in rows]),
                        "spearman_ce_ffn": spearman([values[pos] for pos, row in enumerate(rows) if row["unit_type"] == task020.TYPE_FFN], [_float(row, "mean_ce_increase") for row in rows if row["unit_type"] == task020.TYPE_FFN]),
                        "spearman_margin_all": spearman(values, [_float(row, "mean_margin_drop") for row in rows]),
                        "spearman_margin_ffn": spearman([values[pos] for pos, row in enumerate(rows) if row["unit_type"] == task020.TYPE_FFN], [_float(row, "mean_margin_drop") for row in rows if row["unit_type"] == task020.TYPE_FFN]),
                        **rec, "attention1549_percentile": index_to_p.get(1549, ""), "attention20908_percentile": index_to_p.get(20908, ""), "attention24002_percentile": index_to_p.get(24002, ""),
                        "replacement_ffn_median_percentile": float(np.median(replacement)) if replacement else "", "ordinary_ffn_median_percentile": float(np.median(ordinary)) if ordinary else "", "surviving_ffn_median_percentile": float(np.median(surviving)) if surviving else ""})
    retention_summary = _retention_summary(rows, retrieval)
    atomic_csv(destination / "information_retention_summary.csv", tuple(retention_summary[0]), retention_summary)
    atomic_csv(destination / "task021_summary.csv", tuple(summary[0]), summary)
    figure_status = _write_figures(destination, rows, trajectory, retrieval, bins, domains)
    # This identity file belongs to the Task021 output directory.  It records
    # the exact immutable Task020 inputs and Git source identity used for the
    # diagnosis; it is never written back into the Task020 output directory.
    input_hashes = {
        name: sha256_file(root / name)
        for name in REQUIRED_TASK020_FILES
        if (root / name).is_file()
    }
    atomic_json(destination / "artifact_identity.json", {
        "code_version": CODE_VERSION,
        "task020_identity_pass": True,
        "cohort_identity_pass": True,
        "task020_code_version": identity["code_version"],
        "task020_root": identity["task020_root"],
        "task020_identity_sha256": identity["task020_identity_sha256"],
        "task020_completion_sha256": identity["task020_completion_sha256"],
        "task020_input_sha256": input_hashes,
        "task020_cohort_sha256": identity["cohort_sha256"],
        "task020_unit_task_importance_sha256": identity["unit_task_importance_sha256"],
        "production_source_git_blob_sha": identity["production_source_git_blob_sha"],
        "descriptor_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS,
        "attention_global_indices": list(ATTENTION_IDENTITIES),
        "replacement_ffn_global_indices": identity["replacement_ffn_global_indices"],
        "task020_artifacts_modified": False,
        "production_pruning_code_modified": False,
        "gpu_pruning_or_validation_executed": False,
    })
    atomic_json(destination / "task021_completion.json", {
        "status": "PASS", "code_version": CODE_VERSION,
        "task020_identity_pass": True, "cohort_identity_pass": True,
        "information_ladder_complete": True, "stagewise_correlation_complete": True,
        "rank_loss_complete": True, "within_functional_risk_analysis_complete": True,
        "matched_risk_analysis_complete": True, "type_confound_control_complete": True,
        "diagnostic_proxy_analysis_complete": True, "analysis_complete": True,
        "production_pruning_code_modified": False, "gpu_pruning_or_validation_executed": False,
        "task020_artifacts_modified": False, "raw_field_pair_analysis_complete": False,
        "raw_field_pair_analysis_status": raw_status, "figure_status": figure_status,
        "percentile_population_scope": context["scope"], "percentile_population_size": context["size"] or len(rows),
        "task020_identity_sha256": identity["task020_identity_sha256"],
    })
    diagnosis = _diagnosis_markdown(rows, bins, loss, context)
    (destination / "diagnosis.md").write_text(diagnosis, encoding="utf-8")
    return {"identity": identity, "output_dir": str(destination.resolve()), "figure_status": figure_status,
            "raw_field_status": raw_status, "rows": len(rows)}


def _diagnosis_markdown(rows: Sequence[Mapping[str, object]], bins: Sequence[Mapping[str, object]],
                       loss: Sequence[Mapping[str, object]], context: Mapping[str, object]) -> str:
    by_index = {int(row["global_index"]): row for row in rows}
    def value(index: int, key: str) -> str:
        return f"{_float(by_index[index], key):.8g}" if index in by_index else "unavailable"
    energy_corr = spearman([_float(row, "functional_energy") for row in rows], [_float(row, "mean_ce_increase") for row in rows])
    dabs_corr = spearman([_float(row, "D_abs") for row in rows], [_float(row, "mean_ce_increase") for row in rows])
    drel_corr = spearman([_float(row, "D_rel") for row in rows], [_float(row, "mean_ce_increase") for row in rows])
    delta_corr = spearman([_float(row, "delta_total_at_28") for row in rows], [_float(row, "mean_ce_increase") for row in rows])
    conditional = [float(row["spearman_energy_task_ce"]) for row in bins if math.isfinite(float(row["spearman_energy_task_ce"]))]
    conditional_text = f"{float(np.mean(conditional)):.8g}" if conditional else "unavailable"
    return f"""# Task021: Importance Information Retention Diagnosis

This is an offline, descriptive diagnosis over the exact Task020 investigated units. It does not rerun a model, validation, pruning, descriptor extraction, BMS, or Contribution Field computation. Global percentiles use `{context['scope']}` context with size `{context['size'] or len(rows)}`; task-risk labels remain restricted to Task020 rows.

## Answers

1. **Where do Attention 1549 and 20908 change?** Their early and final values are reported in `attention_information_trajectory.csv`; the change is identified by comparing the D_abs/D_rel/D_dyn and energy percentiles with Delta_total and task-CE percentiles, without imposing a conclusion. Attention 1549 energy is `{value(1549, 'functional_energy')}`, and Attention 20908 energy is `{value(20908, 'functional_energy')}`.
2. **Energy conditional on similar Delta_total:** the mean within-bin Spearman value is `{conditional_text}`. A positive value supports retained magnitude information; a weak or missing value does not.
3. **D_abs versus Delta_total:** Spearman(D_abs, task CE) is `{dabs_corr:.8g}`, while Spearman(Delta_total, task CE) is `{delta_corr:.8g}`.
4. **D_rel versus Delta_total:** Spearman(D_rel, task CE) is `{drel_corr:.8g}`, while Spearman(Delta_total, task CE) is `{delta_corr:.8g}`.
5. **D_dyn:** its cohort-level correlation is in `stagewise_task_correlation.csv`; the three-head Attention population is marked descriptive-only.
6. **L2 normalization:** it is implicated only when energy retains task association within similar Delta_total bins; this is recorded in `information_loss_evidence.csv`, not asserted a priori.
7. **Unequal substitute magnitude:** raw signed field pairs were not loaded or recomputed. `attention_best_substitute_decomposition.csv` reports saved energy norms and leaves signed cosine blank.
8. **Cross-type versus within-FFN:** `type_confound_control.csv` reports Attention/FFN medians and FFN-only correlations; Attention-vs-FFN separation alone is not treated as evidence.
9. **Best pre-functional signal:** compare `task_risk_retrieval.csv` and `stagewise_task_correlation.csv`; no signal is selected as a pruning score.
10. **Parameter-free proxies:** products appear only in `diagnostic_proxy_comparison.csv` as offline probes; they are not used for pruning.
11. **New task dimension:** Task021 does not justify one by construction. The evidence classification is in `information_loss_evidence.csv`.
12. **Next mechanism:** this task does not implement a next mechanism; any follow-up must be chosen from the evidence and separately approved.

## Scope and safeguards

- Task020 artifacts were read-only inputs.
- Production pruning sources, BMS, descriptors, Coverage, Delta_total, validation, and cache identities were not changed.
- No GPU pruning or validation was executed.
- Raw Contribution Field decomposition status: `NOT_RUN`.
"""


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task020-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_analysis(args.task020_dir, args.output_dir, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

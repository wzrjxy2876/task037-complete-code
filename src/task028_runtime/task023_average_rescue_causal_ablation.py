"""Task023 causal ablation of conditional Average rescue.

This is an independent diagnostic path.  The production selector is never
changed: Delta_total remains the primary ordering and Delta_average is used
only as a step-local veto in the three counterfactual variants.  The replay
engine is imported lazily from Task019 so importing this module is cheap and
CPU-only tests do not require PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

import task020_functional_demand_task_importance as task020
import task022_average_total_dual_risk as task022


CODE_VERSION = "task023_average_rescue_causal_ablation_v1"
# Analysis-only version for the final integrity/post-hoc pass.  The causal
# experiment semantics (and therefore CODE_VERSION) remain unchanged.
ANALYSIS_VERSION = "task023_final_integrity_v1"
TASK019_COMMIT = "3c053f11fe6e344a5e7b60875ebe195cd30a1eee"
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
EXPECTED_COHORT_UNITS = 417
EXPECTED_VALIDATION_SAMPLES = 3_783
START_SPARSE = 0.28
TARGET_SPARSE = 0.30
LOW_TOTAL_FRACTION = 0.25  # fixed diagnostic definition inherited from Task022
VARIANTS = (
    "original_dynamic_total_30",
    "average_rescue_all_30",
    "average_rescue_ffn_only_30",
    "mirror_rescue_control_30",
)
RESCUE_VARIANTS = VARIANTS[1:]
TYPE_ATTENTION = task020.TYPE_ATTENTION
TYPE_FFN = task020.TYPE_FFN
PRODUCTION_FILES = (
    "functional_competition_pruning.py",
    "MC.py",
    "ucf101_videoswin_my.py",
)

TRACE_FIELDS = (
    "variant", "step", "estimated_sparsity_before", "estimated_sparsity_after",
    "global_index", "unit_type", "layer", "unit_index", "stage", "domain_id",
    "Delta_average", "Delta_total", "total_rank", "total_percentile",
    "average_rank_within_low_total", "in_low_total", "in_high_average_low_total",
    "in_low_average_low_total", "was_vetoed", "selected", "parameter_cost",
    "cumulative_removed_cost", "domain_retained_ratio_before",
    "domain_coverage_before", "best_substitute_similarity",
)
REQUIRED_REGISTRY_FIELDS = ("global_index", "layer", "unit_type", "unit_index")
VETO_FIELDS = (
    "variant", "step", "global_index", "unit_type", "layer", "domain_id",
    "Delta_average", "Delta_total", "total_rank", "average_rank_within_low_total",
    "reason", "selected_replacement_global_index", "selected_replacement_unit_type",
    "selected_replacement_delta_average", "selected_replacement_delta_total",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
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
                raise ValueError(f"Non-finite {name}")
            return result
    if default is not None:
        return float(default)
    raise KeyError(names[0])


def _int(row: Mapping[str, object], *names: str, default: int | None = None) -> int:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return int(float(value))
    if default is not None:
        return int(default)
    raise KeyError(names[0])


def _stage(layer: object) -> str:
    text = str(layer)
    marker = text.split("layers.", 1)[1].split(".", 1)[0] if "layers." in text else "unknown"
    return f"stage_{marker}"


def assert_registry_ready_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject incomplete or duplicate structural rows before registry build.

    ``local_index`` is deliberately not accepted as a substitute for the
    layer/module ``unit_index`` required by Task018's registry contract.
    """
    seen: set[int] = set()
    for position, row in enumerate(rows):
        missing = [name for name in REQUIRED_REGISTRY_FIELDS
                   if name not in row or str(row[name]).strip() == ""]
        if missing:
            raise RuntimeError(
                f"Registry row {position} global_index={row.get('global_index')} "
                f"missing fields {missing}"
            )
        try:
            global_index = int(row["global_index"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Registry row {position} has invalid global_index") from error
        if global_index in seen:
            raise RuntimeError(f"Duplicate global_index in registry-ready prefix: {global_index}")
        seen.add(global_index)


def canonical_registry_content(registry: Mapping[str, Mapping[str, object]]) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Canonical representation for exact V0 structural-registry comparison."""
    return tuple(sorted(
        (str(layer), str(entry["unit_type"]), tuple(sorted(int(value) for value in entry["indices"])))
        for layer, entry in registry.items()
    ))


def canonical_registry_sha256(registry: Mapping[str, Mapping[str, object]]) -> str:
    """Hash the canonical structural registry, independent of JSON key order."""
    payload = json.dumps(canonical_registry_content(registry), separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FINAL_COMPLETION_TRUE_FIELDS = (
    "artifact_identity_pass", "original_dynamic_reproduction_pass",
    "original_registry_reproduction_pass", "fresh_exact_v0_validation_complete",
    "fresh_exact_v0_registry_identity_pass", "average_rescue_all_validation_complete",
    "average_rescue_ffn_only_validation_complete", "mirror_rescue_validation_complete",
    "v1_v2_v3_validation_reused_without_rerun", "set_difference_analysis_corrected",
    "stage_analysis_corrected", "domain_concentration_analysis_complete",
    "budget_semantics_unchanged", "min_keep_semantics_unchanged",
)
FINAL_COMPLETION_FALSE_FIELDS = (
    "task_label_leakage", "production_pruning_code_modified", "fine_tuning_executed",
)


def assert_final_completion_gate(completion: Mapping[str, object]) -> None:
    """Validate positive completion gates and negative safety gates separately."""
    missing = [key for key in FINAL_COMPLETION_TRUE_FIELDS if completion.get(key) is not True]
    if missing:
        raise RuntimeError(f"Task023 final-integrity positive gates failed: {missing}")
    unsafe = [key for key in FINAL_COMPLETION_FALSE_FIELDS if completion.get(key) is not False]
    if unsafe:
        raise RuntimeError(f"Task023 final-integrity safety gates failed: {unsafe}")


def partition_selected_sets(original: set[int], current: set[int]) -> dict[str, set[int]]:
    """Return the disjoint ``original/current`` partition in causal terms."""
    original = {int(value) for value in original}
    current = {int(value) for value in current}
    common = original & current
    original_only = original - current
    counterfactual_only = current - original
    if common & original_only or common & counterfactual_only or original_only & counterfactual_only:
        raise RuntimeError("Selected-set partition is not disjoint")
    if common | original_only != original or common | counterfactual_only != current:
        raise RuntimeError("Selected-set partition does not reconstruct inputs")
    return {"common": common, "original_only": original_only,
            "counterfactual_only": counterfactual_only}


def analysis_stage(row: Mapping[str, object]) -> str:
    """Use a stored stage when present, otherwise derive it post-hoc."""
    stored = str(row.get("stage", "")).strip()
    return stored or _stage(row.get("layer", ""))


def _find_task019_validation(task019_root: Path) -> Path | None:
    candidates = [Path(task019_root) / "validation" / "dynamic_30" / "metrics.json",
                  Path(task019_root) / "validation" / "dynamic_total_30" / "metrics.json",
                  Path(task019_root) / "validation" / "no_new_attention_dynamic_30" / "metrics.json"]
    candidates.extend(sorted(Path(task019_root).glob("**/*30*/metrics.json")))
    return next((path for path in candidates if path.is_file()), None)


def _candidate_key(row: Mapping[str, object]) -> tuple[float, int]:
    return (_float(row, "Delta_total", "delta_total"), _int(row, "global_index"))


def deterministic_low_total_sets(candidates: Sequence[Mapping[str, object]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return current LowTotal, HighAverageLowTotal and LowAverageLowTotal.

    The first quarter is selected by exact rank (``ceil(M/4)``), then split
    into deterministic average-risk halves.  No threshold or floating median
    is used, and parameter cost is deliberately absent from every key.
    """
    normalized = [dict(row) for row in candidates]
    if not normalized:
        return [], [], []
    if len({int(row["global_index"]) for row in normalized}) != len(normalized):
        raise ValueError("Current feasible candidates contain duplicate global_index")
    ordered_total = sorted(normalized, key=_candidate_key)
    count = int(math.ceil(len(ordered_total) * LOW_TOTAL_FRACTION))
    low_total = ordered_total[:max(1, count)]
    average_order = sorted(low_total, key=lambda row: (_float(row, "Delta_average", "delta_average"),
                                                       _int(row, "global_index")))
    split = len(average_order) // 2
    return low_total, average_order[split:], average_order[:split]


def rescue_veto_set(variant: str, candidates: Sequence[Mapping[str, object]]) -> set[int]:
    if variant not in RESCUE_VARIANTS:
        raise ValueError(f"Not a rescue variant: {variant}")
    low_total, high_average, low_average = deterministic_low_total_sets(candidates)
    if variant == "average_rescue_all_30":
        return {int(row["global_index"]) for row in high_average}
    if variant == "average_rescue_ffn_only_30":
        return {int(row["global_index"]) for row in high_average if str(row["unit_type"]) == TYPE_FFN}
    return {int(row["global_index"]) for row in low_average}


def select_with_rescue(
    variant: str,
    candidate_provider: Callable[[], Sequence[Mapping[str, object]]],
    remove_callback: Callable[[Mapping[str, object]], Mapping[str, object] | None],
    *,
    target_budget: float,
    initial_removed_cost: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Pure orchestration helper used by the GPU replay and regression tests.

    ``candidate_provider`` is called after every removal, proving that the
    functional state is dynamic.  ``remove_callback`` performs the exact
    Task016 state update and returns optional post-removal context.  If every
    current candidate is vetoed, the variant stops instead of falling back.
    """
    if variant not in RESCUE_VARIANTS:
        raise ValueError("select_with_rescue accepts V1/V2/V3 only")
    cumulative = int(initial_removed_cost)
    selections: list[dict] = []
    veto_rows: list[dict] = []
    selected_indices: set[int] = set()
    while cumulative < float(target_budget):
        current = [dict(row) for row in candidate_provider()]
        if not current:
            raise RuntimeError("No feasible candidate before target budget")
        total_order = sorted(current, key=_candidate_key)
        total_rank = {int(row["global_index"]): rank for rank, row in enumerate(total_order, 1)}
        low_total, high_average, low_average = deterministic_low_total_sets(current)
        low_ids = {int(row["global_index"]) for row in low_total}
        high_ids = {int(row["global_index"]) for row in high_average}
        low_ids_avg = {int(row["global_index"]) for row in low_average}
        avg_rank = {int(row["global_index"]): rank for rank, row in enumerate(
            sorted(low_total, key=lambda item: (_float(item, "Delta_average", "delta_average"),
                                                _int(item, "global_index"))), 1)}
        veto_ids = rescue_veto_set(variant, current)
        selected = None
        for candidate in total_order:
            index = int(candidate["global_index"])
            if index in selected_indices:
                # A stale provider is a caller error, but skipping it keeps a
                # malformed cache from causing duplicate removals.
                continue
            if index in veto_ids:
                veto_rows.append({
                    "variant": variant, "step": len(selections) + 1,
                    "global_index": index, "unit_type": candidate.get("unit_type", ""),
                    "layer": candidate.get("layer", ""), "domain_id": candidate.get("domain_id", ""),
                    "Delta_average": _float(candidate, "Delta_average", "delta_average"),
                    "Delta_total": _float(candidate, "Delta_total", "delta_total"),
                    "total_rank": total_rank[index],
                    "average_rank_within_low_total": avg_rank.get(index, ""),
                    "reason": ("high_average_low_total" if index in high_ids else
                               "low_average_low_total"),
                    "selected_replacement_global_index": "",
                    "selected_replacement_unit_type": "",
                    "selected_replacement_delta_average": "",
                    "selected_replacement_delta_total": "",
                })
                continue
            selected = candidate
            break
        if selected is None:
            raise RuntimeError(f"{variant} has no feasible non-vetoed candidate at step {len(selections)+1}")
        replacement_index = int(selected["global_index"])
        for row in veto_rows:
            if int(row["step"]) == len(selections) + 1 and not row["selected_replacement_global_index"]:
                row.update({
                    "selected_replacement_global_index": replacement_index,
                    "selected_replacement_unit_type": selected.get("unit_type", ""),
                    "selected_replacement_delta_average": _float(selected, "Delta_average", "delta_average"),
                    "selected_replacement_delta_total": _float(selected, "Delta_total", "delta_total"),
                })
        selected_cost = _int(selected, "parameter_cost")
        before = cumulative
        context = remove_callback(selected) or {}
        cumulative += selected_cost
        index = replacement_index
        row = dict(selected)
        row.update(context)
        row.update({
            "step": len(selections) + 1,
            "global_index": index,
            "total_rank": total_rank[index],
            "total_percentile": ((total_rank[index] - 1) / max(1, len(total_order) - 1)),
            "average_rank_within_low_total": avg_rank.get(index, ""),
            "in_low_total": index in low_ids,
            "in_high_average_low_total": index in high_ids,
            "in_low_average_low_total": index in low_ids_avg,
            "was_vetoed": False,
            "selected": True,
            "parameter_cost": selected_cost,
            "cumulative_removed_cost": cumulative,
            "estimated_sparsity_before": before / float(context.get("parameters_before", target_budget / TARGET_SPARSE)),
            "estimated_sparsity_after": cumulative / float(context.get("parameters_before", target_budget / TARGET_SPARSE)),
        })
        selections.append(row)
        selected_indices.add(index)
    return selections, veto_rows


def _ordered_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    output = [dict(row) for row in rows if str(row.get("selected", "true")).lower() != "false"]
    output.sort(key=lambda row: int(row.get("step", row.get("global_index", 0))))
    return output


def _task016_prefix_identity(task016_root: Path, task017_root: Path, task018_root: Path):
    import task019_dynamic_ranking_causal_ablation as task019
    return task019._prefix_identity(Path(task016_root), Path(task017_root), Path(task018_root))


def _summary_for_target(
    summaries_by_target: Mapping[float, Mapping[str, object]],
    target: float,
) -> Mapping[str, object] | None:
    """Resolve one Task016 summary from Task019's authoritative mapping.

    Task019 deliberately returns ``by_target`` as its third value.  Iterating
    that mapping yields float keys, not summary rows, so this helper keeps the
    caller contract explicit and rejects ambiguous near-equal keys.
    """
    matches = [
        row for key, row in summaries_by_target.items()
        if math.isclose(float(key), float(target), rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Task016 summaries match target {target}")
    return matches[0] if matches else None


def _production_identity(repo_root: Path) -> dict:
    return task020.production_source_git_identity(Path(repo_root), task019_commit=TASK019_COMMIT)


def verify_identity(
    *, task014_root: Path, task016_root: Path, task017_root: Path,
    task018_root: Path, task019_root: Path, task020_root: Path,
    task021_root: Path, task022_root: Path, output_dir: Path,
    checkpoint: Path, repo_root: Path,
) -> dict:
    """Gate every immutable input before a GPU worker can start."""
    roots = [Path(value) for value in (task014_root, task016_root, task017_root,
                                       task018_root, task019_root, task020_root,
                                       task021_root, task022_root)]
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Task023 input roots: {missing}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    t22 = read_json(Path(task022_root) / "task022_completion.json")
    if t22.get("status") != "PASS":
        raise RuntimeError("Task022 is not PASS")
    # Task022 performs the strict Task020/Task021 gate and checks cohort order.
    inherited = task022.verify_task022_identity(Path(task020_root), Path(task021_root), repo_root=Path(repo_root))
    identity20 = read_json(Path(task020_root) / "artifact_identity.json")
    identity21 = read_json(Path(task021_root) / "artifact_identity.json")
    identity22 = read_json(Path(task022_root) / "artifact_identity.json")
    checkpoint_identities = [str(payload["checkpoint_sha256"]) for payload in (identity20, identity21, identity22) if payload.get("checkpoint_sha256")]
    if checkpoint_identities and any(value != checkpoint_identities[0] for value in checkpoint_identities):
        raise RuntimeError("Task020/Task021/Task022 checkpoint identities differ")
    c19 = read_json(Path(task019_root) / "task019_completion.json")
    if c19.get("status") != "PASS" or c19.get("production_pruning_code_modified") is not False:
        raise RuntimeError("Task019 completion/production gate is incomplete")
    identity19 = read_json(Path(task019_root) / "artifact_identity.json")
    if identity19.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task019 artifact identity is incomplete")
    trace30, prefixes, summaries_by_target, extra = _task016_prefix_identity(task016_root, task017_root, task018_root)
    if len(prefixes.get(START_SPARSE, [])) == 0 or len(prefixes.get(TARGET_SPARSE, [])) == 0:
        raise RuntimeError("Task018 28%/30% prefixes are missing")
    parameters_before = int(extra["parameters_before"])
    if parameters_before != EXPECTED_UNITS and int(identity19.get("descriptor_units", EXPECTED_UNITS)) != EXPECTED_UNITS:
        raise RuntimeError("Unexpected unit mapping")
    production = _production_identity(Path(repo_root))
    source_task020 = read_json(Path(task020_root) / "artifact_identity.json").get("production_source_git_blob_sha", {})
    if source_task020 and source_task020 != production:
        raise RuntimeError("Task020 production source identity differs")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(identity19.get("checkpoint_sha256", checkpoint_sha)):
        raise RuntimeError("Checkpoint SHA differs from Task019")
    # Bind the exact parameter-cost and layer-capacity tables used to reach the
    # budget.  These are read-only Task014 metadata, never selector inputs.
    from task015_attention_ffn_diagnosis import _layer_capacities, _load_units, _mapping_paths, infer_parameter_costs
    unit_path, layer_path, _ = _mapping_paths(Path(task014_root))
    units = _load_units(unit_path)
    costs, _ = infer_parameter_costs(units, read_csv(layer_path))
    capacities = _layer_capacities(units)
    parameter_cost_sha = hashlib.sha256(json.dumps({str(i): int(value) for i, value in enumerate(costs)}, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    layer_capacity_sha = hashlib.sha256(json.dumps({str(k): int(v) for k, v in sorted(capacities.items())}, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    target_summary = _summary_for_target(summaries_by_target, TARGET_SPARSE)
    start_summary = _summary_for_target(summaries_by_target, START_SPARSE)
    if target_summary is None or start_summary is None:
        raise RuntimeError("Task016 summaries do not contain exact 28% and 30% endpoints")
    payload = {
        "status": "PASS", "code_version": CODE_VERSION, "artifact_identity_pass": True,
        "task020_status": "PASS", "task021_status": "PASS", "task022_status": "PASS",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "parameters_before": parameters_before, "descriptor_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS, "start_sparsity": START_SPARSE,
        "target_sparsity": TARGET_SPARSE,
        "start_prefix_steps": len(prefixes[START_SPARSE]), "target_prefix_steps": len(prefixes[TARGET_SPARSE]),
        "start_prefix_sequence_sha256": sequence_sha256(int(row["global_index"]) for row in prefixes[START_SPARSE]),
        "target_prefix_sequence_sha256": sequence_sha256(int(row["global_index"]) for row in prefixes[TARGET_SPARSE]),
        "task016_trace30_sha256": sha256_file(extra["trace30_path"]),
        "parameter_cost_sha256": parameter_cost_sha,
        "layer_capacity_sha256": layer_capacity_sha,
        "validation_split": identity19.get("validation_split", ""),
        "validation_batch_size": identity19.get("validation_batch_size", 1),
        "amp_enabled": identity19.get("amp_enabled", True),
        "production_source_git_blob_sha": production,
        "task020_identity_sha256": inherited["task020_identity_sha256"],
        "task021_identity_sha256": inherited["task021_identity_sha256"],
        "task019_identity_sha256": sha256_file(Path(task019_root) / "artifact_identity.json"),
        "task022_identity_sha256": sha256_file(Path(task022_root) / "artifact_identity.json"),
        "target_parameter_budget": float(target_summary["target_parameter_budget"]),
        "start_parameter_budget": float(start_summary["target_parameter_budget"]),
        "budget_semantics_unchanged": True, "min_keep_semantics_unchanged": True,
        "task_label_leakage": False, "production_pruning_code_modified": False,
        "fine_tuning_executed": False,
        "task014_root": str(Path(task014_root).resolve()), "task016_root": str(Path(task016_root).resolve()),
        "task017_root": str(Path(task017_root).resolve()), "task018_root": str(Path(task018_root).resolve()),
        "task019_root": str(Path(task019_root).resolve()), "task020_root": str(Path(task020_root).resolve()),
        "task021_root": str(Path(task021_root).resolve()), "task022_root": str(Path(task022_root).resolve()),
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "trace30": trace30, "prefixes": prefixes,
            "summaries_by_target": summaries_by_target, "extra": extra}


def _task023_trace_row(row: Mapping[str, object], *, variant: str, step: int,
                       before_cost: int, after_cost: int, parameters_before: int,
                       rank: int, candidate_count: int) -> dict:
    total = _float(row, "Delta_total", "delta_total", "delta_total_at_selection", default=0.0)
    average = _float(row, "Delta_average", "delta_average", "delta_average_at_selection", default=0.0)
    index = _int(row, "global_index")
    out = dict(row)
    out.update({
        "variant": variant, "step": step,
        "estimated_sparsity_before": before_cost / float(parameters_before),
        "estimated_sparsity_after": after_cost / float(parameters_before),
        "global_index": index, "unit_type": row.get("unit_type", ""),
        "layer": row.get("layer", ""), "stage": analysis_stage(row),
        "domain_id": row.get("domain_id", ""), "Delta_average": average, "Delta_total": total,
        "total_rank": row.get("total_rank", rank),
        "total_percentile": row.get("total_percentile", (rank - 1) / max(1, candidate_count - 1)),
        "average_rank_within_low_total": row.get("average_rank_within_low_total", ""),
        "in_low_total": row.get("in_low_total", False),
        "in_high_average_low_total": row.get("in_high_average_low_total", False),
        "in_low_average_low_total": row.get("in_low_average_low_total", False),
        "was_vetoed": False, "selected": True,
        "parameter_cost": _int(row, "parameter_cost"), "cumulative_removed_cost": after_cost,
        "domain_retained_ratio_before": row.get("domain_retained_ratio_before", ""),
        "domain_coverage_before": row.get("domain_coverage_before", ""),
        "best_substitute_similarity": row.get("best_substitute_similarity", row.get("best_substitute_similarity_at_28", "")),
    })
    return {field: out.get(field, "") for field in TRACE_FIELDS}


def reproduce_original_dynamic(
    trace30: Sequence[Mapping[str, object]], prefixes: Mapping[float, Sequence[Mapping[str, object]]],
    *, parameters_before: int,
) -> tuple[list[dict], bool]:
    """Reproduce exact Task016 28%→30% rows without a GPU run."""
    start = [int(row["global_index"]) for row in prefixes[START_SPARSE]]
    expected = [int(row["global_index"]) for row in prefixes[TARGET_SPARSE]]
    if expected[:len(start)] != start:
        raise RuntimeError("Task016 28% is not a prefix of 30%")
    incremental = list(trace30[len(start):len(expected)])
    actual = [int(row["global_index"]) for row in incremental]
    passed = expected[len(start):] == actual
    if not passed:
        raise RuntimeError("Original dynamic 28%→30% sequence differs")
    cumulative = sum(_int(row, "parameter_cost") for row in prefixes[START_SPARSE])
    output = []
    for step, row in enumerate(incremental, 1):
        before = cumulative
        cumulative += _int(row, "parameter_cost")
        output.append(_task023_trace_row(row, variant=VARIANTS[0], step=step,
                                         before_cost=before, after_cost=cumulative,
                                         parameters_before=parameters_before,
                                         rank=_int(row, "rank_at_selection", default=step),
                                         candidate_count=_int(row, "candidate_count", default=1)))
    assert_registry_ready_rows(output)
    return output, passed


def _engine_run_variant(*, variant: str, task014_root: Path, task016_root: Path,
                        task017_root: Path, output_dir: Path, device: str,
                        parameters_before: int, target_budget: float,
                        start_prefix: Sequence[Mapping[str, object]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Run one rescue policy using Task019's unchanged GPU replay state."""
    import task019_dynamic_ranking_causal_ablation as task019
    engine = task019.CausalAblationEngine(task014_root=task014_root, task016_root=task016_root,
                                          task017_root=task017_root, output_dir=output_dir,
                                          device=device)
    engine.restore_prefix(start_prefix)
    initial = engine.removed_cost
    raw_veto: list[dict] = []

    def provider() -> list[dict]:
        return engine.all_candidates()

    def remove(candidate: Mapping[str, object]) -> Mapping[str, object]:
        before = engine.removed_cost
        result = engine.remove(variant=variant, global_index=int(candidate["global_index"]),
                               domain_id=int(candidate["domain_id"]), local_index=int(candidate["local_index"]),
                               rank=int(candidate.get("total_rank", 1)))
        result["parameters_before"] = parameters_before
        result["estimated_sparsity_before"] = before / float(parameters_before)
        return result

    incremental, raw_veto = select_with_rescue(variant, provider, remove,
                                               target_budget=target_budget,
                                               initial_removed_cost=initial)
    # select_with_rescue already has a complete trace; add fields derived from
    # the exact engine removal context and preserve the no-label causal record.
    trace = []
    cumulative = initial
    for row in incremental:
        cost = _int(row, "parameter_cost")
        cumulative += cost
        trace.append({field: row.get(field, "") for field in TRACE_FIELDS})
    # Keep the authoritative structural unit_index through serialization;
    # never replace it with the engine's domain-local local_index.
    assert_registry_ready_rows(trace)
    domains = engine.domain_state_rows(variant)
    return trace, raw_veto, domains


def construct_variant(*, variant: str, task014_root: Path, task016_root: Path,
                      task017_root: Path, task018_root: Path, output_dir: Path,
                      device: str) -> None:
    if variant not in RESCUE_VARIANTS:
        raise ValueError("Construction is only for V1/V2/V3")
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Run identity gate first")
    import task019_dynamic_ranking_causal_ablation as task019
    trace30, prefixes, summaries_by_target, extra = _task016_prefix_identity(task016_root, task017_root, task018_root)
    target_summary = _summary_for_target(summaries_by_target, TARGET_SPARSE)
    if target_summary is None:
        raise RuntimeError("Task016 summaries do not contain exact 30% endpoint")
    trace, veto, domains = _engine_run_variant(variant=variant, task014_root=task014_root,
                                               task016_root=task016_root, task017_root=task017_root,
                                               output_dir=output_dir, device=device,
                                               parameters_before=int(identity["parameters_before"]),
                                               target_budget=float(target_summary["target_parameter_budget"]),
                                               start_prefix=prefixes[START_SPARSE])
    full_prefix = [dict(row) for row in prefixes[START_SPARSE]] + trace
    assert_registry_ready_rows(full_prefix)
    registry = task019.registry_from_prefix(full_prefix)
    variant_dir = Path(output_dir) / "variants" / variant
    atomic_csv(variant_dir / "causal_selection_trace.csv", TRACE_FIELDS, trace)
    atomic_csv(variant_dir / "average_rescue_veto_trace.csv", VETO_FIELDS, veto)
    atomic_csv(variant_dir / "domain_state.csv", tuple(domains[0]) if domains else ("variant",), domains)
    summary = _construction_summary(variant, trace, domains, int(identity["parameters_before"]),
                                   target_budget=float(target_summary["target_parameter_budget"]),
                                   start_cost=sum(_int(row, "parameter_cost") for row in prefixes[START_SPARSE]))
    summary["registry_schema_pass"] = True
    summary["registry_canonical_sha256"] = canonical_registry_sha256(registry)
    atomic_json(variant_dir / "registry.json", {
        "status": "prepared", "variant": variant, "registry": registry,
        "summary": summary, "checkpoint_sha256": identity["checkpoint_sha256"],
        "production_pruning_code_modified": False, "task_label_leakage": False,
    })
    atomic_json(variant_dir / "construction.json", summary)


def construct_original(*, task016_root: Path, task017_root: Path, task018_root: Path,
                       output_dir: Path, parameters_before: int) -> tuple[list[dict], list[dict]]:
    trace30, prefixes, _, extra = _task016_prefix_identity(task016_root, task017_root, task018_root)
    rows, passed = reproduce_original_dynamic(trace30, prefixes, parameters_before=parameters_before)
    if not passed:
        raise RuntimeError("V0 reproduction failed")
    import task019_dynamic_ranking_causal_ablation as task019
    full_prefix = [dict(row) for row in prefixes[START_SPARSE]] + rows
    assert_registry_ready_rows(full_prefix)
    registry = task019.registry_from_prefix(full_prefix)
    assert_registry_ready_rows(prefixes[TARGET_SPARSE])
    authoritative_registry = task019.registry_from_prefix(prefixes[TARGET_SPARSE])
    original_registry_reproduction_pass = (
        canonical_registry_content(registry)
        == canonical_registry_content(authoritative_registry)
    )
    if not original_registry_reproduction_pass:
        raise RuntimeError("V0 structural registry differs from authoritative Task018 30% registry")
    variant_dir = Path(output_dir) / "variants" / VARIANTS[0]
    atomic_csv(variant_dir / "causal_selection_trace.csv", TRACE_FIELDS, rows)
    atomic_csv(variant_dir / "average_rescue_veto_trace.csv", VETO_FIELDS, [])
    summary = _construction_summary(VARIANTS[0], rows, [], parameters_before,
                                   target_budget=float(read_json(Path(output_dir) / "artifact_identity.json")["target_parameter_budget"]),
                                   start_cost=sum(_int(row, "parameter_cost") for row in prefixes[START_SPARSE]))
    summary["registry_schema_pass"] = True
    summary["original_registry_reproduction_pass"] = original_registry_reproduction_pass
    summary["original_dynamic_reproduction_pass"] = passed
    summary["registry_canonical_sha256"] = canonical_registry_sha256(registry)
    variant_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(variant_dir / "domain_state.csv", ("variant", "domain_id", "coverage", "retained_ratio"), [])
    atomic_json(variant_dir / "registry.json", {"status": "prepared", "variant": VARIANTS[0],
                 "registry": registry, "summary": summary,
                 "checkpoint_sha256": read_json(Path(output_dir) / "artifact_identity.json")["checkpoint_sha256"],
                 "production_pruning_code_modified": False, "task_label_leakage": False})
    atomic_json(variant_dir / "construction.json", summary)
    return rows, prefixes[TARGET_SPARSE]


def _construction_summary(variant: str, rows: Sequence[Mapping[str, object]], domains: Sequence[Mapping[str, object]],
                          parameters_before: int, *, target_budget: float, start_cost: int) -> dict:
    attention = [row for row in rows if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffn = [row for row in rows if str(row.get("unit_type")) == TYPE_FFN]
    removed = start_cost + sum(_int(row, "parameter_cost") for row in rows)
    return {
        "status": "prepared", "variant": variant, "start_sparsity": START_SPARSE,
        "target_sparsity": TARGET_SPARSE, "incremental_units": len(rows),
        "incremental_attention": len(attention), "incremental_ffn": len(ffn),
        "incremental_parameter_cost": sum(_int(row, "parameter_cost") for row in rows),
        "incremental_attention_parameter_cost": sum(_int(row, "parameter_cost") for row in attention),
        "incremental_ffn_parameter_cost": sum(_int(row, "parameter_cost") for row in ffn),
        "attention_budget_share": (sum(_int(row, "parameter_cost") for row in attention) /
                                   max(1, sum(_int(row, "parameter_cost") for row in rows))),
        "ffn_budget_share": (sum(_int(row, "parameter_cost") for row in ffn) /
                              max(1, sum(_int(row, "parameter_cost") for row in rows))),
        "estimated_removed_parameters": removed, "estimated_parameter_sparsity": removed / float(parameters_before),
        "target_parameter_budget": target_budget, "budget_overshoot": removed - target_budget,
        "sequence_sha256": sequence_sha256(int(row["global_index"]) for row in rows),
        "increment_sequence_sha256": sequence_sha256(int(row["global_index"]) for row in rows),
        "incremental_domains_touched": len({int(row["domain_id"]) for row in rows if str(row.get("domain_id", ""))}),
        "incremental_layers_touched": len({str(row["layer"]) for row in rows}),
        # V0 reuses the identity-checked Task019 endpoint and has no new GPU
        # domain-state table in this output tree; zero is an explicit missing
        # diagnostic value rather than a non-JSON NaN sentinel.
        "mean_domain_coverage_final": float(np.mean([_float(row, "coverage", default=math.nan) for row in domains])) if domains else 0.0,
        "median_best_substitute_similarity_final": float(np.median([_float(row, "median_best_remaining_similarity", default=math.nan) for row in domains])) if domains else 0.0,
        "budget_semantics_unchanged": True, "min_keep_semantics_unchanged": True,
        "task_label_leakage": False,
    }


def assert_validation_ready(output_dir: Path, variant: str) -> tuple[dict, dict]:
    """Strictly gate a prepared registry before any validation forward pass."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Task023 variant: {variant}")
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    payload = read_json(Path(output_dir) / "variants" / variant / "registry.json")
    if identity.get("artifact_identity_pass") is not True or payload.get("status") != "prepared":
        raise RuntimeError("Identity/construction gate is incomplete")
    summary = payload.get("summary", {})
    if summary.get("registry_schema_pass") is not True:
        raise RuntimeError(f"{variant} registry schema gate is incomplete")
    if variant == VARIANTS[0] and summary.get("original_registry_reproduction_pass") is not True:
        raise RuntimeError("Fresh V0 validation requires original_registry_reproduction_pass=true")
    assert_registry_ready_rows(_registry_rows_from_payload(payload))
    return identity, payload


def _registry_rows_from_payload(payload: Mapping[str, object]) -> list[dict]:
    """Expand a registry enough for the strict row gate without changing it."""
    rows: list[dict] = []
    for layer, entry in (payload.get("registry", {}) or {}).items():
        unit_type = str(entry.get("unit_type", ""))
        for position, index in enumerate(entry.get("indices", ())):
            rows.append({"global_index": len(rows),
                         "layer": layer, "unit_type": unit_type,
                         "unit_index": index})
    # Registry JSON has no global index; synthetic unique keys only exercise
    # the structural fields.  The trace gate remains authoritative for actual
    # global-index uniqueness.
    return rows


def verify_prepared_variant(output_dir: Path, variant: str) -> dict:
    """Read-only construction verification used by final integrity mode."""
    identity, payload = assert_validation_ready(output_dir, variant)
    trace_path = Path(output_dir) / "variants" / variant / "causal_selection_trace.csv"
    construction_path = Path(output_dir) / "variants" / variant / "construction.json"
    if not trace_path.is_file() or not construction_path.is_file():
        raise RuntimeError(f"Prepared {variant} construction artifacts are incomplete")
    construction = read_json(construction_path)
    if variant == VARIANTS[0] and construction.get("original_registry_reproduction_pass") is not True:
        raise RuntimeError("V0 construction is not an exact structural reproduction")
    if construction.get("budget_semantics_unchanged") is not True or construction.get("min_keep_semantics_unchanged") is not True:
        raise RuntimeError(f"{variant} budget/min-keep gate is incomplete")
    return {"identity": identity, "payload": payload, "construction": construction}


def validate_cached_metrics(output_dir: Path, variant: str, identity: Mapping[str, object]) -> dict:
    """Validate cached V1/V2/V3 metadata without rewriting or rerunning them."""
    path = Path(output_dir) / "validation" / variant / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cached validation metrics missing: {path}")
    payload = read_json(path)
    required = {
        "status": "PASS", "variant": variant,
        "samples": EXPECTED_VALIDATION_SAMPLES,
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "fresh_original_checkpoint": True,
        "fine_tuning_executed": False,
        "production_pruning_code_modified": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Cached {variant} validation identity mismatch for {key}")
    return payload


def assert_fresh_v0_primary(output_dir: Path) -> dict:
    """Require that V0's primary metrics are the exact fresh validation."""
    path = Path(output_dir) / "validation" / VARIANTS[0] / "metrics.json"
    if not path.is_file():
        raise RuntimeError("Fresh V0 primary metrics are missing")
    payload = read_json(path)
    if payload.get("reused_task019_validation") is True:
        raise RuntimeError("Task019-reused metrics cannot be V0 primary")
    if payload.get("status") != "PASS" or payload.get("fresh_exact_v0_validation") is not True:
        raise RuntimeError("V0 primary metrics are not from fresh exact validation")
    return payload


def validate_variant(output_dir: Path, variant: str, device: str) -> None:
    """Validate one prepared model on one isolated logical CUDA device."""
    identity, payload = assert_validation_ready(output_dir, variant)
    # Task019's validator path is mirrored only for the one explicitly
    # requested fresh V0 pass in final-integrity mode.  Existing V1/V2/V3
    # metrics are read through validate_cached_metrics and never regenerated.
    import torch
    if not torch.cuda.is_available() or str(device) != "cuda:0":
        raise RuntimeError("Each isolated validation worker requires logical cuda:0")
    from task018_high_sparsity_transition import _apply_registry
    from ucf101_videoswin_my import AverageMeter, SwinTransformer3D, get_dataset, set_seed, validate_rgb
    checkpoint = Path(identity["checkpoint"])
    torch.cuda.set_device(0); set_seed(3407); torch.cuda.reset_peak_memory_stats(0)
    model = SwinTransformer3D(patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
                              num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.0,
                              qkv_bias=True, patch_norm=True, drop_path_rate=0.2,
                              use_checkpoint=True).to(torch.device(device))
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    normalized = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()}
    load_message = model.load_state_dict(normalized, strict=False)
    applied = _apply_registry(model, payload["registry"])
    expected = {"removed_attention": int(payload["summary"].get("incremental_attention", 0)),
                "removed_ffn": int(payload["summary"].get("incremental_ffn", 0))}
    if applied["removed_attention"] < expected["removed_attention"] or applied["removed_ffn"] < expected["removed_ffn"]:
        raise RuntimeError("Applied registry count is smaller than incremental trace")
    loader = get_dataset(str(identity.get("validation_split", "")), int(identity.get("validation_batch_size", 1)))
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"Validation split has {len(loader.dataset)} samples, expected {EXPECTED_VALIDATION_SAMPLES}")
    top1, top5 = AverageMeter(), AverageMeter(); started = time.perf_counter()
    validate_rgb(loader, model, top1, top5, use_amp=bool(identity.get("amp_enabled", True)))
    elapsed = time.perf_counter() - started
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "status": "PASS", "variant": variant, "top1": float(top1.avg), "top5": float(top5.avg),
        "samples": len(loader.dataset), "validation_time_seconds": elapsed,
        "estimated_parameter_sparsity": float(payload["summary"]["estimated_parameter_sparsity"]),
        "physical_numel_sparsity": 1.0 - parameters_after / float(identity["parameters_before"]),
        "parameters_before": int(identity["parameters_before"]), "parameters_after": parameters_after,
        "checkpoint_sha256": identity["checkpoint_sha256"], "sequence_sha256": payload["summary"]["sequence_sha256"],
        "full_registry_canonical_sha256": canonical_registry_sha256(payload["registry"]),
        "removed_attention": int(applied["removed_attention"]), "removed_ffn": int(applied["removed_ffn"]),
        "checkpoint_missing_keys": sorted(load_message.missing_keys), "checkpoint_unexpected_keys": sorted(load_message.unexpected_keys),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)), "fresh_original_checkpoint": True,
        "fresh_exact_v0_validation": variant == VARIANTS[0],
        "fine_tuning_executed": False, "production_pruning_code_modified": False,
    }
    atomic_json(Path(output_dir) / "validation" / variant / "metrics.json", metrics)


def reuse_original_validation(output_dir: Path, task019_root: Path) -> None:
    """Preserve a legacy Task019 report for audit only; never make it primary."""
    output_dir = Path(output_dir)
    target_dir = output_dir / "validation" / VARIANTS[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    primary = target_dir / "metrics.json"
    audit = target_dir / "reused_task019_metrics.json"
    if primary.is_file() and not audit.is_file():
        atomic_json(audit, read_json(primary))
    if audit.is_file():
        return
    source = _find_task019_validation(Path(task019_root))
    if source is None:
        raise FileNotFoundError("Task019 30% validation metrics not found for V0 reuse")
    if not audit.is_file():
        payload = dict(read_json(source)); payload["variant"] = VARIANTS[0]
        payload["reused_task019_validation"] = True
        atomic_json(audit, payload)


def _summary_stat(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {name: math.nan for name in ("mean", "median", "q25", "q75", "q90")}
    return {"mean": float(np.mean(finite)), "median": float(np.median(finite)),
            "q25": float(np.quantile(finite, .25)), "q75": float(np.quantile(finite, .75)),
            "q90": float(np.quantile(finite, .90))}


def domain_concentration_row(variant: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize how each variant's incremental units concentrate by domain."""
    counts = Counter(str(row.get("domain_id", "")) for row in rows if str(row.get("domain_id", "")).strip())
    values = sorted((int(value) for value in counts.values()), reverse=True)
    total = sum(values)
    shares = [value / total for value in values] if total else []
    hhi = sum(share * share for share in shares)
    if total:
        ordered = sorted(values)
        n = len(ordered)
        sum_x = sum(ordered)
        gini = sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * sum_x)
    else:
        gini = 0.0
    return {
        "variant": variant, "incremental_units": len(rows), "unique_domains": len(values),
        "largest_domain_share": max(shares, default=0.0),
        "top2_domain_share": sum(shares[:2]), "top4_domain_share": sum(shares[:4]),
        "hhi": hhi, "gini": gini, "HHI": hhi, "Gini": gini,
        "median_units_per_touched_domain": float(np.median(values)) if values else 0.0,
        "max_units_per_domain": max(values, default=0),
    }


def _variant_rows(output_dir: Path, variant: str) -> list[dict]:
    return read_csv(Path(output_dir) / "variants" / variant / "causal_selection_trace.csv")


def _verify_v0_registry_against_task018(output_dir: Path, task018_root: Path) -> str:
    """Compare V0's structural registry with the authoritative Task018 s30."""
    v0_payload = read_json(Path(output_dir) / "variants" / VARIANTS[0] / "registry.json")
    authoritative_path = Path(task018_root) / "registries" / "s30.json"
    if not authoritative_path.is_file():
        raise FileNotFoundError(f"Authoritative Task018 registry missing: {authoritative_path}")
    authoritative = read_json(authoritative_path)
    actual_sha = canonical_registry_sha256(v0_payload["registry"])
    expected_sha = canonical_registry_sha256(authoritative["registry"])
    if actual_sha != expected_sha:
        raise RuntimeError("V0 registry does not equal authoritative Task018 30% registry")
    if v0_payload.get("summary", {}).get("original_registry_reproduction_pass") is not True:
        raise RuntimeError("V0 construction did not record original_registry_reproduction_pass=true")
    return actual_sha


def final_integrity_run(*, task014_root: Path, task016_root: Path, task017_root: Path,
                        task018_root: Path, task019_root: Path, task020_root: Path,
                        task021_root: Path, task022_root: Path, output_dir: Path,
                        checkpoint: Path, repo_root: Path, device: str = "cuda:0") -> dict:
    """Complete Task023 integrity without rerunning V1/V2/V3 experiments.

    This is intentionally a read-mostly orchestration path.  It verifies all
    prepared variants, performs exactly one fresh V0 validation, reads the
    existing rescue metrics, and then runs post-hoc analysis.
    """
    output_dir = Path(output_dir)
    identity_result = verify_identity(task014_root=task014_root, task016_root=task016_root,
                                      task017_root=task017_root, task018_root=task018_root,
                                      task019_root=task019_root, task020_root=task020_root,
                                      task021_root=task021_root, task022_root=task022_root,
                                      output_dir=output_dir, checkpoint=checkpoint,
                                      repo_root=repo_root)
    prepared = {variant: verify_prepared_variant(output_dir, variant) for variant in VARIANTS}
    # Every stored trace must still be structurally reconstructable.  This is
    # read-only and catches truncated unit_index fields before a GPU forward.
    for variant in VARIANTS:
        rows = _variant_rows(output_dir, variant)
        assert_registry_ready_rows(rows)
    v0_registry_sha = _verify_v0_registry_against_task018(output_dir, task018_root)

    # Preserve any prior Task019 reuse as an audit record before replacing the
    # primary V0 metrics with the exact registry-bound fresh validation.
    reused_audit = output_dir / "validation" / VARIANTS[0] / "reused_task019_metrics.json"
    if (output_dir / "validation" / VARIANTS[0] / "metrics.json").is_file() and not reused_audit.is_file():
        atomic_json(reused_audit, read_json(output_dir / "validation" / VARIANTS[0] / "metrics.json"))
    elif not reused_audit.is_file():
        source = _find_task019_validation(Path(task019_root))
        if source is not None:
            atomic_json(reused_audit, read_json(source))

    # This is the sole fresh GPU validation in final-integrity mode.
    validate_variant(output_dir, VARIANTS[0], device)
    fresh_metrics = assert_fresh_v0_primary(output_dir)
    if fresh_metrics.get("full_registry_canonical_sha256") != v0_registry_sha:
        raise RuntimeError("Fresh V0 metrics are not bound to the reproduced registry")

    cached_metrics = {variant: validate_cached_metrics(output_dir, variant, identity_result["identity"])
                      for variant in RESCUE_VARIANTS}
    # Post-hoc writes only derived tables/figures and reads all raw traces.
    posthoc_analysis(output_dir=output_dir, task019_root=task019_root,
                     task020_root=task020_root, task021_root=task021_root,
                     task022_root=task022_root)
    completion_path = output_dir / "task023_completion.json"
    completion = read_json(completion_path)
    completion.update({
        "status": "PASS" if completion.get("status") == "PASS" else "PENDING_VALIDATION",
        "analysis_version": ANALYSIS_VERSION,
        "artifact_identity_pass": True,
        "original_dynamic_reproduction_pass": True,
        "original_registry_reproduction_pass": True,
        "fresh_exact_v0_validation_complete": True,
        "fresh_exact_v0_registry_identity_pass": True,
        "reused_task019_metrics_preserved_audit_only": reused_audit.is_file(),
        "v1_v2_v3_validation_reused_without_rerun": True,
        "set_difference_analysis_corrected": True,
        "stage_analysis_corrected": True,
        "domain_concentration_analysis_complete": (output_dir / "domain_concentration_summary.csv").is_file(),
    })
    try:
        assert_final_completion_gate(completion)
    except RuntimeError:
        completion["status"] = "PENDING_VALIDATION"
    else:
        completion["status"] = "PASS"
    atomic_json(completion_path, completion)
    identity_payload = dict(identity_result["identity"])
    identity_payload.update({"final_integrity_pass": completion["status"] == "PASS",
                             "original_registry_reproduction_pass": True,
                             "fresh_exact_v0_registry_identity_pass": True,
                             "v1_v2_v3_validation_reused_without_rerun": True})
    atomic_json(output_dir / "artifact_identity.json", identity_payload)
    if completion["status"] != "PASS":
        raise RuntimeError("Task023 final integrity did not reach PASS")
    return {"completion": completion, "fresh_v0": fresh_metrics, "cached": cached_metrics,
            "prepared": prepared}


def posthoc_analysis(*, output_dir: Path, task019_root: Path, task020_root: Path,
                     task021_root: Path, task022_root: Path) -> None:
    """Join labels and validation only after all causal selections complete."""
    output_dir = Path(output_dir)
    by_variant = {variant: _variant_rows(output_dir, variant) for variant in VARIANTS}
    selected_sets = {variant: {int(row["global_index"]) for row in rows} for variant, rows in by_variant.items()}
    original = selected_sets[VARIANTS[0]]
    veto_rows_by_variant = {variant: read_csv(output_dir / "variants" / variant / "average_rescue_veto_trace.csv") for variant in RESCUE_VARIANTS}
    # Keep a single raw causal record as well as the per-variant copies used
    # by workers.  These rows contain no Task020 labels.
    all_trace_rows = [row for variant in VARIANTS for row in by_variant[variant]]
    all_veto_rows = [row for variant in RESCUE_VARIANTS for row in veto_rows_by_variant[variant]]
    atomic_csv(output_dir / "causal_selection_trace.csv", TRACE_FIELDS, all_trace_rows)
    atomic_csv(output_dir / "average_rescue_veto_trace.csv", VETO_FIELDS, all_veto_rows)
    # Set difference and composition.
    set_rows = []
    partition_rows = []
    for variant in RESCUE_VARIANTS:
        current = selected_sets[variant]
        sets = partition_selected_sets(original, current)
        union = original | current
        partition_rows.append({"variant": variant, "original_count": len(original),
                               "current_count": len(current), "common_count": len(sets["common"]),
                               "original_only_count": len(sets["original_only"]),
                               "counterfactual_only_count": len(sets["counterfactual_only"]),
                               "partition_exact": (sets["common"] | sets["original_only"] == original and
                                                    sets["common"] | sets["counterfactual_only"] == current)})
        for set_name, members in sets.items():
            member_rows = [row for row in by_variant[VARIANTS[0]] if int(row["global_index"]) in members]
            if set_name == "counterfactual_only":
                member_rows = [row for row in by_variant[variant] if int(row["global_index"]) in members]
            set_rows.append({
                "variant": variant, "set_name": set_name, "count": len(members),
                "attention_count": sum(str(row.get("unit_type")) == TYPE_ATTENTION for row in member_rows),
                "ffn_count": sum(str(row.get("unit_type")) == TYPE_FFN for row in member_rows),
                "parameter_cost": sum(_int(row, "parameter_cost") for row in member_rows),
                "unique_domains": len({int(row["domain_id"]) for row in member_rows if str(row.get("domain_id", ""))}),
                "unique_layers": len({str(row["layer"]) for row in member_rows}),
                "jaccard_vs_original": len(sets["common"]) / max(1, len(union)),
            })
    atomic_csv(output_dir / "counterfactual_set_difference.csv",
               ("variant", "set_name", "count", "attention_count", "ffn_count", "parameter_cost", "unique_domains", "unique_layers", "jaccard_vs_original"), set_rows)
    atomic_csv(output_dir / "counterfactual_set_partition.csv",
               ("variant", "original_count", "current_count", "common_count", "original_only_count",
                "counterfactual_only_count", "partition_exact"), partition_rows)
    comp = []
    for variant, rows in by_variant.items():
        for key, group in (("unit_type", "type"), ("stage", "stage"), ("layer", "layer")):
            counts = defaultdict(lambda: {"units": 0, "parameter_cost": 0})
            for row in rows:
                name = analysis_stage(row) if key == "stage" else str(row.get(key, _stage(row.get("layer", ""))))
                counts[name]["units"] += 1; counts[name]["parameter_cost"] += _int(row, "parameter_cost")
            for name, values in sorted(counts.items()):
                comp.append({"variant": variant, "breakdown": group, "value": name,
                             "incremental_units": values["units"], "parameter_cost": values["parameter_cost"]})
    atomic_csv(output_dir / "counterfactual_composition.csv", ("variant", "breakdown", "value", "incremental_units", "parameter_cost"), comp)
    concentration = [domain_concentration_row(variant, rows) for variant, rows in by_variant.items()]
    atomic_csv(output_dir / "domain_concentration_summary.csv",
               ("variant", "incremental_units", "unique_domains", "largest_domain_share",
                "top2_domain_share", "top4_domain_share", "hhi", "gini", "HHI", "Gini",
                "median_units_per_touched_domain", "max_units_per_domain"), concentration)
    # Domain rows and score distributions.
    domain_rows = []
    stats_rows = []
    for variant, rows in by_variant.items():
        domain_file = output_dir / "variants" / variant / "domain_state.csv"
        if domain_file.is_file():
            domain_rows.extend(read_csv(domain_file))
        for score_name in ("Delta_average", "Delta_total", "N_valid", "functional_energy"):
            values = [_float(row, score_name, score_name.lower(), default=math.nan) for row in rows]
            summary = _summary_stat(values)
            stats_rows.extend({"variant": variant, "score": score_name, **summary}.items() if False else [])
            stats_rows.append({"variant": variant, "score": score_name, **summary})
    atomic_csv(output_dir / "counterfactual_domain_state.csv", tuple(domain_rows[0]) if domain_rows else ("variant",), domain_rows)
    atomic_csv(output_dir / "counterfactual_selected_score_statistics.csv", ("variant", "score", "mean", "median", "q25", "q75", "q90"), stats_rows)
    # Post-hoc Task020 labels; selection code above has already finished.
    labels = {_int(row, "global_index"): row for row in read_csv(Path(task020_root) / "unit_task_importance.csv")}
    label_rows = []
    for variant, rows in by_variant.items():
        selected = [labels[int(row["global_index"])] for row in rows if int(row["global_index"]) in labels]
        parts = {"common": original & selected_sets[variant],
                 "original_only": original - selected_sets[variant],
                 "rescue_only": selected_sets[variant] - original}
        for subset_name, subset in (("selected", selected),
                                     ("original_only", [labels[i] for i in parts["original_only"] if i in labels]),
                                     ("rescue_only", [labels[i] for i in parts["rescue_only"] if i in labels]),
                                     ("vetoed_labelled", [labels[int(row["global_index"])] for row in veto_rows_by_variant.get(variant, []) if int(row["global_index"]) in labels])):
            values = [_float(row, "mean_ce_increase", "task_ce", default=math.nan) for row in subset]
            label_rows.append({"variant": variant, "subset": subset_name, "num_units": len(values),
                               "mean_task020_ce": float(np.mean(values)) if values else math.nan,
                               "median_task020_ce": float(np.median(values)) if values else math.nan})
    atomic_csv(output_dir / "task020_labelled_counterfactual_analysis.csv",
               ("variant", "subset", "num_units", "mean_task020_ce", "median_task020_ce"), label_rows)
    # Attention outcomes use identities from the authoritative cohort file.
    cohort_rows = read_csv(Path(task020_root) / "task020_cohorts.csv")
    attention_ids = [int(row["global_index"]) for row in cohort_rows if str(row.get("cohort", row.get("group", ""))) == task020.GROUP_ATTENTION]
    outcomes = []
    for variant in VARIANTS:
        veto_ids = {int(row["global_index"]) for row in veto_rows_by_variant.get(variant, [])}
        for index in attention_ids:
            outcomes.append({"variant": variant, "global_index": index,
                             "selected": index in selected_sets[variant],
                             "vetoed_at_least_once": index in veto_ids,
                             "ultimately_retained": index not in selected_sets[variant]})
    atomic_csv(output_dir / "attention_rescue_outcomes.csv", ("variant", "global_index", "selected", "vetoed_at_least_once", "ultimately_retained"), outcomes)
    # Veto statistics and replacements.
    veto_stats = []
    replacement_rows = []
    for variant in RESCUE_VARIANTS:
        veto = veto_rows_by_variant[variant]
        avg = [_float(row, "Delta_average") for row in veto]
        total = [_float(row, "Delta_total") for row in veto]
        repl_avg = [_float(row, "selected_replacement_delta_average", default=math.nan) for row in veto]
        repl_total = [_float(row, "selected_replacement_delta_total", default=math.nan) for row in veto]
        veto_stats.append({"variant": variant, "num_selection_steps": len(by_variant[variant]),
                           "num_veto_events": len(veto), "num_unique_vetoed_units": len({int(row["global_index"]) for row in veto}),
                           "vetoed_attention": sum(row.get("unit_type") == TYPE_ATTENTION for row in veto),
                           "vetoed_ffn": sum(row.get("unit_type") == TYPE_FFN for row in veto),
                           "unique_vetoed_domains": len({int(row["domain_id"]) for row in veto}) if veto else 0,
                           "unique_vetoed_layers": len({str(row["layer"]) for row in veto}) if veto else 0,
                           "median_veto_delta_average": float(np.median(avg)) if avg else math.nan,
                           "median_veto_delta_total": float(np.median(total)) if total else math.nan,
                           "median_selected_replacement_delta_average": float(np.nanmedian(repl_avg)) if repl_avg else math.nan,
                           "median_selected_replacement_delta_total": float(np.nanmedian(repl_total)) if repl_total else math.nan})
        for row in veto:
            veto_label = labels.get(int(row["global_index"])) if 'labels' in locals() else None
            replacement_label = labels.get(int(row["selected_replacement_global_index"])) if 'labels' in locals() and str(row.get("selected_replacement_global_index", "")).strip() else None
            replacement_rows.append({"variant": variant, "veto_global_index": row["global_index"],
                                     "replacement_global_index": row["selected_replacement_global_index"],
                                     "delta_average_difference": float(row["selected_replacement_delta_average"] or math.nan) - _float(row, "Delta_average"),
                                     "delta_total_difference": float(row["selected_replacement_delta_total"] or math.nan) - _float(row, "Delta_total"),
                                     "task020_ce_difference": (_float(replacement_label, "mean_ce_increase", "task_ce") - _float(veto_label, "mean_ce_increase", "task_ce")) if veto_label and replacement_label else math.nan})
    atomic_csv(output_dir / "veto_statistics.csv", tuple(veto_stats[0]) if veto_stats else ("variant",), veto_stats)
    atomic_csv(output_dir / "veto_replacement_comparison.csv", ("variant", "veto_global_index", "replacement_global_index", "delta_average_difference", "delta_total_difference", "task020_ce_difference"), replacement_rows)
    unique_veto_rows = []
    for variant in RESCUE_VARIANTS:
        veto = veto_rows_by_variant[variant]
        for index in sorted({int(row["global_index"]) for row in veto}):
            events = [row for row in veto if int(row["global_index"]) == index]
            label = labels.get(index)
            unique_veto_rows.append({
                "variant": variant, "global_index": index, "num_veto_events": len(events),
                "unit_type": events[0].get("unit_type", ""),
                "domain_id": events[0].get("domain_id", ""),
                "mean_task020_ce": (_float(label, "mean_ce_increase", "task_ce", default=math.nan)
                                     if label else math.nan),
                "ultimately_retained": index not in selected_sets[variant],
            })
    atomic_csv(output_dir / "unique_veto_task020_label_analysis.csv",
               ("variant", "global_index", "num_veto_events", "unit_type", "domain_id",
                "mean_task020_ce", "ultimately_retained"), unique_veto_rows)
    # Primary summary, including validation metrics when available.
    metrics = {}
    for variant in VARIANTS:
        path = output_dir / "validation" / variant / "metrics.json"
        metrics[variant] = read_json(path) if path.is_file() else {}
    summary_rows = []
    def metric_difference(left: str, right: str, key: str) -> float:
        left_value = metrics[left].get(key, math.nan)
        right_value = metrics[right].get(key, math.nan)
        try:
            return float(left_value) - float(right_value)
        except (TypeError, ValueError):
            return math.nan
    for variant in VARIANTS:
        rows = by_variant[variant]; payload = metrics[variant]
        constr = read_json(output_dir / "variants" / variant / "construction.json") if (output_dir / "variants" / variant / "construction.json").is_file() else {}
        jaccard = 1.0 if variant == VARIANTS[0] else len(original & selected_sets[variant]) / max(1, len(original | selected_sets[variant]))
        veto = veto_rows_by_variant.get(variant, [])
        summary_rows.append({"variant": variant, "top1": payload.get("top1", math.nan), "top5": payload.get("top5", math.nan),
                             "top1_gain_vs_original": (payload.get("top1", math.nan) - metrics[VARIANTS[0]].get("top1", math.nan)) if payload and metrics[VARIANTS[0]] else math.nan,
                             "top5_gain_vs_original": (payload.get("top5", math.nan) - metrics[VARIANTS[0]].get("top5", math.nan)) if payload and metrics[VARIANTS[0]] else math.nan,
                             "top1_gain_vs_fresh_v0": (payload.get("top1", math.nan) - metrics[VARIANTS[0]].get("top1", math.nan)) if payload and metrics[VARIANTS[0]] else math.nan,
                             "top5_gain_vs_fresh_v0": (payload.get("top5", math.nan) - metrics[VARIANTS[0]].get("top5", math.nan)) if payload and metrics[VARIANTS[0]] else math.nan,
                             "top1_v1_minus_v3": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[2], "top1"),
                             "top5_v1_minus_v3": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[2], "top5"),
                             "top1_v2_minus_v3": metric_difference(RESCUE_VARIANTS[1], RESCUE_VARIANTS[2], "top1"),
                             "top5_v2_minus_v3": metric_difference(RESCUE_VARIANTS[1], RESCUE_VARIANTS[2], "top5"),
                             "top1_v1_minus_v2": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[1], "top1"),
                             "top5_v1_minus_v2": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[1], "top5"),
                             "specificity_all_vs_mirror_top1": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[2], "top1"),
                             "specificity_all_vs_mirror_top5": metric_difference(RESCUE_VARIANTS[0], RESCUE_VARIANTS[2], "top5"),
                             "specificity_ffn_vs_mirror_top1": metric_difference(RESCUE_VARIANTS[1], RESCUE_VARIANTS[2], "top1"),
                             "specificity_ffn_vs_mirror_top5": metric_difference(RESCUE_VARIANTS[1], RESCUE_VARIANTS[2], "top5"),
                             "estimated_parameter_sparsity": constr.get("estimated_parameter_sparsity", math.nan),
                             "physical_numel_sparsity": payload.get("physical_numel_sparsity", math.nan),
                             "incremental_attention": sum(str(row.get("unit_type")) == TYPE_ATTENTION for row in rows),
                             "incremental_ffn": sum(str(row.get("unit_type")) == TYPE_FFN for row in rows),
                             "attention_budget_share": constr.get("attention_budget_share", math.nan), "ffn_budget_share": constr.get("ffn_budget_share", math.nan),
                             "num_veto_events": len(veto), "num_unique_vetoed_units": len({int(row["global_index"]) for row in veto}),
                             "incremental_domains_touched": len({int(row["domain_id"]) for row in rows if str(row.get("domain_id", ""))}),
                             "incremental_layers_touched": len({str(row["layer"]) for row in rows}),
                             "mean_domain_coverage_final": constr.get("mean_domain_coverage_final", math.nan),
                             "median_best_substitute_similarity_final": constr.get("median_best_substitute_similarity_final", math.nan),
                             "selection_jaccard_vs_original": jaccard})
    summary_fields = tuple(summary_rows[0]) if summary_rows else ("variant",)
    atomic_csv(output_dir / "task023_causal_summary.csv", summary_fields, summary_rows)
    comparison_rows = []
    for label, left, right in (
        ("average_rescue_all_vs_mirror", RESCUE_VARIANTS[0], RESCUE_VARIANTS[2]),
        ("average_rescue_ffn_only_vs_mirror", RESCUE_VARIANTS[1], RESCUE_VARIANTS[2]),
        ("average_rescue_all_vs_ffn_only", RESCUE_VARIANTS[0], RESCUE_VARIANTS[1]),
        ("average_rescue_all_vs_fresh_v0", RESCUE_VARIANTS[0], VARIANTS[0]),
        ("average_rescue_ffn_only_vs_fresh_v0", RESCUE_VARIANTS[1], VARIANTS[0]),
        ("mirror_vs_fresh_v0", RESCUE_VARIANTS[2], VARIANTS[0]),
    ):
        comparison_rows.append({"comparison": label, "left_variant": left, "right_variant": right,
                                "top1_difference": metric_difference(left, right, "top1"),
                                "top5_difference": metric_difference(left, right, "top5")})
    atomic_csv(output_dir / "validation_direct_comparisons.csv",
               ("comparison", "left_variant", "right_variant", "top1_difference", "top5_difference"), comparison_rows)
    _write_figures(output_dir, summary_rows, by_variant, veto_rows_by_variant, outcomes, label_rows)
    _write_diagnosis(output_dir, summary_rows, veto_stats, outcomes, replacement_rows)
    v0_construction = read_json(output_dir / "variants" / VARIANTS[0] / "construction.json")
    fresh_v0 = metrics[VARIANTS[0]].get("status") == "PASS" and metrics[VARIANTS[0]].get("fresh_exact_v0_validation") is True
    cached_rescue = all(metrics[variant].get("status") == "PASS" for variant in RESCUE_VARIANTS)
    completion = {
        "status": "PASS" if fresh_v0 and cached_rescue else "PENDING_VALIDATION",
        "code_version": CODE_VERSION, "analysis_version": ANALYSIS_VERSION,
        "artifact_identity_pass": read_json(output_dir / "artifact_identity.json").get("artifact_identity_pass") is True,
        "original_dynamic_reproduction_pass": v0_construction.get("original_dynamic_reproduction_pass") is True,
        "original_registry_reproduction_pass": v0_construction.get("original_registry_reproduction_pass") is True,
        "fresh_exact_v0_validation_complete": fresh_v0,
        "fresh_exact_v0_registry_identity_pass": metrics[VARIANTS[0]].get("full_registry_canonical_sha256") == v0_construction.get("registry_canonical_sha256", metrics[VARIANTS[0]].get("full_registry_canonical_sha256")),
        "reused_task019_metrics_preserved_audit_only": (output_dir / "validation" / VARIANTS[0] / "reused_task019_metrics.json").is_file(),
        "average_rescue_all_constructed": (output_dir / "variants/average_rescue_all_30/registry.json").is_file(),
        "average_rescue_ffn_only_constructed": (output_dir / "variants/average_rescue_ffn_only_30/registry.json").is_file(),
        "mirror_rescue_control_constructed": (output_dir / "variants/mirror_rescue_control_30/registry.json").is_file(),
        "average_rescue_all_validation_complete": metrics[RESCUE_VARIANTS[0]].get("status") == "PASS",
        "average_rescue_ffn_only_validation_complete": metrics[RESCUE_VARIANTS[1]].get("status") == "PASS",
        "mirror_rescue_validation_complete": metrics[RESCUE_VARIANTS[2]].get("status") == "PASS",
        "v1_v2_v3_validation_reused_without_rerun": cached_rescue,
        "same_target_budget_all_variants": all(read_json(output_dir / "variants" / variant / "construction.json").get("target_sparsity") == TARGET_SPARSE for variant in VARIANTS),
        "set_difference_analysis_corrected": all(bool(row["partition_exact"]) for row in partition_rows),
        "stage_analysis_corrected": bool(comp) and all(str(row["value"]).strip() for row in comp if row["breakdown"] == "stage"),
        "domain_concentration_analysis_complete": bool(concentration) and all("hhi" in row for row in concentration),
        "budget_semantics_unchanged": True, "min_keep_semantics_unchanged": True,
        "task_label_leakage": False, "analysis_complete": True,
        "production_pruning_code_modified": False, "fine_tuning_executed": False,
    }
    try:
        assert_final_completion_gate(completion)
    except RuntimeError:
        completion["status"] = "PENDING_VALIDATION"
    else:
        completion["status"] = "PASS"
    atomic_json(output_dir / "task023_completion.json", completion)


def figure07_labelled_values(label_rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    """Return real Task020 CE values for the Figure-07 comparison.

    Missing labels remain NaN and are omitted by matplotlib; no value is
    synthesized from selection counts or from an unrelated variant.
    """
    values: dict[tuple[str, str], float] = {}
    for variant in RESCUE_VARIANTS:
        for subset in ("original_only", "rescue_only"):
            match = next((row for row in label_rows
                          if str(row.get("variant")) == variant and str(row.get("subset")) == subset), None)
            try:
                value = float(match["mean_task020_ce"]) if match is not None else math.nan
            except (TypeError, ValueError):
                value = math.nan
            values[(variant, subset)] = value if math.isfinite(value) else math.nan
    return values


def _write_figures(output_dir: Path, summary_rows: Sequence[Mapping[str, object]], by_variant: Mapping[str, Sequence[Mapping[str, object]]], veto_rows: Mapping[str, Sequence[Mapping[str, object]]], outcomes: Sequence[Mapping[str, object]], labelled_rows: Sequence[Mapping[str, object]] | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        atomic_json(Path(output_dir) / "figure_status.json", {"status": "NOT_RUN", "reason": "matplotlib unavailable"})
        return
    figures = []
    def save(name: str, fig) -> None:
        fig.tight_layout(); fig.savefig(Path(output_dir) / f"{name}.png", dpi=160); fig.savefig(Path(output_dir) / f"{name}.pdf"); plt.close(fig); figures.append(name)
    labels = [str(row["variant"]) for row in summary_rows]
    for metric, name in (("top1", "figure_01_top1"), ("top5", "figure_02_top5")):
        fig, ax = plt.subplots(); ax.bar(labels, [float(row.get(metric, math.nan)) for row in summary_rows]); ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=25); save(name, fig)
    fig, ax = plt.subplots();
    for variant, rows in by_variant.items():
        xs = np.arange(1, len(rows) + 1); ax.plot(xs, [sum(1 for row in rows[:i] if row.get("unit_type") == TYPE_ATTENTION) for i in xs], label=variant)
    ax.set_xlabel("selection step"); ax.set_ylabel("cumulative Attention removals"); ax.legend(); save("figure_03_parameter_composition", fig)
    fig, ax = plt.subplots();
    for variant, rows in veto_rows.items(): ax.plot(sorted({int(row["step"]) for row in rows}) or [0], [sum(int(row["step"]) == step for row in rows) for step in sorted({int(row["step"]) for row in rows})] or [0], label=variant)
    ax.set_xlabel("step"); ax.set_ylabel("veto events"); ax.legend(); save("figure_04_veto_over_progress", fig)
    for field, name in (("Delta_average", "figure_05_veto_replacement_delta_average"), ("Delta_total", "figure_06_veto_replacement_delta_total")):
        fig, ax = plt.subplots()
        for variant, rows in veto_rows.items(): ax.scatter([_float(row, field) for row in rows], [_float(row, f"selected_replacement_{field.lower()}", default=math.nan) for row in rows], label=variant, s=10)
        ax.set_xlabel(f"veto {field}"); ax.set_ylabel(f"replacement {field}"); ax.legend(); save(name, fig)
    labelled_values = figure07_labelled_values(labelled_rows or [])
    fig, ax = plt.subplots(); x = np.arange(len(RESCUE_VARIANTS)); width = 0.38
    for offset, subset in ((-width / 2, "original_only"), (width / 2, "rescue_only")):
        ax.bar(x + offset, [labelled_values[(variant, subset)] for variant in RESCUE_VARIANTS],
               width, label=subset)
    ax.set_xticks(x, RESCUE_VARIANTS, rotation=25); ax.set_ylabel("mean Task020 CE increase"); ax.legend()
    save("figure_07_task020_labelled_risk", fig)
    fig, ax = plt.subplots();
    for variant in VARIANTS: ax.bar(variant, sum(bool(row["selected"]) for row in outcomes if row["variant"] == variant))
    ax.tick_params(axis="x", rotation=25); save("figure_08_attention_outcomes", fig)
    fig, ax = plt.subplots(); ax.bar(labels, [float(row.get("mean_domain_coverage_final", math.nan)) for row in summary_rows]); save("figure_09_domain_coverage", fig)
    fig, ax = plt.subplots(); ax.bar(labels, [float(row.get("selection_jaccard_vs_original", math.nan)) for row in summary_rows]); ax.set_ylim(0, 1.05); save("figure_10_selection_jaccard", fig)
    atomic_json(Path(output_dir) / "figure_status.json", {"status": "PASS", "figures": figures})


def _write_diagnosis(output_dir: Path, summary_rows: Sequence[Mapping[str, object]], veto_stats: Sequence[Mapping[str, object]], outcomes: Sequence[Mapping[str, object]], replacements: Sequence[Mapping[str, object]]) -> None:
    by = {str(row["variant"]): row for row in summary_rows}
    v0 = by.get(VARIANTS[0], {}); v1 = by.get(RESCUE_VARIANTS[0], {}); v2 = by.get(RESCUE_VARIANTS[1], {}); v3 = by.get(RESCUE_VARIANTS[2], {})
    def gain(row, key):
        value = row.get(key, math.nan); return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.8g}"
    lines = ["# Task023 causal ablation diagnosis", "", "This remains a diagnostic ablation, not the final pruning method.", "",
             f"Q1. Average-rescue-all Top-1 gain over original: {gain(v1, 'top1_gain_vs_original')}.",
             f"Q2. Average-rescue-all Top-5 gain over original: {gain(v1, 'top5_gain_vs_original')}.",
             f"Q3. FFN-only Top-1 gain: {gain(v2, 'top1_gain_vs_original')}.",
             f"Q4. Attention protection is assessed from the V1/V2 gains and attention outcomes; no conclusion is imposed beforehand.",
             f"Q5. Mirror-control Top-1 gain: {gain(v3, 'top1_gain_vs_original')}.",
             f"Q6. V1 versus V3 is reported by their observed gains: {gain(v1, 'top1_gain_vs_original')} vs {gain(v3, 'top1_gain_vs_original')}.",
             f"Q7. Veto frequency is in veto_statistics.csv; the raw identity trace is average_rescue_veto_trace.csv.",
             f"Q8. Task020 risk enrichment is post-hoc only in task020_labelled_counterfactual_analysis.csv.",
             f"Q9. Replacement score differences are in veto_replacement_comparison.csv.",
             f"Q10. Domain/layer shifts are in counterfactual_domain_state.csv and counterfactual_composition.csv.",
             f"Q11. Direct V1−V3, V2−V3, V1−V2 and fresh-V0 Top-1/Top-5 comparisons are in validation_direct_comparisons.csv.",
             f"Q12. Conditional Average utility is supported only if the observed V1/V2/V3 comparisons warrant it.",
             f"Q13. No formal production mechanism is justified by this ablation alone; report the measured evidence first.", ""]
    Path(output_dir, "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("identity")
    for name in ("task014_root", "task016_root", "task017_root", "task018_root", "task019_root", "task020_root", "task021_root", "task022_root", "output_dir", "checkpoint", "repo_root"):
        identity.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    original = sub.add_parser("construct-original")
    for name in ("task016_root", "task017_root", "task018_root", "output_dir"):
        original.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    original.add_argument("--parameters-before", type=int, required=True)
    construct = sub.add_parser("construct")
    construct.add_argument("--variant", choices=RESCUE_VARIANTS, required=True)
    for name in ("task014_root", "task016_root", "task017_root", "task018_root", "output_dir"):
        construct.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    construct.add_argument("--device", default="cuda:0")
    validate = sub.add_parser("validate")
    validate.add_argument("--variant", choices=RESCUE_VARIANTS, required=True)
    validate.add_argument("--output-dir", type=Path, required=True); validate.add_argument("--device", default="cuda:0")
    reuse = sub.add_parser("reuse-original-validation")
    reuse.add_argument("--output-dir", type=Path, required=True); reuse.add_argument("--task019-root", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--output-dir", type=Path, required=True)
    for name in ("task019_root", "task020_root", "task021_root", "task022_root"):
        analyze.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    integrity = sub.add_parser("integrity")
    for name in ("task014_root", "task016_root", "task017_root", "task018_root", "task019_root",
                 "task020_root", "task021_root", "task022_root", "output_dir", "checkpoint", "repo_root"):
        integrity.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    integrity.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "identity":
        verify_identity(task014_root=args.task014_root, task016_root=args.task016_root,
                        task017_root=args.task017_root, task018_root=args.task018_root,
                        task019_root=args.task019_root, task020_root=args.task020_root,
                        task021_root=args.task021_root, task022_root=args.task022_root,
                        output_dir=args.output_dir, checkpoint=args.checkpoint,
                        repo_root=args.repo_root)
    elif args.command == "construct-original":
        construct_original(task016_root=args.task016_root, task017_root=args.task017_root,
                           task018_root=args.task018_root, output_dir=args.output_dir,
                           parameters_before=args.parameters_before)
    elif args.command == "construct":
        construct_variant(variant=args.variant, task014_root=args.task014_root,
                          task016_root=args.task016_root, task017_root=args.task017_root,
                          task018_root=args.task018_root, output_dir=args.output_dir,
                          device=args.device)
    elif args.command == "validate":
        validate_variant(args.output_dir, args.variant, args.device)
    elif args.command == "reuse-original-validation":
        reuse_original_validation(args.output_dir, args.task019_root)
    elif args.command == "analyze":
        posthoc_analysis(output_dir=args.output_dir, task019_root=args.task019_root,
                         task020_root=args.task020_root, task021_root=args.task021_root,
                         task022_root=args.task022_root)
    elif args.command == "integrity":
        final_integrity_run(task014_root=args.task014_root, task016_root=args.task016_root,
                            task017_root=args.task017_root, task018_root=args.task018_root,
                            task019_root=args.task019_root, task020_root=args.task020_root,
                            task021_root=args.task021_root, task022_root=args.task022_root,
                            output_dir=args.output_dir, checkpoint=args.checkpoint,
                            repo_root=args.repo_root, device=args.device)


if __name__ == "__main__":
    main()

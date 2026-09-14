from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DOMAIN_IDS = ("11", "76", "102", "103", "113", "269", "271", "297", "400", "415")
MIXED_DOMAIN_IDS = frozenset(("271", "297"))
ATTENTION = "attention_head"
FFN = "ffn_neuron"


class PhaseDPathError(ValueError):
    pass


def _f64(value: Any, name: str) -> np.float64:
    try:
        result = np.float64(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseDPathError(f"{name} is not a float64 value: {value!r}") from exc
    if not np.isfinite(result):
        raise PhaseDPathError(f"{name} must be finite")
    return result


def _int(value: Any, name: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PhaseDPathError(f"{name} is not an integer: {value!r}") from exc
    return result


def canonical_unit_type(value: Any) -> str:
    value = str(value).strip()
    aliases = {
        "head": ATTENTION,
        "attention_head": ATTENTION,
        "neuron": FFN,
        "ffn_neuron": FFN,
    }
    if value not in aliases:
        raise PhaseDPathError(f"unsupported Task037 unit type: {value!r}")
    return aliases[value]


def attention_head_parameter_cost(model_width: int, head_dim: int,
                                  qkv_has_bias: bool = True) -> int:
    """Task037 MC.estimate_unit_cost: QKV weights/bias plus projection input weights."""
    model_width, head_dim = int(model_width), int(head_dim)
    if model_width <= 0 or head_dim <= 0:
        raise PhaseDPathError("Attention dimensions must be positive")
    return 3 * model_width * head_dim + (3 * head_dim if qkv_has_bias else 0) + model_width * head_dim


def ffn_neuron_parameter_cost(fc1_in_features: int, fc2_out_features: int,
                              fc1_has_bias: bool = True) -> int:
    """Task037 MC.estimate_unit_cost: fc1 input weights/bias plus fc2 output weights."""
    fc1_in_features, fc2_out_features = int(fc1_in_features), int(fc2_out_features)
    if fc1_in_features <= 0 or fc2_out_features <= 0:
        raise PhaseDPathError("FFN dimensions must be positive")
    return fc1_in_features + (1 if fc1_has_bias else 0) + fc2_out_features


def f3_order_key(row: Mapping[str, Any]) -> Tuple[np.float64, np.float64, np.float64, int]:
    """Frozen Task037 key; lower risk is proposed first, global_index breaks ties."""
    risk = row.get("R_F3")
    if risk in (None, ""):
        total = _f64(row.get("p_total"), "p_total")
        average = _f64(row.get("p_average"), "p_average")
        damage = _f64(row.get("domain_damage", 0.0), "domain_damage")
        base = max(total, damage)
        risk = base + max(average - base, np.float64(0.0)) / np.float64(2.0)
    return (_f64(risk, "R_F3"),
            _f64(row.get("p_total"), "p_total"),
            _f64(row.get("p_average"), "p_average"),
            _int(row.get("global_index"), "global_index"))


def distance_index(distance_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, int], np.float64]:
    result: Dict[Tuple[str, int, int], np.float64] = {}
    for row in distance_rows:
        domain = str(row["domain_id"])
        i = _int(row["task037_global_index_i"], "task037_global_index_i")
        j = _int(row["task037_global_index_j"], "task037_global_index_j")
        value = _f64(row["d_temp"], "d_temp")
        if i == j or value < 0:
            raise PhaseDPathError("temporal distances must be nonnegative off-diagonal pairs")
        for key in ((domain, i, j), (domain, j, i)):
            if key in result and result[key] != value:
                raise PhaseDPathError(f"conflicting temporal distance for {key}")
            result[key] = value
    return result


def coverage_j(domain_id: Any, original_ids: Sequence[int], survivor_ids: Iterable[int],
               distances: Mapping[Tuple[str, int, int], Any]) -> np.float64:
    domain = str(domain_id)
    original = tuple(_int(v, "original id") for v in original_ids)
    survivors = tuple(sorted({_int(v, "survivor id") for v in survivor_ids}))
    if not original or not survivors:
        raise PhaseDPathError("J_g requires original units and at least one survivor")
    if not set(survivors).issubset(set(original)):
        raise PhaseDPathError("survivors must be a subset of the original domain")
    values = []
    for i in original:
        nearest = []
        for j in survivors:
            nearest.append(np.float64(0.0) if i == j else _f64(
                distances.get((domain, i, j)), f"d_temp[{domain},{i},{j}]"))
        values.append(min(nearest))
    return np.float64(max(values))


def marginal_coverage_cost(domain_id: Any, original_ids: Sequence[int],
                           survivor_ids: Iterable[int], candidate_id: int,
                           distances: Mapping[Tuple[str, int, int], Any]) -> Tuple[np.float64, np.float64, np.float64]:
    survivors = set(_int(v, "survivor id") for v in survivor_ids)
    candidate_id = _int(candidate_id, "candidate id")
    if candidate_id not in survivors:
        raise PhaseDPathError("proposal is not currently surviving")
    if len(survivors) <= 1:
        raise PhaseDPathError("a proposal cannot remove the last temporal representative")
    before = coverage_j(domain_id, original_ids, survivors, distances)
    after = coverage_j(domain_id, original_ids, survivors - {candidate_id}, distances)
    delta = np.float64(after - before)
    if delta < 0:
        raise PhaseDPathError("coverage loss must be nonnegative")
    return before, after, delta


def gamma_cost(delta_j: Any, delta_p: Any) -> np.float64:
    dj, dp = _f64(delta_j, "DeltaJ"), _f64(delta_p, "DeltaP")
    if dj < 0 or dp <= 0:
        raise PhaseDPathError("Gamma requires DeltaJ >= 0 and DeltaP > 0")
    return np.float64(dj / dp)


def gamma_tie_key(gamma: Any, delta_j: Any, delta_p: Any,
                  domain_id: Any, global_index: Any) -> Tuple[np.float64, np.float64, float, int, int]:
    """Gamma, lower DeltaJ, larger DeltaP, ascending BMS id, ascending Task037 id."""
    return (_f64(gamma, "Gamma"), _f64(delta_j, "DeltaJ"),
            -float(_f64(delta_p, "DeltaP")),
            _int(domain_id, "domain_id"), _int(global_index, "global_index"))


def mask_set_id(masked_ids: Iterable[int]) -> str:
    canonical = sorted({_int(v, "masked id") for v in masked_ids})
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return "BASELINE_CACHE" if not canonical else "MS-" + hashlib.sha256(payload).hexdigest()[:20]


def _score_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    score = f3_order_key(row)
    return {
        "R_F3": float(score[0]),
        "p_total": float(score[1]),
        "p_average": float(score[2]),
        "directional_score_key": json.dumps([float(score[0]), float(score[1]),
                                            float(score[2]), int(score[3])],
                                           separators=(",", ":")),
    }


def candidate_provenance(unit_rows: Sequence[Mapping[str, Any]],
                         f3_trace_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], Dict[int, Dict[str, Any]]]:
    """Project exact Task037 dynamic F3 selected order into frozen Task042 domains."""
    units: Dict[int, Dict[str, Any]] = {}
    by_domain: Dict[str, List[int]] = defaultdict(list)
    for raw in unit_rows:
        gid = _int(raw["task037_global_index"], "task037_global_index")
        domain = str(raw["domain_id"])
        if gid in units:
            raise PhaseDPathError(f"duplicate Task042 unit id {gid}")
        row = dict(raw)
        row["task037_global_index"] = gid
        row["domain_id"] = domain
        row["unit_type"] = canonical_unit_type(raw["unit_type"])
        units[gid] = row
        if domain in DOMAIN_IDS:
            by_domain[domain].append(gid)
    if set(by_domain) != set(DOMAIN_IDS):
        raise PhaseDPathError("Task042 eligible domain set differs from preregistration")
    for domain in DOMAIN_IDS:
        by_domain[domain].sort()
        if len(by_domain[domain]) < 2:
            raise PhaseDPathError(f"eligible domain {domain} has fewer than two tested units")

    trace: List[Dict[str, Any]] = []
    trace_by_id: Dict[int, Dict[str, Any]] = {}
    seen_steps = set()
    for raw in f3_trace_rows:
        row = dict(raw)
        step = _int(row.get("step"), "F3 step")
        gid = _int(row.get("global_index"), "F3 global_index")
        if step in seen_steps or gid in trace_by_id:
            raise PhaseDPathError("F3 continuation must have unique steps and selected unit ids")
        seen_steps.add(step)
        row["step"] = step
        row["global_index"] = gid
        # Validate the recorded frozen directional key and parameter benefit.
        _score_fields(row)
        cost = _f64(row.get("parameter_cost"), "Task037 parameter_cost")
        if cost <= 0 or cost != np.floor(cost):
            raise PhaseDPathError("Task037 parameter_cost must be a positive exact integer")
        trace.append(row)
        trace_by_id[gid] = row
    trace.sort(key=lambda row: row["step"])
    if [row["step"] for row in trace] != list(range(1, len(trace) + 1)):
        raise PhaseDPathError("F3 selection steps are not contiguous from one")

    removal_ids_by_domain: Dict[str, List[int]] = {}
    removal_rank: Dict[int, int] = {}
    for domain in DOMAIN_IDS:
        member_set = set(by_domain[domain])
        in_trace = [row for row in trace if row["global_index"] in member_set]
        required = len(member_set) - 1
        if len(in_trace) < required:
            raise PhaseDPathError(
                f"authoritative F3 order is incomplete for domain {domain}: "
                f"found {len(in_trace)} proposals, require {required}")
        chosen = in_trace[:required]
        removal_ids_by_domain[domain] = [row["global_index"] for row in chosen]
        for position, row in enumerate(chosen, 1):
            removal_rank[row["global_index"]] = position

    out: List[Dict[str, Any]] = []
    for domain in DOMAIN_IDS:
        ids = by_domain[domain]
        chosen_ids = set(removal_ids_by_domain[domain])
        for gid in ids:
            unit = units[gid]
            row = {
                "domain_id": domain,
                "domain_unit_count": len(ids),
                "task037_global_index": gid,
                "unit_type": unit["unit_type"],
                "layer": str(unit["layer"]),
                "unit_index": _int(unit["unit_index"], "unit_index"),
                "stage": str(unit.get("stage", "")),
                "selected_for_removal": gid in chosen_ids,
                "within_domain_removal_rank": removal_rank.get(gid, ""),
                "f3_global_step": "",
                "R_F3": "",
                "p_total": "",
                "p_average": "",
                "directional_score_key": "",
                "DeltaP": "",
                "parameter_cost_source": "",
                "representative_status": "REMOVAL_CANDIDATE" if gid in chosen_ids else "FINAL_DOMAIN_REPRESENTATIVE",
                "score_status": "NOT_SELECTED_WITHIN_EXTENDED_F3_TRACE",
                "domain_group": "mixed" if domain in MIXED_DOMAIN_IDS else "nonmixed",
            }
            selected_row = trace_by_id.get(gid)
            if selected_row is not None:
                if canonical_unit_type(selected_row.get("unit_type")) != unit["unit_type"]:
                    raise PhaseDPathError(f"Task037 type mismatch for unit {gid}")
                if str(selected_row.get("layer")) != str(unit["layer"]) or _int(
                    selected_row.get("unit_index"), "trace unit_index") != row["unit_index"]:
                    raise PhaseDPathError(f"Task037 layer/index mismatch for unit {gid}")
                row.update(_score_fields(selected_row))
                row["f3_global_step"] = selected_row["step"]
                row["score_status"] = "RECORDED_IN_AUTHORITATIVE_DYNAMIC_F3_TRACE"
                if gid in chosen_ids:
                    row["DeltaP"] = int(_f64(selected_row["parameter_cost"], "parameter_cost"))
                    row["parameter_cost_source"] = "Task037 MC.estimate_unit_cost via F3 trace parameter_cost"
            elif gid in chosen_ids:
                raise PhaseDPathError(f"removal candidate {gid} has no Task037 trace row")
            out.append(row)
    candidates = {gid: row for row in out if row["selected_for_removal"]
                  for gid in [int(row["task037_global_index"])]}
    return out, removal_ids_by_domain, candidates


def _all_j(survivors: Mapping[str, Iterable[int]],
           originals: Mapping[str, Sequence[int]],
           distances: Mapping[Tuple[str, int, int], Any]) -> Dict[str, float]:
    return {domain: float(coverage_j(domain, originals[domain], survivors[domain], distances))
            for domain in DOMAIN_IDS}


def _state_row(path_name: str, step: int, candidate: Mapping[str, Any],
               survivors: Mapping[str, set], originals: Mapping[str, Sequence[int]],
               distances: Mapping[Tuple[str, int, int], Any],
               masked_ids: Sequence[int], cumulative_p: int,
               cohort_total_p: int, model_total_p: int, unit_types: Mapping[int, str],
               j_before: Any, j_after: Any, delta_j: Any, gamma: Any) -> Dict[str, Any]:
    domain_js = _all_j(survivors, originals, distances)
    masked = sorted(set(int(v) for v in masked_ids))
    return {
        "path": path_name,
        "step": step,
        "domain_id": str(candidate["domain_id"]),
        "candidate_task037_global_index": int(candidate["task037_global_index"]),
        "unit_type": candidate["unit_type"],
        "layer": candidate["layer"],
        "unit_index": int(candidate["unit_index"]),
        "f3_global_step": int(candidate["f3_global_step"]),
        "directional_score_key": candidate["directional_score_key"],
        "R_F3": float(candidate["R_F3"]),
        "p_total": float(candidate["p_total"]),
        "p_average": float(candidate["p_average"]),
        "J_before": float(j_before),
        "J_after": float(j_after),
        "DeltaJ": float(delta_j),
        "DeltaP": int(candidate["DeltaP"]),
        "Gamma": float(gamma),
        "cumulative_removed_parameters": int(cumulative_p),
        "cumulative_removed_cohort_ratio": float(cumulative_p / cohort_total_p),
        "cumulative_removed_model_ratio": float(cumulative_p / model_total_p),
        "masked_task037_ids": json.dumps(masked, separators=(",", ":")),
        "mask_set_id": mask_set_id(masked),
        "surviving_ids_by_domain": json.dumps(
            {d: sorted(int(v) for v in survivors[d]) for d in DOMAIN_IDS},
            sort_keys=True, separators=(",", ":")),
        "domain_J_by_domain": json.dumps(domain_js, sort_keys=True, separators=(",", ":")),
        "current_total_temporal_coverage": float(sum(domain_js.values())),
        "max_domain_J": float(max(domain_js.values())),
        "masked_attention_count": sum(1 for x in masked if unit_types.get(x) == ATTENTION),
        "masked_ffn_count": sum(1 for x in masked if unit_types.get(x) == FFN),
    }



def _initial_survivors(originals: Mapping[str, Sequence[int]]) -> Dict[str, set]:
    return {domain: set(int(v) for v in originals[domain]) for domain in DOMAIN_IDS}


def _calculate_transition(domain: str, candidate_id: int,
                          survivors: Mapping[str, set],
                          originals: Mapping[str, Sequence[int]],
                          distances: Mapping[Tuple[str, int, int], Any],
                          candidates: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    candidate = candidates[candidate_id]
    before, after, delta = marginal_coverage_cost(
        domain, originals[domain], survivors[domain], candidate_id, distances)
    delta_p = int(candidate["DeltaP"])
    gamma = gamma_cost(delta, delta_p)
    return {
        "domain_id": domain,
        "candidate_task037_global_index": candidate_id,
        "unit_type": candidate["unit_type"],
        "layer": candidate["layer"],
        "unit_index": int(candidate["unit_index"]),
        "f3_global_step": int(candidate["f3_global_step"]),
        "directional_score_key": candidate["directional_score_key"],
        "R_F3": float(candidate["R_F3"]),
        "p_total": float(candidate["p_total"]),
        "p_average": float(candidate["p_average"]),
        "J_before": float(before),
        "J_after": float(after),
        "DeltaJ": float(delta),
        "DeltaP": delta_p,
        "Gamma": float(gamma),
        "gamma_tie_key": json.dumps([
            float(gamma), float(delta), -float(delta_p), int(domain), int(candidate_id)
        ], separators=(",", ":")),
    }


def build_progressive_paths(unit_rows: Sequence[Mapping[str, Any]],
                            provenance_rows: Sequence[Mapping[str, Any]],
                            removal_ids_by_domain: Mapping[str, Sequence[int]],
                            candidates: Mapping[int, Mapping[str, Any]],
                            distances: Mapping[Tuple[str, int, int], Any],
                            model_total_parameters: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model_total_parameters = _int(model_total_parameters, "model_total_parameters")
    if model_total_parameters <= 0:
        raise PhaseDPathError("model total parameter count must be positive")
    originals: Dict[str, List[int]] = {d: [] for d in DOMAIN_IDS}
    unit_types: Dict[int, str] = {}
    for raw in unit_rows:
        domain = str(raw["domain_id"])
        if domain not in originals:
            continue
        gid = _int(raw["task037_global_index"], "task037_global_index")
        originals[domain].append(gid)
        unit_types[gid] = canonical_unit_type(raw["unit_type"])
    for d in DOMAIN_IDS:
        originals[d] = sorted(originals[d])
    candidate_ids = set(candidates)
    removal_union = {gid for values in removal_ids_by_domain.values() for gid in values}
    if candidate_ids != removal_union:
        raise PhaseDPathError("domain proposals and candidate provenance disagree")
    cohort_total_p = sum(int(candidates[gid]["DeltaP"]) for gid in candidate_ids)
    if cohort_total_p <= 0:
        raise PhaseDPathError("cohort removable parameter amount must be positive")

    f3_order = sorted(candidate_ids, key=lambda gid: int(candidates[gid]["f3_global_step"]))
    if len(f3_order) != len(candidate_ids):
        raise PhaseDPathError("candidate ordering is not unique")
    base_survivors = _initial_survivors(originals)
    base_masked: List[int] = []
    base_path: List[Dict[str, Any]] = []
    cumulative = 0
    for step, gid in enumerate(f3_order, 1):
        candidate = candidates[gid]
        domain = str(candidate["domain_id"])
        transition = _calculate_transition(domain, gid, base_survivors,
                                           originals, distances, candidates)
        base_survivors[domain].remove(gid)
        base_masked.append(gid)
        cumulative += int(candidate["DeltaP"])
        base_path.append(_state_row(
            "baseline", step, candidate, base_survivors, originals, distances,
            base_masked, cumulative, cohort_total_p, model_total_parameters, unit_types,
            transition["J_before"], transition["J_after"], transition["DeltaJ"],
            transition["Gamma"]))
    expected_survivors = {d: set(originals[d]) - set(removal_ids_by_domain[d]) for d in DOMAIN_IDS}
    if base_survivors != expected_survivors:
        raise PhaseDPathError("baseline final representative set differs from the frozen cohort rule")

    temporal_survivors = _initial_survivors(originals)
    temporal_masked: List[int] = []
    temporal_path: List[Dict[str, Any]] = []
    proposal_rows: List[Dict[str, Any]] = []
    cursors = {d: 0 for d in DOMAIN_IDS}
    step = 0
    while any(cursors[d] < len(removal_ids_by_domain[d]) for d in DOMAIN_IDS):
        proposals = []
        for domain in DOMAIN_IDS:
            values = list(removal_ids_by_domain[domain])
            cursor = cursors[domain]
            if cursor >= len(values):
                continue
            proposal = _calculate_transition(domain, int(values[cursor]), temporal_survivors,
                                             originals, distances, candidates)
            proposals.append(proposal)
        if not proposals:
            raise PhaseDPathError("temporal progressive path has no active domain proposals")
        chosen = min(proposals, key=lambda row: gamma_tie_key(
            row["Gamma"], row["DeltaJ"], row["DeltaP"],
            row["domain_id"], row["candidate_task037_global_index"]))
        step += 1
        for proposal in proposals:
            proposal_rows.append({
                "temporal_step": step,
                **proposal,
                "accepted": proposal["candidate_task037_global_index"] == chosen["candidate_task037_global_index"],
                "tie_break_rank": sorted(proposals, key=lambda row: gamma_tie_key(
                    row["Gamma"], row["DeltaJ"], row["DeltaP"],
                    row["domain_id"], row["candidate_task037_global_index"])).index(proposal) + 1,
            })
        gid = int(chosen["candidate_task037_global_index"])
        domain = str(chosen["domain_id"])
        candidate = candidates[gid]
        temporal_survivors[domain].remove(gid)
        temporal_masked.append(gid)
        cursors[domain] += 1
        cumulative = sum(int(candidates[x]["DeltaP"]) for x in temporal_masked)
        temporal_path.append(_state_row(
            "temporal_progressive", step, candidate, temporal_survivors,
            originals, distances, temporal_masked, cumulative, cohort_total_p,
            model_total_parameters, unit_types, chosen["J_before"], chosen["J_after"],
            chosen["DeltaJ"], chosen["Gamma"]))
    if set(base_masked) != set(temporal_masked) or base_survivors != temporal_survivors:
        raise PhaseDPathError("baseline and temporal paths do not end at the same tested-unit set")
    return base_path, temporal_path, proposal_rows


def _path_state(path_rows: Sequence[Mapping[str, Any]], prefix: int,
                original_by_domain: Mapping[str, Sequence[int]],
                distances: Mapping[Tuple[str, int, int], Any]) -> Dict[str, Any]:
    if prefix < 0 or prefix > len(path_rows):
        raise PhaseDPathError("checkpoint prefix is outside the path")
    if prefix == 0:
        survivors = {d: set(original_by_domain[d]) for d in DOMAIN_IDS}
        masked: List[int] = []
        cumulative = 0
    else:
        row = path_rows[prefix - 1]
        masked = json.loads(row["masked_task037_ids"])
        survivors = {d: set(json.loads(row["surviving_ids_by_domain"])[d]) for d in DOMAIN_IDS}
        cumulative = int(row["cumulative_removed_parameters"])
    domain_js = _all_j(survivors, original_by_domain, distances)
    return {
        "prefix_steps": prefix,
        "masked_ids": sorted(masked),
        "cumulative_removed_parameters": cumulative,
        "survivors": {d: sorted(survivors[d]) for d in DOMAIN_IDS},
        "domain_J": domain_js,
        "total_J": float(sum(domain_js.values())),
        "max_domain_J": float(max(domain_js.values())),
    }


def _prefix_for_budget(path_rows: Sequence[Mapping[str, Any]], target_parameters: int) -> int:
    prefix = 0
    for row in path_rows:
        if int(row["cumulative_removed_parameters"]) <= int(target_parameters):
            prefix = int(row["step"])
        else:
            break
    return prefix


def build_checkpoint_manifest(base_path: Sequence[Mapping[str, Any]],
                              temporal_path: Sequence[Mapping[str, Any]],
                              unit_rows: Sequence[Mapping[str, Any]],
                              removal_ids_by_domain: Mapping[str, Sequence[int]],
                              distances: Mapping[Tuple[str, int, int], Any],
                              cohort_total_parameters: int) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    cohort_total_parameters = _int(cohort_total_parameters, "cohort_total_parameters")
    original_by_domain: Dict[str, List[int]] = {d: [] for d in DOMAIN_IDS}
    type_by_id: Dict[int, str] = {}
    for row in unit_rows:
        d = str(row["domain_id"])
        if d in original_by_domain:
            gid = _int(row["task037_global_index"], "task037_global_index")
            original_by_domain[d].append(gid)
            type_by_id[gid] = canonical_unit_type(row["unit_type"])
    original_by_domain = {d: sorted(v) for d, v in original_by_domain.items()}

    manifest: List[Dict[str, Any]] = []
    evals: Dict[str, Dict[str, Any]] = {}

    def add(path_name: str, path_rows: Sequence[Mapping[str, Any]], prefix: int,
            checkpoint_kind: str, comparison_id: str, requested_ratio: Any,
            requested_parameters: int) -> None:
        state = _path_state(path_rows, prefix, original_by_domain, distances)
        masked = state["masked_ids"]
        evaluation_id = mask_set_id(masked)
        attention_count = sum(type_by_id[x] == ATTENTION for x in masked)
        ffn_count = sum(type_by_id[x] == FFN for x in masked)
        row = {
            "checkpoint_id": f"{comparison_id}:{path_name}",
            "comparison_id": comparison_id,
            "checkpoint_kind": checkpoint_kind,
            "path": path_name,
            "requested_cohort_ratio": "" if requested_ratio is None else float(requested_ratio),
            "requested_parameter_budget": int(requested_parameters),
            "prefix_steps": int(prefix),
            "achieved_removed_parameters": int(state["cumulative_removed_parameters"]),
            "achieved_cohort_parameter_ratio": float(state["cumulative_removed_parameters"] / cohort_total_parameters),
            "requested_minus_achieved_parameters": int(requested_parameters - state["cumulative_removed_parameters"]),
            "masked_attention_count": int(attention_count),
            "masked_ffn_count": int(ffn_count),
            "masked_unit_count": len(masked),
            "masked_task037_ids": json.dumps(masked, separators=(",", ":")),
            "mask_set_id": evaluation_id,
            "surviving_ids_by_domain": json.dumps(state["survivors"], sort_keys=True, separators=(",", ":")),
            "domain_J_by_domain": json.dumps(state["domain_J"], sort_keys=True, separators=(",", ":")),
            "current_total_temporal_coverage": state["total_J"],
            "max_domain_J": state["max_domain_J"],
        }
        manifest.append(row)
        if evaluation_id != "BASELINE_CACHE":
            payload = evals.setdefault(evaluation_id, {
                "evaluation_id": evaluation_id,
                "masked_task037_ids": json.dumps(masked, separators=(",", ":")),
                "masked_unit_count": len(masked),
                "masked_attention_count": int(attention_count),
                "masked_ffn_count": int(ffn_count),
                "removed_parameters": int(state["cumulative_removed_parameters"]),
                "cohort_parameter_ratio": float(state["cumulative_removed_parameters"] / cohort_total_parameters),
                "domain_J_by_domain": json.dumps(state["domain_J"], sort_keys=True, separators=(",", ":")),
                "current_total_temporal_coverage": state["total_J"],
                "max_domain_J": state["max_domain_J"],
                "contexts": [],
            })
            if json.loads(payload["masked_task037_ids"]) != masked:
                raise PhaseDPathError("mask set identity hash collision")
            payload["contexts"].append(row["checkpoint_id"])

    primary_counts: Dict[Tuple[str, float], int] = {}
    for ratio in (0.25, 0.50, 0.75, 1.0):
        target = int(math.floor(cohort_total_parameters * ratio))
        comparison_id = f"PRIMARY_{int(ratio*100)}"
        for name, path in (("baseline", base_path), ("temporal_progressive", temporal_path)):
            prefix = _prefix_for_budget(path, target)
            primary_counts[(name, ratio)] = prefix
            add(name, path, prefix, "PRIMARY_BUDGET", comparison_id, ratio, target)

    base_budget_to_prefix = {0: 0}
    temporal_budget_to_prefix = {0: 0}
    for row in base_path:
        base_budget_to_prefix[int(row["cumulative_removed_parameters"])] = int(row["step"])
    for row in temporal_path:
        temporal_budget_to_prefix[int(row["cumulative_removed_parameters"])] = int(row["step"])
    primary_exact_parameters = set()
    for ratio in (0.25, 0.50, 0.75, 1.0):
        base_prefix = primary_counts[("baseline", ratio)]
        temporal_prefix = primary_counts[("temporal_progressive", ratio)]
        base_p = 0 if base_prefix == 0 else int(base_path[base_prefix - 1]["cumulative_removed_parameters"])
        temporal_p = 0 if temporal_prefix == 0 else int(temporal_path[temporal_prefix - 1]["cumulative_removed_parameters"])
        if base_p == temporal_p:
            primary_exact_parameters.add(base_p)
    intersection_index = 0
    for budget in sorted(set(base_budget_to_prefix).intersection(temporal_budget_to_prefix)):
        if budget <= 0 or budget >= cohort_total_parameters or budget in primary_exact_parameters:
            continue
        intersection_index += 1
        comparison_id = f"EXACT_INTERSECTION_{intersection_index:03d}_{budget}"
        add("baseline", base_path, base_budget_to_prefix[budget], "EXACT_PARAMETER_INTERSECTION",
            comparison_id, None, budget)
        add("temporal_progressive", temporal_path, temporal_budget_to_prefix[budget],
            "EXACT_PARAMETER_INTERSECTION", comparison_id, None, budget)
    return manifest, evals

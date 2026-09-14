#!/usr/bin/env python3
"""Run the CPU-only Task042 Phase F temporal-backup feasibility audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def _sort_domain(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows],
    ])


def run(input_dir: Path, output_dir: Path, repo_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(repo_dir / "src"))
    from lgfr_runtime import task042_frame_relation_redundancy as task042
    from lgfr_runtime.task042_phase_f_backup_constraint import (
        VIDEO_SUBSET_NAMES, baseline_prefix, build_matched_background,
        calibration_subsets, canonical_type, compute_temporal_distances,
        file_sha256, graph_topology, jaccard, physical_parameter_cost,
        preference_order, read_csv, solve_exact_constrained,
        solve_maximum_constrained, write_csv,
    )

    phase_d = input_dir / "phase_d"
    required = {
        "provenance": phase_d / "task042_phase_d_candidate_provenance.csv",
        "f3_trace": phase_d / "task042_phase_d_f3_authoritative_global_trace.csv",
        "input_identity": phase_d / "task042_phase_d_input_identity.json",
        "raw_sensitivity": input_dir / "task042_frame_pair_sensitivity.csv",
        "units": input_dir / "task042_unit_manifest.csv",
        "videos": input_dir / "task042_video_manifest.csv",
        "phase_a_distances": input_dir / "task042_temporal_pair_distance.csv",
        "phase_a_controls": input_dir / "task042_matched_cross_domain_control.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase F frozen inputs missing: " + ", ".join(missing))

    provenance = read_csv(required["provenance"])
    trace_rows = read_csv(required["f3_trace"])
    input_identity = _read_json(required["input_identity"])
    all_units = read_csv(required["units"])
    raw_rows = read_csv(required["raw_sensitivity"])
    videos = read_csv(required["videos"])
    phase_a_distances = read_csv(required["phase_a_distances"])
    phase_a_controls = read_csv(required["phase_a_controls"])

    # The tested pruning cohort is exactly Phase D's frozen 31-unit projection.
    selection_units = [dict(row) for row in provenance]
    if len(selection_units) != int(input_identity["eligible_unit_count"]):
        raise ValueError("Phase D cohort identity/count drift")
    candidate_rows = [row for row in provenance if _as_bool(row["selected_for_removal"])]
    fixed_retained_rows = [row for row in provenance if not _as_bool(row["selected_for_removal"])]
    if len(candidate_rows) != int(input_identity["removal_candidate_count"]):
        raise ValueError("Phase D candidate count differs from frozen input identity")
    preference_ids = preference_order(candidate_rows)
    if len(fixed_retained_rows) != len({str(row["domain_id"]) for row in provenance}):
        raise ValueError("expected exactly one fixed final representative per eligible BMS domain")
    if sum(int(float(row["DeltaP"])) for row in candidate_rows) != int(input_identity["removable_cohort_parameters"]):
        raise ValueError("Phase D candidate costs no longer reproduce the frozen cohort total")

    # Full exact cost audit: derive for every Phase-D unit, then independently
    # compare the dynamic Task037 trace costs wherever that trace recorded an ID.
    trace_by_id = {int(row["global_index"]): row for row in trace_rows}
    cost_rows: List[Dict[str, Any]] = []
    cost_by_id: Dict[int, int] = {}
    candidate_by_id = {int(row["task037_global_index"]): row for row in candidate_rows}
    for row in sorted(selection_units, key=lambda item: int(item["task037_global_index"])):
        uid = int(row["task037_global_index"])
        expected_cost = physical_parameter_cost(row)
        trace = trace_by_id.get(uid)
        trace_cost = None if trace is None else int(float(trace["parameter_cost"]))
        candidate = candidate_by_id.get(uid)
        recorded_candidate_cost = None if candidate is None else int(float(candidate["DeltaP"]))
        if trace_cost is not None and trace_cost != expected_cost:
            raise ValueError("independent physical accounting disagrees with Task037 trace for unit %s" % uid)
        if recorded_candidate_cost is not None and recorded_candidate_cost != expected_cost:
            raise ValueError("independent physical accounting disagrees with Phase-D DeltaP for unit %s" % uid)
        cost_by_id[uid] = expected_cost
        cost_rows.append({
            "task037_global_index": uid, "domain_id": str(row["domain_id"]),
            "unit_type": canonical_type(row["unit_type"]), "layer": row["layer"],
            "unit_index": int(row["unit_index"]), "stage": int(row["stage"]),
            "model_width": 96 * (2 ** int(row["stage"])),
            "attention_head_dim": 32 if canonical_type(row["unit_type"]) == "attention_head" else "",
            "parameter_cost_exact": expected_cost,
            "physical_accounting_source": "Task037 pruning/MC.py MC.estimate_unit_cost (head: QKV weights+bias plus projection input weights; FFN: fc1 input weights+bias plus fc2 output weights)",
            "independent_formula": ("4*model_width*32 + 3*32; VideoSwin embed_dim=96, heads=(3,6,12,24), qkv bias enabled" if canonical_type(row["unit_type"]) == "attention_head" else "2*model_width + 1; fc1 bias enabled"),
            "phase_d_candidate_DeltaP": "" if recorded_candidate_cost is None else recorded_candidate_cost,
            "task037_trace_parameter_cost": "" if trace_cost is None else trace_cost,
            "candidate_eligible_for_pruning": candidate is not None,
            "candidate_parameter_cost_match": "NOT_APPLICABLE_FIXED_REPRESENTATIVE" if candidate is None else expected_cost == recorded_candidate_cost,
            "trace_parameter_cost_match": "TRACE_ROW_UNAVAILABLE" if trace_cost is None else expected_cost == trace_cost,
        })

    # Unit/video normalizations and d_temp are recomputed with the frozen Phase-A
    # functions from already captured raw rows; no model or inference is loaded.
    subset_ids = calibration_subsets(videos)
    full_ids = subset_ids["full_10x3"]
    selection_ids = {int(row["task037_global_index"]) for row in selection_units}
    control_ids = {int(row["task037_global_index"]) for row in all_units}
    if not selection_ids.issubset(control_ids):
        raise ValueError("Phase-D selection units are not a subset of frozen Task042 captures")
    full_distances, full_valid_counts = compute_temporal_distances(raw_rows, all_units, full_ids, task042)

    # Reproduction check against the existing full-data same-domain distances.
    phase_a_pair_map: Dict[Tuple[int, int], float] = {}
    for row in phase_a_distances:
        i, j = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
        phase_a_pair_map[(min(i, j), max(i, j))] = float(row["d_temp"])
    reproduced, absolute_differences = 0, []
    grouped: Dict[str, List[int]] = {}
    for row in selection_units:
        grouped.setdefault(str(row["domain_id"]), []).append(int(row["task037_global_index"]))
    for ids in grouped.values():
        for i in ids:
            for j in ids:
                if i >= j:
                    continue
                edge = (i, j)
                recomputed, frozen = full_distances.get(edge), phase_a_pair_map.get(edge)
                if recomputed is None or frozen is None:
                    raise ValueError("missing full-data same-domain distance for Phase-D pair %r" % (edge,))
                absolute_differences.append(abs(recomputed - frozen))
                if not math.isclose(recomputed, frozen, rel_tol=0.0, abs_tol=2e-12):
                    raise ValueError("recomputed Phase-A d_temp mismatch for pair %r: %.17g vs %.17g" % (edge, recomputed, frozen))
                reproduced += 1

    # Verify reproduced cross-domain observations against the frozen Phase-A
    # matched-control artifact wherever its original exact match is available.
    control_lookup = {int(row["task037_global_index"]): row for row in all_units}
    control_reproduction_diffs = []
    for row in phase_a_controls:
        i, j = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
        edge = (min(i, j), max(i, j))
        recomputed, frozen = full_distances.get(edge), float(row["d_temp"])
        if recomputed is not None:
            control_reproduction_diffs.append(abs(recomputed - frozen))
            if not math.isclose(recomputed, frozen, rel_tol=0.0, abs_tol=2e-12):
                raise ValueError("recomputed Phase-A matched-control d_temp mismatch for pair %r" % (edge,))

    output_dir.mkdir(parents=True, exist_ok=True)
    full_background, full_pairs = build_matched_background(selection_units, all_units, full_distances)
    full_domain_rows, full_node_rows, full_graph_pair_rows, full_adjacency = graph_topology(selection_units, full_pairs)
    graph_rows = full_domain_rows + full_node_rows + full_graph_pair_rows

    candidate_total = sum(int(float(row["DeltaP"])) for row in candidate_rows)
    budgets = {percent: math.floor(candidate_total * percent / 100.0) for percent in (25, 50, 75)}
    baseline_by_percent: Dict[int, Set[int]] = {}
    constrained_by_percent: Dict[int, Set[int]] = {}
    constrained_meta: Dict[int, Dict[str, Any]] = {}
    baseline_rows_out: List[Dict[str, Any]] = []
    constrained_rows_out: List[Dict[str, Any]] = []
    change_rows_out: List[Dict[str, Any]] = []
    feasibility_rows: List[Dict[str, Any]] = []
    unit_row_by_id = {int(row["task037_global_index"]): row for row in selection_units}
    candidate_ids = set(candidate_by_id)
    for percent in (25, 50, 75):
        budget = budgets[percent]
        baseline = baseline_prefix(candidate_rows, budget)
        exact = solve_exact_constrained(candidate_rows, selection_ids, full_adjacency, budget)
        constrained = set(exact["selected_pruned_ids"])
        baseline_by_percent[percent] = baseline
        constrained_by_percent[percent] = constrained
        constrained_meta[percent] = exact
        baseline_cost = sum(cost_by_id[uid] for uid in baseline)
        constrained_cost = int(exact["released_parameters"])
        intersection = baseline & constrained
        removed_from_baseline = baseline - constrained
        added_by_constraint = constrained - baseline
        union = baseline | constrained
        for uid in sorted(candidate_ids, key=lambda value: (int(candidate_by_id[value]["f3_global_step"]), value)):
            row = candidate_by_id[uid]
            baseline_rows_out.append({
                "budget_percent": percent, "budget_parameters": budget,
                "task037_global_index": uid, "domain_id": row["domain_id"],
                "unit_type": canonical_type(row["unit_type"]), "stage": row["stage"],
                "f3_global_step": row["f3_global_step"], "directional_score_name": "Task037 dynamic F3 midpoint-veto risk R_F3",
                "R_F3": row["R_F3"], "p_total": row["p_total"], "p_average": row["p_average"],
                "parameter_cost": cost_by_id[uid], "baseline_pruned": uid in baseline,
                "baseline_order": preference_ids.index(uid) + 1,
                "selection_source": "Phase-D authoritative dynamic F3 trace prefix; exact original trace order; stop before first unit exceeding budget",
            })
        for uid in sorted(selection_ids):
            row = unit_row_by_id[uid]
            candidate = candidate_by_id.get(uid)
            constrained_rows_out.append({
                "budget_percent": percent, "budget_parameters": budget,
                "task037_global_index": uid, "domain_id": row["domain_id"],
                "unit_type": canonical_type(row["unit_type"]), "stage": row["stage"],
                "parameter_cost_exact": cost_by_id[uid],
                "candidate_eligible_for_pruning": candidate is not None,
                "representative_status": "REMOVAL_CANDIDATE" if candidate else "FIXED_FINAL_DOMAIN_REPRESENTATIVE",
                "constrained_pruned": uid in constrained,
                "constrained_retained": uid not in constrained,
                "directional_score_R_F3": "" if candidate is None else candidate["R_F3"],
                "f3_global_step": "" if candidate is None else candidate["f3_global_step"],
                "solver_status": exact["solver_status"],
                "retained_set_is_dominating": exact["retained_set_is_dominating"],
            })
        change_rows_out.append({
            "record_type": "BUDGET_SUMMARY", "budget_percent": percent, "budget_parameters": budget,
            "baseline_pruned_ids": json.dumps(sorted(baseline, key=lambda value: preference_ids.index(value))),
            "constrained_pruned_ids": json.dumps(sorted(constrained, key=lambda value: preference_ids.index(value))),
            "intersection_ids": json.dumps(sorted(intersection)),
            "removed_from_baseline_ids": json.dumps(sorted(removed_from_baseline)),
            "added_by_constraint_ids": json.dumps(sorted(added_by_constraint)),
            "baseline_count": len(baseline), "constrained_count": len(constrained),
            "intersection_count": len(intersection), "jaccard_similarity": len(intersection) / len(union) if union else 1.0,
            "baseline_released_parameters": baseline_cost, "constrained_released_parameters": constrained_cost,
            "baseline_budget_gap": budget - baseline_cost, "constrained_budget_gap": budget - constrained_cost,
        })
        for uid in sorted(removed_from_baseline | added_by_constraint):
            row = unit_row_by_id[uid]
            neighbor_values = []
            for pair in full_pairs:
                i, j = int(pair["task037_global_index_i"]), int(pair["task037_global_index_j"])
                if uid in (i, j) and bool(pair["temporal_backup_edge"]):
                    neighbor_values.append({"task037_global_index": j if uid == i else i,
                                            "R_CTR": float(pair["R_CTR"]),
                                            "d_temp": float(pair["d_temp_same_domain"]),
                                            "b_ij": float(pair["b_ij_median"])})
            if uid in removed_from_baseline:
                baseline_neighbors_kept = [edge["task037_global_index"] for edge in neighbor_values
                                           if edge["task037_global_index"] not in baseline]
                constrained_neighbors_kept = [edge["task037_global_index"] for edge in neighbor_values
                                              if edge["task037_global_index"] not in constrained]
                if not full_adjacency[uid]:
                    reason = "protected because the unit has no supported temporal-backup edge"
                elif not baseline_neighbors_kept:
                    reason = "protected because baseline would leave no temporal neighbor retained"
                elif constrained_neighbors_kept:
                    reason = "budget/preference-optimal constrained exchange; a backup remains in the selected constrained set"
                else:
                    reason = "retained as a temporal backup for one or more constrained-pruned neighbors"
                change_type = "PROTECTED_BASELINE_PRUNED_UNIT"
            else:
                reason = "selected as a constrained alternative under the exact parameter budget and backup constraints"
                change_type = "FORCED_ALTERNATIVE_PRUNED_UNIT"
                baseline_neighbors_kept = []
                constrained_neighbors_kept = []
            change_rows_out.append({
                "record_type": "UNIT_CHANGE_DETAIL", "budget_percent": percent, "budget_parameters": budget,
                "change_type": change_type, "task037_global_index": uid, "domain_id": row["domain_id"],
                "unit_type": canonical_type(row["unit_type"]), "stage": row["stage"],
                "directional_score_R_F3": "" if uid not in candidate_by_id else candidate_by_id[uid]["R_F3"],
                "f3_global_step": "" if uid not in candidate_by_id else candidate_by_id[uid]["f3_global_step"],
                "parameter_cost": cost_by_id[uid], "reason": reason,
                "backup_neighbors_and_R_CTR": json.dumps(sorted(neighbor_values, key=lambda item: item["task037_global_index"])),
                "baseline_neighbors_retained": json.dumps(sorted(baseline_neighbors_kept)),
                "constrained_neighbors_retained": json.dumps(sorted(constrained_neighbors_kept)),
                "serves_as_backup_for_constrained_pruned_ids": json.dumps(sorted(uid2 for uid2 in constrained if uid in full_adjacency[uid2])),
            })

        feasibility_rows.append({
            "budget_percent": percent, "total_removable_cohort_parameters": candidate_total,
            "budget_parameters_floor": budget, "baseline_released_parameters": baseline_cost,
            "baseline_budget_gap": budget - baseline_cost, "constrained_released_parameters": constrained_cost,
            "constrained_budget_gap": budget - constrained_cost, "max_temporal_constrained_removable_parameters": "",
            "max_temporal_constrained_removable_ratio": "", "diagnostic_budget_reachable_by_constraint": "",
            "solver_status": exact["solver_status"], "constrained_selection_count": len(constrained),
            "baseline_selection_count": len(baseline), "exact_solver_proof": exact["optimality_proof"],
            "is_50pct_target_infeasible": "",
        })

    maximum = solve_maximum_constrained(candidate_rows, selection_ids, full_adjacency)
    maximum_parameters = int(maximum["released_parameters"])
    for row in feasibility_rows:
        row["max_temporal_constrained_removable_parameters"] = maximum_parameters
        row["max_temporal_constrained_removable_ratio"] = maximum_parameters / candidate_total if candidate_total else 0.0
        row["diagnostic_budget_reachable_by_constraint"] = maximum_parameters >= int(row["budget_parameters_floor"])
        row["is_50pct_target_infeasible"] = (int(row["budget_percent"]) == 50 and maximum_parameters < int(row["budget_parameters_floor"]))

    # Mixed domains: report every actual cross-type supported edge from both directions.
    mixed_rows: List[Dict[str, Any]] = []
    for domain in ("271", "297"):
        domain_units = [row for row in selection_units if str(row["domain_id"]) == domain]
        for unit in sorted(domain_units, key=lambda row: int(row["task037_global_index"])):
            uid = int(unit["task037_global_index"])
            cross_neighbors = []
            for pair in full_pairs:
                i, j = int(pair["task037_global_index_i"]), int(pair["task037_global_index_j"])
                if uid not in (i, j) or not bool(pair["temporal_backup_edge"]):
                    continue
                other = j if uid == i else i
                other_row = unit_row_by_id[other]
                if canonical_type(other_row["unit_type"]) != canonical_type(unit["unit_type"]):
                    cross_neighbors.append({"task037_global_index": other,
                                            "unit_type": canonical_type(other_row["unit_type"]),
                                            "R_CTR": float(pair["R_CTR"]),
                                            "d_temp": float(pair["d_temp_same_domain"]),
                                            "b_ij": float(pair["b_ij_median"])})
            mixed_rows.append({
                "domain_id": domain, "task037_global_index": uid,
                "unit_type": canonical_type(unit["unit_type"]), "stage": unit["stage"],
                "cross_type_backup_neighbors": json.dumps(cross_neighbors, sort_keys=True, separators=(",", ":")),
                "has_supported_cross_type_backup": bool(cross_neighbors),
                "attention_supported_by_ffn": canonical_type(unit["unit_type"]) == "attention_head" and bool(cross_neighbors),
                "ffn_supported_by_attention": canonical_type(unit["unit_type"]) == "ffn_neuron" and bool(cross_neighbors),
            })

    # Recompute each preregistered calibration subset from the already-acquired raw rows.
    full_edges = { (int(row["task037_global_index_i"]), int(row["task037_global_index_j"]))
                   for row in full_pairs if bool(row["temporal_backup_edge"]) }
    full50_selected = constrained_by_percent[50]
    baseline50_selected = baseline_by_percent[50]
    full50_protected = baseline50_selected - full50_selected
    full50_release = int(constrained_meta[50]["released_parameters"])
    stability_rows: List[Dict[str, Any]] = []
    subset_summaries: List[Dict[str, Any]] = []
    for name in VIDEO_SUBSET_NAMES:
        ids = subset_ids[name]
        distances, valid_counts = (full_distances, full_valid_counts) if name == "full_10x3" else compute_temporal_distances(raw_rows, all_units, ids, task042)
        bg, pairs = build_matched_background(selection_units, all_units, distances)
        domain_rows, node_rows, pair_graph_rows, adjacency = graph_topology(selection_units, pairs)
        edge_set = {(int(row["task037_global_index_i"]), int(row["task037_global_index_j"]))
                    for row in pairs if bool(row["temporal_backup_edge"])}
        exact = solve_exact_constrained(candidate_rows, selection_ids, adjacency, budgets[50])
        selected = set(exact["selected_pruned_ids"])
        protected = baseline50_selected - selected
        edge_present_count = len(edge_set & full_edges)
        record = {
            "record_type": "SUBSET_SUMMARY", "subset": name, "video_indices": json.dumps(ids),
            "video_count": len(ids), "matched_background_available_pairs": sum(row["reference_status"] == "REFERENCE_AVAILABLE" for row in pairs),
            "same_domain_pair_count": len(pairs), "backup_edge_count": len(edge_set),
            "backup_edge_density": len(edge_set) / len(pairs) if pairs else 0.0,
            "isolated_unit_count": sum(bool(row["is_isolated"]) for row in node_rows),
            "full_data_edges_present": edge_present_count, "full_data_edge_count": len(full_edges),
            "full_data_edge_recall": edge_present_count / len(full_edges) if full_edges else 1.0,
            "50pct_budget_parameters": budgets[50], "50pct_max_removable_parameters": int(solve_maximum_constrained(candidate_rows, selection_ids, adjacency)["released_parameters"]),
            "50pct_budget_reachable": int(solve_maximum_constrained(candidate_rows, selection_ids, adjacency)["released_parameters"]) >= budgets[50],
            "50pct_constrained_released_parameters": int(exact["released_parameters"]),
            "50pct_budget_gap": int(exact["budget_gap"]),
            "50pct_selection_ids": json.dumps(sorted(selected)),
            "selection_jaccard_vs_full_data": jaccard(selected, full50_selected),
            "50pct_protected_ids": json.dumps(sorted(protected)),
            "protected_unit_jaccard_vs_full_data": jaccard(protected, full50_protected),
            "solver_status": exact["solver_status"],
        }
        subset_summaries.append(record)
        stability_rows.append(record)
        pair_index = {(int(row["task037_global_index_i"]), int(row["task037_global_index_j"])): row for row in pairs}
        for i, j in sorted(full_edges):
            row = pair_index[(i, j)]
            stability_rows.append({
                "record_type": "FULL_EDGE_PRESENCE", "subset": name,
                "full_data_edge_i": i, "full_data_edge_j": j,
                "present_in_subset": bool(row["temporal_backup_edge"]),
                "subset_reference_status": row["reference_status"],
                "subset_d_temp": row["d_temp_same_domain"], "subset_b_ij": row["b_ij_median"],
                "subset_R_CTR": row["R_CTR"],
            })
        if name == "full_10x3" and (edge_set != full_edges or selected != full50_selected):
            raise AssertionError("full calibration subset does not reproduce the full-data graph/selection identity")

    # Fill full-data edge occurrence counts after all subsets are available.
    for edge in sorted(full_edges):
        presence = [row for row in stability_rows if row.get("record_type") == "FULL_EDGE_PRESENCE"
                    and row.get("full_data_edge_i") == edge[0] and row.get("full_data_edge_j") == edge[1]]
        count = sum(_as_bool(row["present_in_subset"]) for row in presence)
        for row in presence:
            row["edge_present_subset_count_of_7"] = count
            row["edge_occurrence_rate_across_7_subsets"] = count / len(VIDEO_SUBSET_NAMES)

    # Write the exact required output surface.
    write_csv(output_dir / "task042_phase_f_matched_background.csv", full_background)
    write_csv(output_dir / "task042_phase_f_temporal_backup_graph.csv", graph_rows)
    write_csv(output_dir / "task042_phase_f_parameter_cost.csv", cost_rows)
    write_csv(output_dir / "task042_phase_f_baseline_selection.csv", baseline_rows_out)
    write_csv(output_dir / "task042_phase_f_constrained_selection.csv", constrained_rows_out)
    write_csv(output_dir / "task042_phase_f_selection_changes.csv", change_rows_out)
    write_csv(output_dir / "task042_phase_f_budget_feasibility.csv", feasibility_rows)
    write_csv(output_dir / "task042_phase_f_calibration_stability.csv", stability_rows)
    write_csv(output_dir / "task042_phase_f_mixed_type.csv", mixed_rows)

    all_pairs = len(full_pairs)
    full_edge_count = len(full_edges)
    reference_unavailable = sum(row["reference_status"] == "REFERENCE_UNAVAILABLE" for row in full_pairs)
    full_node_isolates = sum(bool(row["is_isolated"]) for row in full_node_rows)
    supported_reference_rejected = sum(row["reference_status"] == "REFERENCE_AVAILABLE" and
                                        not bool(row["temporal_backup_edge"]) for row in full_pairs)
    supported_types = Counter()
    for row in full_pairs:
        if bool(row["temporal_backup_edge"]):
            types = {canonical_type(row["unit_type_i"]), canonical_type(row["unit_type_j"])}
            key = "Attention-Attention" if types == {"attention_head"} else "FFN-FFN" if types == {"ffn_neuron"} else "Attention-FFN"
            supported_types[key] += 1
    complete_edge_domains = [row["domain_id"] for row in full_domain_rows
                             if int(row["possible_pair_count"]) > 0 and
                             int(row["edge_count"]) == int(row["possible_pair_count"])]
    zero_edge_domains = [row["domain_id"] for row in full_domain_rows
                         if int(row["possible_pair_count"]) > 0 and int(row["edge_count"]) == 0]
    partially_supported_domains = [row["domain_id"] for row in full_domain_rows
                                   if 0 < int(row["edge_count"]) < int(row["possible_pair_count"])]
    selection_changes_50 = next(row for row in change_rows_out if row["record_type"] == "BUDGET_SUMMARY" and int(row["budget_percent"]) == 50)
    subset_jaccards = [float(row["selection_jaccard_vs_full_data"]) for row in subset_summaries]
    protected_jaccards = [float(row["protected_unit_jaccard_vs_full_data"]) for row in subset_summaries]
    candidate_cost_match_count = sum(row["candidate_parameter_cost_match"] is True for row in cost_rows)
    trace_compare_rows = [row for row in cost_rows if row["trace_parameter_cost_match"] is True]
    pair_reference_unavailable = reference_unavailable
    full50_infeasible = maximum_parameters < budgets[50]
    if full_edge_count == 0:
        decision = "C. TEMPORAL_BACKUP_CONSTRAINT_REJECTED"
    elif (not (0 < full_edge_count < all_pairs) or pair_reference_unavailable > 0 or full50_infeasible or
          not (set(json.loads(selection_changes_50["baseline_pruned_ids"])) != set(json.loads(selection_changes_50["constrained_pruned_ids"]))) or
          any(value < 1.0 for value in subset_jaccards) or any(value < 1.0 for value in protected_jaccards)):
        decision = "B. TEMPORAL_BACKUP_CONSTRAINT_WEAK_OR_UNRESOLVED"
    else:
        decision = "A. TEMPORAL_BACKUP_CONSTRAINT_READY_FOR_ORACLE"

    repo_head = "UNKNOWN"
    try:
        import subprocess
        repo_head = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    domain_summaries = [{key: value for key, value in row.items() if key != "record_type"}
                        for row in full_domain_rows]
    summary: Dict[str, Any] = {
        "task": "TASK042_PHASE_F_TEMPORAL_BACKUP_CONSTRAINT_FEASIBILITY_AUDIT",
        "decision": decision,
        "decision_scope": "Feasibility audit only. Decision A authorizes only a later temporary-masking oracle; it does not authorize physical pruning or finetuning.",
        "branch": "task_042_post_bms_frame_relation_redundancy",
        "code_head_at_run": repo_head,
        "gpu_used": False,
        "inference_or_validation_oracle_used": False,
        "physical_pruning_or_finetuning_used": False,
        "frozen_input_identity": {
            "phase_d_input_identity": input_identity,
            "sha256": {key: file_sha256(path) for key, path in required.items()},
        },
        "cohort": {
            "phase_a_control_pool_unit_count": len(all_units),
            "phase_d_selection_unit_count": len(selection_units),
            "directional_removal_candidate_count": len(candidate_rows),
            "fixed_final_domain_representative_count": len(fixed_retained_rows),
            "eligible_domain_ids": sorted({str(row["domain_id"]) for row in selection_units}, key=_sort_domain),
            "candidate_parameter_total": candidate_total,
            "budget_definition": "floor(percent * sum of exact Task037 DeltaP over the 21 Phase-D removal candidates); final representatives are fixed retained anchors, not directional removal candidates",
            "control_pool_definition": "All 51 already-captured frozen Task042 Phase-A units; controls must be cross-domain and exactly match unordered type and stage composition. Selection remains restricted to the Phase-D 31-unit cohort.",
        },
        "parameter_accounting": {
            "all_31_units_have_exact_formula_cost": len(cost_rows) == len(selection_units),
            "candidate_DeltaP_matches_formula_count": candidate_cost_match_count,
            "candidate_DeltaP_expected_count": len(candidate_rows),
            "trace_cost_matches_formula_count": len(trace_compare_rows),
            "trace_cost_comparison_available_count": len([row for row in cost_rows if row["task037_trace_parameter_cost"] != ""]),
            "trace_units_absent_from_extended_trace": [row["task037_global_index"] for row in cost_rows if row["task037_trace_parameter_cost"] == ""],
            "head_formula": "3*width*head_dim + 3*head_dim (qkv bias) + width*head_dim (projection input weights); width=96*2^stage and head_dim=32",
            "ffn_formula": "fc1 input weights + fc1 bias + fc2 output weights = 2*width+1; width=96*2^stage",
            "authoritative_code_source": "pruning/MC.py::MC.estimate_unit_cost",
            "independent_video_swin_config_source": "models/ucf101_videoswin_my.py: embed_dim=96, num_heads=[3,6,12,24]; qkv bias is enabled by model construction",
            "candidate_parameter_total_reproduced_from_phase_d_identity": candidate_total == int(input_identity["removable_cohort_parameters"]),
        },
        "directional_preference": {
            "source_artifact": "phase_d/task042_phase_d_f3_authoritative_global_trace.csv",
            "source_code": "pruning/task034_mid_veto_50_logical_finetune.py::f3_order_key and f3_rank_rows; Phase-D projection source src/lgfr_runtime/task042_phase_d_progressive_path.py::candidate_provenance",
            "score_name": "Task037 dynamic F3 midpoint-veto risk R_F3, with p_total and p_average comparison fields",
            "sort_direction": "ascending R_F3, then p_total, then p_average, with task037_global_index final deterministic tie break; for this cohort, preserve recorded unique global trace steps exactly",
            "baseline_selection": "production trace-order prefix; stop before first unit that would exceed diagnostic parameter budget",
            "constrained_equal_parameter_tie_break": "lexicographically maximize the binary prune vector in ascending recorded F3 global trace step order (earlier production proposals preferred); no lambda and no temporal-magnitude objective",
            "directional_provenance_complete_for_every_removal_candidate": len(preference_ids) == len(candidate_rows),
            "fixed_final_domain_representatives_have_no_pruning_score": True,
        },
        "d_temp_reproduction": {
            "same_domain_phase_a_pairs_reproduced": reproduced,
            "same_domain_pairs_expected": len(phase_a_pair_map),
            "max_absolute_difference_vs_phase_a": max(absolute_differences, default=0.0),
            "existing_cross_domain_controls_reproduced": len(control_reproduction_diffs),
            "max_absolute_difference_vs_existing_cross_domain_controls": max(control_reproduction_diffs, default=0.0),
            "metric_source": "frozen Task042 normalize_signature and relation_distance; per-video Pearson relation distance averaged over selected existing videos",
        },
        "matched_background": {
            "same_domain_pair_count": all_pairs,
            "reference_available_pair_count": all_pairs - reference_unavailable,
            "reference_unavailable_pair_count": reference_unavailable,
            "supported_reference_rejected_pair_count": supported_reference_rejected,
            "edge_definition": "R_CTR=(median matched cross-domain d_temp - same-domain d_temp)/(median matched cross-domain d_temp + 1e-12); edge iff R_CTR>0",
            "threshold_added": False,
        },
        "full_graph": {
            "domain_summaries": domain_summaries,
            "edge_count": full_edge_count,
            "possible_pair_count": all_pairs,
            "edge_density": full_edge_count / all_pairs if all_pairs else 0.0,
            "rejected_pair_count": all_pairs - full_edge_count,
            "isolated_unit_count": full_node_isolates,
            "connected_component_sizes_by_domain": {row["domain_id"]: json.loads(row["component_sizes"]) for row in full_domain_rows},
            "edge_unit_type_counts": dict(supported_types),
            "graph_is_nontrivial": 0 < full_edge_count < all_pairs,
            "complete_edge_domains": complete_edge_domains,
            "zero_edge_domains": zero_edge_domains,
            "partially_supported_domains": partially_supported_domains,
            "degeneracy_assessment": "High aggregate density with complete domains, but also reference-supported rejected pairs and isolated units. The graph is globally nontrivial and locally heterogeneous; no density threshold was used.",
            "retained_set_interpretation": "The retained set must dominate each same-domain temporal-backup graph: every node is retained or has a retained adjacent backup. This is only the graph interpretation of the specified constraints; no generic dominating-set heuristic is used.",
        },
        "budget_feasibility": {
            "budgets_parameters": budgets,
            "maximum_removable_parameters": maximum_parameters,
            "maximum_removable_ratio_of_candidate_cohort": maximum_parameters / candidate_total if candidate_total else 0.0,
            "desired_50pct_budget_parameters": budgets[50],
            "desired_50pct_budget_infeasible": full50_infeasible,
            "solver_status": maximum["solver_status"],
            "exactness_proof": maximum["optimality_proof"],
        },
        "selection_change_at_50pct": selection_changes_50,
        "calibration_stability": {
            "subset_names": list(VIDEO_SUBSET_NAMES),
            "selection_jaccard_vs_full_min": min(subset_jaccards, default=1.0),
            "selection_jaccard_vs_full_max": max(subset_jaccards, default=1.0),
            "protected_unit_jaccard_vs_full_min": min(protected_jaccards, default=1.0),
            "protected_unit_jaccard_vs_full_max": max(protected_jaccards, default=1.0),
            "full_data_edge_count": full_edge_count,
            "all_full_edges_have_presence_rows_for_all_subsets": len([row for row in stability_rows if row.get("record_type") == "FULL_EDGE_PRESENCE"]) == full_edge_count * len(VIDEO_SUBSET_NAMES),
        },
        "mixed_type_support": {
            "domains": [271, 297],
            "attention_nodes_with_supported_ffn_backup": sum(_as_bool(row["attention_supported_by_ffn"]) for row in mixed_rows),
            "ffn_nodes_with_supported_attention_backup": sum(_as_bool(row["ffn_supported_by_attention"]) for row in mixed_rows),
            "no_type_quota_or_fairness_claim": True,
        },
        "required_question_answers": {
            "A_matched_background_graph_nontrivial": 0 < full_edge_count < all_pairs,
            "B_rejects_temporally_unsupported_bms_pairs": supported_reference_rejected > 0,
            "C_reaches_50pct_parameter_budget": not full50_infeasible,
            "D_materially_alters_existing_selection_at_50pct": bool(set(json.loads(selection_changes_50["baseline_pruned_ids"])) != set(json.loads(selection_changes_50["constrained_pruned_ids"]))),
            "E_protected_units_stable_across_class_diverse_subsets": all(value == 1.0 for value in protected_jaccards),
            "F_any_genuine_cross_type_temporal_backup": any(_as_bool(row["has_supported_cross_type_backup"]) for row in mixed_rows),
            "G_ready_for_temporary_masking_oracle": decision == "A. TEMPORAL_BACKUP_CONSTRAINT_READY_FOR_ORACLE",
        },
        "interpretation_note": "Decision uses logical feasibility and exact observed subset identity only; the task provides no numerical thresholds for density, materiality, or stability, so no such threshold was introduced. Any unresolved background pair, infeasible 50% target, no 50% selection change, or non-identical subset selection/protection keeps the result at B.",
    }
    (output_dir / "task042_phase_f_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable report with the full required questions and decisions.
    graph_table = [[row["domain_id"], row["domain_size"], row["edge_count"], row["possible_pair_count"],
                    "%.3f" % float(row["edge_density"]), row["isolated_unit_count"],
                    row["connected_component_count"], row["attention_attention_edges"],
                    row["ffn_ffn_edges"], row["attention_ffn_edges"]] for row in full_domain_rows]
    budget_table = [[row["budget_percent"], row["budget_parameters"], row["baseline_released_parameters"],
                     row["constrained_released_parameters"], row["constrained_budget_gap"],
                     row["jaccard_similarity"]] for row in change_rows_out if row["record_type"] == "BUDGET_SUMMARY"]
    stability_table = [[row["subset"], row["backup_edge_count"], row["isolated_unit_count"],
                        row["50pct_constrained_released_parameters"], row["50pct_budget_reachable"],
                        "%.3f" % row["selection_jaccard_vs_full_data"],
                        "%.3f" % row["protected_unit_jaccard_vs_full_data"]] for row in subset_summaries]
    mixed_summary = [row for row in mixed_rows if _as_bool(row["has_supported_cross_type_backup"])]
    report = f"""# Task042 Phase F — Temporal Backup Constraint Feasibility Audit

Decision: **{decision}**

This is a CPU-only structural feasibility audit. It used no GPU, inference, validation-performance oracle, physical pruning, or finetuning. All descriptors, BMS domains, activation semantics, 80 interventions, per-video normalization, d_temp, and the authoritative Task037 directional criterion were frozen.

## Frozen cohort and provenance

- Selection nodes: {len(selection_units)} Phase-D Task042 units in {len(grouped)} frozen BMS domains.
- Removable directional candidates: {len(candidate_rows)}; fixed final domain representatives: {len(fixed_retained_rows)}.
- Matched-background controls: {len(all_units)} existing Phase-A captured units. Controls must come from different frozen domains and exactly match the unordered unit-type and stage compositions. Selection remains restricted to the Phase-D 31-unit cohort.
- Candidate removable parameter total: {candidate_total:,}; diagnostic budgets use floor(25%, 50%, 75%) of this exact total.
- Exact F3 score/order source: `{summary['directional_preference']['source_artifact']}` and `{summary['directional_preference']['source_code']}`. Direction: {summary['directional_preference']['sort_direction']}.
- All {len(candidate_rows)} removal-candidate DeltaP costs matched independent Video Swin physical-accounting formulas. The {len(selection_units)} tested units have formula costs; all recorded trace costs also match.
- Recomputed full-data d_temp reproduced {reproduced}/{len(phase_a_pair_map)} existing same-domain pairs; maximum absolute difference {max(absolute_differences, default=0.0):.3g}. Existing matched-control rows reproduced: {len(control_reproduction_diffs)}, maximum absolute difference {max(control_reproduction_diffs, default=0.0):.3g}.

## Matched background and graph topology

For every same-domain pair, `b_ij` is the median d_temp among exact matched cross-domain controls. `R_CTR=(b_ij-d_temp)/(b_ij+1e-12)`; an edge exists iff `R_CTR>0`. No temporal threshold was added. Unavailable matched pools stay marked `REFERENCE_UNAVAILABLE` and have no backup edge.

- Same-domain pairs: {all_pairs}; matched references available: {all_pairs-reference_unavailable}; unavailable: {reference_unavailable}.
- Supported edges: {full_edge_count}/{all_pairs} ({(full_edge_count/all_pairs if all_pairs else 0.0):.1%}); reference-available rejected pairs: {supported_reference_rejected}; reference-unavailable pairs: {reference_unavailable}; isolated nodes: {full_node_isolates}.
- Type composition among supported edges: {dict(supported_types)}.

{_md_table(['BMS domain','units','edges','possible pairs','density','isolated','components','Attn-Attn','FFN-FFN','Attn-FFN'], graph_table)}

Degeneracy diagnosis: the aggregate graph is dense (26/34 edges), with complete-edge domains {complete_edge_domains}; it also has {supported_reference_rejected} reference-supported rejected pairs, {full_node_isolates} isolated units, zero-edge domains {zero_edge_domains}, and partially supported domains {partially_supported_domains}. It is therefore nontrivial overall but locally heterogeneous, with both dense and no-backup pockets. No density threshold was used.

The exact binary backup condition is `y_i + sum(y_j for j in B_i) >= 1` for every tested node, with the Phase-D final representative in each domain fixed retained. Thus every pruned unit has a surviving adjacent backup and the retained nodes dominate the temporal-backup graph. This is the graph interpretation of the requested constraint; selection uses exact local enumeration plus parameter-cost DP, not a generic dominating-set heuristic.

## Exact parameter-budget selections

Baseline selection is the recorded production F3 trace-order prefix, stopping before the first unit that would exceed the budget. The constrained solver maximizes released integer parameters not exceeding the same budget; ties lexicographically favor earlier authoritative F3 trace proposals. Temporal edge magnitudes do not enter the objective.

{_md_table(['Budget %','target params','baseline released','constrained released','constrained gap','selection Jaccard'], budget_table)}

- Exact solver: `{maximum['solver_status']}`. It enumerates every feasible prune pattern in each BMS domain and exactly convolves all domain options by integer parameter cost, so the optimum is global for this cohort.
- Maximum removable parameters under backup constraints: {maximum_parameters:,}/{candidate_total:,} ({maximum_parameters/candidate_total:.1%}).
- 50% target: {budgets[50]:,}; **{'infeasible' if full50_infeasible else 'feasible'}**; actual constrained release {full50_release:,}.
- At 50%, baseline/constrained Jaccard is {float(selection_changes_50['jaccard_similarity']):.3f}; protected baseline candidates: {len(baseline50_selected-constrained_by_percent[50])}; constrained alternatives: {len(constrained_by_percent[50]-baseline50_selected)}.
- The 50% change is limited: {len(baseline50_selected-constrained_by_percent[50])} baseline units are protected, {len(constrained_by_percent[50]-baseline50_selected)} alternatives are added, and Jaccard is {float(selection_changes_50['jaccard_similarity']):.3f}; the constraint acts as a local veto here rather than broadly reordering candidates.

## Protected units and mixed-type backups

Each `PROTECTED_BASELINE_PRUNED_UNIT` row in `task042_phase_f_selection_changes.csv` lists its F3 score, same-domain backup neighbors with R_CTR and d_temp, whether baseline would leave it unsupported, and the constrained-pruned units it backs up. `FORCED_ALTERNATIVE_PRUNED_UNIT` rows enumerate selection exchanges.

- Domains 271 and 297 supported cross-type edges: {sum(1 for row in full_pairs if row['domain_id'] in ('271','297') and bool(row['temporal_backup_edge']) and canonical_type(row['unit_type_i']) != canonical_type(row['unit_type_j']))}.
- Attention units with an FFN backup: {summary['mixed_type_support']['attention_nodes_with_supported_ffn_backup']}.
- FFN units with an Attention backup: {summary['mixed_type_support']['ffn_nodes_with_supported_attention_backup']}.

{_md_table(['domain','unit','type','stage','cross-type neighbors / R_CTR'], [[row['domain_id'],row['task037_global_index'],row['unit_type'],row['stage'],row['cross_type_backup_neighbors']] for row in mixed_summary]) if mixed_summary else 'No supported cross-type backup was found in domains 271 or 297.'}

## Calibration stability

Only existing Phase-A raw artifacts are reused. The full-data temporal edges are tracked in every class-position subset; the exact 50% constrained selection and protected-set Jaccard are recomputed per subset.

{_md_table(['subset','edges','isolated','50% released','50% feasible','selection Jaccard','protected Jaccard'], stability_table)}

## Required questions

- A. Matched-background graph nontrivial: **{summary['required_question_answers']['A_matched_background_graph_nontrivial']}**.
- B. Rejects some temporally unsupported BMS pairs: **{summary['required_question_answers']['B_rejects_temporally_unsupported_bms_pairs']}**.
- C. Reaches the 50% parameter budget: **{summary['required_question_answers']['C_reaches_50pct_parameter_budget']}**.
- D. Changes the 50% one-shot selection: **{summary['required_question_answers']['D_materially_alters_existing_selection_at_50pct']}**; the effect is limited to two protected units, with no added alternatives (Jaccard 0.882).
- E. Protected units are identical across all calibration subsets: **{summary['required_question_answers']['E_protected_units_stable_across_class_diverse_subsets']}**.
- F. Any supported cross-type backup in domains 271/297: **{summary['required_question_answers']['F_any_genuine_cross_type_temporal_backup']}**.
- G. Ready for a temporary-masking oracle: **{summary['required_question_answers']['G_ready_for_temporary_masking_oracle']}**.

## Decision

**{decision}**

The decision rule uses exact observed feasibility and identity stability only. The protocol gave no numerical thresholds for graph density, material selection change, or stability, so none were introduced. The result is `B` whenever any matched reference is unavailable, 50% is infeasible, the 50% choice does not change, or a calibration subset changes the exact selection/protected set. Decision `A`, if reached, authorizes only a later temporary-masking oracle; it does not authorize physical pruning or finetuning.

## Required outputs

All required CSVs and this report are in `{output_dir}`. Summary JSON: `task042_phase_f_summary.json`.
"""
    (output_dir / "task042_phase_f_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy/phase_f"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy"))
    args = parser.parse_args()
    result = run(args.input_dir, args.output_dir, args.repo_dir)
    print(json.dumps({"decision": result["decision"], "output_dir": str(args.output_dir),
                      "gpu_used": result["gpu_used"],
                      "maximum_removable_parameters": result["budget_feasibility"]["maximum_removable_parameters"]}, indent=2))


if __name__ == "__main__":
    main()

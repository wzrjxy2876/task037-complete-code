from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "src" / "lgfr_runtime" / "task042_phase_f_backup_constraint.py").is_file():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lgfr_runtime.task042_phase_f_backup_constraint import (  # noqa: E402
        baseline_prefix,
        build_matched_background,
        calibration_subsets,
        graph_topology,
        jaccard,
        pair_key,
        physical_parameter_cost,
        solve_exact_constrained,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task042_phase_f_backup_constraint import (  # noqa: E402
    baseline_prefix,
    build_matched_background,
    calibration_subsets,
    graph_topology,
    jaccard,
    pair_key,
    physical_parameter_cost,
    solve_exact_constrained,
    )


class FrozenCore:
    @staticmethod
    def normalize_signature(values):
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        std = var ** 0.5
        if std <= 1e-12:
            return None, mean, std, True
        return [(value - mean) / std for value in values], mean, std, False

    @staticmethod
    def relation_distance(xs, ys):
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        dx, dy = [x - mx for x in xs], [y - my for y in ys]
        vx, vy = sum(x * x for x in dx), sum(y * y for y in dy)
        if vx <= 0 or vy <= 0:
            return None
        rho = sum(x * y for x, y in zip(dx, dy)) / (vx * vy) ** 0.5
        return (1 - max(-1.0, min(1.0, rho))) / 2


def unit(uid, domain, kind="ffn_neuron", stage="0"):
    return {"task037_global_index": uid, "domain_id": str(domain),
            "unit_type": kind, "stage": str(stage), "layer": "layer",
            "unit_index": uid}


def candidate(uid, domain, cost, step):
    return {**unit(uid, domain), "selected_for_removal": True,
            "DeltaP": str(cost), "f3_global_step": str(step),
            "R_F3": "0.%d" % step, "p_total": "0.1", "p_average": "0.2"}


def test_matched_background_uses_exact_unordered_type_stage_and_cross_domain_pool():
    pair_units = [unit(1, 10, "ffn_neuron", "0"), unit(2, 10, "ffn_neuron", "1")]
    controls = pair_units + [unit(3, 20, "neuron", "0"), unit(4, 30, "ffn_neuron", "1"),
                             unit(5, 20, "ffn_neuron", "0"), unit(6, 20, "ffn_neuron", "2")]
    # Exact unordered match candidates are (1,4), (2,3), and (3,4); the latter
    # uses two different domains and the unit-type/stage multisets match.
    distances = {pair_key(1, 4): 0.1, pair_key(2, 3): 0.3, pair_key(3, 4): 0.5,
                 pair_key(1, 2): 0.2, pair_key(1, 3): 0.9, pair_key(2, 4): 0.9}
    matched, rows = build_matched_background(pair_units, controls, distances)
    assert rows[0]["reference_status"] == "REFERENCE_AVAILABLE"
    assert rows[0]["matched_control_count"] == 3
    assert rows[0]["b_ij_median"] == 0.3
    assert rows[0]["R_CTR"] > 0
    assert rows[0]["temporal_backup_edge"] is True
    assert matched[0]["reference_status"] == "REFERENCE_AVAILABLE"


def test_reference_unavailable_has_no_fallback():
    pair_units = [unit(1, 10, "ffn_neuron", "0"), unit(2, 10, "attention_head", "2")]
    controls = [unit(3, 20, "ffn_neuron", "0"), unit(4, 30, "ffn_neuron", "0")]
    rows, pairs = build_matched_background(pair_units, controls, {pair_key(1, 2): 0.1})
    assert pairs[0]["reference_status"] == "REFERENCE_UNAVAILABLE"
    assert pairs[0]["b_ij_median"] == ""
    assert pairs[0]["R_CTR"] == ""
    assert pairs[0]["temporal_backup_edge"] is False


def test_r_ctr_arithmetic_edge_symmetry_and_isolated_node_behavior():
    units = [unit(1, 10), unit(2, 10), unit(3, 10)]
    controls = [unit(1, 10), unit(2, 10), unit(3, 10), unit(4, 20), unit(5, 30)]
    distances = {pair_key(1, 2): 0.2, pair_key(1, 4): 0.4, pair_key(2, 5): 0.3}
    _matched, pairs = build_matched_background(units, controls, distances)
    first = pairs[0]
    assert abs(float(first["R_CTR"]) - (0.35 - 0.2) / (0.35 + 1e-12)) < 1e-14
    assert first["temporal_backup_edge"] is True
    summaries, nodes, _edge_rows, adjacency = graph_topology(units, pairs)
    assert adjacency[1] == {2} and adjacency[2] == {1} and adjacency[3] == set()
    by_id = {row["task037_global_index"]: row for row in nodes}
    assert by_id[3]["is_isolated"] is True
    assert by_id[3]["connected_component_size"] == 1
    assert summaries[0]["isolated_unit_count"] == 1


def test_physical_parameter_accounting_head_and_ffn():
    assert physical_parameter_cost(unit(1, 1, "ffn_neuron", "0")) == 193
    assert physical_parameter_cost(unit(2, 1, "ffn_neuron", "3")) == 1537
    assert physical_parameter_cost(unit(3, 1, "attention_head", "0")) == 12384
    assert physical_parameter_cost(unit(4, 1, "attention_head", "3")) == 98400


def _is_feasible(pruned, candidates, adjacency):
    all_ids = set(adjacency)
    retained = all_ids - set(pruned)
    return all(bool(adjacency[uid] & retained) for uid in set(pruned) & set(candidates))


def test_backup_constraint_exact_objective_and_optimality_against_bruteforce():
    all_ids = {1, 2, 3, 4, 5, 6}
    # Two independent BMS domains. Unit 3 and 6 are fixed representatives.
    adjacency = {1: {2}, 2: {1, 3}, 3: {2}, 4: {5}, 5: {4, 6}, 6: {5}}
    candidates = [candidate(1, 10, 7, 1), candidate(2, 10, 8, 2),
                  candidate(4, 20, 5, 3), candidate(5, 20, 6, 4)]
    budget = 14
    result = solve_exact_constrained(candidates, all_ids, adjacency, budget)
    ids = [1, 2, 4, 5]
    best_cost = max(sum(int(row["DeltaP"]) for row in candidates if row["task037_global_index"] in subset)
                    for r in range(len(ids) + 1)
                    for subset in itertools.combinations(ids, r)
                    if sum(int(row["DeltaP"]) for row in candidates if row["task037_global_index"] in subset) <= budget
                    and _is_feasible(set(subset), set(ids), adjacency))
    assert result["released_parameters"] == best_cost
    assert result["released_parameters"] <= budget
    assert result["retained_set_is_dominating"] is True
    assert result["solver_status"] == "OPTIMAL_EXACT_DOMAIN_ENUMERATION_DP"
    assert _is_feasible(result["selected_pruned_ids"], set(ids), adjacency)


def test_exact_solver_is_deterministic_and_production_tie_preference_is_preserved():
    adjacency = {1: {2}, 2: {1, 3}, 3: {2}}
    candidates = [candidate(1, 10, 5, 1), candidate(2, 10, 5, 2)]
    first = solve_exact_constrained(candidates, {1, 2, 3}, adjacency, 5)
    second = solve_exact_constrained(candidates, {1, 2, 3}, adjacency, 5)
    assert first["selected_pruned_ids"] == second["selected_pruned_ids"] == {1}
    assert first["tie_preference_mask"] == second["tie_preference_mask"]


def test_baseline_is_exact_trace_order_prefix_and_does_not_skip_oversized_unit():
    rows = [candidate(1, 10, 4, 1), candidate(2, 10, 6, 2), candidate(3, 20, 3, 3)]
    assert baseline_prefix(rows, 10) == {1, 2}
    assert baseline_prefix(rows, 12) == {1, 2}
    assert baseline_prefix(rows, 9) == {1}


def test_calibration_subset_identity_matches_class_position_partition():
    videos = []
    for label in range(10):
        for pos in range(3):
            idx = label * 3 + pos
            videos.append({"video_index": idx, "dataset_index": idx, "label": label})
    subsets = calibration_subsets(videos)
    assert list(subsets) == ["A_position1", "B_position2", "C_position3",
                             "AB_positions12", "AC_positions13", "BC_positions23", "full_10x3"]
    assert subsets["A_position1"] == list(range(0, 30, 3))
    assert subsets["B_position2"] == list(range(1, 30, 3))
    assert subsets["C_position3"] == list(range(2, 30, 3))
    assert len(subsets["AB_positions12"]) == len(subsets["AC_positions13"]) == len(subsets["BC_positions23"]) == 20
    assert subsets["full_10x3"] == list(range(30))


def test_jaccard_empty_and_nonempty_sets():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3

"""Offline helpers for the Task042 Phase F temporal-backup feasibility audit.

The module only consumes frozen Task042 activation artifacts and Phase-D
selection provenance. It never loads a model, performs inference, changes a
mask, or evaluates validation performance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Edge = Tuple[int, int]
EPS = 1e-12
VIDEO_SUBSET_NAMES = (
    "A_position1", "B_position2", "C_position3", "AB_positions12",
    "AC_positions13", "BC_positions23", "full_10x3",
)


def pair_key(i: Any, j: Any) -> Edge:
    a, b = int(i), int(j)
    if a == b:
        raise ValueError("a pair must contain two distinct units")
    return (min(a, b), max(a, b))


def canonical_type(value: Any) -> str:
    value = str(value).strip()
    aliases = {"head": "attention_head", "attention_head": "attention_head",
               "neuron": "ffn_neuron", "ffn_neuron": "ffn_neuron"}
    if value not in aliases:
        raise ValueError("unsupported Task037 unit type: %r" % value)
    return aliases[value]


def _stage_key(value: Any) -> Tuple[int, str]:
    text = str(value).strip()
    try:
        return (0, "%08d" % int(text))
    except ValueError:
        return (1, text)


def structural_composition(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, str], ...]]:
    types = tuple(sorted((canonical_type(a["unit_type"]), canonical_type(b["unit_type"]))))
    stages = tuple(sorted((_stage_key(a["stage"]), _stage_key(b["stage"]))))
    return types, stages


def calibration_subsets(video_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    by_class: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for row in video_rows:
        by_class[int(row["label"])].append((int(row["video_index"]), int(row["dataset_index"])))
    if len(by_class) != 10 or any(len(rows) != 3 for rows in by_class.values()):
        raise ValueError("calibration cohort must be exactly 10 classes x 3 videos")
    positions: Dict[int, List[int]] = {1: [], 2: [], 3: []}
    for rows in by_class.values():
        rows.sort()
        for pos, (video_index, _dataset_index) in enumerate(rows, 1):
            positions[pos].append(video_index)
    all_ids = sorted(int(row["video_index"]) for row in video_rows)
    result = {
        "A_position1": sorted(positions[1]),
        "B_position2": sorted(positions[2]),
        "C_position3": sorted(positions[3]),
        "AB_positions12": sorted(positions[1] + positions[2]),
        "AC_positions13": sorted(positions[1] + positions[3]),
        "BC_positions23": sorted(positions[2] + positions[3]),
        "full_10x3": all_ids,
    }
    if tuple(result) != VIDEO_SUBSET_NAMES:
        raise AssertionError("calibration subset identity order changed")
    return result


def compute_temporal_distances(
    raw_rows: Sequence[Mapping[str, Any]],
    all_units: Sequence[Mapping[str, Any]],
    video_indices: Sequence[int],
    task042_core: Any,
) -> Tuple[Dict[Edge, Optional[float]], Dict[Edge, int]]:
    """Recompute frozen d_temp from the recorded 80 Phase-A interventions."""
    video_set = set(map(int, video_indices))
    ids = sorted(int(row["task037_global_index"]) for row in all_units)
    conditions: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)
    for row in raw_rows:
        vi = int(row["video_index"])
        uid = int(row["task037_global_index"])
        if vi in video_set and uid in set(ids):
            conditions[(vi, uid)].append((int(row["span"]), int(row["pair_index"]),
                                          float(row["relative_sensitivity"])))
    expected_identity = [(span, pair) for span in (1, 2, 4, 8, 16) for pair in range(16)]
    signatures: Dict[Tuple[int, int], Optional[List[float]]] = {}
    for vi in sorted(video_set):
        for uid in ids:
            items = sorted(conditions.get((vi, uid), []), key=lambda item: (item[0], item[1]))
            if [(span, pair) for span, pair, _ in items] != expected_identity:
                raise ValueError("unit/video does not contain the exact 80 frozen Phase-A interventions: %s/%s" % (vi, uid))
            z, _mean, _std, degenerate = task042_core.normalize_signature([value for _, _, value in items])
            signatures[(vi, uid)] = None if degenerate else z
    distances: Dict[Edge, Optional[float]] = {}
    valid_counts: Dict[Edge, int] = {}
    for i, j in combinations(ids, 2):
        values: List[float] = []
        for vi in sorted(video_set):
            zi, zj = signatures[(vi, i)], signatures[(vi, j)]
            d = None if zi is None or zj is None else task042_core.relation_distance(zi, zj)
            if d is not None:
                values.append(float(d))
        edge = (i, j)
        distances[edge] = sum(values) / len(values) if values else None
        valid_counts[edge] = len(values)
    return distances, valid_counts


def build_matched_background(
    selection_units: Sequence[Mapping[str, Any]],
    control_units: Sequence[Mapping[str, Any]],
    distances: Mapping[Edge, Optional[float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build exact unordered type/stage matched cross-domain pools per BMS pair."""
    units = {int(row["task037_global_index"]): dict(row) for row in selection_units}
    controls = {int(row["task037_global_index"]): dict(row) for row in control_units}
    matched_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    groups: Dict[str, List[int]] = defaultdict(list)
    for uid, row in units.items():
        groups[str(row["domain_id"])].append(uid)
    for domain in sorted(groups, key=lambda value: (int(value) if value.isdigit() else value)):
        ids = sorted(groups[domain])
        for i, j in combinations(ids, 2):
            target = structural_composition(units[i], units[j])
            pool = []
            for a, b in combinations(sorted(controls), 2):
                ca, cb = controls[a], controls[b]
                if str(ca["domain_id"]) == str(cb["domain_id"]):
                    continue
                if structural_composition(ca, cb) != target:
                    continue
                d = distances.get((a, b))
                if d is None:
                    continue
                pool.append({"control_i": a, "control_j": b, "domain_i": str(ca["domain_id"]),
                             "domain_j": str(cb["domain_id"]), "d_temp": float(d)})
            if pool:
                background = float(statistics.median(row["d_temp"] for row in pool))
                status = "REFERENCE_AVAILABLE"
                d_same = distances.get((i, j))
                r_ctr = None if d_same is None else (background - float(d_same)) / (background + EPS)
                is_edge = bool(r_ctr is not None and r_ctr > 0.0)
            else:
                background, status, d_same, r_ctr, is_edge = None, "REFERENCE_UNAVAILABLE", distances.get((i, j)), None, False
            record = {
                "domain_id": domain, "task037_global_index_i": i, "task037_global_index_j": j,
                "unit_type_i": canonical_type(units[i]["unit_type"]), "unit_type_j": canonical_type(units[j]["unit_type"]),
                "stage_i": str(units[i]["stage"]), "stage_j": str(units[j]["stage"]),
                "unordered_unit_type_composition": json.dumps(target[0]),
                "unordered_stage_composition": json.dumps([value[1].lstrip("0") or "0" for value in target[1]]),
                "matched_control_count": len(pool), "reference_status": status,
                "d_temp_same_domain": "" if d_same is None else float(d_same),
                "b_ij_median": "" if background is None else background,
                "R_CTR": "" if r_ctr is None else r_ctr,
                "temporal_backup_edge": is_edge,
                "matched_control_pairs": json.dumps(pool, sort_keys=True, separators=(",", ":")),
            }
            pair_rows.append(record)
            matched_rows.append({key: record[key] for key in (
                "domain_id", "task037_global_index_i", "task037_global_index_j",
                "unordered_unit_type_composition", "unordered_stage_composition",
                "reference_status", "matched_control_count", "b_ij_median", "matched_control_pairs")})
    return matched_rows, pair_rows


def graph_topology(
    selection_units: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Set[int]]]:
    """Return domain, node, pair diagnostics and symmetric adjacency."""
    units = {int(row["task037_global_index"]): dict(row) for row in selection_units}
    groups: Dict[str, List[int]] = defaultdict(list)
    for uid, row in units.items():
        groups[str(row["domain_id"])].append(uid)
    adjacency: Dict[int, Set[int]] = {uid: set() for uid in units}
    for row in pair_rows:
        if bool(row["temporal_backup_edge"]):
            i, j = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
            adjacency[i].add(j)
            adjacency[j].add(i)
    node_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for domain in sorted(groups, key=lambda value: (int(value) if value.isdigit() else value)):
        ids = sorted(groups[domain])
        remaining = set(ids)
        components: List[List[int]] = []
        while remaining:
            root = min(remaining)
            stack, seen = [root], set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(sorted(adjacency[current] & remaining, reverse=True))
            remaining -= seen
            components.append(sorted(seen))
        components.sort(key=lambda component: component[0])
        component_by_id = {uid: (idx, len(component)) for idx, component in enumerate(components, 1) for uid in component}
        domain_pairs = [row for row in pair_rows if str(row["domain_id"]) == domain]
        possible = len(domain_pairs)
        edges = [row for row in domain_pairs if bool(row["temporal_backup_edge"])]
        rejected = [row for row in domain_pairs if not bool(row["temporal_backup_edge"])]
        unavailable = [row for row in domain_pairs if row["reference_status"] == "REFERENCE_UNAVAILABLE"]
        reference_rejected = [row for row in rejected if row["reference_status"] == "REFERENCE_AVAILABLE"]
        type_counts = {"Attention-Attention": 0, "FFN-FFN": 0, "Attention-FFN": 0}
        for row in edges:
            pair_types = {canonical_type(row["unit_type_i"]), canonical_type(row["unit_type_j"])}
            if pair_types == {"attention_head"}:
                type_counts["Attention-Attention"] += 1
            elif pair_types == {"ffn_neuron"}:
                type_counts["FFN-FFN"] += 1
            else:
                type_counts["Attention-FFN"] += 1
        isolates = [uid for uid in ids if not adjacency[uid]]
        summary_rows.append({
            "record_type": "DOMAIN_SUMMARY", "domain_id": domain, "domain_size": len(ids),
            "edge_count": len(edges), "possible_pair_count": possible,
            "edge_density": len(edges) / possible if possible else 0.0,
            "rejected_pair_count": len(rejected), "reference_unavailable_pair_count": len(unavailable),
            "supported_reference_rejected_pair_count": len(reference_rejected),
            "isolated_unit_count": len(isolates), "isolated_task037_global_indices": json.dumps(isolates),
            "connected_component_count": len(components),
            "component_sizes": json.dumps([len(component) for component in components]),
            "attention_attention_edges": type_counts["Attention-Attention"],
            "ffn_ffn_edges": type_counts["FFN-FFN"],
            "attention_ffn_edges": type_counts["Attention-FFN"],
            "degeneracy_diagnostic": ("NO_POSSIBLE_PAIRS" if not possible else
                "ALL_PAIRS_SUPPORTED" if len(edges) == possible else
                "NO_PAIRS_SUPPORTED" if len(edges) == 0 else "SUPPORTED_AND_REJECTED_PAIRS"),
        })
        for uid in ids:
            comp_idx, comp_size = component_by_id[uid]
            node_rows.append({
                "record_type": "NODE", "domain_id": domain, "task037_global_index": uid,
                "unit_type": canonical_type(units[uid]["unit_type"]), "stage": str(units[uid]["stage"]),
                "degree": len(adjacency[uid]), "is_isolated": not adjacency[uid],
                "connected_component_id": comp_idx, "connected_component_size": comp_size,
                "temporal_backup_neighbor_ids": json.dumps(sorted(adjacency[uid])),
            })
    edge_rows = []
    for row in pair_rows:
        edge_rows.append({"record_type": "PAIR", **dict(row)})
    return summary_rows, node_rows, edge_rows, adjacency


def physical_parameter_cost(unit: Mapping[str, Any]) -> int:
    """Exact Task037 MC.estimate_unit_cost for Video Swin Base, embed_dim=96."""
    stage = int(unit["stage"])
    if stage < 0 or stage > 3:
        raise ValueError("unexpected Video Swin stage: %s" % stage)
    width = 96 * (2 ** stage)
    kind = canonical_type(unit["unit_type"])
    if kind == "ffn_neuron":
        return width + 1 + width  # fc1 input weights + fc1 bias + fc2 output weights
    head_dim = 32  # VideoSwin embed_dim=96, heads=(3,6,12,24): width/heads=32
    return 3 * width * head_dim + 3 * head_dim + width * head_dim  # qkv weight+bias, projection input weights


def preference_order(candidate_rows: Sequence[Mapping[str, Any]]) -> List[int]:
    rows = list(candidate_rows)
    if not rows:
        return []
    for row in rows:
        for field in ("f3_global_step", "R_F3", "p_total", "p_average", "DeltaP"):
            if row.get(field) in (None, ""):
                raise ValueError("authoritative directional provenance is incomplete for %s" % row.get("task037_global_index"))
    rows.sort(key=lambda row: (int(row["f3_global_step"]), int(row["task037_global_index"])))
    if len({int(row["f3_global_step"]) for row in rows}) != len(rows):
        raise ValueError("F3 global steps must uniquely order tested removal candidates")
    return [int(row["task037_global_index"]) for row in rows]


def baseline_prefix(candidate_rows: Sequence[Mapping[str, Any]], budget: int) -> Set[int]:
    """Exact no-constraint production order prefix that fits the parameter budget."""
    order = preference_order(candidate_rows)
    by_id = {int(row["task037_global_index"]): row for row in candidate_rows}
    selected: Set[int] = set()
    released = 0
    for uid in order:
        cost = int(float(by_id[uid]["DeltaP"]))
        if released + cost > int(budget):
            break
        selected.add(uid)
        released += cost
    return selected


def _preference_mask(selected: Iterable[int], preference_ids: Sequence[int]) -> int:
    chosen = set(map(int, selected))
    n = len(preference_ids)
    return sum(1 << (n - 1 - idx) for idx, uid in enumerate(preference_ids) if uid in chosen)


def solve_exact_constrained(
    candidate_rows: Sequence[Mapping[str, Any]],
    all_unit_ids: Iterable[int],
    adjacency: Mapping[int, Set[int]],
    budget: int,
) -> Dict[str, Any]:
    """Exact per-domain option enumeration plus parameter-cost dynamic programming.

    Each BMS domain has at most four tested units here, so all local pruning
    patterns are enumerated. Domain-local constraints make convolution over
    exact parameter sums exhaustive. Equal-sum states keep the lexicographically
    strongest production preference bit vector (earlier F3 trace entries first).
    """
    ids = set(map(int, all_unit_ids))
    candidates = {int(row["task037_global_index"]): row for row in candidate_rows}
    if not set(candidates).issubset(ids) or ids != set(adjacency):
        raise ValueError("candidate, node, and graph identities do not align")
    if int(budget) < 0:
        raise ValueError("budget must be nonnegative")
    preference_ids = preference_order(candidate_rows)
    groups: Dict[str, List[int]] = defaultdict(list)
    row_by_id = {int(row["task037_global_index"]): row for row in candidate_rows}
    # Candidate rows contain domain identity; non-candidate representatives stay retained.
    for uid, row in row_by_id.items():
        groups[str(row["domain_id"])].append(uid)
    domain_options: List[List[Tuple[int, int, Tuple[int, ...]]]] = []
    for domain in sorted(groups, key=lambda value: (int(value) if value.isdigit() else value)):
        local = sorted(groups[domain])
        options: List[Tuple[int, int, Tuple[int, ...]]] = []
        for bits in range(1 << len(local)):
            removed = tuple(local[pos] for pos in range(len(local)) if bits & (1 << pos))
            removed_set = set(removed)
            feasible = all(any(neighbor not in removed_set for neighbor in adjacency[uid])
                           for uid in removed)
            if not feasible:
                continue
            cost = sum(int(float(row_by_id[uid]["DeltaP"])) for uid in removed)
            mask = _preference_mask(removed, preference_ids)
            options.append((cost, mask, removed))
        domain_options.append(options)
    states: Dict[int, int] = {0: 0}
    for options in domain_options:
        next_states: Dict[int, int] = {}
        for prior_cost, prior_mask in states.items():
            for local_cost, local_mask, _removed in options:
                total = prior_cost + local_cost
                if total > int(budget):
                    continue
                mask = prior_mask | local_mask
                if mask > next_states.get(total, -1):
                    next_states[total] = mask
        states = next_states
    if not states:
        raise AssertionError("the all-retained zero-cost solution must remain feasible")
    best_cost = max(states)
    best_mask = states[best_cost]
    selected = {uid for idx, uid in enumerate(preference_ids)
                if best_mask & (1 << (len(preference_ids) - 1 - idx))}
    retained = ids - selected
    violations = [uid for uid in selected if not (adjacency[uid] & retained)]
    if violations:
        raise AssertionError("exact solver emitted units without temporal backups: %s" % violations)
    dominating = all(uid in retained or bool(adjacency[uid] & retained) for uid in ids)
    if not dominating:
        raise AssertionError("retained set is not a dominating set of the backup graph")
    return {
        "selected_pruned_ids": selected,
        "released_parameters": best_cost,
        "budget": int(budget),
        "budget_gap": int(budget) - best_cost,
        "preference_order": preference_ids,
        "tie_preference_mask": best_mask,
        "solver_status": "OPTIMAL_EXACT_DOMAIN_ENUMERATION_DP",
        "optimality_proof": "Enumerates every feasible prune subset in each BMS domain, then exact-DP convolves all domain cost options up to budget; no cross-domain backup edge exists by construction.",
        "retained_set_is_dominating": dominating,
    }


def solve_maximum_constrained(
    candidate_rows: Sequence[Mapping[str, Any]], all_unit_ids: Iterable[int],
    adjacency: Mapping[int, Set[int]],
) -> Dict[str, Any]:
    total = sum(int(float(row["DeltaP"])) for row in candidate_rows)
    return solve_exact_constrained(candidate_rows, all_unit_ids, adjacency, total)


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    left, right = set(map(int, a)), set(map(int, b))
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def connected_neighbors_json(uid: int, adjacency: Mapping[int, Set[int]], pair_rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {}
    for row in pair_rows:
        i, j = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
        if uid in (i, j) and bool(row["temporal_backup_edge"]):
            other = j if uid == i else i
            lookup[other] = {"R_CTR": float(row["R_CTR"]), "d_temp": float(row["d_temp_same_domain"]),
                             "b_ij": float(row["b_ij_median"])}
    return json.dumps({str(key): lookup[key] for key in sorted(adjacency.get(uid, set()))},
                      sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (list, tuple, set, dict)) else value
                             for key, value in row.items()})

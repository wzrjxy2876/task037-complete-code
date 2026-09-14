"""Pure structural helpers for Task042 Phase E.2.

This module deliberately reuses the frozen Task042 relation metric and contains
no model, loss, or pruning-score implementation of its own.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


Edge = Tuple[int, int]


def pair_key(i: int, j: int) -> Edge:
    i, j = int(i), int(j)
    if i == j:
        raise ValueError("self-pairs are not temporal redundancy edges")
    return (min(i, j), max(i, j))


def validate_identity_partition(all_ids: Iterable[int], candidate_ids: Iterable[int], survivor_ids: Iterable[int]) -> bool:
    all_set, candidate_set, survivor_set = set(map(int, all_ids)), set(map(int, candidate_ids)), set(map(int, survivor_ids))
    if not candidate_set or not survivor_set or candidate_set & survivor_set or candidate_set | survivor_set != all_set:
        raise ValueError("candidate/survivor identities are not an exact disjoint partition")
    return True


def score_available(row: Mapping[str, Any]) -> bool:
    return all(row.get(k) not in (None, "") for k in ("R_F3", "p_total", "p_average"))


def choose_directional_candidate(
    pair: Edge,
    provenance: Mapping[int, Mapping[str, Any]],
    order_key: Callable[[Mapping[str, Any]], Any],
) -> Tuple[int, int, str]:
    """Use only frozen Phase-D F3 direction; temporal distances are not accepted."""
    i, j = pair_key(*pair)
    a, b = provenance[int(i)], provenance[int(j)]
    available = [uid for uid, row in ((i, a), (j, b)) if score_available(row)]
    if not available:
        raise ValueError("reciprocal pair has no authoritative F3 direction")
    if len(available) == 1:
        candidate = available[0]
        rule = "only_pair_member_with_recorded_authoritative_F3_score"
    else:
        # Adapt provenance identity to the canonical Task037 function's field name.
        keyed = []
        for uid in (i, j):
            row = dict(provenance[uid])
            row["global_index"] = uid
            keyed.append((order_key(row), uid))
        candidate = min(keyed, key=lambda x: x[0])[1]
        rule = "canonical_Task037_F3_order_key_lower_first"
    peer = j if candidate == i else i
    return candidate, peer, rule


def directed_nearest_neighbors(
    units: Sequence[Mapping[str, Any]],
    distances: Mapping[Edge, Optional[float]],
) -> List[Dict[str, Any]]:
    """Same-domain nearest neighbors; exact distance ties break by global ID."""
    groups: Dict[str, List[int]] = defaultdict(list)
    unit_by_id = {int(r["task037_global_index"]): r for r in units}
    for uid, row in unit_by_id.items():
        groups[str(row["domain_id"])].append(uid)
    out: List[Dict[str, Any]] = []
    for domain in sorted(groups, key=lambda x: (int(x) if x.isdigit() else x)):
        ids = sorted(groups[domain])
        for uid in ids:
            choices = []
            for other in ids:
                if other == uid:
                    continue
                value = distances.get(pair_key(uid, other))
                if value is not None and math.isfinite(float(value)):
                    choices.append((float(value), int(other)))
            choices.sort(key=lambda x: (x[0], x[1]))
            if not choices:
                out.append({"domain_id": domain, "task037_global_index": uid,
                            "nn_task037_global_index": "", "nn_distance": "",
                            "second_nn_distance": "", "nn_margin": "",
                            "exact_nearest_tie": False})
                continue
            first = choices[0]
            second = choices[1] if len(choices) > 1 else None
            out.append({
                "domain_id": domain,
                "task037_global_index": uid,
                "nn_task037_global_index": first[1],
                "nn_distance": first[0],
                "second_nn_distance": "" if second is None else second[0],
                "nn_margin": "" if second is None else second[0] - first[0],
                "exact_nearest_tie": bool(second is not None and second[0] == first[0]),
            })
    by_uid = {int(r["task037_global_index"]): r for r in out}
    for row in out:
        uid = int(row["task037_global_index"])
        other = row["nn_task037_global_index"]
        partner = ""
        if other != "" and int(by_uid[int(other)]["nn_task037_global_index"] or -1) == uid:
            partner = int(other)
        row["reciprocal_partner_task037_global_index"] = partner
        row["is_reciprocal_edge"] = partner != ""
    return out


def reciprocal_edges(graph_rows: Sequence[Mapping[str, Any]]) -> Set[Edge]:
    return {pair_key(int(r["task037_global_index"]), int(r["reciprocal_partner_task037_global_index"]))
            for r in graph_rows if r.get("is_reciprocal_edge") and r.get("reciprocal_partner_task037_global_index") != ""}


def graph_change_rows(
    initial_edges: Iterable[Edge],
    final_edges: Iterable[Edge],
    survivor_ids: Iterable[int],
    unit_domain: Mapping[int, str],
) -> Tuple[List[Dict[str, Any]], Set[Edge], Set[Edge], Set[Edge]]:
    """Account for impossible E0 edges separately from genuine lost/new edges."""
    e0, e1, survivors = {pair_key(*e) for e in initial_edges}, {pair_key(*e) for e in final_edges}, set(map(int, survivor_ids))
    possible = {e for e in e0 if set(e) <= survivors}
    impossible = e0 - possible
    retained, lost, new = possible & e1, possible - e1, e1 - possible
    rows: List[Dict[str, Any]] = []
    domains = sorted(set(unit_domain.values()), key=lambda x: (int(x) if str(x).isdigit() else str(x)))
    for domain in domains + ["ALL_DOMAINS"]:
        select = lambda edges: edges if domain == "ALL_DOMAINS" else {e for e in edges if unit_domain[e[0]] == domain}
        p, q, r, x = select(possible), select(e1), select(retained), select(lost)
        n, imp = select(new), select(impossible)
        union = p | q
        rows.append({"domain_id": domain, "initial_edges": len(select(e0)),
                     "initial_edges_possible_among_survivors": len(p),
                     "initial_edges_impossible_removed_endpoint": len(imp),
                     "post_edges": len(q), "retained_edges": len(r),
                     "lost_possible_edges": len(x), "new_edges": len(n),
                     "jaccard_on_possible_edges": (len(r) / len(union)) if union else 1.0,
                     "initial_edge_ids": sorted(select(e0)),
                     "possible_initial_edge_ids": sorted(p), "post_edge_ids": sorted(q),
                     "retained_edge_ids": sorted(r), "lost_edge_ids": sorted(x),
                     "new_edge_ids": sorted(n), "impossible_edge_ids": sorted(imp)})
    return rows, retained, lost, new


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    position = (len(xs) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return xs[lo] + (xs[hi] - xs[lo]) * (position - lo)


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return {"count": len(xs), "mean": statistics.mean(xs) if xs else None,
            "median": statistics.median(xs) if xs else None,
            "q25": percentile(xs, 0.25), "q75": percentile(xs, 0.75),
            "max_absolute_change": max(map(abs, xs)) if xs else None}


def calibration_subsets(video_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    """Recreate Phase-A class-position subsets from its frozen video identities."""
    by_class: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for row in video_rows:
        by_class[int(row["label"])].append((int(row["video_index"]), int(row["dataset_index"])))
    if len(by_class) != 10 or any(len(v) != 3 for v in by_class.values()):
        raise ValueError("calibration cohort must be exactly 10 classes x 3 videos")
    positions: Dict[int, List[int]] = {1: [], 2: [], 3: []}
    for label, entries in by_class.items():
        entries.sort()
        for pos, (video_index, _dataset_index) in enumerate(entries, 1):
            positions[pos].append(video_index)
    if any(len(v) != 10 for v in positions.values()):
        raise ValueError("class-position calibration subsets are incomplete")
    all_ids = sorted(int(r["video_index"]) for r in video_rows)
    return {
        "A_position1": sorted(positions[1]),
        "B_position2": sorted(positions[2]),
        "C_position3": sorted(positions[3]),
        "AB_positions12": sorted(positions[1] + positions[2]),
        "AC_positions13": sorted(positions[1] + positions[3]),
        "BC_positions23": sorted(positions[2] + positions[3]),
        "full_10x3": all_ids,
    }


def aggregate_relation_distances(
    raw_rows: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    video_indices: Sequence[int],
    task042: Any,
) -> Tuple[Dict[Edge, Optional[float]], Dict[Edge, int], Dict[Tuple[int, Edge], Optional[float]]]:
    """Exact Phase-A per-unit/per-video z-normalization then Pearson d_temp average."""
    expected_vids = set(map(int, video_indices))
    expected_ids = {int(r["task037_global_index"]) for r in units}
    conditions: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)
    for row in raw_rows:
        vi, uid = int(row["video_index"]), int(row["task037_global_index"])
        if vi not in expected_vids or uid not in expected_ids:
            continue
        conditions[(vi, uid)].append((int(row["span"]), int(row["pair_index"]), float(row["relative_sensitivity"])))
    vectors: Dict[Tuple[int, int], Optional[List[float]]] = {}
    for vi in expected_vids:
        for uid in expected_ids:
            items = sorted(conditions.get((vi, uid), []), key=lambda x: (x[0], x[1]))
            if len(items) != 80 or [(s, p) for s, p, _ in items] != [(s, p) for s in (1, 2, 4, 8, 16) for p in range(16)]:
                raise ValueError("capture must contain the exact 80 Phase-A intervention identities")
            z, _mean, _std, degenerate = task042.normalize_signature([v for _, _, v in items])
            vectors[(vi, uid)] = None if degenerate else z
    domains: Dict[str, List[int]] = defaultdict(list)
    for row in units:
        domains[str(row["domain_id"])].append(int(row["task037_global_index"]))
    distances: Dict[Edge, Optional[float]] = {}
    counts: Dict[Edge, int] = {}
    by_video: Dict[Tuple[int, Edge], Optional[float]] = {}
    for ids in domains.values():
        ids.sort()
        for pos, i in enumerate(ids):
            for j in ids[pos + 1:]:
                edge = pair_key(i, j)
                vals = []
                for vi in sorted(expected_vids):
                    zi, zj = vectors[(vi, i)], vectors[(vi, j)]
                    d = None if zi is None or zj is None else task042.relation_distance(zi, zj)
                    by_video[(vi, edge)] = d
                    if d is not None:
                        vals.append(float(d))
                distances[edge] = (sum(vals) / len(vals)) if vals else None
                counts[edge] = len(vals)
    return distances, counts, by_video

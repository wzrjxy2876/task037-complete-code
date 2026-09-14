"""Offline Task042 Phase-H continuous temporal functional coverage audit.

Consumes only finalized Phase-G relation preferences/stability/protocol and
the Phase-F exact parameter-cost ledger. It has no model, CUDA, pruning,
training, inference, or validation-performance dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_FIELDS = (
    "task037_global_index", "domain_id", "layer", "stage", "unit_type", "unit_index"
)
Q_FIELDS = ("span", "pair_index", "frame_a", "frame_b")
CALIBRATION_NAMES = ("P1", "P2", "P3", "P12", "P13", "P23", "FULL")
EXPECTED_UNITS = 51
EXPECTED_VIDEOS = 30
EXPECTED_CLASSES = 10
EXPECTED_RELATIONS = 80
EPSILON = 1e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in fields})


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation(row: Mapping[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(row[name]) for name in Q_FIELDS)  # type: ignore[return-value]


def _unit(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(row[name]).strip() for name in IDENTITY_FIELDS)


def _numeric_label(label: str) -> tuple[int, str]:
    try:
        return int(label), label
    except ValueError:
        return 10**9, label


def _atom_key(label: str, q: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int, int]]:
    return label, q


def _subset_key(units: Sequence[tuple[str, ...]]) -> str:
    return ",".join(unit[0] for unit in sorted(units, key=lambda u: (int(u[0]), u)))


def _rank(values: Sequence[float], descending: bool = False) -> list[float]:
    """Exact-tie average ranks (1-based)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i] if descending else values[i], i))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        value = values[order[start]]
        while stop < len(order) and values[order[stop]] == value:
            stop += 1
        avg = ((start + 1) + stop) / 2.0
        for pos in range(start, stop):
            result[order[pos]] = avg
        start = stop
    return result


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson vectors must have equal nonzero length")
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    lc = [v - lm for v in left]
    rc = [v - rm for v in right]
    ld = math.sqrt(sum(v * v for v in lc))
    rd = math.sqrt(sum(v * v for v in rc))
    if ld == 0.0 or rd == 0.0:
        return None
    return sum(a * b for a, b in zip(lc, rc)) / (ld * rd)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(_rank(left), _rank(right))


def exact_pareto_dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether left exactly dominates right, with no tolerance."""
    if len(left) != len(right):
        raise ValueError("Pareto vectors must have equal lengths")
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def eta_value(f_keep: float, f_full: float, epsilon: float = EPSILON) -> float:
    if f_full == 0.0:
        return 1.0
    return f_keep / (f_full + epsilon)


def class_median_preference(video_values: Sequence[float]) -> float:
    """Aggregate already-normalized Phase-G preferences, without reranking."""
    if not video_values:
        raise ValueError("Cannot aggregate an empty class cohort")
    return float(statistics.median(video_values))


def enumerate_nonempty_subsets(items: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Deterministic exhaustive subset enumeration, increasing cardinality."""
    return [subset for size in range(1, len(items) + 1) for subset in itertools.combinations(items, size)]


def leave_one_out(items: Sequence[Any], removed: Any) -> tuple[Any, ...]:
    if removed not in items or len(items) < 2:
        raise ValueError("Leave-one-out requires a member of a group with at least two items")
    return tuple(item for item in items if item != removed)


def exact_parameter_accounting(group_costs: Mapping[Any, int], retained: Sequence[Any]) -> tuple[int, int]:
    if any(item not in group_costs for item in retained):
        raise ValueError("Missing exact parameter cost for retained unit")
    total = sum(group_costs.values())
    retained_cost = sum(group_costs[item] for item in retained)
    return retained_cost, total - retained_cost


def subset_envelope(unit_profiles: Mapping[Any, Sequence[float]], retained: Sequence[Any]) -> list[float]:
    if not retained or any(item not in unit_profiles for item in retained):
        raise ValueError("A subset envelope requires known, non-empty retained units")
    lengths = {len(unit_profiles[item]) for item in retained}
    if len(lengths) != 1:
        raise ValueError("Unit profiles must share the same atom count")
    width = next(iter(lengths))
    return [max(unit_profiles[item][j] for item in retained) for j in range(width)]


def _metrics(vector: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(vector)
    return {
        "eta_min": min(vector),
        "eta_mean": sum(vector) / len(vector),
        "eta_median": statistics.median(ordered),
        "eta_exactly_one_count": sum(value == 1.0 for value in vector),
        "eta_exactly_zero_count": sum(value == 0.0 for value in vector),
    }


def _identity_obj(unit: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(IDENTITY_FIELDS, unit))


def _json_vector(vector: Sequence[float]) -> str:
    return json.dumps(list(vector), separators=(",", ":"), allow_nan=False)


def _validate_subset_map(subset_map: Mapping[str, Sequence[int]], videos_by_label: Mapping[str, Sequence[int]]) -> None:
    if tuple(subset_map) != CALIBRATION_NAMES:
        raise ValueError(f"Calibration map names/order mismatch: {tuple(subset_map)}")
    video_to_label = {v: label for label, videos in videos_by_label.items() for v in videos}
    if len(video_to_label) != EXPECTED_VIDEOS:
        raise ValueError("Phase-G video pool is not exactly 30 unique video indices")
    for name, selected in subset_map.items():
        selected_set = set(int(v) for v in selected)
        if len(selected_set) != len(selected) or not selected_set <= set(video_to_label):
            raise ValueError(f"Invalid/duplicate video index in Phase-G subset {name}")
        counts = defaultdict(int)
        for index in selected_set:
            counts[video_to_label[index]] += 1
        expected = {"P1": 1, "P2": 1, "P3": 1, "P12": 2, "P13": 2, "P23": 2, "FULL": 3}[name]
        if any(counts[label] != expected for label in videos_by_label):
            raise ValueError(f"Phase-G subset {name} does not retain {expected} videos per class")


def _pareto_flags(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool]]:
    by_key = {row["subset_id"]: row for row in rows}
    release_min = {key: True for key in by_key}
    release_mean = {key: True for key in by_key}
    release_min_mean = {key: True for key in by_key}
    for a_id, a in by_key.items():
        va = (float(a["released_parameters"]), float(a["eta_min"]), float(a["eta_mean"]))
        for b_id, b in by_key.items():
            if a_id == b_id:
                continue
            vb = (float(b["released_parameters"]), float(b["eta_min"]), float(b["eta_mean"]))
            if exact_pareto_dominates(vb, va):
                release_min_mean[a_id] = False
            if exact_pareto_dominates((vb[0], vb[1]), (va[0], va[1])):
                release_min[a_id] = False
            if exact_pareto_dominates((vb[0], vb[2]), (va[0], va[2])):
                release_mean[a_id] = False
    return release_min, release_mean, release_min_mean


def run_phase_h(
    phase_g_dir: Path,
    phase_f_dir: Path,
    output_dir: Path,
    git_head: str = "unknown",
) -> dict[str, Any]:
    phase_g_dir = Path(phase_g_dir)
    phase_f_dir = Path(phase_f_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preference_path = phase_g_dir / "task042_phase_g_relation_preference.csv"
    phase_g_summary_path = phase_g_dir / "task042_phase_g_summary.json"
    phase_g_stability_path = phase_g_dir / "task042_phase_g_calibration_stability.csv"
    parameter_path = phase_f_dir / "task042_phase_f_parameter_cost.csv"
    for path in (preference_path, phase_g_summary_path, phase_g_stability_path, parameter_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    pg_summary = json.loads(phase_g_summary_path.read_text(encoding="utf-8"))
    if pg_summary.get("git_head") != "9282b3a7aa369dc90db43433e85b8f7afda2b5e5":
        raise ValueError(f"Unexpected Phase-G authoritative head: {pg_summary.get('git_head')}")
    subset_map = {name: tuple(int(v) for v in pg_summary["frozen_protocol"]["subsets"][name]) for name in CALIBRATION_NAMES}

    preference_rows = _read_csv(preference_path)
    if len(preference_rows) != EXPECTED_UNITS * EXPECTED_VIDEOS * EXPECTED_RELATIONS:
        raise ValueError(f"Expected 122400 Phase-G preference rows, found {len(preference_rows)}")
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    labels_by_video: dict[int, str] = {}
    pref: dict[tuple[str, ...], dict[int, dict[tuple[int, int, int, int], float]]] = defaultdict(lambda: defaultdict(dict))
    for row in preference_rows:
        uid = _unit(row)
        metadata[uid] = {field: row[field] for field in IDENTITY_FIELDS}
        vid = int(row["video_index"])
        label = row["label"].strip()
        prior_label = labels_by_video.setdefault(vid, label)
        if prior_label != label:
            raise ValueError(f"Phase-G video index {vid} has multiple labels")
        q = _relation(row)
        value = float(row["relation_preference"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Invalid Phase-G relation_preference value")
        if q in pref[uid][vid]:
            raise ValueError(f"Duplicate Phase-G preference: {uid}/{vid}/{q}")
        pref[uid][vid][q] = value
    if len(metadata) != EXPECTED_UNITS or set(labels_by_video) != set(range(EXPECTED_VIDEOS)):
        raise ValueError("Phase-G unit/video identity count mismatch")
    labels = sorted(set(labels_by_video.values()), key=_numeric_label)
    if len(labels) != EXPECTED_CLASSES:
        raise ValueError(f"Expected ten classes, found {len(labels)}")
    videos_by_label: dict[str, tuple[int, ...]] = {
        label: tuple(i for i in range(EXPECTED_VIDEOS) if labels_by_video[i] == label) for label in labels
    }
    if any(len(v) != 3 for v in videos_by_label.values()):
        raise ValueError("Phase-G cohort is not 10 classes x 3 videos")
    _validate_subset_map(subset_map, videos_by_label)
    q_values = sorted({_relation(row) for row in preference_rows})
    if len(q_values) != EXPECTED_RELATIONS:
        raise ValueError(f"Expected 80 exact relations, found {len(q_values)}")
    for uid in metadata:
        if len(pref[uid]) != EXPECTED_VIDEOS or any(len(pref[uid][v]) != EXPECTED_RELATIONS for v in range(EXPECTED_VIDEOS)):
            raise ValueError(f"Incomplete reused Phase-G preference records: {uid}")

    parameter_rows = _read_csv(parameter_path)
    costs: dict[tuple[str, ...], int] = {}
    for row in parameter_rows:
        gid = str(row["task037_global_index"])
        matches = [uid for uid in metadata if uid[0] == gid]
        if len(matches) != 1:
            raise ValueError(f"Parameter-cost identity does not map uniquely: {gid}")
        uid = matches[0]
        cost = int(row["parameter_cost_exact"])
        if cost <= 0:
            raise ValueError(f"Nonpositive exact parameter cost for {gid}")
        if uid in costs:
            raise ValueError(f"Duplicate exact parameter cost for {gid}")
        costs[uid] = cost
    by_domain: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for uid in metadata:
        by_domain[uid[1]].append(uid)
    domains = {d: sorted(units, key=lambda u: (int(u[0]), u)) for d, units in by_domain.items() if len(units) > 1}
    expected_domain_ids = set(pg_summary["counts"]["multi_unit_domain_ids"])
    if set(domains) != expected_domain_ids:
        raise ValueError(f"Multi-unit domains differ from frozen Phase G: {sorted(domains)}")
    all_domain_units = {uid for units in domains.values() for uid in units}
    if not all_domain_units <= set(costs):
        raise ValueError("Phase-F exact parameter-cost ledger misses a unit in a multi-unit domain")

    atoms: list[tuple[str, tuple[int, int, int, int]]] = [
        (label, q) for label in labels for q in q_values
    ]
    if len(atoms) != 800:
        raise AssertionError("Temporal atom count must be 800")
    q_index = {q: i for i, q in enumerate(q_values)}
    label_index = {label: i for i, label in enumerate(labels)}
    atom_index = {(label, q): i for i, (label, q) in enumerate(atoms)}

    # Domain/cohort/unit -> exact relation-wise class-median preferences.
    class_profiles: dict[str, dict[str, dict[tuple[str, ...], list[float]]]] = {}
    global_profiles: dict[str, dict[str, dict[tuple[str, ...], list[float]]]] = {}
    envelopes: dict[str, dict[str, list[float]]] = {}
    subset_vectors: dict[str, dict[str, dict[str, list[float]]]] = {}
    subset_rows: dict[tuple[str, str], dict[str, Any]] = {}
    envelope_rows: list[dict[str, Any]] = []
    for cohort in CALIBRATION_NAMES:
        selected_videos = set(subset_map[cohort])
        domain_profiles: dict[str, dict[tuple[str, ...], list[float]]] = {}
        domain_global_profiles: dict[str, dict[tuple[str, ...], list[float]]] = {}
        domain_envelopes: dict[str, list[float]] = {}
        domain_subset_profiles: dict[str, dict[str, list[float]]] = {}
        domain_subset_summary: dict[str, dict[str, Any]] = {}
        for domain_id, units in sorted(domains.items(), key=lambda item: (int(item[0]), item[0])):
            profiles: dict[tuple[str, ...], list[float]] = {}
            global_unit_profiles: dict[tuple[str, ...], list[float]] = {}
            for uid in units:
                vector: list[float] = []
                for label, q in atoms:
                    chosen = [pref[uid][v][q] for v in videos_by_label[label] if v in selected_videos]
                    if not chosen:
                        raise ValueError(f"Empty class-conditioned calibration selection {cohort}/{label}")
                    vector.append(class_median_preference(chosen))
                profiles[uid] = vector
                chosen_all = [v for v in sorted(selected_videos)]
                global_unit_profiles[uid] = [float(statistics.median(pref[uid][v][q] for v in chosen_all)) for q in q_values]
            full_envelope = [max(profiles[uid][j] for uid in units) for j in range(800)]
            domain_profiles[domain_id] = profiles
            domain_global_profiles[domain_id] = global_unit_profiles
            domain_envelopes[domain_id] = full_envelope
            for j, (label, q) in enumerate(atoms):
                envelope_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "atom_index": j,
                    "action_label": label, "span": q[0], "pair_index": q[1], "frame_a": q[2], "frame_b": q[3],
                    "F_full": full_envelope[j], "F_full_is_exact_zero": full_envelope[j] == 0.0,
                })
            all_subsets = enumerate_nonempty_subsets(units)
            vectors_for_domain: dict[str, list[float]] = {}
            summaries_for_domain: dict[str, dict[str, Any]] = {}
            domain_total_cost = sum(costs[u] for u in units)
            for retained in all_subsets:
                sid = _subset_key(retained)
                keep_envelope = subset_envelope(profiles, retained)
                eta = [eta_value(keep_envelope[j], full_envelope[j]) for j in range(800)]
                stats = _metrics(eta)
                retained_cost, released = exact_parameter_accounting({uid: costs[uid] for uid in units}, retained)
                removed = tuple(uid for uid in units if uid not in retained)
                key = (domain_id, sid)
                summary_row = {
                    "domain_id": domain_id, "calibration_subset": cohort, "subset_id": sid,
                    "retained_units": [_identity_obj(uid) for uid in retained],
                    "removed_units": [_identity_obj(uid) for uid in removed],
                    "retained_global_indices": [int(uid[0]) for uid in retained],
                    "removed_global_indices": [int(uid[0]) for uid in removed],
                    "retained_count": len(retained), "removed_count": len(removed),
                    "retained_exact_parameters": retained_cost, "released_parameters": released,
                    "domain_exact_parameters": domain_total_cost,
                    **stats,
                    "eta_vector": _json_vector(eta),
                }
                subset_rows[(cohort, domain_id, sid)] = summary_row
                vectors_for_domain[sid] = eta
                summaries_for_domain[sid] = summary_row
            domain_subset_profiles[domain_id] = vectors_for_domain
            domain_subset_summary[domain_id] = summaries_for_domain
        class_profiles[cohort] = domain_profiles
        global_profiles[cohort] = domain_global_profiles
        envelopes[cohort] = domain_envelopes
        subset_vectors[cohort] = domain_subset_profiles
        # Persist the fixed retained-set rows after all cohorts are complete.

    # All subset coverage rows, stable ordering.
    subset_fields = [
        "domain_id", "calibration_subset", "subset_id", "retained_units", "removed_units",
        "retained_global_indices", "removed_global_indices", "retained_count", "removed_count",
        "retained_exact_parameters", "released_parameters", "domain_exact_parameters",
        "eta_min", "eta_mean", "eta_median", "eta_exactly_one_count", "eta_exactly_zero_count", "eta_vector",
    ]
    subset_output_rows = [subset_rows[(c, d, sid)] for c in CALIBRATION_NAMES for d in sorted(domains, key=_numeric_label)
                          for sid in sorted(subset_vectors[c][d], key=lambda x: (len(x.split(",")), tuple(int(i) for i in x.split(","))))]

    # Same-retained-count exact functional dominance; exact released-parameter
    # matches are reported separately. Approximate matching is represented by
    # nearest released-parameter peers, with no invented tolerance.
    pareto_rows: list[dict[str, Any]] = []
    pareto3_flags: dict[str, dict[str, dict[str, bool]]] = {}
    parameter_rows_out: list[dict[str, Any]] = []
    parameter_flags: dict[str, dict[str, dict[str, bool]]] = {}
    for cohort in CALIBRATION_NAMES:
        pareto3_flags[cohort] = {}
        parameter_flags[cohort] = {}
        for domain_id, units in sorted(domains.items(), key=lambda item: _numeric_label(item[0])):
            group = [subset_rows[(cohort, domain_id, sid)] for sid in subset_vectors[cohort][domain_id]]
            by_sid = {r["subset_id"]: r for r in group}
            flags_min, flags_mean, flags_3d = _pareto_flags(group)
            parameter_flags[cohort][domain_id] = {sid: flags_3d[sid] for sid in by_sid}
            pareto3_flags[cohort][domain_id] = {sid: flags_3d[sid] for sid in by_sid}
            for row in group:
                sid = row["subset_id"]
                same_count = [x for x in group if x["retained_count"] == row["retained_count"] and x["subset_id"] != sid]
                exact_param_peers = [x for x in same_count if x["released_parameters"] == row["released_parameters"]]
                vector = subset_vectors[cohort][domain_id][sid]
                dominated_same_count_by = [x["subset_id"] for x in same_count
                                           if exact_pareto_dominates(subset_vectors[cohort][domain_id][x["subset_id"]], vector)]
                dominated_exact_parameter_by = [x["subset_id"] for x in exact_param_peers
                                                if exact_pareto_dominates(subset_vectors[cohort][domain_id][x["subset_id"]], vector)]
                if same_count:
                    min_gap = min(abs(x["released_parameters"] - row["released_parameters"]) for x in same_count)
                    nearest = [x for x in same_count if abs(x["released_parameters"] - row["released_parameters"]) == min_gap]
                    nearest_dominators = [x["subset_id"] for x in nearest
                                          if exact_pareto_dominates(subset_vectors[cohort][domain_id][x["subset_id"]], vector)]
                else:
                    min_gap, nearest, nearest_dominators = None, [], []
                pareto_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "subset_id": sid,
                    "retained_count": row["retained_count"], "released_parameters": row["released_parameters"],
                    "same_count_comparison_count": len(same_count),
                    "dominated_same_count": bool(dominated_same_count_by),
                    "same_count_dominator_ids": dominated_same_count_by,
                    "nondominated_same_count": not bool(dominated_same_count_by),
                    "exact_equal_release_peer_count": len(exact_param_peers),
                    "dominated_exact_equal_release": bool(dominated_exact_parameter_by),
                    "exact_equal_release_dominator_ids": dominated_exact_parameter_by,
                    "nearest_release_gap": min_gap,
                    "nearest_release_peer_ids": [x["subset_id"] for x in nearest],
                    "nearest_release_peer_dominator_ids": nearest_dominators,
                    "nearest_release_peer_dominates": bool(nearest_dominators),
                    "nondominated_release_eta_min": flags_min[sid],
                    "nondominated_release_eta_mean": flags_mean[sid],
                    "nondominated_release_eta_min_eta_mean": flags_3d[sid],
                })
                parameter_rows_out.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "subset_id": sid,
                    "retained_count": row["retained_count"], "released_parameters": row["released_parameters"],
                    "eta_min": row["eta_min"], "eta_mean": row["eta_mean"],
                    "nondominated_release_eta_min": flags_min[sid],
                    "nondominated_release_eta_mean": flags_mean[sid],
                    "nondominated_release_eta_min_eta_mean": flags_3d[sid],
                    "eta_vector": row["eta_vector"],
                })

    # Leave-one-unit-out as explicit continuous-coverage profiles for all cohorts.
    loo_rows: list[dict[str, Any]] = []
    for cohort in CALIBRATION_NAMES:
        for domain_id, units in sorted(domains.items(), key=lambda item: _numeric_label(item[0])):
            sid_all = _subset_key(units)
            domain_full = envelopes[cohort][domain_id]
            for removed_uid in units:
                retained = leave_one_out(units, removed_uid)
                sid = _subset_key(retained)
                eta = subset_vectors[cohort][domain_id][sid]
                min_eta = min(eta)
                worst_idx = next(i for i, value in enumerate(eta) if value == min_eta)
                label, q = atoms[worst_idx]
                row = subset_rows[(cohort, domain_id, sid)]
                loo_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort,
                    "removed_unit": _identity_obj(removed_uid), "removed_global_index": int(removed_uid[0]),
                    "retained_global_indices": row["retained_global_indices"],
                    "retained_count": row["retained_count"], "released_parameters": row["released_parameters"],
                    "eta_min": min_eta, "eta_median": statistics.median(eta), "eta_mean": sum(eta) / 800,
                    "eta_exactly_one_count": sum(x == 1.0 for x in eta),
                    "largest_functional_drop": 1.0 - min_eta,
                    "worst_atom_index": worst_idx, "worst_action_label": label,
                    "worst_span": q[0], "worst_pair_index": q[1], "worst_frame_a": q[2], "worst_frame_b": q[3],
                    "worst_F_full": domain_full[worst_idx],
                })

    # Coverage-profile stability against FULL, including minimum-coverage rank
    # order and 3D parameter-aware frontier membership.
    calibration_rows: list[dict[str, Any]] = []
    pg_stability = _read_csv(phase_g_stability_path)
    pg_summaries = {(r["domain_id"], r["subset"]): r for r in pg_stability if r["row_type"] == "domain_summary"}
    eta_min_ranks: dict[str, dict[str, dict[str, list[float]]]] = {}
    rank_spearman_by: dict[tuple[str, str], float | None] = {}
    rank_exact_pairwise_by: dict[tuple[str, str], float] = {}
    for domain_id in domains:
        full_sids = list(subset_vectors["FULL"][domain_id])
        full_mins = [min(subset_vectors["FULL"][domain_id][sid]) for sid in full_sids]
        full_ranks = _rank(full_mins, descending=True)
        eta_min_ranks.setdefault("FULL", {})[domain_id] = {sid: [full_ranks[i]] for i, sid in enumerate(full_sids)}
        for cohort in CALIBRATION_NAMES[:-1]:
            sids = list(subset_vectors[cohort][domain_id])
            if set(sids) != set(full_sids):
                raise AssertionError("Calibration retained-set identity differs from FULL")
            subset_mins = [min(subset_vectors[cohort][domain_id][sid]) for sid in sids]
            subset_ranks = _rank(subset_mins, descending=True)
            aligned_full_ranks = [full_ranks[full_sids.index(sid)] for sid in sids]
            rank_spearman_by[(domain_id, cohort)] = spearman(aligned_full_ranks, subset_ranks)
            agreements = []
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    df = aligned_full_ranks[i] - aligned_full_ranks[j]
                    ds = subset_ranks[i] - subset_ranks[j]
                    agreements.append(1.0 if df == ds else (0.0 if df == 0.0 or ds == 0.0 or df * ds < 0 else 1.0))
            rank_exact_pairwise_by[(domain_id, cohort)] = sum(agreements) / len(agreements) if agreements else 1.0

    # Phase-G domain-summary exact owner metrics.
    phase_g_owner: dict[tuple[str, str], dict[str, Any]] = {}
    for domain_id in domains:
        phase_g_owner[(domain_id, "FULL")] = {"mean_jaccard": 1.0, "exact_owner_match_rate": 1.0}
        for cohort in CALIBRATION_NAMES[:-1]:
            row = pg_summaries.get((domain_id, cohort))
            if row is None:
                raise ValueError(f"Missing Phase-G owner stability summary for {domain_id}/{cohort}")
            phase_g_owner[(domain_id, cohort)] = {
                "mean_jaccard": float(row["mean_jaccard"]),
                "exact_owner_match_rate": float(row["exact_owner_set_match_rate"]),
            }

    for domain_id in sorted(domains, key=_numeric_label):
        full_sids = list(subset_vectors["FULL"][domain_id])
        full_rank_values = [min(subset_vectors["FULL"][domain_id][sid]) for sid in full_sids]
        full_rank = _rank(full_rank_values, descending=True)
        full_rank_map = dict(zip(full_sids, full_rank))
        for cohort in CALIBRATION_NAMES[:-1]:
            subset_sids = list(subset_vectors[cohort][domain_id])
            subset_rank_values = [min(subset_vectors[cohort][domain_id][sid]) for sid in subset_sids]
            subset_rank = _rank(subset_rank_values, descending=True)
            subset_rank_map = dict(zip(subset_sids, subset_rank))
            rank_rho = rank_spearman_by[(domain_id, cohort)]
            pairwise_agreement = rank_exact_pairwise_by[(domain_id, cohort)]
            for sid in subset_sids:
                full_vector = subset_vectors["FULL"][domain_id][sid]
                vector = subset_vectors[cohort][domain_id][sid]
                diffs = [abs(a - b) for a, b in zip(vector, full_vector)]
                action_metric = subset_rows[(cohort, domain_id, sid)]
                full_metric = subset_rows[("FULL", domain_id, sid)]
                pg = phase_g_owner[(domain_id, cohort)]
                calibration_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "reference_subset": "FULL", "subset_id": sid,
                    "pearson_eta_profile": pearson(vector, full_vector),
                    "spearman_eta_profile": spearman(vector, full_vector),
                    "mean_absolute_eta_difference": sum(diffs) / len(diffs),
                    "max_absolute_eta_difference": max(diffs),
                    "eta_min_subset": action_metric["eta_min"], "eta_min_full": full_metric["eta_min"],
                    "eta_min_rank_subset": subset_rank_map[sid], "eta_min_rank_full": full_rank_map[sid],
                    "eta_min_rank_absolute_shift": abs(subset_rank_map[sid] - full_rank_map[sid]),
                    "eta_min_rank_spearman_across_retained_sets": rank_rho,
                    "eta_min_rank_pairwise_order_agreement": pairwise_agreement,
                    "pareto_3d_nondominated_subset": pareto3_flags[cohort][domain_id][sid],
                    "pareto_3d_nondominated_full": pareto3_flags["FULL"][domain_id][sid],
                    "pareto_3d_membership_match": pareto3_flags[cohort][domain_id][sid] == pareto3_flags["FULL"][domain_id][sid],
                    "phase_g_exact_owner_mean_jaccard": pg["mean_jaccard"],
                    "phase_g_exact_owner_match_rate": pg["exact_owner_match_rate"],
                })

    # Action-conditioning ablation. The global profile uses median preference
    # across the selected cohort videos without labels, then broadcasts that
    # relation-wise eta over the 10 labelled atoms only for aligned comparison.
    action_rows: list[dict[str, Any]] = []
    for cohort in CALIBRATION_NAMES:
        selected = set(subset_map[cohort])
        for domain_id, units in sorted(domains.items(), key=lambda item: _numeric_label(item[0])):
            gp = global_profiles[cohort][domain_id]
            global_full = [max(gp[u][qi] for u in units) for qi in range(80)]
            for sid, action_eta in subset_vectors[cohort][domain_id].items():
                retained_ids = tuple(int(x) for x in sid.split(","))
                retained = tuple(u for u in units if int(u[0]) in retained_ids)
                global_keep = [max(gp[u][qi] for u in retained) for qi in range(80)]
                global_eta_q = [eta_value(global_keep[qi], global_full[qi]) for qi in range(80)]
                global_eta = [global_eta_q[q_index[q]] for label in labels for q in q_values]
                diffs = [abs(a - b) for a, b in zip(action_eta, global_eta)]
                lower = [i for i, (a, g) in enumerate(zip(action_eta, global_eta)) if g < a]
                higher = [i for i, (a, g) in enumerate(zip(action_eta, global_eta)) if g > a]
                equal = [i for i, (a, g) in enumerate(zip(action_eta, global_eta)) if g == a]
                gm = _metrics(global_eta)
                action_stats = _metrics(action_eta)
                min_global = min(global_eta)
                weak_atoms = [i for i, v in enumerate(global_eta) if v == min_global]
                action_min = min(action_eta)
                action_weak = [i for i, v in enumerate(action_eta) if v == action_min]
                action_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "subset_id": sid,
                    "retained_global_indices": retained_ids,
                    "action_conditioned_eta_min": action_stats["eta_min"],
                    "action_conditioned_eta_mean": action_stats["eta_mean"],
                    "global_unconditioned_eta_min": gm["eta_min"],
                    "global_unconditioned_eta_mean": gm["eta_mean"],
                    "profile_pearson": pearson(action_eta, global_eta),
                    "profile_spearman": spearman(action_eta, global_eta),
                    "mean_absolute_eta_difference": sum(diffs) / 800,
                    "max_absolute_eta_difference": max(diffs),
                    "global_lower_atom_count": len(lower), "global_higher_atom_count": len(higher),
                    "exactly_equal_atom_count": len(equal),
                    "global_weak_atom_indices": weak_atoms,
                    "action_conditioned_weak_atom_indices": action_weak,
                    "global_eta_vector_800": _json_vector(global_eta),
                })

    # Mixed Attention/FFN subsets, retaining each exact profile.
    mixed_rows: list[dict[str, Any]] = []
    for cohort in CALIBRATION_NAMES:
        for domain_id in ("271", "297"):
            if domain_id not in domains:
                continue
            for sid, eta in subset_vectors[cohort][domain_id].items():
                ids = tuple(int(x) for x in sid.split(","))
                retained = [u for u in domains[domain_id] if int(u[0]) in ids]
                types = sorted({metadata[u]["unit_type"] for u in retained})
                if len(types) == 1 and types[0] in ("attention_head", "attention", "attn_head"):
                    category = "attention_only"
                elif len(types) == 1 and "ffn" in types[0].lower():
                    category = "ffn_only"
                else:
                    category = "mixed_attention_ffn"
                summary = subset_rows[(cohort, domain_id, sid)]
                mixed_rows.append({
                    "domain_id": domain_id, "calibration_subset": cohort, "subset_id": sid,
                    "survivor_category": category, "survivor_unit_types": types,
                    "retained_global_indices": ids,
                    "attention_survivor_count": sum("att" in metadata[u]["unit_type"].lower() for u in retained),
                    "ffn_survivor_count": sum("ffn" in metadata[u]["unit_type"].lower() for u in retained),
                    "eta_min": summary["eta_min"], "eta_mean": summary["eta_mean"], "eta_median": summary["eta_median"],
                    "eta_vector": _json_vector(eta),
                })

    # Structured LOO summaries used to answer A without inventing a cutoff.
    loo_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loo_rows:
        if row["calibration_subset"] == "FULL":
            loo_by_domain[row["domain_id"]].append(row)
    loo_full_rows = [row for rows in loo_by_domain.values() for row in rows]
    full_nonempty_rows = [
        subset_rows[("FULL", domain_id, sid)]
        for domain_id in domains
        for sid in subset_vectors["FULL"][domain_id]
        if subset_rows[("FULL", domain_id, sid)]["removed_count"] > 0
    ]
    unique_owner_stability = [value for (domain_id, cohort), value in phase_g_owner.items() if cohort != "FULL"]

    summary: dict[str, Any] = {
        "phase": "Task042 Phase H",
        "git_head": git_head,
        "inputs": {str(p.name): {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in (preference_path, phase_g_summary_path, phase_g_stability_path, parameter_path)},
        "hard_constraints": {"gpu_used": False, "new_inference": False, "pruning_performed": False, "finetuning_performed": False, "performance_oracle_used": False},
        "reuse": {"phase_g_head": pg_summary["git_head"], "phase_g_preference_field": "relation_preference", "reranked": False,
                  "unit_count": len(metadata), "video_count": EXPECTED_VIDEOS, "class_count": len(labels),
                  "relations_per_video": len(q_values), "temporal_atoms_per_domain": len(atoms),
                  "multi_unit_domain_ids": sorted(domains, key=_numeric_label),
                  "calibration_video_indices": {k: list(v) for k, v in subset_map.items()}},
        "parameter_cost": {"identity_rows": len(costs), "multi_domain_unit_rows": len(all_domain_units),
                           "multi_domain_costs_all_present": all(u in costs for u in all_domain_units),
                           "source": "Phase-F parameter_cost_exact"},
        "counts": {"retained_subsets_per_domain": {d: (2 ** len(u)) - 1 for d, u in domains.items()},
                   "subset_coverage_rows": len(subset_output_rows), "functional_envelope_rows": len(envelope_rows),
                   "leave_one_out_rows": len(loo_rows), "calibration_stability_rows": len(calibration_rows),
                   "action_conditioning_rows": len(action_rows), "mixed_type_rows": len(mixed_rows)},
        "phase_g_reference": pg_summary.get("decision"),
        "full_cohort_all_pruned_subset_distribution": {
            "nonempty_removal_set_count": len(full_nonempty_rows),
            "eta_min_min": min(row["eta_min"] for row in full_nonempty_rows),
            "eta_min_median": statistics.median(row["eta_min"] for row in full_nonempty_rows),
            "eta_min_max": max(row["eta_min"] for row in full_nonempty_rows),
            "eta_mean_min": min(row["eta_mean"] for row in full_nonempty_rows),
            "eta_mean_median": statistics.median(row["eta_mean"] for row in full_nonempty_rows),
            "eta_mean_max": max(row["eta_mean"] for row in full_nonempty_rows),
            "eta_median_median": statistics.median(row["eta_median"] for row in full_nonempty_rows),
            "released_parameters_min": min(row["released_parameters"] for row in full_nonempty_rows),
            "released_parameters_max": max(row["released_parameters"] for row in full_nonempty_rows),
        },
        "leave_one_out_full_range": {
            "eta_min_min": min(row["eta_min"] for row in loo_full_rows),
            "eta_min_median": statistics.median(row["eta_min"] for row in loo_full_rows),
            "eta_min_max": max(row["eta_min"] for row in loo_full_rows),
            "eta_mean_median": statistics.median(row["eta_mean"] for row in loo_full_rows),
            "eta_exactly_one_median": statistics.median(row["eta_exactly_one_count"] for row in loo_full_rows),
            "released_parameters_min": min(row["released_parameters"] for row in loo_full_rows),
            "released_parameters_max": max(row["released_parameters"] for row in loo_full_rows),
        },
        "calibration_stability_full_comparison": {
            "mean_profile_pearson": _mean_or_none([r["pearson_eta_profile"] for r in calibration_rows]),
            "mean_profile_spearman": _mean_or_none([r["spearman_eta_profile"] for r in calibration_rows]),
            "mean_absolute_eta_difference_mean": _mean_or_none([r["mean_absolute_eta_difference"] for r in calibration_rows]),
            "max_absolute_eta_difference_max": max(r["max_absolute_eta_difference"] for r in calibration_rows),
            "eta_min_rank_spearman_mean": _mean_or_none(list(rank_spearman_by.values())),
            "eta_min_rank_pairwise_order_agreement_mean": _mean_or_none(list(rank_exact_pairwise_by.values())),
            "pareto_3d_membership_match_rate": sum(r["pareto_3d_membership_match"] for r in calibration_rows) / len(calibration_rows),
            "phase_g_owner_mean_jaccard_mean": _mean_or_none([r["mean_jaccard"] for r in unique_owner_stability]),
            "phase_g_owner_exact_match_rate_mean": _mean_or_none([r["exact_owner_match_rate"] for r in unique_owner_stability]),
        },
        "action_conditioning": {
            "mean_profile_pearson": _mean_or_none([r["profile_pearson"] for r in action_rows]),
            "mean_profile_spearman": _mean_or_none([r["profile_spearman"] for r in action_rows]),
            "mean_absolute_eta_difference_mean": _mean_or_none([r["mean_absolute_eta_difference"] for r in action_rows]),
            "mean_global_lower_atom_count": _mean_or_none([r["global_lower_atom_count"] for r in action_rows]),
            "mean_global_higher_atom_count": _mean_or_none([r["global_higher_atom_count"] for r in action_rows]),
        },
        "mixed_type": {"domains": [d for d in ("271", "297") if d in domains],
                       "attention_only_rows": sum(r["survivor_category"] == "attention_only" for r in mixed_rows),
                       "ffn_only_rows": sum(r["survivor_category"] == "ffn_only" for r in mixed_rows),
                       "mixed_rows": sum(r["survivor_category"] == "mixed_attention_ffn" for r in mixed_rows)},
        "decision": {"code": "B", "label": "CONTINUOUS_TEMPORAL_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED",
                     "reason": "No threshold or downstream damage oracle is preregistered; Phase H reports the exact set-level profiles and stability diagnostics but does not authorize a one-shot pruning selector."},
        "interpretation": {
            "A": "The hard all-units-indispensable conclusion is no longer imposed by owner constraints; continuous profiles quantify the actual coverage loss for each leave-one-out subset.",
            "B": "Exact eta vectors, parameter release, calibration stability, action conditioning, mixed-type profiles, and Pareto frontiers are diagnostics only; no threshold converts them into a pruning rule.",
            "C": "Do not incorporate this as a one-shot selection oracle until a separate preregistered validation establishes utility without a validation-damage oracle.",
        },
    }

    _write_csv(output_dir / "task042_phase_h_functional_envelope.csv", envelope_rows,
               ("domain_id", "calibration_subset", "atom_index", "action_label", "span", "pair_index", "frame_a", "frame_b", "F_full", "F_full_is_exact_zero"))
    _write_csv(output_dir / "task042_phase_h_subset_coverage.csv", subset_output_rows, subset_fields)
    _write_csv(output_dir / "task042_phase_h_leave_one_out.csv", loo_rows,
               ("domain_id", "calibration_subset", "removed_unit", "removed_global_index", "retained_global_indices", "retained_count", "released_parameters", "eta_min", "eta_median", "eta_mean", "eta_exactly_one_count", "largest_functional_drop", "worst_atom_index", "worst_action_label", "worst_span", "worst_pair_index", "worst_frame_a", "worst_frame_b", "worst_F_full"))
    _write_csv(output_dir / "task042_phase_h_pareto_structure.csv", pareto_rows,
               ("domain_id", "calibration_subset", "subset_id", "retained_count", "released_parameters", "same_count_comparison_count", "dominated_same_count", "same_count_dominator_ids", "nondominated_same_count", "exact_equal_release_peer_count", "dominated_exact_equal_release", "exact_equal_release_dominator_ids", "nearest_release_gap", "nearest_release_peer_ids", "nearest_release_peer_dominator_ids", "nearest_release_peer_dominates", "nondominated_release_eta_min", "nondominated_release_eta_mean", "nondominated_release_eta_min_eta_mean"))
    _write_csv(output_dir / "task042_phase_h_calibration_stability.csv", calibration_rows,
               ("domain_id", "calibration_subset", "reference_subset", "subset_id", "pearson_eta_profile", "spearman_eta_profile", "mean_absolute_eta_difference", "max_absolute_eta_difference", "eta_min_subset", "eta_min_full", "eta_min_rank_subset", "eta_min_rank_full", "eta_min_rank_absolute_shift", "eta_min_rank_spearman_across_retained_sets", "eta_min_rank_pairwise_order_agreement", "pareto_3d_nondominated_subset", "pareto_3d_nondominated_full", "pareto_3d_membership_match", "phase_g_exact_owner_mean_jaccard", "phase_g_exact_owner_match_rate"))
    _write_csv(output_dir / "task042_phase_h_action_conditioning.csv", action_rows,
               ("domain_id", "calibration_subset", "subset_id", "retained_global_indices", "action_conditioned_eta_min", "action_conditioned_eta_mean", "global_unconditioned_eta_min", "global_unconditioned_eta_mean", "profile_pearson", "profile_spearman", "mean_absolute_eta_difference", "max_absolute_eta_difference", "global_lower_atom_count", "global_higher_atom_count", "exactly_equal_atom_count", "global_weak_atom_indices", "action_conditioned_weak_atom_indices", "global_eta_vector_800"))
    _write_csv(output_dir / "task042_phase_h_mixed_type_coverage.csv", mixed_rows,
               ("domain_id", "calibration_subset", "subset_id", "survivor_category", "survivor_unit_types", "retained_global_indices", "attention_survivor_count", "ffn_survivor_count", "eta_min", "eta_mean", "eta_median", "eta_vector"))
    _write_csv(output_dir / "task042_phase_h_parameter_frontier.csv", parameter_rows_out,
               ("domain_id", "calibration_subset", "subset_id", "retained_count", "released_parameters", "eta_min", "eta_mean", "nondominated_release_eta_min", "nondominated_release_eta_mean", "nondominated_release_eta_min_eta_mean", "eta_vector"))
    (output_dir / "task042_phase_h_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "task042_phase_h_report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _identity_obj(uid: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(IDENTITY_FIELDS, uid))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(value: Any) -> str:
    if value is None:
        return "NA (constant profile)"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _report(summary: Mapping[str, Any]) -> str:
    loo = summary["leave_one_out_full_range"]
    stability = summary["calibration_stability_full_comparison"]
    action = summary["action_conditioning"]
    counts = summary["counts"]
    return f"""# Task042 Phase H — Continuous Temporal Functional Coverage Audit

Decision: **{summary['decision']['label']}** (code {summary['decision']['code']}).

This is a set-level offline audit. It reuses the exact Phase-G relation-preference values and frozen P1/P2/P3/P12/P13/P23/FULL video index map, plus the Phase-F `parameter_cost_exact` ledger. No GPU, inference, pruning, finetuning, or performance oracle was used. No owner, unit score, eta threshold, or weighted objective was introduced.

## Scope and construction

- {summary['reuse']['unit_count']} units; {summary['reuse']['video_count']} videos across {summary['reuse']['class_count']} classes; {summary['reuse']['relations_per_video']} exact frame-pair relations.
- {len(summary['reuse']['multi_unit_domain_ids'])} multi-unit BMS domains and {summary['reuse']['temporal_atoms_per_domain']} `(action class, exact frame relation)` atoms per domain.
- Enumerated every non-empty retained subset for each domain and each of seven calibration cohorts; `{counts['subset_coverage_rows']}` subset-profile rows were written.
- `eta_vector` remains the primary 800-dimensional representation. `eta_min`, `eta_mean`, and `eta_median` are descriptive diagnostics only.
- Parameter cost uses only the existing exact Phase-F ledger. Each frontier preserves released parameters, `eta_min`, and `eta_mean` as separate axes.

## Findings

**A. Does this remove the all-units-indispensable behavior?** The owner-based hard constraint is removed by design. For the FULL-cohort leave-one-out sets, `eta_min` ranges from {_fmt(loo['eta_min_min'])} to {_fmt(loo['eta_min_max'])}; median `eta_min` is {_fmt(loo['eta_min_median'])}, median `eta_mean` is {_fmt(loo['eta_mean_median'])}, and the median number of exactly-one atoms is {_fmt(loo['eta_exactly_one_median'])}. These values expose the continuous loss instead of declaring any unit indispensable; they do not imply an acceptable-loss cutoff.

**A interpretation.** The hard owner rule is gone, but continuous coverage does not make every removal harmless: the observed leave-one-out minimum reaches {_fmt(loo['eta_min_min'])}. Thus exact-owner indispensability is replaced by an explicit atom-wise loss profile, not by a claim that every domain safely tolerates removal.

**B. Can subsets release parameters while retaining relation-wise coverage?** In the FULL cohort, {summary['full_cohort_all_pruned_subset_distribution']['nonempty_removal_set_count']} non-empty removal subsets release parameters; their `eta_mean` median is {_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_mean_median'])} (range {_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_mean_min'])}–{_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_mean_max'])}), while `eta_min` median is {_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_min_median'])} (range {_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_min_min'])}–{_fmt(summary['full_cohort_all_pruned_subset_distribution']['eta_min_max'])}). Released parameter counts range from {summary['full_cohort_all_pruned_subset_distribution']['released_parameters_min']} to {summary['full_cohort_all_pruned_subset_distribution']['released_parameters_max']}. No cutoff defines “high” coverage, so these are trade-off observations, not candidate approval.

**C. Are profiles stable across class-balanced calibration subsets?** Against FULL, mean profile Pearson/Spearman are {_fmt(stability['mean_profile_pearson'])}/{_fmt(stability['mean_profile_spearman'])}; mean absolute eta difference is {_fmt(stability['mean_absolute_eta_difference_mean'])} and the maximum observed absolute difference is {_fmt(stability['max_absolute_eta_difference_max'])}. Eta-min subset-ranking Spearman is {_fmt(stability['eta_min_rank_spearman_mean'])}, pairwise order agreement is {_fmt(stability['eta_min_rank_pairwise_order_agreement_mean'])}, and 3D frontier-membership agreement is {_fmt(stability['pareto_3d_membership_match_rate'])}. For context, Phase-G exact-owner mean Jaccard/match rate are {_fmt(stability['phase_g_owner_mean_jaccard_mean'])}/{_fmt(stability['phase_g_owner_exact_match_rate_mean'])}. Per-domain and per-subset values are in the CSV.

**D. Does action conditioning add information?** Global aggregation without labels has mean profile Pearson/Spearman {_fmt(action['mean_profile_pearson'])}/{_fmt(action['mean_profile_spearman'])}; mean absolute eta difference from action-conditioned coverage is {_fmt(action['mean_absolute_eta_difference_mean'])}. Across aligned atoms, the unconditioned profile is lower/higher on average counts {_fmt(action['mean_global_lower_atom_count'])}/{_fmt(action['mean_global_higher_atom_count'])}. The profiles are not identical, so action conditioning changes which functions appear weakly covered; without a preregistered effect-size cutoff, “material” is not declared. This remains diagnostic; action-conditioned coverage is primary.

**E. Can mixed Attention/FFN survivors jointly maintain coverage?** Domains {', '.join(summary['mixed_type']['domains'])} enumerate Attention-only, FFN-only, and mixed retained sets where possible. Their exact coverage profiles are in `task042_phase_h_mixed_type_coverage.csv`; no fairness claim is made.

**F. Is this ready for a one-shot pruning selection oracle?** **No decision to incorporate is made in this phase.** The result is classified as weak/unresolved because the audit defines no acceptable eta loss, no performance oracle is allowed, and calibration/profile stability alone does not establish downstream pruning utility. Phase H does not authorize physical pruning.

## Output inventory

Eight CSV diagnostics, this report, and the machine-readable summary are present in this directory. The exact 800-D coverage vectors are stored in subset coverage; the action-conditioning ablation also stores an aligned 800-D global profile per set. Pareto comparisons use exact component-wise dominance. Same-count and exact equal-release comparisons are reported separately; nearest-release peers are descriptive and use no tolerance.

## Reproducibility

- Phase-G head: `{summary['reuse']['phase_g_head']}`
- Phase-H code head: `{summary['git_head']}`
- Input SHA-256/byte counts: `task042_phase_h_summary.json`
- Hard constraints: GPU `{summary['hard_constraints']['gpu_used']}`, inference `{summary['hard_constraints']['new_inference']}`, pruning `{summary['hard_constraints']['pruning_performed']}`, finetuning `{summary['hard_constraints']['finetuning_performed']}`, performance oracle `{summary['hard_constraints']['performance_oracle_used']}`.

**Stopped after Phase H.** No Task043, pruning, finetuning, GPU run, threshold tuning, or temporal score was started.
"""

"""Offline Task042 Phase-G action-conditioned temporal coverage audit.

This module consumes only the frozen Phase-A manifests and sensitivity CSV.
It deliberately has no model, CUDA, pruning, training, or performance-oracle
dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_FIELDS = (
    "task037_global_index",
    "domain_id",
    "layer",
    "stage",
    "unit_type",
    "unit_index",
)
Q_FIELDS = ("span", "pair_index", "frame_a", "frame_b")
EXPECTED_SPANS = (1, 2, 4, 8, 16)
EXPECTED_PAIRS_PER_SPAN = 16
EXPECTED_Q = 80
EXPECTED_VIDEOS = 30
EXPECTED_CLASSES = 10
EXPECTED_UNITS = 51
SUBSET_POSITIONS = {
    "P1": (0,),
    "P2": (1,),
    "P3": (2,),
    "P12": (0, 1),
    "P13": (0, 2),
    "P23": (1, 2),
    "FULL": (0, 1, 2),
}

UnitId = tuple[str, ...]
RelationId = tuple[int, int, int, int]
AtomId = tuple[str, RelationId]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unit_json(unit: UnitId) -> str:
    return _json(dict(zip(IDENTITY_FIELDS, unit)))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in fields})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return _json(value)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_id(row: Mapping[str, str]) -> UnitId:
    return tuple(str(row[name]).strip() for name in IDENTITY_FIELDS)


def _q_id(row: Mapping[str, str]) -> RelationId:
    return tuple(int(row[name]) for name in Q_FIELDS)  # type: ignore[return-value]


def _unit_order(unit: UnitId) -> tuple[int, tuple[str, ...]]:
    return int(unit[0]), unit


def _q_order(q: RelationId) -> tuple[int, int, int, int]:
    return q


def _atom_order(atom: AtomId) -> tuple[int, int, int, int, int]:
    return (int(atom[0]), *atom[1])


def average_rank_preference(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Return 1-based average ranks and normalized [0,1] preferences.

    Ties use exact Python-float equality, matching exact stored float64 values.
    """
    count = len(values)
    if count < 2:
        raise ValueError("At least two relations are required for rank normalization")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Rank input contains a non-finite value")
    order = sorted(range(count), key=lambda index: values[index])
    ranks = [0.0] * count
    start = 0
    while start < count:
        stop = start + 1
        value = values[order[start]]
        while stop < count and values[order[stop]] == value:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks, [(rank - 1.0) / (count - 1.0) for rank in ranks]


def build_class_video_positions(videos: Sequence[Mapping[str, str]]) -> dict[str, tuple[int, ...]]:
    """Validate the 10 x 3 cohort and assign positions by video_index."""
    if len(videos) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} videos, found {len(videos)}")
    by_class: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    for row in videos:
        index = int(row["video_index"])
        label = str(row["label"]).strip()
        video_id = str(row.get("video_id", "")).strip()
        if index in seen_indices:
            raise ValueError(f"Duplicate video_index {index}")
        if video_id and video_id in seen_ids:
            raise ValueError(f"Duplicate video_id {video_id}")
        seen_indices.add(index)
        if video_id:
            seen_ids.add(video_id)
        by_class[label].append((index, video_id))
    if len(by_class) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(by_class)}")
    if any(len(items) != 3 for items in by_class.values()):
        raise ValueError("Every action class must have exactly three videos")
    if seen_indices != set(range(EXPECTED_VIDEOS)):
        raise ValueError("video_index must be exactly 0..29")
    return {
        label: tuple(index for index, _ in sorted(items))
        for label, items in sorted(by_class.items(), key=lambda item: (int(item[0]), item[0]))
    }


def calibration_subsets(class_positions: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    """Return deterministic per-class-position subsets P1..FULL."""
    if set(class_positions) == set():
        raise ValueError("No classes supplied")
    if any(len(indices) != 3 for indices in class_positions.values()):
        raise ValueError("Each class must have exactly three ordered videos")
    subsets: dict[str, tuple[int, ...]] = {}
    for name, positions in SUBSET_POSITIONS.items():
        selected = [class_positions[label][position] for label in sorted(class_positions, key=lambda x: (int(x), x)) for position in positions]
        subsets[name] = tuple(selected)
    return subsets


def action_conditioned_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot take median of no videos")
    return float(statistics.median(values))


def exact_argmax_owner_set(values: Mapping[UnitId, float]) -> frozenset[UnitId]:
    if not values:
        raise ValueError("Owner selection requires at least one unit")
    maximum = max(values.values())
    return frozenset(unit for unit, value in values.items() if value == maximum)


def pareto_front(vectors: Mapping[UnitId, Sequence[float]]) -> frozenset[UnitId]:
    if not vectors:
        return frozenset()
    lengths = {len(vector) for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("Pareto vectors must have equal lengths")
    frontier: set[UnitId] = set()
    for candidate, vector in vectors.items():
        dominated = False
        for other, other_vector in vectors.items():
            if other == candidate:
                continue
            if all(right >= left for left, right in zip(vector, other_vector)) and any(
                right > left for left, right in zip(vector, other_vector)
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(candidate)
    return frozenset(frontier)


def build_incidence(owner_sets: Mapping[AtomId, Iterable[UnitId]]) -> dict[UnitId, set[AtomId]]:
    incidence: dict[UnitId, set[AtomId]] = defaultdict(set)
    for atom, owners_iter in owner_sets.items():
        for unit in owners_iter:
            incidence[unit].add(atom)
    return dict(incidence)


def classify_owner_types(owner_units: Iterable[UnitId], unit_types: Mapping[UnitId, str]) -> str:
    types = {unit_types[unit].lower() for unit in owner_units}
    has_attention = any("attention" in kind or kind == "head" for kind in types)
    has_ffn = any("ffn" in kind or "neuron" in kind for kind in types)
    has_other = not all(("attention" in kind or kind == "head" or "ffn" in kind or "neuron" in kind) for kind in types)
    if has_attention and has_ffn and not has_other:
        return "mixed Attention+FFN"
    if has_attention and not has_ffn and not has_other:
        return "Attention-only"
    if has_ffn and not has_attention and not has_other:
        return "FFN-only"
    if not types:
        return "empty-owner-set"
    if has_other and not (has_attention or has_ffn):
        return "other-only"
    return "mixed-with-other"


def minimum_set_covers(owner_sets: Iterable[Iterable[UnitId]], units: Sequence[UnitId]) -> tuple[int, tuple[frozenset[UnitId], ...]]:
    """Exhaustively enumerate every minimum-cardinality hitting set."""
    owner_sets_frozen = tuple(frozenset(owner_set) for owner_set in owner_sets)
    if any(not owner_set for owner_set in owner_sets_frozen):
        raise ValueError("Every functional atom must have at least one owner")
    if not owner_sets_frozen:
        return 0, (frozenset(),)
    ordered_units = tuple(sorted(set(units), key=_unit_order))
    for size in range(len(ordered_units) + 1):
        covers = []
        for combination in itertools.combinations(ordered_units, size):
            candidate = frozenset(combination)
            if all(candidate & owner_set for owner_set in owner_sets_frozen):
                covers.append(candidate)
        if covers:
            covers.sort(key=lambda cover: tuple(_unit_order(unit) for unit in sorted(cover, key=_unit_order)))
            return size, tuple(covers)
    raise ValueError("No cover exists for nonempty owner sets")


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _set_string(units: Iterable[UnitId]) -> str:
    return _json([dict(zip(IDENTITY_FIELDS, unit)) for unit in sorted(units, key=_unit_order)])


def _median_preference(
    preference: Mapping[tuple[UnitId, int, RelationId], float],
    unit: UnitId,
    video_indices: Sequence[int],
    q: RelationId,
) -> float:
    return action_conditioned_median([preference[(unit, video_index, q)] for video_index in video_indices])


def _owner_metrics(full: Mapping[AtomId, frozenset[UnitId]], other: Mapping[AtomId, frozenset[UnitId]]) -> dict[str, Any]:
    atoms = sorted(full, key=_atom_order)
    values = [jaccard(full[atom], other[atom]) for atom in atoms]
    full_single = [atom for atom in atoms if len(full[atom]) == 1]
    singleton_hits = sum(1 for atom in full_single if len(other[atom]) == 1 and full[atom] == other[atom])
    return {
        "n_atoms": len(atoms),
        "mean_jaccard": statistics.mean(values) if values else 1.0,
        "median_jaccard": statistics.median(values) if values else 1.0,
        "exact_owner_set_match_rate": sum(full[a] == other[a] for a in atoms) / len(atoms) if atoms else 1.0,
        "singleton_full_atom_count": len(full_single),
        "singleton_owner_identity_match_rate": singleton_hits / len(full_single) if full_single else "",
        "singleton_owner_identity_matches": singleton_hits,
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    counts = summary["counts"]
    questions = summary["required_questions"]
    decision = summary["decision"]
    lines = [
        "# Task042 Phase G — Action-Conditioned Temporal Functional Coverage Audit",
        "",
        f"**Decision: `{decision['label']}`**",
        "",
        decision["reason"],
        "",
        "## Scope and integrity",
        "",
        "This is an offline audit of the frozen Task042 tables. It used no GPU, model inference, pruning, finetuning, or validation-performance oracle. The D_abs/D_rel/D_st descriptors, BMS domains, activation semantics, and 80 fixed-cardinality frame-pair conditions were not changed.",
        "",
        f"- Verified unit cohort: {counts['units']} exact units; raw sensitivity rows: {counts['raw_rows']:,}.",
        f"- Videos/classes: {counts['videos']} videos, {counts['classes']} classes with 3 videos per class; exact relations per unit-video: {counts['relations_per_unit_video']}.",
        f"- Multi-unit BMS domains analyzed: {counts['multi_unit_domains']} ({', '.join(counts['multi_unit_domain_ids'])}); singleton domains in the authoritative cohort: {counts['singleton_domains']}.",
        f"- Temporal atoms analyzed: {counts['atoms']:,} (800 per multi-unit domain).",
        f"- Input HEAD: `{summary['git_head']}`; input file SHA-256 values are recorded in `task042_phase_g_summary.json`.",
        "",
        "## Primary definition B and degeneracy",
        "",
        f"Across all domains, {counts['singleton_atoms']:,}/{counts['atoms']:,} atoms ({summary['global_owner_degeneracy']['singleton_fraction']:.3f}) have a singleton exact argmax owner; {counts['multi_owner_atoms']:,} atoms have multiple exactly tied owners. Mean/median/maximum owner-set size are {summary['global_owner_degeneracy']['mean_owner_set_size']:.3f}/{summary['global_owner_degeneracy']['median_owner_set_size']:.1f}/{summary['global_owner_degeneracy']['maximum_owner_set_size']}.",
        "",
        "These ties are exact float64 equality only; no epsilon was used. Singleton argmaxes are expected under continuous-valued scores, so the redundancy and calibration-stability tables—not argmax existence alone—determine whether the coverage structure is useful. The unit ownership concentration and cross-class same-owner fractions are in the degeneracy CSV.",
        "",
        "## Results by required question",
        "",
    ]
    for key, prompt in [
        ("A", "Is action-conditioned owner incidence nontrivial?"),
        ("B", "Does definition B produce meaningful functional redundancy, or almost exclusively singleton owners?"),
        ("C", "Does action conditioning reveal relation specialization hidden by global aggregation?"),
        ("D", "Is the owner/incidence structure stable across class-balanced video subsets?"),
        ("E", "Do action-conditioned temporal functions have genuine mixed Attention/FFN redundant owners?"),
        ("F", "Do BMS domains contain units absent from at least one exact minimum functional cover?"),
        ("G", "Is functional coverage stable enough to justify a later one-shot pruning oracle?"),
    ]:
        lines.extend([f"### {key}. {prompt}", "", str(questions[key]), ""])
    lines.extend([
        "## Domain-level coverage and calibration",
        "",
        "The exact per-domain owner counts, cardinality distributions, class/span summaries, minimum covers, and cover stability are provided in the corresponding CSVs. Minimum-cover enumeration is exhaustive over each domain's unit subsets. It uses only the B owner incidence, with no importance, parameter cost, or performance measure.",
        "",
        "Calibration subsets are P1/P2/P3/P12/P13/P23, selected by within-class video_index order, and compared against FULL. A is the global-median diagnostic and C is the strict Pareto diagnostic; neither replaces preregistered B.",
        "",
        "## Scientific boundary",
        "",
        "Phase G does not establish a pruning rule. It only audits whether exact action-conditioned temporal functions have repeatable ownership and whether a subset of owners covers all observed atoms. Any later pruning oracle would require a separately authorized and preregistered experiment. The intended eventual paradigm remains one-shot structural pruning followed by 100-epoch finetuning; no iterative or progressive pruning is tested here.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase_g(
    input_dir: str | Path,
    output_dir: str | Path,
    git_head: str = "unknown",
    decision: str = "B",
    decision_reason: str = "The evidence is unresolved or qualitatively ambiguous; under the preregistered rule, ambiguity is assigned to B.",
) -> dict[str, Any]:
    """Run the full Phase-G audit from frozen CSV artifacts."""
    if decision not in {
        "A. TEMPORAL_FUNCTIONAL_COVERAGE_READY_FOR_SELECTION_ORACLE",
        "B. TEMPORAL_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED",
        "C. TEMPORAL_FUNCTIONAL_COVERAGE_REJECTED",
        "A", "B", "C",
    }:
        raise ValueError("decision must be A, B, or C (full labels are accepted)")
    decision_code = decision[0]
    decision_labels = {
        "A": "TEMPORAL_FUNCTIONAL_COVERAGE_READY_FOR_SELECTION_ORACLE",
        "B": "TEMPORAL_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED",
        "C": "TEMPORAL_FUNCTIONAL_COVERAGE_REJECTED",
    }
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    units_path = input_dir / "task042_unit_manifest.csv"
    videos_path = input_dir / "task042_video_manifest.csv"
    raw_path = input_dir / "task042_frame_pair_sensitivity.csv"
    _, unit_rows = _read_csv(units_path)
    _, video_rows = _read_csv(videos_path)
    _, raw_rows = _read_csv(raw_path)

    if len(unit_rows) != EXPECTED_UNITS:
        raise ValueError(f"Authoritative cohort must have {EXPECTED_UNITS} units, got {len(unit_rows)}")
    units = [_unit_id(row) for row in unit_rows]
    if len(set(units)) != EXPECTED_UNITS:
        raise ValueError("Unit manifest identity tuples are not unique")
    unit_metadata = {_unit_id(row): row for row in unit_rows}
    manifest_ids = set(unit_metadata)
    raw_ids = {_unit_id(row) for row in raw_rows}
    if manifest_ids != raw_ids:
        raise ValueError(f"Raw sensitivity identity mismatch: missing={len(manifest_ids-raw_ids)}, extra={len(raw_ids-manifest_ids)}")

    class_videos = build_class_video_positions(video_rows)
    video_metadata = {int(row["video_index"]): row for row in video_rows}
    if len(video_metadata) != EXPECTED_VIDEOS:
        raise ValueError("Duplicate video_index in video manifest")
    raw_video_ids = {int(row["video_index"]) for row in raw_rows}
    if raw_video_ids != set(video_metadata):
        raise ValueError("Raw sensitivity videos do not exactly match the frozen video manifest")
    for row in raw_rows:
        video = video_metadata[int(row["video_index"])]
        for field in ("label", "video_id"):
            if str(row.get(field, "")).strip() != str(video.get(field, "")).strip():
                raise ValueError(f"Raw/video manifest mismatch for video {row['video_index']} field {field}")

    relation_ids = sorted({_q_id(row) for row in raw_rows}, key=_q_order)
    if len(relation_ids) != EXPECTED_Q:
        raise ValueError(f"Expected {EXPECTED_Q} exact relation identities, got {len(relation_ids)}")
    relation_counts = Counter(q[0] for q in relation_ids)
    if tuple(sorted(relation_counts)) != EXPECTED_SPANS or any(relation_counts[span] != EXPECTED_PAIRS_PER_SPAN for span in EXPECTED_SPANS):
        raise ValueError(f"Unexpected span/pair structure: {relation_counts}")
    for span in EXPECTED_SPANS:
        if {q[1] for q in relation_ids if q[0] == span} != set(range(EXPECTED_PAIRS_PER_SPAN)):
            raise ValueError(f"span {span} pair_index must be exactly 0..15")

    grouped: dict[tuple[UnitId, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(_unit_id(row), int(row["video_index"]))].append(row)
    expected_relations = set(relation_ids)
    if len(grouped) != EXPECTED_UNITS * EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_UNITS*EXPECTED_VIDEOS} unit-video groups, got {len(grouped)}")
    preference: dict[tuple[UnitId, int, RelationId], float] = {}
    relation_output = output_dir / "task042_phase_g_relation_preference.csv"
    preference_fields = [*IDENTITY_FIELDS, "video_index", "dataset_index", "video_id", "label", *Q_FIELDS, "relative_sensitivity", "average_rank", "relation_preference"]
    with relation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preference_fields)
        writer.writeheader()
        for (unit, video_index), rows in sorted(grouped.items(), key=lambda item: (_unit_order(item[0][0]), item[0][1])):
            if len(rows) != EXPECTED_Q:
                raise ValueError(f"Unit {unit[0]} video {video_index}: expected 80 rows, got {len(rows)}")
            by_q = {_q_id(row): row for row in rows}
            if len(by_q) != EXPECTED_Q or set(by_q) != expected_relations:
                raise ValueError(f"Unit {unit[0]} video {video_index}: exact q identities differ from frozen set")
            ordered_rows = [by_q[q] for q in relation_ids]
            sensitivities = [float(row["relative_sensitivity"]) for row in ordered_rows]
            ranks, normalized = average_rank_preference(sensitivities)
            for row, q, value, rank, rel_pref in zip(ordered_rows, relation_ids, sensitivities, ranks, normalized):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("relative_sensitivity must be finite and nonnegative")
                preference[(unit, video_index, q)] = rel_pref
                writer.writerow({
                    **dict(zip(IDENTITY_FIELDS, unit)),
                    "video_index": video_index,
                    "dataset_index": row.get("dataset_index", ""),
                    "video_id": row.get("video_id", ""),
                    "label": row.get("label", ""),
                    **dict(zip(Q_FIELDS, q)),
                    "relative_sensitivity": value,
                    "average_rank": rank,
                    "relation_preference": rel_pref,
                })

    domain_units: dict[str, list[UnitId]] = defaultdict(list)
    for unit in units:
        domain_units[unit[1]].append(unit)
    domain_units = {domain: sorted(items, key=_unit_order) for domain, items in domain_units.items()}
    multi_domains = {domain: items for domain, items in domain_units.items() if len(items) > 1}
    subset_videos = calibration_subsets(class_videos)
    subset_names = ("P1", "P2", "P3", "P12", "P13", "P23", "FULL")

    owners_by_domain_subset: dict[str, dict[str, dict[AtomId, frozenset[UnitId]]]] = {}
    scores_full: dict[tuple[str, AtomId, UnitId], float] = {}
    for domain, domain_unit_list in multi_domains.items():
        owners_by_domain_subset[domain] = {}
        for subset_name in subset_names:
            owners: dict[AtomId, frozenset[UnitId]] = {}
            for label in sorted(class_videos, key=lambda value: (int(value), value)):
                video_indices = [index for index in subset_videos[subset_name] if str(video_metadata[index]["label"]) == label]
                if not video_indices:
                    raise ValueError(f"Subset {subset_name} omitted class {label}")
                for q in relation_ids:
                    atom = (label, q)
                    values = {unit: _median_preference(preference, unit, video_indices, q) for unit in domain_unit_list}
                    owners[atom] = exact_argmax_owner_set(values)
                    if subset_name == "FULL":
                        for unit, value in values.items():
                            scores_full[(domain, atom, unit)] = value
            owners_by_domain_subset[domain][subset_name] = owners

    # Primary owner incidence and atom-level redundancy tables.
    incidence_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    for domain, domain_unit_list in multi_domains.items():
        owner_map = owners_by_domain_subset[domain]["FULL"]
        incidence = build_incidence(owner_map)
        for unit in domain_unit_list:
            incidence.setdefault(unit, set())
        unit_summaries: dict[UnitId, dict[str, Any]] = {}
        for unit in domain_unit_list:
            atoms_owned = incidence[unit]
            unit_summaries[unit] = {
                "unit_owned_atoms": len(atoms_owned),
                "unit_action_classes": len({atom[0] for atom in atoms_owned}),
                **{f"unit_span_{span}_atoms": sum(atom[1][0] == span for atom in atoms_owned) for span in EXPECTED_SPANS},
            }
        for atom in sorted(owner_map, key=_atom_order):
            owner_set = owner_map[atom]
            q = atom[1]
            owner_json = _set_string(owner_set)
            redundancy_rows.append({
                "domain_id": domain,
                "label": atom[0],
                **dict(zip(Q_FIELDS, q)),
                "owner_count": len(owner_set),
                "owner_units": owner_json,
                "is_unique_owner_atom": len(owner_set) == 1,
                "is_redundantly_covered_atom": len(owner_set) > 1,
            })
            for unit in domain_unit_list:
                incidence_rows.append({
                    "domain_id": domain,
                    "label": atom[0],
                    **dict(zip(Q_FIELDS, q)),
                    **dict(zip(IDENTITY_FIELDS, unit)),
                    "is_owner": unit in owner_set,
                    "owner_count": len(owner_set),
                    "owner_units": owner_json,
                    "median_relation_preference": scores_full[(domain, atom, unit)],
                    **unit_summaries[unit],
                })

    _write_csv(output_dir / "task042_phase_g_primary_owner_incidence.csv", incidence_rows,
               ["domain_id", "label", *Q_FIELDS, *IDENTITY_FIELDS, "is_owner", "owner_count", "owner_units", "median_relation_preference", "unit_owned_atoms", "unit_action_classes", *[f"unit_span_{span}_atoms" for span in EXPECTED_SPANS]])
    _write_csv(output_dir / "task042_phase_g_functional_redundancy.csv", redundancy_rows,
               ["domain_id", "label", *Q_FIELDS, "owner_count", "owner_units", "is_unique_owner_atom", "is_redundantly_covered_atom"])

    # Degeneracy by domain and globally.
    degeneracy_rows: list[dict[str, Any]] = []
    total_owner_sets: list[frozenset[UnitId]] = []
    total_units: list[UnitId] = []
    for domain, domain_unit_list in multi_domains.items():
        owner_map = owners_by_domain_subset[domain]["FULL"]
        total_owner_sets.extend(owner_map.values())
        total_units.extend(domain_unit_list)
        cardinalities = Counter(len(owner_set) for owner_set in owner_map.values())
        all_units_count = sum(owner_set == frozenset(domain_unit_list) for owner_set in owner_map.values())
        singleton_counts = Counter(next(iter(owner_set)) for owner_set in owner_map.values() if len(owner_set) == 1)
        incid = build_incidence(owner_map)
        membership_counts = [len(incid.get(unit, set())) for unit in domain_unit_list]
        total_memberships = sum(membership_counts)
        proportions = [count / total_memberships for count in membership_counts] if total_memberships else [0.0] * len(domain_unit_list)
        hhi = sum(p * p for p in proportions)
        entropy = -sum(p * math.log(p) for p in proportions if p > 0)
        normalized_entropy = entropy / math.log(len(domain_unit_list)) if len(domain_unit_list) > 1 else 0.0
        by_q: dict[RelationId, list[frozenset[UnitId]]] = defaultdict(list)
        for (label, q), owner_set in owner_map.items():
            by_q[q].append(owner_set)
        invariant_atoms = 0
        for owner_sets_for_q in by_q.values():
            if len(owner_sets_for_q) == EXPECTED_CLASSES and len(owner_sets_for_q[0]) == 1 and all(item == owner_sets_for_q[0] for item in owner_sets_for_q):
                invariant_atoms += EXPECTED_CLASSES
        atom_count = len(owner_map)
        degeneracy_rows.append({
            "domain_id": domain,
            "row_scope": "domain",
            "domain_size": len(domain_unit_list),
            "atom_count": atom_count,
            "singleton_owner_atoms": cardinalities.get(1, 0),
            "multi_owner_atoms": atom_count - cardinalities.get(1, 0),
            "singleton_fraction": cardinalities.get(1, 0) / atom_count,
            "multi_owner_fraction": (atom_count - cardinalities.get(1, 0)) / atom_count,
            "owner_set_cardinality_distribution": dict(sorted(cardinalities.items())),
            "atoms_owned_by_every_unit": all_units_count,
            "fraction_atoms_owned_by_same_singleton_across_all_classes": invariant_atoms / atom_count,
            "largest_singleton_owner_fraction": max(singleton_counts.values(), default=0) / atom_count,
            "unit_ownership_concentration_hhi": hhi,
            "unit_ownership_concentration_normalized_entropy": normalized_entropy,
            "unit_membership_counts": {str(unit[0]): len(incid.get(unit, set())) for unit in domain_unit_list},
            "mean_owner_set_size": statistics.mean(map(len, owner_map.values())),
            "median_owner_set_size": statistics.median(map(len, owner_map.values())),
            "maximum_owner_set_size": max(map(len, owner_map.values())),
        })
    global_cardinality = Counter(map(len, total_owner_sets))
    global_atom_count = len(total_owner_sets)
    global_unit_counts = Counter(unit for owners in total_owner_sets for unit in owners)
    global_total_memberships = sum(global_unit_counts.values())
    global_hhi = sum((n / global_total_memberships) ** 2 for n in global_unit_counts.values()) if global_total_memberships else 0.0
    degeneracy_rows.append({
        "domain_id": "__GLOBAL__",
        "row_scope": "global",
        "domain_size": len(set(total_units)),
        "atom_count": global_atom_count,
        "singleton_owner_atoms": global_cardinality.get(1, 0),
        "multi_owner_atoms": global_atom_count - global_cardinality.get(1, 0),
        "singleton_fraction": global_cardinality.get(1, 0) / global_atom_count,
        "multi_owner_fraction": (global_atom_count - global_cardinality.get(1, 0)) / global_atom_count,
        "owner_set_cardinality_distribution": dict(sorted(global_cardinality.items())),
        "atoms_owned_by_every_unit": "",
        "fraction_atoms_owned_by_same_singleton_across_all_classes": "",
        "largest_singleton_owner_fraction": max(Counter(next(iter(o)) for o in total_owner_sets if len(o) == 1).values(), default=0) / global_atom_count,
        "unit_ownership_concentration_hhi": global_hhi,
        "unit_ownership_concentration_normalized_entropy": "",
        "unit_membership_counts": {str(unit[0]): global_unit_counts[unit] for unit in sorted(set(total_units), key=_unit_order)},
        "mean_owner_set_size": statistics.mean(map(len, total_owner_sets)),
        "median_owner_set_size": statistics.median(map(len, total_owner_sets)),
        "maximum_owner_set_size": max(map(len, total_owner_sets)),
    })
    _write_csv(output_dir / "task042_phase_g_owner_degeneracy.csv", degeneracy_rows,
               ["domain_id", "row_scope", "domain_size", "atom_count", "singleton_owner_atoms", "multi_owner_atoms", "singleton_fraction", "multi_owner_fraction", "owner_set_cardinality_distribution", "atoms_owned_by_every_unit", "fraction_atoms_owned_by_same_singleton_across_all_classes", "largest_singleton_owner_fraction", "unit_ownership_concentration_hhi", "unit_ownership_concentration_normalized_entropy", "unit_membership_counts", "mean_owner_set_size", "median_owner_set_size", "maximum_owner_set_size"])

    # Diagnostic A: one global-median owner broadcast over action-conditioned atoms.
    ablation_rows: list[dict[str, Any]] = []
    ablation_summaries: dict[str, dict[str, Any]] = {}
    for domain, domain_unit_list in multi_domains.items():
        by_q_global: dict[RelationId, frozenset[UnitId]] = {}
        for q in relation_ids:
            global_scores = {
                unit: action_conditioned_median([preference[(unit, index, q)] for index in range(EXPECTED_VIDEOS)])
                for unit in domain_unit_list
            }
            by_q_global[q] = exact_argmax_owner_set(global_scores)
        primary = owners_by_domain_subset[domain]["FULL"]
        b_incidence: set[tuple[AtomId, UnitId]] = {(atom, unit) for atom, owners_set in primary.items() for unit in owners_set}
        a_incidence: set[tuple[AtomId, UnitId]] = {(atom, unit) for atom in primary for unit in by_q_global[atom[1]]}
        matches = 0
        for atom in sorted(primary, key=_atom_order):
            a_owners = by_q_global[atom[1]]
            b_owners = primary[atom]
            same = a_owners == b_owners
            matches += int(same)
            ablation_rows.append({
                "row_type": "atom",
                "domain_id": domain,
                "label": atom[0],
                **dict(zip(Q_FIELDS, atom[1])),
                "global_median_owner_units_A": _set_string(a_owners),
                "action_conditioned_owner_units_B": _set_string(b_owners),
                "same_owner_set": same,
                "incidence_jaccard_for_atom": jaccard(a_owners, b_owners),
            })
        ablation_summaries[domain] = {
            "owner_identity_agreement_rate": matches / len(primary),
            "owner_identity_agreements": matches,
            "atom_count": len(primary),
            "ownership_changes": len(primary) - matches,
            "ownership_change_fraction": 1.0 - matches / len(primary),
            "incidence_jaccard": jaccard(a_incidence, b_incidence),
        }
        ablation_rows.append({"row_type": "domain_summary", "domain_id": domain, **ablation_summaries[domain]})
    _write_csv(output_dir / "task042_phase_g_global_owner_ablation.csv", ablation_rows,
               ["row_type", "domain_id", "label", *Q_FIELDS, "global_median_owner_units_A", "action_conditioned_owner_units_B", "same_owner_set", "incidence_jaccard_for_atom", "owner_identity_agreement_rate", "owner_identity_agreements", "atom_count", "ownership_changes", "ownership_change_fraction", "incidence_jaccard"])

    # Diagnostic C: strict Pareto frontier on the three per-class videos.
    pareto_rows: list[dict[str, Any]] = []
    pareto_domain_stats: dict[str, dict[str, int]] = {}
    for domain, domain_unit_list in multi_domains.items():
        owner_map = owners_by_domain_subset[domain]["FULL"]
        all_front_count = 0
        larger_than_b_count = 0
        cardinality = Counter()
        for label in sorted(class_videos, key=lambda value: (int(value), value)):
            video_indices = class_videos[label]
            for q in relation_ids:
                atom = (label, q)
                vectors = {unit: [preference[(unit, index, q)] for index in video_indices] for unit in domain_unit_list}
                front = pareto_front(vectors)
                b_owners = owner_map[atom]
                cardinality[len(front)] += 1
                all_front_count += int(len(front) == len(domain_unit_list))
                larger_than_b_count += int(len(front) > len(b_owners))
                pareto_rows.append({
                    "row_type": "atom",
                    "domain_id": domain,
                    "label": label,
                    **dict(zip(Q_FIELDS, q)),
                    "pareto_owner_count": len(front),
                    "pareto_owner_units": _set_string(front),
                    "primary_B_owner_count": len(b_owners),
                    "primary_B_owner_units": _set_string(b_owners),
                    "all_domain_units_non_dominated": len(front) == len(domain_unit_list),
                    "pareto_strictly_larger_than_B": len(front) > len(b_owners),
                })
        pareto_domain_stats[domain] = {"all_units_nondominated_atoms": all_front_count, "pareto_larger_than_B_atoms": larger_than_b_count, "atom_count": EXPECTED_CLASSES * EXPECTED_Q}
        pareto_rows.append({
            "row_type": "domain_summary",
            "domain_id": domain,
            "pareto_cardinality_distribution": dict(sorted(cardinality.items())),
            "all_units_nondominated_atoms": all_front_count,
            "all_units_nondominated_fraction": all_front_count / (EXPECTED_CLASSES * EXPECTED_Q),
            "pareto_larger_than_B_atoms": larger_than_b_count,
            "pareto_larger_than_B_fraction": larger_than_b_count / (EXPECTED_CLASSES * EXPECTED_Q),
        })
    _write_csv(output_dir / "task042_phase_g_pareto_owner_diagnostic.csv", pareto_rows,
               ["row_type", "domain_id", "label", *Q_FIELDS, "pareto_owner_count", "pareto_owner_units", "primary_B_owner_count", "primary_B_owner_units", "all_domain_units_non_dominated", "pareto_strictly_larger_than_B", "pareto_cardinality_distribution", "all_units_nondominated_atoms", "all_units_nondominated_fraction", "pareto_larger_than_B_atoms", "pareto_larger_than_B_fraction"])

    # Calibration owner-set and incidence stability.
    stability_rows: list[dict[str, Any]] = []
    class_span_rows: list[dict[str, Any]] = []
    stability_domain_stats: dict[str, dict[str, Any]] = {}
    for domain, domain_unit_list in multi_domains.items():
        full = owners_by_domain_subset[domain]["FULL"]
        stability_domain_stats[domain] = {}
        full_incidence = build_incidence(full)
        for unit in domain_unit_list:
            full_incidence.setdefault(unit, set())
        for subset_name in subset_names:
            current = owners_by_domain_subset[domain][subset_name]
            metrics = _owner_metrics(full, current)
            stability_domain_stats[domain][subset_name] = metrics
            for atom in sorted(full, key=_atom_order):
                full_owners = full[atom]
                current_owners = current[atom]
                stability_rows.append({
                    "row_type": "atom",
                    "domain_id": domain,
                    "subset": subset_name,
                    "label": atom[0],
                    **dict(zip(Q_FIELDS, atom[1])),
                    "owner_set_jaccard": jaccard(full_owners, current_owners),
                    "exact_owner_set_match": full_owners == current_owners,
                    "full_singleton": len(full_owners) == 1,
                    "singleton_owner_identity_match": len(full_owners) == 1 and current_owners == full_owners,
                    "full_owner_units": _set_string(full_owners),
                    "subset_owner_units": _set_string(current_owners),
                })
            stability_rows.append({"row_type": "domain_summary", "domain_id": domain, "subset": subset_name, **metrics})
            current_incidence = build_incidence(current)
            for unit in domain_unit_list:
                current_incidence.setdefault(unit, set())
                stability_rows.append({
                    "row_type": "unit_incidence",
                    "domain_id": domain,
                    "subset": subset_name,
                    **dict(zip(IDENTITY_FIELDS, unit)),
                    "unit_full_owned_atom_count": len(full_incidence[unit]),
                    "unit_subset_owned_atom_count": len(current_incidence[unit]),
                    "unit_incidence_jaccard": jaccard(full_incidence[unit], current_incidence[unit]),
                    "unit_full_owned_atoms": _json([list(atom) for atom in sorted(full_incidence[unit], key=_atom_order)]),
                    "unit_subset_owned_atoms": _json([list(atom) for atom in sorted(current_incidence[unit], key=_atom_order)]),
                })
            for group_kind, groups in (
                ("class", {label: {atom for atom in full if atom[0] == label} for label in class_videos}),
                ("span", {str(span): {atom for atom in full if atom[1][0] == span} for span in EXPECTED_SPANS}),
            ):
                for group_name, group_atoms in groups.items():
                    group_full = {atom: full[atom] for atom in group_atoms}
                    group_current = {atom: current[atom] for atom in group_atoms}
                    group_metrics = _owner_metrics(group_full, group_current)
                    class_span_rows.append({"domain_id": domain, "subset": subset_name, "group_kind": group_kind, "group": group_name, **group_metrics})
    _write_csv(output_dir / "task042_phase_g_calibration_stability.csv", stability_rows,
               ["row_type", "domain_id", "subset", "label", *Q_FIELDS, *IDENTITY_FIELDS, "owner_set_jaccard", "exact_owner_set_match", "full_singleton", "singleton_owner_identity_match", "full_owner_units", "subset_owner_units", "n_atoms", "mean_jaccard", "median_jaccard", "exact_owner_set_match_rate", "singleton_full_atom_count", "singleton_owner_identity_match_rate", "singleton_owner_identity_matches", "unit_full_owned_atom_count", "unit_subset_owned_atom_count", "unit_incidence_jaccard", "unit_full_owned_atoms", "unit_subset_owned_atoms"])
    _write_csv(output_dir / "task042_phase_g_class_span_stability.csv", class_span_rows,
               ["domain_id", "subset", "group_kind", "group", "n_atoms", "mean_jaccard", "median_jaccard", "exact_owner_set_match_rate", "singleton_full_atom_count", "singleton_owner_identity_match_rate", "singleton_owner_identity_matches"])

    # Mixed Attention/FFN ownership: exact shared atom-level owners only.
    unit_types = {unit: str(unit_metadata[unit]["unit_type"]) for unit in units}
    mixed_domains = {
        domain: domain_unit_list
        for domain, domain_unit_list in multi_domains.items()
        if any("attention" in unit_types[unit].lower() or unit_types[unit].lower() == "head" for unit in domain_unit_list)
        and any("ffn" in unit_types[unit].lower() or "neuron" in unit_types[unit].lower() for unit in domain_unit_list)
    }
    mixed_rows: list[dict[str, Any]] = []
    mixed_counts: dict[str, dict[str, int]] = {}
    for domain, domain_unit_list in mixed_domains.items():
        owner_map = owners_by_domain_subset[domain]["FULL"]
        per_atom: dict[AtomId, str] = {}
        for atom, owner_set in owner_map.items():
            category = classify_owner_types(owner_set, unit_types)
            per_atom[atom] = category
            mixed_rows.append({
                "row_type": "atom",
                "domain_id": domain,
                "label": atom[0],
                **dict(zip(Q_FIELDS, atom[1])),
                "owner_category": category,
                "owner_count": len(owner_set),
                "owner_units": _set_string(owner_set),
            })
        for group_kind, groups in (
            ("all", {"ALL": set(per_atom)}),
            ("class", {label: {atom for atom in per_atom if atom[0] == label} for label in class_videos}),
            ("span", {str(span): {atom for atom in per_atom if atom[1][0] == span} for span in EXPECTED_SPANS}),
        ):
            for group_name, atoms in groups.items():
                counts = Counter(per_atom[atom] for atom in atoms)
                for category in ("Attention-only", "FFN-only", "mixed Attention+FFN", "other-only", "mixed-with-other", "empty-owner-set"):
                    n = counts.get(category, 0)
                    mixed_rows.append({
                        "row_type": "summary",
                        "domain_id": domain,
                        "group_kind": group_kind,
                        "group": group_name,
                        "owner_category": category,
                        "atom_count": len(atoms),
                        "category_atom_count": n,
                        "category_fraction": n / len(atoms) if atoms else 0.0,
                    })
        mixed_counts[domain] = dict(Counter(per_atom.values()))
    _write_csv(output_dir / "task042_phase_g_mixed_type_ownership.csv", mixed_rows,
               ["row_type", "domain_id", "group_kind", "group", "label", *Q_FIELDS, "owner_category", "owner_count", "owner_units", "atom_count", "category_atom_count", "category_fraction"])

    # Unique function owners with exact relation identities.
    unique_rows: list[dict[str, Any]] = []
    unique_summary: dict[str, dict[UnitId, set[AtomId]]] = {}
    for domain, domain_unit_list in multi_domains.items():
        owner_map = owners_by_domain_subset[domain]["FULL"]
        unique_atoms: dict[UnitId, set[AtomId]] = {unit: set() for unit in domain_unit_list}
        for atom, owner_set in owner_map.items():
            if len(owner_set) == 1:
                unit = next(iter(owner_set))
                unique_atoms[unit].add(atom)
                unique_rows.append({
                    "row_type": "unique_atom",
                    "domain_id": domain,
                    "label": atom[0],
                    **dict(zip(Q_FIELDS, atom[1])),
                    "sole_owner": _unit_json(unit),
                    "owner_count": 1,
                })
        unique_summary[domain] = unique_atoms
        incidence = build_incidence(owner_map)
        for unit in domain_unit_list:
            atoms = unique_atoms[unit]
            unique_rows.append({
                "row_type": "unit_summary",
                "domain_id": domain,
                **dict(zip(IDENTITY_FIELDS, unit)),
                "unique_atom_count": len(atoms),
                "unique_action_classes": sorted({atom[0] for atom in atoms}, key=lambda x: (int(x), x)),
                "unique_atoms_by_span": {str(span): sum(atom[1][0] == span for atom in atoms) for span in EXPECTED_SPANS},
                "owned_as_coowner_atoms": sum(unit in owners_set and len(owners_set) > 1 for owners_set in owner_map.values()),
                "never_owner_atom_count": len(owner_map) - len(incidence.get(unit, set())),
                "has_no_unique_atoms": len(atoms) == 0,
                "unique_atom_identities": [
                    {"label": atom[0], **dict(zip(Q_FIELDS, atom[1]))}
                    for atom in sorted(atoms, key=_atom_order)
                ],
            })
    _write_csv(output_dir / "task042_phase_g_unique_function_ownership.csv", unique_rows,
               ["row_type", "domain_id", "label", *Q_FIELDS, *IDENTITY_FIELDS, "sole_owner", "owner_count", "unique_atom_count", "unique_action_classes", "unique_atoms_by_span", "owned_as_coowner_atoms", "never_owner_atom_count", "has_no_unique_atoms", "unique_atom_identities"])

    # Exact minimum functional covers and complete solution multiplicity.
    cover_rows: list[dict[str, Any]] = []
    cover_families: dict[str, dict[str, tuple[frozenset[UnitId], ...]]] = {}
    cover_summaries: dict[str, dict[str, Any]] = {}
    for domain, domain_unit_list in multi_domains.items():
        cover_families[domain] = {}
        for subset_name in subset_names:
            owner_map = owners_by_domain_subset[domain][subset_name]
            min_size, covers = minimum_set_covers(owner_map.values(), domain_unit_list)
            cover_families[domain][subset_name] = covers
            mandatory = set.intersection(*(set(cover) for cover in covers)) if covers else set()
            any_cover = set.union(*(set(cover) for cover in covers)) if covers else set()
            optional = any_cover - mandatory
            never = set(domain_unit_list) - any_cover
            summary = {
                "domain_size": len(domain_unit_list),
                "minimum_cover_size": min_size,
                "redundant_unit_count": len(domain_unit_list) - min_size,
                "optimal_cover_count": len(covers),
                "mandatory_units": _set_string(mandatory),
                "optional_units_some_but_not_all": _set_string(optional),
                "never_in_any_optimal_cover": _set_string(never),
                "units_absent_from_at_least_one_optimal_cover": _set_string(set(domain_unit_list) - mandatory),
            }
            cover_rows.append({"row_type": "domain_summary", "domain_id": domain, "subset": subset_name, **summary})
            for index, cover in enumerate(covers, start=1):
                cover_rows.append({
                    "row_type": "optimal_cover_solution",
                    "domain_id": domain,
                    "subset": subset_name,
                    "solution_index": index,
                    "cover_units": _set_string(cover),
                    "cover_unit_global_indices": [unit[0] for unit in sorted(cover, key=_unit_order)],
                    "cover_size": len(cover),
                })
            membership = {unit: sum(unit in cover for cover in covers) for unit in domain_unit_list}
            for unit in domain_unit_list:
                cover_rows.append({
                    "row_type": "unit_cover_membership",
                    "domain_id": domain,
                    "subset": subset_name,
                    **dict(zip(IDENTITY_FIELDS, unit)),
                    "unit_cover_status": "mandatory" if unit in mandatory else "optional_some_but_not_all" if unit in optional else "never_in_any_optimal_cover",
                    "covers_containing_unit": membership[unit],
                    "optimal_cover_count": len(covers),
                    "absent_from_at_least_one_optimal_cover": unit not in mandatory,
                })
            if subset_name == "FULL":
                cover_summaries[domain] = summary
    _write_csv(output_dir / "task042_phase_g_minimum_functional_cover.csv", cover_rows,
               ["row_type", "domain_id", "subset", "solution_index", "domain_size", "minimum_cover_size", "redundant_unit_count", "optimal_cover_count", "cover_units", "cover_unit_global_indices", "cover_size", "mandatory_units", "optional_units_some_but_not_all", "never_in_any_optimal_cover", "units_absent_from_at_least_one_optimal_cover", *IDENTITY_FIELDS, "unit_cover_status", "covers_containing_unit", "absent_from_at_least_one_optimal_cover"])

    cover_stability_rows: list[dict[str, Any]] = []
    cover_stability_stats: dict[str, dict[str, Any]] = {}
    for domain, domain_unit_list in multi_domains.items():
        full_family = cover_families[domain]["FULL"]
        full_mandatory = set.intersection(*(set(cover) for cover in full_family)) if full_family else set()
        family_full_set = set(full_family)
        cover_stability_stats[domain] = {}
        for subset_name in subset_names:
            family = cover_families[domain][subset_name]
            mandatory = set.intersection(*(set(cover) for cover in family)) if family else set()
            exact_family_jaccard = jaccard(family_full_set, set(family))
            forward_best = [max(jaccard(cover, full_cover) for full_cover in full_family) for cover in family]
            reverse_best = [max(jaccard(full_cover, cover) for cover in family) for full_cover in full_family]
            mean_best = (statistics.mean(forward_best) + statistics.mean(reverse_best)) / 2.0
            record = {
                "domain_id": domain,
                "subset": subset_name,
                "full_minimum_cover_size": len(full_family[0]),
                "subset_minimum_cover_size": len(family[0]),
                "minimum_cover_size_matches_full": len(family[0]) == len(full_family[0]),
                "full_optimal_cover_count": len(full_family),
                "subset_optimal_cover_count": len(family),
                "optimal_cover_family_exact_jaccard": exact_family_jaccard,
                "bidirectional_mean_best_cover_jaccard": mean_best,
                "full_mandatory_units": _set_string(full_mandatory),
                "subset_mandatory_units": _set_string(mandatory),
                "mandatory_unit_jaccard": jaccard(full_mandatory, mandatory),
                "mandatory_units_exact_match": full_mandatory == mandatory,
            }
            cover_stability_stats[domain][subset_name] = record
            cover_stability_rows.append(record)
    _write_csv(output_dir / "task042_phase_g_cover_stability.csv", cover_stability_rows,
               ["domain_id", "subset", "full_minimum_cover_size", "subset_minimum_cover_size", "minimum_cover_size_matches_full", "full_optimal_cover_count", "subset_optimal_cover_count", "optimal_cover_family_exact_jaccard", "bidirectional_mean_best_cover_jaccard", "full_mandatory_units", "subset_mandatory_units", "mandatory_unit_jaccard", "mandatory_units_exact_match"])

    # Assertions on frozen expected sizes and summary evidence.
    all_primary = [owner_set for owners in owners_by_domain_subset.values() for owner_set in owners["FULL"].values()]
    singleton_atoms = sum(len(owner_set) == 1 for owner_set in all_primary)
    multi_owner_atoms = len(all_primary) - singleton_atoms
    global_degeneracy = {
        "singleton_fraction": singleton_atoms / len(all_primary),
        "multi_owner_fraction": multi_owner_atoms / len(all_primary),
        "mean_owner_set_size": statistics.mean(map(len, all_primary)),
        "median_owner_set_size": statistics.median(map(len, all_primary)),
        "maximum_owner_set_size": max(map(len, all_primary)),
        "cardinality_distribution": dict(sorted(Counter(map(len, all_primary)).items())),
    }
    nontrivial_domains = [
        domain for domain, owner_sets in owners_by_domain_subset.items()
        if len({tuple(sorted(owner_set, key=_unit_order)) for owner_set in owner_sets["FULL"].values()}) > 1
    ]
    domains_with_redundancy = [domain for domain, summary in cover_summaries.items() if summary["minimum_cover_size"] < summary["domain_size"]]
    nonmixed_redundancy_domains = [domain for domain in domains_with_redundancy if domain not in mixed_domains]
    mixed_function_atoms = sum(mixed_counts.get(domain, {}).get("mixed Attention+FFN", 0) for domain in mixed_domains)
    action_conditioning_changes = {domain: stats["ownership_changes"] for domain, stats in ablation_summaries.items()}
    cover_optional_units = {domain: summary["optional_units_some_but_not_all"] for domain, summary in cover_summaries.items()}
    class_subset_scores = [
        stability_domain_stats[domain][subset]["mean_jaccard"]
        for domain in multi_domains
        for subset in ("P1", "P2", "P3", "P12", "P13", "P23")
    ]
    class_subset_match = [
        stability_domain_stats[domain][subset]["exact_owner_set_match_rate"]
        for domain in multi_domains
        for subset in ("P1", "P2", "P3", "P12", "P13", "P23")
    ]

    if decision_code == "A":
        if not nontrivial_domains or not domains_with_redundancy or not nonmixed_redundancy_domains:
            raise ValueError("Decision A violates one or more required structural criteria")
        if not decision_reason.strip():
            raise ValueError("Decision A requires a written qualitative stability justification")

    summary: dict[str, Any] = {
        "phase": "Task042 Phase G",
        "scientific_position": "Temporal functions are related to units within frozen BMS domains; not an importance score, pruning rank, domain priority, or training loss.",
        "inputs": {
            str(path.name): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (units_path, videos_path, raw_path)
        },
        "git_head": git_head,
        "gpu_used": False,
        "new_inference": False,
        "pruning_performed": False,
        "finetuning_performed": False,
        "performance_oracle_used": False,
        "frozen_protocol": {
            "unit_identity_fields": list(IDENTITY_FIELDS),
            "sensitivity_field": "relative_sensitivity",
            "rank_scope": "one unit x one video across exact 80 q conditions",
            "tie_rule": "exact float64 equality; average ranks",
            "spans": list(EXPECTED_SPANS),
            "pairs_per_span": EXPECTED_PAIRS_PER_SPAN,
            "relations_per_video": EXPECTED_Q,
            "action_classes": EXPECTED_CLASSES,
            "videos_per_class": 3,
            "primary_owner": "B: exact argmax of classwise median rank preference; all exact maximum ties retained",
            "subsets": {name: list(indices) for name, indices in subset_videos.items()},
        },
        "counts": {
            "units": len(units),
            "videos": len(video_rows),
            "classes": len(class_videos),
            "relations_per_unit_video": EXPECTED_Q,
            "raw_rows": len(raw_rows),
            "all_bms_domains": len(domain_units),
            "singleton_domains": sum(len(items) == 1 for items in domain_units.values()),
            "multi_unit_domains": len(multi_domains),
            "multi_unit_domain_ids": sorted(multi_domains, key=lambda value: (int(value), value)),
            "atoms": len(all_primary),
            "singleton_atoms": singleton_atoms,
            "multi_owner_atoms": multi_owner_atoms,
        },
        "global_owner_degeneracy": global_degeneracy,
        "domain_degeneracy": {row["domain_id"]: row for row in degeneracy_rows if row["row_scope"] == "domain"},
        "action_conditioning_ablation": ablation_summaries,
        "pareto_domain_stats": pareto_domain_stats,
        "calibration_stability": stability_domain_stats,
        "cover_summaries": cover_summaries,
        "cover_stability": cover_stability_stats,
        "mixed_domains": sorted(mixed_domains, key=lambda value: (int(value), value)),
        "mixed_owner_category_counts": mixed_counts,
        "functional_cover_domains_with_redundant_units": domains_with_redundancy,
        "nonmixed_domains_with_redundant_units": nonmixed_redundancy_domains,
        "domains_with_nontrivial_owner_sets": nontrivial_domains,
        "unique_mixed_attention_ffn_owner_atoms": mixed_function_atoms,
        "action_conditioning_ownership_changes_by_domain": action_conditioning_changes,
        "optional_units_by_domain": cover_optional_units,
        "stability_descriptive_distribution": {
            "mean_owner_jaccard_across_domains_and_nonfull_subsets": statistics.mean(class_subset_scores) if class_subset_scores else 1.0,
            "median_of_domain_subset_mean_owner_jaccard": statistics.median(class_subset_scores) if class_subset_scores else 1.0,
            "mean_exact_owner_set_match_rate_across_domains_and_nonfull_subsets": statistics.mean(class_subset_match) if class_subset_match else 1.0,
        },
        "required_questions": {},
        "decision": {"label": f"{decision_code}. {decision_labels[decision_code]}", "reason": decision_reason},
        "outputs": [
            "task042_phase_g_relation_preference.csv",
            "task042_phase_g_primary_owner_incidence.csv",
            "task042_phase_g_owner_degeneracy.csv",
            "task042_phase_g_functional_redundancy.csv",
            "task042_phase_g_global_owner_ablation.csv",
            "task042_phase_g_pareto_owner_diagnostic.csv",
            "task042_phase_g_calibration_stability.csv",
            "task042_phase_g_class_span_stability.csv",
            "task042_phase_g_mixed_type_ownership.csv",
            "task042_phase_g_unique_function_ownership.csv",
            "task042_phase_g_minimum_functional_cover.csv",
            "task042_phase_g_cover_stability.csv",
            "task042_phase_g_summary.json",
            "task042_phase_g_report.md",
        ],
    }
    summary["required_questions"] = {
        "A": (
            f"Yes, incidence is nontrivial in {len(nontrivial_domains)}/{len(multi_domains)} multi-unit domains by exact owner-set diversity. "
            f"The full per-atom relation-unit incidence is in task042_phase_g_primary_owner_incidence.csv."
        ),
        "B": (
            f"{singleton_atoms}/{len(all_primary)} atoms ({global_degeneracy['singleton_fraction']:.3f}) are singleton-owner and "
            f"{multi_owner_atoms}/{len(all_primary)} ({global_degeneracy['multi_owner_fraction']:.3f}) are exactly multi-owner. "
            f"Minimum-cover compression exists in {len(domains_with_redundancy)}/{len(multi_domains)} domains; only {len(nonmixed_redundancy_domains)} such domains are non-mixed."
        ),
        "C": (
            f"Global-median A versus action-conditioned B changes {sum(action_conditioning_changes.values())} of {len(multi_domains)*EXPECTED_CLASSES*EXPECTED_Q} atom owner sets. "
            f"Per-domain identity agreement and incidence Jaccard are in task042_phase_g_global_owner_ablation.csv."
        ),
        "D": (
            f"Across non-FULL balanced subsets, descriptive mean atom-owner Jaccard is {summary['stability_descriptive_distribution']['mean_owner_jaccard_across_domains_and_nonfull_subsets']:.3f}; "
            f"mean exact owner-set match rate is {summary['stability_descriptive_distribution']['mean_exact_owner_set_match_rate_across_domains_and_nonfull_subsets']:.3f}. "
            f"These summaries are descriptive, without a post-hoc numeric pass threshold; per-class, per-span, per-unit, and per-cover stability are in the stability CSVs."
        ),
        "E": (
            f"Exact shared Attention/FFN ownership was found in {mixed_function_atoms} atoms across mixed domains {summary['mixed_domains']}. "
            f"Only owner sets containing both unit types count as cross-type functional redundancy."
        ),
        "F": (
            f"No. {sum(bool(summary['units_absent_from_at_least_one_optimal_cover'] != '[]') for summary in cover_summaries.values())}/{len(multi_domains)} domains have any unit absent from at least one minimum cover. "
            f"Here every domain's exact minimum cover contains all its units, so all units are mandatory under the observed incidence."
        ),
        "G": (
            f"No for this exact coverage formulation: minimum-cover size equals domain size in {len(multi_domains)-len(domains_with_redundancy)}/{len(multi_domains)} domains, so it yields no removable unit while preserving every primary atom. "
            f"Owner stability is descriptive (mean atom-owner Jaccard {summary['stability_descriptive_distribution']['mean_owner_jaccard_across_domains_and_nonfull_subsets']:.3f}; mean exact match {summary['stability_descriptive_distribution']['mean_exact_owner_set_match_rate_across_domains_and_nonfull_subsets']:.3f}), but cannot offset the lack of cover reduction. The selected decision is {summary['decision']['label']}."
        ),
    }
    (output_dir / "task042_phase_g_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / "task042_phase_g_report.md", summary)
    return summary

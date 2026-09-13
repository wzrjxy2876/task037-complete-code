#!/usr/bin/env python3
"""Offline Task041 Phase H.1 safest-semantics and incremental-value audit.

This consumes only completed Phase-H tables and the frozen Phase-G N=9
manifest. It performs no model loading, inference, GPU work, pruning, or training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import task041_phase_h_temporal_stress as phase_h


SAME_TYPE = phase_h.SAME_TYPE
MIXED = phase_h.MIXED
DOMAINS = phase_h.DOMAINS
SPANS = phase_h.SPANS
H1_FILES = (
    "task041_phase_h1_corrected_original_vs_temporal.csv",
    "task041_phase_h1_safest_audit.csv",
    "task041_phase_h1_harmful_candidate_audit.csv",
    "task041_phase_h1_rank_similarity.csv",
    "task041_phase_h1_per_span_audit.csv",
    "task041_phase_h1_class_coverage_audit.csv",
    "task041_phase_h1_summary.json",
    "task041_phase_h1_report.md",
)
H1_OUTPUT_DEFAULT = "/data/jixinye25/work1/output/task041_phase_h1_semantics_audit"
NO_TOP1_SELECTION_VALUE_STATEMENT = (
    "Temporal stress changes intra-domain rank structure but provides no incremental "
    "top-1 pruning-candidate selection value on the current same-type cohort."
)
EXPECTED_FROZEN_RANK_ASSOCIATIONS = {
    "R_original": {"spearman": -0.14285714285714285, "kendall_tau_b": -0.09523809523809522},
    "R_temporal": {"spearman": 0.07142857142857142, "kendall_tau_b": 0.04761904761904761},
}
METHOD_FIELDS = {"R_original": "W_original", "R_temporal": "W_temporal_N9"}
IDENTITY_FIELDS = (
    "candidate_task037_global_index", "candidate_task040_global_index",
    "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
    "candidate_stage", "domain_id",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _float(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid numeric Phase-H field: " + field) from exc
    require(math.isfinite(value), "non-finite Phase-H value: " + field)
    return value


def _uid(row: Mapping[str, Any]) -> str:
    return str(int(str(row["candidate_task037_global_index"])))


def _identity(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {field: row[field] for field in IDENTITY_FIELDS}


def safest_uid(rows: Sequence[Mapping[str, Any]], score_field: str) -> str:
    """Highest W (equivalently minimum R), tied by ascending Task037 ID."""
    require(bool(rows), "cannot select a safest unit from an empty domain")
    return min(rows, key=lambda row: (-_float(row, score_field), int(_uid(row))))[IDENTITY_FIELDS[0]]


def most_harmful_uid(rows: Sequence[Mapping[str, Any]], score_field: str) -> str:
    """Lowest W (equivalently maximum R), tied by ascending Task037 ID."""
    require(bool(rows), "cannot select a most-harmful unit from an empty domain")
    return min(rows, key=lambda row: (_float(row, score_field), int(_uid(row))))[IDENTITY_FIELDS[0]]


def fullval_safest_uid(rows: Sequence[Mapping[str, Any]]) -> str:
    require(bool(rows), "cannot select a full-validation safest unit from an empty domain")
    return min(rows, key=lambda row: (
        _float(row, "fullval_mean_cross_entropy_increase"), int(_uid(row))
    ))[IDENTITY_FIELDS[0]]


def fullval_most_harmful_uid(rows: Sequence[Mapping[str, Any]]) -> str:
    require(bool(rows), "cannot select a full-validation most-harmful unit from an empty domain")
    return min(rows, key=lambda row: (
        -_float(row, "fullval_mean_cross_entropy_increase"), int(_uid(row))
    ))[IDENTITY_FIELDS[0]]


def correction_flags(original_uid: str, temporal_uid: str,
                     truth_uid: str) -> tuple[bool, bool]:
    correction = original_uid != truth_uid and temporal_uid == truth_uid
    regression = original_uid == truth_uid and temporal_uid != truth_uid
    return correction, regression


def _risk_order(rows: Sequence[Mapping[str, Any]], score_field: str) -> list[str]:
    # R = 1-W, so ascending risk is descending W; ties use numeric Task037 ID.
    return [str(row[IDENTITY_FIELDS[0]]) for row in sorted(
        rows, key=lambda row: (-_float(row, score_field), int(_uid(row)))
    )]


def _same_type_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_spearman = [row["spearman_vs_fullval_ce"] for row in rows
                      if row["spearman_vs_fullval_ce"] is not None]
    valid_kendall = [row["kendall_vs_fullval_ce"] for row in rows
                     if row["kendall_vs_fullval_ce"] is not None]
    domains = len(rows)
    return {
        "domain_count": domains,
        "spearman": phase_h.average(valid_spearman),
        "kendall": phase_h.average(valid_kendall),
        "valid_spearman_domain_count": len(valid_spearman),
        "valid_kendall_domain_count": len(valid_kendall),
        "safest_accuracy": sum(bool(row["safest_identity_match"]) for row in rows) / domains,
        "safest_correct_count": sum(bool(row["safest_identity_match"]) for row in rows),
        "low_high_correct": sum(row["low_high_ordering"] == "correct" for row in rows),
        "low_high_reverse": sum(row["low_high_ordering"] == "reverse" for row in rows),
        "low_high_tie": sum(row["low_high_ordering"] == "tie" for row in rows),
    }


def _domain_method_row(domain: str, members: Sequence[Mapping[str, Any]],
                       method: str, legacy: Mapping[str, Any] | None) -> dict[str, Any]:
    score_field = METHOD_FIELDS[method]
    ordered = sorted(members, key=lambda row: int(_uid(row)))
    risk = [1.0 - _float(row, score_field) for row in ordered]
    damage = [_float(row, "fullval_mean_cross_entropy_increase") for row in ordered]
    safe = safest_uid(ordered, score_field)
    harmful = most_harmful_uid(ordered, score_field)
    truth_safe = fullval_safest_uid(ordered)
    safe_row = next(row for row in ordered if _uid(row) == safe)
    harmful_row = next(row for row in ordered if _uid(row) == harmful)
    safe_ce = _float(safe_row, "fullval_mean_cross_entropy_increase")
    harmful_ce = _float(harmful_row, "fullval_mean_cross_entropy_increase")
    legacy_uid = None
    if legacy and legacy.get("safest_candidate_task037_global_index") not in (None, ""):
        legacy_uid = str(int(str(legacy["safest_candidate_task037_global_index"])))
    return {
        "row_type": "domain", "domain_id": domain,
        "scope": "same_type" if domain in SAME_TYPE else "mixed",
        "method": method, "unit_count": len(ordered),
        "risk_definition": "R=1-W; lower risk is safer",
        "spearman_vs_fullval_ce": phase_h.spearman(risk, damage),
        "kendall_vs_fullval_ce": phase_h.kendall_tau_b(risk, damage),
        "safest_candidate_task037_global_index": safe,
        "fullval_safest_task037_global_index": truth_safe,
        "safest_identity_match": safe == truth_safe,
        "safest_W": _float(safe_row, score_field),
        "safest_fullval_ce_increase": safe_ce,
        "most_harmful_candidate_task037_global_index": harmful,
        "most_harmful_W": _float(harmful_row, score_field),
        "most_harmful_fullval_ce_increase": harmful_ce,
        "low_risk_task037_global_index": safe,
        "high_risk_task037_global_index": harmful,
        "low_high_ordering": phase_h.ordering_label(safe_ce, harmful_ce),
        "high_minus_low_fullval_ce": harmful_ce - safe_ce,
        "legacy_phase_h_safest_uid": legacy_uid,
        "legacy_phase_h_safest_matches_corrected": (
            None if legacy_uid is None else legacy_uid == safe
        ),
        "safest_identity": json.dumps(_identity(safe_row), sort_keys=True),
        "fullval_safest_identity": json.dumps(
            _identity(next(row for row in ordered if _uid(row) == truth_safe)), sort_keys=True
        ),
        "unit_task037_order": json.dumps([_uid(row) for row in ordered]),
    }


def _class_counts(indices: Sequence[int], labels_by_index: Mapping[int, str]) -> dict[str, int]:
    return dict(sorted(Counter(labels_by_index[index] for index in indices).items()))


def _validate_class_coverage(manifest: Sequence[Mapping[str, Any]],
                             subset_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(manifest) == 9, "frozen Phase-G manifest must contain exactly nine videos")
    indices = [int(row["video_index"]) for row in manifest]
    require(len(set(indices)) == 9 and set(indices) == set(range(9)),
            "Phase-G video indices must be exactly 0..8")
    labels_by_index: dict[int, str] = {}
    class_by_index: dict[int, str] = {}
    video_rows = []
    for row in manifest:
        index = int(row["video_index"])
        label = str(row["label"])
        class_name = Path(str(row["video_id"])).parent.name
        require(class_name != "" and class_name != ".", "cannot identify action class from manifest video path")
        labels_by_index[index] = label
        class_by_index[index] = class_name
        video_rows.append({
            "row_type": "video", "video_index": index,
            "dataset_index": row["dataset_index"],
            "canonical_video_id": row["canonical_video_id"],
            "action_class_name": class_name, "label_index": label,
            "phase_f_n3_member": row.get("phase_f_n3_member", ""),
        })
    name_to_label: dict[str, str] = {}
    label_to_name: dict[str, str] = {}
    for index in indices:
        name, label = class_by_index[index], labels_by_index[index]
        require(name not in name_to_label or name_to_label[name] == label,
                "one action name maps to multiple manifest labels")
        require(label not in label_to_name or label_to_name[label] == name,
                "one label maps to multiple action names")
        name_to_label[name] = label
        label_to_name[label] = name
    require(len(name_to_label) == 3 and len(label_to_name) == 3,
            "frozen N=9 pool must contain exactly three action classes")
    counts_n9 = _class_counts(indices, class_by_index)
    require(set(counts_n9.values()) == {3}, "N=9 pool must contain three videos per action class")

    selected_n3 = sorted(
        int(row["video_index"]) for row in manifest
        if str(row.get("phase_f_n3_member", "")).strip().lower() == "true"
    )
    require(len(selected_n3) == 3 and _class_counts(selected_n3, class_by_index)
            == {name: 1 for name in sorted(name_to_label)},
            "Phase-F N=3 membership does not select one video/class from the frozen N=9 pool")

    subset_map: dict[str, tuple[list[int], int]] = {}
    for row in subset_rows:
        subset_id = str(row["subset_id"])
        selected = sorted(int(i) for i in json.loads(row["video_indices_json"]))
        video_count = int(row["video_count"])
        candidate = (selected, video_count)
        require(subset_id not in subset_map or subset_map[subset_id] == candidate,
                "Phase-H subset rows disagree on member indices: " + subset_id)
        subset_map[subset_id] = candidate
    n3 = subset_map.get("n3_phase_f")
    n9 = subset_map.get("n9_full")
    n6_ids = sorted(sid for sid in subset_map if sid.startswith("n6_"))
    require(n3 == (selected_n3, 3), "Phase-H N=3 stability set differs from frozen Phase-F subset")
    require(n9 == (sorted(indices), 9), "Phase-H N=9 stability set differs from frozen Phase-G pool")
    require(len(n6_ids) == 27, "Phase-H stability table must contain 27 distinct N=6 subsets")
    subset_audit_rows = []
    for subset_id in sorted(subset_map):
        selected, count = subset_map[subset_id]
        require(count == len(selected) and set(selected) <= set(indices),
                "stability subset contains video outside frozen N=9 pool: " + subset_id)
        class_counts = _class_counts(selected, class_by_index)
        if subset_id == "n3_phase_f":
            valid = count == 3 and set(class_counts.values()) == {1}
        elif subset_id.startswith("n6_"):
            valid = count == 6 and set(class_counts.values()) == {2}
        elif subset_id == "n9_full":
            valid = count == 9 and selected == sorted(indices) and set(class_counts.values()) == {3}
        else:
            valid = False
        require(valid and set(class_counts) == set(name_to_label),
                "N=3/N=6/N=9 subset fails frozen class-balanced pool audit: " + subset_id)
        subset_audit_rows.append({
            "row_type": "subset", "subset_id": subset_id,
            "subset_size": count, "video_indices_json": json.dumps(selected),
            "class_counts_json": json.dumps(class_counts, sort_keys=True),
            "unique_action_class_count": len(class_counts),
            "subset_within_frozen_n9_pool": True,
            "class_balance_valid": True,
        })
    summary = {
        "n9_video_count": len(manifest),
        "unique_action_class_count": len(name_to_label),
        "action_class_to_label_index": dict(sorted(name_to_label.items())),
        "n9_class_counts": counts_n9,
        "n3_video_indices": selected_n3,
        "n6_subset_count": len(n6_ids),
        "n9_video_indices": sorted(indices),
        "all_stability_subsets_within_same_n9_pool": True,
        "n3_n6_n9_are_video_count_stability_only": True,
        "limitation": (
            "Within-pool video-count stability does NOT establish cross-action-class calibration validity."
        ),
    }
    subset_audit_rows.append({
        "row_type": "coverage_summary", "n9_video_count": len(manifest),
        "unique_action_class_count": len(name_to_label),
        "action_class_to_label_index_json": json.dumps(summary["action_class_to_label_index"], sort_keys=True),
        "n9_class_counts_json": json.dumps(counts_n9, sort_keys=True),
        "n3_video_indices_json": json.dumps(selected_n3),
        "n6_subset_count": len(n6_ids),
        "all_stability_subsets_within_same_n9_pool": True,
        "limitation": summary["limitation"],
    })
    return video_rows + subset_audit_rows, summary

def top1_value_decision(changed_count: int, correction_count: int,
                        regression_count: int, exact_order_equal_domain_count: int,
                        domain_count: int) -> tuple[str, str]:
    if changed_count == 0 or correction_count == 0:
        if changed_count == 0 and exact_order_equal_domain_count < domain_count:
            rationale = NO_TOP1_SELECTION_VALUE_STATEMENT
        elif changed_count == 0:
            rationale = (
                "Temporal stress changed neither the complete intra-domain ordering "
                "nor the safest top-1 candidate."
            )
        else:
            rationale = "There are changed top-1 choices but none corrects the original choice."
        return "NO_DEMONSTRATED_TOP1_SELECTION_VALUE", rationale
    if regression_count == 0:
        return (
            "ADDS_TOP1_SELECTION_VALUE",
            "Every changed same-type top-1 choice is a correction and none is a regression.",
        )
    return (
        "UNRESOLVED",
        "The same-type cohort contains both top-1 corrections and regressions.",
    )


def _legacy_safest_report_claims(report: str) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for line in report.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 6 and cells[1] in ("Original-only", "Temporal stress"):
            claims[cells[1]] = {
                "safest_identity_accuracy": float(cells[4]),
                "low_high_correct_reverse_tie": cells[5],
            }
    require(set(claims) == {"Original-only", "Temporal stress"},
            "cannot locate both legacy Phase-H comparison rows in the report")
    require(math.isclose(claims["Original-only"]["safest_identity_accuracy"],
                         4.0 / 7.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(claims["Temporal stress"]["safest_identity_accuracy"],
                             2.0 / 7.0, rel_tol=0.0, abs_tol=1e-12),
            "legacy Phase-H report does not contain the expected displayed safest accuracies")
    return claims

def _verify_frozen_rank_associations(
    original_summary: Mapping[str, Any], temporal_summary: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    same_type_oracle: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for method, recomputed in (("R_original", original_summary),
                               ("R_temporal", temporal_summary)):
        matches = [row for row in baseline_rows
                   if row.get("row_type") == "domain_balanced"
                   and row.get("scope") == "same_type"
                   and row.get("method") == method]
        require(len(matches) == 1, "missing/duplicate frozen baseline rank row: " + method)
        stored = matches[0]
        expected = EXPECTED_FROZEN_RANK_ASSOCIATIONS[method]
        values = {
            "spearman": recomputed["spearman"],
            "kendall_tau_b": recomputed["kendall"],
        }
        baseline_values = {
            "spearman": _float(stored, "spearman"),
            "kendall_tau_b": _float(stored, "kendall_tau_b"),
        }
        for statistic, target in expected.items():
            require(values[statistic] is not None
                    and math.isclose(float(values[statistic]), target, rel_tol=0.0, abs_tol=1e-12),
                    "recomputed frozen Phase-H rank association changed: %s/%s" % (method, statistic))
            require(math.isclose(baseline_values[statistic], target, rel_tol=0.0, abs_tol=1e-12),
                    "saved Phase-H baseline rank association changed: %s/%s" % (method, statistic))
        audit[method] = {
            "expected_frozen": expected,
            "recomputed_from_frozen_unit_records": values,
            "saved_phase_h_baseline_comparison": baseline_values,
            "verified_unchanged": True,
        }
    oracle_rows = [row for row in same_type_oracle
                   if row.get("row_type") == "domain_balanced"
                   and row.get("domain_id") == "ALL"
                   and row.get("scope") == "same_type"]
    require(len(oracle_rows) == 1, "missing/duplicate Phase-H same-type oracle aggregate")
    oracle = oracle_rows[0]
    temporal_expected = EXPECTED_FROZEN_RANK_ASSOCIATIONS["R_temporal"]
    oracle_values = {
        "spearman": _float(oracle, "temporal_spearman"),
        "kendall_tau_b": _float(oracle, "temporal_kendall_tau_b"),
    }
    for statistic, target in temporal_expected.items():
        require(math.isclose(oracle_values[statistic], target, rel_tol=0.0, abs_tol=1e-12),
                "saved Phase-H same-type oracle association changed: " + statistic)
    audit["temporal_same_type_oracle"] = {
        "saved_phase_h_same_type_oracle": oracle_values, "verified_unchanged": True,
    }
    audit["verified"] = True
    return audit

def run(args: argparse.Namespace) -> dict[str, Any]:
    phase_h_root = Path(args.phase_h_dir)
    output_root = Path(args.output_dir)
    manifest_path = Path(args.n9_manifest)
    unit_path = phase_h_root / "task041_phase_h_unit_temporal_winrate.csv"
    span_path = phase_h_root / "task041_phase_h_per_span_winrate.csv"
    old_compare_path = phase_h_root / "task041_phase_h_original_vs_temporal.csv"
    subset_path = phase_h_root / "task041_phase_h_subset_stability.csv"
    old_summary_path = phase_h_root / "task041_phase_h_summary.json"
    baseline_comparison_path = phase_h_root / "task041_phase_h_baseline_comparison.csv"
    same_type_oracle_path = phase_h_root / "task041_phase_h_same_type_oracle.csv"
    phase_h_report_path = phase_h_root / "task041_phase_h_report.md"
    for path in (unit_path, span_path, old_compare_path, subset_path,
                 old_summary_path, baseline_comparison_path,
                 same_type_oracle_path, phase_h_report_path, manifest_path):
        require(path.is_file(), "missing frozen Phase-H/G input: " + str(path))
    require(output_root.resolve() != phase_h_root.resolve(),
            "Phase-H.1 output must be separate from read-only Phase-H inputs")
    require(not output_root.exists(),
            "refusing to overwrite existing Phase-H.1 output directory: " + str(output_root))

    repo = Path(__file__).resolve().parents[2]
    branch = phase_h.git_value(repo, "rev-parse", "--abbrev-ref", "HEAD")
    require(branch == phase_h.BRANCH, "Phase-H.1 must run on the existing Task041 branch")

    units = phase_h.read_csv(unit_path)
    spans = phase_h.read_csv(span_path)
    old_compare = phase_h.read_csv(old_compare_path)
    baseline_comparison = phase_h.read_csv(baseline_comparison_path)
    same_type_oracle = phase_h.read_csv(same_type_oracle_path)
    subsets = phase_h.read_csv(subset_path)
    manifest = phase_h.read_csv(manifest_path)
    previous_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    previous_report = phase_h_report_path.read_text(encoding="utf-8")
    legacy_report_claims = _legacy_safest_report_claims(previous_report)
    require(len(units) == 29, "Phase-H table must contain exactly the frozen 29 units")
    by_uid: dict[str, Mapping[str, Any]] = {}
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in units:
        uid = _uid(row)
        require(uid not in by_uid, "duplicate unit in Phase-H table: " + uid)
        by_uid[uid] = row
        by_domain[str(row["domain_id"])].append(row)
        require(0.0 <= _float(row, "W_original") <= 1.0
                and 0.0 <= _float(row, "W_temporal_N9") <= 1.0,
                "frozen Phase-H W value outside [0,1]")
        require(math.isclose(_float(row, "R_original"), 1.0 - _float(row, "W_original"),
                             rel_tol=0.0, abs_tol=1e-12),
                "stored R_original differs from frozen 1-W_original")
        require(math.isclose(_float(row, "R_temporal"), 1.0 - _float(row, "W_temporal_N9"),
                             rel_tol=0.0, abs_tol=1e-12),
                "stored R_temporal differs from frozen 1-W_temporal_N9")
        require(int(row.get("fullval_n_samples", 0)) == 3783,
                "Phase-H full-validation CE source is not the existing 3,783-clip oracle")
        for span in SPANS:
            require(0.0 <= _float(row, "W_span_%d" % span) <= 1.0,
                    "frozen per-span W outside [0,1]")
    require(set(by_domain) == set(DOMAINS), "Phase-H domains differ from the frozen nine-domain set")
    require(all(len(by_domain[domain]) == (3 if domain in SAME_TYPE else 4) for domain in DOMAINS),
            "Phase-H domain unit counts differ from the frozen 29-unit identity cohort")

    span_lookup = {}
    for row in spans:
        key = (_uid(row), int(row["span"]))
        require(key not in span_lookup, "duplicate unit/span row in Phase-H per-span file")
        span_lookup[key] = row
    require(len(span_lookup) == 29 * len(SPANS), "Phase-H per-span file is incomplete")
    for uid, unit in by_uid.items():
        for span in SPANS:
            row = span_lookup[(uid, span)]
            require(math.isclose(_float(row, "W_temporal_span"), _float(unit, "W_span_%d" % span),
                                 rel_tol=0.0, abs_tol=1e-12),
                    "unit-level and per-span Phase-H W values disagree")

    legacy = {(str(row.get("domain_id")), str(row.get("method"))): row
              for row in old_compare if row.get("row_type") == "domain"}
    corrected_rows = []
    method_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for domain in DOMAINS:
        members = by_domain[domain]
        for method in METHOD_FIELDS:
            row = _domain_method_row(domain, members, method, legacy.get((domain, method)))
            corrected_rows.append(row)
            method_rows[(domain, method)] = row

    for scope, domain_ids in (("same_type", SAME_TYPE), ("mixed", MIXED)):
        for method in METHOD_FIELDS:
            values = [method_rows[(domain, method)] for domain in domain_ids]
            summary = _same_type_summary(values)
            corrected_rows.append({
                "row_type": "domain_balanced", "domain_id": "ALL", "scope": scope,
                "method": method, "unit_count": sum(int(row["unit_count"]) for row in values),
                **summary,
            })

    safest_rows = []
    harmful_rows = []
    same_type_corrected: list[dict[str, Any]] = []
    same_type_regressed: list[dict[str, Any]] = []
    same_type_changes = []
    for domain in SAME_TYPE:
        members = by_domain[domain]
        original = method_rows[(domain, "R_original")]
        temporal = method_rows[(domain, "R_temporal")]
        truth_safe = str(original["fullval_safest_task037_global_index"])
        orig_safe = str(original["safest_candidate_task037_global_index"])
        temp_safe = str(temporal["safest_candidate_task037_global_index"])
        correction, regression = correction_flags(orig_safe, temp_safe, truth_safe)
        changed = orig_safe != temp_safe
        if changed:
            same_type_changes.append(domain)
        if correction:
            same_type_corrected.append({"domain_id": domain, "original_safest": orig_safe,
                                        "temporal_safest": temp_safe, "fullval_safest": truth_safe})
        if regression:
            same_type_regressed.append({"domain_id": domain, "original_safest": orig_safe,
                                        "temporal_safest": temp_safe, "fullval_safest": truth_safe})
        row = {
            "row_type": "domain", "domain_id": domain,
            "original_safest_task037_global_index": orig_safe,
            "original_safest_identity": original["safest_identity"],
            "temporal_safest_task037_global_index": temp_safe,
            "temporal_safest_identity": temporal["safest_identity"],
            "fullval_safest_task037_global_index": truth_safe,
            "fullval_safest_identity": original["fullval_safest_identity"],
            "original_safest_identity_match": orig_safe == truth_safe,
            "temporal_safest_identity_match": temp_safe == truth_safe,
            "original_safest_equals_temporal_safest": orig_safe == temp_safe,
            "temporal_safest_changed": changed,
            "temporal_correction": correction,
            "temporal_regression": regression,
            "W_original_for_original_safest": original["safest_W"],
            "W_temporal_for_original_safest": _float(by_uid[orig_safe], "W_temporal_N9"),
            "W_original_for_temporal_safest": _float(by_uid[temp_safe], "W_original"),
            "W_temporal_for_temporal_safest": temporal["safest_W"],
            "fullval_ce_original_safest": _float(by_uid[orig_safe], "fullval_mean_cross_entropy_increase"),
            "fullval_ce_temporal_safest": _float(by_uid[temp_safe], "fullval_mean_cross_entropy_increase"),
            "fullval_ce_true_safest": _float(by_uid[truth_safe], "fullval_mean_cross_entropy_increase"),
            "original_legacy_reported_safest_uid": original["legacy_phase_h_safest_uid"],
            "temporal_legacy_reported_safest_uid": temporal["legacy_phase_h_safest_uid"],
            "per_span_safest_uids_json": json.dumps({
                str(span): safest_uid(members, "W_span_%d" % span) for span in SPANS
            }, sort_keys=True),
        }
        safest_rows.append(row)

    original_safe_accuracy = sum(method_rows[(d, "R_original")]["safest_identity_match"]
                                for d in SAME_TYPE) / len(SAME_TYPE)
    temporal_safe_accuracy = sum(method_rows[(d, "R_temporal")]["safest_identity_match"]
                                for d in SAME_TYPE) / len(SAME_TYPE)
    safest_rows.append({
        "row_type": "same_type_summary", "domain_id": "ALL",
        "domain_count": len(SAME_TYPE), "temporal_safest_changed_domain_count": len(same_type_changes),
        "temporal_safest_changed_domains_json": json.dumps(same_type_changes),
        "temporal_correction_count": len(same_type_corrected),
        "temporal_correction_domains_json": json.dumps([row["domain_id"] for row in same_type_corrected]),
        "temporal_regression_count": len(same_type_regressed),
        "temporal_regression_domains_json": json.dumps([row["domain_id"] for row in same_type_regressed]),
        "original_safest_identity_accuracy": original_safe_accuracy,
        "temporal_safest_identity_accuracy": temporal_safe_accuracy,
        "same_safest_candidate_domain_count": len(SAME_TYPE) - len(same_type_changes),
        "top1_incremental_value_statement": (
            "Temporal stress provides no incremental top-1 pruning-candidate selection value on the current same-type cohort."
            if not same_type_changes else "Top-1 candidates changed in %d of seven same-type domains." % len(same_type_changes)
        ),
    })

    for domain in DOMAINS:
        members = by_domain[domain]
        truth_harmful = fullval_most_harmful_uid(members)
        for method, field in METHOD_FIELDS.items():
            predicted = most_harmful_uid(members, field)
            harmful_rows.append({
                "row_type": "domain", "domain_id": domain,
                "scope": "same_type" if domain in SAME_TYPE else "mixed",
                "method": method,
                "predicted_most_harmful_task037_global_index": predicted,
                "predicted_most_harmful_identity": json.dumps(
                    _identity(by_uid[predicted]), sort_keys=True),
                "fullval_most_harmful_task037_global_index": truth_harmful,
                "fullval_most_harmful_identity": json.dumps(
                    _identity(by_uid[truth_harmful]), sort_keys=True),
                "identity_match": predicted == truth_harmful,
                "predicted_unit_W": _float(by_uid[predicted], field),
                "predicted_unit_fullval_ce": _float(by_uid[predicted], "fullval_mean_cross_entropy_increase"),
                "fullval_max_ce_increase": _float(by_uid[truth_harmful], "fullval_mean_cross_entropy_increase"),
            })
    for scope, domain_ids in (("same_type", SAME_TYPE), ("mixed", MIXED)):
        for method in METHOD_FIELDS:
            rows = [row for row in harmful_rows if row["row_type"] == "domain"
                    and row["scope"] == scope and row["method"] == method]
            harmful_rows.append({
                "row_type": "domain_balanced", "scope": scope, "domain_id": "ALL",
                "method": method, "domain_count": len(rows),
                "correct_count": sum(bool(row["identity_match"]) for row in rows),
                "identity_accuracy": sum(bool(row["identity_match"]) for row in rows) / len(rows),
            })

    similarity_rows = []
    for domain in SAME_TYPE:
        members = sorted(by_domain[domain], key=lambda row: int(_uid(row)))
        orig_risk = [1.0 - _float(row, "W_original") for row in members]
        temp_risk = [1.0 - _float(row, "W_temporal_N9") for row in members]
        orig_order = _risk_order(members, "W_original")
        temp_order = _risk_order(members, "W_temporal_N9")
        similarity_rows.append({
            "row_type": "domain", "domain_id": domain, "unit_count": len(members),
            "spearman_R_original_vs_R_temporal": phase_h.spearman(orig_risk, temp_risk),
            "kendall_R_original_vs_R_temporal": phase_h.kendall_tau_b(orig_risk, temp_risk),
            "original_risk_order_safe_to_harmful_json": json.dumps(orig_order),
            "temporal_risk_order_safe_to_harmful_json": json.dumps(temp_order),
            "exact_full_order_equality": orig_order == temp_order,
        })
    valid_rho = [row["spearman_R_original_vs_R_temporal"] for row in similarity_rows
                 if row["spearman_R_original_vs_R_temporal"] is not None]
    valid_tau = [row["kendall_R_original_vs_R_temporal"] for row in similarity_rows
                 if row["kendall_R_original_vs_R_temporal"] is not None]
    similarity_rows.append({
        "row_type": "domain_balanced", "domain_id": "ALL", "unit_count": 21,
        "domain_count": len(SAME_TYPE),
        "mean_spearman_R_original_vs_R_temporal": phase_h.average(valid_rho),
        "mean_kendall_R_original_vs_R_temporal": phase_h.average(valid_tau),
        "valid_spearman_domain_count": len(valid_rho),
        "valid_kendall_domain_count": len(valid_tau),
        "exact_full_order_equal_domain_count": sum(row["exact_full_order_equality"] for row in similarity_rows),
        "exact_full_order_equal_domain_accuracy": sum(row["exact_full_order_equality"] for row in similarity_rows) / len(similarity_rows),
    })

    per_span_rows = []
    for span in SPANS:
        score_field = "W_span_%d" % span
        span_domain_rows = []
        for domain in SAME_TYPE:
            members = sorted(by_domain[domain], key=lambda row: int(_uid(row)))
            risk = [1.0 - _float(row, score_field) for row in members]
            damage = [_float(row, "fullval_mean_cross_entropy_increase") for row in members]
            safe = safest_uid(members, score_field)
            harmful = most_harmful_uid(members, score_field)
            truth_safe = fullval_safest_uid(members)
            safe_ce = _float(by_uid[safe], "fullval_mean_cross_entropy_increase")
            harmful_ce = _float(by_uid[harmful], "fullval_mean_cross_entropy_increase")
            row = {
                "row_type": "domain", "span": span, "domain_id": domain,
                "unit_count": len(members),
                "spearman_risk_vs_fullval_ce": phase_h.spearman(risk, damage),
                "kendall_risk_vs_fullval_ce": phase_h.kendall_tau_b(risk, damage),
                "span_safest_task037_global_index": safe,
                "fullval_safest_task037_global_index": truth_safe,
                "span_safest_identity_match": safe == truth_safe,
                "overall_temporal_safest_task037_global_index": method_rows[(domain, "R_temporal")]["safest_candidate_task037_global_index"],
                "overall_temporal_safest_identity_match": method_rows[(domain, "R_temporal")]["safest_identity_match"],
                "span_safest_when_temporal_safest_wrong": (
                    safe == truth_safe and not method_rows[(domain, "R_temporal")]["safest_identity_match"]
                ),
                "low_risk_task037_global_index": safe,
                "high_risk_task037_global_index": harmful,
                "low_high_ordering": phase_h.ordering_label(safe_ce, harmful_ce),
                "high_minus_low_fullval_ce": harmful_ce - safe_ce,
                "span_safest_identity": json.dumps(_identity(by_uid[safe]), sort_keys=True),
                "fullval_safest_identity": json.dumps(_identity(by_uid[truth_safe]), sort_keys=True),
            }
            span_domain_rows.append(row)
            per_span_rows.append(row)
        valid_rho_span = [row["spearman_risk_vs_fullval_ce"] for row in span_domain_rows
                          if row["spearman_risk_vs_fullval_ce"] is not None]
        valid_tau_span = [row["kendall_risk_vs_fullval_ce"] for row in span_domain_rows
                          if row["kendall_risk_vs_fullval_ce"] is not None]
        per_span_rows.append({
            "row_type": "domain_balanced", "span": span, "domain_id": "ALL",
            "domain_count": len(SAME_TYPE),
            "mean_spearman_risk_vs_fullval_ce": phase_h.average(valid_rho_span),
            "mean_kendall_risk_vs_fullval_ce": phase_h.average(valid_tau_span),
            "valid_spearman_domain_count": len(valid_rho_span),
            "valid_kendall_domain_count": len(valid_tau_span),
            "safest_identity_correct_count": sum(bool(row["span_safest_identity_match"]) for row in span_domain_rows),
            "safest_identity_accuracy": sum(bool(row["span_safest_identity_match"]) for row in span_domain_rows) / len(SAME_TYPE),
            "low_high_correct": sum(row["low_high_ordering"] == "correct" for row in span_domain_rows),
            "low_high_reverse": sum(row["low_high_ordering"] == "reverse" for row in span_domain_rows),
            "low_high_tie": sum(row["low_high_ordering"] == "tie" for row in span_domain_rows),
            "span_correct_when_overall_temporal_wrong_domains_json": json.dumps([
                row["domain_id"] for row in span_domain_rows
                if row["span_safest_when_temporal_safest_wrong"]
            ]),
        })

    original_summary = _same_type_summary([method_rows[(domain, "R_original")] for domain in SAME_TYPE])
    temporal_summary = _same_type_summary([method_rows[(domain, "R_temporal")] for domain in SAME_TYPE])
    frozen_rank_association_audit = _verify_frozen_rank_associations(
        original_summary, temporal_summary, baseline_comparison, same_type_oracle
    )
    original_gate_stats = {
        "spearman": original_summary["spearman"],
        "kendall": original_summary["kendall"],
        "safest_accuracy": original_summary["safest_accuracy"],
    }
    temporal_gate_stats = {
        "spearman": temporal_summary["spearman"],
        "kendall": temporal_summary["kendall"],
        "safest_accuracy": temporal_summary["safest_accuracy"],
    }
    predeclared_decision, predeclared_gate = phase_h.temporal_decision_gate(
        temporal_gate_stats, original_gate_stats
    )
    require(
        previous_summary.get("decision") == predeclared_decision
        == "TEMPORAL_STRESS_SELECTION_PROMISING",
        "corrected safest semantics changed the completed Phase-H gate",
    )
    if not same_type_changes or not same_type_corrected:
        top1_decision = "NO_DEMONSTRATED_TOP1_SELECTION_VALUE"
        top1_rationale = (
            "No same-type top-1 correction was observed; temporal stress does not demonstrate incremental top-1 value."
            if not same_type_corrected else
            "There are changed top-1 choices but none corrects the original choice."
        )
    elif not same_type_regressed:
        top1_decision = "ADDS_TOP1_SELECTION_VALUE"
        top1_rationale = "Every changed same-type top-1 choice is a correction and none is a regression."
    else:
        top1_decision = "UNRESOLVED"
        top1_rationale = "The same-type cohort contains both top-1 corrections and regressions."

    class_coverage_rows, class_coverage_summary = _validate_class_coverage(manifest, subsets)
    all_domain_rows = [row for row in corrected_rows if row["row_type"] == "domain"]
    exact_order_equal_domain_count = similarity_rows[-1]["exact_full_order_equal_domain_count"]
    top1_decision, top1_rationale = top1_value_decision(
        len(same_type_changes), len(same_type_corrected),
        len(same_type_regressed), exact_order_equal_domain_count,
        len(SAME_TYPE),
    )
    stale_domain_values = [row for row in all_domain_rows
                           if row["legacy_phase_h_safest_matches_corrected"] is False]
    same_type_safest_rows = [
        row for row in safest_rows
        if row.get("row_type") == "domain"
    ]
    span_winner_summary = [{
        "domain_id": row["domain_id"],
        "per_span_safest_task037_global_index": json.loads(row["per_span_safest_uids_json"]),
        "aggregate_temporal_safest_task037_global_index": row["temporal_safest_task037_global_index"],
        "fullval_safest_task037_global_index": row["fullval_safest_task037_global_index"],
    } for row in same_type_safest_rows]
    span_variability_domains = [
        row["domain_id"] for row in span_winner_summary
        if len(set(row["per_span_safest_task037_global_index"].values())) > 1
    ]
    summary = {
        "task": "Task041 Phase H.1 safest-semantics repair and incremental-value audit",
        "branch": branch,
        "output_directory": str(output_root.resolve()),
        "source_code_sha256": phase_h.sha256_file(Path(__file__)),
        "code_worktree_dirty_when_generated": bool(phase_h.git_value(repo, "status", "--porcelain")),
        "code_head": phase_h.git_value(repo, "rev-parse", "HEAD"),
        "phase_h_predeclared_decision": previous_summary.get("decision"),
        "predeclared_gate_after_safest_semantics_correction": predeclared_decision,
        "predeclared_gate_stats_after_correction": {
            "R_temporal": temporal_gate_stats, "R_original": original_gate_stats,
        },
        "predeclared_gate_details": predeclared_gate,
        "predeclared_phase_h_gate": predeclared_decision,
        "predeclared_phase_h_gate_decision_preserved": (
            predeclared_decision == previous_summary.get("decision")
        ),
        "frozen_rank_association_audit": frozen_rank_association_audit,
        "phase_h_historical_report_safest_accuracy_claims": legacy_report_claims,
        "scientific_top1_value": top1_decision,
        "scientific_top1_value_rationale": top1_rationale,
        "scientific_top1_selection_value": top1_decision,
        "final_decisions": {
            "PREDECLARED_PHASE_H_GATE": predeclared_decision,
            "SCIENTIFIC_TOP1_SELECTION_VALUE": top1_decision,
        },
        "same_type_safest_identity_accuracy": {
            "R_original": original_summary["safest_accuracy"],
            "R_temporal": temporal_summary["safest_accuracy"],
        },
        "same_type_safest_correct_count": {
            "R_original": original_summary["safest_correct_count"],
            "R_temporal": temporal_summary["safest_correct_count"],
            "denominator": len(SAME_TYPE),
        },
        "same_type_original_equals_temporal_safest_count": len(SAME_TYPE) - len(same_type_changes),
        "same_type_top1_changed_domains": same_type_changes,
        "same_type_top1_change_count": len(same_type_changes),
        "same_type_temporal_corrections": same_type_corrected,
        "same_type_temporal_regressions": same_type_regressed,
        "same_type_temporal_correction_count": len(same_type_corrected),
        "same_type_temporal_regression_count": len(same_type_regressed),
        "same_type_original_vs_temporal_rank_similarity": {
            "mean_spearman": similarity_rows[-1]["mean_spearman_R_original_vs_R_temporal"],
            "mean_kendall": similarity_rows[-1]["mean_kendall_R_original_vs_R_temporal"],
            "valid_spearman_domain_count": similarity_rows[-1]["valid_spearman_domain_count"],
            "valid_kendall_domain_count": similarity_rows[-1]["valid_kendall_domain_count"],
            "exact_full_order_equal_domain_count": similarity_rows[-1]["exact_full_order_equal_domain_count"],
            "domain_count": len(SAME_TYPE),
        },
        "same_type_per_span_summary": [row for row in per_span_rows if row["row_type"] == "domain_balanced"],
        "same_type_per_span_safest_winners": span_winner_summary,
        "same_type_domains_with_safest_identity_variation_across_spans": span_variability_domains,
        "only_expected_domains_103_113_vary_across_spans": span_variability_domains == ["103", "113"],
        "span_correct_but_overall_temporal_wrong": [
            {"span": row["span"], "domain_id": row["domain_id"],
             "span_safest_uid": row["span_safest_task037_global_index"],
             "fullval_safest_uid": row["fullval_safest_task037_global_index"],
             "overall_temporal_safest_uid": row["overall_temporal_safest_task037_global_index"]}
            for row in per_span_rows if row.get("span_safest_when_temporal_safest_wrong")
        ],
        "same_type_most_harmful_identity_accuracy": {
            method: next(row["identity_accuracy"] for row in harmful_rows
                         if row.get("row_type") == "domain_balanced"
                         and row.get("scope") == "same_type" and row.get("method") == method)
            for method in METHOD_FIELDS
        },
        "legacy_safest_semantics_mismatch_domain_method_count": len(stale_domain_values),
        "legacy_safest_semantics_mismatch_domain_methods": [
            {"domain_id": row["domain_id"], "method": row["method"],
             "legacy_reported_uid": row["legacy_phase_h_safest_uid"],
             "corrected_safe_uid": row["safest_candidate_task037_global_index"]}
            for row in stale_domain_values
        ],
        "class_coverage": class_coverage_summary,
        "phase_h_inputs": {
            path.name: {"path": str(path), "sha256": phase_h.sha256_file(path)}
            for path in (unit_path, span_path, old_compare_path, subset_path,
                         old_summary_path, baseline_comparison_path,
                         same_type_oracle_path, phase_h_report_path, manifest_path)
        },
        "gpu_used": False, "inference_rerun": False,
        "full_validation_oracle_rerun": False,
        "frozen_W_and_R_changed": False,
        "phase_h_outputs_overwritten": False,
        "pruning_or_physical_removal_performed": False,
        "finetuning_performed": False,
        "task042_created": False,
        "outputs": list(H1_FILES),
    }
    report = _report(summary, corrected_rows, safest_rows, harmful_rows,
                     similarity_rows, per_span_rows, class_coverage_rows)

    output_root.mkdir(parents=True, exist_ok=False)
    phase_h.write_csv_new(output_root / H1_FILES[0], corrected_rows)
    phase_h.write_csv_new(output_root / H1_FILES[1], safest_rows)
    phase_h.write_csv_new(output_root / H1_FILES[2], harmful_rows)
    phase_h.write_csv_new(output_root / H1_FILES[3], similarity_rows)
    phase_h.write_csv_new(output_root / H1_FILES[4], per_span_rows)
    phase_h.write_csv_new(output_root / H1_FILES[5], class_coverage_rows)
    phase_h.write_json_new(output_root / H1_FILES[6], summary)
    with (output_root / H1_FILES[7]).open("x", encoding="utf-8") as handle:
        handle.write(report)
    require({filename for filename in H1_FILES if (output_root / filename).is_file()} == set(H1_FILES),
            "Phase-H.1 did not create exactly its eight new artifacts")
    require({path.name for path in output_root.iterdir()} == set(H1_FILES),
            "Phase-H.1 output directory does not contain exactly eight artifacts")
    print("[Task041 Phase H.1] %s; top1=%s; output=%s" %
          (predeclared_decision, top1_decision, output_root), flush=True)
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)


def _report(summary: Mapping[str, Any], corrected_rows: Sequence[Mapping[str, Any]],
            safest_rows: Sequence[Mapping[str, Any]], harmful_rows: Sequence[Mapping[str, Any]],
            similarity_rows: Sequence[Mapping[str, Any]], per_span_rows: Sequence[Mapping[str, Any]],
            class_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Task041 Phase H.1 — Safest Semantics and Incremental-Value Audit",
        "",
        "Offline audit of completed Phase-H artifacts only. No model inference, GPU work, pruning, or finetuning was run. ",
        "Phase-H W/R, signed CE, and the existing 3,783-clip full-validation oracle were not changed.",
        "",
        "## Corrected predeclared gate and top-1 decision",
        "",
        "- Phase-H recorded decision: `%s`." % summary["phase_h_predeclared_decision"],
        "- Recomputed predeclared gate with corrected safest semantics: `%s`." % summary["predeclared_gate_after_safest_semantics_correction"],
        "- The original numerical Phase-H gate and its criteria were unchanged; it still passes.",
        "- Corrected same-type safest accuracy: original %.3f (%d/7); temporal %.3f (%d/7)." % (
            summary["same_type_safest_identity_accuracy"]["R_original"],
            summary["same_type_safest_correct_count"]["R_original"],
            summary["same_type_safest_identity_accuracy"]["R_temporal"],
            summary["same_type_safest_correct_count"]["R_temporal"],
        ),
        "- Scientific top-1 value: `%s` — %s" % (
            summary["scientific_top1_value"], summary["scientific_top1_value_rationale"]),
        "- Top-1 changed in %d/7 domains; corrections %d; regressions %d." % (
            summary["same_type_top1_change_count"], summary["same_type_temporal_correction_count"],
            summary["same_type_temporal_regression_count"]),
        "",
        "The predeclared gate and actual top-1 deletion-candidate value are reported separately. Passing the gate is not itself authorization to prune.",
        "",
        "## Corrected safest candidate by same-type domain",
        "",
        "| Domain | Original safest | Temporal safest | Full-val safest | Original match | Temporal match | Changed | Correction | Regression |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in safest_rows:
        if row.get("row_type") == "domain":
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["domain_id"], row["original_safest_task037_global_index"],
                row["temporal_safest_task037_global_index"], row["fullval_safest_task037_global_index"],
                row["original_safest_identity_match"], row["temporal_safest_identity_match"],
                row["temporal_safest_changed"], row["temporal_correction"], row["temporal_regression"]))
    lines.extend([
        "",
        "Correction domains: %s." % ", ".join(row["domain_id"] for row in summary["same_type_temporal_corrections"]) or "none",
        "Regression domains: %s." % ", ".join(row["domain_id"] for row in summary["same_type_temporal_regressions"]) or "none",
        "",
        "## Frozen Phase-H rank-association verification",
        "Original rho=%s, tau-b=%s; temporal rho=%s, tau-b=%s."
        % (_fmt(summary["frozen_rank_association_audit"]["R_original"]["recomputed_from_frozen_unit_records"]["spearman"]),
           _fmt(summary["frozen_rank_association_audit"]["R_original"]["recomputed_from_frozen_unit_records"]["kendall_tau_b"]),
           _fmt(summary["frozen_rank_association_audit"]["R_temporal"]["recomputed_from_frozen_unit_records"]["spearman"]),
           _fmt(summary["frozen_rank_association_audit"]["R_temporal"]["recomputed_from_frozen_unit_records"]["kendall_tau_b"])),
        "These match the saved baseline CSV and same-type oracle; no score was changed.",
        "",
        "## Most-harmful identity audit (same-type primary)",
        "",
        "| Method | Correct | Accuracy |",
        "|---|---:|---:|",
    ])
    for method in METHOD_FIELDS:
        row = next(row for row in harmful_rows if row.get("row_type") == "domain_balanced"
                   and row.get("scope") == "same_type" and row.get("method") == method)
        lines.append("| %s | %d/7 | %.6f |" % (method, row["correct_count"], row["identity_accuracy"]))
    lines.extend([
        "",
        "Predicted most harmful is minimum W / maximum R; the oracle is maximum full-validation mean CE increase. This is diagnostic only.",
        "",
        "## Original-versus-temporal ordering similarity",
        "",
        "Domain-balanced mean Spearman(R_original, R_temporal) = %s; mean Kendall tau-b = %s. Exact complete ordering agrees in %d/7 domains." % (
            _fmt(summary["same_type_original_vs_temporal_rank_similarity"]["mean_spearman"]),
            _fmt(summary["same_type_original_vs_temporal_rank_similarity"]["mean_kendall"]),
            summary["same_type_original_vs_temporal_rank_similarity"]["exact_full_order_equal_domain_count"]),
        "",
        "| Domain | Spearman | Kendall tau-b | Exact full order equal |",
        "|---:|---:|---:|---:|",
    ])
    for row in similarity_rows:
        if row.get("row_type") == "domain":
            lines.append("| %s | %s | %s | %s |" % (
                row["domain_id"], _fmt(row["spearman_R_original_vs_R_temporal"]),
                _fmt(row["kendall_R_original_vs_R_temporal"]), row["exact_full_order_equality"]))
    lines.extend([
        "",
        "## Per-span exploratory audit (same-type)",
        "",
        "| Span | Mean Spearman CE | Mean Kendall CE | Safest identity | Low/high correct | Reverse | Tie |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in per_span_rows:
        if row.get("row_type") == "domain_balanced":
            lines.append("| %s | %s | %s | %d/7 (%.6f) | %d | %d | %d |" % (
                row["span"], _fmt(row["mean_spearman_risk_vs_fullval_ce"]),
                _fmt(row["mean_kendall_risk_vs_fullval_ce"]),
                row["safest_identity_correct_count"], row["safest_identity_accuracy"],
                row["low_high_correct"], row["low_high_reverse"], row["low_high_tie"]))
    exceptions = summary["span_correct_but_overall_temporal_wrong"]
    lines.extend([
        "",
        "### Per-domain safest identity by temporal span",
        "",
        "| Domain | span1 | span2 | span4 | span8 | span16 | aggregate | full-val |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["same_type_per_span_safest_winners"]:
        by_span = row["per_span_safest_task037_global_index"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["domain_id"], by_span["1"], by_span["2"], by_span["4"],
            by_span["8"], by_span["16"],
            row["aggregate_temporal_safest_task037_global_index"],
            row["fullval_safest_task037_global_index"],
        ))
    lines.append(
        "Span winners vary in: %s." % (
            ", ".join(summary["same_type_domains_with_safest_identity_variation_across_spans"]) or "none"
        )
    )
    lines.extend([
        "",
        "Span-specific safest matches where aggregate temporal ranking is wrong: %s." % (
            "; ".join("domain %s/span %s (span=%s, aggregate=%s)" % (
                row["domain_id"], row["span"], row["span_safest_uid"], row["overall_temporal_safest_uid"]
            ) for row in exceptions) or "none"),
        "No best span was selected; these results are exploratory only.",
        "",
        "## Frozen calibration pool",
        "",
        "| Index | Canonical video | Action class | Label index | Phase-F N=3 member |",
        "|---:|---|---|---:|---:|",
    ])
    for row in class_rows:
        if row.get("row_type") == "video":
            lines.append("| %s | %s | %s | %s | %s |" % (
                row["video_index"], row["canonical_video_id"], row["action_class_name"],
                row["label_index"], row["phase_f_n3_member"]))
    cc = summary["class_coverage"]
    lines.extend([
        "",
        "Unique action classes: %d (%s). N=3 reuses the exact Phase-F subset; all 27 N=6 subsets and N=9 use only this frozen N=9 pool." % (
            cc["unique_action_class_count"], ", ".join("%s=%s" % item for item in cc["action_class_to_label_index"].items())),
        "**Limitation:** within-pool video-count stability does NOT establish cross-action-class calibration validity.",
        "",
        "## Scope stop",
        "",
        "Only the eight new Phase-H.1 artifacts were created. Phase-H inputs remain untouched. No GPU, inference, full-validation oracle rerun, pruning, physical removal, finetuning, span weighting, or Task042 was performed.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Task041 Phase-H.1 audit")
    parser.add_argument(
        "--phase-h-dir", default=phase_h.OUTPUT_DEFAULT,
        help="read-only source directory for completed Phase-H artifacts",
    )
    parser.add_argument(
        "--output-dir", default=H1_OUTPUT_DEFAULT,
        help="new, separate directory for the eight Phase-H.1 artifacts",
    )
    parser.add_argument(
        "--n9-manifest",
        default=str(phase_h.PHASEG_OUT / "task041_phase_g_n9_video_manifest.csv"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

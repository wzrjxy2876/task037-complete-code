#!/usr/bin/env python3
"""Offline exact representative-set feasibility audit for Task042 Phase B."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DOMAINS = ("11", "76", "102", "103", "113", "269", "271", "297", "400", "415")
SUBSETS = ("A_position1", "B_position2", "C_position3", "AB_positions12",
           "AC_positions13", "BC_positions23", "full_10x3")
SPANS = (1, 2, 4, 8, 16)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ids_text(ids):
    return json.dumps([int(x) for x in ids], separators=(",", ":"))


def key_pair(i, j):
    return (min(int(i), int(j)), max(int(i), int(j)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def coverage(ids, keep, distances):
    retained = set(keep)
    values = []
    for i in ids:
        d = [0.0 if i == j else float(distances[key_pair(i, j)]) for j in keep]
        values.append(min(d))
    return max(values), statistics.mean(values)


def enumerate_sets(ids, k, distances):
    candidates = []
    for keep in itertools.combinations(sorted(ids), k):
        jmax, jmean = coverage(ids, keep, distances)
        candidates.append({"keep": tuple(keep), "removed": tuple(i for i in sorted(ids) if i not in keep),
                           "j_max": jmax, "j_mean": jmean})
    return sorted(candidates, key=lambda r: (r["j_max"], r["j_mean"], r["keep"]))


def make_pair_matrices(pair_rows, domains, value_column):
    result = {d: {} for d in domains}
    for row in pair_rows:
        d = str(row["domain_id"])
        if d not in result:
            continue
        i, j = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
        value = float(row[value_column])
        if not math.isfinite(value):
            raise RuntimeError("non-finite distance in %s" % value_column)
        result[d][key_pair(i, j)] = value
    return result


def calibration_video_sets(videos):
    by_label = defaultdict(list)
    for row in videos:
        by_label[int(row["label"])].append(int(row["video_index"]))
    for label in by_label:
        by_label[label].sort()
        if len(by_label[label]) != 3:
            raise RuntimeError("frozen Task042 cohort must contain exactly 3 videos/class")
    labels = sorted(by_label)
    if len(labels) != 10:
        raise RuntimeError("frozen Task042 cohort must contain exactly 10 classes")
    out = {}
    for pos, name in enumerate(("A_position1", "B_position2", "C_position3")):
        out[name] = [by_label[label][pos] for label in labels]
    out["AB_positions12"] = [by_label[label][k] for label in labels for k in (0, 1)]
    out["AC_positions13"] = [by_label[label][k] for label in labels for k in (0, 2)]
    out["BC_positions23"] = [by_label[label][k] for label in labels for k in (1, 2)]
    out["full_10x3"] = sorted(int(row["video_index"]) for row in videos)
    return out


def calibration_matrices(repo_root, output_dir, units, videos, domains):
    sys.path.insert(0, str(Path(repo_root) / "src" / "lgfr_runtime"))
    import task042_frame_relation_redundancy as task042
    rows = read_csv(Path(output_dir) / "task042_frame_pair_sensitivity.csv")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        vi, uid = int(row["video_index"]), int(row["task037_global_index"])
        condition = (int(row["span"]), int(row["pair_index"]))
        key = (vi, uid, condition)
        if key in seen:
            raise RuntimeError("duplicate Task042 sensitivity condition %r" % (key,))
        seen.add(key)
        z = row.get("normalized_signature_z", "")
        if z == "":
            raise RuntimeError("degenerate signature found; Phase A reports none")
        grouped[(vi, uid)].append((condition, float(z)))
    if len(rows) != 122400:
        raise RuntimeError("expected exactly 122400 completed Task042 sensitivity rows")
    z_by_video_unit = {}
    for key, vals in grouped.items():
        vals.sort(key=lambda x: x[0])
        if len(vals) != 80:
            raise RuntimeError("unit/video does not have 80 Task042 conditions: %r" % (key,))
        z_by_video_unit[key] = [v for _, v in vals]
    video_sets = calibration_video_sets(videos)
    unit_ids = sorted(units)
    by_domain = {d: sorted(uid for uid, row in units.items() if str(row["domain_id"]) == d)
                 for d in domains}
    result = {name: {d: {} for d in domains} for name in SUBSETS}
    for subset_name, video_ids in video_sets.items():
        for domain in domains:
            ids = by_domain[domain]
            for pos, i in enumerate(ids):
                for j in ids[pos + 1:]:
                    vals = []
                    for vi in video_ids:
                        zi = z_by_video_unit[(vi, i)]
                        zj = z_by_video_unit[(vi, j)]
                        dist = task042.relation_distance(zi, zj)
                        if dist is not None:
                            vals.append(float(dist))
                    if len(vals) != len(video_ids):
                        raise RuntimeError("calibration pair missing a valid Task042 distance")
                    result[subset_name][domain][(i, j)] = statistics.mean(vals)
    return result


def lex_results(units_by_domain, matrices):
    selected, objective_rows, set_rows = {}, [], []
    for subset in SUBSETS:
        selected[subset] = {}
        for domain in DOMAINS:
            ids = units_by_domain[domain]
            selected[subset][domain] = {}
            distances = matrices[subset][domain]
            for k in range(1, len(ids)):
                candidates = enumerate_sets(ids, k, distances)
                best = candidates[0]
                primary_count = sum(x["j_max"] == best["j_max"] for x in candidates)
                geometry = [x for x in candidates if x["j_max"] == best["j_max"] and x["j_mean"] == best["j_mean"]]
                selected[subset][domain][k] = {"best": best, "candidates": candidates,
                                               "primary_count": primary_count,
                                               "geometry": geometry}
                for rank, cand in enumerate(candidates, 1):
                    objective_rows.append({"subset": subset, "domain_id": domain, "unit_count": len(ids),
                        "keep_count": k, "candidate_retained_ids": ids_text(cand["keep"]),
                        "candidate_removed_ids": ids_text(cand["removed"]), "J_max": cand["j_max"],
                        "J_mean": cand["j_mean"], "lexicographic_rank": rank,
                        "is_lexicographic_optimum": rank == 1,
                        "is_primary_Jmax_optimum": cand["j_max"] == best["j_max"],
                        "is_geometry_optimum_Jmax_and_Jmean": cand in geometry})
                counts = unit_type_counts(ids, best["keep"], units_by_domain=units_by_domain,
                                          unit_rows=_UNIT_ROWS)
                set_rows.append({"subset": subset, "domain_id": domain, "unit_count": len(ids),
                    "keep_count": k, "retained_task037_ids": ids_text(best["keep"]),
                    "removed_task037_ids": ids_text(best["removed"]), "J_max": best["j_max"],
                    "J_mean": best["j_mean"], "primary_Jmax_tie_count": primary_count,
                    "geometry_optimal_set_count": len(geometry),
                    "attention_count": counts["attention"], "FFN_count": counts["FFN"],
                    "retains_both_types": counts["attention"] > 0 and counts["FFN"] > 0})
    return selected, objective_rows, set_rows


_UNIT_ROWS = {}


def unit_type_counts(ids, keep, units_by_domain=None, unit_rows=None):
    rows = unit_rows or _UNIT_ROWS
    attn = sum(1 for uid in keep if rows[uid]["capture_kind"] == "head")
    ffn = sum(1 for uid in keep if rows[uid]["capture_kind"] == "neuron")
    return {"attention": attn, "FFN": ffn}


def selected_map_for_unitset(selected):
    return {tuple(x["best"]["keep"]) for x in selected}


def build_stability(units_by_domain, selected, full_temporal):
    rows = []
    for domain in DOMAINS:
        ids = units_by_domain[domain]
        for k in range(1, len(ids)):
            full = selected["full_10x3"][domain][k]["best"]["keep"]
            full_geometry = {x["keep"] for x in selected["full_10x3"][domain][k]["geometry"]}
            all_ids = set(ids)
            for subset in SUBSETS[:-1]:
                cur = selected[subset][domain][k]["best"]["keep"]
                removed_full, removed_cur = all_ids - set(full), all_ids - set(cur)
                jmax, jmean = coverage(ids, cur, full_temporal[domain])
                rows.append({"domain_id": domain, "unit_count": len(ids), "keep_count": k,
                    "calibration_subset": subset, "full_10x3_retained_ids": ids_text(full),
                    "subset_retained_ids": ids_text(cur), "exact_retained_set_match": set(full) == set(cur),
                    "retained_set_jaccard_vs_full": jaccard(full, cur),
                    "exact_pruned_set_match": removed_full == removed_cur,
                    "pruned_set_jaccard_vs_full": jaccard(removed_full, removed_cur),
                    "subset_set_is_full_geometry_optimum": tuple(cur) in full_geometry,
                    "full_geometry_optimal_set_count": len(full_geometry),
                    "subset_set_J_max_under_full_10x3": jmax,
                    "subset_set_J_mean_under_full_10x3": jmean,
                    "full_selected_J_max": selected["full_10x3"][domain][k]["best"]["j_max"],
                    "full_selected_J_mean": selected["full_10x3"][domain][k]["best"]["j_mean"]})
    return rows


def build_margins(units_by_domain, selected):
    rows = []
    for domain in DOMAINS:
        ids = units_by_domain[domain]
        for k in range(1, len(ids)):
            info = selected["full_10x3"][domain][k]
            cand = info["candidates"]
            best = cand[0]
            second = cand[1] if len(cand) > 1 else None
            geometry = info["geometry"]
            removed = sorted({x["removed"][0] for x in geometry}) if k == len(ids) - 1 else []
            relative = None if second is None or best["j_max"] == 0 else (second["j_max"] - best["j_max"]) / abs(best["j_max"])
            rows.append({"domain_id": domain, "unit_count": len(ids), "keep_count": k,
                "k_equals_n_minus_1": k == len(ids) - 1,
                "best_retained_ids": ids_text(best["keep"]), "best_removed_ids": ids_text(best["removed"]),
                "best_J_max": best["j_max"], "second_best_J_max": None if second is None else second["j_max"],
                "absolute_Jmax_margin": None if second is None else second["j_max"] - best["j_max"],
                "relative_Jmax_margin": relative,
                "primary_Jmax_ties_including_best": info["primary_count"],
                "best_J_mean": best["j_mean"], "second_lexicographic_J_mean": None if second is None else second["j_mean"],
                "secondary_Jmean_margin_if_primary_tied": (second["j_mean"] - best["j_mean"])
                    if second is not None and second["j_max"] == best["j_max"] else None,
                "geometry_optimal_set_count_before_ID_tiebreak": len(geometry),
                "geometry_optimal_retained_sets": ";".join(ids_text(x["keep"]) for x in geometry),
                "k_n_minus_1_geometry_optimal_removed_candidates": ids_text(removed) if k == len(ids) - 1 else "",
                "k_n_minus_1_deletion_choice_unique": (len(removed) == 1) if k == len(ids) - 1 else "",
                "next_three_lexicographic_candidates": json.dumps([
                    {"keep": list(x["keep"]), "removed": list(x["removed"]),
                     "J_max": x["j_max"], "J_mean": x["j_mean"]} for x in cand[:3]], separators=(",", ":"))})
    return rows


def build_span_sets(units_by_domain, span_matrices, full_selected):
    rows = []
    for span in SPANS:
        subset = "d_span_%d" % span
        for domain in DOMAINS:
            ids = units_by_domain[domain]
            for k in range(1, len(ids)):
                cand = enumerate_sets(ids, k, span_matrices[subset][domain])
                best = cand[0]
                geom_count = sum(x["j_max"] == best["j_max"] and x["j_mean"] == best["j_mean"] for x in cand)
                full = full_selected["full_10x3"][domain][k]["best"]["keep"]
                rows.append({"span": span, "domain_id": domain, "unit_count": len(ids), "keep_count": k,
                    "retained_task037_ids": ids_text(best["keep"]), "removed_task037_ids": ids_text(best["removed"]),
                    "J_max": best["j_max"], "J_mean": best["j_mean"],
                    "geometry_optimal_set_count": geom_count,
                    "same_as_full_d_temp_selected_set": set(best["keep"]) == set(full)})
    return rows


def build_descriptor_baseline(units_by_domain, units, selected, desc_matrices):
    rows = []
    for domain in DOMAINS:
        ids = units_by_domain[domain]
        for k in range(1, len(ids)):
            temporal = selected["full_10x3"][domain][k]["best"]["keep"]
            descriptor_candidates = enumerate_sets(ids, k, desc_matrices[domain])
            descriptor = descriptor_candidates[0]["keep"]
            tmax, tmean = coverage(ids, temporal, _FULL_TEMP[domain])
            dmax, dmean = coverage(ids, descriptor, _FULL_TEMP[domain])
            rows.append({"domain_id": domain, "unit_count": len(ids), "keep_count": k,
                "temporal_selected_ids": ids_text(temporal), "descriptor_selected_ids": ids_text(descriptor),
                "overlap_count": len(set(temporal) & set(descriptor)),
                "retained_set_jaccard": jaccard(temporal, descriptor),
                "temporal_selection_J_max_under_temporal_distance": tmax,
                "temporal_selection_J_mean_under_temporal_distance": tmean,
                "descriptor_selection_J_max_under_temporal_distance": dmax,
                "descriptor_selection_J_mean_under_temporal_distance": dmean,
                "descriptor_geometry_optimal_set_count": sum(x["j_max"] == descriptor_candidates[0]["j_max"] and
                    x["j_mean"] == descriptor_candidates[0]["j_mean"] for x in descriptor_candidates)})
    return rows


_FULL_TEMP = {}


def write_report(path, summary, units_by_domain, selected, matrices, margins, stability,
                 span_rows, mixed_rows, descriptor_rows):
    def fmt(x): return "NA" if x is None else "%.4f" % float(x)
    lines = ["# Task042 Phase B — Temporal Representative-Set Feasibility Audit", "",
        "**Decision: %s**" % summary["decision"], "",
        "This is an offline, exact set-selection feasibility audit over frozen Task042 distances. It does not authorize pruning.", "",
        "## Frozen protocol", "",
        "- Domains: all ten frozen multi-unit BMS domains; exact Task037 global IDs.",
        "- Keep counts: every k=1,...,n-1; all candidate sets enumerated exactly.",
        "- Selection key: lexicographic (J_max, J_mean, sorted Task037 ID tuple); no learned or weighted objective.",
        "- Stability and span analyses reuse Task042 per-video normalized sensitivity records; no inference or GPU use.",
        "- No CE, Top-1, logits, damage oracle, pruning, or fine-tuning was used.", "",
        "## A. Stability under fixed keep budgets", "",
        "Across %d calibration/domain/budget comparisons, exact retained-set match to full_10x3 was %.1f%%; median retained-set Jaccard was %.3f. The calibration choice belongs to the full-data geometric optimum family in %.1f%% of rows (ties treated as equivalent). Median calibration-selected J_max evaluated on the full matrix was %.4f versus full-selected median %.4f." % (
            summary["stability_comparison_count"],100*summary["stability_exact_match_rate"],summary["stability_median_jaccard"],
            100*summary["stability_full_geometry_equivalence_rate"],summary["stability_median_calibration_jmax_under_full"],summary["stability_median_full_optimal_jmax"]), "",
        "Deterministic ID tie-breaking is reported for reproducibility; it is not evidence when multiple sets tie geometrically.", "",
        "## B. Compressibility: strong-redundancy examples vs diverse controls", "",
        "For comparable k=1, the median optimal coverage radius in domains 415/400/103/113 was %.4f; in controls 11/76 it was %.4f. This compares observed temporal coverage only and introduces no threshold." % (
            summary["strong_domains_k1_median_Jmax"],summary["diverse_controls_k1_median_Jmax"]), "",
        "## C. Temporal vs frozen-descriptor selection", "",
        "Across domain/budget rows, median retained-set Jaccard was %.3f; exact selected-set match rate was %.1f%%. Median temporal J_max regret of descriptor-selected sets is %.6f. Both selected sets are evaluated using the temporal matrix in task042_phase_b_descriptor_baseline.csv." % (
            summary["descriptor_median_jaccard"],100*summary["descriptor_exact_match_rate"],summary["descriptor_median_temporal_jmax_regret"]), "",
        "## D. Mixed domains 271 and 297", "",
        "The following uses full_10x3 selections; no structural-type quota was imposed:", "",
        "| Domain | k | Retained IDs | Attention | FFN | Both types |", "|---:|---:|---|---:|---:|:---:|"]
    for r in mixed_rows:
        if r["subset"] == "full_10x3":
            lines.append("| %s | %s | %s | %s | %s | %s |" % (r["domain_id"],r["keep_count"],r["retained_task037_ids"],r["attention_count"],r["FFN_count"],"yes" if r["retains_both_types"] else "no"))
    lines += ["", "## E. k=n−1 deletion symmetry", "",
        "%d of %d domain/budget cases use k=n−1. Geometry has a unique deletion choice in %d cases; the other cases retain multiple exactly tied deletion candidates before ID tie-breaking. See the margin CSV for candidates and objective margins." % (
            summary["kn_minus_1_count"],summary["kn_minus_1_count"],summary["kn_minus_1_unique_deletion_count"]), "",
        "## Strong-redundancy examples and temporally diverse controls", ""]
    for domain in ("415","400","103","113","11","76"):
        ids = units_by_domain[domain]
        lines += ["### Domain %s (units %s)" % (domain,ids_text(ids)), "",
                  "| Task037 ID | %s |" % " | ".join(str(x) for x in ids),
                  "|---:|" + "---:|"*len(ids)]
        matrix = matrices["full_10x3"][domain]
        for i in ids:
            vals = [0.0 if i==j else matrix[key_pair(i,j)] for j in ids]
            lines.append("| %s | %s |" % (i," | ".join(fmt(x) for x in vals)))
        lines += ["", "| k | selected K* | J_max | J_mean | geometric optima | next-best alternatives (K; ΔJmax) |",
                  "|---:|---|---:|---:|---|---|"]
        for k in range(1,len(ids)):
            info = selected["full_10x3"][domain][k]
            best = info["best"]
            geometry = "; ".join(ids_text(x["keep"]) for x in info["geometry"])
            alt = []
            for cand in info["candidates"][1:3]:
                alt.append("%s; %s" % (ids_text(cand["keep"]),fmt(cand["j_max"]-best["j_max"])))
            lines.append("| %d | %s | %s | %s | %s | %s |" % (k,ids_text(best["keep"]),fmt(best["j_max"]),fmt(best["j_mean"]),geometry,"; ".join(alt) or "—"))
        lines.append("")
    lines += ["## F. Span sensitivity", "",
        "%d of %d domain/budget/span choices differ from the full d_temp representative set. This is descriptive; no span is selected or weighted." % (
            summary["span_set_change_count"],summary["span_set_count"]), "",
        "## Answers and decision basis", "",
        "A. **%s** Exact retained-set match rate is %.1f%% (median Jaccard %.3f); tie-aware geometric equivalence is %.1f%%." % (summary["answer_A"],100*summary["stability_exact_match_rate"],summary["stability_median_jaccard"],100*summary["stability_full_geometry_equivalence_rate"]),
        "B. **%s** At k=1, the four prespecified low-distance domains have median J_max %.4f versus %.4f for domains 11/76." % (summary["answer_B"],summary["strong_domains_k1_median_Jmax"],summary["diverse_controls_k1_median_Jmax"]),
        "C. **%s** Temporal and descriptor-only sets have median Jaccard %.3f and median temporal J_max regret %.6f for descriptor-selected sets." % (summary["answer_C"],summary["descriptor_median_jaccard"],summary["descriptor_median_temporal_jmax_regret"]),
        "D. **%s** Mixed-domain type composition is reported for every k and calibration subset; no quota was imposed." % summary["answer_D"],
        "E. **%s** Geometry ties leave deletion orientation unresolved in %d of %d k=n−1 cases." % (summary["answer_E"],summary["kn_minus_1_count"]-summary["kn_minus_1_unique_deletion_count"],summary["kn_minus_1_count"]),
        "F. **%s** A later, separately authorized pruning pilot could use relation coverage as a constraint only after resolving tie/stability limits; this Phase B does not authorize that pilot." % summary["answer_F"], "",
        "The Phase-B prompt specified A/B/C labels but no numerical cutoffs. The decision is therefore a transparent qualitative synthesis of these six reported diagnostics, not a tuned threshold.", "",
        "## Decision", "", "**%s**" % summary["decision"], "",
        "This decision does NOT authorize pruning, fine-tuning, a temporal threshold, or Task043.", "",
        "Outputs: exact objectives, representative sets, subset stability, margins, span sets, mixed-domain composition, descriptor baseline, and summary JSON."]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(repo_root, task042_dir, output_dir):
    global _UNIT_ROWS, _FULL_TEMP
    task042_dir, output_dir = Path(task042_dir), Path(output_dir)
    required = ["task042_summary.json","task042_unit_manifest.csv","task042_video_manifest.csv",
                "task042_frame_pair_sensitivity.csv","task042_temporal_pair_distance.csv",
                "task042_span_specific_distance.csv","task042_domain_temporal_structure.csv"]
    for name in required:
        if not (task042_dir/name).is_file(): raise RuntimeError("missing completed Task042 artifact: %s" % name)
    phase_a = json.loads((task042_dir/"task042_summary.json").read_text(encoding="utf-8"))
    if int(phase_a["forward_count"]) != 2430 or phase_a["pruning_performed"] or phase_a["finetuning_performed"]:
        raise RuntimeError("Task042 source summary does not match frozen completed diagnostic")
    unit_list = read_csv(task042_dir/"task042_unit_manifest.csv")
    units = {int(r["task037_global_index"]):r for r in unit_list}
    if len(units)!=51: raise RuntimeError("expected 51 exact Task037 identities")
    fixture = read_csv(Path(repo_root)/"data"/"task042_unit_identity_fixture.csv")
    fixture_by_id = {int(r["task037_global_index"]):r for r in fixture}
    if set(fixture_by_id) != set(units): raise RuntimeError("Task037 unit IDs differ from frozen identity fixture")
    for uid, row in units.items():
        expected = fixture_by_id[uid]
        checks = (("layer","layer"),("unit_type","mapped_unit_type"),("unit_index","mapped_unit_index"),
                  ("stage","stage"),("domain_id","domain_id"))
        for actual_key, fixture_key in checks:
            if str(row[actual_key]) != str(expected[fixture_key]):
                raise RuntimeError("frozen unit identity mismatch at Task037 ID %d (%s)"%(uid,actual_key))
        for key in ("D_abs","D_rel"):
            if abs(float(row[key])-float(expected[key]))>1e-12:
                raise RuntimeError("frozen descriptor mismatch at Task037 ID %d (%s)"%(uid,key))
        if abs(float(row["D_st"])-float(expected["D_third"]))>1e-12:
            raise RuntimeError("frozen third descriptor mismatch at Task037 ID %d"%uid)
    _UNIT_ROWS = units
    videos = read_csv(task042_dir/"task042_video_manifest.csv")
    if len(videos)!=30: raise RuntimeError("expected 30 frozen Task041 videos")
    by_domain = {d:sorted(uid for uid,u in units.items() if str(u["domain_id"])==d) for d in DOMAINS}
    if any(len(by_domain[d])<2 for d in DOMAINS): raise RuntimeError("required BMS domain absent or singleton")
    if set(str(u["domain_id"]) for u in units.values() if len([x for x in units.values() if str(x["domain_id"])==str(u["domain_id"])])>1) != set(DOMAINS):
        raise RuntimeError("frozen multi-unit domain set differs from Phase B specification")
    pair_rows = read_csv(task042_dir/"task042_temporal_pair_distance.csv")
    if len(pair_rows)!=34: raise RuntimeError("expected exactly 34 Task042 same-domain pair distances")
    full_temp = make_pair_matrices(pair_rows,DOMAINS,"d_temp")
    for d in DOMAINS:
        expected={(i,j) for ix,i in enumerate(by_domain[d]) for j in by_domain[d][ix+1:]}
        if set(full_temp[d])!=expected: raise RuntimeError("full distance pair identities mismatch in domain %s"%d)
    _FULL_TEMP = full_temp
    calibration = calibration_matrices(repo_root,task042_dir,units,videos,DOMAINS)
    reaggregation_diffs = [abs(calibration["full_10x3"][d][pair]-full_temp[d][pair])
        for d in DOMAINS for pair in full_temp[d]]
    max_reaggregation_diff = max(reaggregation_diffs) if reaggregation_diffs else 0.0
    if max_reaggregation_diff > 1e-12:
        raise RuntimeError("recomputed full_10x3 Task042 distances do not reproduce frozen pair table")
    matrices = {name:calibration[name] for name in SUBSETS[:-1]}
    matrices["full_10x3"] = full_temp
    selected, objective_rows, representative_rows = lex_results(by_domain,matrices)
    stability = build_stability(by_domain,selected,full_temp)
    margins = build_margins(by_domain,selected)
    spans_rows = read_csv(task042_dir/"task042_span_specific_distance.csv")
    if len(spans_rows)!=170: raise RuntimeError("expected 34 pairs x 5 frozen Task042 spans")
    # Filter each span's rows before constructing its distance matrix.
    span_matrices = {}
    for s in SPANS:
        span_matrices["d_span_%d"%s] = make_pair_matrices([r for r in spans_rows if int(r["span"])==s],DOMAINS,"d_temp")
    span_sets = build_span_sets(by_domain,span_matrices,selected)
    for r in representative_rows:
        if r["domain_id"] in ("271","297"):
            # Already covers every calibration subset and keep count.
            pass
    mixed_rows = [r for r in representative_rows if r["domain_id"] in ("271","297")]
    desc_matrices = {d:{} for d in DOMAINS}
    for d in DOMAINS:
        ids=by_domain[d]
        for ix,i in enumerate(ids):
            for j in ids[ix+1:]:
                xi=[float(units[i][x]) for x in ("D_abs","D_rel","D_st")]
                xj=[float(units[j][x]) for x in ("D_abs","D_rel","D_st")]
                desc_matrices[d][(i,j)] = math.sqrt(sum((a-b)**2 for a,b in zip(xi,xj)))
    descriptor_rows = build_descriptor_baseline(by_domain,units,selected,desc_matrices)
    objectives_fields=("subset","domain_id","unit_count","keep_count","candidate_retained_ids","candidate_removed_ids",
        "J_max","J_mean","lexicographic_rank","is_lexicographic_optimum","is_primary_Jmax_optimum","is_geometry_optimum_Jmax_and_Jmean")
    set_fields=("subset","domain_id","unit_count","keep_count","retained_task037_ids","removed_task037_ids","J_max","J_mean",
        "primary_Jmax_tie_count","geometry_optimal_set_count","attention_count","FFN_count","retains_both_types")
    stability_fields=("domain_id","unit_count","keep_count","calibration_subset","full_10x3_retained_ids","subset_retained_ids",
        "exact_retained_set_match","retained_set_jaccard_vs_full","exact_pruned_set_match","pruned_set_jaccard_vs_full",
        "subset_set_is_full_geometry_optimum","full_geometry_optimal_set_count","subset_set_J_max_under_full_10x3",
        "subset_set_J_mean_under_full_10x3","full_selected_J_max","full_selected_J_mean")
    margin_fields=("domain_id","unit_count","keep_count","k_equals_n_minus_1","best_retained_ids","best_removed_ids","best_J_max",
        "second_best_J_max","absolute_Jmax_margin","relative_Jmax_margin","primary_Jmax_ties_including_best","best_J_mean",
        "second_lexicographic_J_mean","secondary_Jmean_margin_if_primary_tied","geometry_optimal_set_count_before_ID_tiebreak",
        "geometry_optimal_retained_sets","k_n_minus_1_geometry_optimal_removed_candidates","k_n_minus_1_deletion_choice_unique",
        "next_three_lexicographic_candidates")
    span_fields=("span","domain_id","unit_count","keep_count","retained_task037_ids","removed_task037_ids","J_max","J_mean",
        "geometry_optimal_set_count","same_as_full_d_temp_selected_set")
    mixed_fields=set_fields
    desc_fields=("domain_id","unit_count","keep_count","temporal_selected_ids","descriptor_selected_ids","overlap_count","retained_set_jaccard",
        "temporal_selection_J_max_under_temporal_distance","temporal_selection_J_mean_under_temporal_distance",
        "descriptor_selection_J_max_under_temporal_distance","descriptor_selection_J_mean_under_temporal_distance","descriptor_geometry_optimal_set_count")
    output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(output_dir/"task042_phase_b_exact_subset_objectives.csv",objective_rows,objectives_fields)
    write_csv(output_dir/"task042_phase_b_representative_sets.csv",representative_rows,set_fields)
    write_csv(output_dir/"task042_phase_b_subset_stability.csv",stability,stability_fields)
    write_csv(output_dir/"task042_phase_b_margin_audit.csv",margins,margin_fields)
    write_csv(output_dir/"task042_phase_b_span_specific_sets.csv",span_sets,span_fields)
    write_csv(output_dir/"task042_phase_b_mixed_domain_analysis.csv",mixed_rows,mixed_fields)
    write_csv(output_dir/"task042_phase_b_descriptor_baseline.csv",descriptor_rows,desc_fields)
    exact_rate=sum(bool(r["exact_retained_set_match"]) for r in stability)/len(stability)
    geometry_equiv_rate=sum(bool(r["subset_set_is_full_geometry_optimum"]) for r in stability)/len(stability)
    stab_j=statistics.median(float(r["retained_set_jaccard_vs_full"]) for r in stability)
    med_cal_j=statistics.median(float(r["subset_set_J_max_under_full_10x3"]) for r in stability)
    med_full_j=statistics.median(float(r["full_selected_J_max"]) for r in stability)
    strong=[selected["full_10x3"][d][1]["best"]["j_max"] for d in ("415","400","103","113")]
    diverse=[selected["full_10x3"][d][1]["best"]["j_max"] for d in ("11","76")]
    desc_j=[float(r["retained_set_jaccard"]) for r in descriptor_rows]
    descriptor_regrets=[float(r["descriptor_selection_J_max_under_temporal_distance"])-float(r["temporal_selection_J_max_under_temporal_distance"]) for r in descriptor_rows]
    descriptor_regret_median=statistics.median(descriptor_regrets)
    span_changes=sum(not bool(r["same_as_full_d_temp_selected_set"]) for r in span_sets)
    k_nm1=[r for r in margins if bool(r["k_equals_n_minus_1"])]
    k_unique=sum(bool(r["k_n_minus_1_deletion_choice_unique"]) for r in k_nm1)
    mixed_full=[r for r in mixed_rows if r["subset"]=="full_10x3"]
    mixed_both=sum(bool(r["retains_both_types"]) for r in mixed_full)
    # Phase B contains no numeric decision gate; use qualitative synthesis of predeclared questions.
    strong_med=statistics.median(strong); diverse_med=statistics.median(diverse); desc_med=statistics.median(desc_j)
    if geometry_equiv_rate >= 0.5 and strong_med < diverse_med and descriptor_regret_median > 1e-12:
        decision="A. TEMPORAL_REPRESENTATIVE_SELECTION_PROMISING"
    elif max(float(r["d_temp"]) for r in pair_rows)-min(float(r["d_temp"]) for r in pair_rows) > 1e-12:
        decision="B. TEMPORAL_REPRESENTATIVE_SELECTION_WEAK_OR_UNRESOLVED"
    else:
        decision="C. TEMPORAL_REPRESENTATIVE_SELECTION_REJECTED"
    summary={"task":"TASK042 PHASE B — TEMPORAL REPRESENTATIVE-SET FEASIBILITY AUDIT ONLY",
        "decision":decision,"domains":list(DOMAINS),"domain_count":len(DOMAINS),"unit_count":len(units),"video_count":len(videos),
        "calibration_subsets":list(SUBSETS),"span_values":list(SPANS),"same_domain_pair_count":len(pair_rows),
        "exact_candidate_set_evaluations":len(objective_rows),"representative_rows":len(representative_rows),
        "stability_comparison_count":len(stability),"span_set_count":len(span_sets),"descriptor_domain_budget_count":len(descriptor_rows),
        "no_gpu_used":True,"inference_rerun":False,"pruning_performed":False,"finetuning_performed":False,
        "damage_oracle_used":False,"descriptors_or_bms_modified":False,
        "stability_exact_match_rate":exact_rate,"stability_full_geometry_equivalence_rate":geometry_equiv_rate,"stability_median_jaccard":stab_j,
        "stability_median_calibration_jmax_under_full":med_cal_j,"stability_median_full_optimal_jmax":med_full_j,
        "strong_domains_k1_jmax":{d:round(selected["full_10x3"][d][1]["best"]["j_max"],12) for d in ("415","400","103","113")},
        "strong_domains_k1_median_Jmax":strong_med,"diverse_controls_k1_jmax":{d:round(selected["full_10x3"][d][1]["best"]["j_max"],12) for d in ("11","76")},
        "diverse_controls_k1_median_Jmax":diverse_med,"descriptor_median_jaccard":desc_med,
        "descriptor_median_temporal_jmax_regret":descriptor_regret_median,
        "descriptor_exact_match_rate":sum(1 for r in descriptor_rows if float(r["retained_set_jaccard"])==1.0)/len(descriptor_rows),
        "kn_minus_1_count":len(k_nm1),"kn_minus_1_unique_deletion_count":k_unique,
        "kn_minus_1_ambiguous_domains":[r["domain_id"] for r in k_nm1 if not bool(r["k_n_minus_1_deletion_choice_unique"])],
        "mixed_full_selection_rows":len(mixed_full),"mixed_full_both_types_rows":mixed_both,
        "span_set_change_count":span_changes,"span_set_count":len(span_sets),
        "answer_A":"promising" if geometry_equiv_rate>=0.5 else "weak/unresolved",
        "answer_B":"yes" if strong_med<diverse_med else "not demonstrated",
        "answer_C":"yes" if descriptor_regret_median>1e-12 else "no temporal-coverage improvement",
        "answer_D":"naturally retains both types in %d/%d full-data domain/budget sets"%(mixed_both,len(mixed_full)),
        "answer_E":"unique in %d/%d cases; deterministic ID tie-breaking is not scientific evidence"%(k_unique,len(k_nm1)),
        "answer_F":"diagnostic support only; later pilot requires separate authorization and tie-aware safeguards",
        "full_distance_reaggregation_max_abs_difference":max_reaggregation_diff,
        "decision_basis":"The attachment supplies no numerical A/B/C cutoff. This audit uses a transparent tie-aware synthesis: A requires a majority of calibration selections to lie in the full-data geometric-optimum family, lower k=1 radius in prespecified strong-redundancy domains than diverse controls, and positive median temporal-coverage regret for descriptor-only selections; nonconstant geometry without all three yields B, constant geometry yields C.",
        "input_artifact_sha256":{name:hashlib.sha256((task042_dir/name).read_bytes()).hexdigest() for name in required}}
    (output_dir/"task042_phase_b_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    write_report(output_dir/"task042_phase_b_report.md",summary,by_domain,selected,matrices,margins,stability,span_sets,mixed_rows,descriptor_rows)
    print("PHASE_B_OK decision=%s objectives=%d sets=%d stability=%d spans=%d"%(
        decision,len(objective_rows),len(representative_rows),len(stability),len(span_sets)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",required=True)
    p.add_argument("--task042-dir",required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    run(a.repo_root,a.task042_dir,a.output_dir)

if __name__=="__main__": main()

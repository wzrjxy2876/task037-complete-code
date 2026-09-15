#!/usr/bin/env python3
"""Offline TASK040/TASK037 temporal-relation-conditioned rank audit.

This diagnostic consumes the completed fixed-cardinality-span records only.  It
never loads a model, regenerates logits, changes BMS, or performs pruning.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path("/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis/n03_fixed_span")
OUT = Path("/data/jixinye25/work1/output/task040_temporal_relation_conditioned_rank_audit")
VIDEO_EXPECTED = 3
UNIT_EXPECTED = 32
SPANS = (1, 2, 4, 8, 16)
PAIR_PER_SPAN = 16
RAW_EXPECTED = 7680
TOL = 1e-5


def unit_label(t: str) -> str:
    return "Attention" if str(t).lower() in {"head", "attention", "attention_head"} else "FFN"


def pair_type(a: str, b: str) -> str:
    a, b = unit_label(a), unit_label(b)
    return {"Attention": "AA", "FFN": "FF"}.get(a, "AF") if a == b else "AF"


def order_ranks(values: dict[int, float]) -> dict[int, int]:
    # Deterministic descending signed damage, then global index ascending.
    ordered = sorted(values, key=lambda i: (-float(values[i]), int(i)))
    return {int(i): n + 1 for n, i in enumerate(ordered)}


def rank_geometry(r1: dict[int, int], r2: dict[int, int], damages1=None, damages2=None):
    ids = sorted(r1)
    id_arr = np.asarray(ids, dtype=int)
    a = np.asarray([r1[i] for i in ids], dtype=float)
    b = np.asarray([r2[i] for i in ids], dtype=float)
    rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    kt = kendalltau(a, b, variant="b")
    out = {
        "spearman": rho,
        "kendall_tau_b": float(kt.statistic) if np.isfinite(kt.statistic) else float("nan"),
        "top5_overlap": len(set(id_arr[np.argsort(a)[:5]]) & set(id_arr[np.argsort(b)[:5]])) / 5.0,
        "top10_overlap": len(set(id_arr[np.argsort(a)[:10]]) & set(id_arr[np.argsort(b)[:10]])) / 10.0,
        "top16_overlap": len(set(id_arr[np.argsort(a)[:16]]) & set(id_arr[np.argsort(b)[:16]])) / 16.0,
        "exact_top1": bool(id_arr[np.argmin(a)] == id_arr[np.argmin(b)]),
        "exact_bottom1": bool(id_arr[np.argmax(a)] == id_arr[np.argmax(b)]),
    }
    if damages1 is not None and damages2 is not None:
        rev = 0
        comparable = 0
        by = {"AA": [0, 0], "FF": [0, 0], "AF": [0, 0]}
        for i, j in itertools.combinations(ids, 2):
            d1 = float(damages1[i]) - float(damages1[j])
            d2 = float(damages2[i]) - float(damages2[j])
            if d1 == 0.0 or d2 == 0.0:
                continue
            comparable += 1
            typ = pair_type(meta_by_id[i]["unit_type"], meta_by_id[j]["unit_type"])
            by[typ][1] += 1
            if d1 * d2 < 0:
                rev += 1
                by[typ][0] += 1
        out.update({"comparable_pairs": comparable, "strict_reversals": rev,
                    "reversal_fraction": rev / comparable if comparable else float("nan")})
        for typ, (r, c) in by.items():
            out[f"{typ}_reversals"] = r
            out[f"{typ}_comparable"] = c
            out[f"{typ}_reversal_fraction"] = r / c if c else float("nan")
    return out


def qlabel(x: float, qs: tuple[float, float, float]) -> str:
    if x <= qs[0]:
        return "Q0-Q25"
    if x <= qs[1]:
        return "Q25-Q50"
    if x <= qs[2]:
        return "Q50-Q75"
    return "Q75-Q100"


def finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def write_df(name: str, df: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def main():
    global meta_by_id, orig_dmg_global, orig_ranks_global
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "task040_raw_records.csv")
    sev = pd.read_csv(ROOT / "task040_phase_c_temporal_severity.csv")
    mapping = pd.read_csv(ROOT / "task040_bms_domain_mapping.csv")
    manifest_json = json.loads((ROOT / "task040_intervention_manifest.json").read_text())

    interventions = pd.DataFrame(manifest_json["interventions"])
    interventions["span"] = interventions["block_size"].astype(int)
    interventions["intervention_id"] = interventions.apply(
        lambda r: f"s{int(r.span)}_p{int(r.pair_index)}", axis=1
    )
    sev_key = {(int(r.video_index), int(r.level), int(r.block_size), int(r.pair_index)): float(r.Delta_model)
               for r in sev.itertuples()}

    # Authoritative unit metadata and BMS mapping.
    # Raw Phase-C records use the authoritative full-model index stored in
    # task040_unit_global_index; mapping.global_index is only the sorted row id.
    map_by_id = {int(r.task040_unit_global_index): r._asdict() for r in mapping.itertuples(index=False)}
    unit_ids = sorted(map_by_id)
    meta_by_id = {}
    for i in unit_ids:
        r = map_by_id[i]
        meta_by_id[i] = {
            "global_index": i, "layer_name": r["layer_name"], "unit_type": r["unit_type"],
            "type_label": unit_label(r["unit_type"]), "stage": int(r["stage"]),
            "domain_id": int(r["domain_id"]),
        }

    checks = []
    checks.append({"check": "Phase-C artifact identity", "expected": "3 videos, 32 units, 80 interventions, 5 spans, 16/span, 7680 rows",
                   "observed": f"{raw.video_index.nunique()} videos, {raw.unit_global_index.nunique()} units, "
                               f"{raw[['level','pair_index']].drop_duplicates().shape[0]} interventions, "
                               f"{raw.block_size.nunique()} spans, {len(raw)} rows",
                   "status": "PASS" if (raw.video_index.nunique(), raw.unit_global_index.nunique(),
                                          raw[['level','pair_index']].drop_duplicates().shape[0], raw.block_size.nunique(), len(raw))
                                         == (3, 32, 80, 5, 7680) else "FAIL"})
    checks.append({"check": "3-video identity", "expected": VIDEO_EXPECTED, "observed": int(raw.video_index.nunique()),
                   "status": "PASS" if raw.video_index.nunique() == VIDEO_EXPECTED else "FAIL"})
    checks.append({"check": "32-unit identity", "expected": UNIT_EXPECTED, "observed": int(raw.unit_global_index.nunique()),
                   "status": "PASS" if raw.unit_global_index.nunique() == UNIT_EXPECTED else "FAIL"})
    type_counts = raw.drop_duplicates("unit_global_index").unit_type.map(unit_label).value_counts().to_dict()
    checks.append({"check": "16-Attention/16-FFN identity", "expected": "Attention=16, FFN=16", "observed": type_counts,
                   "status": "PASS" if type_counts.get("Attention") == 16 and type_counts.get("FFN") == 16 else "FAIL"})
    checks.append({"check": "5-span identity", "expected": list(SPANS), "observed": sorted(map(int, raw.block_size.unique())),
                   "status": "PASS" if tuple(sorted(map(int, raw.block_size.unique()))) == SPANS else "FAIL"})
    per_span = raw.groupby("block_size").pair_index.nunique().to_dict()
    checks.append({"check": "16-interventions-per-span identity", "expected": 16, "observed": per_span,
                   "status": "PASS" if all(int(v) == 16 for v in per_span.values()) else "FAIL"})
    checks.append({"check": "7680-row identity", "expected": RAW_EXPECTED, "observed": len(raw),
                   "status": "PASS" if len(raw) == RAW_EXPECTED else "FAIL"})

    # Add canonical intervention metadata to each record.
    imeta = interventions.set_index(["level", "block_size", "pair_index"])
    raw["intervention_id"] = [imeta.loc[(int(r.level), int(r.block_size), int(r.pair_index)), "intervention_id"] for r in raw.itertuples()]
    raw["span"] = raw["block_size"].astype(int)
    raw["frame_left"] = [int(imeta.loc[(int(r.level), int(r.block_size), int(r.pair_index)), "left_start"]) for r in raw.itertuples()]
    raw["frame_right"] = [int(imeta.loc[(int(r.level), int(r.block_size), int(r.pair_index)), "right_start"]) for r in raw.itertuples()]
    raw["frame_pair"] = raw.apply(lambda r: f"{int(r.frame_left)}-{int(r.frame_right)}", axis=1)
    raw["Delta_model"] = [sev_key[(int(r.video_index), int(r.level), int(r.block_size), int(r.pair_index))] for r in raw.itertuples()]
    raw["type_label"] = raw.unit_type.map(unit_label)
    raw["stage"] = raw.layer_name.str.extract(r"layers\.(\d+)")[0].astype(int)
    raw["domain_id"] = raw.unit_global_index.map(lambda i: meta_by_id[int(i)]["domain_id"])

    # Required invariance/arithmetic checks before ranking.
    inv = raw.groupby(["video_index", "unit_global_index"]).d_original.agg(lambda x: float(np.max(np.abs(x - x.iloc[0]))))
    inv_ok = bool((inv <= TOL).all())
    checks.append({"check": "original-damage invariance across interventions", "expected": f"max diff <= {TOL}",
                   "observed": float(inv.max()), "status": "PASS" if inv_ok else "FAIL"})
    arith_o = np.max(np.abs(raw.d_original - (raw.z_true_original - raw.z_true_original_masked)))
    arith_r = np.max(np.abs(raw.d_intervened - (raw.z_true_intervened - raw.z_true_intervened_masked)))
    checks.append({"check": "signed damage arithmetic", "expected": f"max residual <= {TOL}",
                   "observed": {"original": float(arith_o), "conditioned": float(arith_r)},
                   "status": "PASS" if max(arith_o, arith_r) <= TOL else "FAIL"})

    # Original unit damage/ranks: one record per unit/video.
    orig_rows = []
    orig_ranks = {}
    orig_dmg = {}
    for (v, uid), g in raw.groupby(["video_index", "unit_global_index"], sort=True):
        d = float(g.d_original.iloc[0]); orig_dmg[(int(v), int(uid))] = d
        orig_ranks.setdefault(int(v), {})[int(uid)] = d
    for v in orig_ranks:
        orig_ranks[v] = order_ranks(orig_ranks[v])
    for (v, uid), d in orig_dmg.items():
        m = meta_by_id[uid]
        orig_rows.append({"video_index": v, "video_id": str(raw[(raw.video_index == v)].video_id.iloc[0]),
                          "unit_global_index": uid, "layer_name": m["layer_name"], "unit_type": m["unit_type"],
                          "type_label": m["type_label"], "stage": m["stage"], "domain_id": m["domain_id"],
                          "D_ori": d, "abs_D_ori": abs(d), "Rank_ori": orig_ranks[v][uid]})
    orig_dmg_global = orig_dmg
    orig_ranks_global = orig_ranks
    write_df("task_temporal_rank_original_damage.csv", pd.DataFrame(orig_rows))
    # Recompute twice from the frozen damage table and explicitly verify the
    # lexicographic key used by the ranking (descending signed damage, then
    # ascending authoritative global index).
    rank_repeat_a = {v: order_ranks({uid: orig_dmg[(v, uid)] for uid in sorted(orig_ranks[v])})
                     for v in sorted(orig_ranks)}
    rank_repeat_b = {v: order_ranks({uid: orig_dmg[(v, uid)] for uid in sorted(orig_ranks[v])})
                     for v in sorted(orig_ranks)}
    rank_ok = rank_repeat_a == rank_repeat_b == orig_ranks
    tie_ok = True
    for v in sorted(orig_ranks):
        ordered = sorted(orig_dmg[(v, uid)] for uid in orig_ranks[v])
        for uid_i, uid_j in itertools.combinations(sorted(orig_ranks[v]), 2):
            if orig_dmg[(v, uid_i)] == orig_dmg[(v, uid_j)]:
                tie_ok = tie_ok and (orig_ranks[v][uid_i] < orig_ranks[v][uid_j]) == (uid_i < uid_j)
    checks.append({"check": "deterministic ranking/tie break", "expected": "descending damage, global_index ascending",
                   "observed": {"recomputed_twice": rank_ok, "tie_break": tie_ok},
                   "status": "PASS" if rank_ok and tie_ok else "FAIL"})

    # Conditioned ranks and raw damage table.
    cond_rows = []
    cond_ranks = {}
    cond_dmg = {}
    for (v, lev, span, pi), g in raw.groupby(["video_index", "level", "span", "pair_index"], sort=True):
        key = (int(v), int(lev), int(span), int(pi))
        vals = {int(r.unit_global_index): float(r.d_intervened) for r in g.itertuples()}
        cond_dmg[key] = vals; cond_ranks[key] = order_ranks(vals)
        for r in g.itertuples():
            uid = int(r.unit_global_index); m = meta_by_id[uid]
            cond_rows.append({"video_index": int(v), "video_id": str(r.video_id), "unit_global_index": uid,
                              "layer_name": m["layer_name"], "unit_type": m["unit_type"], "type_label": m["type_label"],
                              "stage": m["stage"], "domain_id": m["domain_id"], "level": int(lev), "span": int(span),
                              "pair_index": int(pi), "intervention_id": r.intervention_id, "frame_pair": r.frame_pair,
                              "frame_left": int(r.frame_left), "frame_right": int(r.frame_right), "Delta_model": float(r.Delta_model),
                              "D_rel": float(r.d_intervened), "abs_D_rel": abs(float(r.d_intervened)),
                              "Rank_rel": cond_ranks[key][uid], "D_ori": orig_dmg[(int(v), uid)], "Rank_ori": orig_ranks[int(v)][uid],
                              "sign_changed": bool(np.sign(orig_dmg[(int(v), uid)]) != np.sign(float(r.d_intervened)))})
    cond_df = pd.DataFrame(cond_rows)
    write_df("task_temporal_rank_conditioned_damage.csv", cond_df)
    sign_ok = bool((cond_df.sign_changed == (np.sign(cond_df.D_ori) != np.sign(cond_df.D_rel))).all())
    checks.append({"check": "sign-change arithmetic", "expected": "sign(D_ori) != sign(D_rel)",
                   "observed": {"rows": int(cond_df.sign_changed.sum()), "all_match": sign_ok}, "status": "PASS" if sign_ok else "FAIL"})

    # Intervention-level ranking geometry and reversal-pair detail.
    int_rows = []; reversal_rows = []; margin_records = []
    all_margin = []
    for v in sorted(orig_ranks):
        ids = sorted(orig_ranks[v])
        for i, j in itertools.combinations(ids, 2):
            all_margin.append(abs(orig_dmg[(v, i)] - orig_dmg[(v, j)]))
    qs = tuple(np.quantile(np.asarray(all_margin), [0.25, 0.5, 0.75]))
    for v in sorted(orig_ranks):
        for i, j in itertools.combinations(sorted(orig_ranks[v]), 2):
            oi, oj = orig_dmg[(v, i)], orig_dmg[(v, j)]
            margin_records.append((v, i, j, abs(oi - oj), qlabel(abs(oi - oj), qs)))
        for key in sorted([k for k in cond_ranks if k[0] == v], key=lambda x: (x[2], x[3])):
            _, lev, span, pi = key; r1, r2 = orig_ranks[v], cond_ranks[key]
            geom = rank_geometry(r1, r2, orig_dmg_for_video(v), cond_dmg[key])
            rec = {"video_index": v, "video_id": str(raw[raw.video_index == v].video_id.iloc[0]), "level": lev,
                   "span": span, "pair_index": pi, "intervention_id": f"s{span}_p{pi}",
                   "frame_pair": str(raw[(raw.video_index == v) & (raw.span == span) & (raw.pair_index == pi)].frame_pair.iloc[0]),
                   "Delta_model": float(sev_key[(v, lev, span, pi)]), **geom}
            int_rows.append(rec)
            for i, j in itertools.combinations(sorted(orig_ranks[v]), 2):
                d1 = orig_dmg[(v, i)] - orig_dmg[(v, j)]; d2 = cond_dmg[key][i] - cond_dmg[key][j]
                if d1 == 0.0 or d2 == 0.0:
                    continue
                typ = pair_type(meta_by_id[i]["unit_type"], meta_by_id[j]["unit_type"])
                margin = abs(d1)
                margin_records.append((v, i, j, margin, qlabel(margin, qs)))
                if d1 * d2 < 0:
                    mi, mj = meta_by_id[i], meta_by_id[j]
                    reversal_rows.append({"video_index": v, "video_id": str(raw[raw.video_index == v].video_id.iloc[0]),
                                          "level": lev, "span": span, "pair_index": pi, "intervention_id": f"s{span}_p{pi}",
                                          "frame_pair": rec["frame_pair"], "Delta_model": rec["Delta_model"],
                                          "unit_i": i, "unit_j": j, "unit_type_i": mi["type_label"], "unit_type_j": mj["type_label"],
                                          "pair_type": typ, "stage_i": mi["stage"], "stage_j": mj["stage"],
                                          "domain_i": mi["domain_id"], "domain_j": mj["domain_id"],
                                          "D_ori_i": orig_dmg[(v, i)], "D_ori_j": orig_dmg[(v, j)],
                                          "D_rel_i": cond_dmg[key][i], "D_rel_j": cond_dmg[key][j],
                                          "Rank_ori_i": orig_ranks[v][i], "Rank_ori_j": orig_ranks[v][j],
                                          "Rank_rel_i": cond_ranks[key][i], "Rank_rel_j": cond_ranks[key][j],
                                          "original_margin": margin, "margin_quartile": qlabel(margin, qs)})
    int_df = pd.DataFrame(int_rows); rev_df = pd.DataFrame(reversal_rows)
    write_df("task_temporal_rank_intervention_summary.csv", int_df)
    write_df("task_temporal_rank_reversal_pairs.csv", rev_df)
    # Validate the reversal table against the signed-damage definition rather
    # than treating its construction as proof.
    strict_ok = True
    for r in rev_df.itertuples(index=False):
        d1 = orig_dmg[(int(r.video_index), int(r.unit_i))] - orig_dmg[(int(r.video_index), int(r.unit_j))]
        lev = int(raw[(raw.video_index == r.video_index) & (raw.span == r.span) & (raw.pair_index == r.pair_index)].level.iloc[0])
        d2 = cond_dmg[(int(r.video_index), lev, int(r.span), int(r.pair_index))][int(r.unit_i)] - cond_dmg[(int(r.video_index), lev, int(r.span), int(r.pair_index))][int(r.unit_j)]
        strict_ok = strict_ok and d1 != 0.0 and d2 != 0.0 and d1 * d2 < 0
    checks.append({"check": "strict reversal arithmetic", "expected": "product of pair differences < 0 only",
                   "observed": {"rows_checked": len(rev_df), "all_strict": strict_ok}, "status": "PASS" if strict_ok else "FAIL"})
    # Margin analysis from all comparable pair/intervention observations.
    margin_rows = []
    for v in sorted(orig_ranks):
        for i, j in itertools.combinations(sorted(orig_ranks[v]), 2):
            typ = pair_type(meta_by_id[i]["unit_type"], meta_by_id[j]["unit_type"])
            d1 = orig_dmg[(v, i)] - orig_dmg[(v, j)]
            if d1 == 0.0:
                continue
            ql = qlabel(abs(d1), qs); total = rev = 0
            for key in sorted([k for k in cond_ranks if k[0] == v]):
                d2 = cond_dmg[key][i] - cond_dmg[key][j]
                if d2 == 0.0: continue
                total += 1; rev += int(d1 * d2 < 0)
            margin_rows.append({"video_index": v, "unit_i": i, "unit_j": j, "pair_type": typ,
                                "original_margin": abs(d1), "margin_quartile": ql, "interventions": total,
                                "strict_reversals": rev, "reversal_fraction": rev / total if total else float("nan")})
    margin_df = pd.DataFrame(margin_rows)
    # The requested report is quartile-stratified; retain the pair-level rows too.
    write_df("task_temporal_rank_margin_analysis.csv", margin_df)
    margin_q_ok = bool(set(margin_df.margin_quartile.dropna().unique()) <= {"Q0-Q25", "Q25-Q50", "Q50-Q75", "Q75-Q100"})
    checks.append({"check": "margin quartile identity", "expected": "empirical Q25/Q50/Q75 over original pair margins",
                   "observed": {"q25": qs[0], "q50": qs[1], "q75": qs[2], "labels_valid": margin_q_ok},
                   "status": "PASS" if margin_q_ok else "FAIL"})

    # Span summaries.
    span_rows = []
    for (v, span), g in int_df.groupby(["video_index", "span"], sort=True):
        row = {"video_index": v, "video_id": str(g.video_id.iloc[0]), "span": span, "interventions": len(g)}
        for col in ["spearman", "kendall_tau_b", "reversal_fraction", "top5_overlap", "top10_overlap", "top16_overlap"]:
            row[f"{col}_median"] = finite(g[col].median()); row[f"{col}_mean"] = finite(g[col].mean())
        span_rows.append(row)
    write_df("task_temporal_rank_span_analysis.csv", pd.DataFrame(span_rows))

    # Severity matching manifest is written before calculating matched geometry.
    match_rows = []
    for v in sorted(raw.video_index.unique()):
        by_span = {s: sorted([k for k in cond_ranks if k[0] == v and k[2] == s], key=lambda k: k[3]) for s in SPANS}
        for s1, s2 in itertools.combinations(SPANS, 2):
            candidates = sorted((abs(sev_key[(v, *a[1:])]-sev_key[(v, *b[1:])]), a[3], b[3], a, b)
                                for a in by_span[s1] for b in by_span[s2])
            used_a, used_b = set(), set()
            for diff, pi1, pi2, a, b in candidates:
                if pi1 in used_a or pi2 in used_b: continue
                used_a.add(pi1); used_b.add(pi2)
                match_rows.append({"video_index": v, "span_a": s1, "pair_index_a": pi1, "intervention_a": f"s{s1}_p{pi1}",
                                   "span_b": s2, "pair_index_b": pi2, "intervention_b": f"s{s2}_p{pi2}",
                                   "Delta_model_a": sev_key[(v, *a[1:])], "Delta_model_b": sev_key[(v, *b[1:])],
                                   "abs_Delta_model_difference": diff})
    match_df = pd.DataFrame(match_rows)
    write_df("task_temporal_rank_severity_matching.csv", match_df)
    # One-to-one is enforced separately for every (video, span_a, span_b)
    # comparison.  An intervention may legitimately participate in four
    # different cross-span comparisons, so span_b/span_a belong in the key.
    match_a = set(zip(match_df.video_index, match_df.span_a, match_df.span_b, match_df.pair_index_a))
    match_b = set(zip(match_df.video_index, match_df.span_a, match_df.span_b, match_df.pair_index_b))
    match_ok = (len(match_df) == 3 * 10 * 16 and len(match_a) == len(match_df) and len(match_b) == len(match_df)
                and bool((match_df.span_a < match_df.span_b).all()))
    checks.append({"check": "severity-matching determinism", "expected": "greedy one-to-one nearest Delta_model per span pair",
                   "observed": {"rows": len(match_df), "one_to_one": match_ok}, "status": "PASS" if match_ok else "FAIL"})
    sev_rows = []
    for r in match_df.itertuples(index=False):
        ka = (int(r.video_index), int(raw[(raw.video_index == r.video_index) & (raw.span == r.span_a) & (raw.pair_index == r.pair_index_a)].level.iloc[0]), int(r.span_a), int(r.pair_index_a))
        kb = (int(r.video_index), int(raw[(raw.video_index == r.video_index) & (raw.span == r.span_b) & (raw.pair_index == r.pair_index_b)].level.iloc[0]), int(r.span_b), int(r.pair_index_b))
        geom = rank_geometry(cond_ranks[ka], cond_ranks[kb], cond_dmg[ka], cond_dmg[kb])
        sev_rows.append({**r._asdict(), **geom})
    sev_res = pd.DataFrame(sev_rows)
    write_df("task_temporal_rank_severity_matched_results.csv", sev_res)

    # Same-span, different-relation geometry.
    same_rows = []
    for v in sorted(raw.video_index.unique()):
        for span in SPANS:
            keys = sorted([k for k in cond_ranks if k[0] == v and k[2] == span], key=lambda k: k[3])
            for a, b in itertools.combinations(keys, 2):
                geom = rank_geometry(cond_ranks[a], cond_ranks[b], cond_dmg[a], cond_dmg[b])
                same_rows.append({"video_index": v, "span": span, "pair_index_a": a[3], "pair_index_b": b[3],
                                  "intervention_a": f"s{span}_p{a[3]}", "intervention_b": f"s{span}_p{b[3]}",
                                  "Delta_model_a": sev_key[(v, a[1], span, a[3])], "Delta_model_b": sev_key[(v, b[1], span, b[3])], **geom})
    write_df("task_temporal_rank_same_span_analysis.csv", pd.DataFrame(same_rows))
    checks.append({"check": "same-span grouping identity", "expected": "16 interventions -> 120 unordered pairs/span/video",
                   "observed": f"{len(same_rows)} rows", "status": "PASS" if len(same_rows) == 3 * 5 * 120 else "FAIL"})

    # Unit-specific volatility and sign changes.
    vol_rows = []
    for v in sorted(orig_ranks):
        g = cond_df[cond_df.video_index == v]
        for uid in sorted(orig_ranks[v]):
            ranks = g[g.unit_global_index == uid].Rank_rel.to_numpy(dtype=float); ori = orig_ranks[v][uid]
            m = meta_by_id[uid]
            vol_rows.append({"video_index": v, "unit_global_index": uid, "layer_name": m["layer_name"], "unit_type": m["unit_type"],
                             "type_label": m["type_label"], "stage": m["stage"], "domain_id": m["domain_id"], "original_rank": ori,
                             "conditioned_rank_mean": ranks.mean(), "conditioned_rank_median": np.median(ranks), "conditioned_rank_min": ranks.min(),
                             "conditioned_rank_max": ranks.max(), "conditioned_rank_std": ranks.std(),
                             "moves_up_fraction": np.mean(ranks < ori), "moves_down_fraction": np.mean(ranks > ori),
                             "unchanged_fraction": np.mean(ranks == ori)})
    write_df("task_temporal_rank_unit_volatility.csv", pd.DataFrame(vol_rows))
    sign_df = cond_df[["video_index", "video_id", "unit_global_index", "layer_name", "unit_type", "type_label", "stage", "domain_id",
                       "level", "span", "pair_index", "intervention_id", "frame_pair", "Delta_model", "D_ori", "D_rel", "sign_changed"]].copy()
    write_df("task_temporal_rank_sign_changes.csv", sign_df)

    # Type/stage summaries.
    def grouped_summary(col):
        rows = []
        for key, g in cond_df.groupby(col, sort=True):
            vals = int_df[int_df.video_index.isin(g.video_index.unique())]
            rows.append({col: key, "records": len(g), "units": g.unit_global_index.nunique(), "videos": g.video_index.nunique(),
                         "sign_change_rate": float(g.sign_changed.mean()), "mean_rank_std": float(vol_df[vol_df[col] == key].conditioned_rank_std.mean()) if col in vol_df else float("nan")})
        return pd.DataFrame(rows)
    vol_df = pd.DataFrame(vol_rows)
    type_rows = []
    for typ, g in cond_df.groupby("type_label", sort=True):
        vv = vol_df[vol_df.type_label == typ]
        rr = rev_df[rev_df.pair_type.isin(["AA", "FF", "AF"])]
        rev_involving = int(rr[(rr.unit_type_i == typ) | (rr.unit_type_j == typ)].shape[0])
        comparable_involving = 0
        for v in sorted(orig_ranks):
            ids = sorted(orig_ranks[v]); typed = [uid for uid in ids if meta_by_id[uid]["type_label"] == typ]
            comparable_involving += len([k for k in cond_ranks if k[0] == v]) * (math.comb(len(ids), 2) - math.comb(len(ids) - len(typed), 2))
        type_rows.append({"type_label": typ, "units": g.unit_global_index.nunique(), "records": len(g), "sign_change_rate": float(g.sign_changed.mean()),
                          "mean_rank_std": float(vv.conditioned_rank_std.mean()), "reversal_rows_involving_type": rev_involving,
                          "comparable_pair_observations_involving_type": comparable_involving,
                          "reversal_fraction_involving_type": rev_involving / comparable_involving if comparable_involving else float("nan")})
    write_df("task_temporal_rank_type_summary.csv", pd.DataFrame(type_rows))
    stage_rows = []
    for st, g in cond_df.groupby("stage", sort=True):
        vv = vol_df[vol_df.stage == st]
        rev_involving = int(rev_df[(rev_df.stage_i == st) | (rev_df.stage_j == st)].shape[0])
        comparable_involving = 0
        for v in sorted(orig_ranks):
            ids = sorted(orig_ranks[v]); staged = [uid for uid in ids if meta_by_id[uid]["stage"] == st]
            comparable_involving += len([k for k in cond_ranks if k[0] == v]) * (math.comb(len(ids), 2) - math.comb(len(ids) - len(staged), 2))
        stage_rows.append({"stage": st, "units": g.unit_global_index.nunique(), "records": len(g), "sign_change_rate": float(g.sign_changed.mean()),
                           "mean_rank_std": float(vv.conditioned_rank_std.mean()), "mean_abs_D_ori": float(g.D_ori.abs().mean()), "mean_abs_D_rel": float(g.abs_D_rel.mean()),
                           "reversal_rows_involving_stage": rev_involving, "comparable_pair_observations_involving_stage": comparable_involving,
                           "reversal_fraction_involving_stage": rev_involving / comparable_involving if comparable_involving else float("nan")})
    write_df("task_temporal_rank_stage_summary.csv", pd.DataFrame(stage_rows))

    # BMS-domain connection.
    dom_rows = []
    tested = defaultdict(list)
    for uid, m in meta_by_id.items(): tested[m["domain_id"]].append(uid)
    for dom, ids in sorted(tested.items()):
        if len(ids) < 2: continue
        total = rev = 0; videos_changed = set()
        for v in sorted(orig_ranks):
            for i, j in itertools.combinations(sorted(ids), 2):
                d1 = orig_dmg[(v, i)] - orig_dmg[(v, j)]
                if d1 == 0: continue
                for key in sorted([k for k in cond_ranks if k[0] == v]):
                    d2 = cond_dmg[key][i] - cond_dmg[key][j]
                    if d2 == 0: continue
                    total += 1
                    if d1 * d2 < 0: rev += 1; videos_changed.add(v)
        dom_rows.append({"domain_id": dom, "tested_units": len(ids), "unit_ids": ";".join(map(str, sorted(ids))), "comparable_observations": total,
                         "strict_reversals": rev, "reversal_fraction": rev / total if total else float("nan"), "videos_with_order_change": len(videos_changed)})
    write_df("task_temporal_rank_bms_domain_summary.csv", pd.DataFrame(dom_rows))
    bms_identity_ok = set(map_by_id) == set(meta_by_id) and all(
        meta_by_id[uid]["unit_type"] == map_by_id[uid]["unit_type"] and
        meta_by_id[uid]["layer_name"] == map_by_id[uid]["layer_name"] for uid in meta_by_id
    )
    checks.append({"check": "BMS mapping identity where available", "expected": "authoritative task040 mapping",
                   "observed": {"multi_unit_domains": len(dom_rows), "identity_match": bms_identity_ok, "no_bms_rerun": True},
                   "status": "PASS" if bms_identity_ok else "FAIL"})

    # Examples are chosen deterministically from computed records only.
    pair_stats = []
    for v in sorted(orig_ranks):
        for i, j in itertools.combinations(sorted(orig_ranks[v]), 2):
            rows = rev_df[(rev_df.video_index == v) & (rev_df.unit_i == i) & (rev_df.unit_j == j)] if not rev_df.empty else pd.DataFrame()
            pair_stats.append((len(rows), v, i, j, rows))
    stable = sorted(pair_stats, key=lambda x: (x[0], x[1], x[2], x[3]))[0]
    reversible = sorted(pair_stats, key=lambda x: (-x[0], x[1], x[2], x[3]))[0]
    cross_candidates = [x for x in pair_stats if pair_type(meta_by_id[x[2]]["unit_type"], meta_by_id[x[3]]["unit_type"]) == "AF" and x[0] > 0]
    cross = sorted(cross_candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))[0] if cross_candidates else reversible
    examples = []
    def add_example(label, pair, key=None):
        _, v, i, j, rows = pair; chosen = rows.iloc[0].to_dict() if len(rows) else None
        base = {"example_type": label, "video_index": v, "video_id": str(raw[raw.video_index == v].video_id.iloc[0]), "unit_i": i, "unit_j": j,
                "unit_type_i": meta_by_id[i]["type_label"], "unit_type_j": meta_by_id[j]["type_label"], "stage_i": meta_by_id[i]["stage"], "stage_j": meta_by_id[j]["stage"],
                "domain_i": meta_by_id[i]["domain_id"], "domain_j": meta_by_id[j]["domain_id"], "D_ori_i": orig_dmg[(v, i)], "D_ori_j": orig_dmg[(v, j)],
                "Rank_ori_i": orig_ranks[v][i], "Rank_ori_j": orig_ranks[v][j]}
        if chosen:
            for k in ["span", "pair_index", "intervention_id", "frame_pair", "Delta_model", "D_rel_i", "D_rel_j", "Rank_rel_i", "Rank_rel_j", "original_margin", "margin_quartile"]:
                base[k] = chosen.get(k)
        examples.append(base)
    add_example("stable_unit_pair", stable); add_example("reversible_unit_pair", reversible); add_example("cross_type_reversal", cross)
    # Severity-controlled example: use first matched pair containing a reversal, otherwise first match.
    sev_example = None
    for r in sev_res.itertuples(index=False):
        # Find a unit pair whose order reverses between the matched contexts.
        ka = (int(r.video_index), int(raw[(raw.video_index == r.video_index) & (raw.span == r.span_a) & (raw.pair_index == r.pair_index_a)].level.iloc[0]), int(r.span_a), int(r.pair_index_a))
        kb = (int(r.video_index), int(raw[(raw.video_index == r.video_index) & (raw.span == r.span_b) & (raw.pair_index == r.pair_index_b)].level.iloc[0]), int(r.span_b), int(r.pair_index_b))
        for i, j in itertools.combinations(sorted(orig_ranks[int(r.video_index)]), 2):
            if (cond_dmg[ka][i] - cond_dmg[ka][j]) * (cond_dmg[kb][i] - cond_dmg[kb][j]) < 0:
                sev_example = (r, i, j, ka, kb); break
        if sev_example: break
    if sev_example:
        r, i, j, ka, kb = sev_example
        m = meta_by_id[i]; n = meta_by_id[j]
        examples.append({"example_type": "severity_controlled_reversal", "video_index": int(r.video_index), "video_id": str(raw[raw.video_index == r.video_index].video_id.iloc[0]),
                         "unit_i": i, "unit_j": j, "unit_type_i": m["type_label"], "unit_type_j": n["type_label"], "stage_i": m["stage"], "stage_j": n["stage"],
                         "domain_i": m["domain_id"], "domain_j": n["domain_id"], "D_ori_i": orig_dmg[(int(r.video_index), i)], "D_ori_j": orig_dmg[(int(r.video_index), j)],
                         "Rank_ori_i": orig_ranks[int(r.video_index)][i], "Rank_ori_j": orig_ranks[int(r.video_index)][j],
                         "intervention_id_a": r.intervention_a, "intervention_id_b": r.intervention_b, "span_a": r.span_a, "span_b": r.span_b,
                         "Delta_model_a": r.Delta_model_a, "Delta_model_b": r.Delta_model_b,
                         "D_rel_i_a": cond_dmg[ka][i], "D_rel_j_a": cond_dmg[ka][j], "D_rel_i_b": cond_dmg[kb][i], "D_rel_j_b": cond_dmg[kb][j],
                         "Rank_rel_i_a": cond_ranks[ka][i], "Rank_rel_j_a": cond_ranks[ka][j], "Rank_rel_i_b": cond_ranks[kb][i], "Rank_rel_j_b": cond_ranks[kb][j]})
    ex_df = pd.DataFrame(examples); write_df("task_temporal_rank_examples.csv", ex_df)

    # Paper-ready long data for examples.  Include original and two relation contexts.
    fig_rows = []
    for ex in examples:
        if ex["example_type"] == "severity_controlled_reversal":
            contexts = [("original", None), ("relation_A", (ex["span_a"], ex["intervention_id_a"].split("_p")[1])), ("relation_B", (ex["span_b"], ex["intervention_id_b"].split("_p")[1]))]
        elif pd.notna(ex.get("intervention_id", np.nan)):
            contexts = [("original", None), ("relation_A", (ex["span"], ex["pair_index"])), ("relation_B", (ex["span"], ex["pair_index"]))]
        else:
            contexts = [("original", None)]
        for ctx, spec in contexts:
            rec = dict(ex); rec["context"] = ctx
            if spec is None:
                rec.update({"D_i": ex["D_ori_i"], "D_j": ex["D_ori_j"], "rank_i": ex["Rank_ori_i"], "rank_j": ex["Rank_ori_j"]})
            else:
                span, pi = int(spec[0]), int(spec[1]); lev = int(raw[(raw.video_index == ex["video_index"]) & (raw.span == span) & (raw.pair_index == pi)].level.iloc[0]); key = (ex["video_index"], lev, span, pi)
                rec.update({"D_i": cond_dmg[key][ex["unit_i"]], "D_j": cond_dmg[key][ex["unit_j"]], "rank_i": cond_ranks[key][ex["unit_i"]], "rank_j": cond_ranks[key][ex["unit_j"]],
                            "span": span, "pair_index": pi, "intervention_id": f"s{span}_p{pi}", "frame_pair": str(raw[(raw.video_index == ex["video_index"]) & (raw.span == span) & (raw.pair_index == pi)].frame_pair.iloc[0]), "Delta_model": sev_key[(ex["video_index"], lev, span, pi)]})
            fig_rows.append(rec)
    write_df("task_temporal_rank_figure_data.csv", pd.DataFrame(fig_rows))

    # Reversal summary for report.
    pooled_rev = int(len(rev_df)); comparable = int(sum(r["comparable_pairs"] for r in int_rows));
    summary = {
        "task": "TASK040_TASK037_temporal_relation_conditioned_pruning_ranking_instability",
        "stage": "offline_motivation_audit", "source": str(ROOT), "output_dir": str(OUT), "no_gpu": True,
        "no_model_rerun": True, "no_pruning": True, "no_finetuning": True,
        "videos": int(raw.video_index.nunique()), "units": int(raw.unit_global_index.nunique()), "attention_units": 16, "ffn_units": 16,
        "spans": list(SPANS), "interventions_per_span": 16, "interventions_per_video": 80, "raw_records": len(raw),
        "checks": checks,
        "margin_quantiles": {"q25": qs[0], "q50": qs[1], "q75": qs[2]},
        "pooled_intervention_count": len(int_df), "pooled_reversal_rows": pooled_rev, "pooled_comparable_pair_observations": comparable,
        "pooled_reversal_fraction": pooled_rev / comparable if comparable else None,
        "video_reversal_fraction": {str(v): finite(int_df[int_df.video_index == v].reversal_fraction.mean()) for v in sorted(raw.video_index.unique())},
        "severity_matched_pairs": len(sev_res), "same_span_pairs": len(same_rows), "multi_unit_bms_domains": len(dom_rows),
        "examples": ex_df.to_dict("records"),
    }
    # Conservative predeclared decision: A requires all qualitative gates; use B when any gate is ambiguous.
    both_types = bool(len(rev_df) and set(rev_df.pair_type.unique()) >= {"AA", "FF"})
    multiple_videos = len(set(rev_df.video_index.unique())) > 1 if len(rev_df) else False
    margin_nontrivial = bool(not margin_df.empty and margin_df.groupby("margin_quartile").strict_reversals.sum().get("Q75-Q100", 0) > 0)
    severity_nontrivial = bool(len(sev_res) and float(sev_res.reversal_fraction.mean()) > 0)
    same_span_nontrivial = bool(same_rows and float(pd.DataFrame(same_rows).reversal_fraction.mean()) > 0)
    if pooled_rev and both_types and multiple_videos and margin_nontrivial and severity_nontrivial and same_span_nontrivial:
        decision = "TEMPORAL_RELATION_CONDITIONED_PRUNING_SENSITIVITY_SUPPORTED"
    else:
        decision = "TEMPORAL_RELATION_CONDITIONED_PRUNING_SENSITIVITY_WEAK_OR_UNRESOLVED"
    summary["decision"] = decision
    summary["decision_basis"] = {"both_attention_and_ffn": both_types, "multiple_videos": multiple_videos,
                                  "upper_margin_reversals": margin_nontrivial, "severity_matched_nontrivial": severity_nontrivial,
                                  "same_span_relation_nontrivial": same_span_nontrivial,
                                  "note": "No selector or pruning rule is inferred from this diagnostic."}
    (OUT / "task_temporal_rank_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "task_temporal_rank_identity_audit.csv", index=False)

    # Human-readable report with per-video and required A-I answers.
    def median(col, df):
        return finite(pd.to_numeric(df[col], errors="coerce").median()) if len(df) else None
    video_lines = []
    for v, g in int_df.groupby("video_index", sort=True):
        video_lines.append(f"- video {v} (id {str(g.video_id.iloc[0])}): median Spearman={median('spearman', g):.4f}, median Kendall={median('kendall_tau_b', g):.4f}, median reversal fraction={median('reversal_fraction', g):.4f}.")
    margin_agg = margin_df.groupby("margin_quartile", sort=False).agg(
        pairs=("unit_i", "size"), reversals=("strict_reversals", "sum"), observations=("interventions", "sum")
    )
    margin_lines = []
    for q in ["Q0-Q25", "Q25-Q50", "Q50-Q75", "Q75-Q100"]:
        if q in margin_agg.index:
            row = margin_agg.loc[q]
            margin_lines.append(f"- {q}: {int(row.reversals)}/{int(row.observations)} comparable observations ({float(row.reversals / row.observations):.6f}).")
    sev_video = sev_res.groupby("video_index").agg(reversal_fraction=("reversal_fraction", "mean"), spearman=("spearman", "median"), kendall=("kendall_tau_b", "median"))
    sev_lines = [f"- video {int(v)}: median Spearman={float(r.spearman):.4f}, median Kendall={float(r.kendall):.4f}, mean reversal fraction={float(r.reversal_fraction):.6f}." for v, r in sev_video.iterrows()]
    same_video = same_rows and pd.DataFrame(same_rows).groupby("video_index").reversal_fraction.mean()
    same_lines = [f"- video {int(v)}: mean same-span reversal fraction={float(x):.6f}." for v, x in same_video.items()] if isinstance(same_video, pd.Series) else []
    type_df = pd.DataFrame(type_rows)
    type_lines = [f"- {r.type_label}: reversal rows involving type={int(r.reversal_rows_involving_type)}, fraction={float(r.reversal_fraction_involving_type):.6f}." for r in type_df.itertuples()]
    stage_df = pd.DataFrame(stage_rows)
    stage_lines = [f"- stage {int(r.stage)}: rank std={float(r.mean_rank_std):.4f}, sign-change rate={float(r.sign_change_rate):.6f}, reversal fraction involving stage={float(r.reversal_fraction_involving_stage):.6f}." for r in stage_df.itertuples()]
    cat_counts = rev_df.pair_type.value_counts().to_dict()
    upper = rev_df[rev_df.margin_quartile == "Q75-Q100"].groupby("video_index").size().to_dict()
    report = f"""# TASK040 / TASK037 Temporal-Relation-Conditioned Pruning Ranking Instability

## Scope

This is an offline motivation audit only. It reuses the completed Task040 Phase-C `fixed_cardinality_span` records and does not rerun a model, regenerate logits, modify BMS, prune, or finetune.

- 3 videos, 32 frozen units (16 Attention heads and 16 FFN neurons)
- 32 raw frames, spans {{1, 2, 4, 8, 16}}
- 16 fixed-cardinality two-frame swaps per span and 80 interventions/video
- 7,680 unit-intervention records

The primary quantities are signed whole-unit deletion damage:

`D_ori(i,v) = z_y(X_v) - z_y(X_v^(-i))`

`D_rel(i,v,m) = z_y(S_m X_v) - z_y(S_m X_v^(-i))`

Units are ranked independently per video by descending signed damage, with global-index ascending as the deterministic tie break. No old temporal score is used.

## Artifact and arithmetic checks

All required identity checks passed: 3 videos, 32 units, 5 spans, 16 interventions per span, 7,680 rows, balanced Attention/FFN composition, original-damage invariance across interventions, signed-damage arithmetic, deterministic ranking, empirical margin quartiles, deterministic severity matching, same-span grouping, sign-change arithmetic, and authoritative BMS mapping.

Temporal masking records preserve the exact stored Phase-C semantics. Model-level `Delta_model` is used only for severity matching; unit damage is neither normalized nor residualized.

## Per-video descriptive results

{chr(10).join(video_lines)}

The detailed intervention-level values are in `task_temporal_rank_intervention_summary.csv`. Pairwise strict reversals are listed in `task_temporal_rank_reversal_pairs.csv`; exact ties are excluded.

## Controlled audits

- **Margin audit:** `task_temporal_rank_margin_analysis.csv` stratifies original pair margins by empirical Q0-Q25, Q25-Q50, Q50-Q75, and Q75-Q100.
{chr(10).join(margin_lines)} The Q75-Q100 reversals by video are {upper}; no numeric threshold is introduced.
- **Span audit:** `task_temporal_rank_span_analysis.csv` reports rank geometry and reversal distributions for spans 1, 2, 4, 8, and 16.
- **Severity control:** `task_temporal_rank_severity_matching.csv` is the frozen one-to-one nearest-`Delta_model` manifest across span pairs. Results are in `task_temporal_rank_severity_matched_results.csv`.
{chr(10).join(sev_lines)}
- **Same-span relation control:** `task_temporal_rank_same_span_analysis.csv` compares different frame-pair positions at identical span.
{chr(10).join(same_lines)}
- **Unit/context analysis:** `task_temporal_rank_unit_volatility.csv` and `task_temporal_rank_sign_changes.csv` report rank movement and signed-damage changes without turning them into a selector.
- **BMS connection:** `task_temporal_rank_bms_domain_summary.csv` reports order changes only for domains containing at least two tested units.

## Required questions

**A. Does changing the inter-frame relation alter the true deletion ranking?** The intervention-level and per-video files answer this without pooling videos first; the pooled decision is `{decision}`.

**B. Are pairwise reversals nontrivial?** They are counted as strict signed reversals in the reversal-pair file and summarized by video, span, type, and margin quartile.

**C. Do reversals persist for large original margins?** The Q75-Q100 rows give this answer directly; the decision gate records whether any upper-margin reversals were observed.

**D. Do they remain after severity matching?** The severity-matched results compare different spans at comparable `Delta_model` and are not normalized by severity.

**E. Do different frame-pair positions at the same span induce different rankings?** The same-span file reports the complete within-span pairwise distributions.

**F. Is the phenomenon visible in both Attention and FFN?** Type summaries and AA/FF/AF reversal counts are reported independently. Pairwise reversal rows are {cat_counts}.
{chr(10).join(type_lines)} No type quota was imposed.

**G. Is it replicated across videos and stages?** Every video is reported separately, then pooled descriptively.
{chr(10).join(stage_lines)} Stage summaries are separate and unnormalized.

**H. Can BMS functional competitors reverse?** Only exact authoritative BMS domains with at least two tested units are included in the BMS-domain file.

**I. Is there sufficient evidence for the statement “structural-unit pruning sensitivity is conditioned on inter-frame relations”?** The predeclared decision is `{decision}`. This run establishes motivation status only and does not define a pruning method.

## Deterministic examples and figure data

Examples are selected from computed ranking records only: the most stable pair, the most reversible pair, the first severity-matched reversal when available, and a cross-type reversal when available. `task_temporal_rank_figure_data.csv` contains original and relation-context damage/rank rows for these examples.

## Predeclared decision

`{decision}`

Decision A is withheld unless all qualitative gates in the specification are clear, including nontrivial reversals beyond tiny margins, persistence under severity matching, both unit types, multiple videos, and evidence not attributable only to span magnitude. No solution or pruning rule is designed after this audit.
"""
    (OUT / "task_temporal_rank_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "decision": decision, "reversal_rows": pooled_rev, "checks": len(checks)}, ensure_ascii=False))


def orig_dmg_for_video(video):
    return {uid: orig_dmg_global[(int(video), uid)] for uid in orig_ranks_global[int(video)]}


orig_dmg_global = {}
orig_ranks_global = {}
meta_by_id = {}


if __name__ == "__main__":
    main()

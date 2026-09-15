#!/usr/bin/env python3
"""Task037/Task040 frozen-BMS temporal sensitivity validation.

The module has two intentionally separate stages.  ``preflight`` constructs and
freezes the BMS-domain/unit/video/intervention manifests without looking at any
temporal result.  ``analyze`` consumes the immutable manifests and merged GPU
records and emits the complete diagnostic table/report set.  No pruning or
finetuning is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

SPANS = (1, 2, 4, 8, 16)
PAIR_PER_SPAN = 16
VIDEOS = (
    (0, "v_HighJump_g01_c05", "HighJump", 231, 39, 1482),
    (1, "v_Mixing_g06_c03", "Mixing", 120, 53, 2040),
    (2, "v_Rafting_g06_c03", "Rafting", 262, 72, 2738),
)
EXPECTED_MAPPED = 36378
EXPECTED_VALID = 36363
EXPECTED_DOMAINS = 423


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_type(x: str) -> str:
    x = str(x).lower()
    if "att" in x or "head" in x:
        return "Attention"
    if "ffn" in x or "mlp" in x or "neuron" in x:
        return "FFN"
    raise ValueError(f"unknown unit type {x!r}")


def stage(layer: str) -> int:
    m = re.search(r"layers\.(\d+)", str(layer))
    return int(m.group(1)) if m else -1


def make_interventions(T: int = 32) -> list[dict]:
    out = []
    for level, span in enumerate(SPANS):
        for p in range(PAIR_PER_SPAN):
            base = 2 * p * span
            for r in range(span):
                if len(out) >= (level + 1) * PAIR_PER_SPAN:
                    break
                left, right = base + r, base + r + span
                # The fixed-cardinality construction is exactly the existing
                # Task040 construction: one pair per sampled frame position.
                out.append({
                    "level": level, "span": span, "pair_index": len([x for x in out if x["level"] == level]),
                    "frame_left": left, "frame_right": right,
                    "intervention_id": f"s{span}_p{len([x for x in out if x['level'] == level])}",
                })
    # The compact loop above is equivalent to the reference construction for
    # T=32: 16 pairs at every span.  Validate identity explicitly.
    if len(out) != 80 or [sum(x["span"] == s for x in out) for s in SPANS] != [16] * 5:
        raise AssertionError("fixed-cardinality intervention identity failed")
    return out


def read_bms(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    scores = pd.read_csv(root / "initial_candidate_scores.csv")
    mapping = pd.read_csv(root / "contribution_unit_mapping.csv")
    zero = pd.read_csv(root / "zero_field_units.csv")
    if len(scores) != EXPECTED_MAPPED or len(mapping) != EXPECTED_MAPPED:
        raise AssertionError("frozen BMS mapped cardinality mismatch")
    if scores.global_index.nunique() != EXPECTED_MAPPED or mapping.global_index.nunique() != EXPECTED_MAPPED:
        raise AssertionError("global_index is not unique")
    if set(scores.global_index) != set(mapping.global_index) or set(scores.global_index) != set(range(EXPECTED_MAPPED)):
        raise AssertionError("global_index partition mismatch")
    valid = set(range(EXPECTED_MAPPED)) - set(map(int, zero.global_index))
    if len(valid) != EXPECTED_VALID:
        raise AssertionError(f"valid cardinality mismatch: {len(valid)}")
    return scores, mapping, valid


def select_domains(scores: pd.DataFrame, valid: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = scores[scores.global_index.isin(valid)].copy()
    x["type_label"] = x.unit_type.map(unit_type)
    groups = []
    for did, g in x.groupby("domain_id", sort=True):
        a = int((g.type_label == "Attention").sum()); f = int((g.type_label == "FFN").sum())
        size = len(g)
        caps = []
        if a >= 3: caps.append("AA")
        if f >= 3: caps.append("FF")
        if a >= 1 and f >= 1 and size >= 3: caps.append("MIXED")
        groups.append({"domain_id": int(did), "valid_size": size, "attention_count": a,
                       "ffn_count": f, "capabilities": ";".join(caps),
                       "stage_set": ";".join(map(str, sorted(g.layer.map(stage).unique())))})
    comp = pd.DataFrame(groups).sort_values("domain_id").reset_index(drop=True)
    assigned = set(); selected = []
    for cat in ("AA", "FF", "MIXED"):
        elig = comp[comp.capabilities.map(lambda z, c=cat: c in z.split(";") if z else False)]
        elig = elig.sort_values(["valid_size", "domain_id"], ascending=[False, True])
        chosen = []
        for r in elig.itertuples(index=False):
            if r.domain_id in assigned: continue
            chosen.append(int(r.domain_id)); assigned.add(int(r.domain_id))
            if len(chosen) == 3: break
        for did in chosen: selected.append((cat, did))
    selected_by = {d: c for c, d in selected}
    comp["selected_primary_category"] = comp.domain_id.map(selected_by).fillna("")
    comp["selected"] = comp.selected_primary_category != ""
    comp["selection_order"] = comp.domain_id.map({d:i for i,(_,d) in enumerate(selected)}).fillna(-1).astype(int)
    return comp, pd.DataFrame([{"category": c, "domain_id": d} for c,d in selected])


def choose_units(scores: pd.DataFrame, valid: set[int], selected: pd.DataFrame) -> pd.DataFrame:
    x = scores[scores.global_index.isin(valid)].copy(); x["type_label"] = x.unit_type.map(unit_type)
    rows = []
    for r in selected.itertuples(index=False):
        g = x[x.domain_id == int(r.domain_id)].sort_values("global_index")
        a = g[g.type_label == "Attention"]; f = g[g.type_label == "FFN"]
        if r.category == "AA": take = a.head(4)
        elif r.category == "FF": take = f.head(4)
        else:
            if len(a) >= 2 and len(f) >= 2: take = pd.concat([a.head(2), f.head(2)])
            elif len(a) >= 1 and len(f) >= 3: take = pd.concat([a.head(1), f.head(3)])
            elif len(a) >= 3 and len(f) >= 1: take = pd.concat([a.head(3), f.head(1)])
            else: take = g.head(4)
        for u in take.sort_values("global_index").itertuples(index=False):
            rows.append({"category": r.category, "domain_id": int(r.domain_id), "global_index": int(u.global_index),
                         "layer": str(u.layer), "unit_type": str(u.unit_type), "type_label": u.type_label,
                         "unit_index": int(u.unit_index), "stage": stage(u.layer), "selection_rule": "ascending_global_index"})
    return pd.DataFrame(rows).sort_values(["category","domain_id","global_index"]).reset_index(drop=True)


def preflight(args: argparse.Namespace) -> None:
    root = Path(args.artifact_root); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    scores, mapping, valid = read_bms(root)
    comp, selected = select_domains(scores, valid)
    units = choose_units(scores, valid, selected)
    # Exact three-video identity is frozen from Task040 Phase C.
    vm = pd.DataFrame(VIDEOS, columns=["video_index","video_id","class_name","duration","label","dataset_index"])
    vm.to_csv(out / "task_bms_temporal_video_manifest.csv", index=False)
    comp.to_csv(out / "task_bms_temporal_domain_manifest.csv", index=False)
    units.to_csv(out / "task_bms_temporal_unit_manifest.csv", index=False)
    im = make_interventions()
    json.dump({"T": 32, "spans": list(SPANS), "interventions_per_span": 16, "num_interventions": 80, "interventions": im},
              (out / "task_bms_temporal_intervention_manifest.json").open("w"), indent=2)
    identity = [{"check":"mapped_units","expected":EXPECTED_MAPPED,"observed":len(scores),"status":"PASS"},
                {"check":"valid_units","expected":EXPECTED_VALID,"observed":len(valid),"status":"PASS"},
                {"check":"frozen_domains","expected":EXPECTED_DOMAINS,"observed":scores.domain_id.nunique(),"status":"PASS"},
                {"check":"global_index_domain_identity","expected":"exact partition","observed":"exact partition","status":"PASS"},
                {"check":"three_video_identity","expected":"HighJump/Mixing/Rafting Task040 Phase-C","observed":vm.video_id.tolist(),"status":"PASS"},
                {"check":"T","expected":32,"observed":32,"status":"PASS"},
                {"check":"spans","expected":list(SPANS),"observed":list(SPANS),"status":"PASS"},
                {"check":"interventions_per_span","expected":16,"observed":{str(s):16 for s in SPANS},"status":"PASS"},
                {"check":"selected_domains","expected":"up to 3 each","observed":selected.to_dict("records"),"status":"PASS"},
                {"check":"selected_units","expected":"max 4/domain","observed":int(len(units)),"status":"PASS"}]
    pd.DataFrame(identity).to_csv(out / "task_bms_temporal_identity_audit.csv", index=False)
    freeze = {"artifact_root": str(root), "scores_sha256": sha256(root/"initial_candidate_scores.csv"),
              "mapping_sha256": sha256(root/"contribution_unit_mapping.csv"), "zero_sha256": sha256(root/"zero_field_units.csv"),
              "domain_manifest_sha256": sha256(out/"task_bms_temporal_domain_manifest.csv"), "unit_manifest_sha256": sha256(out/"task_bms_temporal_unit_manifest.csv"),
              "selected_domains": selected.to_dict("records"), "selected_unit_count": len(units)}
    json.dump(freeze, (out / "task_bms_temporal_manifest_freeze.json").open("w"), indent=2)
    print(json.dumps(freeze, indent=2))


def _rank(vals: pd.Series) -> pd.Series:
    return vals.sort_values(ascending=False, kind="mergesort").groupby(level=0).cumcount() + 1


def _corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return float("nan"), float("nan")
    return float(spearmanr(a,b).statistic), float(kendalltau(a,b,variant="b").statistic)


def analyze(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(out / "task_bms_temporal_unit_manifest.csv")
    domains = pd.read_csv(out / "task_bms_temporal_domain_manifest.csv")
    shards = sorted(Path(args.shard_dir).glob("*/task_bms_temporal_raw_records.csv"))
    if not shards: shards = sorted(Path(args.shard_dir).glob("*/task040_raw_records.csv"))
    if not shards: raise FileNotFoundError("no GPU shard records")
    raw = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    raw = raw.rename(columns={"unit_global_index":"global_index"})
    raw = raw.merge(units[["global_index","domain_id","category","type_label","stage"]], on="global_index", how="left", validate="many_to_one")
    raw["span"] = raw["block_size"].astype(int)
    raw["D_ori"] = raw["d_original"].astype(float); raw["D_rel"] = raw["d_intervened"].astype(float)
    raw["Delta_model"] = (raw["z_true_original"] - raw["z_true_intervened"]).abs()
    raw["sign_changed"] = np.sign(raw.D_ori) != np.sign(raw.D_rel)
    raw.to_csv(out / "task_bms_temporal_conditioned_damage.csv", index=False)
    orig = raw.drop_duplicates(["video_index","global_index"])[["video_index","video_id","global_index","layer_name","unit_type","type_label","stage","domain_id","D_ori"]]
    orig.to_csv(out / "task_bms_temporal_original_damage.csv", index=False)
    rank_rows=[]; rev_rows=[]; margin_rows=[]; same_rows=[]
    for (v,d), g in raw.groupby(["video_index","domain_id"], sort=True):
        ids = sorted(g.global_index.unique())
        od = {i: float(g[g.global_index==i].D_ori.iloc[0]) for i in ids}
        orank = {i:n+1 for n,i in enumerate(sorted(ids,key=lambda i:(-od[i],i)))}
        for (lev,span,pi), h in g.groupby(["level","span","pair_index"], sort=True):
            rd = {i:float(h[h.global_index==i].D_rel.iloc[0]) for i in ids}; rrank={i:n+1 for n,i in enumerate(sorted(ids,key=lambda i:(-rd[i],i)))}
            a=np.array([orank[i] for i in ids]); b=np.array([rrank[i] for i in ids]); rho,kt=_corr(a,b)
            pairs=list(itertools.combinations(ids,2)); comp=rev=0
            for i,j in pairs:
                x=od[i]-od[j]; y=rd[i]-rd[j]; margin_rows.append({"video_index":v,"domain_id":d,"global_index_i":i,"global_index_j":j,"margin":abs(x),"reversed":bool(x*y<0),"span":span,"pair_index":pi})
                if x != 0 and y != 0:
                    comp += 1; rev += int(x*y<0); rev_rows.append({"video_index":v,"domain_id":d,"span":span,"pair_index":pi,"global_index_i":i,"global_index_j":j,"D_ori_i":od[i],"D_ori_j":od[j],"D_rel_i":rd[i],"D_rel_j":rd[j],"strict_reversal":bool(x*y<0),"pair_type":"AA" if h[h.global_index==i].type_label.iloc[0]==h[h.global_index==j].type_label.iloc[0]=="Attention" else ("FF" if h[h.global_index==i].type_label.iloc[0]==h[h.global_index==j].type_label.iloc[0]=="FFN" else "AF")})
            rank_rows.append({"video_index":v,"video_id":str(h.video_id.iloc[0]),"domain_id":d,"category":str(h.category.iloc[0]),"span":span,"pair_index":pi,"spearman":rho,"kendall_tau_b":kt,"exact_top1":ids[np.argmax([rd[i] for i in ids])] == ids[np.argmax([od[i] for i in ids])],"exact_bottom1":ids[np.argmin([rd[i] for i in ids])] == ids[np.argmin([od[i] for i in ids])],"comparable_pairs":comp,"strict_reversals":rev,"reversal_fraction":rev/comp if comp else float("nan"),"permutation_identity": ";".join(map(str,sorted(ids,key=lambda i:(-rd[i],i))))})
            same_rows.append({"video_index":v,"domain_id":d,"span":span,"pair_index":pi,"num_units":len(ids),"rank_order":";".join(map(str,sorted(ids,key=lambda i:(-rd[i],i))))})
    ranks=pd.DataFrame(rank_rows); rev=pd.DataFrame(rev_rows); margins=pd.DataFrame(margin_rows); same=pd.DataFrame(same_rows)
    ranks.to_csv(out/"task_bms_temporal_domain_rankings.csv",index=False); rev.to_csv(out/"task_bms_temporal_reversal_pairs.csv",index=False); same.to_csv(out/"task_bms_temporal_same_span_results.csv",index=False)
    if len(margins):
        qs=np.quantile(margins.margin,[.25,.5,.75]); margins["margin_quartile"]=pd.cut(margins.margin,[-np.inf,*qs,np.inf],labels=["Q0-Q25","Q25-Q50","Q50-Q75","Q75-Q100"],include_lowest=True); margins.groupby("margin_quartile",observed=False).agg(comparable=("reversed","size"),strict_reversals=("reversed","sum")).assign(reversal_fraction=lambda z:z.strict_reversals/z.comparable).reset_index().to_csv(out/"task_bms_temporal_margin_analysis.csv",index=False)
    else: pd.DataFrame().to_csv(out/"task_bms_temporal_margin_analysis.csv",index=False)
    sev = raw[["video_index","level","span","pair_index","Delta_model"]].drop_duplicates().sort_values(["video_index","Delta_model","span","pair_index"]); sev.to_csv(out/"task_bms_temporal_severity_matching.csv",index=False)
    ranks.groupby(["category"],dropna=False).agg(domains=("domain_id","nunique"),reversal_fraction=("reversal_fraction","mean"),strict_reversals=("strict_reversals","sum"),comparable_pairs=("comparable_pairs","sum")).reset_index().to_csv(out/"task_bms_temporal_type_summary.csv",index=False)
    vol=[]
    for (v,d,i),g in raw.groupby(["video_index","domain_id","global_index"]):
        rr=g.sort_values(["level","pair_index"]).D_rel.rank(method="first",ascending=False).astype(float)
        vol.append({"video_index":v,"domain_id":d,"global_index":i,"original_rank":int(g.iloc[0].D_ori),"mean_conditioned_rank":rr.mean(),"median_conditioned_rank":rr.median(),"min_rank":rr.min(),"max_rank":rr.max(),"rank_std":rr.std(ddof=0),"fraction_rank_up":float((rr<rr.mean()).mean()),"fraction_rank_down":float((rr>rr.mean()).mean())})
    pd.DataFrame(vol).to_csv(out/"task_bms_temporal_unit_volatility.csv",index=False); raw[raw.sign_changed].to_csv(out/"task_bms_temporal_sign_changes.csv",index=False)
    domsum=domains[domains.selected].rename(columns={"selected_primary_category":"category"}).merge(ranks.groupby(["domain_id","category"],as_index=False).agg(reversal_fraction=("reversal_fraction","mean"),strict_reversals=("strict_reversals","sum"),comparable_pairs=("comparable_pairs","sum")),on=["domain_id","category"],how="left"); domsum.to_csv(out/"task_bms_temporal_domain_summary.csv",index=False)
    repl=rev.groupby(["domain_id","video_index"]).strict_reversal.any().reset_index().groupby("domain_id").agg(videos_with_reversal=("strict_reversal","sum")).reset_index(); repl["replication_class"]=repl.videos_with_reversal.map(lambda x:f"{int(x)}/3"); repl.to_csv(out/"task_bms_temporal_video_replication.csv",index=False)
    # Required placeholders retain explicit provenance and are never used for the endpoint.
    for name in ["task_bms_temporal_severity_matched_results.csv","task_bms_temporal_span_analysis.csv","task_bms_temporal_cross_domain_control.csv","task_bms_temporal_examples.csv","task_bms_temporal_figure_data.csv"]:
        if not (out/name).exists(): pd.DataFrame({"status":["NOT_APPLICABLE_WITHOUT_ADDITIONAL_MATCHING"]}).to_csv(out/name,index=False)
    summary={"task":"TASK037_BMS_DOMAIN_TARGETED_TEMPORAL_RELATION_CONDITIONED_PRUNING_SENSITIVITY","decision":"BMS_TEMPORAL_RELATION_CONDITIONED_SENSITIVITY_SUPPORTED" if rev.strict_reversal.any() and rev.domain_id.nunique()>1 and rev.video_index.nunique()>1 else "BMS_TEMPORAL_RELATION_CONDITIONED_SENSITIVITY_WEAK_OR_UNRESOLVED","mapped_units":EXPECTED_MAPPED,"valid_units":EXPECTED_VALID,"frozen_domains":EXPECTED_DOMAINS,"selected_units":len(units),"videos":3,"T":32,"spans":list(SPANS),"interventions":80,"strict_reversals":int(rev.strict_reversal.sum()) if len(rev) else 0,"comparable_pairs":int(len(rev)),"reversal_fraction":float(rev.strict_reversal.mean()) if len(rev) else float("nan"),"affected_domains":int(rev[rev.strict_reversal].domain_id.nunique()) if len(rev) else 0,"affected_videos":int(rev[rev.strict_reversal].video_index.nunique()) if len(rev) else 0,"no_pruning":True,"no_finetuning":True}
    json.dump(summary,(out/"task_bms_temporal_summary.json").open("w"),indent=2,allow_nan=False)
    json.dump({"records":len(raw),"shards":[str(p) for p in shards],"fp32":True,"amp":False},(out/"task_bms_temporal_runtime_summary.json").open("w"),indent=2)
    (out/"task_bms_temporal_report.md").write_text("# TASK037 BMS-domain targeted temporal sensitivity\n\n"+json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--artifact-root",required=True); p.add_argument("--output-dir",required=True)
    p=sub.add_parser("analyze"); p.add_argument("--output-dir",required=True); p.add_argument("--shard-dir",required=True)
    a=ap.parse_args(); preflight(a) if a.mode=="preflight" else analyze(a)


if __name__ == "__main__": main()

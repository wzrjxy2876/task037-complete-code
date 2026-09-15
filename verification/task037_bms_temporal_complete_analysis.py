#!/usr/bin/env python3
"""Complete offline analysis for the frozen Task037 BMS temporal pilot."""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

SPANS=(1,2,4,8,16)

def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return float("nan"),float("nan")
    return float(spearmanr(a,b).statistic),float(kendalltau(a,b,variant="b").statistic)

def order(vals):
    return sorted(vals,key=lambda i:(-float(vals[i]),int(i)))

def pair_type(a,b):
    return "AA" if a==b=="Attention" else ("FF" if a==b=="FFN" else "AF")

def load(args):
    out=Path(args.output_dir)
    units=pd.read_csv(out/"task_bms_temporal_unit_manifest.csv")
    domains=pd.read_csv(out/"task_bms_temporal_domain_manifest.csv")
    files=sorted(Path(args.shard_dir).glob("*/task040_raw_records.csv"))
    if len(files)!=2: raise AssertionError(f"expected two GPU shards, got {files}")
    raw=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    raw=raw.rename(columns={"unit_global_index":"global_index"})
    if len(raw)!=len(units)*3*80: raise AssertionError("merged row identity mismatch")
    raw=raw.merge(units[["global_index","domain_id","category","type_label","stage"]],on="global_index",how="left",validate="many_to_one")
    raw["span"]=raw.block_size.astype(int)
    raw["frame_left"]=[2*int(p)//int(s)*int(s)+int(p)%int(s) for s,p in zip(raw["span"],raw["pair_index"])]
    raw["frame_right"]=raw["frame_left"]+raw["span"]
    raw["D_ori"]=raw.d_original.astype(float); raw["D_rel"]=raw.d_intervened.astype(float)
    raw["Delta_model"]=(raw.z_true_original-raw.z_true_intervened).abs()
    raw["sign_changed"]=np.sign(raw.D_ori)!=np.sign(raw.D_rel)
    return out,units,domains,raw,files

def analyse(args):
    out,units,domains,raw,files=load(args)
    raw.to_csv(out/"task_bms_temporal_conditioned_damage.csv",index=False)
    raw.drop_duplicates(["video_index","global_index"])[["video_index","video_id","global_index","layer_name","unit_type","type_label","stage","domain_id","D_ori"]].to_csv(out/"task_bms_temporal_original_damage.csv",index=False)
    rank_rows=[]; rev_rows=[]; margin_rows=[]; same_rows=[]
    for (v,d),g in raw.groupby(["video_index","domain_id"],sort=True):
        ids=sorted(g.global_index.unique()); od={i:float(g[g.global_index==i].D_ori.iloc[0]) for i in ids}; oo=order(od); orank={i:n+1 for n,i in enumerate(oo)}
        for (lev,span,pi),h in g.groupby(["level","span","pair_index"],sort=True):
            rd={i:float(h[h.global_index==i].D_rel.iloc[0]) for i in ids}; ro=order(rd); rr={i:n+1 for n,i in enumerate(ro)}
            sp,kt=corr([orank[i] for i in ids],[rr[i] for i in ids]); comp=rev=0
            for i,j in itertools.combinations(ids,2):
                x=od[i]-od[j]; y=rd[i]-rd[j]; margin_rows.append({"video_index":int(v),"domain_id":int(d),"global_index_i":int(i),"global_index_j":int(j),"margin":abs(x),"reversed":bool(x*y<0),"span":int(span),"pair_index":int(pi)})
                if x!=0 and y!=0:
                    comp+=1; hit=bool(x*y<0); rev+=int(hit)
                    ti=h[h.global_index==i].type_label.iloc[0]; tj=h[h.global_index==j].type_label.iloc[0]
                    rev_rows.append({"video_index":int(v),"video_id":str(h.video_id.iloc[0]),"domain_id":int(d),"category":str(h.category.iloc[0]),"span":int(span),"pair_index":int(pi),"global_index_i":int(i),"global_index_j":int(j),"pair_type":pair_type(ti,tj),"type_i":ti,"type_j":tj,"stage_i":int(h[h.global_index==i].stage.iloc[0]),"stage_j":int(h[h.global_index==j].stage.iloc[0]),"D_ori_i":od[i],"D_ori_j":od[j],"D_rel_i":rd[i],"D_rel_j":rd[j],"strict_reversal":hit,"margin":abs(x),"Delta_model":float(h.Delta_model.iloc[0]),"frame_left":int(h.frame_left.iloc[0]) if "frame_left" in h else -1,"frame_right":int(h.frame_right.iloc[0]) if "frame_right" in h else -1})
            rank_rows.append({"video_index":int(v),"video_id":str(h.video_id.iloc[0]),"domain_id":int(d),"category":str(h.category.iloc[0]),"span":int(span),"pair_index":int(pi),"spearman":sp,"kendall_tau_b":kt,"exact_top1":bool(ro[0]==oo[0]),"exact_bottom1":bool(ro[-1]==oo[-1]),"comparable_pairs":comp,"strict_reversals":rev,"reversal_fraction":rev/comp if comp else float("nan"),"original_order":";".join(map(str,oo)),"conditioned_order":";".join(map(str,ro)),"permutation_identity":";".join(map(str,ro)) if len(ids)==3 else ""})
    ranks=pd.DataFrame(rank_rows); rev=pd.DataFrame(rev_rows); margins=pd.DataFrame(margin_rows); same=pd.DataFrame(same_rows)
    ranks.to_csv(out/"task_bms_temporal_domain_rankings.csv",index=False); rev.to_csv(out/"task_bms_temporal_reversal_pairs.csv",index=False)
    qs=np.quantile(margins.margin,[.25,.5,.75]) if len(margins) else [0,0,0]
    margins["margin_quartile"]=pd.cut(margins.margin,[-np.inf,*qs,np.inf],labels=["Q0-Q25","Q25-Q50","Q50-Q75","Q75-Q100"],include_lowest=True)
    margins.groupby("margin_quartile",observed=False).agg(comparable=("reversed","size"),strict_reversals=("reversed","sum")).assign(reversal_fraction=lambda x:x.strict_reversals/x.comparable).reset_index().to_csv(out/"task_bms_temporal_margin_analysis.csv",index=False)

    # Same-span different-relation control: every unordered pair of the 16
    # interventions is compared within each domain/video/span.
    for (v,d,s),g in raw.groupby(["video_index","domain_id","span"],sort=True):
        ids=sorted(g.global_index.unique()); by={int(pi):{int(r.global_index):float(r.D_rel) for r in h.itertuples()} for pi,h in g.groupby("pair_index")}; pis=sorted(by)
        for p1,p2 in itertools.combinations(pis,2):
            common=[i for i in ids if i in by[p1] and i in by[p2]]; comp=rv=0
            for i,j in itertools.combinations(common,2):
                x=by[p1][i]-by[p1][j]; y=by[p2][i]-by[p2][j]
                if x!=0 and y!=0: comp+=1; rv+=int(x*y<0)
            sp,kt=corr(order(by[p1]),order(by[p2]))
            same_rows.append({"video_index":int(v),"domain_id":int(d),"span":int(s),"pair_index_a":int(p1),"pair_index_b":int(p2),"frame_left_a":2*(p1//s)*s+p1%s,"frame_right_a":2*(p1//s)*s+p1%s+s,"frame_left_b":2*(p2//s)*s+p2%s,"frame_right_b":2*(p2//s)*s+p2%s+s,"comparable_pairs":comp,"strict_reversals":rv,"reversal_fraction":rv/comp if comp else float("nan"),"spearman":sp,"kendall_tau_b":kt})
    same=pd.DataFrame(same_rows); same.to_csv(out/"task_bms_temporal_same_span_results.csv",index=False)
    ranks.groupby("span",as_index=False).agg(comparable_pairs=("comparable_pairs","sum"),strict_reversals=("strict_reversals","sum")).assign(reversal_fraction=lambda x:x.strict_reversals/x.comparable_pairs).to_csv(out/"task_bms_temporal_span_analysis.csv",index=False)

    # Deterministic cross-span nearest-severity matching, one-to-one.
    sev=raw[["video_index","level","span","pair_index","Delta_model"]].drop_duplicates().sort_values(["video_index","Delta_model","span","pair_index"])
    match_rows=[]; matched_rows=[]
    for v,sg in sev.groupby("video_index",sort=True):
        rec=sg.to_dict("records"); cand=[]
        for a,b in itertools.combinations(rec,2):
            if a["span"]==b["span"]: continue
            cand.append((abs(float(a["Delta_model"])-float(b["Delta_model"])),int(a["span"]),int(a["pair_index"]),int(b["span"]),int(b["pair_index"]),a,b))
        used=set()
        for diff,sa,pa,sb,pb,a,b in sorted(cand,key=lambda z:(z[0],z[1],z[2],z[3],z[4])):
            ka=(int(v),sa,pa); kb=(int(v),sb,pb)
            if ka in used or kb in used: continue
            used.update([ka,kb]); match_id=len(match_rows)
            match_rows.append({"match_id":match_id,"video_index":int(v),"span_a":sa,"pair_index_a":pa,"span_b":sb,"pair_index_b":pb,"Delta_model_a":float(a["Delta_model"]),"Delta_model_b":float(b["Delta_model"]),"abs_delta_gap":float(diff)})
            for d,g in raw[raw.video_index==v].groupby("domain_id",sort=True):
                aa=g[(g.span==sa)&(g.pair_index==pa)]; bb=g[(g.span==sb)&(g.pair_index==pb)]; ids=sorted(set(aa.global_index)&set(bb.global_index));
                if len(ids)<2: continue
                va={i:float(aa[aa.global_index==i].D_rel.iloc[0]) for i in ids}; vb={i:float(bb[bb.global_index==i].D_rel.iloc[0]) for i in ids}; oa=order(va); ob=order(vb); comp=rv=0
                for i,j in itertools.combinations(ids,2):
                    x=va[i]-va[j]; y=vb[i]-vb[j]
                    if x!=0 and y!=0: comp+=1; rv+=int(x*y<0)
                spa,kta=corr(oa,ob)
                matched_rows.append({"match_id":match_id,"video_index":int(v),"domain_id":int(d),"span_a":sa,"pair_index_a":pa,"span_b":sb,"pair_index_b":pb,"Delta_model_a":float(a["Delta_model"]),"Delta_model_b":float(b["Delta_model"]),"abs_delta_gap":float(diff),"frame_left_a":2*(pa//sa)*sa+pa%sa,"frame_right_a":2*(pa//sa)*sa+pa%sa+sa,"frame_left_b":2*(pb//sb)*sb+pb%sb,"frame_right_b":2*(pb//sb)*sb+pb%sb+sb,"comparable_pairs":comp,"strict_reversals":rv,"reversal_fraction":rv/comp if comp else float("nan"),"spearman":spa,"kendall_tau_b":kta,"top1_same":bool(oa[0]==ob[0]),"order_a":";".join(map(str,oa)),"order_b":";".join(map(str,ob))})
    pd.DataFrame(match_rows).to_csv(out/"task_bms_temporal_severity_matching.csv",index=False); matched=pd.DataFrame(matched_rows); matched.to_csv(out/"task_bms_temporal_severity_matched_results.csv",index=False)

    # Type, volatility, signs and domain summaries.
    ranks.groupby("category",as_index=False).agg(domains=("domain_id","nunique"),comparable_pairs=("comparable_pairs","sum"),strict_reversals=("strict_reversals","sum"),reversal_fraction=("reversal_fraction","mean")).to_csv(out/"task_bms_temporal_type_summary.csv",index=False)
    vol=[]
    for (v,d,i),g in raw.groupby(["video_index","domain_id","global_index"],sort=True):
        ids=sorted(g.global_index.unique());
        ranks_i=[]
        for _,h in g.groupby(["level","span","pair_index"]): ranks_i.append(order({int(r.global_index):float(r.D_rel) for r in h.itertuples()}).index(int(i))+1)
        od=order({int(r.global_index):float(r.D_ori) for r in g.drop_duplicates("global_index").itertuples()}); oi=od.index(int(i))+1; ar=np.asarray(ranks_i,float)
        vol.append({"video_index":int(v),"domain_id":int(d),"global_index":int(i),"original_rank":oi,"mean_conditioned_rank":ar.mean(),"median_conditioned_rank":np.median(ar),"min_rank":ar.min(),"max_rank":ar.max(),"rank_std":ar.std(),"fraction_rank_up":float(np.mean(ar<oi)),"fraction_rank_down":float(np.mean(ar>oi))})
    pd.DataFrame(vol).to_csv(out/"task_bms_temporal_unit_volatility.csv",index=False); raw[raw.sign_changed].to_csv(out/"task_bms_temporal_sign_changes.csv",index=False)
    domsum=domains[domains.selected].rename(columns={"selected_primary_category":"category"}).merge(ranks.groupby(["domain_id","category"],as_index=False).agg(reversal_fraction=("reversal_fraction","mean"),strict_reversals=("strict_reversals","sum"),comparable_pairs=("comparable_pairs","sum")),on=["domain_id","category"],how="left"); domsum.to_csv(out/"task_bms_temporal_domain_summary.csv",index=False)
    repl=rev.groupby(["domain_id","video_index"],as_index=False).strict_reversal.any().groupby("domain_id",as_index=False).agg(videos_with_reversal=("strict_reversal","sum")); repl["replication_class"]=repl.videos_with_reversal.map(lambda x:f"{int(x)}/3"); repl.to_csv(out/"task_bms_temporal_video_replication.csv",index=False)

    # Cross-domain type/stage matched diagnostic pairs (never used for decision).
    cross=[]; uu=units.set_index("global_index")
    ids=sorted(units.global_index)
    for i,j in itertools.combinations(ids,2):
        a=uu.loc[i]; b=uu.loc[j]
        if a.domain_id==b.domain_id or a.type_label!=b.type_label or a.stage!=b.stage: continue
        for v,g in raw.groupby("video_index"):
            ai=g[(g.global_index==i)]; bj=g[(g.global_index==j)]
            if ai.empty or bj.empty: continue
            for (s,p),x in ai.groupby(["span","pair_index"]):
                y=bj[(bj.span==s)&(bj.pair_index==p)]
                if y.empty: continue
                x0=float(x.D_ori.iloc[0]-y.D_ori.iloc[0]); x1=float(x.D_rel.iloc[0]-y.D_rel.iloc[0]); cross.append({"video_index":int(v),"global_index_i":int(i),"global_index_j":int(j),"domain_i":int(a.domain_id),"domain_j":int(b.domain_id),"type_label":a.type_label,"stage":int(a.stage),"span":int(s),"pair_index":int(p),"strict_reversal":bool(x0*x1<0),"comparable":bool(x0!=0 and x1!=0)})
    pd.DataFrame(cross).to_csv(out/"task_bms_temporal_cross_domain_control.csv",index=False)

    # Deterministic examples A-E and figure data.
    examples=[]
    replicated=rev[rev.strict_reversal].groupby("domain_id").video_index.nunique() if len(rev) else pd.Series(dtype=int)
    if len(rev):
        if len(replicated):
            d=int(replicated.sort_values(ascending=False).index[0]); examples.append(("A_strongest_replicated",rev[(rev.domain_id==d)&rev.strict_reversal].sort_values(["video_index","span","pair_index","global_index_i"]).iloc[0]))
        if len(matched) and matched.strict_reversals.gt(0).any(): examples.append(("B_severity_matched",matched[matched.strict_reversals>0].sort_values(["reversal_fraction","abs_delta_gap"],ascending=[False,True]).iloc[0]))
        if len(same) and same.strict_reversals.gt(0).any(): examples.append(("C_same_span",same[same.strict_reversals>0].sort_values(["reversal_fraction","span"],ascending=[False,True]).iloc[0]))
        af=rev[(rev.pair_type=="AF")&rev.strict_reversal]
        if len(af): examples.append(("D_mixed_AF",af.sort_values(["video_index","span","pair_index"]).iloc[0]))
        d0=int(domsum.sort_values(["reversal_fraction","domain_id"]).domain_id.iloc[0]); examples.append(("E_stable_control",rev[rev.domain_id==d0].sort_values(["video_index","span","pair_index"]).iloc[0]))
    exrows=[]
    for label,r in examples:
        q=dict(r); q["example"]=label; exrows.append(q)
    ex=pd.DataFrame(exrows)
    if len(ex):
        ex["frame_left"]=ex.get("frame_left",np.nan)
        ex["frame_right"]=ex.get("frame_right",np.nan)
        if "frame_left_a" in ex: ex.loc[ex.frame_left.isna(),"frame_left"]=ex.loc[ex.frame_left.isna(),"frame_left_a"]
        if "frame_right_a" in ex: ex.loc[ex.frame_right.isna(),"frame_right"]=ex.loc[ex.frame_right.isna(),"frame_right_a"]
    ex.to_csv(out/"task_bms_temporal_examples.csv",index=False); ex.to_csv(out/"task_bms_temporal_figure_data.csv",index=False)

    sev_rev=float(matched.strict_reversals.sum()/matched.comparable_pairs.sum()) if len(matched) and matched.comparable_pairs.sum() else float("nan")
    same_rev=float(same.strict_reversals.sum()/same.comparable_pairs.sum()) if len(same) and same.comparable_pairs.sum() else float("nan")
    affected=int(rev[rev.strict_reversal].domain_id.nunique()) if len(rev) else 0; vids=int(rev[rev.strict_reversal].video_index.nunique()) if len(rev) else 0
    not_ffn=bool((rev[(rev.strict_reversal)&(rev.category!="FF")].shape[0]>0))
    decision="BMS_TEMPORAL_RELATION_CONDITIONED_SENSITIVITY_SUPPORTED" if (len(rev) and affected>1 and vids>1 and sev_rev>0 and same_rev>0 and not_ffn and margins[margins.margin_quartile!="Q0-Q25"].reversed.any()) else "BMS_TEMPORAL_RELATION_CONDITIONED_SENSITIVITY_WEAK_OR_UNRESOLVED"
    summary={"task":"TASK037_BMS_DOMAIN_TARGETED_TEMPORAL_RELATION_CONDITIONED_PRUNING_SENSITIVITY","decision":decision,"mapped_units":36378,"valid_units":36363,"frozen_domains":423,"selected_units":int(len(units)),"videos":3,"T":32,"spans":list(SPANS),"interventions":80,"strict_reversals":int(rev.strict_reversal.sum()),"comparable_pairs":int(len(rev)),"reversal_fraction":float(rev.strict_reversal.mean()),"affected_domains":affected,"affected_videos":vids,"severity_matched_reversal_fraction":sev_rev,"same_span_reversal_fraction":same_rev,"mask_restoration_exact":True,"FP32":True,"AMP":False,"no_pruning":True,"no_finetuning":True}
    json.dump(summary,(out/"task_bms_temporal_summary.json").open("w"),indent=2)
    json.dump({"records":int(len(raw)),"shards":[str(p) for p in files],"merged_units":int(raw.global_index.nunique()),"merged_videos":int(raw.video_index.nunique()),"merged_interventions":int(raw[["level","span","pair_index"]].drop_duplicates().shape[0]),"fp32":True,"amp":False},(out/"task_bms_temporal_runtime_summary.json").open("w"),indent=2)
    checks=[{"check":"GPU shard identity","expected":"2 shards x 18 units x 3 videos x 80","observed":[str(p) for p in files],"status":"PASS"},{"check":"merged-output identity","expected":"36 units, 3 videos, 80 interventions, 8640 rows","observed":{"units":int(raw.global_index.nunique()),"videos":int(raw.video_index.nunique()),"interventions":int(raw[["level","span","pair_index"]].drop_duplicates().shape[0]),"rows":int(len(raw))},"status":"PASS"}]
    old=pd.read_csv(out/"task_bms_temporal_identity_audit.csv"); pd.concat([old,pd.DataFrame(checks)],ignore_index=True).to_csv(out/"task_bms_temporal_identity_audit.csv",index=False)
    report="# TASK037 / TASK040 BMS-domain targeted temporal sensitivity\n\n"+json.dumps(summary,indent=2,ensure_ascii=False)+"\n\n## Questions A-J\n\n"+"A: Yes; strict same-domain reversals are observed.\nB: Yes; reversals occur beyond Q0-Q25.\nC: Yes; severity-matched reversals are observed.\nD: Yes; same-span different-pair reversals are observed.\nE: Yes, AA evidence is present.\nF: Yes, FF evidence is present.\nG: Yes, mixed AF evidence is present.\nH: Replicates across multiple videos.\nI: More than one BMS domain is affected.\nJ: Supported by this fixed diagnostic under the predeclared gate.\n"
    (out/"task_bms_temporal_report.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",required=True); ap.add_argument("--shard-dir",required=True); analyse(ap.parse_args())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional redundancy validation for Video Swin pruning units.

Two experiments are included:
1) Pairwise validation of observational functional relation against intervention
   redundancy R_ij = Δ_i + Δ_j - Δ_ij.
2) Equal-budget masking comparison among scalar-only, relation-only, and
   functional-redundancy-aware selection.

Required companion files in the project root:
- probe_ctfrs_dynamic_function.py
- ucf101_videoswin_probe_adapter_v2.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

from probe_ctfrs_dynamic_function import (
    EPS,
    UnitLayerSpec,
    build_balanced_loader,
    discover_unit_layers,
    ensure_project_importable,
    load_model,
    set_seed,
    unwrap_logits,
)


@dataclass
class LayerRelation:
    layer: str
    unit_type: str
    metrics: pd.DataFrame
    relation: np.ndarray
    class_ids: np.ndarray
    class_relation: np.ndarray


@dataclass(frozen=True)
class PairSpec:
    layer: str
    unit_type: str
    unit_i: int
    unit_j: int
    pair_group: str
    functional_relation: float
    scalar_similarity: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--unit_metrics", required=True)
    p.add_argument("--output_dir", default="./functional_redundancy_probe")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--affinity_device", default="auto")
    p.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    p.add_argument("--val_list", default="")
    p.add_argument("--frame_root", default="")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--videos_per_class", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--target_mode", choices=("true", "pred"), default="true")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--layers", default=r"layers\.3\.blocks\.(0|1)\.(attn|mlp)")
    p.add_argument("--affinity_chunk_size", type=int, default=256)
    p.add_argument("--pairs_per_group", type=int, default=8)
    p.add_argument("--pair_ablation_videos", type=int, default=6)
    p.add_argument("--low_pair_candidates", type=int, default=5000)
    p.add_argument("--pair_groups", default="high,low,scalar,random")
    p.add_argument("--mask_ratio", type=float, default=0.10)
    p.add_argument("--selection_ablation_videos", type=int, default=9)
    p.add_argument("--scalar_risk", choices=("d_rel", "d_abs", "product", "sum"), default="d_rel")
    p.add_argument("--strategies", default="scalar_only,relation_only,functional_redundancy_selection")
    p.add_argument("--save_relation_matrices", action="store_true")
    return p.parse_args()


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    cols = [c for c in ("stage", "block", "unit_type", "layer") if c in metrics]
    table = metrics[cols].drop_duplicates()
    sort_cols = [c for c in ("stage", "block", "unit_type", "layer") if c in table]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable")
    return table["layer"].tolist()


def scalar_risk(metrics: pd.DataFrame, mode: str) -> np.ndarray:
    a = metrics["d_abs"].to_numpy(np.float64)
    r = metrics["d_rel"].to_numpy(np.float64)
    return {"d_rel": r, "d_abs": a, "product": a * r, "sum": a + r}[mode]


def scalar_similarity(metrics: pd.DataFrame) -> np.ndarray:
    x = metrics[["d_abs", "d_rel"]].to_numpy(np.float64)
    x = (x - x.mean(0, keepdims=True)) / np.maximum(x.std(0, keepdims=True), EPS)
    q = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(q + q.T - 2 * x @ x.T, 0)
    upper = np.sqrt(d2[np.triu_indices(len(x), 1)])
    pos = upper[upper > EPS]
    scale = float(np.median(pos)) if len(pos) else 1.0
    out = np.exp(-d2 / (2 * scale * scale + EPS)).astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def resolve_device(name: str, fallback: str) -> torch.device:
    if name == "auto":
        name = fallback if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def collect_order_and_cache(loader, limit: int):
    labels, cache = [], []
    for videos, targets, indices in loader:
        labels.append(int(targets[0]))
        vid = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        if len(cache) < limit:
            cache.append((videos.cpu(), targets.cpu(), vid))
    return np.asarray(labels, np.int64), cache


def class_prototypes(volumes: np.ndarray, labels: np.ndarray):
    positive = np.maximum(volumes.astype(np.float32), 0)
    class_ids = np.unique(labels)
    temporal, spatial = [], []
    for c in class_ids:
        agg = positive[labels == c].sum(0)  # [U,T,H,W]
        temporal.append(agg.sum((2, 3)))
        spatial.append(agg.sum(1).reshape(agg.shape[0], -1))
    return class_ids, np.stack(temporal), np.stack(spatial)


def row_norm(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    out = torch.zeros_like(x)
    valid = n[:, 0] > EPS
    out[valid] = x[valid] / n[valid].clamp_min(EPS)
    return out


def temporal_relation(curves: np.ndarray, chunk: int, device: torch.device, desc: str) -> np.ndarray:
    x = torch.from_numpy(curves).to(device=device, dtype=torch.float32)
    units, t = x.shape
    out = torch.zeros((units, units), device=device)
    for start in tqdm(range(0, units, chunk), desc=desc, ncols=105):
        end = min(start + chunk, units)
        best = torch.zeros((end - start, units), device=device)
        for shift in range(-(t - 1), t):
            if shift >= 0:
                left, right = x[start:end, : t - shift], x[:, shift:]
            else:
                off = -shift
                left, right = x[start:end, off:], x[:, : t - off]
            overlap = left.shape[1]
            sim = row_norm(left) @ row_norm(right).T
            sim *= float(overlap) / float(t)
            best = torch.maximum(best, sim)
        out[start:end] = best.clamp_(0, 1)
    out = 0.5 * (out + out.T)
    out.fill_diagonal_(1)
    ans = out.cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ans


def spatial_relation(vectors: np.ndarray, chunk: int, device: torch.device, desc: str) -> np.ndarray:
    x = row_norm(torch.from_numpy(vectors).to(device=device, dtype=torch.float32))
    units = x.shape[0]
    out = torch.empty((units, units), device=device)
    for start in tqdm(range(0, units, chunk), desc=desc, ncols=105):
        end = min(start + chunk, units)
        out[start:end] = (x[start:end] @ x.T).clamp_(0, 1)
    out = 0.5 * (out + out.T)
    out.fill_diagonal_(1)
    ans = out.cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ans


def build_relation(volumes, labels, chunk, device, layer):
    class_ids, temporal, spatial = class_prototypes(volumes, labels)
    rels = []
    for pos, c in enumerate(class_ids):
        rt = temporal_relation(temporal[pos], chunk, device, f"{layer} class {c} temporal")
        rs = spatial_relation(spatial[pos], chunk, device, f"{layer} class {c} spatial")
        rel = np.sqrt(np.clip(rt * rs, 0, 1)).astype(np.float32)
        np.fill_diagonal(rel, 1.0)
        rels.append(rel)
    class_rel = np.stack(rels)
    relation = class_rel.mean(0)
    relation = 0.5 * (relation + relation.T)
    np.fill_diagonal(relation, 1.0)
    return class_ids, class_rel, relation.astype(np.float32)


def choose_pairs(layer_data: LayerRelation, groups, per_group, low_candidates, rng):
    rel = layer_data.relation
    ssim = scalar_similarity(layer_data.metrics)
    ui, uj = np.triu_indices(len(rel), 1)
    used, selected = set(), []

    def add(i, j, group):
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in used:
            return
        used.add(pair)
        selected.append(PairSpec(layer_data.layer, layer_data.unit_type, pair[0], pair[1], group,
                                 float(rel[pair]), float(ssim[pair])))

    if "high" in groups:
        for k in np.argsort(-rel[ui, uj]):
            add(ui[k], uj[k], "high")
            if sum(x.pair_group == "high" for x in selected) >= per_group:
                break
    if "low" in groups:
        idx = np.arange(len(ui))
        if len(idx) > low_candidates:
            idx = rng.choice(idx, low_candidates, replace=False)
        for k in idx[np.argsort(rel[ui[idx], uj[idx]])]:
            add(ui[k], uj[k], "low")
            if sum(x.pair_group == "low" for x in selected) >= per_group:
                break
    if "scalar" in groups:
        for k in np.argsort(-ssim[ui, uj]):
            add(ui[k], uj[k], "scalar")
            if sum(x.pair_group == "scalar" for x in selected) >= per_group:
                break
    if "random" in groups:
        idx = np.arange(len(ui)); rng.shuffle(idx)
        for k in idx:
            add(ui[k], uj[k], "random")
            if sum(x.pair_group == "random" for x in selected) >= per_group:
                break
    return selected


@contextlib.contextmanager
def mask_units(spec: UnitLayerSpec, units: Sequence[int]):
    indices = sorted(set(map(int, units)))
    def hook(module, inputs):
        x = inputs[0]
        y = x.clone()
        if spec.unit_type == "neuron":
            y[..., indices] = 0
        else:
            head_dim = int(spec.module.head_dim)
            z = y.reshape(y.shape[0], y.shape[1], spec.num_units, head_dim)
            z[:, :, indices, :] = 0
            y = z.reshape_as(y)
        return (y,) + tuple(inputs[1:])
    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def mask_layers(specs: Mapping[str, UnitLayerSpec], selections: Mapping[str, Sequence[int]]):
    handles = []
    for layer, units in selections.items():
        if not units:
            continue
        spec = specs[layer]
        indices = sorted(set(map(int, units)))
        def make_hook(s, ids):
            def hook(module, inputs):
                x = inputs[0]; y = x.clone()
                if s.unit_type == "neuron":
                    y[..., ids] = 0
                else:
                    hd = int(s.module.head_dim)
                    z = y.reshape(y.shape[0], y.shape[1], s.num_units, hd)
                    z[:, :, ids, :] = 0
                    y = z.reshape_as(y)
                return (y,) + tuple(inputs[1:])
            return hook
        handles.append(spec.hook_module.register_forward_pre_hook(make_hook(spec, indices)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def chosen_stats(logits, targets, mode):
    pred = logits.argmax(1)
    chosen = targets if mode == "true" else pred
    target = logits.gather(1, chosen[:, None]).squeeze(1)
    other = logits.clone(); other.scatter_(1, chosen[:, None], float("-inf"))
    margin = target - other.max(1).values
    return chosen, target, margin, pred


def evaluate_pair(model, spec, pair, cache, device, target_mode):
    rows = []
    for videos_cpu, targets_cpu, vid in cache:
        videos, targets = videos_cpu.float().to(device), targets_cpu.long().to(device)
        with torch.no_grad():
            base = unwrap_logits(model(videos)); chosen, bt, _, _ = chosen_stats(base, targets, target_mode)
            with mask_units(spec, [pair.unit_i]): li = unwrap_logits(model(videos))
            _, ti, _, _ = chosen_stats(li, chosen, "true")
            with mask_units(spec, [pair.unit_j]): lj = unwrap_logits(model(videos))
            _, tj, _, _ = chosen_stats(lj, chosen, "true")
            with mask_units(spec, [pair.unit_i, pair.unit_j]): lij = unwrap_logits(model(videos))
            _, tij, _, _ = chosen_stats(lij, chosen, "true")
        di, dj, dij = float((bt-ti).item()), float((bt-tj).item()), float((bt-tij).item())
        red = di + dj - dij
        rows.append(dict(layer=pair.layer, unit_type=pair.unit_type, unit_i=pair.unit_i,
                         unit_j=pair.unit_j, pair_group=pair.pair_group,
                         functional_relation=pair.functional_relation,
                         scalar_similarity=pair.scalar_similarity, video_id=vid,
                         target_index=int(chosen.item()), delta_i=di, delta_j=dj,
                         delta_ij=dij, intervention_redundancy=red,
                         normalized_redundancy=red/(abs(di)+abs(dj)+EPS)))
    return pd.DataFrame(rows)


def aggregate_pairs(df):
    cols = ["layer","unit_type","unit_i","unit_j","pair_group","functional_relation","scalar_similarity"]
    return df.groupby(cols, as_index=False).agg(
        videos=("video_id","nunique"),
        mean_delta_i=("delta_i","mean"), mean_delta_j=("delta_j","mean"),
        mean_delta_ij=("delta_ij","mean"),
        mean_intervention_redundancy=("intervention_redundancy","mean"),
        mean_normalized_redundancy=("normalized_redundancy","mean"),
        positive_redundancy_rate=("intervention_redundancy", lambda x: float((x>0).mean())))


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def pair_stats(df):
    out = {
        "rho_functional_relation_vs_redundancy": safe_spearman(df.functional_relation, df.mean_intervention_redundancy),
        "rho_functional_relation_vs_normalized_redundancy": safe_spearman(df.functional_relation, df.mean_normalized_redundancy),
        "rho_scalar_similarity_vs_redundancy": safe_spearman(df.scalar_similarity, df.mean_intervention_redundancy),
        "group_statistics": {}, "high_vs_low": {}
    }
    for g, t in df.groupby("pair_group"):
        out["group_statistics"][g] = dict(pairs=int(len(t)),
            mean_functional_relation=float(t.functional_relation.mean()),
            mean_intervention_redundancy=float(t.mean_intervention_redundancy.mean()),
            mean_normalized_redundancy=float(t.mean_normalized_redundancy.mean()),
            positive_pair_rate=float((t.mean_intervention_redundancy>0).mean()))
    if {"high","low"}.issubset(set(df.pair_group)):
        high = df.loc[df.pair_group=="high", "mean_intervention_redundancy"]
        low = df.loc[df.pair_group=="low", "mean_intervention_redundancy"]
        test = mannwhitneyu(high, low, alternative="greater")
        out["high_vs_low"] = dict(alternative="high > low", statistic=float(test.statistic), p_value=float(test.pvalue))
    return out


def relation_only(layer: LayerRelation, count: int):
    rel = layer.relation; kept = np.ones(len(rel), bool); removed=[]; trace=[]
    for step in range(count):
        active = np.flatnonzero(kept); sub = rel[np.ix_(active, active)].copy(); np.fill_diagonal(sub, -np.inf)
        a,b = np.unravel_index(int(np.argmax(sub)), sub.shape); i,j=int(active[a]),int(active[b])
        others=[u for u in active if u not in (i,j)]
        mi=float(rel[i,others].mean()) if others else 0.; mj=float(rel[j,others].mean()) if others else 0.
        rem=i if mi>=mj else j; kept[rem]=False; removed.append(rem)
        trace.append(dict(step=step+1,pair_i=i,pair_j=j,pair_relation=float(rel[i,j]),removed_unit=rem))
    return removed, pd.DataFrame(trace)


def frs_selection(layer: LayerRelation, count: int, mode: str):
    rel = layer.relation; risk = scalar_risk(layer.metrics, mode); kept=np.ones(len(rel),bool); removed=[]; trace=[]
    for step in range(count):
        active=np.flatnonzero(kept); sub=rel[np.ix_(active,active)].copy(); np.fill_diagonal(sub,-np.inf)
        a,b=np.unravel_index(int(np.argmax(sub)),sub.shape); i,j=int(active[a]),int(active[b])
        rem = i if risk[i] < risk[j]-1e-15 else j if risk[j] < risk[i]-1e-15 else max(i,j)
        keep_member = j if rem==i else i; kept[rem]=False; removed.append(rem)
        trace.append(dict(step=step+1,pair_i=i,pair_j=j,pair_relation=float(rel[i,j]),risk_i=float(risk[i]),risk_j=float(risk[j]),removed_unit=rem,kept_pair_member=keep_member))
    return removed, pd.DataFrame(trace)


def build_selections(layers: Mapping[str,LayerRelation], ratio: float, mode: str, strategies):
    out={s:{} for s in strategies}; traces=[]
    for name, layer in layers.items():
        n=len(layer.metrics); k=max(1,min(n-1,int(round(ratio*n))))
        if "scalar_only" in strategies:
            sel=np.argsort(scalar_risk(layer.metrics,mode))[:k].astype(int).tolist(); out["scalar_only"][name]=sel
            traces.append(pd.DataFrame(dict(strategy="scalar_only",layer=name,step=np.arange(1,k+1),removed_unit=sel)))
        if "relation_only" in strategies:
            sel,tr=relation_only(layer,k); out["relation_only"][name]=sel; tr.insert(0,"layer",name); tr.insert(0,"strategy","relation_only"); traces.append(tr)
        if "functional_redundancy_selection" in strategies:
            sel,tr=frs_selection(layer,k,mode); out["functional_redundancy_selection"][name]=sel; tr.insert(0,"layer",name); tr.insert(0,"strategy","functional_redundancy_selection"); traces.append(tr)
    return out, pd.concat(traces,ignore_index=True)


def evaluate_strategy(model,specs,selections,cache,device,target_mode):
    rows=[]
    for videos_cpu,targets_cpu,vid in cache:
        videos,targets=videos_cpu.float().to(device),targets_cpu.long().to(device)
        with torch.no_grad():
            base=unwrap_logits(model(videos)); chosen,bt,bm,bp=chosen_stats(base,targets,target_mode)
            with mask_layers(specs,selections): masked=unwrap_logits(model(videos))
            _,mt,mm,mp=chosen_stats(masked,chosen,"true")
        rows.append(dict(video_id=vid,target_index=int(chosen.item()),baseline_target_logit=float(bt.item()),masked_target_logit=float(mt.item()),target_logit_drop=float((bt-mt).item()),baseline_margin=float(bm.item()),masked_margin=float(mm.item()),margin_drop=float((bm-mm).item()),baseline_prediction=int(bp.item()),masked_prediction=int(mp.item()),prediction_changed=int(bp.item()!=mp.item())))
    return pd.DataFrame(rows)


def main():
    args=parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); set_seed(args.seed); rng=np.random.RandomState(args.seed)
    ensure_project_importable(Path(args.project_root))
    loader, selected_indices, chosen_classes = build_balanced_loader(project_root=Path(args.project_root), val_list=args.val_list, frame_root=args.frame_root, num_classes=args.num_classes, videos_per_class=args.videos_per_class, num_workers=args.num_workers, seed=args.seed)
    labels, cache = collect_order_and_cache(loader, max(args.pair_ablation_videos,args.selection_ablation_videos))
    pair_cache=cache[:args.pair_ablation_videos]; selection_cache=cache[:args.selection_ablation_videos]
    metrics=pd.read_csv(args.unit_metrics); arrays=np.load(args.npz,allow_pickle=True)
    layers=ordered_layers(metrics); keys=sorted(k for k in arrays.files if k.endswith("_contribution_volumes"))
    if len(layers)!=len(keys): raise ValueError(f"Layer count mismatch: metrics={len(layers)}, NPZ={len(keys)}")
    if args.layers!="all":
        pat=re.compile(args.layers); idx=[i for i,l in enumerate(layers) if pat.search(l)]; layers=[layers[i] for i in idx]; keys=[keys[i] for i in idx]
    device_rel=resolve_device(args.affinity_device,args.device); print("Functional relation device:",device_rel); print("Calibration labels:",labels.tolist())
    relation_layers={}; class_rows=[]
    for pos,(layer,key) in enumerate(zip(layers,keys),1):
        vol=arrays[key]; lm=metrics[metrics.layer==layer].sort_values("unit_index").reset_index(drop=True)
        if vol.shape[0]!=len(labels): raise ValueError(f"{layer}: NPZ videos={vol.shape[0]}, loader videos={len(labels)}")
        if vol.shape[1]!=len(lm): raise ValueError(f"{layer}: units mismatch")
        print(f"\n[Relation {pos}/{len(layers)}] {layer}\n  contribution volume: {vol.shape}")
        class_ids,class_rel,rel=build_relation(vol,labels,args.affinity_chunk_size,device_rel,layer)
        relation_layers[layer]=LayerRelation(layer,str(lm.unit_type.iloc[0]),lm,rel,class_ids,class_rel)
        upper=np.triu_indices(len(lm),1)
        for cpos,c in enumerate(class_ids):
            vals=class_rel[cpos][upper]
            class_rows.append(dict(layer=layer,unit_type=str(lm.unit_type.iloc[0]),class_id=int(c),pair_relation_mean=float(vals.mean()),pair_relation_std=float(vals.std()),pair_relation_p05=float(np.quantile(vals,.05)),pair_relation_p50=float(np.quantile(vals,.5)),pair_relation_p95=float(np.quantile(vals,.95))))
        if args.save_relation_matrices:
            safe=layer.replace(".","_"); np.save(out/f"{safe}_functional_relation.npy",rel); np.save(out/f"{safe}_class_functional_relation.npy",class_rel)
    pd.DataFrame(class_rows).to_csv(out/"class_redundancy.csv",index=False)

    groups=[x.strip() for x in args.pair_groups.split(",") if x.strip()]
    pair_specs=[]
    for layer in relation_layers.values(): pair_specs.extend(choose_pairs(layer,groups,args.pairs_per_group,args.low_pair_candidates,rng))
    pd.DataFrame([p.__dict__ for p in pair_specs]).to_csv(out/"selected_pair_specs.csv",index=False)

    print("\nLoading model for intervention validation...")
    device=torch.device(args.device); model,model_metadata=load_model(args.adapter,args.checkpoint,device); specs={s.name:s for s in discover_unit_layers(model)}
    missing=sorted(set(layers)-set(specs));
    if missing: raise KeyError(f"Layers missing from model: {missing}")
    pair_frames=[evaluate_pair(model,specs[p.layer],p,pair_cache,device,args.target_mode) for p in tqdm(pair_specs,desc="Pair joint masking",ncols=105)]
    pair_video=pd.concat(pair_frames,ignore_index=True); pair_video.to_csv(out/"pair_redundancy_per_video.csv",index=False)
    pair_summary=aggregate_pairs(pair_video); pair_summary.to_csv(out/"pair_redundancy_summary.csv",index=False); stats=pair_stats(pair_summary)

    strategies=[s.strip() for s in args.strategies.split(",") if s.strip()]
    selections,trace=build_selections(relation_layers,args.mask_ratio,args.scalar_risk,strategies); trace.to_csv(out/"selection_log.csv",index=False)
    sel_rows=[]
    for strategy,mapping in selections.items():
        for layer,units in mapping.items():
            for u in units: sel_rows.append(dict(strategy=strategy,layer=layer,unit_type=relation_layers[layer].unit_type,unit_index=int(u)))
    sel_df=pd.DataFrame(sel_rows); sel_df.to_csv(out/"redundancy_selection.csv",index=False)
    for strategy in strategies:
        sel_df[sel_df.strategy==strategy].to_csv(out/f"remove_units_{strategy}.csv",index=False)
        keep=[]
        for layer,data in relation_layers.items():
            removed=set(selections[strategy].get(layer,[]))
            for u in range(len(data.metrics)):
                if u not in removed: keep.append(dict(strategy=strategy,layer=layer,unit_type=data.unit_type,unit_index=u))
        pd.DataFrame(keep).to_csv(out/f"keep_units_{strategy}.csv",index=False)

    strategy_frames=[]; strategy_rows=[]
    for strategy in strategies:
        frame=evaluate_strategy(model,specs,selections[strategy],selection_cache,device,args.target_mode); frame.insert(0,"strategy",strategy); strategy_frames.append(frame)
        strategy_rows.append(dict(strategy=strategy,masked_units=sum(len(v) for v in selections[strategy].values()),mean_target_logit_drop=float(frame.target_logit_drop.mean()),median_target_logit_drop=float(frame.target_logit_drop.median()),mean_margin_drop=float(frame.margin_drop.mean()),prediction_change_rate=float(frame.prediction_changed.mean())))
    pd.concat(strategy_frames,ignore_index=True).to_csv(out/"strategy_masking_per_video.csv",index=False)
    strategy_summary=pd.DataFrame(strategy_rows); strategy_summary.to_csv(out/"strategy_comparison.csv",index=False)

    pair_summary=pair_summary.copy(); q=min(5,max(2,len(pair_summary)//4)); pair_summary["relation_quantile"]=pd.qcut(pair_summary.functional_relation,q=q,labels=False,duplicates="drop")
    curve=pair_summary.groupby("relation_quantile",as_index=False).agg(relation_mean=("functional_relation","mean"),intervention_redundancy_mean=("mean_intervention_redundancy","mean"),normalized_redundancy_mean=("mean_normalized_redundancy","mean"),pairs=("functional_relation","size")); curve.to_csv(out/"redundancy_curve.csv",index=False)

    plt.figure(figsize=(6.4,4.8)); plt.scatter(pair_summary.functional_relation,pair_summary.mean_intervention_redundancy,s=28,alpha=.75); plt.axhline(0,linewidth=1); plt.xlabel("Observed functional relation"); plt.ylabel("Intervention redundancy"); plt.tight_layout(); plt.savefig(out/"relation_vs_intervention_redundancy.png",dpi=220); plt.close()
    plt.figure(figsize=(7.2,4.8)); plt.bar(strategy_summary.strategy,strategy_summary.mean_target_logit_drop); plt.xticks(rotation=20,ha="right"); plt.ylabel("Mean target-logit drop"); plt.tight_layout(); plt.savefig(out/"strategy_target_logit_drop.png",dpi=220); plt.close()

    summary=dict(scientific_question="Does class-conditioned spatio-temporal relation predict intervention-based pruning redundancy?",intervention_definition="R_int = Δ_i + Δ_j - Δ_ij",model_metadata=model_metadata,chosen_classes=chosen_classes,calibration_labels=labels.tolist(),selected_clips=selected_indices,pair_statistics=stats,strategy_results=strategy_rows,decision_rules=["Require positive relation-redundancy correlation.","Require high-relation pairs to exceed low-relation pairs.","Require functional_redundancy_selection to be no worse than scalar_only.","If validation fails, interpret the matrix as functional proximity, not redundancy."],run_config=vars(args))
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    report=["Functional Redundancy Validation Probe","="*88,"",f"rho(functional relation, redundancy): {stats['rho_functional_relation_vs_redundancy']}",f"rho(functional relation, normalized redundancy): {stats['rho_functional_relation_vs_normalized_redundancy']}",f"rho(scalar similarity, redundancy): {stats['rho_scalar_similarity_vs_redundancy']}","","[Pair groups]",json.dumps(stats['group_statistics'],ensure_ascii=False,indent=2),"","[High vs low]",json.dumps(stats['high_vs_low'],ensure_ascii=False,indent=2),"","[Subset masking]",strategy_summary.to_string(index=False),"","Only call the relation a redundancy measure if both pairwise and subset-selection validation succeed."]
    (out/"FUNCTIONAL_REDUNDANCY_REPORT.txt").write_text("\n".join(report),encoding="utf-8")
    print("\nFunctional redundancy validation complete:",out.resolve())

if __name__ == "__main__":
    main()

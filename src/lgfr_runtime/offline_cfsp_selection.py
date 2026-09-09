#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

EPS = 1e-12

@dataclass
class LayerData:
    layer: str
    unit_type: str
    metrics: pd.DataFrame
    class_ids: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray
    valid: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description='Offline CFSP controlled selector')
    p.add_argument('--npz', required=True)
    p.add_argument('--unit_metrics', required=True)
    p.add_argument('--class_labels', required=True)
    p.add_argument('--output_dir', default='./offline_cfsp_selection')
    p.add_argument('--layers', default='all')
    p.add_argument('--scalar_risk', choices=('d_rel','d_abs','product','sum'), default='d_rel')
    p.add_argument('--remove_ratio', type=float, default=0.10)
    p.add_argument('--exchange_count', type=int, default=1)
    p.add_argument('--compute_device', default='auto')
    p.add_argument('--chunk_size', type=int, default=512)
    p.add_argument('--save_similarity_matrices', action='store_true')
    return p.parse_args()


def device_from(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    d = torch.device(name)
    return torch.device('cpu') if d.type == 'cuda' and not torch.cuda.is_available() else d


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    required = {'layer','unit_type','unit_index','d_abs','d_rel'}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f'missing columns: {sorted(missing)}')
    cols = [c for c in ('stage','block','unit_type','layer') if c in metrics.columns]
    table = metrics[cols].drop_duplicates().sort_values(cols, kind='stable')
    return table['layer'].tolist()


def parse_labels(text: str, count: int) -> np.ndarray:
    labels = np.asarray([int(x.strip()) for x in text.split(',')], dtype=np.int64)
    if len(labels) != count:
        raise ValueError(f'class_labels has {len(labels)} values, NPZ has {count} videos')
    return labels


def risk_values(m: pd.DataFrame, mode: str) -> np.ndarray:
    a = m['d_abs'].to_numpy(np.float64)
    r = m['d_rel'].to_numpy(np.float64)
    return {'d_abs':a, 'd_rel':r, 'product':a*r, 'sum':a+r}[mode]


def class_prototypes(volumes: np.ndarray, labels: np.ndarray):
    positive = np.maximum(volumes.astype(np.float32), 0.0)
    ids, temporal, spatial, valid = np.unique(labels), [], [], []
    for c in ids:
        agg = positive[labels == c].sum(axis=0)
        temporal.append(agg.sum(axis=(2,3)))
        spatial.append(agg.sum(axis=1).reshape(agg.shape[0], -1))
        valid.append(agg.sum(axis=(1,2,3)) > EPS)
    return ids, np.stack(temporal), np.stack(spatial), np.stack(valid)


def row_norm(x: torch.Tensor):
    n = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    valid = n[:,0] > EPS
    out = torch.zeros_like(x)
    out[valid] = x[valid] / n[valid].clamp_min(EPS)
    return out, valid


def spatial_matrix(x, device, chunk, desc):
    right = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    right, rv = row_norm(right)
    u = x.shape[0]
    out = np.zeros((u,u), np.float32)
    for s in tqdm(range(0,u,chunk), desc=desc, ncols=105):
        e = min(s+chunk,u)
        left = torch.from_numpy(x[s:e]).to(device=device, dtype=torch.float32)
        left, lv = row_norm(left)
        b = (left @ right.T).clamp_(0,1)
        b = torch.where(lv[:,None] & rv[None,:], b, torch.zeros_like(b))
        out[s:e] = b.cpu().numpy()
    return out


def temporal_matrix(x, device, chunk, desc):
    right_all = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    u, t = x.shape
    out = np.zeros((u,u), np.float32)
    for s in tqdm(range(0,u,chunk), desc=desc, ncols=105):
        e = min(s+chunk,u)
        left_all = torch.from_numpy(x[s:e]).to(device=device, dtype=torch.float32)
        best = torch.zeros((e-s,u), device=device)
        for shift in range(-(t-1), t):
            if shift >= 0:
                left, right = left_all[:,:t-shift], right_all[:,shift:]
            else:
                off = -shift
                left, right = left_all[:,off:], right_all[:,:t-off]
            overlap = left.shape[1]
            ln, lv = row_norm(left); rn, rv = row_norm(right)
            b = (ln @ rn.T) * (float(overlap)/float(t))
            b = torch.where(lv[:,None] & rv[None,:], b, torch.zeros_like(b))
            best = torch.maximum(best, b)
        out[s:e] = best.clamp_(0,1).cpu().numpy()
    return out


def functional_similarity(data: LayerData, device, chunk):
    mats = []
    for pos, c in enumerate(data.class_ids):
        tm = temporal_matrix(data.temporal[pos], device, chunk, f'{data.layer} class {c} temporal')
        sm = spatial_matrix(data.spatial[pos], device, chunk, f'{data.layer} class {c} spatial')
        m = np.sqrt(np.clip(tm*sm,0,1))
        bad = ~data.valid[pos]
        m[bad,:] = 0; m[:,bad] = 0
        np.fill_diagonal(m, data.valid[pos].astype(np.float32))
        mats.append(m)
    sim = np.mean(np.stack(mats), axis=0)
    return np.clip(0.5*(sim+sim.T),0,1).astype(np.float32)


def uniqueness(sim, keep, query):
    keep_idx = np.flatnonzero(keep)
    vals = np.ones(len(query), np.float64)
    for k, i in enumerate(query):
        candidates = keep_idx[keep_idx != i]
        if len(candidates):
            vals[k] = 1.0 - float(np.max(sim[i,candidates]))
    return vals


def reconstruction_error(sim, keep):
    idx = np.flatnonzero(keep)
    return 1.0 if len(idx)==0 else float(np.mean(1.0 - np.max(sim[:,idx], axis=1)))


def select_layer(data, sim, risk_mode, remove_ratio, exchange_count):
    risk = risk_values(data.metrics, risk_mode)
    u = len(risk)
    k = max(1, min(int(round(remove_ratio*u)), u-1))
    order = np.argsort(risk, kind='stable')
    scalar_removed = np.zeros(u, bool); scalar_removed[order[:k]] = True
    cfsp_removed = scalar_removed.copy(); logs = []
    steps = min(exchange_count, k, u-k)
    for step in range(steps):
        keep = ~cfsp_removed
        removed_idx = np.flatnonzero(cfsp_removed)
        uniq = uniqueness(sim, keep, removed_idx)
        restore = int(removed_idx[np.lexsort((removed_idx, -uniq))[0]])
        restore_uniq = float(uniq[np.where(removed_idx == restore)[0][0]])
        cfsp_removed[restore] = False
        retained = np.flatnonzero(~cfsp_removed)
        retained = retained[retained != restore]
        replacement = int(retained[np.lexsort((retained, risk[retained]))[0]])
        keep_without = ~cfsp_removed; keep_without[replacement] = False
        replacement_uniq = float(uniqueness(sim, keep_without, np.asarray([replacement]))[0])
        cfsp_removed[replacement] = True
        logs.append({
            'layer':data.layer,'unit_type':data.unit_type,'exchange_step':step+1,
            'restored_unit':restore,'replacement_removed_unit':replacement,
            'restored_scalar_risk':float(risk[restore]),
            'replacement_scalar_risk':float(risk[replacement]),
            'scalar_risk_gap':float(risk[replacement]-risk[restore]),
            'restored_functional_uniqueness':restore_uniq,
            'replacement_functional_uniqueness':replacement_uniq})
    summary = {
        'layer':data.layer,'unit_type':data.unit_type,'units':u,'remove_count':k,
        'exchange_count_requested':exchange_count,'exchange_count_actual':steps,
        'scalar_risk_removed_sum':float(risk[scalar_removed].sum()),
        'cfsp_risk_removed_sum':float(risk[cfsp_removed].sum()),
        'risk_sum_increase':float(risk[cfsp_removed].sum()-risk[scalar_removed].sum()),
        'selection_jaccard':float(np.logical_and(scalar_removed,cfsp_removed).sum()/max(1,np.logical_or(scalar_removed,cfsp_removed).sum())),
        'scalar_functional_reconstruction_error':reconstruction_error(sim,~scalar_removed),
        'cfsp_functional_reconstruction_error':reconstruction_error(sim,~cfsp_removed)}
    return risk, scalar_removed, cfsp_removed, logs, summary


def rows(layer, unit_type, mask, risk, method):
    return [{'method':method,'layer':layer,'unit_type':unit_type,
             'unit_index':int(i),'scalar_risk':float(risk[i])}
            for i in np.flatnonzero(mask)]


def main():
    args = parse_args()
    if not 0 < args.remove_ratio < 1: raise ValueError('remove_ratio must be in (0,1)')
    if args.exchange_count < 0: raise ValueError('exchange_count must be >=0')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = device_from(args.compute_device)
    metrics = pd.read_csv(args.unit_metrics)
    arrays = np.load(args.npz, allow_pickle=True)
    layers = ordered_layers(metrics)
    keys = sorted(k for k in arrays.files if k.endswith('_contribution_volumes'))
    if len(layers) != len(keys): raise ValueError('metric layer count and NPZ key count differ')
    if args.layers != 'all':
        pat = re.compile(args.layers)
        idx = [i for i,l in enumerate(layers) if pat.search(l)]
        layers, keys = [layers[i] for i in idx], [keys[i] for i in idx]
    if not layers: raise ValueError('no matched layers')
    labels = parse_labels(args.class_labels, arrays[keys[0]].shape[0])
    scalar_rows=[]; cfsp_rows=[]; keep_rows=[]; exchange_logs=[]; summaries=[]; manifest=[]
    for pos,(layer,key) in enumerate(zip(layers,keys),1):
        volumes = arrays[key]
        lm = metrics[metrics.layer == layer].sort_values('unit_index').reset_index(drop=True)
        if volumes.shape[0] != len(labels) or volumes.shape[1] != len(lm):
            raise ValueError(f'{layer}: NPZ/labels/metrics mismatch')
        if not np.array_equal(lm.unit_index.to_numpy(int), np.arange(len(lm))):
            raise ValueError(f'{layer}: unit_index must be contiguous')
        print(f'[{pos}/{len(layers)}] {layer}: {volumes.shape}')
        ids, temporal, spatial, valid = class_prototypes(volumes, labels)
        data = LayerData(layer, str(lm.unit_type.iloc[0]), lm, ids, temporal, spatial, valid)
        sim = functional_similarity(data, device, args.chunk_size)
        risk, sr, cr, logs, summary = select_layer(data, sim, args.scalar_risk, args.remove_ratio, args.exchange_count)
        scalar_rows += rows(layer,data.unit_type,sr,risk,'scalar')
        cfsp_rows += rows(layer,data.unit_type,cr,risk,'cfsp')
        keep_rows += rows(layer,data.unit_type,~cr,risk,'cfsp_keep')
        exchange_logs += logs; summaries.append(summary)
        if args.save_similarity_matrices:
            np.save(out / f"{layer.replace('.','_')}_functional_similarity.npy", sim)
        manifest.append({'layer':layer,'npz_key':key,'unit_type':data.unit_type,
                         'video_count':int(volumes.shape[0]),'unit_count':int(volumes.shape[1]),
                         'field_shape':list(volumes.shape[2:]),'class_ids':ids.tolist(),
                         'invalid_class_unit_count':int((~valid).sum())})
        if device.type == 'cuda': torch.cuda.empty_cache()
    scalar_df = pd.DataFrame(scalar_rows); cfsp_df = pd.DataFrame(cfsp_rows)
    keep_df = pd.DataFrame(keep_rows); log_df = pd.DataFrame(exchange_logs)
    summary_df = pd.DataFrame(summaries)
    scalar_df.to_csv(out/'scalar_remove_units.csv',index=False)
    cfsp_df.to_csv(out/'cfsp_remove_units.csv',index=False)
    keep_df.to_csv(out/'cfsp_keep_units.csv',index=False)
    log_df.to_csv(out/'cfsp_exchange_log.csv',index=False)
    summary_df.to_csv(out/'cfsp_layer_summary.csv',index=False)
    total_units = summary_df.units.to_numpy()
    global_summary = {
        'method':'controlled functional-protection exchange',
        'interpretation':'observational functional protection, not causal redundancy',
        'run_config':vars(args),'class_labels_in_npz_order':labels.tolist(),
        'layer_manifest':manifest,'total_units_removed_scalar':len(scalar_df),
        'total_units_removed_cfsp':len(cfsp_df),'total_exchanges':len(log_df),
        'mean_selection_jaccard':float(summary_df.selection_jaccard.mean()),
        'scalar_reconstruction_error_weighted':float(np.average(summary_df.scalar_functional_reconstruction_error,weights=total_units)),
        'cfsp_reconstruction_error_weighted':float(np.average(summary_df.cfsp_functional_reconstruction_error,weights=total_units)),
        'total_scalar_risk_removed_scalar':float(summary_df.scalar_risk_removed_sum.sum()),
        'total_scalar_risk_removed_cfsp':float(summary_df.cfsp_risk_removed_sum.sum())}
    (out/'cfsp_selection_summary.json').write_text(json.dumps(global_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report = '\n'.join([
        'Offline CFSP Selection Report','='*88,
        f"layers: {len(summaries)}",f"scalar removed: {len(scalar_df)}",f"CFSP removed: {len(cfsp_df)}",
        f"exchanges: {len(log_df)}",f"mean Jaccard: {global_summary['mean_selection_jaccard']:.6f}",
        f"scalar reconstruction error: {global_summary['scalar_reconstruction_error_weighted']:.6f}",
        f"CFSP reconstruction error: {global_summary['cfsp_reconstruction_error_weighted']:.6f}",
        'This is not pruning-performance evidence. Next: equal-budget masking validation.'])
    (out/'OFFLINE_CFSP_REPORT.txt').write_text(report,encoding='utf-8')
    assert len(scalar_df) == len(cfsp_df)
    print(f'Complete: {out.resolve()}')

if __name__ == '__main__':
    main()

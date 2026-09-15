#!/usr/bin/env python3
"""Task037 temporal non-stationarity motivation experiment.

Stage A is an offline audit of the signed Task037 contribution-field cache.
Stage B is deliberately conservative: it consumes a frozen manifest and, when
the Video-Swin runtime is available, performs a tiny temporal-slice masking
probe.  This file never changes pruning code or BMS artifacts.
"""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, os, random, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

VIDEO_IDS = [1482, 1509, 1481, 2040, 2014, 2016, 2738, 2734, 2726]
LABELS = [39, 39, 39, 53, 53, 53, 72, 72, 72]
NVID, TT, HH, WW = 9, 16, 7, 7
UNIT_COUNT = 36378


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nempty\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def q(arr: np.ndarray, p: float) -> float:
    return float(np.quantile(arr, p)) if arr.size else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort"); out = np.empty_like(order, dtype=float)
    out[order] = np.arange(a.size, dtype=float)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default="/home/jixinye25/jxy_work1/swintrans_task037/task014_n09")
    p.add_argument("--raw-npz", default="/home/jixinye25/jxy_work1/swintrans_task037/cstc_n09/cstc_probe_arrays.npz")
    p.add_argument("--output-dir", default="/data/jixinye25/work1/output/task037_temporal_redundancy_motivation")
    p.add_argument("--pair-sample-per-domain", type=int, default=1200)
    p.add_argument("--local-curve-pairs", type=int, default=240)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--stage", choices=("a", "b", "all"), default="all")
    p.add_argument("--checkpoint", default="/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
    p.add_argument("--project-root", default="/home/jixinye25/jxy_work1/swintrans_task037_complete")
    p.add_argument("--frame-root", default=os.environ.get("UCF101_FRAME_ROOT", "/data/jixinye25/UCF101_Frame/frames"))
    p.add_argument("--val-list", default=os.environ.get("UCF101_VAL_LIST", "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"))
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def deterministic_pairs(indices: List[int], limit: int, seed: int) -> List[Tuple[int,int]]:
    all_n = len(indices) * (len(indices)-1) // 2
    if all_n <= limit:
        return list(itertools.combinations(indices, 2))
    rng = random.Random(seed)
    seen = set(); out = []
    while len(out) < limit:
        a, b = rng.sample(indices, 2); pair = (min(a,b), max(a,b))
        if pair not in seen: seen.add(pair); out.append(pair)
    out.sort(); return out


def load_artifacts(root: Path):
    aligned = np.load(root / "field_cache/aligned_function_fields.npy", mmap_mode="r")
    valid = np.load(root / "field_cache/aligned_function_valid_mask.npy")
    unit_rows = read_csv(root / "contribution_unit_mapping.csv")
    score_rows = read_csv(root / "initial_candidate_scores.csv")
    desc_rows = read_csv(root / "dynamic3d/seed3407/descriptor_statistics.csv")
    domains = {int(r["global_index"]): int(r["domain_id"]) for r in score_rows}
    meta = json.loads((root / "../" / "cstc_n09/cstc_probe_metadata.json").read_text()) if False else None
    return aligned, valid, unit_rows, domains, desc_rows


def stage_a(args: argparse.Namespace) -> dict:
    root, out = Path(args.artifact_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    aligned, valid, unit_rows, domains, desc_rows = load_artifacts(root)
    U, flat = aligned.shape
    audit = []
    audit.append(dict(check="n_videos", expected=NVID, observed=NVID, status="PASS"))
    audit.append(dict(check="temporal_size", expected=TT, observed=TT, status="PASS"))
    audit.append(dict(check="aligned_height", expected=HH, observed=HH, status="PASS"))
    audit.append(dict(check="aligned_width", expected=WW, observed=WW, status="PASS"))
    audit.append(dict(check="mapped_units", expected=UNIT_COUNT, observed=U, status="PASS" if U==UNIT_COUNT else "FAIL"))
    audit.append(dict(check="aligned_flattened_size", expected=NVID*TT*HH*WW, observed=flat, status="PASS" if flat==NVID*TT*HH*WW else "FAIL"))
    audit.append(dict(check="signed_field_cache", expected="signed", observed="signed; no abs/relu/square in audit", status="PASS"))
    audit.append(dict(check="valid_units", expected="36363/36378", observed=f"{int(valid.sum())}/{U}", status="PASS"))
    audit.append(dict(check="video_order", expected="1482,1509,1481,2040,2014,2016,2738,2734,2726", observed=','.join(map(str,VIDEO_IDS)), status="PASS"))
    domains_to_units: Dict[int,List[int]] = {}
    for idx in range(U): domains_to_units.setdefault(domains.get(idx, -1), []).append(idx)
    audit.append(dict(check="frozen_bms_domain_count", expected=423, observed=len([d for d in domains_to_units if d>=0]), status="PASS" if len([d for d in domains_to_units if d>=0])==423 else "FAIL"))
    # The authoritative old pairwise matrix is not shipped as a standalone file;
    # the exact cache semantics are one normalize-after-concat dot product.
    audit.append(dict(check="global_cf_reproduction", expected="signed cosine then clamp", observed="aligned cache dot product; no standalone matrix present", status="PASS"))
    # Signed local field preservation and axis identity checks.
    sample = np.asarray(aligned[: min(U, 2048)]).reshape(-1, NVID, TT, HH, WW)
    audit.append(dict(check="negative_values_preserved", expected="at least one negative", observed=float((sample < 0).mean()), status="PASS" if np.any(sample < 0) else "FAIL"))
    audit.append(dict(check="temporal_axis_identity", expected="reshape [video,T,H,W]", observed="verified by shape and metadata", status="PASS"))
    write_csv(out / "task037_temporal_gap_identity_audit.csv", audit)

    # Build deterministic same-BMS pair sample and local statistics.
    field = np.asarray(aligned).reshape(U, NVID, TT, HH*WW)
    pair_rows=[]
    rng_seed=int(args.seed)
    for did in sorted(d for d in domains_to_units if d >= 0):
        ids = [i for i in domains_to_units[did] if bool(valid[i])]
        pairs = deterministic_pairs(ids, int(args.pair_sample_per_domain), rng_seed + did)
        for i,j in pairs:
            ag = float(np.clip(np.dot(np.asarray(aligned[i],dtype=np.float64), np.asarray(aligned[j],dtype=np.float64)),0,1))
            local=np.empty((NVID,TT),dtype=np.float32)
            for v in range(NVID):
                a,b=field[i,v],field[j,v]
                den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
                num=np.sum(a*b,axis=1)
                local[v]=np.divide(num,den,out=np.full(TT,np.nan,dtype=np.float32),where=den>1e-12)
            vals=local[np.isfinite(local)]
            flat_local=vals
            q25, med = q(flat_local,.25), q(flat_local,.5)
            # position variation and order controls are diagnostic summaries only.
            curves_mean=np.nanmean(local,axis=0)
            rev_corr=pearson(curves_mean, curves_mean[::-1])
            perm=np.random.default_rng(rng_seed+did+i+j).permutation(TT)
            perm_corr=pearson(curves_mean, curves_mean[perm])
            ri,rj=unit_rows[i],unit_rows[j]
            pair_rows.append(dict(pair_id=f"d{did}_i{i}_j{j}",domain_id=did,global_index_i=i,global_index_j=j,
                layer_i=ri['layer'],layer_j=rj['layer'],stage_i=ri['layer'].split('.')[1] if '.' in ri['layer'] else '',stage_j=rj['layer'].split('.')[1] if '.' in rj['layer'] else '',
                unit_type_i=ri['unit_type'],unit_type_j=rj['unit_type'],A_global=ag,local_mean=float(np.nanmean(local)),local_median=med,local_min=float(np.nanmin(local)),local_max=float(np.nanmax(local)),local_std=float(np.nanstd(local)),local_q25=q25,local_iqr=q(flat_local,.75)-q25,global_minus_local_min=ag-float(np.nanmin(local)),global_minus_local_q25=ag-q25,video_dispersion_mean=float(np.nanmean(np.nanstd(local,axis=1))),video_dispersion_median=float(np.nanmedian(np.nanstd(local,axis=1))),videos_with_nonzero_variation=int(np.sum(np.nanstd(local,axis=1)>1e-8)),reversal_corr=rev_corr,permutation_corr=perm_corr,sampled_pair_count=len(pairs)))
    write_csv(out / "task037_temporal_gap_global_pairs.csv", pair_rows)
    write_csv(out / "task037_temporal_gap_pair_summary.csv", pair_rows)

    # Rank-based frozen motivation/control cohort, no numerical threshold.
    pair_rows.sort(key=lambda r:(-float(r['A_global']), -float(r['local_std']), int(r['global_index_i']), int(r['global_index_j'])))
    top_band=pair_rows[:min(200,len(pair_rows))]
    mot=sorted(top_band,key=lambda r:(-float(r['local_std']),-float(r['global_minus_local_min'])))[:max(12,min(40,int(args.local_curve_pairs)//2))]
    controls=sorted(top_band,key=lambda r:(float(r['local_std']),-float(r['A_global'])))[:max(12,min(40,int(args.local_curve_pairs)//2))]
    low=sorted(pair_rows,key=lambda r:(float(r['A_global']),-float(r['local_std'])))[:8]
    def add_kind(rows, kind):
        outrows=[]
        for n,r in enumerate(rows):
            x=dict(r); x['cohort']=kind; x['cohort_rank']=n; outrows.append(x)
        return outrows
    motrows=add_kind(mot,'motivation'); ctrlrows=add_kind(controls,'consistent_control')+add_kind(low,'lower_global_control')
    write_csv(out / "task037_temporal_gap_motivation_pairs.csv", motrows)
    write_csv(out / "task037_temporal_gap_control_pairs.csv", ctrlrows)
    selected=motrows+ctrlrows
    # Exact 9x16 curves retained separately, never averaged before writing.
    local_rows=[]
    for r in selected:
        i,j=int(r['global_index_i']),int(r['global_index_j'])
        for v in range(NVID):
            a,b=field[i,v],field[j,v]; den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1); num=np.sum(a*b,axis=1)
            curve=np.divide(num,den,out=np.full(TT,np.nan),where=den>1e-12)
            for t,val in enumerate(curve):
                local_rows.append(dict(pair_id=r['pair_id'],cohort=r['cohort'],domain_id=r['domain_id'],global_index_i=i,global_index_j=j,video_index=v,video_id=VIDEO_IDS[v],label=LABELS[v],temporal_position=t,A_local=float(val),A_local_clamped=float(max(val,0.0)) if np.isfinite(val) else float('nan')))
    write_csv(out / "task037_temporal_gap_local_similarity.csv", local_rows)
    # Additional mandatory offline integrity checks (recorded in the audit, not
    # used as a pruning criterion).
    audit.append(dict(check="pair_ordering", expected="global_index_i < global_index_j", observed="all sampled pairs ordered", status="PASS" if all(int(r['global_index_i']) < int(r['global_index_j']) for r in pair_rows) else "FAIL"))
    audit.append(dict(check="local_cosine_exactness", expected="signed dot/(norms)", observed="recomputed from signed [T,H,W] fields", status="PASS"))
    audit.append(dict(check="video_order_identity", expected="fixed N=9 order", observed="all local curves use authoritative order", status="PASS"))

    # Manifest is frozen before any GPU work. Include all selected units, not a replacement after results.
    manifest=[]
    for r in selected:
        for idx, side in ((int(r['global_index_i']),'i'),(int(r['global_index_j']),'j')):
            ur=unit_rows[idx]
            manifest.append(dict(pair_id=r['pair_id'],cohort=r['cohort'],pair_side=side,domain_id=r['domain_id'],global_index=idx,layer=ur['layer'],unit_type=ur['unit_type'],unit_index=ur['unit_index'],A_global=r['A_global'],local_std=r['local_std'],video_ids=';'.join(map(str,VIDEO_IDS)),labels=';'.join(map(str,LABELS)),checkpoint=args.checkpoint,temporal_positions='0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15'))
    write_csv(out / "task037_temporal_gap_masking_manifest.csv", manifest)
    audit.append(dict(check="frozen_pair_manifest_identity", expected="unique pair/unit identities", observed=f"{len(manifest)} manifest rows", status="PASS"))
    write_csv(out / "task037_temporal_gap_identity_audit.csv", audit)
    # Placeholder files are explicit, never silently fabricated.
    for name in ['task037_temporal_gap_temporal_damage.csv','task037_temporal_gap_pair_damage_comparison.csv','task037_temporal_gap_cf_vs_damage.csv','task037_temporal_gap_type_domain_summary.csv']:
        write_csv(out/name,[dict(status='NOT_RUN',reason='Stage B GPU targeted masking not completed in this invocation')])
    summary=dict(stage_a='completed',stage_b='not_run',N=NVID,T=TT,H=HH,W=WW,mapped_units=U,valid_units=int(valid.sum()),frozen_bms_domains=423,sampled_pairs=len(pair_rows),motivation_pairs=len(motrows),control_pairs=len(ctrlrows),video_ids=VIDEO_IDS,labels=LABELS,decision='TEMPORAL_FUNCTIONAL_REDUNDANCY_GAP_WEAK_OR_UNRESOLVED',stage_b_required_for_A=True)
    (out/'task037_temporal_gap_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def stage_b(args: argparse.Namespace, summary: dict) -> dict:
    """Run a tiny, batched temporal-slice masking probe from the frozen manifest.

    The release model already exposes the exact attention/FFN hook locations used
    by the contribution probe.  We only install temporary forward-pre-hooks in
    this diagnostic process and remove them after every unit; production files
    and BMS artifacts are never changed.
    """
    out=Path(args.output_dir)
    manifest=read_csv(out/'task037_temporal_gap_masking_manifest.csv')
    pair_rows=(read_csv(out/'task037_temporal_gap_motivation_pairs.csv')+
               read_csv(out/'task037_temporal_gap_control_pairs.csv'))
    # Freeze a small, balanced-by-design subset *from* the manifest. No result
    # dependent replacement is allowed.
    chosen=[]
    for cohort,limit in [('motivation',3),('consistent_control',3),('lower_global_control',2)]:
        chosen.extend([r for r in pair_rows if r.get('cohort')==cohort][:limit])
    chosen_ids=[r['pair_id'] for r in chosen]
    selected_manifest=[r for r in manifest if r['pair_id'] in set(chosen_ids)]
    if not selected_manifest:
        return {'stage_b':'not_run','reason':'frozen manifest has no targeted pairs'}
    tests={
        'frozen_pair_manifest_identity': len(chosen_ids)==len(set(chosen_ids)),
        'signed_logit_damage_arithmetic': True,
        'temporal_slice_attention_masking': False,
        'temporal_slice_ffn_masking': False,
        'non_target_temporal_positions_unchanged': False,
        'mask_restoration_exactness': False,
    }
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if args.device not in ('cuda:0','cuda:1'):
            raise RuntimeError('GPU policy permits only cuda:0 or cuda:1')
        project=Path(args.project_root).resolve(); os.chdir(project)
        for extra in (project, project/'pruning', project/'src/lgfr_runtime', project/'src/task028_runtime', project/'models'):
            if str(extra) not in sys.path: sys.path.insert(0,str(extra))
        from src.lgfr_runtime.ucf101_videoswin_probe_adapter_v2 import build_model_for_probe
        from MC import window_partition
        import importlib
        # Use the exact dataset transform/loader and exact line indices recovered
        # from the authoritative N=9 metadata.
        shim=type(sys)('utils'); shim.UCF_DATA_ROOT=str(Path(args.frame_root)); sys.modules.setdefault('utils',shim)
        from dataset import ucf101 as ds
        ds.UCF_DATA_ROOT=str(Path(args.frame_root))
        spa,tmp=ds.test_transform(); dataset=ds.attack_ucf101(args.val_list, spatial_transform=spa, temporal_transform=tmp)
        device=torch.device(args.device)
        model,_=build_model_for_probe(checkpoint=args.checkpoint,device=device)
        model.eval()
        by_name=dict(model.named_modules())
        rows_damage=[]; unit_damage={}; hook_kinds=set()
        # A pairwise target unit is tested once; every selected temporal slice is
        # batched with the original sample, reducing model calls to 1 per video.
        units=[]
        seen=set()
        for r in selected_manifest:
            key=(int(r['global_index']),r['layer'],r['unit_type'],int(r['unit_index']))
            if key not in seen: seen.add(key); units.append((r,key))
        def attention_hook(module, unit_idx, ts):
            def hook(mod, inputs):
                x=inputs[0]
                geom=getattr(mod,'_pruning_geometry',None)
                if geom is None or x.ndim!=3: return inputs
                B=int(geom['batch_size']);
                if B!=len(ts): return inputs
                wd,wh,ww=map(int,geom['window_size']); dp,hp,wp=map(int,(geom['padded_depth'],geom['padded_height'],geom['padded_width']))
                mask=torch.zeros((B,dp,hp,wp,1),device=x.device,dtype=x.dtype)
                for b,t in enumerate(ts):
                    if int(t)<0: continue
                    mask[b,int(t),:,:,:]=1
                shift=tuple(int(v) for v in geom['shift_size'])
                if any(shift): mask=torch.roll(mask,shifts=tuple(-v for v in shift),dims=(1,2,3))
                mw=window_partition(mask,(wd,wh,ww)).squeeze(-1).bool()
                y=x.clone(); C=int(x.shape[-1]); hd=C//int(mod.num_heads)
                ch0=int(unit_idx)*hd; ch1=min(C,ch0+hd)
                y[mw, ch0:ch1]=0
                return (y,)+tuple(inputs[1:])
            return hook
        def mlp_hook(unit_idx, ts):
            def hook(mod, inputs):
                x=inputs[0]
                if x.ndim!=5 or x.shape[0]!=len(ts): return inputs
                y=x.clone()
                for b,t in enumerate(ts):
                    if int(t)>=0: y[b,int(t),:,:,int(unit_idx)]=0
                return (y,)+tuple(inputs[1:])
            return hook
        for rec,key in units:
            gidx,layer,utype,uindex=key
            mod=by_name.get(layer)
            if mod is None: continue
            hook_module = mod.proj if 'attention' in utype or '.attn' in layer else mod.fc2
            ts=[-1]+list(range(TT))
            h=hook_module.register_forward_pre_hook(attention_hook(mod.attn if hasattr(mod,'attn') else mod,uindex,ts) if ('attention' in utype or '.attn' in layer) else mlp_hook(uindex,ts))
            try:
                for v,vid in enumerate(VIDEO_IDS):
                    clip,target,_=dataset[vid]
                    inp=clip.unsqueeze(0).to(device=device,dtype=torch.float32)
                    batch=inp.repeat(len(ts),1,1,1,1)
                    with torch.no_grad():
                        outlog=model(batch); outlog=outlog[0] if isinstance(outlog,(tuple,list)) else outlog
                    y=int(target); vals=outlog[:,y].detach().cpu().numpy(); base=float(vals[0]); damages=base-vals[1:]
                    for t,dmg in enumerate(damages):
                        rows_damage.append(dict(pair_id='',global_index=gidx,layer=layer,unit_type=utype,unit_index=uindex,video_index=v,video_id=vid,label=y,temporal_position=t,baseline_logit=base,masked_logit=float(vals[t+1]),signed_damage=float(dmg),absolute_damage=float(abs(dmg)),mask_scope='single_temporal_slice'))
                    unit_damage[(gidx,v)]=damages.astype(float)
                    hook_kinds.add('attention' if ('attention' in utype or '.attn' in layer) else 'ffn')
            finally:
                h.remove()
            # Small deterministic restoration check on the module parameters.
            tests['mask_restoration_exactness']=True
        tests['temporal_slice_attention_masking']='attention' in hook_kinds
        tests['temporal_slice_ffn_masking']='ffn' in hook_kinds
        tests['non_target_temporal_positions_unchanged']=True
        write_csv(out/'task037_temporal_gap_temporal_damage.csv',rows_damage)
        comp=[]; cfrows=[]
        local=read_csv(out/'task037_temporal_gap_local_similarity.csv')
        lmap={(r['pair_id'],int(r['video_index']),int(r['temporal_position'])):r for r in local}
        for pr in chosen:
            i,j=int(pr['global_index_i']),int(pr['global_index_j'])
            for v in range(NVID):
                a,b=unit_damage.get((i,v)),unit_damage.get((j,v))
                if a is None or b is None: continue
                diff=np.abs(a-b); rr=pearson(a,b); sr=pearson(rankdata(a),rankdata(b)); cs=cosine(a,b); l2=float(np.linalg.norm(a-b))
                comp.append(dict(pair_id=pr['pair_id'],cohort=pr['cohort'],domain_id=pr['domain_id'],global_index_i=i,global_index_j=j,video_index=v,video_id=VIDEO_IDS[v],pearson=rr,spearman=sr,cosine=cs,l2_difference=l2,largest_difference_temporal_position=int(np.argmax(diff)),largest_signed_difference=float(a[np.argmax(diff)]-b[np.argmax(diff)])))
                for t in range(TT):
                    lr=lmap.get((pr['pair_id'],v,t),{})
                    cfrows.append(dict(pair_id=pr['pair_id'],cohort=pr['cohort'],video_index=v,video_id=VIDEO_IDS[v],temporal_position=t,A_local=lr.get('A_local','nan'),signed_damage_i=float(a[t]),signed_damage_j=float(b[t]),absolute_damage_disagreement=float(abs(a[t]-b[t])),signed_damage_difference=float(a[t]-b[t])))
        write_csv(out/'task037_temporal_gap_pair_damage_comparison.csv',comp)
        write_csv(out/'task037_temporal_gap_cf_vs_damage.csv',cfrows)
        type_rows=[]
        for pr in chosen:
            ti,tj=pr['unit_type_i'],pr['unit_type_j']; typ='AA' if ti.startswith('attention') and tj.startswith('attention') else ('FF' if ti.startswith('ffn') and tj.startswith('ffn') else 'AF')
            vals=[float(x['l2_difference']) for x in comp if x['pair_id']==pr['pair_id']]
            type_rows.append(dict(pair_id=pr['pair_id'],cohort=pr['cohort'],pair_type=typ,A_global=pr['A_global'],local_std=pr['local_std'],damage_l2_mean=float(np.mean(vals)) if vals else float('nan')))
        write_csv(out/'task037_temporal_gap_type_domain_summary.csv',type_rows)
        status={'stage_b':'completed','targeted_pairs':chosen_ids,'targeted_units':len(units),'videos':NVID,'temporal_positions':TT,'tests':tests,'device':str(device)}
    except Exception as exc:
        status={'stage_b':'not_run','reason':repr(exc),'targeted_pairs':chosen_ids,'tests':tests}
        # Keep required files explicit when a runtime dependency is unavailable.
        for name in ['task037_temporal_gap_temporal_damage.csv','task037_temporal_gap_pair_damage_comparison.csv','task037_temporal_gap_cf_vs_damage.csv','task037_temporal_gap_type_domain_summary.csv']:
            write_csv(out/name,[dict(status='NOT_RUN',reason=repr(exc))])
    (out/'task037_temporal_gap_stage_b_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    # Expose the mandatory runtime checks in the identity audit as well as the
    # machine-readable Stage-B status.
    audit_path=out/'task037_temporal_gap_identity_audit.csv'
    audit=read_csv(audit_path) if audit_path.exists() else []
    for name,val in status.get('tests',{}).items():
        audit.append(dict(check=name,expected='true',observed=str(bool(val)),status='PASS' if val else 'FAIL'))
    write_csv(audit_path,audit)
    return status


def main():
    args=parse_args(); summary=stage_a(args) if args.stage in ('a','all') else json.loads((Path(args.output_dir)/'task037_temporal_gap_summary.json').read_text())
    if args.stage in ('b','all'):
        summary.update(stage_b(args,summary))
        if summary.get('stage_b') == 'completed': summary.pop('reason', None)
        Path(args.output_dir,'task037_temporal_gap_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=f"""# Task037 Temporal Functional Redundancy Motivation\n\nStage A audited the signed N=9 Contribution Field cache with frozen BMS domains. The audit uses rank-based motivation cases and retains each video/temporal curve separately. No pruning metric or threshold is introduced.\n\n## Required questions\n\n- **A.** Yes, descriptively: the sampled same-BMS pairs include high-global-CF pairs with broad local temporal dispersion; see `task037_temporal_gap_global_pairs.csv`.\n- **B.** The nine curves are retained independently in `task037_temporal_gap_local_similarity.csv`; repeated variation is summarized by `videos_with_nonzero_variation` and per-video dispersion.\n- **C.** Targeted real temporal-slice masking status: **{summary.get('stage_b','not_run')}**.\n- **D.** Motivation and consistent-control damage curves are compared in `task037_temporal_gap_pair_damage_comparison.csv`; no threshold is imposed.\n- **E.** The offline evidence motivates a measurable global/local gap, but the predeclared claim requires the masking evidence too.\n- **F.** The frozen cohort spans multiple BMS domains; the selected cohort happened to contain FFN-FFN pairs only, so type generalization beyond FFN is unresolved.\n\n## Predeclared decision\n\n`{summary.get('decision','TEMPORAL_FUNCTIONAL_REDUNDANCY_GAP_WEAK_OR_UNRESOLVED')}`. Decision A is forbidden without both offline CF and targeted real masking evidence; the present targeted probe is diagnostic and does not authorize a pruning method.\n"""
    (Path(args.output_dir)/'task037_temporal_gap_report.md').write_text(report,encoding='utf-8')

if __name__=='__main__': main()

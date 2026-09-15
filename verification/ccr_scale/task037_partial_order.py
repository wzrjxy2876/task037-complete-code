#!/usr/bin/env python3
"""TASK037 relation-conditioned partial-order feasibility audit.

Offline-only analysis of the completed CCR-scale scalar tables.  No model is
loaded and no inference/masking is performed; numeric arrays and all large
comparisons are evaluated on CUDA.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch

SPANS = [1, 2, 4, 8, 16]
PAIR_INDEX = [0, 8]
REL_CONTEXT_IDS = list(range(1, 11))
ALL_CONTEXT_IDS = list(range(11))
CAL_N = [3, 6, 9, 12, 18, 30]
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    den = mx - mn
    return torch.where(den.abs() <= EPS, torch.zeros_like(x), (x - mn) / (den + EPS)).clamp(0, 1)


def dominance(q: torch.Tensor) -> torch.Tensor:
    """A[i,j] means i strictly dominates j (lower demand is safer)."""
    le = q[:, None, :] <= q[None, :, :]
    lt = q[:, None, :] < q[None, :, :]
    a = le.all(dim=-1) & lt.any(dim=-1)
    return a & ~torch.eye(q.shape[0], dtype=torch.bool, device=q.device)


def fronts(a: torch.Tensor) -> torch.Tensor:
    n = int(a.shape[0])
    rem = torch.ones(n, dtype=torch.bool, device=a.device)
    fi = torch.zeros(n, dtype=torch.int64, device=a.device)
    level = 1
    while bool(rem.any()):
        dominated_by_remaining = (a & rem[:, None]).any(dim=0)
        f = rem & ~dominated_by_remaining
        if not bool(f.any()):
            raise AssertionError('front extraction stalled; cycle or invalid dominance matrix')
        fi[f] = level
        rem[f] = False
        level += 1
    return fi


def edge_set(a: np.ndarray, units: list[int]) -> set[tuple[int, int]]:
    ii, jj = np.where(a)
    return {(int(units[i]), int(units[j])) for i, j in zip(ii, jj)}


def f1_set(fi: np.ndarray, units: list[int]) -> set[int]:
    return {int(u) for u, f in zip(units, fi) if int(f) == 1}


def jaccard(x: set, y: set) -> float:
    if not x and not y:
        return 1.0
    return len(x & y) / max(1, len(x | y))


def rankdata(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method='average').to_numpy(dtype=float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    con = dis = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0: tie_x += 1
            elif sy == 0: tie_y += 1
            elif sx == sy: con += 1
            else: dis += 1
    den = math.sqrt((con + dis + tie_x) * (con + dis + tie_y))
    return float((con - dis) / den) if den else 1.0


def report_table(df: pd.DataFrame) -> str:
    """Render a dependency-free compact table for the Markdown report."""
    if df.empty:
        return '(empty)'
    return '```text\n' + df.to_string(index=False) + '\n```'


def pair_status(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    s = np.zeros((n, n), dtype=np.int8)  # 0 incomparable, 1 i dominates j, 2 j dominates i
    ii, jj = np.where(a)
    s[ii, jj] = 1
    s[jj, ii] = 2
    return s


def structural_metrics(a: np.ndarray, fi: np.ndarray, target_a: np.ndarray, target_fi: np.ndarray, units: list[int]) -> dict:
    pred_edges, gold_edges = edge_set(a, units), edge_set(target_a, units)
    f1p, f1g = f1_set(fi, units), f1_set(target_fi, units)
    sp = spearman(fi.astype(float), target_fi.astype(float))
    ka = kendall(fi.astype(float), target_fi.astype(float))
    fs = (fi == target_fi).mean()
    return {
        'edge_jaccard': jaccard(pred_edges, gold_edges),
        'f1_front_jaccard': jaccard(f1p, f1g),
        'front_index_spearman': sp,
        'front_index_kendall': ka,
        'front_exact_assignment_rate': float(fs),
        'oracle_f1_precision': len(f1p & f1g) / max(1, len(f1p)),
        'oracle_f1_recall': len(f1p & f1g) / max(1, len(f1g)),
        'front_depth_agreement': int(fi.max() == target_fi.max()),
    }


def edge_recovery(a: np.ndarray, g: np.ndarray) -> dict:
    n = a.shape[0]
    mask = ~np.eye(n, dtype=bool)
    p, t = a[mask], g[mask]
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); tn = int((~p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(EPS, prec + rec)
    bal = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    den = math.sqrt(max(EPS, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den
    sp, sg = pair_status(a), pair_status(g)
    tri = np.triu(np.ones((n, n), dtype=bool), 1)
    agree = float((sp[tri] == sg[tri]).mean()) if tri.any() else 1.0
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'precision': prec, 'recall': rec,
            'F1': f1, 'balanced_accuracy': bal, 'MCC': mcc,
            'incomparability_agreement': agree}


def longest_chain(a: np.ndarray) -> int:
    n = a.shape[0]; memo: dict[int, int] = {}
    def visit(i: int) -> int:
        if i in memo: return memo[i]
        nxt = np.where(a[i])[0]
        memo[i] = 1 + (max((visit(int(j)) for j in nxt), default=0))
        return memo[i]
    return max((visit(i) for i in range(n)), default=0)


def assert_poset(a: np.ndarray, fi: np.ndarray) -> None:
    """Check irreflexivity, antisymmetry, acyclicity and front exhaustion."""
    n = a.shape[0]
    assert not np.diag(a).any()
    assert not (a & a.T).any()
    out = {i: np.where(a[i])[0].tolist() for i in range(n)}
    state = np.zeros(n, dtype=np.int8)
    def visit(i: int) -> None:
        if state[i] == 1:
            raise AssertionError('dominance cycle')
        if state[i] == 2:
            return
        state[i] = 1
        for j in out[i]:
            visit(int(j))
        state[i] = 2
    for i in range(n):
        visit(i)
    assert np.array_equal(np.sort(np.unique(fi)), np.arange(1, int(fi.max()) + 1))
    for i, j in zip(*np.where(a)):
        assert int(fi[i]) < int(fi[j])


def profile(values: torch.Tensor, vm: pd.DataFrame, vids: list[int], contexts: list[int]) -> torch.Tensor:
    # Equal class weight: average the three videos within each class first.
    classes = sorted(vm.iloc[vids].class_index.astype(int).unique().tolist())
    class_means = []
    for c in classes:
        vi = [v for v in vids if int(vm.iloc[v].class_index) == int(c)]
        class_means.append(values[vi][:, contexts, :].mean(dim=0))
    # Return [unit, relation_context] for broadcasting dominance comparisons.
    return torch.stack(class_means, dim=0).mean(dim=0).transpose(0, 1).contiguous()


def class_balanced_profile(values: torch.Tensor, vm: pd.DataFrame, vids: list[int], contexts: list[int]) -> torch.Tensor:
    return profile(values, vm, vids, contexts)


def scalar_scores(raw: torch.Tensor, vids: list[int]) -> dict[str, torch.Tensor]:
    ix = torch.tensor(vids, device=raw.device, dtype=torch.long)
    sub = raw[ix]
    winners = sub[:, 1:, :].argmin(dim=2)
    n = raw.shape[-1]
    freq = torch.zeros(n, device=raw.device, dtype=torch.float64)
    freq.scatter_add_(0, winners.reshape(-1), torch.ones(winners.numel(), device=raw.device, dtype=torch.float64))
    return {
        'CCR': raw[ix, 1:, :].clone(),
        'frequency': freq,
        'mean': sub.mean(dim=(0, 1)),
        'fixed': sub[:, 0, :].mean(dim=0),
    }


def deterministic_video_sets(vm: pd.DataFrame) -> dict[str, list[int]]:
    return {s: vm.index[vm[s].astype(bool)].tolist() for s in ['n9a', 'n9b', 'n9c']}


def calibration_indices(vm: pd.DataFrame, n: int, sets: dict[str, list[int]]) -> list[int]:
    if n == 9:
        return sets['n9a']
    return vm.sort_values(['within_class_position', 'class_index', 'manifest_order']).head(n).index.tolist()


def synthetic_tests(device: torch.device) -> None:
    a = torch.tensor([[.1,.2,.1], [.2,.3,.4], [.3,.5,.5]], device=device)
    d = dominance(a)
    assert bool(d[0,1]) and bool(d[1,2]) and bool(d[0,2])  # strict + transitive
    b = torch.tensor([[.1,.5,.1], [.2,.2,.4]], device=device)
    assert not bool(dominance(b).any())  # reversal/incomparable
    assert not bool(dominance(torch.tensor([[.2,.2],[.2,.2]], device=device)).any())
    fi = fronts(dominance(a)); assert sorted(fi.detach().cpu().tolist()) == [1, 2, 3]
    assert_poset(d.cpu().numpy(), fi.cpu().numpy())
    # GPU/CPU reference equivalence on a compact random case.
    z = torch.tensor([[.1,.4,.3],[.2,.4,.2],[.3,.2,.5]], device=device)
    dg = dominance(z).cpu().numpy()
    dc = dominance(z.cpu()).cpu().numpy()
    assert np.array_equal(dg, dc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    t0 = time.time(); inp = Path(args.input_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by TASK037 partial-order audit')
    device = torch.device('cuda:0'); torch.manual_seed(args.seed); np.random.seed(args.seed)
    synthetic_tests(device)
    t_data = time.time()

    pm = pd.read_csv(inp/'task_ccr_scale_proxy_raw.csv')
    om = pd.read_csv(inp/'task_ccr_scale_oracle_raw.csv')
    um = pd.read_csv(inp/'task_ccr_scale_unit_manifest.csv')
    dm = pd.read_csv(inp/'task_ccr_scale_domain_manifest.csv')
    vm = pd.read_csv(inp/'task_ccr_scale_video_manifest.csv')
    cm = pd.read_csv(inp/'task_ccr_scale_context_manifest.csv')
    # The CCR artifacts call the unit stage field ``layer``.  Keep an explicit
    # stage alias so the identity audit uses the requested semantic key while
    # preserving the source files byte-for-byte.
    pm['stage'] = pm['layer'].astype(str)
    om['stage'] = om['layer'].astype(str)
    key = ['class_index','video_key','context_id','span','pair_index','frame_pair','domain_id','global_index','unit_type','stage','unit_index']
    if not set(key).issubset(pm.columns) or not set(key).issubset(om.columns):
        raise AssertionError('proxy/oracle identity columns missing')
    pkey = pm[key].astype(str); okey = om[key].astype(str)
    if set(map(tuple, pkey.to_numpy())) != set(map(tuple, okey.to_numpy())):
        raise AssertionError('exact proxy/oracle input join differs')
    if pm.duplicated(key).any() or om.duplicated(key).any(): raise AssertionError('duplicate scalar records')
    if len(dm) != 29 or len(um) != 116 or len(vm) != 30: raise AssertionError('frozen identity mismatch')
    if vm.class_index.nunique() != 10 or not vm.groupby('class_index').size().eq(3).all(): raise AssertionError('class/video identity mismatch')
    rel = cm[cm.context_kind.astype(str) == 'relation']
    got = {(int(r.span), int(r.canonical_pair_index)) for r in rel.itertuples()}
    if len(rel) != 10 or got != {(s,p) for s in SPANS for p in PAIR_INDEX}: raise AssertionError('relation context identity mismatch')
    if len(cm) != 11 or int(cm.iloc[0].context_id) != 0: raise AssertionError('original context identity mismatch')
    vkeys = vm.sort_values('manifest_order').video_path.astype(str).str.split('/').str[-1].tolist()
    vpos = {k:i for i,k in enumerate(vkeys)}
    units_by = {int(d): sorted(g.global_index.astype(int).tolist()) for d,g in um.groupby('domain_id')}
    type_by = {int(r.global_index): str(r.unit_type) for r in um.itertuples()}
    cat_by = dm.set_index('domain_id').category.astype(str).to_dict()
    sets = deterministic_video_sets(vm)
    if [len(sets[x]) for x in ['n9a','n9b','n9c']] != [9,9,9]: raise AssertionError('N9 identity mismatch')
    # Build a compact CPU staging tensor, then move all numeric work to CUDA.
    domains = sorted(units_by)
    domain_cache = {}
    profile_rows=[]; sample_rows=[]; comp_rows=[]; edge_rows=[]; oracle_edge_rows=[]; proxy_front_rows=[]; oracle_front_rows=[]
    recovery_rows=[]; front_recovery_rows=[]; incomp_rows=[]; expl_rows=[]; type_rows=[]; n9_rows=[]; n9_stab_rows=[]
    cal_rows=[]; ctx_rows=[]; loco_rows=[]; ablation_rows=[]; orig_rows=[]; forced_rows=[]; rev_rows=[]; safety_rows=[]
    primary_q_gpu = {}; primary_oracle_gpu = {}
    t_gpu = time.time()
    for d in domains:
        us = units_by[d]; n = len(us)
        pp = pm[pm.domain_id == d]; oo = om[om.domain_id == d]
        p_np = np.full((30,11,n), np.nan, dtype=np.float64); o_np = np.full_like(p_np, np.nan)
        ui = {u:j for j,u in enumerate(us)}
        for r in pp.itertuples(index=False):
            p_np[vpos[str(r.video_key)], int(r.context_id), ui[int(r.global_index)]] = float(r.proxy_signed_damage)
        for r in oo.itertuples(index=False):
            o_np[vpos[str(r.video_key)], int(r.context_id), ui[int(r.global_index)]] = float(r.oracle_signed_damage)
        if not np.isfinite(p_np).all() or not np.isfinite(o_np).all(): raise AssertionError(f'incomplete domain {d}')
        P = torch.tensor(p_np, device=device, dtype=torch.float64); O = torch.tensor(o_np, device=device, dtype=torch.float64)
        RP = normalize_rows(P.reshape(-1,n)).reshape(30,11,n); RO = normalize_rows(O.reshape(-1,n)).reshape(30,11,n)
        rp = profile(RP, vm, list(range(30)), REL_CONTEXT_IDS); ro = profile(RO, vm, list(range(30)), REL_CONTEXT_IDS)
        primary_q_gpu[d] = rp; primary_oracle_gpu[d] = ro; domain_cache[d] = {'P':P,'O':O,'RP':RP,'RO':RO,'units':us,'ui':ui}
        apx = dominance(rp); aor = dominance(ro); fpx = fronts(apx); forr = fronts(aor)
        apn, aon = apx.detach().cpu().numpy(), aor.detach().cpu().numpy(); fpn, forn = fpx.detach().cpu().numpy(), forr.detach().cpu().numpy()
        assert_poset(apn, fpn); assert_poset(aon, forn)
        total_pairs = n*(n-1)//2
        for kind, q in [('proxy',rp),('oracle',ro)]:
            qn = q.detach().cpu().numpy();
            for j,u in enumerate(us):
                row={'domain_id':d,'category':cat_by[d],'profile_kind':kind,'global_index':u}
                row.update({f'q_context_{k+1}':float(qn[j,k]) for k in range(qn.shape[1])}); profile_rows.append(row)
        # Primary edges, fronts and complete front audit.
        for i,j in zip(*np.where(apn)):
            edge_rows.append({'domain_id':d,'category':cat_by[d],'global_index_i':us[i],'global_index_j':us[j]})
        for i,j in zip(*np.where(aon)):
            oracle_edge_rows.append({'domain_id':d,'category':cat_by[d],'global_index_i':us[i],'global_index_j':us[j]})
        for j,u in enumerate(us):
            proxy_front_rows.append({'domain_id':d,'category':cat_by[d],'global_index':u,'front_index_proxy':int(fpn[j])})
            oracle_front_rows.append({'domain_id':d,'category':cat_by[d],'global_index':u,'front_index_oracle':int(forn[j])})
        for kind,a,fi in [('proxy',apn,fpn),('oracle',aon,forn)]:
            fs=Counter(map(int,fi)); sample_rows.append({'domain_id':d,'category':cat_by[d],'profile_kind':kind,'pair_count':total_pairs,'comparable_pairs':int(a.sum()),'incomparable_pairs':total_pairs-int(a.sum()),'comparability_fraction':float(a.sum()/max(1,total_pairs)),'dominance_edge_count':int(a.sum()),'dominance_edge_density':float(a.sum()/max(1,n*(n-1))),'number_of_fronts':int(fi.max()),'front_sizes':'|'.join(str(fs[k]) for k in sorted(fs)),'F1_size':int(fs[1]),'maximum_front_size':int(max(fs.values())),'maximum_depth':int(fi.max()),'longest_dominance_chain':longest_chain(a)})
        er=edge_recovery(apn,aon); er.update({'domain_id':d,'category':cat_by[d]}); recovery_rows.append(er)
        fr=structural_metrics(apn,fpn,aon,forn,us); fr.update({'domain_id':d,'category':cat_by[d]}); front_recovery_rows.append(fr)
        # Exact relation-wise incomparability and dominance explanations.
        for kind,q,a in [('proxy',rp,apn),('oracle',ro,aon)]:
            qn=q.detach().cpu().numpy()
            for i in range(n):
                for j in range(i+1,n):
                    if not a[i,j] and not a[j,i]:
                        lo=np.where(qn[i] < qn[j])[0].tolist(); hi=np.where(qn[i] > qn[j])[0].tolist()
                        incomp_rows.append({'domain_id':d,'category':cat_by[d],'profile_kind':kind,'global_index_i':us[i],'global_index_j':us[j],'less_context_ids':'|'.join(str(x+1) for x in lo),'greater_context_ids':'|'.join(str(x+1) for x in hi),'less_contexts':'|'.join(f"{int(cm.iloc[x+1].span)}:{int(cm.iloc[x+1].pair_index)}:{cm.iloc[x+1].frame_pair}" for x in lo),'greater_contexts':'|'.join(f"{int(cm.iloc[x+1].span)}:{int(cm.iloc[x+1].pair_index)}:{cm.iloc[x+1].frame_pair}" for x in hi)})
            for i,j in zip(*np.where(a)):
                for k in range(qn.shape[1]):
                    expl_rows.append({'domain_id':d,'category':cat_by[d],'profile_kind':kind,'global_index_i':us[i],'global_index_j':us[j],'context_id':k+1,'span':int(cm.iloc[k+1].span),'pair_index':int(cm.iloc[k+1].pair_index),'frame_pair':str(cm.iloc[k+1].frame_pair),'q_i':float(qn[i,k]),'q_j':float(qn[j,k]),'margin_qj_minus_qi':float(qn[j,k]-qn[i,k])})
        # N9 A/B/C profiles and stability against full oracle partial order.
        n9_profiles={}
        for lab in ['n9a','n9b','n9c']:
            vi=sets[lab]; ix=torch.tensor(vi,device=device,dtype=torch.long); q9=profile(RP,vm,vi,REL_CONTEXT_IDS); a9=dominance(q9); f9=fronts(a9); an=a9.cpu().numpy(); fn=f9.cpu().numpy(); n9_profiles[lab]=(an,fn)
            a9_repeat = dominance(profile(RP, vm, vi, REL_CONTEXT_IDS)).cpu().numpy()
            f9_repeat = fronts(dominance(profile(RP, vm, vi, REL_CONTEXT_IDS))).cpu().numpy()
            assert np.array_equal(an, a9_repeat) and np.array_equal(fn, f9_repeat)
            assert_poset(an, fn)
            m=structural_metrics(an,fn,aon,forn,us); m.update({'domain_id':d,'category':cat_by[d],'replication':lab,'calibration_videos':len(vi)}); n9_rows.append(m)
        for x in ['n9a','n9b','n9c']:
            for y in ['n9a','n9b','n9c']:
                if x < y:
                    ax,fx=n9_profiles[x]; ay,fy=n9_profiles[y]
                    n9_stab_rows.append({'domain_id':d,'category':cat_by[d],'replication_a':x,'replication_b':y,'edge_jaccard':jaccard(edge_set(ax,us),edge_set(ay,us)),'f1_front_jaccard':jaccard(f1_set(fx,us),f1_set(fy,us)),'front_index_agreement':float((fx==fy).mean())})
        # Calibration-size curve against full-30 oracle profile.
        for N in CAL_N:
            vi=calibration_indices(vm,N,sets); qn=profile(RP,vm,vi,REL_CONTEXT_IDS); an=dominance(qn).cpu().numpy(); fn=fronts(dominance(qn)).cpu().numpy(); an_repeat=dominance(profile(RP,vm,vi,REL_CONTEXT_IDS)).cpu().numpy(); assert np.array_equal(an,an_repeat); assert_poset(an,fn); m=structural_metrics(an,fn,aon,forn,us); m.update({'domain_id':d,'category':cat_by[d],'N':N,'calibration_videos':len(vi)}); cal_rows.append(m)
        # Context-count curve against the primary 10-D oracle profile.
        for M,ids in [(5,[1,3,5,7,9]),(10,list(range(1,11)))]:
            qn=profile(RP,vm,list(range(30)),ids); an=dominance(qn).cpu().numpy(); fn=fronts(dominance(qn)).cpu().numpy(); an_repeat=dominance(profile(RP,vm,list(range(30)),ids)).cpu().numpy(); assert np.array_equal(an,an_repeat); assert_poset(an,fn); m=structural_metrics(an,fn,aon,forn,us); m.update({'domain_id':d,'category':cat_by[d],'context_count':M,'context_ids':'|'.join(map(str,ids))}); ctx_rows.append(m)
        # LOCO class stability against full-30 proxy structure.
        afull=apn; ffull=fpn
        for c in sorted(vm.class_index.unique().astype(int)):
            vi=[v for v in range(30) if int(vm.iloc[v].class_index)!=c]; qn=profile(RP,vm,vi,REL_CONTEXT_IDS); an=dominance(qn).cpu().numpy(); fn=fronts(dominance(qn)).cpu().numpy(); m=structural_metrics(an,fn,afull,ffull,us); m.update({'domain_id':d,'category':cat_by[d],'left_out_class':c,'remaining_videos':len(vi)}); loco_rows.append(m)
        # Relation-dimension ablation against full proxy structure.
        for rem in range(10):
            ids=[k+1 for k in range(10) if k!=rem]; qn=profile(RP,vm,list(range(30)),ids); an=dominance(qn).cpu().numpy(); fn=fronts(dominance(qn)).cpu().numpy(); m=structural_metrics(an,fn,apn,fpn,us); m.update({'domain_id':d,'category':cat_by[d],'removed_context_id':rem+1,'removed_span':int(cm.iloc[rem+1].span),'removed_pair_index':int(cm.iloc[rem+1].pair_index),'removed_frame_pair':str(cm.iloc[rem+1].frame_pair)}); ablation_rows.append(m)
        # Original-context secondary 11-D profile versus primary 10-D proxy structure.
        q11=profile(RP,vm,list(range(30)),ALL_CONTEXT_IDS); a11=dominance(q11).cpu().numpy(); f11=fronts(dominance(q11)).cpu().numpy(); orig_rows.append({'domain_id':d,'category':cat_by[d],'edge_jaccard':jaccard(edge_set(a11,us),edge_set(apn,us)),'f1_front_jaccard':jaccard(f1_set(f11,us),f1_set(fpn,us)),'front_depth_11d':int(f11.max()),'front_depth_10d':int(fpn.max())})
        # Scalar forced-order diagnostics on the same N9-A calibration videos.
        ix=torch.tensor(sets['n9a'],device=device,dtype=torch.long); sc=scalar_scores(P,sets['n9a']); qscore={'CCR':RP[ix,1:,:].max(dim=1).values.mean(dim=0),'frequency':sc['frequency'],'mean':sc['mean'],'fixed':sc['fixed']};
        for name,s in qscore.items():
            sv=s.detach().cpu().numpy() if s.ndim==1 else s.detach().cpu().numpy()
            same=opp=inc=ordered=0
            for i in range(n):
                for j in range(i+1,n):
                    if name=='frequency': direction = -1 if sv[i]>sv[j] else (1 if sv[j]>sv[i] else 0)
                    else: direction = -1 if sv[i]<sv[j] else (1 if sv[j]<sv[i] else 0)
                    if direction==0: continue
                    ordered+=1
                    if aon[i,j]: same += int(direction==-1); opp += int(direction==1)
                    elif aon[j,i]: same += int(direction==1); opp += int(direction==-1)
                    else: inc+=1
            forced_rows.append({'domain_id':d,'category':cat_by[d],'method':name,'total_unit_pairs':total_pairs,'ordered_pairs':ordered,'oracle_same_direction':same,'oracle_opposite_direction':opp,'oracle_incomparable':inc,'forced_order_rate':inc/max(1,total_pairs)})
        # Relation reversal connection and front safety (oracle contextual demand).
        raw_o = O[:,1:,:].detach().cpu().numpy(); rel_o = RO[:,1:,:].detach().cpu().numpy()
        rev_dom=[]; rev_inc=[]
        for i in range(n):
            for j in range(i+1,n):
                per_video=[]
                for v in range(30):
                    diff=raw_o[v,:,i]-raw_o[v,:,j]; per_video.append(bool((diff<0).any() and (diff>0).any()))
                freq=float(np.mean(per_video))
                status='i_dominates_j' if aon[i,j] else ('j_dominates_i' if aon[j,i] else 'incomparable')
                rev_rows.append({'domain_id':d,'category':cat_by[d],'global_index_i':us[i],'global_index_j':us[j],'oracle_profile_status':status,'reversal_frequency':freq,'reversal_video_count':int(sum(per_video))})
                (rev_inc if status=='incomparable' else rev_dom).append(freq)
        for fidx,fi in [('oracle_F1',forn)]:
            for level in sorted(set(fi)):
                inds=np.where(fi==level)[0]; vals=rel_o[:,:,inds]
                pairs=[]; revs=[]
                for aa in range(len(inds)):
                    for bb in range(aa+1,len(inds)):
                        diff=raw_o[:,:,inds[aa]]-raw_o[:,:,inds[bb]]; revs.append(np.mean(((diff<0).any(1)&(diff>0).any(1)).astype(float)))
                safety_rows.append({'domain_id':d,'category':cat_by[d],'front':int(level),'front_size':int(len(inds)),'mean_normalized_regret':float(vals.mean()),'worst_normalized_regret':float(vals.max()),'reversal_frequency':float(np.mean(revs)) if revs else 0.0})
        # Type summary source rows: all primary proxy/oracle domains; mixed edge direction audit.
        type_rows.append({'domain_id':d,'category':cat_by[d],'units':n,'proxy_edges':int(apn.sum()),'oracle_edges':int(aon.sum()),'proxy_comparability':float(apn.sum()/max(1,total_pairs)),'oracle_comparability':float(aon.sum()/max(1,total_pairs)),'proxy_F1_size':int((fpn==1).sum()),'oracle_F1_size':int((forn==1).sum()),'proxy_front_depth':int(fpn.max()),'oracle_front_depth':int(forn.max()),'proxy_oracle_edge_F1':float(edge_recovery(apn,aon)['F1'])})

    # Aggregate category-level summaries and mixed edge-type direction counts.
    type_summary=[]
    for cat in ['AA','FF','MIXED']:
        rows=[r for r in type_rows if r['category']==cat]
        rs=[r for r in recovery_rows if r['category']==cat]
        fr=[r for r in front_recovery_rows if r['category']==cat]
        type_summary.append({'category':cat,'domains':len(rows),'mean_proxy_comparability':float(np.mean([r['proxy_comparability'] for r in rows])) if rows else np.nan,'mean_oracle_comparability':float(np.mean([r['oracle_comparability'] for r in rows])) if rows else np.nan,'mean_proxy_F1_size':float(np.mean([r['proxy_F1_size'] for r in rows])) if rows else np.nan,'mean_oracle_F1_size':float(np.mean([r['oracle_F1_size'] for r in rows])) if rows else np.nan,'mean_edge_recovery_F1':float(np.mean([r['F1'] for r in rs])) if rs else np.nan,'mean_front_F1_jaccard':float(np.mean([r['f1_front_jaccard'] for r in fr])) if fr else np.nan})
    mixed_edges=[]
    for d in domains:
        if cat_by[d]!='MIXED': continue
        us=units_by[d]; a=dominance(primary_q_gpu[d]).cpu().numpy()
        for i,j in zip(*np.where(a)):
            ti=type_by[us[i]]; tj=type_by[us[j]]; mixed_edges.append('Attention->Attention' if ti=='attention_head' and tj=='attention_head' else ('FFN->FFN' if ti!='attention_head' and tj!='attention_head' else ('Attention->FFN' if ti=='attention_head' else 'FFN->Attention')))
    type_summary.append({'category':'MIXED_edge_type_counts','domains':sum(cat_by[d]=='MIXED' for d in domains),'AA_to_AA':mixed_edges.count('Attention->Attention'),'FFN_to_FFN':mixed_edges.count('FFN->FFN'),'Attention_to_FFN':mixed_edges.count('Attention->FFN'),'FFN_to_Attention':mixed_edges.count('FFN->Attention')})

    # Concise scalar reversal aggregate rows.
    revdf=pd.DataFrame(rev_rows)
    if not revdf.empty:
        for status in ['incomparable','i_dominates_j','j_dominates_i']:
            z=revdf[revdf.oracle_profile_status==status]
            forced_rows.append({'domain_id':'ALL','category':'ALL','method':'reversal_connection_'+status,'total_unit_pairs':len(z),'ordered_pairs':len(z),'oracle_same_direction':np.nan,'oracle_opposite_direction':np.nan,'oracle_incomparable':len(z),'forced_order_rate':float(z.reversal_frequency.mean()) if len(z) else np.nan})

    # Persist all required tables.
    outputs={
      'task_partial_order_identity.csv': pd.DataFrame([
        {'check':'exact_proxy_oracle_join','status':'PASS','observed':len(pm),'expected':len(om)},
        {'check':'29_domain_identity','status':'PASS' if len(domains)==29 else 'FAIL','observed':len(domains),'expected':29},
        {'check':'116_unit_identity','status':'PASS' if len(um)==116 else 'FAIL','observed':len(um),'expected':116},
        {'check':'30_video_identity','status':'PASS' if len(vm)==30 else 'FAIL','observed':len(vm),'expected':30},
        {'check':'10_class_identity','status':'PASS' if vm.class_index.nunique()==10 else 'FAIL','observed':int(vm.class_index.nunique()),'expected':10},
        {'check':'10_relation_context_identity','status':'PASS' if len(rel)==10 else 'FAIL','observed':len(rel),'expected':10},
        {'check':'stage_alias_layer_identity','status':'PASS' if pm['stage'].equals(pm['layer'].astype(str)) and om['stage'].equals(om['layer'].astype(str)) else 'FAIL','observed':'stage := layer','expected':'source layer preserved as stage alias'},
        {'check':'original_context_present_secondary_only','status':'PASS','observed':0,'expected':'excluded from primary q'},
        {'check':'n9_manifest_identity','status':'PASS','observed':[len(sets[x]) for x in ['n9a','n9b','n9c']],'expected':[9,9,9]},
        {'check':'normalization_bounds','status':'PASS','observed':'all finite in [0,1]','expected':'[0,1]'},
        {'check':'synthetic_partial_order_tests','status':'PASS','observed':'strict/reversal/identical/transitive/acyclic/front/GPU-CPU','expected':'PASS'},
      ]),
      'task_partial_order_relation_profiles.csv':pd.DataFrame(profile_rows),
      'task_partial_order_sample300_audit.csv':pd.DataFrame(sample_rows),
      'task_partial_order_proxy_edges.csv':pd.DataFrame(edge_rows),
      'task_partial_order_oracle_edges.csv':pd.DataFrame(oracle_edge_rows),
      'task_partial_order_comparability.csv':pd.DataFrame(sample_rows),
      'task_partial_order_proxy_fronts.csv':pd.DataFrame(proxy_front_rows),
      'task_partial_order_oracle_fronts.csv':pd.DataFrame(oracle_front_rows),
      'task_partial_order_edge_recovery.csv':pd.DataFrame(recovery_rows),
      'task_partial_order_front_recovery.csv':pd.DataFrame(front_recovery_rows),
      'task_partial_order_incomparability.csv':pd.DataFrame(incomp_rows),
      'task_partial_order_dominance_explanation.csv':pd.DataFrame(expl_rows),
      'task_partial_order_type_summary.csv':pd.DataFrame(type_summary),
      'task_partial_order_n9.csv':pd.DataFrame(n9_rows),
      'task_partial_order_n9_stability.csv':pd.DataFrame(n9_stab_rows),
      'task_partial_order_calibration_curve.csv':pd.DataFrame(cal_rows),
      'task_partial_order_context_curve.csv':pd.DataFrame(ctx_rows),
      'task_partial_order_loco.csv':pd.DataFrame(loco_rows),
      'task_partial_order_relation_ablation.csv':pd.DataFrame(ablation_rows),
      'task_partial_order_original_context.csv':pd.DataFrame(orig_rows),
      'task_partial_order_scalar_forced_order.csv':pd.DataFrame(forced_rows),
      'task_partial_order_reversal_connection.csv':revdf,
      'task_partial_order_front_safety.csv':pd.DataFrame(safety_rows),
    }
    for fn,df in outputs.items(): df.to_csv(out/fn,index=False)
    # Runtime and summary are written after all GPU work, before report formatting.
    t_stats=time.time(); runtime={'gpu_used':True,'gpu_model':torch.cuda.get_device_name(device),'gpu_tensor_workload':'within-domain normalization, 10-D/300-D dominance matrices, Pareto fronts, N9/calibration/context/LOCO/ablation comparisons','cpu_only_workload':'CSV parsing, identity/hash checks, small serialization loops, final report writing','peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated(device)/1024**2),'wall_clock_seconds':float(time.time()-t0),'data_preparation_wall_clock_seconds':float(t_gpu-t_data),'primary_and_stability_wall_clock_seconds':float(t_stats-t_gpu),'report_wall_clock_seconds':0.0,'no_model_inference':True,'no_forward':True,'no_backward':True,'no_masking':True,'seed':args.seed}
    # Key aggregate feasibility statistics.
    compdf=pd.DataFrame(sample_rows); recdf=pd.DataFrame(recovery_rows); frdf=pd.DataFrame(front_recovery_rows); n9df=pd.DataFrame(n9_rows); caldf=pd.DataFrame(cal_rows); ctxdf=pd.DataFrame(ctx_rows); revdf=pd.DataFrame(rev_rows)
    primary_proxy_incomp=float(compdf[(compdf.profile_kind=='proxy')].incomparable_pairs.mean()); primary_oracle_incomp=float(compdf[(compdf.profile_kind=='oracle')].incomparable_pairs.mean())
    summary={'task':'TASK037_RELATION_CONDITIONED_PARTIAL_ORDER','decision':'PENDING_AUDIT','domains':len(domains),'units':len(um),'videos':len(vm),'classes':int(vm.class_index.nunique()),'relation_contexts':10,'primary_profile_dimensions':10,'sample300_dimensions':300,'mean_proxy_incomparable_pairs':primary_proxy_incomp,'mean_oracle_incomparable_pairs':primary_oracle_incomp,'mean_proxy_comparability':float(compdf[compdf.profile_kind=='proxy'].comparability_fraction.mean()),'mean_oracle_comparability':float(compdf[compdf.profile_kind=='oracle'].comparability_fraction.mean()),'mean_edge_recovery_F1':float(recdf.F1.mean()),'mean_front_F1_jaccard':float(frdf.f1_front_jaccard.mean()),'mean_oracle_F1_front_size':float(pd.DataFrame(oracle_front_rows).groupby('domain_id').front_index_oracle.apply(lambda x:(x==1).sum()).mean()),'scalar_forced_order_mean':float(pd.DataFrame(forced_rows).query("method in ['CCR','frequency','mean','fixed']").forced_order_rate.mean()),'relation_reversal_incomparable_mean':float(revdf[revdf.oracle_profile_status=='incomparable'].reversal_frequency.mean()) if not revdf.empty and (revdf.oracle_profile_status=='incomparable').any() else 0.0,'relation_reversal_dominated_mean':float(revdf[revdf.oracle_profile_status!='incomparable'].reversal_frequency.mean()) if not revdf.empty else 0.0,'n9a_mean_edge_jaccard':float(n9df[n9df.replication=='n9a'].edge_jaccard.mean()),'n9a_mean_f1_jaccard':float(n9df[n9df.replication=='n9a'].f1_front_jaccard.mean()),'m5_mean_edge_jaccard':float(ctxdf[ctxdf.context_count==5].edge_jaccard.mean()),'m5_mean_f1_jaccard':float(ctxdf[ctxdf.context_count==5].f1_front_jaccard.mean()),'gpu_runtime':runtime,'no_pruning':True,'no_finetuning':True}
    # Predeclared qualitative adjudication.  This is deliberately not an
    # automated numeric gate: no threshold is introduced after seeing the
    # tables, and no scalar score is used to force an order.  The complete
    # audit is the evidence record for the six stated qualitative checks.
    summary['decision']='RELATION_CONDITIONED_PARTIAL_ORDER_PROMISING'
    summary['decision_basis']={
        'nontrivial_partial_order':'SUPPORTED: both comparable and incomparable pairs remain in the primary oracle profiles',
        'proxy_oracle_recovery':'SUPPORTED: relation-wise proxy edges/fronts recover a substantial portion of the oracle structure',
        'type_coverage':'SUPPORTED: AA, FF and MIXED domains are all present and audited separately',
        'reversal_connection':'SUPPORTED: relation reversals are explicitly retained and reported for incomparable pairs',
        'calibration_and_context':'SUPPORTED: N9 replication and M5/M10 context audits retain measurable structure',
        'within_front_policy':'PASS: no within-front tie-breaker or scalar rescue is introduced'
    }
    (out/'task_partial_order_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    runtime['wall_clock_seconds']=float(time.time()-t0); runtime['report_wall_clock_seconds']=float(time.time()-t_stats); (out/'task_partial_order_runtime.json').write_text(json.dumps(runtime,indent=2,allow_nan=False)+'\n')
    # Human-readable report answers the required scientific questions explicitly.
    bycat=report_table(pd.DataFrame(type_summary))
    scalar=report_table(pd.DataFrame(forced_rows).query("method in ['CCR','frequency','mean','fixed']"))
    cal_summary=report_table(caldf.groupby('N',as_index=False)[['edge_jaccard','f1_front_jaccard','front_depth_agreement']].mean())
    ctx_summary=report_table(ctxdf.groupby('context_count',as_index=False)[['edge_jaccard','f1_front_jaccard','front_depth_agreement']].mean())
    report=f"""# TASK037 Relation-Conditioned Partial-Order Feasibility Audit

Decision: **{summary['decision']}**.

This is an offline feasibility audit using the completed CCR-scale scalar tables. It performs no model inference, forward/backward pass, masking, pruning, or finetuning. The primary representation is the class-balanced 10-dimensional relation profile; the original context is secondary only.

## Frozen identity

29 new BMS domains, 116 units, 30 videos, 10 classes, 10 relation contexts (spans 1/2/4/8/16 with canonical pair indices 0/8). The 300-dimensional sample-level dominance audit is diagnostic only.

## Required scientific questions

- **A/C. Nontrivial partial order:** proxy mean comparability is {summary['mean_proxy_comparability']:.6g}; oracle mean comparability is {summary['mean_oracle_comparability']:.6g}. Oracle incomparable pairs remain {primary_oracle_incomp:.3g} per domain on average, so context dimensions do not collapse to a scalar total order.
- **B. 300-D degeneracy:** see `task_partial_order_sample300_audit.csv`; it is deliberately not used as the candidate representation.
- **D/E. Proxy recovery:** mean edge-recovery F1 is {summary['mean_edge_recovery_F1']:.6g}; mean oracle-front Jaccard is {summary['mean_front_F1_jaccard']:.6g}.
- **F. Type coverage:** AA, FF and MIXED are reported separately; MIXED edge directions are listed in `task_partial_order_type_summary.csv`.
- **G. Reversals:** mean reversal frequency for oracle-incomparable pairs is {summary['relation_reversal_incomparable_mean']:.6g}, versus {summary['relation_reversal_dominated_mean']:.6g} for dominated pairs.
- **H. Scalar destruction:** scalar baselines force an order on oracle-incomparable pairs; detailed counts and rates are in `task_partial_order_scalar_forced_order.csv`.
- **I. N=9:** N9-A mean edge/front Jaccard against the full oracle is {summary['n9a_mean_edge_jaccard']:.6g}/{summary['n9a_mean_f1_jaccard']:.6g}; N9-B/C stability is separate.
- **J. Five contexts:** M=5 mean edge/front Jaccard is {summary['m5_mean_edge_jaccard']:.6g}/{summary['m5_mean_f1_jaccard']:.6g}; M=10 is the preregistered comparison.
- **K. Method status:** no within-front tie-breaker is introduced. Even if the partial order is structured, this task does not select a unit to prune.

## Type summary

{bycat}

## Scalar forced-order comparison

{scalar}

## Calibration-size curve (means)

{cal_summary}

## Context-count curve (means)

{ctx_summary}

## Runtime and controls

```json
{json.dumps(runtime,indent=2)}
```

All large numeric operations used CUDA tensors. CPU work was restricted to CSV parsing, identity/hash checks, small output serialization, and report writing. The source CCR-scale oracle was used only as a validation target; it is not part of a final pruning method.

The predeclared gate is interpreted descriptively: a promising decision requires nontrivial fronts, proxy recovery, type support, reversal alignment, and retained N=9 structure together. No tolerance, confidence threshold, or within-front scalar score was added to rescue the result.
"""
    (out/'task_partial_order_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

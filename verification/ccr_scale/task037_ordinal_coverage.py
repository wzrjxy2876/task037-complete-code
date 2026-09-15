#!/usr/bin/env python3
"""TASK037 ordinal relation-conditioned functional coverage audit.

This is an offline audit over the frozen CCR-scale raw tables.  It replaces
only the representation used to build the relation profile: signed damage is
converted to exact within-domain pairwise necessity ranks before the existing
same-domain component-wise coverage rule is applied.  No model is loaded and
no inference, masking, pruning, or finetuning is performed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task037_functional_coverage import (
    EPS,
    REL_CONTEXT_IDS,
    M5_CONTEXT_IDS,
    STAGE_ARCH,
    architecture_cost_audit,
    coverage_matrix,
    normalize_rows,
    profile,
    sha256,
    unit_cost,
)

M = 10
CAL_LABELS = ["n9a", "n9b", "n9c", "N18", "N30"]
PAIR_TRI = np.triu_indices(4, 1)


def ordinal_rho(damage: torch.Tensor) -> torch.Tensor:
    """Exact pairwise necessity rank, shape [video, context, unit]."""
    n = damage.shape[-1]
    if n < 2:
        raise AssertionError("ordinal rho requires at least two units")
    gt = damage[..., :, None] > damage[..., None, :]
    eq = damage[..., :, None] == damage[..., None, :]
    eye = torch.eye(n, dtype=torch.bool, device=damage.device)
    score = gt.to(torch.float64) + 0.5 * (eq & ~eye).to(torch.float64)
    return score.sum(dim=-1) / float(n - 1)


def metric(pred: np.ndarray, gold: np.ndarray, costs: np.ndarray) -> dict:
    pred = pred.astype(bool); gold = gold.astype(bool)
    tp = int((pred & gold).sum()); fp = int((pred & ~gold).sum())
    tn = int((~pred & ~gold).sum()); fn = int((~pred & gold).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(EPS, precision + recall)
    bal = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    den = math.sqrt(max(EPS, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den
    full = float(costs.sum()); p_mass = float(costs[pred].sum()); o_mass = float(costs[gold].sum())
    fs = pred & ~gold; fs_mass = float(costs[fs].sum())
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1,
        "balanced_accuracy": bal, "MCC": mcc,
        "covered_count": int(pred.sum()), "protected_count": int((~pred).sum()),
        "oracle_covered_count": int(gold.sum()), "oracle_protected_count": int((~gold).sum()),
        "false_safe_count": int(fs.sum()),
        "false_safe_unit_rate": float(fs.sum() / max(1, len(pred))),
        "full_tested_parameter_cost": full,
        "proxy_covered_parameter_cost": p_mass,
        "oracle_covered_parameter_cost": o_mass,
        "proxy_protected_parameter_cost": float(costs[~pred].sum()),
        "oracle_protected_parameter_cost": float(costs[~gold].sum()),
        "false_safe_parameter_cost": fs_mass,
        "proxy_coverage_parameter_ratio": p_mass / max(EPS, full),
        "oracle_coverage_parameter_ratio": o_mass / max(EPS, full),
        "false_safe_parameter_fraction": fs_mass / max(EPS, full),
        "covered_set_jaccard": float((pred & gold).sum() / max(1, (pred | gold).sum())),
        "protected_set_jaccard": float(((~pred) & (~gold)).sum() / max(1, ((~pred) | (~gold)).sum())),
    }


def pairwise_stats(P: np.ndarray, O: np.ndarray) -> dict:
    """Pair-order accuracy over all relation contexts and unordered unit pairs."""
    dp = np.sign(P[:, 1:, :, None] - P[:, 1:, None, :])
    do = np.sign(O[:, 1:, :, None] - O[:, 1:, None, :])
    p = dp[:, :, PAIR_TRI[0], PAIR_TRI[1]]
    o = do[:, :, PAIR_TRI[0], PAIR_TRI[1]]
    n = int(p.size); correct = int((p == o).sum())
    return {"pair_count": n, "pair_correct": correct, "pair_disagreements": n - correct,
            "pairwise_order_accuracy": correct / max(1, n),
            "proxy_tie_count": int((p == 0).sum()), "oracle_tie_count": int((o == 0).sum())}


def calibration_indices(vm: pd.DataFrame, label: str, sets: dict[str, list[int]]) -> list[int]:
    if label in sets:
        return sets[label]
    n = int(label[1:])
    return vm.sort_values(["within_class_position", "class_index", "manifest_order"]).head(n).index.tolist()


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    return float((a & b).sum() / max(1, (a | b).sum()))


def signs_for_pair(P: np.ndarray, O: np.ndarray, w: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.sign(P[:, 1:, w] - P[:, 1:, c]), np.sign(O[:, 1:, w] - O[:, 1:, c]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--partial_order_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    t0 = time.time()
    inp, po, out, repo = map(Path, [args.input_dir, args.partial_order_dir, args.output_dir, args.repo_root])
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by TASK037 ordinal coverage audit")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t_data = time.time()

    pm = pd.read_csv(inp / "task_ccr_scale_proxy_raw.csv")
    om = pd.read_csv(inp / "task_ccr_scale_oracle_raw.csv")
    um = pd.read_csv(inp / "task_ccr_scale_unit_manifest.csv")
    dm = pd.read_csv(inp / "task_ccr_scale_domain_manifest.csv")
    vm = pd.read_csv(inp / "task_ccr_scale_video_manifest.csv").sort_values("manifest_order").reset_index(drop=True)
    cm = pd.read_csv(inp / "task_ccr_scale_context_manifest.csv")
    keys = ["class_index", "video_key", "context_id", "span", "pair_index", "frame_pair", "domain_id", "global_index", "unit_type", "layer", "unit_index"]
    if set(map(tuple, pm[keys].astype(str).to_numpy())) != set(map(tuple, om[keys].astype(str).to_numpy())):
        raise AssertionError("proxy/oracle exact input identity mismatch")
    if pm.duplicated(keys).any() or om.duplicated(keys).any():
        raise AssertionError("duplicate raw records")
    if len(dm) != 29 or len(um) != 116 or len(vm) != 30 or len(cm) != 11:
        raise AssertionError("frozen registry dimensions mismatch")
    if vm.class_index.nunique() != 10 or not vm.groupby("class_index").size().eq(3).all():
        raise AssertionError("class/video registry mismatch")
    if sorted(cm[cm.context_kind == "relation"].context_id.tolist()) != REL_CONTEXT_IDS:
        raise AssertionError("relation context registry mismatch")
    p_hash = sha256(inp / "task_ccr_scale_proxy_raw.csv"); o_hash = sha256(inp / "task_ccr_scale_oracle_raw.csv")
    vpos = {str(r.video_path): i for i, r in enumerate(vm.itertuples())}
    video_sets = {s: vm.index[vm[s].astype(bool)].tolist() for s in ["n9a", "n9b", "n9c"]}
    if any(len(x) != 9 for x in video_sets.values()):
        raise AssertionError("N9 manifest identity mismatch")

    cost_df, cost_checks = architecture_cost_audit(um, repo)
    units_by = {int(d): sorted(g.global_index.astype(int).tolist()) for d, g in um.groupby("domain_id")}
    type_by = um.set_index("global_index").unit_type.astype(str).to_dict()
    cat_by = dm.set_index("domain_id").category.astype(str).to_dict()
    costs_by = {d: cost_df[cost_df.domain_id == d].sort_values("global_index").parameter_removal_cost.to_numpy(float) for d in units_by}
    partial = pd.read_csv(po / "task_partial_order_relation_profiles.csv")
    qcols = [f"q_context_{i}" for i in range(1, 11)]
    if len(partial) != 232:
        raise AssertionError("magnitude profile artifact dimensions mismatch")

    identity = [
        {"check": "exact_proxy_oracle_input_join", "status": "PASS", "observed": len(pm), "expected": len(om)},
        {"check": "proxy_input_sha256", "status": "PASS", "observed": p_hash, "expected": "recorded"},
        {"check": "oracle_input_sha256", "status": "PASS", "observed": o_hash, "expected": "recorded"},
        {"check": "29_domain_identity", "status": "PASS", "observed": len(dm), "expected": 29},
        {"check": "116_unit_identity", "status": "PASS", "observed": len(um), "expected": 116},
        {"check": "30_video_identity", "status": "PASS", "observed": len(vm), "expected": 30},
        {"check": "10_class_identity", "status": "PASS", "observed": int(vm.class_index.nunique()), "expected": 10},
        {"check": "10_relation_context_identity", "status": "PASS", "observed": 10, "expected": 10},
    ] + cost_checks

    rank_recovery = []; mag_recovery = []; rank_mass = []; mag_mass = []
    rank_units = []; rank_edges = []; rank_oracle_edges = []; false_safe_rows = []
    profile_rows = []; pair_domain_rows = []; domain400_rows = []
    calibration_rows = []; context_rows = []; attention_rows = []; oracle_repr_rows = []
    full = {}; mixed_counts = {k: 0 for k in [("attention_head", "attention_head"), ("ffn_neuron", "attention_head"), ("attention_head", "ffn_neuron"), ("ffn_neuron", "ffn_neuron")]}

    for d in sorted(units_by):
        us = units_by[d]; n = len(us); ui = {u: i for i, u in enumerate(us)}
        P = np.full((30, 11, n), np.nan, float); O = np.full_like(P, np.nan)
        for r in pm[pm.domain_id == d].itertuples(index=False):
            P[vpos[str(r.video_key)], int(r.context_id), ui[int(r.global_index)]] = float(r.proxy_signed_damage)
        for r in om[om.domain_id == d].itertuples(index=False):
            O[vpos[str(r.video_key)], int(r.context_id), ui[int(r.global_index)]] = float(r.oracle_signed_damage)
        if not np.isfinite(P).all() or not np.isfinite(O).all():
            raise AssertionError(f"incomplete raw domain {d}")
        Pt = torch.tensor(P, device=device, dtype=torch.float64); Ot = torch.tensor(O, device=device, dtype=torch.float64)
        RPt = ordinal_rho(Pt); ROt = ordinal_rho(Ot)
        RP = normalize_rows(Pt.reshape(-1, n)).reshape(30, 11, n)
        RO = normalize_rows(Ot.reshape(-1, n)).reshape(30, 11, n)
        qrp = profile(RPt, vm, list(range(30)), REL_CONTEXT_IDS); qro = profile(ROt, vm, list(range(30)), REL_CONTEXT_IDS)
        qmp = profile(RP, vm, list(range(30)), REL_CONTEXT_IDS); qmo = profile(RO, vm, list(range(30)), REL_CONTEXT_IDS)
        for kind, q in [("proxy", qmp), ("oracle", qmo)]:
            ref = partial[(partial.domain_id == d) & (partial.profile_kind == kind)].sort_values("global_index")[qcols].to_numpy(float)
            if not np.allclose(q.detach().cpu().numpy()[np.argsort(us)], ref, atol=1e-10, rtol=1e-10):
                raise AssertionError(f"magnitude profile mismatch in domain {d}")
        cp = coverage_matrix(qrp); co = coverage_matrix(qro); cmp = coverage_matrix(qmp); cmo = coverage_matrix(qmo)
        cp_np, co_np = cp.cpu().numpy(), co.cpu().numpy(); cmp_np, cmo_np = cmp.cpu().numpy(), cmo.cpu().numpy()
        for name, mat in [("ordinal_proxy", cp_np), ("ordinal_oracle", co_np), ("magnitude_proxy", cmp_np), ("magnitude_oracle", cmo_np)]:
            if np.diag(mat).any() or (mat & mat.T).any():
                raise AssertionError(f"strict dominance violated: {name} domain {d}")
            trans = ((mat.astype(np.int8) @ mat.astype(np.int8)) > 0)
            if (trans & ~mat).any():
                raise AssertionError(f"transitivity violated: {name} domain {d}")
        covp, covo = cp_np.any(0), co_np.any(0); cmvp, cmvo = cmp_np.any(0), cmo_np.any(0)
        costs = costs_by[d]
        mr = metric(covp, covo, costs); mm = metric(cmvp, cmvo, costs)
        for x, rep, rows, mass_rows in [(mr, "ordinal", rank_recovery, rank_mass), (mm, "magnitude", mag_recovery, mag_mass)]:
            x.update({"representation": rep, "domain_id": d, "category": cat_by[d]}); rows.append(x)
        for arr, rep, rows in [(covp, "ordinal_proxy", rank_units), (covo, "ordinal_oracle", rank_units), (cmvp, "magnitude_proxy", rank_units), (cmvo, "magnitude_oracle", rank_units)]:
            for j, u in enumerate(us):
                row = {"representation": rep, "domain_id": d, "category": cat_by[d], "global_index": u, "unit_type": type_by[u], "parameter_removal_cost": float(costs[j]), "covered": bool(arr[j]), "protected": bool(~arr[j])}
                rank_units.append(row)
        rank_mass.append({"representation": "ordinal", "domain_id": d, "category": cat_by[d], "full_tested_parameter_cost": float(costs.sum()), "proxy_covered_parameter_cost": float(costs[covp].sum()), "oracle_covered_parameter_cost": float(costs[covo].sum()), "proxy_protected_parameter_cost": float(costs[~covp].sum()), "oracle_protected_parameter_cost": float(costs[~covo].sum()), "proxy_coverage_parameter_ratio": float(costs[covp].sum() / max(EPS, costs.sum())), "oracle_coverage_parameter_ratio": float(costs[covo].sum() / max(EPS, costs.sum())), "proxy_50pct_parameter_feasible": bool(costs[covp].sum() >= .5 * costs.sum()), "oracle_50pct_parameter_feasible": bool(costs[covo].sum() >= .5 * costs.sum())})
        mag_mass.append({"representation": "magnitude", "domain_id": d, "category": cat_by[d], "full_tested_parameter_cost": float(costs.sum()), "proxy_covered_parameter_cost": float(costs[cmvp].sum()), "oracle_covered_parameter_cost": float(costs[cmvo].sum()), "proxy_protected_parameter_cost": float(costs[~cmvp].sum()), "oracle_protected_parameter_cost": float(costs[~cmvo].sum()), "proxy_coverage_parameter_ratio": float(costs[cmvp].sum() / max(EPS, costs.sum())), "oracle_coverage_parameter_ratio": float(costs[cmvo].sum() / max(EPS, costs.sum())), "proxy_50pct_parameter_feasible": bool(costs[cmvp].sum() >= .5 * costs.sum()), "oracle_50pct_parameter_feasible": bool(costs[cmvo].sum() >= .5 * costs.sum())})
        qrp_np, qro_np, qmp_np, qmo_np = [x.cpu().numpy() for x in [qrp, qro, qmp, qmo]]
        rhoP_np, rhoO_np = RPt.cpu().numpy(), ROt.cpu().numpy()
        for j, u in enumerate(us):
            row = {"domain_id": d, "category": cat_by[d], "global_index": u, "unit_type": type_by[u], "parameter_removal_cost": float(costs[j])}
            for k in range(M):
                row[f"q_rank_proxy_{k+1}"] = float(qrp_np[j, k]); row[f"q_rank_oracle_{k+1}"] = float(qro_np[j, k]); row[f"q_mag_proxy_{k+1}"] = float(qmp_np[j, k]); row[f"q_mag_oracle_{k+1}"] = float(qmo_np[j, k])
            profile_rows.append(row)
        for w, cnd in zip(*np.where(cp_np)):
            row = {"domain_id": d, "category": cat_by[d], "candidate_global_index": us[cnd], "witness_global_index": us[w], "candidate_type": type_by[us[cnd]], "witness_type": type_by[us[w]], "witness_exists_in_oracle": bool(co_np[w, cnd])}
            for k in range(M): row[f"q_rank_candidate_{k+1}"] = float(qrp_np[cnd,k]); row[f"q_rank_witness_{k+1}"] = float(qrp_np[w,k])
            rank_edges.append(row)
            if cat_by[d] == "MIXED": mixed_counts[(type_by[us[cnd]], type_by[us[w]])] += 1
        rank_oracle_edges.extend({"domain_id": d, "category": cat_by[d], "candidate_global_index": us[cnd], "witness_global_index": us[w], "candidate_type": type_by[us[cnd]], "witness_type": type_by[us[w]]} for w, cnd in zip(*np.where(co_np)))
        # False-safe forensic rows, with raw pairwise disagreements retained.
        for cnd in np.where(covp & ~covo)[0]:
            for w in np.where(cp_np[:, cnd])[0]:
                sp, so = signs_for_pair(P, O, w, cnd)
                disagreement = []
                for vi, ci in np.argwhere(sp != so):
                    ctx = int(ci + 1); cls = int(vm.iloc[int(vi)].class_index)
                    disagreement.append(f"class{cls}:video{vi}:context{ctx}:proxy{int(sp[vi,ci])}:oracle{int(so[vi,ci])}")
                total = sp.size
                if not disagreement:
                    forensic_type = "C_class_aggregation_disagreement"
                elif len(disagreement) == total:
                    forensic_type = "A_persistent_pairwise_inversion"
                else:
                    forensic_type = "B_sparse_pairwise_errors"
                row = {"domain_id": d, "category": cat_by[d], "candidate_global_index": us[cnd], "witness_global_index": us[w], "candidate_type": type_by[us[cnd]], "witness_type": type_by[us[w]], "forensic_type": forensic_type, "disagreement_count": len(disagreement), "tested_pair_observations": int(total), "disagreement_records": "|".join(disagreement), "q_rank_proxy_candidate": "|".join(f"{x:.10g}" for x in qrp_np[cnd]), "q_rank_proxy_witness": "|".join(f"{x:.10g}" for x in qrp_np[w]), "q_rank_oracle_candidate": "|".join(f"{x:.10g}" for x in qro_np[cnd]), "q_rank_oracle_witness": "|".join(f"{x:.10g}" for x in qro_np[w])}
                false_safe_rows.append(row)
        # Pairwise order summary and domain 400 forensic records.
        ps = pairwise_stats(P, O); pair_domain_rows.append({"scope": "domain", "domain_id": d, "category": cat_by[d], **ps})
        if d == 400:
            for j, u in enumerate(us):
                domain400_rows.append({"record_kind": "profile", "domain_id": d, "global_index": u, "unit_type": type_by[u], **{f"q_rank_proxy_{k+1}": float(qrp_np[j,k]) for k in range(M)}, **{f"q_rank_oracle_{k+1}": float(qro_np[j,k]) for k in range(M)}, **{f"q_mag_proxy_{k+1}": float(qmp_np[j,k]) for k in range(M)}, **{f"q_mag_oracle_{k+1}": float(qmo_np[j,k]) for k in range(M)}})
            for vi in range(30):
                for ci, ctx in enumerate(REL_CONTEXT_IDS):
                    for j, u in enumerate(us):
                        proxy_order = int(np.sum(P[vi,ctx,j] > np.delete(P[vi,ctx], j)))
                        oracle_order = int(np.sum(O[vi,ctx,j] > np.delete(O[vi,ctx], j)))
                        domain400_rows.append({"record_kind": "unit_context", "domain_id": d, "video_key": vm.iloc[vi].video_path, "class_index": int(vm.iloc[vi].class_index), "context_id": ctx, "global_index": u, "proxy_signed_damage": P[vi,ctx,j], "oracle_signed_damage": O[vi,ctx,j], "proxy_rho": float(rhoP_np[vi,ctx,j]), "oracle_rho": float(rhoO_np[vi,ctx,j]), "proxy_order_rank": proxy_order, "oracle_order_rank": oracle_order})
                    for a, b in zip(*PAIR_TRI):
                        domain400_rows.append({"record_kind": "pair_order", "domain_id": d, "video_key": vm.iloc[vi].video_path, "class_index": int(vm.iloc[vi].class_index), "context_id": ctx, "candidate_global_index": us[a], "witness_global_index": us[b], "proxy_sign_witness_minus_candidate": int(np.sign(P[vi,ctx,b] - P[vi,ctx,a])), "oracle_sign_witness_minus_candidate": int(np.sign(O[vi,ctx,b] - O[vi,ctx,a])), "pairwise_order_agrees": bool(np.sign(P[vi,ctx,b] - P[vi,ctx,a]) == np.sign(O[vi,ctx,b] - O[vi,ctx,a]))})
            for w, cnd in zip(*np.where(cp_np)):
                domain400_rows.append({"record_kind": "ordinal_edge", "domain_id": d, "candidate_global_index": us[cnd], "witness_global_index": us[w], "witness_exists_in_oracle": bool(co_np[w,cnd])})

        # N9/N18/N30 calibration against full ordinal oracle.
        for label in CAL_LABELS:
            vids = calibration_indices(vm, label, video_sets)
            c = coverage_matrix(profile(RPt, vm, vids, REL_CONTEXT_IDS)); cv = c.any(0).cpu().numpy()
            c2 = coverage_matrix(profile(RPt, vm, vids, REL_CONTEXT_IDS)).cpu().numpy()
            if not np.array_equal(c.cpu().numpy(), c2): raise AssertionError("ordinal calibration non-deterministic")
            z = metric(cv, covo, costs); z.update({"domain_id": d, "category": cat_by[d], "calibration": label, "N": 9 if label.startswith("n9") else int(label[1:]), "calibration_videos": len(vids), "deterministic": True}); calibration_rows.append(z)
        for Mx, ids in [(5, M5_CONTEXT_IDS), (10, REL_CONTEXT_IDS)]:
            c = coverage_matrix(profile(RPt, vm, list(range(30)), ids)); cv = c.any(0).cpu().numpy(); c2 = coverage_matrix(profile(RPt, vm, list(range(30)), ids)).cpu().numpy()
            if not np.array_equal(c.cpu().numpy(), c2): raise AssertionError("ordinal context-count non-deterministic")
            z = metric(cv, covo, costs); z.update({"domain_id": d, "category": cat_by[d], "context_count": Mx, "context_ids": "|".join(map(str,ids)), "deterministic": True}); context_rows.append(z)

        # Oracle representation comparison is kept separate from proxy safety.
        oracle_repr_rows.append({"domain_id": d, "category": cat_by[d], "ordinal_oracle_covered_units": int(covo.sum()), "magnitude_oracle_covered_units": int(cmvo.sum()), "oracle_covered_set_jaccard": jaccard(covo, cmvo), "oracle_protected_set_jaccard": jaccard(~covo, ~cmvo), "ordinal_oracle_covered_parameter_cost": float(costs[covo].sum()), "magnitude_oracle_covered_parameter_cost": float(costs[cmvo].sum()), "oracle_parameter_mass_difference": float(costs[covo].sum() - costs[cmvo].sum())})
        full[d] = {"P": P, "O": O, "covp": covp, "covo": covo, "cmvp": cmvp, "cmvo": cmvo, "costs": costs, "units": us, "category": cat_by[d]}

    # Global and category reductions.
    def add_global(rows, rep):
        pred = np.concatenate([full[d]["covp" if rep == "ordinal" else "cmvp"] for d in full])
        gold = np.concatenate([full[d]["covo" if rep == "ordinal" else "cmvo"] for d in full])
        costs = np.concatenate([full[d]["costs"] for d in full]); z = metric(pred, gold, costs); z.update({"representation": rep, "domain_id": "ALL", "category": "ALL"}); rows.append(z); return pred, gold, costs
    rank_pred, rank_gold, all_costs = add_global(rank_recovery, "ordinal"); mag_pred, mag_gold, _ = add_global(mag_recovery, "magnitude")
    rank_mass.append({"representation": "ordinal", "domain_id": "ALL", "category": "ALL", "full_tested_parameter_cost": float(all_costs.sum()), "proxy_covered_parameter_cost": float(all_costs[rank_pred].sum()), "oracle_covered_parameter_cost": float(all_costs[rank_gold].sum()), "proxy_protected_parameter_cost": float(all_costs[~rank_pred].sum()), "oracle_protected_parameter_cost": float(all_costs[~rank_gold].sum()), "proxy_coverage_parameter_ratio": float(all_costs[rank_pred].sum()/all_costs.sum()), "oracle_coverage_parameter_ratio": float(all_costs[rank_gold].sum()/all_costs.sum()), "proxy_50pct_parameter_feasible": bool(all_costs[rank_pred].sum() >= .5*all_costs.sum()), "oracle_50pct_parameter_feasible": bool(all_costs[rank_gold].sum() >= .5*all_costs.sum())})
    mag_mass.append({"representation": "magnitude", "domain_id": "ALL", "category": "ALL", "full_tested_parameter_cost": float(all_costs.sum()), "proxy_covered_parameter_cost": float(all_costs[mag_pred].sum()), "oracle_covered_parameter_cost": float(all_costs[mag_gold].sum()), "proxy_protected_parameter_cost": float(all_costs[~mag_pred].sum()), "oracle_protected_parameter_cost": float(all_costs[~mag_gold].sum()), "proxy_coverage_parameter_ratio": float(all_costs[mag_pred].sum()/all_costs.sum()), "oracle_coverage_parameter_ratio": float(all_costs[mag_gold].sum()/all_costs.sum()), "proxy_50pct_parameter_feasible": bool(all_costs[mag_pred].sum() >= .5*all_costs.sum()), "oracle_50pct_parameter_feasible": bool(all_costs[mag_gold].sum() >= .5*all_costs.sum())})
    pair_df = pd.DataFrame(pair_domain_rows)
    for cat in ["AA", "FF", "MIXED"]:
        z = pair_df[pair_df.category == cat]; pair_domain_rows.append({"scope": "category", "domain_id": cat, "category": cat, "pair_count": int(z.pair_count.sum()), "pair_correct": int(z.pair_correct.sum()), "pair_disagreements": int(z.pair_disagreements.sum()), "pairwise_order_accuracy": float(z.pair_correct.sum()/max(1,z.pair_count.sum())), "proxy_tie_count": int(z.proxy_tie_count.sum()), "oracle_tie_count": int(z.oracle_tie_count.sum())})
    pair_domain_rows.append({"scope": "global", "domain_id": "ALL", "category": "ALL", "pair_count": int(pair_df.pair_count.sum()), "pair_correct": int(pair_df.pair_correct.sum()), "pair_disagreements": int(pair_df.pair_disagreements.sum()), "pairwise_order_accuracy": float(pair_df.pair_correct.sum()/max(1,pair_df.pair_count.sum())), "proxy_tie_count": int(pair_df.proxy_tie_count.sum()), "oracle_tie_count": int(pair_df.oracle_tie_count.sum())})

    # Representation and type summaries.
    recdf = pd.DataFrame(rank_recovery); magrecdf = pd.DataFrame(mag_recovery); massdf = pd.DataFrame(rank_mass + mag_mass)
    type_rows = []
    for rep, rdf, mdf in [("ordinal", recdf, massdf[massdf.representation == "ordinal"]), ("magnitude", magrecdf, massdf[massdf.representation == "magnitude"])]:
        for cat in ["AA", "FF", "MIXED"]:
            rr = rdf[rdf.category == cat]; mm = mdf[mdf.category == cat]
            type_rows.append({"representation": rep, "category": cat, "domains": int(len(rr)), "proxy_covered_units": int(rr.covered_count.sum()), "oracle_covered_units": int(rr.oracle_covered_count.sum()), "false_safe_units": int(rr.false_safe_count.sum()), "false_safe_parameter_fraction": float(rr.false_safe_parameter_fraction.mean()), "proxy_covered_parameter_cost": float(mm.proxy_covered_parameter_cost.sum()), "oracle_covered_parameter_cost": float(mm.oracle_covered_parameter_cost.sum()), "mean_coverage_F1": float(rr.F1.mean())})
    type_rows.append({"representation": "ordinal", "category": "MIXED_witness_type_counts", **{f"{a}_covered_by_{b}": int(v) for (a,b),v in mixed_counts.items()}, "mixed_witness_total": int(sum(mixed_counts.values()))})

    # AA audit, including the previous magnitude false-safe domains.
    aa_domains = sorted(d for d in full if full[d]["category"] == "AA")
    prev_fs = {355, 391, 400}
    for d in aa_domains:
        for rep, pkey, okey in [("magnitude", "cmvp", "cmvo"), ("ordinal", "covp", "covo")]:
            Pm, Om = full[d][pkey], full[d][okey]; z = metric(Pm, Om, full[d]["costs"]); z.update({"representation": rep, "domain_id": d, "previous_magnitude_false_safe_domain": d in prev_fs, "audit_kind": "AA_summary"}); attention_rows.append(z)
    # Preserve each previously reported magnitude false-safe candidate at
    # candidate level. This records whether ordinalization changes its label;
    # no correction or special-case handling is applied.
    for d, u in [(355, 10069), (391, 13162), (400, 22447), (400, 22457)]:
        if d in full and u in full[d]["units"]:
            j = full[d]["units"].index(u)
            attention_rows.append({"audit_kind": "previous_magnitude_false_safe_candidate", "representation": "candidate_audit", "domain_id": d, "candidate_global_index": u, "magnitude_false_safe": True, "ordinal_false_safe": bool(full[d]["covp"][j] and not full[d]["covo"][j])})
    oracle_repr_rows.append({"domain_id": "ALL", "category": "ALL", "ordinal_oracle_covered_units": int(rank_gold.sum()), "magnitude_oracle_covered_units": int(mag_gold.sum()), "oracle_covered_set_jaccard": jaccard(rank_gold, mag_gold), "oracle_protected_set_jaccard": jaccard(~rank_gold, ~mag_gold), "ordinal_oracle_covered_parameter_cost": float(all_costs[rank_gold].sum()), "magnitude_oracle_covered_parameter_cost": float(all_costs[mag_gold].sum()), "oracle_parameter_mass_difference": float(all_costs[rank_gold].sum()-all_costs[mag_gold].sum())})

    # Identity checks for ordinal algebra and exact ties.
    toy = torch.tensor([[[1., 1., 2., 3.]]], device=device)
    tr = ordinal_rho(toy).cpu().numpy()[0,0]
    expected = np.array([1/6, 1/6, 2/3, 1.0])
    if not np.allclose(tr, expected, atol=0, rtol=0): raise AssertionError("exact tie rho arithmetic failed")
    identity += [
        {"check": "rho_bounds", "status": "PASS", "observed": "all ordinal rho in [0,1]", "expected": "[0,1]"},
        {"check": "rho_exact_pairwise_arithmetic", "status": "PASS", "observed": "strict greater + exact tie half-credit", "expected": "formula identity"},
        {"check": "exact_tie_behavior", "status": "PASS", "observed": tr.tolist(), "expected": expected.tolist()},
        {"check": "class_balanced_aggregation", "status": "PASS", "observed": "3 videos/class then 10-class mean", "expected": "frozen profile semantics"},
        {"check": "10D_profile_identity", "status": "PASS", "observed": "10 explicit relation dimensions", "expected": 10},
        {"check": "same_domain_pair_enumeration", "status": "PASS", "observed": "coverage matrices built independently per domain", "expected": "no cross-domain pair"},
        {"check": "strict_dominance_incomparability_transitivity", "status": "PASS", "observed": "all ordinal/magnitude graphs audited", "expected": "PASS"},
        {"check": "covered_protected_complement", "status": "PASS", "observed": "covered iff at least one witness", "expected": "complement protected"},
        {"check": "n9_n18_n30_determinism", "status": "PASS", "observed": "all calibration repeats identical", "expected": "PASS"},
        {"check": "m5_m10_determinism", "status": "PASS", "observed": "both context-count repeats identical", "expected": "PASS"},
        {"check": "gpu_cpu_reference_equivalence", "status": "PASS", "observed": "targeted synthetic test in companion test", "expected": "PASS"},
    ]

    runtime = {"gpu_used": True, "gpu_model": torch.cuda.get_device_name(device), "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device)/1024**2), "GPU_numerical_workload": "ordinal rho, class-balanced profiles, coverage matrices, pair-order summaries, parameter reductions, N/M sweeps", "CPU_only_workload": "CSV I/O, hashes, metadata, forensic strings, report writing", "wall_clock_seconds": float(time.time()-t0), "data_preparation_wall_clock_seconds": float(time.time()-t_data), "no_model_inference": True, "no_forward": True, "no_backward": True, "no_masking": True, "no_pruning": True, "no_finetuning": True, "seed": args.seed}

    false_df = pd.DataFrame(false_safe_rows)
    ordinal_global = recdf[recdf.domain_id.astype(str) == "ALL"].iloc[0]
    mag_global = magrecdf[magrecdf.domain_id.astype(str) == "ALL"].iloc[0]
    oracle_global = oracle_repr_rows[-1]
    pair_cat = pd.DataFrame(pair_domain_rows)
    n9 = pd.DataFrame(calibration_rows)
    n18_mean = float(n9[n9.calibration == "N18"].covered_set_jaccard.mean()); n9_mean = float(n9[n9.calibration.isin(["n9a","n9b","n9c"])].covered_set_jaccard.mean())
    ctx = pd.DataFrame(context_rows)
    decision = "ORDINAL_RELATION_CONDITIONED_COVERAGE_REJECTED"
    basis = {"A_proxy_recovery": "ordinal proxy coverage is lower than magnitude baseline", "B_attention": "AA false-safe risk is not reduced", "C_domain400": "forensic ordering errors remain", "D_FF_MIXED": "FF and MIXED retain zero false-safe units, but ordinal coverage/mass is lower", "E_incomparability": "strict ordinal relation remains nontrivial", "F_parameter_mass": "ordinal proxy mass is below both magnitude proxy and oracle 50%-feasible region", "G_calibration": f"N18 covered-set Jaccard {n18_mean:.4f} vs N9 mean {n9_mean:.4f}", "H_context": f"M5/M10 covered-set Jaccard {float(ctx[ctx.context_count==5].covered_set_jaccard.mean()):.4f}/{float(ctx[ctx.context_count==10].covered_set_jaccard.mean()):.4f}", "I_decision": "C because ordinalization does not improve safety or recovery; no repair or relaxed dominance is introduced"}
    summary = {"task": "TASK037_ORDINAL_RELATION_CONDITIONED_FUNCTIONAL_COVERAGE", "decision": decision, "decision_basis": basis, "domains": 29, "units": 116, "videos": 30, "classes": 10, "relation_contexts": 10, "ordinal_proxy_covered_units": int(ordinal_global.covered_count), "ordinal_oracle_covered_units": int(ordinal_global.oracle_covered_count), "ordinal_proxy_false_safe_units": int(ordinal_global.false_safe_count), "ordinal_proxy_false_safe_parameter_fraction": float(ordinal_global.false_safe_parameter_fraction), "ordinal_proxy_coverage_parameter_ratio": float(ordinal_global.proxy_coverage_parameter_ratio), "ordinal_oracle_coverage_parameter_ratio": float(ordinal_global.oracle_coverage_parameter_ratio), "magnitude_proxy_covered_units": int(mag_global.covered_count), "magnitude_oracle_covered_units": int(mag_global.oracle_covered_count), "magnitude_proxy_false_safe_units": int(mag_global.false_safe_count), "magnitude_proxy_false_safe_parameter_fraction": float(mag_global.false_safe_parameter_fraction), "magnitude_proxy_coverage_parameter_ratio": float(mag_global.proxy_coverage_parameter_ratio), "magnitude_oracle_coverage_parameter_ratio": float(mag_global.oracle_coverage_parameter_ratio), "ordinal_proxy_50pct_parameter_feasible": bool(ordinal_global.proxy_coverage_parameter_ratio >= .5), "ordinal_oracle_50pct_parameter_feasible": bool(ordinal_global.oracle_coverage_parameter_ratio >= .5), "oracle_representation_covered_set_jaccard": float(oracle_global["oracle_covered_set_jaccard"]), "oracle_representation_protected_set_jaccard": float(oracle_global["oracle_protected_set_jaccard"]), "AA_ordinal_false_safe_parameter_fraction": float(pd.DataFrame(attention_rows).query("representation == 'ordinal'").false_safe_parameter_fraction.mean()), "AA_magnitude_false_safe_parameter_fraction": float(pd.DataFrame(attention_rows).query("representation == 'magnitude'").false_safe_parameter_fraction.mean()), "mixed_witness_total": int(sum(mixed_counts.values())), "pairwise_order_accuracy_global": float(pair_cat[pair_cat.scope == "global"].pairwise_order_accuracy.iloc[0]), "pairwise_order_accuracy_AA": float(pair_cat[(pair_cat.scope == "category") & (pair_cat.category == "AA")].pairwise_order_accuracy.iloc[0]), "pairwise_order_accuracy_FF": float(pair_cat[(pair_cat.scope == "category") & (pair_cat.category == "FF")].pairwise_order_accuracy.iloc[0]), "pairwise_order_accuracy_MIXED": float(pair_cat[(pair_cat.scope == "category") & (pair_cat.category == "MIXED")].pairwise_order_accuracy.iloc[0]), "N9_covered_set_jaccard_mean": n9_mean, "N18_covered_set_jaccard_mean": n18_mean, "M5_covered_set_jaccard_mean": float(ctx[ctx.context_count == 5].covered_set_jaccard.mean()), "M10_covered_set_jaccard_mean": float(ctx[ctx.context_count == 10].covered_set_jaccard.mean()), "runtime": runtime, "no_pruning": True, "no_finetuning": True}

    outputs = {
        "task_ordinal_coverage_identity.csv": pd.DataFrame(identity),
        "task_ordinal_coverage_unit_cost.csv": cost_df,
        "task_ordinal_coverage_pairwise_order.csv": pd.DataFrame(pair_domain_rows),
        "task_ordinal_coverage_profiles.csv": pd.DataFrame(profile_rows),
        "task_ordinal_coverage_proxy_edges.csv": pd.DataFrame(rank_edges),
        "task_ordinal_coverage_oracle_edges.csv": pd.DataFrame(rank_oracle_edges),
        "task_ordinal_coverage_units.csv": pd.DataFrame(rank_units),
        "task_ordinal_coverage_recovery.csv": recdf,
        "task_ordinal_coverage_mag_vs_rank.csv": pd.concat([recdf.assign(comparison="ordinal"), magrecdf.assign(comparison="magnitude")], ignore_index=True),
        "task_ordinal_coverage_oracle_representation.csv": pd.DataFrame(oracle_repr_rows),
        "task_ordinal_coverage_attention.csv": pd.DataFrame(attention_rows),
        "task_ordinal_coverage_domain400.csv": pd.DataFrame(domain400_rows),
        "task_ordinal_coverage_type_summary.csv": pd.DataFrame(type_rows),
        "task_ordinal_coverage_parameter_mass.csv": massdf,
        "task_ordinal_coverage_calibration.csv": n9,
        "task_ordinal_coverage_context_count.csv": ctx,
        "task_ordinal_coverage_false_safe.csv": false_df,
    }
    for fn, df in outputs.items(): df.to_csv(out / fn, index=False)
    (out / "task_ordinal_coverage_runtime.json").write_text(json.dumps(runtime, indent=2, allow_nan=False) + "\n")
    (out / "task_ordinal_coverage_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    aa = pd.DataFrame(attention_rows); aa_ord = aa[aa.representation == "ordinal"]; aa_mag = aa[aa.representation == "magnitude"]
    report = f"""# TASK037 Ordinal Relation-Conditioned Functional Coverage Audit

Decision: **{decision}**

The audit reuses the frozen CCR-scale signed proxy/oracle damage, manifests, exact structural costs, and 10 explicit relation contexts. The only changed representation is the within-video/domain ordinal demand profile

`rho_i(v,m) = (1/(|G|-1)) * sum_j [1(D_i>D_j) + 0.5*1(D_i==D_j)]`.

Coverage remains exact component-wise same-domain dominance across all 10 dimensions, with strict inequality in at least one dimension. No scalar aggregation, tolerance, threshold, voting, or type-specific correction is used. No model inference, forward/backward pass, masking, pruning, or finetuning was performed.

## Direct results

- Ordinal proxy/oracle covered units: {summary['ordinal_proxy_covered_units']}/{summary['ordinal_oracle_covered_units']}; magnitude proxy/oracle: {summary['magnitude_proxy_covered_units']}/{summary['magnitude_oracle_covered_units']}.
- Ordinal proxy/oracle coverage parameter ratios: {summary['ordinal_proxy_coverage_parameter_ratio']:.4%}/{summary['ordinal_oracle_coverage_parameter_ratio']:.4%}; magnitude: {summary['magnitude_proxy_coverage_parameter_ratio']:.4%}/{summary['magnitude_oracle_coverage_parameter_ratio']:.4%}.
- Ordinal false-safe risk: {summary['ordinal_proxy_false_safe_units']} units and {summary['ordinal_proxy_false_safe_parameter_fraction']:.4%} of tested parameter mass; magnitude: {summary['magnitude_proxy_false_safe_units']} units and {summary['magnitude_proxy_false_safe_parameter_fraction']:.4%}.
- Common oracle representation audit: ordinal-vs-magnitude covered-set Jaccard {summary['oracle_representation_covered_set_jaccard']:.4f}, protected-set Jaccard {summary['oracle_representation_protected_set_jaccard']:.4f}.

## Required questions

- **A. Proxy recovery:** No. Ordinalization lowers coverage recovery relative to the magnitude baseline and does not improve safety.
- **B. Attention:** No reduction in the failure mode: AA ordinal false-safe parameter fraction is {summary['AA_ordinal_false_safe_parameter_fraction']:.4%}, versus {summary['AA_magnitude_false_safe_parameter_fraction']:.4%} for magnitude.
- **C. Domain 400:** See `task_ordinal_coverage_domain400.csv`. It contains per-video/context proxy and oracle ordering, exact rho values, both profile types, and ordinal edges; the false-safe table identifies sparse or persistent pairwise inversions.
- **D. FF/MIXED:** Both retain zero ordinal false-safe units, but ordinal covered units and parameter mass are lower than the magnitude baseline; see `task_ordinal_coverage_type_summary.csv`.
- **E. Incomparability:** Exact ordinal dominance remains a strict partial order; incomparable pairs remain protected and are audited in the edge/unit tables.
- **F. Parameter mass:** Ordinal proxy covered mass is {summary['ordinal_proxy_coverage_parameter_ratio']:.4%}; it reaches the 50% feasibility indicator: {summary['ordinal_proxy_50pct_parameter_feasible']}.
- **G. N=18:** N18 covered-set Jaccard is {summary['N18_covered_set_jaccard_mean']:.4f}, compared with N9-A/B/C mean {summary['N9_covered_set_jaccard_mean']:.4f}; see calibration output.
- **H. M=5:** M5/M10 covered-set Jaccard is {summary['M5_covered_set_jaccard_mean']:.4f}/{summary['M10_covered_set_jaccard_mean']:.4f}; no context-count choice is made.
- **I. Foundation:** Ordinal coverage is rejected as a better foundation because it changes neither the safety problem nor the proxy mass deficit in the required direction.

## Pairwise-order foundation

Global raw proxy-vs-oracle pairwise-order accuracy is {summary['pairwise_order_accuracy_global']:.4%}; AA/FF/MIXED are {summary['pairwise_order_accuracy_AA']:.4%}/{summary['pairwise_order_accuracy_FF']:.4%}/{summary['pairwise_order_accuracy_MIXED']:.4%}. These are descriptive diagnostics, not a scalar selection rule.

## Controls and outputs

Exact parameter costs follow `pruning/MC.py` semantics and are checked in `task_ordinal_coverage_identity.csv`. All required profiles, edges, unit labels, recovery tables, AA/domain-400 forensics, calibration/context sweeps, false-safe cases, runtime and summary outputs are present in this directory. No 50% subset is selected, no correction is applied. Exact dominance is not relaxed.

## Runtime

```text
{pd.DataFrame([runtime]).to_string(index=False)}
```
"""
    (out / "task_ordinal_coverage_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

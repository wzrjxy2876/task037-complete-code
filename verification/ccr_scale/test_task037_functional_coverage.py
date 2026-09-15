#!/usr/bin/env python3
"""Targeted invariant tests for TASK037 functional-coverage outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task037_functional_coverage import coverage_matrix, unit_cost


REQUIRED = [
    "task_context_coverage_identity.csv",
    "task_context_coverage_unit_cost.csv",
    "task_context_coverage_proxy_edges.csv",
    "task_context_coverage_oracle_edges.csv",
    "task_context_coverage_proxy_units.csv",
    "task_context_coverage_oracle_units.csv",
    "task_context_coverage_witnesses.csv",
    "task_context_coverage_recovery.csv",
    "task_context_coverage_false_safe.csv",
    "task_context_coverage_parameter_mass.csv",
    "task_context_coverage_type_summary.csv",
    "task_context_coverage_attention_audit.csv",
    "task_context_coverage_calibration.csv",
    "task_context_coverage_context_count.csv",
    "task_context_coverage_stability.csv",
    "task_context_coverage_summary.json",
    "task_context_coverage_report.md",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    for name in REQUIRED:
        assert (out / name).exists(), name

    summary = json.loads((out / "task_context_coverage_summary.json").read_text())
    assert summary["decision"] == "RELATION_CONDITIONED_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED"
    assert summary["domains"] == 29 and summary["units"] == 116
    assert summary["videos"] == 30 and summary["classes"] == 10
    assert summary["relation_contexts"] == 10 and summary["primary_profile_dimensions"] == 10
    assert summary["no_pruning"] and summary["no_finetuning"]
    assert summary["runtime"]["no_model_inference"]
    assert summary["runtime"]["no_forward"] and summary["runtime"]["no_backward"]
    assert summary["runtime"]["no_masking"]
    assert summary["runtime"]["gpu_used"] and "RTX 3090" in summary["runtime"]["gpu_model"]

    identity = pd.read_csv(out / "task_context_coverage_identity.csv")
    assert len(identity) >= 20 and set(identity["status"]) == {"PASS"}

    costs = pd.read_csv(out / "task_context_coverage_unit_cost.csv")
    assert len(costs) == 116 and costs.global_index.is_unique
    assert costs.parameter_removal_cost.gt(0).all()
    for r in costs.itertuples(index=False):
        expected, stage = unit_cost(r.layer, r.unit_type)
        assert int(r.parameter_removal_cost) == expected
        assert r.stage == stage

    proxy = pd.read_csv(out / "task_context_coverage_proxy_units.csv")
    oracle = pd.read_csv(out / "task_context_coverage_oracle_units.csv")
    assert len(proxy) == len(oracle) == 116
    assert set(proxy.global_index) == set(oracle.global_index) == set(costs.global_index)
    assert np.array_equal(proxy.protected_proxy.to_numpy(bool), ~proxy.covered_proxy.to_numpy(bool))
    assert np.array_equal(oracle.protected_oracle.to_numpy(bool), ~oracle.covered_oracle.to_numpy(bool))
    assert proxy.domain_id.nunique() == 29

    pedges = pd.read_csv(out / "task_context_coverage_proxy_edges.csv")
    oedges = pd.read_csv(out / "task_context_coverage_oracle_edges.csv")
    for df in [pedges, oedges]:
        assert (df.domain_id == df.domain_id).all()
        assert (df.candidate_global_index != df.witness_global_index).all()
        manifest = proxy.set_index("global_index")["domain_id"]
        assert all(manifest.loc[a] == manifest.loc[b] for a, b in zip(df.candidate_global_index, df.witness_global_index))

    witnesses = pd.read_csv(out / "task_context_coverage_witnesses.csv")
    assert len(witnesses) == len(pedges)
    for r in witnesses.itertuples(index=False):
        cvals = np.array([getattr(r, f"q_candidate_{i}") for i in range(1, 11)])
        wvals = np.array([getattr(r, f"q_witness_{i}") for i in range(1, 11)])
        assert np.all(wvals >= cvals - 1e-12) and np.any(wvals > cvals + 1e-12)
        assert int(r.witness_exists_in_oracle) in (0, 1)

    recovery = pd.read_csv(out / "task_context_coverage_recovery.csv")
    global_rec = recovery[recovery.domain_id.astype(str) == "ALL"].iloc[0]
    assert int(global_rec.TP + global_rec.FP + global_rec.TN + global_rec.FN) == 116
    assert int(global_rec.FP) == summary["proxy_false_safe_units"]
    assert int(global_rec.covered_count) == summary["proxy_covered_units"]
    assert int(global_rec.oracle_covered_count) == summary["oracle_covered_units"]
    assert abs(float(global_rec.proxy_coverage_parameter_ratio) - summary["proxy_coverage_parameter_ratio"]) < 1e-12

    false_safe = pd.read_csv(out / "task_context_coverage_false_safe.csv")
    assert len(false_safe) == summary["proxy_false_safe_witness_rows"] > 0
    assert false_safe.oracle_contradiction_context_ids.astype(str).str.len().gt(0).all()
    assert false_safe.candidate_global_index.nunique() == summary["proxy_false_safe_units"]
    assert false_safe.candidate_type.eq("attention_head").all()

    mass = pd.read_csv(out / "task_context_coverage_parameter_mass.csv")
    gm = mass[mass.domain_id.astype(str) == "ALL"].iloc[0]
    assert abs(gm.full_tested_parameter_cost - costs.parameter_removal_cost.sum()) < 1e-6
    assert abs(gm.proxy_coverage_parameter_ratio - summary["proxy_coverage_parameter_ratio"]) < 1e-12
    assert abs(gm.oracle_coverage_parameter_ratio - summary["oracle_coverage_parameter_ratio"]) < 1e-12
    assert bool(gm.proxy_50pct_parameter_feasible) is False
    assert bool(gm.oracle_50pct_parameter_feasible) is True

    types = pd.read_csv(out / "task_context_coverage_type_summary.csv")
    assert set(["AA", "FF", "MIXED"]).issubset(set(types.category))
    mixed = types[types.category == "MIXED_witness_type_counts"].iloc[0]
    assert int(mixed.Attention_covered_by_Attention + mixed.Attention_covered_by_FFN + mixed.FFN_covered_by_Attention + mixed.FFN_covered_by_FFN) == summary["mixed_witness_total"]
    aa = pd.read_csv(out / "task_context_coverage_attention_audit.csv")
    assert len(aa) == 11 and "ALL_AA" in aa.domain_id.astype(str).tolist()

    cal = pd.read_csv(out / "task_context_coverage_calibration.csv")
    assert set(cal.calibration) == {"n9a", "n9b", "n9c", "N18", "N30"}
    assert cal.deterministic.all()
    ctx = pd.read_csv(out / "task_context_coverage_context_count.csv")
    assert set(ctx.context_count) == {5, 10} and ctx.deterministic.all()
    stab = pd.read_csv(out / "task_context_coverage_stability.csv")
    assert set(stab.stability_kind) == {"calibration_vs_full_oracle", "N9_replication_pair", "leave_one_class_out"}
    assert stab.query("stability_kind == 'calibration_vs_full_oracle'").replication.isin(["n9a", "n9b", "n9c", "N18", "N30"]).all()

    # Small exact tests for strict dominance, incomparability, transitivity,
    # maximal protected witness, and CPU/GPU equivalence.
    q = torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2], [0.0, 0.3]], dtype=torch.float64)
    c_cpu = coverage_matrix(q).cpu().numpy()
    assert not np.diag(c_cpu).any()
    assert c_cpu[1, 0] and c_cpu[2, 1] and c_cpu[2, 0]
    assert not c_cpu[0, 1] and not c_cpu[3, 0] and not c_cpu[0, 3]
    assert ((c_cpu.astype(np.int8) @ c_cpu.astype(np.int8)) > 0)[2, 0]
    protected = ~c_cpu.any(axis=0)
    assert bool(protected[2]) and bool(protected[3])
    assert bool(c_cpu[2, 0])
    if torch.cuda.is_available():
        c_gpu = coverage_matrix(q.cuda()).cpu().numpy()
        assert np.array_equal(c_cpu, c_gpu)
        gpu_equivalence = "PASS"
    else:
        gpu_equivalence = "SKIP_LOCAL_NO_CUDA"

    report = (out / "task_context_coverage_report.md").read_text()
    for token in ["RELATION_CONDITIONED_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED", "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "Exact dominance is not relaxed"]:
        assert token in report

    print(json.dumps({"status": "PASS", "gpu_cpu_synthetic": gpu_equivalence, "required_files": len(REQUIRED), "units": len(costs), "false_safe_units": summary["proxy_false_safe_units"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Targeted invariant tests for the ordinal coverage audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task037_ordinal_coverage import ordinal_rho, coverage_matrix, unit_cost


REQUIRED = [
    "task_ordinal_coverage_identity.csv", "task_ordinal_coverage_pairwise_order.csv",
    "task_ordinal_coverage_profiles.csv", "task_ordinal_coverage_proxy_edges.csv",
    "task_ordinal_coverage_oracle_edges.csv", "task_ordinal_coverage_units.csv",
    "task_ordinal_coverage_recovery.csv", "task_ordinal_coverage_mag_vs_rank.csv",
    "task_ordinal_coverage_oracle_representation.csv", "task_ordinal_coverage_attention.csv",
    "task_ordinal_coverage_domain400.csv", "task_ordinal_coverage_type_summary.csv",
    "task_ordinal_coverage_parameter_mass.csv", "task_ordinal_coverage_calibration.csv",
    "task_ordinal_coverage_context_count.csv", "task_ordinal_coverage_false_safe.csv",
    "task_ordinal_coverage_unit_cost.csv",
    "task_ordinal_coverage_runtime.json", "task_ordinal_coverage_summary.json",
    "task_ordinal_coverage_report.md",
]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output_dir", required=True); args = ap.parse_args()
    out = Path(args.output_dir)
    for f in REQUIRED: assert (out / f).exists(), f
    s = json.loads((out / "task_ordinal_coverage_summary.json").read_text())
    assert s["decision"] == "ORDINAL_RELATION_CONDITIONED_COVERAGE_REJECTED"
    assert s["domains"] == 29 and s["units"] == 116 and s["videos"] == 30 and s["classes"] == 10
    assert s["relation_contexts"] == 10 and s["no_pruning"] and s["no_finetuning"]
    rt = s["runtime"]
    assert rt["gpu_used"] and "RTX 3090" in rt["gpu_model"]
    for k in ["no_model_inference", "no_forward", "no_backward", "no_masking", "no_pruning", "no_finetuning"]: assert rt[k]

    identity = pd.read_csv(out / "task_ordinal_coverage_identity.csv")
    assert len(identity) >= 20 and set(identity.status) == {"PASS"}
    costs = pd.read_csv(out / "task_ordinal_coverage_parameter_mass.csv")
    units_cost = pd.read_csv(out / "task_ordinal_coverage_profiles.csv")
    assert len(units_cost) == 116 and units_cost.global_index.is_unique
    for r in pd.read_csv(out / "task_ordinal_coverage_unit_cost.csv").itertuples(index=False):
        expected, stage = unit_cost(r.layer, r.unit_type)
        assert int(r.parameter_removal_cost) == expected and r.stage == stage

    profiles = pd.read_csv(out / "task_ordinal_coverage_profiles.csv")
    qcols = [c for c in profiles.columns if c.startswith("q_")]
    assert len(qcols) == 40 and profiles[qcols].applymap(lambda x: 0 <= x <= 1).all().all()

    units = pd.read_csv(out / "task_ordinal_coverage_units.csv")
    assert set(units.representation) == {"ordinal_proxy", "ordinal_oracle", "magnitude_proxy", "magnitude_oracle"}
    for rep in units.representation.unique():
        x = units[units.representation == rep]
        assert len(x) == 116 and np.array_equal(x.protected.to_numpy(bool), ~x.covered.to_numpy(bool))

    pu = pd.read_csv(out / "task_ordinal_coverage_proxy_edges.csv")
    ou = pd.read_csv(out / "task_ordinal_coverage_oracle_edges.csv")
    manifest = profiles.set_index("global_index").domain_id
    for df in [pu, ou]:
        assert (df.candidate_global_index != df.witness_global_index).all()
        assert all(manifest.loc[a] == manifest.loc[b] for a, b in zip(df.candidate_global_index, df.witness_global_index))
    for r in pu.itertuples(index=False):
        c = np.array([getattr(r, f"q_rank_candidate_{i}") for i in range(1, 11)])
        w = np.array([getattr(r, f"q_rank_witness_{i}") for i in range(1, 11)])
        assert np.all(w >= c) and np.any(w > c)

    rec = pd.read_csv(out / "task_ordinal_coverage_recovery.csv")
    g = rec[rec.domain_id.astype(str) == "ALL"].iloc[0]
    assert int(g.TP + g.FP + g.TN + g.FN) == 116
    assert int(g.FP) == s["ordinal_proxy_false_safe_units"]
    assert int(g.covered_count) == s["ordinal_proxy_covered_units"]
    assert int(g.oracle_covered_count) == s["ordinal_oracle_covered_units"]
    assert abs(g.proxy_coverage_parameter_ratio - s["ordinal_proxy_coverage_parameter_ratio"]) < 1e-12

    fs = pd.read_csv(out / "task_ordinal_coverage_false_safe.csv")
    assert len(fs) > 0 and fs.candidate_global_index.nunique() == s["ordinal_proxy_false_safe_units"]
    assert set(fs.forensic_type).issubset({"A_persistent_pairwise_inversion", "B_sparse_pairwise_errors", "C_class_aggregation_disagreement"})
    assert fs.disagreement_count.gt(0).all()

    pm = pd.read_csv(out / "task_ordinal_coverage_parameter_mass.csv")
    gm = pm[(pm.representation == "ordinal") & (pm.domain_id.astype(str) == "ALL")].iloc[0]
    assert abs(gm.full_tested_parameter_cost - profiles.parameter_removal_cost.sum()) < 1e-6
    assert bool(gm.proxy_50pct_parameter_feasible) is False
    assert bool(gm.oracle_50pct_parameter_feasible) is False

    typ = pd.read_csv(out / "task_ordinal_coverage_type_summary.csv")
    assert set(["AA", "FF", "MIXED"]).issubset(set(typ.category))
    assert int(typ[typ.category == "MIXED_witness_type_counts"].mixed_witness_total.iloc[0]) == s["mixed_witness_total"]
    att = pd.read_csv(out / "task_ordinal_coverage_attention.csv")
    cand = att[att.audit_kind == "previous_magnitude_false_safe_candidate"]
    assert set(cand.domain_id.astype(int)) == {355, 391, 400}
    assert set(cand.candidate_global_index.astype(int)) == {10069, 13162, 22447, 22457}

    pair = pd.read_csv(out / "task_ordinal_coverage_pairwise_order.csv")
    assert len(pair[pair.scope == "domain"]) == 29
    assert pair[pair.scope == "domain"].pair_count.eq(1800).all()
    assert pair[pair.scope == "global"].pairwise_order_accuracy.iloc[0] == s["pairwise_order_accuracy_global"]
    dom400 = pd.read_csv(out / "task_ordinal_coverage_domain400.csv")
    assert set(dom400.record_kind) >= {"profile", "unit_context", "pair_order", "ordinal_edge"}
    assert len(dom400[dom400.record_kind == "profile"]) == 4
    assert len(dom400[dom400.record_kind == "unit_context"]) == 1200

    cal = pd.read_csv(out / "task_ordinal_coverage_calibration.csv")
    assert set(cal.calibration) == {"n9a", "n9b", "n9c", "N18", "N30"} and cal.deterministic.all()
    ctx = pd.read_csv(out / "task_ordinal_coverage_context_count.csv")
    assert set(ctx.context_count) == {5, 10} and ctx.deterministic.all()

    # Exact rho formula, tie half-credit, strict dominance/incomparability,
    # transitivity, protected complement, and GPU/CPU equivalence.
    toy = torch.tensor([[[1., 1., 2., 3.]]], dtype=torch.float64)
    assert np.array_equal(ordinal_rho(toy).numpy()[0, 0], np.array([1/6, 1/6, 2/3, 1.0]))
    q = torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2], [0.0, 0.3]], dtype=torch.float64)
    cpu = coverage_matrix(q).numpy()
    assert cpu[1, 0] and cpu[2, 1] and cpu[2, 0] and not cpu[3, 0] and not cpu[0, 3]
    assert not np.diag(cpu).any() and not (cpu & cpu.T).any()
    assert ((cpu.astype(np.int8) @ cpu.astype(np.int8)) > 0)[2, 0]
    assert np.array_equal(~cpu.any(axis=0), np.array([False, False, True, True]))
    if torch.cuda.is_available():
        assert np.array_equal(cpu, coverage_matrix(q.cuda()).cpu().numpy()); gpu_cpu = "PASS"
    else:
        gpu_cpu = "SKIP_LOCAL_NO_CUDA"

    report = (out / "task_ordinal_coverage_report.md").read_text()
    for token in ["ORDINAL_RELATION_CONDITIONED_COVERAGE_REJECTED", "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "Exact dominance is not relaxed"]: assert token in report
    print(json.dumps({"status": "PASS", "gpu_cpu_reference": gpu_cpu, "required_files": len(REQUIRED), "false_safe_units": s["ordinal_proxy_false_safe_units"]}, indent=2))


if __name__ == "__main__":
    main()

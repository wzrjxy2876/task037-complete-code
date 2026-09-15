import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(os.environ.get("TASK037_CCR_OUTPUT", "outputs/task037_ccr"))

REQUIRED = [
    "task_ccr_identity.csv", "task_ccr_context_regret.csv", "task_ccr_unit_scores.csv",
    "task_ccr_oracle_scores.csv", "task_ccr_domain_rankings.csv", "task_ccr_selection_gap.csv",
    "task_ccr_frequency_baseline.csv", "task_ccr_mean_baseline.csv", "task_ccr_fixed_baseline.csv",
    "task_ccr_raw_baseline.csv", "task_ccr_method_comparison.csv", "task_ccr_type_summary.csv",
    "task_ccr_position_stability.csv", "task_ccr_loco.csv", "task_ccr_calibration_size.csv",
    "task_ccr_context_count.csv", "task_ccr_severity_audit.csv", "task_ccr_same_span.csv",
    "task_ccr_oracle_safety_curves.csv", "task_ccr_disagreement_cases.csv", "task_ccr_bootstrap.csv",
    "task_ccr_summary.json", "task_ccr_report.md",
]


def test_outputs():
    assert not [n for n in REQUIRED if not (OUT / n).exists() or (OUT / n).stat().st_size == 0]


def test_identity_and_regret_invariants():
    ident = pd.read_csv(OUT / "task_ccr_identity.csv")
    assert set(ident.status.astype(str)) == {"PASS"}
    d = pd.read_csv(OUT / "task_ccr_context_regret.csv")
    assert len(d) == 22680
    assert d.video_key.nunique() == 30 and d.domain_id.nunique() == 9 and d.global_index.nunique() == 36
    assert d.groupby("video_key").context_id.nunique().eq(21).all()
    assert d.proxy_regret.between(0, 1).all() and d.oracle_regret.between(0, 1).all()
    assert np.allclose(d.groupby(["video_key", "context_id", "domain_id"]).proxy_regret.min().to_numpy(), 0.0)
    assert np.allclose(d.groupby(["video_key", "context_id", "domain_id"]).oracle_regret.min().to_numpy(), 0.0)


def test_selection_gap_and_baselines():
    comp = pd.read_csv(OUT / "task_ccr_selection_gap.csv")
    assert set(comp.method) == {"CCR", "frequency", "mean_damage", "fixed_original", "raw_regret"}
    assert len(comp) == 45
    assert np.allclose(comp.selected_oracle_CCR - comp.oracle_best_CCR, comp.SelectionGap)
    summ = json.loads((OUT / "task_ccr_summary.json").read_text())
    assert summ["CCR_top1_identity_rate"] >= 0
    assert summ["bootstrap_replicates"] == 10000 and summ["bootstrap_seed"] == 3407


def test_aggregation_arithmetic():
    d = pd.read_csv(OUT / "task_ccr_context_regret.csv")
    rel = d[d.context_id > 0]
    vmax = rel.groupby(["video_key", "domain_id", "global_index"], as_index=False).proxy_regret.max().groupby(["domain_id", "global_index"], as_index=False).proxy_regret.mean()
    got = pd.read_csv(OUT / "task_ccr_unit_scores.csv")[["domain_id", "global_index", "CCR"]]
    chk = got.merge(vmax, on=["domain_id", "global_index"])
    assert np.allclose(chk.CCR, chk.proxy_regret)
    oracle = pd.read_csv(OUT / "task_ccr_oracle_scores.csv")
    ov = rel.groupby(["video_key", "domain_id", "global_index"], as_index=False).oracle_regret.max().groupby(["domain_id", "global_index"], as_index=False).oracle_regret.mean()
    assert np.allclose(oracle.merge(ov, on=["domain_id", "global_index"]).oracle_CCR, oracle.merge(ov, on=["domain_id", "global_index"]).oracle_regret)
    winners = d.sort_values(["video_key", "context_id", "domain_id", "proxy_signed_damage", "global_index"]).groupby(["video_key", "context_id", "domain_id"], as_index=False).first()
    freq = winners.groupby(["domain_id", "global_index"]).size().rename("frequency").reset_index()
    assert np.allclose(pd.read_csv(OUT / "task_ccr_frequency_baseline.csv").merge(freq, on=["domain_id", "global_index"]).frequency_x, pd.read_csv(OUT / "task_ccr_frequency_baseline.csv").merge(freq, on=["domain_id", "global_index"]).frequency_y)
    mean = d.groupby(["domain_id", "global_index"]).proxy_signed_damage.mean().reset_index(name="mean_damage")
    assert np.allclose(pd.read_csv(OUT / "task_ccr_mean_baseline.csv").merge(mean, on=["domain_id", "global_index"]).mean_damage_x, pd.read_csv(OUT / "task_ccr_mean_baseline.csv").merge(mean, on=["domain_id", "global_index"]).mean_damage_y)
    fixed = d[d.context_id == 0].groupby(["domain_id", "global_index"]).proxy_signed_damage.mean().reset_index(name="fixed_original")
    assert np.allclose(pd.read_csv(OUT / "task_ccr_fixed_baseline.csv").merge(fixed, on=["domain_id", "global_index"]).fixed_original_x, pd.read_csv(OUT / "task_ccr_fixed_baseline.csv").merge(fixed, on=["domain_id", "global_index"]).fixed_original_y)


def test_deterministic_splits_and_bootstrap():
    pos = pd.read_csv(OUT / "task_ccr_position_stability.csv")
    assert set(pos.split) == {"position_1", "position_2", "position_3"}
    sz = pd.read_csv(OUT / "task_ccr_calibration_size.csv")
    assert set(sz.N) == {3, 6, 9, 12, 18, 30}
    cc = pd.read_csv(OUT / "task_ccr_context_count.csv")
    assert set(cc.context_count) == {5, 10, 15, 20}
    boot = pd.read_csv(OUT / "task_ccr_bootstrap.csv")
    assert len(boot) == 10000 and boot.replicate.iloc[0] == 0 and boot.replicate.iloc[-1] == 9999
    assert boot.notna().all().all()


def test_synthetic_divergence():
    syn = pd.DataFrame(json.loads((OUT / "task_ccr_synthetic_sanity.json").read_text()))
    assert bool(syn.loc[syn.unit == "A", "frequency_selected"].iloc[0])
    assert bool(syn.loc[syn.unit == "B", "CCR_selected"].iloc[0])

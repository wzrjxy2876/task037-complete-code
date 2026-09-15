import json
import os
from pathlib import Path

import pandas as pd


OUT = Path(os.environ.get("TASK037_PROXY_OUTPUT", "outputs/task037_context_proxy"))


REQUIRED = [
    "task_context_proxy_manifest.csv",
    "task_context_proxy_identity_audit.csv",
    "task_context_proxy_raw.csv",
    "task_context_proxy_original_validation.csv",
    "task_context_proxy_conditioned_validation.csv",
    "task_context_proxy_pairwise_order.csv",
    "task_context_proxy_reversal_detection.csv",
    "task_context_proxy_static_baseline.csv",
    "task_context_proxy_perfect_static_reference.csv",
    "task_context_proxy_margin_analysis.csv",
    "task_context_proxy_conditioned_margin.csv",
    "task_context_proxy_span_analysis.csv",
    "task_context_proxy_severity_analysis.csv",
    "task_context_proxy_same_span.csv",
    "task_context_proxy_type_summary.csv",
    "task_context_proxy_stage_summary.csv",
    "task_context_proxy_video_summary.csv",
    "task_context_proxy_class_summary.csv",
    "task_context_proxy_bootstrap.csv",
    "task_context_proxy_aligned_cf_audit.csv",
    "task_context_proxy_examples.csv",
    "task_context_proxy_runtime_summary.json",
    "task_context_proxy_summary.json",
    "task_context_proxy_report.md",
]


def test_required_outputs_exist_and_nonempty():
    missing = [name for name in REQUIRED if not (OUT / name).exists() or (OUT / name).stat().st_size == 0]
    assert not missing, missing


def test_identity_audit_and_raw_cardinality():
    audit = pd.read_csv(OUT / "task_context_proxy_identity_audit.csv")
    assert set(audit.status.astype(str)) == {"PASS"}
    raw = pd.read_csv(OUT / "task_context_proxy_raw.csv")
    assert len(raw) == 22680
    assert raw.video_key.nunique() == 30
    assert raw.global_index.nunique() == 36
    assert raw.context_id.nunique() == 21
    assert raw.groupby("video_key").context_id.nunique().eq(21).all()


def test_gpu_shards_are_disjoint_and_complete():
    s0 = pd.read_csv(OUT / "shards/gpu0/task_context_proxy_raw.csv")
    s1 = pd.read_csv(OUT / "shards/gpu1/task_context_proxy_raw.csv")
    assert len(s0) == len(s1) == 11340
    assert set(s0.video_key).isdisjoint(set(s1.video_key))
    assert set(s0.video_key) | set(s1.video_key) == set(pd.read_csv(OUT / "task_context_proxy_raw.csv").video_key)


def test_summary_and_bootstrap():
    summary = json.loads((OUT / "task_context_proxy_summary.json").read_text())
    assert summary["decision"] == "CONTEXTUAL_FIRST_ORDER_PROXY_PROMISING"
    assert summary["conditioned_pairwise_accuracy_mean"] > summary["fixed_pairwise_accuracy_mean"]
    assert summary["bootstrap_ci"]["improvement_over_fixed"][0] > 0
    boot = pd.read_csv(OUT / "task_context_proxy_bootstrap.csv")
    assert len(boot) == 10000
    assert boot.replicate.iloc[0] == 0 and boot.replicate.iloc[-1] == 9999


def test_static_and_stratified_outputs_are_populated():
    for name in [
        "task_context_proxy_static_baseline.csv",
        "task_context_proxy_perfect_static_reference.csv",
        "task_context_proxy_type_summary.csv",
        "task_context_proxy_stage_summary.csv",
        "task_context_proxy_video_summary.csv",
        "task_context_proxy_class_summary.csv",
    ]:
        assert len(pd.read_csv(OUT / name)) > 0

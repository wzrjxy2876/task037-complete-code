"""Targeted integrity tests for the Task037 motivation artifacts."""
from __future__ import annotations
import csv, json, os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("TASK037_GAP_OUTPUT", "/data/jixinye25/work1/output/task037_temporal_redundancy_motivation"))


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_identity_and_frozen_manifest():
    s = json.loads((OUT / "task037_temporal_gap_summary.json").read_text())
    assert (s["N"], s["T"], s["H"], s["W"], s["mapped_units"]) == (9, 16, 7, 7, 36378)
    m = read("task037_temporal_gap_masking_manifest.csv")
    keys = [(r["pair_id"], r["pair_side"], r["global_index"]) for r in m]
    assert len(keys) == len(set(keys))


def test_local_curve_cardinality_and_signed_clamp():
    rows = read("task037_temporal_gap_local_similarity.csv")
    assert rows
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
        a = float(r["A_local"])
        c = float(r["A_local_clamped"])
        if np.isfinite(a): assert abs(c - max(a, 0.0)) < 1e-6
    assert all(len(v) == 9 * 16 for v in by_pair.values())


def test_pair_ordering_and_required_outputs():
    rows = read("task037_temporal_gap_global_pairs.csv")
    assert all(int(r["global_index_i"]) < int(r["global_index_j"]) for r in rows)
    required = [
        "task037_temporal_gap_identity_audit.csv", "task037_temporal_gap_global_pairs.csv",
        "task037_temporal_gap_local_similarity.csv", "task037_temporal_gap_pair_summary.csv",
        "task037_temporal_gap_motivation_pairs.csv", "task037_temporal_gap_control_pairs.csv",
        "task037_temporal_gap_masking_manifest.csv", "task037_temporal_gap_temporal_damage.csv",
        "task037_temporal_gap_pair_damage_comparison.csv", "task037_temporal_gap_cf_vs_damage.csv",
        "task037_temporal_gap_type_domain_summary.csv", "task037_temporal_gap_summary.json",
        "task037_temporal_gap_report.md",
    ]
    assert all((OUT / x).exists() for x in required)


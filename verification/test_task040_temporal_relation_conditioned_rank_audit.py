#!/usr/bin/env python3
"""Targeted offline tests for the TASK040 motivation audit outputs.

The tests intentionally consume only the frozen Phase-C CSV artifacts and the
diagnostic CSV/JSON outputs.  They never instantiate a model or touch CUDA.
"""
from __future__ import annotations

import itertools
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "TASK040_ROOT",
    "/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis/n03_fixed_span",
))
OUT = Path(os.environ.get(
    "TASK040_OUT",
    "/data/jixinye25/work1/output/task040_temporal_relation_conditioned_rank_audit",
))
SPANS = {1, 2, 4, 8, 16}


def signed_rank(values: dict[int, float]) -> dict[int, int]:
    order = sorted(values, key=lambda uid: (-float(values[uid]), int(uid)))
    return {int(uid): idx + 1 for idx, uid in enumerate(order)}


class Task040AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "task040_raw_records.csv")
        cls.out = json.loads((OUT / "task_temporal_rank_summary.json").read_text())
        cls.orig = pd.read_csv(OUT / "task_temporal_rank_original_damage.csv")
        cls.cond = pd.read_csv(OUT / "task_temporal_rank_conditioned_damage.csv")
        cls.rev = pd.read_csv(OUT / "task_temporal_rank_reversal_pairs.csv")
        cls.match = pd.read_csv(OUT / "task_temporal_rank_severity_matching.csv")
        cls.same = pd.read_csv(OUT / "task_temporal_rank_same_span_analysis.csv")
        cls.sign = pd.read_csv(OUT / "task_temporal_rank_sign_changes.csv")

    def test_required_outputs_and_scope(self):
        required = (
            "identity_audit", "original_damage", "conditioned_damage",
            "intervention_summary", "reversal_pairs", "margin_analysis",
            "span_analysis", "severity_matching", "severity_matched_results",
            "same_span_analysis", "unit_volatility", "sign_changes",
            "type_summary", "stage_summary", "bms_domain_summary", "examples",
            "figure_data", "summary", "report",
        )
        for stem in required:
            suffix = ".json" if stem == "summary" else ".md" if stem == "report" else ".csv"
            self.assertTrue((OUT / f"task_temporal_rank_{stem}{suffix}").exists(), stem)
        self.assertTrue(self.out["no_gpu"])
        self.assertTrue(self.out["no_model_rerun"])
        self.assertTrue(self.out["no_pruning"])
        self.assertTrue(self.out["no_finetuning"])
        self.assertEqual(self.out["decision"], "TEMPORAL_RELATION_CONDITIONED_PRUNING_SENSITIVITY_SUPPORTED")

    def test_identity_checks_all_pass(self):
        checks = pd.read_csv(OUT / "task_temporal_rank_identity_audit.csv")
        self.assertEqual(set(checks["status"]), {"PASS"})
        self.assertEqual(self.raw.video_index.nunique(), 3)
        self.assertEqual(self.raw.unit_global_index.nunique(), 32)
        self.assertEqual(len(self.raw), 7680)
        self.assertEqual(set(self.raw.block_size.unique()), SPANS)
        self.assertTrue((self.raw.groupby("block_size").pair_index.nunique() == 16).all())

    def test_original_damage_invariance_and_signed_arithmetic(self):
        inv = self.raw.groupby(["video_index", "unit_global_index"]).d_original.agg(lambda x: float(np.max(np.abs(x - x.iloc[0]))))
        self.assertLessEqual(float(inv.max()), 1e-5)
        self.assertLessEqual(float(np.max(np.abs(self.raw.d_original - (self.raw.z_true_original - self.raw.z_true_original_masked)))), 1e-5)
        self.assertLessEqual(float(np.max(np.abs(self.raw.d_intervened - (self.raw.z_true_intervened - self.raw.z_true_intervened_masked)))), 1e-5)

    def test_deterministic_ranking_and_tie_break(self):
        for video, g in self.orig.groupby("video_index"):
            damages = dict(zip(g.unit_global_index.astype(int), g.D_ori.astype(float)))
            expected = signed_rank(damages)
            observed = dict(zip(g.unit_global_index.astype(int), g.Rank_ori.astype(int)))
            self.assertEqual(observed, expected)
            for i, j in itertools.combinations(sorted(damages), 2):
                if damages[i] == damages[j]:
                    self.assertLess(observed[i], observed[j]) if i < j else self.assertLess(observed[j], observed[i])
        for key, g in self.cond.groupby(["video_index", "level", "span", "pair_index"]):
            damages = dict(zip(g.unit_global_index.astype(int), g.D_rel.astype(float)))
            observed = dict(zip(g.unit_global_index.astype(int), g.Rank_rel.astype(int)))
            self.assertEqual(observed, signed_rank(damages))

    def test_strict_reversal_arithmetic(self):
        orig = {(int(r.video_index), int(r.unit_global_index)): float(r.D_ori) for r in self.orig.itertuples()}
        cond = {(int(r.video_index), int(r.level), int(r.span), int(r.pair_index), int(r.unit_global_index)): float(r.D_rel)
                for r in self.cond.itertuples()}
        levels = self.raw.drop_duplicates(["video_index", "block_size", "pair_index"])
        level_map = {(int(r.video_index), int(r.block_size), int(r.pair_index)): int(r.level) for r in levels.itertuples()}
        for r in self.rev.itertuples():
            d1 = orig[(int(r.video_index), int(r.unit_i))] - orig[(int(r.video_index), int(r.unit_j))]
            lev = level_map[(int(r.video_index), int(r.span), int(r.pair_index))]
            d2 = cond[(int(r.video_index), lev, int(r.span), int(r.pair_index), int(r.unit_i))] - cond[(int(r.video_index), lev, int(r.span), int(r.pair_index), int(r.unit_j))]
            self.assertNotEqual(d1, 0.0)
            self.assertNotEqual(d2, 0.0)
            self.assertLess(d1 * d2, 0.0)

    def test_margin_quartiles_and_upper_margin_reversals(self):
        margins = []
        for _, g in self.orig.groupby("video_index"):
            vals = dict(zip(g.unit_global_index.astype(int), g.D_ori.astype(float)))
            margins.extend(abs(vals[i] - vals[j]) for i, j in itertools.combinations(sorted(vals), 2))
        q25, q50, q75 = np.quantile(np.asarray(margins), [0.25, 0.5, 0.75])
        summary = self.out["margin_quantiles"]
        self.assertAlmostEqual(summary["q25"], q25, places=12)
        self.assertAlmostEqual(summary["q50"], q50, places=12)
        self.assertAlmostEqual(summary["q75"], q75, places=12)
        self.assertIn("Q75-Q100", set(self.rev.margin_quartile))

    def test_severity_matching_and_same_span_grouping(self):
        self.assertEqual(len(self.match), 480)
        self.assertEqual(self.match[["video_index", "span_a", "span_b", "pair_index_a"]].drop_duplicates().shape[0], len(self.match))
        self.assertEqual(self.match[["video_index", "span_a", "span_b", "pair_index_b"]].drop_duplicates().shape[0], len(self.match))
        self.assertTrue((self.match.span_a < self.match.span_b).all())
        self.assertEqual(len(self.same), 1800)
        self.assertTrue((self.same.groupby(["video_index", "span"]).size() == 120).all())

    def test_sign_changes_and_balanced_types(self):
        expected = np.sign(self.sign.D_ori.to_numpy()) != np.sign(self.sign.D_rel.to_numpy())
        observed = self.sign.sign_changed.astype(bool).to_numpy()
        np.testing.assert_array_equal(observed, expected)
        counts = self.orig.drop_duplicates("unit_global_index").type_label.value_counts().to_dict()
        self.assertEqual(counts, {"Attention": 16, "FFN": 16})

    def test_bms_mapping_identity_and_control_evidence(self):
        mapping = pd.read_csv(ROOT / "task040_bms_domain_mapping.csv")
        self.assertEqual(set(self.orig.unit_global_index), set(mapping.task040_unit_global_index))
        self.assertGreaterEqual(len(pd.read_csv(OUT / "task_temporal_rank_bms_domain_summary.csv")), 1)
        # The controlled audit must contain both homogeneous reversal classes.
        self.assertTrue({"AA", "FF"}.issubset(set(self.rev.pair_type)))
        self.assertGreater(len(self.rev.video_index.unique()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

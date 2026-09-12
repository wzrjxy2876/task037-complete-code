import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "lgfr_runtime"
sys.path.insert(0, str(SOURCE_ROOT))

import task041_phase_h_temporal_stress as phase_h


class PairwiseRankingTests(unittest.TestCase):
    def test_pairwise_win_tie_is_half(self):
        self.assertEqual(phase_h.pairwise_win(0.2, 0.2), 0.5)
        self.assertEqual(phase_h.pairwise_win(-1.0, 0.0), 1.0)
        self.assertEqual(phase_h.pairwise_win(1.0, 0.0), 0.0)

    def test_win_rates_use_only_same_domain_peers_and_average_conditions(self):
        values = {
            "1": {"a": 0.0, "b": 1.0},
            "2": {"a": 1.0, "b": 0.0},
            "3": {"a": 100.0},
            "4": {"a": 0.0},
        }
        domains = {"1": "A", "2": "A", "3": "B", "4": "B"}
        result = phase_h.group_win_rates(values, domains)
        self.assertEqual(result, {"1": 0.5, "2": 0.5, "3": 0.0, "4": 1.0})

    def test_win_rates_reject_condition_mismatch_within_domain(self):
        values = {"1": {"a": 0.0}, "2": {"b": 1.0}}
        with self.assertRaisesRegex(RuntimeError, "condition set mismatch"):
            phase_h.group_win_rates(values, {"1": "A", "2": "A"})


class RankStatisticTests(unittest.TestCase):
    def test_average_ranks_preserve_ties(self):
        self.assertEqual(phase_h.average_ranks([2.0, 1.0, 2.0]), [2.5, 1.0, 2.5])

    def test_spearman_and_kendall_tau_b(self):
        self.assertAlmostEqual(phase_h.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(
            phase_h.kendall_tau_b([1, 1, 2], [1, 2, 2]), 0.5
        )
        self.assertIsNone(phase_h.spearman([1, 1], [2, 3]))

    def test_historical_baseline_missingness_is_not_imputed(self):
        row = phase_h._baseline_domain_row(
            "269", ["1", "2", "3"],
            {"1": 0.1, "2": 0.2, "3": None},
            {"1": 0.3, "2": 0.2, "3": -0.1},
            "frozen_metric",
        )
        self.assertEqual(row["unit_count"], 2)
        self.assertEqual(row["missing_baseline_count"], 1)
        self.assertAlmostEqual(row["spearman"], -1.0)
        self.assertEqual(row["low_high_ordering"], "reverse")


class FrozenSubsetTests(unittest.TestCase):
    def test_all_27_class_balanced_n6_subsets(self):
        manifest = [
            {"label": label, "video_index": index}
            for label, start in ((0, 0), (1, 3), (2, 6))
            for index in range(start, start + 3)
        ]
        subsets = phase_h.class_balanced_n6_subsets(manifest)
        self.assertEqual(len(subsets), 27)
        self.assertEqual(len({row["subset_id"] for row in subsets}), 27)
        for row in subsets:
            self.assertEqual(len(row["video_indices"]), 6)
            self.assertEqual(
                [sum(index in row["video_indices"] for index in range(start, start + 3))
                 for start in (0, 3, 6)],
                [2, 2, 2],
            )

    def test_low_high_ties_are_deterministic_by_numeric_unit_id(self):
        self.assertEqual(
            phase_h.ordered_extremes({"12": 0.5, "3": 0.5, "8": 0.5}),
            ("3", "12"),
        )


class DecisionGateTests(unittest.TestCase):
    def test_A_requires_positive_association_and_noninferior_comparison(self):
        decision, gate = phase_h.temporal_decision_gate(
            {"spearman": 0.2, "kendall": 0.1, "safest_accuracy": 0.5},
            {"spearman": 0.1, "kendall": 0.1, "safest_accuracy": 0.5},
        )
        self.assertEqual(decision, "TEMPORAL_STRESS_SELECTION_PROMISING")
        self.assertTrue(gate["gate_passed"])
        self.assertFalse(gate["rejection_threshold_added"])

    def test_nonpositive_association_cannot_pass(self):
        decision, gate = phase_h.temporal_decision_gate(
            {"spearman": 0.0, "kendall": 0.5, "safest_accuracy": 1.0},
            {"spearman": -0.1, "kendall": 0.0, "safest_accuracy": 0.0},
        )
        self.assertEqual(decision, "TEMPORAL_STRESS_ADDS_NO_CLEAR_VALUE")
        self.assertFalse(gate["gate_passed"])

    def test_improvement_with_all_remaining_measures_worse_cannot_pass(self):
        decision, gate = phase_h.temporal_decision_gate(
            {"spearman": 0.4, "kendall": 0.1, "safest_accuracy": 0.2},
            {"spearman": 0.3, "kendall": 0.2, "safest_accuracy": 0.3},
        )
        self.assertEqual(decision, "TEMPORAL_STRESS_ADDS_NO_CLEAR_VALUE")
        self.assertTrue(gate["better_than_original_on_at_least_one_primary_measure"])
        self.assertFalse(gate["not_worse_on_all_remaining_primary_measures"])


if __name__ == "__main__":
    unittest.main()

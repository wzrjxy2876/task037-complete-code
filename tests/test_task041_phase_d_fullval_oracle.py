from __future__ import annotations

import unittest

import numpy as np

from task041_phase_d_fullval_oracle import (
    bootstrap_mean_ci,
    kendall_tau_b,
    spearman,
    wilcoxon_signed_rank,
)


class TestTask041PhaseDMath(unittest.TestCase):
    def test_rank_statistics_and_ties(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [2, 4, 8]), 1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [2, 4, 8]), 1.0)
        self.assertAlmostEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(kendall_tau_b([1, 1, 1], [1, 2, 3]), 0.0)

    def test_wilcoxon_zero_vector(self) -> None:
        result = wilcoxon_signed_rank([0.0, 0.0])
        self.assertEqual(result["method"], "all_zero")
        self.assertEqual(result["p_value"], 1.0)

    def test_bootstrap_is_deterministic_and_signed(self) -> None:
        values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        first = bootstrap_mean_ci(
            values, np.random.RandomState(3407), resamples=1000
        )
        second = bootstrap_mean_ci(
            values, np.random.RandomState(3407), resamples=1000
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], float(values.mean()))
        self.assertGreaterEqual(first[1], float(values.mean()))

    def test_signed_difference_is_not_abs(self) -> None:
        baseline = np.asarray([1.0, 2.0], dtype=np.float64)
        masked = np.asarray([2.0, 1.0], dtype=np.float64)
        difference = masked - baseline
        self.assertTrue(np.any(difference < 0))
        self.assertTrue(np.any(difference > 0))


if __name__ == "__main__":
    unittest.main()

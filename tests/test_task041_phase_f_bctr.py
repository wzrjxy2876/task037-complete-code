from __future__ import annotations

import unittest

import numpy as np

from task041_phase_f_bctr_audit import (
    bounded_least_squares,
    choose_low_high,
    dimension_key,
    rank_statistics,
    residual_ratio,
)


class TestTask041PhaseFBCTR(unittest.TestCase):
    def test_bounded_solver_does_not_amplify_predictor_above_one(self) -> None:
        y = np.asarray([2.0, 0.0], dtype=np.float64)
        x = np.asarray([[1.0], [0.0]], dtype=np.float64)
        fit = bounded_least_squares(y, x)
        self.assertTrue(fit["success"])
        self.assertLessEqual(float(fit["alpha"][0]), 1.0)
        self.assertGreaterEqual(float(fit["alpha"][0]), 0.0)
        self.assertAlmostEqual(float(fit["alpha"][0]), 1.0, places=10)
        self.assertAlmostEqual(residual_ratio(y, fit["reconstruction"]), 0.5, places=10)

    def test_collective_fit_contains_pairwise_model_without_sum_constraint(self) -> None:
        y = np.asarray([1.0, 1.0], dtype=np.float64)
        x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        fit = bounded_least_squares(y, x)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(float(fit["alpha"][0]), 1.0, places=10)
        self.assertAlmostEqual(float(fit["alpha"][1]), 1.0, places=10)
        self.assertAlmostEqual(residual_ratio(y, fit["reconstruction"]), 0.0, places=10)
        self.assertGreater(float(np.sum(fit["alpha"])), 1.0)

    def test_dimension_key_keeps_video_then_intervention_identity(self) -> None:
        row = {
            "video_index": "1",
            "video_id": "clip-B",
            "level": "3",
            "block_size": "8",
            "pair_index": "12",
        }
        self.assertEqual(dimension_key(row), (1, "clip-B", 3, 8, 12))

    def test_rank_statistics_direction_and_degenerate_inputs(self) -> None:
        rho, tau = rank_statistics([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertEqual(rho, -1.0)
        self.assertEqual(tau, -1.0)
        self.assertEqual(rank_statistics([1.0, 1.0], [2.0, 3.0]), (None, None))

    def test_score_ties_use_ascending_task037_identity(self) -> None:
        rows = [
            {"candidate_task037_global_index": "9", "score": 1.0},
            {"candidate_task037_global_index": "2", "score": 1.0},
            {"candidate_task037_global_index": "5", "score": 1.0},
        ]
        low, high = choose_low_high(rows, "score")
        self.assertEqual(low["candidate_task037_global_index"], "2")
        self.assertEqual(high["candidate_task037_global_index"], "2")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
_REPO_SRC = HERE.parents[1] / "src"
_LOCAL_SRC = HERE.parent / "src"
SOURCE_ROOT = _REPO_SRC if (_REPO_SRC / "lgfr_runtime").is_dir() else _LOCAL_SRC
sys.path.insert(0, str(SOURCE_ROOT))

from lgfr_runtime.task042_phase_h_continuous_coverage import (
    _validate_subset_map,
    class_median_preference,
    enumerate_nonempty_subsets,
    eta_value,
    exact_pareto_dominates,
    exact_parameter_accounting,
    leave_one_out,
    subset_envelope,
)


class PhaseHContinuousCoverageTests(unittest.TestCase):
    def test_phase_g_preference_reuse_without_reranking(self) -> None:
        # These are already-normalized Phase-G relation_preference values.
        values = [0.12658227848101267, 0.45569620253164556, 0.3291139240506329]
        self.assertEqual(class_median_preference(values), 0.3291139240506329)

    def test_zero_envelope_handling(self) -> None:
        self.assertEqual(eta_value(0.0, 0.0), 1.0)

    def test_subset_envelope_exactness(self) -> None:
        profiles = {"a": [0.2, 0.8, 0.0], "b": [0.6, 0.1, 0.0], "c": [0.4, 0.9, 0.2]}
        self.assertEqual(subset_envelope(profiles, ["a", "b"]), [0.6, 0.8, 0.0])

    def test_eta_exact_arithmetic(self) -> None:
        expected = 0.4 / (0.8 + 1e-12)
        self.assertEqual(eta_value(0.4, 0.8), expected)

    def test_exact_pareto_dominance(self) -> None:
        self.assertTrue(exact_pareto_dominates([0.8, 0.5, 1.0], [0.8, 0.4, 1.0]))
        self.assertFalse(exact_pareto_dominates([0.8, 0.5], [0.8, 0.5]))
        # No tolerance: a tiny component-wise loss means no dominance.
        self.assertFalse(exact_pareto_dominates([0.8, 0.5 - 1e-15], [0.8, 0.5]))

    def test_leave_one_out_identity(self) -> None:
        group = ["11", "22", "33"]
        self.assertEqual(leave_one_out(group, "22"), ("11", "33"))

    def test_parameter_accounting(self) -> None:
        retained, released = exact_parameter_accounting({"a": 10, "b": 20, "c": 5}, ["a", "c"])
        self.assertEqual((retained, released), (15, 20))

    def test_subset_determinism(self) -> None:
        items = ["a", "b", "c"]
        expected = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]
        self.assertEqual(enumerate_nonempty_subsets(items), expected)
        self.assertEqual(enumerate_nonempty_subsets(items), expected)

    def test_calibration_subset_identity(self) -> None:
        class_videos = {str(c): (c * 3, c * 3 + 1, c * 3 + 2) for c in range(10)}
        mapping = {
            "P1": tuple(v[0] for v in class_videos.values()),
            "P2": tuple(v[1] for v in class_videos.values()),
            "P3": tuple(v[2] for v in class_videos.values()),
            "P12": tuple(x for v in class_videos.values() for x in v[:2]),
            "P13": tuple(x for v in class_videos.values() for x in (v[0], v[2])),
            "P23": tuple(x for v in class_videos.values() for x in v[1:]),
            "FULL": tuple(x for v in class_videos.values() for x in v),
        }
        _validate_subset_map(mapping, class_videos)
        self.assertEqual(set(mapping["FULL"]), set(range(30)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from task041_collective_temporal_coverage_oracle import (
    compute_collective_coverage,
    kendall_tau_b,
    select_low_high,
    spearman,
    wilcoxon_signed_rank,
)


def profile(domain: str, task037: int, task040: int, unit_type: str, values: list[float]):
    return {
        "domain_id": domain,
        "task037_global_index": str(task037),
        "unit_global_index": str(task040),
        "layer_name": f"layers.0.blocks.{task037}.{'attn' if unit_type == 'head' else 'mlp'}",
        "unit_type": unit_type,
        "unit_index": "0",
        "stage": "0",
        **{f"g_span_{span}": str(value) for span, value in zip((1, 2, 4, 8, 16), values)},
    }


class TestTask041Coverage(unittest.TestCase):
    def test_leave_one_out_formula_and_float64(self) -> None:
        rows = [
            profile("271", 10, 20, "head", [2, 0, 1, 4, 3]),
            profile("271", 11, 21, "neuron", [1, 3, 1, 2, 3]),
            profile("271", 12, 22, "neuron", [0, 2, 5, 1, 1]),
        ]
        result = compute_collective_coverage(rows, domains=("271",), eps=1e-12)
        by_id = {row["candidate_task037_global_index"]: row for row in result}
        self.assertAlmostEqual(by_id[10]["delta_span_1"], 0.5)
        self.assertEqual(by_id[10]["delta_span_2"], 0.0)
        self.assertAlmostEqual(by_id[12]["delta_span_4"], 1.0)
        expected = np.sqrt(np.mean(np.square([0.5, 0.0, 0.0, 0.5, 0.0])))
        self.assertAlmostEqual(by_id[10]["R_MCTC"], expected)
        self.assertIsInstance(by_id[10]["R_MCTC"], float)

    def test_low_high_uses_task037_index_as_tie_break(self) -> None:
        rows = []
        for task037, r in ((9, 0.0), (3, 0.0), (12, 0.5)):
            rows.append(
                {
                    "domain_id": "271",
                    "candidate_task037_global_index": task037,
                    "R_MCTC": r,
                    "mixed_type": False,
                }
            )
        selected = select_low_high(rows, domains=("271",))
        self.assertEqual(selected[0]["candidate_role"], "low")
        self.assertEqual(selected[0]["candidate_task037_global_index"], 3)
        self.assertEqual(selected[1]["candidate_role"], "high")
        self.assertEqual(selected[1]["candidate_task037_global_index"], 12)

    def test_rank_statistics_are_deterministic_and_tie_safe(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [2, 4, 8]), 1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [2, 4, 8]), 1.0)
        self.assertAlmostEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(kendall_tau_b([1, 1, 1], [1, 2, 3]), 0.0)

    def test_wilcoxon_zero_vector_is_explicit(self) -> None:
        result = wilcoxon_signed_rank([0.0, 0.0])
        self.assertEqual(result["method"], "all_zero")
        self.assertEqual(result["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()

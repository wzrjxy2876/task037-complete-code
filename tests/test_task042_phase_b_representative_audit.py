import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import task042_phase_b_representative_audit as phase_b


class RepresentativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.ids = [1, 2, 3]
        self.distances = {(1, 2): 0.1, (1, 3): 0.8, (2, 3): 0.7}

    def test_exact_enumeration_finds_minimax_medoid(self):
        candidates = phase_b.enumerate_sets(self.ids, 1, self.distances)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["keep"], (2,))
        self.assertAlmostEqual(candidates[0]["j_max"], 0.7)
        self.assertAlmostEqual(candidates[0]["j_mean"], (0.1 + 0.7) / 3.0)

    def test_k_n_minus_one_exposes_symmetric_geometric_tie(self):
        candidates = phase_b.enumerate_sets(self.ids, 2, self.distances)
        self.assertEqual(candidates[0]["keep"], (1, 3))  # ID tie-break only.
        tied = [x for x in candidates if x["j_max"] == candidates[0]["j_max"]
                and x["j_mean"] == candidates[0]["j_mean"]]
        self.assertEqual({x["removed"] for x in tied}, {(1,), (2,)})
        self.assertEqual(candidates[0]["j_max"], 0.1)

    def test_coverage_includes_zero_distance_for_retained_units(self):
        self.assertEqual(phase_b.coverage(self.ids, (1, 3), self.distances),
                         (0.1, 0.1 / 3.0))

    def test_jaccard_uses_set_overlap(self):
        self.assertEqual(phase_b.jaccard((1, 2), (2, 3)), 1.0 / 3.0)
        self.assertEqual(phase_b.jaccard((), ()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

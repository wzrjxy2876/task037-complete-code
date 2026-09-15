import unittest
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src" / "lgfr_runtime", ROOT / "work"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
import task042_phase_l_interaction as phase_l


class PhaseLTests(unittest.TestCase):
    def test_temporal_position_identity(self):
        self.assertEqual(phase_l.temporal_positions(16), tuple(range(2, 16)))
        self.assertEqual(len(phase_l.PAIRS), 91)
        with self.assertRaises(ValueError):
            phase_l.temporal_positions(32)

    def test_factorial_state_identity_and_order(self):
        binds = phase_l.factorial_state_bindings(2, 15)
        self.assertEqual(list(binds), ["F11", "F10", "F01", "F00"])
        self.assertEqual(binds, {"F11": (), "F10": (15,), "F01": (2,), "F00": (2, 15)})
        with self.assertRaises(ValueError):
            phase_l.factorial_state_bindings(4, 4)

    def test_single_suppression_reuse_for_all_pairs(self):
        use_count = {p: 0 for p in phase_l.POSITIONS}
        for s, t in phase_l.PAIRS:
            binds = phase_l.factorial_state_bindings(s, t)
            self.assertEqual(binds["F11"], ())
            self.assertEqual(binds["F10"], (t,))
            self.assertEqual(binds["F01"], (s,))
            self.assertEqual(binds["F00"], (s, t))
            use_count[s] += 1
            use_count[t] += 1
        self.assertEqual(set(use_count.values()), {13})
        self.assertEqual(len(use_count), 14)

    def test_real_manifest_binding_and_forward_reuse_identity(self):
        vi = 7
        manifest = [{"video_index": vi, "forward_id": f"v{vi}:baseline", "forward_kind": "original_F11_reused"}]
        manifest += [{"video_index": vi, "forward_id": f"v{vi}:single:{p}",
                      "forward_kind": "single_suppression_reused"} for p in phase_l.POSITIONS]
        states = []
        for s, t in phase_l.PAIRS:
            ids = {"F11": f"v{vi}:baseline", "F10": f"v{vi}:single:{t}",
                   "F01": f"v{vi}:single:{s}", "F00": f"v{vi}:double:{s}:{t}"}
            manifest.append({"video_index": vi, "forward_id": ids["F00"], "forward_kind": "double_suppression"})
            for state, suppress in phase_l.factorial_state_bindings(s, t).items():
                states.append({"video_index": vi, "s": s, "t": t, "state": state,
                               "suppressed_positions": list(suppress), "forward_id": ids[state],
                               "reused_cached_state": state != "F00"})
        phase_l.validate_factorial_manifests(manifest, states, [vi])

    def test_double_suppression_uses_original_tensor_and_is_local(self):
        import torch
        z = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1, 1)
        changed, _, detail = phase_l.apply_temporal_suppressions(z, (2, 3))
        self.assertTrue(detail["outside_positions_exact"])
        self.assertEqual(detail["replacement_max_abs_error"], 0.0)
        self.assertEqual(float(changed[0, 0, 1, 0, 0]), 1.0)
        self.assertEqual(float(changed[0, 0, 2, 0, 0]), 2.0)
        self.assertTrue(torch.equal(changed[:, :, 3:], z[:, :, 3:]))
        self.assertTrue(torch.equal(changed[:, :, :1], z[:, :, :1]))

    def test_additive_synthetic_interaction_is_zero(self):
        base = np.array([2.0, -1.0, 4.0])
        a = np.array([1.0, 3.0, -2.0])
        b = np.array([-4.0, 2.0, 5.0])
        f11, f10, f01, f00 = base + a + b, base + a, base + b, base
        j, ms, mt = phase_l.factorial_contrasts(f11, f10, f01, f00)
        np.testing.assert_array_equal(j, np.zeros_like(j))
        np.testing.assert_allclose(ms, a, atol=0.0)
        np.testing.assert_allclose(mt, b, atol=0.0)

    def test_synthetic_interaction_term_is_recovered(self):
        base = np.array([1.0, 3.0])
        a = np.array([2.0, -2.0])
        b = np.array([-1.0, 4.0])
        c = np.array([7.0, -3.0])
        f11, f10, f01, f00 = base + a + b + c, base + a, base + b, base
        j, _, _ = phase_l.factorial_contrasts(f11, f10, f01, f00)
        np.testing.assert_array_equal(j, c)

    def test_main_effect_arithmetic(self):
        f11 = np.array([9.0, 8.0])
        f10 = np.array([7.0, 5.0])
        f01 = np.array([6.0, 4.0])
        f00 = np.array([2.0, 1.0])
        j, ms, mt = phase_l.factorial_contrasts(f11, f10, f01, f00)
        np.testing.assert_array_equal(j, np.array([-2.0, 0.0]))
        np.testing.assert_array_equal(ms, np.array([4.0, 4.0]))
        np.testing.assert_array_equal(mt, np.array([3.0, 3.0]))

    def test_interaction_energy_arithmetic_and_raw_norm(self):
        j = np.array([3.0, 4.0])
        ms = np.array([1.0, 0.0])
        mt = np.array([0.0, 2.0])
        values = phase_l.interaction_energy(j, ms, mt)
        self.assertEqual(values["interaction_norm"], 5.0)
        self.assertEqual(values["E_I"], 25.0)
        self.assertEqual(values["E_s"], 1.0)
        self.assertEqual(values["E_t"], 4.0)
        self.assertAlmostEqual(values["R"], 25.0 / 30.0)

    def test_r_bounds(self):
        for vals in ((np.zeros(2), np.ones(2), np.ones(2)),
                     (np.ones(3), np.zeros(3), np.zeros(3))):
            value = phase_l.interaction_energy(*vals)["R"]
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_91_pair_order(self):
        self.assertEqual(phase_l.PAIRS[0], (2, 3))
        self.assertEqual(phase_l.PAIRS[-1], (14, 15))
        self.assertEqual(len(set(phase_l.PAIRS)), 91)
        self.assertTrue(all(s < t for s, t in phase_l.PAIRS))

    def test_pair_symmetry_and_zero_diagonal(self):
        vector = np.arange(91, dtype=np.float64)
        matrix = phase_l.symmetric_matrix_from_upper(vector)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(14))
        np.testing.assert_array_equal(matrix[np.triu_indices(14, 1)], vector)

    def test_ten_video_concatenation_order(self):
        vectors = {i: np.full(91, float(i)) for i in range(10)}
        result = phase_l.concat_video_vectors(vectors, list(range(10)))
        self.assertEqual(result.size, 910)
        for i in range(10):
            np.testing.assert_array_equal(result[i * 91:(i + 1) * 91], np.full(91, float(i)))
        reversed_result = phase_l.concat_video_vectors(vectors, [1, 0])
        self.assertEqual(reversed_result[0], 1.0)
        self.assertEqual(reversed_result[91], 0.0)

    def test_simplex_weights_nonnegative(self):
        target = np.array([1.0, 1.0])
        competitors = np.array([[0.0, 1.0], [2.0, 1.0]])
        alpha, _, _ = phase_l.simplex_cover(target, competitors)
        self.assertTrue(np.all(alpha >= 0.0))

    def test_simplex_weights_sum_to_one(self):
        target = np.array([1.0, 1.0])
        competitors = np.array([[0.0, 1.0], [2.0, 1.0]])
        alpha, _, _ = phase_l.simplex_cover(target, competitors)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=12)

    def test_leave_one_out_reconstruction_identity(self):
        a, b = np.array([1.0, 0.0, 2.0]), np.array([0.0, 2.0, 1.0])
        target = 0.25 * a + 0.75 * b
        alpha, residual, delta = phase_l.simplex_cover(target, [a, b])
        np.testing.assert_allclose(alpha, [0.25, 0.75], atol=1e-6)
        np.testing.assert_allclose(residual, np.zeros_like(target), atol=1e-6)
        self.assertLess(delta, 1e-6)

    def test_residual_reshape_identity(self):
        v = np.arange(91, dtype=np.float64)
        matrix = phase_l.symmetric_matrix_from_upper(v)
        pairs = list(phase_l.PAIRS)
        for i, (s, t) in enumerate(pairs):
            self.assertEqual(matrix[s - 2, t - 2], v[i])
            self.assertEqual(matrix[t - 2, s - 2], v[i])


if __name__ == "__main__":
    unittest.main()

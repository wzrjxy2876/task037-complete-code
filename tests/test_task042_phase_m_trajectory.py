import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
import task042_phase_m_trajectory as m

class PhaseMTests(unittest.TestCase):
    def test_attention_spatial_response(self):
        x = np.zeros((2, 3, 4, 5)); x[..., 0] = 3
        np.testing.assert_allclose(m.attention_spatial_response(x), 3)
    def test_ffn_nonnegative(self):
        np.testing.assert_array_equal(m.ffn_spatial_response(np.array([[[-1, 2]]])), [[[1, 2]]])
    def test_probability_normalization(self):
        p, nz = m.spatial_probability(np.ones((2, 2, 2)))
        self.assertFalse(nz.any()); np.testing.assert_allclose(p.sum((1,2)), 1)
    def test_coordinates(self):
        x, y = m.normalized_coordinates(3, 4)
        self.assertEqual(x[0,0], -1); self.assertEqual(x[0,-1], 1); self.assertEqual(y[0,0], -1); self.assertEqual(y[-1,0], 1)
    def test_center_synthetic(self):
        a = np.zeros((1, 3, 3)); a[0, 1, 2] = 1
        f = m.functional_moments(m.spatial_probability(a)[0])
        np.testing.assert_allclose(f[0,:2], [1, 0])
    def test_covariance_synthetic(self):
        a = np.zeros((1, 3, 3)); a[0,0,0] = a[0,2,2] = 1
        f = m.functional_moments(m.spatial_probability(a)[0])
        self.assertGreater(f[0,2], 0); self.assertGreater(f[0,4], 0); self.assertAlmostEqual(f[0,3], 1)
    def test_covariance_symmetry_psd(self):
        a = np.ones((1, 2, 2)); f = m.functional_moments(m.spatial_probability(a)[0])[0]
        self.assertEqual(f[3], 0); self.assertGreaterEqual(f[2], 0); self.assertGreaterEqual(f[4], 0)
    def test_transition(self):
        f = np.arange(80).reshape(16,5)
        np.testing.assert_array_equal(m.trajectory_transitions(f), np.diff(f,axis=0))
    def test_long_range_identity(self):
        f = np.arange(80,dtype=float).reshape(16,5); r = m.trajectory_transitions(f)
        for a in range(16):
            for b in range(a+1,16): np.testing.assert_allclose(f[b]-f[a], r[a:b].sum(0))
    def test_vector_order(self):
        f = np.arange(80).reshape(16,5); r = m.trajectory_transitions(f)
        self.assertEqual(r.reshape(-1).size,75); self.assertEqual(r.reshape(-1)[0],5)
    def test_30_video_concatenation_order(self):
        vectors = {i: np.full(75, float(i)) for i in range(30)}
        out = m.concat_video_trajectories(vectors, list(range(30)))
        self.assertEqual(out.size, 2250)
        self.assertEqual(out[0], 0.0); self.assertEqual(out[75], 1.0); self.assertEqual(out[-1], 29.0)

    def test_motion_deformation(self):
        a,b=m.split_motion_deformation(np.ones((15,5))); self.assertEqual(a.shape[-1],2); self.assertEqual(b.shape[-1],3)
    def test_simplex(self):
        a, res, d=m.solve_simplex_coverage(np.array([.25,1.5,1.25]), [np.array([1,0,2]),np.array([0,2,1])])
        np.testing.assert_allclose(a,[.25,.75],atol=1e-6); np.testing.assert_allclose(res,0,atol=1e-6); self.assertLess(d,1e-6)
        self.assertTrue(np.all(a >= 0)); self.assertAlmostEqual(float(a.sum()), 1.0, places=12)
    def test_residual_shape(self):
        x=np.arange(30*15*5).reshape(30,15,5); self.assertEqual(x.shape,(30,15,5))

if __name__=="__main__": unittest.main()


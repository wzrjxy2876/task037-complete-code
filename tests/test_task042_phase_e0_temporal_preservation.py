import sys
import unittest
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
_runtime = _here if (_here / "task042_phase_e0_temporal_preservation.py").is_file() else _here.parent / "src" / "lgfr_runtime"
sys.path.insert(0, str(_runtime))
import task042_phase_e0_temporal_preservation as phase_e0


class GatePrimitiveTests(unittest.TestCase):
    def test_gate_state_is_nontrainable_and_validated(self):
        state = phase_e0.GateState([10, 20])
        self.assertFalse(any(isinstance(x, torch.nn.Parameter)
                            for x in vars(state).values()))
        with state.temporary(10, 0.5):
            self.assertEqual(state.values, {10: 0.5, 20: 1.0})
        self.assertEqual(state.values, {10: 1.0, 20: 1.0})
        with self.assertRaises(phase_e0.PhaseE0Error):
            state.set_one(10, 1.01)

    def test_attention_gate_scales_one_preprojection_head_exactly(self):
        torch.manual_seed(3)
        attn = torch.rand(2, 3, 5, 5, dtype=torch.float64)
        value = torch.rand(2, 3, 5, 4, dtype=torch.float64)
        baseline_head_output = torch.matmul(attn, value)
        gated_attn = phase_e0.scale_attention_probabilities(attn, {1: 0.25})
        gated_head_output = torch.matmul(gated_attn, value)
        self.assertTrue(torch.equal(gated_head_output[:, 0], baseline_head_output[:, 0]))
        self.assertTrue(torch.allclose(gated_head_output[:, 1],
                                       0.25 * baseline_head_output[:, 1]))
        self.assertTrue(torch.equal(gated_head_output[:, 2], baseline_head_output[:, 2]))

    def test_ffn_gate_scales_one_post_gelu_neuron(self):
        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        gated = phase_e0.scale_ffn_activations(x, {2: 0.5})
        self.assertTrue(torch.equal(gated[..., 0], x[..., 0]))
        self.assertTrue(torch.equal(gated[..., 1], x[..., 1]))
        self.assertTrue(torch.equal(gated[..., 2], 0.5 * x[..., 2]))
        self.assertTrue(torch.equal(gated[..., 3], x[..., 3]))

    def test_gate_restoration_returns_exact_identity_output(self):
        state = phase_e0.GateState([7])
        x = torch.randn(2, 3, 4)
        baseline = phase_e0.scale_ffn_activations(x, {1: state.values[7]})
        with state.temporary(7, 0.0):
            removed = phase_e0.scale_ffn_activations(x, {1: state.values[7]})
            self.assertTrue(torch.equal(removed[..., 1], torch.zeros_like(removed[..., 1])))
        restored = phase_e0.scale_ffn_activations(x, {1: state.values[7]})
        self.assertTrue(torch.equal(restored, baseline))


class RelationLossTests(unittest.TestCase):
    def test_relation_distance_is_symmetric(self):
        x = torch.tensor([0.0, 0.4, 1.0, 1.6, 2.0])
        y = torch.tensor([1.0, 0.8, 0.1, 0.7, 1.9])
        self.assertTrue(torch.allclose(phase_e0.relation_distance(x, y),
                                       phase_e0.relation_distance(y, x)))

    def test_constant_pearson_stabilization_is_finite(self):
        x = torch.ones(5, requires_grad=True)
        y = torch.ones(5) * 2.0
        distance = phase_e0.relation_distance(x, y)
        self.assertTrue(torch.isfinite(distance).item())
        self.assertAlmostEqual(float(distance.item()), 0.5, places=6)
        distance.backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_teacher_detached_and_student_differentiable(self):
        scale = torch.tensor(0.7, requires_grad=True)
        student = torch.stack([
            torch.stack([scale, 1.0 + scale, scale * 0.0 + 3.0,
                         scale * 0.0 + 4.5, scale * 0.0 + 5.0]),
            torch.tensor([0.0, 1.2, 2.0, 4.0, 6.0]),
        ]).unsqueeze(0)
        teacher = torch.tensor([[[0.1, 1.0, 2.8, 4.2, 5.4],
                                 [0.2, 1.4, 2.1, 3.7, 6.1]]])
        self.assertFalse(teacher.requires_grad)
        loss, details = phase_e0.relation_loss(student, teacher,
                                               ["same", "same"], [True, True], [11, 12])
        self.assertEqual(len(details), 1)
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad).item())
        self.assertGreater(abs(float(scale.grad.item())), 0.0)
        self.assertIsNone(teacher.grad)

    def test_memory_bounded_chain_rule_vjp_matches_direct_loss_gradient(self):
        teacher = torch.tensor([[[0.1, 0.8, 2.6, 4.3, 5.2],
                                 [0.3, 1.7, 2.1, 4.4, 6.0],
                                 [1.3, 0.5, 2.8, 3.1, 5.7]]])
        base = torch.tensor([[[0.0, 1.0, 2.0, 4.0, 5.0],
                              [0.0, 1.5, 2.0, 3.0, 6.0],
                              [1.0, 0.4, 2.5, 3.2, 5.3]]])
        direct = base.clone().requires_grad_(True)
        loss, _ = phase_e0.relation_loss(direct, teacher,
                                         ["d", "d", "d"], [True, True, True], [1, 2, 3])
        direct_gradient = torch.autograd.grad(loss, direct)[0]
        meta = base.clone().requires_grad_(True)
        meta_loss, _ = phase_e0.relation_loss(meta, teacher,
                                              ["d", "d", "d"], [True, True, True], [1, 2, 3])
        coefficients = torch.autograd.grad(meta_loss, meta)[0].detach()
        differentiable_student = base.clone().requires_grad_(True)
        vjp_surrogate = (coefficients * differentiable_student).sum()
        vjp_gradient = torch.autograd.grad(vjp_surrogate, differentiable_student)[0]
        self.assertTrue(torch.allclose(vjp_gradient, direct_gradient, atol=1e-7, rtol=1e-6))

    def test_zero_gate_excludes_removed_unit_but_trains_survivors(self):
        student = torch.tensor([[[0.0, 0.1, 9.0, 0.2, 0.3],
                                 [0.0, 1.0, 2.0, 4.0, 6.0],
                                 [1.0, 0.2, 0.6, 2.0, 5.0]]], requires_grad=True)
        teacher = torch.tensor([[[8.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.2, 0.8, 2.2, 3.5, 6.2],
                                 [1.2, 0.3, 0.4, 2.5, 4.7]]])
        loss, details = phase_e0.relation_loss(student, teacher,
                                               ["d", "d", "d"],
                                               [False, True, True], [1, 2, 3])
        self.assertEqual(len(details), 1)
        self.assertEqual((details[0]["unit_i"], details[0]["unit_j"]), (2, 3))
        loss.backward()
        self.assertTrue(torch.equal(student.grad[0, 0], torch.zeros(5)))
        self.assertGreater(float(student.grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(student.grad[0, 2].abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(student.grad).all().item())


if __name__ == "__main__":
    unittest.main()

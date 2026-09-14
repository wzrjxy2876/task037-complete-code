import unittest

from scripts.task042_phase_e1_progressive_pilot import (
    FrozenLossScale,
    GATE_STAGES,
    median_gradient_calibration,
    normalized_trapezoid_auc,
    validate_gate_schedule,
    validate_identical_candidate_sets,
)


class PhaseE1ProtocolTests(unittest.TestCase):
    def test_multi_batch_scale_calibration_is_deterministic(self):
        ce = [2.0, 4.0, 3.0, 5.0]
        tr = [10.0, 20.0, 15.0, 25.0]
        first = median_gradient_calibration(ce, tr)
        second = median_gradient_calibration(list(ce), list(tr))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 3.5)
        self.assertAlmostEqual(first[1], 17.5)
        self.assertAlmostEqual(first[2], 0.2)

    def test_scale_rejects_single_batch_or_zero_relation_gradient(self):
        with self.assertRaises(RuntimeError):
            median_gradient_calibration([1.0], [2.0])
        with self.assertRaises(RuntimeError):
            median_gradient_calibration([1.0, 2.0], [0.0, 0.0])

    def test_lambda_scale_is_immutable(self):
        scale = FrozenLossScale(0.125)
        self.assertEqual(scale.value, 0.125)
        with self.assertRaises(AttributeError):
            scale._value = 0.25

    def test_gate_schedule_is_exact(self):
        validate_gate_schedule((1.0, 0.75, 0.5, 0.25, 0.0))
        self.assertEqual(GATE_STAGES, (1.0, 0.75, 0.5, 0.25, 0.0))
        with self.assertRaises(RuntimeError):
            validate_gate_schedule((1.0, 0.5, 0.0))

    def test_final_candidate_identity_and_order_are_shared_across_arms(self):
        ids = [30185, 33306, 16627, 411]
        validate_identical_candidate_sets(ids, {
            'A_ONE_SHOT': ids[:], 'B_PROGRESSIVE_CE': ids[:],
            'C_PROGRESSIVE_CE_TR': ids[:],
        })
        with self.assertRaises(RuntimeError):
            validate_identical_candidate_sets(ids, {
                'A_ONE_SHOT': ids[:], 'B_PROGRESSIVE_CE': ids[::-1],
                'C_PROGRESSIVE_CE_TR': ids[:],
            })

    def test_equal_stage_update_budget_is_four_equal_noninitial_stages(self):
        r = 30
        per_arm = [r for _ in GATE_STAGES[1:]]
        self.assertEqual(per_arm, [30, 30, 30, 30])
        self.assertEqual(sum(per_arm), 120)

    def test_recovery_auc_uses_normalized_matched_step_interval(self):
        self.assertAlmostEqual(normalized_trapezoid_auc([0, 1, 2], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(normalized_trapezoid_auc([0, 2], [0, 2]), 1.0)
        with self.assertRaises(RuntimeError):
            normalized_trapezoid_auc([0, 0], [1, 2])


if __name__ == '__main__':
    unittest.main()

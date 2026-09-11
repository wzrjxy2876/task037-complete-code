from __future__ import annotations

import unittest

import torch

from task040_htor_core import (
    apply_temporal_intervention,
    apply_temporal_interventions,
    compute_htor,
    compute_level_rms,
    compute_tau,
    enumerate_hierarchical_interventions,
    validate_temporal_length,
    verify_intervention_identity,
)


class TestTask040Core(unittest.TestCase):
    def test_intervention_count_and_identity(self) -> None:
        for temporal_length in (2, 4, 8, 16, 32):
            with self.subTest(temporal_length=temporal_length):
                specs = enumerate_hierarchical_interventions(temporal_length)
                self.assertEqual(len(specs), temporal_length - 1)
                verify_intervention_identity(specs, temporal_length)
                self.assertEqual(len({spec.permutation for spec in specs}), temporal_length - 1)

    def test_power_of_two_validation(self) -> None:
        for temporal_length in (0, 1, 3, 6, 12):
            with self.subTest(temporal_length=temporal_length):
                with self.assertRaisesRegex(ValueError, "power of two"):
                    validate_temporal_length(temporal_length)

    def test_block_order_and_frame_identity(self) -> None:
        specs = enumerate_hierarchical_interventions(8)
        level_one = next(spec for spec in specs if spec.block_size == 2 and spec.pair_index == 0)
        clip = torch.arange(8)
        transformed = apply_temporal_intervention(clip, level_one, time_dim=0)
        self.assertEqual(transformed.tolist(), [2, 3, 0, 1, 4, 5, 6, 7])
        self.assertEqual(sorted(transformed.tolist()), list(range(8)))

    def test_tau_formula_and_range(self) -> None:
        self.assertEqual(compute_tau(1.0, 1.0).item(), 0.0)
        self.assertAlmostEqual(compute_tau(1.0, 0.0).item(), 1.0, places=10)
        self.assertAlmostEqual(compute_tau(1.0, -1.0).item(), 1.0, places=10)
        self.assertEqual(compute_tau(0.0, 0.0).item(), 0.0)
        values = compute_tau(torch.tensor([-2.0, 0.0, 2.0]), torch.tensor([2.0, 0.0, -2.0]))
        self.assertTrue(bool(torch.all((values >= 0.0) & (values <= 1.0))))

    def test_rms_and_equal_level_weighting(self) -> None:
        self.assertAlmostEqual(compute_level_rms(torch.tensor([1.0, 0.0])).item(), 2**-0.5)
        # The lower level has many zeros and the higher level has one large value.
        # HTOR averages the two level RMS values, not all raw interventions together.
        result = compute_htor({
            0: compute_level_rms([0.0, 0.0, 0.0, 0.0]),
            1: compute_level_rms([1.0]),
        })
        self.assertAlmostEqual(result.item(), 2**-0.5)

    def test_interventions_are_deterministic(self) -> None:
        self.assertEqual(
            enumerate_hierarchical_interventions(16),
            enumerate_hierarchical_interventions(16),
        )

    def test_batched_temporal_interventions_match_single_path(self) -> None:
        for temporal_length in (8, 32):
            clip = torch.arange(3 * temporal_length * 2 * 2, dtype=torch.float32).reshape(
                3, temporal_length, 2, 2
            )
            specs = enumerate_hierarchical_interventions(temporal_length)
            selected = specs[: min(5, len(specs))]
            batched = apply_temporal_interventions(clip, selected, time_dim=1)
            self.assertEqual(
                tuple(batched.shape),
                (len(selected), 3, temporal_length, 2, 2),
            )
            self.assertEqual(batched.dtype, clip.dtype)
            self.assertEqual(batched.device, clip.device)
            for index, spec in enumerate(selected):
                self.assertTrue(
                    torch.equal(
                        batched[index],
                        apply_temporal_intervention(clip, spec, time_dim=1),
                    )
                )


if __name__ == "__main__":
    unittest.main()


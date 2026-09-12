from __future__ import annotations

import unittest

from task041_phase_g_n9_validation import (
    interaction,
    n9_gate,
    signature_row,
    validate_n9_subset,
)


def manifest_row(index: int, class_name: str, label: int, within_class: int) -> dict[str, object]:
    return {
        "video_index": index,
        "dataset_index": 1000 + index,
        "video_id": f"/data/UCF101_Frame/frames/{class_name}/v_{class_name}_g01_c{within_class:02d}",
        "duration": 20 + within_class,
        "label": label,
    }


class TestTask041PhaseGN9(unittest.TestCase):
    def setUp(self) -> None:
        self.n9 = [
            manifest_row(0, "HighJump", 39, 1),
            manifest_row(1, "HighJump", 39, 2),
            manifest_row(2, "HighJump", 39, 3),
            manifest_row(3, "Mixing", 53, 1),
            manifest_row(4, "Mixing", 53, 2),
            manifest_row(5, "Mixing", 53, 3),
            manifest_row(6, "Rafting", 72, 1),
            manifest_row(7, "Rafting", 72, 2),
            manifest_row(8, "Rafting", 72, 3),
        ]
        self.n3 = [self.n9[0], self.n9[3], self.n9[6]]

    def test_exact_n3_manifest_subset_passes(self) -> None:
        validate_n9_subset(self.n9, self.n3)

    def test_subset_with_changed_dataset_identity_fails(self) -> None:
        changed = [dict(row) for row in self.n3]
        changed[0]["dataset_index"] = 9999
        with self.assertRaises(RuntimeError):
            validate_n9_subset(self.n9, changed)

    def test_manifest_requires_three_videos_per_class(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_n9_subset(self.n9[:-1], self.n3)

    def test_predeclared_gate_passes_only_when_all_four_conditions_pass(self) -> None:
        passed, checks = n9_gate(5, 0.01, 5, 5)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))

    def test_predeclared_gate_is_not_weakened(self) -> None:
        cases = [
            (4, 0.1, 7, 7),
            (5, 0.0, 7, 7),
            (5, 0.1, 4, 7),
            (5, 0.1, 7, 4),
            (7, None, 7, 7),
        ]
        for values in cases:
            with self.subTest(values=values):
                passed, checks = n9_gate(*values)
                self.assertFalse(passed)
                self.assertFalse(all(checks.values()))

    def test_raw_interaction_uses_frozen_signed_order(self) -> None:
        self.assertEqual(interaction(0.25, 0.05, 0.10, 0.02), 0.12)

    def test_signature_keeps_span_and_canonical_dimension_identity(self) -> None:
        unit = {
            "candidate_task037_global_index": "6",
            "candidate_task040_global_index": "285",
            "candidate_layer_name": "layers.0.blocks.0.mlp",
            "candidate_unit_type": "neuron",
            "candidate_unit_index": "3",
            "candidate_stage": "0",
            "domain_id": "113",
        }
        row = signature_row(unit, 4, 5, "/data/frames/Mixing/v_Mixing_g01_c01", 2, 0.125)
        self.assertEqual(row["level"], 2)
        self.assertEqual(row["dimension_index"], 82)
        self.assertEqual(row["C_signed"], 0.125)


if __name__ == "__main__":
    unittest.main()

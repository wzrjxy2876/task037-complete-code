from __future__ import annotations

import unittest

import numpy as np

from task041_phase_e_signed_interaction import (
    kendall_tau_b,
    make_span_statistics,
    parse_raw,
    spearman,
)


class TestTask041PhaseESignedInteraction(unittest.TestCase):
    def test_reconstructs_signed_interaction_from_four_logits(self) -> None:
        frozen = {
            ("7", "layers.0.blocks.0.mlp", "neuron", "3"): {
                "candidate_task037_global_index": "70",
                "candidate_task040_global_index": "7",
                "candidate_layer_name": "layers.0.blocks.0.mlp",
                "candidate_unit_type": "neuron",
                "candidate_unit_index": 3,
                "candidate_stage": 0,
                "domain_id": "102",
            }
        }
        raw = {
            "unit_global_index": "7",
            "layer_name": "layers.0.blocks.0.mlp",
            "unit_type": "neuron",
            "unit_index": "3",
            "stage": "0",
            "block_size": "4",
            "pair_index": "2",
            "level": "2",
            "video_index": "0",
            "video_id": "clip-0",
            "z_true_original": "2.0",
            "z_true_original_masked": "0.5",
            "z_true_intervened": "1.0",
            "z_true_intervened_masked": "0.2",
            "C_interaction": "0.7",
        }
        row = parse_raw(raw, "phase_c", frozen)
        self.assertIsNotNone(row)
        self.assertEqual(row["C_signed"], 0.7)

    def test_rms_keeps_magnitude_while_signed_mean_cancels(self) -> None:
        records = []
        base = {
            "candidate_task037_global_index": "70",
            "candidate_task040_global_index": "7",
            "candidate_layer_name": "layers.0.blocks.0.mlp",
            "candidate_unit_type": "neuron",
            "candidate_unit_index": 3,
            "candidate_stage": 0,
            "domain_id": "102",
        }
        for pair_index in range(48):
            records.append({
                **base,
                "span": 1,
                "pair_index": pair_index,
                "C_signed": 2.0 if pair_index < 24 else -2.0,
            })
        row = make_span_statistics(records)[0]
        self.assertEqual(row["mu"], 0.0)
        self.assertEqual(row["mean_abs"], 2.0)
        self.assertEqual(row["rms"], 2.0)
        self.assertEqual(row["sign_balance"], 0.0)
        self.assertEqual(row["interaction_count"], 48)

    def test_rank_statistics_handle_direction_and_ties(self) -> None:
        self.assertEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(kendall_tau_b([1, 1], [2, 2]))


if __name__ == "__main__":
    unittest.main()

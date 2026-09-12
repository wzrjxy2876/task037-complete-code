import json
import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "lgfr_runtime"
sys.path.insert(0, str(SOURCE_ROOT))

import task041_phase_h_temporal_stress as phase_h
import task041_phase_h1_safest_semantics_audit as phase_h1


class SafestAndHarmfulSemanticsTests(unittest.TestCase):
    def test_safest_is_maximum_win_rate_with_numeric_id_tiebreak(self):
        rows = [
            {"candidate_task037_global_index": "12", "W": 0.8},
            {"candidate_task037_global_index": "2", "W": 0.8},
            {"candidate_task037_global_index": "1", "W": 0.4},
        ]
        self.assertEqual(phase_h1.safest_uid(rows, "W"), "2")

    def test_predicted_most_harmful_is_minimum_win_rate(self):
        rows = [
            {"candidate_task037_global_index": "10", "W": 0.9},
            {"candidate_task037_global_index": "4", "W": 0.1},
            {"candidate_task037_global_index": "2", "W": 0.1},
        ]
        self.assertEqual(phase_h1.most_harmful_uid(rows, "W"), "2")

    def test_fullval_extremes_keep_signed_ce_and_numeric_ties(self):
        rows = [
            {"candidate_task037_global_index": "8", "fullval_mean_cross_entropy_increase": -0.1},
            {"candidate_task037_global_index": "3", "fullval_mean_cross_entropy_increase": -0.1},
            {"candidate_task037_global_index": "2", "fullval_mean_cross_entropy_increase": 0.4},
            {"candidate_task037_global_index": "1", "fullval_mean_cross_entropy_increase": 0.4},
        ]
        self.assertEqual(phase_h1.fullval_safest_uid(rows), "3")
        self.assertEqual(phase_h1.fullval_most_harmful_uid(rows), "1")

    def test_correction_and_regression_definitions_are_exact(self):
        self.assertEqual(phase_h1.correction_flags("1", "3", "3"), (True, False))
        self.assertEqual(phase_h1.correction_flags("3", "1", "3"), (False, True))
        self.assertEqual(phase_h1.correction_flags("3", "3", "3"), (False, False))

    def test_safe_to_harmful_order_uses_descending_w_then_numeric_id(self):
        rows = [
            {"candidate_task037_global_index": "12", "W": 0.5},
            {"candidate_task037_global_index": "3", "W": 0.8},
            {"candidate_task037_global_index": "2", "W": 0.5},
        ]
        self.assertEqual(phase_h1._risk_order(rows, "W"), ["3", "2", "12"])


class FrozenClassPoolTests(unittest.TestCase):
    def test_n3_n6_n9_members_all_stay_inside_one_frozen_three_class_pool(self):
        manifest = [
            {
                "video_index": str(index),
                "dataset_index": str(index + 100),
                "video_id": f"/dataset/{class_name}/v_{class_name}_g01_c{index:02d}",
                "canonical_video_id": f"v_{class_name}_g01_c{index:02d}",
                "label": str(label),
                "phase_f_n3_member": str(index in (0, 3, 6)),
            }
            for label, class_name, start in ((10, "ActionA", 0), (20, "ActionB", 3), (30, "ActionC", 6))
            for index in range(start, start + 3)
        ]
        subsets = [
            {"subset_id": "n3_phase_f", "video_count": "3", "video_indices_json": "[0, 3, 6]"},
            {"subset_id": "n9_full", "video_count": "9", "video_indices_json": "[0,1,2,3,4,5,6,7,8]"},
        ]
        subsets.extend(
            {"subset_id": row["subset_id"], "video_count": "6",
             "video_indices_json": json.dumps(row["video_indices"])}
            for row in phase_h.class_balanced_n6_subsets(manifest)
        )
        video_rows, summary = phase_h1._validate_class_coverage(manifest, subsets)
        self.assertEqual(summary["unique_action_class_count"], 3)
        self.assertEqual(summary["n6_subset_count"], 27)
        self.assertTrue(summary["all_stability_subsets_within_same_n9_pool"])
        self.assertEqual(len([row for row in video_rows if row["row_type"] == "video"]), 9)
        self.assertIn("does NOT establish cross-action-class", summary["limitation"])


if __name__ == "__main__":
    unittest.main()

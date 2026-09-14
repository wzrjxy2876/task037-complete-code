import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
from task040_htor_probe import temporary_unit_mask
from task042_phase_c_joint_mask import joint_temporary_unit_masks
from task042_phase_d_progressive_path import (
    DOMAIN_IDS,
    attention_head_parameter_cost,
    build_checkpoint_manifest,
    build_progressive_paths,
    candidate_provenance,
    coverage_j,
    distance_index,
    f3_order_key,
    ffn_neuron_parameter_cost,
    gamma_cost,
    gamma_tie_key,
    marginal_coverage_cost,
    mask_set_id,
)


class IdentityUnits(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.num_units = width

    def forward(self, values):
        return values


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = IdentityUnits(4)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, values):
        return self.units(values) * self.scale


class Task042PhaseDProgressivePathTests(unittest.TestCase):
    def test_authoritative_f3_direction_and_global_id_tiebreak(self):
        rows = [
            {"R_F3": 0.2, "p_total": 0.4, "p_average": 0.3, "global_index": 9},
            {"R_F3": 0.2, "p_total": 0.4, "p_average": 0.3, "global_index": 3},
            {"R_F3": 0.1, "p_total": 0.9, "p_average": 0.2, "global_index": 8},
        ]
        self.assertEqual(sorted(rows, key=f3_order_key), [rows[2], rows[1], rows[0]])

    def test_exact_temporal_coverage_and_marginal_loss(self):
        ids = [1, 2, 3]
        distances = distance_index([
            {"domain_id": "11", "task037_global_index_i": 1, "task037_global_index_j": 2, "d_temp": 2.0},
            {"domain_id": "11", "task037_global_index_i": 1, "task037_global_index_j": 3, "d_temp": 8.0},
            {"domain_id": "11", "task037_global_index_i": 2, "task037_global_index_j": 3, "d_temp": 5.0},
        ])
        self.assertEqual(float(coverage_j("11", ids, ids, distances)), 0.0)
        self.assertEqual(float(coverage_j("11", ids, [1, 3], distances)), 2.0)
        before, after, delta = marginal_coverage_cost("11", ids, [1, 3], 3, distances)
        self.assertEqual((float(before), float(after), float(delta)), (2.0, 8.0, 6.0))

    def test_exact_attention_ffn_parameter_cost_and_gamma(self):
        self.assertEqual(attention_head_parameter_cost(96, 32, True), 12384)
        self.assertEqual(attention_head_parameter_cost(96, 32, False), 12288)
        self.assertEqual(ffn_neuron_parameter_cost(384, 96, True), 481)
        self.assertEqual(ffn_neuron_parameter_cost(384, 96, False), 480)
        self.assertEqual(float(gamma_cost(6.0, 3)), 2.0)

    def test_progressive_tie_break_is_numeric_and_paths_share_final_set(self):
        units, trace, distances_rows = [], [], []
        gid = 1000
        for step, domain in enumerate(DOMAIN_IDS, 1):
            candidate, representative = gid, gid + 1
            gid += 2
            for index, unit_id in enumerate((candidate, representative)):
                units.append({"domain_id": domain, "task037_global_index": unit_id,
                              "unit_type": "ffn_neuron", "layer": "layers.0.mlp",
                              "unit_index": index, "stage": "stage_0"})
            trace.append({"step": step, "global_index": candidate, "domain_id": domain,
                          "unit_type": "ffn_neuron", "layer": "layers.0.mlp",
                          "unit_index": 0, "parameter_cost": 1, "R_F3": 0.1,
                          "p_total": 0.2, "p_average": 0.3, "domain_damage": 0.0})
            distances_rows.append({"domain_id": domain, "task037_global_index_i": candidate,
                                   "task037_global_index_j": representative, "d_temp": 1.0})
        provenance, removals, candidates = candidate_provenance(units, trace)
        distances = distance_index(distances_rows)
        baseline, temporal, proposals = build_progressive_paths(
            units, provenance, removals, candidates, distances, 100)
        self.assertEqual(len(baseline), len(DOMAIN_IDS))
        self.assertEqual(len(temporal), len(DOMAIN_IDS))
        self.assertEqual(baseline[0]["domain_id"], "11")
        self.assertEqual(temporal[0]["domain_id"], "11")
        self.assertEqual(set(baseline[-1]["masked_task037_ids"].strip("[]").split(",")),
                         set(temporal[-1]["masked_task037_ids"].strip("[]").split(",")))
        self.assertEqual(set(json_ids(baseline[-1]["surviving_ids_by_domain"])),
                         set(json_ids(temporal[-1]["surviving_ids_by_domain"])))
        self.assertEqual(len(proposals), len(DOMAIN_IDS) * (len(DOMAIN_IDS) + 1) // 2)
        manifest, evaluations = build_checkpoint_manifest(
            baseline, temporal, units, removals, distances, len(DOMAIN_IDS))
        self.assertEqual(len(manifest), 8 + 2 * 6)
        self.assertTrue(all(row["achieved_removed_parameters"] <= row["requested_parameter_budget"]
                            for row in manifest if row["checkpoint_kind"] == "PRIMARY_BUDGET"))
        self.assertEqual(mask_set_id([3, 1, 2]), mask_set_id([2, 3, 1]))
        self.assertGreaterEqual(len(evaluations), 1)

    def test_joint_mask_restoration_is_exact(self):
        model = Fixture().eval()
        spec = SimpleNamespace(name="layers.0.mlp", unit_type="neuron", num_units=4,
                               module=model.units, hook_module=model.units)
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
        before = model(values).detach().clone()
        version = int(model.scale._version)
        hooks = tuple(model.units._forward_pre_hooks.items())
        record = {"task037_global_index": 77, "unit_index": 2, "spec": spec}
        with joint_temporary_unit_masks(model, [record], temporary_unit_mask, torch) as calls:
            masked = model(values)
            self.assertTrue(torch.equal(masked[:, 2], torch.zeros_like(masked[:, 2])))
        self.assertTrue(torch.equal(model(values), before))
        self.assertEqual(int(model.scale._version), version)
        self.assertEqual(tuple(model.units._forward_pre_hooks.items()), hooks)
        self.assertEqual(len(calls), 1)


def json_ids(value):
    import json
    return json.loads(value)


if __name__ == "__main__":
    unittest.main()

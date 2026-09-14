import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
from task040_htor_probe import temporary_unit_mask
from task042_phase_c_joint_mask import (
    JointMaskError,
    build_pair_evaluations,
    deterministic_mask_id,
    joint_temporary_unit_masks,
)


class IdentityUnits(nn.Module):
    def forward(self, values):
        return values


class AttentionFixture(nn.Module):
    def __init__(self, heads=3, head_dim=2):
        super().__init__()
        self.units = IdentityUnits()
        self.units.num_units = heads
        self.units.head_dim = head_dim
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, values):
        return self.units(values) * self.scale


class FFNFixture(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.units = IdentityUnits()
        self.units.num_units = width
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, values):
        return self.units(values) * self.scale


class MixedFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttentionFixture()
        self.ffn = FFNFixture()

    def forward(self, values):
        return self.attn(values[..., :6]), self.ffn(values[..., 6:])


def make_spec(module, name, unit_type, num_units):
    return SimpleNamespace(name=name, unit_type=unit_type, num_units=num_units,
                           module=module.units, hook_module=module.units)


def record(uid, spec, index):
    return {"task037_global_index": uid, "spec": spec, "unit_index": index}


class Task042PhaseCJointMaskTests(unittest.TestCase):
    def run_joint(self, model, records, inputs):
        before = model(inputs)
        with joint_temporary_unit_masks(model, records, temporary_unit_mask, torch):
            masked = model(inputs)
        after = model(inputs)
        if isinstance(before, tuple):
            for a, b in zip(before, after):
                self.assertTrue(torch.equal(a, b))
        else:
            self.assertTrue(torch.equal(before, after))
        return before, masked

    def test_multi_attention_heads_masked_together_and_restored(self):
        model = AttentionFixture().eval()
        spec = make_spec(model, "layers.0.attn", "head", 3)
        values = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1
        before, masked = self.run_joint(model, [record(10, spec, 0), record(11, spec, 2)], values)
        self.assertTrue(torch.equal(masked[:, :2], torch.zeros_like(masked[:, :2])))
        self.assertTrue(torch.equal(masked[:, 4:6], torch.zeros_like(masked[:, 4:6])))
        self.assertTrue(torch.equal(masked[:, 2:4], before[:, 2:4]))

    def test_multi_ffn_neurons_masked_together_and_restored(self):
        model = FFNFixture().eval()
        spec = make_spec(model, "layers.0.mlp", "neuron", 4)
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
        before, masked = self.run_joint(model, [record(21, spec, 1), record(22, spec, 3)], values)
        self.assertTrue(torch.equal(masked[:, 1], torch.zeros_like(masked[:, 1])))
        self.assertTrue(torch.equal(masked[:, 3], torch.zeros_like(masked[:, 3])))
        self.assertTrue(torch.equal(masked[:, [0, 2]], before[:, [0, 2]]))

    def test_mixed_attention_and_ffn_masks_and_restores(self):
        model = MixedFixture().eval()
        attn = make_spec(model.attn, "layers.0.attn", "head", 3)
        ffn = make_spec(model.ffn, "layers.0.mlp", "neuron", 4)
        values = torch.arange(20, dtype=torch.float32).reshape(2, 10) + 1
        before, masked = self.run_joint(model, [record(31, attn, 1), record(32, ffn, 2)], values)
        self.assertTrue(torch.equal(masked[0][:, 2:4], torch.zeros_like(masked[0][:, 2:4])))
        self.assertTrue(torch.equal(masked[1][:, 2], torch.zeros_like(masked[1][:, 2])))
        self.assertTrue(torch.equal(masked[0][:, :2], before[0][:, :2]))
        self.assertTrue(torch.equal(masked[1][:, [0, 1, 3]], before[1][:, [0, 1, 3]]))

    def test_duplicate_mask_rejected_before_hook_mutation(self):
        model = AttentionFixture().eval()
        spec = make_spec(model, "layers.0.attn", "head", 3)
        before = copy.copy(model.units._forward_pre_hooks)
        with self.assertRaisesRegex(JointMaskError, "duplicate-mask"):
            with joint_temporary_unit_masks(model, [record(1, spec, 0), record(1, spec, 1)],
                                           temporary_unit_mask, torch):
                pass
        self.assertEqual(tuple(before.items()), tuple(model.units._forward_pre_hooks.items()))

    def test_deterministic_set_identity_and_temporal_descriptor_manifest(self):
        first = deterministic_mask_id("415", 1, [40, 30])
        self.assertEqual(first, deterministic_mask_id(415, 1, [30, 40]))
        self.assertNotEqual(first, deterministic_mask_id("415", 1, [30, 41]))
        types = {1: "ffn_neuron", 2: "attention_head", 3: "ffn_neuron"}
        pair, evals = build_pair_evaluations("415", "strong_structured", [1, 2, 3], 1,
                                              [1], [2], types, "TEMPORAL_ORIENTATION_UNIQUE")
        self.assertEqual(pair["set_relation"], "A_TEMPORAL_DIFFERS")
        self.assertTrue(pair["informative_directional_pair"])
        self.assertEqual([row["method"] for row in evals], ["temporal", "descriptor"])
        self.assertNotEqual(evals[0]["mask_evaluation_id"], evals[1]["mask_evaluation_id"])
        shared_pair, shared = build_pair_evaluations("415", "strong_structured", [1, 2, 3], 1,
                                                      [1], [1], types, "TEMPORAL_ORIENTATION_UNIQUE")
        self.assertEqual(shared_pair["set_relation"], "B_SETS_EQUAL")
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["shared_methods"], "temporal|descriptor")
        tied_pair, _ = build_pair_evaluations("415", "strong_structured", [1, 2, 3], 1,
                                               [1], [2], types, "TEMPORAL_ORIENTATION_TIED")
        self.assertFalse(tied_pair["informative_directional_pair"])


if __name__ == "__main__":
    unittest.main()

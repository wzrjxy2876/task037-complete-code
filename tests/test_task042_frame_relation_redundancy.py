import csv
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
import task042_frame_relation_redundancy as t42


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv = nn.Linear(4, 12, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, 2, 2).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (2 ** 0.5), dim=-1)
        head_out = torch.matmul(self.attn_drop(weights), v)
        merged = head_out.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(merged)


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 7)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.0)
        self.fc2 = nn.Linear(7, 4)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.mlp = TinyMlp()

    def forward(self, x):
        return self.mlp(self.attn(x))


def fake_task040_core():
    interventions = []
    for level, span in enumerate((1, 2, 4, 8, 16)):
        pair_index = 0
        for q in range(32 // (2 * span)):
            base = 2 * q * span
            for r in range(span):
                a, b = base + r, base + r + span
                perm = list(range(32))
                perm[a], perm[b] = perm[b], perm[a]
                interventions.append(SimpleNamespace(
                    temporal_length=32, level=level, block_size=span,
                    pair_index=pair_index, left_start=a, left_end=a + 1,
                    right_start=b, right_end=b + 1, permutation=tuple(perm)))
                pair_index += 1
    return SimpleNamespace(enumerate_fixed_cardinality_temporal_pairs=lambda _t: interventions)


def synthetic_51_rows():
    profiles, desc, mapping = [], [], []
    for uid in range(51):
        is_head = uid < 22
        kind = "head" if is_head else "neuron"
        task037_type = "attention_head" if is_head else "ffn_neuron"
        stage = uid % 4
        block = uid // 4
        leaf = "attn" if is_head else "mlp"
        layer = "layers.%d.blocks.%d.%s" % (stage, block, leaf)
        local_index = uid % (8 if is_head else 768)
        profile = {"task037_global_index": str(uid), "layer_name": layer,
                   "unit_type": kind, "unit_index": str(local_index),
                   "stage": str(stage), "domain_id": str(uid // 2),
                   "unit_global_index": str(1000 + uid)}
        d = {"global_index": str(uid), "layer": layer, "unit_type": task037_type,
             "unit_index": str(local_index), "D_abs": "0.1", "D_rel": "0.2",
             "D_third": "0.3", "third_descriptor_name": "D_dyn", "D_dyn": "0.3"}
        m = {"global_index": str(uid), "layer": layer, "unit_type": task037_type,
             "unit_index": str(local_index)}
        profiles.append(profile)
        desc.append(d)
        mapping.append(m)
    return profiles, desc, mapping


class Task042Tests(unittest.TestCase):
    def test_attention_capture_is_preconcat_preprojection_head_output(self):
        torch.manual_seed(7)
        model = TinyModel().eval()
        units = [
            {"layer": "attn", "capture_kind": "head", "unit_index": 0, "task037_global_index": 11},
            {"layer": "attn", "capture_kind": "head", "unit_index": 1, "task037_global_index": 12},
        ]
        capture = t42.ActivationCapture(model, units, torch)
        x = torch.randn(1, 5, 4)
        with torch.inference_mode():
            _ = model(x)
        output = capture.values
        qkv = model.attn.qkv(x).reshape(1, 5, 3, 2, 2).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (2 ** 0.5), dim=-1)
        expected = torch.matmul(model.attn.attn_drop(weights), v)
        self.assertTrue(torch.allclose(output[11], expected[:, 0], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(output[12], expected[:, 1], atol=1e-7, rtol=1e-6))
        # The captured shape retains the head dimension and is not the [B,N,C]
        # concatenated tensor or the projected attention output.
        self.assertEqual(tuple(output[11].shape), (1, 5, 2))
        capture.close()

    def test_ffn_capture_is_post_gelu_pre_fc2_neuron(self):
        torch.manual_seed(13)
        model = TinyModel().eval()
        units = [{"layer": "mlp", "capture_kind": "neuron", "unit_index": 4,
                  "task037_global_index": 77}]
        capture = t42.ActivationCapture(model, units, torch)
        x = torch.randn(2, 6, 4)
        with torch.inference_mode():
            _ = model(x)
        expected = model.mlp.act(model.mlp.fc1(model.attn(x)))[..., 4]
        self.assertTrue(torch.allclose(capture.values[77], expected, atol=1e-7, rtol=1e-6))
        self.assertEqual(tuple(capture.values[77].shape), (2, 6))
        capture.close()

    def test_all_51_frozen_unit_identities_join_exactly(self):
        fixture_path = ROOT / "data" / "task042_unit_identity_fixture.csv"
        with fixture_path.open("r", encoding="utf-8-sig", newline="") as f:
            fixture = list(csv.DictReader(f))
        profiles = [{"task037_global_index": r["task037_global_index"],
                     "layer_name": r["layer"], "unit_type": r["profile_type"],
                     "unit_index": r["unit_index"], "stage": r["stage"],
                     "domain_id": r["domain_id"],
                     "unit_global_index": r["task040_unit_global_index"]} for r in fixture]
        desc = [{"global_index": r["task037_global_index"], "layer": r["layer"],
                 "unit_type": r["unit_type"], "unit_index": r["unit_index"],
                 "D_abs": r["D_abs"], "D_rel": r["D_rel"], "D_third": r["D_third"],
                 "third_descriptor_name": r["third_descriptor_name"], "D_dyn": r["D_dyn"]} for r in fixture]
        mapping = [{"global_index": r["task037_global_index"], "layer": r["mapped_layer"],
                    "unit_type": r["mapped_unit_type"], "unit_index": r["mapped_unit_index"]} for r in fixture]
        joined = t42.build_unit_manifest(profiles, desc, mapping)
        self.assertEqual(len(joined), 51)
        self.assertEqual(len({r["task037_global_index"] for r in joined}), 51)
        joined_by_id = {r["task037_global_index"]: r for r in joined}
        for expected in fixture:
            got = joined_by_id[int(expected["task037_global_index"])]
            self.assertEqual(int(expected["task037_global_index"]), got["task037_global_index"])
            self.assertEqual(expected["layer"], got["layer"])
            self.assertEqual(int(expected["unit_index"]), got["unit_index"])
            self.assertEqual(int(expected["stage"]), got["stage"])
            self.assertEqual(expected["domain_id"], got["domain_id"])
            self.assertEqual(expected["mapped_layer"], got["layer"])
            self.assertEqual(expected["mapped_unit_type"], got["unit_type"])
            self.assertEqual(int(expected["mapped_unit_index"]), got["unit_index"])
            self.assertEqual(float(expected["D_third"]), got["D_st"])
            self.assertEqual(expected["third_descriptor_name"], got["third_descriptor_name"])
        with self.assertRaises(RuntimeError):
            t42.build_unit_manifest(profiles[:-1], desc, mapping)

    def test_exact_80_fixed_cardinality_intervention_identities(self):
        rows = t42.fixed_cardinality_identities(fake_task040_core())
        self.assertEqual(len(rows), 80)
        self.assertEqual({s: sum(r["span"] == s for r in rows) for s in t42.SPANS},
                         {1: 16, 2: 16, 4: 16, 8: 16, 16: 16})
        for row in rows:
            self.assertNotEqual(row["frame_a"], row["frame_b"])

    def test_intervention_does_not_change_capture_shape(self):
        model = TinyModel().eval()
        units = [{"layer": "attn", "capture_kind": "head", "unit_index": 0,
                  "task037_global_index": 0},
                 {"layer": "mlp", "capture_kind": "neuron", "unit_index": 1,
                  "task037_global_index": 1}]
        capture = t42.ActivationCapture(model, units, torch)
        x = torch.randn(1, 8, 4)
        with torch.inference_mode():
            _ = model(x)
        baseline_shapes = dict(capture.shapes)
        capture.clear()
        with torch.inference_mode():
            _ = model(x[:, [1, 0, 2, 3, 4, 5, 6, 7], :])
        self.assertEqual(capture.shapes, baseline_shapes)
        capture.close()

    def test_two_gpu_video_shard_merge_is_identity_preserving(self):
        videos = [{"video_index": i, "label": i // 3} for i in range(30)]
        shards = t42.deterministic_video_shards(videos)
        self.assertEqual(len(shards[0]), 15)
        self.assertEqual(len(shards[1]), 15)
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(set(shards[0] + shards[1]), set(range(30)))

    def test_signature_normalization_and_degenerate_detection(self):
        z, mean, std, degenerate = t42.normalize_signature([1.0, 2.0, 3.0])
        self.assertFalse(degenerate)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, (2.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(sum(z) / len(z), 0.0)
        self.assertAlmostEqual(sum(x * x for x in z) / len(z), 1.0)
        z0, _, std0, degenerate0 = t42.normalize_signature([4.0] * 80)
        self.assertTrue(degenerate0)
        self.assertIsNone(z0)
        self.assertLessEqual(std0, 1e-12)

    def test_pair_distance_is_symmetric_and_self_distance_zero(self):
        a, b = [-2.0, -1.0, 1.0, 2.0], [2.0, 1.0, -1.0, -2.0]
        self.assertAlmostEqual(t42.relation_distance(a, b), t42.relation_distance(b, a))
        self.assertEqual(t42.relation_distance(a, a), 0.0)

    def test_nearest_neighbor_ties_break_by_smallest_task037_index(self):
        distances = {(10, 20): 0.25, (10, 15): 0.25, (10, 30): 0.4}
        self.assertEqual(t42.nearest_neighbor([10, 15, 20, 30], distances, 10), (15, 0.25))


if __name__ == "__main__":
    unittest.main(verbosity=2)

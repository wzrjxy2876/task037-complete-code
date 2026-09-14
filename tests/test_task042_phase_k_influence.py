from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import task042_phase_j_conditional as phase_j
import task042_phase_k_influence as phase_k


class PhaseKInfluenceTests(unittest.TestCase):
    def test_temporal_position_identity_and_tubelet_raw_support(self):
        sampled = list(range(2, 34))
        self.assertEqual(phase_k.temporal_position_support(sampled, 1), (2, 3))
        self.assertEqual(phase_k.temporal_position_support(sampled, 8), (16, 17))
        self.assertEqual(phase_k.temporal_position_support(sampled, 16), (32, 33))
        with self.assertRaises(ValueError):
            phase_k.temporal_position_support(sampled, 17)
        with self.assertRaises(ValueError):
            phase_k.temporal_position_support(sampled[:31], 1)

    def test_intervention_axis_identity_and_non_source_locality(self):
        z = torch.arange(1 * 2 * 16 * 3 * 4, dtype=torch.float32).reshape(1, 2, 16, 3, 4)
        changed, innovation, magnitude, detail = phase_k.replace_source_innovation(z, 6)
        self.assertEqual(changed.shape, z.shape)
        self.assertEqual(changed.ndim, 5)
        self.assertEqual(detail["temporal_axis"], 2)
        self.assertTrue(torch.equal(changed[:, :, :6], z[:, :, :6]))
        self.assertTrue(torch.equal(changed[:, :, 7:], z[:, :, 7:]))
        self.assertTrue(torch.equal(changed[:, :, 6], (z[:, :, 5] + z[:, :, 7]) * 0.5))
        expected_innovation = z[:, :, 6] - ((z[:, :, 5] + z[:, :, 7]) * 0.5)
        self.assertTrue(torch.equal(innovation, expected_innovation))
        expected = torch.linalg.vector_norm(innovation) / (torch.linalg.vector_norm(z[:, :, 6]) + phase_k.EPS)
        self.assertTrue(torch.allclose(magnitude, expected))
        self.assertTrue(detail["non_source_exact"])
        self.assertEqual(detail["source_replacement_max_abs_error"], 0.0)

    def test_interior_source_enumeration_and_boundary_exclusion(self):
        self.assertEqual(phase_k.source_positions(16), tuple(range(1, 15)))
        z = torch.zeros((1, 1, 16, 1, 1))
        with self.assertRaises(ValueError):
            phase_k.replace_source_innovation(z, 0)
        with self.assertRaises(ValueError):
            phase_k.replace_source_innovation(z, 15)

    def test_local_background_and_temporal_innovation_arithmetic(self):
        z = torch.tensor([0., 2., 10., 6., 9.]).reshape(1, 1, 5, 1, 1)
        modified, innovation, _, _ = phase_k.replace_source_innovation(z, 2)
        self.assertEqual(modified[0, 0, 2, 0, 0].item(), 4.0)
        self.assertEqual(innovation[0, 0, 0, 0].item(), 6.0)

    def test_noop_exactness_and_tolerance_gate(self):
        baseline = {7: torch.tensor([[1., 2.], [3., 4.]])}
        replay = {7: baseline[7].clone()}
        result = phase_k.no_op_match(baseline, replay)
        self.assertTrue(result["exact_bitwise_all_units"])
        self.assertEqual(result["max_abs_all_units"], 0.0)
        replay[7][0, 0] += 1e-3
        with self.assertRaises(RuntimeError):
            phase_k.no_op_match(baseline, replay)

    def test_attention_target_extraction_preserves_spatial_and_head_dimensions(self):
        geometry = {"window_size": (2, 1, 2), "batch_size": 1,
                    "padded_depth": 2, "padded_height": 1, "padded_width": 2,
                    "shift_size": (0, 0, 0), "depth": 2, "height": 1, "width": 2}
        windows = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        features = phase_k.extract_attention_target_frames(windows, geometry, phase_j)
        self.assertEqual(tuple(features.shape), (2, 6))
        self.assertTrue(torch.equal(features, windows.reshape(2, 6)))

    def test_ffn_target_extraction_preserves_spatial_positions(self):
        activation = torch.arange(1 * 3 * 2 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 2, 4)
        features = phase_k.extract_ffn_target_frames(activation, 2, phase_j)
        self.assertEqual(tuple(features.shape), (3, 4))
        self.assertTrue(torch.equal(features, activation[0, ..., 2].reshape(3, 4)))

    def test_directed_influence_exact_synthetic_value_and_bounds(self):
        original = torch.zeros((5, 2), dtype=torch.float64)
        intervened = original.clone()
        original[3] = torch.tensor([3., 4.])
        intervened[3] = torch.tensor([0., 4.])
        result = phase_k.directed_influence(original, intervened, 2)
        self.assertTrue(torch.isnan(result[2]))
        self.assertAlmostEqual(result[3].item(), 1.0 / 3.0)
        valid = result[torch.isfinite(result)]
        self.assertTrue(bool(torch.all(valid >= 0)))
        self.assertTrue(bool(torch.all(valid <= 1)))

    def test_directed_relation_order_and_lag_identity(self):
        coords = phase_k._relation_coordinates(4)
        self.assertEqual(coords[:3], [(0, 0, 2, 1, 1), (0, 2, 2, 3, 1), (0, 3, 2, 4, 2)])
        self.assertEqual(len(coords), 2 * 3)
        self.assertTrue(all(source_one != target_one for _, _, source_one, target_one, _ in coords))

    def test_asymmetry_arithmetic(self):
        absolute, relative = phase_k.asymmetry(0.8, 0.2)
        self.assertAlmostEqual(absolute, 0.6)
        self.assertAlmostEqual(relative, 0.6)

    def test_cuda_event_elapsed_time_uses_start_then_end(self):
        class Event:
            def __init__(self, milliseconds):
                self.milliseconds = milliseconds
            def elapsed_time(self, other):
                return other.milliseconds - self.milliseconds
        class Cuda:
            @staticmethod
            def synchronize():
                return None
        class FakeTorch:
            cuda = Cuda()
        seconds = phase_k._event_seconds(FakeTorch, [{"start": Event(100.), "end": Event(142.5)}])
        self.assertAlmostEqual(seconds, 0.0425)

    def test_simplex_weights_and_leave_one_out_residual_identity(self):
        target = np.array([1., 0., 0.])
        competitors = np.array([[0., 1., 0.], [0., 0., 1.]])
        alpha, residual, delta = phase_j.solve_simplex_coverage(target, competitors)
        self.assertTrue(np.all(alpha >= 0))
        self.assertAlmostEqual(float(alpha.sum()), 1.0)
        reconstructed = alpha @ competitors
        self.assertTrue(np.allclose(residual, target - reconstructed))
        self.assertAlmostEqual(delta, np.linalg.norm(residual) / (np.linalg.norm(target) + phase_k.EPS))

    def test_residual_reshape_identity(self):
        coordinates = phase_k._relation_coordinates(4)
        residual = np.arange(2 * len(coordinates), dtype=np.float64)
        shaped = phase_k.residual_to_relation_map(residual, video_count=2, t_count=4)
        self.assertEqual(shaped.shape, (2, 2, 4))
        cursor = 0
        for video in range(2):
            for source_slot, target_index, *_ in coordinates:
                self.assertEqual(shaped[video, source_slot, target_index], residual[cursor])
                cursor += 1
            for source_slot, source in enumerate(phase_k.source_positions(4)):
                self.assertTrue(np.isnan(shaped[video, source_slot, source]))
        self.assertEqual(cursor, residual.size)

    def test_synthetic_analysis_writes_required_outputs(self):
        rng = np.random.default_rng(3407)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            phase_root = root / "phase_k"
            phase_j_root = root / "phase_j"
            phase_root.mkdir()
            phase_j_root.mkdir()
            unit_ids = [uid for domain in phase_k.DOMAINS for uid in phase_k.EXPECTED_DOMAIN_UNITS[domain]]
            unit_identities = []
            unit_manifest_rows = []
            type_by_id = {}
            for domain in phase_k.DOMAINS:
                for index, uid in enumerate(phase_k.EXPECTED_DOMAIN_UNITS[domain]):
                    kind = "attention_head" if domain != "271" or index < 2 else "ffn_neuron"
                    type_by_id[uid] = kind
                    item = {"task037_global_index": uid, "domain_id": domain, "layer": "layer.%d" % index,
                            "stage": "stage", "unit_type": kind, "unit_index": index}
                    unit_identities.append(item)
                    unit_manifest_rows.append({**item, "D_abs": index + 0.1, "D_rel": index + 0.2, "D_st": index + 0.3})
            videos = []
            for label in range(10):
                for position in range(1, 4):
                    vi = label * 3 + position - 1
                    videos.append({"video_index": vi, "video_id": "class%02d/video%02d" % (label, position),
                                   "label": label, "class_name": "class%02d" % label, "class_position": position})
            config = {"required_branch": phase_k.BRANCH, "base_output": str(root), "git_head": "test-head",
                      "task": "synthetic Phase K integration test", "unit_identities": unit_identities,
                      "video_identity_order": videos, "unit_manifest": str(root / "unit_manifest.csv")}
            phase_k.write_json(phase_root / "task042_phase_k_run_config.json", config)
            phase_k.write_csv(root / "unit_manifest.csv", unit_manifest_rows)
            phase_k.write_json(phase_root / "task042_phase_k_noop_gate.json",
                               {"passed": True, "max_abs_all_units": 0.0, "atol": 1e-6, "rtol": 1e-6})
            phasej_rows = []
            for uid in unit_ids:
                for video in videos:
                    for t in range(16):
                        for tp in range(16):
                            phasej_rows.append({"task037_global_index": uid, "video_index": video["video_index"],
                                                "frame_t": t, "frame_t_prime": tp,
                                                "conditional_C": float(rng.normal())})
            phase_k.write_csv(phase_j_root / "task042_phase_j_conditional_relation.csv", phasej_rows)
            for shard, video_indices in ((0, list(range(15))), (1, list(range(15, 30)))):
                directed = rng.random((15, len(unit_ids), 14, 16), dtype=np.float32) * 0.2
                sources = phase_k.source_positions(16)
                for si, source in enumerate(sources):
                    directed[:, :, si, source] = np.nan
                source_mag = rng.random((15, 14), dtype=np.float32) * 0.5
                np.savez_compressed(phase_root / ("task042_phase_k_shard%d.npz" % shard),
                                    video_indices=np.asarray(video_indices, dtype=np.int16),
                                    unit_ids=np.asarray(unit_ids, dtype=np.int64),
                                    directed=directed, source_magnitude=source_mag)
                manifest, innovations = [], []
                for vi in video_indices:
                    video = videos[vi]
                    manifest.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                                     "class_position": video["class_position"], "forward_kind": "original",
                                     "source_position": "", "non_source_positions_exact": "true",
                                     "source_replacement_max_abs_error": "0.0", "analysis_forward": "true"})
                    for si, source in enumerate(sources):
                        manifest.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                                         "class_position": video["class_position"], "forward_kind": "source",
                                         "source_position": source + 1, "non_source_positions_exact": "true",
                                         "source_replacement_max_abs_error": "0.0", "analysis_forward": "true"})
                        innovations.append({"video_index": vi, "video_id": video["video_id"],
                                            "class_name": video["class_name"], "class_position": video["class_position"],
                                            "source_position": source + 1,
                                            "source_innovation_relative_norm": float(source_mag[vi - video_indices[0], si])})
                phase_k.write_csv(phase_root / ("task042_phase_k_shard%d_intervention_manifest.csv" % shard), manifest)
                phase_k.write_csv(phase_root / ("task042_phase_k_shard%d_source_innovation.csv" % shard), innovations)
                phase_k.write_json(phase_root / ("task042_phase_k_shard%d_runtime.json" % shard), {
                    "analysis_forward_count": 225, "no_op_replay_forward_count": 1 if shard == 0 else 0,
                    "physical_gpu": shard,
                    "noop_summary": {"max_abs_all_units": 0.0} if shard == 0 else None,
                    "total_forward_count": 226 if shard == 0 else 225,
                    "cuda_analysis_forward_event_seconds": 1.0, "cuda_noop_forward_event_seconds": 0.01,
                    "worker_wall_seconds": 1.0, "peak_gpu_memory_allocated_bytes": 100,
                    "peak_gpu_memory_reserved_bytes": 200, "shard_file_bytes": 1000,
                    "patch_embedding_geometry": {"module_path": "patch_embed", "temporal_axis": 2}})
            original_runtime = phase_k._task042_runtime
            phase_k._task042_runtime = lambda _repo: (None, phase_j, None)
            try:
                phase_k.analyze_phase_k(SimpleNamespace(repo_root=root, phase_root=phase_root))
            finally:
                phase_k._task042_runtime = original_runtime
            for name in (
                "task042_phase_k_intervention_manifest.csv", "task042_phase_k_source_innovation.csv",
                "task042_phase_k_directed_relation.csv", "task042_phase_k_directionality.csv",
                "task042_phase_k_source_magnitude_audit.csv", "task042_phase_k_target_specificity.csv",
                "task042_phase_k_lag_analysis.csv", "task042_phase_k_video_stability.csv",
                "task042_phase_k_domain_geometry.csv", "task042_phase_k_coverability.csv",
                "task042_phase_k_residual_relation_map.csv", "task042_phase_k_mixed_domain.csv",
                "task042_phase_k_phasej_comparison.csv", "task042_phase_k_descriptor_complementarity.csv",
                "task042_phase_k_runtime_summary.json", "task042_phase_k_summary.json", "task042_phase_k_report.md",
            ):
                self.assertTrue((phase_root / name).is_file(), name)
            summary = json.loads((phase_root / "task042_phase_k_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["decision"], "B")
            self.assertEqual(summary["phase_j_comparison"]["nearest_neighbor_comparison_count"], 13)


if __name__ == "__main__":
    unittest.main()

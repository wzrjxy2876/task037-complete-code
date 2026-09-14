#!/usr/bin/env python3
"""Task042 Phase K: directed temporal-innovation intervention diagnosis.

The only intervention is at the common Swin patch-embedding representation.
No unit masks, pruning, training, prediction model, or performance oracle is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BRANCH = "task_042_post_bms_frame_relation_redundancy"
DOMAINS = ("415", "103", "76", "271")
EXPECTED_DOMAIN_UNITS = {
    "415": (779, 1549, 1553),
    "103": (254, 2334, 16627),
    "76": (133, 33306, 36376),
    "271": (328, 1611, 7845, 30209),
}
CHECKPOINT_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DEFAULT_REPO = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
DEFAULT_BASE = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
T_INPUT = 32
T_MODEL = 16
EPS = 1e-12
NOOP_ATOL = 1e-6
NOOP_RTOL = 1e-6


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 Phase-K gate failed: " + message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in fields})


def _cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def source_positions(t_count: int) -> tuple[int, ...]:
    """Return zero-based interior temporal positions, excluding both boundaries."""
    if t_count < 3:
        raise ValueError("at least three temporal positions are required")
    return tuple(range(1, int(t_count) - 1))


def temporal_position_support(sampled_input_frames: Sequence[int], temporal_position_one_based: int,
                              tubelet_width: int = 2) -> tuple[int, ...]:
    """Map a non-overlapping temporal tubelet position to selected raw-frame indices."""
    p = int(temporal_position_one_based)
    width = int(tubelet_width)
    if width != 2 or p < 1 or len(sampled_input_frames) != T_INPUT:
        raise ValueError("Phase-K expects T=32 input slots and non-overlapping width-2 tubelets")
    start = (p - 1) * width
    if start + width > len(sampled_input_frames):
        raise ValueError("temporal position is outside the model input support")
    return tuple(int(x) for x in sampled_input_frames[start:start + width])


def extract_attention_target_frames(head_windows: Any, geometry: Mapping[str, Any], phase_j_module: Any) -> Any:
    """Restore exact per-head spatial output and flatten spatial/head dimensions only."""
    return phase_j_module.attention_frame_features(head_windows, geometry)


def extract_ffn_target_frames(post_gelu_activation: Any, neuron_index: int, phase_j_module: Any) -> Any:
    """Select one post-GELU/pre-fc2 neuron, preserving every spatial position."""
    return phase_j_module.ffn_frame_features(post_gelu_activation, int(neuron_index))


def replace_source_innovation(z: Any, source_index: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Apply do(q_s=0) to channels-first patch embeddings [1,C,T,H,W]."""
    import torch
    if not torch.is_tensor(z) or z.ndim != 5 or int(z.shape[0]) != 1:
        raise ValueError("patch embedding output must have shape [1,C,T,H,W]")
    t_count = int(z.shape[2])
    if source_index not in source_positions(t_count):
        raise ValueError("source index must be an interior zero-based temporal position")
    background = (z[:, :, source_index - 1] + z[:, :, source_index + 1]) * 0.5
    innovation = z[:, :, source_index] - background
    modified = z.clone()
    modified[:, :, source_index] = background
    before = torch.cat((z[:, :, :source_index], z[:, :, source_index + 1:]), dim=2)
    after = torch.cat((modified[:, :, :source_index], modified[:, :, source_index + 1:]), dim=2)
    non_source_exact = bool(torch.equal(before, after))
    source_error = float(torch.max(torch.abs(modified[:, :, source_index] - background)).detach().cpu())
    magnitude = torch.linalg.vector_norm(innovation) / (torch.linalg.vector_norm(z[:, :, source_index]) + EPS)
    detail = {"non_source_exact": non_source_exact, "source_replacement_max_abs_error": source_error,
              "source_position_zero_based": int(source_index), "temporal_axis": 2,
              "input_shape": list(z.shape), "output_shape": list(modified.shape)}
    if not non_source_exact or source_error != 0.0:
        raise RuntimeError("intervention locality failed at patch embedding")
    return modified, innovation, magnitude, detail


def directed_influence(original: Any, intervened: Any, source_index: int) -> Any:
    """Compute A(s->t) for [T,D] target features; source self-effect is NaN."""
    import torch
    if not torch.is_tensor(original) or not torch.is_tensor(intervened):
        raise TypeError("target representations must be tensors")
    if original.ndim != 2 or intervened.shape != original.shape:
        raise ValueError("target representations must share [T,D] shape")
    if source_index not in source_positions(int(original.shape[0])):
        raise ValueError("source index must be interior")
    delta = torch.linalg.vector_norm(original - intervened, dim=1)
    denom = (torch.linalg.vector_norm(original, dim=1) +
             torch.linalg.vector_norm(intervened, dim=1) + EPS)
    result = delta / denom
    result[source_index] = float("nan")
    valid = torch.isfinite(result)
    if bool(torch.any(result[valid] < -1e-7)) or bool(torch.any(result[valid] > 1.0 + 1e-6)):
        raise RuntimeError("directed influence is outside [0,1]")
    return result


def no_op_match(reference: Mapping[int, Any], replay: Mapping[int, Any], atol: float = NOOP_ATOL,
                rtol: float = NOOP_RTOL) -> dict[str, Any]:
    import torch
    if set(reference) != set(replay):
        raise RuntimeError("no-op replay target-unit set differs from ordinary forward")
    maxima: dict[str, float] = {}
    exact_all = True
    for uid in sorted(reference):
        a, b = reference[uid], replay[uid]
        if a.shape != b.shape:
            raise RuntimeError("no-op replay changed target shape for unit " + str(uid))
        exact_all = exact_all and bool(torch.equal(a, b))
        maxima[str(uid)] = float(torch.max(torch.abs(a - b)).detach().cpu())
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            raise RuntimeError("no-op replay exceeded strict FP32 tolerance for unit " + str(uid))
    return {"exact_bitwise_all_units": exact_all, "atol": atol, "rtol": rtol,
            "max_abs_by_unit": maxima, "max_abs_all_units": max(maxima.values(), default=0.0)}


def asymmetry(a_forward: float, a_reverse: float) -> tuple[float, float]:
    absolute = abs(float(a_forward) - float(a_reverse))
    relative = absolute / (float(a_forward) + float(a_reverse) + EPS)
    return absolute, relative


def residual_to_relation_map(residual: Sequence[float], video_count: int, t_count: int) -> np.ndarray:
    """Restore packed source-target residuals to [video,source,T], diagonal NaN."""
    sources = source_positions(t_count)
    expected = int(video_count) * len(sources) * (t_count - 1)
    vector = np.asarray(residual, dtype=np.float64)
    if vector.size != expected:
        raise ValueError("packed residual length does not match video/source/target identity")
    out = np.full((video_count, len(sources), t_count), np.nan, dtype=np.float64)
    cursor = 0
    for vi in range(video_count):
        for si, source in enumerate(sources):
            for target in range(t_count):
                if target == source:
                    continue
                out[vi, si, target] = vector[cursor]
                cursor += 1
    if cursor != vector.size:
        raise RuntimeError("residual reshape did not consume the complete vector")
    return out


class PatchInnovationIntervention:
    """A read-only forward hook with explicit baseline/no-op/source modes."""
    def __init__(self, module: Any, torch: Any):
        self.torch = torch
        self.mode = "baseline"
        self.source_index: int | None = None
        self.z_original: Any = None
        self.innovation: Any = None
        self.magnitude: Any = None
        self.detail: dict[str, Any] = {}
        self.handle = module.register_forward_hook(self._hook)

    def begin(self, mode: str, source_index: int | None = None) -> None:
        if mode not in ("baseline", "noop", "source"):
            raise ValueError("unknown patch intervention mode")
        self.mode, self.source_index = mode, source_index
        self.z_original = self.innovation = self.magnitude = None
        self.detail = {}

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not self.torch.is_tensor(output) or output.ndim != 5 or int(output.shape[0]) != 1:
            raise RuntimeError("common patch embedding must output [1,C,T,H,W]")
        if int(output.shape[2]) != T_MODEL:
            raise RuntimeError("common patch embedding temporal length is not 16")
        self.z_original = output.detach().clone()
        if self.mode == "baseline" or self.mode == "noop":
            self.detail = {"non_source_exact": True, "source_replacement_max_abs_error": 0.0,
                           "output_shape": list(output.shape), "temporal_axis": 2}
            return output
        if self.source_index is None:
            raise RuntimeError("source mode requires an interior source index")
        modified, self.innovation, self.magnitude, self.detail = replace_source_innovation(output, self.source_index)
        return modified

    def close(self) -> None:
        self.handle.remove()


def _normalize_video_order(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        required = ("video_index", "video_id", "label", "class_name", "class_position")
        require(all(key in row for key in required), "Phase-J frozen video identity is incomplete")
        result.append({**dict(row), "video_index": int(row["video_index"]), "label": int(row["label"]),
                       "class_position": int(row["class_position"])})
    require(sorted(r["video_index"] for r in result) == list(range(30)), "video identity is not exactly 30 videos")
    classes = {str(r["class_name"]) for r in result}
    require(len(classes) == 10, "video identity is not 10 action classes")
    for cls in classes:
        rows_cls = [r for r in result if str(r["class_name"]) == cls]
        require(len(rows_cls) == 3 and {r["class_position"] for r in rows_cls} == {1, 2, 3},
                "class/within-class video identity is not exactly positions 1,2,3")
        require(len({int(r["label"]) for r in rows_cls}) == 1, "class numeric label is inconsistent")
    return sorted(result, key=lambda r: r["video_index"])


def _plain_state(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain_state(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _plain_state(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if callable(value):
        return {"callable": getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", type(value).__name__)}
    if hasattr(value, "tolist"):
        try:
            return _plain_state(value.tolist())
        except Exception:
            pass
    return str(value)


def _transform_identity(loader: Any, repo: Path) -> dict[str, Any]:
    dataset = loader.dataset
    subset_layers = []
    while hasattr(dataset, "dataset"):
        subset_layers.append({"class": type(dataset).__module__ + "." + type(dataset).__name__,
                              "indices": _plain_state(getattr(dataset, "indices", []))})
        dataset = dataset.dataset
    spatial = getattr(dataset, "spatial_transform", None)
    temporal = getattr(dataset, "temporal_transform", None)
    spatial_parts = list(getattr(spatial, "transforms", []))
    root = repo / "src" / "lgfr_runtime" / "dataset"
    probe_path = repo / "src" / "lgfr_runtime" / "probe_ctfrs_dynamic_function.py"
    return {
        "dataset_class": type(dataset).__module__ + "." + type(dataset).__name__,
        "subset_layers": subset_layers,
        "spatial_pipeline": [{"class": type(x).__module__ + "." + type(x).__name__,
                              "state": _plain_state(vars(x))} for x in spatial_parts],
        "temporal_transform": {"class": type(temporal).__module__ + "." + type(temporal).__name__,
                               "state": _plain_state(vars(temporal))},
        "dataset_loader": _plain_state(dataset.loader),
        "loader": {"batch_size": loader.batch_size, "drop_last": loader.drop_last,
                   "sampler": type(loader.sampler).__module__ + "." + type(loader.sampler).__name__,
                   "num_workers": loader.num_workers},
        "source_sha256": {
            "ucf101_dataset": sha256_file(root / "ucf101.py"),
            "transforms": sha256_file(root / "transforms.py"),
            "balanced_loader_builder": sha256_file(probe_path),
        },
    }


def _prepare(repo: Path, base_root: Path, phase_root: Path) -> dict[str, Any]:
    if phase_root.exists():
        raise RuntimeError("Phase-K output already exists; refusing overwrite")
    require(subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip() == BRANCH,
            "checkout is not the frozen Task042 branch")
    require(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip() == "",
            "Phase-K support must be committed and checkout clean before server execution")
    phase_j = base_root / "phase_j"
    jcfg_path, jsummary_path = phase_j / "task042_phase_j_run_config.json", phase_j / "task042_phase_j_summary.json"
    require(jcfg_path.is_file() and jsummary_path.is_file(), "completed Phase-J run configuration/summary is missing")
    jcfg, jsummary = json.loads(jcfg_path.read_text(encoding="utf-8")), json.loads(jsummary_path.read_text(encoding="utf-8"))
    require(jsummary.get("analysis_status") == "completed" and jsummary.get("decision") == "B",
            "Phase J did not complete with its frozen weak/unresolved decision")
    require(jcfg.get("checkpoint_sha256") == CHECKPOINT_SHA, "Phase-J checkpoint identity mismatch")
    require(int(jsummary.get("temporal_resolution", -1)) == T_MODEL, "Phase-J audit did not establish T=16")
    base_cfg_path = base_root / "task042_run_config.json"
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    require(cfg.get("required_branch") == BRANCH and Path(cfg["repo_root"]).resolve() == repo.resolve(),
            "frozen Task042 run configuration identity mismatch")
    require(sha256_file(Path(cfg["checkpoint_path"])) == CHECKPOINT_SHA, "checkpoint SHA256 mismatch")
    for key, path_key in (("profile", "profile_path"), ("descriptors", "descriptor_path"),
                          ("unit_mapping", "unit_mapping_path"), ("val_list", "val_list"),
                          ("video_manifest", "video_manifest")):
        expected_sha = cfg.get("input_sha256", {}).get(key)
        require(expected_sha and sha256_file(Path(cfg[path_key])) == expected_sha,
                "frozen Task042 input hash mismatch: " + key)
    require(sha256_file(Path(cfg["exact_video_list"])) == sha256_file(Path(jcfg["exact_video_list"])),
            "Phase-J and Phase-K exact video lists differ")
    phase_j_units = jcfg.get("unit_identities", [])
    selected = [row for row in read_csv(Path(cfg["unit_manifest"])) if str(row["domain_id"]) in DOMAINS]
    selected.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    expected_ids = {uid for domain in DOMAINS for uid in EXPECTED_DOMAIN_UNITS[domain]}
    require({int(r["task037_global_index"]) for r in selected} == expected_ids and len(selected) == 13,
            "selected Task037 global-index cohort differs from frozen Phase-K identities")
    for domain in DOMAINS:
        actual_domain_ids = {int(r["task037_global_index"]) for r in selected if str(r["domain_id"]) == domain}
        require(actual_domain_ids == set(EXPECTED_DOMAIN_UNITS[domain]),
                "frozen Task037 global-index/domain assignment differs for BMS domain " + domain)
    by_id = {int(r["task037_global_index"]): r for r in selected}
    mapping = {int(r["global_index"]): r for r in read_csv(Path(cfg["unit_mapping_path"]))}
    require(expected_ids.issubset(mapping), "Task037 mapping lacks selected units")
    require({int(r["task037_global_index"]) for r in phase_j_units} == expected_ids,
            "Phase-J selected unit set differs from Phase-K frozen identities")
    phase_j_by_id = {int(x["task037_global_index"]): x for x in phase_j_units}
    for uid in sorted(expected_ids):
        row, source = by_id[uid], mapping[uid]
        for key in ("global_index",):
            require(int(source[key]) == uid, "Task037 global-index identity mismatch")
        require(str(row["layer"]) == str(source["layer"]) and str(row["unit_type"]) == str(source["unit_type"])
                and int(row["unit_index"]) == int(source["unit_index"]),
                "Task037 layer/type/unit identity mismatch for " + str(uid))
        phase_identity = phase_j_by_id[uid]
        for key in ("domain_id", "layer", "stage", "unit_type", "unit_index"):
            require(str(row[key]) == str(phase_identity[key]),
                    "Phase-J and Phase-K %s identity mismatch for %s" % (key, uid))
    _, _, ctfrs = _runtime_modules(repo)
    base, _, _ = _task042_runtime(repo)
    loader = base._get_loader(ctfrs, cfg, workers=0)
    videos = _normalize_video_order(jcfg["video_identity_order"])
    phase_j = _task042_runtime(repo)[1]
    manifest_joined = phase_j._derive_video_identities(base.read_csv(Path(cfg["video_manifest"])), Path(cfg["exact_video_list"]))
    manifest_by_name = {Path(str(v["video_id"])).name: v for v in manifest_joined}
    require({Path(str(v["video_id"])).name for v in videos} == set(manifest_by_name),
            "Phase-J video identities differ from the authoritative video manifest")
    for video in videos:
        authoritative = manifest_by_name[Path(str(video["video_id"])).name]
        for key in ("label", "class_name", "class_position"):
            require(str(video[key]) == str(authoritative[key]),
                    "Phase-J and authoritative video %s identity differs for %s" % (key, video["video_id"]))
    duration_by_stem = {Path(str(v["video_id"])).name: int(v["duration"]) for v in manifest_joined}
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    temporal = dataset.temporal_transform
    require(type(temporal).__name__ == "LoopPadding" and int(temporal.size) == T_INPUT,
            "authoritative Task042 temporal transform is no longer LoopPadding(32)")
    slot_rows = []
    sampled_by_stem = {}
    for video in videos:
        stem = Path(str(video["video_id"])).name
        duration = duration_by_stem[stem]
        sampled = [int(i) for i in temporal(list(range(1, duration + 1)))]
        require(len(sampled) == T_INPUT and min(sampled) >= 1 and max(sampled) <= duration,
                "authoritative temporal transform did not produce 32 valid frame indices")
        sampled_by_stem[stem] = sampled
        for temporal_pos in range(1, T_MODEL + 1):
            slot1, slot2 = 2 * temporal_pos - 1, 2 * temporal_pos
            raw_support = temporal_position_support(sampled, temporal_pos)
            slot_rows.append({"video_index": int(video["video_index"]), "video_id": video["video_id"],
                              "class_name": video["class_name"], "class_position": video["class_position"],
                              "duration": duration, "temporal_position": temporal_pos,
                              "model_input_slot_1": slot1, "model_input_slot_2": slot2,
                              "raw_frame_index_1": raw_support[0], "raw_frame_index_2": raw_support[1],
                              "raw_frame_file_1": "image_%05d.jpg" % raw_support[0],
                              "raw_frame_file_2": "image_%05d.jpg" % raw_support[1],
                              "identity_semantics": "patch-embedding temporal position covers these two LoopPadding-selected input slots"})
    require(len(loader.dataset) == 30, "authoritative loader did not yield exactly 30 videos")
    transform = _transform_identity(loader, repo)
    transform_blob = json.dumps(transform, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    run_cfg = {
        "task": "TASK042 PHASE K — DIRECTED TEMPORAL INNOVATION INFLUENCE DIAGNOSTIC",
        "required_branch": BRANCH, "repo_root": str(repo.resolve()), "base_output": str(base_root.resolve()),
        "phase_root": str(phase_root.resolve()), "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "phase_j_git_head": jsummary.get("git_head"), "phase_j_decision": jsummary.get("decision"),
        "checkpoint_path": cfg["checkpoint_path"], "checkpoint_sha256": CHECKPOINT_SHA,
        "unit_manifest": cfg["unit_manifest"], "video_manifest": cfg["video_manifest"],
        "unit_count": len(selected), "unit_identities": [{k: r[k] for k in ("task037_global_index", "domain_id", "layer", "stage", "unit_type", "unit_index")} for r in selected],
        "video_count": len(videos), "video_identity_order": videos, "duration_by_stem": duration_by_stem,
        "temporal_input_indices_by_video": sampled_by_stem, "temporal_position_map": slot_rows,
        "input_temporal_slots": T_INPUT, "model_temporal_positions": T_MODEL,
        "temporal_intervention_module": "patch_embed output hook; exact module path verified at runtime",
        "expected_temporal_patch_kernel_stride": 2,
        "transform_identity": transform, "transform_identity_sha256": hashlib.sha256(transform_blob.encode("utf-8")).hexdigest(),
        "frame_root": cfg["frame_root"], "exact_video_list": cfg["exact_video_list"], "val_list": cfg["val_list"],
        "input_sha256": {"task042_run_config": sha256_file(base_cfg_path), "phase_j_run_config": sha256_file(jcfg_path),
                         "phase_j_summary": sha256_file(jsummary_path), "unit_manifest": sha256_file(Path(cfg["unit_manifest"])),
                         "video_manifest": sha256_file(Path(cfg["video_manifest"])), "exact_video_list": sha256_file(Path(cfg["exact_video_list"])),
                         "val_list": sha256_file(Path(cfg["val_list"])), "checkpoint": CHECKPOINT_SHA},
        "dtype": "float32", "amp": False, "no_op_tolerance": {"atol": NOOP_ATOL, "rtol": NOOP_RTOL},
        "source_positions_one_based": list(range(2, T_MODEL)), "boundary_positions_excluded": [1, T_MODEL],
        "target_positions_one_based": list(range(1, T_MODEL + 1)), "valid_relations_per_video_unit": 210,
        "analysis_forwards_expected": 450, "no_op_replay_forwards_expected": 1,
        "mask_free": True, "pruning": False, "finetuning": False, "performance_oracle": False,
    }
    phase_root.mkdir(parents=True)
    write_json(phase_root / "task042_phase_k_run_config.json", run_cfg)
    write_csv(phase_root / "task042_phase_k_temporal_position_audit.csv", slot_rows)
    return run_cfg


def _task042_runtime(repo: Path) -> tuple[Any, Any, Any]:
    runtime = str(repo / "src" / "lgfr_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task042_frame_relation_redundancy as base
    import task042_phase_j_conditional as phase_j
    return base, phase_j, phase_j._task042_runtime(repo)[2]


def _runtime_modules(repo: Path) -> tuple[Any, Any, Any]:
    return _task042_runtime(repo)


def _capture_rows(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in units:
        item = dict(row)
        item["capture_kind"] = "head" if str(row["unit_type"]) == "attention_head" else "neuron"
        result.append(item)
    return result


def _model_on_gpu(config: Mapping[str, Any], gpu: int):
    import torch
    require(gpu in (0, 1), "only physical GPUs 0 and 1 are authorized")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "worker must be isolated to its authorized physical GPU")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker must see exactly one isolated GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3407 + gpu)
    torch.cuda.manual_seed_all(3407 + gpu)
    torch.cuda.reset_peak_memory_stats()
    repo = Path(config["repo_root"])
    base, phase_j, ctfrs = _task042_runtime(repo)
    phase_j_cfg = json.loads((Path(config["base_output"]) / "phase_j" /
                              "task042_phase_j_run_config.json").read_text(encoding="utf-8"))
    _, _, _, _, model, specs, identity = phase_j._load_model_and_specs(phase_j_cfg, gpu)
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA and identity.get("classifier_head", {}).get("status") == "loaded",
            "checkpoint / authoritative classifier identity mismatch")
    require(not identity.get("missing_keys") and not identity.get("unexpected_keys") and not identity.get("shape_mismatches"),
            "checkpoint did not load exactly")
    model.eval()
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    patch_candidates = [(name, mod) for name, mod in modules.items()
                        if name.endswith("patch_embed") and hasattr(mod, "proj") and hasattr(mod.proj, "kernel_size")]
    require(len(patch_candidates) == 1, "expected exactly one common patch_embed module")
    patch_name, patch_module = patch_candidates[0]
    patch = patch_module.proj
    geometry = {"module_path": patch_name, "output_tensor_layout": "[B,C,T,H,W]",
                "temporal_axis": 2, "spatial_axes": [3, 4], "channel_axis": 1,
                "patch_size": [int(x) for x in patch_module.patch_size],
                "conv_kernel_size": [int(x) for x in patch.kernel_size],
                "conv_stride": [int(x) for x in patch.stride], "conv_padding": [int(x) for x in patch.padding],
                "conv_dilation": [int(x) for x in patch.dilation], "normalization": type(patch_module.norm).__name__ if patch_module.norm is not None else None}
    require(geometry["patch_size"][0] == 2 and geometry["conv_kernel_size"][0] == 2 and geometry["conv_stride"][0] == 2,
            "temporal patch embedding no longer uses kernel/stride 2")
    require(geometry["conv_padding"][0] == 0 and geometry["conv_dilation"][0] == 1,
            "patch embed temporal support is not the expected non-overlapping two-slot tubelet")
    return torch, base, phase_j, model, specs, identity, patch_name, patch_module, geometry


def _forward_features(model: Any, capture: Any, patch_hook: PatchInnovationIntervention, clip: Any,
                      torch: Any, mode: str, source_index: int | None = None) -> tuple[dict[int, Any], Any, dict[str, Any], Any, Any]:
    capture.begin(collect_features=True)
    patch_hook.begin(mode, source_index)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        _ = model(clip.unsqueeze(0))
    end.record()
    expected = [int(r["task037_global_index"]) for r in capture.rows]
    require(set(capture.features) == set(expected), "a selected target unit was not captured")
    require(set(capture.metadata) == set(expected), "selected target metadata is incomplete")
    phase_j = sys.modules.get("task042_phase_j_conditional")
    if phase_j is not None:
        phase_j.validate_mask_free_identity(expected, list(capture.features), mask_hooks_registered=False)
    return ({int(uid): feature.detach() for uid, feature in capture.features.items()},
            patch_hook.z_original, dict(patch_hook.detail), patch_hook.innovation,
            {"start": start, "end": end})


def _event_seconds(torch: Any, events: Sequence[Mapping[str, Any]]) -> float:
    if not events:
        return 0.0
    torch.cuda.synchronize()
    return float(sum(float(item["start"].elapsed_time(item["end"])) for item in events) / 1000.0)


def _video_identity_map(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["video_index"]): dict(row) for row in config["video_identity_order"]}


def run_shard(args: argparse.Namespace) -> None:
    import torch
    repo, phase_root = Path(args.repo_root), Path(args.phase_root)
    cfg_path = phase_root / "task042_phase_k_run_config.json"
    require(cfg_path.is_file(), "run Phase-K prepare-only before GPU shards")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == config["git_head"],
            "checkout moved after Phase-K preparation")
    require(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip() == "",
            "Phase-K worker requires committed clean support")
    gpu, shard = int(args.gpu), int(args.shard)
    require(shard in (0, 1) and gpu == shard, "video shard 0/1 must run on the matching physical GPU")
    if shard == 1:
        gate_path = phase_root / "task042_phase_k_noop_gate.json"
        deadline = time.monotonic() + 900.0
        while not gate_path.is_file() and time.monotonic() < deadline:
            time.sleep(1.0)
        require(gate_path.is_file() and json.loads(gate_path.read_text()).get("passed") is True,
                "GPU1 waited 15 minutes but GPU0 no-op gate did not pass")
    torch, base, phase_j, model, specs, identity, patch_name, patch_module, patch_geometry = _model_on_gpu(config, gpu)
    unit_rows = [r for r in read_csv(Path(config["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    unit_rows.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    capture_rows = _capture_rows(unit_rows)
    capture = phase_j.PhaseJCapture(model, capture_rows, specs, torch)
    patch_hook = PatchInnovationIntervention(patch_module, torch)
    _, clips, loader_rows = phase_j._load_all_videos(config, workers=2)
    expected_video_indices = list(range(0, 15)) if shard == 0 else list(range(15, 30))
    require(set(clips) == set(range(30)), "authoritative loader did not reproduce all 30 exact clips")
    videos = _video_identity_map(config)
    unit_ids = [int(r["task037_global_index"]) for r in capture_rows]
    source_indices = source_positions(T_MODEL)
    directed = np.full((15, len(unit_ids), len(source_indices), T_MODEL), np.nan, dtype=np.float32)
    source_mag = np.full((15, len(source_indices)), np.nan, dtype=np.float32)
    video_index_slot = {vi: slot for slot, vi in enumerate(expected_video_indices)}
    manifest_rows, innovation_rows, analysis_events, noop_events = [], [], [], []
    patch_audit = None
    noop_summary = None
    main_forward_count = 0
    noop_forward_count = 0
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for video_index in expected_video_indices:
            video = videos[video_index]
            loader_video = loader_rows[video_index]
            require(str(video["video_id"]) == str(loader_video["video_id"]) and int(video["label"]) == int(loader_video["label"]),
                    "loader video identity/label differs from the frozen 30-video manifest")
            clip = clips[video_index].to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            require(tuple(clip.shape[1:]) == (T_INPUT, int(clip.shape[2]), int(clip.shape[3])),
                    "transformed model input does not have 32 temporal slots")
            base_features, z, patch_detail, _, events = _forward_features(model, capture, patch_hook, clip, torch, "baseline")
            require(z is not None and tuple(z.shape[2:3]) == (T_MODEL,), "common patch embedding output T is not 16")
            analysis_events.append(events)
            main_forward_count += 1
            if patch_audit is None:
                require(int(clip.shape[1]) == T_INPUT, "input tensor temporal axis is not 32")
                patch_audit = {**patch_geometry, "input_tensor_shape_B_C_T_H_W": [1, int(clip.shape[0]), int(clip.shape[1]),
                                                                                   int(clip.shape[2]), int(clip.shape[3])],
                               "patch_embedding_output_shape_B_C_T_H_W": list(z.shape),
                               "output_temporal_length": int(z.shape[2]), "output_spatial_shape": list(z.shape[3:]),
                               "output_channels": int(z.shape[1]), "temporal_patch_has_padding": False,
                               "raw_frame_support_rule": "Conv3d temporal kernel=2,stride=2; output position p covers transformed input slots 2p-1 and 2p (one-based)."}
                require(int(z.shape[2]) == T_MODEL and int(patch_geometry["patch_size"][0]) == 2,
                        "patch embedding does not define the declared T=16 tubelet grid")
            manifest_rows.append({"video_index": video_index, "video_id": video["video_id"], "class_name": video["class_name"],
                                  "class_position": video["class_position"], "forward_kind": "original", "source_position": "",
                                  "intervention_module": patch_name, "tensor_shape": list(z.shape), "temporal_axis": 2,
                                  "non_source_positions_exact": True, "source_replacement_max_abs_error": 0.0,
                                  "analysis_forward": True, "physical_gpu": gpu})
            if shard == 0 and video_index == 0:
                noop_features, noop_z, noop_detail, _, noop_event = _forward_features(model, capture, patch_hook, clip, torch, "noop")
                noop_events.append(noop_event)
                noop_forward_count += 1
                require(noop_z is not None and torch.equal(noop_z, z), "no-op intervention changed patch embedding output")
                try:
                    noop_summary = no_op_match(base_features, noop_features)
                    noop_gate = {"passed": True, "video_index": video_index, "video_id": video["video_id"],
                                 "target_unit_count": len(unit_ids), "patch_output_exact": True, **noop_summary}
                except Exception as exc:
                    noop_gate = {"passed": False, "video_index": video_index, "video_id": video["video_id"],
                                 "target_unit_count": len(unit_ids), "error": str(exc)}
                    write_json(phase_root / "task042_phase_k_noop_gate.json", noop_gate)
                    raise
                write_json(phase_root / "task042_phase_k_noop_gate.json", noop_gate)
                manifest_rows.append({"video_index": video_index, "video_id": video["video_id"], "class_name": video["class_name"],
                                      "class_position": video["class_position"], "forward_kind": "noop_replay", "source_position": "",
                                      "intervention_module": patch_name, "tensor_shape": list(noop_z.shape), "temporal_axis": 2,
                                      "non_source_positions_exact": True, "source_replacement_max_abs_error": 0.0,
                                      "analysis_forward": False, "physical_gpu": gpu, "noop_passed": True})
            base_features = {uid: feature.detach().clone() for uid, feature in base_features.items()}
            slot = video_index_slot[video_index]
            for si, source_index in enumerate(source_indices):
                target_features, z_source, detail, innovation, events = _forward_features(
                    model, capture, patch_hook, clip, torch, "source", source_index)
                require(z_source is not None and innovation is not None, "source innovation hook did not run")
                require(tuple(z_source.shape) == tuple(z.shape), "intervention changed common tensor shape")
                analysis_events.append(events)
                main_forward_count += 1
                m_value = float(patch_hook.magnitude.detach().cpu())
                source_mag[slot, si] = m_value
                position = source_index + 1
                innovation_rows.append({"video_index": video_index, "video_id": video["video_id"],
                                        "class_name": video["class_name"], "class_position": video["class_position"],
                                        "source_position": position, "source_input_slot_1": 2 * position - 1,
                                        "source_input_slot_2": 2 * position, "source_innovation_relative_norm": m_value,
                                        "non_source_positions_exact": detail["non_source_exact"],
                                        "source_replacement_max_abs_error": detail["source_replacement_max_abs_error"],
                                        "physical_gpu": gpu})
                manifest_rows.append({"video_index": video_index, "video_id": video["video_id"], "class_name": video["class_name"],
                                      "class_position": video["class_position"], "forward_kind": "source_innovation_suppressed",
                                      "source_position": position, "intervention_module": patch_name,
                                      "tensor_shape": detail["output_shape"], "temporal_axis": 2,
                                      "non_source_positions_exact": detail["non_source_exact"],
                                      "source_replacement_max_abs_error": detail["source_replacement_max_abs_error"],
                                      "source_innovation_relative_norm": m_value, "analysis_forward": True,
                                      "physical_gpu": gpu})
                for ui, uid in enumerate(unit_ids):
                    require(uid in base_features and uid in target_features, "selected unit missing from intervention capture")
                    a = directed_influence(base_features[uid], target_features[uid], source_index)
                    directed[slot, ui, si] = a.detach().to(dtype=torch.float32).cpu().numpy()
            print("PHASE_K_SHARD%d video=%d/15 source_interventions=%d" % (shard, slot + 1, len(source_indices)), flush=True)
        require(main_forward_count == len(expected_video_indices) * (1 + len(source_indices)),
                "analysis forward count differs from 15 per video")
    finally:
        capture.close()
        patch_hook.close()
    torch.cuda.synchronize()
    main_gpu_seconds = _event_seconds(torch, analysis_events)
    noop_gpu_seconds = _event_seconds(torch, noop_events)
    wall_seconds = time.perf_counter() - wall_start
    out_npz = phase_root / ("task042_phase_k_shard%d.npz" % shard)
    np.savez_compressed(out_npz, video_indices=np.asarray(expected_video_indices, dtype=np.int16),
                        unit_ids=np.asarray(unit_ids, dtype=np.int64), directed=directed, source_magnitude=source_mag)
    write_csv(phase_root / ("task042_phase_k_shard%d_intervention_manifest.csv" % shard), manifest_rows)
    write_csv(phase_root / ("task042_phase_k_shard%d_source_innovation.csv" % shard), innovation_rows)
    runtime = {"shard": shard, "physical_gpu": gpu, "video_indices": expected_video_indices,
               "analysis_forward_count": main_forward_count, "no_op_replay_forward_count": noop_forward_count,
               "total_forward_count": main_forward_count + noop_forward_count,
               "cuda_analysis_forward_event_seconds": main_gpu_seconds,
               "cuda_noop_forward_event_seconds": noop_gpu_seconds, "worker_wall_seconds": wall_seconds,
               "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "relation_tensor_shape": list(directed.shape), "saved_relation_tensor_bytes": int(directed.nbytes + source_mag.nbytes),
               "shard_file_bytes": out_npz.stat().st_size, "patch_embedding_geometry": patch_audit,
               "noop_summary": noop_summary, "git_head": config["git_head"]}
    write_json(phase_root / ("task042_phase_k_shard%d_runtime.json" % shard), runtime)
    print("PHASE_K_SHARD_COMPLETE shard=%d main_forwards=%d noop=%d seconds=%.2f" %
          (shard, main_forward_count, noop_forward_count, wall_seconds), flush=True)


def _corr(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    result: dict[str, Any] = {"n": int(a.size), "pearson": None, "pearson_p": None,
                              "spearman": None, "spearman_p": None}
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return result
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    result.update(pearson=float(pr), pearson_p=float(pp), spearman=float(sr), spearman_p=float(sp))
    return result


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-30:
        return float("nan")
    return float(1.0 - np.dot(x, y) / denom)


def _vectorize_relation(matrix: np.ndarray, sources: Sequence[int]) -> np.ndarray:
    values = []
    for si, source in enumerate(sources):
        for target in range(matrix.shape[-1]):
            if target != source:
                values.append(float(matrix[si, target]))
    return np.asarray(values, dtype=np.float64)


def _relation_coordinates(t_count: int = T_MODEL) -> list[tuple[int, int, int, int, int]]:
    result = []
    for si, source in enumerate(source_positions(t_count)):
        for target in range(t_count):
            if target != source:
                result.append((si, target, source + 1, target + 1, abs((source + 1) - (target + 1))))
    require(len(result) == (t_count - 2) * (t_count - 1),
            "Phase-K directed relation coordinate count is not (T-2)*(T-1)")
    return result


def _stats(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "std": None, "min": None, "q25": None,
                "median": None, "q75": None, "max": None}
    return {"n": int(x.size), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "min": float(np.min(x)), "q25": float(np.quantile(x, 0.25)),
            "median": float(np.median(x)), "q75": float(np.quantile(x, 0.75)), "max": float(np.max(x))}


def _domain_map(unit_rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {int(row["task037_global_index"]): str(row["domain_id"]) for row in unit_rows}


def _phase_j_vectors(phase_j_csv: Path, unit_ids: Sequence[int]) -> tuple[dict[int, np.ndarray], int]:
    rows = read_csv(phase_j_csv)
    require(bool(rows), "Phase-J conditional-relation artifact is empty")
    fields = set(rows[0])
    cfield = "conditional_C" if "conditional_C" in fields else "conditional_c"
    aliases = (("frame_t", "t"), ("frame_t_prime", "frame_t2", "t_prime", "t2"))
    tfield = next((name for name in aliases[0] if name in fields), None)
    tpfield = next((name for name in aliases[1] if name in fields), None)
    require("task037_global_index" in fields and "video_index" in fields and cfield in fields
            and tfield is not None and tpfield is not None,
            "Phase-J conditional relation schema is missing identity/coordinate/value fields")
    grouped: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
    for row in rows:
        uid = int(row["task037_global_index"])
        if uid not in set(map(int, unit_ids)):
            continue
        grouped[uid].append((int(row["video_index"]), int(row[tfield]), int(row[tpfield]), float(row[cfield])))
    vectors: dict[int, np.ndarray] = {}
    lengths = set()
    for uid in unit_ids:
        entries = grouped.get(int(uid), [])
        require(bool(entries), "Phase-J artifact lacks selected unit " + str(uid))
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        identity = [(v, t, tp) for v, t, tp, _ in entries]
        require(len(set(identity)) == len(identity), "Phase-J conditional relation has duplicate coordinates")
        vectors[int(uid)] = np.asarray([value for _, _, _, value in entries], dtype=np.float64)
        lengths.add(len(entries))
    require(len(lengths) == 1, "Phase-J conditional relation vectors have inconsistent lengths")
    return vectors, next(iter(lengths))


def analyze_phase_k(args: argparse.Namespace) -> None:
    from scipy.stats import rankdata
    repo, phase_root = Path(args.repo_root), Path(args.phase_root)
    config = json.loads((phase_root / "task042_phase_k_run_config.json").read_text(encoding="utf-8"))
    require(config.get("required_branch") == BRANCH, "Phase-K analysis branch identity mismatch")
    require((phase_root / "task042_phase_k_noop_gate.json").is_file(), "no-op gate result is missing")
    noop_gate = json.loads((phase_root / "task042_phase_k_noop_gate.json").read_text(encoding="utf-8"))
    require(noop_gate.get("passed") is True, "no-op intervention gate failed; relation interpretation is prohibited")
    unit_rows = list(config["unit_identities"])
    unit_rows.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    unit_ids = [int(row["task037_global_index"]) for row in unit_rows]
    domains = _domain_map(unit_rows)
    videos = _video_identity_map(config)
    sources = source_positions(T_MODEL)
    directed = np.full((30, len(unit_ids), len(sources), T_MODEL), np.nan, dtype=np.float32)
    source_mag = np.full((30, len(sources)), np.nan, dtype=np.float32)
    seen = set()
    shard_runtime = []
    manifest_rows, innovation_rows = [], []
    for shard in (0, 1):
        npz_path = phase_root / ("task042_phase_k_shard%d.npz" % shard)
        runtime_path = phase_root / ("task042_phase_k_shard%d_runtime.json" % shard)
        im_path = phase_root / ("task042_phase_k_shard%d_intervention_manifest.csv" % shard)
        src_path = phase_root / ("task042_phase_k_shard%d_source_innovation.csv" % shard)
        require(npz_path.is_file() and runtime_path.is_file() and im_path.is_file() and src_path.is_file(),
                "one Phase-K GPU shard is incomplete: " + str(shard))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        require(runtime.get("analysis_forward_count") == 225, "shard analysis forward count is not 225")
        if shard == 0:
            require(runtime.get("no_op_replay_forward_count") == 1 and runtime.get("noop_summary") is not None,
                    "GPU0 did not record the separate no-op replay")
        else:
            require(runtime.get("no_op_replay_forward_count") == 0, "GPU1 unexpectedly ran a no-op replay")
        shard_runtime.append(runtime)
        with np.load(npz_path, allow_pickle=False) as data:
            video_indices = data["video_indices"].astype(int).tolist()
            shard_units = data["unit_ids"].astype(int).tolist()
            require(shard_units == unit_ids, "shard unit order differs from frozen manifest")
            require(video_indices == (list(range(15)) if shard == 0 else list(range(15, 30))),
                    "shard video identities differ from frozen 15/15 split")
            require(tuple(data["directed"].shape) == (15, len(unit_ids), 14, 16), "saved relation tensor has wrong shape")
            for slot, vi in enumerate(video_indices):
                require(vi not in seen, "duplicate video across Phase-K shards")
                seen.add(vi)
                directed[vi] = data["directed"][slot]
                source_mag[vi] = data["source_magnitude"][slot]
        manifest_rows.extend(read_csv(im_path))
        innovation_rows.extend(read_csv(src_path))
    require(seen == set(range(30)), "Phase-K shards did not cover all 30 videos")
    require(np.all(np.isfinite(source_mag)) and np.all(source_mag >= 0), "source innovation magnitudes are incomplete")
    for si, source in enumerate(sources):
        require(np.all(np.isnan(directed[:, :, si, source])), "source self-effect must remain NA")
        valid = np.delete(directed[:, :, si, :], source, axis=2)
        require(np.all(np.isfinite(valid)) and np.all(valid >= -1e-7) and np.all(valid <= 1.0 + 1e-6),
                "directed influence values are incomplete or outside [0,1]")
    # The shared clip forward, rather than one forward per unit, captures all 13 units.
    coords = _relation_coordinates()
    packed: dict[int, np.ndarray] = {}
    unit_meta = {int(r["task037_global_index"]): dict(r) for r in unit_rows}
    unit_video_vectors: dict[int, dict[int, np.ndarray]] = {uid: {} for uid in unit_ids}
    for ui, uid in enumerate(unit_ids):
        packed[uid] = np.stack([_vectorize_relation(directed[vi, ui], sources) for vi in range(30)])
        for vi in range(30):
            unit_video_vectors[uid][vi] = packed[uid][vi]

    write_csv(phase_root / "task042_phase_k_intervention_manifest.csv", manifest_rows)
    write_csv(phase_root / "task042_phase_k_source_innovation.csv", innovation_rows)
    relation_rows = []
    for vi in range(30):
        video = videos[vi]
        for ui, uid in enumerate(unit_ids):
            for si, ti, source_pos, target_pos, lag in coords:
                relation_rows.append({"task037_global_index": uid, "domain_id": domains[uid],
                                      "unit_type": unit_meta[uid]["unit_type"], "video_index": vi,
                                      "video_id": video["video_id"], "class_name": video["class_name"],
                                      "class_position": video["class_position"], "source_position": source_pos,
                                      "target_position": target_pos, "lag": lag,
                                      "source_innovation_relative_norm": float(source_mag[vi, si]),
                                      "A_source_to_target": float(directed[vi, ui, si, ti])})
    write_csv(phase_root / "task042_phase_k_directed_relation.csv", relation_rows)

    # Directionality compares reciprocal relations only when both endpoints are interior.
    direction_rows, asym_values = [], defaultdict(list)
    for vi in range(30):
        video = videos[vi]
        for ui, uid in enumerate(unit_ids):
            for s in range(2, 16):
                for t in range(s + 1, 16):
                    forward = float(directed[vi, ui, s - 2, t - 1])
                    reverse = float(directed[vi, ui, t - 2, s - 1])
                    absolute, relative = asymmetry(forward, reverse)
                    asym_values[("unit", uid)].append(relative)
                    asym_values[("domain", domains[uid])].append(relative)
                    asym_values[("video", vi)].append(relative)
                    direction_rows.append({"task037_global_index": uid, "domain_id": domains[uid],
                                           "unit_type": unit_meta[uid]["unit_type"], "video_index": vi,
                                           "video_id": video["video_id"], "class_name": video["class_name"],
                                           "source_position": s, "target_position": t,
                                           "A_s_to_t": forward, "A_t_to_s": reverse,
                                           "absolute_directional_difference": absolute,
                                           "relative_asymmetry": relative})
    write_csv(phase_root / "task042_phase_k_directionality.csv", direction_rows)

    # Source magnitude confound and source-wise outgoing structure.
    magnitude_rows = []
    mag_correlation_rows = []
    all_magnitude = {uid: [] for uid in unit_ids}
    all_out_mean = {uid: [] for uid in unit_ids}
    all_out_max = {uid: [] for uid in unit_ids}
    all_out_energy = {uid: [] for uid in unit_ids}
    target_rows = []
    source_profiles = {}
    for ui, uid in enumerate(unit_ids):
        for vi in range(30):
            out_means, out_maxes, out_energies = [], [], []
            for si, source in enumerate(sources):
                vals = np.asarray([directed[vi, ui, si, t] for t in range(T_MODEL) if t != source], dtype=np.float64)
                mval = float(source_mag[vi, si])
                mean_v, max_v, energy_v = float(np.mean(vals)), float(np.max(vals)), float(np.dot(vals, vals))
                out_means.append(mean_v); out_maxes.append(max_v); out_energies.append(energy_v)
                all_magnitude[uid].append(mval); all_out_mean[uid].append(mean_v)
                all_out_max[uid].append(max_v); all_out_energy[uid].append(energy_v)
                magnitude_rows.append({"row_type": "source_video_unit", "task037_global_index": uid,
                                       "domain_id": domains[uid], "video_index": vi,
                                       "video_id": videos[vi]["video_id"], "class_name": videos[vi]["class_name"],
                                       "source_position": source + 1, "source_innovation_relative_norm": mval,
                                       "outgoing_mean_A": mean_v, "outgoing_max_A": max_v,
                                       "outgoing_squared_energy": energy_v})
                probs = vals / (float(vals.sum()) + EPS)
                entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
                max_share = float(np.max(probs)) if probs.size else None
                target_rows.append({"row_type": "source_outgoing_specificity", "task037_global_index": uid,
                                    "domain_id": domains[uid], "video_index": vi, "video_id": videos[vi]["video_id"],
                                    "class_name": videos[vi]["class_name"], "source_position": source + 1,
                                    "target_position": "", "n_relations": int(vals.size), "variance": float(np.var(vals)),
                                    "entropy_nats": entropy, "max_target_share": max_share,
                                    "effective_target_count": float(np.exp(entropy)),
                                    "top_target_position": int(np.asarray([t for t in range(T_MODEL) if t != source])[int(np.argmax(vals))] + 1),
                                    "top_target_A": max_v})
            source_profiles[(uid, vi)] = (out_means, out_maxes, out_energies)
            for label, vals in (("mean_A", out_means), ("max_A", out_maxes), ("squared_energy", out_energies)):
                mag_correlation_rows.append({"scope": "unit_video", "task037_global_index": uid,
                                             "domain_id": domains[uid], "video_index": vi,
                                             "metric": label, **_corr(source_mag[vi], vals)})
        for label, other in (("mean_A", all_out_mean[uid]), ("max_A", all_out_max[uid]),
                             ("squared_energy", all_out_energy[uid])):
            mag_correlation_rows.append({"scope": "unit_all_videos", "task037_global_index": uid,
                                         "domain_id": domains[uid], "video_index": "", "metric": label,
                                         **_corr(all_magnitude[uid], other)})
    for ui, uid in enumerate(unit_ids):
        for vi in range(30):
            for target in range(T_MODEL):
                incoming = [float(directed[vi, ui, si, target]) for si, source in enumerate(sources) if source != target]
                target_rows.append({"row_type": "target_incoming_source_variance", "task037_global_index": uid,
                                    "domain_id": domains[uid], "video_index": vi, "video_id": videos[vi]["video_id"],
                                    "class_name": videos[vi]["class_name"], "source_position": "",
                                    "target_position": target + 1, "n_relations": len(incoming),
                                    "variance": float(np.var(incoming)), "mean_incoming_A": float(np.mean(incoming)),
                                    "max_incoming_A": float(np.max(incoming))})
    write_csv(phase_root / "task042_phase_k_source_magnitude_audit.csv", magnitude_rows + mag_correlation_rows)
    write_csv(phase_root / "task042_phase_k_target_specificity.csv", target_rows)

    # Influence distribution by each exact lag, without merging lag bins.
    lag_rows = []
    for lag in range(1, T_MODEL):
        vals = [float(directed[vi, ui, si, ti]) for vi in range(30) for ui in range(len(unit_ids))
                for si, ti, _, _, this_lag in coords if this_lag == lag]
        lag_rows.append({"scope": "global", "scope_id": "all", "lag": lag, **_stats(vals)})
        for domain in DOMAINS:
            domain_units = [ui for ui, uid in enumerate(unit_ids) if domains[uid] == domain]
            dvals = [float(directed[vi, ui, si, ti]) for vi in range(30) for ui in domain_units
                     for si, ti, _, _, this_lag in coords if this_lag == lag]
            lag_rows.append({"scope": "domain", "scope_id": domain, "lag": lag, **_stats(dvals)})
        for ui, uid in enumerate(unit_ids):
            uvals = [float(directed[vi, ui, si, ti]) for vi in range(30)
                     for si, ti, _, _, this_lag in coords if this_lag == lag]
            lag_rows.append({"scope": "unit", "scope_id": uid, "task037_global_index": uid,
                             "domain_id": domains[uid], "lag": lag, **_stats(uvals)})
    write_csv(phase_root / "task042_phase_k_lag_analysis.csv", lag_rows)

    # Video pair reproducibility and class-position-balanced concatenated prototypes.
    stability_rows = []
    class_names = sorted({str(v["class_name"]) for v in videos.values()})
    video_by_class_pos = {(str(v["class_name"]), int(v["class_position"])): vi for vi, v in videos.items()}
    for uid in unit_ids:
        vecs = unit_video_vectors[uid]
        pair_metrics = {"same": {"pearson": [], "spearman": [], "cosine": []},
                        "different": {"pearson": [], "spearman": [], "cosine": []}}
        for va in range(30):
            for vb in range(va + 1, 30):
                ma, mb = videos[va], videos[vb]
                corr = _corr(vecs[va], vecs[vb])
                cosine = 1.0 - _cosine_distance(vecs[va], vecs[vb])
                same = str(ma["class_name"]) == str(mb["class_name"])
                label = "same" if same else "different"
                pair_metrics[label]["pearson"].append(corr["pearson"])
                pair_metrics[label]["spearman"].append(corr["spearman"])
                pair_metrics[label]["cosine"].append(cosine)
                stability_rows.append({"row_type": "video_pair", "task037_global_index": uid,
                                       "domain_id": domains[uid], "video_index_a": va, "video_index_b": vb,
                                       "class_a": ma["class_name"], "class_b": mb["class_name"],
                                       "same_class": same, "pearson": corr["pearson"], "spearman": corr["spearman"],
                                       "cosine": cosine})
        for condition in ("same", "different"):
            for metric, values in pair_metrics[condition].items():
                stability_rows.append({"row_type": "pairwise_summary", "task037_global_index": uid,
                                       "domain_id": domains[uid], "pair_condition": condition, "metric": metric,
                                       **_stats([v for v in values if v is not None])})
        prototypes = {}
        for pos in (1, 2, 3):
            prototypes["P%d" % pos] = np.concatenate([vecs[video_by_class_pos[(cls, pos)]] for cls in class_names])
        for left, right in (("P1", "P2"), ("P1", "P3"), ("P2", "P3")):
            corr = _corr(prototypes[left], prototypes[right])
            stability_rows.append({"row_type": "balanced_position_pair", "task037_global_index": uid,
                                   "domain_id": domains[uid], "left": left, "right": right,
                                   "pearson": corr["pearson"], "spearman": corr["spearman"],
                                   "cosine": 1.0 - _cosine_distance(prototypes[left], prototypes[right])})
        pair_groups = {"P12": (1, 2), "P13": (1, 3), "P23": (2, 3)}
        for name, positions in pair_groups.items():
            prototypes[name] = np.concatenate([np.mean([vecs[video_by_class_pos[(cls, p)]] for p in positions], axis=0)
                                               for cls in class_names])
        for left, right in (("P12", "P3"), ("P13", "P2"), ("P23", "P1")):
            corr = _corr(prototypes[left], prototypes[right])
            stability_rows.append({"row_type": "balanced_average_vs_heldout", "task037_global_index": uid,
                                   "domain_id": domains[uid], "left": left, "right": right,
                                   "pearson": corr["pearson"], "spearman": corr["spearman"],
                                   "cosine": 1.0 - _cosine_distance(prototypes[left], prototypes[right])})
    write_csv(phase_root / "task042_phase_k_video_stability.csv", stability_rows)

    # Directed relation geometry in the frozen BMS groups.
    domain_geometry_rows, pair_distance = [], {}
    for domain in DOMAINS:
        ids = [uid for uid in unit_ids if domains[uid] == domain]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                distance = _cosine_distance(packed[i].reshape(-1), packed[j].reshape(-1))
                pair_distance[(i, j)] = distance
                domain_geometry_rows.append({"domain_id": domain, "task037_global_index_i": i,
                                             "task037_global_index_j": j, "unit_type_i": unit_meta[i]["unit_type"],
                                             "unit_type_j": unit_meta[j]["unit_type"], "cosine_distance_d_dir": distance,
                                             "cosine_similarity": 1.0 - distance if math.isfinite(distance) else None,
                                             "feature_count": int(packed[i].size)})
    write_csv(phase_root / "task042_phase_k_domain_geometry.csv", domain_geometry_rows)

    # Convex collective leave-one-out coverage, on full video-concatenated directed patterns.
    _, phase_j, _ = _task042_runtime(repo)
    cover_rows, residual_rows, mixed_rows = [], [], []
    residual_npz_data = {}
    for domain in DOMAINS:
        ids = [uid for uid in unit_ids if domains[uid] == domain]
        matrix = np.stack([packed[uid].reshape(-1) for uid in ids])
        for uid in ids:
            competitors = [other for other in ids if other != uid]
            alpha, residual, delta = phase_j.solve_simplex_coverage(packed[uid].reshape(-1),
                                                                     [packed[o].reshape(-1) for o in competitors])
            reconstructed = packed[uid].reshape(-1) - residual
            require(np.all(alpha >= -1e-10) and abs(float(alpha.sum()) - 1.0) <= 1e-8,
                    "simplex coverage coefficients violate the convex constraints")
            require(np.allclose(residual, packed[uid].reshape(-1) - reconstructed, rtol=1e-12, atol=1e-12),
                    "leave-one-out residual identity failed")
            residual_map = residual_to_relation_map(residual, 30, T_MODEL)
            residual_npz_data[str(uid)] = residual_map.astype(np.float32)
            flat_abs = np.abs(residual)
            top_idx = np.argsort(flat_abs)[-10:][::-1]
            top_relations = []
            for index in top_idx:
                video_idx, in_video = divmod(int(index), 14 * 15)
                source_slot, rem = divmod(in_video, 15)
                source_pos = sources[source_slot] + 1
                target_candidates = [t for t in range(1, T_MODEL + 1) if t != source_pos]
                target_pos = target_candidates[rem]
                top_relations.append({"video_index": video_idx, "source_position": source_pos,
                                      "target_position": target_pos, "signed_residual": float(residual[index]),
                                      "absolute_residual": float(flat_abs[index])})
            norm = float(np.linalg.norm(packed[uid].reshape(-1)))
            row = {"domain_id": domain, "task037_global_index": uid, "unit_type": unit_meta[uid]["unit_type"],
                   "competitor_unit_ids": competitors, "simplex_weights": {str(other): float(w) for other, w in zip(competitors, alpha)},
                   "target_relation_norm": norm, "residual_norm": float(np.linalg.norm(residual)),
                   "normalized_leave_one_out_residual_delta": delta,
                   "top_residual_relations": top_relations}
            cover_rows.append(row)
            if domain == "271":
                mixed_rows.append({"row_type": "leave_one_out_coverability", "domain_id": domain,
                                   "task037_global_index": uid, "unit_type": unit_meta[uid]["unit_type"],
                                   "competitor_unit_ids": competitors, "simplex_weights": row["simplex_weights"],
                                   "delta": delta, "residual_norm": float(np.linalg.norm(residual)),
                                   "top_residual_relations": top_relations})
            coords_cursor = 0
            for vi in range(30):
                for si, source in enumerate(sources):
                    for target in range(T_MODEL):
                        if target == source:
                            continue
                        value = float(residual[coords_cursor]); coords_cursor += 1
                        residual_rows.append({"domain_id": domain, "task037_global_index": uid,
                                              "video_index": vi, "video_id": videos[vi]["video_id"],
                                              "class_name": videos[vi]["class_name"], "source_position": source + 1,
                                              "target_position": target + 1, "lag": abs(target - source),
                                              "signed_residual": value, "absolute_residual": abs(value)})
            require(coords_cursor == residual.size, "residual relation-map reshape lost or added entries")
        if domain == "271":
            for left_index in range(len(ids)):
                for right_index in range(left_index + 1, len(ids)):
                    i, j = ids[left_index], ids[right_index]
                    ti, tj = str(unit_meta[i]["unit_type"]), str(unit_meta[j]["unit_type"])
                    if ti == "attention_head" and tj == "attention_head": pair_type = "Attention_to_Attention"
                    elif ti != "attention_head" and tj != "attention_head": pair_type = "FFN_to_FFN"
                    else: pair_type = "Attention_to_FFN"
                    mixed_rows.append({"row_type": "pairwise_relation_distance", "domain_id": domain,
                                       "task037_global_index_i": i, "task037_global_index_j": j,
                                       "unit_type_i": ti, "unit_type_j": tj, "pair_type": pair_type,
                                       "cosine_distance_d_dir": _cosine_distance(packed[i].reshape(-1), packed[j].reshape(-1))})
    write_csv(phase_root / "task042_phase_k_coverability.csv", cover_rows)
    write_csv(phase_root / "task042_phase_k_residual_relation_map.csv", residual_rows)
    np.savez_compressed(phase_root / "task042_phase_k_residual_relation_map.npz", **residual_npz_data)
    write_csv(phase_root / "task042_phase_k_mixed_domain.csv", mixed_rows)

    # Compare unit-space geometry with the frozen Phase-J output only.
    phase_j_root = Path(config["base_output"]) / "phase_j"
    phasej_vectors, phasej_length = _phase_j_vectors(phase_j_root / "task042_phase_j_conditional_relation.csv", unit_ids)
    phasej_rows = []
    all_j, all_k, nearest_changes, pair_rank_changes = [], [], [], []
    for domain in DOMAINS:
        ids = [uid for uid in unit_ids if domains[uid] == domain]
        pair_rows = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                dj = _cosine_distance(phasej_vectors[i], phasej_vectors[j])
                dk = pair_distance[(i, j)]
                pair_rows.append((i, j, dj, dk))
                all_j.append(dj); all_k.append(dk)
        jrank = rankdata([r[2] for r in pair_rows]) if pair_rows else []
        krank = rankdata([r[3] for r in pair_rows]) if pair_rows else []
        for idx, (i, j, dj, dk) in enumerate(pair_rows):
            rank_change = float(abs(jrank[idx] - krank[idx]))
            pair_rank_changes.append(rank_change)
            phasej_rows.append({"row_type": "unit_pair_geometry", "domain_id": domain,
                                "task037_global_index_i": i, "task037_global_index_j": j,
                                "phase_j_vector_length": len(phasej_vectors[i]), "phase_k_vector_length": packed[i].size,
                                "phase_j_cosine_distance": dj, "phase_k_cosine_distance": dk,
                                "phase_j_within_domain_pair_rank": float(jrank[idx]),
                                "phase_k_within_domain_pair_rank": float(krank[idx]),
                                "absolute_rank_change": rank_change})
        for uid in ids:
            peers = [other for other in ids if other != uid]
            j_nearest = min(peers, key=lambda other: _cosine_distance(phasej_vectors[uid], phasej_vectors[other]))
            k_nearest = min(peers, key=lambda other: pair_distance[tuple(sorted((uid, other)))])
            nearest_changes.append(j_nearest != k_nearest)
            phasej_rows.append({"row_type": "nearest_neighbor", "domain_id": domain,
                                "task037_global_index": uid, "phase_j_nearest_unit": j_nearest,
                                "phase_k_nearest_unit": k_nearest, "nearest_neighbor_changed": j_nearest != k_nearest})
    pair_corr = _corr(all_j, all_k)
    phasej_rows.append({"row_type": "pair_distance_correlation", **pair_corr,
                        "phase_j_vector_length": phasej_length, "phase_k_vector_length": int(packed[unit_ids[0]].size),
                        "nearest_neighbor_change_count": int(sum(nearest_changes)),
                        "nearest_neighbor_comparison_count": len(nearest_changes),
                        "mean_absolute_within_domain_rank_change": float(np.mean(pair_rank_changes)) if pair_rank_changes else None})
    write_csv(phase_root / "task042_phase_k_phasej_comparison.csv", phasej_rows)

    # Descriptor complementarity; report separate associations, never a combined score.
    descriptor_names = ("D_abs", "D_rel", "D_st")
    descriptor_values = {int(r["task037_global_index"]): {name: float(r[name]) for name in descriptor_names}
                         for r in read_csv(Path(config["unit_manifest"])) if int(r["task037_global_index"]) in set(unit_ids)}
    desc_rows = []
    for descriptor in descriptor_names:
        dd, td = [], []
        for domain in DOMAINS:
            ids = [uid for uid in unit_ids if domains[uid] == domain]
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    i, j = ids[x], ids[y]
                    dd.append(pair_distance[(i, j)])
                    td.append(abs(descriptor_values[i][descriptor] - descriptor_values[j][descriptor]))
        desc_rows.append({"comparison": "pairwise_directed_distance_vs_descriptor_difference",
                          "descriptor": descriptor, **_corr(dd, td)})
        desc_rows.append({"comparison": "unit_coverability_delta_vs_descriptor_value", "descriptor": descriptor,
                          **_corr([r["normalized_leave_one_out_residual_delta"] for r in cover_rows],
                                  [descriptor_values[int(r["task037_global_index"])][descriptor] for r in cover_rows])})
    write_csv(phase_root / "task042_phase_k_descriptor_complementarity.csv", desc_rows)

    asym_global = [row["relative_asymmetry"] for row in direction_rows]
    specificity_values = [row["effective_target_count"] for row in target_rows if row["row_type"] == "source_outgoing_specificity"]
    domain_cover = {domain: _stats([r["normalized_leave_one_out_residual_delta"] for r in cover_rows if r["domain_id"] == domain])
                    for domain in DOMAINS}
    same_corr = {metric: _stats([row[metric] for row in stability_rows if row.get("row_type") == "video_pair"
                                 and row.get("same_class") is True])
                 for metric in ("pearson", "spearman", "cosine")}
    diff_corr = {metric: _stats([row[metric] for row in stability_rows if row.get("row_type") == "video_pair"
                                 and row.get("same_class") is False])
                 for metric in ("pearson", "spearman", "cosine")}
    mag_agg = [r for r in mag_correlation_rows if r["scope"] == "unit_all_videos"]
    summary = {
        "task": config["task"], "analysis_status": "completed", "branch": BRANCH,
        "git_head": config["git_head"], "checkpoint_sha256": CHECKPOINT_SHA,
        "unit_count": len(unit_ids), "unit_identities": config["unit_identities"], "domains": list(DOMAINS),
        "video_count": 30, "class_count": 10, "videos_per_class": 3,
        "no_op_gate": noop_gate,
        "intervention_locality": {"exact_non_source_positions": all(str(r.get("non_source_positions_exact", "")).lower() == "true" for r in manifest_rows),
                                   "source_replacement_max_abs_error": max(float(r.get("source_replacement_max_abs_error") or 0.0) for r in manifest_rows),
                                   "source_self_relations_excluded": True},
        "directed_asymmetry": _stats(asym_global),
        "target_specificity_effective_target_count": _stats(specificity_values),
        "source_magnitude_correlations_unit_all_videos": mag_agg,
        "lag_influence": {str(lag): next(r for r in lag_rows if r["scope"] == "global" and int(r["lag"]) == lag)
                          for lag in range(1, T_MODEL)},
        "video_stability_same_class": same_corr, "video_stability_different_class": diff_corr,
        "coverability_by_domain": domain_cover, "coverability_values": [r["normalized_leave_one_out_residual_delta"] for r in cover_rows],
        "domain_271_mixed_unit_count": sum(1 for uid in unit_ids if domains[uid] == "271"),
        "phase_j_comparison": {"pair_distance_correlation": pair_corr,
                               "nearest_neighbor_change_count": int(sum(nearest_changes)),
                               "nearest_neighbor_comparison_count": len(nearest_changes),
                               "mean_absolute_within_domain_rank_change": float(np.mean(pair_rank_changes)) if pair_rank_changes else None,
                               "phase_j_vector_length": phasej_length, "phase_k_vector_length": int(packed[unit_ids[0]].size)},
        "descriptor_complementarity": desc_rows,
        "required_questions": {
            "A": {"answer": "PASS", "evidence": "A common patch_embed output hook implements local replacement; no-op replay passed; all non-source positions were bitwise unchanged; source replacement error was zero."},
            "B": {"answer": "REVIEW_NUMERIC_DISTRIBUTION", "evidence": _stats(asym_global)},
            "C": {"answer": "REVIEW_NUMERIC_DISTRIBUTION", "evidence": _stats(specificity_values)},
            "D": {"answer": "REVIEW_NUMERIC_DISTRIBUTION", "evidence": mag_agg},
            "E": {"answer": "REVIEW_LAG_DISTRIBUTIONS", "evidence": {str(lag): next(r for r in lag_rows if r["scope"] == "global" and int(r["lag"]) == lag) for lag in range(1, T_MODEL)}},
            "F": {"answer": "REVIEW_CLASS_BALANCED_STABILITY", "same_class": same_corr, "different_class": diff_corr},
            "G": {"answer": "REVIEW_RAW_DOMAIN_COVERABILITY", "evidence": domain_cover},
            "H": {"answer": "REVIEW_DOMAIN_271_PAIR_TYPES", "evidence": "Pair distances and leave-one-out simplex coefficients are in task042_phase_k_mixed_domain.csv."},
            "I": {"answer": "REVIEW_PHASEJ_GEOMETRY_CHANGE", "evidence": "See phase_j_comparison below."},
            "J": {"answer": "REVIEW_DESCRIPTOR_CORRELATIONS", "evidence": desc_rows},
        },
        "decision": "B",
        "decision_label": "DIRECTED_TEMPORAL_INNOVATION_RELATION_WEAK_OR_UNRESOLVED",
        "decision_rationale": "Conservative predeclared choice pending a full joint reading of the required qualitative evidence; no post-hoc numeric threshold or pruning claim is used.",
        "pruning": False, "finetuning": False, "performance_oracle": False,
    }
    summary["required_questions"]["I"]["evidence"] = summary["phase_j_comparison"]
    runtime_summary = {
        "analysis_status": "completed", "analysis_forwards_expected": 450,
        "analysis_forwards_observed": int(sum(r["analysis_forward_count"] for r in shard_runtime)),
        "no_op_replay_forwards_expected": 1,
        "no_op_replay_forwards_observed": int(sum(r["no_op_replay_forward_count"] for r in shard_runtime)),
        "total_forwards_observed": int(sum(r["total_forward_count"] for r in shard_runtime)),
        "cuda_analysis_forward_event_seconds_sum": float(sum(r["cuda_analysis_forward_event_seconds"] for r in shard_runtime)),
        "cuda_noop_forward_event_seconds_sum": float(sum(r["cuda_noop_forward_event_seconds"] for r in shard_runtime)),
        "parallel_shard_wall_seconds_max": float(max(r["worker_wall_seconds"] for r in shard_runtime)),
        "peak_gpu_memory_allocated_bytes_by_gpu": {str(r["physical_gpu"]): r["peak_gpu_memory_allocated_bytes"] for r in shard_runtime},
        "peak_gpu_memory_reserved_bytes_by_gpu": {str(r["physical_gpu"]): r["peak_gpu_memory_reserved_bytes"] for r in shard_runtime},
        "relation_tensor_shape": [30, 13, 14, 16],
        "valid_directed_relations": int(30 * 13 * 14 * 15),
        "saved_relation_tensor_uncompressed_bytes": int(directed.nbytes + source_mag.nbytes),
        "shard_file_bytes": int(sum(r["shard_file_bytes"] for r in shard_runtime)),
        "gpu_ids_used": [0, 1], "amp": False, "dtype": "float32",
        "patch_embedding_geometry": shard_runtime[0]["patch_embedding_geometry"],
    }
    write_json(phase_root / "task042_phase_k_summary.json", summary)
    write_json(phase_root / "task042_phase_k_runtime_summary.json", runtime_summary)
    report = _render_report(summary, runtime_summary, cover_rows, lag_rows, desc_rows, mag_correlation_rows,
                            direction_rows, target_rows, stability_rows, phasej_rows, mixed_rows)
    (phase_root / "task042_phase_k_report.md").write_text(report, encoding="utf-8")
    print("PHASE_K_ANALYSIS_COMPLETE decision=%s" % summary["decision_label"], flush=True)


def _render_report(summary: Mapping[str, Any], runtime: Mapping[str, Any], cover_rows: Sequence[Mapping[str, Any]],
                   lag_rows: Sequence[Mapping[str, Any]], desc_rows: Sequence[Mapping[str, Any]],
                   mag_rows: Sequence[Mapping[str, Any]], direction_rows: Sequence[Mapping[str, Any]],
                   target_rows: Sequence[Mapping[str, Any]], stability_rows: Sequence[Mapping[str, Any]],
                   phasej_rows: Sequence[Mapping[str, Any]], mixed_rows: Sequence[Mapping[str, Any]]) -> str:
    asym = summary["directed_asymmetry"]
    target = summary["target_specificity_effective_target_count"]
    lines = [
        "# Task042 Phase K — Directed Temporal Innovation Influence Diagnostic", "",
        "## Frozen scope and intervention", "",
        "This is a representation diagnostic on 13 frozen units in BMS domains 415, 103, 76, and 271 and the authoritative 30-video, 10-class calibration cohort. It performs no unit masking, pruning, fine-tuning, accuracy/CE evaluation, or validation-performance oracle.", "",
        "The intervention is applied once at the common `patch_embed` output `[B,C,T,H,W]`, with `T=16`, temporal axis 2, and a temporal Conv3d kernel/stride of 2. Each temporal position therefore covers two adjacent transformed input slots (a tubelet), not one raw video frame. The audit CSV maps each model temporal position to the exact LoopPadding-selected input slots and raw frame filenames.", "",
        "For each interior source position, the local temporal background is the arithmetic mean of its two neighboring positions. The source innovation is replaced with that background; all other positions are checked bitwise unchanged. The diagonal source-to-self entries are excluded from the directed relation vectors.", "",
        "## Computational and implementation checks", "",
        f"- No-op replay gate: `passed={summary['no_op_gate'].get('passed')}`, maximum target-feature absolute difference `{summary['no_op_gate'].get('max_abs_all_units')}` (FP32 tolerance 1e-6).",
        f"- Locality: all non-source positions exact = `{summary['intervention_locality']['exact_non_source_positions']}`; maximum source replacement error = `{summary['intervention_locality']['source_replacement_max_abs_error']}`.",
        f"- Forward count: `{runtime['analysis_forwards_observed']}` analysis forwards plus `{runtime['no_op_replay_forwards_observed']}` no-op replay.",
        f"- GPU event time sum `{runtime['cuda_analysis_forward_event_seconds_sum']:.3f}` seconds; shard wall time max `{runtime['parallel_shard_wall_seconds_max']:.3f}` seconds; GPUs `{runtime['gpu_ids_used']}`.", "",
        "## Required questions", "",
        f"**A. Exact and local?** Yes, by construction and passed no-op/locality checks above.",
        f"**B. Meaningfully asymmetric?** Relative asymmetry summary across reciprocal interior pairs: median `{asym.get('median')}`, mean `{asym.get('mean')}`, quartiles `{asym.get('q25')}`–`{asym.get('q75')}`. Interpret from full per-unit/domain/video distributions in `task042_phase_k_directionality.csv`; no threshold was introduced.",
        f"**C. Target-specific rather than uniform?** Effective target count summary: median `{target.get('median')}` of 15 possible targets; see entropy, maximum target share, and full source/target variance records in `task042_phase_k_target_specificity.csv`.",
        f"**D. More than source magnitude?** Per-unit/video and pooled magnitude correlations are recorded for mean, max, and squared outgoing energy in `task042_phase_k_source_magnitude_audit.csv`; these are diagnostics only and do not residualize influence.",
        "**E. Local and long-range influence?** The exact lag 1–15 distributions are reported separately in `task042_phase_k_lag_analysis.csv`; no hand-designed lag bins were applied.",
        "**F. Reproducible across videos?** Same-class and different-class Pearson, Spearman, and cosine comparisons, plus P1/P2/P3 class-balanced comparisons, are in `task042_phase_k_video_stability.csv`.",
        "**G. Nontrivial same-domain coverability?** Raw leave-one-unit-out normalized residuals and simplex weights are in `task042_phase_k_coverability.csv`; residuals are expanded to video/source/target maps in `task042_phase_k_residual_relation_map.csv` and `.npz`.",
        "**H. Attention and FFN in a common relation space?** Domain 271 reports Attention↔Attention, FFN↔FFN, and cross-type distances and per-unit collective weights in `task042_phase_k_mixed_domain.csv`.",
        "**I. New geometry beyond Phase J?** The frozen Phase-J artifact is compared without inference reruns; pair-distance correlation, nearest-neighbor changes, and ranking changes are in `task042_phase_k_phasej_comparison.csv`.",
        "**J. Complementary to descriptors?** Separate correlations against D_abs, D_rel, and D_st are in `task042_phase_k_descriptor_complementarity.csv`; no combined score is formed.", "",
        "## Predeclared decision", "",
        f"`{summary['decision_label']}`. This conservative B outcome is used unless the complete required evidence qualitatively supports every condition for A; no post-hoc numeric thresholds are introduced. Phase K stops here and makes no pruning or training recommendation.", "",
        "## Artifact inventory", "",
        "The required CSV/JSON outputs are stored alongside this report in the Phase-K output directory. The intervention and source-position audits preserve video/class identity and temporal-position-to-input-frame support.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--phase-root", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.phase_root is None:
        args.phase_root = args.base_output / "phase_k"
    if args.prepare_only:
        _prepare(args.repo_root, args.base_output, args.phase_root)
        print("PHASE_K_PREPARED phase_root=%s" % args.phase_root, flush=True)
    elif args.shard is not None:
        if args.gpu is None:
            parser.error("--shard requires --gpu")
        run_shard(args)
    elif args.analyze:
        analyze_phase_k(args)
    else:
        parser.error("select --prepare-only, --shard, or --analyze")


if __name__ == "__main__":
    main()

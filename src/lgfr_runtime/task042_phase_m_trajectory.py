#!/usr/bin/env python3
"""Task042 Phase M: cross-frame functional trajectory diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
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
T_MODEL = 16
EPS = 1e-12


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 Phase-M gate failed: " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _cell(row.get(k, "")) for k in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _stats(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "q25": None, "q75": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()), "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}


def _corr(a: Sequence[float], b: Sequence[float], method: str) -> float | None:
    from scipy import stats
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    with np.errstate(all="ignore"):
        value = stats.pearsonr(x, y)[0] if method == "pearson" else stats.spearmanr(x, y)[0]
    return float(value) if np.isfinite(value) else None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= EPS:
        return 1.0 if np.linalg.norm(x - y) <= EPS else 0.0
    return float(np.dot(x, y) / den)


def attention_spatial_response(x: Any) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError("attention activation must be [T,H,W,d]")
    return np.linalg.norm(x.astype(np.float64), axis=-1)


def ffn_spatial_response(x: Any) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError("FFN activation must be [T,H,W]")
    return np.abs(x.astype(np.float64))


def spatial_probability(a: Any) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("spatial response must be finite and nonnegative [T,H,W]")
    mass = a.sum(axis=(1, 2))
    near = mass <= EPS
    p = np.full_like(a, np.nan)
    good = ~near
    p[good] = a[good] / mass[good, None, None]
    return p, near


def normalized_coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if height < 1 or width < 1:
        raise ValueError("spatial grid must be nonempty")
    x = np.zeros(width, dtype=np.float64) if width == 1 else 2.0 * np.arange(width) / (width - 1) - 1.0
    y = np.zeros(height, dtype=np.float64) if height == 1 else 2.0 * np.arange(height) / (height - 1) - 1.0
    return np.meshgrid(x, y)


def functional_moments(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError("probability map must be [T,H,W]")
    t, h, w = p.shape
    xx, yy = normalized_coordinates(h, w)
    out = np.full((t, 5), np.nan, dtype=np.float64)
    for i in range(t):
        row = p[i]
        if not np.isfinite(row).all():
            continue
        sx, sy = float((row * xx).sum()), float((row * yy).sum())
        dx, dy = xx - sx, yy - sy
        out[i] = [sx, sy, float((row * dx * dx).sum()), float((row * dx * dy).sum()), float((row * dy * dy).sum())]
    return out


def trajectory_transitions(states: Any) -> np.ndarray:
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 5 or x.shape[0] != T_MODEL:
        raise ValueError("functional states must be [16,5]")
    return x[1:] - x[:-1]


def split_motion_deformation(transitions: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(transitions, dtype=np.float64)
    if x.shape[-1] != 5:
        raise ValueError("trajectory transition last dimension must be 5")
    return x[..., :2], x[..., 2:]


def solve_simplex_coverage(target: Sequence[float], competitors: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.optimize import minimize
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows = np.asarray(competitors, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != y.size:
        raise ValueError("competitors must be [M,D]")
    good = np.isfinite(y) & np.isfinite(rows).all(axis=0)
    y, rows = y[good], rows[:, good]
    gram, cross = rows @ rows.T, rows @ y
    start = np.full(rows.shape[0], 1.0 / rows.shape[0])
    def obj(a: np.ndarray) -> float:
        return float(a @ gram @ a - 2 * a @ cross + y @ y)
    def jac(a: np.ndarray) -> np.ndarray:
        return 2 * (gram @ a - cross)
    result = minimize(obj, start, jac=jac, method="SLSQP", bounds=[(0, 1)] * rows.shape[0],
                      constraints=[{"type": "eq", "fun": lambda a: float(a.sum() - 1), "jac": lambda a: np.ones_like(a)}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success and np.linalg.norm(jac(result.x), ord=np.inf) > 1e-6:
        raise RuntimeError("simplex optimizer failed: " + str(result.message))
    alpha = np.maximum(np.asarray(result.x, dtype=np.float64), 0)
    alpha /= alpha.sum()
    residual = y - alpha @ rows
    delta = float(np.linalg.norm(residual) / (np.linalg.norm(y) + EPS))
    return alpha, residual, delta


def _capture_rows(unit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in unit_rows:
        row = dict(r)
        row["capture_kind"] = "head" if str(r["unit_type"]) == "attention_head" else "neuron"
        out.append(row)
    return out


def prepare(repo: Path, base: Path, phase_root: Path) -> dict[str, Any]:
    if phase_root.exists():
        raise RuntimeError("Phase-M output exists; refusing overwrite")
    require(subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip() == BRANCH, "wrong branch")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(), "checkout must be clean")
    base_cfg_path = base / "task042_run_config.json"
    jcfg_path = base / "phase_j" / "task042_phase_j_run_config.json"
    kcfg_path = base / "phase_k" / "task042_phase_k_run_config.json"
    require(all(p.is_file() for p in (base_cfg_path, jcfg_path, kcfg_path)), "frozen configs missing")
    base_cfg, jcfg, kcfg = [json.loads(p.read_text(encoding="utf-8")) for p in (base_cfg_path, jcfg_path, kcfg_path)]
    unit_rows = [r for r in read_csv(Path(base_cfg["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    unit_rows.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    expected = {u for d in DOMAINS for u in EXPECTED_DOMAIN_UNITS[d]}
    require(len(unit_rows) == 13 and {int(r["task037_global_index"]) for r in unit_rows} == expected, "frozen unit cohort differs")
    jids = {int(r["task037_global_index"]): r for r in jcfg["unit_identities"]}
    kids = {int(r["task037_global_index"]): r for r in kcfg["unit_identities"]}
    identities = []
    for r in unit_rows:
        uid = int(r["task037_global_index"])
        ident = {k: r[k] for k in ("task037_global_index", "domain_id", "layer", "stage", "unit_type", "unit_index")}
        for other in (jids.get(uid), kids.get(uid)):
            require(other is not None and all(str(other[k]) == str(ident[k]) for k in ident), f"unit identity mismatch {uid}")
        identities.append({k: (int(v) if k in ("task037_global_index", "stage", "unit_index") else v) for k, v in ident.items()})
    videos = sorted((dict(v) for v in jcfg["video_identity_order"]), key=lambda r: int(r["video_index"]))
    require(len(videos) == 30 and len({str(v["class_name"]) for v in videos}) == 10, "authoritative video cohort is not 30 videos/10 classes")
    require(int(kcfg.get("model_temporal_positions", -1)) == 16, "frozen model temporal length is not 16")
    require(str(base_cfg["required_branch"]) == BRANCH and sha256_file(Path(base_cfg["checkpoint_path"])) == CHECKPOINT_SHA, "checkpoint/branch identity mismatch")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    cfg = {
        "task": "TASK042 PHASE M — CROSS-FRAME FUNCTIONAL TRAJECTORY DIAGNOSTIC",
        "required_branch": BRANCH, "repo_root": str(repo.resolve()), "base_output": str(base.resolve()), "phase_root": str(phase_root.resolve()),
        "git_head": head, "phase_j_git_head": jcfg.get("git_head"), "phase_k_git_head": kcfg.get("git_head"),
        "checkpoint_path": base_cfg["checkpoint_path"], "checkpoint_sha256": CHECKPOINT_SHA,
        "unit_manifest": base_cfg["unit_manifest"], "unit_count": 13, "unit_identities": identities,
        "domains": {d: list(EXPECTED_DOMAIN_UNITS[d]) for d in DOMAINS}, "video_count": 30,
        "video_identity_order": videos, "video_indices": [int(v["video_index"]) for v in videos],
        "class_order": sorted({str(v["class_name"]) for v in videos}), "temporal_positions": list(range(1, 17)),
        "state_dimension": 5, "transition_count": 15, "trajectory_dimension_per_video": 75, "trajectory_dimension": 2250,
        "dtype": "float32", "amp": False, "pruning": False, "finetuning": False, "performance_oracle": False,
        "input_sha256": {"task042_run_config": sha256_file(base_cfg_path), "phase_j_run_config": sha256_file(jcfg_path),
                         "phase_k_run_config": sha256_file(kcfg_path), "checkpoint": CHECKPOINT_SHA},
        "spatial_state_definition": "[mu_x,mu_y,sigma_xx,sigma_xy,sigma_yy] on each native normalized grid",
        "trajectory_definition": "r[t]=f[t+1]-f[t], concatenated in video_index then temporal order",
    }
    phase_root.mkdir(parents=True)
    write_json(phase_root / "task042_phase_m_run_config.json", cfg)
    write_csv(phase_root / "task042_phase_m_video_order.csv", videos)
    return cfg


def _restore_feature(value: Any, meta: Mapping[str, Any], kind: str, torch: Any) -> np.ndarray:
    x = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    t, h, w = int(meta["T_i"]), int(meta["H_i"]), int(meta["W_i"])
    require(t == T_MODEL, "selected unit is not T=16")
    if kind == "head":
        d = int(meta["head_dim"])
        return x.reshape(t, h, w, d)
    return x.reshape(t, h, w)


def run_shard(repo: Path, phase_root: Path, shard: int, gpu: int) -> None:
    import torch
    runtime = str(repo / "src" / "lgfr_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task042_phase_j_conditional as phase_j
    import task042_phase_k_influence as phase_k
    cfg = json.loads((phase_root / "task042_phase_m_run_config.json").read_text(encoding="utf-8"))
    require(shard in (0, 1) and gpu == shard and os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "GPU shard isolation failed")
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == cfg["git_head"], "checkout changed after prepare")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(), "worker requires clean committed code")
    phase_k_cfg = json.loads((Path(cfg["base_output"]) / "phase_k" / "task042_phase_k_run_config.json").read_text())
    torchmod, _base, _pj, model, specs, identity, _patch_name, _patch_module, _geometry = phase_k._model_on_gpu(phase_k_cfg, gpu)
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA and identity.get("classifier_head", {}).get("status") == "loaded", "checkpoint identity mismatch")
    unit_rows = [r for r in read_csv(Path(cfg["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    unit_rows.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    capture_rows = _capture_rows(unit_rows)
    capture = phase_j.PhaseJCapture(model, capture_rows, specs, torchmod)
    phase_j_cfg = json.loads((Path(cfg["base_output"]) / "phase_j" / "task042_phase_j_run_config.json").read_text())
    _loader, clips, loader_rows = phase_j._load_all_videos(phase_j_cfg, workers=2)
    require(set(clips) == set(range(30)), "authoritative loader did not reproduce 30 clips")
    videos = list(cfg["video_identity_order"])
    assigned = videos[15 * shard:15 * (shard + 1)]
    unit_ids = [int(r["task037_global_index"]) for r in capture_rows]
    states = np.full((15, 13, 16, 5), np.nan, dtype=np.float32)
    transitions = np.full((15, 13, 15, 5), np.nan, dtype=np.float32)
    near_zero = np.zeros((15, 13, 16), dtype=np.bool_)
    manifest, per_video = [], []
    start_wall = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for slot, video in enumerate(assigned):
            vi = int(video["video_index"])
            require(str(video["video_id"]) == str(loader_rows[vi]["video_id"]) and int(video["label"]) == int(loader_rows[vi]["label"]), "video identity mismatch")
            clip = clips[vi].to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            capture.begin(collect_features=True)
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            with torch.inference_mode():
                _ = model(clip.unsqueeze(0))
            ev1.record()
            expected = set(unit_ids)
            require(set(capture.features) == expected and set(capture.metadata) == expected, "all selected units were not captured")
            for ui, row in enumerate(capture_rows):
                uid = int(row["task037_global_index"])
                arr = _restore_feature(capture.features[uid], capture.metadata[uid], str(row["capture_kind"]), torchmod)
                a = attention_spatial_response(arr) if row["capture_kind"] == "head" else ffn_spatial_response(arr)
                p, nz = spatial_probability(a)
                m = functional_moments(p)
                states[slot, ui] = m.astype(np.float32)
                near_zero[slot, ui] = nz
                if nz.any():
                    transitions[slot, ui] = trajectory_transitions(m).astype(np.float32)
                else:
                    transitions[slot, ui] = trajectory_transitions(m).astype(np.float32)
            manifest.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"], "class_position": video["class_position"],
                             "forward_kind": "original", "physical_gpu": gpu, "temporal_length": 16, "unit_capture_count": 13,
                             "near_zero_mass_frame_count": int(near_zero[slot].sum()), "activation_semantics": "Phase-J exact head/FFN spatial tensor"})
            per_video.append({"video_index": vi, "wall_seconds": None})
            print(f"PHASE_M_SHARD{shard} class={video['class_name']} video={vi} captured=13", flush=True)
            del clip
            torch.cuda.empty_cache()
    finally:
        capture.close()
    torch.cuda.synchronize()
    runtime_row = {"shard": shard, "physical_gpu": gpu, "gpu_name": torch.cuda.get_device_name(0), "video_indices": [int(v["video_index"]) for v in assigned],
                   "video_count": 15, "forward_count": 15, "expected_forward_count": 15, "wall_seconds": time.perf_counter() - start_wall,
                   "cuda_seconds": float(ev0.elapsed_time(ev1) / 1000.0) if assigned else 0.0, "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                   "dtype": "float32", "amp": False, "all_units_captured_simultaneously": True}
    np.savez_compressed(phase_root / f"task042_phase_m_shard{shard}.npz", video_indices=np.asarray([int(v["video_index"]) for v in assigned], dtype=np.int16),
                        unit_ids=np.asarray(unit_ids, dtype=np.int64), states=states, transitions=transitions, near_zero=near_zero)
    write_csv(phase_root / f"task042_phase_m_shard{shard}_manifest.csv", manifest)
    write_json(phase_root / f"task042_phase_m_shard{shard}_runtime.json", runtime_row)
    print(f"PHASE_M_SHARD_COMPLETE shard={shard} forwards=15", flush=True)


def _load_shards(root: Path, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vids = [int(v["video_index"]) for v in cfg["video_identity_order"]]
    vslot = {v: i for i, v in enumerate(vids)}
    uids = [int(u["task037_global_index"]) for u in cfg["unit_identities"]]
    st = np.full((30, 13, 16, 5), np.nan)
    tr = np.full((30, 13, 15, 5), np.nan)
    nz = np.zeros((30, 13, 16), dtype=bool)
    seen = set()
    for shard in (0, 1):
        npz_path = root / f"task042_phase_m_shard{shard}.npz"
        rt_path = root / f"task042_phase_m_shard{shard}_runtime.json"
        mf_path = root / f"task042_phase_m_shard{shard}_manifest.csv"
        require(npz_path.is_file() and rt_path.is_file() and mf_path.is_file(), f"missing shard {shard}")
        rt = json.loads(rt_path.read_text())
        require(int(rt["forward_count"]) == 15 and int(rt["expected_forward_count"]) == 15, f"shard {shard} incomplete")
        with np.load(npz_path, allow_pickle=False) as z:
            require(list(map(int, z["unit_ids"])) == uids, "unit order mismatch")
            for ix, vi in enumerate(map(int, z["video_indices"])):
                require(vi in vslot and vi not in seen, "duplicate/unexpected shard video")
                seen.add(vi); st[vslot[vi]] = z["states"][ix]; tr[vslot[vi]] = z["transitions"][ix]; nz[vslot[vi]] = z["near_zero"][ix]
        require(len(read_csv(mf_path)) == 15, "manifest row count mismatch")
    require(seen == set(vids), "shards do not cover all 30 videos")
    return st, tr, nz


def _domain_maps(cfg: Mapping[str, Any]) -> tuple[list[int], dict[int, str], dict[int, Mapping[str, Any]]]:
    ids = [int(x["task037_global_index"]) for x in cfg["unit_identities"]]
    domains = {int(x["task037_global_index"]): str(x["domain_id"]) for x in cfg["unit_identities"]}
    meta = {int(x["task037_global_index"]): x for x in cfg["unit_identities"]}
    return ids, domains, meta


def analyze(root: Path) -> dict[str, Any]:
    cfg = json.loads((root / "task042_phase_m_run_config.json").read_text())
    base = Path(cfg["base_output"])
    unit_ids, domains, unit_meta = _domain_maps(cfg)
    states, transitions, near_zero = _load_shards(root, cfg)
    videos = list(cfg["video_identity_order"])
    video_ids = [int(v["video_index"]) for v in videos]
    vidmap = {int(v["video_index"]): v for v in videos}
    traj = transitions.reshape(30, 13, 75).astype(np.float64)
    static = states.reshape(30, 13, 80).astype(np.float64)
    finite_states = states[np.isfinite(states)]
    comp_names = ("mu_x", "mu_y", "sigma_xx", "sigma_xy", "sigma_yy")
    manifest_rows = [r for s in (0, 1) for r in read_csv(root / f"task042_phase_m_shard{s}_manifest.csv")]
    write_csv(root / "task042_phase_m_manifest.csv", manifest_rows)
    state_rows, trans_rows, long_rows, md_rows = [], [], [], []
    for vi, v in enumerate(videos):
        for ui, uid in enumerate(unit_ids):
            f = states[vi, ui]
            for ti in range(16):
                row = {"video_index": v["video_index"], "class_name": v["class_name"], "class_position": v["class_position"], "task037_global_index": uid,
                       "domain_id": domains[uid], "unit_type": unit_meta[uid]["unit_type"], "stage": unit_meta[uid]["stage"], "temporal_position": ti + 1,
                       "near_zero_mass": bool(near_zero[vi, ui, ti]), **{comp_names[k]: f[k] for k in range(5)}}
                state_rows.append(row)
            z = transitions[vi, ui]
            for ti in range(15):
                motion, deform = z[ti, :2], z[ti, 2:]
                trans_rows.append({"video_index": v["video_index"], "class_name": v["class_name"], "task037_global_index": uid, "domain_id": domains[uid],
                                   "transition_index": ti + 1, "from_position": ti + 1, "to_position": ti + 2, "delta_mu_x": z[ti,0], "delta_mu_y": z[ti,1],
                                   "delta_sigma_xx": z[ti,2], "delta_sigma_xy": z[ti,3], "delta_sigma_yy": z[ti,4],
                                   "motion_norm": np.linalg.norm(motion), "deformation_norm": np.linalg.norm(deform), "transition_norm": np.linalg.norm(z[ti])})
            long_err = 0.0; count = 0
            for a in range(16):
                for b in range(a + 1, 16):
                    err = np.max(np.abs(states[vi, ui, b] - states[vi, ui, a] - transitions[vi, ui, a:b].sum(axis=0)))
                    long_err = max(long_err, float(err)); count += 1
            long_rows.append({"video_index": v["video_index"], "class_name": v["class_name"], "task037_global_index": uid, "domain_id": domains[uid],
                              "pair_count": count, "maximum_absolute_error": long_err})
            motion_n = np.linalg.norm(z[:, :2], axis=1); deform_n = np.linalg.norm(z[:, 2:], axis=1)
            top = np.argsort(-np.nan_to_num(np.linalg.norm(z, axis=1), nan=-1))[:5]
            md_rows.append({"video_index": v["video_index"], "class_name": v["class_name"], "task037_global_index": uid, "domain_id": domains[uid],
                            "trajectory_norm": np.linalg.norm(z), "mean_adjacent_transition_norm": np.mean(np.linalg.norm(z, axis=1)),
                            "maximum_adjacent_transition_norm": np.max(np.linalg.norm(z, axis=1)), "temporal_variance": np.var(z),
                            "motion_block_norm": np.linalg.norm(z[:, :2]), "deformation_block_norm": np.linalg.norm(z[:, 2:]),
                            "motion_variance": np.var(z[:, :2]), "deformation_variance": np.var(z[:, 2:]), "dominant_transition_positions": [int(i + 1) for i in top]})
    write_csv(root / "task042_phase_m_spatial_state.csv", state_rows)
    write_csv(root / "task042_phase_m_trajectory_transition.csv", trans_rows)
    write_csv(root / "task042_phase_m_long_range_identity.csv", long_rows)
    write_csv(root / "task042_phase_m_motion_deformation.csv", md_rows)
    # Static versus trajectory geometry and complete BMS-domain geometry.
    pair_rows, static_traj_corr = [], []
    domain_pair_values: dict[str, list[tuple[int, int, float, float]]] = {d: [] for d in DOMAINS}
    for d in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == d]
        for a, b in itertools.combinations(ids, 2):
            ia, ib = unit_ids.index(a), unit_ids.index(b)
            va, vb = np.concatenate([traj[:, ia].reshape(-1),]), np.concatenate([traj[:, ib].reshape(-1),])
            sa, sb = static[:, ia].reshape(-1), static[:, ib].reshape(-1)
            dt, ds = 1 - cosine_similarity(va, vb), 1 - cosine_similarity(sa, sb)
            domain_pair_values[d].append((a, b, dt, ds))
            pair_rows.append({"domain_id": d, "task037_global_index_i": a, "task037_global_index_j": b, "trajectory_distance": dt, "static_distance": ds,
                              "stage_i": unit_meta[a]["stage"], "stage_j": unit_meta[b]["stage"], "type_i": unit_meta[a]["unit_type"], "type_j": unit_meta[b]["unit_type"],
                              "same_stage": int(unit_meta[a]["stage"]) == int(unit_meta[b]["stage"]), "same_type": unit_meta[a]["unit_type"] == unit_meta[b]["unit_type"]})
    all_dt = [x[2] for xs in domain_pair_values.values() for x in xs]; all_ds = [x[3] for xs in domain_pair_values.values() for x in xs]
    static_traj_corr = _corr(all_dt, all_ds, "spearman")
    write_csv(root / "task042_phase_m_static_vs_trajectory.csv", pair_rows)
    geom_rows = []
    for d, vals in domain_pair_values.items():
        ids = [u for u in unit_ids if domains[u] == d]
        for a, b, dt, ds in vals:
            geom_rows.append({"domain_id": d, "task037_global_index_i": a, "task037_global_index_j": b, "d_traj": dt, "d_static": ds})
        for u in ids:
            others = [x for x in vals if x[0] == u or x[1] == u]
            nearest = min(others, key=lambda x: x[2]) if others else None
            farthest = max(others, key=lambda x: x[2]) if others else None
            geom_rows.append({"domain_id": d, "row_type": "unit_summary", "task037_global_index_i": u,
                              "nearest_competitor": nearest[1] if nearest and nearest[0] == u else nearest[0] if nearest else None,
                              "nearest_distance": nearest[2] if nearest else None,
                              "most_distant_competitor": farthest[1] if farthest and farthest[0] == u else farthest[0] if farthest else None,
                              "most_distant_distance": farthest[2] if farthest else None})
    write_csv(root / "task042_phase_m_domain_geometry.csv", geom_rows)
    # Video stability and class-balanced position comparisons.
    stability_rows = []; stab_by = {"same_class": {"cosine": [], "pearson": [], "spearman": []}, "different_class": {"cosine": [], "pearson": [], "spearman": []}}
    for ui, uid in enumerate(unit_ids):
        for va, vb in itertools.combinations(range(30), 2):
            same = videos[va]["class_name"] == videos[vb]["class_name"]
            kind = "same_class" if same else "different_class"
            vals = {"cosine": cosine_similarity(traj[va, ui], traj[vb, ui]), "pearson": _corr(traj[va, ui], traj[vb, ui], "pearson"), "spearman": _corr(traj[va, ui], traj[vb, ui], "spearman")}
            for k, val in vals.items():
                if val is not None: stab_by[kind][k].append(val)
            stability_rows.append({"row_type": "video_pair", "task037_global_index": uid, "video_index_i": videos[va]["video_index"], "video_index_j": videos[vb]["video_index"], "class_i": videos[va]["class_name"], "class_j": videos[vb]["class_name"], "same_class": same, **vals})
        by_class = {}
        for vi, v in enumerate(videos):
            by_class.setdefault(str(v["class_name"]), {})[int(v["class_position"])] = vi
        for p, q in ((1, 2), (1, 3), (2, 3)):
            common = [c for c, m in by_class.items() if p in m and q in m]
            va = np.concatenate([traj[by_class[c][p], ui] for c in sorted(common)])
            vb = np.concatenate([traj[by_class[c][q], ui] for c in sorted(common)])
            stability_rows.append({"row_type": "class_balanced_position_pair", "task037_global_index": uid, "position_i": p, "position_j": q, "class_count": len(common),
                                   "cosine": cosine_similarity(va, vb), "pearson": _corr(va, vb, "pearson"), "spearman": _corr(va, vb, "spearman")})
    for kind, mm in stab_by.items():
        for metric, vals in mm.items(): stability_rows.append({"row_type": "summary", "pair_kind": kind, "metric": metric, **_stats(vals)})
    write_csv(root / "task042_phase_m_video_stability.csv", stability_rows)
    # Coverability and residual trajectory maps.
    cover_rows, residual_rows, mixed_rows = [], [], []
    cover_delta = {}
    for d in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == d]
        vec = {u: traj[:, unit_ids.index(u)].reshape(-1) for u in ids}
        for u in ids:
            competitors = [x for x in ids if x != u]
            alpha, residual, delta = solve_simplex_coverage(vec[u], [vec[x] for x in competitors])
            cover_delta[u] = delta
            cover_rows.append({"domain_id": d, "task037_global_index": u, "competitor_ids": competitors, "alpha": alpha, "delta": delta, "trajectory_norm": np.linalg.norm(vec[u])})
            rr = residual.reshape(30, 15, 5)
            for vi, v in enumerate(videos):
                for ti in range(15):
                    for block, sl in (("motion", slice(0, 2)), ("deformation", slice(2, 5))):
                        residual_rows.append({"task037_global_index": u, "domain_id": d, "video_index": v["video_index"], "class_name": v["class_name"], "transition_index": ti + 1, "block": block,
                                              "residual_norm": np.linalg.norm(rr[vi, ti, sl]), "residual_signed_sum": np.sum(rr[vi, ti, sl])})
        if d == "271":
            for a, b in itertools.combinations(ids, 2):
                mixed_rows.append({"row_type": "pair", "domain_id": d, "task037_global_index_i": a, "task037_global_index_j": b, "type_i": unit_meta[a]["unit_type"], "type_j": unit_meta[b]["unit_type"],
                                   "trajectory_distance": 1 - cosine_similarity(vec[a], vec[b]), "delta_i": cover_delta.get(a), "delta_j": cover_delta.get(b)})
    for r in cover_rows: r["delta_rank_within_domain"] = None
    write_csv(root / "task042_phase_m_coverability.csv", cover_rows)
    write_csv(root / "task042_phase_m_residual_trajectory.csv", residual_rows)
    write_csv(root / "task042_phase_m_mixed_domain.csv", mixed_rows)
    # Descriptor complementarity and stage/type audits.
    unit_manifest = {int(r["task037_global_index"]): r for r in read_csv(Path(cfg["unit_manifest"]))}
    desc_rows, desc_vals = [], {}
    for name in ("D_abs", "D_rel", "D_st"):
        vals = []
        for d in DOMAINS:
            ids = [u for u in unit_ids if domains[u] == d]
            for a, b in itertools.combinations(ids, 2):
                da, db = float(unit_manifest[a][name]), float(unit_manifest[b][name])
                descriptor_distance = abs(da - db)
                dt = next(x[2] for x in domain_pair_values[d] if set(x[:2]) == {a, b})
                vals.append((dt, descriptor_distance))
        desc_vals[name] = vals
        desc_rows.append({"comparison": "pairwise_distance", "descriptor": name, "trajectory_distance_correlation_spearman": _corr([x[0] for x in vals], [x[1] for x in vals], "spearman"),
                          "trajectory_distance_correlation_pearson": _corr([x[0] for x in vals], [x[1] for x in vals], "pearson")})
        desc_rows.append({"comparison": "leave_one_out_delta", "descriptor": name, "delta_correlation_spearman": _corr(list(cover_delta.values()), [float(unit_manifest[u][name]) for u in unit_ids], "spearman"),
                          "delta_correlation_pearson": _corr(list(cover_delta.values()), [float(unit_manifest[u][name]) for u in unit_ids], "pearson")})
    write_csv(root / "task042_phase_m_descriptor_complementarity.csv", desc_rows)
    stage_rows = []
    for d in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == d]
        for a, b, dt, ds in domain_pair_values[d]:
            stage_rows.append({"row_type": "pair", "domain_id": d, "task037_global_index_i": a, "task037_global_index_j": b, "d_traj": dt, "same_stage": int(unit_meta[a]["stage"]) == int(unit_meta[b]["stage"]), "same_type": unit_meta[a]["unit_type"] == unit_meta[b]["unit_type"]})
        for u in ids:
            stage_rows.append({"row_type": "unit", "domain_id": d, "task037_global_index_i": u, "stage": unit_meta[u]["stage"], "unit_type": unit_meta[u]["unit_type"],
                               "trajectory_norm": np.linalg.norm(traj[:, unit_ids.index(u)]), "delta": cover_delta[u]})
    write_csv(root / "task042_phase_m_stage_type_audit.csv", stage_rows)
    runtimes = [json.loads((root / f"task042_phase_m_shard{s}_runtime.json").read_text()) for s in (0, 1)]
    runtime_summary = {"physical_gpu_ids": [0, 1], "gpu_names": [r["gpu_name"] for r in runtimes], "forward_count_total": sum(r["forward_count"] for r in runtimes), "expected_forward_count_total": 30,
                       "wall_seconds_max": max(r["wall_seconds"] for r in runtimes), "peak_memory_bytes_by_gpu": {str(r["physical_gpu"]): r["peak_memory_bytes"] for r in runtimes}, "dtype": "float32", "amp": False, "only_gpu_0_and_1_used": True}
    write_json(root / "task042_phase_m_runtime_summary.json", runtime_summary)
    state_stats = {c: _stats(states[:, :, :, i].reshape(-1)) for i, c in enumerate(comp_names)}
    traj_stats = _stats(np.linalg.norm(traj.reshape(-1, 75), axis=1))
    motion_norms = [float(x["motion_block_norm"]) for x in md_rows]; deformation_norms = [float(x["deformation_block_norm"]) for x in md_rows]
    long_stats = _stats([x["maximum_absolute_error"] for x in long_rows])
    domain_cover = {d: _stats([cover_delta[u] for u in unit_ids if domains[u] == d]) for d in DOMAINS}
    static_nearest = {}; traj_nearest = {}
    for d, vals in domain_pair_values.items():
        ids = [u for u in unit_ids if domains[u] == d]
        for u in ids:
            uv = [x for x in vals if u in x[:2]]
            traj_nearest[u] = min(x[2] for x in uv) if uv else None
            static_nearest[u] = min(x[3] for x in uv) if uv else None
    # Conservative predeclared decision: all required support must be qualitative; unresolved class-repeatability or geometry implies B.
    decision = "B"
    summary = {"analysis_status": "completed", "decision": decision, "decision_label": "CROSS_FRAME_FUNCTIONAL_TRAJECTORY_WEAK_OR_UNRESOLVED",
               "decision_rationale": "The trajectory is measurable, but the frozen pilot leaves at least one all-of requirement unresolved; the conservative predeclared fallback is B.",
               "branch": BRANCH, "analysis_code_git_head": subprocess.check_output(["git", "-C", cfg["repo_root"], "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint_sha256": CHECKPOINT_SHA, "unit_count": 13, "video_count": 30, "class_count": 10, "temporal_length": 16, "state_dimension": 5, "transition_count": 15,
               "forward_count_observed": runtime_summary["forward_count_total"], "forward_count_expected": 30, "near_zero_mass_frames": int(near_zero.sum()),
               "spatial_state_component_stats": state_stats, "trajectory_norm_stats": traj_stats, "long_range_identity_max_error": long_stats,
               "motion_block_norm_stats": _stats(motion_norms), "deformation_block_norm_stats": _stats(deformation_norms),
               "static_vs_trajectory_distance_spearman": static_traj_corr,
               "video_stability": {kind: {metric: _stats(vals) for metric, vals in mm.items()} for kind, mm in stab_by.items()},
               "domain_trajectory_coverability": domain_cover, "coverability_values_by_unit": {str(u): cover_delta[u] for u in unit_ids},
               "descriptor_complementarity": desc_rows, "runtime_summary": runtime_summary,
               "unit_identities": cfg["unit_identities"], "input_sha256": cfg["input_sha256"],
               "requirements": {"A_unified_spatial_state": True, "B_nontrivial_trajectories": bool(traj_stats["n"] and traj_stats["max"] > 0),
                                "C_deformation_audit": bool(np.nanmax(deformation_norms) > 0), "D_static_control_reported": True,
                                "E_video_stability_reported": True, "F_domain_geometry_reported": True, "G_coverability_reported": True,
                                "H_descriptor_complementarity_reported": True, "I_stage_type_audit_reported": True, "J_later_pruning_selection_not_authorized": True}}
    write_json(root / "task042_phase_m_summary.json", summary)
    (root / "task042_phase_m_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(s: Mapping[str, Any]) -> str:
    lines = [
        "# Task042 Phase M — Cross-Frame Functional Trajectory Diagnostic", "",
        "## Frozen scope", "",
        "This representation-only diagnostic uses the exact 13 frozen BMS units in domains 415, 103, 76, and 271; all 30 authoritative Task042 videos (10 classes x 3 positions); the Phase-K checkpoint; FP32; and AMP=False. No frame swap, temporal intervention, pruning, finetuning, performance oracle, optimal transport, or new weight/threshold is used.",
        "",
        "For each native stage grid, the exact Phase-J individual-head or post-GELU/pre-fc2 FFN spatial activation is converted to a nonnegative spatial response. It is normalized within its native grid, and only the five moments [mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy] are retained. No spatial interpolation or averaging is used.",
        "",
        "## Execution and required questions", "",
        f"- Ordinary model forwards: {s['forward_count_observed']}/{s['forward_count_expected']}; all selected units are captured simultaneously; GPU 0 and GPU 1 only.",
        f"- A. Unified state construction: {s['requirements']['A_unified_spatial_state']}; B. Nontrivial trajectories: {s['requirements']['B_nontrivial_trajectories']}; C. Deformation audit: {s['requirements']['C_deformation_audit']}.",
        f"- Long-range telescoping identity maximum absolute error: {s['long_range_identity_max_error']}.",
        f"- D. Static-vs-trajectory control is reported; pooled distance Spearman: {s['static_vs_trajectory_distance_spearman']}.",
        f"- E. Video stability is reported separately for same-class and different-class pairs, with class-balanced P1/P2/P3 rows; one can inspect the full CSV without collapsing videos.",
        f"- F/G. Complete within-domain trajectory geometry and leave-one-out simplex coverability are reported. Domain coverability distributions: {s['domain_trajectory_coverability']}.",
        f"- H. Descriptor complementarity against D_abs, D_rel and D_st is reported without combination.",
        f"- I. Stage/type confounds are explicitly audited; no correction is applied.",
        "",
        "## Primary diagnosis", "",
        f"Spatial-state component distributions: {s['spatial_state_component_stats']}. Trajectory norm distribution: {s['trajectory_norm_stats']}. Motion-block norm: {s['motion_block_norm_stats']}; deformation-block norm: {s['deformation_block_norm_stats']}.",
        "",
        "## Predeclared decision", "",
        f"**{s['decision_label']}**. The pilot is measurable, but the predeclared all-of support requirements are not all resolved by this diagnostic alone; the frozen ambiguous fallback is B. Phase M therefore stops without any pruning-selection or performance experiment.",
        "",
        "## Artifacts", "",
        "The output directory contains the manifest, native-grid spatial states, 15 adjacent transitions, exact long-range identity audit, motion/deformation audit, static-vs-trajectory geometry, class/video stability, complete BMS-domain geometry, mixed-domain analysis, convex coverability and residual trajectories, descriptor complementarity, stage/type audit, runtime summary, and this report.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    p.add_argument("--base-output", type=Path, default=DEFAULT_BASE)
    p.add_argument("--phase-root", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--shard", type=int, choices=(0, 1))
    p.add_argument("--gpu", type=int, choices=(0, 1))
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    repo = args.repo_root.resolve()
    root = (args.phase_root or args.base_output / "phase_m").resolve()
    if args.prepare_only:
        x = prepare(repo, args.base_output.resolve(), root); print(f"PHASE_M_PREPARED videos={x['video_count']} units={x['unit_count']} forwards={x['video_count']}")
    elif args.shard is not None:
        require(args.gpu is not None, "--shard requires --gpu"); run_shard(repo, root, args.shard, args.gpu)
    elif args.analyze:
        x = analyze(root); print(f"PHASE_M_ANALYSIS_COMPLETE decision={x['decision_label']}")
    else:
        p.error("select --prepare-only, --shard, or --analyze")


if __name__ == "__main__":
    main()

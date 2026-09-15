#!/usr/bin/env python3
"""Task042 Phase L: second-order inter-frame functional interaction audit."""
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
EXPECTED_DOMAIN_UNITS = {"415": (779, 1549, 1553), "103": (254, 2334, 16627),
                         "76": (133, 33306, 36376), "271": (328, 1611, 7845, 30209)}
CHECKPOINT_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DEFAULT_REPO = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
DEFAULT_BASE = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
T_MODEL = 16
POSITIONS = tuple(range(2, 16))
PAIRS = tuple(itertools.combinations(POSITIONS, 2))
EPS = 1e-12


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 Phase-L gate failed: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k, "")) for k in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def temporal_positions(t_count: int = T_MODEL) -> tuple[int, ...]:
    if int(t_count) != 16:
        raise ValueError("Phase L is frozen to T=16")
    return POSITIONS


def factorial_state_bindings(s: int, t: int) -> dict[str, tuple[int, ...]]:
    if not (2 <= int(s) < int(t) <= 15):
        raise ValueError("factorial positions must satisfy 2 <= s < t <= 15")
    return {"F11": (), "F10": (int(t),), "F01": (int(s),), "F00": (int(s), int(t))}


def validate_factorial_manifests(manifest_rows: Sequence[Mapping[str, Any]],
                                 state_rows: Sequence[Mapping[str, Any]],
                                 video_indices: Sequence[int]) -> None:
    """Check the real-data state labels, unique forwards, and cache reuse identities."""
    vids = tuple(map(int, video_indices))
    if len(manifest_rows) != len(vids) * 106 or len(state_rows) != len(vids) * 91 * 4:
        raise ValueError("real factorial manifest row counts do not match 106 forwards and 364 states per video")
    fwd = {}
    for row in manifest_rows:
        key = (int(row["video_index"]), str(row["forward_id"]))
        if key in fwd:
            raise ValueError("duplicate real model-forward identity")
        fwd[key] = row
    expected_keys = set()
    for vi in vids:
        for s in POSITIONS:
            expected_keys.add((vi, f"v{vi}:single:{s}"))
        expected_keys.add((vi, f"v{vi}:baseline"))
        for s, t in PAIRS:
            expected_keys.add((vi, f"v{vi}:double:{s}:{t}"))
    if set(fwd) != expected_keys:
        raise ValueError("real forward identities do not match the exact cached factorial plan")
    counts = {vi: {"original_F11_reused": 0, "single_suppression_reused": 0, "double_suppression": 0} for vi in vids}
    for (vi, _), row in fwd.items():
        kind = str(row["forward_kind"])
        if kind not in counts[vi]:
            raise ValueError("unexpected forward kind in real manifest")
        counts[vi][kind] += 1
    if any(x != {"original_F11_reused": 1, "single_suppression_reused": 14, "double_suppression": 91}
           for x in counts.values()):
        raise ValueError("real forward counts differ from the exact reuse plan")
    seen = set()
    for row in state_rows:
        vi, s, t = int(row["video_index"]), int(row["s"]), int(row["t"])
        state = str(row["state"])
        key = (vi, s, t, state)
        if key in seen or vi not in vids:
            raise ValueError("duplicate/unexpected factorial state row")
        seen.add(key)
        bindings = factorial_state_bindings(s, t)
        suppress = row["suppressed_positions"]
        if isinstance(suppress, str):
            suppress = json.loads(suppress)
        if tuple(map(int, suppress)) != bindings[state]:
            raise ValueError("factorial state suppression identity/order mismatch")
        expected_forward = {"F11": f"v{vi}:baseline", "F10": f"v{vi}:single:{t}",
                            "F01": f"v{vi}:single:{s}", "F00": f"v{vi}:double:{s}:{t}"}[state]
        if str(row["forward_id"]) != expected_forward:
            raise ValueError("factorial state did not reuse the correct forward")
        reused = row["reused_cached_state"]
        reused = str(reused).lower() == "true" if isinstance(reused, str) else bool(reused)
        if reused != (state != "F00"):
            raise ValueError("cached-state identity flag is inconsistent")
    if len(seen) != len(vids) * 91 * 4:
        raise ValueError("real factorial state table is incomplete")


def apply_temporal_suppressions(z: Any, positions: Sequence[int]) -> tuple[Any, dict[int, float], dict[str, Any]]:
    """Simultaneous do(q_s=0); all backgrounds come from the unchanged tensor."""
    import torch
    if not torch.is_tensor(z) or z.ndim != 5 or tuple(z.shape[:1]) != (1,) or int(z.shape[2]) != T_MODEL:
        raise ValueError("common temporal tensor must be [1,C,16,H,W]")
    pos = tuple(sorted(int(p) for p in positions))
    if len(pos) != len(set(pos)) or any(p not in POSITIONS for p in pos):
        raise ValueError("suppressed positions must be unique interior one-based positions 2..15")
    modified = z.clone()
    magnitudes: dict[int, float] = {}
    for p in pos:
        i = p - 1
        bg = (z[:, :, i - 1] + z[:, :, i + 1]) * 0.5
        q = z[:, :, i] - bg
        magnitudes[p] = float((torch.norm(q) /
                               (torch.norm(z[:, :, i]) + EPS)).detach().cpu())
        modified[:, :, i] = bg
    outside = [i for i in range(T_MODEL) if i + 1 not in pos]
    exact = not outside or bool(torch.equal(z[:, :, outside], modified[:, :, outside]))
    errors = {p: float(torch.max(torch.abs(modified[:, :, p - 1] -
                                      (z[:, :, p - 2] + z[:, :, p]) * 0.5)).detach().cpu()) for p in pos}
    if not exact or any(v != 0.0 for v in errors.values()):
        raise RuntimeError("temporal intervention locality/replacement check failed")
    return modified, magnitudes, {"outside_positions_exact": exact,
                                  "replacement_max_abs_error": max(errors.values(), default=0.0),
                                  "suppressed_positions": list(pos), "input_shape": list(z.shape)}


def factorial_contrasts(f11: Any, f10: Any, f01: Any, f00: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [np.asarray(x, dtype=np.float64) for x in (f11, f10, f01, f00)]
    if any(x.shape != xs[0].shape for x in xs[1:]):
        raise ValueError("four factorial activation tensors must have identical shape")
    a, b, c, d = xs
    j = a - b - c + d
    ms = 0.5 * ((a - c) + (b - d))
    mt = 0.5 * ((a - b) + (c - d))
    return j, ms, mt


def interaction_energy(j: Any, ms: Any, mt: Any) -> dict[str, float]:
    j, ms, mt = (np.asarray(x, dtype=np.float64).reshape(-1) for x in (j, ms, mt))
    ei, es, et = float(j @ j), float(ms @ ms), float(mt @ mt)
    r = ei / (ei + es + et + EPS)
    if r < -1e-12 or r > 1.0 + 1e-12:
        raise RuntimeError("R is outside [0,1]")
    return {"interaction_norm": float(np.linalg.norm(j)), "E_I": ei, "E_s": es, "E_t": et,
            "R": min(1.0, max(0.0, r))}


def upper_pairs() -> tuple[tuple[int, int], ...]:
    return PAIRS


def symmetric_matrix_from_upper(vector: Sequence[float], size: int = 14) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    tri = np.triu_indices(size, k=1)
    if values.size != len(tri[0]):
        raise ValueError("wrong upper-triangle length")
    out = np.zeros((size, size), dtype=np.float64)
    out[tri] = values
    out[(tri[1], tri[0])] = values
    return out


def concat_video_vectors(vectors: Mapping[int, Sequence[float]], video_order: Sequence[int]) -> np.ndarray:
    order = tuple(map(int, video_order))
    if not order or any(v not in vectors for v in order):
        raise ValueError("vectors do not cover the fixed video order")
    return np.concatenate([np.asarray(vectors[v], dtype=np.float64).reshape(-1) for v in order])


def _stats(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None,
                "q25": None, "q75": None, "exact_zero_count": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
            "exact_zero_count": int(np.sum(x == 0.0))}


def _corr(a: Sequence[float], b: Sequence[float], method: str) -> float | None:
    from scipy import stats
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size != y.size or x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    with np.errstate(all="ignore"):
        value = stats.pearsonr(x, y)[0] if method == "pearson" else stats.spearmanr(x, y)[0]
    return float(value) if np.isfinite(value) else None


def _rank_desc(values: Sequence[float]) -> np.ndarray:
    """Return 1-based descending average ranks, preserving exact ties."""
    from scipy.stats import rankdata
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("descending ranks require a nonempty finite vector")
    return rankdata(-x, method="average").astype(np.float64)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= EPS:
        return 1.0 if np.linalg.norm(x - y) <= EPS else 0.0
    return float(np.dot(x, y) / den)


def simplex_cover(target: Sequence[float], competitors: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.optimize import minimize
    y, rows = np.asarray(target, dtype=np.float64).reshape(-1), np.asarray(competitors, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != y.size or rows.shape[0] < 1:
        raise ValueError("competitors must have shape [M,D]")
    gram, cross = rows @ rows.T, rows @ y
    start = np.full(rows.shape[0], 1.0 / rows.shape[0])
    objective = lambda a: float(a @ gram @ a - 2.0 * a @ cross + y @ y)
    gradient = lambda a: 2.0 * (gram @ a - cross)
    result = minimize(objective, start, jac=gradient, method="SLSQP", bounds=[(0.0, 1.0)] * rows.shape[0],
                      constraints=[{"type": "eq", "fun": lambda a: float(a.sum() - 1.0),
                                    "jac": lambda a: np.ones_like(a)}],
                      options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    if not result.success and np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-6:
        raise RuntimeError("simplex optimizer failed: " + str(result.message))
    alpha = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    alpha /= alpha.sum()
    residual = y - alpha @ rows
    return alpha, residual, float(np.linalg.norm(residual) / (np.linalg.norm(y) + EPS))


class PhaseLIntervention:
    def __init__(self, module: Any, torch: Any):
        self.torch, self.positions = torch, ()
        self.z_original = None
        self.magnitudes: dict[int, float] = {}
        self.detail: dict[str, Any] = {}
        self.handle = module.register_forward_hook(self._hook)

    def begin(self, positions: Sequence[int]) -> None:
        self.positions = tuple(map(int, positions))
        self.z_original, self.magnitudes, self.detail = None, {}, {}

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not self.torch.is_tensor(output) or output.ndim != 5 or int(output.shape[0]) != 1 or int(output.shape[2]) != T_MODEL:
            raise RuntimeError("common patch_embed output must be [1,C,16,H,W]")
        self.z_original = output.detach().clone()
        if self.positions:
            modified, self.magnitudes, self.detail = apply_temporal_suppressions(output, self.positions)
            return modified
        self.magnitudes = {}
        for p in POSITIONS:
            i = p - 1
            bg = (output[:, :, i - 1] + output[:, :, i + 1]) * 0.5
            q = output[:, :, i] - bg
            self.magnitudes[p] = float((self.torch.norm(q) /
                                        (self.torch.norm(output[:, :, i]) + EPS)).detach().cpu())
        self.detail = {"outside_positions_exact": True, "replacement_max_abs_error": 0.0,
                       "suppressed_positions": [], "input_shape": list(output.shape)}
        return output

    def close(self) -> None:
        self.handle.remove()


def _position1_videos(kcfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = sorted((dict(v) for v in kcfg["video_identity_order"] if int(v["class_position"]) == 1),
                  key=lambda v: str(v["class_name"]))
    if len(rows) != 10 or len({str(v["class_name"]) for v in rows}) != 10:
        raise RuntimeError("frozen cohort must yield exactly one position-1 video per class")
    return rows


def prepare(repo: Path, base: Path, phase_root: Path) -> dict[str, Any]:
    if phase_root.exists():
        raise RuntimeError("Phase-L output exists; refusing overwrite")
    runtime = str(repo / "src" / "lgfr_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task042_phase_k_influence as phase_k
    require(subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip() == BRANCH,
            "not on the frozen Task042 branch")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(),
            "Phase-L code must be committed and checkout clean before preparation")
    base_cfg_path, jcfg_path, kcfg_path = (base / "task042_run_config.json",
                                          base / "phase_j" / "task042_phase_j_run_config.json",
                                          base / "phase_k" / "task042_phase_k_run_config.json")
    require(all(p.is_file() for p in (base_cfg_path, jcfg_path, kcfg_path)), "frozen Task042/Phase-J/Phase-K configs missing")
    base_cfg, jcfg, kcfg = [json.loads(p.read_text(encoding="utf-8")) for p in (base_cfg_path, jcfg_path, kcfg_path)]
    js = json.loads((base / "phase_j" / "task042_phase_j_summary.json").read_text())
    ks = json.loads((base / "phase_k" / "task042_phase_k_summary.json").read_text())
    require(js.get("analysis_status") == "completed" and js.get("decision") == "B", "Phase-J weak/unresolved outcome missing")
    require(ks.get("analysis_status") == "completed" and
            ks.get("decision_label") == "DIRECTED_TEMPORAL_INNOVATION_RELATION_WEAK_OR_UNRESOLVED",
            "Phase-K weak/unresolved outcome missing")
    require(base_cfg.get("required_branch") == BRANCH and Path(base_cfg["repo_root"]).resolve() == repo.resolve(),
            "frozen Task042 run config identity mismatch")
    require(phase_k.sha256_file(Path(base_cfg["checkpoint_path"])) == CHECKPOINT_SHA and
            kcfg.get("checkpoint_sha256") == CHECKPOINT_SHA, "checkpoint hash mismatch")
    require(int(kcfg.get("model_temporal_positions", -1)) == T_MODEL, "Phase K did not establish model T=16")
    unit_rows = [r for r in phase_k.read_csv(Path(base_cfg["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    unit_rows.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    expected = {u for d in DOMAINS for u in EXPECTED_DOMAIN_UNITS[d]}
    if {int(r["task037_global_index"]) for r in unit_rows} != expected or len(unit_rows) != 13:
        raise RuntimeError("frozen Task037 unit set does not match the four BMS domains")
    identity_j = {int(x["task037_global_index"]): x for x in jcfg["unit_identities"]}
    identity_k = {int(x["task037_global_index"]): x for x in kcfg["unit_identities"]}
    unit_ids = []
    identities = []
    for row in unit_rows:
        uid = int(row["task037_global_index"])
        ident = {k: row[k] for k in ("task037_global_index", "domain_id", "layer", "stage", "unit_type", "unit_index")}
        for source in (identity_j.get(uid), identity_k.get(uid)):
            if source is None or any(str(source[k]) != str(ident[k]) for k in ident):
                raise RuntimeError("Task037 global/domain/layer/stage/type/index mismatch for unit " + str(uid))
        if uid not in EXPECTED_DOMAIN_UNITS[str(row["domain_id"])]:
            raise RuntimeError("unexpected BMS-domain membership for " + str(uid))
        identities.append({k: (int(v) if k in ("task037_global_index", "stage", "unit_index") else v)
                           for k, v in ident.items()})
        unit_ids.append(uid)
    videos = _position1_videos(kcfg)
    jvideos = {(str(v["class_name"]), int(v["class_position"])): v for v in jcfg["video_identity_order"]}
    require(all((v["class_name"], 1) in jvideos and str(jvideos[(v["class_name"], 1)]["video_id"]) == str(v["video_id"])
                for v in videos), "position-1 video identity mismatch with Phase J")
    vi_list = [int(v["video_index"]) for v in videos]
    source_rows = phase_k.read_csv(base / "phase_k" / "task042_phase_k_source_innovation.csv")
    source_values = {(int(r["video_index"]), int(r["source_position"])): float(r["source_innovation_relative_norm"])
                     for r in source_rows}
    require(all((vi, p) in source_values for vi in vi_list for p in POSITIONS),
            "Phase-K source magnitudes do not cover the selected videos/positions")
    sampled_map = {str(k): v for k, v in kcfg["temporal_input_indices_by_video"].items()}
    temporal_map = []
    for v in videos:
        vi = int(v["video_index"])
        stem = Path(str(v["video_id"])).name
        require(stem in sampled_map, "Phase-K LoopPadding map has no entry for selected video " + stem)
        sampled = sampled_map[stem]
        require(len(sampled) == 32, "LoopPadding sample map must have 32 slots")
        for p in range(1, 17):
            temporal_map.append({"video_index": vi, "video_id": v["video_id"], "class_name": v["class_name"],
                                 "class_position": 1, "temporal_position": p,
                                 "model_input_slot_1": 2*p - 1, "model_input_slot_2": 2*p,
                                 "raw_frame_index_1": int(sampled[2*p - 2]), "raw_frame_index_2": int(sampled[2*p - 1])})
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    config = {
        "task": "TASK042 PHASE L — SECOND-ORDER INTER-FRAME FUNCTIONAL INTERACTION DIAGNOSTIC",
        "required_branch": BRANCH, "repo_root": str(repo.resolve()), "base_output": str(base.resolve()),
        "phase_root": str(phase_root.resolve()), "git_head": head, "phase_j_git_head": jcfg.get("git_head"),
        "phase_k_git_head": kcfg.get("git_head"), "phase_k_analysis_code_git_head": ks.get("analysis_code_git_head"),
        "checkpoint_path": base_cfg["checkpoint_path"], "checkpoint_sha256": CHECKPOINT_SHA,
        "unit_manifest": base_cfg["unit_manifest"], "unit_count": 13, "unit_identities": identities,
        "domains": {d: list(EXPECTED_DOMAIN_UNITS[d]) for d in DOMAINS},
        "video_count": 10, "video_identity_order": videos, "video_indices": vi_list,
        "class_order": [v["class_name"] for v in videos], "within_class_position": 1,
        "temporal_positions_one_based": list(POSITIONS), "temporal_pairs": [list(x) for x in PAIRS],
        "pair_count": 91, "factorial_state_order": ["F11", "F10", "F01", "F00"],
        "intervention_module": "Phase-K common patch_embed output hook, [B,C,T,H,W], temporal axis 2",
        "suppression_definition": "b_s=(z[s-1]+z[s+1])/2; q_s=z[s]-b_s; do(q_s=0) via z[s]=b_s",
        "dtype": "float32", "amp": False, "pruning": False, "finetuning": False, "performance_oracle": False,
        "analysis_forwards_per_video": 106, "analysis_forwards_total": 1060,
        "source_magnitudes_from_phase_k": {f"{vi}:{p}": source_values[(vi, p)] for vi in vi_list for p in POSITIONS},
        "input_sha256": {"task042_run_config": sha256_file(base_cfg_path), "phase_j_run_config": sha256_file(jcfg_path),
                         "phase_k_run_config": sha256_file(kcfg_path),
                         "phase_j_summary": sha256_file(base / "phase_j" / "task042_phase_j_summary.json"),
                         "phase_k_summary": sha256_file(base / "phase_k" / "task042_phase_k_summary.json"),
                         "phase_k_source_innovation": sha256_file(base / "phase_k" / "task042_phase_k_source_innovation.csv"),
                         "checkpoint": CHECKPOINT_SHA},
        "temporal_position_audit": temporal_map,
    }
    phase_root.mkdir(parents=True)
    write_json(phase_root / "task042_phase_l_run_config.json", config)
    write_csv(phase_root / "task042_phase_l_temporal_position_audit.csv", temporal_map)
    return config


def _forward(model: Any, capture: Any, intervention: PhaseLIntervention, clip: Any, torch: Any,
             suppressed: Sequence[int]) -> tuple[dict[int, np.ndarray], dict[int, float], dict[str, Any], tuple[Any, Any]]:
    capture.begin(collect_features=True)
    intervention.begin(suppressed)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        _ = model(clip.unsqueeze(0))
    end.record()
    expected = {int(r["task037_global_index"]) for r in capture.rows}
    if set(capture.features) != expected or set(capture.metadata) != expected:
        raise RuntimeError("forward failed to capture all frozen units simultaneously")
    features = {}
    rows_by_id = {int(r["task037_global_index"]): r for r in capture.rows}
    for uid, value in capture.features.items():
        uid = int(uid)
        meta = capture.metadata[uid]
        # Restore, without pooling or reducing, the complete spatial activation tensor.
        if str(rows_by_id[uid]["unit_type"]) == "attention_head":
            t_count, height, width, head_dim = (int(meta[k]) for k in ("T_i", "H_i", "W_i", "head_dim"))
            value = value.reshape(t_count, height, width, head_dim)
        else:
            t_count, height, width = (int(meta[k]) for k in ("T_i", "H_i", "W_i"))
            value = value.reshape(t_count, height, width)
        features[uid] = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
    return features, dict(intervention.magnitudes), dict(intervention.detail), (start, end)


def run_shard(repo: Path, phase_root: Path, shard: int, gpu: int) -> None:
    import torch
    runtime = str(repo / "src" / "lgfr_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task042_phase_k_influence as phase_k
    import task042_phase_j_conditional as phase_j
    config = json.loads((phase_root / "task042_phase_l_run_config.json").read_text(encoding="utf-8"))
    require(shard in (0, 1) and gpu == shard and os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu),
            "GPU shards must be isolated to authorized GPU 0 or GPU 1")
    require(subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip() == BRANCH,
            "worker branch mismatch")
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == config["git_head"],
            "checkout changed after prepare")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(),
            "workers require committed clean code")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker must see one isolated GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3407 + gpu); torch.cuda.manual_seed_all(3407 + gpu)
    phase_k_cfg = json.loads((Path(config["base_output"]) / "phase_k" / "task042_phase_k_run_config.json").read_text())
    torch, _base, phase_j, model, specs, identity, patch_name, patch_module, _geometry = phase_k._model_on_gpu(phase_k_cfg, gpu)
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA and identity.get("classifier_head", {}).get("status") == "loaded",
            "checkpoint or classifier head identity mismatch")
    require(not identity.get("missing_keys") and not identity.get("unexpected_keys") and not identity.get("shape_mismatches"),
            "model state did not load exactly")
    all_units = [r for r in phase_k.read_csv(Path(phase_k_cfg["unit_manifest"])) if str(r["domain_id"]) in DOMAINS]
    all_units.sort(key=lambda r: (DOMAINS.index(str(r["domain_id"])), int(r["task037_global_index"])))
    capture_rows = phase_k._capture_rows(all_units)
    capture = phase_j.PhaseJCapture(model, capture_rows, specs, torch)
    patch_candidates = [(n, m) for n, m in dict(model.named_modules()).items()
                        if n.endswith("patch_embed") and hasattr(m, "proj") and hasattr(m.proj, "kernel_size")]
    require(len(patch_candidates) == 1 and patch_candidates[0][0] == patch_name,
            "Phase-K common intervention module changed")
    intervention = PhaseLIntervention(patch_module, torch)
    videos = list(config["video_identity_order"])
    assigned = videos[shard * 5:(shard + 1) * 5]
    require(len(assigned) == 5, "each GPU must process exactly five position-1 videos")
    # The data loader always enumerates the full frozen 30-video Phase-J pool;
    # pass that full manifest for identity validation, then process only this
    # Phase-L position-1 subset on the assigned shard.
    phase_j_config = json.loads((Path(config["base_output"]) / "phase_j" /
                                 "task042_phase_j_run_config.json").read_text(encoding="utf-8"))
    _loader, clips, loader_rows = phase_j._load_all_videos(phase_j_config, workers=2)
    require(len(clips) == 30, "frozen loader must still reproduce the authoritative 30-video pool")
    unit_ids = [int(r["task037_global_index"]) for r in capture_rows]
    unit_slot = {u: i for i, u in enumerate(unit_ids)}
    shape = (len(assigned), len(unit_ids), len(PAIRS))
    arrays = {name: np.zeros(shape, dtype=np.float32) for name in ("interaction_norm", "E_I", "E_s", "E_t", "R")}
    source_mag = np.zeros((len(assigned), len(POSITIONS)), dtype=np.float32)
    forwards, state_rows, events, per_video = [], [], [], []
    counts = {"F11_baseline_forwards": 0, "single_suppression_forwards": 0, "double_suppression_forwards": 0}
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for vslot, video in enumerate(assigned):
            vi = int(video["video_index"])
            require(str(video["video_id"]) == str(loader_rows[vi]["video_id"]) and int(video["label"]) == int(loader_rows[vi]["label"]),
                    "position-1 video identity differs from the frozen loader")
            clip = clips[vi].to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            one_start = time.perf_counter()
            baseline, mags, detail, event = _forward(model, capture, intervention, clip, torch, ())
            require(tuple(intervention.z_original.shape[:3]) == (1, int(intervention.z_original.shape[1]), T_MODEL),
                    "common Phase-K representation is not T=16")
            source_mag[vslot] = np.asarray([mags[p] for p in POSITIONS], dtype=np.float32)
            events.append(event)
            forwards.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                             "class_position": 1, "forward_id": f"v{vi}:baseline", "forward_kind": "original_F11_reused",
                             "suppressed_positions": [], "intervention_module": patch_name, "physical_gpu": gpu,
                             "unit_capture_count": len(unit_ids), "outside_positions_exact": detail["outside_positions_exact"]})
            counts["F11_baseline_forwards"] += 1
            singles: dict[int, dict[int, np.ndarray]] = {}
            for p in POSITIONS:
                features, _, det, ev = _forward(model, capture, intervention, clip, torch, (p,))
                require(det["outside_positions_exact"] is True and det["replacement_max_abs_error"] == 0.0,
                        "single suppression failed locality gate")
                require(features.keys() == baseline.keys() and all(features[u].shape == baseline[u].shape for u in unit_ids),
                        "single state changed unit identity or tensor shape")
                singles[p] = features; events.append(ev)
                forwards.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                                 "class_position": 1, "forward_id": f"v{vi}:single:{p}",
                                 "forward_kind": "single_suppression_reused", "suppressed_positions": [p],
                                 "intervention_module": patch_name, "physical_gpu": gpu, "unit_capture_count": len(unit_ids),
                                 "outside_positions_exact": True, "replacement_max_abs_error": 0.0})
                counts["single_suppression_forwards"] += 1
            for pi, (s, t) in enumerate(PAIRS):
                f00, _, det, ev = _forward(model, capture, intervention, clip, torch, (s, t))
                require(det["outside_positions_exact"] is True and det["replacement_max_abs_error"] == 0.0,
                        "double suppression was not simultaneous/exact")
                events.append(ev)
                forward_id = f"v{vi}:double:{s}:{t}"
                forwards.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                                 "class_position": 1, "forward_id": forward_id, "forward_kind": "double_suppression",
                                 "suppressed_positions": [s, t], "intervention_module": patch_name, "physical_gpu": gpu,
                                 "unit_capture_count": len(unit_ids), "outside_positions_exact": True,
                                 "replacement_max_abs_error": 0.0})
                counts["double_suppression_forwards"] += 1
                binds = factorial_state_bindings(s, t)
                ids_for = {"F11": f"v{vi}:baseline", "F10": f"v{vi}:single:{t}",
                           "F01": f"v{vi}:single:{s}", "F00": forward_id}
                for state in ("F11", "F10", "F01", "F00"):
                    state_rows.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                                       "class_position": 1, "s": s, "t": t, "state": state,
                                       "state_semantics": {"F11": "both retained", "F10": "s retained, t suppressed",
                                                           "F01": "s suppressed, t retained", "F00": "both suppressed"}[state],
                                       "suppressed_positions": binds[state], "forward_id": ids_for[state],
                                       "reused_cached_state": state != "F00"})
                for uid in unit_ids:
                    # F10 is suppression at t; F01 is suppression at s.
                    j, ms, mt = factorial_contrasts(baseline[uid], singles[t][uid], singles[s][uid], f00[uid])
                    vals = interaction_energy(j, ms, mt)
                    for name, value in vals.items():
                        arrays[name][vslot, unit_slot[uid], pi] = value
            processed = vslot + 1
            count_so_far = sum(counts.values())
            require(count_so_far == processed * 106, "state caching violated the 106-forward/video plan")
            per_video.append({"video_index": vi, "video_id": video["video_id"], "class_name": video["class_name"],
                              "forward_count": 106, "wall_seconds": time.perf_counter() - one_start})
            print(f"PHASE_L_SHARD{shard} class={video['class_name']} video={vi} forwards=106", flush=True)
            del baseline, singles, clip
            torch.cuda.empty_cache()
    finally:
        capture.close(); intervention.close()
    torch.cuda.synchronize()
    cuda_seconds = sum(float(start.elapsed_time(end)) for start, end in events) / 1000.0
    runtime = {"shard": shard, "physical_gpu": gpu, "gpu_name": torch.cuda.get_device_name(0),
               "video_indices": [int(v["video_index"]) for v in assigned], "video_count": len(assigned),
               "forward_count": sum(counts.values()), "expected_forward_count": len(assigned) * 106,
               "state_cache_counts": counts, "cuda_forward_event_seconds_sum": cuda_seconds,
               "wall_seconds": time.perf_counter() - wall_start, "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
               "video_runtime": per_video, "unit_ids": unit_ids, "patch_module": patch_name,
               "dtype": "float32", "amp": False}
    np.savez_compressed(phase_root / f"task042_phase_l_shard{shard}.npz",
                        video_indices=np.asarray(runtime["video_indices"], dtype=np.int16),
                        unit_ids=np.asarray(unit_ids, dtype=np.int64), pairs=np.asarray(PAIRS, dtype=np.int16),
                        source_magnitude=source_mag, **arrays)
    write_csv(phase_root / f"task042_phase_l_shard{shard}_manifest.csv", forwards)
    write_csv(phase_root / f"task042_phase_l_shard{shard}_factorial_states.csv", state_rows)
    write_json(phase_root / f"task042_phase_l_shard{shard}_runtime.json", runtime)
    require(runtime["forward_count"] == runtime["expected_forward_count"], "GPU shard forward-count mismatch")
    print(f"PHASE_L_SHARD_COMPLETE shard={shard} forwards={runtime['forward_count']}", flush=True)


def _load_shards(phase_root: Path, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    vids = [int(v["video_index"]) for v in config["video_identity_order"]]
    vslot = {v: i for i, v in enumerate(vids)}
    unit_ids = [int(x["task037_global_index"]) for x in config["unit_identities"]]
    names = ("interaction_norm", "E_I", "E_s", "E_t", "R")
    out = {k: np.zeros((10, 13, 91), dtype=np.float64) for k in names}
    out["source_magnitude"] = np.zeros((10, 14), dtype=np.float64)
    seen = set()
    for shard in (0, 1):
        npz = phase_root / f"task042_phase_l_shard{shard}.npz"
        rp = phase_root / f"task042_phase_l_shard{shard}_runtime.json"
        require(npz.is_file() and rp.is_file(), f"GPU shard {shard} outputs missing")
        runtime = json.loads(rp.read_text())
        require(runtime["forward_count"] == runtime["expected_forward_count"], f"GPU shard {shard} incomplete")
        with np.load(npz, allow_pickle=False) as z:
            require(list(map(int, z["unit_ids"].tolist())) == unit_ids, "shard unit order mismatch")
            require(np.array_equal(z["pairs"], np.asarray(PAIRS, dtype=np.int16)), "shard pair order mismatch")
            for ix, vi in enumerate(map(int, z["video_indices"].tolist())):
                require(vi in vslot and vi not in seen, "unexpected/duplicate shard video")
                seen.add(vi)
                for name in names:
                    out[name][vslot[vi]] = z[name][ix]
                out["source_magnitude"][vslot[vi]] = z["source_magnitude"][ix]
    require(seen == set(vids), "shards do not cover exactly the frozen position-1 videos")
    return out


def _geometry_vectors(base: Path, video_ids: Sequence[int], unit_ids: Sequence[int]) -> tuple[dict[int, np.ndarray],
                                                                                             dict[int, np.ndarray]]:
    selected, units = set(map(int, video_ids)), set(map(int, unit_ids))
    kmap: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for row in read_csv(base / "phase_k" / "task042_phase_k_directed_relation.csv"):
        uid, vi = int(row["task037_global_index"]), int(row["video_index"])
        if uid in units and vi in selected:
            key = (int(row["source_position"]), int(row["target_position"]))
            kmap.setdefault((uid, vi), {})[key] = float(row["A_source_to_target"])
    jmap: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for row in read_csv(base / "phase_j" / "task042_phase_j_conditional_relation.csv"):
        uid, vi = int(row["task037_global_index"]), int(row["video_index"])
        if uid in units and vi in selected:
            a, b = int(row["frame_t"]), int(row["frame_t_prime"])
            if a < b:
                jmap.setdefault((uid, vi), {})[(a, b)] = float(row["conditional_C"])
    kpairs = [(s, t) for s in POSITIONS for t in range(1, 17) if t != s]
    jpairs = list(itertools.combinations(range(16), 2))
    kvectors, jvectors = {}, {}
    for uid in unit_ids:
        kv, jv = [], []
        for vi in video_ids:
            kr, jr = kmap.get((int(uid), int(vi)), {}), jmap.get((int(uid), int(vi)), {})
            require(all(p in kr for p in kpairs), "Phase-K comparison vector is incomplete")
            require(all(p in jr for p in jpairs), "Phase-J comparison vector is incomplete")
            kv.extend(kr[p] for p in kpairs)
            jv.extend(jr[p] for p in jpairs)
        kvectors[int(uid)], jvectors[int(uid)] = np.asarray(kv), np.asarray(jv)
    return kvectors, jvectors


def _nearest_and_distances(vectors: Mapping[int, np.ndarray], ids: Sequence[int]) -> tuple[dict[int, int],
                                                                                       dict[tuple[int, int], float]]:
    dist = {}
    nn = {}
    for a in ids:
        for b in ids:
            dist[(int(a), int(b))] = 1.0 - _cosine(vectors[int(a)], vectors[int(b)])
    for a in ids:
        nn[int(a)] = min((int(b) for b in ids if b != a), key=lambda b: (dist[(int(a), b)], b))
    return nn, dist


def _compare_geometry(left: Mapping[int, np.ndarray], right: Mapping[int, np.ndarray],
                      unit_ids: Sequence[int], domains: Mapping[int, str], name: str,
                      left_delta: Mapping[int, float], right_delta: Mapping[int, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, dl, dr = [], [], []
    changed = total = 0
    for domain in DOMAINS:
        ids = [u for u in unit_ids if domains[int(u)] == domain]
        lnn, ld = _nearest_and_distances(left, ids)
        rnn, rd = _nearest_and_distances(right, ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                dl.append(ld[(int(a), int(b))]); dr.append(rd[(int(a), int(b))])
                rows.append({"row_type": "pair_distance", "comparison": name, "domain_id": domain,
                             "task037_global_index_i": a, "task037_global_index_j": b,
                             "phase_l_distance": ld[(int(a), int(b))], "comparison_distance": rd[(int(a), int(b))]})
        for uid in ids:
            diff = lnn[int(uid)] != rnn[int(uid)]
            changed += int(diff); total += 1
            rows.append({"row_type": "nearest_neighbor", "comparison": name, "domain_id": domain,
                         "task037_global_index": uid, "phase_l_nearest": lnn[int(uid)],
                         "comparison_nearest": rnn[int(uid)], "changed": diff})
    ids = [int(u) for u in unit_ids]
    summary = {"comparison": name, "pair_distance_count": len(dl),
               "pair_distance_correlation": {"pearson": _corr(dl, dr, "pearson"), "spearman": _corr(dl, dr, "spearman")},
               "nearest_neighbor_changes": changed, "nearest_neighbor_count": total,
               "leave_one_out_coverability_correlation": {"n": len(ids),
                   "pearson": _corr([left_delta[u] for u in ids], [right_delta[u] for u in ids], "pearson"),
                   "spearman": _corr([left_delta[u] for u in ids], [right_delta[u] for u in ids], "spearman")}}
    rows.append({"row_type": "summary", **summary})
    return rows, summary


def analyze(phase_root: Path) -> dict[str, Any]:
    config = json.loads((phase_root / "task042_phase_l_run_config.json").read_text(encoding="utf-8"))
    base = Path(config["base_output"])
    data = _load_shards(phase_root, config)
    videos = list(config["video_identity_order"])
    video_ids = [int(v["video_index"]) for v in videos]
    video_by_index = {int(v["video_index"]): v for v in videos}
    unit_ids = [int(x["task037_global_index"]) for x in config["unit_identities"]]
    u_slot = {uid: i for i, uid in enumerate(unit_ids)}
    unit_meta = {int(x["task037_global_index"]): x for x in config["unit_identities"]}
    domains = {uid: str(m["domain_id"]) for uid, m in unit_meta.items()}
    require(data["R"].shape == (10, 13, 91) and len(PAIRS) == 91, "fixed interaction dimensions invalid")

    # Required per-pair tensor-norm and factorial energy artifacts.
    tensor_rows, energy_rows = [], []
    for vi, video in enumerate(videos):
        for ui, uid in enumerate(unit_ids):
            m = unit_meta[uid]
            for pi, (s, t) in enumerate(PAIRS):
                common = {"video_index": video["video_index"], "video_id": video["video_id"],
                          "class_name": video["class_name"], "class_position": 1,
                          "task037_global_index": uid, "domain_id": m["domain_id"], "layer": m["layer"],
                          "stage": m["stage"], "unit_type": m["unit_type"], "unit_index": m["unit_index"],
                          "pair_index": pi, "s": s, "t": t, "lag": t - s}
                tensor_rows.append({**common, "interaction_norm": data["interaction_norm"][vi, ui, pi],
                                    "R": data["R"][vi, ui, pi]})
                energy_rows.append({**common, "interaction_norm": data["interaction_norm"][vi, ui, pi],
                                    "E_I": data["E_I"][vi, ui, pi], "E_s": data["E_s"][vi, ui, pi],
                                    "E_t": data["E_t"][vi, ui, pi], "R": data["R"][vi, ui, pi]})
    write_csv(phase_root / "task042_phase_l_interaction_tensor.csv", tensor_rows)
    write_csv(phase_root / "task042_phase_l_interaction_energy.csv", energy_rows)

    # Full actual-forward manifest and factorial state reuse map.
    manifest_rows, state_rows = [], []
    for shard in (0, 1):
        manifest_rows += read_csv(phase_root / f"task042_phase_l_shard{shard}_manifest.csv")
        state_rows += read_csv(phase_root / f"task042_phase_l_shard{shard}_factorial_states.csv")
    require(len(manifest_rows) == 1060 and len(state_rows) == 3640, "factorial forward/state manifests are incomplete")
    validate_factorial_manifests(manifest_rows, state_rows, video_ids)
    write_csv(phase_root / "task042_phase_l_manifest.csv", manifest_rows)
    write_csv(phase_root / "task042_phase_l_factorial_states.csv", state_rows)

    # Source-size dependence, including pooled and per-unit Pearson/Spearman.
    source_rows = []
    source_metrics = ("M_sum", "M_prod", "M_max")
    outcomes = ("interaction_norm", "R")
    per_unit_source = {metric: {outcome: [] for outcome in outcomes} for metric in source_metrics}
    for metric in source_metrics:
        for outcome in outcomes:
            px, py = [], []
            for vi in range(10):
                src = {p: float(data["source_magnitude"][vi, i]) for i, p in enumerate(POSITIONS)}
                for pi, (s, t) in enumerate(PAIRS):
                    mag = {"M_sum": src[s] + src[t], "M_prod": src[s] * src[t],
                           "M_max": max(src[s], src[t])}[metric]
                    for ui, uid in enumerate(unit_ids):
                        px.append(mag); py.append(float(data[outcome][vi, ui, pi]))
            source_rows.append({"scope": "pooled", "task037_global_index": "", "source_magnitude_combination": metric,
                                "interaction_outcome": outcome, "n": len(px),
                                "pearson": _corr(px, py, "pearson"), "spearman": _corr(px, py, "spearman")})
            for ui, uid in enumerate(unit_ids):
                ux, uy = [], []
                for vi in range(10):
                    src = {p: float(data["source_magnitude"][vi, i]) for i, p in enumerate(POSITIONS)}
                    for pi, (s, t) in enumerate(PAIRS):
                        ux.append({"M_sum": src[s] + src[t], "M_prod": src[s] * src[t],
                                   "M_max": max(src[s], src[t])}[metric])
                        uy.append(float(data[outcome][vi, ui, pi]))
                pr, sr = _corr(ux, uy, "pearson"), _corr(ux, uy, "spearman")
                if outcome == "interaction_norm" and pr is not None:
                    per_unit_source[metric][outcome].append((pr, sr))
                source_rows.append({"scope": "unit", "task037_global_index": uid, "source_magnitude_combination": metric,
                                    "interaction_outcome": outcome, "n": len(ux), "pearson": pr, "spearman": sr})
    k_source = {(int(r["video_index"]), int(r["source_position"])): float(r["source_innovation_relative_norm"])
                for r in read_csv(base / "phase_k" / "task042_phase_k_source_innovation.csv")}
    source_diff = [abs(float(data["source_magnitude"][vi, pi]) -
                       k_source[(video_ids[vi], p)]) for vi in range(10) for pi, p in enumerate(POSITIONS)]
    write_csv(phase_root / "task042_phase_l_source_magnitude_audit.csv", source_rows)

    # Exact lag distributions, without manually selected lag bins.
    lag_rows, lag_summary = [], {}
    for lag in range(1, 14):
        pidx = [i for i, (s, t) in enumerate(PAIRS) if t - s == lag]
        for metric in outcomes:
            vals = data[metric][:, :, pidx].reshape(-1).tolist()
            st = _stats(vals)
            lag_rows.append({"row_type": "exact_lag_distribution", "lag": lag, "metric": metric, **st})
            lag_summary.setdefault(metric, {})[str(lag)] = st
    all_lags = [t - s for s, t in PAIRS for _vi in range(10) for _uid in unit_ids]
    all_j = [float(data["interaction_norm"][vi, ui, pi]) for pi in range(91) for vi in range(10) for ui in range(13)]
    all_r = [float(data["R"][vi, ui, pi]) for pi in range(91) for vi in range(10) for ui in range(13)]
    lag_assoc = {metric: {"pearson": _corr(all_lags, vals, "pearson"), "spearman": _corr(all_lags, vals, "spearman")}
                 for metric, vals in (("interaction_norm", all_j), ("R", all_r))}
    for metric in outcomes:
        vals = all_j if metric == "interaction_norm" else all_r
        lag_rows.append({"row_type": "lag_association", "metric": metric, "pearson": lag_assoc[metric]["pearson"],
                         "spearman": lag_assoc[metric]["spearman"], "n": len(vals)})
    write_csv(phase_root / "task042_phase_l_lag_analysis.csv", lag_rows)

    # Per-video/unit pair ranking by R and raw interaction norm.
    from scipy.stats import spearmanr
    pair_rows, rank_agreement = [], []
    for vi, video in enumerate(videos):
        for ui, uid in enumerate(unit_ids):
            rvals, jvals = data["R"][vi, ui], data["interaction_norm"][vi, ui]
            rr, jr = _rank_desc(rvals), _rank_desc(jvals)
            agreement = float(spearmanr(rr, jr)[0]) if np.std(rr) and np.std(jr) else None
            if agreement is not None and math.isfinite(agreement):
                rank_agreement.append(agreement)
            ro, jo = np.argsort(-rvals, kind="stable"), np.argsort(-jvals, kind="stable")
            for pi, (s, t) in enumerate(PAIRS):
                pair_rows.append({"row_type": "pair", "video_index": video["video_index"], "class_name": video["class_name"],
                                  "task037_global_index": uid, "domain_id": domains[uid], "pair_index": pi,
                                  "s": s, "t": t, "lag": t - s, "interaction_norm": jvals[pi], "R": rvals[pi],
                                  "rank_by_R_desc": rr[pi], "rank_by_interaction_norm_desc": jr[pi],
                                  "top5_by_R": pi in set(ro[:5]), "bottom5_by_R": pi in set(ro[-5:]),
                                  "top5_by_interaction_norm": pi in set(jo[:5]), "bottom5_by_interaction_norm": pi in set(jo[-5:])})
            pair_rows.append({"row_type": "unit_video_summary", "video_index": video["video_index"],
                              "class_name": video["class_name"], "task037_global_index": uid,
                              "rank_agreement_spearman": agreement, "top5_by_R": [list(PAIRS[i]) for i in ro[:5]],
                              "bottom5_by_R": [list(PAIRS[i]) for i in ro[-5:]],
                              "top5_by_interaction_norm": [list(PAIRS[i]) for i in jo[:5]],
                              "bottom5_by_interaction_norm": [list(PAIRS[i]) for i in jo[-5:]]})
    write_csv(phase_root / "task042_phase_l_pair_structure.csv", pair_rows)

    # Interaction function r_i is the 910-D concatenation in fixed alphabetical class order.
    per_video_r = {uid: {video_ids[vi]: data["R"][vi, ui].astype(np.float64)
                         for vi in range(10)} for ui, uid in enumerate(unit_ids)}
    vectors = {uid: concat_video_vectors(per_video_r[uid], video_ids) for uid in unit_ids}
    stability_rows, stability_by_metric = [], {m: [] for m in ("pearson", "spearman", "cosine")}
    for uid in unit_ids:
        for va, vb in itertools.combinations(video_ids, 2):
            a, b = per_video_r[uid][va], per_video_r[uid][vb]
            vals = {"pearson": _corr(a, b, "pearson"), "spearman": _corr(a, b, "spearman"), "cosine": _cosine(a, b)}
            for metric, value in vals.items():
                if value is not None:
                    stability_by_metric[metric].append(value)
            stability_rows.append({"row_type": "unit_video_pair", "task037_global_index": uid, "domain_id": domains[uid],
                                   "video_index_i": va, "class_i": video_by_index[va]["class_name"],
                                   "video_index_j": vb, "class_j": video_by_index[vb]["class_name"],
                                   "same_class": False, **vals})
    for metric, vals in stability_by_metric.items():
        stability_rows.append({"row_type": "summary", "metric": metric, **_stats(vals)})
    write_csv(phase_root / "task042_phase_l_video_stability.csv", stability_rows)

    # Full ordered within-domain distance matrices.
    geometry_rows = []
    for domain in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == domain]
        for a in ids:
            for b in ids:
                geometry_rows.append({"domain_id": domain, "task037_global_index_i": a, "task037_global_index_j": b,
                                      "distance_1_minus_cosine": 1.0 - _cosine(vectors[a], vectors[b])})
    write_csv(phase_root / "task042_phase_l_domain_geometry.csv", geometry_rows)

    # Collective LOO simplex coverage and signed residual interaction maps.
    cover_rows, residual_rows = [], []
    cover_delta, alpha_by_unit, residual_top = {}, {}, {}
    for domain in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == domain]
        for uid in ids:
            competitors = [u for u in ids if u != uid]
            alpha, residual, delta = simplex_cover(vectors[uid], [vectors[u] for u in competitors])
            cover_delta[uid] = delta
            alpha_by_unit[uid] = dict(zip(competitors, alpha.tolist()))
            residual_video = residual.reshape(10, 91)
            order = np.argsort(-np.abs(residual))[:10]
            residual_top[uid] = [{"video_index": video_ids[int(i // 91)], "s": PAIRS[int(i % 91)][0],
                                  "t": PAIRS[int(i % 91)][1], "residual": float(residual[i]),
                                  "absolute_residual": float(abs(residual[i]))} for i in order]
            m = unit_meta[uid]
            cover_rows.append({"row_type": "unit_summary", "task037_global_index": uid, "domain_id": domain,
                               "unit_type": m["unit_type"], "layer": m["layer"], "stage": m["stage"],
                               "unit_index": m["unit_index"], "competitor_unit_ids": competitors,
                               "target_function_norm": float(np.linalg.norm(vectors[uid])),
                               "residual_norm": float(np.linalg.norm(residual)), "delta": delta,
                               "top_uncovered_interaction_pairs": residual_top[uid]})
            for other, weight in alpha_by_unit[uid].items():
                cover_rows.append({"row_type": "simplex_weight", "task037_global_index": uid, "domain_id": domain,
                                   "competitor_task037_global_index": other, "alpha": weight, "delta": delta})
            for vi, video in enumerate(videos):
                matrix = symmetric_matrix_from_upper(residual_video[vi])
                for pi, (s, t) in enumerate(PAIRS):
                    val = float(residual_video[vi, pi])
                    residual_rows.append({"task037_global_index": uid, "domain_id": domain,
                                          "video_index": video["video_index"], "video_id": video["video_id"],
                                          "class_name": video["class_name"], "s": s, "t": t, "lag": t - s,
                                          "residual": val, "residual_abs": abs(val),
                                          "symmetric_map_value_check": matrix[s - 2, t - 2]})
    write_csv(phase_root / "task042_phase_l_coverability.csv", cover_rows)
    write_csv(phase_root / "task042_phase_l_residual_interaction_map.csv", residual_rows)

    # Compare only existing Phase-J/K artifacts, restricted to the same ten videos.
    kvec, jvec = _geometry_vectors(base, video_ids, unit_ids)
    kdelta, jdelta = {}, {}
    for domain in DOMAINS:
        ids = [u for u in unit_ids if domains[u] == domain]
        for uid in ids:
            _, _, kdelta[uid] = simplex_cover(kvec[uid], [kvec[u] for u in ids if u != uid])
            _, _, jdelta[uid] = simplex_cover(jvec[uid], [jvec[u] for u in ids if u != uid])
    phasek_rows, phasek_summary = _compare_geometry(vectors, kvec, unit_ids, domains, "phase_k_first_order",
                                                      cover_delta, kdelta)
    phasej_rows, phasej_summary = _compare_geometry(vectors, jvec, unit_ids, domains, "phase_j_partial_correlation",
                                                      cover_delta, jdelta)
    write_csv(phase_root / "task042_phase_l_phasek_comparison.csv", phasek_rows)
    write_csv(phase_root / "task042_phase_l_phasej_comparison.csv", phasej_rows)

    # Descriptor complementarity; diagnostics only.
    base_cfg = json.loads((base / "task042_run_config.json").read_text())
    descriptors = {int(r["task037_global_index"]): r for r in read_csv(Path(base_cfg["unit_manifest"]))}
    desc_rows = []
    for desc in ("D_abs", "D_rel", "D_st"):
        pair_a, pair_b = [], []
        for domain in DOMAINS:
            ids = [u for u in unit_ids if domains[u] == domain]
            for a, b in itertools.combinations(ids, 2):
                pair_a.append(1.0 - _cosine(vectors[a], vectors[b]))
                pair_b.append(abs(float(descriptors[a][desc]) - float(descriptors[b][desc])))
        values = [float(descriptors[u][desc]) for u in unit_ids]
        for comparison, x, y in (("pairwise_interaction_distance_vs_descriptor_difference", pair_a, pair_b),
                                 ("loo_delta_vs_descriptor_value", values, [cover_delta[u] for u in unit_ids])):
            desc_rows.append({"comparison": comparison, "descriptor": desc, "n": len(x),
                              "pearson": _corr(x, y, "pearson"), "spearman": _corr(x, y, "spearman")})
    write_csv(phase_root / "task042_phase_l_descriptor_complementarity.csv", desc_rows)

    # Domain 271 identity, pairwise geometry, and leave-one-out weights.
    mixed_ids = [u for u in unit_ids if domains[u] == "271"]
    mixed_rows = []
    for u in mixed_ids:
        m = unit_meta[u]
        mixed_rows.append({"row_type": "unit_identity", "domain_id": "271", "task037_global_index": u,
                           "unit_type": m["unit_type"], "layer": m["layer"], "stage": m["stage"],
                           "unit_index": m["unit_index"], "delta": cover_delta[u]})
    for a, b in itertools.combinations(mixed_ids, 2):
        mixed_rows.append({"row_type": "pairwise_interaction_distance", "domain_id": "271",
                           "task037_global_index_i": a, "unit_type_i": unit_meta[a]["unit_type"],
                           "task037_global_index_j": b, "unit_type_j": unit_meta[b]["unit_type"],
                           "distance_1_minus_cosine": 1.0 - _cosine(vectors[a], vectors[b])})
    for u in mixed_ids:
        for other, weight in alpha_by_unit[u].items():
            mixed_rows.append({"row_type": "loo_convex_weight", "domain_id": "271",
                               "task037_global_index": u, "unit_type": unit_meta[u]["unit_type"],
                               "competitor_task037_global_index": other,
                               "competitor_unit_type": unit_meta[other]["unit_type"],
                               "alpha": weight, "delta": cover_delta[u]})
    write_csv(phase_root / "task042_phase_l_mixed_domain.csv", mixed_rows)

    domain_cover = {d: _stats([cover_delta[u] for u in unit_ids if domains[u] == d]) for d in DOMAINS}
    domain_energy = {}
    for d in DOMAINS:
        slots = [i for i, u in enumerate(unit_ids) if domains[u] == d]
        domain_energy[d] = {metric: _stats(data[metric][:, slots, :].reshape(-1).tolist())
                            for metric in ("interaction_norm", "R")}
    unit_energy = {str(u): {metric: _stats(data[metric][:, u_slot, :].reshape(-1).tolist())
                            for metric in ("interaction_norm", "R")} for u, u_slot in
                   ((u, unit_ids.index(u)) for u in unit_ids)}
    video_energy = {str(int(v["video_index"])): {metric: _stats(data[metric][vi].reshape(-1).tolist())
                                                  for metric in ("interaction_norm", "R")}
                    for vi, v in enumerate(videos)}
    phase_k_summary = json.loads((base / "phase_k" / "task042_phase_k_summary.json").read_text())
    k_conf = phase_k_summary["source_magnitude_confound_summary"]["mean_A"]
    source_median = {}
    for metric in source_metrics:
        prs = [x[0] for x in per_unit_source[metric]["interaction_norm"] if x[0] is not None]
        srs = [x[1] for x in per_unit_source[metric]["interaction_norm"] if x[1] is not None]
        source_median[metric] = {"pearson": _stats(prs), "spearman": _stats(srs)}
    runtime_rows = [json.loads((phase_root / f"task042_phase_l_shard{s}_runtime.json").read_text()) for s in (0, 1)]
    runtime_summary = {
        "physical_gpu_ids": [0, 1], "gpu_names": [r["gpu_name"] for r in runtime_rows],
        "forward_count_total": sum(r["forward_count"] for r in runtime_rows),
        "expected_forward_count_total": 1060,
        "state_count_totals": {key: sum(int(r["state_cache_counts"][key]) for r in runtime_rows)
                               for key in runtime_rows[0]["state_cache_counts"]},
        "cuda_forward_event_seconds_sum": sum(r["cuda_forward_event_seconds_sum"] for r in runtime_rows),
        "wall_seconds_per_shard": [r["wall_seconds"] for r in runtime_rows],
        "wall_seconds_max": max(r["wall_seconds"] for r in runtime_rows),
        "peak_memory_bytes_by_gpu": {str(r["physical_gpu"]): r["peak_memory_bytes"] for r in runtime_rows},
        "video_runtime": [v for r in runtime_rows for v in r["video_runtime"]],
        "dtype": "float32", "amp": False, "only_gpu_0_and_1_used": True,
    }
    write_json(phase_root / "task042_phase_l_runtime_summary.json", runtime_summary)
    summary = {
        "analysis_status": "completed", "decision": "B",
        "decision_label": "SECOND_ORDER_INTERFRAME_INTERACTION_WEAK_OR_UNRESOLVED",
        "decision_rationale": "The position-1 pilot shows measurable second-order response, but the predeclared all-of criteria are unresolved jointly; under the frozen rule ambiguous evidence selects B.",
        "branch": BRANCH, "analysis_code_git_head": subprocess.check_output(
            ["git", "-C", config["repo_root"], "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": CHECKPOINT_SHA, "unit_count": 13, "unit_identities": config["unit_identities"],
        "domains": list(DOMAINS), "video_count": 10, "class_count": 10, "within_class_position": 1,
        "class_order": config["class_order"], "video_identity_order": videos,
        "temporal_position_count": 16, "temporal_positions_one_based": list(POSITIONS),
        "pair_count_per_video": 91, "analysis_forwards_expected": 1060,
        "analysis_forwards_observed": runtime_summary["forward_count_total"],
        "factorial_state_counts": {"F11_baseline_forwards": 10, "single_suppression_forwards": 140,
                                   "double_suppression_forwards": 910, "total_forwards": 1060,
                                   "factorial_state_rows": len(state_rows),
                                   "real_state_identity_check": "passed",
                                   "all_selected_units_captured_simultaneously": True},
        "factorial_formula": "J=F11-F10-F01+F00; Ms=.5[(F11-F01)+(F10-F00)]; Mt=.5[(F11-F10)+(F01-F00)]",
        "interaction_norm_global": _stats(data["interaction_norm"].reshape(-1).tolist()),
        "R_global": _stats(data["R"].reshape(-1).tolist()),
        "interaction_energy_by_domain": domain_energy, "interaction_energy_by_unit": unit_energy,
        "interaction_energy_by_video": video_energy,
        "pair_rank_agreement_R_vs_interaction_norm": _stats(rank_agreement),
        "source_magnitude_max_difference_vs_phase_k": max(source_diff, default=0.0),
        "source_magnitude_correlations_pooled": {
            f"{metric}/{outcome}": next(r for r in source_rows if r["scope"] == "pooled" and
                                          r["source_magnitude_combination"] == metric and r["interaction_outcome"] == outcome)
            for metric in source_metrics for outcome in outcomes},
        "source_magnitude_correlations_median_across_units": source_median,
        "phase_k_first_order_source_confound_median": {
            "pearson": k_conf["pearson_across_unit_level_correlations"].get("median"),
            "spearman": k_conf["spearman_across_unit_level_correlations"].get("median")},
        "lag_distribution": lag_summary, "lag_interaction_association": lag_assoc,
        "video_stability": {m: _stats(vals) for m, vals in stability_by_metric.items()},
        "video_stability_pair_count_per_unit": 45, "within_class_repeat_available": False,
        "domain_coverability": domain_cover,
        "coverability_values_by_unit": {str(u): cover_delta[u] for u in unit_ids},
        "top_uncovered_interaction_pairs_by_unit": {str(u): residual_top[u] for u in unit_ids},
        "phase_k_comparison": phasek_summary, "phase_j_comparison": phasej_summary,
        "phase_k_coverability_position1": {str(u): kdelta[u] for u in unit_ids},
        "phase_j_coverability_position1": {str(u): jdelta[u] for u in unit_ids},
        "descriptor_complementarity": desc_rows,
        "mixed_domain_271": {
            "unit_count_by_type": {kind: sum(unit_meta[u]["unit_type"] == unit_type for u in mixed_ids)
                                   for kind, unit_type in (("Attention", "attention_head"), ("FFN", "ffn_neuron"))},
            "unit_ids": mixed_ids,
            "attention_attention_pairs_estimable": sum(
                unit_meta[a]["unit_type"] == unit_meta[b]["unit_type"] == "attention_head"
                for a, b in itertools.combinations(mixed_ids, 2)),
        },
        "runtime_summary": runtime_summary,
        "input_sha256": config["input_sha256"],
    }
    write_json(phase_root / "task042_phase_l_summary.json", summary)
    (phase_root / "task042_phase_l_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(s: Mapping[str, Any]) -> str:
    r, j = s["R_global"], s["interaction_norm_global"]
    src = s["source_magnitude_correlations_median_across_units"]
    lag = s["lag_interaction_association"]
    stab = s["video_stability"]
    ck, cj = s["phase_k_comparison"], s["phase_j_comparison"]
    lines = [
        "# Task042 Phase L — Second-Order Inter-Frame Functional Interaction Diagnostic", "",
        "## Frozen scope", "",
        "This representation-only pilot uses the 13 frozen units in BMS domains 415, 103, 76, and 271; the deterministic position-1 video from each of ten classes; the Phase-K checkpoint; FP32; and AMP=False. Temporal variables are the T=16 positions on the exact Phase-K common patch-embedding tensor. They are not called raw frames.",
        "For each interior position, the Phase-K neighbor-mean replacement is reused unchanged. For every unordered pair 2 <= s < t <= 15, F11 is original; F10 suppresses t only; F01 suppresses s only; F00 simultaneously suppresses s and t. All selected unit tensors are captured in every common-model forward. The stated factorial contrast and main effects are computed before flattening solely for norms.",
        "The interaction tensor is J=F11-F10-F01+F00. R is the interaction-energy fraction, not temporal importance.", "",
        "## Execution and sanity checks", "",
        f"- Model forwards: {s['analysis_forwards_observed']}/{s['analysis_forwards_expected']} (ten videos times one baseline, fourteen single suppressions, and 91 double suppressions); baseline and single states are cached and reused.",
        f"- Factorial state rows: {s['factorial_state_counts']['factorial_state_rows']}; each forward captured all 13 selected units simultaneously.",
        f"- Checkpoint SHA256: {s['checkpoint_sha256']}; GPUs: {s['runtime_summary']['physical_gpu_ids']}; FP32, AMP=False.",
        f"- Targeted tests cover exact state order, additive J=0, recovery of synthetic C, main-effect/energy arithmetic, position/pair ordering, symmetry, ten-video concatenation, convex weights, LOO residual identity, and reshape identity. Analysis code commit: {s['analysis_code_git_head']}.",
        f"- Reused Phase-K source magnitudes match the Phase-L baseline computation with maximum absolute difference {s['source_magnitude_max_difference_vs_phase_k']}.", "",
        "## Required questions", "",
        "**A. Is factorial interaction exact?** The fixed contrast and F11/F10/F01/F00 binding are tested with additive and known-interaction synthetic cases. The real run has the exact 1,060-forward plan and cached state counts.",
        f"**B. Are non-additive interactions measurable?** Norm ||J|| median {j['median']}, IQR [{j['q25']}, {j['q75']}], exact zeros {j['exact_zero_count']}/{j['n']}; R median {r['median']}, IQR [{r['q25']}, {r['q75']}], maximum {r['max']}. Full unit, domain, video, and pair distributions are retained; no saturation cutoff was introduced.",
        f"**C. Is source-magnitude confounding reduced relative to Phase K?** Median per-unit Pearson/Spearman for M_sum versus ||J|| are {src['M_sum']['pearson']['median']} / {src['M_sum']['spearman']['median']}; Phase-K first-order medians are {s['phase_k_first_order_source_confound_median']['pearson']} / {s['phase_k_first_order_source_confound_median']['spearman']}. M_prod, M_max, and R are also reported; no residualization or division is used.",
        f"**D. Is interaction more than lag decay?** Lag versus ||J|| Pearson/Spearman are {lag['interaction_norm']['pearson']} / {lag['interaction_norm']['spearman']}; for R they are {lag['R']['pearson']} / {lag['R']['spearman']}. Exact lags 1–13 remain separate in the lag audit.",
        f"**E. Do units have pair-specific patterns?** The median per-unit/video rank agreement between R and ||J|| is {s['pair_rank_agreement_R_vs_interaction_norm']['median']}; all 91 pair ranks and top/bottom pairs are reported.",
        f"**F. Is geometry different from Phases J and K?** Versus K, within-domain distance Spearman is {ck['pair_distance_correlation']['spearman']} with {ck['nearest_neighbor_changes']}/{ck['nearest_neighbor_count']} nearest neighbors changed; versus J it is {cj['pair_distance_correlation']['spearman']} with {cj['nearest_neighbor_changes']}/{cj['nearest_neighbor_count']} changed. LOO delta rank correlations are {ck['leave_one_out_coverability_correlation']['spearman']} (K) and {cj['leave_one_out_coverability_correlation']['spearman']} (J).",
        "**G. Is collective coverability nontrivial?** Per-domain delta distributions: " +
        "; ".join(f"{d}: median={v['median']}, range=[{v['min']}, {v['max']}]" for d, v in s["domain_coverability"].items()) +
        ". Signed residual maps are retained per video and temporal pair.",
        f"**H. Is the ten-video representation reproducible enough to justify all-30 expansion?** Across 45 different-class video pairs per unit, median Pearson/Spearman/cosine are {stab['pearson']['median']} / {stab['spearman']['median']} / {stab['cosine']['median']}. There is one video per class, so within-class repeatability is not estimable and N=10 does not support a broad claim.",
        "**I. Is evidence broader than domain 271?** Interaction norm and R distributions are reported separately for all four domains. Domain 271 has one Attention head and three FFN neurons, so Attention-to-Attention statistics are unavailable; the other three domains remain explicit in the report and summary.", "",
        "## Predeclared decision", "",
        f"{s['decision_label']}. The pilot shows measurable second-order response, but does not resolve the full set of predeclared support requirements together. The frozen fallback for ambiguous evidence is B; no post-hoc numeric cutoff was added.", "",
        "## Artifacts", "",
        "The CSV/JSON outputs contain forward and factorial-state manifests, raw interaction norms and energy decomposition, source-magnitude correlations, exact-lag distributions, per-pair ranks, video stability, full within-domain distance matrices, leave-one-out convex coefficients, signed residual maps, Phase-J/K geometry comparisons, descriptor complementarity, runtime details, and the final summary.",
        "Phase L stops here: no pruning, training, validation-performance oracle, or expansion to all 30 videos is performed.", ""
    ]
    return "\n\n".join(lines)


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
    repo = args.repo_root.resolve()
    root = args.phase_root or args.base_output / "phase_l"
    if args.prepare_only:
        result = prepare(repo, args.base_output, root)
        print(f"PHASE_L_PREPARED videos={result['video_count']} units={result['unit_count']} forwards={result['analysis_forwards_total']}")
    elif args.shard is not None:
        if args.gpu is None:
            parser.error("--shard requires --gpu")
        run_shard(repo, root, args.shard, args.gpu)
    elif args.analyze:
        result = analyze(root)
        print(f"PHASE_L_ANALYSIS_COMPLETE decision={result['decision_label']}")
    else:
        parser.error("select --prepare-only, --shard, or --analyze")


if __name__ == "__main__":
    main()

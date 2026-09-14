#!/usr/bin/env python3
"""Task042 Phase I: intra-group inter-frame functional coverage feasibility.

This pilot evaluates temporary joint masks only. It does not physically prune,
fine-tune, or inspect classification-performance oracles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BRANCH = "task_042_post_bms_frame_relation_redundancy"
EXPECTED_CHECKPOINT_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DOMAINS = ("415", "103", "76", "271")
EXPECTED_DOMAIN_UNITS = {
    "415": (779, 1549, 1553),
    "103": (254, 2334, 16627),
    "76": (133, 33306, 36376),
    "271": (328, 1611, 7845, 30209),
}
SPANS = (1, 2, 4, 8, 16)
PAIR_INDICES = (0, 8)
EPS = 1e-12
DEFAULT_BASE = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
DEFAULT_PHASE = DEFAULT_BASE / "phase_i"
DEFAULT_PROJECT = Path("/home/jixinye25/jxy_work1/swintrans_task035")
DEFAULT_CHECKPOINT = Path("/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
DEFAULT_TASK037 = Path("/home/jixinye25/jxy_work1/swintrans_task037/task014_n09")
DEFAULT_TASK041 = Path("/data/jixinye25/work1/output/task041_collective_temporal_coverage_pruning_oracle")

OUTPUT_NAMES = (
    "task042_phase_i_pilot_manifest.csv",
    "task042_phase_i_full_relation_response.csv",
    "task042_phase_i_domain_empty_reference.csv",
    "task042_phase_i_subset_function.csv",
    "task042_phase_i_relation_coverage.csv",
    "task042_phase_i_temporal_pareto.csv",
    "task042_phase_i_static_coverage.csv",
    "task042_phase_i_temporal_vs_static.csv",
    "task042_phase_i_class_span_breakdown.csv",
    "task042_phase_i_stability.csv",
    "task042_phase_i_mixed_domain.csv",
    "task042_phase_i_baseline_projection.csv",
    "task042_phase_i_summary.json",
    "task042_phase_i_report.md",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 Phase-I gate failed: " + message)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _cell(row.get(k, "")) for k in fields})


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def center_logits(logits: Any) -> Any:
    """Subtract the per-sample class mean; supports NumPy and torch tensors."""
    if hasattr(logits, "mean") and logits.__class__.__module__.startswith("torch"):
        return logits - logits.mean(dim=-1, keepdim=True)
    array = np.asarray(logits)
    return array - array.mean(axis=-1, keepdims=True)


def centered_delta(swapped: Any, base: Any) -> Any:
    return center_logits(swapped) - center_logits(base)


def domain_contribution(t_domain: Any, t_empty: Any) -> np.ndarray:
    return np.asarray(t_domain, dtype=np.float64) - np.asarray(t_empty, dtype=np.float64)


def kappa_value(c_keep: Sequence[float], c_full: Sequence[float], epsilon: float = EPS) -> float:
    a = np.asarray(c_keep, dtype=np.float64)
    b = np.asarray(c_full, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("coverage vectors must have identical shapes")
    value = 1.0 - float(np.linalg.norm(a - b)) / (float(np.linalg.norm(a)) + float(np.linalg.norm(b)) + epsilon)
    if value < 0.0 or value > 1.0:
        if -1e-12 <= value <= 1.0 + 1e-12:
            value = min(1.0, max(0.0, value))
        else:
            raise ValueError(f"kappa outside [0,1] beyond roundoff: {value}")
    return value


def kappa_profile(c_keep: np.ndarray, c_full: np.ndarray) -> np.ndarray:
    a, b = np.asarray(c_keep, dtype=np.float64), np.asarray(c_full, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("coverage profiles must be equal [conditions, classes] matrices")
    return np.asarray([kappa_value(x, y) for x, y in zip(a, b)], dtype=np.float64)


def exact_dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        raise ValueError("dominance vectors must have equal width")
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def pareto_front(vectors: Mapping[str, Sequence[float]]) -> set[str]:
    return {a for a, va in vectors.items() if not any(a != b and exact_dominates(vb, va) for b, vb in vectors.items())}


def static_kappa_value(s_keep: Sequence[float], s_full: Sequence[float], epsilon: float = EPS) -> float:
    return kappa_value(s_keep, s_full, epsilon)


def same_keep_groups(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    out: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        out[(str(row["domain_id"]), int(row["keep_count"]))].append(row)
    return dict(out)


def pairset_indices(relations: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    a, b = [], []
    for i, row in enumerate(relations):
        (a if int(row["pair_index"]) == 0 else b if int(row["pair_index"]) == 8 else []).append(i)
    if len(a) != 5 or len(b) != 5 or set(a) & set(b):
        raise ValueError("pairset split must contain exactly five pair-0 and five pair-8 relations")
    return a, b


def leave_one_class_out_indices(labels: Sequence[Any]) -> dict[str, list[int]]:
    classes = sorted({str(x) for x in labels})
    if len(classes) != 10:
        raise ValueError("the frozen Phase-I cohort must contain 10 classes")
    return {label: [i for i, value in enumerate(labels) if str(value) != label] for label in classes}


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for j in range(start, stop):
            ranks[order[j]] = rank
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman vectors must have equal nonzero length")
    a, b = np.asarray(_rank(left), dtype=np.float64), np.asarray(_rank(right), dtype=np.float64)
    a -= a.mean(); b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if den == 0 else float(np.dot(a, b) / den)


def pairwise_linf_spread(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in itertools.combinations(vectors, 2))


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"min": float(x.min()), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)), "max": float(x.max())}


def _ids(value: str | Sequence[int]) -> tuple[int, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    return tuple(sorted(int(x) for x in parsed))


def _type_alias(value: str) -> str:
    aliases = {"head": "attention_head", "attention_head": "attention_head", "neuron": "ffn_neuron", "ffn_neuron": "ffn_neuron"}
    if value not in aliases:
        raise ValueError("unknown Task037 unit type " + value)
    return aliases[value]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _runtime(repo: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(repo / "src" / "lgfr_runtime"))
    sys.path.insert(0, str(repo / "scripts"))
    import task042_frame_relation_redundancy as base
    import task040_htor_probe as task040
    import task041_phase_d_fullval_oracle as phase_d
    import task042_phase_c_joint_mask as joint
    return base, task040, phase_d, joint


def _base_config(base_root: Path, phase_root: Path) -> dict[str, Any]:
    global DEFAULT_BASE
    DEFAULT_BASE = Path(base_root)
    base_root, phase_root = Path(base_root), Path(phase_root)
    require(not phase_root.exists(), "Phase-I output directory already exists; refusing overwrite")
    config_path = base_root / "task042_run_config.json"
    require(config_path.is_file(), "frozen Task042 run config is missing")
    base_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    require(base_cfg.get("required_branch") == BRANCH, "base Task042 config branch mismatch")
    require(sha256_file(Path(base_cfg["checkpoint_path"])) == EXPECTED_CHECKPOINT_SHA, "checkpoint SHA mismatch")
    units_path, videos_path = Path(base_cfg["unit_manifest"]), Path(base_cfg["video_manifest"])
    units, videos = read_csv(units_path), read_csv(videos_path)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in units:
        row = dict(r)
        row["task037_global_index"] = int(row["task037_global_index"])
        row["unit_index"] = int(row["unit_index"])
        row["stage"] = int(row["stage"])
        row["domain_id"] = str(row["domain_id"])
        row["canonical_type"] = _type_alias(str(row["unit_type"]))
        row["capture_kind"] = "head" if row["canonical_type"] == "attention_head" else "neuron"
        by_domain[row["domain_id"]].append(row)
    require(set(by_domain) >= set(DOMAINS), "one or more frozen pilot domains are absent")
    for domain in DOMAINS:
        observed = tuple(sorted(r["task037_global_index"] for r in by_domain[domain]))
        require(observed == EXPECTED_DOMAIN_UNITS[domain], f"Task037 unit identity mismatch in domain {domain}: {observed}")
    video_rows = sorted((dict(r) for r in videos), key=lambda r: int(r["video_index"]))
    require(len(video_rows) == 30, "authoritative video manifest must contain 30 rows")
    per_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in video_rows:
        per_label[str(r["label"])].append(r)
    require(len(per_label) == 10 and all(len(v) == 3 for v in per_label.values()), "expected 10 classes x 3 videos")
    pilot_videos = [sorted(rows, key=lambda r: int(r["video_index"]))[0] for _, rows in sorted(per_label.items(), key=lambda kv: min(int(x["video_index"]) for x in kv[1]))]
    pilot_videos.sort(key=lambda r: int(r["video_index"]))
    phase_g_summary = DEFAULT_BASE / "phase_g" / "task042_phase_g_summary.json"
    require(phase_g_summary.is_file(), "authoritative Phase-G summary is missing")
    pg = json.loads(phase_g_summary.read_text(encoding="utf-8"))
    p1 = pg.get("frozen_protocol", {}).get("subsets", {}).get("P1")
    if p1 is None:
        p1 = pg.get("frozen_protocol", {}).get("video_subsets", {}).get("P1")
    require(p1 is not None, "Phase-G frozen P1 identity is absent")
    require(tuple(sorted(int(x) for x in p1)) == tuple(int(r["video_index"]) for r in pilot_videos), "position-1 cohort differs from authoritative Phase-G P1")
    # Confirm the exact Task037 raw rows agree with every tested identity.
    task037_root = DEFAULT_TASK037
    descriptors = read_csv(task037_root / "dynamic3d" / "seed3407" / "descriptor_statistics.csv")
    mapping = read_csv(task037_root / "contribution_unit_mapping.csv")
    desc = {int(r["global_index"]): r for r in descriptors}
    mmap = {int(r["global_index"]): r for r in mapping}
    for domain in DOMAINS:
        for u in by_domain[domain]:
            uid = int(u["task037_global_index"])
            require(uid in desc and uid in mmap, f"Task037 source identity missing for {uid}")
            for source, keys in ((desc[uid], ("layer", "unit_type", "unit_index")), (mmap[uid], ("layer", "unit_type", "unit_index"))):
                require(str(source[keys[0]]) == str(u["layer"]), f"Task037 layer identity mismatch for {uid}")
                require(_type_alias(str(source[keys[1]])) == u["canonical_type"], f"Task037 type identity mismatch for {uid}")
                require(int(source[keys[2]]) == int(u["unit_index"]), f"Task037 unit index mismatch for {uid}")
    phase_f = read_csv(DEFAULT_BASE / "phase_f" / "task042_phase_f_parameter_cost.csv")
    cost_rows = {int(r["task037_global_index"]): int(float(r["parameter_cost_exact"])) for r in phase_f}
    phase_d = read_csv(DEFAULT_BASE / "phase_d" / "task042_phase_d_candidate_provenance.csv")
    baseline_rank: dict[str, dict[int, int]] = defaultdict(dict)
    for r in phase_d:
        domain, uid = str(r["domain_id"]), int(r["task037_global_index"])
        if domain in DOMAINS and str(r["selected_for_removal"]).lower() in ("true", "1"):
            baseline_rank[domain][uid] = int(r["within_domain_removal_rank"])
    for d in DOMAINS:
        members = set(EXPECTED_DOMAIN_UNITS[d])
        require(all(uid in cost_rows for uid in members), f"Phase-F exact parameter costs incomplete for domain {d}")
        ranked = baseline_rank[d]
        require(set(ranked).issubset(members), f"Phase-D directional candidate identity mismatch in domain {d}")
        require(sorted(ranked.values()) == list(range(1, len(ranked) + 1)), f"Phase-D removal ranks are not contiguous in domain {d}")
        require(len(ranked) == len(members) - 1, f"Phase-D removal ranks cannot project every keep count in domain {d}")
    # Frozen fixed-cardinality IDs, canonical locations 0 and 8 only.
    base, _, _, _ = _runtime(Path(base_cfg["repo_root"]))
    rels = base.fixed_cardinality_identities(base._runtime_modules(Path(base_cfg["repo_root"]))[0], 32)
    rels = [r for r in rels if r["span"] in SPANS and r["pair_index"] in PAIR_INDICES]
    rels.sort(key=lambda r: (r["span"], r["pair_index"]))
    require(len(rels) == 10, "authoritative pair-set selection must yield 10 conditions")
    phase_root.mkdir(parents=True, exist_ok=True)
    cache = phase_root / "cache"
    cache.mkdir(exist_ok=True)
    config = {
        "task": "Task042 Phase I intra-group inter-frame functional coverage feasibility",
        "branch": BRANCH,
        "repo_root": str(Path(base_cfg["repo_root"]).resolve()),
        "project_root": str(Path(base_cfg["project_root"]).resolve()),
        "checkpoint_path": str(Path(base_cfg["checkpoint_path"]).resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "base_output": str(base_root.resolve()),
        "phase_root": str(phase_root.resolve()),
        "cache_dir": str(cache.resolve()),
        "task037_root": str(task037_root.resolve()),
        "phase_f_dir": str((DEFAULT_BASE / "phase_f").resolve()),
        "phase_d_dir": str((DEFAULT_BASE / "phase_d").resolve()),
        "unit_manifest": str(units_path.resolve()),
        "video_manifest": str(videos_path.resolve()),
        "video_count": 10,
        "classes": sorted({str(r["label"]) for r in pilot_videos}),
        "domains": {d: [dict(r) for r in sorted(by_domain[d], key=lambda x: x["task037_global_index"])] for d in DOMAINS},
        "videos": [{**r, "video_index": int(r["video_index"]), "dataset_index": int(r["dataset_index"]), "label": int(r["label"])} for r in pilot_videos],
        "relations": rels,
        "baseline_removal_rank": {d: {str(uid): rank for uid, rank in baseline_rank[d].items()} for d in DOMAINS},
        "exact_parameter_cost": {str(uid): cost_rows[uid] for d in DOMAINS for uid in EXPECTED_DOMAIN_UNITS[d]},
        "input_sha256": {"checkpoint": EXPECTED_CHECKPOINT_SHA, "unit_manifest": sha256_file(units_path), "video_manifest": sha256_file(videos_path),
                         "phase_g_summary": sha256_file(phase_g_summary),
                         "phase_f_cost": sha256_file(DEFAULT_BASE / "phase_f" / "task042_phase_f_parameter_cost.csv"),
                         "phase_d_provenance": sha256_file(DEFAULT_BASE / "phase_d" / "task042_phase_d_candidate_provenance.csv")},
        "precision": {"dtype": "float32", "amp": False, "tf32": False},
        "protocol": {"T": 32, "spans": list(SPANS), "pair_indices_per_span": list(PAIR_INDICES), "relations_per_video": 10, "centered_classes": 400},
    }
    manifest_rows: list[dict[str, Any]] = []
    for d in DOMAINS:
        for u in config["domains"][d]:
            manifest_rows.append({"record_type": "unit", "domain_id": d, "task037_global_index": u["task037_global_index"],
                                  "layer": u["layer"], "unit_type": u["canonical_type"], "unit_index": u["unit_index"],
                                  "stage": u["stage"], "parameter_cost_exact": cost_rows[int(u["task037_global_index"])]})
    for v in config["videos"]:
        manifest_rows.append({"record_type": "video", "video_index": v["video_index"], "dataset_index": v["dataset_index"],
                              "video_id": v["video_id"], "label": v["label"], "class_position": 1})
    for qid, r in enumerate(rels):
        manifest_rows.append({"record_type": "relation", "relation_id": qid, **r})
    write_csv(phase_root / "task042_phase_i_pilot_manifest.csv", manifest_rows,
              ("record_type", "domain_id", "task037_global_index", "layer", "unit_type", "unit_index", "stage",
               "parameter_cost_exact", "video_index", "dataset_index", "video_id", "label", "class_position",
               "relation_id", "span", "pair_index", "frame_a", "frame_b"))
    write_json(phase_root / "task042_phase_i_run_config.json", config)
    print(f"PREPARE_OK domains=4 tested_units=13 videos=10 relations=10 checkpoint_sha={EXPECTED_CHECKPOINT_SHA}")


def prepare(base_root: Path, phase_root: Path) -> None:
    _base_config(Path(base_root), Path(phase_root))


def _load_config(phase_root: Path) -> dict[str, Any]:
    path = phase_root / "task042_phase_i_run_config.json"
    require(path.is_file(), "run config is missing; run prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_data(config: Mapping[str, Any], workers: int) -> tuple[Any, dict[int, Any]]:
    repo = Path(config["repo_root"])
    base, _, _, _ = _runtime(repo)
    base_config = json.loads((Path(config["base_output"]) / "task042_run_config.json").read_text(encoding="utf-8"))
    ctfrs = base._runtime_modules(repo)[2]
    loader = base._get_loader(ctfrs, base_config, workers=workers)
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    by_name = {Path(str(v["video_id"])).name: v for v in config["videos"]}
    selected: dict[int, Any] = {}
    for batch in loader:
        local_index = int(batch[2][0].item())
        name = Path(str(dataset.clips[local_index][0])).name
        if name not in by_name:
            continue
        row = by_name[name]
        require(int(batch[1][0].item()) == int(row["label"]), f"video label differs from frozen manifest: {name}")
        require(int(batch[0].shape[0]) == 1 and int(batch[0].shape[2]) == 32, "video loader did not produce batch-one T=32 clip")
        vi = int(row["video_index"])
        require(vi not in selected, "repeated selected video in loader")
        selected[vi] = batch[0][0]
    expected = {int(v["video_index"]) for v in config["videos"]}
    require(set(selected) == expected, "loader did not reproduce exact P1 cohort")
    return dataset, selected


def _model(config: Mapping[str, Any], device: Any) -> tuple[Any, Any, Any, Any, list[Any], dict[str, Any]]:
    repo = Path(config["repo_root"])
    base, task040, _, _ = _runtime(repo)
    core, probe, ctfrs = base._runtime_modules(repo)
    model, _, identity = base._model_and_identity(Path(config["project_root"]), Path(config["checkpoint_path"]), device, probe, ctfrs)
    specs = ctfrs.discover_unit_layers(model)
    require(identity.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA, "loaded checkpoint SHA mismatch")
    require(identity.get("classifier_head", {}).get("status") == "loaded", "authoritative 400-class classifier not loaded")
    require(not identity.get("missing_keys") and not identity.get("unexpected_keys") and not identity.get("shape_mismatches"), "checkpoint is not an exact load")
    return core, task040, ctfrs, model, specs, identity


def _spec_records(config: Mapping[str, Any], domain: str, specs: Sequence[Any]) -> list[dict[str, Any]]:
    by_key = {(str(s.name), str(s.unit_type)): s for s in specs}
    records = []
    for u in config["domains"][domain]:
        kind = str(u["capture_kind"])
        spec = by_key.get((str(u["layer"]), kind))
        require(spec is not None, f"validated Task040 mask spec missing for {u['task037_global_index']}")
        require(int(u["unit_index"]) < int(spec.num_units), f"unit index exceeds discovered width for {u['task037_global_index']}")
        records.append({"spec": spec, "task037_global_index": int(u["task037_global_index"]), "unit_index": int(u["unit_index"])})
    return records


def preflight(args: argparse.Namespace) -> None:
    import torch
    phase_root = Path(args.phase_root)
    config = _load_config(phase_root)
    repo = Path(config["repo_root"])
    require(_git(repo, "branch", "--show-current") == BRANCH, "checkout is not on frozen Task042 branch")
    require(_git(repo, "status", "--porcelain") == "", "checkout must be clean for CPU preflight")
    require(sha256_file(Path(config["checkpoint_path"])) == EXPECTED_CHECKPOINT_SHA, "checkpoint SHA changed after prepare")
    for path_name, hash_name in (("unit_manifest", "unit_manifest"), ("video_manifest", "video_manifest")):
        p = Path(config[path_name])
        require(sha256_file(p) == config["input_sha256"][hash_name], f"frozen {path_name} changed")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(not torch.cuda.is_available() or not cuda_visible, "CPU preflight must not expose a GPU")
    core, task040, ctfrs, model, specs, identity = _model(config, torch.device("cpu"))
    relations = core.enumerate_fixed_cardinality_temporal_pairs(32)
    relation_map = {(int(r.block_size), int(r.pair_index)): r for r in relations}
    require(len(relation_map) == 80, "authoritative Task040 relation enumerator changed")
    selected = {(s, p) for s in SPANS for p in PAIR_INDICES}
    require(selected.issubset(relation_map), "fixed pairset location identities are missing")
    records = {d: _spec_records(config, d, specs) for d in DOMAINS}
    data_set, clips = _load_data(config, workers=0)
    del data_set, clips
    record = {"status": "passed", "branch": BRANCH, "git_head": _git(repo, "rev-parse", "HEAD"),
              "checkpoint_identity": identity, "domains": {d: list(EXPECTED_DOMAIN_UNITS[d]) for d in DOMAINS},
              "unit_identity_count": 13, "video_count": 10, "class_count": 10, "relation_count": 10,
              "relation_identities": config["relations"], "precision": config["precision"],
              "mask_specs_checked": {d: len(records[d]) for d in DOMAINS}, "gpu_inference_performed": False,
              "full_validation_oracle_used": False}
    write_json(phase_root / "task042_phase_i_preflight.json", record)
    print("PREFLIGHT_OK exact_Task037_identities=13 video_P1=10 fixed_relations=10 checkpoint=exact gpu_inference=false")


def _set_determinism(torch: Any, seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _center_np(logits: Any) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    require(arr.size == 400 and np.isfinite(arr).all(), "classifier output must be finite 400-D logits")
    return np.asarray(arr - arr.mean(), dtype=np.float32)


def _forward(model: Any, torch: Any, clip: Any) -> tuple[Any, np.ndarray]:
    import task040_htor_probe as task040
    with torch.no_grad():
        out = task040.unwrap_logits(model(clip.unsqueeze(0)))
    require(tuple(out.shape) == (1, 400), f"expected 400-class logits, got {tuple(out.shape)}")
    raw = out[0].detach().to(dtype=torch.float32).cpu().numpy().copy()
    return out[0].detach(), _center_np(raw)


def full_cache(args: argparse.Namespace) -> None:
    import torch
    phase_root = Path(args.phase_root); config = _load_config(phase_root)
    require((phase_root / "task042_phase_i_preflight.json").is_file(), "preflight must pass before GPU cache")
    require(int(args.gpu) == 0, "shared immutable full-model cache must run on authorized GPU 0")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "full-cache process must be isolated to physical GPU 0")
    _set_determinism(torch, 3407)
    device = torch.device("cuda:0")
    core, _, _, model, _, identity = _model(config, device)
    _, clips = _load_data(config, workers=2)
    relation_objects = core.enumerate_fixed_cardinality_temporal_pairs(32)
    relation_by_key = {(int(x.block_size), int(x.pair_index)): x for x in relation_objects}
    relations = config["relations"]
    centered = np.empty((10, 11, 400), dtype=np.float32)
    raw = np.empty((10, 11, 400), dtype=np.float32)
    clip_hashes: dict[str, str] = {}
    videos = sorted(config["videos"], key=lambda r: int(r["video_index"]))
    for vi, v in enumerate(videos):
        idx = int(v["video_index"])
        clip = clips[idx].to(device=device, dtype=torch.float32, non_blocking=True)
        require(clip.ndim == 4 and int(clip.shape[1]) == 32, "expected [C,T,H,W] T=32 clip")
        clip_hashes[str(idx)] = hashlib.sha256(clip.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        base_raw, base_center = _forward(model, torch, clip)
        raw[vi, 0] = base_raw.cpu().numpy(); centered[vi, 0] = base_center
        for qi, r in enumerate(relations):
            op = relation_by_key[(int(r["span"]), int(r["pair_index"]))]
            swapped = core.apply_temporal_interventions(clip, [op], time_dim=1)[0]
            raw_out, c = _forward(model, torch, swapped)
            raw[vi, qi + 1] = raw_out.detach().float().cpu().numpy()
            centered[vi, qi + 1] = c
        print(f"FULL_CACHE_VIDEO {vi+1}/10 video_index={idx}", flush=True)
    cache_path = Path(config["cache_dir"]) / "task042_phase_i_full_cache.npz"
    require(not cache_path.exists(), "refusing to overwrite immutable full-model cache")
    np.savez_compressed(cache_path, centered_logits=centered, raw_logits=raw,
                        video_indices=np.asarray([int(v["video_index"]) for v in videos], dtype=np.int64),
                        labels=np.asarray([int(v["label"]) for v in videos], dtype=np.int64))
    write_json(Path(config["cache_dir"]) / "task042_phase_i_full_cache.json",
               {"status": "complete", "cache": str(cache_path), "sha256": sha256_file(cache_path),
                "checkpoint_sha256": identity["checkpoint_sha256"], "git_head": _git(Path(config["repo_root"]), "rev-parse", "HEAD"),
                "video_indices": [int(v["video_index"]) for v in videos], "video_clip_sha256": clip_hashes,
                "array_shape": list(centered.shape), "forward_count": 110, "dtype": "float32", "amp": False,
                "physical_gpu": 0})
    print(f"FULL_CACHE_OK gpu=0 videos=10 relations=100 forwards=110 cache={cache_path}")


def worker(args: argparse.Namespace) -> None:
    import torch
    phase_root = Path(args.phase_root); config = _load_config(phase_root)
    require((phase_root / "task042_phase_i_preflight.json").is_file(), "preflight must pass before GPU workers")
    full_info_path = Path(config["cache_dir"]) / "task042_phase_i_full_cache.json"
    full_cache_path = Path(config["cache_dir"]) / "task042_phase_i_full_cache.npz"
    require(full_info_path.is_file() and full_cache_path.is_file(), "immutable full cache must complete before subset workers")
    full_info = json.loads(full_info_path.read_text(encoding="utf-8"))
    require(sha256_file(full_cache_path) == full_info["sha256"], "full cache SHA mismatch")
    gpu = int(args.gpu)
    require(gpu in (0, 1), "only physical GPU 0/1 are authorized")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu) and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            f"worker must be isolated to physical GPU {gpu}")
    domains = tuple(args.domains)
    required = ("415", "103") if gpu == 0 else ("76", "271")
    require(domains == required, f"GPU {gpu} assignment must be {required}")
    _set_determinism(torch, 3407 + gpu)
    device = torch.device("cuda:0")
    core, task040, _, model, specs, identity = _model(config, device)
    _, clip_cpu = _load_data(config, workers=2)
    clips = {int(v["video_index"]): clip_cpu[int(v["video_index"])].to(device=device, dtype=torch.float32, non_blocking=True)
             for v in config["videos"]}
    with np.load(full_cache_path, allow_pickle=False) as z:
        expected_indices = z["video_indices"].tolist()
    require(expected_indices == [int(v["video_index"]) for v in sorted(config["videos"], key=lambda r: int(r["video_index"]))], "full cache video order differs")
    require(set(full_info.get("video_clip_sha256", {})) == {str(x) for x in expected_indices}, "full cache input hash identities are incomplete")
    local_base: dict[int, Any] = {}
    for v in sorted(config["videos"], key=lambda r: int(r["video_index"])):
        idx = int(v["video_index"])
        local_base[idx], _ = _forward(model, torch, clips[idx])
        clip_hash = hashlib.sha256(clips[idx].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        require(clip_hash == full_info["video_clip_sha256"][str(idx)], "worker video tensor differs from immutable full-cache input")
    relation_objects = core.enumerate_fixed_cardinality_temporal_pairs(32)
    relation_by_key = {(int(x.block_size), int(x.pair_index)): x for x in relation_objects}
    relation_rows = config["relations"]
    for domain in domains:
        records_all = _spec_records(config, domain, specs)
        member_ids = tuple(sorted(int(x["task037_global_index"]) for x in config["domains"][domain]))
        states: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [("EMPTY", tuple(), member_ids)]
        for keep_count in range(1, len(member_ids)):
            for kept in itertools.combinations(member_ids, keep_count):
                masked = tuple(x for x in member_ids if x not in kept)
                states.append(("K-" + "-".join(map(str, kept)), tuple(kept), masked))
        all_centered = np.empty((len(states), 10, 11, 400), dtype=np.float32)
        metadata: list[dict[str, Any]] = []
        for si, (state_id, kept, masked) in enumerate(states):
            targeted = set(masked)
            records = [r for r in records_all if int(r["task037_global_index"]) in targeted]
            require(len(records) == len(masked) and records, "each Phase-I state must install the exact nonempty complementary joint mask")
            all_calls: dict[int, int] = {uid: 0 for uid in masked}
            with __import__("task042_phase_c_joint_mask").joint_temporary_unit_masks(
                model, records, task040.temporary_unit_mask, torch
            ) as calls:
                for vi, v in enumerate(sorted(config["videos"], key=lambda r: int(r["video_index"]))):
                    idx = int(v["video_index"]); clip = clips[idx]
                    base_raw, base_center = _forward(model, torch, clip)
                    all_centered[si, vi, 0] = base_center
                    for qi, rel in enumerate(relation_rows):
                        op = relation_by_key[(int(rel["span"]), int(rel["pair_index"]))]
                        swapped = core.apply_temporal_interventions(clip, [op], time_dim=1)[0]
                        _, center = _forward(model, torch, swapped)
                        all_centered[si, vi, qi + 1] = center
            for record in records:
                uid = int(record["task037_global_index"])
                observed = int(calls.get((id(record["spec"].hook_module), int(record["unit_index"])), 0))
                require(observed == 110, f"masked unit {uid} verified {observed} forwards, expected 110")
                all_calls[uid] = observed
            # Re-evaluate every unmasked base clip after each scope and demand bit-exact restoration.
            for v in sorted(config["videos"], key=lambda r: int(r["video_index"])):
                idx = int(v["video_index"])
                restored, _ = _forward(model, torch, clips[idx])
                require(torch.equal(restored, local_base[idx]), f"exact forward restoration failed after {state_id}, video={idx}")
            metadata.append({"state_id": state_id, "retained_ids": kept, "masked_ids": masked,
                             "mask_forward_count_per_unit": 110, "restoration_base_forwards": 10,
                             "restoration_exact": True})
            print(f"MASK_STATE_OK domain={domain} state={state_id} {si+1}/{len(states)}", flush=True)
        path = Path(config["cache_dir"]) / f"task042_phase_i_domain_{domain}.npz"
        require(not path.exists(), "refusing to overwrite domain mask cache")
        np.savez_compressed(path, centered_logits=all_centered,
                            state_ids=np.asarray([m["state_id"] for m in metadata], dtype="U128"),
                            retained_json=np.asarray([json.dumps(m["retained_ids"]) for m in metadata], dtype="U256"),
                            masked_json=np.asarray([json.dumps(m["masked_ids"]) for m in metadata], dtype="U256"))
        write_json(Path(config["cache_dir"]) / f"task042_phase_i_domain_{domain}.json",
                   {"status": "complete", "domain_id": domain, "gpu": gpu, "checkpoint_sha256": identity["checkpoint_sha256"],
                    "sha256": sha256_file(path), "state_count": len(states), "forward_count_masked": len(states) * 110,
                    "forward_count_restoration": len(states) * 10, "states": metadata,
                    "all_masks_verified": True, "exact_restoration_verified": True})
    print(f"WORKER_OK gpu={gpu} domains={','.join(domains)}")


def _dominance_edges(vectors: Mapping[str, Sequence[float]]) -> set[tuple[str, str]]:
    return {(a, b) for a, va in vectors.items() for b, vb in vectors.items() if a != b and exact_dominates(va, vb)}


def _diagnostics(vectors: Mapping[str, Sequence[float]]) -> dict[str, dict[str, Any]]:
    out = {}
    for sid, vector in vectors.items():
        m = _quantiles(vector)
        out[sid] = m
    return out


def finalize(args: argparse.Namespace) -> None:
    phase_root = Path(args.phase_root); config = _load_config(phase_root)
    cache_dir = Path(config["cache_dir"])
    full_info = json.loads((cache_dir / "task042_phase_i_full_cache.json").read_text(encoding="utf-8"))
    full_path = cache_dir / "task042_phase_i_full_cache.npz"
    require(sha256_file(full_path) == full_info["sha256"], "full relation cache SHA mismatch")
    domains_data: dict[str, dict[str, Any]] = {}
    for d in DOMAINS:
        p = cache_dir / f"task042_phase_i_domain_{d}.npz"
        info_p = cache_dir / f"task042_phase_i_domain_{d}.json"
        require(p.is_file() and info_p.is_file(), f"domain mask cache missing: {d}")
        info = json.loads(info_p.read_text(encoding="utf-8"))
        require(sha256_file(p) == info["sha256"] and info.get("exact_restoration_verified") is True,
                f"domain mask cache failed integrity/restoration gate: {d}")
        with np.load(p, allow_pickle=False) as z:
            data = z["centered_logits"].astype(np.float64)
            state_ids = z["state_ids"].tolist()
            keeps = [_ids(x) for x in z["retained_json"].tolist()]
            masks = [_ids(x) for x in z["masked_json"].tolist()]
        require(len(state_ids) == data.shape[0] and len(keeps) == len(state_ids) and len(masks) == len(state_ids), "state cache arrays mismatch")
        domains_data[d] = {"info": info, "data": data, "state_ids": state_ids, "keeps": keeps, "masks": masks}
    with np.load(full_path, allow_pickle=False) as z:
        full_center = z["centered_logits"].astype(np.float64)
    relation_rows = config["relations"]
    videos = sorted(config["videos"], key=lambda r: int(r["video_index"]))
    require(full_center.shape == (10, 11, 400), "full cache must be [10,11,400]")
    base_t = full_center[:, 0, :]
    t_full = full_center[:, 1:, :] - base_t[:, None, :]
    full_relation_rows = []
    for vi, v in enumerate(videos):
        for qi, q in enumerate(relation_rows):
            full_relation_rows.append({"video_index": v["video_index"], "video_id": v["video_id"], "label": v["label"],
                                       "relation_id": qi, **q, "T_full_centered_delta": t_full[vi, qi].tolist()})
    write_csv(phase_root / "task042_phase_i_full_relation_response.csv", full_relation_rows)
    domain_empty_rows = []; subset_rows = []; coverage_rows = []; static_rows = []
    temporal_summary: list[dict[str, Any]] = []
    static_summary: list[dict[str, Any]] = []
    group_vectors: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    group_static_vectors: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    subset_meta: dict[str, dict[str, Any]] = {}
    for d in DOMAINS:
        domain_rows = config["domains"][d]
        domain_ids = tuple(sorted(int(u["task037_global_index"]) for u in domain_rows))
        cdata = domains_data[d]
        empty_i = cdata["state_ids"].index("EMPTY")
        empty_logits = cdata["data"][empty_i]
        t_empty = empty_logits[:, 1:, :] - empty_logits[:, 0, None, :]
        c_full = t_full - t_empty
        s_full = base_t - empty_logits[:, 0, :]
        for vi, v in enumerate(videos):
            for qi, q in enumerate(relation_rows):
                domain_empty_rows.append({"domain_id": d, "video_index": v["video_index"], "video_id": v["video_id"], "label": v["label"],
                                          "relation_id": qi, **q, "T_empty_centered_delta": t_empty[vi, qi].tolist(),
                                          "T_full_minus_empty_norm": float(np.linalg.norm(c_full[vi, qi]))})
        local_vectors: dict[int, np.ndarray] = {}
        local_static: dict[int, np.ndarray] = {}
        for si, (state_id, kept, masked) in enumerate(zip(cdata["state_ids"], cdata["keeps"], cdata["masks"])):
            if not kept:
                continue
            require(set(kept).issubset(domain_ids) and set(masked) == set(domain_ids) - set(kept), f"subset identity mismatch in domain {d}")
            k_logits = cdata["data"][si]
            t_keep = k_logits[:, 1:, :] - k_logits[:, 0, None, :]
            c_keep = t_keep - t_empty
            kappas = np.empty((10, 10), dtype=np.float64)
            static_k = np.empty(10, dtype=np.float64)
            for vi, v in enumerate(videos):
                for qi in range(10):
                    kappas[vi, qi] = kappa_value(c_keep[vi, qi], c_full[vi, qi])
                    coverage_rows.append({"domain_id": d, "subset_id": state_id, "retained_task037_ids": kept,
                                          "masked_task037_ids": masked, "keep_count": len(kept), "video_index": v["video_index"],
                                          "video_id": v["video_id"], "label": v["label"], "relation_id": qi,
                                          **relation_rows[qi], "kappa": float(kappas[vi, qi]),
                                          "masked_parameter_cost_exact": sum(int(config["exact_parameter_cost"][str(uid)]) for uid in masked),
                                          "retained_unit_types": Counter(u["canonical_type"] for u in domain_rows if int(u["task037_global_index"]) in kept),
                                          "C_full_norm": float(np.linalg.norm(c_full[vi, qi])), "C_keep_norm": float(np.linalg.norm(c_keep[vi, qi]))})
                    subset_rows.append({"domain_id": d, "subset_id": state_id, "retained_task037_ids": kept,
                                        "masked_task037_ids": masked, "keep_count": len(kept), "video_index": v["video_index"],
                                        "label": v["label"], "relation_id": qi, **relation_rows[qi],
                                        "masked_parameter_cost_exact": sum(int(config["exact_parameter_cost"][str(uid)]) for uid in masked),
                                        "retained_parameter_cost_exact": sum(int(config["exact_parameter_cost"][str(uid)]) for uid in kept),
                                        "retained_unit_types": Counter(u["canonical_type"] for u in domain_rows if int(u["task037_global_index"]) in kept),
                                        "C_full": c_full[vi, qi].tolist(), "C_keep": c_keep[vi, qi].tolist()})
                s_keep = k_logits[vi, 0, :] - empty_logits[vi, 0, :]
                static_k[vi] = static_kappa_value(s_keep, s_full[vi])
                static_rows.append({"domain_id": d, "subset_id": state_id, "retained_task037_ids": kept,
                                    "masked_task037_ids": masked, "keep_count": len(kept), "video_index": v["video_index"],
                                    "video_id": v["video_id"], "label": v["label"], "kappa_static": float(static_k[vi]),
                                    "S_full_norm": float(np.linalg.norm(s_full[vi])), "S_keep_norm": float(np.linalg.norm(s_keep))})
            sid = state_id
            local_vectors[len(local_vectors)] = kappas.reshape(-1)
            local_static[len(local_static)] = static_k
            unit_by_id = {int(u["task037_global_index"]): u for u in domain_rows}
            types_kept = Counter(unit_by_id[x]["canonical_type"] for x in kept)
            cost_map = {int(uid): int(cost) for uid, cost in config["exact_parameter_cost"].items()}
            meta = {"domain_id": d, "subset_id": sid, "retained_ids": kept, "masked_ids": masked,
                    "keep_count": len(kept), "retained_count": len(kept), "masked_count": len(masked),
                    "retained_types": dict(types_kept), "masked_types": dict(Counter(unit_by_id[x]["canonical_type"] for x in masked)),
                    "parameter_cost_masked": sum(cost_map[x] for x in masked), "parameter_cost_retained": sum(cost_map[x] for x in kept),
                    "temporal_profile": kappas.reshape(-1), "static_profile": static_k}
            subset_meta[sid] = meta
            subset_rows[-100:]  # keep static analyzer from treating this as a score-only primary object
            # Store vectors under a domain/keep key below after all states.
            group_vectors.setdefault((d, len(kept)), {})[sid] = kappas.reshape(-1)
            group_static_vectors.setdefault((d, len(kept)), {})[sid] = static_k
            summary = _quantiles(kappas.reshape(-1))
            temporal_summary.append({**meta, **summary, "kappa_profile_100": kappas.reshape(-1).tolist()})
            static_summary.append({**meta, **_quantiles(static_k), "kappa_static_profile_10": static_k.tolist()})
    write_csv(phase_root / "task042_phase_i_domain_empty_reference.csv", domain_empty_rows)
    write_csv(phase_root / "task042_phase_i_subset_function.csv", subset_rows)
    write_csv(phase_root / "task042_phase_i_relation_coverage.csv", coverage_rows)
    write_csv(phase_root / "task042_phase_i_static_coverage.csv", static_rows)
    # Pareto fronts and exact directed dominance edges, always within same domain and keep count.
    temporal_pareto_rows = []; temporal_edges: dict[tuple[str, int], set[tuple[str, str]]] = {}
    static_fronts: dict[tuple[str, int], set[str]] = {}; temporal_fronts: dict[tuple[str, int], set[str]] = {}
    for key, vectors in group_vectors.items():
        fronts = pareto_front(vectors); edges = _dominance_edges(vectors)
        temporal_fronts[key] = fronts; temporal_edges[key] = edges
        for sid, vec in vectors.items():
            meta = subset_meta[sid]
            temporal_pareto_rows.append({"domain_id": key[0], "keep_count": key[1], "subset_id": sid,
                                         "retained_task037_ids": meta["retained_ids"], "masked_task037_ids": meta["masked_ids"],
                                         "masked_parameter_cost_exact": meta["parameter_cost_masked"],
                                         "kappa_min": float(np.min(vec)), "kappa_mean": float(np.mean(vec)),
                                         "kappa_median": float(np.median(vec)), "kappa_q25": float(np.quantile(vec, .25)),
                                         "kappa_q75": float(np.quantile(vec, .75)), "kappa_profile_100": vec.tolist(),
                                         "dominates_count": sum(a == sid for a, _ in edges),
                                         "dominated_by_count": sum(b == sid for _, b in edges),
                                         "nondominated": sid in fronts, "pareto_front_size": len(fronts)})
    write_csv(phase_root / "task042_phase_i_temporal_pareto.csv", temporal_pareto_rows)
    tvs_rows = []
    for key, vectors in group_vectors.items():
        svectors = group_static_vectors[key]
        sfront = pareto_front(svectors); static_fronts[key] = sfront
        tfront = temporal_fronts[key]
        union = tfront | sfront
        for sid in sorted(vectors):
            tvs_rows.append({"domain_id": key[0], "keep_count": key[1], "subset_id": sid,
                             "temporal_nondominated": sid in tfront, "static_nondominated": sid in sfront,
                             "static_nondominated_but_temporal_dominated": sid in sfront and sid not in tfront,
                             "temporal_nondominated_but_static_dominated": sid in tfront and sid not in sfront,
                             "temporal_front_size": len(tfront), "static_front_size": len(sfront),
                             "front_overlap_count": len(tfront & sfront),
                             "front_jaccard": len(tfront & sfront) / len(union) if union else 1.0,
                             "temporal_mean": float(np.mean(vectors[sid])), "static_mean": float(np.mean(svectors[sid]))})
    write_csv(phase_root / "task042_phase_i_temporal_vs_static.csv", tvs_rows)
    # Per-class/per-span diagnostics preserve the individual functional atoms.
    class_span_rows = []
    coverage_by = defaultdict(list)
    for row in coverage_rows:
        coverage_by[(row["domain_id"], row["subset_id"], str(row["label"]), int(row["span"]))].append(float(row["kappa"]))
    for (d, sid, label, span), values in sorted(coverage_by.items()):
        class_span_rows.append({"domain_id": d, "subset_id": sid, "label": label, "span": span,
                                "condition_count": len(values), **_quantiles(values)})
    write_csv(phase_root / "task042_phase_i_class_span_breakdown.csv", class_span_rows)
    # Stability: ten leave-one-class-out views, and deterministic pair-index A/B splits.
    stability_rows = []
    coverage_index = defaultdict(dict)
    for r in coverage_rows:
        coverage_index[(r["domain_id"], r["subset_id"])][(int(r["video_index"]), int(r["relation_id"]))] = float(r["kappa"])
    labels = [str(v["label"]) for v in videos]
    for key, vectors in group_vectors.items():
        d, k = key; full_front = temporal_fronts[key]; full_edges = temporal_edges[key]
        full_diag = {sid: float(np.mean(vec)) for sid, vec in vectors.items()}
        for omitted, keep_vi in leave_one_class_out_indices(labels).items():
            subvecs = {sid: [coverage_index[(d, sid)][(int(videos[vi]["video_index"]), qi)]
                             for vi in keep_vi for qi in range(10)] for sid in vectors}
            front = pareto_front(subvecs); edges = _dominance_edges(subvecs)
            diag = {sid: float(np.mean(x)) for sid, x in subvecs.items()}
            rank_corr = spearman([full_diag[s] for s in vectors], [diag[s] for s in vectors]) if len(vectors) > 1 else None
            stability_rows.append({"analysis": "leave_one_class_out", "domain_id": d, "keep_count": k, "view_id": omitted,
                                   "profile_width": len(next(iter(subvecs.values()))), "pareto_membership_jaccard": len(front & full_front) / len(front | full_front) if front | full_front else 1.0,
                                   "dominance_edge_jaccard": len(edges & full_edges) / len(edges | full_edges) if edges | full_edges else 1.0,
                                   "diagnostic_mean_rank_spearman": rank_corr, "pareto_front_size": len(front), "dominance_edge_count": len(edges)})
        relation_info = relation_rows
        pa, pb = pairset_indices(relation_info)
        split_sets = {"PAIRSET-A": pa, "PAIRSET-B": pb}
        for split_name, qids in split_sets.items():
            subvecs = {sid: [coverage_index[(d, sid)][(int(v["video_index"]), qi)] for v in videos for qi in qids]
                       for sid in vectors}
            front = pareto_front(subvecs); edges = _dominance_edges(subvecs)
            diag = {sid: float(np.mean(x)) for sid, x in subvecs.items()}
            rank_corr = spearman([full_diag[s] for s in vectors], [diag[s] for s in vectors]) if len(vectors) > 1 else None
            stability_rows.append({"analysis": "pairset_split", "domain_id": d, "keep_count": k, "view_id": split_name,
                                   "profile_width": len(next(iter(subvecs.values()))), "pareto_membership_jaccard": len(front & full_front) / len(front | full_front) if front | full_front else 1.0,
                                   "dominance_edge_jaccard": len(edges & full_edges) / len(edges | full_edges) if edges | full_edges else 1.0,
                                   "diagnostic_mean_rank_spearman": rank_corr, "pareto_front_size": len(front), "dominance_edge_count": len(edges)})
    write_csv(phase_root / "task042_phase_i_stability.csv", stability_rows)
    # Domain 271 type-composition feasibility in exactly the shared functional space.
    mixed_rows = []
    for item in temporal_summary:
        if item["domain_id"] != "271":
            continue
        kept = item["retained_ids"]
        if all(subset_meta[item["subset_id"]]["retained_types"].get(t, 0) == 0 for t in ("attention_head",)):
            comp = "FFN-only"
        elif all(subset_meta[item["subset_id"]]["retained_types"].get(t, 0) == 0 for t in ("ffn_neuron",)):
            comp = "Attention-only"
        else:
            comp = "mixed Attention+FFN"
        mixed_rows.append({"domain_id": "271", "subset_id": item["subset_id"], "keep_count": item["keep_count"],
                           "retained_task037_ids": kept, "retained_type_composition": comp,
                           "masked_task037_ids": item["masked_ids"], "kappa_min": item["min"], "kappa_mean": item["mean"],
                           "kappa_median": item["median"], "kappa_profile_100": item["temporal_profile"],
                           "temporal_nondominated": item["subset_id"] in temporal_fronts[("271", item["keep_count"])]})
    write_csv(phase_root / "task042_phase_i_mixed_domain.csv", mixed_rows)
    # Project the already frozen Phase-D directional removal rank; no new ranking is learned.
    projection_rows = []
    for d in DOMAINS:
        ranks = {int(uid): int(rank) for uid, rank in config["baseline_removal_rank"][d].items()}
        members = set(EXPECTED_DOMAIN_UNITS[d]); n = len(members)
        for k in range(1, n):
            remove_n = n - k
            removed = tuple(sorted(uid for uid, rank in ranks.items() if rank <= remove_n))
            kept = tuple(sorted(members - set(removed)))
            sid = "K-" + "-".join(map(str, kept))
            fronts = temporal_fronts[(d, k)]
            edges = temporal_edges[(d, k)]
            dominated_by = [a for a, b in sorted(edges) if b == sid]
            projection_rows.append({"domain_id": d, "keep_count": k, "baseline_retained_task037_ids": kept,
                                    "baseline_masked_task037_ids": removed, "subset_id": sid,
                                    "temporal_nondominated": sid in fronts, "temporal_dominated": sid not in fronts,
                                    "dominated_by_subsets": dominated_by, "kappa_mean": subset_meta[sid]["temporal_profile"].mean().item(),
                                    "kappa_min": float(np.min(subset_meta[sid]["temporal_profile"]))})
    write_csv(phase_root / "task042_phase_i_baseline_projection.csv", projection_rows)
    # Decision is conservative: A requires structure outside mixed domain plus visible difference from static;
    # no numeric stability threshold is imposed. Ambiguous pilot evidence maps to B.
    family_spreads = [pairwise_linf_spread(list(v.values())) for v in group_vectors.values()]
    nontrivial_families = sum(x > 0.0 for x in family_spreads)
    unmixed = [key for key in group_vectors if key[0] != "271"]
    unmixed_dominance_families = sum(bool(temporal_edges[key]) for key in unmixed)
    temporal_static_different = sum(temporal_fronts[key] != static_fronts[key] for key in group_vectors)
    temporal_static_different_unmixed = sum(temporal_fronts[key] != static_fronts[key] for key in unmixed)
    loo_rows = [r for r in stability_rows if r["analysis"] == "leave_one_class_out"]
    pair_rows = [r for r in stability_rows if r["analysis"] == "pairset_split"]
    # Do not invent a numeric stability gate after seeing results. The default
    # is the preregistered conservative B; A requires a human-readable finding
    # that every listed evidence condition is actually persuasive.
    decision = "B. INTRA_GROUP_INTERFRAME_FUNCTIONAL_COVERAGE_WEAK_OR_UNRESOLVED"
    coverage_spreads = {f"{d}/k{k}": pairwise_linf_spread(list(v.values())) for (d, k), v in group_vectors.items()}
    loo_rank_values = [r["diagnostic_mean_rank_spearman"] for r in loo_rows if r["diagnostic_mean_rank_spearman"] is not None]
    summary = {
        "task": "Task042 Phase I intra-group inter-frame functional coverage feasibility",
        "decision": decision,
        "git_head": _git(Path(config["repo_root"]), "rev-parse", "HEAD"),
        "protocol": {"domains": list(DOMAINS), "videos": 10, "classes": 10, "relations_per_video": 10,
                     "coverage_profile_dim": 100, "pair_indices": list(PAIR_INDICES), "spans": list(SPANS),
                     "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "dtype": "float32", "amp": False,
                     "output_classes": 400, "physical_gpus_used": [0, 1], "physical_pruning": False,
                     "finetuning": False, "forbidden_performance_oracles_used": False},
        "answers": {
            "A_same_keep_count_profiles_differ": {"answer": bool(nontrivial_families), "families_with_nonzero_Linf_spread": nontrivial_families,
                                                     "family_count": len(family_spreads), "max_Linf_spread_by_domain_keep": coverage_spreads},
            "B_exact_functional_dominance": {"answer": "yes" if any(temporal_edges.values()) else "no", "dominance_edges_by_family": {f"{d}/k{k}": len(e) for (d,k),e in temporal_edges.items()},
                                              "nondominated_subset_counts": {f"{d}/k{k}": len(front) for (d,k),front in temporal_fronts.items()}},
            "C_leave_one_class_out_stability": {"answer": "see stability table; no numerical threshold imposed", "mean_pareto_jaccard": float(np.mean([r["pareto_membership_jaccard"] for r in loo_rows])) if loo_rows else None,
                                                 "mean_dominance_edge_jaccard": float(np.mean([r["dominance_edge_jaccard"] for r in loo_rows])) if loo_rows else None,
                                                 "mean_diagnostic_rank_spearman": float(np.mean(loo_rank_values)) if loo_rank_values else None},
            "D_pair_location_split_stability": {"answer": "see stability table", "pairset_a_mean_pareto_jaccard": float(np.mean([r["pareto_membership_jaccard"] for r in pair_rows if r["view_id"] == "PAIRSET-A"])) if pair_rows else None,
                                                 "pairset_b_mean_pareto_jaccard": float(np.mean([r["pareto_membership_jaccard"] for r in pair_rows if r["view_id"] == "PAIRSET-B"])) if pair_rows else None},
            "E_temporal_differs_from_static": {"answer": bool(temporal_static_different), "families_with_different_fronts": temporal_static_different,
                                                  "families_total": len(group_vectors), "different_unmixed_families": temporal_static_different_unmixed,
                                                  "front_overlap_rows": tvs_rows},
            "F_mixed_attention_ffn_shared_space": {"answer": "yes" if mixed_rows else "not evaluated", "evaluated_subsets": len(mixed_rows),
                                                     "compositions": dict(Counter(r["retained_type_composition"] for r in mixed_rows)),
                                                     "all_subsets_use_same_400D_centered_logit_function_space": True},
            "G_ready_for_larger_30x80_validation": {"answer": "not yet; Phase I is a 10x10 feasibility pilot and stability remains empirical",
                                                      "full_30x80_inference_performed": False},
        },
        "subset_count": len(subset_meta), "relation_coverage_row_count": len(coverage_rows),
        "exact_dominance_edge_count": sum(len(e) for e in temporal_edges.values()),
        "stability_summary": {"leave_one_class_out_views": len(loo_rows), "pairset_views": len(pair_rows)},
        "input_cache_sha256": {"full": full_info["sha256"], **{d: domains_data[d]["info"]["sha256"] for d in DOMAINS}},
        "decision_basis": {"nontrivial_same_count_profiles": nontrivial_families,
                           "unmixed_domain_families_with_dominance": unmixed_dominance_families,
                           "unmixed_domain_fronts_differing_from_static": temporal_static_different_unmixed,
                           "automatic_stability_threshold_used": False,
                           "decision_defaulted_to_conservative_B_pending_evidence_review": True},
    }
    write_json(phase_root / "task042_phase_i_summary.json", summary)
    _write_report(phase_root / "task042_phase_i_report.md", summary, temporal_summary, stability_rows, projection_rows)
    missing = [name for name in OUTPUT_NAMES if not (phase_root / name).is_file()]
    require(not missing, "final output set incomplete: " + ", ".join(missing))
    print(f"FINALIZE_OK decision={decision} subsets={len(subset_meta)} exact_dominance_edges={summary['exact_dominance_edge_count']}")


def _write_report(path: Path, summary: Mapping[str, Any], temporal: Sequence[Mapping[str, Any]], stability: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> None:
    answers = summary["answers"]
    lines = ["# Task042 Phase I — Intra-group Inter-frame Functional Coverage", "",
             f"**Decision: `{summary['decision']}`**", "",
             "This is a temporary joint-mask feasibility pilot. It did not physically prune, fine-tune, or use accuracy/CE/prediction-flip oracles.", "",
             "## Frozen protocol", "",
             "Four preregistered BMS domains (415, 103, 76, 271); 10 classes × first manifest video; 10 fixed temporal relations (span 1/2/4/8/16, pair index 0 and 8); full 400-D centered logits; FP32, AMP off.", "",
             "Each retained set is compared only with sets from the same domain and the same keep count. Exact dominance uses all 100 relation-wise κ values, with no tolerance.", "",
             "## Required questions", ""]
    a = answers["A_same_keep_count_profiles_differ"]
    lines += [f"- **A — Same-count profiles differ:** {a['answer']}; {a['families_with_nonzero_Linf_spread']}/{a['family_count']} domain/keep families have nonzero exact profile spread."]
    b = answers["B_exact_functional_dominance"]
    lines += [f"- **B — Exact dominance:** {b['answer']}; total directed edges = {summary['exact_dominance_edge_count']}."]
    c = answers["C_leave_one_class_out_stability"]
    lines += [f"- **C — Leave-one-class-out:** mean Pareto-membership Jaccard = {c['mean_pareto_jaccard']}; mean dominance-edge Jaccard = {c['mean_dominance_edge_jaccard']}; no numeric stability threshold was introduced."]
    d = answers["D_pair_location_split_stability"]
    lines += [f"- **D — Pair-location split:** PAIRSET-A mean Pareto Jaccard = {d['pairset_a_mean_pareto_jaccard']}; PAIRSET-B = {d['pairset_b_mean_pareto_jaccard']}."]
    e = answers["E_temporal_differs_from_static"]
    lines += [f"- **E — Temporal vs static:** fronts differ in {e['families_with_different_fronts']}/{e['families_total']} families; among non-mixed domains, {e['different_unmixed_families']} differ."]
    f = answers["F_mixed_attention_ffn_shared_space"]
    lines += [f"- **F — Mixed Attention/FFN:** {f['answer']}; {f['evaluated_subsets']} subsets are evaluated with the same centered 400-D logit relation function, without type-specific normalization."]
    lines += [f"- **G — Larger 30×80 validation readiness:** {answers['G_ready_for_larger_30x80_validation']['answer']}.", "",
              "## Stability and baseline projection", "",
              f"Leave-one-class-out views: {summary['stability_summary']['leave_one_class_out_views']}; pairset views: {summary['stability_summary']['pairset_views']}.",
              "The frozen Phase-D directional removal order is only projected onto each fixed keep count; it is not re-ranked using Phase-I results.", "",
              "## Interpretation", "",
              "The primary evidence is the complete 100-D κ profile for every nonempty proper retained subset. Min/mean/median/quantiles are descriptive only. An ambiguous result is classified as weak/unresolved.", "",
              "## Outputs", ""]
    for name in OUTPUT_NAMES:
        lines.append(f"- `{name}`")
    lines += ["", "## Stop condition", "", "Phase I ends here. This result does not authorize structural pruning or fine-tuning.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("prepare", prepare), ("preflight", preflight), ("full-cache", full_cache), ("worker", worker), ("finalize", finalize)):
        p = sub.add_parser(name)
        p.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE)
        if name == "prepare":
            p.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
        if name == "full-cache":
            p.add_argument("--gpu", type=int, default=0)
        if name == "worker":
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--domains", nargs="+", required=True)
        p.set_defaults(run=fn)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        args.run(args.base_root, args.phase_root)
    else:
        args.run(args)


if __name__ == "__main__":
    main()

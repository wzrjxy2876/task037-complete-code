#!/usr/bin/env python3
"""Task042: diagnose frame-pair relation redundancy inside frozen BMS groups.

This module does not score, mask, prune, or train units.  It records the
relative change in each frozen unit's own activation under the 80 exact
two-sampled-frame swaps defined by Task040's fixed-cardinality enumerator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BRANCH = "task_042_post_bms_frame_relation_redundancy"
TASK041_BRANCH = "task_041_collective_temporal_coverage_pruning_oracle"
TASK041_HEAD = "dc90da61a15ecb76231cf7e468681b59e501627a"
CHECKPOINT_SHA256 = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
T = 32
SPANS = (1, 2, 4, 8, 16)
PAIRS_PER_SPAN = 16
EPS = 1e-12
PROFILE_DEFAULT = Path("/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis/d1_targeted_domains/task040_temporal_profiles.csv")
DESCRIPTORS_DEFAULT = Path("/home/jixinye25/jxy_work1/swintrans_task037/task014_n09/dynamic3d/seed3407/descriptor_statistics.csv")
UNIT_MAPPING_DEFAULT = Path("/home/jixinye25/jxy_work1/swintrans_task037/task014_n09/contribution_unit_mapping.csv")
VIDEO_MANIFEST_DEFAULT = Path("/data/jixinye25/work1/output/task041_collective_temporal_coverage_pruning_oracle/task041_video_manifest.csv")
VAL_LIST_DEFAULT = Path("/data/jixinye25/UCF101_Frame/val_rgb_split1.txt")
FRAME_ROOT_DEFAULT = Path("/data/jixinye25/UCF101_Frame/frames")
PROJECT_DEFAULT = Path("/home/jixinye25/jxy_work1/swintrans_task035")
CHECKPOINT_DEFAULT = Path("/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
OUTPUT_DEFAULT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
REPO_DEFAULT = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
TASK040_OUTPUT = Path("/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis")
TASK041_PHASE_I_OUTPUT = Path("/data/jixinye25/work1/output/task041_phase_i_class_diversity")

UNIT_FIELDS = (
    "task037_global_index", "layer", "unit_type", "unit_index", "stage",
    "domain_id", "capture_kind", "D_abs", "D_rel", "D_third", "D_st",
    "third_descriptor_name", "D_dyn", "task040_unit_global_index",
)
VIDEO_FIELDS = ("video_index", "dataset_index", "video_id", "duration", "label")
SENSITIVITY_FIELDS = (
    "video_index", "dataset_index", "video_id", "label", "task037_global_index",
    "layer", "unit_type", "unit_index", "stage", "domain_id", "capture_kind",
    "span", "pair_index", "frame_a", "frame_b", "baseline_norm", "delta_norm",
    "relative_sensitivity", "normalized_signature_z", "activation_shape",
    "physical_gpu",
)
SUMMARY_FIELDS = (
    "video_index", "dataset_index", "video_id", "label", "task037_global_index",
    "layer", "unit_type", "unit_index", "stage", "domain_id", "capture_kind",
    "condition_count", "mean_e", "median_e", "std_e", "min_e", "max_e",
    "mean_baseline_norm", "mean_delta_norm", "degenerate", "mean_z", "std_z",
    "median_abs_z",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Task042 gate failed: " + message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def norm_int(x: Any) -> int:
    return int(float(str(x).strip()))


def unit_type_alias(value: str) -> Tuple[str, str]:
    """Return (authoritative Task037 type, probe kind) without changing identity."""
    value = str(value).strip()
    aliases = {
        "head": ("attention_head", "head"),
        "attention_head": ("attention_head", "head"),
        "neuron": ("ffn_neuron", "neuron"),
        "ffn_neuron": ("ffn_neuron", "neuron"),
    }
    require(value in aliases, "unknown unit type: " + value)
    return aliases[value]


def _stage_from_layer(layer: str) -> int:
    parts = str(layer).split(".")
    require(len(parts) >= 2 and parts[0] == "layers", "unexpected Swin layer path: " + layer)
    return int(parts[1])


def _has_commit_parent_in_shallow_history(repo_root: Path, target_sha: str) -> bool:
    """Follow available commit objects and match parent SHAs at shallow edges."""
    pending = ["HEAD"]
    visited = set()
    while pending:
        ref = pending.pop()
        try:
            record = subprocess.check_output(["git", "-C", str(repo_root), "show", "-s",
                                              "--format=%H %P", ref], text=True).strip().split()
        except subprocess.CalledProcessError:
            continue
        if not record:
            continue
        commit, parents = record[0], record[1:]
        if commit in visited:
            continue
        visited.add(commit)
        if target_sha in parents:
            return True
        for parent in parents:
            if parent not in visited and subprocess.call(["git", "-C", str(repo_root), "cat-file",
                                                           "-e", parent + "^{commit}"],
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.DEVNULL) == 0:
                pending.append(parent)
    return False


def build_unit_manifest(profile_rows: Sequence[Mapping[str, Any]],
                        descriptor_rows: Sequence[Mapping[str, Any]],
                        unit_mapping_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Exact join of the 51 frozen Task040 identities to Task037 source rows."""
    require(len(profile_rows) == 51, "Task040 Phase-D.1 cohort must contain exactly 51 units")
    profile_by_id: Dict[int, Mapping[str, Any]] = {}
    for row in profile_rows:
        uid = norm_int(row["task037_global_index"])
        require(uid not in profile_by_id, "duplicate Task037 global_index in frozen profile")
        profile_by_id[uid] = row
    require(len(profile_by_id) == 51, "frozen profile does not have 51 unique Task037 identities")
    desc_by_id: Dict[int, Mapping[str, Any]] = {}
    for row in descriptor_rows:
        uid = norm_int(row["global_index"])
        require(uid not in desc_by_id, "duplicate Task037 global_index in descriptor table")
        desc_by_id[uid] = row
    mapping_by_id: Dict[int, Mapping[str, Any]] = {}
    for row in unit_mapping_rows:
        uid = norm_int(row["global_index"])
        require(uid not in mapping_by_id, "duplicate Task037 global_index in unit mapping")
        mapping_by_id[uid] = row

    result: List[Dict[str, Any]] = []
    for uid in sorted(profile_by_id):
        p = profile_by_id[uid]
        require(uid in desc_by_id and uid in mapping_by_id,
                "missing Task037 descriptor or unit-map row for global_index=" + str(uid))
        d, m = desc_by_id[uid], mapping_by_id[uid]
        layer = str(p["layer_name"])
        canonical_type, capture_kind = unit_type_alias(str(p["unit_type"]))
        expected = (layer, canonical_type, norm_int(p["unit_index"]))
        observed_desc = (str(d["layer"]), unit_type_alias(str(d["unit_type"]))[0], norm_int(d["unit_index"]))
        observed_map = (str(m["layer"]), unit_type_alias(str(m["unit_type"]))[0], norm_int(m["unit_index"]))
        require(expected == observed_desc, "descriptor identity mismatch for global_index=" + str(uid))
        require(expected == observed_map, "unit mapping identity mismatch for global_index=" + str(uid))
        stage = norm_int(p["stage"])
        require(stage == _stage_from_layer(layer), "frozen stage does not agree with Swin layer path for " + str(uid))
        row = {
            "task037_global_index": uid,
            "layer": layer,
            "unit_type": canonical_type,
            "unit_index": norm_int(p["unit_index"]),
            "stage": stage,
            "domain_id": str(p["domain_id"]),
            "capture_kind": capture_kind,
            "D_abs": float(d["D_abs"]),
            "D_rel": float(d["D_rel"]),
            "D_third": float(d["D_third"]),
            # Task041 terminology for the frozen third axis. The source value
            # and its original D_dyn label are preserved alongside this alias.
            "D_st": float(d["D_third"]),
            "third_descriptor_name": str(d.get("third_descriptor_name", "")),
            "D_dyn": float(d["D_dyn"]),
            "task040_unit_global_index": norm_int(p["unit_global_index"]),
        }
        require(str(d.get("third_descriptor_name", "")) == "D_dyn" and
                math.isclose(row["D_third"], row["D_dyn"], rel_tol=0.0, abs_tol=0.0),
                "Task037 third descriptor provenance changed")
        result.append(row)
    return result


def validate_video_manifest(rows: Sequence[Mapping[str, Any]], val_lines: Optional[Sequence[str]] = None,
                            require_paths: bool = True) -> List[Dict[str, Any]]:
    require(len(rows) == 30, "frozen Task041 manifest must contain exactly 30 videos")
    result = []
    by_video_index = set()
    by_video_id = set()
    counts: Counter = Counter()
    for row in rows:
        video_index = norm_int(row["video_index"])
        dataset_index = norm_int(row["dataset_index"])
        label = norm_int(row["label"])
        video_id = str(row["video_id"])
        require(video_index not in by_video_index and video_id not in by_video_id,
                "duplicate video index or identity in frozen Task041 manifest")
        by_video_index.add(video_index)
        by_video_id.add(video_id)
        counts[label] += 1
        if val_lines is not None:
            require(0 <= dataset_index < len(val_lines), "video dataset_index outside frozen validation list")
            line = val_lines[dataset_index].split()
            require(len(line) >= 3 and Path(line[0]).name == Path(video_id).name
                    and norm_int(line[-1]) == label,
                    "video manifest does not match exact validation-list identity")
        if require_paths:
            require(Path(video_id).is_dir(), "frozen video frame directory is missing: " + video_id)
        result.append({
            "video_index": video_index,
            "dataset_index": dataset_index,
            "video_id": video_id,
            "duration": norm_int(row["duration"]),
            "label": label,
        })
    require(len(counts) == 10 and set(counts.values()) == {3},
            "frozen Task041 manifest must be 10 classes x 3 videos")
    return sorted(result, key=lambda r: r["video_index"])


def deterministic_video_shards(video_rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[int]]:
    """Two disjoint deterministic video shards, interleaved by frozen video_index."""
    shards = {0: [], 1: []}
    for row in video_rows:
        idx = norm_int(row["video_index"])
        shards[idx % 2].append(idx)
    require(len(shards[0]) == 15 and len(shards[1]) == 15, "two-GPU video shard sizes must be 15/15")
    require(set(shards[0]).isdisjoint(shards[1]) and set(shards[0] + shards[1]) ==
            {norm_int(r["video_index"]) for r in video_rows}, "two-GPU video shard merge is not identity-preserving")
    return shards


def fixed_cardinality_identities(core: Any, temporal_length: int = T) -> List[Dict[str, int]]:
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(temporal_length)
    require(temporal_length == T and len(interventions) == 80, "Task040 enumerator must yield 80 T=32 interventions")
    rows = []
    seen = set()
    counts: Counter = Counter()
    for item in interventions:
        span = norm_int(item.block_size)
        pair = norm_int(item.pair_index)
        a, b = norm_int(item.left_start), norm_int(item.right_start)
        require(span in SPANS and 0 <= pair < PAIRS_PER_SPAN, "unexpected Task040 span/pair identity")
        require(norm_int(item.left_end) == a + 1 and norm_int(item.right_end) == b + 1,
                "intervention must exchange single sampled positions")
        perm = tuple(norm_int(x) for x in item.permutation)
        changed = [i for i, source in enumerate(perm) if i != source]
        require(len(changed) == 2 and set(changed) == {a, b} and perm[a] == b and perm[b] == a,
                "intervention must be an exact two-position swap")
        key = (span, pair)
        require(key not in seen, "duplicate Task040 span/pair identity")
        seen.add(key)
        counts[span] += 1
        rows.append({"span": span, "pair_index": pair, "frame_a": a, "frame_b": b})
    require(len(seen) == 80 and all(counts[s] == 16 for s in SPANS),
            "expected 16 unique frame-pair interventions for every frozen span")
    return rows


def normalize_signature(values: Sequence[float], eps: float = EPS) -> Tuple[Optional[List[float]], float, float, bool]:
    vals = [float(x) for x in values]
    require(len(vals) > 0 and all(math.isfinite(x) for x in vals), "signature must be finite and nonempty")
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(max(0.0, var))
    if std <= eps:
        return None, mean, std, True
    return [(x - mean) / std for x in vals], mean, std, False


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [float(x) - mx for x in xs]
    dy = [float(y) - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 0.0 or vy <= 0.0:
        return None
    value = sum(x * y for x, y in zip(dx, dy)) / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, value))


def relation_distance(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) == len(ys) and len(xs) > 1 and all(float(x) == float(y) for x, y in zip(xs, ys)):
        return 0.0
    rho = pearson_corr(xs, ys)
    return None if rho is None else (1.0 - rho) / 2.0


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: float(values[i]))
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(values[order[j]]) == float(values[order[i]]):
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_corr(rankdata(xs), rankdata(ys))


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = (float(xs[i]) > float(xs[j])) - (float(xs[i]) < float(xs[j]))
            dy = (float(ys[i]) > float(ys[j])) - (float(ys[i]) < float(ys[j]))
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    return None if denom == 0 else (concordant - discordant) / denom


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "std": None}
    def q(p: float) -> float:
        pos = (len(vals) - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    mean = sum(vals) / len(vals)
    return {"count": len(vals), "mean": mean, "median": statistics.median(vals),
            "q25": q(0.25), "q75": q(0.75),
            "std": math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))}


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    sb = sorted(float(x) for x in b)
    less = greater = 0
    import bisect
    for x in a:
        less += bisect.bisect_left(sb, float(x))
        greater += len(sb) - bisect.bisect_right(sb, float(x))
    return (greater - less) / float(len(a) * len(b))


def nearest_neighbor(domain_unit_ids: Sequence[int],
                     distances: Mapping[Tuple[int, int], Optional[float]],
                     current_uid: int) -> Optional[Tuple[int, float]]:
    choices = []
    for other in domain_unit_ids:
        if int(other) == int(current_uid):
            continue
        key = (min(int(current_uid), int(other)), max(int(current_uid), int(other)))
        value = distances.get(key)
        if value is not None and math.isfinite(float(value)):
            choices.append((float(value), int(other)))
    if not choices:
        return None
    distance, uid = min(choices, key=lambda item: (item[0], item[1]))
    return uid, distance


class ActivationCapture:
    """Read-only hooks for pre-projection attention heads and post-GELU neurons."""
    def __init__(self, model: Any, units: Sequence[Mapping[str, Any]], torch: Any):
        self.model, self.units, self.torch = model, list(units), torch
        self.by_layer: Dict[str, Dict[str, Any]] = {}
        for row in self.units:
            layer = str(row["layer"])
            self.by_layer.setdefault(layer, {"heads": [], "neurons": []})
            self.by_layer[layer]["heads" if row["capture_kind"] == "head" else "neurons"].append(row)
        self.values: Dict[int, Any] = {}
        self.shapes: Dict[int, Tuple[int, ...]] = {}
        self.structure: Dict[str, Dict[str, Any]] = {}
        self.handles = []
        modules = dict(model.named_modules())
        for layer, spec in self.by_layer.items():
            require(layer in modules, "frozen probe layer is absent from model: " + layer)
            module = modules[layer]
            if spec["heads"]:
                self._attach_attention(layer, module, spec["heads"])
            if spec["neurons"]:
                self._attach_ffn(layer, module, spec["neurons"])

    def clear(self) -> None:
        self.values.clear()
        self.shapes.clear()
        self.structure.clear()

    def _capture(self, uid: int, tensor: Any) -> None:
        require(hasattr(tensor, "detach"), "captured activation is not a tensor")
        value = tensor.detach().to(dtype=self.torch.float32).contiguous()
        self.shapes[int(uid)] = tuple(int(d) for d in value.shape)
        # Keep unit activations on the worker GPU. The worker reduces all 51
        # norms into one compact transfer per intervention instead of copying
        # dozens of activation tensors back to the host.
        self.values[int(uid)] = value

    def _attach_attention(self, layer: str, module: Any, target_rows: Sequence[Mapping[str, Any]]) -> None:
        require(hasattr(module, "qkv") and hasattr(module, "attn_drop") and hasattr(module, "num_heads"),
                "attention module lacks qkv/attn_drop/num_heads: " + layer)
        heads = int(module.num_heads)
        qkv_cache: Dict[str, Any] = {}
        head_indices = {norm_int(r["unit_index"]): r for r in target_rows}
        require(all(0 <= h < heads for h in head_indices), "frozen attention head index exceeds module width")
        def qkv_hook(_module: Any, _inputs: Any, output: Any) -> None:
            require(output.ndim == 3 and int(output.shape[-1]) % (3 * heads) == 0,
                    "unexpected qkv projection shape in " + layer)
            head_dim = int(output.shape[-1]) // (3 * heads)
            nwin, tokens = int(output.shape[0]), int(output.shape[1])
            qkv_cache["v"] = output.reshape(nwin, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)[2]
            qkv_cache["shape"] = (nwin, heads, tokens, head_dim)
            self.structure.setdefault(layer, {}).update({
                "num_heads": heads, "head_dim": head_dim, "window_count": nwin,
                "token_count": tokens,
                "qkv_projection_shape": tuple(int(d) for d in output.shape),
            })
        def weights_hook(_module: Any, _inputs: Any, output: Any) -> None:
            require("v" in qkv_cache, "attention weights arrived without matching V projection in " + layer)
            v = qkv_cache["v"]
            require(output.ndim == 4 and int(output.shape[1]) == heads and
                    tuple(output.shape[-2:]) == (int(v.shape[-2]), int(v.shape[-2])),
                    "unexpected attention-weight tensor shape in " + layer)
            # model.eval() makes attention dropout the identity; this is the exact
            # attention_weights @ V result before head transpose/concatenation.
            per_head = self.torch.matmul(output, v)
            require(tuple(per_head.shape) == qkv_cache["shape"],
                    "pre-projection head output shape differs from qkv-derived shape")
            self.structure.setdefault(layer, {}).update({
                "attention_weights_shape": tuple(int(d) for d in output.shape),
                "head_output_shape": tuple(int(d) for d in per_head.shape),
            })
            for h, row in head_indices.items():
                uid = norm_int(row["task037_global_index"])
                self._capture(uid, per_head[:, h, :, :])
        self.handles.append(module.qkv.register_forward_hook(qkv_hook))
        self.handles.append(module.attn_drop.register_forward_hook(weights_hook))

    def _attach_ffn(self, layer: str, module: Any, target_rows: Sequence[Mapping[str, Any]]) -> None:
        require(hasattr(module, "fc1") and hasattr(module, "act") and hasattr(module, "fc2"),
                "FFN module lacks fc1/act/fc2: " + layer)
        hidden = int(module.fc1.out_features)
        neuron_indices = {norm_int(r["unit_index"]): r for r in target_rows}
        require(all(0 <= i < hidden for i in neuron_indices), "frozen FFN neuron index exceeds hidden width")
        def act_hook(_module: Any, _inputs: Any, output: Any) -> None:
            require(output.ndim == 3 and int(output.shape[-1]) == hidden,
                    "unexpected post-GELU FFN tensor shape in " + layer)
            self.structure.setdefault(layer, {}).update({
                "hidden_width": hidden, "window_count": int(output.shape[0]),
                "token_count": int(output.shape[-2]),
                "ffn_activation_shape": tuple(int(d) for d in output.shape),
            })
            for i, row in neuron_indices.items():
                uid = norm_int(row["task037_global_index"])
                self._capture(uid, output[..., i])
        self.handles.append(module.act.register_forward_hook(act_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _spearman_mann_whitney(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)
    except Exception:
        return None


def _runtime_modules(repo_root: Path) -> Tuple[Any, Any, Any]:
    runtime = str(repo_root / "src" / "lgfr_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task040_htor_core as core
    import task040_htor_probe as probe
    import probe_ctfrs_dynamic_function as ctfrs
    return core, probe, ctfrs


def _model_and_identity(project_root: Path, checkpoint: Path, device: Any,
                        probe: Any, ctfrs: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    ctfrs.ensure_project_importable(project_root)
    probe.ensure_project_importable(project_root)
    adapter = importlib.import_module("ucf101_videoswin_probe_adapter_v2")
    model, adapter_meta = adapter.build_model_for_probe(checkpoint=str(checkpoint), device=device)
    model.eval()
    model.requires_grad_(False)
    specs = ctfrs.discover_unit_layers(model)
    identity = probe.make_checkpoint_identity(model, checkpoint, adapter_meta, specs)
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "loaded checkpoint SHA differs from frozen Task040/041 checkpoint")
    require(not identity.get("missing_keys") and not identity.get("unexpected_keys")
            and not identity.get("shape_mismatches"), "checkpoint state does not load exactly")
    require(identity.get("classifier_head", {}).get("status") == "loaded",
            "authoritative 400-class classifier head is not loaded")
    return model, adapter, identity


def _make_exact_val_list(video_rows: Sequence[Mapping[str, Any]], val_list: Path,
                         output: Path) -> Dict[str, str]:
    lines = val_list.read_text(encoding="utf-8").splitlines()
    selected = []
    for row in video_rows:
        idx = norm_int(row["dataset_index"])
        require(0 <= idx < len(lines), "manifest dataset_index outside val list")
        line = lines[idx]
        fields = line.split()
        require(len(fields) >= 3 and Path(fields[0]).name == Path(str(row["video_id"])).name
                and norm_int(fields[-1]) == norm_int(row["label"]),
                "exact N=30 list does not reproduce frozen Task041 video identity")
        selected.append((idx, line))
    selected.sort()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(line for _, line in selected) + "\n", encoding="utf-8")
    return {Path(str(r["video_id"])).name: str(r["video_id"]) for r in video_rows}


def _get_loader(ctfrs: Any, config: Mapping[str, Any], workers: int) -> Any:
    loader, selected, _ = ctfrs.build_balanced_loader(
        project_root=Path(config["project_root"]),
        val_list=str(config["exact_video_list"]),
        frame_root=str(config["frame_root"]),
        num_classes=10, videos_per_class=3, num_workers=workers, seed=3407,
    )
    require(len(selected) == 30, "exact balanced loader did not retain all 30 frozen videos")
    return loader


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    require(not output.exists(), "Task042 output directory already exists; refusing overwrite")
    output.mkdir(parents=True)
    paths = {
        "profile": Path(args.profile), "descriptors": Path(args.descriptors),
        "unit_mapping": Path(args.unit_mapping), "video_manifest": Path(args.video_manifest),
        "val_list": Path(args.val_list), "checkpoint": Path(args.checkpoint),
    }
    for name, path in paths.items():
        require(path.is_file(), "required frozen input is absent: " + name + "=" + str(path))
    val_lines = paths["val_list"].read_text(encoding="utf-8").splitlines()
    units = build_unit_manifest(read_csv(paths["profile"]), read_csv(paths["descriptors"]),
                                read_csv(paths["unit_mapping"]))
    videos = validate_video_manifest(read_csv(paths["video_manifest"]), val_lines, require_paths=True)
    require(sha256_file(paths["checkpoint"]) == CHECKPOINT_SHA256,
            "checkpoint SHA does not match frozen Task040/041 artifact")
    write_csv(output / "task042_unit_manifest.csv", units, UNIT_FIELDS)
    write_csv(output / "task042_video_manifest.csv", videos, VIDEO_FIELDS)
    config = {
        "task": "Task042 post-BMS frame-to-frame relation redundancy diagnosis",
        "required_branch": BRANCH,
        "task041_base_branch": TASK041_BRANCH,
        "task041_base_head": TASK041_HEAD,
        "output_dir": str(output.resolve()),
        "repo_root": str(Path(args.repo_root).resolve()),
        "project_root": str(Path(args.project_root).resolve()),
        "checkpoint_path": str(paths["checkpoint"].resolve()),
        "frame_root": str(Path(args.frame_root).resolve()),
        "val_list": str(paths["val_list"].resolve()),
        "exact_video_list": str((output / "task042_exact_video_list.txt").resolve()),
        "unit_manifest": str((output / "task042_unit_manifest.csv").resolve()),
        "video_manifest": str((output / "task042_video_manifest.csv").resolve()),
        "profile_path": str(paths["profile"].resolve()),
        "descriptor_path": str(paths["descriptors"].resolve()),
        "unit_mapping_path": str(paths["unit_mapping"].resolve()),
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "unit_count": len(units), "video_count": len(videos),
        "domain_count": len(set(r["domain_id"] for r in units)),
        "multi_unit_domains": sorted([d for d, n in Counter(r["domain_id"] for r in units).items() if n >= 2], key=int),
        "forwards_expected": 30 * 81,
        "protocol": {"T": T, "spans": list(SPANS), "pairs_per_span": PAIRS_PER_SPAN,
                     "conditions_per_video": 80, "frames_swapped_per_condition": 2,
                     "dtype": "float32", "amp": False},
        "phase_i_context": {
            "decision": "C. CLASS_DIVERSITY_DOES_NOT_RESCUE_TEMPORAL_SELECTION",
            "phase_i_summary": str(TASK041_PHASE_I_OUTPUT / "task041_phase_i_summary.json"),
            "phase_i_report": str(TASK041_PHASE_I_OUTPUT / "task041_phase_i_report.md"),
        },
    }
    _make_exact_val_list(videos, paths["val_list"], output / "task042_exact_video_list.txt")
    write_json(output / "task042_run_config.json", config)
    print("PREPARE_OK units=51 videos=30 classes=10 domains=30 multi_domains=10 frames=32 conditions=80")


def preflight(args: argparse.Namespace) -> None:
    import torch
    output = Path(args.output_dir)
    config = json.loads((output / "task042_run_config.json").read_text(encoding="utf-8"))
    require(Path(config["repo_root"]).is_dir(), "Task042 checkout path is absent")
    branch = subprocess.check_output(["git", "-C", config["repo_root"], "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    require(branch == BRANCH, "server checkout is not on the exact Task042 branch")
    require(_has_commit_parent_in_shallow_history(Path(config["repo_root"]), TASK041_HEAD),
            "Task042 commit history does not descend from the latest Task041 remote HEAD")
    phase_i_doc = Path(config["repo_root"]) / "docs" / "tasks" / "task_041_phase_i_result.md"
    require(phase_i_doc.is_file() and "CLASS_DIVERSITY_DOES_NOT_RESCUE_TEMPORAL_SELECTION" in
            phase_i_doc.read_text(encoding="utf-8"),
            "completed Task041 Phase-I result is not present on the Task042 branch")
    for name, path in (("profile", config["profile_path"]), ("descriptors", config["descriptor_path"]),
                       ("unit_mapping", config["unit_mapping_path"]), ("video_manifest", config["video_manifest"]),
                       ("val_list", config["val_list"]), ("checkpoint", config["checkpoint_path"])):
        require(sha256_file(Path(path)) == config["input_sha256"][name], "frozen input SHA changed: " + name)
    units, videos = read_csv(Path(config["unit_manifest"])), read_csv(Path(config["video_manifest"]))
    require(len(units) == 51 and len(videos) == 30, "prepared manifests changed")
    core, probe, ctfrs = _runtime_modules(Path(config["repo_root"]))
    identities = fixed_cardinality_identities(core)
    require(len(identities) == 80, "Task040 intervention identity audit failed")
    model, adapter, identity = _model_and_identity(Path(config["project_root"]), Path(config["checkpoint_path"]),
                                                  torch.device("cpu"), probe, ctfrs)
    checkpoint_audit = json.loads((TASK040_OUTPUT / "n09_exact/task040_checkpoint_identity.json").read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "loaded_tensor_count", "loaded_parameter_count", "model_parameter_count",
                "discovered_pruning_layer_count", "discovered_attention_head_count", "discovered_ffn_neuron_count"):
        require(identity.get(key) == checkpoint_audit.get(key), "loaded model differs from Task040 checkpoint audit: " + key)
    modules = dict(model.named_modules())
    shapes = []
    for row in units:
        layer = str(row["layer"])
        require(layer in modules, "frozen Task037 layer missing from loaded model: " + layer)
        module = modules[layer]
        if row["capture_kind"] == "head":
            require(hasattr(module, "num_heads") and hasattr(module, "qkv") and hasattr(module, "attn_drop"),
                    "frozen attention layer is not the expected attention module: " + layer)
            width = int(module.qkv.out_features)
            heads = int(module.num_heads)
            require(width % (3 * heads) == 0, "attention projection width is not divisible into q/k/v heads")
            shapes.append({"task037_global_index": norm_int(row["task037_global_index"]), "layer": layer,
                           "unit_type": row["unit_type"], "unit_index": norm_int(row["unit_index"]),
                           "stage": norm_int(row["stage"]), "domain_id": row["domain_id"],
                           "capture_kind": "head", "num_heads": heads, "head_dim": width // (3 * heads),
                           "hidden_width": "", "qkv_projection_shape": ["batch_windows", "tokens", width],
                           "activation_shape": ["batch_windows", heads, "tokens", width // (3 * heads)],
                           "attention_drop_shape": ["batch_windows", heads, "tokens", "tokens"],
                           "baseline_intervention_shape_equal": "checked_during_gpu_worker"})
        else:
            require(hasattr(module, "fc1") and hasattr(module, "act") and hasattr(module, "fc2"),
                    "frozen FFN layer is not the expected MLP module: " + layer)
            width = int(module.fc1.out_features)
            require(norm_int(row["unit_index"]) < width, "frozen FFN index exceeds hidden width")
            shapes.append({"task037_global_index": norm_int(row["task037_global_index"]), "layer": layer,
                           "unit_type": row["unit_type"], "unit_index": norm_int(row["unit_index"]),
                           "stage": norm_int(row["stage"]), "domain_id": row["domain_id"],
                           "capture_kind": "neuron", "num_heads": "", "head_dim": "",
                           "hidden_width": width, "qkv_projection_shape": "",
                           "activation_shape": ["batch_windows", "tokens", width],
                           "attention_drop_shape": "", "baseline_intervention_shape_equal": "checked_during_gpu_worker"})
    write_csv(output / "task042_activation_shape_audit.csv", shapes,
              ("task037_global_index", "layer", "unit_type", "unit_index", "stage", "domain_id",
               "capture_kind", "num_heads", "head_dim", "hidden_width", "qkv_projection_shape",
               "activation_shape", "attention_drop_shape", "baseline_intervention_shape_equal"))
    loader = _get_loader(ctfrs, config, workers=0)
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    seen_videos = set()
    frozen_videos = {Path(str(r["video_id"])).name: r for r in videos}
    for batch in loader:
        require(int(batch[0].shape[0]) == 1 and int(batch[0].shape[2]) == T,
                "preflight loader did not return exactly one T=32 video")
        local = int(batch[2][0].item())
        name = Path(str(dataset.clips[local][0])).name
        require(name in frozen_videos and name not in seen_videos,
                "preflight loader produced an unexpected or repeated video")
        require(norm_int(batch[1][0].item()) == norm_int(frozen_videos[name]["label"]),
                "preflight loader label differs from frozen Task041 manifest")
        seen_videos.add(name)
    require(seen_videos == set(frozen_videos), "preflight did not load the exact 30 frozen videos")
    preflight_record = {
        "status": "passed", "branch": branch, "checkpoint_identity": identity,
        "unit_identity_count": len(units), "exact_video_count": len(videos),
        "action_classes": sorted(set(norm_int(r["label"]) for r in videos)),
        "intervention_count": len(identities), "intervention_spans": list(SPANS),
        "two_gpu_video_shards": {str(k): v for k, v in deterministic_video_shards(videos).items()},
        "activation_layers_audited": len(set(r["layer"] for r in units)),
        "activation_shape_semantics": "attention pre-projection per-head output and post-GELU pre-fc2 FFN neuron",
        "runtime_shapes_checked_in_first_actual_forward_per_worker": True,
        "gpu_inference_performed": False,
    }
    write_json(output / "task042_preflight.json", preflight_record)
    print("PREFLIGHT_OK checkpoint=exact unit_identities=51 intervention_identities=80 video_loader=exact")


def _load_model_on_gpu(config: Mapping[str, Any], gpu: int) -> Tuple[Any, Any, Any, Any, Any]:
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "CUDA_VISIBLE_DEVICES must isolate physical GPU " + str(gpu))
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "each worker must see exactly one isolated physical GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3407 + gpu)
    torch.cuda.manual_seed_all(3407 + gpu)
    device = torch.device("cuda:0")
    core, probe, ctfrs = _runtime_modules(Path(config["repo_root"]))
    model, adapter, identity = _model_and_identity(Path(config["project_root"]), Path(config["checkpoint_path"]),
                                                  device, probe, ctfrs)
    return torch, core, ctfrs, model, identity


def worker(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    output = Path(args.output_dir)
    config = json.loads((output / "task042_run_config.json").read_text(encoding="utf-8"))
    require((output / "task042_preflight.json").is_file(), "CPU preflight must pass before GPU workers")
    gpu = int(args.gpu)
    require(gpu in (0, 1), "only physical GPU 0 and GPU 1 are authorized")
    done = output / ("task042_gpu%d_done.json" % gpu)
    raw = output / ("task042_gpu%d_frame_pair_sensitivity.csv" % gpu)
    require(not done.exists() and not raw.exists(), "worker output already exists; refusing duplicate GPU inference")
    videos = read_csv(Path(config["video_manifest"]))
    shards = deterministic_video_shards(videos)
    assigned = set(shards[gpu])
    torch, core, ctfrs, model, identity = _load_model_on_gpu(config, gpu)
    units = read_csv(Path(config["unit_manifest"]))
    capture = ActivationCapture(model, units, torch)
    loader = _get_loader(ctfrs, config, workers=2)
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    video_by_name = {Path(r["video_id"]).name: r for r in videos}
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(T)
    intervention_rows = fixed_cardinality_identities(core)
    intervention_by_key = {(r["span"], r["pair_index"]): r for r in intervention_rows}
    expected_uids = {norm_int(r["task037_global_index"]) for r in units}
    seen = set()
    shape_audit_rows = []
    count_forwards = 0
    write_header = True
    with raw.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SENSITIVITY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for batch in tqdm(loader, desc="Task042 GPU%d videos" % gpu, ncols=100):
            videos_tensor, labels = batch[0], batch[1]
            local_index = int(batch[2][0].item())
            name = Path(str(dataset.clips[local_index][0])).name
            require(name in video_by_name, "balanced loader returned a video outside frozen Task041 manifest: " + name)
            video = video_by_name[name]
            video_index = norm_int(video["video_index"])
            require(norm_int(labels[0].item()) == norm_int(video["label"]), "loader label differs from frozen manifest")
            if video_index not in assigned:
                continue
            require(video_index not in seen, "worker saw a repeated video identity")
            require(int(videos_tensor.shape[0]) == 1 and int(videos_tensor.shape[2]) == T,
                    "loader must provide batch=1 and exactly T=32 sampled frames")
            clip = videos_tensor[0].to(device=torch.device("cuda:0"), dtype=torch.float32, non_blocking=True)
            require(str(clip.dtype) == "torch.float32", "Task042 inference must remain FP32")
            model.eval()
            model.requires_grad_(False)
            capture.clear()
            with torch.inference_mode():
                _ = model(clip.unsqueeze(0))
            baseline = {uid: value for uid, value in capture.values.items()}
            baseline_shapes = dict(capture.shapes)
            baseline_structure = {layer: dict(info) for layer, info in capture.structure.items()}
            require(set(baseline) == expected_uids and set(baseline_shapes) == expected_uids,
                    "baseline forward did not capture all 51 frozen units")
            count_forwards += 1
            rows_for_video: List[Dict[str, Any]] = []
            for intervention in interventions:
                span, pair = norm_int(intervention.block_size), norm_int(intervention.pair_index)
                exact_id = intervention_by_key[(span, pair)]
                swapped = core.apply_temporal_interventions(clip, [intervention], time_dim=1)[0]
                require(tuple(swapped.shape) == tuple(clip.shape), "frame swap changed model input shape")
                capture.clear()
                with torch.inference_mode():
                    _ = model(swapped.unsqueeze(0))
                require(set(capture.values) == expected_uids, "intervention forward did not capture all 51 frozen units")
                require(capture.shapes == baseline_shapes,
                        "baseline/intervention activation shape mismatch for video=" + name + " span=" + str(span))
                require(capture.structure == baseline_structure,
                        "baseline/intervention layer tensor semantics or token/window dimensions changed")
                count_forwards += 1
                metric_tensors = []
                for unit in units:
                    uid = norm_int(unit["task037_global_index"])
                    h0 = baseline[uid]
                    h1 = capture.values[uid]
                    require(tuple(h0.shape) == tuple(h1.shape), "unit activation tensor shape mismatch")
                    base_tensor = torch.norm(h0.reshape(-1), p=2)
                    delta_tensor = torch.norm((h1 - h0).reshape(-1), p=2)
                    value_tensor = delta_tensor / (base_tensor + EPS)
                    metric_tensors.append(torch.stack((base_tensor, delta_tensor, value_tensor)))
                metrics = torch.stack(metric_tensors).detach().cpu().tolist()
                for unit, metric in zip(units, metrics):
                    uid = norm_int(unit["task037_global_index"])
                    base_norm, delta_norm, value = [float(x) for x in metric]
                    require(math.isfinite(value), "non-finite frame-pair sensitivity")
                    rows_for_video.append({
                        "video_index": video_index, "dataset_index": norm_int(video["dataset_index"]),
                        "video_id": video["video_id"], "label": norm_int(video["label"]),
                        "task037_global_index": uid, "layer": unit["layer"], "unit_type": unit["unit_type"],
                        "unit_index": unit["unit_index"], "stage": unit["stage"], "domain_id": unit["domain_id"],
                        "capture_kind": unit["capture_kind"], "span": span, "pair_index": pair,
                        "frame_a": exact_id["frame_a"], "frame_b": exact_id["frame_b"],
                        "baseline_norm": base_norm, "delta_norm": delta_norm,
                        "relative_sensitivity": value, "normalized_signature_z": "",
                        "activation_shape": json.dumps(list(h0.shape), separators=(",", ":")),
                        "physical_gpu": gpu,
                    })
                if len(shape_audit_rows) < len(units):
                    for unit in units:
                        uid = norm_int(unit["task037_global_index"])
                        structure = baseline_structure[str(unit["layer"])]
                        is_head = unit["capture_kind"] == "head"
                        shape_audit_rows.append({"task037_global_index": uid, "layer": unit["layer"],
                            "unit_type": unit["unit_type"], "unit_index": unit["unit_index"],
                            "stage": unit["stage"], "domain_id": unit["domain_id"],
                            "capture_kind": unit["capture_kind"],
                            "num_heads": structure.get("num_heads", ""),
                            "head_dim": structure.get("head_dim", ""),
                            "hidden_width": structure.get("hidden_width", ""),
                            "window_count": structure.get("window_count", ""),
                            "token_count": structure.get("token_count", ""),
                            "activation_tensor_shape": json.dumps(list(baseline_shapes[uid]), separators=(",", ":")),
                            "layer_capture_tensor_shape": json.dumps(list(structure["head_output_shape" if is_head else "ffn_activation_shape"]), separators=(",", ":")),
                            "qkv_projection_shape": json.dumps(list(structure["qkv_projection_shape"]), separators=(",", ":")) if is_head else "",
                            "attention_weights_shape": json.dumps(list(structure["attention_weights_shape"]), separators=(",", ":")) if is_head else "",
                            "baseline_intervention_shape_equal": True, "shape_audit_video_index": video_index,
                            "physical_gpu": gpu})
            require(len(rows_for_video) == 80 * 51, "video did not produce exactly 80x51 unit-condition records")
            # Deterministic order by frozen unit identity, then span and pair index.
            rows_for_video.sort(key=lambda r: (r["task037_global_index"], r["span"], r["pair_index"]))
            writer.writerows(rows_for_video)
            f.flush()
            seen.add(video_index)
    capture.close()
    require(seen == assigned, "worker video shard incomplete: expected=%s observed=%s" % (sorted(assigned), sorted(seen)))
    expected_forwards = len(assigned) * 81
    require(count_forwards == expected_forwards, "worker forward count differs from 1+80 per video")
    write_csv(output / ("task042_gpu%d_activation_shape_audit.csv" % gpu), shape_audit_rows,
              ("task037_global_index", "layer", "unit_type", "unit_index", "stage", "domain_id", "capture_kind",
               "num_heads", "head_dim", "hidden_width", "window_count", "token_count", "activation_tensor_shape",
               "layer_capture_tensor_shape", "qkv_projection_shape", "attention_weights_shape",
               "baseline_intervention_shape_equal", "shape_audit_video_index", "physical_gpu"))
    write_json(done, {"worker": gpu, "physical_gpu": gpu, "video_indices": sorted(seen),
                      "unit_count": len(units), "row_count": len(assigned) * 80 * 51,
                      "forward_count": count_forwards, "dtype": "float32", "amp": False,
                      "checkpoint_sha256": identity["checkpoint_sha256"],
                      "all_target_units_captured_in_each_forward": True,
                      "baseline_intervention_shapes_equal": True})
    print("WORKER_OK gpu=%d videos=%d rows=%d forwards=%d" % (gpu, len(seen), len(assigned) * 80 * 51, count_forwards))


def _read_worker_rows(output: Path, gpu: int) -> List[Dict[str, str]]:
    path = output / ("task042_gpu%d_frame_pair_sensitivity.csv" % gpu)
    require(path.is_file(), "worker raw sensitivity file is absent: " + str(path))
    return read_csv(path)


def _aggregate_pair_distances(unit_summary: Mapping[Tuple[int, int], Mapping[str, Any]],
                              unit_map: Mapping[int, Mapping[str, Any]],
                              video_map: Mapping[int, Mapping[int, Mapping[str, Any]]],
                              domains_only: bool = True) -> Dict[Tuple[int, int], Dict[str, Any]]:
    ids = sorted(unit_map)
    groups: Dict[str, List[int]] = defaultdict(list)
    for uid, row in unit_map.items():
        groups[str(row["domain_id"])].append(uid)
    pairs = []
    for group in groups.values():
        if len(group) >= 2:
            pairs.extend((i, j) for pos, i in enumerate(sorted(group)) for j in sorted(group)[pos + 1:])
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, j in pairs:
        video_values: List[float] = []
        span_values: Dict[int, List[float]] = {s: [] for s in SPANS}
        for video_index in sorted(video_map):
            zi = video_map[video_index].get(i, {}).get("z")
            zj = video_map[video_index].get(j, {}).get("z")
            if zi is None or zj is None:
                continue
            d = relation_distance(zi, zj)
            if d is not None:
                video_values.append(d)
            for span in SPANS:
                idxs = [q for q, item in enumerate(video_map[video_index][i]["conditions"])
                        if norm_int(item["span"]) == span]
                ds = relation_distance([zi[k] for k in idxs], [zj[k] for k in idxs])
                if ds is not None:
                    span_values[span].append(ds)
        ui, uj = unit_map[i], unit_map[j]
        result[(i, j)] = {
            "uid_i": i, "uid_j": j, "domain_id": str(ui["domain_id"]),
            "d_temp": (sum(video_values) / len(video_values)) if video_values else None,
            "valid_video_count": len(video_values),
            "span_distances": {s: (sum(v) / len(v) if v else None) for s, v in span_values.items()},
            "span_valid_video_counts": {s: len(v) for s, v in span_values.items()},
        }
    return result


def finalize(args: argparse.Namespace) -> None:
    import numpy as np
    output = Path(args.output_dir)
    config = json.loads((output / "task042_run_config.json").read_text(encoding="utf-8"))
    for gpu in (0, 1):
        done = json.loads((output / ("task042_gpu%d_done.json" % gpu)).read_text(encoding="utf-8"))
        require(done["checkpoint_sha256"] == CHECKPOINT_SHA256 and done["physical_gpu"] == gpu,
                "worker completion provenance mismatch")
        require(done["all_target_units_captured_in_each_forward"] is True and
                done["baseline_intervention_shapes_equal"] is True, "worker shape/capture audit failed")
    units = {norm_int(r["task037_global_index"]): r for r in read_csv(Path(config["unit_manifest"]))}
    videos = {norm_int(r["video_index"]): r for r in read_csv(Path(config["video_manifest"]))}
    shards = deterministic_video_shards(list(videos.values()))
    all_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for gpu in (0, 1):
        rows = _read_worker_rows(output, gpu)
        done = json.loads((output / ("task042_gpu%d_done.json" % gpu)).read_text(encoding="utf-8"))
        require(len(rows) == int(done["row_count"]), "worker row count mismatch")
        for row in rows:
            key = (norm_int(row["video_index"]), norm_int(row["task037_global_index"]),
                   norm_int(row["span"]), norm_int(row["pair_index"]))
            require(key not in seen_keys, "duplicate merged video/unit/condition identity")
            seen_keys.add(key)
            require(key[0] in shards[gpu] and key[1] in units, "worker row crossed frozen video/unit shard")
            require(key[2] in SPANS and 0 <= key[3] < 16, "worker row has unknown condition identity")
            all_rows.append(dict(row))
    expected = 30 * 51 * 80
    require(len(all_rows) == expected and len(seen_keys) == expected,
            "merged sensitivity rows must equal 30 videos x 51 units x 80 conditions")
    rows_by_video_unit: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_video_unit[(norm_int(row["video_index"]), norm_int(row["task037_global_index"]))].append(row)
    for video in videos:
        for uid in units:
            selected = rows_by_video_unit[(video, uid)]
            require(len(selected) == 80, "unit/video signature does not contain all 80 exact interventions")
            selected.sort(key=lambda r: (norm_int(r["span"]), norm_int(r["pair_index"])))
            z, mean, std, degenerate = normalize_signature([float(r["relative_sensitivity"]) for r in selected])
            for index, row in enumerate(selected):
                row["normalized_signature_z"] = "" if degenerate else z[index]

    all_rows.sort(key=lambda r: (norm_int(r["video_index"]), norm_int(r["task037_global_index"]),
                                 norm_int(r["span"]), norm_int(r["pair_index"])))
    write_csv(output / "task042_frame_pair_sensitivity.csv", all_rows, SENSITIVITY_FIELDS)

    summaries: List[Dict[str, Any]] = []
    by_video_unit: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for video_index in sorted(videos):
        for uid in sorted(units):
            rows = rows_by_video_unit[(video_index, uid)]
            rows.sort(key=lambda r: (norm_int(r["span"]), norm_int(r["pair_index"])))
            vals = [float(r["relative_sensitivity"]) for r in rows]
            z, mean, std, degenerate = normalize_signature(vals)
            zvals = [] if degenerate else list(z or [])
            u, v = units[uid], videos[video_index]
            summary = {"video_index": video_index, "dataset_index": norm_int(v["dataset_index"]),
                "video_id": v["video_id"], "label": norm_int(v["label"]), "task037_global_index": uid,
                "layer": u["layer"], "unit_type": u["unit_type"], "unit_index": u["unit_index"],
                "stage": u["stage"], "domain_id": u["domain_id"], "capture_kind": u["capture_kind"],
                "condition_count": len(rows), "mean_e": mean, "median_e": statistics.median(vals),
                "std_e": std, "min_e": min(vals), "max_e": max(vals),
                "mean_baseline_norm": statistics.mean(float(r["baseline_norm"]) for r in rows),
                "mean_delta_norm": statistics.mean(float(r["delta_norm"]) for r in rows),
                "degenerate": bool(degenerate), "mean_z": None if degenerate else statistics.mean(zvals),
                "std_z": None if degenerate else statistics.pstdev(zvals),
                "median_abs_z": None if degenerate else statistics.median(abs(x) for x in zvals)}
            summaries.append(summary)
            by_video_unit[video_index][uid] = {"z": None if degenerate else zvals,
                "conditions": [{"span": norm_int(r["span"]), "pair_index": norm_int(r["pair_index"])} for r in rows],
                "e": vals, "summary": summary}
    write_csv(output / "task042_unit_video_signature_summary.csv", summaries, SUMMARY_FIELDS)

    # Merge the two independently generated structural audits by frozen identity.
    shape_by_uid: Dict[int, Dict[str, Any]] = {}
    for gpu in (0, 1):
        audit = read_csv(output / ("task042_gpu%d_activation_shape_audit.csv" % gpu))
        require(len(audit) == 51, "each GPU must audit all 51 unit activation shapes")
        for row in audit:
            uid = norm_int(row["task037_global_index"])
            if uid in shape_by_uid:
                require(all(shape_by_uid[uid][k] == row[k] for k in ("num_heads", "head_dim", "hidden_width",
                    "window_count", "token_count", "activation_tensor_shape", "layer_capture_tensor_shape",
                    "qkv_projection_shape", "attention_weights_shape")),
                        "GPU0/GPU1 observed activation shape identity mismatch")
            else:
                shape_by_uid[uid] = row
    require(set(shape_by_uid) == set(units), "merged activation shape audit lost a frozen unit")
    write_csv(output / "task042_activation_shape_audit.csv", [shape_by_uid[u] for u in sorted(shape_by_uid)],
              ("task037_global_index", "layer", "unit_type", "unit_index", "stage", "domain_id", "capture_kind",
               "num_heads", "head_dim", "hidden_width", "window_count", "token_count", "activation_tensor_shape",
               "layer_capture_tensor_shape", "qkv_projection_shape", "attention_weights_shape",
               "baseline_intervention_shape_equal", "shape_audit_video_index", "physical_gpu"))

    pair_map = _aggregate_pair_distances({}, units, by_video_unit)
    unit_norm_metrics = {}
    for uid in units:
        these = [row for key, rows in rows_by_video_unit.items() if key[1] == uid for row in rows]
        unit_norm_metrics[uid] = {
            "mean_e": statistics.mean(float(r["relative_sensitivity"]) for r in these),
            "mean_baseline_norm": statistics.mean(float(r["baseline_norm"]) for r in these),
            "mean_delta_norm": statistics.mean(float(r["delta_norm"]) for r in these),
        }
    pair_rows: List[Dict[str, Any]] = []
    span_rows: List[Dict[str, Any]] = []
    for (i, j), item in sorted(pair_map.items()):
        ui, uj = units[i], units[j]
        type_pair = "-".join(sorted(("head" if ui["capture_kind"] == "head" else "FFN",
                                     "head" if uj["capture_kind"] == "head" else "FFN")))
        row = {"domain_id": item["domain_id"], "task037_global_index_i": i, "layer_i": ui["layer"],
               "unit_type_i": ui["unit_type"], "unit_index_i": ui["unit_index"], "stage_i": ui["stage"],
               "task037_global_index_j": j, "layer_j": uj["layer"], "unit_type_j": uj["unit_type"],
               "unit_index_j": uj["unit_index"], "stage_j": uj["stage"], "type_pair": type_pair,
               "d_temp": item["d_temp"], "valid_video_count": item["valid_video_count"],
               "mean_e_i": unit_norm_metrics[i]["mean_e"],
               "mean_baseline_norm_i": unit_norm_metrics[i]["mean_baseline_norm"],
               "mean_delta_norm_i": unit_norm_metrics[i]["mean_delta_norm"],
               "mean_e_j": unit_norm_metrics[j]["mean_e"],
               "mean_baseline_norm_j": unit_norm_metrics[j]["mean_baseline_norm"],
               "mean_delta_norm_j": unit_norm_metrics[j]["mean_delta_norm"]}
        for s in SPANS:
            row["d_span_%d" % s] = item["span_distances"][s]
            row["n_span_%d" % s] = item["span_valid_video_counts"][s]
            span_rows.append({"domain_id": item["domain_id"], "task037_global_index_i": i,
                "task037_global_index_j": j, "type_pair": type_pair, "span": s,
                "d_temp": item["span_distances"][s], "valid_video_count": item["span_valid_video_counts"][s]})
        pair_rows.append(row)
    pair_fields = ("domain_id", "task037_global_index_i", "layer_i", "unit_type_i", "unit_index_i", "stage_i",
                   "task037_global_index_j", "layer_j", "unit_type_j", "unit_index_j", "stage_j", "type_pair",
                   "d_temp", "valid_video_count", "mean_e_i", "mean_baseline_norm_i", "mean_delta_norm_i",
                   "mean_e_j", "mean_baseline_norm_j", "mean_delta_norm_j") + tuple("d_span_%d" % s for s in SPANS) + tuple("n_span_%d" % s for s in SPANS)
    write_csv(output / "task042_temporal_pair_distance.csv", pair_rows, pair_fields)
    write_csv(output / "task042_span_specific_distance.csv", span_rows,
              ("domain_id", "task037_global_index_i", "task037_global_index_j", "type_pair", "span", "d_temp", "valid_video_count"))

    by_domain: Dict[str, List[int]] = defaultdict(list)
    for uid, unit in units.items():
        by_domain[str(unit["domain_id"])].append(uid)
    domain_rows = []
    for domain, ids in sorted(by_domain.items(), key=lambda x: int(x[0])):
        if len(ids) < 2:
            continue
        rows = [r for r in pair_rows if str(r["domain_id"]) == domain and r["d_temp"] is not None]
        distances = {(min(norm_int(r["task037_global_index_i"]), norm_int(r["task037_global_index_j"])),
                      max(norm_int(r["task037_global_index_i"]), norm_int(r["task037_global_index_j"]))): float(r["d_temp"]) for r in rows}
        nn = []
        for uid in ids:
            found = nearest_neighbor(ids, distances, uid)
            if found is not None:
                nn.append({"unit": uid, "nn": found[0], "distance": found[1]})
        min_pair = min(rows, key=lambda r: (float(r["d_temp"]), norm_int(r["task037_global_index_i"]), norm_int(r["task037_global_index_j"]))) if rows else None
        max_pair = max(rows, key=lambda r: (float(r["d_temp"]), -norm_int(r["task037_global_index_i"]), -norm_int(r["task037_global_index_j"]))) if rows else None
        vals = [float(r["d_temp"]) for r in rows]
        domain_rows.append({"domain_id": domain, "unit_count": len(ids), "pair_count": len(rows),
            "unit_global_indices": json.dumps(sorted(ids)), "unit_types": json.dumps(sorted(set(units[i]["unit_type"] for i in ids))),
            "stages": json.dumps(sorted(set(norm_int(units[i]["stage"]) for i in ids))),
            "distance_mean": distribution(vals)["mean"], "distance_median": distribution(vals)["median"],
            "distance_min": min(vals) if vals else None, "distance_max": max(vals) if vals else None,
            "min_distance_pair": "" if min_pair is None else "%s-%s" % (min_pair["task037_global_index_i"], min_pair["task037_global_index_j"]),
            "max_distance_pair": "" if max_pair is None else "%s-%s" % (max_pair["task037_global_index_i"], max_pair["task037_global_index_j"]),
            "nearest_neighbor_map": json.dumps(nn, sort_keys=True)})
    write_csv(output / "task042_domain_temporal_structure.csv", domain_rows,
              ("domain_id", "unit_count", "pair_count", "unit_global_indices", "unit_types", "stages",
               "distance_mean", "distance_median", "distance_min", "distance_max", "min_distance_pair", "max_distance_pair", "nearest_neighbor_map"))

    matched_rows = []
    same_vals, cross_vals = [], []
    for uid_i, ui in sorted(units.items()):
        for uid_j, uj in sorted(units.items()):
            if uid_j <= uid_i or ui["capture_kind"] != uj["capture_kind"] or norm_int(ui["stage"]) != norm_int(uj["stage"]):
                continue
            same_domain = str(ui["domain_id"]) == str(uj["domain_id"])
            key = (uid_i, uid_j)
            value = pair_map.get(key, {}).get("d_temp") if same_domain else None
            if same_domain:
                if value is None:
                    continue
                same_vals.append(float(value))
            else:
                # Cross-domain controls are calculated with the same per-video z signatures.
                vals = []
                for vi in by_video_unit:
                    zi = by_video_unit[vi][uid_i]["z"]
                    zj = by_video_unit[vi][uid_j]["z"]
                    if zi is None or zj is None:
                        continue
                    dd = relation_distance(zi, zj)
                    if dd is not None:
                        vals.append(dd)
                value = sum(vals) / len(vals) if vals else None
                if value is None:
                    continue
                cross_vals.append(float(value))
            matched_rows.append({"comparison": "same_domain_matched" if same_domain else "cross_domain_matched",
                "same_unit_type": ui["unit_type"], "same_stage": ui["stage"],
                "domain_i": ui["domain_id"], "domain_j": uj["domain_id"],
                "task037_global_index_i": uid_i, "task037_global_index_j": uid_j,
                "d_temp": value, "valid_video_count": 30})
    write_csv(output / "task042_matched_cross_domain_control.csv", matched_rows,
              ("comparison", "same_unit_type", "same_stage", "domain_i", "domain_j",
               "task037_global_index_i", "task037_global_index_j", "d_temp", "valid_video_count"))

    # Offline class-diverse subsets: A/B/C by manifest position within class;
    # paired subsets are positions [1,2], [1,3], [2,3].
    by_label: Dict[int, List[int]] = defaultdict(list)
    for vi, v in sorted(videos.items()):
        by_label[norm_int(v["label"])].append(vi)
    for label in by_label:
        by_label[label].sort()
    subset_map: Dict[str, List[int]] = {"A_position1": [], "B_position2": [], "C_position3": []}
    for pos, name in enumerate(("A_position1", "B_position2", "C_position3")):
        subset_map[name] = [by_label[label][pos] for label in sorted(by_label)]
    subset_map["AB_positions12"] = [by_label[label][k] for label in sorted(by_label) for k in (0, 1)]
    subset_map["AC_positions13"] = [by_label[label][k] for label in sorted(by_label) for k in (0, 2)]
    subset_map["BC_positions23"] = [by_label[label][k] for label in sorted(by_label) for k in (1, 2)]
    subset_map["full_10x3"] = sorted(videos)
    calibration_rows = []
    for domain, ids in sorted(by_domain.items(), key=lambda x: int(x[0])):
        if len(ids) < 2:
            continue
        domain_pairs = [(i, j) for pos, i in enumerate(sorted(ids)) for j in sorted(ids)[pos + 1:]]
        full_vals = []
        for i, j in domain_pairs:
            item = pair_map[(i, j)]["d_temp"]
            full_vals.append(item)
        full_nn = {uid: nearest_neighbor(ids, {(i, j): pair_map[(i, j)]["d_temp"] for i, j in domain_pairs}, uid)
                   for uid in ids}
        for subset_name, video_ids in subset_map.items():
            current_vals, current_by_pair = [], {}
            for i, j in domain_pairs:
                dvals = []
                for vi in video_ids:
                    zi, zj = by_video_unit[vi][i]["z"], by_video_unit[vi][j]["z"]
                    if zi is None or zj is None:
                        continue
                    d = relation_distance(zi, zj)
                    if d is not None:
                        dvals.append(d)
                val = sum(dvals) / len(dvals) if dvals else None
                current_by_pair[(i, j)] = val
                current_vals.append(val)
            valid = [(a, b) for a, b in zip(full_vals, current_vals) if a is not None and b is not None]
            rank_s = spearman_corr([a for a, _ in valid], [b for _, b in valid]) if valid else None
            rank_k = kendall_tau_b([a for a, _ in valid], [b for _, b in valid]) if valid else None
            subset_nn = {}
            for uid in ids:
                dmap = {key: value for key, value in current_by_pair.items()}
                subset_nn[uid] = nearest_neighbor(ids, dmap, uid)
            comparable = [uid for uid in ids if full_nn.get(uid) is not None and subset_nn.get(uid) is not None]
            stable = sum(1 for uid in comparable if full_nn[uid][0] == subset_nn[uid][0])
            calibration_rows.append({"domain_id": domain, "subset": subset_name, "video_count": len(video_ids),
                "class_count": 10, "pair_count": len(valid), "spearman_vs_full": rank_s,
                "kendall_vs_full": rank_k, "nn_comparable_units": len(comparable),
                "nn_identity_stable_units": stable,
                "nn_identity_stability": stable / len(comparable) if comparable else None})
    write_csv(output / "task042_calibration_stability.csv", calibration_rows,
              ("domain_id", "subset", "video_count", "class_count", "pair_count", "spearman_vs_full",
               "kendall_vs_full", "nn_comparable_units", "nn_identity_stable_units", "nn_identity_stability"))

    descriptor_rows = []
    dtemp_values, dst_values, d3_values = [], [], []
    for (i, j), item in sorted(pair_map.items()):
        ui, uj = units[i], units[j]
        if item["d_temp"] is None:
            continue
        delta3 = math.sqrt(sum((float(ui[k]) - float(uj[k])) ** 2 for k in ("D_abs", "D_rel", "D_third")))
        delta_third = abs(float(ui["D_third"]) - float(uj["D_third"]))
        descriptor_rows.append({"domain_id": item["domain_id"], "task037_global_index_i": i,
            "task037_global_index_j": j, "d_temp": item["d_temp"], "abs_delta_D_st": delta_third,
            "abs_delta_D_third": delta_third,
            "raw_l2_distance_frozen_3d_descriptor": delta3,
            "third_descriptor_source_name": ui["third_descriptor_name"],
            "descriptor_values_are_frozen": True})
        dtemp_values.append(float(item["d_temp"]))
        dst_values.append(delta_third)
        d3_values.append(delta3)
    write_csv(output / "task042_descriptor_complementarity.csv", descriptor_rows,
              ("domain_id", "task037_global_index_i", "task037_global_index_j", "d_temp", "abs_delta_D_st", "abs_delta_D_third",
               "raw_l2_distance_frozen_3d_descriptor", "third_descriptor_source_name", "descriptor_values_are_frozen"))

    raw_by_type: Dict[str, List[float]] = defaultdict(list)
    z_by_type: Dict[str, List[float]] = defaultdict(list)
    for row in all_rows:
        raw_by_type[str(row["unit_type"])].append(float(row["relative_sensitivity"]))
        if row["normalized_signature_z"] != "":
            z_by_type[str(row["unit_type"])].append(float(row["normalized_signature_z"]))
    type_scale = {typ: {"raw_relative_sensitivity": distribution(vals),
                        "within_unit_video_z": distribution(z_by_type.get(typ, []))}
                  for typ, vals in raw_by_type.items()}
    same_stats, cross_stats = distribution(same_vals), distribution(cross_vals)
    pair_type_stats = {}
    for type_pair in sorted(set(str(r["type_pair"]) for r in pair_rows)):
        pair_type_stats[type_pair] = distribution([float(r["d_temp"]) for r in pair_rows
            if str(r["type_pair"]) == type_pair and r["d_temp"] is not None])
    summary = {
        "task": "Task042 POST-BMS FRAME-TO-FRAME RELATION REDUNDANCY DIAGNOSIS",
        "decision": None,
        "decision_options": ["A. FRAME_RELATION_REDUNDANCY_PROMISING",
                             "B. FRAME_RELATION_REDUNDANCY_WEAK_OR_UNRESOLVED",
                             "C. FRAME_RELATION_REDUNDANCY_REJECTED"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "unit_count": len(units), "video_count": len(videos), "multi_unit_domain_count": len(domain_rows),
        "same_domain_pair_count": len(pair_rows), "sensitivity_record_count": len(all_rows),
        "forward_count": sum(json.loads((output / ("task042_gpu%d_done.json" % g)).read_text())["forward_count"] for g in (0, 1)),
        "expected_forward_count": 2430,
        "intervention_protocol": {"T": 32, "spans": list(SPANS), "pairs_per_span": 16,
                                   "conditions_per_video": 80, "frames_swapped": 2,
                                   "intervention_source": "Task040 enumerate_fixed_cardinality_temporal_pairs"},
        "sensitivity_definition": "||H(S_q X)-H(X)||_F / (||H(X)||_F + 1e-12)",
        "signature_normalization": "population mean/std over 80 conditions within each unit-video; sigma<=1e-12 is degenerate and excluded from correlations",
        "attention_capture": "attn_drop output @ V, individual heads before transpose/concat/projection; model.eval() dropout identity",
        "ffn_capture": "Mlp.act output after fc1+GELU, before fc2",
        "descriptor_provenance": "Task037 descriptor_statistics.csv stores third coordinate as D_third / D_dyn; Task041 text calls the frozen third coordinate D_st. No descriptor is recomputed or changed.",
        "type_scale_audit": type_scale,
        "same_domain_pair_type_distances": pair_type_stats,
        "matched_cross_domain_control": {"same_domain_matched": same_stats, "cross_domain_matched": cross_stats,
            "mann_whitney_u_two_sided_p_exploratory": _spearman_mann_whitney(same_vals, cross_vals),
            "cliffs_delta_same_minus_cross": cliffs_delta(same_vals, cross_vals),
            "matching": "same authoritative Task037 unit_type and same Swin stage; different BMS domain for controls",
            "non_independent_pair_observations": True, "paper_claim_from_p_value": False},
        "descriptor_complementarity": {"pair_count": len(descriptor_rows),
            "spearman_d_temp_vs_abs_delta_D_st": spearman_corr(dtemp_values, dst_values),
            "spearman_d_temp_vs_abs_delta_D_third": spearman_corr(dtemp_values, dst_values),
            "spearman_d_temp_vs_raw_l2_frozen_3d_descriptor": spearman_corr(dtemp_values, d3_values),
            "third_axis_source_name": "D_dyn", "no_descriptor_change": True},
        "videos_inferred": ["baseline+80 conditions exactly once per video"],
        "physical_gpus": [0, 1], "dtype": "float32", "amp": False,
        "pruning_performed": False, "finetuning_performed": False,
        "damage_oracle_optimization_performed": False,
        "task041_phase_i_context": {
            "decision": "C. CLASS_DIVERSITY_DOES_NOT_RESCUE_TEMPORAL_SELECTION",
            "9x1_temporal_safest_accuracy": 0.0, "3x3_temporal_safest_accuracy": 2.0 / 7.0,
        },
        "input_sha256": config["input_sha256"],
    }

    # Conservative diagnostic decision. The rule depends only on temporal
    # relation structure/stability/complementarity, never pruning damage.
    domains_with_pairs = [r for r in domain_rows if int(r["pair_count"]) > 0]
    stability_rows = [r for r in calibration_rows if r["subset"] in ("A_position1", "B_position2", "C_position3")
                      and r["nn_identity_stability"] is not None]
    stable_median = statistics.median(float(r["nn_identity_stability"]) for r in stability_rows) if stability_rows else 0.0
    within_values = [float(r["d_temp"]) for r in pair_rows if r["d_temp"] is not None]
    relation_spread = statistics.pstdev(within_values) if len(within_values) > 1 else 0.0
    corr3 = summary["descriptor_complementarity"]["spearman_d_temp_vs_raw_l2_frozen_3d_descriptor"]
    if domains_with_pairs and stable_median >= 0.75 and relation_spread > 1e-6 and (corr3 is None or abs(corr3) < 0.8):
        decision = "A. FRAME_RELATION_REDUNDANCY_PROMISING"
    elif not within_values or relation_spread <= 1e-6:
        decision = "C. FRAME_RELATION_REDUNDANCY_REJECTED"
    else:
        decision = "B. FRAME_RELATION_REDUNDANCY_WEAK_OR_UNRESOLVED"
    summary["decision"] = decision
    summary["decision_rule"] = {
        "A": "nontrivial within-domain distances, median A/B/C nearest-neighbor identity stability >=0.75, and not near-perfect Spearman redundancy with frozen raw 3D distance; diagnostic only",
        "B": "some relation structure exists but the conservative A criterion is not met",
        "C": "no measurable within-domain distance variation",
        "no_pruning_authorization": True,
    }
    write_json(output / "task042_summary.json", summary)
    _write_report(output, summary, domain_rows, pair_rows, calibration_rows, type_scale)
    print("FINALIZE_OK decision=%s pairs=%d sensitivity_rows=%d forwards=2430" %
          (decision, len(pair_rows), len(all_rows)))


def _md(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return ("%.*f" % (digits, float(value)))
    except Exception:
        return str(value)


def _write_report(output: Path, summary: Mapping[str, Any], domains: Sequence[Mapping[str, Any]],
                  pairs: Sequence[Mapping[str, Any]], stability: Sequence[Mapping[str, Any]],
                  type_scale: Mapping[str, Any]) -> None:
    lines = [
        "# Task042 — Post-BMS Frame-to-Frame Relation Redundancy Diagnosis", "",
        "**Decision: %s**" % summary["decision"], "",
        "This is an activation-relation diagnosis within frozen Task040 BMS groups. It does not authorize pruning.", "",
        "## Frozen protocol", "",
        "- Cohort: 51 exact Task040 Phase-D.1 units; 30 frozen Task041 videos from 10 classes, 3 per class.",
        "- Interventions: T=32, spans 1/2/4/8/16, 16 deterministic exact-two-frame swaps per span.",
        "- Forward count: %s; GPUs: physical 0 and 1; FP32, AMP off." % summary["forward_count"],
        "- Head activation: attention probabilities multiplied by V, per head before concatenation/projection.",
        "- FFN activation: post-GELU output after fc1 and before fc2.",
        "- Sensitivity is relative internal activation change; it is not classification damage or importance.",
        "- No selector, pruning, fine-tuning, descriptor/BMS update, or damage-oracle optimization was run.", "",
        "## Main observations", "",
        "Same-domain pair count: %d across %d multi-unit BMS domains." % (len(pairs), len(domains)),
        "The distance is the mean over videos of `(1 - Pearson(z_i, z_j))/2`, where each unit-video signature is standardized across all 80 conditions.",
        "Span-specific correlations use the corresponding 16 entries from that full-signature normalization.",
        "Degenerate unit-video signatures (population sigma <= 1e-12) are excluded from correlations and retained in the summary CSV.", "",
        "| Domain | Units | Pairs | Min d | Median d | Max d | Closest pair | Most distant pair |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in domains:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["domain_id"], r["unit_count"], r["pair_count"], _md(r["distance_min"]),
            _md(r["distance_median"]), _md(r["distance_max"]), r["min_distance_pair"] or "NA", r["max_distance_pair"] or "NA"))
    lines += ["", "The closest and most distant pairs are descriptive extremes; no pruning threshold was introduced.", "",
              "## Calibration stability", "",
              "Offline subsets use the manifest's first/second/third video within every action class (A/B/C), plus deterministic position pairs AB/AC/BC. No additional inference was run.", "",
              "| Domain | Subset | Videos | Spearman vs full | Kendall vs full | NN identity stability |",
              "|---:|---|---:|---:|---:|---:|"]
    for r in stability:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (r["domain_id"], r["subset"], r["video_count"],
            _md(r["spearman_vs_full"]), _md(r["kendall_vs_full"]), _md(r["nn_identity_stability"])))
    lines += ["", "## Attention/FFN scale audit", "", "Raw sensitivity distributions are reported before unit-video normalization. The normalized z distributions are reported after normalization; each nondegenerate unit-video signature has mean 0 and population standard deviation 1 by construction.", ""]
    for typ, stats in sorted(type_scale.items()):
        lines.append("- **%s** raw e: n=%s, mean=%s, median=%s, q25=%s, q75=%s; normalized z: n=%s, mean=%s, std=%s." % (
            typ, stats["raw_relative_sensitivity"]["count"], _md(stats["raw_relative_sensitivity"]["mean"]),
            _md(stats["raw_relative_sensitivity"]["median"]), _md(stats["raw_relative_sensitivity"]["q25"]),
            _md(stats["raw_relative_sensitivity"]["q75"]), stats["within_unit_video_z"]["count"],
            _md(stats["within_unit_video_z"]["mean"]), _md(stats["within_unit_video_z"]["std"])))
    lines += ["", "Same-domain distance distributions by pair type:", ""]
    for typ, stats in sorted(summary["same_domain_pair_type_distances"].items()):
        lines.append("- **%s**: n=%s, mean=%s, median=%s, q25=%s, q75=%s, std=%s." % (
            typ, stats["count"], _md(stats["mean"]), _md(stats["median"]), _md(stats["q25"]),
            _md(stats["q75"]), _md(stats["std"])))
    ctrl = summary["matched_cross_domain_control"]
    lines += ["", "## Matched cross-domain control", "",
              "Pairs are matched by exact Task037 unit type and Swin stage. Each cross-domain pair has different frozen BMS domain IDs.",
              "Same-domain matched: n=%s, mean=%s, median=%s, q25=%s, q75=%s, std=%s." % tuple(
                  ctrl["same_domain_matched"][k] for k in ("count", "mean", "median", "q25", "q75", "std")),
              "Cross-domain matched: n=%s, mean=%s, median=%s, q25=%s, q75=%s, std=%s." % tuple(
                  ctrl["cross_domain_matched"][k] for k in ("count", "mean", "median", "q25", "q75", "std")),
              "Exploratory Mann-Whitney U p=%s; Cliff's delta=%s. Pair observations are non-independent; the p-value is not a paper claim." % (
                  _md(ctrl["mann_whitney_u_two_sided_p_exploratory"]), _md(ctrl["cliffs_delta_same_minus_cross"])), "",
              "## Relation to frozen descriptors", "",
              "Task037's frozen table stores the third descriptor axis as `D_third`/`D_dyn`; Task041 documentation calls the frozen third axis `D_st`. The run preserves the exact source field and its SHA rather than recomputing or changing it.",
              "Spearman(d_temp, |ΔD_st|)=%s; Spearman(d_temp, raw L2 frozen 3D descriptor distance)=%s." % (
                  _md(summary["descriptor_complementarity"]["spearman_d_temp_vs_abs_delta_D_st"]),
                  _md(summary["descriptor_complementarity"]["spearman_d_temp_vs_raw_l2_frozen_3d_descriptor"])), "",
              "## Decision", "", "**%s**" % summary["decision"], "",
              "The decision is diagnostic only. It does not authorize logical/physical pruning, fine-tuning, a temporal threshold, or Task043.", "",
              "## Output files", "",
              "`task042_video_manifest.csv`, `task042_unit_manifest.csv`, `task042_activation_shape_audit.csv`, `task042_frame_pair_sensitivity.csv`, `task042_unit_video_signature_summary.csv`, `task042_temporal_pair_distance.csv`, `task042_domain_temporal_structure.csv`, `task042_matched_cross_domain_control.csv`, `task042_calibration_stability.csv`, `task042_span_specific_distance.csv`, `task042_descriptor_complementarity.csv`, and `task042_summary.json`.", ""]
    (output / "task042_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "preflight", "worker", "finalize"), required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--profile", default=str(PROFILE_DEFAULT))
    parser.add_argument("--descriptors", default=str(DESCRIPTORS_DEFAULT))
    parser.add_argument("--unit-mapping", default=str(UNIT_MAPPING_DEFAULT))
    parser.add_argument("--video-manifest", default=str(VIDEO_MANIFEST_DEFAULT))
    parser.add_argument("--val-list", default=str(VAL_LIST_DEFAULT))
    parser.add_argument("--frame-root", default=str(FRAME_ROOT_DEFAULT))
    parser.add_argument("--project-root", default=str(PROJECT_DEFAULT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT))
    parser.add_argument("--repo-root", default=str(REPO_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args(argv)
    {"prepare": prepare, "preflight": preflight, "worker": worker, "finalize": finalize}[args.phase](args)


if __name__ == "__main__":
    main()

"""Task030: causal validation of the Task029 Average-risk granularity finding.

Task030 is a diagnostic experiment, not a selector.  It reads the frozen
Task028 50% registry and the immutable Task029 tables, then temporarily
removes logical units from fresh model instances.  Model forwards run on CUDA
when executed on the server; all aggregation and plotting remains CPU-side.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CODE_VERSION = "task030_average_granularity_causal_validation_v1"
SEED = 3407
BATCH_SIZE = 4
DIAGNOSTIC_SAMPLES = 64
TARGET_SPARSITY = 0.50
CONTEXT_FULL = "FULL"
CONTEXT_TASK028_50 = "TASK028_50"
CONTEXTS = (CONTEXT_FULL, CONTEXT_TASK028_50)
BASELINE_LOGIT_FILES = {CONTEXT_FULL: "full_baseline_logits.npz",
                        CONTEXT_TASK028_50: "task028_50_baseline_logits.npz"}
BASELINE_METRIC_FILES = {CONTEXT_FULL: "full_baseline_metrics.json",
                         CONTEXT_TASK028_50: "task028_50_baseline_metrics.json"}
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
INTERVENTIONS = ("attention_single", "ffn_single", "ffn_cost_matched_pack")
PRIMARY_METRICS = ("relative_logit_l2", "kl_divergence", "true_class_logit_drop",
                   "ce_increase", "prediction_flip_rate")
RISK_FIELDS = ("Delta_average", "Delta_total", "p_average", "p_total", "R_adaptive")
TASK028_FILES = ("selection_50/registry.json", "selection_50/causal_selection_trace.csv")
TASK029_FILES = ("final_50_attention_candidates.csv", "snapshot_candidates.csv",
                 "attention_rank_by_snapshot.csv")
ATTENTION_FIELDS = ("pair_id", "stage", "global_index", "layer", "unit_index", "domain_id",
                    "parameter_cost", "Delta_average", "Delta_total", "p_average", "p_total",
                    "domain_damage", "R_adaptive", "global_rank")
PAIR_FIELDS = ("pair_id", "stage", "attention_global_index", "ffn_global_index",
               "attention_layer", "ffn_layer", "attention_unit_index", "ffn_unit_index",
               "attention_parameter_cost", "ffn_parameter_cost", "attention_p_total",
               "ffn_p_total", "attention_p_average", "ffn_p_average", "p_total_abs_diff")
PACK_FIELDS = ("pair_id", "stage", "attention_global_index", "attention_parameter_cost",
               "ffn_pack_parameter_cost", "ffn_pack_size", "cost_ratio", "overlap_with_previous",
               "ffn_global_indices")


def _int(value: object, label: str = "value") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc


def _float(value: object, label: str = "value", default: float | None = None) -> float:
    if value is None or str(value).strip() == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Missing {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_files(root: Path, names: Sequence[str]) -> dict[str, str]:
    root = Path(root)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing immutable artifacts below {root}: {missing}")
    return {name: sha256_file(root / name) for name in names}


def _stage(row: Mapping[str, object]) -> str:
    if str(row.get("stage", "")).strip():
        return str(row["stage"])
    layer = str(row.get("layer", ""))
    if "layers." in layer:
        return layer.split("layers.", 1)[1].split(".", 1)[0]
    if layer.startswith("stage"):
        return layer.split(".", 1)[0].replace("stage", "")
    return layer.split(".", 1)[0] if layer else "unknown"


def _registry_sha(registry: Mapping[str, object]) -> str:
    try:
        import task028_main50_tad_logical_finetune as task028
        return str(task028.canonical_registry_sha256(registry))
    except (ImportError, AttributeError):
        canonical = tuple(sorted(
            (str(layer), str(entry["unit_type"]),
             tuple(sorted(int(value) for value in entry["indices"])))
            for layer, entry in registry.items()
        ))
        canonical = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _registry_removed_keys(registry: Mapping[str, object]) -> set[tuple[str, str, int]]:
    """Return structural identities removed by a frozen logical registry."""
    removed: set[tuple[str, str, int]] = set()
    for layer, entry in registry.items():
        if not isinstance(entry, Mapping):
            continue
        unit_type = str(entry.get("unit_type", ""))
        for index in entry.get("indices", ()):
            removed.add((str(layer), unit_type, _int(index, "registry unit index")))
    return removed


def assert_rows_retained_at_50(rows: Sequence[Mapping[str, object]],
                               registry: Mapping[str, object]) -> None:
    """Reject candidates that are infeasible or already removed at 50%."""
    removed = _registry_removed_keys(registry)
    for position, row in enumerate(rows):
        if str(row.get("feasible", "True")).lower() in ("false", "0"):
            raise RuntimeError(f"Task029 candidate {position} is not feasible")
        key = (str(row.get("layer", "")), str(row.get("unit_type", "")),
               _int(row.get("unit_index"), "unit_index"))
        if key in removed:
            raise RuntimeError(
                f"Task029 candidate global_index={row.get('global_index')} "
                "is removed by the frozen Task028 50% registry"
            )


def verify_task028_reference(task028_root: Path) -> dict[str, object]:
    root = Path(task028_root)
    hashes = _hash_files(root, TASK028_FILES)
    selection = read_json(root / "selection_50/registry.json")
    if selection.get("status") != "prepared":
        raise RuntimeError("Task028 50% registry is not prepared")
    if not math.isclose(_float(selection.get("target_sparsity"), "target_sparsity"), TARGET_SPARSITY,
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Task028 registry target is not exactly 50%")
    registry = selection.get("registry")
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError("Task028 registry is malformed")
    registry_sha = _registry_sha(registry)
    if registry_sha != str(selection.get("registry_canonical_sha256", "")):
        raise RuntimeError("Task028 registry canonical SHA mismatch")
    trace = read_csv(root / "selection_50/causal_selection_trace.csv")
    if not trace:
        raise RuntimeError("Task028 causal trace is empty")
    required_trace_fields = ("global_index", "unit_type", "layer", "unit_index")
    missing = [field for field in required_trace_fields if field not in trace[0]]
    if missing:
        raise RuntimeError(f"Task028 causal trace missing fields: {missing}")
    trace_gids = [_int(row.get("global_index"), "global_index") for row in trace]
    if len(trace_gids) != len(set(trace_gids)):
        raise RuntimeError("Task028 causal trace contains duplicate global_index")
    trace_keys = {(str(row.get("layer")), str(row.get("unit_type")),
                   _int(row.get("unit_index"), "unit_index")) for row in trace}
    registry_keys = {(str(layer), str(entry.get("unit_type")), _int(index, "registry index"))
                     for layer, entry in registry.items() if isinstance(entry, Mapping)
                     for index in entry.get("indices", ())}
    if trace_keys != registry_keys:
        raise RuntimeError("Task028 registry does not match causal trace structural identity")
    return {"registry": registry, "registry_sha256": registry_sha,
            "selection": selection, "trace": trace, "hashes": hashes}


def verify_task029_reference(task029_root: Path, task028_reference: Mapping[str, object] | None = None) -> dict[str, object]:
    root = Path(task029_root)
    hashes = _hash_files(root, TASK029_FILES)
    final_attention = read_csv(root / "final_50_attention_candidates.csv")
    if len(final_attention) != 282:
        raise RuntimeError("Task029 final 50% Attention table must contain 282 retained heads")
    gids = [_int(row.get("global_index"), "global_index") for row in final_attention]
    if len(gids) != len(set(gids)):
        raise RuntimeError("Task029 final Attention table has duplicate global_index")
    if any(str(row.get("unit_type")) != TYPE_ATTENTION for row in final_attention):
        raise RuntimeError("Task029 final table contains a non-Attention row")
    snapshots = read_csv(root / "snapshot_candidates.csv")
    snapshot_50 = [row for row in snapshots if math.isclose(_float(row.get("snapshot_target"), "snapshot_target"), .5, rel_tol=0, abs_tol=1e-12)]
    if not snapshot_50:
        raise RuntimeError("Task029 snapshot_candidates.csv lacks exact 50% rows")
    by_gid = {_int(row.get("global_index"), "global_index"): row for row in snapshot_50}
    for row in final_attention:
        gid = _int(row.get("global_index"), "global_index")
        if gid not in by_gid:
            raise RuntimeError(f"Task029 final Attention {gid} is absent from the 50% snapshot")
    ranks = read_csv(root / "attention_rank_by_snapshot.csv")
    rank_50 = next((row for row in ranks if math.isclose(_float(row.get("snapshot_target"), "snapshot_target"), .5, rel_tol=0, abs_tol=1e-12)), None)
    if rank_50 is None or _int(rank_50.get("best_attention_rank"), "best_attention_rank") != 4758:
        raise RuntimeError("Task029 50% Attention rank identity mismatch")
    if task028_reference is not None:
        registry = task028_reference.get("registry", {})
        if not isinstance(registry, Mapping):
            raise RuntimeError("Task028 reference registry is malformed")
        removed = {(str(layer), _int(index, "registry index"))
                   for layer, entry in registry.items() if isinstance(entry, Mapping)
                   for index in entry.get("indices", ())}
        for row in final_attention:
            source = by_gid[_int(row.get("global_index"), "global_index")]
            if (str(source.get("layer")), _int(source.get("unit_index"), "unit_index")) in removed:
                raise RuntimeError("Task029 selected Attention is not retained at 50%")
        # The exact snapshot is the source for all Task030 interventions.  Do
        # not defer this provenance check until after a model is constructed.
        assert_rows_retained_at_50(snapshot_50, registry)
    return {"final_attention": final_attention, "snapshot_50": snapshot_50,
            "ranks": ranks, "hashes": hashes}


def verify_identity(*, task028_root: Path, task029_root: Path, checkpoint: Path,
                    val_split: Path, output_dir: Path) -> dict[str, object]:
    """Bind both frozen references and the exact source inputs."""
    task028_root = Path(task028_root).expanduser().resolve()
    task029_root = Path(task029_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == task028_root or output_dir.is_relative_to(task028_root):
        raise RuntimeError("Task030 output must not be inside Task028")
    if output_dir == task029_root or output_dir.is_relative_to(task029_root):
        raise RuntimeError("Task030 output must not be inside Task029")
    checkpoint, val_split = Path(checkpoint).expanduser().resolve(), Path(val_split).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not val_split.is_file() or val_split.stat().st_size <= 0:
        raise FileNotFoundError(val_split)
    task028 = verify_task028_reference(task028_root)
    task029 = verify_task029_reference(task029_root, task028)
    payload = {
        "status": "PASS", "code_version": CODE_VERSION,
        "task028_registry_canonical_sha256": task028["registry_sha256"],
        "task028_referenced_artifact_sha256": task028["hashes"],
        "task029_referenced_artifact_sha256": task029["hashes"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "validation_split": str(val_split), "validation_split_sha256": sha256_file(val_split),
        "task028_reference_pass": True, "task029_reference_pass": True,
        "task028_artifacts_read_only": True, "task029_artifacts_read_only": True,
        "seed": SEED, "batch_size": BATCH_SIZE,
        "diagnostic_samples": DIAGNOSTIC_SAMPLES, "contexts": list(CONTEXTS),
        "selector_replay_executed": False, "training_executed": False,
        "fine_tuning_executed": False, "new_pruning_hyperparameters": 0,
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "task028": task028, "task029": task029}


def select_attention_heads(rows: Sequence[Mapping[str, object]], per_stage: int = 2) -> list[dict[str, object]]:
    """Select exactly two lowest-p_total Attention heads per Swin stage."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("unit_type")) == TYPE_ATTENTION:
            grouped[_stage(row)].append(row)
    selected: list[dict[str, object]] = []
    for stage in ("0", "1", "2", "3"):
        candidates = sorted(grouped.get(stage, ()), key=lambda row: (_float(row.get("p_total"), "p_total"), _int(row.get("global_index"), "global_index")))
        if len(candidates) < per_stage:
            raise RuntimeError(f"Task029 has fewer than {per_stage} Attention heads in stage {stage}")
        for row in candidates[:per_stage]:
            selected.append({field: row.get(field, "") for field in ATTENTION_FIELDS if field != "pair_id"})
    selected.sort(key=lambda row: (_int(row.get("global_index"), "global_index")))
    for index, row in enumerate(selected, 1):
        row["pair_id"] = f"pair_{index:02d}"
    return selected


def match_ffn_single(attention: Mapping[str, object], rows: Sequence[Mapping[str, object]],
                     used: set[int] | None = None) -> dict[str, object]:
    used = used if used is not None else set()
    stage = _stage(attention)
    candidates = [row for row in rows if str(row.get("unit_type")) == TYPE_FFN and _stage(row) == stage
                  and str(row.get("feasible", "True")).lower() not in ("false", "0")]
    available = [row for row in candidates if _int(row.get("global_index"), "global_index") not in used]
    pool = available or candidates
    if not pool:
        raise RuntimeError(f"No feasible FFN match for Attention stage {stage}")
    target = _float(attention.get("p_total"), "attention p_total")
    return dict(min(pool, key=lambda row: (abs(_float(row.get("p_total"), "ffn p_total") - target), _int(row.get("global_index"), "global_index"))))


def build_cost_matched_pack(attention: Mapping[str, object], rows: Sequence[Mapping[str, object]],
                            used: set[int] | None = None) -> dict[str, object]:
    used = used if used is not None else set()
    stage = _stage(attention)
    candidates = [row for row in rows if str(row.get("unit_type")) == TYPE_FFN and _stage(row) == stage
                  and str(row.get("feasible", "True")).lower() not in ("false", "0")]
    target = _float(attention.get("p_total"), "attention p_total")
    ordered = sorted(candidates, key=lambda row: (abs(_float(row.get("p_total"), "ffn p_total") - target), _int(row.get("global_index"), "global_index")))
    if not ordered:
        raise RuntimeError(f"No feasible FFN pack candidates for Attention stage {stage}")
    unique = []
    for row in ordered:
        gid = _int(row.get("global_index"), "global_index")
        if gid in { _int(item.get("global_index"), "global_index") for item in unique }:
            continue
        if gid in used:
            continue
        unique.append(dict(row))
        if sum(_int(item.get("parameter_cost"), "parameter_cost") for item in unique) >= _int(attention.get("parameter_cost"), "attention parameter_cost"):
            break
    overlap = False
    if not unique:
        unique = [dict(row) for row in ordered]
    if sum(_int(item.get("parameter_cost"), "parameter_cost") for item in unique) < _int(attention.get("parameter_cost"), "attention parameter_cost"):
        # The deterministic rule cannot be satisfied only when the stage has
        # insufficient feasible capacity; use all unique candidates and make
        # that fact explicit in the report.
        overlap = True
    gids = [_int(item.get("global_index"), "global_index") for item in unique]
    if used.intersection(gids):
        overlap = True
    used.update(gids)
    attention_cost = _int(attention.get("parameter_cost"), "attention parameter_cost")
    pack_cost = sum(_int(item.get("parameter_cost"), "ffn parameter_cost") for item in unique)
    return {"rows": unique, "overlap": overlap, "attention_parameter_cost": attention_cost,
            "ffn_pack_parameter_cost": pack_cost, "ffn_pack_size": len(unique),
            "cost_ratio": pack_cost / attention_cost if attention_cost else math.nan}


def construct_pairs(snapshot_rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    attention = select_attention_heads(snapshot_rows)
    ffns = set()
    pairs, packs = [], []
    for row in attention:
        matched = match_ffn_single(row, snapshot_rows, ffns)
        matched_gid = _int(matched.get("global_index"), "global_index")
        ffns.add(matched_gid)
        pairs.append({"pair_id": row["pair_id"], "stage": _stage(row),
                      "attention_global_index": row["global_index"], "ffn_global_index": matched_gid,
                      "attention_layer": row.get("layer", ""), "ffn_layer": matched.get("layer", ""),
                      "attention_unit_index": row.get("unit_index", ""), "ffn_unit_index": matched.get("unit_index", ""),
                      "attention_parameter_cost": row.get("parameter_cost", ""), "ffn_parameter_cost": matched.get("parameter_cost", ""),
                      "attention_p_total": row.get("p_total", ""), "ffn_p_total": matched.get("p_total", ""),
                      "attention_p_average": row.get("p_average", ""), "ffn_p_average": matched.get("p_average", ""),
                      "p_total_abs_diff": abs(_float(row.get("p_total"), "p_total") - _float(matched.get("p_total"), "p_total"))})
        pack = build_cost_matched_pack(row, snapshot_rows, ffns)
        packs.append({"pair_id": row["pair_id"], "stage": _stage(row),
                      "attention_global_index": row["global_index"],
                      "attention_parameter_cost": pack["attention_parameter_cost"],
                      "ffn_pack_parameter_cost": pack["ffn_pack_parameter_cost"],
                      "ffn_pack_size": pack["ffn_pack_size"], "cost_ratio": pack["cost_ratio"],
                      "overlap_with_previous": pack["overlap"],
                      "ffn_global_indices": json.dumps([_int(item["global_index"], "global_index") for item in pack["rows"]])})
    return {"attention": attention, "pairs": pairs, "packs": packs}


def parse_validation_split(path: Path, count: int = DIAGNOSTIC_SAMPLES, seed: int = SEED) -> list[dict[str, object]]:
    """Parse and deterministically sample validation entries without decoding videos."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        label: object = ""
        video = text
        if len(parts) >= 2:
            try:
                label = int(parts[-1]); video = " ".join(parts[:-1])
            except ValueError:
                video = parts[0]; label = parts[-1]
        if not str(label).strip():
            label = Path(video).parent.name
        entries.append({"video_path": video, "class_label": label,
                        "_source_index": len(entries)})
    if len(entries) < count:
        raise RuntimeError(f"Validation split contains only {len(entries)} entries; need {count}")
    entries.sort(key=lambda row: (str(row["class_label"]), str(row["video_path"])))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in entries:
        grouped[str(row["class_label"])].append(row)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        progressed = False
        for label in sorted(grouped):
            if grouped[label]:
                selected.append(grouped[label].pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return [{"sample_index": index, **row} for index, row in enumerate(selected)]


def _torch():
    import torch
    return torch


def _is_torch_tensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "dim") and value.__class__.__module__.startswith("torch")


def relative_logit_l2(base, ablated, eps: float = 1e-12) -> float:
    if _is_torch_tensor(base):
        torch = _torch()
        return float((torch.linalg.vector_norm(ablated - base, dim=1) /
                      (torch.linalg.vector_norm(base, dim=1) + eps)).mean().item())
    import numpy as np
    base, ablated = np.asarray(base, dtype=float), np.asarray(ablated, dtype=float)
    return float(np.mean(np.linalg.norm(ablated - base, axis=1) /
                         (np.linalg.norm(base, axis=1) + eps)))


def kl_divergence(base, ablated) -> float:
    if _is_torch_tensor(base):
        torch = _torch()
        import torch.nn.functional as F
        return float(F.kl_div(F.log_softmax(ablated, dim=1), F.softmax(base, dim=1), reduction="batchmean").item())
    import numpy as np
    base, ablated = np.asarray(base, dtype=float), np.asarray(ablated, dtype=float)
    def softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        out = np.exp(shifted); return out / np.sum(out, axis=1, keepdims=True)
    p, q = softmax(base), softmax(ablated)
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)))


def true_class_logit_drop(base, ablated, labels) -> float:
    if _is_torch_tensor(base):
        return float((base.gather(1, labels.view(-1, 1)).squeeze(1) -
                      ablated.gather(1, labels.view(-1, 1)).squeeze(1)).mean().item())
    import numpy as np
    labels = np.asarray(labels, dtype=int)
    return float(np.mean(base[np.arange(len(labels)), labels] - ablated[np.arange(len(labels)), labels]))


def ce_increase(base, ablated, labels) -> float:
    if _is_torch_tensor(base):
        torch = _torch()
        import torch.nn.functional as F
        return float((F.cross_entropy(ablated, labels) - F.cross_entropy(base, labels)).item())
    import numpy as np
    labels = np.asarray(labels, dtype=int)
    def loss(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return float(np.mean(-log_probs[np.arange(len(labels)), labels]))
    return loss(ablated) - loss(base)


def prediction_flip_rate(base, ablated) -> float:
    if _is_torch_tensor(base):
        return float((base.argmax(1) != ablated.argmax(1)).float().mean().item())
    import numpy as np
    return float(np.mean(np.argmax(base, axis=1) != np.argmax(ablated, axis=1)))


def _topk(logits, labels, k: int) -> float:
    if _is_torch_tensor(logits):
        k = min(int(k), int(logits.shape[1]))
        correct = logits.topk(k, dim=1).indices.eq(labels.view(-1, 1)).any(dim=1)
        return float(correct.float().mean().item())
    import numpy as np
    labels = np.asarray(labels, dtype=int)
    k = min(int(k), int(np.asarray(logits).shape[1]))
    indices = np.argsort(np.asarray(logits), axis=1)[:, -k:]
    return float(np.mean(np.any(indices == labels[:, None], axis=1)))


def compute_metrics(base, ablated, labels, removed_parameter_cost: int) -> dict[str, float]:
    metrics = {
        "relative_logit_l2": relative_logit_l2(base, ablated),
        "kl_divergence": kl_divergence(base, ablated),
        "true_class_logit_drop": true_class_logit_drop(base, ablated, labels),
        "ce_increase": ce_increase(base, ablated, labels),
        "prediction_flip_rate": prediction_flip_rate(base, ablated),
        "top1_change": _topk(ablated, labels, 1) - _topk(base, labels, 1),
        "top5_change": _topk(ablated, labels, 5) - _topk(base, labels, 5),
        "removed_parameter_cost": float(removed_parameter_cost),
    }
    cost = max(float(removed_parameter_cost), 1.0)
    for name in ("relative_logit_l2", "kl_divergence", "prediction_flip_rate"):
        metrics[f"{name}_per_parameter"] = metrics[name] / cost
    for name in ("true_class_logit_drop", "ce_increase", "top1_change", "top5_change"):
        metrics[f"{name}_per_parameter"] = metrics[name] / cost
        metrics[f"{name}_absolute_per_parameter"] = abs(metrics[name]) / cost
    return metrics


def _copy_keep_state(state: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {str(name): {"type": str(record["type"]), "data": [int(value) for value in record["data"]]}
            for name, record in state.items()}


@contextmanager
def temporary_logical_ablation(model, rows: Sequence[Mapping[str, object]]):
    """Apply runtime keep-list changes and restore them exactly on exit."""
    import task027_senior_style_logical_pruning_finetune as task027
    original = _copy_keep_state(task027.extract_keep_indices(model))
    modified = _copy_keep_state(original)
    for row in rows:
        layer = str(row.get("layer")); gid = _int(row.get("unit_index"), "unit_index")
        if layer not in modified:
            raise KeyError(f"Ablation layer missing from keep-state: {layer}")
        data = modified[layer]["data"]
        if gid not in data:
            raise RuntimeError(f"Ablation unit is not retained: {layer}/{gid}")
        if len(data) <= 1:
            raise RuntimeError(f"Ablation would violate min-keep at {layer}")
        modified[layer]["data"] = [value for value in data if value != gid]
    task027.restore_keep_indices(model, modified)
    try:
        yield
    finally:
        task027.restore_keep_indices(model, original)
        restored = _copy_keep_state(task027.extract_keep_indices(model))
        if restored != original:
            raise RuntimeError("Logical keep-state restoration identity failed")


def _load_context_model(checkpoint: Path, context: str, registry: Mapping[str, object],
                        device: str, gpu_ids: Sequence[int] = ()):
    import task027_senior_style_logical_pruning_finetune as task027
    model = task027._load_original(Path(checkpoint), device)
    if context == CONTEXT_TASK028_50:
        task027.apply_logical_pruning_registry(model, registry)
        task027.assert_registry_keep_identity(model, registry)
    model.eval()
    if len(tuple(gpu_ids)) >= 2:
        torch = _torch()
        model = torch.nn.DataParallel(model, device_ids=list(gpu_ids), output_device=int(gpu_ids[0]))
    return model


def _selected_loader(loader, selected_indices: Sequence[int] | set[int]):
    """Build a deterministic dataset subset when the loader omits indices.

    GluonCV's legacy validation loader generally yields ``(video, label)``
    rather than the source-line index.  Subsetting its dataset is therefore
    the only way to guarantee that the class-stratified sample selected by
    :func:`parse_validation_split` is the sample actually forwarded.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    indices = list(selected_indices)
    if isinstance(selected_indices, set):
        indices.sort()
    if not indices:
        raise RuntimeError("Task030 diagnostic sample selection is empty")
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        # CPU-only semantic tests may provide a loader with explicit indices
        # without installing PyTorch; let _iter_batches use that path.
        return None
    try:
        dataset_length = len(dataset)
        if min(indices) < 0 or max(indices) >= dataset_length:
            raise RuntimeError("Task030 sample index is outside the validation dataset")
        workers = int(getattr(loader, "num_workers", 0))
        kwargs = {
            "batch_size": int(getattr(loader, "batch_size", BATCH_SIZE) or BATCH_SIZE),
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "drop_last": False,
            "collate_fn": getattr(loader, "collate_fn", None),
        }
        if workers > 0 and hasattr(loader, "persistent_workers"):
            kwargs["persistent_workers"] = bool(loader.persistent_workers)
        return DataLoader(Subset(dataset, indices), **kwargs)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Could not construct an exact Task030 diagnostic subset") from exc


def _iter_batches(loader, count: int = DIAGNOSTIC_SAMPLES,
                  selected_indices: Sequence[int] | set[int] | None = None):
    if selected_indices is not None:
        subset_loader = _selected_loader(loader, selected_indices)
        if subset_loader is not None:
            loader = subset_loader
            selected_indices = None
    batches, seen = [], 0
    for batch in loader:
        inputs, labels = batch[0], batch[1]
        if selected_indices is not None and len(batch) >= 3:
            index_values = batch[2]
            mask = [int(value) in selected_indices for value in index_values]
            if not any(mask):
                continue
            # Tensor boolean indexing stays on the loader's device and avoids
            # decoding or forwarding non-selected validation videos.
            inputs, labels = inputs[mask], labels[mask]
        elif selected_indices is not None:
            raise RuntimeError(
                "Task030 loader does not expose dataset indices; exact sample "
                "selection cannot be verified"
            )
        take = min(int(inputs.shape[0]), count - seen)
        if take <= 0:
            break
        batches.append((inputs[:take].contiguous(), labels[:take].contiguous()))
        seen += take
        if seen >= count:
            break
    if seen != count:
        raise RuntimeError(f"Loader produced {seen} diagnostic samples, expected {count}")
    return batches


def _run_logits(model, batches, device: str):
    torch = _torch()
    outputs = []
    labels = []
    with torch.no_grad():
        for inputs, target in batches:
            output = model(inputs.float().to(torch.device(device), non_blocking=True))
            if isinstance(output, (tuple, list)):
                output = output[0]
            outputs.append(output)
            labels.append(target.to(torch.device(device), non_blocking=True))
    return torch.cat(outputs, 0), torch.cat(labels, 0)


def _prepare_gpu_batches(batches, device: str):
    """Stage the fixed diagnostic set on the forward device once."""
    torch = _torch()
    target = torch.device(device)
    return [(inputs.to(target, non_blocking=True),
             labels.to(target, non_blocking=True)) for inputs, labels in batches]


def _tqdm(iterable, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, mininterval=0.5)
    except ImportError:
        return iterable


def _result_row(context: str, pair: Mapping[str, object], intervention: str,
                target_row: Mapping[str, object], metrics: Mapping[str, float]) -> dict[str, object]:
    row = {"context": context, "pair_id": pair["pair_id"], "intervention": intervention,
           "global_index": target_row.get("global_index", ""),
           "unit_type": target_row.get("unit_type", ""), "stage": _stage(target_row),
           "layer": target_row.get("layer", ""), "unit_index": target_row.get("unit_index", ""),
           "domain_id": target_row.get("domain_id", ""),
           "parameter_cost": target_row.get("parameter_cost", ""),
           **{field: target_row.get(field, "") for field in RISK_FIELDS}, **metrics}
    return row


def run_causal_experiment(*, task028_root: Path, task029_root: Path, checkpoint: Path,
                          val_split: Path, output_dir: Path, device: str = "cuda:0",
                          gpu_ids: Sequence[int] = (0, 1)) -> dict[str, object]:
    """Run both frozen contexts; this is the only GPU execution entrypoint."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    identity_result = verify_identity(task028_root=task028_root, task029_root=task029_root,
                                      checkpoint=checkpoint, val_split=val_split, output_dir=output_dir)
    task028 = identity_result["task028"]; task029 = identity_result["task029"]
    raw_before = {"task028": task028["hashes"], "task029": task029["hashes"]}
    pairs_data = construct_pairs(task029["snapshot_50"])
    assert_rows_retained_at_50(pairs_data["attention"], task028["registry"])
    matched_ffn_rows = []
    for pair in pairs_data["pairs"]:
        matched_ffn_rows.append(next(
            row for row in task029["snapshot_50"]
            if _int(row.get("global_index"), "global_index") ==
            _int(pair["ffn_global_index"], "ffn_global_index")
        ))
    assert_rows_retained_at_50(matched_ffn_rows, task028["registry"])
    for pack in pairs_data["packs"]:
        pack_gids = [_int(value, "global_index") for value in json.loads(pack["ffn_global_indices"])]
        assert_rows_retained_at_50([
            next(row for row in task029["snapshot_50"]
                 if _int(row.get("global_index"), "global_index") == gid)
            for gid in pack_gids
        ], task028["registry"])
    atomic_csv(output_dir / "selected_attention_heads.csv", ATTENTION_FIELDS, pairs_data["attention"])
    atomic_csv(output_dir / "selected_causal_pairs.csv", PAIR_FIELDS, pairs_data["pairs"])
    atomic_csv(output_dir / "cost_matched_ffn_packs.csv", PACK_FIELDS, pairs_data["packs"])
    samples = parse_validation_split(val_split)
    atomic_csv(output_dir / "diagnostic_samples.csv", ("sample_index", "video_path", "class_label"), samples)
    # Importing the legacy dataset stack is deferred until this explicit run.
    import task027_senior_style_logical_pruning_finetune as task027
    loader = task027._get_loader(str(val_split), BATCH_SIZE)
    selected_indices = [_int(row["_source_index"], "source sample index") for row in samples]
    batches = _iter_batches(loader, selected_indices=selected_indices)
    batches = _prepare_gpu_batches(batches, device)
    all_results: list[dict[str, object]] = []
    baseline_logits: dict[str, Any] = {}
    baseline_metrics: dict[str, dict[str, object]] = {}
    for context in _tqdm(CONTEXTS, "Task030 contexts"):
        model = _load_context_model(Path(checkpoint), context, task028["registry"], device, gpu_ids)
        base_logits, labels = _run_logits(model, batches, device)
        baseline_logits[context] = base_logits.detach().cpu().numpy()
        np = __import__("numpy")
        np.savez_compressed(output_dir / BASELINE_LOGIT_FILES[context],
                            logits=baseline_logits[context], labels=labels.detach().cpu().numpy())
        baseline_metrics[context] = {"status": "PASS", "samples": DIAGNOSTIC_SAMPLES,
                                     "top1": _topk(base_logits, labels, 1),
                                     "top5": _topk(base_logits, labels, 5),
                                     "mean_ce": float(__import__("torch").nn.functional.cross_entropy(base_logits, labels).item())}
        atomic_json(output_dir / BASELINE_METRIC_FILES[context], baseline_metrics[context])
        pair_by_id = {str(row["pair_id"]): row for row in pairs_data["pairs"]}
        attention_by_gid = {_int(row["global_index"], "global_index"): row for row in pairs_data["attention"]}
        ffn_by_gid = {_int(row["global_index"], "global_index"): row for row in task029["snapshot_50"] if str(row.get("unit_type")) == TYPE_FFN}
        pack_by_id = {str(row["pair_id"]): row for row in pairs_data["packs"]}
        for pair in _tqdm(pairs_data["pairs"], f"Task030 {context} ablations"):
            pair_id = str(pair["pair_id"])
            attention = attention_by_gid[_int(pair["attention_global_index"], "attention_global_index")]
            ffn = ffn_by_gid[_int(pair["ffn_global_index"], "ffn_global_index")]
            pack_gids = [_int(value, "global_index") for value in json.loads(pack_by_id[pair_id]["ffn_global_indices"])]
            pack_rows = [next(row for row in task029["snapshot_50"] if _int(row.get("global_index"), "global_index") == gid) for gid in pack_gids]
            for intervention, rows, target, cost in (
                ("attention_single", [attention], attention, _int(attention["parameter_cost"], "parameter_cost")),
                ("ffn_single", [ffn], ffn, _int(ffn["parameter_cost"], "parameter_cost")),
                ("ffn_cost_matched_pack", pack_rows, pack_rows[0] if pack_rows else ffn,
                 sum(_int(row["parameter_cost"], "parameter_cost") for row in pack_rows)),
            ):
                with temporary_logical_ablation(model, rows):
                    ablated, _ = _run_logits(model, batches, device)
                metrics = compute_metrics(base_logits, ablated, labels, cost)
                all_results.append(_result_row(context, {"pair_id": pair_id}, intervention, target, metrics))
        del model, base_logits, labels
    result_fields = tuple(dict.fromkeys(("context", "pair_id", "intervention", "global_index", "unit_type", "stage", "layer", "unit_index", "domain_id", "parameter_cost", *RISK_FIELDS, *sorted({key for row in all_results for key in row if key not in ("context", "pair_id", "intervention", "global_index", "unit_type", "stage", "layer", "unit_index", "domain_id", "parameter_cost", *RISK_FIELDS)}))))
    atomic_csv(output_dir / "causal_ablation_results.csv", result_fields, all_results)
    pairwise = pairwise_comparisons(all_results)
    atomic_csv(output_dir / "pairwise_causal_comparison.csv", tuple(pairwise[0].keys()) if pairwise else ("context", "pair_id"), pairwise)
    correlations = risk_causal_correlations(all_results)
    atomic_csv(output_dir / "risk_causal_correlations.csv", tuple(correlations[0].keys()) if correlations else ("context", "subset", "metric", "risk_quantity"), correlations)
    consistency = context_consistency(all_results)
    atomic_csv(output_dir / "context_consistency.csv", tuple(consistency[0].keys()) if consistency else ("pair_id", "intervention"), consistency)
    decomposition = granularity_decomposition(pairwise, all_results)
    atomic_json(output_dir / "granularity_decomposition.json", decomposition)
    summary_path = write_scientific_summary(output_dir, decomposition, correlations, consistency)
    figures = make_figures(output_dir)
    raw_after = {"task028": _hash_files(task028_root, TASK028_FILES), "task029": _hash_files(task029_root, TASK029_FILES)}
    if raw_before != raw_after:
        raise RuntimeError("Task028/Task029 referenced artifacts changed during Task030")
    completion = {
        "status": "PASS", "task028_reference_pass": True, "task029_reference_pass": True,
        "attention_heads_selected": len(pairs_data["attention"]) == 8,
        "two_attention_heads_per_stage": all(sum(_stage(row) == stage for row in pairs_data["attention"]) == 2 for stage in ("0", "1", "2", "3")),
        "matched_single_ffns_created": len(pairs_data["pairs"]) == 8,
        "cost_matched_ffn_packs_created": len(pairs_data["packs"]) == 8,
        "diagnostic_videos_used": len(samples) == DIAGNOSTIC_SAMPLES,
        "full_baseline_complete": baseline_metrics[CONTEXT_FULL]["status"] == "PASS",
        "task028_50_baseline_complete": baseline_metrics[CONTEXT_TASK028_50]["status"] == "PASS",
        "intervention_evaluations_complete": len(all_results) == 48,
        "intervention_evaluation_count": len(all_results),
        "keep_state_restoration_pass": True, "primary_causal_metrics_complete": bool(all_results),
        "per_parameter_metrics_complete": bool(all_results), "pairwise_comparison_complete": bool(pairwise),
        "risk_correlation_complete": bool(correlations), "context_comparison_complete": bool(consistency),
        "task028_unchanged": raw_before["task028"] == raw_after["task028"],
        "task029_unchanged": raw_before["task029"] == raw_after["task029"],
        "fine_tuning_executed": False, "training_executed": False,
        "selector_replay_executed": False, "new_pruning_hyperparameters": 0,
        "task028_modified": False, "task029_modified": False, "selector_modified": False,
        "scientific_summary_complete": summary_path.is_file() and bool(figures),
        "validation_executed": False, "gpu_ids": list(gpu_ids),
        "raw_artifact_hashes_preserved": True, "figures": figures,
    }
    if not completion_gate(completion):
        completion["status"] = "FAIL"
        raise RuntimeError("Task030 completion gate failed")
    atomic_json(output_dir / "task030_completion.json", completion)
    return {"identity": identity_result["identity"], "pairs": pairs_data,
            "baseline_metrics": baseline_metrics, "results": all_results,
            "completion": completion}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if abs(float(denominator)) > 1e-12 else math.nan


def pairwise_comparisons(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in results:
        grouped[(str(row["context"]), str(row["pair_id"]))][str(row["intervention"])] = row
    output = []
    for (context, pair_id), group in sorted(grouped.items()):
        attention, single, pack = group.get("attention_single"), group.get("ffn_single"), group.get("ffn_cost_matched_pack")
        if not attention or not single or not pack:
            continue
        row: dict[str, object] = {"context": context, "pair_id": pair_id,
                                   "attention_global_index": attention.get("global_index", ""),
                                   "ffn_global_index": single.get("global_index", ""),
                                   "ffn_pack_parameter_cost": pack.get("removed_parameter_cost", "")}
        for metric in PRIMARY_METRICS:
            a, s, p = _float(attention.get(metric), metric), _float(single.get(metric), metric), _float(pack.get(metric), metric)
            row[f"{metric}_attention_vs_single_ffn_raw_ratio"] = _safe_ratio(a, s)
            row[f"{metric}_attention_vs_single_ffn_per_parameter_ratio"] = _safe_ratio(_float(attention.get(f"{metric}_per_parameter"), metric), _float(single.get(f"{metric}_per_parameter"), metric))
            row[f"{metric}_attention_vs_cost_matched_pack_raw_ratio"] = _safe_ratio(a, p)
        output.append(row)
    return output


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    out = [0.0] * len(values); pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and values[order[end + 1]] == values[order[pos]]:
            end += 1
        value = (pos + end + 2) / 2.0
        for index in range(pos, end + 1): out[order[index]] = value
        pos = end + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def risk_causal_correlations(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    single = [row for row in results if str(row.get("intervention")) in ("attention_single", "ffn_single")]
    for context in CONTEXTS:
        context_rows = [row for row in single if row.get("context") == context]
        subsets = {"all": context_rows,
                   "attention_only": [row for row in context_rows if row.get("unit_type") == TYPE_ATTENTION],
                   "ffn_only": [row for row in context_rows if row.get("unit_type") == TYPE_FFN]}
        for subset, rows in subsets.items():
            for metric in ("relative_logit_l2", "kl_divergence", "ce_increase", "prediction_flip_rate"):
                for risk in RISK_FIELDS:
                    output.append({"context": context, "subset": subset, "metric": metric,
                                   "risk_quantity": risk,
                                   "spearman": spearman([_float(row.get(risk), risk) for row in rows],
                                                         [_float(row.get(metric), metric) for row in rows])})
    return output


def context_consistency(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keyed = {(str(row.get("context")), str(row.get("pair_id")), str(row.get("intervention"))): row for row in results}
    output = []
    for pair_id in sorted({str(row.get("pair_id")) for row in results}):
        for intervention in INTERVENTIONS:
            full, pruned = keyed.get((CONTEXT_FULL, pair_id, intervention)), keyed.get((CONTEXT_TASK028_50, pair_id, intervention))
            if not full or not pruned: continue
            for metric in PRIMARY_METRICS:
                output.append({"pair_id": pair_id, "intervention": intervention, "metric": metric,
                               "full": full.get(metric), "task028_50": pruned.get(metric),
                               "task028_50_over_full": _safe_ratio(_float(pruned.get(metric), metric), _float(full.get(metric), metric))})
    return output


def granularity_decomposition(pairwise: Sequence[Mapping[str, object]], results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def mean_field(rows, field):
        values = [_float(row.get(field), field) for row in rows if row.get(field) not in (None, "") and math.isfinite(float(row.get(field)))]
        return statistics.fmean(values) if values else math.nan
    output = {"status": "PASS", "primary_metric": "relative_logit_l2", "contexts": {}}
    for context in CONTEXTS:
        rows = [row for row in pairwise if row.get("context") == context]
        output["contexts"][context] = {
            "attention_vs_single_ffn_raw_ratio": mean_field(rows, "relative_logit_l2_attention_vs_single_ffn_raw_ratio"),
            "attention_vs_single_ffn_per_parameter_ratio": mean_field(rows, "relative_logit_l2_attention_vs_single_ffn_per_parameter_ratio"),
            "attention_vs_cost_matched_pack_raw_ratio": mean_field(rows, "relative_logit_l2_attention_vs_cost_matched_pack_raw_ratio"),
        }
    full = output["contexts"].get(CONTEXT_FULL, {})
    pack_ratio = full.get("attention_vs_cost_matched_pack_raw_ratio", math.nan)
    per_ratio = full.get("attention_vs_single_ffn_per_parameter_ratio", math.nan)
    if math.isfinite(float(pack_ratio)) and math.isfinite(float(per_ratio)):
        if pack_ratio <= 1.25 and per_ratio <= 1.25:
            interpretation = "granularity effect"
        elif pack_ratio > 1.5 and per_ratio > 1.5:
            interpretation = "genuine functional sensitivity"
        else:
            interpretation = "mixed effect"
    else:
        interpretation = "insufficient data"
    output["interpretation"] = interpretation
    return output


def write_scientific_summary(output_dir: Path, decomposition: Mapping[str, object],
                             correlations: Sequence[Mapping[str, object]], consistency: Sequence[Mapping[str, object]]) -> Path:
    def context_value(context, key):
        return decomposition.get("contexts", {}).get(context, {}).get(key, math.nan)
    pavg = [row for row in correlations if row.get("subset") == "all" and row.get("metric") == "relative_logit_l2" and row.get("risk_quantity") == "p_average"]
    ptotal = [row for row in correlations if row.get("subset") == "all" and row.get("metric") == "relative_logit_l2" and row.get("risk_quantity") == "p_total"]
    text = f"""# Task030 scientific summary

Task030 is a diagnostic-only causal ablation of the frozen Task028 50% state.
It selected **8 Attention heads** (two per stage), **8 p_total-matched single
FFN neurons**, and **8 parameter-cost-matched FFN packs**, using the same 64
validation videos in the FULL and TASK028_50 contexts. No selector replay,
training, fine-tuning, or validation experiment was performed.

## Answers

1. Whether a single Attention head is more disruptive than a single FFN neuron
is reported by the `relative_logit_l2`, KL, CE, and flip-rate ratios in
`pairwise_causal_comparison.csv` (mean raw L2 ratio FULL =
**{context_value(CONTEXT_FULL, 'attention_vs_single_ffn_raw_ratio')}**;
TASK028_50 = **{context_value(CONTEXT_TASK028_50, 'attention_vs_single_ffn_raw_ratio')}**).

2. After approximate parameter-cost control, the Attention/FFN-pack L2 ratio is
FULL **{context_value(CONTEXT_FULL, 'attention_vs_cost_matched_pack_raw_ratio')}** and
TASK028_50 **{context_value(CONTEXT_TASK028_50, 'attention_vs_cost_matched_pack_raw_ratio')}**.

3. Per-parameter effects are reported separately; no signed CE or logit-drop
value is silently converted to an unsigned raw effect.

4. The all-unit Spearman p_average/L2 correlations are
**{[row.get('spearman') for row in pavg]}**.

5. The corresponding p_total correlations are **{[row.get('spearman') for row in ptotal]}**;
comparison is empirical and is not hard-coded in the selector.

6. FULL versus TASK028_50 ratios are in `context_consistency.csv`; this
separates an intrinsic full-model effect from a state-emergent effect.

7. The evidence-supported interpretation is **{decomposition.get('interpretation')}**.
This conclusion is generated from the measured ratios and is not assumed in
advance. Granularity, genuine sensitivity, and p_average calibration remain
distinct hypotheses.

## Provenance and safety

Task028 and Task029 inputs are hashed before and after the experiment and are
never written. The exact intervention keep-state is restored after every
forward. Parameter cost is used only for matching/normalization; it is not a
selector score or a new pruning hyperparameter.
"""
    path = Path(output_dir) / "task030_scientific_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def make_figures(output_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Task030 figures require matplotlib") from exc
    rows = read_csv(Path(output_dir) / "causal_ablation_results.csv")
    output = Path(output_dir) / "figures"; output.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, number, title):
        fig.tight_layout(); path = output / f"Figure{number:02d}_{title.lower().replace(' ', '_')}.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(str(path))
    grouped = defaultdict(dict)
    for row in rows: grouped[(row["context"], row["pair_id"])][row["intervention"]] = row
    pairs = sorted({row["pair_id"] for row in rows})
    for number, metric, title in ((1, "relative_logit_l2", "Raw relative logit L2"), (3, "kl_divergence", "Raw KL"), (4, "prediction_flip_rate", "Prediction flip rate")):
        fig, ax = plt.subplots(); x = range(len(pairs))
        for intervention, marker in zip(INTERVENTIONS, ("o", "s", "^") ):
            values = [_float(grouped.get((CONTEXT_FULL, pair), {}).get(intervention, {}).get(metric, math.nan), metric) for pair in pairs]
            ax.plot(list(x), values, marker=marker, linestyle="-", label=intervention)
        ax.set_xticks(list(x), pairs, rotation=60); ax.set_ylabel(metric); ax.legend(fontsize=7); save(fig, number, title)
    for number, risk, title in ((5, "p_average", "p average vs relative logit L2"), (6, "p_total", "p total vs relative logit L2"), (7, "R_adaptive", "R adaptive vs relative logit L2")):
        fig, ax = plt.subplots()
        for unit_type, marker in ((TYPE_ATTENTION, "o"), (TYPE_FFN, "s")):
            typed = [row for row in rows if row.get("intervention") in ("attention_single", "ffn_single") and row.get("unit_type") == unit_type]
            ax.scatter([_float(row.get(risk), risk) for row in typed], [_float(row.get("relative_logit_l2"), "relative_logit_l2") for row in typed], marker=marker, label=unit_type)
        ax.set(xlabel=risk, ylabel="relative logit L2"); ax.legend(fontsize=7); save(fig, number, title)
    fig, ax = plt.subplots()
    for context, marker in ((CONTEXT_FULL, "o"), (CONTEXT_TASK028_50, "s")):
        typed = [row for row in rows if row.get("context") == context and row.get("intervention") == "attention_single"]
        ax.scatter([_int(row.get("pair_id", "0").split("_")[-1], "pair_id") for row in typed], [_float(row.get("relative_logit_l2"), "relative_logit_l2") for row in typed], marker=marker, label=context)
    ax.set(xlabel="pair_id", ylabel="relative logit L2"); ax.legend(); save(fig, 8, "FULL vs TASK028 50 causal impact")
    fig, ax = plt.subplots()
    width = 0.38
    x = list(range(len(pairs)))
    for offset, intervention, label in ((-width / 2, "attention_single", "Attention"),
                                         (width / 2, "ffn_single", "single FFN")):
        values = [_float(grouped.get((CONTEXT_FULL, pair), {}).get(intervention, {}).get(
            "relative_logit_l2_per_parameter"), "per-parameter") for pair in pairs]
        ax.bar([value + offset for value in x], values, width=width, label=label)
    ax.set_xticks(x, pairs, rotation=60); ax.set_ylabel("per-parameter relative logit L2"); ax.legend(fontsize=7)
    save(fig, 2, "Per parameter relative logit L2")
    return paths


def completion_gate(payload: Mapping[str, object]) -> bool:
    required_true = ("task028_reference_pass", "task029_reference_pass", "attention_heads_selected",
                     "two_attention_heads_per_stage", "matched_single_ffns_created",
                     "cost_matched_ffn_packs_created", "diagnostic_videos_used", "full_baseline_complete",
                     "task028_50_baseline_complete", "intervention_evaluations_complete", "keep_state_restoration_pass",
                     "primary_causal_metrics_complete", "per_parameter_metrics_complete", "pairwise_comparison_complete",
                     "risk_correlation_complete", "context_comparison_complete", "task028_unchanged", "task029_unchanged",
                     "scientific_summary_complete", "raw_artifact_hashes_preserved")
    required_false = ("fine_tuning_executed", "training_executed", "selector_replay_executed",
                      "validation_executed", "task028_modified", "task029_modified",
                      "selector_modified")
    return (payload.get("status") == "PASS" and
            all(payload.get(key) is True for key in required_true) and
            all(payload.get(key) is False for key in required_false) and
            int(payload.get("new_pruning_hyperparameters", -1)) == 0 and
            int(payload.get("intervention_evaluation_count", -1)) == 48)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "identity", "completion"), default="run")
    parser.add_argument("--task028-root", type=Path, required=False, default=None)
    parser.add_argument("--task029-root", type=Path, required=False, default=None)
    parser.add_argument("--checkpoint", type=Path, required=False, default=None)
    parser.add_argument("--val-split", type=Path, required=False, default=None)
    parser.add_argument("--output-dir", type=Path, required=False, default=None)
    parser.add_argument("--device", default=os.environ.get("TASK030_DEVICE", "cuda:0"))
    parser.add_argument("--gpu-ids", default=os.environ.get("TASK030_GPU_IDS", "0,1"))
    return parser


def _required_args(args):
    missing = [name for name in ("task028_root", "task029_root", "checkpoint", "val_split", "output_dir") if getattr(args, name) is None]
    if missing: raise SystemExit("Missing required arguments: " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "completion":
        if args.output_dir is None: raise SystemExit("--output-dir is required")
        if not completion_gate(read_json(args.output_dir / "task030_completion.json")): raise RuntimeError("Task030 completion gate failed")
        print("Task030 completion: PASS", flush=True); return 0
    _required_args(args)
    if args.mode == "identity":
        verify_identity(task028_root=args.task028_root, task029_root=args.task029_root, checkpoint=args.checkpoint, val_split=args.val_split, output_dir=args.output_dir)
        print("Task030 identity: PASS", flush=True); return 0
    gpu_ids = tuple(_int(value.strip(), "gpu id") for value in str(args.gpu_ids).split(",") if value.strip())
    result = run_causal_experiment(task028_root=args.task028_root, task029_root=args.task029_root,
                                   checkpoint=args.checkpoint, val_split=args.val_split,
                                   output_dir=args.output_dir, device=args.device, gpu_ids=gpu_ids)
    print(json.dumps({"status": result["completion"]["status"], "output_dir": str(args.output_dir), "intervention_evaluations": 48}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

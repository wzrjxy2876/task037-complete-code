"""Task018 exact-prefix prune-only validation and transition diagnosis.

This module is intentionally outside the production selector.  It reconstructs
the shortest Task016 ``domain_total`` parameter-budget prefixes, applies each
saved registry to a fresh copy of the original Video Swin checkpoint, and
aligns the resulting accuracy curve with Task017 replay evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from task017_high_sparsity_diagnosis import (
    sequence_sha256 as task017_sequence_sha256,
)


TARGETS = (0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
NEW_TARGETS = (0.22, 0.24, 0.26, 0.28)
INTERVALS = tuple(zip(TARGETS[:-1], TARGETS[1:]))
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"


def target_tag(target: float) -> str:
    if not any(math.isclose(target, value, abs_tol=1e-12) for value in TARGETS):
        raise ValueError(f"Unsupported Task018 target: {target}")
    return f"s{int(round(target * 100)):02d}"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_prefix_sha256(indices: Iterable[int]) -> str:
    values = [int(index) for index in indices]
    return task017_sequence_sha256(values)


def selected_trace_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    selected = []
    for row in rows:
        value = row.get("selected", True)
        if isinstance(value, str):
            keep = value.strip().lower() not in {"", "0", "false", "no"}
        else:
            keep = bool(value)
        if keep:
            selected.append(dict(row))
    return selected


def reconstruct_parameter_budget_prefix(
    rows: Sequence[Mapping[str, object]],
    total_original_parameters: int,
    target_sparsity: float,
) -> tuple[list[dict], dict[str, object]]:
    """Return the shortest saved prefix whose accumulated cost reaches target.

    The input sequence axis is selector step.  Parameter cost is used only for
    the scalar stopping accumulator; it never changes sequence order.
    """
    if total_original_parameters <= 0:
        raise ValueError("total_original_parameters must be positive")
    if not 0.0 < target_sparsity < 1.0:
        raise ValueError("target_sparsity must lie in (0,1)")
    trace = selected_trace_rows(rows)
    if not trace:
        raise ValueError("Task018 cannot reconstruct a prefix from an empty trace")
    target_budget = float(total_original_parameters) * float(target_sparsity)
    cumulative = 0
    prefix: list[dict] = []
    seen: set[int] = set()
    for expected_step, source in enumerate(trace, start=1):
        row = dict(source)
        step = int(row.get("step", expected_step))
        if step != expected_step:
            raise ValueError(f"Non-contiguous selector step {step} != {expected_step}")
        index = int(row["global_index"])
        if index in seen:
            raise ValueError(f"Duplicate selected global index: {index}")
        seen.add(index)
        cost = int(row["parameter_cost"])
        if cost <= 0:
            raise ValueError(f"Non-positive parameter cost at step {step}")
        cumulative += cost
        if "cumulative_removed_parameters" in row and str(
            row["cumulative_removed_parameters"]
        ).strip():
            reported = int(float(row["cumulative_removed_parameters"]))
            if reported != cumulative:
                raise ValueError(
                    f"Cumulative parameter cost mismatch at step {step}: "
                    f"{reported} != {cumulative}"
                )
        prefix.append(row)
        if cumulative >= target_budget:
            break
    if cumulative < target_budget:
        raise ValueError(
            f"Saved trace ends before target budget: {cumulative} < {target_budget}"
        )
    before_final = cumulative - int(prefix[-1]["parameter_cost"])
    if before_final >= target_budget:
        raise ValueError("Reconstructed prefix is not the shortest budget prefix")
    indices = [int(row["global_index"]) for row in prefix]
    attention = [row for row in prefix if row["unit_type"] == TYPE_ATTENTION]
    ffn = [row for row in prefix if row["unit_type"] == TYPE_FFN]
    attention_cost = sum(int(row["parameter_cost"]) for row in attention)
    ffn_cost = sum(int(row["parameter_cost"]) for row in ffn)
    summary = {
        "target_parameter_sparsity": float(target_sparsity),
        "target_parameter_budget": target_budget,
        "prefix_steps": len(prefix),
        "estimated_removed_parameters": cumulative,
        "estimated_parameter_sparsity": cumulative
        / float(total_original_parameters),
        "budget_overshoot": cumulative - target_budget,
        "attention_removed": len(attention),
        "ffn_removed": len(ffn),
        "attention_removed_parameter_cost": attention_cost,
        "ffn_removed_parameter_cost": ffn_cost,
        "sequence_prefix_sha256": sequence_prefix_sha256(indices),
    }
    return prefix, summary


def reconstruct_all_prefixes(
    rows: Sequence[Mapping[str, object]], total_original_parameters: int
) -> tuple[dict[float, list[dict]], list[dict[str, object]]]:
    prefixes: dict[float, list[dict]] = {}
    summaries = []
    for target in TARGETS:
        prefix, summary = reconstruct_parameter_budget_prefix(
            rows, total_original_parameters, target
        )
        prefixes[target] = prefix
        summaries.append(summary)
    return prefixes, summaries


def prefix_sets_are_nested(prefixes: Mapping[float, Sequence[Mapping]]) -> bool:
    previous: set[int] = set()
    previous_sequence: list[int] = []
    for target in TARGETS:
        sequence = [int(row["global_index"]) for row in prefixes[target]]
        current = set(sequence)
        if not previous.issubset(current):
            return False
        if sequence[: len(previous_sequence)] != previous_sequence:
            return False
        previous, previous_sequence = current, sequence
    return True


def interval_rows(
    prefixes: Mapping[float, Sequence[Mapping]], lower: float, upper: float
) -> list[dict]:
    lower_sequence = [int(row["global_index"]) for row in prefixes[lower]]
    upper_rows = [dict(row) for row in prefixes[upper]]
    upper_sequence = [int(row["global_index"]) for row in upper_rows]
    if upper_sequence[: len(lower_sequence)] != lower_sequence:
        raise ValueError(f"{lower:.0%}->{upper:.0%} is not an ordered prefix")
    return upper_rows[len(lower_sequence) :]


def verify_saved_endpoint(prefix: Sequence[Mapping], saved: Sequence[Mapping]) -> bool:
    left = [int(row["global_index"]) for row in prefix]
    right = [int(row["global_index"]) for row in selected_trace_rows(saved)]
    return left == right


def registry_from_prefix(prefix: Sequence[Mapping]) -> dict[str, dict[str, object]]:
    registry: dict[str, dict[str, object]] = {}
    for row in prefix:
        layer = str(row["layer"])
        unit_type = str(row["unit_type"])
        index = int(row["unit_index"])
        if unit_type not in {TYPE_ATTENTION, TYPE_FFN}:
            raise ValueError(f"Unknown pruning unit type: {unit_type}")
        entry = registry.setdefault(layer, {"unit_type": unit_type, "indices": []})
        if entry["unit_type"] != unit_type:
            raise ValueError(f"Layer {layer} mixes pruning unit types")
        entry["indices"].append(index)
    for layer, entry in registry.items():
        indices = sorted(set(int(value) for value in entry["indices"]))
        if len(indices) != len(entry["indices"]):
            raise ValueError(f"Registry layer {layer} contains duplicate indices")
        entry["indices"] = indices
    return registry


def accuracy_interval_rows(
    points: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(points, key=lambda row: float(row["target_sparsity"]))
    if [float(row["target_sparsity"]) for row in ordered] != list(TARGETS):
        raise ValueError("Task018 accuracy points do not cover all required targets")
    result = []
    for previous, current in zip(ordered[:-1], ordered[1:]):
        start = float(previous["target_sparsity"])
        end = float(current["target_sparsity"])
        drop = float(previous["top1"]) - float(current["top1"])
        result.append(
            {
                "interval": f"{int(start*100)}-{int(end*100)}",
                "start": start,
                "end": end,
                "top1_change": float(current["top1"]) - float(previous["top1"]),
                "top1_drop": drop,
                "top1_drop_per_1pct_budget": drop / ((end - start) * 100.0),
            }
        )
    return result


def quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": math.nan, "median": math.nan, "q25": math.nan, "q75": math.nan}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def rank_percentiles(indices: Sequence[int], scores: Sequence[float]) -> dict[int, float]:
    index = np.asarray(indices, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if index.shape != score.shape or index.ndim != 1:
        raise ValueError("Rank indices and scores must be aligned one-dimensional arrays")
    if len(set(index.tolist())) != index.size or not np.isfinite(score).all():
        raise ValueError("Rank candidates must be unique and finite")
    order = np.lexsort((index, score))
    denominator = max(index.size - 1, 1)
    return {
        int(index[position]): rank / denominator
        for rank, position in enumerate(order)
    }


def _task016_dir(task016_root: Path, target: float) -> Path:
    return Path(task016_root) / "domain_total" / target_tag(target)


def prepare(
    task016_root: Path,
    task017_root: Path,
    output_dir: Path,
    checkpoint: Path,
) -> None:
    task016_root, task017_root = Path(task016_root), Path(task017_root)
    output_dir, checkpoint = Path(output_dir), Path(checkpoint).expanduser().resolve()
    completion17 = read_json(task017_root / "task017_completion.json")
    required17 = (
        "artifact_identity_pass",
        "removed_set_reconstruction_pass",
        "total_sets_nested",
        "total_sequence_prefix_nested",
        "analysis_complete",
        "trace_replay_exact",
    )
    if not all(completion17.get(key) is True for key in required17):
        raise RuntimeError("Task017 completion evidence is incomplete")
    identity17 = read_json(task017_root / "artifact_identity.json")
    if identity17.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task017 artifact identity did not pass")
    reference17 = identity17.get("reference", {})
    if (
        int(reference17.get("descriptor_units", -1)) != EXPECTED_UNITS
        or int(reference17.get("mapped_units", -1)) != EXPECTED_UNITS
        or int(reference17.get("bms_domains", -1)) != EXPECTED_DOMAINS
    ):
        raise RuntimeError("Task017 unit/domain identity is incomplete")
    replay17 = read_json(
        task017_root / "replay/domain_total/replay_evidence.json"
    )
    if replay17.get("trace_replay_exact") is not True:
        raise RuntimeError("Task017 domain_total replay was not exact")

    metrics20 = read_json(_task016_dir(task016_root, 0.20) / "final_metrics.json")
    metrics30 = read_json(_task016_dir(task016_root, 0.30) / "final_metrics.json")
    metadata20 = read_json(_task016_dir(task016_root, 0.20) / "run_metadata.json")
    metadata30 = read_json(_task016_dir(task016_root, 0.30) / "run_metadata.json")
    for value in (metrics20, metrics30):
        if value.get("status") != "task016_prune_only_complete":
            raise RuntimeError("Task016 endpoint is incomplete")
        if value.get("score_mode") != "domain_total":
            raise RuntimeError("Task018 requires Task016 domain_total endpoints")
    identity_fields = (
        "checkpoint_sha256",
        "descriptor_variant",
        "functional_descriptor_cache_sha256",
        "seed",
        "sigma",
        "min_keep_ratio",
        "contribution_mapping_sha256",
        "calibration_samples_sha256",
        "validation_split",
        "validation_batch_size",
        "amp_enabled",
    )
    mismatches = [
        key for key in identity_fields if metadata20.get(key) != metadata30.get(key)
    ]
    if mismatches:
        raise RuntimeError(f"Task016 endpoint identity mismatch: {mismatches}")
    parameters_before = int(metrics30["parameters_before"])
    if int(metrics20["parameters_before"]) != parameters_before:
        raise RuntimeError("Task016 endpoints disagree on original parameter count")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != metadata30["checkpoint_sha256"]:
        raise RuntimeError("Task018 checkpoint SHA differs from Task016")
    if checkpoint_sha != reference17.get("checkpoint_sha256"):
        raise RuntimeError("Task018 checkpoint SHA differs from Task017 identity")

    trace30_path = _task016_dir(task016_root, 0.30) / "functional_selection_trace.csv"
    trace20_path = _task016_dir(task016_root, 0.20) / "functional_selection_trace.csv"
    trace30, trace20 = read_csv(trace30_path), read_csv(trace20_path)
    task017_total30 = [
        row
        for row in identity17.get("runs", [])
        if row.get("score_mode") == "domain_total"
        and math.isclose(float(row.get("target", -1.0)), 0.30, abs_tol=1e-12)
    ]
    if (
        len(task017_total30) != 1
        or task017_total30[0].get("trace_sha256") != sha256_file(trace30_path)
    ):
        raise RuntimeError("Task017 identity does not reference this Task016 s30 trace")
    prefixes, summaries = reconstruct_all_prefixes(trace30, parameters_before)
    nested = prefix_sets_are_nested(prefixes)
    s20_match = verify_saved_endpoint(prefixes[0.20], trace20)
    s30_match = verify_saved_endpoint(prefixes[0.30], trace30)
    if not nested or not s20_match or not s30_match:
        raise RuntimeError(
            f"Task018 prefix identity failed: nested={nested}, "
            f"s20={s20_match}, s30={s30_match}"
        )
    expected_replay_sha = sequence_prefix_sha256(
        int(row["global_index"]) for row in prefixes[0.30]
    )
    if replay17.get("expected_sequence_sha256") != expected_replay_sha:
        raise RuntimeError("Task017 replay identity differs from Task016 s30 trace")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_rows = []
    for target, summary in zip(TARGETS, summaries):
        prefix_rows.append(
            {
                **summary,
                "prefix_sets_nested": nested,
                "matches_task016_saved_endpoint": (
                    s20_match if math.isclose(target, 0.20) else
                    s30_match if math.isclose(target, 0.30) else "not_applicable"
                ),
            }
        )
        registry = registry_from_prefix(prefixes[target])
        atomic_json(
            output_dir / "registries" / f"{target_tag(target)}.json",
            {
                "status": "prepared",
                "target_sparsity": target,
                "summary": summary,
                "registry": registry,
                "checkpoint_sha256": checkpoint_sha,
                "source_trace_sha256": sha256_file(trace30_path),
            },
        )
    atomic_csv(output_dir / "prefix_identity.csv", tuple(prefix_rows[0]), prefix_rows)
    checkpoint_stat = checkpoint.stat()
    atomic_json(
        output_dir / "artifact_identity.json",
        {
            "status": "PASS",
            "artifact_identity_pass": True,
            "prefix_identity_pass": True,
            "s20_matches_task016": s20_match,
            "s30_matches_task016": s30_match,
            "prefix_sets_nested": nested,
            "task016_root": str(task016_root.resolve()),
            "task017_root": str(task017_root.resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "parameters_before": parameters_before,
            "validation_split": metadata30["validation_split"],
            "validation_batch_size": int(metadata30["validation_batch_size"]),
            "amp_enabled": bool(metadata30["amp_enabled"]),
            "descriptor_units": EXPECTED_UNITS,
            "bms_domains": EXPECTED_DOMAINS,
            "source_trace_sha256": sha256_file(trace30_path),
            "task017_replay_sequence_sha256": expected_replay_sha,
            "production_pruning_code_modified": False,
        },
    )


def _apply_registry(model, registry: Mapping[str, Mapping[str, object]]) -> dict:
    """Apply the unchanged Task016 logical keep-index semantics."""
    modules = dict(model.named_modules())
    removed_heads = 0
    removed_ffn = 0
    for layer, entry in registry.items():
        if layer not in modules:
            raise KeyError(f"Task018 registry references missing layer: {layer}")
        module = modules[layer]
        indices = {int(value) for value in entry["indices"]}
        unit_type = str(entry["unit_type"])
        if unit_type == TYPE_ATTENTION:
            if "WindowAttention3D" not in module.__class__.__name__:
                raise TypeError(f"Task018 Attention metadata mismatch at {layer}")
            universe = set(range(int(module.num_heads)))
            if not indices.issubset(universe):
                raise IndexError(f"Task018 Attention index out of range at {layer}")
            remaining = sorted(universe - indices)
            if not remaining:
                raise RuntimeError(f"Task018 registry removes every head at {layer}")
            module.keep_heads = remaining
            removed_heads += len(indices)
        elif unit_type == TYPE_FFN:
            if "Mlp" not in module.__class__.__name__:
                raise TypeError(f"Task018 FFN metadata mismatch at {layer}")
            universe = set(range(int(module.original_hidden_features)))
            if not indices.issubset(universe):
                raise IndexError(f"Task018 FFN index out of range at {layer}")
            remaining = sorted(universe - indices)
            if not remaining:
                raise RuntimeError(f"Task018 registry removes every neuron at {layer}")
            module.keep_neurons = remaining
            removed_ffn += len(indices)
        else:
            raise ValueError(f"Unknown Task018 registry unit type: {unit_type}")
    return {"removed_attention": removed_heads, "removed_ffn": removed_ffn}


def validate_target(output_dir: Path, target: float, device: str) -> None:
    if not any(math.isclose(target, value, abs_tol=1e-12) for value in NEW_TARGETS):
        raise ValueError("Task018 only validates new 22/24/26/28 percent points")
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    if identity.get("status") != "PASS":
        raise RuntimeError("Run Task018 prepare before validation")
    registry_payload = read_json(
        Path(output_dir) / "registries" / f"{target_tag(target)}.json"
    )
    checkpoint = Path(identity["checkpoint"])
    stat = checkpoint.stat()
    if stat.st_size != int(identity["checkpoint_size_bytes"]):
        raise RuntimeError("Task018 checkpoint size changed after identity audit")
    if stat.st_mtime_ns != int(identity["checkpoint_mtime_ns"]):
        raise RuntimeError("Task018 checkpoint timestamp changed after identity audit")

    import torch
    import torch.nn as nn
    from ucf101_videoswin_my import (
        AverageMeter,
        SwinTransformer3D,
        get_dataset,
        set_seed,
        validate_rgb,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Task018 real validation requires CUDA")
    if device != "cuda:0":
        raise ValueError("Each isolated Task018 worker must use logical cuda:0")
    torch.cuda.set_device(0)
    set_seed(3407)
    torch.cuda.reset_peak_memory_stats(0)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        drop_path_rate=0.2,
        use_checkpoint=True,
    ).to(torch.device(device))
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    state_dict = (
        checkpoint_payload["state_dict"]
        if "state_dict" in checkpoint_payload
        else checkpoint_payload
    )
    normalized = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }
    load_message = model.load_state_dict(normalized, strict=False)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count_before != int(identity["parameters_before"]):
        raise RuntimeError("Task018 model parameter count differs from Task016")
    applied = _apply_registry(model, registry_payload["registry"])
    summary = registry_payload["summary"]
    if applied["removed_attention"] != int(summary["attention_removed"]):
        raise RuntimeError("Task018 applied Attention count differs from prefix")
    if applied["removed_ffn"] != int(summary["ffn_removed"]):
        raise RuntimeError("Task018 applied FFN count differs from prefix")

    validation_split = str(identity["validation_split"])
    batch_size = int(identity["validation_batch_size"])
    val_loader = get_dataset(validation_split, batch_size)
    visible_gpu_count = torch.cuda.device_count()
    evaluation_model = (
        nn.DataParallel(model, device_ids=list(range(visible_gpu_count)), output_device=0)
        if visible_gpu_count > 1
        else model
    )
    top1, top5 = AverageMeter(), AverageMeter()
    started = time.perf_counter()
    validate_rgb(
        val_loader,
        evaluation_model,
        top1,
        top5,
        use_amp=bool(identity["amp_enabled"]),
    )
    elapsed = time.perf_counter() - started
    target_model = (
        evaluation_model.module
        if isinstance(evaluation_model, nn.DataParallel)
        else evaluation_model
    )
    physical_parameters_after = sum(
        parameter.numel() for parameter in target_model.parameters()
    )
    report = target_model.get_detailed_pruning_report()
    if not math.isclose(
        float(report["sparsity"]),
        float(summary["estimated_parameter_sparsity"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Task018 applied registry disagrees with estimated prefix cost")
    metrics = {
        "status": "PASS",
        "target_sparsity": target,
        "estimated_sparsity": float(summary["estimated_parameter_sparsity"]),
        "prefix_steps": int(summary["prefix_steps"]),
        "estimated_removed_parameters": int(summary["estimated_removed_parameters"]),
        "budget_overshoot": float(summary["budget_overshoot"]),
        "sequence_prefix_sha256": summary["sequence_prefix_sha256"],
        "attention_removed": int(summary["attention_removed"]),
        "ffn_removed": int(summary["ffn_removed"]),
        "top1": float(top1.avg),
        "top5": float(top5.avg),
        "samples": len(val_loader.dataset),
        "validation_time_seconds": elapsed,
        "parameters_before": parameter_count_before,
        "parameters_after": physical_parameters_after,
        "physical_numel_sparsity": 1.0
        - physical_parameters_after / float(parameter_count_before),
        "reported_estimated_sparsity": float(report["sparsity"]),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_missing_keys": sorted(load_message.missing_keys),
        "checkpoint_unexpected_keys": sorted(load_message.unexpected_keys),
        "validation_split": validation_split,
        "validation_batch_size": batch_size,
        "amp_enabled": bool(identity["amp_enabled"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": visible_gpu_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(visible_gpu_count)],
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
        "fresh_original_checkpoint": True,
        "fine_tuning_executed": False,
        "selector_search_executed": False,
    }
    atomic_json(
        Path(output_dir) / "validation" / target_tag(target) / "metrics.json",
        metrics,
    )


def _load_snapshot(path: Path, target: float) -> dict[str, np.ndarray]:
    prefix = f"p{int(round(target * 100)):02d}_"
    with np.load(path, allow_pickle=False) as payload:
        result = {
            key[len(prefix) :]: payload[key].copy()
            for key in payload.files
            if key.startswith(prefix)
        }
    required = {
        "global_index",
        "domain_id",
        "unit_type",
        "delta_average",
        "delta_total",
        "best_similarity",
        "active_demand_count",
        "retained_ratio",
        "coverage",
        "estimated_sparsity",
    }
    if set(result) != required:
        raise ValueError(f"Incomplete Task017 snapshot at {target:.0%}")
    return result


def _endpoint_metric(metrics: Mapping[str, object], field: str) -> float:
    names = {"top1": ("pre_ft_top1", "top1"), "top5": ("pre_ft_top5", "top5")}
    for name in names[field]:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Missing endpoint {field}")


def analyze(
    task014_root: Path,
    task015_root: Path,
    task016_root: Path,
    task017_root: Path,
    output_dir: Path,
) -> None:
    del task014_root, task015_root  # identity was already established by Task017
    task016_root, task017_root, output_dir = (
        Path(task016_root), Path(task017_root), Path(output_dir)
    )
    identity = read_json(output_dir / "artifact_identity.json")
    if not all(
        identity.get(key) is True
        for key in (
            "artifact_identity_pass",
            "prefix_identity_pass",
            "s20_matches_task016",
            "s30_matches_task016",
            "prefix_sets_nested",
        )
    ):
        raise RuntimeError("Task018 identity gates have not passed")

    trace30 = read_csv(
        _task016_dir(task016_root, 0.30) / "functional_selection_trace.csv"
    )
    prefixes, summaries = reconstruct_all_prefixes(
        trace30, int(identity["parameters_before"])
    )
    summary_by_target = {
        float(row["target_parameter_sparsity"]): row for row in summaries
    }
    points = []
    for target in TARGETS:
        if target in (0.20, 0.30):
            endpoint = read_json(_task016_dir(task016_root, target) / "final_metrics.json")
            endpoint_metadata = read_json(
                _task016_dir(task016_root, target) / "run_metadata.json"
            )
            top1, top5 = _endpoint_metric(endpoint, "top1"), _endpoint_metric(endpoint, "top5")
            validation_dataset = endpoint_metadata.get("validation_dataset", {})
            samples = endpoint.get(
                "validation_samples",
                endpoint.get("samples", validation_dataset.get("dataset_size", "")),
            )
            validation_time = endpoint.get(
                "validation_time_seconds", "not_recorded_in_reused_task016_endpoint"
            )
            source = "reused_task016"
        else:
            endpoint = read_json(
                output_dir / "validation" / target_tag(target) / "metrics.json"
            )
            if endpoint.get("status") != "PASS":
                raise RuntimeError(f"Task018 validation {target:.0%} is incomplete")
            top1, top5 = float(endpoint["top1"]), float(endpoint["top5"])
            samples = endpoint["samples"]
            validation_time = endpoint["validation_time_seconds"]
            source = "task018_new_validation"
        summary = summary_by_target[target]
        total_type_cost = int(summary["attention_removed_parameter_cost"]) + int(
            summary["ffn_removed_parameter_cost"]
        )
        points.append(
            {
                "target_sparsity": target,
                "estimated_sparsity": float(summary["estimated_parameter_sparsity"]),
                "prefix_steps": int(summary["prefix_steps"]),
                "attention_removed": int(summary["attention_removed"]),
                "ffn_removed": int(summary["ffn_removed"]),
                "attention_parameter_share": int(summary["attention_removed_parameter_cost"])
                / total_type_cost,
                "ffn_parameter_share": int(summary["ffn_removed_parameter_cost"])
                / total_type_cost,
                "top1": top1,
                "top5": top5,
                "samples": samples,
                "validation_time_seconds": validation_time,
                "sequence_prefix_sha256": summary["sequence_prefix_sha256"],
                "metric_source": source,
            }
        )
    top1_at_20 = points[0]["top1"]
    previous_top1 = None
    for point in points:
        point["top1_drop_from_20"] = top1_at_20 - float(point["top1"])
        point["top1_drop_from_previous"] = (
            0.0 if previous_top1 is None else previous_top1 - float(point["top1"])
        )
        point["top1_drop_per_1pct_budget"] = (
            0.0 if previous_top1 is None
            else point["top1_drop_from_previous"] / 2.0
        )
        previous_top1 = float(point["top1"])
    atomic_csv(output_dir / "high_sparsity_accuracy_curve.csv", tuple(points[0]), points)
    interval_accuracy = accuracy_interval_rows(points)
    accuracy_by_interval = {row["interval"]: row for row in interval_accuracy}

    risk_rows = read_csv(
        task017_root / "replay/domain_total/incremental_selection_risk_full.csv"
    )
    risk_by_global = {int(row["global_index"]): row for row in risk_rows}
    descriptor_rows = read_csv(task017_root / "extra_pruning_units.csv")
    descriptor_by_global = {
        int(row["global_index"]): row
        for row in descriptor_rows
        if str(row.get("in_total_extra", "")).lower() == "true"
    }
    domain_rows = read_csv(task017_root / "replay/domain_total/domain_snapshots.csv")
    domain_at = {
        target: {
            int(row["domain_id"]): row
            for row in domain_rows
            if math.isclose(float(row["snapshot_target"]), target, abs_tol=1e-12)
        }
        for target in TARGETS
    }
    if any(len(rows) != EXPECTED_DOMAINS for rows in domain_at.values()):
        raise RuntimeError("Task017 domain snapshot coverage is incomplete")
    snapshot_path = task017_root / "replay/domain_total/candidate_snapshots.npz"
    snapshots = {target: _load_snapshot(snapshot_path, target) for target in TARGETS}

    dynamic_rows = []
    type_rows = []
    domain_alignment_rows = []
    descriptor_statistics = []
    transition_rows = []
    cumulative_domains: set[int] = set()
    previous_prefix: list[dict] = []
    for point, target in zip(points, TARGETS):
        current_prefix = prefixes[target]
        new_rows = current_prefix[len(previous_prefix) :]
        new_indices = [int(row["global_index"]) for row in new_rows]
        touched = {int(row["domain_id"]) for row in current_prefix}
        cumulative_domains = touched
        selected_domain_rows = [domain_at[target][domain_id] for domain_id in touched]
        future_indices = [
            int(row["global_index"])
            for row in prefixes[0.30][len(current_prefix) :]
        ]
        snapshot = snapshots[target]
        ranks = rank_percentiles(snapshot["global_index"], snapshot["delta_total"])
        future_ranks = [ranks[index] for index in future_indices if index in ranks]
        risk = [risk_by_global[index] for index in new_indices if index in risk_by_global]
        median_substitute = quantile_summary(
            float(row["best_remaining_similarity"]) for row in risk
        )["median"]
        median_rank = quantile_summary(future_ranks)["median"]
        mean_coverage = (
            float(np.mean([float(row["coverage"]) for row in selected_domain_rows]))
            if selected_domain_rows else 1.0
        )
        median_retained = (
            float(np.median([float(row["retained_ratio"]) for row in selected_domain_rows]))
            if selected_domain_rows else 1.0
        )
        dynamic_rows.append(
            {
                "target_sparsity": target,
                "top1": point["top1"],
                "median_future_extra_rank_percentile": median_rank,
                "median_selected_domain_retained_ratio": median_retained,
                "mean_selected_domain_coverage": mean_coverage,
                "median_best_substitute_similarity": median_substitute,
                "attention_parameter_share": point["attention_parameter_share"],
                "unique_domains_touched": len(touched),
                "cumulative_attention_heads_removed": point["attention_removed"],
            }
        )
        if target > 0.20:
            lower = TARGETS[TARGETS.index(target) - 1]
            interval = f"{int(lower*100)}-{int(target*100)}"
            start_snapshot = snapshots[lower]
            start_ranks = rank_percentiles(
                start_snapshot["global_index"], start_snapshot["delta_total"]
            )
            newly_ranked = [
                start_ranks[index] for index in new_indices if index in start_ranks
            ]
            attention = [row for row in new_rows if row["unit_type"] == TYPE_ATTENTION]
            ffn = [row for row in new_rows if row["unit_type"] == TYPE_FFN]
            attention_cost = sum(int(row["parameter_cost"]) for row in attention)
            ffn_cost = sum(int(row["parameter_cost"]) for row in ffn)
            cost = attention_cost + ffn_cost
            type_rows.append(
                {
                    "interval": interval,
                    "new_attention_heads": len(attention),
                    "new_ffn_neurons": len(ffn),
                    "attention_parameter_cost": attention_cost,
                    "ffn_parameter_cost": ffn_cost,
                    "attention_budget_share": attention_cost / cost if cost else 0.0,
                    "top1_change": accuracy_by_interval[interval]["top1_change"],
                }
            )
            before_domains = {
                int(row["domain_id"]) for row in prefixes[lower]
            }
            new_domains = touched - before_domains
            newly_touched_rows = [domain_at[target][domain_id] for domain_id in new_domains]
            domain_alignment_rows.append(
                {
                    "interval": interval,
                    "new_unique_bms_domains_touched": len(new_domains),
                    "cumulative_unique_domains_touched": len(touched),
                    "mean_coverage_loss_newly_touched_domains": (
                        float(np.mean([
                            float(row["coverage_drop_from_initial"])
                            for row in newly_touched_rows
                        ])) if newly_touched_rows else 0.0
                    ),
                    "median_retained_ratio_newly_touched_domains": (
                        float(np.median([
                            float(row["retained_ratio"])
                            for row in newly_touched_rows
                        ])) if newly_touched_rows else 1.0
                    ),
                    "top1_change": accuracy_by_interval[interval]["top1_change"],
                }
            )
            interval_descriptors = [descriptor_by_global[index] for index in new_indices]
            quantities = {
                "D_abs": [float(row["D_abs"]) for row in interval_descriptors],
                "D_rel": [float(row["D_rel"]) for row in interval_descriptors],
                "D_dyn": [float(row["D_dyn"]) for row in interval_descriptors],
                "functional_energy": [float(row["functional_energy"]) for row in interval_descriptors],
                "delta_average_at_selection": [float(risk_by_global[index]["delta_average"]) for index in new_indices],
                "delta_total_at_selection": [float(risk_by_global[index]["delta_total"]) for index in new_indices],
                "domain_valid_demand_count": [float(row["domain_valid_demand_count"]) for row in interval_descriptors],
            }
            for quantity, values in quantities.items():
                descriptor_statistics.append(
                    {"interval": interval, "quantity": quantity, **quantile_summary(values)}
                )
            rank_stats = quantile_summary(newly_ranked)
            ranks_at_20 = rank_percentiles(
                snapshots[0.20]["global_index"], snapshots[0.20]["delta_total"]
            )
            at20 = [ranks_at_20[index] for index in new_indices if index in ranks_at_20]
            dynamic20 = quantile_summary(at20)["median"]
            transition_rows.append(
                {
                    "interval": interval,
                    "num_units": len(new_indices),
                    "median_rank_at_interval_start": rank_stats["median"],
                    "q25": rank_stats["q25"],
                    "q75": rank_stats["q75"],
                    "median_rank_gain_since_20": (
                        dynamic20 - rank_stats["median"]
                        if math.isfinite(dynamic20) and math.isfinite(rank_stats["median"])
                        else math.nan
                    ),
                }
            )
        previous_prefix = current_prefix

    atomic_csv(output_dir / "accuracy_vs_dynamic_state.csv", tuple(dynamic_rows[0]), dynamic_rows)
    atomic_csv(output_dir / "interval_type_accuracy_alignment.csv", tuple(type_rows[0]), type_rows)
    atomic_csv(output_dir / "interval_domain_accuracy_alignment.csv", tuple(domain_alignment_rows[0]), domain_alignment_rows)
    atomic_csv(output_dir / "interval_descriptor_statistics.csv", tuple(descriptor_statistics[0]), descriptor_statistics)
    atomic_csv(output_dir / "interval_rank_amplification.csv", tuple(transition_rows[0]), transition_rows)

    descriptor_lookup = {
        (row["interval"], row["quantity"]): row for row in descriptor_statistics
    }
    dynamic_lookup = {float(row["target_sparsity"]): row for row in dynamic_rows}
    type_lookup = {row["interval"]: row for row in type_rows}
    transition_summary = []
    previous = None
    for point in points:
        target = float(point["target_sparsity"])
        interval = "" if previous is None else f"{int(previous*100)}-{int(target*100)}"
        transition_summary.append(
            {
                "target_sparsity": target,
                "estimated_sparsity": point["estimated_sparsity"],
                "Top1": point["top1"],
                "Top5": point["top5"],
                "Top1_drop_from_20": point["top1_drop_from_20"],
                "Top1_drop_from_previous": point["top1_drop_from_previous"],
                "removed_attention": point["attention_removed"],
                "removed_ffn": point["ffn_removed"],
                "new_attention_since_previous": 0 if not interval else type_lookup[interval]["new_attention_heads"],
                "new_ffn_since_previous": 0 if not interval else type_lookup[interval]["new_ffn_neurons"],
                "attention_parameter_share_interval": 0.0 if not interval else type_lookup[interval]["attention_budget_share"],
                "cumulative_domains_touched": dynamic_lookup[target]["unique_domains_touched"],
                "median_new_unit_D_abs": math.nan if not interval else descriptor_lookup[(interval, "D_abs")]["median"],
                "median_new_unit_functional_energy": math.nan if not interval else descriptor_lookup[(interval, "functional_energy")]["median"],
                "median_dynamic_rank_percentile": dynamic_lookup[target]["median_future_extra_rank_percentile"],
                "mean_domain_coverage": dynamic_lookup[target]["mean_selected_domain_coverage"],
                "median_selected_domain_retained_ratio": dynamic_lookup[target]["median_selected_domain_retained_ratio"],
            }
        )
        previous = target
    atomic_csv(output_dir / "task018_transition_summary.csv", tuple(transition_summary[0]), transition_summary)
    _write_figures(output_dir, transition_summary, type_rows, interval_accuracy)
    _write_diagnosis(output_dir, transition_summary, type_rows, domain_alignment_rows, transition_rows, interval_accuracy)
    atomic_json(
        output_dir / "task018_completion.json",
        {
            "artifact_identity_pass": True,
            "prefix_identity_pass": True,
            "s20_matches_task016": True,
            "s30_matches_task016": True,
            "prefix_sets_nested": True,
            "validation_22_complete": True,
            "validation_24_complete": True,
            "validation_26_complete": True,
            "validation_28_complete": True,
            "analysis_complete": True,
            "status": "PASS",
            "production_pruning_code_modified": False,
            "fine_tuning_executed": False,
            "causal_ablation_executed": False,
        },
    )


def _write_figures(
    output_dir: Path,
    summary: Sequence[Mapping[str, object]],
    type_rows: Sequence[Mapping[str, object]],
    interval_accuracy: Sequence[Mapping[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/task018-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(figure, name: str) -> None:
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    x = [100 * float(row["estimated_sparsity"]) for row in summary]
    top1 = [float(row["Top1"]) for row in summary]
    intervals = [row["interval"] for row in interval_accuracy]
    drops = [float(row["top1_drop"]) for row in interval_accuracy]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(x, top1, marker="o", color="#0072B2")
    axis.set(xlabel="Estimated parameter sparsity (%)", ylabel="Prune-only Top-1 (%)")
    axis.grid(alpha=0.25)
    save(figure, "figure1_top1_accuracy_curve")

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.bar(intervals, drops, color="#D55E00")
    axis.set(xlabel="Estimated budget interval (%)", ylabel="Top-1 drop (points)")
    axis.grid(axis="y", alpha=0.25)
    save(figure, "figure2_interval_top1_drop")

    plots = (
        ("removed_attention", "Cumulative Attention heads", "figure3_attention_vs_top1"),
        ("attention_parameter_share_interval", "Interval Attention parameter share", "figure4_attention_share_vs_drop"),
        ("median_dynamic_rank_percentile", "Median future-unit rank percentile", "figure5_dynamic_rank_vs_top1"),
        ("cumulative_domains_touched", "Cumulative domains touched", "figure6_domains_vs_top1"),
        ("median_new_unit_D_abs", "Median D_abs of new units", "figure7_dabs_vs_top1"),
    )
    for field, label, name in plots:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        values = [float(row[field]) for row in summary]
        axis.plot(x, values, marker="o", color="#009E73")
        twin = axis.twinx()
        twin.plot(x, top1, marker="s", color="#0072B2", alpha=0.65)
        axis.set_xlabel("Estimated parameter sparsity (%)")
        axis.set_ylabel(label)
        twin.set_ylabel("Top-1 (%)")
        axis.grid(alpha=0.2)
        save(figure, name)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    coverage = [float(row["mean_domain_coverage"]) for row in summary]
    retained = [float(row["median_selected_domain_retained_ratio"]) for row in summary]
    axis.plot(x, coverage, marker="o", label="Mean selected-domain coverage")
    axis.plot(x, retained, marker="s", label="Median selected-domain retained ratio")
    axis.set(xlabel="Estimated parameter sparsity (%)", ylabel="Functional state")
    axis.grid(alpha=0.2)
    axis.legend()
    save(figure, "figure8_functional_state_vs_top1")


def _write_diagnosis(
    output_dir: Path,
    summary: Sequence[Mapping[str, object]],
    type_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    rank_rows: Sequence[Mapping[str, object]],
    interval_accuracy: Sequence[Mapping[str, object]],
) -> None:
    worst = max(interval_accuracy, key=lambda row: float(row["top1_drop_per_1pct_budget"]))
    worst_type = next(row for row in type_rows if row["interval"] == worst["interval"])
    worst_domain = next(row for row in domain_rows if row["interval"] == worst["interval"])
    worst_rank = next(row for row in rank_rows if row["interval"] == worst["interval"])
    drops = ", ".join(
        f"{row['interval']}={float(row['top1_drop']):.4f}"
        for row in interval_accuracy
    )
    text = f"""# Task018 high-sparsity transition diagnosis

Task018 uses exact shortest parameter-budget prefixes from the saved Task016
`domain_total` sequence. Every target starts from the same original checkpoint.
The reported x-axis is estimated parameter sparsity; unchanged physical `numel`
must not be interpreted as verified physical sparsity.

## Q1. Is the degradation gradual or concentrated?

The observed two-percent interval drops are: {drops}. The largest observed
deterioration is concentrated in `{worst['interval']}`; the complete vector is
reported so this wording does not impose a new threshold.

## Q2. Which interval deteriorates fastest?

`{worst['interval']}`, with {float(worst['top1_drop_per_1pct_budget']):.4f}
Top-1 points lost per additional one percent estimated parameter budget.

## Q3. Does it coincide with increased Attention pruning?

The same interval adds {worst_type['new_attention_heads']} Attention heads and
{worst_type['new_ffn_neurons']} FFN neurons; Attention accounts for
{float(worst_type['attention_budget_share']):.4%} of its estimated parameter
cost. This is temporal alignment, not causal evidence.

## Q4. Does it coincide with new competition domains?

The interval newly touches {worst_domain['new_unique_bms_domains_touched']}
domains and reaches {worst_domain['cumulative_unique_domains_touched']}
cumulative touched domains. Its newly touched domains have mean coverage loss
{float(worst_domain['mean_coverage_loss_newly_touched_domains']):.6g}.

## Q5. Are later units stronger by D_abs or functional energy?

See `interval_descriptor_statistics.csv`; interval medians and quartiles are
reported directly without a fitted cutoff.

## Q6. Does dynamic rank amplification precede accuracy damage?

At the start of the worst interval, its future selected units have median rank
percentile {float(worst_rank['median_rank_at_interval_start']):.6g} and median
rank gain since 20% {float(worst_rank['median_rank_gain_since_20']):.6g}.
This establishes ordering in the saved dynamic state, not causality.

## Q7. Approximate damaging transition

The observed maximum-acceleration region is `{worst['interval']}` percent
estimated parameter sparsity. It localizes the next causal study; it is not a
new pruning threshold.

## Q8. State transition or static score-scale problem?

The accuracy curve is aligned with dynamically recomputed rank, domain and
type state at every two-percent prefix. The evidence therefore tests a
high-sparsity state-transition explanation, while Task018 deliberately leaves
the Task016 score scale unchanged.

## Q9. Next mechanism to investigate

The next task should run a narrowly controlled causal ablation inside the
localized interval, separating newly entered Attention heads from matched FFN
or domain-expansion alternatives. Task018 does not implement that mechanism.

No fine-tuning or counterfactual pruning was executed.
"""
    (output_dir / "diagnosis.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task016-root", type=Path, required=True)
    prepare_parser.add_argument("--task017-root", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--target", type=float, required=True)
    validate_parser.add_argument("--device", default="cuda:0")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--task014-root", type=Path, required=True)
    analyze_parser.add_argument("--task015-root", type=Path, required=True)
    analyze_parser.add_argument("--task016-root", type=Path, required=True)
    analyze_parser.add_argument("--task017-root", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.task016_root, args.task017_root, args.output_dir, args.checkpoint)
    elif args.command == "validate":
        validate_target(args.output_dir, args.target, args.device)
    elif args.command == "analyze":
        analyze(
            args.task014_root,
            args.task015_root,
            args.task016_root,
            args.task017_root,
            args.output_dir,
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

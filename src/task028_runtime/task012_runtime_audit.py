"""Runtime evidence writer for Task012 GPU prune-only diagnostics.

The model forward path, calibration, BMS grouping, pruning registry, and
validation remain in the established Task010/Task011 implementation.  This
module records the resulting scalars, small index lists, and metadata only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from task012_diagnostics import (
    layer_at_min_keep,
    minimum_keep_units,
    parameter_accounting,
)


LAYER_FIELDS = (
    "layer",
    "stage",
    "unit_type",
    "original_units",
    "removed_units",
    "remaining_units",
    "removed_ratio",
    "estimated_removed_parameters",
    "min_keep_units",
    "at_min_keep",
    "original_heads",
    "removed_heads",
    "remaining_heads",
    "original_neurons",
    "removed_neurons",
    "remaining_neurons",
)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cached_checkpoint_sha256(path: Path, cache_path: Path) -> str:
    """Hash a checkpoint once and reuse it while path/size/mtime are stable."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("identity") == identity and cached.get("sha256"):
            return str(cached["sha256"])
    sha256 = _sha256_file(resolved)
    _atomic_json(cache_path, {"identity": identity, "sha256": sha256})
    return sha256


def checkpoint_audit_cached(
    path: Path,
    cleaned_state_dict: Mapping[str, object],
    load_result: object,
    cache_path: Path,
) -> dict:
    """Record load evidence without repeatedly reading the large file on CPU."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    missing = list(getattr(load_result, "missing_keys", ()))
    unexpected = list(getattr(load_result, "unexpected_keys", ()))
    return {
        "path": str(resolved),
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": cached_checkpoint_sha256(resolved, cache_path),
        "cleaned_state_dict_key_count": len(cleaned_state_dict),
        "loaded_key_count_estimate": len(cleaned_state_dict) - len(unexpected),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def baseline_identity(
    *, args, checkpoint_metadata: Mapping[str, object], validation_metadata: dict,
    gpu_ids: Sequence[int]
) -> dict:
    """Build the exact compatibility key for the shared baseline cache."""
    return {
        "git_commit": _git_commit(),
        "checkpoint": checkpoint_metadata["path"],
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "checkpoint_size_bytes": checkpoint_metadata["file_size_bytes"],
        "model": args.model,
        "validation_split": "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt",
        "validation_dataset": validation_metadata,
        "validation_batch_size": int(args.batch_size),
        "amp_enabled": not bool(args.disable_amp),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(index) for index in gpu_ids],
    }


def load_baseline_cache(path: Path, identity: Mapping[str, object]):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("identity") != dict(identity):
        return None
    metrics = value.get("metrics", {})
    try:
        return {
            "baseline_top1": float(metrics["baseline_top1"]),
            "baseline_top5": float(metrics["baseline_top5"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def write_baseline_cache(
    path: Path,
    identity: Mapping[str, object],
    baseline_top1: float,
    baseline_top5: float,
) -> None:
    _atomic_json(
        path,
        {
            "identity": dict(identity),
            "metrics": {
                "baseline_top1": float(baseline_top1),
                "baseline_top5": float(baseline_top5),
            },
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "Evaluated once with the unchanged validation pipeline and both "
                "visible GPUs; reused only for an exact identity match."
            ),
        },
    )


def _stage(layer_name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return match.group(1) if match else ""


def _layer_rows(model: nn.Module, pruner, args):
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    modules = dict(target_model.named_modules())
    rows = []
    attention_cost = 0
    mlp_cost = 0
    for layer_name in pruner.ordered_layer_names:
        module = modules.get(layer_name)
        if module is None:
            continue
        class_name = module.__class__.__name__
        if "WindowAttention3D" in class_name:
            unit_type = "attention_head"
            cost_type = "head"
            original = int(module.num_heads)
            remaining_indices = set(int(index) for index in module.keep_heads)
        elif "Mlp" in class_name:
            unit_type = "ffn_neuron"
            cost_type = "neuron"
            original = int(module.original_hidden_features)
            remaining_indices = set(int(index) for index in module.keep_neurons)
        else:
            continue
        removed = original - len(remaining_indices)
        remaining = len(remaining_indices)
        unit_cost = int(pruner.estimate_unit_cost(module, cost_type))
        estimated_removed = removed * unit_cost
        minimum = minimum_keep_units(
            original,
            float(args.min_keep_ratio),
            unit_type,
            int(args.diagnostic_min_attention_heads),
        )
        if unit_type == "attention_head":
            attention_cost += estimated_removed
        else:
            mlp_cost += estimated_removed
        rows.append(
            {
                "layer": layer_name,
                "stage": _stage(layer_name),
                "unit_type": unit_type,
                "original_units": original,
                "removed_units": removed,
                "remaining_units": remaining,
                "removed_ratio": removed / original,
                "estimated_removed_parameters": estimated_removed,
                "min_keep_units": minimum,
                "at_min_keep": layer_at_min_keep(remaining, minimum),
                "original_heads": original if unit_type == "attention_head" else "",
                "removed_heads": removed if unit_type == "attention_head" else "",
                "remaining_heads": remaining if unit_type == "attention_head" else "",
                "original_neurons": original if unit_type == "ffn_neuron" else "",
                "removed_neurons": removed if unit_type == "ffn_neuron" else "",
                "remaining_neurons": remaining if unit_type == "ffn_neuron" else "",
            }
        )
    return rows, attention_cost, mlp_cost


def write_task012_run_artifacts(
    *,
    run_dir: Path,
    args,
    model: nn.Module,
    pruner,
    actual_report: Mapping[str, object],
    baseline_top1: float,
    baseline_top5: float,
    pre_ft_top1: float,
    pre_ft_top5: float,
    parameter_count_before: int,
    checkpoint_metadata: Mapping[str, object],
    calibration_loader,
    validation_metadata: Mapping[str, object],
    captured_groups: Mapping[str, object],
) -> dict:
    """Write one isolated run; ``final_metrics.json`` is committed last."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    parameter_count_after = sum(parameter.numel() for parameter in target_model.parameters())
    layer_rows, attention_cost, mlp_cost = _layer_rows(model, pruner, args)
    estimated_removed = attention_cost + mlp_cost
    accounting = parameter_accounting(
        parameter_count_before, parameter_count_after, estimated_removed
    )
    reported_sparsity = float(actual_report["sparsity"])
    if abs(float(accounting["estimated_budget_sparsity"]) - reported_sparsity) > 1e-10:
        raise RuntimeError(
            "independent pruning-cost audit disagrees with model report: "
            f"{accounting['estimated_budget_sparsity']} vs {reported_sparsity}"
        )

    groups = captured_groups.get("groups")
    if groups is None:
        raise RuntimeError("Task012 requires captured BMS group membership")
    group_count = len(groups)
    singleton_ratio = (
        sum(1 for group in groups if len(group) == 1) / group_count
        if group_count else 0.0
    )
    calibration = calibration_loader.summary()
    calibration.update(
        {
            "random_seed": int(args.seed),
            "calibration_batches": int(args.calib_batches),
            "calibration_batch_size": int(args.calib_batch_size),
            "split": "/data/jixinye25/UCF101_Frame/train_rgb_split1.txt",
            "clip_sampling_policy": "LoopPadding(32)",
            "spatial_preprocessing": (
                "Scale(224), CornerCrop(224, center), ToTensor, "
                "Normalize(ImageNet mean/std)"
            ),
        }
    )
    _atomic_csv(run_dir / "layer_pruning.csv", LAYER_FIELDS, layer_rows)
    _atomic_json(run_dir / "calibration_samples.json", calibration)

    checkpoint_path = Path(str(checkpoint_metadata["path"]))
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        "status": "task012_prune_only_complete",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "variant": args.descriptor_variant,
        "target_sparsity": float(args.sparsity),
        "sigma": float(args.sigma),
        "gamma_decay": float(args.gamma_decay),
        "min_keep_ratio": float(args.min_keep_ratio),
        "importance_alpha": float(args.importance_alpha),
        "diagnostic_min_attention_heads": int(args.diagnostic_min_attention_heads),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "seed": int(args.seed),
        "selection_mode": args.selection_mode,
        "calibration_device": "cuda:0",
        "calibration_batch_size": int(args.calib_batch_size),
        "calibration_batches": int(args.calib_batches),
        "calibration_samples_sha256": calibration["sample_list_sha256"],
        "validation_split": "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt",
        "validation_dataset": dict(validation_metadata),
        "validation_batch_size": int(args.batch_size),
        "amp_enabled": not bool(args.disable_amp),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "gpu_execution": {
            "calibration_and_bms": "cuda:0",
            "baseline_and_validation": [
                f"cuda:{index}" for index in range(torch.cuda.device_count())
            ],
            "cpu_only": "video decode, sample identifiers, CSV/JSON, offline figures",
        },
        "command_line": list(sys.argv),
        "full_fine_tuning_executed": False,
    }
    _atomic_json(run_dir / "run_metadata.json", metadata)

    removed_heads = int(actual_report["original_heads"]) - int(
        actual_report["remaining_heads"]
    )
    removed_neurons = int(actual_report["original_neurons"]) - int(
        actual_report["remaining_neurons"]
    )
    total_cost = estimated_removed
    metrics = {
        "status": "task012_prune_only_complete",
        "variant": args.descriptor_variant,
        "attention_protection": (
            "original"
            if int(args.diagnostic_min_attention_heads) == 0
            else f"min{int(args.diagnostic_min_attention_heads)}"
        ),
        "target_sparsity": float(args.sparsity),
        "baseline_top1": float(baseline_top1),
        "baseline_top5": float(baseline_top5),
        "pre_ft_top1": float(pre_ft_top1),
        "pre_ft_top5": float(pre_ft_top5),
        "top1_drop": float(baseline_top1 - pre_ft_top1),
        "top5_drop": float(baseline_top5 - pre_ft_top5),
        "estimated_actual_sparsity": reported_sparsity,
        "estimated_budget_sparsity": float(accounting["estimated_budget_sparsity"]),
        "physical_numel_sparsity": float(accounting["physical_numel_sparsity"]),
        "parameters_before": int(accounting["parameters_before"]),
        "parameters_after": int(accounting["parameters_after"]),
        "estimated_removed_parameters": int(accounting["estimated_removed_parameters"]),
        "removed_heads": removed_heads,
        "remaining_heads": int(actual_report["remaining_heads"]),
        "removed_neurons": removed_neurons,
        "remaining_neurons": int(actual_report["remaining_neurons"]),
        "num_groups": group_count,
        "singleton_ratio": float(singleton_ratio),
        "attention_removed_parameter_cost": int(attention_cost),
        "mlp_removed_parameter_cost": int(mlp_cost),
        "attention_budget_fraction": attention_cost / total_cost if total_cost else 0.0,
        "mlp_budget_fraction": mlp_cost / total_cost if total_cost else 0.0,
        "layer_pruning_file": str((run_dir / "layer_pruning.csv").resolve()),
        "run_metadata_file": str((run_dir / "run_metadata.json").resolve()),
        "parameter_accounting_note": (
            "Estimated pruning cost is logical/index-based and is reported "
            "separately from physical tensor numel reduction."
        ),
        "full_fine_tuning_executed": False,
    }
    _atomic_json(run_dir / "final_metrics.json", metrics)
    return metrics

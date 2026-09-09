"""Runtime evidence writer for Task013 two-GPU prune-only experiments."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

import task012_runtime_audit as task012_runtime
from cost_decoupled_selection import selection_trace_snapshots
from task012_diagnostics import layer_at_min_keep, minimum_keep_units, parameter_accounting


checkpoint_audit_cached = task012_runtime.checkpoint_audit_cached
load_baseline_cache = task012_runtime.load_baseline_cache
write_baseline_cache = task012_runtime.write_baseline_cache

LAYER_FIELDS = (
    "layer", "stage", "unit_type", "original_units", "removed_units",
    "remaining_units", "removed_ratio", "removed_parameter_cost", "at_min_keep",
)
SELECTION_FIELDS = (
    "cost_mode", "target_sparsity", "budget_progress", "selection_rank",
    "layer", "unit_type", "group_id", "raw_pruning_score",
    "parameter_cost", "effective_selection_score", "selected",
)
CANDIDATE_FIELDS = SELECTION_FIELDS + ("unit_index",)


def baseline_identity(
    *, args, checkpoint_metadata: Mapping[str, object],
    validation_metadata: dict, gpu_ids: Sequence[int]
) -> dict:
    identity = task012_runtime.baseline_identity(
        args=args,
        checkpoint_metadata=checkpoint_metadata,
        validation_metadata=validation_metadata,
        gpu_ids=gpu_ids,
    )
    identity["purpose"] = "task013_shared_unpruned_baseline"
    return identity


def _stage(layer_name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return match.group(1) if match else ""


def _layer_rows(model: nn.Module, pruner, args):
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    modules = dict(target_model.named_modules())
    rows = []
    costs = {"attention_head": 0, "ffn_neuron": 0}
    totals = {"attention_head": 0, "ffn_neuron": 0}
    removed_totals = {"attention_head": 0, "ffn_neuron": 0}
    for layer_name in pruner.ordered_layer_names:
        module = modules.get(layer_name)
        if module is None:
            continue
        class_name = module.__class__.__name__
        if "WindowAttention3D" in class_name:
            unit_type = "attention_head"
            cost_type = "head"
            original = int(module.num_heads)
            remaining = len({int(index) for index in module.keep_heads})
        elif "Mlp" in class_name:
            unit_type = "ffn_neuron"
            cost_type = "neuron"
            original = int(module.original_hidden_features)
            remaining = len({int(index) for index in module.keep_neurons})
        else:
            continue
        removed = original - remaining
        unit_cost = int(pruner.estimate_unit_cost(module, cost_type))
        removed_cost = removed * unit_cost
        minimum = minimum_keep_units(
            original, float(args.min_keep_ratio), unit_type, 0
        )
        costs[unit_type] += removed_cost
        totals[unit_type] += original
        removed_totals[unit_type] += removed
        rows.append(
            {
                "layer": layer_name,
                "stage": _stage(layer_name),
                "unit_type": unit_type,
                "original_units": original,
                "removed_units": removed,
                "remaining_units": remaining,
                "removed_ratio": removed / original,
                "removed_parameter_cost": removed_cost,
                "at_min_keep": layer_at_min_keep(remaining, minimum),
            }
        )
    return rows, costs, totals, removed_totals


def _selection_rows(pruner, args, parameter_count_before: int):
    candidates = []
    for candidate in pruner.selection_candidates:
        row = {
            "cost_mode": args.selection_cost_mode,
            "target_sparsity": float(args.sparsity),
            "budget_progress": 0.0,
            "selection_rank": int(candidate["selection_rank"]),
            "layer": candidate["layer"],
            "unit_type": candidate["unit_type"],
            "group_id": int(candidate["group_id"]),
            "raw_pruning_score": float(candidate["raw_pruning_score"]),
            "parameter_cost": int(candidate["parameter_cost"]),
            "effective_selection_score": float(
                candidate["effective_selection_score"]
            ),
            "selected": bool(candidate["selected"]),
            "unit_index": int(candidate["unit_index"]),
        }
        candidates.append(row)
    if not candidates:
        raise RuntimeError("Task013 requires a non-empty selection trace")
    snapshots = selection_trace_snapshots(
        candidates, float(parameter_count_before) * float(args.sparsity)
    )
    snapshots = [
        {field: row[field] for field in SELECTION_FIELDS}
        for row in snapshots
    ]
    return candidates, snapshots


def write_task013_run_artifacts(
    *, run_dir: Path, args, model: nn.Module, pruner,
    actual_report: Mapping[str, object], baseline_top1: float,
    baseline_top5: float, pre_ft_top1: float, pre_ft_top5: float,
    parameter_count_before: int, checkpoint_metadata: Mapping[str, object],
    calibration_loader, validation_metadata: Mapping[str, object],
    captured_groups: Mapping[str, object],
) -> dict:
    """Write an isolated run and commit final metrics only after all audits."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    parameter_count_after = sum(p.numel() for p in target_model.parameters())
    layer_rows, costs, totals, removed_totals = _layer_rows(model, pruner, args)
    estimated_removed = sum(costs.values())
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
        raise RuntimeError("Task013 requires captured BMS group membership")
    group_count = len(groups)
    singleton_ids = {
        index for index, group in enumerate(groups) if len(group) == 1
    }
    selected_group_ids = {
        int(row["group_id"])
        for row in pruner.selection_candidates if bool(row["selected"])
    }
    candidates, snapshots = _selection_rows(pruner, args, parameter_count_before)
    task012_runtime._atomic_csv(run_dir / "layer_pruning.csv", LAYER_FIELDS, layer_rows)
    task012_runtime._atomic_csv(
        run_dir / "selection_candidates.csv", CANDIDATE_FIELDS, candidates
    )
    task012_runtime._atomic_csv(
        run_dir / "selection_trace.csv", SELECTION_FIELDS, snapshots
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
    task012_runtime._atomic_json(run_dir / "calibration_samples.json", calibration)

    checkpoint_path = Path(str(checkpoint_metadata["path"]))
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        "status": "task013_prune_only_complete",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": task012_runtime._git_commit(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "descriptor_variant": args.descriptor_variant,
        "selection_mode": args.selection_mode,
        "selection_cost_mode": args.selection_cost_mode,
        "target_sparsity": float(args.sparsity),
        "sigma": float(args.sigma),
        "gamma_decay": float(args.gamma_decay),
        "min_keep_ratio": float(args.min_keep_ratio),
        "importance_alpha": float(args.importance_alpha),
        "seed": int(args.seed),
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
            "calibration_descriptors_bms_pruning": "cuda:0",
            "baseline_and_pruned_validation": [
                f"cuda:{index}" for index in range(torch.cuda.device_count())
            ],
            "cpu_only": "video decode, identifiers/hashes, CSV/JSON, offline plots",
        },
        "command_line": list(sys.argv),
        "full_fine_tuning_executed": False,
    }
    task012_runtime._atomic_json(run_dir / "run_metadata.json", metadata)

    attention_cost = costs["attention_head"]
    mlp_cost = costs["ffn_neuron"]
    total_cost = attention_cost + mlp_cost
    metrics = {
        "status": "task013_prune_only_complete",
        "descriptor_variant": args.descriptor_variant,
        "selection_cost_mode": args.selection_cost_mode,
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
        "original_heads": totals["attention_head"],
        "removed_heads": removed_totals["attention_head"],
        "remaining_heads": totals["attention_head"] - removed_totals["attention_head"],
        "original_neurons": totals["ffn_neuron"],
        "removed_neurons": removed_totals["ffn_neuron"],
        "remaining_neurons": totals["ffn_neuron"] - removed_totals["ffn_neuron"],
        "removed_head_ratio": (
            removed_totals["attention_head"] / totals["attention_head"]
            if totals["attention_head"] else 0.0
        ),
        "removed_neuron_ratio": (
            removed_totals["ffn_neuron"] / totals["ffn_neuron"]
            if totals["ffn_neuron"] else 0.0
        ),
        "attention_removed_parameter_cost": attention_cost,
        "mlp_removed_parameter_cost": mlp_cost,
        "attention_budget_fraction": attention_cost / total_cost if total_cost else 0.0,
        "mlp_budget_fraction": mlp_cost / total_cost if total_cost else 0.0,
        "num_groups": group_count,
        "singleton_ratio": (
            len(singleton_ids) / group_count if group_count else 0.0
        ),
        "selected_groups": len(selected_group_ids),
        "selected_singleton_groups": len(selected_group_ids & singleton_ids),
        "attention_layers_at_min_keep": sum(
            1 for row in layer_rows
            if row["unit_type"] == "attention_head" and row["at_min_keep"]
        ),
        "layer_pruning_file": str((run_dir / "layer_pruning.csv").resolve()),
        "selection_trace_file": str((run_dir / "selection_trace.csv").resolve()),
        "full_fine_tuning_executed": False,
    }
    task012_runtime._atomic_json(run_dir / "final_metrics.json", metrics)
    return metrics

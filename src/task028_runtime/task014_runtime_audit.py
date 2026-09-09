"""Runtime evidence writer for Task014 two-GPU prune-only experiments."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

import task012_runtime_audit as task012_runtime
import task013_runtime_audit as task013_runtime
from task012_diagnostics import parameter_accounting


checkpoint_audit_cached = task012_runtime.checkpoint_audit_cached
load_baseline_cache = task012_runtime.load_baseline_cache
write_baseline_cache = task012_runtime.write_baseline_cache


def baseline_identity(
    *, args, checkpoint_metadata: Mapping[str, object],
    validation_metadata: dict, gpu_ids: Sequence[int]
) -> dict:
    identity = task013_runtime.baseline_identity(
        args=args,
        checkpoint_metadata=checkpoint_metadata,
        validation_metadata=validation_metadata,
        gpu_ids=gpu_ids,
    )
    identity["purpose"] = "task014_shared_unpruned_baseline"
    contribution_path = Path(args.contribution_npz).expanduser().resolve()
    if not contribution_path.is_file():
        raise FileNotFoundError(contribution_path)
    stat = contribution_path.stat()
    identity.update(
        {
            "contribution_npz": str(contribution_path),
            "contribution_npz_size_bytes": int(stat.st_size),
            "contribution_npz_mtime_ns": int(stat.st_mtime_ns),
        }
    )
    return identity


def write_task014_run_artifacts(
    *, run_dir: Path, args, model: nn.Module, pruner,
    actual_report: Mapping[str, object], baseline_top1: float,
    baseline_top5: float, pre_ft_top1: float, pre_ft_top5: float,
    parameter_count_before: int, checkpoint_metadata: Mapping[str, object],
    calibration_loader, validation_metadata: Mapping[str, object],
    captured_groups: Mapping[str, object],
) -> dict:
    """Write one isolated Task014 run after pruning and validation succeed."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    parameter_count_after = sum(parameter.numel() for parameter in target_model.parameters())
    layer_rows, costs, totals, removed_totals = task013_runtime._layer_rows(
        model, pruner, args
    )
    estimated_removed = sum(costs.values())
    accounting = parameter_accounting(
        parameter_count_before, parameter_count_after, estimated_removed
    )
    reported_sparsity = float(actual_report["sparsity"])
    if abs(float(accounting["estimated_budget_sparsity"]) - reported_sparsity) > 1e-10:
        raise RuntimeError(
            "Task014 parameter accounting disagrees with model report: "
            f"{accounting['estimated_budget_sparsity']} vs {reported_sparsity}"
        )
    task012_runtime._atomic_csv(
        run_dir / "layer_pruning.csv", task013_runtime.LAYER_FIELDS, layer_rows
    )

    groups = captured_groups.get("groups")
    if groups is None:
        raise RuntimeError("Task014 requires captured BMS competition domains")
    num_domains = len(groups)
    singleton_ratio = (
        sum(len(group) == 1 for group in groups) / num_domains
        if num_domains else 0.0
    )

    functional_result = pruner.functional_selection_result
    mapping_audit = pruner.functional_mapping_audit
    if args.selection_mode == "functional":
        if functional_result is None or mapping_audit is None:
            raise RuntimeError("Functional run is missing Task014 selection evidence")
        if mapping_audit.mapped_units != 36_378:
            raise RuntimeError(
                f"Task014 mapped {mapping_audit.mapped_units}, expected 36,378"
            )
        num_removed_units = len(functional_result.trace)
        budget_overshoot = float(functional_result.budget_overshoot)
    else:
        num_removed_units = sum(removed_totals.values())
        budget_overshoot = estimated_removed - (
            float(parameter_count_before) * float(args.sparsity)
        )

    calibration = calibration_loader.summary()
    calibration.update(
        {
            "random_seed": int(args.seed),
            "calibration_batches": int(args.calib_batches),
            "calibration_batch_size": int(args.calib_batch_size),
            "split": os.environ.get(
                "UCF101_TRAIN_SPLIT",
                "dataset/UCF101_Frame/train_rgb_split1.txt",
            ),
            "clip_sampling_policy": "LoopPadding(32)",
            "spatial_preprocessing": (
                "Scale(224), CornerCrop(224, center), ToTensor, "
                "Normalize(ImageNet mean/std)"
            ),
        }
    )
    task012_runtime._atomic_json(run_dir / "calibration_samples.json", calibration)

    contribution_path = Path(args.contribution_npz).expanduser().resolve()
    contribution_stat = contribution_path.stat()
    checkpoint_path = Path(str(checkpoint_metadata["path"]))
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        "status": "task014_prune_only_complete",
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
        "contribution_npz": str(contribution_path),
        "contribution_npz_size_bytes": int(contribution_stat.st_size),
        "contribution_npz_mtime_ns": int(contribution_stat.st_mtime_ns),
        "contribution_metadata_sha256": (
            mapping_audit.metadata_sha256 if mapping_audit is not None else None
        ),
        "contribution_mapping_sha256": (
            mapping_audit.mapping_sha256 if mapping_audit is not None else None
        ),
        "contribution_sample_count": 9,
        "contribution_sample_aggregation": "none",
        "aligned_field_shape_per_video": [16, 7, 7],
        "calibration_device": "cuda:0",
        "calibration_batch_size": int(args.calib_batch_size),
        "calibration_batches": int(args.calib_batches),
        "calibration_samples_sha256": calibration["sample_list_sha256"],
        "validation_split": os.environ.get(
            "UCF101_VAL_SPLIT",
            "dataset/UCF101_Frame/val_rgb_split1.txt",
        ),
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
            "calibration_descriptors_bms_functional_selection": "cuda:0",
            "baseline_and_pruned_validation": [
                f"cuda:{index}" for index in range(torch.cuda.device_count())
            ],
            "cpu_only": (
                "video decode, NPZ decompression, identifiers/hashes, mmap IO, "
                "CSV/JSON and offline plots"
            ),
        },
        "command_line": list(sys.argv),
        "full_fine_tuning_executed": False,
    }
    task012_runtime._atomic_json(run_dir / "run_metadata.json", metadata)

    attention_cost = costs["attention_head"]
    mlp_cost = costs["ffn_neuron"]
    total_cost = attention_cost + mlp_cost
    metrics = {
        "status": "task014_prune_only_complete",
        "selection_mode": args.selection_mode,
        "target_sparsity": float(args.sparsity),
        "baseline_top1": float(baseline_top1),
        "baseline_top5": float(baseline_top5),
        "pre_ft_top1": float(pre_ft_top1),
        "pre_ft_top5": float(pre_ft_top5),
        "top1_drop": float(baseline_top1 - pre_ft_top1),
        "top5_drop": float(baseline_top5 - pre_ft_top5),
        "estimated_budget_sparsity": float(accounting["estimated_budget_sparsity"]),
        "physical_numel_sparsity": float(accounting["physical_numel_sparsity"]),
        "parameters_before": int(accounting["parameters_before"]),
        "parameters_after": int(accounting["parameters_after"]),
        "estimated_removed_parameters": int(accounting["estimated_removed_parameters"]),
        "budget_overshoot": budget_overshoot,
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
        "removed_mlp_ratio": (
            removed_totals["ffn_neuron"] / totals["ffn_neuron"]
            if totals["ffn_neuron"] else 0.0
        ),
        "attention_removed_parameter_cost": attention_cost,
        "mlp_removed_parameter_cost": mlp_cost,
        "attention_budget_fraction": attention_cost / total_cost if total_cost else 0.0,
        "mlp_budget_fraction": mlp_cost / total_cost if total_cost else 0.0,
        "num_bms_domains": num_domains,
        "singleton_ratio": singleton_ratio,
        "num_removed_units": num_removed_units,
        "full_fine_tuning_executed": False,
    }
    task012_runtime._atomic_json(run_dir / "final_metrics.json", metrics)
    return metrics

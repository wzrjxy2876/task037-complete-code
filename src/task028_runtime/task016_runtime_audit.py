"""Runtime evidence for Task016 domain-size-calibrated prune-only runs."""

from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch.nn as nn

import task012_runtime_audit as task012_runtime
import task014_runtime_audit as task014_runtime


checkpoint_audit_cached = task014_runtime.checkpoint_audit_cached
load_baseline_cache = task014_runtime.load_baseline_cache
write_baseline_cache = task014_runtime.write_baseline_cache


def actual_parameter_count(model: nn.Module) -> int:
    """Authoritative physical parameter count from the current model."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    return sum(parameter.numel() for parameter in target.parameters())


def baseline_identity(
    *, args, checkpoint_metadata: Mapping[str, object],
    validation_metadata: dict, gpu_ids: Sequence[int]
) -> dict:
    identity = task014_runtime.baseline_identity(
        args=args,
        checkpoint_metadata=checkpoint_metadata,
        validation_metadata=validation_metadata,
        gpu_ids=gpu_ids,
    )
    identity["purpose"] = "task016_shared_unpruned_baseline"
    identity["functional_score_not_used_for_baseline"] = True
    return identity


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_task016_run_artifacts(
    *, run_dir: Path, args, model: nn.Module, pruner,
    actual_report: Mapping[str, object], baseline_top1: float,
    baseline_top5: float, pre_ft_top1: float, pre_ft_top5: float,
    parameter_count_before: int, checkpoint_metadata: Mapping[str, object],
    calibration_loader, validation_metadata: Mapping[str, object],
    captured_groups: Mapping[str, object],
) -> dict:
    """Extend the unchanged Task014 evidence with Task016 scale diagnostics."""
    if args.functional_score not in {"domain_average", "domain_total"}:
        raise ValueError(f"Unexpected Task016 score mode: {args.functional_score}")
    metrics = task014_runtime.write_task014_run_artifacts(
        run_dir=run_dir,
        args=args,
        model=model,
        pruner=pruner,
        actual_report=actual_report,
        baseline_top1=baseline_top1,
        baseline_top5=baseline_top5,
        pre_ft_top1=pre_ft_top1,
        pre_ft_top5=pre_ft_top5,
        parameter_count_before=parameter_count_before,
        checkpoint_metadata=checkpoint_metadata,
        calibration_loader=calibration_loader,
        validation_metadata=validation_metadata,
        captured_groups=captured_groups,
    )
    run_dir = Path(run_dir)
    measured_after = actual_parameter_count(model)
    if measured_after != int(metrics["parameters_after"]):
        raise RuntimeError("Task016 physical parameter count changed during audit")
    metrics.update(
        {
            "status": "task016_prune_only_complete",
            "score_mode": args.functional_score,
            "actual_sparsity": float(metrics["physical_numel_sparsity"]),
            "parameters_removed": int(metrics["parameters_before"])
            - int(metrics["parameters_after"]),
            "attention_removal_ratio": float(metrics["removed_head_ratio"]),
            "ffn_removal_ratio": float(metrics["removed_mlp_ratio"]),
            "ffn_removed_parameters": int(metrics["mlp_removed_parameter_cost"]),
            "parameter_cost_in_ranking": False,
            "type_specific_rule": False,
            "full_fine_tuning_executed": False,
        }
    )
    task012_runtime._atomic_json(run_dir / "final_metrics.json", metrics)

    metadata_path = run_dir / "run_metadata.json"
    metadata = _read_json(metadata_path)
    descriptor_cache = Path(args.functional_descriptor_cache).expanduser().resolve()
    metadata.update(
        {
            "status": "task016_prune_only_complete",
            "task": "Task016",
            "functional_score": args.functional_score,
            "mathematical_change": (
                "domain_total = fixed_non_null_demand_count * domain_average"
            ),
            "descriptor_changed": False,
            "bms_changed": False,
            "contribution_field_changed": False,
            "type_quota_or_weight": False,
            "new_method_hyperparameter": False,
            "parameter_cost_in_ranking": False,
            "functional_descriptor_cache": str(descriptor_cache),
            "functional_descriptor_cache_sha256": _sha256_file(descriptor_cache),
            "functional_descriptor_cache_reused": True,
        }
    )
    task012_runtime._atomic_json(metadata_path, metadata)

    layer_rows = _read_csv(run_dir / "layer_pruning.csv")
    layerwise_rows = []
    for row in layer_rows:
        before = int(row["original_units"])
        after = int(row["remaining_units"])
        removed = int(row["removed_units"])
        parameters_removed = int(row["removed_parameter_cost"])
        per_unit_cost = parameters_removed // removed if removed else 0
        layerwise_rows.append(
            {
                "layer": row["layer"],
                "unit_type": row["unit_type"],
                "units_before": before,
                "units_after": after,
                "units_removed": removed,
                "unit_removal_ratio": removed / before if before else 0.0,
                "parameters_removed": parameters_removed,
                "parameter_removal_ratio": (
                    parameters_removed / float(per_unit_cost * before)
                    if per_unit_cost and before else 0.0
                ),
            }
        )
    task012_runtime._atomic_csv(
        run_dir / "layerwise_pruning_statistics.csv",
        (
            "layer", "unit_type", "units_before", "units_after",
            "units_removed", "unit_removal_ratio", "parameters_removed",
            "parameter_removal_ratio",
        ),
        layerwise_rows,
    )
    task012_runtime._atomic_csv(
        run_dir / "removed_unit_statistics.csv",
        (
            "score_mode", "target_sparsity", "attention_heads_before",
            "attention_heads_after", "attention_heads_removed",
            "attention_head_removal_ratio", "ffn_neurons_before",
            "ffn_neurons_after", "ffn_neurons_removed", "ffn_removal_ratio",
            "attention_removed_parameters", "ffn_removed_parameters",
        ),
        [
            {
                "score_mode": args.functional_score,
                "target_sparsity": float(args.sparsity),
                "attention_heads_before": int(metrics["original_heads"]),
                "attention_heads_after": int(metrics["remaining_heads"]),
                "attention_heads_removed": int(metrics["removed_heads"]),
                "attention_head_removal_ratio": float(metrics["removed_head_ratio"]),
                "ffn_neurons_before": int(metrics["original_neurons"]),
                "ffn_neurons_after": int(metrics["remaining_neurons"]),
                "ffn_neurons_removed": int(metrics["removed_neurons"]),
                "ffn_removal_ratio": float(metrics["removed_mlp_ratio"]),
                "attention_removed_parameters": int(
                    metrics["attention_removed_parameter_cost"]
                ),
                "ffn_removed_parameters": int(metrics["mlp_removed_parameter_cost"]),
            }
        ],
    )
    return metrics

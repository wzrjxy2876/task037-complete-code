"""Runtime-only audit helpers for Task011 controlled prune-only runs.

These helpers observe the existing production path.  They do not implement a
descriptor, BMS update, group score, budget rule, registry rule, or pruning
mask.  The only tensors copied to CPU are small metadata vectors (sample IDs,
labels, keep indices, and scalar reports).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


BASELINE_FIELDS = (
    "variant",
    "checkpoint",
    "baseline_top1",
    "baseline_top5",
    "pre_ft_top1",
    "pre_ft_top5",
    "top1_drop",
    "top5_drop",
)
LAYER_FIELDS = (
    "layer",
    "unit_type",
    "original_units",
    "removed_units",
    "remaining_units",
    "removed_ratio",
    "estimated_removed_parameters",
    "min_keep_constraint",
    "group_ids_involved",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict:
    """Capture RNG state so baseline validation cannot perturb calibration."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CalibrationAuditLoader:
    """Record metadata for only the calibration batches actually consumed."""

    def __init__(self, loader, max_batches: int):
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        self.loader = loader
        self.max_batches = int(max_batches)
        self.samples: list[dict] = []

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        for batch_number, batch in enumerate(self.loader):
            if batch_number < self.max_batches:
                self._record_batch(batch_number, batch)
            yield batch

    def _record_batch(self, batch_number: int, batch) -> None:
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise TypeError("calibration loader must return (video, target, index)")
        targets = torch.as_tensor(batch[1]).detach().cpu().reshape(-1).tolist()
        indices = torch.as_tensor(batch[2]).detach().cpu().reshape(-1).tolist()
        if len(targets) != len(indices):
            raise ValueError("calibration target/index counts differ")
        clips = getattr(self.loader.dataset, "clips", None)
        for position, (target, index) in enumerate(zip(targets, indices)):
            sample_index = int(index)
            video_identifier = ""
            dataset_target = int(target)
            if clips is not None and 0 <= sample_index < len(clips):
                clip = clips[sample_index]
                if isinstance(clip, (tuple, list)) and clip:
                    video_identifier = str(clip[0])
                    if len(clip) >= 3:
                        dataset_target = int(clip[2])
            self.samples.append(
                {
                    "order": len(self.samples),
                    "batch": int(batch_number),
                    "position_in_batch": int(position),
                    "sample_index": sample_index,
                    "class_id": int(target),
                    "dataset_class_id": dataset_target,
                    "video_identifier": video_identifier,
                }
            )

    def summary(self) -> dict:
        canonical = json.dumps(
            self.samples, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        class_distribution = Counter(int(row["class_id"]) for row in self.samples)
        return {
            "num_videos": len(self.samples),
            "sample_order": self.samples,
            "class_distribution": {
                str(key): int(value) for key, value in sorted(class_distribution.items())
            },
            "sample_list_sha256": hashlib.sha256(canonical).hexdigest(),
            "shuffle": True,
            "sampling_note": (
                "Observed indices from the existing shuffled calibration DataLoader; "
                "no loader or sampler behavior was changed."
            ),
        }


def install_bms_group_capture(pruner) -> dict:
    """Wrap the existing BMS call and retain only returned group membership."""
    capture: dict = {"groups": None}
    original = pruner.mean_shift_clustering

    def audited_mean_shift(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError("unexpected mean_shift_clustering return structure")
        groups = result[0]
        capture["groups"] = [
            [int(index) for index in torch.as_tensor(group).detach().cpu().tolist()]
            for group in groups
        ]
        return result

    pruner.mean_shift_clustering = audited_mean_shift
    return capture


def classifier_output_dimension(model: nn.Module) -> int | None:
    target = model.module if isinstance(model, nn.DataParallel) else model
    classifier = getattr(getattr(target, "cls_head", None), "fc_cls", None)
    return int(classifier.out_features) if isinstance(classifier, nn.Linear) else None


def checkpoint_audit(
    path: Path,
    checkpoint: object,
    cleaned_state_dict: dict,
    load_result: object,
    model: nn.Module,
) -> dict:
    raw_state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    classifier_candidates = [
        value
        for key, value in cleaned_state_dict.items()
        if str(key).endswith("cls_head.fc_cls.weight")
        and isinstance(value, torch.Tensor)
        and value.ndim == 2
    ]
    checkpoint_classifier_dim = (
        int(classifier_candidates[0].shape[0]) if classifier_candidates else None
    )
    missing = list(getattr(load_result, "missing_keys", ()))
    unexpected = list(getattr(load_result, "unexpected_keys", ()))
    state_count = len(raw_state) if isinstance(raw_state, dict) else None
    return {
        "path": str(path.resolve()),
        "file_size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "state_dict_key_count": state_count,
        "cleaned_state_dict_key_count": len(cleaned_state_dict),
        "loaded_key_count_estimate": len(cleaned_state_dict) - len(unexpected),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_classifier_output_dimension": checkpoint_classifier_dim,
        "model_classifier_output_dimension": classifier_output_dimension(model),
    }


def dataset_audit(loader) -> dict:
    clips = getattr(loader.dataset, "clips", None)
    if clips is None:
        return {
            "dataset_size": len(loader.dataset),
            "label_min": None,
            "label_max": None,
            "num_unique_labels": None,
        }
    labels = [int(clip[2]) for clip in clips]
    return {
        "dataset_size": len(clips),
        "label_min": min(labels) if labels else None,
        "label_max": max(labels) if labels else None,
        "num_unique_labels": len(set(labels)),
    }


def _read_descriptor_indices(path: Path) -> dict[tuple[str, str, int], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"global_index", "layer", "unit_type", "unit_index"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"descriptor statistics missing columns {sorted(missing)}")
        rows = list(reader)
    result: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (str(row["layer"]), str(row["unit_type"]), int(row["unit_index"]))
        if key in result:
            raise ValueError(f"duplicate descriptor unit key {key}")
        result[key] = int(row["global_index"])
    return result


def _group_index(groups: list[list[int]] | None) -> dict[int, int]:
    if groups is None:
        return {}
    mapping: dict[int, int] = {}
    for group_id, members in enumerate(groups):
        for global_index in members:
            if global_index in mapping:
                raise ValueError(f"unit {global_index} appears in multiple BMS groups")
            mapping[int(global_index)] = int(group_id)
    return mapping


def _update_baseline_table(path: Path, row: dict) -> None:
    rows = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if existing.get("variant") != row["variant"]]
    rows.append(row)
    rows.sort(key=lambda value: value["variant"])
    _atomic_csv(path, BASELINE_FIELDS, rows)


def write_controlled_run_artifacts(
    *,
    diagnosis_dir: Path,
    args,
    model: nn.Module,
    pruner,
    actual_report: dict,
    baseline_top1: float,
    baseline_top5: float,
    pre_ft_top1: float,
    pre_ft_top5: float,
    parameter_count_before: int,
    checkpoint_metadata: dict,
    calibration_loader: CalibrationAuditLoader,
    validation_metadata: dict,
    captured_groups: dict,
) -> dict:
    """Write budget, layer, sample, checkpoint, and metric evidence."""
    diagnosis_dir = Path(diagnosis_dir)
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    parameter_count_after = sum(parameter.numel() for parameter in target_model.parameters())
    physical_sparsity = (
        (parameter_count_before - parameter_count_after) / parameter_count_before
        if parameter_count_before else 0.0
    )
    descriptor_path = Path(args.adv_path) / "descriptor_statistics.csv"
    descriptor_indices = _read_descriptor_indices(descriptor_path)
    groups = captured_groups.get("groups")
    membership = _group_index(groups)
    modules = dict(target_model.named_modules())
    layer_rows = []
    estimated_removable = 0
    selected_removed_cost = 0

    for layer_name in pruner.ordered_layer_names:
        if layer_name not in modules:
            continue
        module = modules[layer_name]
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
        removed_indices = sorted(set(range(original)) - remaining_indices)
        unit_cost = int(pruner.estimate_unit_cost(module, cost_type))
        estimated_removable += original * unit_cost
        estimated_layer_removed = len(removed_indices) * unit_cost
        selected_removed_cost += estimated_layer_removed
        group_ids = set()
        for unit_index in removed_indices:
            key = (layer_name, unit_type, unit_index)
            if key not in descriptor_indices:
                raise KeyError(f"missing descriptor metadata for {key}")
            global_index = descriptor_indices[key]
            if global_index in membership:
                group_ids.add(membership[global_index])
        minimum_keep_units = max(1, int(original * float(args.min_keep_ratio)))
        layer_rows.append(
            {
                "layer": layer_name,
                "unit_type": unit_type,
                "original_units": original,
                "removed_units": len(removed_indices),
                "remaining_units": len(remaining_indices),
                "removed_ratio": len(removed_indices) / original,
                "estimated_removed_parameters": estimated_layer_removed,
                "min_keep_constraint": (
                    f"ratio={float(args.min_keep_ratio):.9g};units={minimum_keep_units}"
                ),
                "group_ids_involved": ";".join(str(value) for value in sorted(group_ids)),
            }
        )

    layer_path = diagnosis_dir / f"{args.descriptor_variant}_layer_pruning.csv"
    _atomic_csv(layer_path, LAYER_FIELDS, layer_rows)
    sample_metadata = calibration_loader.summary()
    sample_metadata.update(
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
    sample_path = (
        diagnosis_dir
        / "controlled_runs"
        / f"{args.descriptor_variant}_calibration_samples.json"
    )
    _atomic_json(sample_path, sample_metadata)

    estimated_sparsity = selected_removed_cost / parameter_count_before
    report_sparsity = float(actual_report["sparsity"])
    if abs(estimated_sparsity - report_sparsity) > 1e-10:
        raise RuntimeError(
            "independent selected-cost sparsity disagrees with model report: "
            f"{estimated_sparsity} vs {report_sparsity}"
        )
    baseline_row = {
        "variant": args.descriptor_variant,
        "checkpoint": checkpoint_metadata["path"],
        "baseline_top1": float(baseline_top1),
        "baseline_top5": float(baseline_top5),
        "pre_ft_top1": float(pre_ft_top1),
        "pre_ft_top5": float(pre_ft_top5),
        "top1_drop": float(baseline_top1 - pre_ft_top1),
        "top5_drop": float(baseline_top5 - pre_ft_top5),
    }
    _update_baseline_table(diagnosis_dir / "baseline_and_pruned_metrics.csv", baseline_row)

    result = {
        "status": "controlled_prune_only_complete",
        "variant": args.descriptor_variant,
        "configuration": {
            "checkpoint": checkpoint_metadata["path"],
            "checkpoint_sha256": checkpoint_metadata["sha256"],
            "target_sparsity": float(args.sparsity),
            "sigma": float(args.sigma),
            "gamma_decay": float(args.gamma_decay),
            "min_keep_ratio": float(args.min_keep_ratio),
            "importance_alpha": float(args.importance_alpha),
            "calib_batch_size": int(args.calib_batch_size),
            "calib_batches": int(args.calib_batches),
            "validation_batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "selection_mode": args.selection_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "checkpoint": checkpoint_metadata,
        "validation_dataset": validation_metadata,
        "calibration_samples": sample_metadata,
        "metrics": baseline_row,
        "parameter_budget": {
            "original_parameter_count": int(parameter_count_before),
            "physical_parameter_count_after": int(parameter_count_after),
            "estimated_removable_parameter_count": int(estimated_removable),
            "target_removed_parameter_count": float(parameter_count_before * args.sparsity),
            "selected_removed_parameter_cost": int(selected_removed_cost),
            "estimated_budget_sparsity": float(estimated_sparsity),
            "reported_estimated_sparsity": report_sparsity,
            "physical_numel_sparsity": float(physical_sparsity),
            "accounting_note": (
                "The current Task010 path is logical/index-based. Estimated budget "
                "sparsity is not physical tensor compaction."
            ),
            "removed_heads": int(
                actual_report["original_heads"] - actual_report["remaining_heads"]
            ),
            "removed_ffn_neurons": int(
                actual_report["original_neurons"] - actual_report["remaining_neurons"]
            ),
        },
        "bms_group_count": None if groups is None else len(groups),
        "layer_audit": str(layer_path.resolve()),
    }
    result_path = diagnosis_dir / "controlled_runs" / f"{args.descriptor_variant}.json"
    _atomic_json(result_path, result)
    return result

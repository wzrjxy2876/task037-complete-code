#!/usr/bin/env python3
"""Task040 exact HTOR diagnosis for the existing Video Swin model.

The probe is intentionally diagnostic-only.  It uses the validated Task037
adapter for model construction, checkpoint loading, balanced UCF101 sampling,
and pruning-unit discovery, but computes a new score from raw true-class
logits and temporary whole-unit masks.  It never physically prunes or trains.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SelectedUnit:
    global_index: int
    layer_name: str
    unit_type: str
    unit_index: int
    spec: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task040 exact HTOR diagnosis")
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--layers", default="representative")
    parser.add_argument(
        "--unit",
        action="append",
        default=[],
        help="Explicit unit: LAYER_REGEX:head|neuron:INDEX; repeatable",
    )
    parser.add_argument("--units_per_layer", type=int, default=3)
    parser.add_argument("--intervention_batch_size", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def ensure_project_importable(project_root: Path) -> None:
    """Add only package-local source roots; no historical checkout is needed."""

    candidates = [
        project_root,
        project_root / "src" / "lgfr_runtime",
        project_root / "src",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Task040 requested CUDA, but CUDA is unavailable")
    if requested.type == "cuda":
        torch.cuda.set_device(requested)
    return requested


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(project_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "backbone.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict/model is not a mapping")
    return {
        normalize_checkpoint_key(key): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def make_checkpoint_identity(
    model: nn.Module,
    checkpoint_path: Path,
    adapter_metadata: Mapping[str, Any],
    all_specs: Sequence[Any],
) -> dict[str, Any]:
    """Audit the already-loaded adapter result without changing load semantics."""

    state = checkpoint_state(checkpoint_path)
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            compatible[key] = value

    missing = sorted(key for key in model_state if key not in compatible)
    non_head_missing = [key for key in missing if not key.startswith("cls_head.")]
    non_head_mismatches = [
        item for item in shape_mismatches if not item["key"].startswith("cls_head.")
    ]
    if non_head_missing or non_head_mismatches:
        raise RuntimeError(
            "Task037 adapter loading semantics were not satisfied: "
            f"non_head_missing={non_head_missing[:8]}, "
            f"non_head_mismatches={non_head_mismatches[:8]}"
        )

    head_state_keys = sorted(
        key for key in state if key.startswith("cls_head.")
    )
    head_loaded_keys = sorted(
        key for key in compatible if key.startswith("cls_head.")
    )
    head_missing_keys = sorted(
        key for key in missing if key.startswith("cls_head.")
    )
    head_mismatch_keys = sorted(
        item["key"] for item in shape_mismatches if item["key"].startswith("cls_head.")
    )
    if head_mismatch_keys:
        head_status = "shape_mismatch_allowed_by_task037_adapter"
    elif head_missing_keys:
        head_status = "missing_allowed_by_task037_adapter"
    else:
        head_status = "loaded"

    head_count = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "head")
    neuron_count = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "neuron")
    return {
        "task": "task040",
        "checkpoint_absolute_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "loaded_tensor_count": len(compatible),
        "loaded_parameter_count": int(sum(value.numel() for value in compatible.values())),
        "missing_keys": missing,
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": shape_mismatches,
        "non_head_missing_keys": non_head_missing,
        "non_head_shape_mismatches": non_head_mismatches,
        "classifier_head": {
            "status": head_status,
            "checkpoint_keys": head_state_keys,
            "loaded_keys": head_loaded_keys,
            "missing_keys": head_missing_keys,
            "shape_mismatch_keys": head_mismatch_keys,
        },
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "discovered_pruning_layer_count": len(all_specs),
        "discovered_attention_head_count": head_count,
        "discovered_ffn_neuron_count": neuron_count,
        "adapter_metadata": dict(adapter_metadata),
    }


def normalize_unit_type(value: str) -> str:
    value = value.strip().lower()
    aliases = {"head": "head", "attention": "head", "neuron": "neuron", "ffn": "neuron"}
    if value not in aliases:
        raise ValueError(f"unsupported unit type: {value!r}")
    return aliases[value]


def representative_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    return sorted(set(np.linspace(0, count - 1, min(count, limit)).round().astype(int).tolist()))


def select_units(
    all_specs: Sequence[Any],
    layer_selector: str,
    explicit_units: Sequence[str],
    units_per_layer: int,
    ctfrs_module: Any,
) -> list[SelectedUnit]:
    selected_specs = ctfrs_module.filter_layers(all_specs, layer_selector)
    head_total = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "head")
    offsets: dict[str, int] = {}
    seen_by_type = {"head": 0, "neuron": 0}
    for spec in all_specs:
        offsets[spec.name] = seen_by_type[spec.unit_type]
        seen_by_type[spec.unit_type] += int(spec.num_units)

    def global_unit_index(spec: Any, unit_index: int) -> int:
        type_offset = 0 if spec.unit_type == "head" else head_total
        return type_offset + offsets[spec.name] + unit_index

    selected: list[SelectedUnit] = []

    if explicit_units:
        for expression in explicit_units:
            try:
                layer_pattern, unit_type, unit_text = expression.rsplit(":", 2)
                pattern = re.compile(layer_pattern)
                unit_type = normalize_unit_type(unit_type)
                unit_index = int(unit_text)
            except ValueError as exc:
                raise ValueError(
                    "--unit must have the form LAYER_REGEX:head|neuron:INDEX"
                ) from exc
            matches = [spec for spec in selected_specs if pattern.search(spec.name)]
            if not matches:
                raise ValueError(f"--unit layer regex matched no selected layer: {layer_pattern!r}")
            for spec in matches:
                if spec.unit_type != unit_type:
                    continue
                if not 0 <= unit_index < int(spec.num_units):
                    raise ValueError(
                        f"unit index {unit_index} is outside {spec.name} size {spec.num_units}"
                    )
                selected.append(
                    SelectedUnit(
                        global_index=global_unit_index(spec, unit_index),
                        layer_name=spec.name,
                        unit_type=unit_type,
                        unit_index=unit_index,
                        spec=spec,
                    )
                )
    else:
        for spec in selected_specs:
            for unit_index in representative_indices(int(spec.num_units), units_per_layer):
                selected.append(
                    SelectedUnit(
                        global_index=global_unit_index(spec, unit_index),
                        layer_name=spec.name,
                        unit_type=spec.unit_type,
                        unit_index=unit_index,
                        spec=spec,
                    )
                )

    unique: dict[tuple[str, str, int], SelectedUnit] = {}
    for unit in selected:
        unique[(unit.layer_name, unit.unit_type, unit.unit_index)] = unit
    selected = list(unique.values())
    selected.sort(key=lambda item: (item.global_index, item.unit_type, item.unit_index))
    if not selected:
        raise ValueError("no representative units were selected")
    selected_types = {unit.unit_type for unit in selected}
    if selected_types != {"head", "neuron"}:
        raise ValueError(
            "Task040 representative selection must include both attention heads and FFN neurons"
        )
    return selected


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_video_manifest(loader: Any, selected_indices: Sequence[int], output_path: Path) -> None:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    rows: list[dict[str, Any]] = []
    for order, index in enumerate(selected_indices):
        directory, duration, label = dataset.clips[int(index)]
        rows.append(
            {
                "video_index": order,
                "dataset_index": int(index),
                "video_id": str(directory),
                "duration": int(duration),
                "label": int(label),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["video_index"])
        writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def temporary_unit_mask(spec: Any, unit_index: int):
    """Match the validated Task037 whole-head/whole-neuron hook semantics."""

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        values = inputs[0].clone()
        if spec.unit_type == "head":
            head_dim = int(spec.module.head_dim)
            expected = int(spec.num_units) * head_dim
            if values.shape[-1] != expected:
                raise ValueError(
                    f"attention projection input shape {tuple(values.shape)} does not match "
                    f"{spec.num_units} heads x {head_dim}"
                )
            reshaped = values.reshape(*values.shape[:-1], int(spec.num_units), head_dim)
            reshaped[..., unit_index, :] = 0.0
            values = reshaped.reshape_as(values)
        elif spec.unit_type == "neuron":
            values[..., unit_index] = 0.0
        else:
            raise ValueError(f"unsupported pruning unit type: {spec.unit_type!r}")
        return (values,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def unwrap_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"could not extract logits from {type(output)!r}")


def true_class_logit(model: nn.Module, videos: torch.Tensor, label: int) -> float:
    with torch.no_grad():
        logits = unwrap_logits(model(videos))
    if logits.ndim != 2 or logits.shape[0] != videos.shape[0]:
        raise ValueError(f"unexpected model logits shape: {tuple(logits.shape)}")
    if not 0 <= int(label) < logits.shape[1]:
        raise ValueError(f"ground-truth label {label} is outside logits dimension {logits.shape[1]}")
    value = logits[:, int(label)].detach().to(dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("non-finite true-class raw logit")
    return float(value[0].item())


def infer_temporal_length(videos: torch.Tensor) -> int:
    if videos.ndim != 5:
        raise ValueError(
            "Task040 expected the existing UCF101 loader layout [B,C,T,H,W]; "
            f"received {tuple(videos.shape)}"
        )
    return int(videos.shape[2])


def intervention_batches(
    clip: torch.Tensor,
    interventions: Sequence[Any],
    batch_size: int,
    core: Any,
) -> list[tuple[list[Any], torch.Tensor]]:
    batches: list[tuple[list[Any], torch.Tensor]] = []
    for start in range(0, len(interventions), batch_size):
        group = list(interventions[start : start + batch_size])
        values = core.apply_temporal_interventions(clip, group, time_dim=1)
        batches.append((group, values))
    return batches


def write_raw_header(handle: Any) -> None:
    fieldnames = [
        "video_index", "video_id", "label", "unit_global_index", "layer_name",
        "unit_type", "unit_index", "level", "block_size", "pair_index",
        "z_true_original", "z_true_original_masked", "z_true_intervened",
        "z_true_intervened_masked", "d_original", "d_intervened", "tau",
    ]
    handle.write(",".join(fieldnames) + "\n")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.intervention_batch_size <= 0:
        raise ValueError("--intervention_batch_size must be positive")

    ensure_project_importable(project_root)
    set_seed(args.seed)
    device = resolve_device(args.device)
    core = importlib.import_module("task040_htor_core")
    ctfrs = importlib.import_module("probe_ctfrs_dynamic_function")
    adapter = importlib.import_module(args.adapter)

    model, adapter_metadata = adapter.build_model_for_probe(
        checkpoint=str(checkpoint_path), device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_specs = ctfrs.discover_unit_layers(model)
    selected_units = select_units(
        all_specs=all_specs,
        layer_selector=args.layers,
        explicit_units=args.unit,
        units_per_layer=args.units_per_layer,
        ctfrs_module=ctfrs,
    )

    identity = make_checkpoint_identity(
        model=model,
        checkpoint_path=checkpoint_path,
        adapter_metadata=adapter_metadata,
        all_specs=all_specs,
    )
    identity.update({
        "device": str(device),
        "dtype": "torch.float32",
        "amp": False,
        "git_branch": git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git_value(project_root, "rev-parse", "HEAD"),
    })
    write_json(output_dir / "task040_checkpoint_identity.json", identity)

    loader, selected_indices, chosen_classes = ctfrs.build_balanced_loader(
        project_root=project_root,
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    write_video_manifest(loader, selected_indices, output_dir / "task040_video_manifest.csv")

    first_batch = next(iter(loader))
    first_videos = first_batch[0].float().to(device, non_blocking=True)
    first_label = int(first_batch[1][0].item())
    actual_t = infer_temporal_length(first_videos)
    interventions = core.enumerate_hierarchical_interventions(actual_t)
    core.verify_intervention_identity(interventions, actual_t)
    identity["baseline_model_sanity"] = {
        "output_shape": list(unwrap_logits(model(first_videos)).shape),
        "actual_T": actual_t,
        "first_label": first_label,
        "finite": True,
    }
    write_json(output_dir / "task040_checkpoint_identity.json", identity)

    manifest = {
        "task": "task040",
        "method": "HTOR",
        "actual_T": actual_t,
        "time_dim_for_model_input": 2,
        "index_base": 0,
        "end_index_is_exclusive": True,
        "num_levels": len({item.level for item in interventions}),
        "num_interventions": len(interventions),
        "interventions": [asdict(item) for item in interventions],
    }
    write_json(output_dir / "task040_intervention_manifest.json", manifest)

    fieldnames = [
        "video_index", "video_id", "label", "unit_global_index", "layer_name",
        "unit_type", "unit_index", "level", "block_size", "pair_index",
        "z_true_original", "z_true_original_masked", "z_true_intervened",
        "z_true_intervened_masked", "d_original", "d_intervened", "tau",
    ]
    tau_by_unit_level: dict[int, dict[int, list[float]]] = {
        unit.global_index: {} for unit in selected_units
    }
    unit_key = lambda item: item.global_index

    # The exact mask oracle is intentionally kept on one device.  A hook tied to
    # one layer is not silently replicated through DataParallel.
    mask_restore_exact = True
    video_count = 0
    with (output_dir / "task040_raw_records.csv").open("w", newline="", encoding="utf-8") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=fieldnames)
        writer.writeheader()
        try:
            from tqdm import tqdm
            iterator: Iterable[Any] = tqdm(loader, desc="Task040 HTOR", ncols=100)
        except ImportError:
            iterator = loader
        for video_index, batch in enumerate(iterator):
            videos, targets, dataset_indices = batch[:3]
            videos = videos.float().to(device, non_blocking=True)
            label = int(targets[0].item())
            dataset_index = int(dataset_indices[0].item())
            clip = videos[0]
            baseline = true_class_logit(model, videos, label)
            cached_batches = intervention_batches(
                clip, interventions, args.intervention_batch_size, core
            )
            intervened_logits: list[float] = []
            for _, values in cached_batches:
                with torch.no_grad():
                    logits = unwrap_logits(model(values))
                intervened_logits.extend(
                    float(logits[:, label].detach().to(dtype=torch.float64).cpu()[index].item())
                    for index in range(logits.shape[0])
                )
            if len(intervened_logits) != len(interventions):
                raise AssertionError("intervention logit cache has the wrong length")

            masked_original: dict[int, float] = {}
            masked_intervened: dict[int, list[float]] = {}
            for unit in selected_units:
                key = unit_key(unit)
                with temporary_unit_mask(unit.spec, unit.unit_index):
                    masked_original[key] = true_class_logit(model, videos, label)
                    values_for_unit: list[float] = []
                    for _, values in cached_batches:
                        with torch.no_grad():
                            logits = unwrap_logits(model(values))
                        values_for_unit.extend(
                            float(logits[:, label].detach().to(dtype=torch.float64).cpu()[index].item())
                            for index in range(logits.shape[0])
                        )
                    masked_intervened[key] = values_for_unit

                restored = true_class_logit(model, videos, label)
                if restored != baseline:
                    mask_restore_exact = False

            dataset = loader.dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            video_id = str(dataset.clips[dataset_index][0])
            for unit in selected_units:
                key = unit_key(unit)
                d_original = baseline - masked_original[key]
                for intervention_index, spec in enumerate(interventions):
                    d_intervened = intervened_logits[intervention_index] - masked_intervened[key][intervention_index]
                    tau = core.compute_tau(d_original, d_intervened, eps=args.eps).item()
                    tau_by_unit_level[key].setdefault(int(spec.level), []).append(float(tau))
                    writer.writerow({
                        "video_index": video_index,
                        "video_id": video_id,
                        "label": label,
                        "unit_global_index": unit.global_index,
                        "layer_name": unit.layer_name,
                        "unit_type": unit.unit_type,
                        "unit_index": unit.unit_index,
                        "level": spec.level,
                        "block_size": spec.block_size,
                        "pair_index": spec.pair_index,
                        "z_true_original": baseline,
                        "z_true_original_masked": masked_original[key],
                        "z_true_intervened": intervened_logits[intervention_index],
                        "z_true_intervened_masked": masked_intervened[key][intervention_index],
                        "d_original": d_original,
                        "d_intervened": d_intervened,
                        "tau": tau,
                    })
            video_count += 1

    if not mask_restore_exact:
        raise RuntimeError("model output changed after a temporary mask was removed")

    max_level = max(item.level for item in interventions)
    summary_fieldnames = [
        "layer_name", "unit_type", "unit_index", "unit_global_index",
        *[f"level_{level}_rms" for level in range(max_level + 1)], "HTOR",
    ]
    with (output_dir / "task040_unit_level_summary.csv").open("w", newline="", encoding="utf-8") as summary_handle:
        writer = csv.DictWriter(summary_handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for unit in selected_units:
            key = unit_key(unit)
            per_level = {
                level: core.compute_level_rms(values)
                for level, values in tau_by_unit_level[key].items()
            }
            htor = core.compute_htor(per_level)
            row = {
                "layer_name": unit.layer_name,
                "unit_type": unit.unit_type,
                "unit_index": unit.unit_index,
                "unit_global_index": unit.global_index,
                "HTOR": float(htor.item()),
            }
            row.update({
                f"level_{level}_rms": float(per_level[level].item())
                for level in range(max_level + 1)
            })
            writer.writerow(row)

    runtime_arguments = vars(args).copy()
    runtime_arguments["resolved_project_root"] = str(project_root)
    runtime_arguments["resolved_checkpoint"] = str(checkpoint_path)
    runtime_arguments["resolved_output_dir"] = str(output_dir)
    runtime_arguments["resolved_device"] = str(device)
    runtime_arguments["selected_classes"] = [int(value) for value in chosen_classes]
    runtime_arguments["selected_indices"] = [int(value) for value in selected_indices]
    runtime_arguments["selected_layers"] = sorted({unit.layer_name for unit in selected_units})
    runtime_arguments["selected_units"] = [
        {
            "global_index": unit.global_index,
            "layer_name": unit.layer_name,
            "unit_type": unit.unit_type,
            "unit_index": unit.unit_index,
        }
        for unit in selected_units
    ]
    summary = {
        "task": "task040",
        "method": "HTOR",
        "seed": int(args.seed),
        "num_classes": int(args.num_classes),
        "videos_per_class": int(args.videos_per_class),
        "num_videos": int(video_count),
        "actual_T": int(actual_t),
        "num_levels": int(max_level + 1),
        "num_interventions": int(len(interventions)),
        "selected_layer_count": int(len(set(unit.layer_name for unit in selected_units))),
        "selected_unit_count": int(len(selected_units)),
        "FP32": True,
        "AMP": False,
        "target": "true_class_raw_logit",
        "checkpoint": str(checkpoint_path),
        "git_branch": identity["git_branch"],
        "git_commit": identity["git_commit"],
        "no_physical_pruning": True,
        "no_finetuning": True,
        "no_contribution_field": True,
        "mask_restore_exact": mask_restore_exact,
        "stage_status": {
            "A": "PASS",
            "B": "PASS",
            "C": "PASS",
            "D": "COMPLETED_UNJUDGED",
        },
        "runtime_arguments": runtime_arguments,
    }
    write_json(output_dir / "task040_summary.json", summary)
    return summary


def main() -> None:
    summary = run_probe(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


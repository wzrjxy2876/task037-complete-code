"""Structural-equivalent trainable parameter accounting for Task038.

Task038 remains logical pruning.  This module simulates the tensor shapes of
the physically channel-sliced SlowFast graph and counts each trainable tensor
exactly once.  It is deliberately independent from the frozen F3 score.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping

from torch import nn


_FAST_RE = re.compile(r"^fast_res([2-5])\.([0-9]+)\.(conv[123])$")
_SLOW_RE = re.compile(r"^slow_res([2-5])\.([0-9]+)\.(conv[123])$")
_LAT_RE = re.compile(r"^lateral_(p1|res[2-4])\.0$")

SHAPE_FIELDS = [
    "module_name", "module_type", "original_in_channels",
    "remaining_in_channels", "original_out_channels", "remaining_out_channels",
    "kernel_t", "kernel_h", "kernel_w", "original_parameters",
    "remaining_parameters", "removed_parameters", "dependency_source",
]


def _is_candidate(name: str) -> bool:
    return bool(_FAST_RE.match(name) or _SLOW_RE.match(name) or _LAT_RE.match(name))


def _kernel(module: nn.Conv3d) -> tuple[int, int, int]:
    value = tuple(int(x) for x in module.kernel_size)
    if len(value) != 3:
        raise ValueError(f"Task038 expects Conv3d kernels, got {value}")
    return value


def _trainable_numel(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters(recurse=False) if p.requires_grad)


def _load_layers(registry_or_keep_state: Any) -> Mapping[str, Any]:
    if registry_or_keep_state is None:
        return {}
    if isinstance(registry_or_keep_state, (str, Path)):
        payload = json.loads(Path(registry_or_keep_state).read_text(encoding="utf-8"))
    else:
        payload = registry_or_keep_state
    if not isinstance(payload, Mapping):
        raise TypeError("registry_or_keep_state must be a mapping or JSON path")
    layers = payload.get("layers", payload)
    if not isinstance(layers, Mapping):
        raise TypeError("registry layers must be a mapping")
    return layers


def _keep_state(model: nn.Module, registry_or_keep_state: Any) -> dict[str, tuple[int, ...]]:
    layers = _load_layers(registry_or_keep_state)
    result: dict[str, tuple[int, ...]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv3d) or not _is_candidate(name):
            continue
        width = int(module.out_channels)
        entry = layers.get(name)
        if entry is None:
            keep = list(range(width))
        elif isinstance(entry, Mapping):
            keep = entry.get("keep")
            if keep is None and entry.get("keep_count") is not None:
                keep_count = int(entry["keep_count"])
                if keep_count < 0 or keep_count > width:
                    raise ValueError(f"keep_count out of range for {name}")
                keep = range(keep_count)
            if keep is None:
                pruned = {int(x) for x in entry.get("pruned", [])}
                keep = [x for x in range(width) if x not in pruned]
        else:
            keep = entry
        values = tuple(sorted({int(x) for x in keep}))
        if any(x < 0 or x >= width for x in values):
            raise ValueError(f"keep index out of range for {name}")
        result[name] = values
    return result


def registry_from_pruned_indices(model: nn.Module, inventory: Any, pruned_indices: set[int]) -> dict[str, Any]:
    """Build a registry-shaped keep state from the canonical unit inventory."""
    modules = dict(model.named_modules())
    grouped: dict[str, list[Any]] = {}
    for unit in inventory.units:
        grouped.setdefault(unit.module_name, []).append(unit)
    layers: dict[str, Any] = {}
    for name, units in sorted(grouped.items()):
        width = int(units[0].out_channels)
        pruned = sorted(int(unit.local_channel_index) for unit in units if int(unit.global_index) in pruned_indices)
        if name not in modules:
            raise RuntimeError(f"inventory module missing: {name}")
        pruned_set = set(pruned)
        keep = [x for x in range(width) if x not in pruned_set]
        layers[name] = {
            "total": width, "pruned": pruned, "keep": keep,
            "pruned_count": len(pruned), "keep_count": len(keep),
        }
    return {"layers": layers}


def _conv_row_spec(specs: dict[str, dict[str, Any]], name: str, in_channels: int,
                   out_channels: int, dependency_source: str, adjusted: bool = True) -> None:
    specs[name] = {"remaining_in_channels": int(in_channels),
                   "remaining_out_channels": int(out_channels),
                   "dependency_source": dependency_source, "adjusted": adjusted}


def _bn_row_spec(specs: dict[str, dict[str, Any]], name: str, channels: int,
                 dependency_source: str, adjusted: bool = True) -> None:
    specs[name] = {"remaining_in_channels": int(channels),
                   "remaining_out_channels": int(channels),
                   "dependency_source": dependency_source, "adjusted": adjusted}


def _block_names(model: nn.Module, pathway: str, stage: int) -> list[str]:
    prefix = f"{pathway}_res{stage}"
    sequence = getattr(model, prefix)
    return [f"{prefix}.{i}" for i in range(len(sequence))]


def _append_pathway(model: nn.Module, modules: Mapping[str, nn.Module],
                    keep: Mapping[str, tuple[int, ...]], specs: dict[str, dict[str, Any]],
                    pathway: str, initial_channels: int) -> dict[int, int]:
    """Add all Bottleneck and downsample shapes for one pathway."""
    outputs: dict[int, int] = {}
    current = int(initial_channels)
    for stage in range(2, 6):
        for block_name in _block_names(model, pathway, stage):
            conv1, conv2, conv3 = (f"{block_name}.conv1", f"{block_name}.conv2", f"{block_name}.conv3")
            k1, k2, k3 = len(keep[conv1]), len(keep[conv2]), len(keep[conv3])
            _conv_row_spec(specs, conv1, current, k1, f"{pathway}:previous_output")
            _bn_row_spec(specs, f"{block_name}.bn1", k1, f"{conv1}:output")
            _conv_row_spec(specs, conv2, k1, k2, f"{conv1}:output")
            _bn_row_spec(specs, f"{block_name}.bn2", k2, f"{conv2}:output")
            _conv_row_spec(specs, conv3, k2, k3, f"{conv2}:output")
            _bn_row_spec(specs, f"{block_name}.bn3", k3, f"{conv3}:output")
            downsample_name = f"{block_name}.downsample.0"
            if downsample_name in modules:
                _conv_row_spec(specs, downsample_name, current, k3, f"{conv3}:residual_output")
                _bn_row_spec(specs, f"{block_name}.downsample.1", k3, f"{downsample_name}:output")
            current = k3
        outputs[stage] = current
    return outputs


def _append_fixed_stems(modules: Mapping[str, nn.Module], specs: dict[str, dict[str, Any]]) -> None:
    for name in ("fast_conv1", "slow_conv1"):
        module = modules[name]
        _conv_row_spec(specs, name, int(module.in_channels), int(module.out_channels), "fixed_stem", adjusted=False)
    for name in ("fast_bn1", "slow_bn1"):
        module = modules[name]
        _bn_row_spec(specs, name, int(module.num_features), "fixed_stem", adjusted=False)


def _row_for_module(name: str, module: nn.Module, spec: Mapping[str, Any]) -> dict[str, Any]:
    adjusted = bool(spec["adjusted"])
    if isinstance(module, nn.Conv3d):
        original_in, original_out = int(module.in_channels), int(module.out_channels)
        kernel_t, kernel_h, kernel_w = _kernel(module)
        remaining_in = int(spec["remaining_in_channels"]) if adjusted else original_in
        remaining_out = int(spec["remaining_out_channels"]) if adjusted else original_out
        bias = int(module.bias is not None and module.bias.requires_grad)
        original = _trainable_numel(module)
        remaining = remaining_out * remaining_in * kernel_t * kernel_h * kernel_w + (remaining_out if bias else 0)
    elif isinstance(module, nn.BatchNorm3d):
        original_in = original_out = int(module.num_features)
        remaining_in = remaining_out = int(spec["remaining_in_channels"]) if adjusted else original_in
        kernel_t = kernel_h = kernel_w = 1
        original = _trainable_numel(module)
        remaining = 0
        if getattr(module, "affine", False):
            if module.weight is not None and module.weight.requires_grad:
                remaining += remaining_out
            if module.bias is not None and module.bias.requires_grad:
                remaining += remaining_out
    elif isinstance(module, nn.Linear):
        original_in, original_out = int(module.in_features), int(module.out_features)
        remaining_in, remaining_out = (int(spec["remaining_in_channels"]) if adjusted else original_in), original_out
        kernel_t = kernel_h = kernel_w = 1
        original = _trainable_numel(module)
        remaining = remaining_out * remaining_in
        if module.bias is not None and module.bias.requires_grad:
            remaining += remaining_out
    else:
        raise TypeError(f"unsupported trainable module {name}: {type(module).__name__}")
    return {
        "module_name": name, "module_type": type(module).__name__,
        "original_in_channels": original_in, "remaining_in_channels": remaining_in,
        "original_out_channels": original_out, "remaining_out_channels": remaining_out,
        "kernel_t": kernel_t, "kernel_h": kernel_h, "kernel_w": kernel_w,
        "original_parameters": int(original), "remaining_parameters": int(remaining),
        "removed_parameters": int(original - remaining),
        "dependency_source": str(spec["dependency_source"]),
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SHAPE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def count_structural_parameters(model: nn.Module, registry_or_keep_state: Any = None,
                                 dependency_graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Count trainable parameters of the equivalent physically sliced graph."""
    del dependency_graph
    modules = dict(model.named_modules())
    keep = _keep_state(model, registry_or_keep_state)
    specs: dict[str, dict[str, Any]] = {}
    _append_fixed_stems(modules, specs)
    fast_initial = int(modules["fast_conv1"].out_channels)
    fast_outputs = _append_pathway(model, modules, keep, specs, "fast", fast_initial)

    lateral_outputs: dict[str, int] = {}
    lateral_sources = {"p1": fast_initial, "res2": fast_outputs[2],
                       "res3": fast_outputs[3], "res4": fast_outputs[4]}
    for suffix in ("p1", "res2", "res3", "res4"):
        conv_name, bn_name = f"lateral_{suffix}.0", f"lateral_{suffix}.1"
        k_out = len(keep[conv_name])
        lateral_outputs[suffix] = k_out
        source = "fast_stem:output" if suffix == "p1" else f"fast_{suffix}:output"
        _conv_row_spec(specs, conv_name, lateral_sources[suffix], k_out, source)
        _bn_row_spec(specs, bn_name, k_out, f"{conv_name}:output")

    slow_initial = int(modules["slow_conv1"].out_channels)
    slow_outputs: dict[int, int] = {}
    current = slow_initial
    for stage in range(2, 6):
        suffix = "p1" if stage == 2 else f"res{stage - 1}"
        stage_input = current + lateral_outputs[suffix]
        first = True
        for block_name in _block_names(model, "slow", stage):
            block_input = stage_input if first else current
            first = False
            conv1, conv2, conv3 = (f"{block_name}.conv1", f"{block_name}.conv2", f"{block_name}.conv3")
            k1, k2, k3 = len(keep[conv1]), len(keep[conv2]), len(keep[conv3])
            _conv_row_spec(specs, conv1, block_input, k1, "slow:previous_output_plus_lateral")
            _bn_row_spec(specs, f"{block_name}.bn1", k1, f"{conv1}:output")
            _conv_row_spec(specs, conv2, k1, k2, f"{conv1}:output")
            _bn_row_spec(specs, f"{block_name}.bn2", k2, f"{conv2}:output")
            _conv_row_spec(specs, conv3, k2, k3, f"{conv2}:output")
            _bn_row_spec(specs, f"{block_name}.bn3", k3, f"{conv3}:output")
            downsample_name = f"{block_name}.downsample.0"
            if downsample_name in modules:
                _conv_row_spec(specs, downsample_name, block_input, k3, f"{conv3}:residual_output")
                _bn_row_spec(specs, f"{block_name}.downsample.1", k3, f"{downsample_name}:output")
            current = k3
        slow_outputs[stage] = current

    fc = modules["fc"]
    final_features = slow_outputs[5] + fast_outputs[5]
    specs["fc"] = {"remaining_in_channels": final_features,
                    "remaining_out_channels": int(fc.out_features),
                    "dependency_source": "slow_res5_output_plus_fast_res5_output",
                    "adjusted": True}

    rows: list[dict[str, Any]] = []
    expected_trainable_modules: set[str] = set()
    for name, module in model.named_modules():
        if _trainable_numel(module):
            expected_trainable_modules.add(name)
            if name not in specs:
                raise RuntimeError(f"unaccounted trainable module: {name}")
            rows.append(_row_for_module(name, module, specs[name]))
    original = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    remaining = sum(int(row["remaining_parameters"]) for row in rows)
    if len(rows) != len(expected_trainable_modules):
        raise RuntimeError("trainable module accounting is not one-to-one")
    if not rows:
        raise RuntimeError("model has no trainable modules")
    actual = sum(int(p.numel()) for p in model.parameters())
    shape_sha = hashlib.sha256(_csv_bytes(rows)).hexdigest()
    return {
        "original_trainable_parameters": original,
        "structural_equivalent_remaining_parameters": remaining,
        "structural_equivalent_removed_parameters": original - remaining,
        "remaining_parameter_ratio": remaining / original,
        "parameter_pruning_ratio": (original - remaining) / original,
        "actual_state_dict_parameters": actual,
        "actual_state_dict_parameter_ratio": actual / original,
        "logical_pruning": True, "physical_pruning_executed": False,
        "parameterized_module_count": len(rows),
        "fixed_module_count": sum(not bool(specs[row["module_name"]]["adjusted"]) for row in rows),
        "shape_adjusted_module_count": sum(bool(specs[row["module_name"]]["adjusted"]) for row in rows),
        "unaccounted_trainable_parameters": 0, "double_counted_parameters": 0,
        "module_shape_sha256": shape_sha,
        "final_fast_feature_channels": int(fast_outputs[5]),
        "final_slow_feature_channels": int(slow_outputs[5]),
        "final_classifier_input_features": int(final_features),
        "module_parameter_shapes": rows,
    }


def write_parameter_accounting(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write the deterministic module table and summary JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(report["module_parameter_shapes"])
    csv_path = out / "module_parameter_shapes.csv"
    csv_path.write_bytes(_csv_bytes(rows))
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if digest != report["module_shape_sha256"]:
        raise RuntimeError("parameter accounting table hash changed while writing")
    summary = {key: value for key, value in report.items() if key != "module_parameter_shapes"}
    summary["module_shape_sha256"] = digest
    (out / "parameter_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["SHAPE_FIELDS", "count_structural_parameters", "registry_from_pruned_indices", "write_parameter_accounting"]

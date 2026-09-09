"""Canonical SlowFast pruning-unit inventory for Task038."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .slowfast_model_task038 import attach_mask, set_mask

_FAST_RE = re.compile(r"^fast_res([2-5])\.([0-9]+)\.(conv[123])$")
_SLOW_RE = re.compile(r"^slow_res([2-5])\.([0-9]+)\.(conv[123])$")
_LAT_RE = re.compile(r"^lateral_(p1|res[2-4])\.0$")


@dataclass(frozen=True)
class Unit:
    global_index: int
    layer_name: str
    module_name: str
    local_channel_index: int
    pathway: str
    stage: str
    block: int | None
    conv_position: str
    unit_type: str
    out_channels: int
    parameter_cost: int
    dependency_group: str
    bn_module_name: str
    downsample_module_name: str | None = None


@dataclass
class UnitInventory:
    units: list[Unit]
    total_model_parameters: int
    candidate_parameter_capacity: int
    max_achievable_analytical_sparsity: float
    min_keep_ratio: float

    @property
    def num_units(self) -> int:
        return len(self.units)

    def to_json(self) -> dict[str, Any]:
        return {
            "total_model_parameters": self.total_model_parameters,
            "candidate_parameter_capacity": self.candidate_parameter_capacity,
            "max_achievable_analytical_sparsity": self.max_achievable_analytical_sparsity,
            "min_keep_ratio": self.min_keep_ratio,
            "num_units": len(self.units),
            "units": [asdict(unit) for unit in self.units],
        }

    def write(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _bn_name(name: str) -> str:
    if name.endswith(".conv1"):
        return name[:-6] + ".bn1"
    if name.endswith(".conv2"):
        return name[:-6] + ".bn2"
    if name.endswith(".conv3"):
        return name[:-6] + ".bn3"
    if name.startswith("lateral_"):
        return name[:-2] + ".1"
    raise ValueError(name)


def _cost(conv: nn.Conv3d, bn: nn.Module | None, downsample: nn.Conv3d | None) -> int:
    per = int(conv.weight[0].numel()) + (1 if conv.bias is not None else 0)
    if bn is not None and getattr(bn, "affine", False):
        per += 2
    if downsample is not None:
        per += int(downsample.weight[0].numel()) + (1 if downsample.bias is not None else 0)
        ds_bn = getattr(downsample, "_task038_bn", None)
        if ds_bn is not None and getattr(ds_bn, "affine", False):
            per += 2
    return int(per)


def _sort_key(item: tuple[str, nn.Conv3d]):
    name, _ = item
    m = _FAST_RE.match(name)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), int(m.group(3)[-1]), 0)
    m = _LAT_RE.match(name)
    if m:
        order = {"p1": 0, "res2": 1, "res3": 2, "res4": 3}[m.group(1)]
        return (1, order, 0, 0, 0)
    m = _SLOW_RE.match(name)
    if m:
        return (2, int(m.group(1)), int(m.group(2)), int(m.group(3)[-1]), 0)
    return (99, name)


def build_inventory(model: nn.Module, min_keep_ratio: float = 0.1) -> UnitInventory:
    if not 0.0 < min_keep_ratio <= 1.0:
        raise ValueError("min_keep_ratio must be in (0,1]")
    modules = dict(model.named_modules())
    candidates: list[tuple[str, nn.Conv3d]] = []
    for name, module in modules.items():
        if not isinstance(module, nn.Conv3d):
            continue
        if _FAST_RE.match(name) or _SLOW_RE.match(name) or _LAT_RE.match(name):
            candidates.append((name, module))
    candidates.sort(key=_sort_key)
    units: list[Unit] = []
    for layer_name, conv in candidates:
        fast = _FAST_RE.match(layer_name)
        slow = _SLOW_RE.match(layer_name)
        lat = _LAT_RE.match(layer_name)
        if lat:
            pathway = "lateral"
            stage = lat.group(1)
            block = None
            position = "lateral"
            bn_name = _bn_name(layer_name)
            ds_name = None
            dependency = f"lateral:{layer_name}"
        else:
            pathway = "fast" if fast else "slow"
            match = fast or slow
            assert match is not None
            stage = f"res{match.group(1)}"
            block = int(match.group(2))
            position = match.group(3)
            bn_name = _bn_name(layer_name)
            ds_name = None
            dependency = layer_name
            if position == "conv3":
                block_name = layer_name.rsplit(".", 1)[0]
                ds = modules.get(block_name + ".downsample.0")
                if isinstance(ds, nn.Conv3d):
                    ds_name = block_name + ".downsample.0"
                dependency = f"{block_name}:conv3_residual"
        bn = modules.get(bn_name)
        if bn is not None and not hasattr(bn, "affine"):
            bn = None
        downsample = modules.get(ds_name) if ds_name else None
        if isinstance(downsample, nn.Conv3d):
            object.__setattr__(downsample, "_task038_bn", modules.get(ds_name[:-1] + "1"))
        else:
            downsample = None
        cost = _cost(conv, bn, downsample)
        attach_mask(conv, int(conv.out_channels))
        if downsample is not None:
            attach_mask(downsample, int(downsample.out_channels))
        for channel in range(int(conv.out_channels)):
            units.append(
                Unit(
                    global_index=len(units),
                    layer_name=layer_name,
                    module_name=layer_name,
                    local_channel_index=channel,
                    pathway=pathway,
                    stage=stage,
                    block=block,
                    conv_position=position,
                    unit_type="conv_channel",
                    out_channels=int(conv.out_channels),
                    parameter_cost=cost,
                    dependency_group=dependency,
                    bn_module_name=bn_name,
                    downsample_module_name=ds_name,
                )
            )
    total = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    capacity = sum(
        unit.parameter_cost
        for unit in units
        if unit.local_channel_index
        < int(math.floor(unit.out_channels * (1.0 - min_keep_ratio) + 1e-12))
    )
    maximum = float(capacity / total) if total else 0.0
    if maximum < 0.5:
        raise RuntimeError(
            f"max achievable analytical sparsity {maximum:.8f} is below 0.50"
        )
    return UnitInventory(units, total, capacity, maximum, min_keep_ratio)


def initialize_masks(model: nn.Module, inventory: UnitInventory) -> None:
    modules = dict(model.named_modules())
    for layer in {u.module_name for u in inventory.units}:
        attach_mask(modules[layer], int(modules[layer].out_channels))
    for u in inventory.units:
        if u.downsample_module_name:
            attach_mask(modules[u.downsample_module_name], u.out_channels)


def apply_pruned_indices(
    model: nn.Module, inventory: UnitInventory, pruned_indices: set[int]
) -> dict[str, list[int]]:
    initialize_masks(model, inventory)
    grouped: dict[str, list[Unit]] = {}
    for u in inventory.units:
        grouped.setdefault(u.module_name, []).append(u)
    modules = dict(model.named_modules())
    for layer, units in grouped.items():
        mask = torch.ones(units[0].out_channels, device=next(model.parameters()).device)
        for u in units:
            if u.global_index in pruned_indices:
                mask[u.local_channel_index] = 0.0
        set_mask(modules[layer], mask)
    for u in inventory.units:
        if u.downsample_module_name:
            set_mask(
                modules[u.downsample_module_name],
                modules[u.module_name].task038_mask,
            )
    return {
        layer: modules[layer].task038_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist()
        for layer in grouped
    }


def validate_canonical_order(inventory: UnitInventory) -> None:
    expected = list(range(len(inventory.units)))
    actual = [u.global_index for u in inventory.units]
    if actual != expected:
        raise AssertionError("global indices are not contiguous canonical order")
    for a, b in zip(inventory.units, inventory.units[1:]):
        if (a.pathway, a.stage, a.block if a.block is not None else -1, a.conv_position, a.local_channel_index) > (
            b.pathway, b.stage, b.block if b.block is not None else -1, b.conv_position, b.local_channel_index
        ):
            raise AssertionError("inventory is not canonical")

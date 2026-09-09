"""Task038 dependency graph and semantic checks."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from torch import nn

from .slowfast_unit_adapter import UnitInventory, validate_canonical_order


def build_dependency_graph(model: nn.Module, inventory: UnitInventory) -> dict[str, Any]:
    modules = dict(model.named_modules())
    edges: list[dict[str, Any]] = []
    for unit in inventory.units:
        if unit.module_name not in modules:
            raise RuntimeError(f"inventory module missing: {unit.module_name}")
        if unit.bn_module_name not in modules:
            raise RuntimeError(f"inventory BN missing: {unit.bn_module_name}")
        if unit.downsample_module_name:
            if unit.downsample_module_name not in modules:
                raise RuntimeError(
                    f"inventory downsample missing: {unit.downsample_module_name}"
                )
            edges.append(
                {
                    "source": unit.layer_name,
                    "target": unit.downsample_module_name,
                    "relation": "conv3_residual_tied",
                    "channel": unit.local_channel_index,
                }
            )
    return {
        "candidate_unit_definition": "one Conv3d output channel",
        "excluded_independent_candidates": [
            "fast_conv1",
            "slow_conv1",
            "downsample Conv3d",
            "BatchNorm3d",
            "fc",
        ],
        "edges": edges,
        "unit_count": inventory.num_units,
        "global_domain_is_required": True,
    }


def validate_dependency_graph(graph: dict[str, Any], inventory: UnitInventory) -> None:
    validate_canonical_order(inventory)
    if int(graph.get("unit_count", -1)) != inventory.num_units:
        raise AssertionError("dependency graph unit count mismatch")
    for edge in graph.get("edges", []):
        if edge["relation"] != "conv3_residual_tied":
            raise AssertionError("unexpected dependency relation")
        if int(edge["channel"]) < 0:
            raise AssertionError("negative dependency channel")


def write_dependency_graph(
    path: str | Path, graph: dict[str, Any], inventory: UnitInventory
) -> None:
    validate_dependency_graph(graph, inventory)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

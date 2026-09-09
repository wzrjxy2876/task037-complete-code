"""Apply a Task038 registry as logical masks and audit invariants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .slowfast_model_task038 import (
    install_lateral_masks,
    zero_pruned_bn_and_protect,
)
from .slowfast_unit_adapter import UnitInventory, apply_pruned_indices


def logical_prune(
    model: nn.Module,
    inventory: UnitInventory,
    registry_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    pruned = {int(x) for x in payload.get("pruned_global_indices", [])}
    if any(x < 0 or x >= inventory.num_units for x in pruned):
        raise RuntimeError("registry contains an out-of-range global index")
    before_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    before_numel = sum(int(p.numel()) for p in model.parameters())
    keep = apply_pruned_indices(model, inventory, pruned)
    install_lateral_masks(model)
    zero_pruned_bn_and_protect(model, inventory)
    modules = dict(model.named_modules())
    for unit in inventory.units:
        if unit.downsample_module_name:
            conv = modules[unit.downsample_module_name]
            bn = modules.get(unit.downsample_module_name[:-1] + "1")
            mask = conv.task038_mask
            if bn is not None:
                with torch.no_grad():
                    if bn.weight is not None:
                        bn.weight.mul_(mask.to(bn.weight.device))
                    if bn.bias is not None:
                        bn.bias.mul_(mask.to(bn.bias.device))
    after_state = model.state_dict()
    after_shapes = {k: tuple(after_state[k].shape) for k in before_shapes}
    after_numel = sum(int(p.numel()) for p in model.parameters())
    if before_shapes != after_shapes:
        raise RuntimeError("logical pruning changed state_dict shapes")
    if before_numel != after_numel:
        raise RuntimeError("logical pruning changed parameter numel")
    legacy_local_cost_sum = sum(inventory.units[i].parameter_cost for i in pruned)
    report = {
        "logical_only": True,
        "pruned_global_indices": sorted(pruned),
        "removed_unit_count": len(pruned),
        "legacy_local_parameter_cost_sum": legacy_local_cost_sum,
        "local_parameter_cost_is_diagnostic_only": True,
        "parameter_count_before": before_numel,
        "parameter_count_after": after_numel,
        "state_dict_shapes_unchanged": True,
        "parameter_numel_unchanged": True,
        "layer_keep_indices": keep,
    }
    return report


def forward_backward_gate(model: nn.Module, device: torch.device) -> dict[str, Any]:
    model = model.to(device).train()
    x = torch.randn(1, 3, 8, 32, 32, device=device)
    logits = model(x)
    loss = logits.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("logical-pruned forward is non-finite")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("logical-pruned backward is non-finite")
    return {"forward_backward_finite": True, "output_shape": list(logits.shape)}

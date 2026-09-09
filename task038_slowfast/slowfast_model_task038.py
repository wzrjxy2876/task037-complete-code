"""Task038 SlowFast architecture adapter.

The archived source supplies the historical SlowFast topology only.  This
module patches the residual/output masking semantics required by Task038 and
does not import or execute the archived pruning implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
LEGACY_PATH = ROOT / "legacy_sources" / "IPslowfast.py"


def _load_legacy_architecture():
    spec = importlib.util.spec_from_file_location(
        "_task038_legacy_slowfast_architecture", LEGACY_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archived architecture: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if '_task038_bottleneck_forward' in globals():
        module.Bottleneck.forward = _task038_bottleneck_forward
    return module


def _mask(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    m = getattr(module, "task038_mask", None)
    if m is None:
        return value
    return value * m.to(device=value.device, dtype=value.dtype).view(
        1, -1, 1, 1, 1
    )


def _task038_bottleneck_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = _mask(self.conv1, out)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = _mask(self.conv2, out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    conv3_mask = getattr(self.conv3, "task038_mask", None)
    if conv3_mask is not None:
        out = out * conv3_mask.to(out.device, out.dtype).view(1, -1, 1, 1, 1)

    if self.downsample is not None:
        identity = self.downsample(x)
        identity = _mask(self.downsample[0], identity)

    # Conv3 is a residual output unit: both branches share its mask, including
    # blocks without a downsample.  The final mask is after ReLU.
    if conv3_mask is not None:
        identity = identity * conv3_mask.to(identity.device, identity.dtype).view(
            1, -1, 1, 1, 1
        )
    out = self.relu(out + identity)
    if conv3_mask is not None:
        out = out * conv3_mask.to(out.device, out.dtype).view(1, -1, 1, 1, 1)
    return out


def patch_residual_semantics() -> None:
    legacy = _load_legacy_architecture()
    legacy.Bottleneck.forward = _task038_bottleneck_forward


patch_residual_semantics()


def slowfast_16x8_resnet101_kinetics400(num_classes: int = 101) -> nn.Module:
    legacy = _load_legacy_architecture()
    legacy.Bottleneck.forward = _task038_bottleneck_forward
    model = legacy.slowfast_16x8_resnet101_kinetics400(num_classes=num_classes)
    model.task038_architecture = "slowfast_16x8_resnet101_kinetics400"
    model.task038_legacy_architecture_sha256 = hashlib.sha256(
        LEGACY_PATH.read_bytes()
    ).hexdigest()
    return model


def attach_mask(module: nn.Module, size: int | None = None) -> torch.Tensor:
    if size is None:
        if not hasattr(module, "out_channels"):
            raise TypeError("size is required for modules without out_channels")
        size = int(module.out_channels)
    if "task038_mask" not in module._buffers:
        module.register_buffer("task038_mask", torch.ones(int(size)))
    return module.task038_mask


def set_mask(module: nn.Module, mask: torch.Tensor) -> None:
    expected = int(getattr(module, "out_channels", mask.numel()))
    if mask.ndim != 1 or mask.numel() != expected:
        raise ValueError(f"mask shape {tuple(mask.shape)} != [{expected}]")
    target = attach_mask(module, expected)
    target.copy_(mask.detach().to(target.device, dtype=target.dtype))
    module.task038_mask.requires_grad_(False)


def get_mask(module: nn.Module) -> torch.Tensor:
    return attach_mask(module).detach()


def mask_state(model: nn.Module) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "task038_mask"):
            output[name] = module.task038_mask.detach().cpu().tolist()
    return output


def install_lateral_masks(model: nn.Module) -> list[Any]:
    handles = []
    for name in ("lateral_p1.0", "lateral_res2.0", "lateral_res3.0", "lateral_res4.0"):
        try:
            module = dict(model.named_modules())[name]
        except KeyError:
            continue
        attach_mask(module, int(module.out_channels))

        def hook(_module, _inputs, output):
            return _mask(_module, output)

        handles.append(module.register_forward_hook(hook))
    model.task038_lateral_mask_handles = handles
    return handles


def zero_pruned_bn_and_protect(model: nn.Module, inventory) -> list[Any]:
    handles = []
    modules = dict(model.named_modules())
    for unit in inventory.units:
        conv = modules[unit.module_name]
        mask = attach_mask(conv, unit.out_channels)
        bn_name = unit.bn_module_name
        if bn_name and bn_name in modules:
            bn = modules[bn_name]
            with torch.no_grad():
                if getattr(bn, "weight", None) is not None:
                    bn.weight.mul_(mask.to(bn.weight.device))
                if getattr(bn, "bias", None) is not None:
                    bn.bias.mul_(mask.to(bn.bias.device))

            def grad_hook(grad, m=mask):
                return grad * m.to(grad.device, grad.dtype)

            if getattr(bn, "weight", None) is not None:
                handles.append(bn.weight.register_hook(grad_hook))
            if getattr(bn, "bias", None) is not None:
                handles.append(bn.bias.register_hook(grad_hook))
    model.task038_bn_protection_handles = handles
    return handles


def load_checkpoint_identity(
    model: nn.Module, checkpoint_path: str | Path, device: torch.device,
    allow_runtime_mask_buffers: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location=device)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint has no state_dict mapping")
    normalized = {}
    for key, value in state.items():
        key = str(key)
        while key.startswith("module.") or key.startswith("backbone."):
            key = key.split(".", 1)[1]
        normalized[key] = value
    model_state = model.state_dict()
    runtime_mask_keys = {key for key in model_state if key.endswith("task038_mask")}
    comparison_state = ({key: value for key, value in model_state.items() if key not in runtime_mask_keys}
                        if allow_runtime_mask_buffers else model_state)
    missing = sorted(set(comparison_state) - set(normalized))
    unexpected = sorted(set(normalized) - set(comparison_state))
    shape_mismatch = sorted(
        key for key in set(comparison_state).intersection(normalized)
        if tuple(comparison_state[key].shape) != tuple(normalized[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "checkpoint identity mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={shape_mismatch[:5]}"
        )
    if allow_runtime_mask_buffers and runtime_mask_keys:
        model.load_state_dict(normalized, strict=False)
    else:
        model.load_state_dict(normalized, strict=True)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "normalized_key_count": len(normalized),
        "runtime_mask_buffers_ignored": sorted(runtime_mask_keys) if allow_runtime_mask_buffers else [],
    }


__all__ = [
    "slowfast_16x8_resnet101_kinetics400",
    "attach_mask",
    "set_mask",
    "get_mask",
    "mask_state",
    "install_lateral_masks",
    "zero_pruned_bn_and_protect",
    "load_checkpoint_identity",
]

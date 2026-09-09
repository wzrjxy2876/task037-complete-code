"""Task038 Dynamic3D descriptor extraction for SlowFast."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .slowfast_unit_adapter import UnitInventory


def _dynamicity(signed: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Exact Task037 temporal dynamicity with a singleton feature dimension.
    if signed.ndim != 6:
        raise ValueError("signed response must be [B,U,T,H,W,D]")
    x = torch.nan_to_num(signed.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean_t = x.mean(2, keepdim=True)
    residual = x - mean_t
    dynamic = residual.norm(p=2, dim=-1).mean((2, 3, 4))
    stable = mean_t.norm(p=2, dim=-1).mean((2, 3, 4))
    return torch.nan_to_num(
        dynamic / (dynamic + stable + eps), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)


def _robust01(value: torch.Tensor, low: float = 0.01, high: float = 0.99) -> torch.Tensor:
    lo = torch.quantile(value.float(), low)
    hi = torch.quantile(value.float(), high)
    scale = hi - lo
    if not bool(torch.isfinite(scale).item()) or float(scale.abs().item()) <= 1e-8:
        return torch.zeros_like(value, dtype=torch.float32)
    return ((value.float() - lo) / (scale + 1e-8)).clamp(0.0, 1.0)


def _schur_relative(abs_score: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Task037 Schur-complement substitutability, then D_rel = D_abs*(1-sub).
    if z.ndim != 2 or z.shape[1] != abs_score.numel():
        raise ValueError("activation matrix and descriptor width disagree")
    centered = z.float() - z.float().mean(0, keepdim=True)
    covariance = centered.T @ centered / z.shape[0]
    covariance = covariance + 1e-5 * torch.eye(
        z.shape[1], device=z.device, dtype=torch.float32
    )
    try:
        inverse = torch.linalg.inv(covariance)
        error_variance = 1.0 / torch.diag(inverse)
        original_variance = torch.diag(covariance)
        substitutable = (
            1.0 - error_variance / (original_variance + 1e-5)
        ).clamp(0.0, 1.0)
    except RuntimeError:
        substitutable = torch.zeros_like(abs_score)
    return abs_score * (1.0 - substitutable)


@dataclass
class _Stats:
    count: int
    sum_abs: torch.Tensor
    cross: torch.Tensor
    dyn_sum: torch.Tensor
    dyn_count: int
    above: torch.Tensor | None = None


def _candidate_capture_names(inventory: UnitInventory) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for unit in inventory.units:
        if unit.conv_position == "conv3":
            hook_name = unit.module_name.rsplit(".", 1)[0]
        else:
            hook_name = unit.module_name
        grouped.setdefault(hook_name, []).append(unit.module_name)
    return grouped


def _new_stats(channels: int, device: torch.device) -> _Stats:
    return _Stats(
        0,
        torch.zeros(channels, device=device, dtype=torch.float32),
        torch.zeros((channels, channels), device=device, dtype=torch.float32),
        torch.zeros(channels, device=device, dtype=torch.float32),
        0,
    )


def _run_pass(
    model: nn.Module,
    loader: Iterable,
    capture_names: dict[str, list[str]],
    stats: dict[str, _Stats],
    device: torch.device,
    batches: int,
    second_pass: bool = False,
) -> None:
    modules = dict(model.named_modules())
    handles = []
    active: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 5:
                raise ValueError(f"{name} activation is not 5D: {tuple(output.shape)}")
            signed = output.float()
            flat = signed.abs().permute(0, 2, 3, 4, 1).reshape(-1, signed.shape[1])
            s = stats[name]
            if second_pass:
                if s.above is None:
                    s.above = torch.zeros(signed.shape[1], device=device)
                active[name] = active.get(name, torch.zeros_like(s.above)) + (
                    (flat > (s.sum_abs / max(s.count, 1))).sum(0).float()
                )
            else:
                s.count += int(flat.shape[0])
                s.sum_abs.add_(flat.sum(0))
                s.cross.add_(flat.T @ flat)
                dyn = _dynamicity(signed.unsqueeze(-1))
                s.dyn_sum.add_(dyn.sum(0))
                s.dyn_count += int(dyn.shape[0])
        return hook

    for hook_name in capture_names:
        handles.append(modules[hook_name].register_forward_hook(make_hook(hook_name)))
    model.eval()
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= batches:
                    break
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device, non_blocking=True).float())
    finally:
        for handle in handles:
            handle.remove()
    if second_pass:
        for name, values in active.items():
            if stats[name].above is None:
                stats[name].above = torch.zeros_like(values)
            stats[name].above.add_(values)


def calibrate_descriptors(
    model: nn.Module,
    loader: Iterable,
    inventory: UnitInventory,
    device: torch.device,
    output_dir: str | Path,
    batches: int = 10,
) -> dict[str, Any]:
    if batches != 10:
        raise ValueError("Task038 calibration is fixed at 10 batches")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules = dict(model.named_modules())
    capture_names = _candidate_capture_names(inventory)
    widths = {name: max(u.local_channel_index for u in inventory.units if (u.module_name.rsplit(".", 1)[0] if u.conv_position == "conv3" else u.module_name) == name) + 1 for name in capture_names}
    stats = {name: _new_stats(widths[name], device) for name in capture_names}
    _run_pass(model, loader, capture_names, stats, device, batches, False)
    _run_pass(model, loader, capture_names, stats, device, batches, True)

    by_layer: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for layer_name in sorted({u.module_name for u in inventory.units}):
        hook_name = layer_name.rsplit(".", 1)[0] if layer_name.endswith(".conv3") else layer_name
        s = stats[hook_name]
        mean_amp = s.sum_abs / max(s.count, 1)
        freq = (s.above if s.above is not None else torch.zeros_like(mean_amp)) / max(s.count, 1)
        amp_n = _robust01(mean_amp)
        freq_n = _robust01(freq)
        d_abs = 0.5 * amp_n + 0.5 * freq_n
        # Reconstructing rows is impossible from sufficient statistics; use the
        # exact covariance implied by the collected cross moments instead.
        covariance = s.cross / max(s.count, 1) - torch.outer(mean_amp, mean_amp)
        covariance = covariance + 1e-5 * torch.eye(
            covariance.shape[0], device=device, dtype=torch.float32
        )
        try:
            inverse = torch.linalg.inv(covariance)
            err = 1.0 / torch.diag(inverse)
            orig = torch.diag(covariance)
            sub = (1.0 - err / (orig + 1e-5)).clamp(0.0, 1.0)
        except RuntimeError:
            sub = torch.zeros_like(d_abs)
        d_rel = d_abs * (1.0 - sub)
        d_dyn = s.dyn_sum / max(s.dyn_count, 1)
        descriptor = torch.stack((d_abs, d_rel, d_dyn), dim=1)
        by_layer[layer_name] = descriptor
        indices = [u for u in inventory.units if u.module_name == layer_name]
        if len(indices) != descriptor.shape[0]:
            raise RuntimeError(f"descriptor mapping mismatch for {layer_name}")
        for unit, values in zip(indices, descriptor):
            rows.append(
                {
                    "global_index": unit.global_index,
                    "layer_name": unit.layer_name,
                    "local_channel_index": unit.local_channel_index,
                    "pathway": unit.pathway,
                    "stage": unit.stage,
                    "conv_position": unit.conv_position,
                    "D_abs": float(values[0].item()),
                    "D_rel": float(values[1].item()),
                    "D_dyn": float(values[2].item()),
                    "calibration_batches": batches,
                    "calibration_device": str(device),
                }
            )
    descriptor = torch.stack(
        [by_layer[u.module_name][u.local_channel_index] for u in inventory.units]
    ).float()
    if not torch.isfinite(descriptor).all():
        raise RuntimeError("descriptor contains non-finite values")
    if bool((descriptor[:, 2] < -1e-6).any()) or bool((descriptor[:, 2] > 1.0 + 1e-6).any()):
        raise RuntimeError("D_dyn outside [0,1]")
    torch.save(descriptor.detach().cpu(), out / "descriptor_vectors.pt")
    with (out / "descriptor_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["global_index"])))
    metadata = {
        "descriptor_variant": "dynamic3d",
        "dimensions": ["D_abs", "D_rel", "D_dyn"],
        "D_abs_normalization": {"method": "robust_quantile", "q_low": 0.01, "q_high": 0.99, "alpha": 0.5},
        "D_rel": "Schur complement substitutability from Task037",
        "D_dyn": "Task037 signed temporal dynamicity",
        "unit_count": inventory.num_units,
        "calibration_batches": batches,
        "device": str(device),
        "all_finite": True,
    }
    (out / "descriptor_identity.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata

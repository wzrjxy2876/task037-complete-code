#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent probe for Class-conditioned Spatio-temporal Contribution Tube Coherence (CSTC).

Purpose
-------
This probe validates a simple third descriptor dimension for video structured pruning:

    activation/gradient contribution field
        -> normalized t-x-y mass distribution
        -> weighted 3x3 covariance
        -> tube anisotropy x temporal extent
        -> CSTC score in [0, 1]

The script does NOT prune or fine-tune the model. It performs four diagnostics:

1. Synthetic sanity tests:
   moving tube, fixed tube, random jumps, single-frame impulse, and zero field.
2. Real contribution-field extraction for selected Video Swin attention heads / FFN neurons.
3. Local masking validation:
   compare activation*gradient values at sampled t-x-y positions with actual target-logit
   drops after masking the same unit position.
4. Whole-unit masking validation:
   compare CSTC, D_abs and D_rel with real target-logit drops after masking a complete unit.

The implementation intentionally reuses the already tested model/dataset/layer-discovery
helpers from probe_ctfrs_dynamic_function.py. Put both scripts in the same project root.

Python 3.9+, PyTorch 1.12 style.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from probe_ctfrs_dynamic_function import (
        EPS,
        UnitLayerSpec,
        build_balanced_loader,
        chunked,
        compute_d_abs_d_rel,
        discover_unit_layers,
        ensure_project_importable,
        filter_layers,
        load_model,
        set_seed,
        spearman,
        unwrap_logits,
        window_reverse_3d_local,
    )
except Exception as exc:
    raise ImportError(
        "probe_cstc_contribution_tube.py must be placed beside the previously fixed "
        "probe_ctfrs_dynamic_function.py. Importing the helper probe failed."
    ) from exc


@dataclass
class LayerCSTCResult:
    spec: UnitLayerSpec
    video_ids: List[int]
    labels: List[int]
    target_logits: List[float]
    contribution_volumes: np.ndarray  # [N,U,T,H,W]
    cstc_per_video: np.ndarray        # [N,U]
    anisotropy_per_video: np.ndarray  # [N,U]
    temporal_extent_per_video: np.ndarray  # [N,U]
    cstc: np.ndarray                  # [U]
    anisotropy: np.ndarray            # [U]
    temporal_extent: np.ndarray       # [U]
    valid_video_rate: np.ndarray      # [U]
    d_abs: np.ndarray
    d_rel: np.ndarray
    substitutability: np.ndarray
    mean_abs_activation: np.ndarray
    active_frequency: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe class-conditioned spatio-temporal contribution tube coherence"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./cstc_probe_output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--target_mode", choices=("true", "pred"), default="true")
    parser.add_argument(
        "--layers",
        default=r"layers\.3\.blocks\.(0|1)\.(attn|mlp)",
        help="'representative', 'all', or regex over pruning layer names",
    )
    parser.add_argument("--layer_batch_size", type=int, default=1)
    parser.add_argument("--max_static_samples", type=int, default=2048)

    # Probe-budget controls. These do not enter the CSTC formula.
    parser.add_argument("--whole_ablation_layers", type=int, default=2)
    parser.add_argument("--whole_ablation_units", type=int, default=2,
                        help="Top and bottom unit count per selected layer")
    parser.add_argument("--ablation_videos", type=int, default=4)
    parser.add_argument("--local_ablation_layers", type=int, default=1)
    parser.add_argument("--local_ablation_units", type=int, default=2)
    parser.add_argument("--local_ablation_points", type=int, default=12,
                        help="Sampled t-x-y points per unit and video")
    parser.add_argument("--skip_local_ablation", action="store_true")
    parser.add_argument("--skip_whole_ablation", action="store_true")
    parser.add_argument("--save_top_volumes", type=int, default=4,
                        help="Top/bottom units per selected layer exported as heatmap GIF-like PNG strips")
    return parser.parse_args()


def _attention_response_volume(x: torch.Tensor, spec: UnitLayerSpec) -> torch.Tensor:
    geometry = getattr(spec.module, "_pruning_geometry", None)
    if geometry is None:
        raise RuntimeError(f"Missing _pruning_geometry for {spec.name}")
    head_dim = int(spec.module.head_dim)
    windows = x.reshape(x.shape[0], x.shape[1], spec.num_units, head_dim).abs().mean(dim=-1)
    return window_reverse_3d_local(windows, geometry)


def _attention_contribution_volume(
    x: torch.Tensor, grad: torch.Tensor, spec: UnitLayerSpec
) -> torch.Tensor:
    geometry = getattr(spec.module, "_pruning_geometry", None)
    if geometry is None:
        raise RuntimeError(f"Missing _pruning_geometry for {spec.name}")
    head_dim = int(spec.module.head_dim)
    local = (x * grad).reshape(
        x.shape[0], x.shape[1], spec.num_units, head_dim
    ).sum(dim=-1)
    return window_reverse_3d_local(local, geometry)


def _mlp_response_volume(x: torch.Tensor) -> torch.Tensor:
    # Project MLP fc2 input is [B,T,H,W,U].
    return x.permute(0, 4, 1, 2, 3).contiguous()


def _mlp_contribution_volume(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return (x * grad).permute(0, 4, 1, 2, 3).contiguous()


class SpatialContributionCollector:
    """Capture [B,U,T,H,W] activation-gradient contribution fields."""

    def __init__(self, specs: Sequence[UnitLayerSpec], max_static_samples: int):
        self.specs = {spec.name: spec for spec in specs}
        self.max_static_samples = int(max_static_samples)
        self.handles: List[Any] = []
        self.current: Dict[str, torch.Tensor] = {}
        self.static_samples: Dict[str, torch.Tensor] = {}

    def register(self) -> None:
        for spec in self.specs.values():
            self.handles.append(
                spec.hook_module.register_forward_pre_hook(self._make_hook(spec))
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_batch(self) -> None:
        self.current = {}

    def _append_static(self, name: str, response: torch.Tensor) -> None:
        # [B,U,T,H,W] -> [positions,U]
        samples = response.detach().abs().permute(0, 2, 3, 4, 1).reshape(
            -1, response.shape[1]
        )
        if samples.shape[0] > self.max_static_samples:
            idx = torch.randperm(samples.shape[0], device=samples.device)[
                : self.max_static_samples
            ]
            samples = samples.index_select(0, idx)
        samples = samples.cpu().float()
        if name in self.static_samples:
            samples = torch.cat([self.static_samples[name], samples], dim=0)
            if samples.shape[0] > self.max_static_samples:
                idx = torch.randperm(samples.shape[0])[: self.max_static_samples]
                samples = samples.index_select(0, idx)
        self.static_samples[name] = samples

    def _make_hook(
        self, spec: UnitLayerSpec
    ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], None]:
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            if not x.requires_grad:
                x.requires_grad_(True)

            if spec.unit_type == "head":
                response = _attention_response_volume(x.detach(), spec)
            else:
                response = _mlp_response_volume(x.detach())
            self._append_static(spec.name, response)
            activation = x

            def gradient_hook(grad: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    if spec.unit_type == "head":
                        field = _attention_contribution_volume(
                            activation.detach(), grad.detach(), spec
                        )
                    else:
                        field = _mlp_contribution_volume(
                            activation.detach(), grad.detach()
                        )
                    self.current[spec.name] = field.cpu().float()
                return grad

            x.register_hook(gradient_hook)

        return hook


def cstc_from_volume(
    volume: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CSTC for volume [...,T,H,W].

    Returns:
        score, normalized_anisotropy, normalized_temporal_extent, valid_mask
    """
    original_shape = volume.shape[:-3]
    t_size, h_size, w_size = volume.shape[-3:]
    positive = torch.relu(volume.float())
    flat = positive.reshape(-1, t_size * h_size * w_size)
    mass = flat.sum(dim=1, keepdim=True)
    valid = mass[:, 0] > EPS
    prob = torch.zeros_like(flat)
    prob[valid] = flat[valid] / mass[valid].clamp_min(EPS)

    def coord(size: int, device: torch.device) -> torch.Tensor:
        if size <= 1:
            return torch.zeros(1, device=device)
        return torch.linspace(0.0, 1.0, size, device=device)

    tt, xx, yy = torch.meshgrid(
        coord(t_size, volume.device),
        coord(h_size, volume.device),
        coord(w_size, volume.device),
        indexing="ij",
    )
    coords = torch.stack([tt, xx, yy], dim=-1).reshape(-1, 3)  # [P,3]
    mean = prob @ coords  # [M,3]
    centered = coords.unsqueeze(0) - mean.unsqueeze(1)  # [M,P,3]
    cov = torch.einsum("mp,mpa,mpb->mab", prob, centered, centered)
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    trace = eigenvalues.sum(dim=-1)
    largest = eigenvalues[:, -1]
    principal_ratio = largest / trace.clamp_min(EPS)
    anisotropy = ((3.0 * principal_ratio - 1.0) / 2.0).clamp(0.0, 1.0)

    temporal_variance = cov[:, 0, 0].clamp_min(0.0)
    temporal_extent = (4.0 * temporal_variance).clamp(0.0, 1.0)
    score = torch.sqrt(anisotropy * temporal_extent)
    score[~valid] = 0.0
    anisotropy[~valid] = 0.0
    temporal_extent[~valid] = 0.0

    return (
        score.reshape(original_shape),
        anisotropy.reshape(original_shape),
        temporal_extent.reshape(original_shape),
        valid.reshape(original_shape),
    )


def synthetic_sanity(output_dir: Path) -> pd.DataFrame:
    """Create five deterministic synthetic fields and evaluate expected ordering."""
    t_size, h_size, w_size = 16, 16, 16

    def gaussian(cx: float, cy: float, sigma: float = 1.5) -> np.ndarray:
        xx, yy = np.meshgrid(np.arange(h_size), np.arange(w_size), indexing="ij")
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    fields: Dict[str, np.ndarray] = {}

    moving = np.zeros((t_size, h_size, w_size), dtype=np.float32)
    fixed = np.zeros_like(moving)
    random_jump = np.zeros_like(moving)
    impulse = np.zeros_like(moving)
    zero = np.zeros_like(moving)
    rng = np.random.RandomState(3407)

    for t in range(t_size):
        moving[t] = gaussian(2 + 11 * t / (t_size - 1), 3 + 9 * t / (t_size - 1))
        fixed[t] = gaussian(8, 8)
        random_jump[t] = gaussian(rng.uniform(1, 14), rng.uniform(1, 14))
    impulse[t_size // 2] = gaussian(8, 8)

    fields["moving_tube"] = moving
    fields["fixed_tube"] = fixed
    fields["random_jump"] = random_jump
    fields["single_frame_impulse"] = impulse
    fields["zero_field"] = zero

    rows: List[Dict[str, Any]] = []
    for name, array in fields.items():
        tensor = torch.from_numpy(array).unsqueeze(0)
        score, anis, extent, valid = cstc_from_volume(tensor)
        rows.append(
            {
                "case": name,
                "cstc": float(score.item()),
                "anisotropy": float(anis.item()),
                "temporal_extent": float(extent.item()),
                "valid": bool(valid.item()),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "synthetic_sanity.csv", index=False)

    plt.figure(figsize=(7.2, 4.3))
    plt.bar(frame["case"], frame["cstc"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("CSTC")
    plt.title("Synthetic contribution-field sanity test")
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_sanity.png", dpi=220)
    plt.close()
    return frame


def run_layer_group(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    specs: Sequence[UnitLayerSpec],
    device: torch.device,
    target_mode: str,
    max_static_samples: int,
) -> List[LayerCSTCResult]:
    collector = SpatialContributionCollector(specs, max_static_samples)
    collector.register()
    volumes: Dict[str, List[torch.Tensor]] = defaultdict(list)
    video_ids: List[int] = []
    labels: List[int] = []
    target_logits: List[float] = []

    try:
        for videos, targets, indices in tqdm(
            loader, desc="CSTC contribution forward/backward", ncols=105
        ):
            collector.reset_batch()
            videos = videos.float().to(device, non_blocking=True)
            targets = targets.long().to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = unwrap_logits(model(videos))
                predicted = logits.argmax(dim=1)
                chosen = targets if target_mode == "true" else predicted
                if int(chosen.max()) >= logits.shape[1]:
                    raise ValueError(
                        f"Target {int(chosen.max())} exceeds classifier width {logits.shape[1]}"
                    )
                objective = logits.gather(1, chosen[:, None]).sum()
                objective.backward()

            missing = [spec.name for spec in specs if spec.name not in collector.current]
            if missing:
                raise RuntimeError(f"No contribution volume captured for: {missing}")

            labels.extend(targets.cpu().tolist())
            target_logits.extend(
                logits.detach().gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
            )
            if torch.is_tensor(indices):
                video_ids.extend(indices.cpu().tolist())
            else:
                video_ids.extend(int(v) for v in indices)

            for spec in specs:
                field = collector.current[spec.name]
                if field.shape[0] != 1:
                    raise RuntimeError("The probe assumes batch_size=1")
                volumes[spec.name].append(field.squeeze(0))

            del videos, targets, logits, objective
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        collector.remove()

    results: List[LayerCSTCResult] = []
    for spec in specs:
        field = torch.stack(volumes[spec.name], dim=0)  # [N,U,T,H,W]
        score, anis, extent, valid = cstc_from_volume(field)
        d_abs, d_rel, sub, mean_amp, frequency = compute_d_abs_d_rel(
            collector.static_samples[spec.name]
        )
        results.append(
            LayerCSTCResult(
                spec=spec,
                video_ids=list(video_ids),
                labels=list(labels),
                target_logits=list(target_logits),
                contribution_volumes=field.numpy(),
                cstc_per_video=score.numpy(),
                anisotropy_per_video=anis.numpy(),
                temporal_extent_per_video=extent.numpy(),
                cstc=score.mean(dim=0).numpy(),
                anisotropy=anis.mean(dim=0).numpy(),
                temporal_extent=extent.mean(dim=0).numpy(),
                valid_video_rate=valid.float().mean(dim=0).numpy(),
                d_abs=d_abs,
                d_rel=d_rel,
                substitutability=sub,
                mean_abs_activation=mean_amp,
                active_frequency=frequency,
            )
        )
    return results


def _global_mask_to_attention_windows(
    global_mask: torch.Tensor, geometry: Mapping[str, Any]
) -> torch.Tensor:
    """Convert [B,T,H,W] global mask to [B*nW,N] matching attention window input."""
    wd, wh, ww = [int(v) for v in geometry["window_size"]]
    batch_size = int(geometry["batch_size"])
    depth = int(geometry["depth"])
    height = int(geometry["height"])
    width = int(geometry["width"])
    dp = int(geometry["padded_depth"])
    hp = int(geometry["padded_height"])
    wp = int(geometry["padded_width"])
    shift = tuple(int(v) for v in geometry["shift_size"])

    if global_mask.shape != (batch_size, depth, height, width):
        raise ValueError(
            f"Global mask shape {tuple(global_mask.shape)} != "
            f"{(batch_size, depth, height, width)}"
        )
    padded = torch.zeros(
        (batch_size, dp, hp, wp),
        device=global_mask.device,
        dtype=global_mask.dtype,
    )
    padded[:, :depth, :height, :width] = global_mask
    if any(v > 0 for v in shift):
        padded = torch.roll(
            padded, shifts=tuple(-v for v in shift), dims=(1, 2, 3)
        )
    windows = padded.view(
        batch_size,
        dp // wd,
        wd,
        hp // wh,
        wh,
        wp // ww,
        ww,
    )
    windows = windows.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return windows.view(-1, wd * wh * ww)


@contextlib.contextmanager
def mask_unit_region(
    spec: UnitLayerSpec,
    unit_index: int,
    region_mask: Optional[torch.Tensor] = None,
):
    """Mask a full unit or selected native t-x-y positions.

    region_mask is [T,H,W] with True at positions to zero. When None, zero the
    complete unit.
    """
    def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        x = inputs[0]
        masked = x.clone()
        if spec.unit_type == "neuron":
            if region_mask is None:
                masked[..., unit_index] = 0.0
            else:
                native = region_mask.to(device=masked.device, dtype=torch.bool)
                if tuple(native.shape) != tuple(masked.shape[1:4]):
                    raise ValueError(
                        f"MLP region {tuple(native.shape)} != native shape {tuple(masked.shape[1:4])}"
                    )
                selected = masked[..., unit_index]
                selected[:, native] = 0.0
                masked[..., unit_index] = selected
        else:
            head_dim = int(spec.module.head_dim)
            reshaped = masked.reshape(
                masked.shape[0], masked.shape[1], spec.num_units, head_dim
            )
            if region_mask is None:
                reshaped[:, :, unit_index, :] = 0.0
            else:
                geometry = getattr(spec.module, "_pruning_geometry", None)
                if geometry is None:
                    raise RuntimeError(f"Missing geometry for {spec.name}")
                global_mask = region_mask.to(masked.device, dtype=torch.bool).unsqueeze(0)
                window_mask = _global_mask_to_attention_windows(
                    global_mask, geometry
                )
                selected = reshaped[:, :, unit_index, :]
                selected = selected.masked_fill(window_mask.unsqueeze(-1), 0.0)
                reshaped[:, :, unit_index, :] = selected
            masked = reshaped.reshape_as(masked)
        return (masked,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_cache(
    loader: torch.utils.data.DataLoader, limit: int
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    cache = []
    for videos, targets, indices in loader:
        index = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        cache.append((videos.cpu(), targets.cpu(), index))
        if len(cache) >= limit:
            break
    return cache


def target_logit(
    model: nn.Module,
    videos: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = unwrap_logits(model(videos))
    predicted = logits.argmax(dim=1)
    chosen = targets if target_mode == "true" else predicted
    return logits.gather(1, chosen[:, None]).squeeze(1), chosen


def evaluate_whole_unit_drop(
    model: nn.Module,
    spec: UnitLayerSpec,
    unit_index: int,
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
) -> Tuple[float, float]:
    drops = []
    relative = []
    for videos_cpu, targets_cpu, _ in cache:
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)
        with torch.no_grad():
            baseline, _ = target_logit(model, videos, targets, target_mode)
            with mask_unit_region(spec, unit_index, None):
                masked, _ = target_logit(model, videos, targets, target_mode)
        drop = float((baseline - masked).item())
        drops.append(drop)
        relative.append(drop / (abs(float(baseline.item())) + EPS))
    return float(np.mean(drops)), float(np.mean(relative))


def select_spatial_points(
    positive_field: np.ndarray,
    count: int,
    seed: int,
) -> List[Tuple[str, int]]:
    """Select high, low-positive and random flattened positions."""
    flat = np.asarray(positive_field, dtype=np.float64).reshape(-1)
    count = min(int(count), len(flat))
    per_group = max(1, count // 3)
    order = np.argsort(flat)
    positive_idx = order[flat[order] > 0]
    low = positive_idx[:per_group].tolist() if len(positive_idx) else order[:per_group].tolist()
    high = order[-per_group:][::-1].tolist()
    rng = np.random.RandomState(seed)
    remaining = max(0, count - len(low) - len(high))
    random_idx = rng.choice(len(flat), size=remaining, replace=False).tolist()
    selected = [("low", int(i)) for i in low]
    selected += [("high", int(i)) for i in high]
    selected += [("random", int(i)) for i in random_idx]
    seen = set()
    unique = []
    for group, idx in selected:
        if idx not in seen:
            seen.add(idx)
            unique.append((group, idx))
    return unique


def run_local_ablation(
    model: nn.Module,
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    results: Sequence[LayerCSTCResult],
    device: torch.device,
    target_mode: str,
    max_layers: int,
    unit_count: int,
    point_count: int,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    selected_results = list(results[:max_layers])
    rows: List[Dict[str, Any]] = []

    for result in tqdm(selected_results, desc="Local t-x-y masking", ncols=105):
        # Use high and low CSTC units.
        order = np.argsort(result.cstc)
        chosen_units = list(order[:unit_count]) + list(order[-unit_count:])
        chosen_units = list(dict.fromkeys(int(v) for v in chosen_units))

        for unit_index in chosen_units:
            for cache_pos, (videos_cpu, targets_cpu, video_id) in enumerate(cache):
                # Match this cached clip to extracted contribution volume by dataset id.
                if video_id not in result.video_ids:
                    continue
                sample_index = result.video_ids.index(video_id)
                field = np.maximum(
                    result.contribution_volumes[sample_index, unit_index], 0.0
                )
                t_size, h_size, w_size = field.shape
                points = select_spatial_points(
                    field, point_count, seed + 97 * unit_index + cache_pos
                )

                videos = videos_cpu.float().to(device)
                targets = targets_cpu.long().to(device)
                with torch.no_grad():
                    baseline, _ = target_logit(model, videos, targets, target_mode)

                for point_group, flat_index in points:
                    t, h, w = np.unravel_index(flat_index, field.shape)
                    region = torch.zeros(
                        (t_size, h_size, w_size), dtype=torch.bool
                    )
                    region[t, h, w] = True
                    with torch.no_grad(), mask_unit_region(
                        result.spec, unit_index, region
                    ):
                        masked, _ = target_logit(
                            model, videos, targets, target_mode
                        )
                    drop = float((baseline - masked).item())
                    rows.append(
                        {
                            "layer": result.spec.name,
                            "unit_type": result.spec.unit_type,
                            "unit_index": unit_index,
                            "video_id": video_id,
                            "point_group": point_group,
                            "t": int(t),
                            "x": int(h),
                            "y": int(w),
                            "estimated_signed_contribution": float(
                                result.contribution_volumes[
                                    sample_index, unit_index, t, h, w
                                ]
                            ),
                            "estimated_positive_contribution": float(
                                field[t, h, w]
                            ),
                            "true_target_logit_drop": drop,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "local_masking_validation.csv", index=False)
    return frame


def run_whole_ablation(
    model: nn.Module,
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    results: Sequence[LayerCSTCResult],
    device: torch.device,
    target_mode: str,
    max_layers: int,
    unit_count: int,
    output_dir: Path,
) -> pd.DataFrame:
    selected_results = list(results)
    if len(selected_results) > max_layers:
        idx = np.linspace(0, len(selected_results) - 1, max_layers).round().astype(int)
        selected_results = [selected_results[i] for i in sorted(set(idx.tolist()))]

    rows = []
    for result in tqdm(selected_results, desc="Whole-unit masking", ncols=105):
        order = np.argsort(result.cstc)
        selected = [("low", int(i)) for i in order[:unit_count]]
        selected += [("high", int(i)) for i in order[-unit_count:][::-1]]
        seen = set()
        for rank_group, unit_index in selected:
            if unit_index in seen:
                continue
            seen.add(unit_index)
            mean_drop, relative_drop = evaluate_whole_unit_drop(
                model, result.spec, unit_index, cache, device, target_mode
            )
            rows.append(
                {
                    "layer": result.spec.name,
                    "unit_type": result.spec.unit_type,
                    "unit_index": unit_index,
                    "rank_group": rank_group,
                    "cstc": float(result.cstc[unit_index]),
                    "anisotropy": float(result.anisotropy[unit_index]),
                    "temporal_extent": float(result.temporal_extent[unit_index]),
                    "d_abs": float(result.d_abs[unit_index]),
                    "d_rel": float(result.d_rel[unit_index]),
                    "true_mean_logit_drop": mean_drop,
                    "true_mean_relative_drop": relative_drop,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "whole_unit_masking_validation.csv", index=False)
    return frame


def export_unit_metrics(
    results: Sequence[LayerCSTCResult], output_dir: Path
) -> pd.DataFrame:
    rows = []
    for result in results:
        for unit_index in range(result.spec.num_units):
            rows.append(
                {
                    "layer": result.spec.name,
                    "stage": result.spec.stage,
                    "block": result.spec.block,
                    "unit_type": result.spec.unit_type,
                    "unit_index": unit_index,
                    "cstc": float(result.cstc[unit_index]),
                    "anisotropy": float(result.anisotropy[unit_index]),
                    "temporal_extent": float(result.temporal_extent[unit_index]),
                    "valid_video_rate": float(result.valid_video_rate[unit_index]),
                    "d_abs": float(result.d_abs[unit_index]),
                    "d_rel": float(result.d_rel[unit_index]),
                    "substitutability": float(result.substitutability[unit_index]),
                    "mean_abs_activation": float(result.mean_abs_activation[unit_index]),
                    "active_frequency": float(result.active_frequency[unit_index]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "unit_cstc_metrics.csv", index=False)
    return frame


def export_arrays(results: Sequence[LayerCSTCResult], output_dir: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata = {}
    for index, result in enumerate(results):
        prefix = f"layer_{index:03d}"
        arrays[f"{prefix}_contribution_volumes"] = result.contribution_volumes
        arrays[f"{prefix}_cstc_per_video"] = result.cstc_per_video
        arrays[f"{prefix}_anisotropy_per_video"] = result.anisotropy_per_video
        arrays[f"{prefix}_temporal_extent_per_video"] = result.temporal_extent_per_video
        metadata[prefix] = {
            "layer": result.spec.name,
            "unit_type": result.spec.unit_type,
            "video_ids": result.video_ids,
            "labels": result.labels,
            "shape": list(result.contribution_volumes.shape),
        }
    np.savez_compressed(output_dir / "cstc_probe_arrays.npz", **arrays)
    with open(output_dir / "cstc_probe_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(6.0, 4.8))
    plt.scatter(x, y, s=8, alpha=0.45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_volume_strips(
    results: Sequence[LayerCSTCResult],
    output_dir: Path,
    count: int,
) -> None:
    vis_dir = output_dir / "contribution_volume_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for layer_index, result in enumerate(results):
        mean_field = np.maximum(result.contribution_volumes, 0.0).mean(axis=0)
        order = np.argsort(result.cstc)
        selected = [("low", int(i)) for i in order[:count]]
        selected += [("high", int(i)) for i in order[-count:][::-1]]
        for rank_group, unit_index in selected:
            volume = mean_field[unit_index]
            vmax = max(float(volume.max()), 1e-12)
            columns = min(volume.shape[0], 8)
            rows = int(math.ceil(volume.shape[0] / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.0 * rows))
            axes = np.asarray(axes).reshape(-1)
            for t in range(volume.shape[0]):
                axes[t].imshow(volume[t], vmin=0.0, vmax=vmax)
                axes[t].set_title(f"t={t}")
                axes[t].axis("off")
            for axis in axes[volume.shape[0]:]:
                axis.axis("off")
            fig.suptitle(
                f"{result.spec.name} | unit {unit_index} | {rank_group} CSTC="
                f"{result.cstc[unit_index]:.4f}"
            )
            fig.tight_layout()
            safe_name = result.spec.name.replace(".", "_")
            fig.savefig(
                vis_dir / f"L{layer_index:02d}_{safe_name}_u{unit_index}_{rank_group}.png",
                dpi=180,
            )
            plt.close(fig)


def build_summary(
    unit_frame: pd.DataFrame,
    synthetic: pd.DataFrame,
    local_frame: pd.DataFrame,
    whole_frame: pd.DataFrame,
    args: argparse.Namespace,
    model_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "method": "CSTC = sqrt(normalized covariance anisotropy * normalized temporal extent)",
        "model_metadata": dict(model_metadata),
        "num_units": int(len(unit_frame)),
        "num_layers": int(unit_frame["layer"].nunique()) if len(unit_frame) else 0,
        "synthetic": synthetic.to_dict(orient="records"),
        "unit_statistics": {},
        "local_masking": {},
        "whole_unit_masking": {},
        "run_config": vars(args),
    }
    if len(unit_frame):
        summary["unit_statistics"] = {
            "cstc_mean": float(unit_frame["cstc"].mean()),
            "cstc_std": float(unit_frame["cstc"].std()),
            "anisotropy_mean": float(unit_frame["anisotropy"].mean()),
            "temporal_extent_mean": float(unit_frame["temporal_extent"].mean()),
            "rho_cstc_d_abs": spearman(unit_frame["cstc"], unit_frame["d_abs"]),
            "rho_cstc_d_rel": spearman(unit_frame["cstc"], unit_frame["d_rel"]),
        }
    if len(local_frame):
        summary["local_masking"] = {
            "num_tests": int(len(local_frame)),
            "rho_signed_estimate_true_drop": spearman(
                local_frame["estimated_signed_contribution"],
                local_frame["true_target_logit_drop"],
            ),
            "rho_positive_estimate_true_drop": spearman(
                local_frame["estimated_positive_contribution"],
                local_frame["true_target_logit_drop"],
            ),
            "sign_agreement_rate": float(
                np.mean(
                    np.sign(local_frame["estimated_signed_contribution"])
                    == np.sign(local_frame["true_target_logit_drop"])
                )
            ),
            "mean_true_drop_high_points": float(
                local_frame.loc[
                    local_frame["point_group"] == "high", "true_target_logit_drop"
                ].mean()
            ),
            "mean_true_drop_low_points": float(
                local_frame.loc[
                    local_frame["point_group"] == "low", "true_target_logit_drop"
                ].mean()
            ),
        }
    if len(whole_frame):
        summary["whole_unit_masking"] = {
            "num_tests": int(len(whole_frame)),
            "rho_cstc_true_drop": spearman(
                whole_frame["cstc"], whole_frame["true_mean_logit_drop"]
            ),
            "rho_d_abs_true_drop": spearman(
                whole_frame["d_abs"], whole_frame["true_mean_logit_drop"]
            ),
            "rho_d_rel_true_drop": spearman(
                whole_frame["d_rel"], whole_frame["true_mean_logit_drop"]
            ),
            "high_cstc_mean_drop": float(
                whole_frame.loc[
                    whole_frame["rank_group"] == "high", "true_mean_logit_drop"
                ].mean()
            ),
            "low_cstc_mean_drop": float(
                whole_frame.loc[
                    whole_frame["rank_group"] == "low", "true_mean_logit_drop"
                ].mean()
            ),
        }
    return summary


def write_report(summary: Mapping[str, Any], output_dir: Path) -> None:
    lines = [
        "CSTC Contribution Tube Probe Report",
        "=" * 86,
        "",
        "CSTC is computed from the positive class-conditioned activation-gradient",
        "contribution field using a normalized t-x-y covariance. No optical flow,",
        "correspondence network, DTW, clustering, or tunable fusion weight is used.",
        "",
        "[Synthetic sanity]",
    ]
    for row in summary.get("synthetic", []):
        lines.append(
            f"{row['case']}: CSTC={row['cstc']:.6f}, "
            f"anisotropy={row['anisotropy']:.6f}, "
            f"temporal_extent={row['temporal_extent']:.6f}"
        )
    lines += ["", "[Unit statistics]"]
    for key, value in summary.get("unit_statistics", {}).items():
        lines.append(f"{key}: {value}")
    lines += ["", "[Local masking validation]"]
    for key, value in summary.get("local_masking", {}).items():
        lines.append(f"{key}: {value}")
    lines += ["", "[Whole-unit masking validation]"]
    for key, value in summary.get("whole_unit_masking", {}).items():
        lines.append(f"{key}: {value}")
    lines += [
        "",
        "Interpretation:",
        "1. Synthetic moving/fixed tubes should score above random jumps and a single-frame impulse.",
        "2. Positive local-mask correlation supports activation*gradient as a local deletion proxy.",
        "3. CSTC should not be almost identical to D_abs or D_rel.",
        "4. Positive CSTC-vs-whole-unit-drop correlation is necessary before integrating CSTC into pruning.",
        "5. A negative or near-zero local masking correlation means the contribution-field estimator",
        "   should be rejected even if the tube visualizations look plausible.",
    ]
    (output_dir / "CSTC_PROBE_REPORT.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    ensure_project_importable(Path(args.project_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    synthetic = synthetic_sanity(output_dir)

    loader, selected_indices, chosen_classes = build_balanced_loader(
        project_root=Path(args.project_root),
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model, metadata = load_model(args.adapter, args.checkpoint, device)
    all_specs = discover_unit_layers(model)
    selected_specs = filter_layers(all_specs, args.layers)

    print(f"Model metadata: {metadata}")
    print(f"Selected classes: {chosen_classes}")
    print(f"Selected clips: {len(selected_indices)}")
    print(f"Selected pruning layers: {len(selected_specs)} / {len(all_specs)}")
    for spec in selected_specs:
        print(
            f"  - {spec.name}: {spec.unit_type}, units={spec.num_units}, "
            f"stage={spec.stage}, block={spec.block}"
        )

    results: List[LayerCSTCResult] = []
    groups = list(chunked(selected_specs, max(1, args.layer_batch_size)))
    for group_index, group in enumerate(groups, start=1):
        print(f"\n[Layer group {group_index}/{len(groups)}] {[s.name for s in group]}")
        results.extend(
            run_layer_group(
                model=model,
                loader=loader,
                specs=group,
                device=device,
                target_mode=args.target_mode,
                max_static_samples=args.max_static_samples,
            )
        )

    unit_frame = export_unit_metrics(results, output_dir)
    export_arrays(results, output_dir)
    save_volume_strips(results, output_dir, max(0, args.save_top_volumes))

    cache = collect_cache(loader, args.ablation_videos)
    if args.skip_local_ablation:
        local_frame = pd.DataFrame()
    else:
        local_frame = run_local_ablation(
            model=model,
            cache=cache,
            results=results,
            device=device,
            target_mode=args.target_mode,
            max_layers=args.local_ablation_layers,
            unit_count=args.local_ablation_units,
            point_count=args.local_ablation_points,
            output_dir=output_dir,
            seed=args.seed,
        )

    if args.skip_whole_ablation:
        whole_frame = pd.DataFrame()
    else:
        whole_frame = run_whole_ablation(
            model=model,
            cache=cache,
            results=results,
            device=device,
            target_mode=args.target_mode,
            max_layers=args.whole_ablation_layers,
            unit_count=args.whole_ablation_units,
            output_dir=output_dir,
        )

    if len(unit_frame):
        plot_scatter(
            unit_frame["cstc"], unit_frame["d_abs"],
            "CSTC", "D_abs", "CSTC vs D_abs",
            output_dir / "cstc_vs_d_abs.png",
        )
        plot_scatter(
            unit_frame["cstc"], unit_frame["d_rel"],
            "CSTC", "D_rel", "CSTC vs D_rel",
            output_dir / "cstc_vs_d_rel.png",
        )
    if len(local_frame):
        plot_scatter(
            local_frame["estimated_positive_contribution"],
            local_frame["true_target_logit_drop"],
            "Positive activation-gradient estimate",
            "True local-mask target-logit drop",
            "Local contribution estimate validation",
            output_dir / "estimated_vs_true_local_drop.png",
        )
    if len(whole_frame):
        plot_scatter(
            whole_frame["cstc"],
            whole_frame["true_mean_logit_drop"],
            "CSTC",
            "True whole-unit target-logit drop",
            "CSTC vs true whole-unit deletion effect",
            output_dir / "cstc_vs_true_unit_drop.png",
        )

    summary = build_summary(
        unit_frame, synthetic, local_frame, whole_frame, args, metadata
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_report(summary, output_dir)
    print(f"\nCSTC probe complete. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

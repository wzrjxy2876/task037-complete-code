#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent probe for Task-conditioned Dynamic Contribution Consistency (TDCC).

This script validates whether a Video Swin pruning unit is better represented as a
*dynamical functional entity* rather than a static activation statistic.

It does NOT prune or fine-tune the model. It performs five diagnostics:

1. Unit diversity: different units should exhibit different contribution-evolution curves.
2. Class-conditioned stability: the same unit should be more stable within the same
   action class than across different classes.
3. Causal deletion check: units with higher TDCC should cause a larger target-logit
   drop when their response is masked.
4. Non-redundancy check: TDCC should not collapse to D_abs or D_rel.
5. Role inspection: export early/synchronous/key-stage/late functional-role metadata.

The probe is intentionally layer-wise and memory-safe. Only a small group of layers is
hooked at once, and the model performs one backward pass per calibration clip.

Python: 3.9+
PyTorch: tested against the project style used by torch 1.12.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import random
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


EPS = 1e-8
MAX_STATIC_SAMPLES = 2048
DEFAULT_LAYER_BATCH_SIZE = 1
DEFAULT_SEED = 3407


@dataclass(frozen=True)
class UnitLayerSpec:
    name: str
    unit_type: str
    num_units: int
    stage: int
    block: int
    module: nn.Module
    hook_module: nn.Module


@dataclass
class LayerProbeResult:
    spec: UnitLayerSpec
    video_ids: List[int]
    labels: List[int]
    target_logits: List[float]
    predicted_labels: List[int]
    signed_curves: np.ndarray       # [N,U,T]
    positive_curves: np.ndarray     # [N,U,T]
    cumulative_curves: np.ndarray   # [N,U,T]
    evolution_consistency: np.ndarray  # [N,U]
    class_stability: np.ndarray     # [U]
    tdcc: np.ndarray                # [U]
    d_abs: np.ndarray               # [U]
    d_rel: np.ndarray               # [U]
    substitutability: np.ndarray    # [U]
    mean_abs_activation: np.ndarray # [U]
    active_frequency: np.ndarray    # [U]
    same_class_similarity: np.ndarray # [U]
    cross_class_similarity: np.ndarray # [U]
    role: List[str]
    peak_position: np.ndarray       # [U]
    temporal_centroid: np.ndarray   # [U]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe task-conditioned dynamic contribution consistency for pruning units"
    )
    parser.add_argument("--project_root", type=str, required=True,
                        help="Directory containing MC.py and dataset/")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Unpruned/fine-tuned Video Swin checkpoint")
    parser.add_argument("--output_dir", type=str, default="./tdcc_probe_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--adapter", type=str,
                        default="ucf101_videoswin_probe_adapter_v2",
                        help="Module providing build_model_for_probe and optionally dataset settings")
    parser.add_argument("--val_list", type=str, default="",
                        help="Override UCF101 validation list")
    parser.add_argument("--frame_root", type=str, default="",
                        help="Override UCF101 frame root")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of action classes sampled for the probe")
    parser.add_argument("--videos_per_class", type=int, default=4,
                        help="Number of clips sampled per class")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target_mode", choices=("true", "pred"), default="true")
    parser.add_argument("--layers", type=str, default="representative",
                        help="'representative', 'all', or a regex over named_modules()")
    parser.add_argument("--layer_batch_size", type=int, default=DEFAULT_LAYER_BATCH_SIZE,
                        help="Number of layers hooked per model pass; 1 is safest")
    parser.add_argument("--ablation_layers", type=int, default=4,
                        help="Maximum number of layers used for true masking validation")
    parser.add_argument("--ablation_units", type=int, default=3,
                        help="Top and bottom TDCC unit count per selected layer")
    parser.add_argument("--ablation_videos", type=int, default=8,
                        help="Maximum clips used for each masking test")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--max_static_samples", type=int, default=MAX_STATIC_SAMPLES)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_project_importable(project_root: Path) -> None:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def parse_stage_block(name: str) -> Tuple[int, int]:
    match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", name)
    if match is None:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def discover_unit_layers(model: nn.Module) -> List[UnitLayerSpec]:
    specs: List[UnitLayerSpec] = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        stage, block = parse_stage_block(name)
        if "WindowAttention3D" in class_name:
            specs.append(UnitLayerSpec(
                name=name,
                unit_type="head",
                num_units=int(module.num_heads),
                stage=stage,
                block=block,
                module=module,
                hook_module=module.proj,
            ))
        elif class_name == "Mlp" or "Mlp" in class_name:
            num_units = int(getattr(module, "original_hidden_features", module.fc2.in_features))
            specs.append(UnitLayerSpec(
                name=name,
                unit_type="neuron",
                num_units=num_units,
                stage=stage,
                block=block,
                module=module,
                hook_module=module.fc2,
            ))
    specs.sort(key=lambda item: (item.stage, item.block, 0 if item.unit_type == "head" else 1))
    if not specs:
        raise RuntimeError("No WindowAttention3D or Mlp pruning layers were found")
    return specs


def representative_layer_selection(specs: Sequence[UnitLayerSpec]) -> List[UnitLayerSpec]:
    """Select the first and last block of every stage for both unit types."""
    grouped: Dict[Tuple[int, str], List[UnitLayerSpec]] = defaultdict(list)
    for spec in specs:
        grouped[(spec.stage, spec.unit_type)].append(spec)
    selected: List[UnitLayerSpec] = []
    seen: set[str] = set()
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: item.block)
        for item in (items[0], items[-1]):
            if item.name not in seen:
                selected.append(item)
                seen.add(item.name)
    selected.sort(key=lambda item: (item.stage, item.block, 0 if item.unit_type == "head" else 1))
    return selected


def filter_layers(specs: Sequence[UnitLayerSpec], selector: str) -> List[UnitLayerSpec]:
    selector = selector.strip()
    if selector == "all":
        return list(specs)
    if selector == "representative":
        return representative_layer_selection(specs)
    pattern = re.compile(selector)
    selected = [spec for spec in specs if pattern.search(spec.name)]
    if not selected:
        raise ValueError(f"Layer regex matched no pruning layers: {selector!r}")
    return selected


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_balanced_loader(
    project_root: Path,
    val_list: str,
    frame_root: str,
    num_classes: int,
    videos_per_class: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, List[int], List[int]]:
    # dataset/ucf101.py only needs UCF_DATA_ROOT from the project utils module.
    # Importing the full utils.py pulls in an old GluonCV stack, which imports
    # torch._six.int_classes and is incompatible with the current PyTorch.
    # Install a minimal temporary shim so the dataset can be imported without
    # loading GluonCV. This probe does not use any other project-utils symbols.
    resolved_frame_root = frame_root or os.environ.get("UCF101_FRAME_ROOT", "")
    if not resolved_frame_root:
        raise ValueError(
            "Frame root is required. Pass --frame_root or set UCF101_FRAME_ROOT."
        )

    previous_utils = sys.modules.get("utils")
    shim_installed = previous_utils is None
    if shim_installed:
        utils_shim = types.ModuleType("utils")
        utils_shim.UCF_DATA_ROOT = resolved_frame_root
        sys.modules["utils"] = utils_shim

    try:
        from dataset import ucf101 as ucf101_dataset
    finally:
        if shim_installed:
            sys.modules.pop("utils", None)

    # ucf101.py imports UCF_DATA_ROOT by value, so overwrite its module global
    # explicitly to guarantee the CLI path is used.
    ucf101_dataset.UCF_DATA_ROOT = resolved_frame_root

    setting_path = val_list or os.environ.get(
        "UCF101_VAL_LIST", "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
    )
    spatial_transform, temporal_transform = ucf101_dataset.test_transform()
    dataset = ucf101_dataset.attack_ucf101(
        setting_path,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
    )

    # Pre-validate clips before balanced sampling. The original UCF101 dataset
    # returns an empty list when the first requested frame is missing; its
    # __getitem__ then fails at torch.stack([]). This commonly happens when
    # --frame_root points to the wrong directory level, or when a clip has
    # duration <= 1 while LoopPadding starts from frame index 2.
    by_class: Dict[int, List[int]] = defaultdict(list)
    invalid_clips: List[Tuple[int, str, int, str]] = []

    for index, (directory, duration, label) in enumerate(dataset.clips):
        frame_indices = list(range(1, int(duration) + 1))
        sampled_indices = (
            temporal_transform(frame_indices)
            if temporal_transform is not None
            else frame_indices
        )

        reason = ""
        if not os.path.isdir(directory):
            reason = "video directory does not exist"
        elif not sampled_indices:
            reason = (
                "temporal transform returned no frames "
                f"(duration={int(duration)}; LoopPadding requires duration >= 2)"
            )
        else:
            missing = [
                frame_idx
                for frame_idx in sampled_indices
                if not os.path.isfile(
                    os.path.join(directory, f"image_{int(frame_idx):05d}.jpg")
                )
            ]
            if missing:
                preview = ",".join(str(v) for v in missing[:5])
                reason = f"missing sampled frame(s): {preview}"

        if reason:
            invalid_clips.append((index, directory, int(duration), reason))
            continue

        by_class[int(label)].append(index)

    print(
        f"Valid clips after path/frame validation: "
        f"{sum(len(v) for v in by_class.values())}/{len(dataset.clips)}"
    )
    if invalid_clips:
        print("Examples of skipped invalid clips:")
        for bad_index, bad_dir, bad_duration, bad_reason in invalid_clips[:8]:
            print(
                f"  - dataset_index={bad_index}, duration={bad_duration}, "
                f"dir={bad_dir!r}, reason={bad_reason}"
            )

    if not by_class:
        example_dir = dataset.clips[0][0] if dataset.clips else "<dataset is empty>"
        raise RuntimeError(
            "No valid UCF101 clips were found. The most likely cause is that "
            "--frame_root points to the wrong directory level. According to "
            "dataset/ucf101.py, the expected layout is "
            "'<frame_root>/<class_name>/<video_name>/image_00002.jpg'. "
            f"The first resolved clip directory was: {example_dir!r}"
        )

    eligible = sorted(
        label for label, indices in by_class.items()
        if len(indices) >= videos_per_class
    )
    if len(eligible) < num_classes:
        valid_counts = sorted(
            ((label, len(indices)) for label, indices in by_class.items()),
            key=lambda item: (-item[1], item[0]),
        )
        raise ValueError(
            f"Only {len(eligible)} valid classes have >= {videos_per_class} clips; "
            f"requested {num_classes} classes. Top valid class counts: "
            f"{valid_counts[:10]}. Invalid clips skipped: {len(invalid_clips)}."
        )

    rng = np.random.RandomState(seed)
    chosen_classes = sorted(rng.choice(eligible, size=num_classes, replace=False).tolist())
    selected_indices: List[int] = []
    selected_labels: List[int] = []
    for label in chosen_classes:
        candidates = np.array(by_class[label], dtype=np.int64)
        chosen = rng.choice(candidates, size=videos_per_class, replace=False).tolist()
        selected_indices.extend(int(index) for index in chosen)
        selected_labels.extend([int(label)] * videos_per_class)

    subset = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    return loader, selected_indices, chosen_classes


def load_model(adapter_name: str, checkpoint: str, device: torch.device) -> Tuple[nn.Module, Mapping[str, Any]]:
    adapter = importlib.import_module(adapter_name)
    if not hasattr(adapter, "build_model_for_probe"):
        raise AttributeError(f"{adapter_name} has no build_model_for_probe()")
    model, metadata = adapter.build_model_for_probe(checkpoint=checkpoint, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def unwrap_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Could not extract logits from model output type {type(output)!r}")


def robust_scale_01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() == 0:
        return values
    median = values.median()
    q1 = torch.quantile(values, 0.25)
    q3 = torch.quantile(values, 0.75)
    scale = (q3 - q1).clamp_min(1e-8)
    z = (values - median) / scale
    return torch.sigmoid(z)


def compute_d_abs_d_rel(static_samples: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the project's D_abs and D_rel definitions for one layer."""
    z = static_samples.float()
    mean_amp = z.abs().mean(dim=0)
    frequency = (z.abs() > 1e-4).float().mean(dim=0)
    mean_scaled = robust_scale_01(mean_amp)
    frequency_scaled = robust_scale_01(frequency)
    d_abs = 0.5 * mean_scaled + 0.5 * frequency_scaled

    centered = z - z.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True)
    standardized = centered / scale.clamp_min(1e-5)
    standardized[:, scale.squeeze(0) <= 1e-5] = 0.0
    cov = standardized.T @ standardized / max(z.shape[0], 1)
    cov = cov + 1e-5 * torch.eye(z.shape[1], dtype=cov.dtype)
    factor, info = torch.linalg.cholesky_ex(cov)
    if int(info.max()) == 0:
        inv_cov = torch.cholesky_inverse(factor)
    else:
        inv_cov = torch.linalg.pinv(cov)
    err_var = 1.0 / torch.diag(inv_cov).clamp_min(1e-5)
    orig_var = torch.diag(cov)
    substitutability = torch.clamp(1.0 - err_var / (orig_var + 1e-5), 0.0, 1.0)
    d_rel = d_abs * (1.0 - substitutability)
    return tuple(array.detach().cpu().numpy() for array in (
        d_abs, d_rel, substitutability, mean_amp, frequency
    ))


def normalize_positive_curve(curve: torch.Tensor) -> torch.Tensor:
    positive = torch.relu(curve)
    denominator = positive.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / max(positive.shape[-1], 1))
    return torch.where(denominator > EPS, positive / denominator.clamp_min(EPS), uniform)


def resample_time(curve: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Resample [...,T] to [...,target_bins] without learnable parameters."""
    if curve.shape[-1] == target_bins:
        return curve
    shape = curve.shape
    flat = curve.reshape(-1, 1, shape[-1])
    resized = torch.nn.functional.interpolate(
        flat, size=target_bins, mode="linear", align_corners=False
    )
    return resized.reshape(*shape[:-1], target_bins)


def window_reverse_3d_local(windows: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    """Reverse [B*nW,N,U] windows to [B,U,T,H,W]."""
    wd, wh, ww = [int(value) for value in geometry["window_size"]]
    batch_size = int(geometry["batch_size"])
    dp = int(geometry["padded_depth"])
    hp = int(geometry["padded_height"])
    wp = int(geometry["padded_width"])
    units = windows.shape[-1]
    x = windows.view(
        batch_size,
        dp // wd,
        hp // wh,
        wp // ww,
        wd,
        wh,
        ww,
        units,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(batch_size, dp, hp, wp, units)
    shift = tuple(int(value) for value in geometry["shift_size"])
    if any(value > 0 for value in shift):
        x = torch.roll(x, shifts=shift, dims=(1, 2, 3))
    depth = int(geometry["depth"])
    height = int(geometry["height"])
    width = int(geometry["width"])
    x = x[:, :depth, :height, :width, :]
    return x.permute(0, 4, 1, 2, 3).contiguous()


class DynamicContributionCollector:
    """Collect signed activation-gradient temporal contribution for selected layers."""

    def __init__(self, specs: Sequence[UnitLayerSpec], max_static_samples: int = MAX_STATIC_SAMPLES):
        self.specs = {spec.name: spec for spec in specs}
        self.max_static_samples = int(max_static_samples)
        self.handles: List[Any] = []
        self.current_contributions: Dict[str, torch.Tensor] = {}
        self.static_samples: Dict[str, torch.Tensor] = {}

    def register(self) -> None:
        for spec in self.specs.values():
            handle = spec.hook_module.register_forward_pre_hook(self._make_hook(spec))
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_batch(self) -> None:
        self.current_contributions = {}

    def _append_static_samples(self, name: str, response: torch.Tensor) -> None:
        # response [B,U,T,H,W] -> [M,U]
        samples = response.detach().abs().permute(0, 2, 3, 4, 1).reshape(-1, response.shape[1])
        if samples.shape[0] > self.max_static_samples:
            indices = torch.randperm(samples.shape[0], device=samples.device)[:self.max_static_samples]
            samples = samples.index_select(0, indices)
        samples = samples.to(device="cpu", dtype=torch.float32)
        previous = self.static_samples.get(name)
        if previous is not None:
            samples = torch.cat((previous, samples), dim=0)
            if samples.shape[0] > self.max_static_samples:
                indices = torch.randperm(samples.shape[0])[:self.max_static_samples]
                samples = samples.index_select(0, indices)
        self.static_samples[name] = samples

    def _attention_response(self, x: torch.Tensor, spec: UnitLayerSpec) -> torch.Tensor:
        geometry = getattr(spec.module, "_pruning_geometry", None)
        if geometry is None:
            raise RuntimeError(f"Missing _pruning_geometry for {spec.name}")
        head_dim = int(spec.module.head_dim)
        windows = x.reshape(x.shape[0], x.shape[1], spec.num_units, head_dim).abs().mean(dim=-1)
        return window_reverse_3d_local(windows, geometry)

    def _attention_contribution(self, x: torch.Tensor, grad: torch.Tensor, spec: UnitLayerSpec) -> torch.Tensor:
        geometry = getattr(spec.module, "_pruning_geometry", None)
        head_dim = int(spec.module.head_dim)
        local = (x * grad).reshape(x.shape[0], x.shape[1], spec.num_units, head_dim).sum(dim=-1)
        response = window_reverse_3d_local(local, geometry)
        return response.mean(dim=(-1, -2))  # [B,U,T]

    @staticmethod
    def _mlp_response(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _mlp_contribution(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        local = x * grad
        return local.mean(dim=(2, 3)).permute(0, 2, 1).contiguous()  # [B,U,T]

    def _make_hook(self, spec: UnitLayerSpec) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], None]:
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            if not x.requires_grad:
                x.requires_grad_(True)
            if spec.unit_type == "head":
                response = self._attention_response(x.detach(), spec)
            else:
                response = self._mlp_response(x.detach())
            self._append_static_samples(spec.name, response)

            activation = x

            def gradient_hook(grad: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    if spec.unit_type == "head":
                        contribution = self._attention_contribution(activation.detach(), grad.detach(), spec)
                    else:
                        contribution = self._mlp_contribution(activation.detach(), grad.detach())
                    self.current_contributions[spec.name] = contribution.detach().cpu().float()
                return grad

            x.register_hook(gradient_hook)

        return hook


def leave_one_out_evolution_consistency(cumulative: torch.Tensor) -> torch.Tensor:
    """Compare each unit curve to leave-one-out layer evidence, shape [U,T] -> [U]."""
    unit_distribution = cumulative[:, -1].new_zeros(cumulative.shape)  # placeholder dtype/device
    del unit_distribution
    # Recover normalized incremental distributions from cumulative curves.
    increments = torch.diff(
        torch.cat((torch.zeros(cumulative.shape[0], 1, device=cumulative.device), cumulative), dim=1),
        dim=1,
    ).clamp_min(0.0)
    total = increments.sum(dim=0, keepdim=True)
    loo = (total - increments).clamp_min(0.0)
    loo_denominator = loo.sum(dim=1, keepdim=True)
    uniform = torch.full_like(loo, 1.0 / max(loo.shape[1], 1))
    loo = torch.where(loo_denominator > EPS, loo / loo_denominator.clamp_min(EPS), uniform)
    loo_cumulative = loo.cumsum(dim=1)
    return 1.0 - torch.mean(torch.abs(cumulative - loo_cumulative), dim=1)


def pair_similarity(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    return float(1.0 - np.mean(np.abs(curve_a - curve_b)))


def compute_class_stability(
    cumulative: np.ndarray, labels: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class-prototype stability and same/cross class pair similarities."""
    n_videos, n_units, _ = cumulative.shape
    labels_np = np.asarray(labels, dtype=np.int64)
    class_stability = np.zeros(n_units, dtype=np.float64)
    same_similarity = np.full(n_units, np.nan, dtype=np.float64)
    cross_similarity = np.full(n_units, np.nan, dtype=np.float64)

    for unit in range(n_units):
        stability_values: List[float] = []
        same_values: List[float] = []
        cross_values: List[float] = []
        for index in range(n_videos):
            same_indices = np.flatnonzero(labels_np == labels_np[index])
            same_indices = same_indices[same_indices != index]
            if same_indices.size:
                prototype = cumulative[same_indices, unit].mean(axis=0)
                stability_values.append(pair_similarity(cumulative[index, unit], prototype))
        for left in range(n_videos):
            for right in range(left + 1, n_videos):
                similarity = pair_similarity(cumulative[left, unit], cumulative[right, unit])
                if labels_np[left] == labels_np[right]:
                    same_values.append(similarity)
                else:
                    cross_values.append(similarity)
        class_stability[unit] = np.mean(stability_values) if stability_values else np.nan
        same_similarity[unit] = np.mean(same_values) if same_values else np.nan
        cross_similarity[unit] = np.mean(cross_values) if cross_values else np.nan
    return class_stability, same_similarity, cross_similarity


def classify_functional_role(mean_distribution: np.ndarray) -> Tuple[str, float, float]:
    time = np.linspace(0.0, 1.0, mean_distribution.shape[-1], dtype=np.float64)
    distribution = mean_distribution / max(mean_distribution.sum(), EPS)
    centroid = float(np.sum(time * distribution))
    peak = float(time[int(np.argmax(distribution))])
    entropy = float(-np.sum(distribution * np.log(distribution + EPS)) / math.log(max(len(distribution), 2)))
    if entropy < 0.72:
        role = "key-stage"
    elif centroid < 0.38:
        role = "early"
    elif centroid > 0.62:
        role = "late"
    else:
        role = "synchronous"
    return role, peak, centroid


def run_layer_group(
    model: nn.Module,
    loader: DataLoader,
    specs: Sequence[UnitLayerSpec],
    device: torch.device,
    target_mode: str,
    max_static_samples: int,
) -> List[LayerProbeResult]:
    collector = DynamicContributionCollector(specs, max_static_samples=max_static_samples)
    collector.register()
    curves_by_layer: Dict[str, List[torch.Tensor]] = defaultdict(list)
    video_ids: List[int] = []
    labels: List[int] = []
    target_logits: List[float] = []
    predicted_labels: List[int] = []

    try:
        for videos, targets, indices in tqdm(loader, desc="Contribution forward/backward", ncols=100):
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
                        f"Target label {int(chosen.max())} exceeds classifier width {logits.shape[1]}"
                    )
                selected_logit = logits.gather(1, chosen[:, None]).sum()
                selected_logit.backward()

            missing = [spec.name for spec in specs if spec.name not in collector.current_contributions]
            if missing:
                raise RuntimeError(f"No backward contribution captured for layers: {missing}")
            target_logits.extend(
                logits.detach().gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
            )
            predicted_labels.extend(predicted.detach().cpu().tolist())
            labels.extend(targets.detach().cpu().tolist())
            if torch.is_tensor(indices):
                video_ids.extend(indices.cpu().tolist())
            else:
                video_ids.extend([int(value) for value in indices])
            for spec in specs:
                curves_by_layer[spec.name].append(collector.current_contributions[spec.name].squeeze(0))
            del videos, targets, logits, selected_logit
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        collector.remove()

    results: List[LayerProbeResult] = []
    for spec in specs:
        raw_curves = torch.stack(curves_by_layer[spec.name], dim=0)  # [N,U,Tnative]
        reference_bins = max(int(curve.shape[-1]) for curve in curves_by_layer[spec.name])
        raw_curves = resample_time(raw_curves, reference_bins)
        positive = normalize_positive_curve(raw_curves)
        cumulative = positive.cumsum(dim=-1)
        evolution = torch.stack([
            leave_one_out_evolution_consistency(cumulative[index])
            for index in range(cumulative.shape[0])
        ], dim=0)

        cumulative_np = cumulative.numpy()
        class_stability, same_similarity, cross_similarity = compute_class_stability(
            cumulative_np, labels
        )
        evolution_mean = evolution.mean(dim=0).numpy()
        tdcc = np.clip(evolution_mean * np.nan_to_num(class_stability, nan=0.0), 0.0, 1.0)
        d_abs, d_rel, substitutability, mean_amp, frequency = compute_d_abs_d_rel(
            collector.static_samples[spec.name]
        )
        mean_distribution = positive.mean(dim=0).numpy()
        roles: List[str] = []
        peak_positions: List[float] = []
        centroids: List[float] = []
        for unit in range(spec.num_units):
            role, peak, centroid = classify_functional_role(mean_distribution[unit])
            roles.append(role)
            peak_positions.append(peak)
            centroids.append(centroid)

        results.append(LayerProbeResult(
            spec=spec,
            video_ids=list(video_ids),
            labels=list(labels),
            target_logits=list(target_logits),
            predicted_labels=list(predicted_labels),
            signed_curves=raw_curves.numpy(),
            positive_curves=positive.numpy(),
            cumulative_curves=cumulative_np,
            evolution_consistency=evolution.numpy(),
            class_stability=class_stability,
            tdcc=tdcc,
            d_abs=d_abs,
            d_rel=d_rel,
            substitutability=substitutability,
            mean_abs_activation=mean_amp,
            active_frequency=frequency,
            same_class_similarity=same_similarity,
            cross_class_similarity=cross_similarity,
            role=roles,
            peak_position=np.asarray(peak_positions),
            temporal_centroid=np.asarray(centroids),
        ))
    return results


def chunked(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


@contextlib.contextmanager
def unit_mask(spec: UnitLayerSpec, unit_index: int):
    """Temporarily zero one pruning unit at the response entering proj/fc2."""
    def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        x = inputs[0]
        masked = x.clone()
        if spec.unit_type == "head":
            head_dim = int(spec.module.head_dim)
            reshaped = masked.reshape(masked.shape[0], masked.shape[1], spec.num_units, head_dim)
            reshaped[:, :, unit_index, :] = 0.0
            masked = reshaped.reshape_as(masked)
        else:
            masked[..., unit_index] = 0.0
        return (masked,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_ablation_cache(loader: DataLoader, limit: int) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    cache: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for videos, targets, indices in loader:
        index = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        cache.append((videos.cpu(), targets.cpu(), index))
        if len(cache) >= limit:
            break
    return cache


def evaluate_unit_ablation(
    model: nn.Module,
    spec: UnitLayerSpec,
    unit_index: int,
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
) -> Tuple[float, float, List[float]]:
    drops: List[float] = []
    relative_drops: List[float] = []
    model.eval()
    for videos_cpu, targets_cpu, _ in cache:
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)
        with torch.no_grad():
            baseline_logits = unwrap_logits(model(videos))
            predicted = baseline_logits.argmax(dim=1)
            chosen = targets if target_mode == "true" else predicted
            baseline = baseline_logits.gather(1, chosen[:, None]).squeeze(1)
            with unit_mask(spec, unit_index):
                masked_logits = unwrap_logits(model(videos))
            masked = masked_logits.gather(1, chosen[:, None]).squeeze(1)
            drop = (baseline - masked).item()
            drops.append(drop)
            relative_drops.append(drop / (abs(baseline.item()) + EPS))
    return float(np.mean(drops)), float(np.mean(relative_drops)), drops


def run_ablation_validation(
    model: nn.Module,
    loader: DataLoader,
    results: Sequence[LayerProbeResult],
    device: torch.device,
    target_mode: str,
    max_layers: int,
    unit_count: int,
    video_count: int,
    output_dir: Path,
) -> pd.DataFrame:
    cache = collect_ablation_cache(loader, video_count)
    selected_results = list(results)
    if len(selected_results) > max_layers:
        # Spread selected layers across network depth.
        indices = np.linspace(0, len(selected_results) - 1, max_layers).round().astype(int)
        selected_results = [selected_results[index] for index in sorted(set(indices.tolist()))]

    rows: List[Dict[str, Any]] = []
    for result in tqdm(selected_results, desc="True unit masking", ncols=100):
        order = np.argsort(result.tdcc)
        bottom = order[:min(unit_count, len(order))]
        top = order[-min(unit_count, len(order)):][::-1]
        selected = [("low", int(index)) for index in bottom] + [("high", int(index)) for index in top]
        seen: set[int] = set()
        for rank_group, unit_index in selected:
            if unit_index in seen:
                continue
            seen.add(unit_index)
            mean_drop, relative_drop, per_video = evaluate_unit_ablation(
                model=model,
                spec=result.spec,
                unit_index=unit_index,
                cache=cache,
                device=device,
                target_mode=target_mode,
            )
            rows.append({
                "layer": result.spec.name,
                "stage": result.spec.stage,
                "block": result.spec.block,
                "unit_type": result.spec.unit_type,
                "unit_index": unit_index,
                "tdcc_group": rank_group,
                "tdcc": float(result.tdcc[unit_index]),
                "d_abs": float(result.d_abs[unit_index]),
                "d_rel": float(result.d_rel[unit_index]),
                "mean_logit_drop": mean_drop,
                "mean_relative_logit_drop": relative_drop,
                "positive_drop_rate": float(np.mean(np.asarray(per_video) > 0.0)),
                "num_videos": len(per_video),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "true_masking_validation.csv", index=False)
    return frame


def export_unit_table(results: Sequence[LayerProbeResult], output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        for unit in range(result.spec.num_units):
            rows.append({
                "layer": result.spec.name,
                "stage": result.spec.stage,
                "block": result.spec.block,
                "unit_type": result.spec.unit_type,
                "unit_index": unit,
                "tdcc": float(result.tdcc[unit]),
                "mean_evolution_consistency": float(result.evolution_consistency[:, unit].mean()),
                "class_stability": float(result.class_stability[unit]),
                "same_class_similarity": float(result.same_class_similarity[unit]),
                "cross_class_similarity": float(result.cross_class_similarity[unit]),
                "same_minus_cross": float(result.same_class_similarity[unit] - result.cross_class_similarity[unit]),
                "d_abs": float(result.d_abs[unit]),
                "d_rel": float(result.d_rel[unit]),
                "substitutability": float(result.substitutability[unit]),
                "mean_abs_activation": float(result.mean_abs_activation[unit]),
                "active_frequency": float(result.active_frequency[unit]),
                "functional_role": result.role[unit],
                "peak_position": float(result.peak_position[unit]),
                "temporal_centroid": float(result.temporal_centroid[unit]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "unit_tdcc_metrics.csv", index=False)
    return frame


def export_curve_arrays(results: Sequence[LayerProbeResult], output_dir: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}
    for index, result in enumerate(results):
        prefix = f"layer_{index:03d}"
        arrays[f"{prefix}_signed"] = result.signed_curves.astype(np.float32)
        arrays[f"{prefix}_positive"] = result.positive_curves.astype(np.float32)
        arrays[f"{prefix}_cumulative"] = result.cumulative_curves.astype(np.float32)
        arrays[f"{prefix}_evolution"] = result.evolution_consistency.astype(np.float32)
        metadata[prefix] = {
            "name": result.spec.name,
            "stage": result.spec.stage,
            "block": result.spec.block,
            "unit_type": result.spec.unit_type,
            "num_units": result.spec.num_units,
            "video_ids": result.video_ids,
            "labels": result.labels,
        }
    np.savez_compressed(output_dir / "tdcc_curve_arrays.npz", **arrays)
    with (output_dir / "tdcc_curve_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def plot_summary(unit_frame: pd.DataFrame, ablation_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(unit_frame["d_abs"], unit_frame["tdcc"], s=12, alpha=0.6)
    plt.xlabel("D_abs")
    plt.ylabel("TDCC")
    plt.title("TDCC versus response strength")
    plt.tight_layout()
    plt.savefig(output_dir / "tdcc_vs_d_abs.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(unit_frame["d_rel"], unit_frame["tdcc"], s=12, alpha=0.6)
    plt.xlabel("D_rel")
    plt.ylabel("TDCC")
    plt.title("TDCC versus relative importance")
    plt.tight_layout()
    plt.savefig(output_dir / "tdcc_vs_d_rel.png", dpi=220)
    plt.close()

    layer_order = list(dict.fromkeys(unit_frame["layer"].tolist()))
    same_means = unit_frame.groupby("layer")["same_class_similarity"].mean().reindex(layer_order)
    cross_means = unit_frame.groupby("layer")["cross_class_similarity"].mean().reindex(layer_order)
    x = np.arange(len(layer_order))
    width = 0.38
    plt.figure(figsize=(max(10, len(layer_order) * 0.55), 5))
    plt.bar(x - width / 2, same_means.values, width=width, label="same class")
    plt.bar(x + width / 2, cross_means.values, width=width, label="cross class")
    plt.xticks(x, layer_order, rotation=75, ha="right")
    plt.ylabel("Contribution-curve similarity")
    plt.title("Within-class versus cross-class functional stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "same_vs_cross_class_stability.png", dpi=220)
    plt.close()

    role_counts = unit_frame["functional_role"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(role_counts.index, role_counts.values)
    plt.ylabel("Number of units")
    plt.title("Dynamic functional roles")
    plt.tight_layout()
    plt.savefig(output_dir / "functional_role_distribution.png", dpi=220)
    plt.close()

    if not ablation_frame.empty:
        plt.figure(figsize=(7, 5))
        for group, group_frame in ablation_frame.groupby("tdcc_group"):
            plt.scatter(
                group_frame["tdcc"], group_frame["mean_logit_drop"],
                s=35, alpha=0.75, label=group,
            )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("TDCC")
        plt.ylabel("Target-logit drop after masking")
        plt.title("True masking validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "tdcc_vs_true_logit_drop.png", dpi=220)
        plt.close()


def build_summary(
    unit_frame: pd.DataFrame,
    ablation_frame: pd.DataFrame,
    results: Sequence[LayerProbeResult],
    chosen_classes: Sequence[int],
    model_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    same_mean = float(unit_frame["same_class_similarity"].mean())
    cross_mean = float(unit_frame["cross_class_similarity"].mean())
    summary: Dict[str, Any] = {
        "model_metadata": dict(model_metadata),
        "selected_classes": list(map(int, chosen_classes)),
        "num_layers": len(results),
        "num_units": int(len(unit_frame)),
        "hypothesis_1_unit_diversity": {
            "tdcc_std": float(unit_frame["tdcc"].std()),
            "temporal_centroid_std": float(unit_frame["temporal_centroid"].std()),
            "peak_position_std": float(unit_frame["peak_position"].std()),
            "supported": bool(unit_frame["tdcc"].std() > 0.01 and unit_frame["temporal_centroid"].std() > 0.02),
        },
        "hypothesis_2_class_conditioned_stability": {
            "same_class_mean": same_mean,
            "cross_class_mean": cross_mean,
            "gap": same_mean - cross_mean,
            "unit_positive_gap_rate": float((unit_frame["same_minus_cross"] > 0).mean()),
            "supported": bool(same_mean > cross_mean),
        },
        "hypothesis_4_non_redundancy": {
            "spearman_tdcc_d_abs": spearman(unit_frame["tdcc"], unit_frame["d_abs"]),
            "spearman_tdcc_d_rel": spearman(unit_frame["tdcc"], unit_frame["d_rel"]),
        },
        "functional_role_counts": {
            str(key): int(value) for key, value in unit_frame["functional_role"].value_counts().items()
        },
    }
    non_redundancy = summary["hypothesis_4_non_redundancy"]
    correlations = [abs(value) for value in non_redundancy.values() if np.isfinite(value)]
    non_redundancy["supported"] = bool(correlations and max(correlations) < 0.90)

    if not ablation_frame.empty:
        high = ablation_frame.loc[ablation_frame["tdcc_group"] == "high", "mean_logit_drop"]
        low = ablation_frame.loc[ablation_frame["tdcc_group"] == "low", "mean_logit_drop"]
        ablation_corr = spearman(ablation_frame["tdcc"], ablation_frame["mean_logit_drop"])
        summary["hypothesis_3_causal_masking"] = {
            "high_tdcc_mean_logit_drop": float(high.mean()) if len(high) else float("nan"),
            "low_tdcc_mean_logit_drop": float(low.mean()) if len(low) else float("nan"),
            "high_minus_low": float(high.mean() - low.mean()) if len(high) and len(low) else float("nan"),
            "spearman_tdcc_logit_drop": ablation_corr,
            "supported": bool(len(high) and len(low) and high.mean() > low.mean()),
        }
    else:
        summary["hypothesis_3_causal_masking"] = {"skipped": True, "supported": None}
    return summary


def write_report(summary: Mapping[str, Any], output_dir: Path) -> None:
    lines = [
        "TDCC DYNAMIC FUNCTION PROBE REPORT",
        "=" * 72,
        "",
        "This probe is diagnostic only. It does not prune or fine-tune the model.",
        "A hypothesis is marked supported only by the simple pre-registered criterion",
        "implemented in this script; final paper claims still require repeated seeds.",
        "",
    ]
    for key in (
        "hypothesis_1_unit_diversity",
        "hypothesis_2_class_conditioned_stability",
        "hypothesis_3_causal_masking",
        "hypothesis_4_non_redundancy",
    ):
        lines.append(key)
        lines.append("-" * len(key))
        values = summary.get(key, {})
        for metric, value in values.items():
            lines.append(f"{metric}: {value}")
        lines.append("")
    lines.append("Interpretation rule:")
    lines.append("- H1: curves vary across units, so units are not static scalar replicas.")
    lines.append("- H2: same-class stability exceeds cross-class stability.")
    lines.append("- H3: masking high-TDCC units causes a larger target-logit drop.")
    lines.append("- H4: |rho(TDCC,D_abs/D_rel)| remains below 0.90.")
    (output_dir / "TDCC_PROBE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    ensure_project_importable(project_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, model_metadata = load_model(args.adapter, args.checkpoint, device)
    all_specs = discover_unit_layers(model)
    selected_specs = filter_layers(all_specs, args.layers)

    loader, selected_indices, chosen_classes = build_balanced_loader(
        project_root=project_root,
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    run_config = vars(args).copy()
    run_config.update({
        "resolved_device": str(device),
        "selected_indices": selected_indices,
        "selected_classes": chosen_classes,
        "selected_layers": [spec.name for spec in selected_specs],
    })
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print(f"Model metadata: {model_metadata}")
    print(f"Selected classes: {chosen_classes}")
    print(f"Selected clips: {len(selected_indices)}")
    print(f"Selected pruning layers: {len(selected_specs)} / {len(all_specs)}")
    for spec in selected_specs:
        print(f"  - {spec.name}: {spec.unit_type}, units={spec.num_units}")

    results: List[LayerProbeResult] = []
    for group_index, layer_group in enumerate(chunked(selected_specs, max(args.layer_batch_size, 1)), start=1):
        print(f"\n[Layer group {group_index}] {[spec.name for spec in layer_group]}")
        group_results = run_layer_group(
            model=model,
            loader=loader,
            specs=layer_group,
            device=device,
            target_mode=args.target_mode,
            max_static_samples=args.max_static_samples,
        )
        results.extend(group_results)

    unit_frame = export_unit_table(results, output_dir)
    export_curve_arrays(results, output_dir)

    if args.skip_ablation:
        ablation_frame = pd.DataFrame()
    else:
        ablation_frame = run_ablation_validation(
            model=model,
            loader=loader,
            results=results,
            device=device,
            target_mode=args.target_mode,
            max_layers=args.ablation_layers,
            unit_count=args.ablation_units,
            video_count=args.ablation_videos,
            output_dir=output_dir,
        )

    plot_summary(unit_frame, ablation_frame, output_dir)
    summary = build_summary(
        unit_frame=unit_frame,
        ablation_frame=ablation_frame,
        results=results,
        chosen_classes=chosen_classes,
        model_metadata=model_metadata,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    write_report(summary, output_dir)

    print("\nProbe complete.")
    print(f"Outputs: {output_dir}")
    for key in (
        "hypothesis_1_unit_diversity",
        "hypothesis_2_class_conditioned_stability",
        "hypothesis_3_causal_masking",
        "hypothesis_4_non_redundancy",
    ):
        print(f"{key}: {summary.get(key)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent probe for Class-conditioned Temporal Functional Role Stability (CTFRS).

This script validates whether a Video Swin pruning unit is better represented as a
*dynamical functional entity* rather than a static activation statistic.

It does NOT prune or fine-tune the model. It performs five diagnostics:

1. Role diversity: different units should exhibit different class-conditioned temporal roles.
2. Within-class stability: each unit should repeat its role across videos of the same class.
3. Causal deletion check: units with higher CTFRS should cause a larger target-logit
   drop when their response is masked.
4. Non-redundancy check: CTFRS should not collapse to D_abs or D_rel.
5. Role inspection: export persistent/early/key-stage/late role metadata.

CTFRS is computed with a parameter-free one-dimensional Wasserstein-1 distance
between each video's normalized positive contribution distribution and the same unit's
class prototype. It does not compare units with a global network evidence curve.

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
    signed_curves: np.ndarray          # [N,U,T]
    role_distributions: np.ndarray     # [N,U,T], normalized positive contribution
    cumulative_curves: np.ndarray      # [N,U,T]
    sample_role_stability: np.ndarray  # [N,U], 1-W1(sample,class prototype)
    within_class_dispersion: np.ndarray # [U], mean W1 to class prototype
    ctfrs: np.ndarray                  # [U], 1-within_class_dispersion
    same_class_similarity: np.ndarray  # [U]
    cross_class_similarity: np.ndarray # [U]
    between_class_distance: np.ndarray # [U]
    class_role_gap: np.ndarray         # [U], between - within
    d_abs: np.ndarray
    d_rel: np.ndarray
    substitutability: np.ndarray
    mean_abs_activation: np.ndarray
    active_frequency: np.ndarray
    role: List[str]
    peak_position: np.ndarray
    temporal_centroid: np.ndarray
    role_concentration: np.ndarray
    class_prototypes: Dict[int, np.ndarray] # class -> [U,T]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe class-conditioned temporal functional role stability for pruning units"
    )
    parser.add_argument("--project_root", type=str, required=True,
                        help="Directory containing MC.py and dataset/")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Unpruned/fine-tuned Video Swin checkpoint")
    parser.add_argument("--output_dir", type=str, default="./ctfrs_probe_output")
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
                        help="Top and bottom CTFRS unit count per selected layer")
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


def wasserstein_1_cdf(distribution_a: np.ndarray, distribution_b: np.ndarray) -> float:
    """Normalized 1-D Wasserstein-1 distance on an equally spaced [0,1] timeline."""
    a = np.asarray(distribution_a, dtype=np.float64)
    b = np.asarray(distribution_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"Expected equal 1-D distributions, got {a.shape} and {b.shape}")
    if a.size <= 1:
        return 0.0
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    a = a / max(float(a.sum()), EPS)
    b = b / max(float(b.sum()), EPS)
    cdf_a = np.cumsum(a)[:-1]
    cdf_b = np.cumsum(b)[:-1]
    return float(np.mean(np.abs(cdf_a - cdf_b)))


def role_similarity(distribution_a: np.ndarray, distribution_b: np.ndarray) -> float:
    return float(np.clip(1.0 - wasserstein_1_cdf(distribution_a, distribution_b), 0.0, 1.0))


def compute_ctfrs_metrics(
    role_distributions: np.ndarray,
    labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Compute CTFRS and diagnostics.

    Args:
        role_distributions: [N,U,T] normalized positive contribution distributions.
        labels: N class labels.

    Returns:
        sample_stability [N,U]
        within_dispersion [U]
        ctfrs [U]
        same_similarity [U]
        cross_similarity [U]
        between_class_distance [U]
        class_prototypes: class -> [U,T]
    """
    distributions = np.asarray(role_distributions, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    n_videos, n_units, _ = distributions.shape
    classes = sorted(int(v) for v in np.unique(labels_np))
    prototypes: Dict[int, np.ndarray] = {}
    for cls in classes:
        mask = labels_np == cls
        prototype = distributions[mask].mean(axis=0)
        prototype /= np.maximum(prototype.sum(axis=-1, keepdims=True), EPS)
        prototypes[cls] = prototype

    sample_stability = np.zeros((n_videos, n_units), dtype=np.float64)
    within_dispersion = np.zeros(n_units, dtype=np.float64)
    same_similarity = np.full(n_units, np.nan, dtype=np.float64)
    cross_similarity = np.full(n_units, np.nan, dtype=np.float64)
    between_distance = np.zeros(n_units, dtype=np.float64)

    for unit in range(n_units):
        distances: List[float] = []
        same_values: List[float] = []
        cross_values: List[float] = []
        for video_idx in range(n_videos):
            cls = int(labels_np[video_idx])
            distance = wasserstein_1_cdf(distributions[video_idx, unit], prototypes[cls][unit])
            distances.append(distance)
            sample_stability[video_idx, unit] = 1.0 - distance
        for left in range(n_videos):
            for right in range(left + 1, n_videos):
                similarity = role_similarity(distributions[left, unit], distributions[right, unit])
                if labels_np[left] == labels_np[right]:
                    same_values.append(similarity)
                else:
                    cross_values.append(similarity)
        class_pair_distances: List[float] = []
        for c_idx, cls_a in enumerate(classes):
            for cls_b in classes[c_idx + 1:]:
                class_pair_distances.append(
                    wasserstein_1_cdf(prototypes[cls_a][unit], prototypes[cls_b][unit])
                )
        within_dispersion[unit] = float(np.mean(distances)) if distances else 1.0
        same_similarity[unit] = float(np.mean(same_values)) if same_values else np.nan
        cross_similarity[unit] = float(np.mean(cross_values)) if cross_values else np.nan
        between_distance[unit] = float(np.mean(class_pair_distances)) if class_pair_distances else 0.0

    ctfrs = np.clip(1.0 - within_dispersion, 0.0, 1.0)
    return (
        sample_stability,
        within_dispersion,
        ctfrs,
        same_similarity,
        cross_similarity,
        between_distance,
        prototypes,
    )


def summarize_role_prototypes(
    prototypes: Mapping[int, np.ndarray],
    class_counts: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted mean prototype, peak, centroid, and concentration per unit."""
    if not prototypes:
        raise ValueError("No class prototypes were computed")
    first = next(iter(prototypes.values()))
    n_units, time_bins = first.shape
    weighted = np.zeros((n_units, time_bins), dtype=np.float64)
    total_weight = 0.0
    for cls, prototype in prototypes.items():
        weight = float(class_counts.get(int(cls), 1))
        weighted += weight * prototype
        total_weight += weight
    weighted /= max(total_weight, EPS)
    weighted /= np.maximum(weighted.sum(axis=-1, keepdims=True), EPS)
    timeline = np.linspace(0.0, 1.0, time_bins, dtype=np.float64)
    centroid = np.sum(weighted * timeline[None, :], axis=1)
    peak = timeline[np.argmax(weighted, axis=1)]
    entropy = -np.sum(weighted * np.log(weighted + EPS), axis=1) / math.log(max(time_bins, 2))
    concentration = np.clip(1.0 - entropy, 0.0, 1.0)
    return weighted, peak, centroid, concentration


def assign_role_labels(centroid: np.ndarray, concentration: np.ndarray) -> List[str]:
    """Analysis-only labels; the median concentration is not a pruning hyperparameter."""
    threshold = float(np.median(concentration))
    labels: List[str] = []
    for mu, kappa in zip(centroid, concentration):
        if float(kappa) < threshold:
            labels.append("persistent")
        elif float(mu) < 1.0 / 3.0:
            labels.append("early")
        elif float(mu) < 2.0 / 3.0:
            labels.append("key-stage")
        else:
            labels.append("late")
    return labels


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
            target_logits.extend(logits.detach().gather(1, chosen[:, None]).squeeze(1).cpu().tolist())
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

    label_array = np.asarray(labels, dtype=np.int64)
    class_counts = {int(cls): int(np.sum(label_array == cls)) for cls in np.unique(label_array)}
    results: List[LayerProbeResult] = []
    for spec in specs:
        raw_curves = torch.stack(curves_by_layer[spec.name], dim=0)  # [N,U,Tnative]
        reference_bins = max(int(curve.shape[-1]) for curve in curves_by_layer[spec.name])
        raw_curves = resample_time(raw_curves, reference_bins)
        role_distributions = normalize_positive_curve(raw_curves)
        cumulative = role_distributions.cumsum(dim=-1)
        (
            sample_stability,
            within_dispersion,
            ctfrs,
            same_similarity,
            cross_similarity,
            between_distance,
            prototypes,
        ) = compute_ctfrs_metrics(role_distributions.numpy(), labels)
        class_role_gap = between_distance - within_dispersion
        _, peak, centroid, concentration = summarize_role_prototypes(prototypes, class_counts)
        roles = assign_role_labels(centroid, concentration)
        d_abs, d_rel, substitutability, mean_amp, frequency = compute_d_abs_d_rel(
            collector.static_samples[spec.name]
        )
        results.append(LayerProbeResult(
            spec=spec,
            video_ids=list(video_ids),
            labels=list(labels),
            target_logits=list(target_logits),
            predicted_labels=list(predicted_labels),
            signed_curves=raw_curves.numpy(),
            role_distributions=role_distributions.numpy(),
            cumulative_curves=cumulative.numpy(),
            sample_role_stability=sample_stability,
            within_class_dispersion=within_dispersion,
            ctfrs=ctfrs,
            same_class_similarity=same_similarity,
            cross_class_similarity=cross_similarity,
            between_class_distance=between_distance,
            class_role_gap=class_role_gap,
            d_abs=d_abs,
            d_rel=d_rel,
            substitutability=substitutability,
            mean_abs_activation=mean_amp,
            active_frequency=frequency,
            role=roles,
            peak_position=peak,
            temporal_centroid=centroid,
            role_concentration=concentration,
            class_prototypes=prototypes,
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
        order = np.argsort(result.ctfrs)
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
                "ctfrs_group": rank_group,
                "ctfrs": float(result.ctfrs[unit_index]),
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
                "ctfrs": float(result.ctfrs[unit]),
                "within_class_w1_dispersion": float(result.within_class_dispersion[unit]),
                "same_class_similarity": float(result.same_class_similarity[unit]),
                "cross_class_similarity": float(result.cross_class_similarity[unit]),
                "same_minus_cross": float(result.same_class_similarity[unit] - result.cross_class_similarity[unit]),
                "between_class_w1_distance": float(result.between_class_distance[unit]),
                "class_role_gap": float(result.class_role_gap[unit]),
                "d_abs": float(result.d_abs[unit]),
                "d_rel": float(result.d_rel[unit]),
                "substitutability": float(result.substitutability[unit]),
                "mean_abs_activation": float(result.mean_abs_activation[unit]),
                "active_frequency": float(result.active_frequency[unit]),
                "functional_role": result.role[unit],
                "peak_position": float(result.peak_position[unit]),
                "temporal_centroid": float(result.temporal_centroid[unit]),
                "role_concentration": float(result.role_concentration[unit]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "unit_ctfrs_metrics.csv", index=False)
    return frame


def export_curve_arrays(results: Sequence[LayerProbeResult], output_dir: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}
    for index, result in enumerate(results):
        prefix = f"layer_{index:03d}"
        arrays[f"{prefix}_signed"] = result.signed_curves.astype(np.float32)
        arrays[f"{prefix}_role_distribution"] = result.role_distributions.astype(np.float32)
        arrays[f"{prefix}_cumulative"] = result.cumulative_curves.astype(np.float32)
        arrays[f"{prefix}_sample_stability"] = result.sample_role_stability.astype(np.float32)
        for cls, prototype in result.class_prototypes.items():
            arrays[f"{prefix}_prototype_class_{int(cls)}"] = prototype.astype(np.float32)
        metadata[prefix] = {
            "name": result.spec.name,
            "stage": result.spec.stage,
            "block": result.spec.block,
            "unit_type": result.spec.unit_type,
            "num_units": result.spec.num_units,
            "video_ids": result.video_ids,
            "labels": result.labels,
            "prototype_classes": sorted(int(k) for k in result.class_prototypes),
        }
    np.savez_compressed(output_dir / "ctfrs_curve_arrays.npz", **arrays)
    with (output_dir / "ctfrs_curve_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def plot_summary(unit_frame: pd.DataFrame, ablation_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(unit_frame["d_abs"], unit_frame["ctfrs"], s=12, alpha=0.6)
    plt.xlabel("D_abs")
    plt.ylabel("CTFRS")
    plt.title("CTFRS versus response strength")
    plt.tight_layout()
    plt.savefig(output_dir / "ctfrs_vs_d_abs.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(unit_frame["d_rel"], unit_frame["ctfrs"], s=12, alpha=0.6)
    plt.xlabel("D_rel")
    plt.ylabel("CTFRS")
    plt.title("CTFRS versus relative importance")
    plt.tight_layout()
    plt.savefig(output_dir / "ctfrs_vs_d_rel.png", dpi=220)
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
    plt.ylabel("1 - temporal Wasserstein distance")
    plt.title("Within-class versus cross-class role similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "same_vs_cross_class_role_similarity.png", dpi=220)
    plt.close()

    role_counts = unit_frame["functional_role"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(role_counts.index, role_counts.values)
    plt.ylabel("Number of units")
    plt.title("Class-conditioned temporal functional roles")
    plt.tight_layout()
    plt.savefig(output_dir / "functional_role_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(unit_frame["temporal_centroid"], unit_frame["role_concentration"], s=12, alpha=0.6)
    plt.xlabel("Temporal role centroid")
    plt.ylabel("Role concentration (1 - normalized entropy)")
    plt.title("Temporal location and stage concentration")
    plt.tight_layout()
    plt.savefig(output_dir / "role_centroid_vs_concentration.png", dpi=220)
    plt.close()

    if not ablation_frame.empty:
        plt.figure(figsize=(7, 5))
        for group, group_frame in ablation_frame.groupby("ctfrs_group"):
            plt.scatter(group_frame["ctfrs"], group_frame["mean_logit_drop"], s=35, alpha=0.75, label=group)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("CTFRS")
        plt.ylabel("Target-logit drop after masking")
        plt.title("True masking validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ctfrs_vs_true_logit_drop.png", dpi=220)
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
        "metric_definition": "CTFRS = 1 - mean within-class W1 distance to the same unit's class prototype",
        "hypothesis_1_role_diversity": {
            "ctfrs_std": float(unit_frame["ctfrs"].std()),
            "temporal_centroid_std": float(unit_frame["temporal_centroid"].std()),
            "role_concentration_std": float(unit_frame["role_concentration"].std()),
            "supported": bool(unit_frame["temporal_centroid"].std() > 0.02),
        },
        "hypothesis_2_class_conditioned_stability": {
            "same_class_mean": same_mean,
            "cross_class_mean": cross_mean,
            "gap": same_mean - cross_mean,
            "unit_positive_gap_rate": float((unit_frame["same_minus_cross"] > 0).mean()),
            "mean_within_class_w1_dispersion": float(unit_frame["within_class_w1_dispersion"].mean()),
            "mean_between_class_w1_distance": float(unit_frame["between_class_w1_distance"].mean()),
            "mean_class_role_gap": float(unit_frame["class_role_gap"].mean()),
            "supported": bool(same_mean > cross_mean),
        },
        "hypothesis_4_non_redundancy": {
            "spearman_ctfrs_d_abs": spearman(unit_frame["ctfrs"], unit_frame["d_abs"]),
            "spearman_ctfrs_d_rel": spearman(unit_frame["ctfrs"], unit_frame["d_rel"]),
        },
        "functional_role_counts": {
            str(key): int(value) for key, value in unit_frame["functional_role"].value_counts().items()
        },
    }
    correlations = [
        abs(value) for value in summary["hypothesis_4_non_redundancy"].values()
        if np.isfinite(value)
    ]
    summary["hypothesis_4_non_redundancy"]["supported"] = bool(correlations and max(correlations) < 0.90)

    if not ablation_frame.empty:
        high = ablation_frame.loc[ablation_frame["ctfrs_group"] == "high", "mean_logit_drop"]
        low = ablation_frame.loc[ablation_frame["ctfrs_group"] == "low", "mean_logit_drop"]
        corr = spearman(ablation_frame["ctfrs"], ablation_frame["mean_logit_drop"])
        summary["hypothesis_3_causal_masking"] = {
            "high_ctfrs_mean_logit_drop": float(high.mean()) if len(high) else float("nan"),
            "low_ctfrs_mean_logit_drop": float(low.mean()) if len(low) else float("nan"),
            "high_minus_low": float(high.mean() - low.mean()) if len(high) and len(low) else float("nan"),
            "spearman_ctfrs_logit_drop": corr,
            "supported": bool(len(high) and len(low) and high.mean() > low.mean()),
        }
    else:
        summary["hypothesis_3_causal_masking"] = {"skipped": True, "supported": None}
    return summary


def write_report(summary: Mapping[str, Any], output_dir: Path) -> None:
    lines = [
        "CTFRS TEMPORAL FUNCTIONAL ROLE PROBE REPORT",
        "=" * 72,
        "",
        "This probe is diagnostic only. It does not prune or fine-tune the model.",
        "CTFRS uses class-conditioned temporal role prototypes and normalized",
        "one-dimensional Wasserstein-1 distance; it does not use a global evidence curve.",
        "Role labels are analysis-only and do not enter CTFRS or pruning.",
        "",
    ]
    for key in (
        "hypothesis_1_role_diversity",
        "hypothesis_2_class_conditioned_stability",
        "hypothesis_3_causal_masking",
        "hypothesis_4_non_redundancy",
    ):
        lines.append(key)
        lines.append("-" * len(key))
        for metric, value in summary.get(key, {}).items():
            lines.append(f"{metric}: {value}")
        lines.append("")
    lines.extend([
        "Interpretation:",
        "- H1: units occupy different temporal centers/concentrations.",
        "- H2: same-class role similarity exceeds cross-class similarity.",
        "- H3: masking high-CTFRS units causes a larger target-logit drop.",
        "- H4: |rho(CTFRS,D_abs/D_rel)| remains below 0.90.",
        "- class_role_gap is diagnostic only: between-class W1 minus within-class W1.",
    ])
    (output_dir / "CTFRS_PROBE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


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
        "hypothesis_1_role_diversity",
        "hypothesis_2_class_conditioned_stability",
        "hypothesis_3_causal_masking",
        "hypothesis_4_non_redundancy",
    ):
        print(f"{key}: {summary.get(key)}")


if __name__ == "__main__":
    main()

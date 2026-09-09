#!/usr/bin/env python3
"""GPU-first layer-wise probe for video-specific third descriptors.

This Task 009 probe is independent of production pruning.  It reuses the exact
Task 007 unit ordering, checkpoint, and probe videos; captures one target layer
at a time; and obtains gradients for every pruning unit in that layer from one
ground-truth-logit backward pass per probe batch.  No unit-wise backward loop,
raw activation dump, pruning, calibration, or model update is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - server environment normally has tqdm.
    def tqdm(iterable=None, **kwargs):
        del kwargs
        return iterable


EPSILON = 1e-8
SEED = 3407
EXPECTED_UNIT_TYPES = frozenset({"attention_head", "ffn_neuron"})
TASK007_COLUMNS = (
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_st",
    "ablation_logit_deviation",
)
CANDIDATE_FIELDS = (
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_st_old",
    "D_temporal_variation",
    "D_action_dynamic",
    "ablation_logit_deviation",
)
SANITY_FIELDS = (
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_temporal_variation_normal",
    "D_temporal_variation_frozen",
    "D_temporal_variation_shuffle",
    "D_action_dynamic_normal",
    "D_action_dynamic_frozen",
    "D_action_dynamic_shuffle",
)
PROBE_OUTPUT_NAMES = (
    "third_descriptor_candidates.csv",
    "third_descriptor_candidates.npz",
    "third_descriptor_video_sanity.csv",
    "progress.json",
    "run_metadata.json",
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_finite(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Task 007 row {row_number} has invalid {column!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Task 007 row {row_number} column {column!r} is not finite"
        )
    return value


def stage_from_layer(layer: str) -> int:
    match = re.search(r"(?:^|\.)layers\.([0-3])(?:\.|$)", layer)
    if match is None:
        raise ValueError(f"cannot derive Video Swin stage from {layer!r}")
    return int(match.group(1))


def read_task007_rows(path: Path) -> list[dict]:
    """Read the full Task 007 table as an ordered unit vector of length ``N``."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(TASK007_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"Task 007 CSV is empty: {path}")

    rows: list[dict] = []
    for row_number, source in enumerate(source_rows, start=2):
        try:
            global_index = int(source["global_index"])
            unit_index = int(source["unit_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Task 007 row {row_number} has invalid integer unit metadata"
            ) from exc
        layer = str(source["layer"]).strip()
        unit_type = str(source["unit_type"]).strip()
        if global_index < 0 or unit_index < 0 or not layer:
            raise ValueError(f"Task 007 row {row_number} has invalid unit metadata")
        if unit_type not in EXPECTED_UNIT_TYPES:
            raise ValueError(
                f"Task 007 row {row_number} has unsupported type {unit_type!r}"
            )
        rows.append(
            {
                "global_index": global_index,
                "layer": layer,
                "unit_type": unit_type,
                "unit_index": unit_index,
                "stage": stage_from_layer(layer),
                "D_abs": _parse_finite(source, "D_abs", row_number),
                "D_rel": _parse_finite(source, "D_rel", row_number),
                "D_st": _parse_finite(source, "D_st", row_number),
                "ablation_logit_deviation": _parse_finite(
                    source, "ablation_logit_deviation", row_number
                ),
            }
        )
    rows.sort(key=lambda item: item["global_index"])
    validate_unit_ordering(rows)
    return rows


def group_rows_by_layer(rows: Sequence[dict]) -> OrderedDict[str, list[dict]]:
    """Map the ordered unit vector ``[N]`` to contiguous per-layer row lists."""
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    closed: set[str] = set()
    previous = None
    for row in rows:
        layer = str(row["layer"])
        if layer != previous:
            if layer in closed:
                raise ValueError(f"Task 007 layer rows are not contiguous: {layer}")
            if previous is not None:
                closed.add(previous)
            grouped.setdefault(layer, [])
            previous = layer
        grouped[layer].append(row)
    return grouped


def validate_unit_ordering(
    rows: Sequence[dict], model: torch.nn.Module | None = None
) -> OrderedDict[str, list[dict]]:
    """Require exact ``global/layer/type/unit`` ordering, optionally against model."""
    if not rows:
        raise ValueError("unit rows must be non-empty")
    indices = [int(row["global_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("global_index must be unique and contiguous from zero")
    keys = [
        (str(row["layer"]), str(row["unit_type"]), int(row["unit_index"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("(layer, unit_type, unit_index) must be unique")

    grouped = group_rows_by_layer(rows)
    modules = dict(model.named_modules()) if model is not None else {}
    for layer, layer_rows in grouped.items():
        unit_types = {str(row["unit_type"]) for row in layer_rows}
        if len(unit_types) != 1:
            raise ValueError(f"layer {layer} mixes unit types")
        actual = [int(row["unit_index"]) for row in layer_rows]
        if actual != list(range(len(layer_rows))):
            raise ValueError(f"layer {layer} unit_index is not contiguous from zero")
        if model is None:
            continue
        if layer not in modules:
            raise ValueError(f"Task 007 layer is absent from current model: {layer}")
        module = modules[layer]
        unit_type = next(iter(unit_types))
        if unit_type == "attention_head":
            if "WindowAttention3D" not in module.__class__.__name__:
                raise ValueError(f"{layer} is not WindowAttention3D")
            expected_count = int(module.num_heads)
            keep = list(getattr(module, "keep_heads", range(expected_count)))
        else:
            if "Mlp" not in module.__class__.__name__:
                raise ValueError(f"{layer} is not Mlp")
            expected_count = int(module.original_hidden_features)
            keep = list(getattr(module, "keep_neurons", range(expected_count)))
        if len(layer_rows) != expected_count:
            raise ValueError(
                f"{layer} Task 007 count {len(layer_rows)} != model count "
                f"{expected_count}"
            )
        if keep != list(range(expected_count)):
            raise ValueError(
                f"{layer} is already structurally pruned; Task 009 requires the "
                "same unpruned model as Task 007"
            )
    return grouped


def temporal_decomposition(
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose ``A [B,T,H,W,U,E]`` into mean and signed residual.

    Axes are probe video, time, height, width, pruning unit, and per-unit
    feature.  The returned temporal mean is ``A_bar [B,1,H,W,U,E]`` and the
    signed residual is ``Delta_A [B,T,H,W,U,E]``.
    """
    if activation.ndim != 6 or any(size <= 0 for size in activation.shape):
        raise ValueError(
            "activation must have non-empty shape [B,T,H,W,U,E], got "
            f"{tuple(activation.shape)}"
        )
    if not torch.isfinite(activation).all():
        raise ValueError("activation contains NaN or infinity")
    temporal_mean = activation.mean(dim=1, keepdim=True)
    dynamic_residual = activation - temporal_mean
    return temporal_mean, dynamic_residual


def temporal_variation_ratio(
    activation: torch.Tensor, eps: float = EPSILON
) -> torch.Tensor:
    """Return Candidate B ``D_var [B,U]`` without reducing feature axis early."""
    temporal_mean, dynamic_residual = temporal_decomposition(activation)
    # ``S [B,U]`` averages ||A_bar(h,w,:)||_2 over H,W; broadcasting over T
    # in the paper definition cancels the factor T in its denominator.
    static_energy = temporal_mean[:, 0].norm(dim=-1).mean(dim=(1, 2))
    # ``M [B,U]`` averages ||Delta_A(t,h,w,:)||_2 over T,H,W.
    dynamic_energy = dynamic_residual.norm(dim=-1).mean(dim=(1, 2, 3))
    ratio = dynamic_energy / (dynamic_energy + static_energy + eps)
    if not torch.isfinite(ratio).all():
        raise ValueError("Candidate B produced NaN or infinity")
    return ratio.clamp(0.0, 1.0)


def action_dynamic_components(
    activation: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dynamic/static class contributions, each shaped ``[B,U]``.

    ``A`` and ``G`` both have shape ``[B,T,H,W,U,E]``.  Their inner product is
    taken over the complete per-unit feature axis ``E`` before any temporal or
    spatial reduction; for an FFN neuron ``E=1``.
    """
    if activation.shape != gradient.shape:
        raise ValueError(
            f"activation/gradient shapes differ: {activation.shape} vs "
            f"{gradient.shape}"
        )
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient contains NaN or infinity")
    temporal_mean, dynamic_residual = temporal_decomposition(activation)
    dynamic_inner = (gradient * dynamic_residual).sum(dim=-1)
    static_inner = (gradient * temporal_mean).sum(dim=-1)
    dynamic_contribution = dynamic_inner.abs().mean(dim=(1, 2, 3))
    static_contribution = static_inner.abs().mean(dim=(1, 2, 3))
    return dynamic_contribution, static_contribution


def action_dynamic_ratio(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Return Candidate C ``D_adc [B,U]`` for ground-truth class gradients."""
    dynamic, static = action_dynamic_components(activation, gradient)
    ratio = dynamic / (dynamic + static + eps)
    if not torch.isfinite(ratio).all():
        raise ValueError("Candidate C produced NaN or infinity")
    return ratio.clamp(0.0, 1.0)


def ground_truth_logit_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return scalar ``Z=sum_n z[n,y_n]`` for ``logits [B,C]``."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B,C], got {tuple(logits.shape)}")
    target = targets.reshape(-1).long()
    if target.numel() != logits.shape[0]:
        raise ValueError("target count differs from logit batch size")
    if int(target.min()) < 0 or int(target.max()) >= logits.shape[1]:
        raise ValueError("ground-truth target is outside the model logit range")
    return logits.gather(1, target[:, None]).sum()


def restore_attention_volume(
    window_tensor: torch.Tensor,
    attention_module: torch.nn.Module,
) -> torch.Tensor:
    """Restore signed head responses ``[B,T,H,W,U,E]`` from window tokens.

    The input ``X [B*nW,N,U*E]`` is the exact pre-``proj`` tensor.  ``U`` is
    the attention-head axis and ``E`` is head embedding dimension; neither is
    interpreted as time.
    """
    from MC import window_reverse

    if window_tensor.ndim != 3:
        raise ValueError(
            f"attention tensor must be [B*nW,N,U*E], got {window_tensor.shape}"
        )
    geometry = getattr(attention_module, "_pruning_geometry", None)
    if geometry is None:
        raise RuntimeError("attention module lacks current forward geometry")
    batch = int(geometry["batch_size"])
    depth = int(geometry["depth"])
    height = int(geometry["height"])
    width = int(geometry["width"])
    padded_depth = int(geometry["padded_depth"])
    padded_height = int(geometry["padded_height"])
    padded_width = int(geometry["padded_width"])
    window = tuple(int(value) for value in geometry["window_size"])
    shift = tuple(int(value) for value in geometry["shift_size"])
    units = int(attention_module.num_heads)
    feature = int(attention_module.head_dim)
    tokens = math.prod(window)
    windows = (
        batch
        * (padded_depth // window[0])
        * (padded_height // window[1])
        * (padded_width // window[2])
    )
    expected = (windows, tokens, units * feature)
    if tuple(window_tensor.shape) != expected:
        raise ValueError(
            f"attention window shape {tuple(window_tensor.shape)} != {expected}"
        )

    local = window_tensor.reshape(windows, *window, units * feature)
    restored = window_reverse(
        local,
        window,
        batch,
        padded_depth,
        padded_height,
        padded_width,
    )
    restored = restored.reshape(
        batch, padded_depth, padded_height, padded_width, units, feature
    )
    if any(value > 0 for value in shift):
        restored = torch.roll(restored, shifts=shift, dims=(1, 2, 3))
    return restored[:, :depth, :height, :width].contiguous()


def restore_mlp_volume(hidden: torch.Tensor) -> torch.Tensor:
    """Restore FFN response ``[B,T,H,W,U,1]`` from pre-``fc2`` hidden input."""
    if hidden.ndim != 5 or any(size <= 0 for size in hidden.shape):
        raise ValueError(
            f"FFN hidden tensor must be [B,T,H,W,U], got {tuple(hidden.shape)}"
        )
    return hidden.unsqueeze(-1)


class LayerActivationCapture:
    """Detach one target layer and retain its exact pre-mixing tensor gradient."""

    def __init__(self, layer_module: torch.nn.Module, unit_type: str):
        if unit_type == "attention_head":
            target = layer_module.proj
        elif unit_type == "ffn_neuron":
            target = layer_module.fc2
        else:
            raise ValueError(f"unsupported unit type {unit_type!r}")
        self.layer_module = layer_module
        self.unit_type = unit_type
        self.activation: torch.Tensor | None = None
        self.handle = target.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        if self.activation is not None:
            raise RuntimeError("target layer executed more than once in one forward")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("target pre-hook did not receive a tensor")
        # All learned parameters are frozen.  Detaching here starts the graph at
        # ``A [..]`` and avoids retaining upstream Video Swin activations.
        captured = inputs[0].detach().requires_grad_(True)
        self.activation = captured
        return (captured, *inputs[1:])

    def reset(self) -> None:
        self.activation = None

    def volumes(self) -> tuple[torch.Tensor, torch.Tensor]:
        activation = self.activation
        if activation is None or activation.grad is None:
            raise RuntimeError("captured activation or its gradient is missing")
        if self.unit_type == "attention_head":
            response = restore_attention_volume(activation, self.layer_module)
            gradient = restore_attention_volume(activation.grad, self.layer_module)
        else:
            response = restore_mlp_volume(activation)
            gradient = restore_mlp_volume(activation.grad)
        return response, gradient

    def remove(self) -> None:
        self.handle.remove()


def transform_video_condition(
    videos: torch.Tensor,
    condition: str,
    shuffle_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform ``X [B,C,T,H,W]`` into normal, frozen, or shuffled video."""
    if videos.ndim != 5:
        raise ValueError(f"videos must be [B,C,T,H,W], got {videos.shape}")
    if condition == "normal":
        return videos
    if condition == "frozen":
        center = videos.shape[2] // 2
        return videos[:, :, center : center + 1].expand_as(videos).contiguous()
    if condition == "shuffle":
        if shuffle_index is None or shuffle_index.numel() != videos.shape[2]:
            raise ValueError("shuffle permutation does not match temporal length")
        return videos.index_select(2, shuffle_index.to(videos.device))
    raise ValueError(f"unknown video condition {condition!r}")


def _extract_videos_targets(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("probe loader must return (videos, targets, ...)")
    videos, targets = batch[0], batch[1]
    if not isinstance(videos, torch.Tensor) or videos.ndim != 5:
        raise ValueError("probe videos must have shape [B,C,T,H,W]")
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)
    return videos, targets.reshape(-1).long()


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    elif isinstance(output, dict):
        logits = output.get("logits", output.get("cls_score"))
    else:
        logits = None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("model output must contain logits [B,C]")
    return logits


def collect_layer_condition(
    model: torch.nn.Module,
    capture: LayerActivationCapture,
    loader: DataLoader,
    device: torch.device,
    condition: str,
    expected_labels: Sequence[int],
    expected_units: int,
    shuffle_index: torch.Tensor | None = None,
    secondary_limit: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Accumulate per-video ratios into two CPU vectors of shape ``[U]``."""
    variation_sum = torch.zeros(expected_units, dtype=torch.float64)
    adc_sum = torch.zeros(expected_units, dtype=torch.float64)
    secondary_var = torch.zeros(expected_units, dtype=torch.float64)
    secondary_adc = torch.zeros(expected_units, dtype=torch.float64)
    count = 0
    secondary_count = 0
    model.eval()

    progress = tqdm(loader, desc=f"{condition} batches", ncols=0, leave=False)
    for batch in progress:
        videos, targets = _extract_videos_targets(batch)
        batch_size = int(videos.shape[0])
        expected = list(expected_labels[count : count + batch_size])
        if targets.cpu().tolist() != expected:
            raise ValueError(
                "dataset targets do not match Task 007 probe_samples.csv order"
            )
        videos = videos.float().to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        videos = transform_video_condition(videos, condition, shuffle_index)

        capture.reset()
        logits = _extract_logits(model(videos))
        objective = ground_truth_logit_sum(logits, targets)
        objective.backward()
        response, gradient = capture.volumes()
        variation = temporal_variation_ratio(response)  # [B,U]
        adc = action_dynamic_ratio(response, gradient)  # [B,U]
        if variation.shape != (batch_size, expected_units):
            raise RuntimeError(
                f"Candidate B shape {variation.shape} != "
                f"({batch_size},{expected_units})"
            )
        if adc.shape != variation.shape:
            raise RuntimeError("Candidate B/C batch shapes differ")

        variation_cpu = variation.detach().double().cpu()
        adc_cpu = adc.detach().double().cpu()
        variation_sum += variation_cpu.sum(dim=0)
        adc_sum += adc_cpu.sum(dim=0)
        if secondary_limit > secondary_count:
            take = min(batch_size, secondary_limit - secondary_count)
            secondary_var += variation_cpu[:take].sum(dim=0)
            secondary_adc += adc_cpu[:take].sum(dim=0)
            secondary_count += take
        count += batch_size

        capture.reset()
        del videos, targets, logits, objective, response, gradient, variation, adc

    if count != len(expected_labels):
        raise RuntimeError(f"processed {count} videos, expected {len(expected_labels)}")
    primary = {
        "variation": variation_sum / count,
        "adc": adc_sum / count,
    }
    secondary = None
    if secondary_limit:
        if secondary_count != secondary_limit:
            raise RuntimeError(
                f"sanity subset processed {secondary_count}, expected {secondary_limit}"
            )
        secondary = {
            "variation": secondary_var / secondary_count,
            "adc": secondary_adc / secondary_count,
        }
    return primary, secondary


def read_probe_samples(path: Path) -> list[dict]:
    required = {
        "sample_index", "class_id", "class_name", "video_identifier"
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"probe sample file is empty: {path}")
    result = []
    for row in rows:
        result.append(
            {
                "sample_index": int(row["sample_index"]),
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "video_identifier": str(row["video_identifier"]),
            }
        )
    if len({row["sample_index"] for row in result}) != len(result):
        raise ValueError("probe_samples.csv contains duplicate sample_index")
    return result


def validate_probe_samples_against_split(
    sample_rows: Sequence[dict], split_rows: Sequence[dict]
) -> None:
    current = {int(row["sample_index"]): row for row in split_rows}
    for sample in sample_rows:
        split = current.get(int(sample["sample_index"]))
        if split is None:
            raise ValueError("probe sample index is absent from validation split")
        if (
            int(split["class_id"]) != int(sample["class_id"])
            or str(split["video_identifier"]) != str(sample["video_identifier"])
        ):
            raise ValueError(
                "probe_samples.csv does not match the current validation split"
            )


def _load_task007_helpers():
    import probe_descriptor_ablation_consistency as task007

    return task007


def _make_loader(
    task007,
    base_loader: DataLoader,
    sample_indices: Sequence[int],
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    task007.set_seed(seed)
    return task007.make_probe_loader(
        base_loader,
        list(sample_indices),
        batch_size,
        workers,
        seed,
    )


def _run_fingerprint(
    task007_sha: str,
    checkpoint_sha: str,
    sample_sha: str,
    split_sha: str,
    seed: int,
    sanity_videos: int,
    device_name: str,
) -> str:
    payload = {
        "task007_csv_sha256": task007_sha,
        "checkpoint_sha256": checkpoint_sha,
        "probe_samples_sha256": sample_sha,
        "validation_split_sha256": split_sha,
        "seed": int(seed),
        "sanity_videos": int(sanity_videos),
        "device_name": device_name,
        "candidate_version": "task009_adc_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _read_csv_if_present(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_resume_state(
    progress: dict | None,
    run_fingerprint: str,
    task007_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    sanity_rows: Sequence[dict],
) -> set[str]:
    """Validate complete layer checkpoints and return completed layer names."""
    if progress is None:
        if candidate_rows or sanity_rows:
            raise ValueError("partial CSV exists without progress.json")
        return set()
    if progress.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "progress.json does not match checkpoint/sample/task007 metadata; "
            "use --overwrite only for an intentional new run"
        )
    completed = set(str(name) for name in progress.get("completed_layers", []))
    known_layers = {str(row["layer"]) for row in task007_rows}
    if not completed.issubset(known_layers):
        raise ValueError("progress.json contains an unknown completed layer")
    expected_by_key = {
        (
            int(row["global_index"]),
            str(row["layer"]),
            str(row["unit_type"]),
            int(row["unit_index"]),
        ): row
        for row in task007_rows
    }

    def validate_saved(rows: Sequence[dict], value_columns: Sequence[str]) -> set:
        keys = set()
        for row in rows:
            key = (
                int(row["global_index"]),
                str(row["layer"]),
                str(row["unit_type"]),
                int(row["unit_index"]),
            )
            if key not in expected_by_key or key in keys:
                raise ValueError("resume CSV contains duplicate or unknown unit")
            for column in value_columns:
                value = float(row[column])
                if not math.isfinite(value) or value < -1e-6 or value > 1.0 + 1e-6:
                    raise ValueError(f"resume value {column} is outside [0,1]")
            keys.add(key)
        return keys

    candidate_keys = validate_saved(
        candidate_rows, ("D_temporal_variation", "D_action_dynamic")
    )
    sanity_keys = validate_saved(
        sanity_rows,
        (
            "D_temporal_variation_normal",
            "D_temporal_variation_frozen",
            "D_temporal_variation_shuffle",
            "D_action_dynamic_normal",
            "D_action_dynamic_frozen",
            "D_action_dynamic_shuffle",
        ),
    )
    if candidate_keys != sanity_keys:
        raise ValueError("candidate and sanity resume CSV unit sets differ")
    expected_by_global = {
        int(row["global_index"]): row for row in task007_rows
    }
    for row in candidate_rows:
        expected = expected_by_global[int(row["global_index"])]
        comparisons = (
            ("D_abs", "D_abs"),
            ("D_rel", "D_rel"),
            ("D_st_old", "D_st"),
            ("ablation_logit_deviation", "ablation_logit_deviation"),
        )
        for saved_column, source_column in comparisons:
            saved = float(row[saved_column])
            source = float(expected[source_column])
            if abs(saved - source) > 1e-7 * max(1.0, abs(saved), abs(source)):
                raise ValueError(
                    f"resume CSV {saved_column} differs from Task 007 at "
                    f"global_index={row['global_index']}"
                )
    expected_keys = {
        key for key in expected_by_key if key[1] in completed
    }
    if candidate_keys != expected_keys:
        raise ValueError("resume CSV rows do not exactly cover completed layers")
    return completed


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in PROBE_OUTPUT_NAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()


def _save_final_npz(path: Path, rows: Sequence[dict]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["global_index"]))
    np.savez(
        path,
        D_abs=np.asarray([float(row["D_abs"]) for row in ordered], dtype=np.float32),
        D_rel=np.asarray([float(row["D_rel"]) for row in ordered], dtype=np.float32),
        D_old=np.asarray(
            [float(row["D_st_old"]) for row in ordered], dtype=np.float32
        ),
        D_var=np.asarray(
            [float(row["D_temporal_variation"]) for row in ordered],
            dtype=np.float32,
        ),
        D_adc=np.asarray(
            [float(row["D_action_dynamic"]) for row in ordered],
            dtype=np.float32,
        ),
        ablation_effect=np.asarray(
            [float(row["ablation_logit_deviation"]) for row in ordered],
            dtype=np.float32,
        ),
        global_index=np.asarray(
            [int(row["global_index"]) for row in ordered], dtype=np.int64
        ),
    )


def run_probe(args: argparse.Namespace) -> None:
    visible_gpus = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Task 009 probe requires CUDA; the default GPU-first path cannot "
            "run the Video Swin model on CPU"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    task007 = _load_task007_helpers()
    task007.set_seed(args.seed)

    task007_csv = Path(args.task007_csv)
    probe_samples_path = Path(args.probe_samples)
    checkpoint_path = Path(args.checkpoint_path)
    validation_split = Path(args.val_split)
    for required in (
        task007_csv, probe_samples_path, checkpoint_path, validation_split
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    task007_rows = read_task007_rows(task007_csv)
    sample_rows = read_probe_samples(probe_samples_path)
    expected_count = args.num_classes * args.videos_per_class
    if len(sample_rows) != expected_count:
        raise ValueError(
            f"probe_samples.csv has {len(sample_rows)} rows, expected {expected_count}"
        )
    class_counts = Counter(int(row["class_id"]) for row in sample_rows)
    if (
        len(class_counts) != args.num_classes
        or set(class_counts.values()) != {args.videos_per_class}
    ):
        raise ValueError("probe sample class balance differs from Task 007 protocol")
    split_rows = task007.parse_split_records(validation_split)
    validate_probe_samples_against_split(sample_rows, split_rows)

    get_dataset, _, SwinTransformer3D = task007._load_repository_components()
    model = task007._build_model(SwinTransformer3D, checkpoint_path, device)
    layer_rows = validate_unit_ordering(task007_rows, model=model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    output_dir = Path(args.output_dir)
    _prepare_output(output_dir, args.overwrite)
    candidate_path = output_dir / "third_descriptor_candidates.csv"
    sanity_path = output_dir / "third_descriptor_video_sanity.csv"
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "run_metadata.json"

    task007_sha = _sha256_file(task007_csv)
    checkpoint_sha = _sha256_file(checkpoint_path)
    sample_sha = _sha256_file(probe_samples_path)
    split_sha = _sha256_file(validation_split)
    device_name = torch.cuda.get_device_name(0)
    sanity_count = min(args.sanity_videos, len(sample_rows))
    run_fingerprint = _run_fingerprint(
        task007_sha,
        checkpoint_sha,
        sample_sha,
        split_sha,
        args.seed,
        sanity_count,
        device_name,
    )

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else None
    )
    existing_candidates = _read_csv_if_present(candidate_path)
    existing_sanity = _read_csv_if_present(sanity_path)
    completed_layers = validate_resume_state(
        progress,
        run_fingerprint,
        task007_rows,
        existing_candidates,
        existing_sanity,
    )
    candidate_by_global = {
        int(row["global_index"]): row for row in existing_candidates
    }
    sanity_by_global = {
        int(row["global_index"]): row for row in existing_sanity
    }

    sample_indices = [int(row["sample_index"]) for row in sample_rows]
    expected_labels = [int(row["class_id"]) for row in sample_rows]
    sanity_indices = sample_indices[:sanity_count]
    sanity_labels = expected_labels[:sanity_count]
    base_loader = get_dataset(args.val_split, 1)

    def new_loader(indices: Sequence[int]) -> DataLoader:
        return _make_loader(
            task007,
            base_loader,
            indices,
            args.probe_batch_size,
            args.workers,
            args.seed,
        )

    first_batch = next(iter(new_loader(sample_indices)))
    first_videos, _ = _extract_videos_targets(first_batch)
    temporal_length = int(first_videos.shape[2])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    shuffle_index = torch.randperm(temporal_length, generator=generator)
    del first_batch, first_videos

    parameter_hash_before = task007.hash_model_parameters(model)
    metadata = {
        "status": "in_progress",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "task007_csv": str(task007_csv.resolve()),
        "task007_csv_sha256": task007_sha,
        "probe_samples": str(probe_samples_path.resolve()),
        "sample_list_hash": sample_sha,
        "validation_split": str(validation_split.resolve()),
        "validation_split_sha256": split_sha,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "execution_device": "cuda:0",
        "gpu_first": True,
        "probe_video_count": len(sample_rows),
        "sanity_video_count": sanity_count,
        "shuffle_temporal_index": shuffle_index.tolist(),
        "command_line": [sys.executable, *sys.argv],
        "run_fingerprint": run_fingerprint,
        "model_parameter_sha256_before": parameter_hash_before,
        "gradient_strategy": (
            "one backward per target layer per probe batch; all units in the "
            "layer are accumulated simultaneously"
        ),
        "precision": "torch.float32 (autocast disabled)",
    }
    if metadata_path.exists() and not args.overwrite:
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("run_fingerprint") != run_fingerprint:
            raise ValueError("existing run_metadata.json belongs to another run")
    _atomic_write_json(metadata_path, metadata)

    modules = dict(model.named_modules())
    peak_memory: dict[str, float] = dict(
        (progress or {}).get("peak_allocated_mib_by_layer", {})
    )
    ordered_layers = list(layer_rows)
    for position, (layer_name, rows) in enumerate(layer_rows.items(), start=1):
        if layer_name in completed_layers:
            print(
                f"Layer {position}/{len(ordered_layers)}: {layer_name} complete; skip"
            )
            continue
        unit_type = str(rows[0]["unit_type"])
        unit_count = len(rows)
        module = modules[layer_name]
        print()
        print(f"Layer {position}/{len(ordered_layers)}: {layer_name}")
        print(f"Type: {unit_type}; units: {unit_count}; device: cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        capture = LayerActivationCapture(module, unit_type)
        try:
            normal, normal_sanity = collect_layer_condition(
                model,
                capture,
                new_loader(sample_indices),
                device,
                "normal",
                expected_labels,
                unit_count,
                shuffle_index=shuffle_index,
                secondary_limit=sanity_count,
            )
            frozen, _ = collect_layer_condition(
                model,
                capture,
                new_loader(sanity_indices),
                device,
                "frozen",
                sanity_labels,
                unit_count,
                shuffle_index=shuffle_index,
            )
            shuffled, _ = collect_layer_condition(
                model,
                capture,
                new_loader(sanity_indices),
                device,
                "shuffle",
                sanity_labels,
                unit_count,
                shuffle_index=shuffle_index,
            )
        finally:
            capture.remove()
        if normal_sanity is None:
            raise RuntimeError("normal sanity subset was not accumulated")

        for local_index, task007_row in enumerate(rows):
            global_index = int(task007_row["global_index"])
            candidate_by_global[global_index] = {
                "global_index": global_index,
                "layer": layer_name,
                "unit_type": unit_type,
                "unit_index": int(task007_row["unit_index"]),
                "D_abs": float(task007_row["D_abs"]),
                "D_rel": float(task007_row["D_rel"]),
                "D_st_old": float(task007_row["D_st"]),
                "D_temporal_variation": float(
                    normal["variation"][local_index]
                ),
                "D_action_dynamic": float(normal["adc"][local_index]),
                "ablation_logit_deviation": float(
                    task007_row["ablation_logit_deviation"]
                ),
            }
            sanity_by_global[global_index] = {
                "global_index": global_index,
                "layer": layer_name,
                "unit_type": unit_type,
                "unit_index": int(task007_row["unit_index"]),
                "D_temporal_variation_normal": float(
                    normal_sanity["variation"][local_index]
                ),
                "D_temporal_variation_frozen": float(
                    frozen["variation"][local_index]
                ),
                "D_temporal_variation_shuffle": float(
                    shuffled["variation"][local_index]
                ),
                "D_action_dynamic_normal": float(
                    normal_sanity["adc"][local_index]
                ),
                "D_action_dynamic_frozen": float(
                    frozen["adc"][local_index]
                ),
                "D_action_dynamic_shuffle": float(
                    shuffled["adc"][local_index]
                ),
            }

        completed_layers.add(layer_name)
        torch.cuda.synchronize(device)
        peak_memory[layer_name] = torch.cuda.max_memory_allocated(device) / (1024**2)
        _atomic_write_csv(
            candidate_path,
            CANDIDATE_FIELDS,
            (
                candidate_by_global[index]
                for index in sorted(candidate_by_global)
            ),
        )
        _atomic_write_csv(
            sanity_path,
            SANITY_FIELDS,
            (sanity_by_global[index] for index in sorted(sanity_by_global)),
        )
        _atomic_write_json(
            progress_path,
            {
                "run_fingerprint": run_fingerprint,
                "completed_layers": [
                    name for name in ordered_layers if name in completed_layers
                ],
                "completed_units": len(candidate_by_global),
                "total_units": len(task007_rows),
                "peak_allocated_mib_by_layer": peak_memory,
            },
        )
        print(
            f"Saved layer; completed units {len(candidate_by_global)}/"
            f"{len(task007_rows)}; peak {peak_memory[layer_name]:.1f} MiB"
        )
        del normal, normal_sanity, frozen, shuffled
        torch.cuda.empty_cache()

    final_candidates = [
        candidate_by_global[index] for index in range(len(task007_rows))
    ]
    final_sanity = [sanity_by_global[index] for index in range(len(task007_rows))]
    assert len(final_candidates) == len(task007_rows)
    assert len(final_sanity) == len(task007_rows)
    for row in final_candidates:
        for column in ("D_temporal_variation", "D_action_dynamic"):
            value = float(row[column])
            if not math.isfinite(value) or not -1e-6 <= value <= 1.0 + 1e-6:
                raise AssertionError(f"final {column} is invalid")
    _save_final_npz(output_dir / "third_descriptor_candidates.npz", final_candidates)

    parameter_hash_after = task007.hash_model_parameters(model)
    if parameter_hash_after != parameter_hash_before:
        raise AssertionError("model parameters changed during Task 009 probe")
    metadata.update(
        {
            "status": "complete",
            "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "units": len(final_candidates),
            "completed_layers": ordered_layers,
            "peak_allocated_mib_by_layer": peak_memory,
            "model_parameter_sha256_after": parameter_hash_after,
            "parameter_state_verified_unchanged": True,
        }
    )
    _atomic_write_json(metadata_path, metadata)
    print("=" * 72)
    print("Task 009 video-specific descriptor probe complete")
    print(f"Units: {len(final_candidates)}")
    print(f"Layers: {len(ordered_layers)}")
    print(f"GPU: {device_name}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 72)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-first Task 009 video-specific descriptor probe"
    )
    parser.add_argument("--gpu", default=None, help="visible physical GPU IDs")
    parser.add_argument(
        "--task007_csv",
        default="descriptor_ablation_validation/unit_ablation_effect.csv",
    )
    parser.add_argument(
        "--probe_samples",
        default="descriptor_ablation_validation/probe_samples.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt",
    )
    parser.add_argument(
        "--val_split",
        default="/data/jixinye25/UCF101_Frame/val_rgb_split1.txt",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--probe_batch_size", type=int, default=1)
    parser.add_argument(
        "--sanity_videos",
        type=int,
        default=6,
        help="runtime-only subset size for normal/frozen/shuffled diagnostics",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output_dir", default="video_descriptor_validation"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    positive = {
        "num_classes": args.num_classes,
        "videos_per_class": args.videos_per_class,
        "probe_batch_size": args.probe_batch_size,
        "sanity_videos": args.sanity_videos,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.workers < 0:
        parser.error(f"invalid non-positive arguments: {invalid}")
    return args


if __name__ == "__main__":
    run_probe(parse_args())

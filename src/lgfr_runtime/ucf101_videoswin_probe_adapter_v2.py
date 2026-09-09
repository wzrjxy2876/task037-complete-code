"""Project adapter for paired cluster prototype CAM validation.

Environment overrides:
  UCF101_FRAME_ROOT, UCF101_TRAIN_LIST, UCF101_VAL_LIST, VIDEOSWIN_NUM_CLASSES.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

DEFAULT_TRAIN_LIST = "/data/jixinye25/UCF101_Frame/train_rgb_split1.txt"
DEFAULT_VAL_LIST = "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"


def _normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    prefixes = ("module.", "backbone.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _load_unpruned_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    device: torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Probe checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a mapping")
    normalized = {
        _normalize_checkpoint_key(key): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    model_state = model.state_dict()
    compatible = {}
    non_head_shape_mismatches = []
    unexpected = []
    for key, value in normalized.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(model_state[key].shape):
            if not key.startswith("cls_head."):
                non_head_shape_mismatches.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
        else:
            compatible[key] = value
    if non_head_shape_mismatches:
        preview = non_head_shape_mismatches[:8]
        raise ValueError(
            "Checkpoint is not shape-compatible with the unpruned Video Swin "
            f"probe model. First mismatches: {preview}"
        )
    result = model.load_state_dict(compatible, strict=False)
    non_head_missing = [
        key for key in result.missing_keys if not key.startswith("cls_head.")
    ]
    if non_head_missing:
        raise ValueError(
            "Checkpoint is missing non-classifier tensors required for strict "
            f"response validation. First missing keys: {non_head_missing[:12]}"
        )
    return {
        "loaded_tensor_count": len(compatible),
        "classifier_missing": [
            key for key in result.missing_keys if key.startswith("cls_head.")
        ],
        "unexpected_count": len(unexpected) + len(result.unexpected_keys),
    }


def build_model_for_probe(
    checkpoint: str,
    device: torch.device | str,
    **_: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the same unpruned Video Swin topology used for formal analysis."""
    from MC import SwinTransformer3D

    device = torch.device(device)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        drop_path_rate=0.2,
        num_classes=int(os.environ.get("VIDEOSWIN_NUM_CLASSES", "400")),
    ).to(device)
    metadata = _load_unpruned_checkpoint(model, checkpoint, device)
    model.eval()
    return model, metadata


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_probe_loader(
    videos: int = 20,
    batch_size: int = 1,
    split: str = "val",
    seed: int = 3407,
    num_workers: int = 4,
    **_: Any,
) -> DataLoader:
    """Build a fixed-order probe subset so both cluster paths see the same clips."""
    from dataset import ucf101 as ucf101_dataset

    split_name = str(split).strip().lower()
    if split_name not in {"train", "val", "validation", "test"}:
        raise ValueError(f"Unsupported UCF101 probe split: {split!r}")
    if os.environ.get("UCF101_FRAME_ROOT"):
        ucf101_dataset.UCF_DATA_ROOT = os.environ["UCF101_FRAME_ROOT"]
    if split_name == "train":
        setting_path = os.environ.get("UCF101_TRAIN_LIST", DEFAULT_TRAIN_LIST)
    else:
        setting_path = os.environ.get("UCF101_VAL_LIST", DEFAULT_VAL_LIST)
    spatial_transform, temporal_transform = ucf101_dataset.test_transform()
    dataset = ucf101_dataset.attack_ucf101(
        setting_path,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
    )
    if len(dataset) < int(videos):
        raise ValueError(
            f"UCF101 split contains {len(dataset)} videos, but {videos} were requested"
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    subset_indices = torch.randperm(len(dataset), generator=generator)[: int(videos)]
    subset = Subset(dataset, subset_indices.tolist())
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )

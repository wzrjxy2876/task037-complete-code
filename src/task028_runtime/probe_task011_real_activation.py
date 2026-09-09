#!/usr/bin/env python3
"""Minimal one-video, two-layer Task011 real-activation trace on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

import probe_video_specific_third_descriptor as task009
from temporal_dynamicity import compute_temporal_dynamicity


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class PreInputCapture:
    """Retain one detached pre-module input for a single forward pass."""

    def __init__(self, module: torch.nn.Module):
        self.value: torch.Tensor | None = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        if self.value is not None:
            raise RuntimeError("target module executed more than once")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("target pre-hook did not receive a tensor")
        self.value = inputs[0].detach()

    def remove(self) -> None:
        self.handle.remove()


def _components(response: torch.Tensor) -> dict:
    """Summarize ``A [B,U,T,H,W,D]`` without saving the response tensor."""
    temporal_mean = response.float().mean(dim=2, keepdim=True)
    residual = response.float() - temporal_mean
    dynamic = residual.norm(p=2, dim=-1).mean(dim=(2, 3, 4))
    stable = temporal_mean.norm(p=2, dim=-1).mean(dim=(2, 3, 4))
    final = dynamic / (dynamic + stable + 1e-8)
    batch, units, temporal, height, width, feature = response.shape
    return {
        "restored_shape": [int(value) for value in response.shape],
        "B": int(batch),
        "U": int(units),
        "T": int(temporal),
        "H": int(height),
        "W": int(width),
        "D": int(feature),
        "temporal_mean_checksum": float(temporal_mean.double().sum().item()),
        "dynamic_energy_checksum": float(dynamic.double().sum().item()),
        "stable_energy_checksum": float(stable.double().sum().item()),
        "final_tdd_checksum": float(final.double().sum().item()),
    }


def run_trace(args: argparse.Namespace) -> None:
    visible = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if not torch.cuda.is_available():
        raise RuntimeError("Task011 real-activation trace requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    task007 = task009._load_task007_helpers()
    task007.set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    sample_path = Path(args.probe_samples)
    validation_split = Path(args.val_split)
    for required in (checkpoint_path, sample_path, validation_split):
        if not required.is_file():
            raise FileNotFoundError(required)

    sample_rows = task009.read_probe_samples(sample_path)
    if not sample_rows:
        raise ValueError("probe sample table is empty")
    sample = sample_rows[0]
    get_dataset, InteractionPruner, SwinTransformer3D = (
        task007._load_repository_components()
    )
    model = task007._build_model(SwinTransformer3D, checkpoint_path, device)
    model.eval()
    modules = dict(model.named_modules())
    for layer_name in (args.attention_layer, args.mlp_layer):
        if layer_name not in modules:
            raise KeyError(f"model does not contain requested layer {layer_name!r}")
    attention = modules[args.attention_layer]
    mlp = modules[args.mlp_layer]
    if "WindowAttention3D" not in attention.__class__.__name__:
        raise TypeError(f"{args.attention_layer} is not an Attention module")
    if "Mlp" not in mlp.__class__.__name__:
        raise TypeError(f"{args.mlp_layer} is not an FFN module")

    base_loader = get_dataset(args.val_split, 1)
    loader = task009._make_loader(
        task007,
        base_loader,
        [int(sample["sample_index"])],
        1,
        args.workers,
        args.seed,
    )
    batch = next(iter(loader))
    videos, targets = task009._extract_videos_targets(batch)
    if int(targets[0]) != int(sample["class_id"]):
        raise ValueError("selected sample label differs from probe_samples.csv")
    videos = videos.float().to(device, non_blocking=True)

    attention_capture = PreInputCapture(attention.proj)
    mlp_capture = PreInputCapture(mlp.fc2)
    try:
        with torch.no_grad():
            model(videos)
    finally:
        attention_capture.remove()
        mlp_capture.remove()
    if attention_capture.value is None or mlp_capture.value is None:
        raise RuntimeError("real-activation hooks did not capture both target layers")

    raw_attention = attention_capture.value
    raw_mlp = mlp_capture.value
    attention009 = task009.restore_attention_volume(raw_attention, attention)
    mlp009 = task009.restore_mlp_volume(raw_mlp)
    # Independently invoke Task010's production restoration helpers.  Task009
    # layout is [B,T,H,W,U,D], whereas Task010 is [B,U,T,H,W,D].
    pruner = InteractionPruner(model, descriptor_variant="dynamic3d")
    attention010 = pruner._attention_signed_response_volume(
        raw_attention, attention, attention.num_heads, attention.head_dim
    )
    mlp010 = pruner._mlp_signed_response_volume(raw_mlp)
    attention009_in_010_layout = attention009.permute(
        0, 4, 1, 2, 3, 5
    ).contiguous()
    mlp009_in_010_layout = mlp009.permute(0, 4, 1, 2, 3, 5).contiguous()
    attention_restoration_difference = float(
        (attention009_in_010_layout - attention010).abs().max().item()
    )
    mlp_restoration_difference = float(
        (mlp009_in_010_layout - mlp010).abs().max().item()
    )

    task009_attention_score = task009.temporal_variation_ratio(attention009)
    task009_mlp_score = task009.temporal_variation_ratio(mlp009)
    task010_attention_score = compute_temporal_dynamicity(attention010)
    task010_mlp_score = compute_temporal_dynamicity(mlp010)
    attention_difference = float(
        (task009_attention_score - task010_attention_score).abs().max().item()
    )
    mlp_difference = float(
        (task009_mlp_score - task010_mlp_score).abs().max().item()
    )

    output = {
        "status": "complete",
        "execution_device": "cuda:0",
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "single_gpu_reason": (
            "One forward pass with replica-local window geometry is intentionally "
            "kept on cuda:0; dual GPUs are used by controlled validation."
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": task009._sha256_file(checkpoint_path),
        "probe_samples": str(sample_path.resolve()),
        "sample": {
            "sample_index": int(sample["sample_index"]),
            "class_id": int(sample["class_id"]),
            "video_identifier": str(sample.get("video_identifier", "")),
        },
        "input_shape": [int(value) for value in videos.shape],
        "attention": {
            "layer": args.attention_layer,
            "hook": "input of WindowAttention3D.proj",
            "raw_shape": [int(value) for value in raw_attention.shape],
            **_components(attention010),
            "task009_checksum": float(task009_attention_score.double().sum().item()),
            "task010_checksum": float(task010_attention_score.double().sum().item()),
            "restoration_max_abs_difference": attention_restoration_difference,
            "max_abs_difference": attention_difference,
        },
        "mlp": {
            "layer": args.mlp_layer,
            "hook": "input of Mlp.fc2",
            "raw_shape": [int(value) for value in raw_mlp.shape],
            **_components(mlp010),
            "task009_checksum": float(task009_mlp_score.double().sum().item()),
            "task010_checksum": float(task010_mlp_score.double().sum().item()),
            "restoration_max_abs_difference": mlp_restoration_difference,
            "max_abs_difference": mlp_difference,
        },
        "same_forward_same_tensor_max_abs_difference": max(
            attention_difference,
            mlp_difference,
            attention_restoration_difference,
            mlp_restoration_difference,
        ),
    }
    output_path = Path(args.output_dir) / "real_activation_trace.json"
    _atomic_json(output_path, output)
    print(f"Task011 real-activation trace written to {output_path.resolve()}")
    print(
        "Maximum Task009/Task010 formula difference: "
        f"{output['same_forward_same_tensor_max_abs_difference']:.9g}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task011 minimal real activation trace")
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--checkpoint_path",
        default="/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt",
    )
    parser.add_argument(
        "--probe_samples",
        default="/home/jixinye25/jxy_work1/work1-pruning/descriptor_ablation_validation/probe_samples.csv",
    )
    parser.add_argument(
        "--val_split", default="/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
    )
    parser.add_argument("--attention_layer", default="layers.0.blocks.0.attn")
    parser.add_argument("--mlp_layer", default="layers.0.blocks.0.mlp")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_dir", default="task011_diagnosis")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("workers must be non-negative")
    return args


if __name__ == "__main__":
    run_trace(parse_args())

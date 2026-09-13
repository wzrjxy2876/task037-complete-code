#!/usr/bin/env python3
"""Task042 Phase A.1 exploratory temporal-relation-retention pilot.

The program evaluates the supplied dense checkpoints and existing structured
pruning selections on the same deterministic UCF101 validation clips.  It does
not train, overwrite checkpoints, or claim matched physical FLOPs reduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F


DEFAULTS = {
    "mamba": {
        "base": "/home/jixinye25/jxy_work1/Code/mamba/mambapruner/videomamba_pretrained/videomamba_small-89%.pth",
        "prune": "/home/jixinye25/jxy_work1/Code/mamba/mambapruner/output_opt/UCF-videomamba_small_-20%-85.88/videomamba_small_s20_pruned_init.pth",
        "source": "/home/jixinye25/jxy_work1/Code/mamba/mambapruner",
        "prune_source": "existing VideoMamba InteractionPruner mask; before fine-tuning",
        "env": "mamba",
    },
    "slowfast": {
        "base": "/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt",
        "prune": "/data/jixinye25/work1/work1_output/UCF-slowfast/UCF-slowfast_sigma=0.1-50%-85.99%/slowfast_InteractionPruner_best.pth",
        "source": "/home/jixinye25/jxy_work1/Code",
        "prune_source": "existing SlowFast Task038 interaction/BMS masks extracted from saved checkpoint",
        "env": "MC_Pruning",
    },
    "swin": {
        "base": "/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt",
        "prune": "/data/jixinye25/work1/work1_output/UCF-swintrans/swintrans_st_mf_s0.50_sigma0.10_seed3407_20260723_163523/swin_pruned_before_finetune.pth",
        "source": "/home/jixinye25/jxy_work1/Code",
        "prune_source": "existing Video Swin BMS keep-index selection; before fine-tuning",
        "env": "MC_Pruning",
    },
}

CONDITIONS = ("clean", "adjacent_swap", "block_reorder", "reverse")
CSV_FIELDS = (
    "video_id", "action_class", "label", "model", "variant", "condition",
    "prediction", "top1_correct", "agrees_with_dense", "js_vs_dense_bits",
    "temporal_js_bits", "relation_retention_trr", "relation_collapse_trc",
)


def _safe_load(path: str) -> Any:
    """Prefer PyTorch's restricted loader when available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _state_dict(obj: Any, model_name: str) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("checkpoint root must be a mapping")
    if model_name == "mamba":
        state = obj.get("state_dict", obj.get("model", obj))
    else:
        state = obj.get("state_dict", obj)
    if not isinstance(state, dict):
        raise TypeError("checkpoint has no state dictionary")
    result = {}
    for key, value in state.items():
        clean = str(key)
        while clean.startswith("module.") or clean.startswith("backbone."):
            clean = clean.split(".", 1)[1]
        if torch.is_tensor(value):
            result[clean] = value
    return result


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_audit(path: str, model_name: str) -> dict[str, Any]:
    obj = _safe_load(path)
    state = _state_dict(obj, model_name)
    tensors = [value for value in state.values() if value.is_floating_point()]
    total = sum(int(value.numel()) for value in tensors)
    zeros = sum(int(torch.count_nonzero(value == 0).item()) for value in tensors)
    classifier = [
        (key, list(value.shape)) for key, value in state.items()
        if value.ndim == 2 and any(token in key.lower() for token in ("head", "fc_cls", "projection"))
    ]
    root_metadata = {}
    if isinstance(obj, dict):
        for key in ("epoch", "target_sparsity", "sparsity", "top1", "top5", "note"):
            if key in obj and isinstance(obj[key], (str, int, float, bool, type(None))):
                root_metadata[key] = obj[key]
    return {
        "path": path,
        "exists": Path(path).is_file(),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "state_tensor_count": len(state),
        "floating_parameter_count": total,
        "exact_zero_fraction": zeros / total if total else None,
        "classifier_weight_shapes": classifier,
        "root_metadata": root_metadata,
    }


def temporal_permutation(length: int, condition: str, device: torch.device) -> torch.Tensor:
    if condition == "clean":
        indices = list(range(length))
    elif condition == "adjacent_swap":
        indices = list(range(length))
        for start in range(0, length - 1, 2):
            indices[start], indices[start + 1] = indices[start + 1], indices[start]
    elif condition == "block_reorder":
        bounds = [round(i * length / 4) for i in range(5)]
        blocks = [list(range(bounds[i], bounds[i + 1])) for i in range(4)]
        indices = blocks[2] + blocks[0] + blocks[3] + blocks[1]
    elif condition == "reverse":
        indices = list(reversed(range(length)))
    else:
        raise ValueError("unknown temporal condition: " + condition)
    return torch.tensor(indices, dtype=torch.long, device=device)


def permute_clip(batch_clip: torch.Tensor, condition: str) -> torch.Tensor:
    if batch_clip.ndim != 5:
        raise ValueError("clip must have shape [B,C,T,H,W]")
    index = temporal_permutation(batch_clip.shape[2], condition, batch_clip.device)
    return batch_clip.index_select(2, index)


def trajectory_relation(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build the multi-frame relation matrix from first/second temporal differences."""
    if features.ndim != 2 or features.shape[0] < 4:
        raise ValueError("features must be [T,C] with T >= 4")
    z = F.layer_norm(features.float(), (features.shape[-1],))
    first = z[1:] - z[:-1]
    second = z[2:] - 2.0 * z[1:-1] + z[:-2]
    q = torch.cat((first[: second.shape[0]], second), dim=-1)
    q = F.normalize(q, dim=-1, eps=eps)
    return q @ q.transpose(0, 1)


def relation_retention(dense: torch.Tensor, pruned: torch.Tensor) -> float:
    if dense.shape != pruned.shape:
        raise ValueError("dense and pruned relation matrices must have identical shapes")
    tri = torch.triu_indices(dense.shape[0], dense.shape[1], offset=1, device=dense.device)
    a = dense[tri[0], tri[1]].reshape(1, -1)
    b = pruned[tri[0], tri[1]].reshape(1, -1)
    return float(F.cosine_similarity(a, b, dim=1, eps=1e-8).item())


def js_divergence_bits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    pa = F.softmax(logits_a.float(), dim=-1)
    pb = F.softmax(logits_b.float(), dim=-1)
    mean = (pa + pb) * 0.5
    log_pa = torch.log(pa.clamp_min(1e-12))
    log_pb = torch.log(pb.clamp_min(1e-12))
    log_m = torch.log(mean.clamp_min(1e-12))
    value = 0.5 * (pa * (log_pa - log_m)).sum(-1) + 0.5 * (pb * (log_pb - log_m)).sum(-1)
    return float((value / math.log(2.0)).mean().item())


def _select_top(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    count = int(scores.numel())
    keep_count = min(count, max(1, int(keep_count)))
    chosen = torch.topk(scores.detach().float().cpu(), keep_count, largest=True, sorted=False).indices
    mask = torch.zeros(count, dtype=torch.float32)
    mask[chosen] = 1.0
    return mask


def _load_model(model_name: str, source_root: str, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, source_root)
    if model_name == "mamba":
        from myvideomamba import videomamba_small
        model = videomamba_small()
    elif model_name == "slowfast":
        from IPslowfast import slowfast_16x8_resnet101_kinetics400
        model = slowfast_16x8_resnet101_kinetics400(num_classes=101)
    elif model_name == "swin":
        from MC import SwinTransformer3D
        model = SwinTransformer3D(
            patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.0,
            qkv_bias=True, patch_norm=True, drop_path_rate=0.2,
        )
    else:
        raise ValueError("unknown model: " + model_name)
    return model.to(device).eval()


def _load_weights(model: torch.nn.Module, path: str, model_name: str) -> tuple[Any, dict[str, Any]]:
    obj = _safe_load(path)
    state = _state_dict(obj, model_name)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model key mismatch; missing=%s unexpected=%s" %
            (result.missing_keys[:12], result.unexpected_keys[:12])
        )
    return obj, state


def _mamba_masks(model: torch.nn.Module, pruning_obj: dict[str, Any], mode: str) -> tuple[list[Any], dict[str, Any]]:
    state = pruning_obj.get("state_dict", {})
    modules = dict(model.named_modules())
    specs = []
    for key, mask in state.items():
        if not str(key).endswith(".interaction_mask") or not torch.is_tensor(mask):
            continue
        module_name = str(key)[: -len(".interaction_mask")]
        module = modules.get(module_name)
        if module is None or not hasattr(module, "weight"):
            continue
        if mask.numel() == module.out_features:
            orientation = "out"
            magnitude_scores = module.weight.detach().float().norm(p=2, dim=1)
        elif mask.numel() == module.in_features:
            orientation = "in"
            magnitude_scores = module.weight.detach().float().norm(p=2, dim=0)
        else:
            raise RuntimeError("Mamba mask shape does not match projection: " + module_name)
        current = mask.detach().float().reshape(-1).cpu()
        magnitude = _select_top(magnitude_scores, int(current.sum().item()))
        specs.append({"name": module_name, "module": module, "orientation": orientation,
                      "current": current, "magnitude": magnitude})
    if not specs:
        raise RuntimeError("no interaction_mask tensors found in VideoMamba prune checkpoint")
    handles = []
    def apply_variant(variant: str) -> None:
        nonlocal handles
        for handle in handles:
            handle.remove()
        handles = []
        if variant == "dense":
            return
        for spec in specs:
            raw_mask = spec["current"] if variant == "existing_prune" else spec["magnitude"]
            mask = raw_mask.to(device=next(model.parameters()).device)
            if spec["orientation"] == "out":
                def post_hook(_module, _inputs, output, mask=mask):
                    return output * mask.view(*([1] * (output.ndim - 1)), -1)
                handles.append(spec["module"].register_forward_hook(post_hook))
            else:
                def pre_hook(_module, inputs, mask=mask):
                    x = inputs[0] * mask.view(*([1] * (inputs[0].ndim - 1)), -1)
                    return (x,) + tuple(inputs[1:])
                handles.append(spec["module"].register_forward_pre_hook(pre_hook))
    kept = sum(int(spec["current"].sum().item()) for spec in specs)
    total = sum(int(spec["current"].numel()) for spec in specs)
    return apply_variant, {"mask_source": "existing InteractionPruner", "masked_module_count": len(specs),
                           "mask_kept_fraction": kept / total, "target_sparsity": pruning_obj.get("target_sparsity")}


def _slowfast_masks(model: torch.nn.Module, pruning_obj: dict[str, Any], mode: str) -> tuple[Any, dict[str, Any]]:
    state = pruning_obj.get("state_dict", {})
    modules = dict(model.named_modules())
    specs = []
    for key, mask in state.items():
        if not str(key).endswith(".interaction_mask") or not torch.is_tensor(mask):
            continue
        module_name = str(key)[: -len(".interaction_mask")]
        module = modules.get(module_name)
        if module is None or not hasattr(module, "weight") or module.weight.ndim < 2:
            continue
        if mask.numel() != module.weight.shape[0]:
            raise RuntimeError("SlowFast mask shape does not match output channels: " + module_name)
        current = mask.detach().float().reshape(-1).cpu()
        scores = module.weight.detach().float().reshape(module.weight.shape[0], -1).norm(p=2, dim=1)
        specs.append({"name": module_name, "module": module, "current": current,
                      "magnitude": _select_top(scores, int(current.sum().item()))})
    if not specs:
        raise RuntimeError("no interaction_mask tensors found in SlowFast prune checkpoint")
    for spec in specs:
        module = spec["module"]
        if "interaction_mask" not in module._buffers:
            module.register_buffer("interaction_mask", torch.ones_like(spec["current"], device=next(model.parameters()).device))
    def apply_variant(variant: str) -> None:
        for spec in specs:
            if variant == "dense":
                mask = torch.ones_like(spec["current"])
            elif variant == "existing_prune":
                mask = spec["current"]
            else:
                mask = spec["magnitude"]
            spec["module"]._buffers["interaction_mask"] = mask.to(device=next(model.parameters()).device)
    kept = sum(int(spec["current"].sum().item()) for spec in specs)
    total = sum(int(spec["current"].numel()) for spec in specs)
    return apply_variant, {"mask_source": "existing SlowFast interaction/BMS checkpoint", "masked_module_count": len(specs),
                           "mask_kept_fraction": kept / total, "checkpoint_sparsity": pruning_obj.get("sparsity")}


def _swin_unit_score(module: torch.nn.Module, unit_type: str) -> torch.Tensor:
    if unit_type == "neuron":
        return module.fc1.weight.detach().float().norm(p=2, dim=1)
    if unit_type == "head":
        dim = module.qkv.in_features
        heads = int(module.num_heads)
        head_dim = dim // heads
        weights = module.qkv.weight.detach().float().reshape(3, heads, head_dim, dim)
        return weights.square().sum(dim=(0, 2, 3)).sqrt()
    raise ValueError("unknown Video Swin unit type: " + unit_type)


def _swin_masks(model: torch.nn.Module, pruning_obj: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    raw = pruning_obj.get("keep_indices")
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("Video Swin pruning checkpoint has no keep_indices registry")
    modules = dict(model.named_modules())
    specs = []
    for module_name, entry in raw.items():
        module = modules.get(module_name)
        if module is None or not isinstance(entry, dict):
            continue
        unit_type = entry.get("unit_type", entry.get("type"))
        indices = entry.get("indices", entry.get("data"))
        if unit_type == "head":
            total = int(module.num_heads)
        elif unit_type == "neuron":
            total = int(module.original_hidden_features)
        else:
            continue
        current = sorted(set(int(i) for i in indices))
        if not current or min(current) < 0 or max(current) >= total:
            raise RuntimeError("invalid keep-index list for " + module_name)
        magnitude = _select_top(_swin_unit_score(module, unit_type), len(current)).nonzero().reshape(-1).tolist()
        specs.append({"name": module_name, "module": module, "unit_type": unit_type,
                      "total": total, "current": current, "magnitude": sorted(int(i) for i in magnitude)})
    if not specs:
        raise RuntimeError("no usable Video Swin keep-index entries found")
    def apply_variant(variant: str) -> None:
        for spec in specs:
            indices = list(range(spec["total"])) if variant == "dense" else (
                spec["current"] if variant == "existing_prune" else spec["magnitude"]
            )
            setattr(spec["module"], "keep_heads" if spec["unit_type"] == "head" else "keep_neurons", indices)
    kept = sum(len(spec["current"]) for spec in specs)
    total = sum(spec["total"] for spec in specs)
    report = pruning_obj.get("pruning_report", {})
    return apply_variant, {"mask_source": "existing Video Swin BMS keep_indices", "masked_module_count": len(specs),
                           "mask_kept_fraction": kept / total,
                           "estimated_parameter_sparsity": report.get("estimated_sparsity"),
                           "target_sparsity": report.get("target_sparsity")}


def _pruning_variants(model: torch.nn.Module, model_name: str, pruning_obj: dict[str, Any]):
    if model_name == "mamba":
        apply, details = _mamba_masks(model, pruning_obj, "existing_prune")
    elif model_name == "slowfast":
        apply, details = _slowfast_masks(model, pruning_obj, "existing_prune")
    else:
        apply, details = _swin_masks(model, pruning_obj)
    return apply, details


class _FeatureCapture:
    def __init__(self, model: torch.nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []
        if model_name == "mamba":
            def capture(_module, _inputs, output):
                hidden, residual = output if isinstance(output, (tuple, list)) else (output, None)
                if residual is not None:
                    hidden = hidden + residual
                self.values["tokens"] = model.norm_f(hidden)
            self.handles.append(model.layers[-1].register_forward_hook(capture))
        elif model_name == "slowfast":
            self.handles.append(model.fast_res5.register_forward_hook(
                lambda _m, _i, out: self.values.update(fast=out)
            ))
            self.handles.append(model.slow_res5.register_forward_hook(
                lambda _m, _i, out: self.values.update(slow=out)
            ))

    def reset(self) -> None:
        self.values.clear()

    def convert(self, input_frames: int) -> torch.Tensor:
        if self.model_name == "mamba":
            tokens = self.values["tokens"]
            b, _, c = tokens.shape
            patches = (tokens.shape[1] - 1) // input_frames
            if tokens.shape[1] != 1 + input_frames * patches:
                raise RuntimeError("unexpected VideoMamba token sequence shape")
            return tokens[:, 1:, :].reshape(b, input_frames, patches, c).mean(dim=2)
        if self.model_name == "slowfast":
            def pool_pathway(value: torch.Tensor, target_t: int) -> torch.Tensor:
                value = value.mean(dim=(-1, -2)).transpose(1, 2)
                if value.shape[1] != target_t:
                    value = F.interpolate(value.transpose(1, 2), size=target_t,
                                          mode="linear", align_corners=False).transpose(1, 2)
                return value
            fast = pool_pathway(self.values["fast"], input_frames)
            slow = pool_pathway(self.values["slow"], input_frames)
            return torch.cat((slow, fast), dim=-1)
        raise RuntimeError("Swin features are returned directly by forward")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _forward(model: torch.nn.Module, model_name: str, capture: _FeatureCapture,
             batch: torch.Tensor, input_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    capture.reset()
    output = model(batch)
    if model_name == "swin":
        logits, stage_features = output
        feat = stage_features[-1]
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = model.norm(feat)
        z = feat.mean(dim=(2, 3))
    else:
        logits = output
        z = capture.convert(input_frames)
    return logits, z


def _data_entries(annotation: str, per_class: int, max_videos: Optional[int]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    with open(annotation, "r", encoding="utf-8") as handle:
        for line in handle:
            cols = line.split()
            if len(cols) < 3:
                continue
            video_id, duration, label = cols[0], int(cols[1]), int(cols[2])
            action = video_id.split("_g", 1)[0][2:]
            groups.setdefault(label, []).append({
                "video_id": video_id, "duration": duration, "label": label,
                "action_class": action,
            })
    chosen = []
    for label in sorted(groups):
        chosen.extend(sorted(groups[label], key=lambda item: item["video_id"])[:per_class])
    if max_videos is not None:
        chosen = chosen[:max_videos]
    if not chosen:
        raise RuntimeError("validation annotation yielded no clips")
    return chosen


def _load_video_input(entry: dict[str, Any], frame_root: str, spatial_transform: Any,
                      temporal_transform: Any, image_loader: Any) -> torch.Tensor:
    video_dir = Path(frame_root) / entry["action_class"] / entry["video_id"]
    if not video_dir.is_dir():
        raise FileNotFoundError("frame directory not found for " + entry["video_id"] + ": " + str(video_dir))
    frame_indices = temporal_transform(list(range(1, entry["duration"] + 1)))
    spatial_transform.randomize_parameters()
    frames = []
    for frame_index in frame_indices:
        path = video_dir / ("image_%05d.jpg" % frame_index)
        if not path.is_file():
            raise FileNotFoundError("frame missing: " + str(path))
        frames.append(spatial_transform(image_loader(str(path))))
    return torch.stack(frames, dim=0).permute(1, 0, 2, 3).contiguous()


def _load_dataset_helpers(source_root: str) -> tuple[Any, Any, str]:
    """Load the repository's exact frame transforms without importing its heavy utils.py.

    The existing utils.py imports GluonCV code that is incompatible with the
    server's PyTorch 1.12 environment. dataset.ucf101 only needs the path
    constant from that module; provide that constant while importing the
    dataset helpers, then restore normal module resolution for model loading.
    """
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    previous_utils = sys.modules.get("utils")
    utils_shim = types.ModuleType("utils")
    utils_shim.UCF_DATA_ROOT = "/data/jixinye25/UCF101_Frame/frames"
    sys.modules["utils"] = utils_shim
    try:
        dataset_module = importlib.import_module("dataset.ucf101")
        return dataset_module.pil_loader, dataset_module.test_transform, utils_shim.UCF_DATA_ROOT
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(items) / len(items) if items else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    defaults = DEFAULTS[args.model]
    base_path = args.base_checkpoint or defaults["base"]
    prune_path = args.prune_checkpoint or defaults["prune"]
    annotation = args.annotation or "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
    summary = {
        "task": "Task042 Phase A.1 temporal relation collapse exploratory pilot",
        "model": args.model,
        "base_checkpoint": checkpoint_audit(base_path, args.model),
        "existing_prune_checkpoint": checkpoint_audit(prune_path, args.model),
        "annotation_path": annotation,
        "annotation_exists": Path(annotation).is_file(),
        "frames_root": args.frames_root or "read from the model source utils.UCF_DATA_ROOT",
        "requested_sample_per_action_class": args.videos_per_class,
        "conditions": list(CONDITIONS),
        "pruning_mask_source": defaults["prune_source"],
        "matched_actual_flops_control": False,
        "finetuning": False,
        "checkpoint_overwrite": False,
    }
    return summary


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    cfg = DEFAULTS[args.model]
    source_root = args.source_root or cfg["source"]
    base_path = args.base_checkpoint or cfg["base"]
    prune_path = args.prune_checkpoint or cfg["prune"]
    annotation = args.annotation or "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model evaluation")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    pil_loader, make_test_transform, default_frame_root = _load_dataset_helpers(source_root)
    spatial_transform, temporal_transform = make_test_transform()
    model = _load_model(args.model, source_root, device)
    _, _ = _load_weights(model, base_path, args.model)
    prune_obj = _safe_load(prune_path)
    configure_pruning, prune_info = _pruning_variants(model, args.model, prune_obj)
    capture = _FeatureCapture(model, args.model)

    entries = _data_entries(annotation, args.videos_per_class, args.max_videos)
    output_dir = Path(args.output_root) / args.run_id / args.model / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    start_time = time.time()

    for entry_number, entry in enumerate(entries, start=1):
        cpu_clip = _load_video_input(entry, args.frames_root or default_frame_root,
                                     spatial_transform, temporal_transform, pil_loader)
        batch = cpu_clip.unsqueeze(0).to(device, non_blocking=True)
        cache = {}
        for variant in ("dense", "magnitude", "existing_prune"):
            configure_pruning(variant)
            cache[variant] = {}
            with torch.no_grad():
                for condition in CONDITIONS:
                    model_input = permute_clip(batch, condition)
                    logits, features = _forward(model, args.model, capture, model_input, batch.shape[2])
                    cache[variant][condition] = {
                        "logits": logits.detach().float().cpu()[0],
                        "features": features.detach().float().cpu()[0],
                    }
        dense_clean_logits = cache["dense"]["clean"]["logits"]
        dense_clean_feature = cache["dense"]["clean"]["features"]
        dense_pred = int(dense_clean_logits.argmax().item())
        dense_relations = {cond: trajectory_relation(cache["dense"][cond]["features"]) for cond in CONDITIONS}
        for variant in ("dense", "magnitude", "existing_prune"):
            clean_logits = cache[variant]["clean"]["logits"]
            clean_pred = int(clean_logits.argmax().item())
            clean_prob = clean_logits
            for condition in CONDITIONS:
                values = cache[variant][condition]
                pred = int(values["logits"].argmax().item())
                trr = relation_retention(
                    dense_relations[condition], trajectory_relation(values["features"])
                )
                temporal_js = 0.0 if condition == "clean" else js_divergence_bits(clean_prob, values["logits"])
                rows.append({
                    "video_id": entry["video_id"], "action_class": entry["action_class"],
                    "label": entry["label"], "model": args.model, "variant": variant,
                    "condition": condition, "prediction": pred,
                    "top1_correct": int(pred == entry["label"]),
                    "agrees_with_dense": int(pred == dense_pred),
                    "js_vs_dense_bits": js_divergence_bits(
                        cache["dense"][condition]["logits"], values["logits"]
                    ),
                    "temporal_js_bits": temporal_js,
                    "relation_retention_trr": trr,
                    "relation_collapse_trc": 1.0 - trr,
                })
        print("[%s] %d/%d %s" % (args.model, entry_number, len(entries), entry["video_id"]), flush=True)

    capture.close()
    _write_csv(output_dir / "per_video_results.csv", rows)
    summaries = {}
    for variant in ("dense", "magnitude", "existing_prune"):
        selected = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {
            "clean_top1_accuracy": _mean(row["top1_correct"] for row in selected if row["condition"] == "clean"),
            "clean_prediction_agreement_with_dense": _mean(row["agrees_with_dense"] for row in selected if row["condition"] == "clean"),
            "clean_relation_retention_trr": _mean(row["relation_retention_trr"] for row in selected if row["condition"] == "clean"),
            "clean_relation_collapse_trc": _mean(row["relation_collapse_trc"] for row in selected if row["condition"] == "clean"),
            "mean_temporal_js_bits_by_condition": {
                condition: _mean(row["temporal_js_bits"] for row in selected if row["condition"] == condition)
                for condition in CONDITIONS
            },
            "mean_js_vs_dense_by_condition": {
                condition: _mean(row["js_vs_dense_bits"] for row in selected if row["condition"] == condition)
                for condition in CONDITIONS
            },
        }
    summary = {
        "task": "Task042 Phase A.1 exploratory temporal relation collapse pilot",
        "model": args.model,
        "variants": summaries,
        "video_count": len(entries),
        "action_class_count": len({entry["label"] for entry in entries}),
        "videos_per_action_class_requested": args.videos_per_class,
        "input_frames": 32,
        "temporal_conditions": list(CONDITIONS),
        "base_checkpoint": {"path": base_path, "sha256": sha256_file(base_path)},
        "prune_checkpoint": {"path": prune_path, "sha256": sha256_file(prune_path)},
        "pruning": prune_info,
        "mask_sources": cfg["prune_source"],
        "magnitude_control": "per-prunable-module L2 magnitude; kept unit counts match existing mask per module",
        "same_pruning_ratio_across_architectures": False,
        "matched_actual_flops_control": False,
        "actual_latency_claim": False,
        "finetuning_performed": False,
        "full_validation": len(entries) == 3783,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root) / args.run_id
    summaries = {}
    for model_name in ("mamba", "slowfast", "swin"):
        path = root / model_name / "evaluation" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError("missing model summary: " + str(path))
        summaries[model_name] = json.loads(path.read_text(encoding="utf-8"))
    comparison = {
        "task": "Task042 Phase A.1 cross-model exploratory temporal relation analysis",
        "models": summaries,
        "interpretation_limitations": [
            "The existing pruned masks have different target rates and were produced by model-specific code.",
            "This is a pilot on one deterministic validation video per action class unless sample size is overridden.",
            "No matched 30% measured FLOPs, physical speedup, or fine-tuning claim is made.",
            "The magnitude control matches each existing selection's per-module kept-unit counts; it is not a matched global FLOPs control.",
        ],
    }
    (root / "cross_model_summary.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Task042 Phase A.1 — Temporal-relation collapse pilot", "",
        "This exploratory run compares each dense checkpoint with a per-module magnitude control and the existing structured-pruning selection on the same 32-frame UCF101 validation clips.",
        "", "| Model | Videos | Dense Top-1 | Magnitude TRR | Existing-prune TRR | Existing-prune TRC | Existing-prune temporal JS (reverse) | Mask kept fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, summary in summaries.items():
        dense = summary["variants"]["dense"]
        mag = summary["variants"]["magnitude"]
        pruned = summary["variants"]["existing_prune"]
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.6f | %.4f |" % (
            model_name, summary["video_count"], dense["clean_top1_accuracy"],
            mag["clean_relation_retention_trr"], pruned["clean_relation_retention_trr"],
            pruned["clean_relation_collapse_trc"],
            pruned["mean_temporal_js_bits_by_condition"]["reverse"],
            summary["pruning"].get("mask_kept_fraction", float("nan")),
        ))
    lines.extend([
        "", "## Readout", "",
        "TRR is the cosine similarity between the upper-triangular entries of the dense and pruned multi-frame relation matrices. TRC is `1 - TRR`. Temporal JS measures the prediction-distribution change between clean and perturbed clips.",
        "", "These values are a screening result. The existing pruning rates differ by architecture, and the saved implementations do not establish a matched physical FLOPs reduction. Do not use this table as a final cross-model superiority claim. The next controlled run should regenerate all masks at a predeclared, measured FLOPs target and then evaluate the full validation split.",
        "", "Per-video observations are in each model's `per_video_results.csv`; complete settings and hashes are in the three model `summary.json` files.",
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "run", "finalize"))
    parser.add_argument("--model", choices=tuple(DEFAULTS))
    parser.add_argument("--output-root", default="/data/jixinye25/work1/output/task042_temporal_relation_collapse")
    parser.add_argument("--run-id", default="phase_a1_pilot")
    parser.add_argument("--source-root")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--prune-checkpoint")
    parser.add_argument("--annotation")
    parser.add_argument("--frames-root")
    parser.add_argument("--videos-per-class", type=int, default=1)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.phase in ("audit", "run") and not args.model:
        parser.error("--model is required for audit and run")
    if args.videos_per_class < 1:
        parser.error("--videos-per-class must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "audit":
        result = audit(args)
        dest = Path(args.output_root) / args.run_id / args.model
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "preflight.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.phase == "run":
        run_model(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure full-unit ablation effects for the existing 3-D descriptor.

This is a validation probe only.  It calls the repository's existing
``InteractionPruner.run_calibration`` and ``build_3d_descriptors`` methods,
then temporarily zeros one head or FFN neuron per replicated branch.  Learned
parameters, pruning registries, BMS, Coverage, and training code are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SEED = 3407
EPSILON = 1e-8
RESULT_FIELDS = [
    "global_index",
    "layer",
    "unit_type",
    "stage",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_st",
    "ablation_logit_deviation",
    "prediction_flip_rate",
]
PROBE_OUTPUT_NAMES = (
    "probe_samples.csv",
    "baseline_logits.pt",
    "unit_ablation_effect.csv",
    "unit_ablation_effect.npz",
    "progress.json",
    "run_metadata.json",
    "descriptor_variant_summary.csv",
    "nearest_neighbor_validation.csv",
    "descriptor_dimension_correlation.csv",
    "analysis_metadata.json",
    "descriptor_variant_spearman.png",
    "descriptor_variant_nnerror.png",
    "descriptor_vs_ablation_distance_3d.png",
    "descriptor_dimension_correlation.png",
    "validation_summary.md",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _install_legacy_import_compatibility() -> None:
    """Mirror the import compatibility used by ucf101_videoswin_my.py."""
    try:
        import PIL
        import PIL.Image as pil_image

        resampling = getattr(pil_image, "Resampling", None)
        if not hasattr(pil_image, "LINEAR"):
            pil_image.LINEAR = resampling.BILINEAR if resampling else 2
            pil_image.BILINEAR = resampling.BILINEAR if resampling else 2
            pil_image.BICUBIC = resampling.BICUBIC if resampling else 3
            pil_image.NEAREST = resampling.NEAREST if resampling else 0
            PIL.Image = pil_image
            sys.modules["PIL.Image"] = pil_image
    except ImportError:
        pass

    try:
        import torch._six as torch_six
    except ImportError:
        from types import ModuleType

        torch_six = ModuleType("torch._six")
        sys.modules["torch._six"] = torch_six
    if not hasattr(torch_six, "int_classes"):
        torch_six.int_classes = (int,)
    if not hasattr(torch_six, "string_classes"):
        torch_six.string_classes = (str,)


def _load_repository_components():
    _install_legacy_import_compatibility()
    from dataset.ucf101 import get_dataset
    from MC import InteractionPruner, SwinTransformer3D

    return get_dataset, InteractionPruner, SwinTransformer3D


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    elif isinstance(output, dict):
        logits = output.get("logits")
        if logits is None:
            logits = output.get("cls_score")
    else:
        logits = None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("model output must contain logits with shape [B,C]")
    return logits


def _unpack_videos(batch: object) -> torch.Tensor:
    videos = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(videos, torch.Tensor) or videos.ndim != 5:
        shape = getattr(videos, "shape", None)
        raise ValueError(f"expected video tensor [B,C,T,H,W], got {shape}")
    return videos


def hash_model_parameters(model: torch.nn.Module) -> str:
    """SHA-256 over learned parameters without retaining a second state_dict."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        if value.ndim == 0:
            value = value.reshape(1)
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def descriptor_fingerprint(
    descriptor: torch.Tensor, unit_records: list[dict]
) -> str:
    digest = hashlib.sha256()
    values = descriptor.detach().cpu().float().contiguous().numpy()  # [N,3]
    digest.update(values.tobytes(order="C"))
    for record in unit_records:
        digest.update(
            f"{record['global_index']}|{record['layer']}|"
            f"{record['unit_type']}|{record['unit_index']}\n".encode("utf-8")
        )
    return digest.hexdigest()


class BranchAblationController:
    """Temporary branch-specific activation ablation for one target layer.

    For a chunk containing ``K`` units and an input batch containing ``B``
    videos, the model sees ``K*B`` samples ordered as ``[K,B,...]``.  The MLP
    hook receives hidden activations ``[K*B,T,H,W,U]`` before ``fc2``.  The
    attention hook receives projected-head inputs
    ``[K*B*nW,N,H*D]`` before ``proj``.  Branch ``k`` zeros only its assigned
    unit.
    """

    def __init__(self, module: torch.nn.Module, unit_type: str):
        if unit_type not in {"attention_head", "ffn_neuron"}:
            raise ValueError(f"unsupported unit type {unit_type!r}")
        self.module = module
        self.unit_type = unit_type
        self.unit_indices: tuple[int, ...] | None = None
        self.batch_size: int | None = None
        target = module.proj if unit_type == "attention_head" else module.fc2
        self.handle = target.register_forward_pre_hook(self._hook)

    def configure(self, unit_indices: Iterable[int], batch_size: int) -> None:
        indices = tuple(int(value) for value in unit_indices)
        if not indices or batch_size <= 0:
            raise ValueError("unit_indices and batch_size must be non-empty")
        limit = (
            int(self.module.num_heads)
            if self.unit_type == "attention_head"
            else int(self.module.fc1.out_features)
        )
        if any(index < 0 or index >= limit for index in indices):
            raise IndexError(f"unit index outside [0,{limit})")
        self.unit_indices = indices
        self.batch_size = int(batch_size)

    def disable(self) -> None:
        self.unit_indices = None
        self.batch_size = None

    def remove(self) -> None:
        self.disable()
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module: torch.nn.Module, inputs: tuple) -> tuple | None:
        del module
        if self.unit_indices is None:
            return None
        value = inputs[0]
        chunk_size = len(self.unit_indices)
        batch_size = int(self.batch_size)
        total_samples = chunk_size * batch_size

        if self.unit_type == "ffn_neuron":
            if value.ndim < 2 or value.shape[0] != total_samples:
                raise ValueError(
                    "FFN ablation input must have first axis K*B; got "
                    f"{tuple(value.shape)}, K={chunk_size}, B={batch_size}"
                )
            # [K*B,...,U] -> [K,B,...,U]
            masked = value.reshape(
                chunk_size, batch_size, *value.shape[1:]
            ).clone()
            for branch, unit_index in enumerate(self.unit_indices):
                masked[branch, ..., unit_index] = 0
            return (masked.reshape_as(value),)

        if value.ndim != 3 or value.shape[0] % total_samples != 0:
            raise ValueError(
                "Attention ablation input must be [K*B*nW,N,H*D]; got "
                f"{tuple(value.shape)}, K={chunk_size}, B={batch_size}"
            )
        num_heads = int(self.module.num_heads)
        head_dim = int(self.module.head_dim)
        if value.shape[-1] != num_heads * head_dim:
            raise ValueError(
                f"attention channel axis {value.shape[-1]} does not equal H*D="
                f"{num_heads * head_dim}"
            )
        windows_per_sample = value.shape[0] // total_samples
        # [K*B*nW,N,H*D] -> [K,B,nW,N,H,D]
        masked = value.reshape(
            chunk_size,
            batch_size,
            windows_per_sample,
            value.shape[1],
            num_heads,
            head_dim,
        ).clone()
        for branch, unit_index in enumerate(self.unit_indices):
            masked[branch, ..., unit_index, :] = 0
        return (masked.reshape_as(value),)


def _infer_class_name(identifier: str) -> str:
    path = Path(identifier)
    parent = path.parent.name
    if parent and parent not in {".", "frames"}:
        return parent
    stem = path.stem
    match = re.match(r"v_(.+?)_g\d+_c\d+", stem)
    return match.group(1) if match else ""


def parse_split_records(split_path: Path) -> list[dict]:
    """Parse labels and identifiers while preserving dataset line indices."""
    records = []
    with split_path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            sample_index = len(records)
            tokens = stripped.split()
            label = None
            for token in reversed(tokens):
                try:
                    label = int(token)
                    break
                except ValueError:
                    continue
            if label is None:
                raise ValueError(
                    f"cannot parse class label from line {source_line}: {stripped}"
                )
            identifier = tokens[0]
            records.append(
                {
                    "sample_index": sample_index,
                    "class_id": label,
                    "class_name": _infer_class_name(identifier),
                    "video_identifier": identifier,
                }
            )
    if not records:
        raise ValueError(f"validation split is empty: {split_path}")
    return records


def choose_probe_samples(
    records: list[dict], num_classes: int, videos_per_class: int, seed: int
) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_class[int(record["class_id"])].append(record)
    eligible = sorted(
        class_id
        for class_id, samples in by_class.items()
        if len(samples) >= videos_per_class
    )
    if len(eligible) < num_classes:
        raise ValueError(
            f"only {len(eligible)} classes contain at least {videos_per_class} videos"
        )
    rng = random.Random(seed)
    chosen_classes = sorted(rng.sample(eligible, num_classes))
    selected = []
    for class_id in chosen_classes:
        samples = rng.sample(by_class[class_id], videos_per_class)
        selected.extend(sorted(samples, key=lambda row: row["sample_index"]))
    if len(selected) != num_classes * videos_per_class:
        raise RuntimeError("probe sample selection has an unexpected size")
    return selected


def _write_probe_samples(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_index", "class_id", "class_name", "video_identifier"
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_probe_samples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample_index"] = int(row["sample_index"])
        row["class_id"] = int(row["class_id"])
    return rows


def make_probe_loader(
    base_loader: DataLoader,
    sample_indices: list[int],
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    """Rebuild the same deterministic subset for baseline and every layer."""
    if not hasattr(base_loader, "dataset"):
        raise TypeError("get_dataset must return a DataLoader with .dataset")
    dataset = base_loader.dataset
    if max(sample_indices) >= len(dataset):
        raise IndexError(
            "probe_samples.csv indices do not match the current validation dataset"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(getattr(base_loader, "pin_memory", True)),
        drop_last=False,
        collate_fn=getattr(base_loader, "collate_fn", None),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def collect_baseline_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    logits = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Baseline logits", ncols=0):
            videos = _unpack_videos(batch).float().to(device, non_blocking=True)
            logits.append(_extract_logits(model(videos)).detach().cpu().float())
            del videos
    result = torch.cat(logits, dim=0)  # [M,C]
    if not torch.isfinite(result).all():
        raise ValueError("baseline logits contain NaN or infinity")
    return result


def _is_cuda_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower() and torch.cuda.is_available()


def measure_layer_ablation(
    model: torch.nn.Module,
    module: torch.nn.Module,
    unit_type: str,
    num_units: int,
    loader: DataLoader,
    baseline_logits: torch.Tensor,
    device: torch.device,
    requested_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Measure ``E_i`` for every unit in one layer with branch batching."""
    effect_sum = np.zeros(num_units, dtype=np.float64)  # [U]
    flip_sum = np.zeros(num_units, dtype=np.float64)  # [U]
    effective_chunk_size = min(requested_chunk_size, num_units)
    controller = BranchAblationController(module, unit_type)
    sample_offset = 0
    last_videos = None
    last_baseline = None
    progress = tqdm(
        total=num_units * baseline_logits.shape[0],
        desc="Unit chunks within layer",
        ncols=0,
    )
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                videos = _unpack_videos(batch).float().to(
                    device, non_blocking=True
                )  # [B,C,T,H,W]
                batch_size = videos.shape[0]
                baseline = baseline_logits[
                    sample_offset:sample_offset + batch_size
                ].to(device, non_blocking=True)  # [B,C_out]
                if baseline.shape[0] != batch_size:
                    raise RuntimeError("probe loader order changed from baseline pass")

                start = 0
                while start < num_units:
                    current_size = min(effective_chunk_size, num_units - start)
                    current_units = tuple(range(start, start + current_size))
                    replicated = None
                    ablated_logits = None
                    try:
                        repeats = (current_size,) + (1,) * (videos.ndim - 1)
                        # [B,C,T,H,W] -> [K*B,C,T,H,W], branch-major.
                        replicated = videos.repeat(repeats)
                        controller.configure(current_units, batch_size)
                        ablated_logits = _extract_logits(model(replicated)).reshape(
                            current_size, batch_size, -1
                        )  # [K,B,C_out]
                    except RuntimeError as error:
                        controller.disable()
                        if not _is_cuda_oom(error) or current_size == 1:
                            raise
                        del replicated, ablated_logits
                        torch.cuda.empty_cache()
                        effective_chunk_size = max(1, current_size // 2)
                        print(
                            "CUDA OOM during ablation; retrying without skipped "
                            f"units at chunk size {effective_chunk_size}"
                        )
                        continue

                    controller.disable()
                    denominator = torch.linalg.vector_norm(
                        baseline, dim=1
                    ).clamp_min(EPSILON)  # [B]
                    deviation = torch.linalg.vector_norm(
                        ablated_logits - baseline.unsqueeze(0), dim=2
                    ) / denominator.unsqueeze(0)  # [K,B]
                    flips = (
                        ablated_logits.argmax(dim=2)
                        != baseline.argmax(dim=1).unsqueeze(0)
                    ).float()  # [K,B]
                    effect_sum[start:start + current_size] += (
                        deviation.sum(dim=1).detach().cpu().double().numpy()
                    )
                    flip_sum[start:start + current_size] += (
                        flips.sum(dim=1).detach().cpu().double().numpy()
                    )
                    progress.update(current_size * batch_size)
                    start += current_size
                    del replicated, ablated_logits, deviation, flips

                sample_offset += batch_size
                last_videos = videos
                last_baseline = baseline
    finally:
        progress.close()
        controller.remove()

    if sample_offset != baseline_logits.shape[0]:
        raise RuntimeError(
            f"layer evaluated {sample_offset} videos, expected {baseline_logits.shape[0]}"
        )
    if last_videos is None:
        raise RuntimeError("probe loader produced no videos")

    # Hook removal must restore the original path for the same input batch.
    with torch.no_grad():
        restored = _extract_logits(model(last_videos)).float()
    max_restoration_error = float((restored - last_baseline).abs().max().item())
    if not torch.allclose(restored, last_baseline, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            "model output did not return to baseline after temporary ablation; "
            f"max error={max_restoration_error:.6g}"
        )

    divisor = float(sample_offset)
    effect = effect_sum / divisor  # [U]
    flip_rate = flip_sum / divisor  # [U]
    if not np.isfinite(effect).all() or not np.isfinite(flip_rate).all():
        raise ValueError("ablation results contain NaN or infinity")
    return effect, flip_rate, effective_chunk_size, max_restoration_error


def _stage_from_layer(layer_name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return match.group(1) if match else ""


def build_unit_records(
    model: torch.nn.Module,
    descriptor: torch.Tensor,
    unit_info: list[dict],
) -> list[dict]:
    modules = dict(model.named_modules())
    records = []
    for global_index, info in enumerate(unit_info):
        layer = str(info["layer"])
        unit_index = int(info["idx"])
        module = modules[layer]
        unit_type = (
            "attention_head"
            if "WindowAttention3D" in module.__class__.__name__
            else "ffn_neuron"
        )
        values = descriptor[global_index].detach().cpu().double().tolist()
        records.append(
            {
                "global_index": global_index,
                "layer": layer,
                "unit_type": unit_type,
                "stage": _stage_from_layer(layer),
                "unit_index": unit_index,
                "D_abs": float(values[0]),
                "D_rel": float(values[1]),
                "D_st": float(values[2]),
            }
        )
    return records


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_resume_rows(
    rows: list[dict], unit_records: list[dict], tolerance: float = 1e-7
) -> set[str]:
    """Validate saved descriptors and return fully completed layer names."""
    expected = {record["global_index"]: record for record in unit_records}
    seen_global = set()
    seen_unit = set()
    by_layer: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        global_index = int(row["global_index"])
        if global_index in seen_global or global_index not in expected:
            raise ValueError("resume CSV contains duplicate or unknown global_index")
        record = expected[global_index]
        unit_key = (row["layer"], int(row["unit_index"]))
        if unit_key in seen_unit:
            raise ValueError("resume CSV contains a duplicate (layer, unit_index)")
        if unit_key != (record["layer"], record["unit_index"]):
            raise ValueError("resume CSV unit metadata does not match descriptor order")
        for column in ("D_abs", "D_rel", "D_st"):
            if not math_isclose(float(row[column]), float(record[column]), tolerance):
                raise ValueError(
                    f"resume descriptor mismatch at global_index={global_index}, {column}"
                )
        seen_global.add(global_index)
        seen_unit.add(unit_key)
        by_layer[record["layer"]].add(record["unit_index"])

    expected_by_layer: dict[str, set[int]] = defaultdict(set)
    for record in unit_records:
        expected_by_layer[record["layer"]].add(record["unit_index"])
    completed = {
        layer for layer, indices in by_layer.items()
        if indices == expected_by_layer[layer]
    }
    partial = set(by_layer) - completed
    if partial:
        raise ValueError(
            "resume CSV contains partial layers; rerun with --overwrite or restore "
            f"the last complete checkpoint: {sorted(partial)}"
        )
    return completed


def math_isclose(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def _write_result_rows(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["global_index"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in PROBE_OUTPUT_NAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()


def _build_model(SwinTransformer3D, checkpoint_path: Path, device: torch.device):
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
        use_checkpoint=True,
    ).to(device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }
    load_result = model.load_state_dict(cleaned, strict=False)
    print(
        "Checkpoint loaded: "
        f"missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    model.eval()
    return model


def _create_run_metadata(
    args: argparse.Namespace,
    descriptor_hash: str,
    parameter_hash: str,
    sample_rows: list[dict],
) -> dict:
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "git_commit": _git_commit(),
        "command_line": [sys.executable, *sys.argv],
        "seed": args.seed,
        "descriptor_fingerprint": descriptor_hash,
        "model_parameter_sha256_before_ablation": parameter_hash,
        "probe_sample_indices": [row["sample_index"] for row in sample_rows],
        "probe_video_count": len(sample_rows),
        "deterministic_kernels_forced": False,
        "note": (
            "Seeds are fixed, but deterministic algorithms are not forced because "
            "required Video Swin CUDA operations may not support them."
        ),
    }


def run_probe(args: argparse.Namespace) -> None:
    if args.model != "swintrans":
        raise ValueError("Task 007 currently validates only --model swintrans")
    visible_gpus = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 007 full probe requires CUDA")
    device = torch.device("cuda:0")
    print(f"CUDA_VISIBLE_DEVICES: {visible_gpus}")
    print(f"Visible GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    print("Probe execution device: cuda:0 (unit ordering is GPU-count independent)")

    output_dir = Path(args.output_dir)
    _prepare_output_directory(output_dir, args.overwrite)
    get_dataset, InteractionPruner, SwinTransformer3D = (
        _load_repository_components()
    )

    model = _build_model(SwinTransformer3D, Path(args.checkpoint_path), device)
    calibration_loader = get_dataset(args.calib_split, args.calib_batch_size)
    pruner = InteractionPruner(model)
    pruner.run_calibration(
        calibration_loader, device=device, num_batches=args.calib_batches
    )
    valid_layer_names = [
        name for name in pruner.ordered_layer_names if name in pruner.activations
    ]
    descriptor, unit_info, _, _, _ = pruner.build_3d_descriptors(
        valid_layer_names
    )
    if descriptor.ndim != 2 or descriptor.shape[1] != 3:
        raise AssertionError(f"descriptor shape must be [N,3], got {descriptor.shape}")
    if len(unit_info) != descriptor.shape[0]:
        raise AssertionError("unit_info length differs from descriptor row count")
    if not torch.isfinite(descriptor).all():
        raise AssertionError("descriptor contains NaN or infinity")

    unit_records = build_unit_records(model, descriptor, unit_info)
    attention_units = sum(
        record["unit_type"] == "attention_head" for record in unit_records
    )
    mlp_units = sum(record["unit_type"] == "ffn_neuron" for record in unit_records)
    print(f"Total descriptor units: {len(unit_records)}")
    print(f"Attention head units: {attention_units}")
    print(f"FFN neuron units: {mlp_units}")
    print(f"Descriptor shape: [{descriptor.shape[0]},3]")

    descriptor_hash = descriptor_fingerprint(descriptor, unit_records)
    result_path = output_dir / "unit_ablation_effect.csv"
    progress_path = output_dir / "progress.json"
    existing_rows = _read_result_rows(result_path)
    completed_layers = validate_resume_rows(existing_rows, unit_records)
    effective_chunks: dict[str, int] = {}
    restoration_errors: dict[str, float] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("descriptor_fingerprint") != descriptor_hash:
            raise ValueError(
                "progress.json descriptor fingerprint differs; use --overwrite "
                "only after confirming the new calibration is intended"
            )
        progress_layers = set(progress.get("completed_layers", []))
        if progress_layers != completed_layers:
            raise ValueError("progress.json and result CSV completed layers disagree")
        effective_chunks.update(
            {
                str(name): int(value)
                for name, value in progress.get("effective_chunk_sizes", {}).items()
            }
        )
        restoration_errors.update(
            {
                str(name): float(value)
                for name, value in progress.get("max_restoration_errors", {}).items()
            }
        )

    split_records = parse_split_records(Path(args.val_split))
    sample_path = output_dir / "probe_samples.csv"
    if sample_path.exists():
        sample_rows = _read_probe_samples(sample_path)
        expected_count = args.num_classes * args.videos_per_class
        if len(sample_rows) != expected_count:
            raise ValueError(
                f"existing probe_samples.csv has {len(sample_rows)} rows, "
                f"expected {expected_count}"
            )
        class_counts = Counter(row["class_id"] for row in sample_rows)
        if (
            len(class_counts) != args.num_classes
            or set(class_counts.values()) != {args.videos_per_class}
        ):
            raise ValueError(
                "existing probe_samples.csv does not match the requested "
                "class-balanced configuration"
            )
        current_split = {row["sample_index"]: row for row in split_records}
        for row in sample_rows:
            current = current_split.get(row["sample_index"])
            if (
                current is None
                or current["class_id"] != row["class_id"]
                or current["video_identifier"] != row["video_identifier"]
            ):
                raise ValueError(
                    "probe_samples.csv no longer matches the validation split"
                )
    else:
        sample_rows = choose_probe_samples(
            split_records,
            args.num_classes,
            args.videos_per_class,
            args.seed,
        )
        _write_probe_samples(sample_path, sample_rows)
    sample_indices = [int(row["sample_index"]) for row in sample_rows]

    base_val_loader = get_dataset(args.val_split, 1)

    def new_probe_loader() -> DataLoader:
        set_seed(args.seed)
        return make_probe_loader(
            base_val_loader,
            sample_indices,
            args.probe_batch_size,
            args.workers,
            args.seed,
        )

    baseline_logits = collect_baseline_logits(model, new_probe_loader(), device)
    if baseline_logits.shape[0] != len(sample_rows):
        raise AssertionError(
            f"baseline rows {baseline_logits.shape[0]} != probe videos {len(sample_rows)}"
        )
    torch.save(baseline_logits, output_dir / "baseline_logits.pt")

    parameter_hash_before = hash_model_parameters(model)
    metadata = _create_run_metadata(
        args, descriptor_hash, parameter_hash_before, sample_rows
    )
    _write_json(output_dir / "run_metadata.json", metadata)

    modules = dict(model.named_modules())
    result_by_global = {
        int(row["global_index"]): row for row in existing_rows
    }
    total_layers = len(valid_layer_names)
    for layer_position, layer_name in enumerate(valid_layer_names, start=1):
        layer_records = [
            record for record in unit_records if record["layer"] == layer_name
        ]
        if layer_name in completed_layers:
            print(
                f"Layer {layer_position}/{total_layers}: {layer_name} already "
                "complete; skipping"
            )
            continue
        layer_records.sort(key=lambda record: record["unit_index"])
        expected_indices = list(range(len(layer_records)))
        actual_indices = [record["unit_index"] for record in layer_records]
        if actual_indices != expected_indices:
            raise AssertionError(
                f"{layer_name} unit indices are not contiguous from zero"
            )

        module = modules[layer_name]
        unit_type = layer_records[0]["unit_type"]
        print()
        print(f"Layer {layer_position}/{total_layers}")
        print(layer_name)
        print(f"units: {len(layer_records)}")
        print(f"completed: 0/{len(layer_records)}")
        print(f"requested chunk size: {args.ablation_chunk_size}")
        effect, flip_rate, effective_chunk, restoration_error = (
            measure_layer_ablation(
                model,
                module,
                unit_type,
                len(layer_records),
                new_probe_loader(),
                baseline_logits,
                device,
                args.ablation_chunk_size,
            )
        )
        for local_index, record in enumerate(layer_records):
            result_by_global[record["global_index"]] = {
                **record,
                "ablation_logit_deviation": float(effect[local_index]),
                "prediction_flip_rate": float(flip_rate[local_index]),
            }
        completed_layers.add(layer_name)
        effective_chunks[layer_name] = effective_chunk
        restoration_errors[layer_name] = restoration_error
        _write_result_rows(result_path, list(result_by_global.values()))
        _write_json(
            progress_path,
            {
                "descriptor_fingerprint": descriptor_hash,
                "completed_layers": [
                    name for name in valid_layer_names if name in completed_layers
                ],
                "completed_units": len(result_by_global),
                "total_units": len(unit_records),
                "effective_chunk_sizes": effective_chunks,
                "max_restoration_errors": restoration_errors,
            },
        )
        print(f"completed: {len(layer_records)}/{len(layer_records)}")
        print(f"effective chunk size: {effective_chunk}")
        print(f"state restoration max error: {restoration_error:.6g}")

    final_rows = sorted(
        result_by_global.values(), key=lambda row: int(row["global_index"])
    )
    assert len(final_rows) == len(unit_info)
    assert len({
        (row["layer"], int(row["unit_index"])) for row in final_rows
    }) == len(unit_info)
    if [int(row["global_index"]) for row in final_rows] != list(range(len(unit_info))):
        raise AssertionError("final result global_index is incomplete or reordered")

    ablation_effect = np.asarray(
        [float(row["ablation_logit_deviation"]) for row in final_rows],
        dtype=np.float32,
    )  # [N]
    np.savez(
        output_dir / "unit_ablation_effect.npz",
        V=descriptor.detach().cpu().float().numpy(),  # [N,3]
        ablation_effect=ablation_effect,
        global_index=np.arange(len(final_rows), dtype=np.int64),
    )

    parameter_hash_after = hash_model_parameters(model)
    if parameter_hash_after != parameter_hash_before:
        raise AssertionError("learned model parameters changed during the probe")
    metadata.update(
        {
            "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "units_evaluated": len(final_rows),
            "attention_heads": attention_units,
            "ffn_neurons": mlp_units,
            "model_parameter_sha256_after_ablation": parameter_hash_after,
            "parameter_state_verified_unchanged": True,
            "effective_chunk_sizes": effective_chunks,
        }
    )
    _write_json(output_dir / "run_metadata.json", metadata)

    print("=" * 72)
    print("Full descriptor--ablation probe complete")
    print(f"Units evaluated: {len(final_rows)}")
    print(f"Attention heads: {attention_units}")
    print(f"FFN neurons: {mlp_units}")
    print("Learned parameter state verified unchanged")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-unit descriptor--ablation consistency probe"
    )
    parser.add_argument("--gpu", default=None, help="visible physical GPU IDs")
    parser.add_argument("--model", default="swintrans")
    parser.add_argument(
        "--checkpoint_path",
        default="/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt",
    )
    parser.add_argument(
        "--calib_split",
        default="/data/jixinye25/UCF101_Frame/train_rgb_split1.txt",
    )
    parser.add_argument(
        "--val_split",
        default="/data/jixinye25/UCF101_Frame/val_rgb_split1.txt",
    )
    parser.add_argument("--calib_batches", type=int, default=10)
    parser.add_argument("--calib_batch_size", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--probe_batch_size", type=int, default=1)
    parser.add_argument(
        "--ablation_chunk_size", type=int, default=8,
        help="runtime/memory control; CUDA OOM halves it without skipping units",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output_dir", default="descriptor_ablation_validation"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = {
        "calib_batches": args.calib_batches,
        "calib_batch_size": args.calib_batch_size,
        "num_classes": args.num_classes,
        "videos_per_class": args.videos_per_class,
        "probe_batch_size": args.probe_batch_size,
        "ablation_chunk_size": args.ablation_chunk_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.workers < 0:
        parser.error(f"invalid non-positive arguments: {invalid}")
    return args


if __name__ == "__main__":
    run_probe(parse_args())

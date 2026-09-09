"""Validate protected representatives against cluster members and removed units.

This script only consumes exported descriptor-cluster protection artifacts. It
does not run clustering, alter labels, prune a model, or train a model.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from representative_redundancy_cam_utils import (
    cosine_similarity,
    normalize_unit_maps,
    parse_unit_id,
    read_csv_rows,
    representative_cluster_metrics,
    select_cluster_members,
    select_representative_clusters,
    write_csv_rows,
    write_json,
)


LOGGER = logging.getLogger("function_representative_cam")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare each protected descriptor-cluster representative with "
            "cluster members and units removed by the pruning budget."
        )
    )
    parser.add_argument("--model_builder", required=True)
    parser.add_argument("--loader_builder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unit_redundancy_csv", required=True)
    parser.add_argument("--cluster_redundancy_csv", required=True)
    parser.add_argument("--formal_descriptors_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_clusters", type=int, default=5)
    parser.add_argument(
        "--cluster_ids",
        default="",
        help="Optional comma-separated descriptor cluster IDs.",
    )
    parser.add_argument("--members_per_cluster", type=int, default=6)
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument(
        "--response_mode",
        choices=("activation", "discriminative"),
        default="activation",
    )
    parser.add_argument(
        "--target_mode",
        choices=("predicted", "ground_truth"),
        default="predicted",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--probe_batch_size", type=int, default=1)
    parser.add_argument("--probe_split", default="val")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="")
    return parser


def import_callable(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError(
            f"Builder {specification!r} must use module:function format"
        )
    module_name, attribute_name = specification.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(value):
        raise TypeError(f"{specification!r} is not callable")
    return value


def call_builder(builder: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    signature = inspect.signature(builder)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        dict(arguments)
        if accepts_kwargs
        else {
            key: value
            for key, value in arguments.items()
            if key in signature.parameters
        }
    )
    return builder(**selected)


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_dir / "representative_redundancy_cam.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        return extract_logits(model_output[0])
    if isinstance(model_output, Mapping):
        for key in ("logits", "output", "predictions"):
            if key in model_output:
                return extract_logits(model_output[key])
    raise TypeError("model_builder must produce tensor logits")


def unpack_probe_batch(
    batch: Any,
    generated_index_start: int,
) -> tuple[torch.Tensor, torch.Tensor | None, list[Any]]:
    if isinstance(batch, Mapping):
        video = batch.get("video", batch.get("input", batch.get("images")))
        target = batch.get("target", batch.get("label"))
        indices = batch.get("index", batch.get("indices", batch.get("video_id")))
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("probe loader yielded an empty batch")
        video = batch[0]
        target = batch[1] if len(batch) > 1 else None
        indices = batch[2] if len(batch) > 2 else None
    else:
        video, target, indices = batch, None, None
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("probe videos must have shape [B,C,T,H,W]")
    if target is not None and not torch.is_tensor(target):
        target = torch.as_tensor(target)
    if indices is None:
        index_list = list(
            range(generated_index_start, generated_index_start + video.shape[0])
        )
    elif torch.is_tensor(indices):
        index_list = indices.detach().cpu().reshape(-1).tolist()
    elif isinstance(indices, np.ndarray):
        index_list = indices.reshape(-1).tolist()
    elif isinstance(indices, (tuple, list)):
        index_list = list(indices)
    else:
        index_list = [indices]
    if len(index_list) != video.shape[0]:
        raise ValueError("probe video IDs are not batch-aligned")
    return video, target, index_list


class ActivationResponseCollector:
    """Collect selected head/neuron response maps with one hook per layer."""

    def __init__(
        self,
        model: torch.nn.Module,
        selected_unit_ids: Sequence[str],
        response_mode: str,
    ) -> None:
        self.model = model
        self.response_mode = response_mode
        self.spec_by_layer: dict[str, dict[str, list[tuple[int, str]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for unit_id in selected_unit_ids:
            layer, unit_type, local_index = parse_unit_id(unit_id)
            self.spec_by_layer[layer][unit_type].append(
                (local_index, str(unit_id))
            )
        self.handles: list[Any] = []
        self._captures: dict[str, tuple[torch.Tensor, Any, str]] = {}
        self._register()

    @staticmethod
    def _window_reverse(
        windows: torch.Tensor,
        geometry: Mapping[str, Any],
    ) -> torch.Tensor:
        batch = int(geometry["batch_size"])
        depth = int(geometry["padded_depth"])
        height = int(geometry["padded_height"])
        width = int(geometry["padded_width"])
        wd, wh, ww = tuple(geometry["window_size"])
        channels = windows.shape[-1]
        response = windows.reshape(
            batch,
            depth // wd,
            height // wh,
            width // ww,
            wd,
            wh,
            ww,
            channels,
        )
        response = response.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        response = response.reshape(batch, depth, height, width, channels)
        shift = tuple(geometry["shift_size"])
        if any(value > 0 for value in shift):
            response = torch.roll(response, shifts=shift, dims=(1, 2, 3))
        response = response[
            :,
            : int(geometry["depth"]),
            : int(geometry["height"]),
            : int(geometry["width"]),
            :,
        ]
        return response.permute(0, 4, 1, 2, 3).contiguous()

    @classmethod
    def _extract_attention(
        cls,
        tensor: torch.Tensor,
        source_module: torch.nn.Module,
        local_indices: Sequence[int],
        discriminative: bool,
    ) -> torch.Tensor:
        geometry = getattr(source_module, "_pruning_geometry", None)
        if geometry is None:
            raise RuntimeError("attention response geometry is unavailable")
        num_heads = int(source_module.num_heads)
        head_dim = int(source_module.head_dim)
        selected = tensor.reshape(
            tensor.shape[0], tensor.shape[1], num_heads, head_dim
        )[:, :, list(local_indices), :]
        response = (
            torch.relu(selected).mean(dim=-1)
            if discriminative
            else selected.abs().mean(dim=-1)
        )
        return cls._window_reverse(response, geometry)

    @staticmethod
    def _extract_mlp(
        tensor: torch.Tensor,
        local_indices: Sequence[int],
        discriminative: bool,
    ) -> torch.Tensor:
        if tensor.ndim != 5:
            raise ValueError("MLP fc2 input must have shape [B,T,H,W,U]")
        selected = tensor[..., list(local_indices)]
        response = torch.relu(selected) if discriminative else selected.abs()
        return response.permute(0, 4, 1, 2, 3).contiguous()

    def _make_hook(
        self,
        layer_name: str,
        source_module: torch.nn.Module,
        unit_type: str,
    ) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            tensor = inputs[0]
            if self.response_mode == "discriminative":
                if not tensor.requires_grad:
                    tensor.requires_grad_(True)
                tensor.retain_grad()
            self._captures[layer_name] = (tensor, source_module, unit_type)

        return hook

    def _register(self) -> None:
        modules = dict(self.model.named_modules())
        for layer_name, type_specs in self.spec_by_layer.items():
            if layer_name not in modules:
                raise KeyError(f"exported layer {layer_name!r} is not in the model")
            if len(type_specs) != 1:
                raise ValueError(f"layer {layer_name!r} mixes unit types")
            unit_type = next(iter(type_specs))
            source_module = modules[layer_name]
            if unit_type == "head":
                target_module = source_module.proj
                upper_bound = int(source_module.num_heads)
            else:
                target_module = source_module.fc2
                upper_bound = int(
                    getattr(
                        source_module,
                        "original_hidden_features",
                        source_module.fc2.in_features,
                    )
                )
            indices = [item[0] for item in type_specs[unit_type]]
            if min(indices) < 0 or max(indices) >= upper_bound:
                raise IndexError(f"unit index is outside layer {layer_name!r}")
            self.handles.append(
                target_module.register_forward_pre_hook(
                    self._make_hook(layer_name, source_module, unit_type)
                )
            )
        LOGGER.info(
            "Registered %d hooks for %d units",
            len(self.handles),
            sum(
                len(entries)
                for type_specs in self.spec_by_layer.values()
                for entries in type_specs.values()
            ),
        )

    def clear(self) -> None:
        self._captures.clear()

    def finish_discriminative(self) -> None:
        if self.response_mode != "discriminative":
            return
        for layer_name, (activation, _, _) in self._captures.items():
            if activation.grad is None:
                raise RuntimeError(f"no activation gradient for {layer_name!r}")

    def export_resized(
        self,
        output_shape: Sequence[int],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        maps: dict[str, np.ndarray] = {}
        energy: dict[str, np.ndarray] = {}
        for layer_name, type_specs in self.spec_by_layer.items():
            if layer_name not in self._captures:
                raise RuntimeError(f"hook for {layer_name!r} did not fire")
            unit_type = next(iter(type_specs))
            entries = type_specs[unit_type]
            activation, source_module, _ = self._captures[layer_name]
            source_tensor = (
                activation.detach() * activation.grad.detach()
                if self.response_mode == "discriminative"
                else activation
            )
            for start in range(0, len(entries), 8):
                selected_entries = entries[start : start + 8]
                local_indices = [item[0] for item in selected_entries]
                response = (
                    self._extract_attention(
                        source_tensor,
                        source_module,
                        local_indices,
                        discriminative=self.response_mode == "discriminative",
                    )
                    if unit_type == "head"
                    else self._extract_mlp(
                        source_tensor,
                        local_indices,
                        discriminative=self.response_mode == "discriminative",
                    )
                )
                if not torch.isfinite(response).all():
                    raise FloatingPointError(
                        f"non-finite response in {layer_name!r}"
                    )
                raw_energy = response.float().mean(dim=(2, 3, 4)).cpu().numpy()
                resized = torch_functional.interpolate(
                    response.float(),
                    size=tuple(int(value) for value in output_shape),
                    mode="trilinear",
                    align_corners=False,
                ).detach().cpu().numpy()
                for offset, (_, unit_id) in enumerate(selected_entries):
                    maps[unit_id] = resized[:, offset]
                    energy[unit_id] = raw_energy[:, offset]
        return maps, energy

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def parse_cluster_ids(specification: str) -> list[int]:
    if not specification.strip():
        return []
    values = [int(item.strip()) for item in specification.split(",")]
    if len(set(values)) != len(values):
        raise ValueError("cluster_ids must not contain duplicates")
    return values


def load_analysis_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    unit_rows = read_csv_rows(args.unit_redundancy_csv)
    cluster_rows = read_csv_rows(args.cluster_redundancy_csv)
    formal = np.load(args.formal_descriptors_npz, allow_pickle=True)
    required = {"unit_ids", "descriptors_raw", "descriptors_normalized", "unit_info"}
    missing = required - set(formal.files)
    if missing:
        raise KeyError(f"formal_descriptors.npz is missing {sorted(missing)}")
    unit_ids = formal["unit_ids"].astype(str)
    row_ids = [str(row["unit_id"]) for row in unit_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("unit_redundancy.csv contains duplicate unit IDs")
    if set(row_ids) != set(unit_ids.tolist()):
        raise ValueError("unit redundancy rows are not aligned with formal IDs")
    selected_clusters = select_representative_clusters(
        cluster_rows,
        args.num_clusters,
        parse_cluster_ids(args.cluster_ids),
    )
    contexts = []
    for order, cluster_row in enumerate(selected_clusters):
        cluster_id = int(float(cluster_row["cluster_id"]))
        context = select_cluster_members(
            unit_rows,
            cluster_id,
            args.members_per_cluster,
        )
        context["order"] = order
        context["cluster_row"] = cluster_row
        contexts.append(context)
    return {
        "unit_ids": unit_ids,
        "unit_rows": unit_rows,
        "cluster_rows": cluster_rows,
        "contexts": contexts,
    }


def _stack_maps(
    unit_ids: Sequence[str],
    batch_position: int,
    maps_by_unit: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [maps_by_unit[str(unit_id)][batch_position] for unit_id in unit_ids],
        dtype=np.float32,
    )


def _display_frame(video: np.ndarray, time_index: int) -> np.ndarray:
    frame = np.asarray(video[:, time_index], dtype=np.float32).transpose(1, 2, 0)
    low = float(frame.min())
    high = float(frame.max())
    if high - low <= 1e-8:
        return np.zeros_like(frame)
    return np.clip((frame - low) / (high - low), 0.0, 1.0)


def plot_representative_comparison(
    path: Path,
    video: np.ndarray,
    representative: np.ndarray,
    cluster_prototype: np.ndarray,
    member_maps: np.ndarray,
    title: str,
    comparison_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temporal_energy = representative.mean(axis=(1, 2)) + cluster_prototype.mean(
        axis=(1, 2)
    )
    time_index = int(np.argmax(temporal_energy))
    frame = _display_frame(video, time_index)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title("Probe frame")
    axes[0, 1].imshow(representative[time_index], cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("Protected representative")
    axes[0, 2].imshow(
        cluster_prototype[time_index],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title(f"{comparison_label} prototype")
    axes[1, 0].imshow(frame)
    axes[1, 0].imshow(
        representative[time_index],
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Representative overlay")
    axes[1, 1].imshow(frame)
    axes[1, 1].imshow(
        cluster_prototype[time_index],
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title(f"{comparison_label} overlay")
    axes[1, 2].imshow(member_maps[:, time_index].mean(axis=0), cmap="viridis")
    axes[1, 2].set_title(
        f"Mean of {member_maps.shape[0]} {comparison_label.lower()}s"
    )
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def pairwise_video_cosine(maps: Sequence[np.ndarray]) -> float:
    values = []
    for left in range(len(maps)):
        for right in range(left + 1, len(maps)):
            values.append(cosine_similarity(maps[left], maps[right]))
    return float(np.mean(values)) if values else 1.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir)
    set_deterministic_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    artifacts = load_analysis_artifacts(args)
    contexts = artifacts["contexts"]
    selected_rows = []
    selected_unit_ids = set()
    for context in contexts:
        representative_id = str(context["representative"]["unit_id"])
        member_ids = [
            str(row["unit_id"]) for row in context["cluster_members"]
        ]
        removed_ids = [
            str(row["unit_id"]) for row in context["removed_members"]
        ]
        selected_unit_ids.add(representative_id)
        selected_unit_ids.update(member_ids)
        selected_unit_ids.update(removed_ids)
        cluster_row = context["cluster_row"]
        selected_rows.append(
            {
                "cluster_order": context["order"] + 1,
                "cluster_id": context["cluster_id"],
                "cluster_size": cluster_row["size"],
                "mean_uniqueness": cluster_row.get("mean_uniqueness", ""),
                "representative_id": representative_id,
                "sampled_cluster_member_count": len(member_ids),
                "cluster_member_count": context["cluster_member_count"],
                "removed_member_count": len(removed_ids),
                "cluster_member_ids": ";".join(member_ids),
                "removed_unit_ids": ";".join(removed_ids),
            }
        )
    write_csv_rows(output_dir / "selected_representative_clusters.csv", selected_rows)

    model_builder = import_callable(args.model_builder)
    loader_builder = import_callable(args.loader_builder)
    model_result = call_builder(
        model_builder,
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "response_mode": args.response_mode,
            "target_mode": args.target_mode,
            "seed": args.seed,
        },
    )
    model = model_result[0] if isinstance(model_result, tuple) else model_result
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_builder must return a torch.nn.Module")
    model = model.to(device).eval()
    loader = call_builder(
        loader_builder,
        {
            "videos": args.videos,
            "batch_size": args.probe_batch_size,
            "split": args.probe_split,
            "seed": args.seed,
            "num_workers": args.num_workers,
        },
    )
    collector = ActivationResponseCollector(
        model,
        sorted(selected_unit_ids),
        args.response_mode,
    )
    per_video_rows = []
    per_unit_rows = []
    cuda_rows = []
    video_ids = []
    representative_maps = [[] for _ in contexts]
    cluster_prototypes = [[] for _ in contexts]
    processed = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for batch in loader:
            if processed >= args.videos:
                break
            videos, targets, batch_ids = unpack_probe_batch(batch, processed)
            remaining = args.videos - processed
            videos = videos[:remaining].to(device, non_blocking=True)
            batch_ids = batch_ids[:remaining]
            if targets is not None:
                targets = targets[:remaining].to(device).long()
            collector.clear()
            model.zero_grad(set_to_none=True)
            gradient_enabled = args.response_mode == "discriminative"
            with torch.set_grad_enabled(gradient_enabled):
                logits = extract_logits(model(videos))
                if gradient_enabled:
                    if args.target_mode == "ground_truth":
                        if targets is None:
                            raise ValueError(
                                "ground_truth target mode requires labels"
                            )
                        selected_targets = targets.reshape(-1)
                    else:
                        selected_targets = logits.detach().argmax(dim=1)
                    logits.gather(1, selected_targets[:, None]).sum().backward()
                    collector.finish_discriminative()
            maps_by_unit, energy_by_unit = collector.export_resized(videos.shape[2:])
            video_cpu = videos.detach().cpu().numpy()

            for batch_position, video_id in enumerate(batch_ids):
                video_order = processed
                video_ids.append(str(video_id))
                for context_index, context in enumerate(contexts):
                    representative_id = str(
                        context["representative"]["unit_id"]
                    )
                    member_ids = [
                        str(row["unit_id"])
                        for row in context["cluster_members"]
                    ]
                    removed_ids = [
                        str(row["unit_id"])
                        for row in context["removed_members"]
                    ]
                    all_ids = list(
                        dict.fromkeys(
                            [representative_id, *member_ids, *removed_ids]
                        )
                    )
                    normalized = normalize_unit_maps(
                        _stack_maps(all_ids, batch_position, maps_by_unit)
                    )
                    maps_by_id = dict(zip(all_ids, normalized))
                    representative = maps_by_id[representative_id]
                    member_maps = np.asarray(
                        [maps_by_id[unit_id] for unit_id in member_ids]
                    )
                    metrics = representative_cluster_metrics(
                        representative,
                        member_maps,
                    )
                    prototype = metrics.pop("member_prototype")
                    removed_metrics = None
                    removed_maps = np.empty(
                        (0, *representative.shape),
                        dtype=np.float32,
                    )
                    removed_prototype = np.empty(
                        (0, *representative.shape),
                        dtype=np.float32,
                    )
                    if removed_ids:
                        removed_maps = np.asarray(
                            [maps_by_id[unit_id] for unit_id in removed_ids]
                        )
                        removed_metrics = representative_cluster_metrics(
                            representative,
                            removed_maps,
                        )
                        removed_prototype = removed_metrics.pop(
                            "member_prototype"
                        )
                    representative_maps[context_index].append(
                        representative.astype(np.float16)
                    )
                    cluster_prototypes[context_index].append(
                        prototype.astype(np.float16)
                    )
                    per_video_rows.append(
                        {
                            "cluster_order": context_index + 1,
                            "cluster_id": context["cluster_id"],
                            "video_order": video_order,
                            "video_id": str(video_id),
                            "representative_id": representative_id,
                            "cluster_member_count": len(member_ids),
                            "removed_unit_count": len(removed_ids),
                            "representative_to_cluster_prototype_cosine": metrics[
                                "representative_to_prototype_cosine"
                            ],
                            "representative_to_cluster_prototype_soft_iou": metrics[
                                "representative_to_prototype_soft_iou"
                            ],
                            "mean_cluster_member_cosine": metrics[
                                "mean_member_cosine"
                            ],
                            "mean_cluster_member_soft_iou": metrics[
                                "mean_member_soft_iou"
                            ],
                            "mean_removed_unit_cosine": (
                                removed_metrics["mean_member_cosine"]
                                if removed_metrics is not None
                                else None
                            ),
                            "mean_removed_unit_soft_iou": (
                                removed_metrics["mean_member_soft_iou"]
                                if removed_metrics is not None
                                else None
                            ),
                            "hotspot_centroid_distance": metrics[
                                "hotspot_centroid_distance"
                            ],
                            "representative_energy": float(
                                energy_by_unit[representative_id][batch_position]
                            ),
                            "cluster_member_energy_mean": float(
                                np.mean(
                                    [
                                        energy_by_unit[unit_id][batch_position]
                                        for unit_id in member_ids
                                    ]
                                )
                            ),
                        }
                    )
                    for row, cosine, iou in zip(
                        context["cluster_members"],
                        metrics["representative_to_member_cosines"],
                        metrics["representative_to_member_soft_ious"],
                    ):
                        per_unit_rows.append(
                            {
                                "cluster_id": context["cluster_id"],
                                "video_order": video_order,
                                "video_id": str(video_id),
                                "representative_id": representative_id,
                                "comparison_set": "cluster_member",
                                "compared_unit_id": row["unit_id"],
                                "removed": str(row.get("kept", "")).lower()
                                in {"false", "0"},
                                "function_uniqueness": row[
                                    "function_uniqueness"
                                ],
                                "descriptor_importance": row[
                                    "descriptor_importance"
                                ],
                                "cam_cosine_to_representative": cosine,
                                "cam_soft_iou_to_representative": iou,
                            }
                        )
                    if removed_metrics is not None:
                        for row, cosine, iou in zip(
                            context["removed_members"],
                            removed_metrics[
                                "representative_to_member_cosines"
                            ],
                            removed_metrics[
                                "representative_to_member_soft_ious"
                            ],
                        ):
                            per_unit_rows.append(
                                {
                                    "cluster_id": context["cluster_id"],
                                    "video_order": video_order,
                                    "video_id": str(video_id),
                                    "representative_id": representative_id,
                                    "comparison_set": "removed_unit",
                                    "compared_unit_id": row["unit_id"],
                                    "removed": True,
                                    "function_uniqueness": row[
                                        "function_uniqueness"
                                    ],
                                    "descriptor_importance": row[
                                        "descriptor_importance"
                                    ],
                                    "cam_cosine_to_representative": cosine,
                                    "cam_soft_iou_to_representative": iou,
                                }
                            )
                    video_dir = (
                        output_dir
                        / f"cluster_{int(context['cluster_id']):03d}"
                        / f"video_{video_order + 1:02d}"
                    )
                    video_dir.mkdir(parents=True, exist_ok=True)
                    plot_representative_comparison(
                        video_dir / "representative_vs_cluster_members.png",
                        video_cpu[batch_position],
                        representative,
                        prototype,
                        member_maps,
                        (
                            f"Descriptor cluster {context['cluster_id']} | "
                            f"video {video_id}"
                        ),
                        "Cluster member",
                    )
                    if removed_metrics is not None:
                        plot_representative_comparison(
                            video_dir / "representative_vs_removed_units.png",
                            video_cpu[batch_position],
                            representative,
                            removed_prototype,
                            removed_maps,
                            (
                                f"Removed units from cluster "
                                f"{context['cluster_id']} | video {video_id}"
                            ),
                            "Removed unit",
                        )
                    np.savez_compressed(
                        video_dir / "response_maps.npz",
                        video_id=np.asarray(str(video_id)),
                        representative_id=np.asarray(representative_id),
                        representative_map=representative.astype(np.float16),
                        cluster_member_ids=np.asarray(member_ids),
                        cluster_member_maps=member_maps.astype(np.float16),
                        cluster_member_prototype=prototype.astype(np.float16),
                        removed_unit_ids=np.asarray(removed_ids),
                        removed_unit_prototype=removed_prototype.astype(np.float16),
                    )
                processed += 1
                if processed >= args.videos:
                    break
            if torch.cuda.is_available():
                cuda_rows.append(
                    {
                        "videos_processed": processed,
                        "cuda_available": True,
                        "allocated_mb": float(
                            torch.cuda.memory_allocated(device) / 2**20
                        ),
                        "reserved_mb": float(
                            torch.cuda.memory_reserved(device) / 2**20
                        ),
                        "max_allocated_mb": float(
                            torch.cuda.max_memory_allocated(device) / 2**20
                        ),
                    }
                )
                torch.cuda.empty_cache()
            LOGGER.info("Processed %d/%d videos", processed, args.videos)
    finally:
        collector.close()

    if processed != args.videos:
        raise RuntimeError(
            f"probe loader ended after {processed} of {args.videos} videos"
        )
    if not cuda_rows:
        cuda_rows.append(
            {
                "videos_processed": processed,
                "cuda_available": False,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
            }
        )
    write_csv_rows(
        output_dir / "per_video_representative_cam_metrics.csv",
        per_video_rows,
    )
    write_csv_rows(
        output_dir / "per_unit_representative_cam_metrics.csv",
        per_unit_rows,
    )
    write_csv_rows(output_dir / "cuda_memory_log.csv", cuda_rows)

    cluster_summaries = []
    for index, context in enumerate(contexts):
        cluster_rows = [
            row
            for row in per_video_rows
            if int(row["cluster_id"]) == int(context["cluster_id"])
        ]
        cluster_summaries.append(
            {
                "cluster_id": context["cluster_id"],
                "representative_id": context["representative"]["unit_id"],
                "sampled_cluster_member_count": len(
                    context["cluster_members"]
                ),
                "removed_unit_count": len(context["removed_members"]),
                "mean_representative_to_cluster_prototype_cosine": float(
                    np.mean(
                        [
                            row["representative_to_cluster_prototype_cosine"]
                            for row in cluster_rows
                        ]
                    )
                ),
                "mean_representative_to_cluster_member_cosine": float(
                    np.mean(
                        [
                            row["mean_cluster_member_cosine"]
                            for row in cluster_rows
                        ]
                    )
                ),
                "mean_representative_to_removed_unit_cosine": (
                    float(
                        np.mean(
                            [
                                row["mean_removed_unit_cosine"]
                                for row in cluster_rows
                            ]
                        )
                    )
                    if context["removed_members"]
                    else None
                ),
                "representative_cross_video_cosine": pairwise_video_cosine(
                    representative_maps[index]
                ),
                "cluster_prototype_cross_video_cosine": pairwise_video_cosine(
                    cluster_prototypes[index]
                ),
            }
        )
    write_csv_rows(output_dir / "per_cluster_cam_summary.csv", cluster_summaries)
    overall_cosines = [
        float(row["mean_cluster_member_cosine"])
        for row in per_video_rows
    ]
    overall_ious = [
        float(row["mean_cluster_member_soft_iou"])
        for row in per_video_rows
    ]
    removed_cosines = [
        float(row["mean_removed_unit_cosine"])
        for row in per_video_rows
        if row["mean_removed_unit_cosine"] is not None
    ]
    summary = {
        "status": "completed",
        "validation_scope": (
            "representative vs descriptor-cluster members and removed units"
        ),
        "descriptor_labels_modified": False,
        "clustering_executed": False,
        "pruning_or_training_executed": False,
        "same_videos_verified": True,
        "response_mode": args.response_mode,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "video_ids": video_ids,
        "cluster_count": len(contexts),
        "selected_unit_count": len(selected_unit_ids),
        "mean_representative_to_cluster_member_cosine": float(
            np.mean(overall_cosines)
        ),
        "mean_representative_to_cluster_member_soft_iou": float(
            np.mean(overall_ious)
        ),
        "mean_representative_to_removed_unit_cosine": (
            float(np.mean(removed_cosines)) if removed_cosines else None
        ),
        "cluster_summaries": cluster_summaries,
        "cuda_peak_allocated_mb": max(
            float(row["max_allocated_mb"]) for row in cuda_rows
        ),
        "source_artifacts": {
            "unit_redundancy_csv": str(
                Path(args.unit_redundancy_csv).resolve()
            ),
            "cluster_redundancy_csv": str(
                Path(args.cluster_redundancy_csv).resolve()
            ),
            "formal_descriptors_npz": str(
                Path(args.formal_descriptors_npz).resolve()
            ),
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    removed_summary_line = (
        f"- Mean representative/removed CAM cosine: "
        f"{float(np.mean(removed_cosines)):.6f}\n"
        if removed_cosines
        else "- Mean representative/removed CAM cosine: unavailable\n"
    )
    (output_dir / "README_results.md").write_text(
        (
            "# Function Representative CAM\n\n"
            "This validation compares each protected descriptor-cluster "
            "representative with sampled cluster members and actually removed "
            "units on exactly the same probe videos. It does not create or "
            "modify cluster labels and does not execute pruning or training.\n\n"
            f"- Clusters: {len(contexts)}\n"
            f"- Videos: {processed}\n"
            f"- Mean representative/member CAM cosine: "
            f"{float(np.mean(overall_cosines)):.6f}\n"
            + removed_summary_line
            + f"- Mean soft IoU: {float(np.mean(overall_ious)):.6f}\n"
        ),
        encoding="utf-8",
    )
    LOGGER.info("Validation complete: %s", output_dir)
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

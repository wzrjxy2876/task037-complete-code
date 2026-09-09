#!/usr/bin/env python3
"""Probe-only normal/frozen/shuffled sanity validation for Task010 TDD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task010_tdd_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from MC import InteractionPruner
import probe_video_specific_third_descriptor as task009


EPSILON = 1e-8
CONDITIONS = ("normal", "frozen", "shuffle")
OUTPUT_FIELDS = (
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_dyn_normal",
    "D_dyn_frozen",
    "D_dyn_shuffle",
    "freeze_ratio",
    "shuffle_sensitivity",
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _condition_transform(
    condition: str, shuffle_index: torch.Tensor
):
    def transform(videos: torch.Tensor) -> torch.Tensor:
        return task009.transform_video_condition(
            videos,
            condition,
            shuffle_index if condition == "shuffle" else None,
        )

    return transform


def _collect_condition(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    condition: str,
    shuffle_index: torch.Tensor,
    batches: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return per-layer ``D_dyn [U]`` while retaining statistics on CUDA."""
    pruner = InteractionPruner(model, descriptor_variant="dynamic3d")
    pruner.collect_activation_samples = False
    pruner.run_calibration(
        loader,
        device=device,
        num_batches=batches,
        video_transform=_condition_transform(condition, shuffle_index),
    )
    result = {}
    for layer_name in pruner.ordered_layer_names:
        if layer_name not in pruner.temporal_dynamicity_scores:
            raise RuntimeError(f"missing TDD scores for {layer_name}")
        # Each entry is [B,U]; averaging all selected videos yields [U].
        result[layer_name] = torch.cat(
            pruner.temporal_dynamicity_scores[layer_name], dim=0
        ).mean(dim=0)
    ordered_layers = list(pruner.ordered_layer_names)
    del pruner
    torch.cuda.empty_cache()
    return result, ordered_layers


def _finite_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        numerator / (denominator + EPSILON),
        nan=0.0,
        posinf=torch.finfo(numerator.dtype).max,
        neginf=0.0,
    )


def run_probe(args: argparse.Namespace) -> None:
    visible_gpus = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    if not torch.cuda.is_available():
        raise RuntimeError("TDD sanity probe requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    task007_csv = Path(args.task007_csv)
    probe_samples = Path(args.probe_samples)
    checkpoint = Path(args.checkpoint_path)
    validation_split = Path(args.val_split)
    for required in (task007_csv, probe_samples, checkpoint, validation_split):
        if not required.is_file():
            raise FileNotFoundError(required)

    task007 = task009._load_task007_helpers()
    task007.set_seed(args.seed)
    rows = task009.read_task007_rows(task007_csv)
    sample_rows = task009.read_probe_samples(probe_samples)
    if args.sanity_videos > len(sample_rows):
        raise ValueError(
            f"sanity_videos={args.sanity_videos} exceeds {len(sample_rows)} samples"
        )
    split_rows = task007.parse_split_records(validation_split)
    task009.validate_probe_samples_against_split(sample_rows, split_rows)

    get_dataset, _, model_class = task007._load_repository_components()
    model = task007._build_model(model_class, checkpoint, device)
    layer_rows = task009.validate_unit_ordering(rows, model=model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    selected = sample_rows[: args.sanity_videos]
    sample_indices = [int(row["sample_index"]) for row in selected]
    base_loader = get_dataset(str(validation_split), 1)

    def new_loader():
        return task009._make_loader(
            task007,
            base_loader,
            sample_indices,
            args.batch_size,
            args.workers,
            args.seed,
        )

    first_batch = next(iter(new_loader()))
    first_videos, _ = task009._extract_videos_targets(first_batch)
    temporal_size = int(first_videos.shape[2])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    shuffle_index = torch.randperm(temporal_size, generator=generator)
    del first_batch, first_videos

    batches = math.ceil(args.sanity_videos / args.batch_size)
    condition_scores = {}
    reference_order = None
    for condition in CONDITIONS:
        scores, order = _collect_condition(
            model,
            new_loader(),
            device,
            condition,
            shuffle_index,
            batches,
        )
        if reference_order is None:
            reference_order = order
        elif order != reference_order:
            raise RuntimeError("layer ordering changed across sanity conditions")
        condition_scores[condition] = scores

    normal_values = []
    frozen_values = []
    shuffle_values = []
    output_rows = []
    for layer_name, expected_rows in layer_rows.items():
        for condition in CONDITIONS:
            values = condition_scores[condition].get(layer_name)
            if values is None or values.numel() != len(expected_rows):
                raise RuntimeError(f"TDD unit count mismatch for {layer_name}")
        normal = condition_scores["normal"][layer_name]
        frozen = condition_scores["frozen"][layer_name]
        shuffled = condition_scores["shuffle"][layer_name]
        freeze_ratio = _finite_ratio(frozen, normal)
        shuffle_sensitivity = _finite_ratio((shuffled - normal).abs(), normal)

        normal_values.append(normal)
        frozen_values.append(frozen)
        shuffle_values.append(shuffled)
        cpu_columns = torch.stack(
            [normal, frozen, shuffled, freeze_ratio, shuffle_sensitivity], dim=1
        ).double().cpu().tolist()
        for row, values in zip(expected_rows, cpu_columns):
            output_rows.append({
                "global_index": int(row["global_index"]),
                "layer": layer_name,
                "unit_type": row["unit_type"],
                "unit_index": int(row["unit_index"]),
                "D_dyn_normal": f"{values[0]:.17g}",
                "D_dyn_frozen": f"{values[1]:.17g}",
                "D_dyn_shuffle": f"{values[2]:.17g}",
                "freeze_ratio": f"{values[3]:.17g}",
                "shuffle_sensitivity": f"{values[4]:.17g}",
            })

    normal = torch.cat(normal_values)
    frozen = torch.cat(frozen_values)
    shuffled = torch.cat(shuffle_values)
    freeze_ratio = _finite_ratio(frozen, normal)
    shuffle_sensitivity = _finite_ratio((shuffled - normal).abs(), normal)
    quantiles = torch.quantile(
        freeze_ratio, torch.tensor([0.25, 0.5, 0.75], device=device)
    )
    summary = {
        "units": int(normal.numel()),
        "videos": args.sanity_videos,
        "seed": args.seed,
        "execution_device": "cuda:0",
        "visible_gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "median_freeze_ratio": float(quantiles[1].item()),
        "freeze_ratio_q25": float(quantiles[0].item()),
        "freeze_ratio_q75": float(quantiles[2].item()),
        "freeze_ratio_iqr": float((quantiles[2] - quantiles[0]).item()),
        "fraction_freeze_ratio_below_0.5": float(
            (freeze_ratio < 0.5).float().mean().item()
        ),
        "median_shuffle_sensitivity": float(
            torch.median(shuffle_sensitivity).item()
        ),
        "shuffle_temporal_index": shuffle_index.tolist(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tdd_video_sanity.csv"
    summary_path = output_dir / "tdd_video_sanity_summary.json"
    figure_path = output_dir / "tdd_video_sanity.png"
    if not args.overwrite:
        existing = [path for path in (csv_path, summary_path, figure_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"sanity outputs already exist; use --overwrite: {existing}"
            )
    _atomic_csv(csv_path, output_rows)
    _atomic_json(summary_path, summary)

    plot_values = [
        normal.detach().cpu().numpy(),
        frozen.detach().cpu().numpy(),
        shuffled.detach().cpu().numpy(),
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.boxplot(plot_values, labels=["Normal", "Frozen", "Shuffle"], showfliers=False)
    axis.set_ylabel("Temporal dynamicity")
    axis.set_title("TDD video sanity check")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("=" * 64)
    print("TDD Video Sanity")
    print(f"Units: {summary['units']}; videos: {summary['videos']}")
    print(
        "Freeze ratio median/IQR: "
        f"{summary['median_freeze_ratio']:.6f} / "
        f"{summary['freeze_ratio_iqr']:.6f}"
    )
    print(
        "Fraction freeze ratio < 0.5: "
        f"{summary['fraction_freeze_ratio_below_0.5']:.6f}"
    )
    print(f"Output: {output_dir.resolve()}")
    print("=" * 64)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-first Task010 TDD sanity probe")
    parser.add_argument("--gpu", default=None)
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
        "--val_split", default="/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
    )
    parser.add_argument("--output_dir", default="tdd_pruning_validation")
    parser.add_argument("--sanity_videos", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.sanity_videos <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("sanity_videos/batch_size must be positive and workers nonnegative")
    return args


if __name__ == "__main__":
    run_probe(parse_args())

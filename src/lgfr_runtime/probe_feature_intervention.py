#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional Descriptor Feature-Intervention Probe (V4).

Scientific question
-------------------
For pruning-unit pairs with similar scalar descriptors, does a larger
Functional Descriptor distance correspond to a larger difference in the
internal representation changes caused by masking each unit?

This probe replaces final-logit intervention with representation intervention.

For each video x and unit u_i:

    h(x)      : baseline downstream representation
    h_-i(x)   : representation after masking u_i
    delta_i(x)= h(x) - h_-i(x)

For a pair (i,j), representation-intervention difference is measured by:

    d_dir(i,j) = mean_x [1 - cos(delta_i(x), delta_j(x))]

and

    d_mag(i,j) = mean_x [
        | ||delta_i(x)||_2 - ||delta_j(x)||_2 | /
        (||delta_i(x)||_2 + ||delta_j(x)||_2 + eps)
    ]

Primary hypothesis
------------------
scalar_close_function_far pairs should have larger representation-intervention
distance than scalar_close_function_close pairs.

Default feature target
----------------------
The script automatically captures the input of the final classifier Linear
layer. This is the last semantic representation before classification.

An explicit module can be selected with:

    --feature_module '<exact module name or regex>'
    --feature_capture input|output

For tensor outputs with more than two dimensions, the feature is reduced to
[B,D] by averaging all dimensions except batch and the last channel dimension.

Required project-side files
---------------------------
- probe_ctfrs_dynamic_function.py
- ucf101_videoswin_probe_adapter_v2.py

Python 3.9+.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

try:
    from probe_ctfrs_dynamic_function import (
        EPS,
        UnitLayerSpec,
        build_balanced_loader,
        discover_unit_layers,
        ensure_project_importable,
        load_model,
        set_seed,
        unwrap_logits,
    )
except Exception as exc:
    raise ImportError(
        "Place this script beside the fixed probe_ctfrs_dynamic_function.py."
    ) from exc


@dataclass(frozen=True)
class UnitKey:
    layer: str
    unit_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Functional Descriptor by feature intervention"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--representative_cases", required=True)
    parser.add_argument("--output_dir", default="./feature_intervention_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--frame_root", required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument(
        "--case_types",
        default=(
            "scalar_close_function_far,"
            "scalar_close_function_close,"
            "scalar_far_function_close"
        ),
    )
    parser.add_argument("--pairs_per_case_layer", type=int, default=8)
    parser.add_argument("--ablation_videos", type=int, default=9)

    parser.add_argument(
        "--feature_module",
        default="auto",
        help=(
            "'auto' captures the input of the final classifier Linear layer; "
            "otherwise provide an exact module name or regular expression."
        ),
    )
    parser.add_argument(
        "--feature_capture",
        choices=("input", "output"),
        default="input",
    )
    parser.add_argument(
        "--feature_reduce",
        choices=("auto", "flatten", "mean"),
        default="auto",
    )
    parser.add_argument(
        "--save_feature_deltas",
        action="store_true",
        help="Save every unit/video feature-delta vector as compressed NPZ.",
    )
    return parser.parse_args()


def parse_case_types(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def choose_cases(
    cases: pd.DataFrame,
    case_types: Sequence[str],
    pairs_per_case_layer: int,
) -> pd.DataFrame:
    required = {
        "layer",
        "unit_type",
        "case_type",
        "unit_i",
        "unit_j",
        "scalar_distance",
        "functional_distance",
    }
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"representative cases missing columns: {sorted(missing)}")

    selected_frames = []
    for (layer, case_type), table in cases.groupby(["layer", "case_type"]):
        if case_type not in case_types:
            continue
        if case_type == "scalar_close_function_far":
            table = table.sort_values(
                ["functional_distance", "scalar_distance"],
                ascending=[False, True],
            )
        elif case_type == "scalar_close_function_close":
            table = table.sort_values(
                ["functional_distance", "scalar_distance"],
                ascending=[True, True],
            )
        else:
            table = table.sort_values(
                ["functional_distance", "scalar_distance"],
                ascending=[True, False],
            )
        selected_frames.append(table.head(pairs_per_case_layer))

    if not selected_frames:
        raise ValueError("No representative pairs matched --case_types")

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.drop_duplicates(
        subset=["layer", "unit_i", "unit_j", "case_type"]
    ).reset_index(drop=True)
    selected.insert(0, "pair_id", np.arange(len(selected), dtype=int))
    return selected


def find_feature_module(
    model: nn.Module,
    requested: str,
) -> Tuple[str, nn.Module]:
    named = list(model.named_modules())

    if requested == "auto":
        linear_modules = [
            (name, module)
            for name, module in named
            if isinstance(module, nn.Linear)
        ]
        if not linear_modules:
            raise RuntimeError(
                "No nn.Linear classifier was found. Supply --feature_module."
            )
        # The last Linear is normally the classification projection.
        return linear_modules[-1]

    exact = dict(named)
    if requested in exact:
        return requested, exact[requested]

    pattern = re.compile(requested)
    matches = [(name, module) for name, module in named if pattern.search(name)]
    if not matches:
        raise KeyError(f"No module matched --feature_module={requested!r}")
    if len(matches) > 1:
        names = [name for name, _ in matches[:20]]
        raise ValueError(
            "feature_module matched multiple modules; use an exact name. "
            f"First matches: {names}"
        )
    return matches[0]


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def reduce_feature(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert arbitrary feature tensor to [B,D]."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor

    if mode == "flatten":
        return tensor.reshape(tensor.shape[0], -1)

    if mode == "mean":
        # Preserve the final channel/embedding dimension.
        reduce_dims = tuple(range(1, tensor.ndim - 1))
        if reduce_dims:
            return tensor.mean(dim=reduce_dims)
        return tensor.reshape(tensor.shape[0], -1)

    # auto:
    # Transformer features often end in channel D, so average all middle axes.
    # CNN features are often [B,C,T,H,W]; flatten is safer than assuming layout.
    if tensor.ndim in (3, 4):
        reduce_dims = tuple(range(1, tensor.ndim - 1))
        return tensor.mean(dim=reduce_dims)
    return tensor.reshape(tensor.shape[0], -1)


class FeatureCapture:
    def __init__(self, module: nn.Module, capture: str, reduce: str):
        self.module = module
        self.capture = capture
        self.reduce = reduce
        self.value: Optional[torch.Tensor] = None
        self.handle = None

    def __enter__(self):
        if self.capture == "input":
            def pre_hook(module, inputs):
                tensor = first_tensor(inputs)
                if tensor is None:
                    raise RuntimeError("Feature module input contains no tensor")
                self.value = reduce_feature(tensor.detach(), self.reduce)
            self.handle = self.module.register_forward_pre_hook(pre_hook)
        else:
            def hook(module, inputs, output):
                tensor = first_tensor(output)
                if tensor is None:
                    raise RuntimeError("Feature module output contains no tensor")
                self.value = reduce_feature(tensor.detach(), self.reduce)
            self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()

    def pop(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("The selected feature hook did not run")
        result = self.value
        self.value = None
        return result


@contextlib.contextmanager
def mask_one_unit(spec: UnitLayerSpec, unit_index: int):
    unit_index = int(unit_index)

    def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        x = inputs[0]
        masked = x.clone()
        if spec.unit_type == "neuron":
            masked[..., unit_index] = 0.0
        else:
            head_dim = int(spec.module.head_dim)
            shaped = masked.reshape(
                masked.shape[0],
                masked.shape[1],
                spec.num_units,
                head_dim,
            )
            shaped[:, :, unit_index, :] = 0.0
            masked = shaped.reshape_as(masked)
        return (masked,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_video_cache(
    loader: torch.utils.data.DataLoader,
    limit: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    cache = []
    for videos, targets, indices in loader:
        video_id = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        cache.append((videos.cpu(), targets.cpu(), video_id))
        if len(cache) >= limit:
            break
    return cache


def forward_feature(
    model: nn.Module,
    capture: FeatureCapture,
    videos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = unwrap_logits(model(videos))
        feature = capture.pop()
    return feature, logits


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= EPS and norm_b <= EPS:
        return 0.0
    if norm_a <= EPS or norm_b <= EPS:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b + EPS))
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def magnitude_gap(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return abs(norm_a - norm_b) / (norm_a + norm_b + EPS)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    if np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def plot_scatter(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6.5, 4.9))
    for case_type, table in summary.groupby("case_type"):
        plt.scatter(
            table["functional_distance"],
            table["feature_direction_distance_mean"],
            s=38,
            alpha=0.75,
            label=case_type,
        )
    plt.xlabel("Functional Descriptor distance")
    plt.ylabel("Representation-intervention direction distance")
    plt.title("Descriptor distance vs representation intervention")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "functional_vs_feature_intervention.png", dpi=220)
    plt.savefig(output_dir / "functional_vs_feature_intervention.pdf")
    plt.close()


def plot_case_box(summary: pd.DataFrame, output_dir: Path) -> None:
    order = [
        name
        for name in (
            "scalar_close_function_far",
            "scalar_close_function_close",
            "scalar_far_function_close",
        )
        if name in set(summary["case_type"])
    ]
    values = [
        summary.loc[
            summary["case_type"] == name,
            "feature_direction_distance_mean",
        ].to_numpy()
        for name in order
    ]
    labels = [
        name.replace("scalar_", "S:").replace("_function_", "\nF:")
        for name in order
    ]
    plt.figure(figsize=(7.4, 4.9))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.ylabel("Representation-intervention direction distance")
    plt.title("Representation intervention by descriptor-pair type")
    plt.tight_layout()
    plt.savefig(output_dir / "case_feature_intervention.png", dpi=220)
    plt.savefig(output_dir / "case_feature_intervention.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    ensure_project_importable(Path(args.project_root))

    cases = pd.read_csv(args.representative_cases)
    selected_cases = choose_cases(
        cases,
        parse_case_types(args.case_types),
        args.pairs_per_case_layer,
    )
    selected_cases.to_csv(
        output_dir / "selected_feature_intervention_pairs.csv",
        index=False,
    )

    loader, selected_indices, chosen_classes = build_balanced_loader(
        project_root=Path(args.project_root),
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    video_cache = collect_video_cache(loader, args.ablation_videos)
    if not video_cache:
        raise RuntimeError("No valid videos were loaded")

    device = torch.device(args.device)
    model, model_metadata = load_model(args.adapter, args.checkpoint, device)
    layer_specs = {spec.name: spec for spec in discover_unit_layers(model)}

    missing = sorted(set(selected_cases["layer"]) - set(layer_specs))
    if missing:
        raise KeyError(f"Selected pruning layers not found in model: {missing}")

    feature_name, feature_module = find_feature_module(
        model, args.feature_module
    )
    print(f"Feature target: {feature_name}")
    print(f"Feature capture: {args.feature_capture}")

    unique_units = sorted(
        {
            UnitKey(str(row.layer), int(row.unit_i))
            for row in selected_cases.itertuples()
        }
        | {
            UnitKey(str(row.layer), int(row.unit_j))
            for row in selected_cases.itertuples()
        },
        key=lambda item: (item.layer, item.unit_index),
    )

    baseline_features: Dict[int, np.ndarray] = {}
    baseline_predictions: Dict[int, int] = {}
    delta_vectors: Dict[Tuple[str, int, int], np.ndarray] = {}
    per_video_rows: List[Dict[str, object]] = []

    with FeatureCapture(
        feature_module,
        args.feature_capture,
        args.feature_reduce,
    ) as capture:
        for videos_cpu, targets_cpu, video_id in tqdm(
            video_cache, desc="Baseline representations", ncols=105
        ):
            videos = videos_cpu.float().to(device)
            feature, logits = forward_feature(model, capture, videos)
            baseline_features[video_id] = (
                feature[0].detach().cpu().numpy().astype(np.float32)
            )
            baseline_predictions[video_id] = int(logits.argmax(dim=1).item())

        feature_dim = int(next(iter(baseline_features.values())).shape[0])
        print(f"Captured feature dimension: {feature_dim}")

        for unit in tqdm(
            unique_units,
            desc="Unit feature interventions",
            ncols=105,
        ):
            spec = layer_specs[unit.layer]
            for videos_cpu, targets_cpu, video_id in video_cache:
                videos = videos_cpu.float().to(device)
                with mask_one_unit(spec, unit.unit_index):
                    masked_feature, masked_logits = forward_feature(
                        model, capture, videos
                    )

                masked = (
                    masked_feature[0].detach().cpu().numpy().astype(np.float32)
                )
                baseline = baseline_features[video_id]
                delta = baseline - masked
                delta_vectors[(unit.layer, unit.unit_index, video_id)] = delta

                baseline_norm = float(np.linalg.norm(baseline))
                delta_norm = float(np.linalg.norm(delta))
                relative_change = delta_norm / (baseline_norm + EPS)

                per_video_rows.append(
                    {
                        "layer": unit.layer,
                        "unit_type": spec.unit_type,
                        "unit_index": unit.unit_index,
                        "video_id": video_id,
                        "target_index": int(targets_cpu[0]),
                        "feature_dimension": feature_dim,
                        "baseline_feature_norm": baseline_norm,
                        "delta_feature_norm": delta_norm,
                        "relative_feature_change": relative_change,
                        "baseline_prediction": baseline_predictions[video_id],
                        "masked_prediction": int(
                            masked_logits.argmax(dim=1).item()
                        ),
                        "prediction_changed": int(
                            baseline_predictions[video_id]
                            != int(masked_logits.argmax(dim=1).item())
                        ),
                    }
                )

    per_video = pd.DataFrame(per_video_rows)
    per_video.to_csv(
        output_dir / "unit_feature_intervention_per_video.csv",
        index=False,
    )

    pair_video_rows = []
    pair_rows = []

    for pair in selected_cases.itertuples():
        video_direction = []
        video_magnitude = []
        delta_i_list = []
        delta_j_list = []

        for _, _, video_id in video_cache:
            delta_i = delta_vectors[
                (str(pair.layer), int(pair.unit_i), video_id)
            ]
            delta_j = delta_vectors[
                (str(pair.layer), int(pair.unit_j), video_id)
            ]

            direction_distance = cosine_distance(delta_i, delta_j)
            mag_gap = magnitude_gap(delta_i, delta_j)
            video_direction.append(direction_distance)
            video_magnitude.append(mag_gap)
            delta_i_list.append(delta_i)
            delta_j_list.append(delta_j)

            pair_video_rows.append(
                {
                    "pair_id": int(pair.pair_id),
                    "layer": str(pair.layer),
                    "unit_type": str(pair.unit_type),
                    "case_type": str(pair.case_type),
                    "unit_i": int(pair.unit_i),
                    "unit_j": int(pair.unit_j),
                    "video_id": int(video_id),
                    "scalar_distance": float(pair.scalar_distance),
                    "functional_distance": float(pair.functional_distance),
                    "feature_direction_distance": direction_distance,
                    "feature_magnitude_gap": mag_gap,
                    "unit_i_delta_norm": float(np.linalg.norm(delta_i)),
                    "unit_j_delta_norm": float(np.linalg.norm(delta_j)),
                }
            )

        mean_delta_i = np.mean(np.stack(delta_i_list), axis=0)
        mean_delta_j = np.mean(np.stack(delta_j_list), axis=0)

        pair_rows.append(
            {
                "pair_id": int(pair.pair_id),
                "layer": str(pair.layer),
                "unit_type": str(pair.unit_type),
                "case_type": str(pair.case_type),
                "unit_i": int(pair.unit_i),
                "unit_j": int(pair.unit_j),
                "scalar_distance": float(pair.scalar_distance),
                "functional_distance": float(pair.functional_distance),
                "feature_direction_distance_mean": float(
                    np.mean(video_direction)
                ),
                "feature_direction_distance_median": float(
                    np.median(video_direction)
                ),
                "feature_magnitude_gap_mean": float(
                    np.mean(video_magnitude)
                ),
                "mean_delta_direction_distance": cosine_distance(
                    mean_delta_i, mean_delta_j
                ),
                "unit_i_mean_delta_norm": float(np.linalg.norm(mean_delta_i)),
                "unit_j_mean_delta_norm": float(np.linalg.norm(mean_delta_j)),
            }
        )

    pair_per_video = pd.DataFrame(pair_video_rows)
    pair_summary = pd.DataFrame(pair_rows)

    pair_per_video.to_csv(
        output_dir / "pair_feature_intervention_per_video.csv",
        index=False,
    )
    pair_summary.to_csv(
        output_dir / "pair_feature_intervention_summary.csv",
        index=False,
    )

    group_rows = []
    for case_type, table in pair_summary.groupby("case_type"):
        group_rows.append(
            {
                "case_type": case_type,
                "pairs": int(len(table)),
                "functional_distance_median": float(
                    table["functional_distance"].median()
                ),
                "feature_direction_distance_median": float(
                    table["feature_direction_distance_mean"].median()
                ),
                "feature_direction_distance_mean": float(
                    table["feature_direction_distance_mean"].mean()
                ),
                "feature_magnitude_gap_median": float(
                    table["feature_magnitude_gap_mean"].median()
                ),
                "mean_delta_direction_distance_median": float(
                    table["mean_delta_direction_distance"].median()
                ),
            }
        )

    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(
        output_dir / "case_type_feature_intervention.csv",
        index=False,
    )

    rho_function = safe_spearman(
        pair_summary["functional_distance"],
        pair_summary["feature_direction_distance_mean"],
    )
    rho_scalar = safe_spearman(
        pair_summary["scalar_distance"],
        pair_summary["feature_direction_distance_mean"],
    )

    far = pair_summary[
        pair_summary["case_type"] == "scalar_close_function_far"
    ]["feature_direction_distance_mean"].to_numpy()
    close = pair_summary[
        pair_summary["case_type"] == "scalar_close_function_close"
    ]["feature_direction_distance_mean"].to_numpy()

    comparison = {}
    if len(far) and len(close):
        test = mannwhitneyu(far, close, alternative="greater")
        comparison = {
            "hypothesis": (
                "scalar-close/function-far pairs have larger representation "
                "intervention distance than scalar-close/function-close pairs"
            ),
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "function_far_median": float(np.median(far)),
            "function_close_median": float(np.median(close)),
        }

    plot_scatter(pair_summary, output_dir)
    plot_case_box(pair_summary, output_dir)

    if args.save_feature_deltas:
        payload = {}
        for (layer, unit_index, video_id), delta in delta_vectors.items():
            safe_layer = layer.replace(".", "_")
            payload[
                f"{safe_layer}__unit_{unit_index}__video_{video_id}"
            ] = delta.astype(np.float32)
        np.savez_compressed(
            output_dir / "feature_delta_vectors.npz",
            **payload,
        )

    summary = {
        "scientific_question": (
            "Do Functional Descriptor differences correspond to different "
            "downstream representation changes under unit masking?"
        ),
        "feature_module": feature_name,
        "feature_capture": args.feature_capture,
        "feature_reduce": args.feature_reduce,
        "feature_dimension": int(
            per_video["feature_dimension"].iloc[0]
        ),
        "model_metadata": model_metadata,
        "chosen_classes": chosen_classes,
        "selected_clips": selected_indices,
        "evaluated_video_ids": [item[2] for item in video_cache],
        "pair_count": int(len(pair_summary)),
        "unique_unit_count": int(len(unique_units)),
        "rho_functional_distance_vs_feature_intervention": rho_function,
        "rho_scalar_distance_vs_feature_intervention": rho_scalar,
        "far_vs_close_test": comparison,
        "case_type_summary": group_rows,
        "decision_rule": {
            "support": (
                "Functional distance is positively associated with downstream "
                "representation-intervention distance."
            ),
            "strong_support": (
                "Function-far pairs have significantly larger feature-change "
                "direction differences than function-close pairs (p < 0.05), "
                "and functional distance is more predictive than scalar distance."
            ),
            "failure": (
                "Functional Descriptor distance does not correspond to different "
                "internal representation changes."
            ),
        },
        "run_config": vars(args),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report = [
        "Functional Descriptor Feature-Intervention Probe",
        "=" * 92,
        "",
        f"Feature module: {feature_name}",
        f"Capture: {args.feature_capture}",
        f"Feature dimension: {summary['feature_dimension']}",
        f"Evaluated pairs: {len(pair_summary)}",
        f"Unique masked units: {len(unique_units)}",
        "",
        f"rho(functional distance, feature intervention): {rho_function}",
        f"rho(scalar distance, feature intervention): {rho_scalar}",
        "",
        "[Case-type comparison]",
        group_summary.to_string(index=False),
        "",
        "[Function-far vs function-close]",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "",
        "[Interpretation]",
        "Pass:",
        "- Functional Descriptor distance predicts differences in downstream",
        "  representation-change direction;",
        "- scalar-close/function-far pairs differ more than",
        "  scalar-close/function-close pairs.",
        "",
        "This validates representation-level functional correspondence.",
        "It does not yet establish final physical-pruning accuracy.",
    ]
    (output_dir / "FEATURE_INTERVENTION_REPORT.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"\nFeature-intervention probe complete: {output_dir.resolve()}")
    print(f"rho(function, feature intervention) = {rho_function}")
    print(f"rho(scalar, feature intervention)   = {rho_scalar}")


if __name__ == "__main__":
    main()

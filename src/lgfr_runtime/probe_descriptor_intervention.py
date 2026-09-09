#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional Descriptor Intervention Probe (V3).

Scientific question
-------------------
Do pruning units with similar scalar descriptors but different Functional
Descriptors produce different class-level intervention effects?

This probe uses representative pairs selected by V2
`probe_descriptor_distinctiveness.py`.

Primary comparison
------------------
1. scalar_close_function_far
2. scalar_close_function_close

For every selected pair (u_i, u_j), each unit is masked independently. The
resulting class-conditioned intervention signature is

    g_i[c] = mean_{video with target c}
             (z_c(x) - z_c(x; mask u_i))

The pairwise intervention-signature distance is cosine distance between g_i and
g_j after separating direction and magnitude:

    direction_distance = 1 - cosine(g_i, g_j)
    magnitude_gap      = | ||g_i||_2 - ||g_j||_2 | /
                         (||g_i||_2 + ||g_j||_2 + eps)

The primary hypothesis is:

    scalar-close/function-far pairs
        have larger intervention-signature distance
    than
    scalar-close/function-close pairs.

This experiment validates functional correspondence, not pruning superiority.

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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
        description="Validate Functional Descriptor using causal unit masking"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--representative_cases", required=True)
    parser.add_argument("--output_dir", default="./descriptor_intervention_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--frame_root", required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--target_mode",
        choices=("true", "pred"),
        default="true",
        help="Class used to define the intervention target.",
    )
    parser.add_argument(
        "--case_types",
        default=(
            "scalar_close_function_far,"
            "scalar_close_function_close,"
            "scalar_far_function_close"
        ),
    )
    parser.add_argument(
        "--pairs_per_case_layer",
        type=int,
        default=8,
        help="Maximum representative pairs for each case type in each layer.",
    )
    parser.add_argument(
        "--ablation_videos",
        type=int,
        default=9,
        help="Maximum number of selected clips used for masking.",
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
    )
    selected.insert(0, "pair_id", np.arange(len(selected), dtype=int))
    return selected


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


def target_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = logits.argmax(dim=1)
    chosen = targets if target_mode == "true" else prediction
    target_logit = logits.gather(1, chosen[:, None]).squeeze(1)

    competitor_logits = logits.clone()
    competitor_logits.scatter_(1, chosen[:, None], float("-inf"))
    competitor = competitor_logits.max(dim=1).values
    margin = target_logit - competitor
    probability = torch.softmax(logits, dim=1).gather(
        1, chosen[:, None]
    ).squeeze(1)
    return chosen, target_logit, margin, probability


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


def evaluate_unit(
    model: nn.Module,
    spec: UnitLayerSpec,
    unit_index: int,
    video_cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
    baseline_cache: Mapping[int, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for videos_cpu, targets_cpu, video_id in video_cache:
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)

        with torch.no_grad(), mask_one_unit(spec, unit_index):
            logits = unwrap_logits(model(videos))
            chosen = torch.tensor(
                [int(baseline_cache[video_id]["target_index"])],
                device=device,
                dtype=torch.long,
            )
            _, target_logit, margin, probability = target_metrics(
                logits, chosen, "true"
            )
            prediction = logits.argmax(dim=1)

        baseline = baseline_cache[video_id]
        rows.append(
            {
                "layer": spec.name,
                "unit_type": spec.unit_type,
                "unit_index": int(unit_index),
                "video_id": int(video_id),
                "target_index": int(baseline["target_index"]),
                "baseline_prediction": int(baseline["prediction"]),
                "masked_prediction": int(prediction.item()),
                "target_logit_drop": float(
                    baseline["target_logit"] - target_logit.item()
                ),
                "margin_drop": float(
                    baseline["margin"] - margin.item()
                ),
                "target_probability_drop": float(
                    baseline["probability"] - probability.item()
                ),
                "prediction_changed": int(
                    int(baseline["prediction"]) != int(prediction.item())
                ),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_cache(
    model: nn.Module,
    video_cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
) -> Dict[int, Dict[str, float]]:
    output: Dict[int, Dict[str, float]] = {}
    for videos_cpu, targets_cpu, video_id in tqdm(
        video_cache, desc="Baseline inference", ncols=100
    ):
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)
        with torch.no_grad():
            logits = unwrap_logits(model(videos))
            chosen, target_logit, margin, probability = target_metrics(
                logits, targets, target_mode
            )
            prediction = logits.argmax(dim=1)
        output[int(video_id)] = {
            "target_index": int(chosen.item()),
            "target_logit": float(target_logit.item()),
            "margin": float(margin.item()),
            "probability": float(probability.item()),
            "prediction": int(prediction.item()),
        }
    return output


def class_signature(
    unit_results: pd.DataFrame,
    class_ids: Sequence[int],
    value_column: str,
) -> np.ndarray:
    values = []
    for class_id in class_ids:
        selected = unit_results[
            unit_results["target_index"] == int(class_id)
        ][value_column]
        values.append(float(selected.mean()) if len(selected) else 0.0)
    return np.asarray(values, dtype=np.float64)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= EPS and norm_b <= EPS:
        return 0.0
    if norm_a <= EPS or norm_b <= EPS:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b + EPS))
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity


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


def plot_function_vs_intervention(
    pair_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(6.4, 4.9))
    for case_type, table in pair_summary.groupby("case_type"):
        plt.scatter(
            table["functional_distance"],
            table["intervention_direction_distance"],
            s=35,
            alpha=0.75,
            label=case_type,
        )
    plt.xlabel("Functional Descriptor distance")
    plt.ylabel("Intervention-signature direction distance")
    plt.title("Descriptor distance vs intervention difference")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        output_dir / "functional_vs_intervention_distance.png", dpi=220
    )
    plt.savefig(
        output_dir / "functional_vs_intervention_distance.pdf"
    )
    plt.close()


def plot_case_comparison(
    pair_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    order = [
        case_type
        for case_type in (
            "scalar_close_function_far",
            "scalar_close_function_close",
            "scalar_far_function_close",
        )
        if case_type in set(pair_summary["case_type"])
    ]
    values = [
        pair_summary.loc[
            pair_summary["case_type"] == case_type,
            "intervention_direction_distance",
        ].to_numpy()
        for case_type in order
    ]
    labels = [
        value.replace("scalar_", "S:").replace("_function_", "\nF:")
        for value in order
    ]
    plt.figure(figsize=(7.4, 4.9))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.ylabel("Intervention-signature direction distance")
    plt.title("Intervention differences across representative pair types")
    plt.tight_layout()
    plt.savefig(output_dir / "case_intervention_distance.png", dpi=220)
    plt.savefig(output_dir / "case_intervention_distance.pdf")
    plt.close()


def plot_signature_examples(
    pair_summary: pd.DataFrame,
    signature_rows: pd.DataFrame,
    class_ids: Sequence[int],
    output_dir: Path,
    max_examples: int = 6,
) -> None:
    gallery_dir = output_dir / "intervention_signature_gallery"
    gallery_dir.mkdir(exist_ok=True)

    priority = pair_summary.sort_values(
        ["case_type", "functional_distance"],
        ascending=[True, False],
    ).head(max_examples)

    for _, pair in priority.iterrows():
        pair_id = int(pair["pair_id"])
        rows = signature_rows[signature_rows["pair_id"] == pair_id]
        unit_i = rows[rows["pair_member"] == "i"].sort_values("class_id")
        unit_j = rows[rows["pair_member"] == "j"].sort_values("class_id")
        if unit_i.empty or unit_j.empty:
            continue

        x = np.arange(len(class_ids))
        width = 0.36
        plt.figure(figsize=(7.2, 4.6))
        plt.bar(
            x - width / 2,
            unit_i["target_logit_drop"].to_numpy(),
            width,
            label=f"unit {int(pair['unit_i'])}",
        )
        plt.bar(
            x + width / 2,
            unit_j["target_logit_drop"].to_numpy(),
            width,
            label=f"unit {int(pair['unit_j'])}",
        )
        plt.xticks(x, [str(c) for c in class_ids])
        plt.xlabel("Target class")
        plt.ylabel("Mean target-logit drop")
        plt.title(
            f"{pair['layer']}\n{pair['case_type']} | pair {pair_id}"
        )
        plt.legend()
        plt.tight_layout()
        safe_layer = str(pair["layer"]).replace(".", "_")
        plt.savefig(
            gallery_dir / f"{safe_layer}_pair_{pair_id}.png",
            dpi=220,
        )
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
        output_dir / "selected_intervention_pairs.csv", index=False
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
    model, model_metadata = load_model(
        args.adapter, args.checkpoint, device
    )
    all_specs = {spec.name: spec for spec in discover_unit_layers(model)}

    required_layers = sorted(selected_cases["layer"].unique())
    missing_layers = [layer for layer in required_layers if layer not in all_specs]
    if missing_layers:
        raise KeyError(f"Selected layers not found in model: {missing_layers}")

    baseline_cache = build_baseline_cache(
        model, video_cache, device, args.target_mode
    )
    class_ids = sorted(
        set(int(value["target_index"]) for value in baseline_cache.values())
    )

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

    unit_frames = []
    for unit in tqdm(
        unique_units, desc="Independent unit masking", ncols=100
    ):
        frame = evaluate_unit(
            model=model,
            spec=all_specs[unit.layer],
            unit_index=unit.unit_index,
            video_cache=video_cache,
            device=device,
            target_mode=args.target_mode,
            baseline_cache=baseline_cache,
        )
        unit_frames.append(frame)

    unit_per_video = pd.concat(unit_frames, ignore_index=True)
    unit_per_video.to_csv(
        output_dir / "unit_intervention_per_video.csv", index=False
    )

    unit_lookup: Dict[Tuple[str, int], pd.DataFrame] = {
        (layer, int(unit_index)): table.copy()
        for (layer, unit_index), table in unit_per_video.groupby(
            ["layer", "unit_index"]
        )
    }

    pair_rows = []
    signature_rows = []

    for pair in selected_cases.itertuples():
        key_i = (str(pair.layer), int(pair.unit_i))
        key_j = (str(pair.layer), int(pair.unit_j))
        results_i = unit_lookup[key_i]
        results_j = unit_lookup[key_j]

        logit_i = class_signature(
            results_i, class_ids, "target_logit_drop"
        )
        logit_j = class_signature(
            results_j, class_ids, "target_logit_drop"
        )
        margin_i = class_signature(results_i, class_ids, "margin_drop")
        margin_j = class_signature(results_j, class_ids, "margin_drop")
        probability_i = class_signature(
            results_i, class_ids, "target_probability_drop"
        )
        probability_j = class_signature(
            results_j, class_ids, "target_probability_drop"
        )

        for member, unit_index, logit, margin, probability in (
            ("i", int(pair.unit_i), logit_i, margin_i, probability_i),
            ("j", int(pair.unit_j), logit_j, margin_j, probability_j),
        ):
            for pos, class_id in enumerate(class_ids):
                signature_rows.append(
                    {
                        "pair_id": int(pair.pair_id),
                        "layer": str(pair.layer),
                        "unit_type": str(pair.unit_type),
                        "case_type": str(pair.case_type),
                        "pair_member": member,
                        "unit_index": unit_index,
                        "class_id": int(class_id),
                        "target_logit_drop": float(logit[pos]),
                        "margin_drop": float(margin[pos]),
                        "target_probability_drop": float(probability[pos]),
                    }
                )

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
                "intervention_direction_distance": cosine_distance(
                    logit_i, logit_j
                ),
                "intervention_magnitude_gap": magnitude_gap(
                    logit_i, logit_j
                ),
                "margin_signature_distance": cosine_distance(
                    margin_i, margin_j
                ),
                "probability_signature_distance": cosine_distance(
                    probability_i, probability_j
                ),
                "unit_i_logit_signature_norm": float(np.linalg.norm(logit_i)),
                "unit_j_logit_signature_norm": float(np.linalg.norm(logit_j)),
                "unit_i_prediction_change_rate": float(
                    results_i["prediction_changed"].mean()
                ),
                "unit_j_prediction_change_rate": float(
                    results_j["prediction_changed"].mean()
                ),
            }
        )

    pair_summary = pd.DataFrame(pair_rows)
    signature_frame = pd.DataFrame(signature_rows)
    pair_summary.to_csv(
        output_dir / "pair_intervention_summary.csv", index=False
    )
    signature_frame.to_csv(
        output_dir / "pair_class_intervention_signatures.csv", index=False
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
                "intervention_direction_distance_median": float(
                    table["intervention_direction_distance"].median()
                ),
                "intervention_direction_distance_mean": float(
                    table["intervention_direction_distance"].mean()
                ),
                "intervention_magnitude_gap_median": float(
                    table["intervention_magnitude_gap"].median()
                ),
                "margin_signature_distance_median": float(
                    table["margin_signature_distance"].median()
                ),
                "probability_signature_distance_median": float(
                    table["probability_signature_distance"].median()
                ),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(
        output_dir / "case_type_intervention_comparison.csv", index=False
    )

    rho_function = safe_spearman(
        pair_summary["functional_distance"],
        pair_summary["intervention_direction_distance"],
    )
    rho_scalar = safe_spearman(
        pair_summary["scalar_distance"],
        pair_summary["intervention_direction_distance"],
    )

    high = pair_summary[
        pair_summary["case_type"] == "scalar_close_function_far"
    ]["intervention_direction_distance"].to_numpy()
    low = pair_summary[
        pair_summary["case_type"] == "scalar_close_function_close"
    ]["intervention_direction_distance"].to_numpy()

    comparison = {}
    if len(high) and len(low):
        test = mannwhitneyu(high, low, alternative="greater")
        comparison = {
            "hypothesis": (
                "scalar_close_function_far has larger intervention distance "
                "than scalar_close_function_close"
            ),
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "far_median": float(np.median(high)),
            "close_median": float(np.median(low)),
        }

    plot_function_vs_intervention(pair_summary, output_dir)
    plot_case_comparison(pair_summary, output_dir)
    plot_signature_examples(
        pair_summary,
        signature_frame,
        class_ids,
        output_dir,
    )

    summary = {
        "scientific_question": (
            "Does Functional Descriptor distance correspond to differences in "
            "class-conditioned causal intervention signatures?"
        ),
        "model_metadata": model_metadata,
        "chosen_classes": chosen_classes,
        "selected_clips": selected_indices,
        "evaluated_video_ids": [item[2] for item in video_cache],
        "class_ids_in_intervention_signature": class_ids,
        "pair_count": int(len(pair_summary)),
        "unique_unit_count": int(len(unique_units)),
        "rho_functional_distance_vs_intervention_distance": rho_function,
        "rho_scalar_distance_vs_intervention_distance": rho_scalar,
        "far_vs_close_test": comparison,
        "case_type_summary": group_rows,
        "decision_rule": {
            "support": (
                "Functional distance is positively associated with intervention "
                "signature distance, and scalar-close/function-far pairs have "
                "larger intervention differences than scalar-close/function-close "
                "pairs."
            ),
            "strong_support": (
                "far-vs-close one-sided Mann-Whitney p < 0.05 and functional "
                "distance is more predictive than scalar distance."
            ),
            "failure": (
                "Descriptor distance does not correspond to causal intervention "
                "differences. In that case it remains descriptive but cannot be "
                "claimed as a functional pruning descriptor."
            ),
        },
        "run_config": vars(args),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_lines = [
        "Functional Descriptor Intervention Probe",
        "=" * 90,
        "",
        "Scientific question:",
        "Do descriptor differences correspond to different class-level causal effects?",
        "",
        f"evaluated pairs: {len(pair_summary)}",
        f"unique units masked: {len(unique_units)}",
        f"functional distance vs intervention distance rho: {rho_function}",
        f"scalar distance vs intervention distance rho: {rho_scalar}",
        "",
        "[Case-type comparison]",
        group_summary.to_string(index=False),
        "",
        "[Primary far-vs-close test]",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "",
        "[Interpretation]",
        "Pass:",
        "- scalar-close/function-far pairs show larger class-level intervention",
        "  differences than scalar-close/function-close pairs;",
        "- functional distance predicts intervention difference better than scalar",
        "  distance.",
        "",
        "This validates functional correspondence, not final pruning accuracy.",
    ]
    (output_dir / "INTERVENTION_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"\nIntervention probe complete: {output_dir.resolve()}")
    print(f"rho(function, intervention) = {rho_function}")
    print(f"rho(scalar, intervention)   = {rho_scalar}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online pair joint-masking probe for contribution-pattern redundancy.

This probe tests whether two units with similar class-conditioned spatio-temporal
contribution patterns are genuinely redundant under pruning.

For each selected pair (i, j), it measures:

    Δ_i   = z_y(original) - z_y(mask i)
    Δ_j   = z_y(original) - z_y(mask j)
    Δ_ij  = z_y(original) - z_y(mask i and j)

and defines pair redundancy / overlap:

    R_ij = Δ_i + Δ_j - Δ_ij

Interpretation:
- R_ij > 0: joint damage is smaller than the sum of individual damages; evidence overlaps.
- R_ij ≈ 0: approximately additive effects.
- R_ij < 0: super-additive joint damage; the pair may be complementary or interacting.

The script compares high contribution-pattern-affinity pairs against low-affinity,
scalar-neighbour, and random pairs. It does not prune or fine-tune the model.

Required companion files in the project root:
- probe_ctfrs_dynamic_function.py (dataset-compatible fixed version)
- ucf101_videoswin_probe_adapter_v2.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
class PairSpec:
    layer: str
    unit_type: str
    unit_i: int
    unit_j: int
    pair_group: str
    functional_affinity: float
    scalar_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate contribution-pattern affinity using pair joint masking"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--nearest_neighbors", required=True)
    parser.add_argument("--output_dir", default="./pair_joint_masking_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", choices=("true", "pred"), default="true")
    parser.add_argument("--seed", type=int, default=3407)

    # Probe budget only; these are not method hyperparameters.
    parser.add_argument("--pairs_per_group", type=int, default=8)
    parser.add_argument("--ablation_videos", type=int, default=6)
    parser.add_argument("--low_pair_candidates", type=int, default=5000)
    parser.add_argument(
        "--groups",
        default="high,low,scalar,random",
        help="Comma-separated subset of high,low,scalar,random",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' or regex applied to layer names present in the metrics file",
    )
    return parser.parse_args()


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    columns = [c for c in ("stage", "block", "unit_type", "layer") if c in metrics]
    table = metrics[columns].drop_duplicates()
    sort_cols = [c for c in ("stage", "block", "unit_type", "layer") if c in table]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable")
    return table["layer"].tolist()


def normalize_patterns(volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[V,U,T,H,W] -> per-video L2-normalized positive patterns [V,U,P]."""
    positive = np.maximum(np.asarray(volumes, dtype=np.float32), 0.0)
    flat = positive.reshape(positive.shape[0], positive.shape[1], -1)
    l1 = flat.sum(axis=-1, keepdims=True)
    valid = l1[..., 0] > EPS
    probability = np.zeros_like(flat, dtype=np.float32)
    probability[valid] = flat[valid] / np.maximum(l1[valid], EPS)
    l2 = np.linalg.norm(probability, axis=-1, keepdims=True)
    normalized = np.zeros_like(probability, dtype=np.float32)
    good = l2[..., 0] > EPS
    normalized[good] = probability[good] / np.maximum(l2[good], EPS)
    return normalized, valid


def functional_affinity_pair(
    patterns: np.ndarray, valid: np.ndarray, i: int, j: int
) -> float:
    pair_valid = valid[:, i] & valid[:, j]
    if not np.any(pair_valid):
        return 0.0
    cosine = np.sum(patterns[pair_valid, i] * patterns[pair_valid, j], axis=1)
    return float(np.mean(cosine))


def standardize_scalar(layer_metrics: pd.DataFrame) -> np.ndarray:
    values = layer_metrics[["d_abs", "d_rel"]].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, EPS)).astype(np.float32)


def estimate_scalar_scale(scalar: np.ndarray, rng: np.random.RandomState) -> float:
    units = len(scalar)
    sample_count = min(50000, max(units * 10, 1000))
    first = rng.randint(0, units, size=sample_count)
    second = rng.randint(0, units, size=sample_count)
    keep = first != second
    distances = np.linalg.norm(scalar[first[keep]] - scalar[second[keep]], axis=1)
    positive = distances[distances > EPS]
    return float(np.median(positive)) if len(positive) else 1.0


def scalar_similarity(scalar: np.ndarray, scale: float, i: int, j: int) -> float:
    d2 = float(np.sum((scalar[i] - scalar[j]) ** 2))
    return float(np.exp(-d2 / (2.0 * scale * scale + EPS)))


def parse_neighbor_list(value: Any) -> List[int]:
    if pd.isna(value):
        return []
    return [int(v) for v in str(value).split(";") if str(v).strip()]


def unique_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def choose_pairs_for_layer(
    layer_name: str,
    unit_type: str,
    layer_metrics: pd.DataFrame,
    layer_nn: pd.DataFrame,
    patterns: np.ndarray,
    valid: np.ndarray,
    groups: Sequence[str],
    pairs_per_group: int,
    low_pair_candidates: int,
    rng: np.random.RandomState,
) -> List[PairSpec]:
    units = len(layer_metrics)
    scalar = standardize_scalar(layer_metrics)
    scale = estimate_scalar_scale(scalar, rng)
    selected: List[PairSpec] = []
    used: set[Tuple[int, int]] = set()

    def add_ranked(
        candidates: Sequence[Tuple[int, int]], group: str, reverse: bool
    ) -> None:
        scored = []
        for i, j in candidates:
            if i == j or not (0 <= i < units and 0 <= j < units):
                continue
            pair = unique_pair(i, j)
            if pair in used:
                continue
            affinity = functional_affinity_pair(patterns, valid, *pair)
            scalar_sim = scalar_similarity(scalar, scale, *pair)
            scored.append((affinity, scalar_sim, pair))
        scored.sort(key=lambda item: item[0], reverse=reverse)
        for affinity, scalar_sim, pair in scored:
            if len([p for p in selected if p.pair_group == group]) >= pairs_per_group:
                break
            if pair in used:
                continue
            used.add(pair)
            selected.append(
                PairSpec(
                    layer=layer_name,
                    unit_type=unit_type,
                    unit_i=pair[0],
                    unit_j=pair[1],
                    pair_group=group,
                    functional_affinity=affinity,
                    scalar_similarity=scalar_sim,
                )
            )

    if "high" in groups:
        candidates = []
        for row in layer_nn.itertuples():
            for neighbour in parse_neighbor_list(row.functional_neighbors):
                candidates.append((int(row.unit_index), neighbour))
        add_ranked(candidates, "high", reverse=True)

    if "scalar" in groups:
        candidates = []
        for row in layer_nn.itertuples():
            for neighbour in parse_neighbor_list(row.scalar_neighbors):
                candidates.append((int(row.unit_index), neighbour))
        # Scalar group is ranked by scalar similarity, not functional similarity.
        scored = []
        for i, j in candidates:
            if i == j or not (0 <= i < units and 0 <= j < units):
                continue
            pair = unique_pair(i, j)
            if pair in used:
                continue
            scored.append(
                (
                    scalar_similarity(scalar, scale, *pair),
                    functional_affinity_pair(patterns, valid, *pair),
                    pair,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        for scalar_sim, affinity, pair in scored:
            if len([p for p in selected if p.pair_group == "scalar"]) >= pairs_per_group:
                break
            if pair in used:
                continue
            used.add(pair)
            selected.append(
                PairSpec(
                    layer_name, unit_type, pair[0], pair[1], "scalar",
                    affinity, scalar_sim
                )
            )

    # Draw a common random pool for low and random groups.
    pool: set[Tuple[int, int]] = set()
    max_possible = units * (units - 1) // 2
    target = min(low_pair_candidates, max_possible)
    attempts = 0
    while len(pool) < target and attempts < target * 20 + 1000:
        i, j = int(rng.randint(0, units)), int(rng.randint(0, units))
        attempts += 1
        if i != j:
            pair = unique_pair(i, j)
            if pair not in used:
                pool.add(pair)

    if "low" in groups:
        add_ranked(list(pool), "low", reverse=False)

    if "random" in groups:
        available = [pair for pair in pool if pair not in used]
        rng.shuffle(available)
        for pair in available[:pairs_per_group]:
            used.add(pair)
            selected.append(
                PairSpec(
                    layer_name,
                    unit_type,
                    pair[0],
                    pair[1],
                    "random",
                    functional_affinity_pair(patterns, valid, *pair),
                    scalar_similarity(scalar, scale, *pair),
                )
            )

    return selected


@contextlib.contextmanager
def mask_units(spec: UnitLayerSpec, unit_indices: Sequence[int]):
    indices = sorted(set(int(v) for v in unit_indices))

    def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        x = inputs[0]
        masked = x.clone()
        if spec.unit_type == "neuron":
            masked[..., indices] = 0.0
        else:
            head_dim = int(spec.module.head_dim)
            reshaped = masked.reshape(
                masked.shape[0], masked.shape[1], spec.num_units, head_dim
            )
            reshaped[:, :, indices, :] = 0.0
            masked = reshaped.reshape_as(masked)
        return (masked,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_cache(
    loader: torch.utils.data.DataLoader, limit: int
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    cache = []
    for videos, targets, indices in loader:
        index = int(indices[0]) if torch.is_tensor(indices) else int(indices[0])
        cache.append((videos.cpu(), targets.cpu(), index))
        if len(cache) >= limit:
            break
    return cache


def chosen_target_logits(
    model: nn.Module,
    videos: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = unwrap_logits(model(videos))
    predicted = logits.argmax(dim=1)
    chosen = targets if target_mode == "true" else predicted
    values = logits.gather(1, chosen[:, None]).squeeze(1)
    return values, chosen


def evaluate_pair(
    model: nn.Module,
    spec: UnitLayerSpec,
    pair: PairSpec,
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for videos_cpu, targets_cpu, video_id in cache:
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)
        with torch.no_grad():
            baseline, chosen = chosen_target_logits(model, videos, targets, target_mode)
            with mask_units(spec, [pair.unit_i]):
                masked_i, _ = chosen_target_logits(model, videos, targets, target_mode)
            with mask_units(spec, [pair.unit_j]):
                masked_j, _ = chosen_target_logits(model, videos, targets, target_mode)
            with mask_units(spec, [pair.unit_i, pair.unit_j]):
                masked_ij, _ = chosen_target_logits(model, videos, targets, target_mode)

        base = float(baseline.item())
        delta_i = float((baseline - masked_i).item())
        delta_j = float((baseline - masked_j).item())
        delta_ij = float((baseline - masked_ij).item())
        redundancy = delta_i + delta_j - delta_ij
        absolute_overlap = abs(delta_i) + abs(delta_j) - abs(delta_ij)
        normalized_redundancy = redundancy / (
            abs(delta_i) + abs(delta_j) + EPS
        )
        rows.append(
            {
                "layer": pair.layer,
                "unit_type": pair.unit_type,
                "unit_i": pair.unit_i,
                "unit_j": pair.unit_j,
                "pair_group": pair.pair_group,
                "functional_affinity": pair.functional_affinity,
                "scalar_similarity": pair.scalar_similarity,
                "video_id": video_id,
                "target_index": int(chosen.item()),
                "baseline_logit": base,
                "delta_i": delta_i,
                "delta_j": delta_j,
                "delta_ij": delta_ij,
                "redundancy": redundancy,
                "absolute_overlap": absolute_overlap,
                "normalized_redundancy": normalized_redundancy,
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "mean_delta_i": float(frame["delta_i"].mean()),
        "mean_delta_j": float(frame["delta_j"].mean()),
        "mean_delta_ij": float(frame["delta_ij"].mean()),
        "mean_redundancy": float(frame["redundancy"].mean()),
        "median_redundancy": float(frame["redundancy"].median()),
        "mean_absolute_overlap": float(frame["absolute_overlap"].mean()),
        "mean_normalized_redundancy": float(
            frame["normalized_redundancy"].mean()
        ),
        "positive_redundancy_rate": float((frame["redundancy"] > 0).mean()),
    }
    return frame, summary


def aggregate_pairs(per_video: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "layer", "unit_type", "unit_i", "unit_j", "pair_group",
        "functional_affinity", "scalar_similarity",
    ]
    return (
        per_video.groupby(group_cols, as_index=False)
        .agg(
            videos=("video_id", "nunique"),
            mean_delta_i=("delta_i", "mean"),
            mean_delta_j=("delta_j", "mean"),
            mean_delta_ij=("delta_ij", "mean"),
            mean_redundancy=("redundancy", "mean"),
            median_redundancy=("redundancy", "median"),
            mean_absolute_overlap=("absolute_overlap", "mean"),
            mean_normalized_redundancy=("normalized_redundancy", "mean"),
            positive_redundancy_rate=("redundancy", lambda x: float((x > 0).mean())),
        )
    )


def statistical_tests(pair_frame: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "rho_affinity_redundancy": safe_spearman(
            pair_frame["functional_affinity"], pair_frame["mean_redundancy"]
        ),
        "rho_affinity_normalized_redundancy": safe_spearman(
            pair_frame["functional_affinity"],
            pair_frame["mean_normalized_redundancy"],
        ),
        "rho_scalar_similarity_redundancy": safe_spearman(
            pair_frame["scalar_similarity"], pair_frame["mean_redundancy"]
        ),
        "group_means": {},
        "high_vs_low_mannwhitney": {},
    }
    for group, table in pair_frame.groupby("pair_group"):
        results["group_means"][group] = {
            "pairs": int(len(table)),
            "functional_affinity": float(table["functional_affinity"].mean()),
            "mean_redundancy": float(table["mean_redundancy"].mean()),
            "median_redundancy": float(table["mean_redundancy"].median()),
            "mean_normalized_redundancy": float(
                table["mean_normalized_redundancy"].mean()
            ),
            "positive_redundancy_rate": float(
                (table["mean_redundancy"] > 0).mean()
            ),
        }

    if {"high", "low"}.issubset(set(pair_frame["pair_group"])):
        high = pair_frame.loc[
            pair_frame["pair_group"] == "high", "mean_redundancy"
        ].to_numpy()
        low = pair_frame.loc[
            pair_frame["pair_group"] == "low", "mean_redundancy"
        ].to_numpy()
        if len(high) and len(low):
            test = mannwhitneyu(high, low, alternative="greater")
            results["high_vs_low_mannwhitney"] = {
                "alternative": "high > low",
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
    return results


def plot_results(pair_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6.2, 4.7))
    plt.scatter(
        pair_frame["functional_affinity"],
        pair_frame["mean_redundancy"],
        s=24,
        alpha=0.7,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Contribution-pattern affinity")
    plt.ylabel("Mean pair redundancy R_ij")
    plt.title("Functional affinity vs joint-masking redundancy")
    plt.tight_layout()
    plt.savefig(output_dir / "affinity_vs_redundancy.png", dpi=220)
    plt.close()

    order = [g for g in ("high", "low", "scalar", "random")
             if g in set(pair_frame["pair_group"])]
    values = [
        pair_frame.loc[pair_frame["pair_group"] == group, "mean_redundancy"]
        for group in order
    ]
    plt.figure(figsize=(6.2, 4.7))
    plt.boxplot(values, labels=order)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Mean pair redundancy R_ij")
    plt.title("Joint-masking redundancy by pair source")
    plt.tight_layout()
    plt.savefig(output_dir / "redundancy_by_pair_group.png", dpi=220)
    plt.close()

    values_norm = [
        pair_frame.loc[
            pair_frame["pair_group"] == group,
            "mean_normalized_redundancy",
        ]
        for group in order
    ]
    plt.figure(figsize=(6.2, 4.7))
    plt.boxplot(values_norm, labels=order)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Normalized redundancy")
    plt.title("Normalized joint-masking redundancy by pair source")
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_redundancy_by_pair_group.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    metrics = pd.read_csv(args.unit_metrics)
    nearest = pd.read_csv(args.nearest_neighbors)
    arrays = np.load(args.npz, allow_pickle=True)
    layers = ordered_layers(metrics)
    volume_keys = sorted(
        key for key in arrays.files if key.endswith("_contribution_volumes")
    )
    if len(layers) != len(volume_keys):
        raise ValueError(
            f"Layer count mismatch: metrics={len(layers)}, npz={len(volume_keys)}"
        )

    if args.layers != "all":
        import re
        pattern = re.compile(args.layers)
        keep = [i for i, layer in enumerate(layers) if pattern.search(layer)]
        layers = [layers[i] for i in keep]
        volume_keys = [volume_keys[i] for i in keep]

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    invalid = set(groups) - {"high", "low", "scalar", "random"}
    if invalid:
        raise ValueError(f"Unknown groups: {sorted(invalid)}")

    pair_specs: List[PairSpec] = []
    for layer_name, volume_key in zip(layers, volume_keys):
        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        layer_nn = nearest[nearest["layer"] == layer_name].copy()
        volumes = arrays[volume_key]
        patterns, valid = normalize_patterns(volumes)
        pair_specs.extend(
            choose_pairs_for_layer(
                layer_name=layer_name,
                unit_type=str(layer_metrics["unit_type"].iloc[0]),
                layer_metrics=layer_metrics,
                layer_nn=layer_nn,
                patterns=patterns,
                valid=valid,
                groups=groups,
                pairs_per_group=args.pairs_per_group,
                low_pair_candidates=args.low_pair_candidates,
                rng=rng,
            )
        )

    pairs_table = pd.DataFrame([p.__dict__ for p in pair_specs])
    pairs_table.to_csv(output_dir / "selected_pairs.csv", index=False)
    print("Selected pair counts:")
    print(pairs_table.groupby(["layer", "pair_group"]).size())

    ensure_project_importable(Path(args.project_root))
    device = torch.device(args.device)
    loader, selected_indices, chosen_classes = build_balanced_loader(
        project_root=Path(args.project_root),
        val_list=args.val_list,
        frame_root=args.frame_root,
        num_classes=args.num_classes,
        videos_per_class=args.videos_per_class,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model, model_metadata = load_model(args.adapter, args.checkpoint, device)
    specs = {spec.name: spec for spec in discover_unit_layers(model)}
    missing = sorted(set(p.layer for p in pair_specs) - set(specs))
    if missing:
        raise KeyError(f"Selected layers not found in model: {missing}")

    cache = collect_cache(loader, args.ablation_videos)
    all_rows = []
    pair_summaries = []
    for pair in tqdm(pair_specs, desc="Pair joint masking", ncols=105):
        frame, summary = evaluate_pair(
            model=model,
            spec=specs[pair.layer],
            pair=pair,
            cache=cache,
            device=device,
            target_mode=args.target_mode,
        )
        all_rows.append(frame)
        pair_summaries.append({**pair.__dict__, **summary})

    per_video = pd.concat(all_rows, ignore_index=True)
    per_video.to_csv(output_dir / "pair_masking_per_video.csv", index=False)
    pair_frame = aggregate_pairs(per_video)
    pair_frame.to_csv(output_dir / "pair_redundancy_summary.csv", index=False)

    tests = statistical_tests(pair_frame)
    summary = {
        "definition": "R_ij = Δ_i + Δ_j - Δ_ij",
        "interpretation": {
            "positive": "overlapping/redundant deletion effects",
            "zero": "approximately additive effects",
            "negative": "super-additive/complementary interaction",
        },
        "model_metadata": model_metadata,
        "chosen_classes": chosen_classes,
        "selected_clips": selected_indices,
        "num_pairs": int(len(pair_frame)),
        "num_videos_per_pair": int(len(cache)),
        "statistics": tests,
        "limitations": [
            "Logit-drop redundancy is local to the selected calibration videos.",
            "Positive R_ij supports overlapping effects but is not a proof that either unit can be safely removed globally.",
            "Pairs are selected within layers; cross-layer pair redundancy is not tested.",
            "A larger class/video sample is required before changing the pruning algorithm.",
        ],
        "run_config": vars(args),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plot_results(pair_frame, output_dir)

    report = [
        "Pair Joint-Masking Redundancy Probe",
        "=" * 88,
        "",
        "Definition: R_ij = Δ_i + Δ_j - Δ_ij",
        "",
        f"Pairs: {len(pair_frame)}",
        f"Videos per pair: {len(cache)}",
        "",
        "[Global correlations]",
        f"rho(functional affinity, redundancy): "
        f"{tests['rho_affinity_redundancy']}",
        f"rho(functional affinity, normalized redundancy): "
        f"{tests['rho_affinity_normalized_redundancy']}",
        f"rho(scalar similarity, redundancy): "
        f"{tests['rho_scalar_similarity_redundancy']}",
        "",
        "[Group means]",
        json.dumps(tests["group_means"], ensure_ascii=False, indent=2),
        "",
        "[High vs low Mann-Whitney]",
        json.dumps(tests["high_vs_low_mannwhitney"], ensure_ascii=False, indent=2),
        "",
        "Decision rule:",
        "- Continue only if high-affinity pairs have consistently larger R_ij than low/random pairs.",
        "- A positive global correlation is necessary but not sufficient.",
        "- Check attention heads and FFN neurons separately; opposite directions invalidate a universal claim.",
    ]
    (output_dir / "PAIR_REDUNDANCY_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Pair redundancy probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

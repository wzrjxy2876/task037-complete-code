#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe contribution-pattern coverage pruning without modifying the model structure.

This probe compares three unit-selection strategies under the same masking budget:

1. scalar_only
   Delete units with the lowest existing scalar pruning risk.

2. pattern_only
   Delete units that cause the smallest loss of contribution-pattern coverage.

3. scalar_candidate_pattern_coverage
   First restrict candidates using the existing scalar risk, then delete units
   greedily by minimum contribution-pattern coverage loss.

The probe does not physically prune or fine-tune the network. It masks the selected
units in-place and measures:
- target-logit drop
- target-margin drop
- top-1 prediction change rate
- retained contribution-pattern coverage
- per-layer / per-type deletion distribution

Required companion files in the project root:
- fixed probe_ctfrs_dynamic_function.py
- ucf101_videoswin_probe_adapter_v2.py

The contribution-pattern affinity is computed within each layer from the saved
positive T×H×W contribution volumes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


@dataclass
class LayerData:
    layer: str
    unit_type: str
    metrics: pd.DataFrame
    affinity: np.ndarray
    unit_cost: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scalar-only and contribution-pattern-coverage unit selection"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--unit_metrics", required=True)
    parser.add_argument("--output_dir", default="./coverage_pruning_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default="ucf101_videoswin_probe_adapter_v2")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", choices=("true", "pred"), default="true")
    parser.add_argument("--seed", type=int, default=3407)

    # Probe budgets only. These do not become pruning-method hyperparameters.
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.10,
        help="Fraction of units masked inside each selected layer for the probe",
    )
    parser.add_argument(
        "--candidate_multiplier",
        type=float,
        default=2.0,
        help="Scalar candidate-pool size relative to the final mask count",
    )
    parser.add_argument("--ablation_videos", type=int, default=6)
    parser.add_argument(
        "--layers",
        default=r"layers\.3\.blocks\.(0|1)\.(attn|mlp)",
        help="'all' or regex over layer names",
    )
    parser.add_argument(
        "--strategies",
        default="scalar_only,pattern_only,scalar_candidate_pattern_coverage",
    )
    parser.add_argument("--affinity_chunk_size", type=int, default=256)
    parser.add_argument(
        "--affinity_device",
        default="auto",
        help="auto, cpu, or cuda:0. auto uses --device when CUDA is available.",
    )
    parser.add_argument(
        "--scalar_risk",
        choices=("d_rel", "d_abs", "product", "sum"),
        default="d_rel",
        help=(
            "Existing scalar risk used only by the probe. "
            "Use the option matching the current pruning implementation."
        ),
    )
    return parser.parse_args()


def ordered_layers(metrics: pd.DataFrame) -> List[str]:
    columns = [c for c in ("stage", "block", "unit_type", "layer") if c in metrics]
    table = metrics[columns].drop_duplicates()
    sort_cols = [c for c in ("stage", "block", "unit_type", "layer") if c in table]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable")
    return table["layer"].tolist()


def normalize_patterns(volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [V,U,T,H,W] volumes into per-video unit L2-normalized patterns."""
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


def full_functional_affinity(
    patterns: np.ndarray,
    valid: np.ndarray,
    chunk_size: int,
    compute_device: str = "cpu",
    desc: str = "functional affinity",
) -> np.ndarray:
    """Average per-video cosine affinity [U,U], accelerated by torch/CUDA."""
    videos, units, _ = patterns.shape
    device = torch.device(compute_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    pattern_t = torch.from_numpy(patterns).to(device=device, dtype=torch.float32)
    valid_t = torch.from_numpy(valid).to(device=device, dtype=torch.bool)
    affinity = torch.zeros((units, units), device=device, dtype=torch.float32)
    denominator = torch.zeros((units, units), device=device, dtype=torch.float32)
    iterator = range(0, units, chunk_size)
    iterator = tqdm(iterator, desc=desc, ncols=105)
    with torch.no_grad():
        for start in iterator:
            end = min(start + chunk_size, units)
            block_num = torch.zeros((end - start, units), device=device)
            block_den = torch.zeros((end - start, units), device=device)
            for video in range(videos):
                sim = pattern_t[video, start:end] @ pattern_t[video].T
                pair_valid = (
                    valid_t[video, start:end, None] & valid_t[video, None, :]
                )
                block_num.add_(sim * pair_valid)
                block_den.add_(pair_valid.float())
            good = block_den > 0
            block = torch.zeros_like(block_num)
            block[good] = block_num[good] / block_den[good]
            affinity[start:end] = block.clamp_(0.0, 1.0)
            denominator[start:end] = block_den
    affinity.fill_diagonal_(1.0)
    output = affinity.cpu().numpy()
    del pattern_t, valid_t, affinity, denominator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output

def infer_unit_cost(metrics: pd.DataFrame) -> np.ndarray:
    """Use available parameter-count column; otherwise use equal unit costs."""
    for column in (
        "parameter_count",
        "param_count",
        "params",
        "unit_parameter_count",
        "cost",
    ):
        if column in metrics.columns:
            values = metrics[column].to_numpy(dtype=np.float64)
            values = np.maximum(values, EPS)
            return values
    return np.ones(len(metrics), dtype=np.float64)


def scalar_risk_values(metrics: pd.DataFrame, mode: str) -> np.ndarray:
    d_abs = metrics["d_abs"].to_numpy(dtype=np.float64)
    d_rel = metrics["d_rel"].to_numpy(dtype=np.float64)
    if mode == "d_rel":
        return d_rel
    if mode == "d_abs":
        return d_abs
    if mode == "product":
        return d_abs * d_rel
    if mode == "sum":
        return d_abs + d_rel
    raise ValueError(mode)


def coverage_value(affinity: np.ndarray, kept: np.ndarray) -> float:
    if len(kept) == 0:
        return 0.0
    return float(np.max(affinity[:, kept], axis=1).sum())


def initial_top_two(
    affinity: np.ndarray, kept_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kept = np.flatnonzero(kept_mask)
    values = affinity[:, kept]
    if values.shape[1] == 1:
        best_local = np.zeros(values.shape[0], dtype=int)
        best_values = values[:, 0]
        second_values = np.zeros(values.shape[0], dtype=np.float32)
        second_units = np.full(values.shape[0], -1, dtype=int)
    else:
        order = np.argpartition(-values, kth=1, axis=1)[:, :2]
        first_vals = values[np.arange(len(values)), order[:, 0]]
        second_vals_raw = values[np.arange(len(values)), order[:, 1]]
        swap = second_vals_raw > first_vals
        best_local = np.where(swap, order[:, 1], order[:, 0])
        second_local = np.where(swap, order[:, 0], order[:, 1])
        best_values = values[np.arange(len(values)), best_local]
        second_values = values[np.arange(len(values)), second_local]
        second_units = kept[second_local]
    best_units = kept[best_local]
    return best_values, second_values, best_units, second_units


def coverage_loss_for_removal(
    affinity: np.ndarray,
    kept_mask: np.ndarray,
    candidate: int,
) -> float:
    kept = np.flatnonzero(kept_mask)
    before = np.max(affinity[:, kept], axis=1)
    remaining = kept[kept != candidate]
    if len(remaining) == 0:
        return float(before.sum())
    after = np.max(affinity[:, remaining], axis=1)
    return float((before - after).sum())


def greedy_pattern_removal(
    affinity: np.ndarray,
    remove_count: int,
    candidates: Optional[Sequence[int]] = None,
    unit_cost: Optional[np.ndarray] = None,
    desc: str = "coverage greedy",
) -> Tuple[List[int], pd.DataFrame]:
    """Exact greedy removal with incremental top-two coverage maintenance.

    For each represented unit m, maintain its best and second-best retained
    representatives. Removing candidate i loses coverage only for rows whose
    current best representative is i. This avoids the original nested
    candidate-by-full-matrix recomputation.
    """
    units = affinity.shape[0]
    kept_mask = np.ones(units, dtype=bool)
    candidate_mask = np.zeros(units, dtype=bool)
    if candidates is None:
        candidate_mask[:] = True
    else:
        candidate_mask[np.asarray(list(candidates), dtype=int)] = True
    costs = (
        np.asarray(unit_cost, dtype=np.float64)
        if unit_cost is not None
        else np.ones(units, dtype=np.float64)
    )

    best_values, second_values, best_units, second_units = initial_top_two(
        affinity, kept_mask
    )
    removed: List[int] = []
    trace: List[Dict[str, float]] = []

    for step in tqdm(range(remove_count), desc=desc, ncols=105):
        available_mask = kept_mask & candidate_mask
        if not np.any(available_mask):
            break

        gaps = np.maximum(best_values - second_values, 0.0)
        losses = np.bincount(best_units, weights=gaps, minlength=units).astype(np.float64)
        ratios = losses / np.maximum(costs, EPS)
        ratios[~available_mask] = np.inf
        best_unit = int(np.argmin(ratios))
        best_loss = float(losses[best_unit])
        best_ratio = float(ratios[best_unit])
        coverage_before = float(best_values.sum())

        kept_mask[best_unit] = False
        removed.append(best_unit)

        affected = (best_units == best_unit) | (second_units == best_unit)
        affected_rows = np.flatnonzero(affected)
        if len(affected_rows):
            kept = np.flatnonzero(kept_mask)
            if len(kept) == 0:
                best_values[affected_rows] = 0.0
                second_values[affected_rows] = 0.0
                best_units[affected_rows] = -1
                second_units[affected_rows] = -1
            else:
                values = affinity[np.ix_(affected_rows, kept)]
                if len(kept) == 1:
                    best_values[affected_rows] = values[:, 0]
                    second_values[affected_rows] = 0.0
                    best_units[affected_rows] = kept[0]
                    second_units[affected_rows] = -1
                else:
                    order = np.argpartition(-values, kth=1, axis=1)[:, :2]
                    v0 = values[np.arange(len(affected_rows)), order[:, 0]]
                    v1 = values[np.arange(len(affected_rows)), order[:, 1]]
                    swap = v1 > v0
                    first_local = np.where(swap, order[:, 1], order[:, 0])
                    second_local = np.where(swap, order[:, 0], order[:, 1])
                    best_values[affected_rows] = values[
                        np.arange(len(affected_rows)), first_local
                    ]
                    second_values[affected_rows] = values[
                        np.arange(len(affected_rows)), second_local
                    ]
                    best_units[affected_rows] = kept[first_local]
                    second_units[affected_rows] = kept[second_local]

        coverage_after = float(best_values.sum())
        trace.append(
            {
                "step": step + 1,
                "removed_unit": best_unit,
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
                "coverage_loss": best_loss,
                "unit_cost": float(costs[best_unit]),
                "loss_per_cost": best_ratio,
            }
        )
    return removed, pd.DataFrame(trace)

def select_units_for_layer(
    layer: LayerData,
    remove_count: int,
    candidate_multiplier: float,
    scalar_risk_mode: str,
) -> Dict[str, Tuple[List[int], pd.DataFrame]]:
    risk = scalar_risk_values(layer.metrics, scalar_risk_mode)
    scalar_order = np.argsort(risk)
    scalar_removed = scalar_order[:remove_count].astype(int).tolist()
    scalar_trace = pd.DataFrame(
        {
            "step": np.arange(1, len(scalar_removed) + 1),
            "removed_unit": scalar_removed,
            "scalar_risk": risk[scalar_removed],
        }
    )

    pattern_removed, pattern_trace = greedy_pattern_removal(
        layer.affinity,
        remove_count,
        candidates=None,
        unit_cost=layer.unit_cost,
        desc=f"{layer.layer} pattern-only",
    )

    candidate_count = min(
        len(layer.metrics),
        max(remove_count, int(math.ceil(candidate_multiplier * remove_count))),
    )
    scalar_candidates = scalar_order[:candidate_count].astype(int).tolist()
    hybrid_removed, hybrid_trace = greedy_pattern_removal(
        layer.affinity,
        remove_count,
        candidates=scalar_candidates,
        unit_cost=layer.unit_cost,
        desc=f"{layer.layer} hybrid",
    )
    hybrid_trace["candidate_pool_size"] = candidate_count

    return {
        "scalar_only": (scalar_removed, scalar_trace),
        "pattern_only": (pattern_removed, pattern_trace),
        "scalar_candidate_pattern_coverage": (hybrid_removed, hybrid_trace),
    }


@contextlib.contextmanager
def mask_strategy_units(
    layer_specs: Mapping[str, UnitLayerSpec],
    selections: Mapping[str, Sequence[int]],
):
    handles = []

    for layer_name, units in selections.items():
        spec = layer_specs[layer_name]
        unit_indices = sorted(set(int(v) for v in units))

        def make_hook(
            local_spec: UnitLayerSpec,
            local_indices: Sequence[int],
        ):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                x = inputs[0]
                masked = x.clone()
                if local_spec.unit_type == "neuron":
                    masked[..., list(local_indices)] = 0.0
                else:
                    head_dim = int(local_spec.module.head_dim)
                    reshaped = masked.reshape(
                        masked.shape[0],
                        masked.shape[1],
                        local_spec.num_units,
                        head_dim,
                    )
                    reshaped[:, :, list(local_indices), :] = 0.0
                    masked = reshaped.reshape_as(masked)
                return (masked,) + tuple(inputs[1:])
            return hook

        handles.append(
            spec.hook_module.register_forward_pre_hook(
                make_hook(spec, unit_indices)
            )
        )

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def collect_cache(
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


def logits_and_targets(
    model: nn.Module,
    videos: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = unwrap_logits(model(videos))
    predicted = logits.argmax(dim=1)
    chosen = targets if target_mode == "true" else predicted
    target_logit = logits.gather(1, chosen[:, None]).squeeze(1)

    masked_logits = logits.clone()
    masked_logits.scatter_(1, chosen[:, None], float("-inf"))
    runner_up = masked_logits.max(dim=1).values
    margin = target_logit - runner_up
    return logits, chosen, margin


def evaluate_strategy(
    model: nn.Module,
    layer_specs: Mapping[str, UnitLayerSpec],
    selections: Mapping[str, Sequence[int]],
    cache: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    target_mode: str,
) -> pd.DataFrame:
    rows = []
    for videos_cpu, targets_cpu, video_id in cache:
        videos = videos_cpu.float().to(device)
        targets = targets_cpu.long().to(device)
        with torch.no_grad():
            base_logits, chosen, base_margin = logits_and_targets(
                model, videos, targets, target_mode
            )
            base_target = base_logits.gather(1, chosen[:, None]).squeeze(1)
            base_pred = base_logits.argmax(dim=1)

            with mask_strategy_units(layer_specs, selections):
                masked_logits, _, masked_margin = logits_and_targets(
                    model, videos, targets, target_mode
                )
            masked_target = masked_logits.gather(1, chosen[:, None]).squeeze(1)
            masked_pred = masked_logits.argmax(dim=1)

        rows.append(
            {
                "video_id": video_id,
                "target_index": int(chosen.item()),
                "baseline_target_logit": float(base_target.item()),
                "masked_target_logit": float(masked_target.item()),
                "target_logit_drop": float((base_target - masked_target).item()),
                "baseline_margin": float(base_margin.item()),
                "masked_margin": float(masked_margin.item()),
                "margin_drop": float((base_margin - masked_margin).item()),
                "baseline_prediction": int(base_pred.item()),
                "masked_prediction": int(masked_pred.item()),
                "prediction_changed": int(base_pred.item() != masked_pred.item()),
            }
        )
    return pd.DataFrame(rows)


def strategy_coverage(
    layer_data: Mapping[str, LayerData],
    selections: Mapping[str, Sequence[int]],
) -> Tuple[float, Dict[str, float]]:
    totals = {}
    retained = {}
    for layer_name, data in layer_data.items():
        all_units = np.arange(len(data.metrics))
        removed = set(int(v) for v in selections.get(layer_name, []))
        kept = np.array([v for v in all_units if int(v) not in removed], dtype=int)
        total = coverage_value(data.affinity, all_units)
        value = coverage_value(data.affinity, kept)
        totals[layer_name] = total
        retained[layer_name] = value / (total + EPS)
    weighted = float(
        np.average(
            list(retained.values()),
            weights=[len(layer_data[name].metrics) for name in retained],
        )
    )
    return weighted, retained


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    metrics = pd.read_csv(args.unit_metrics)
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
        pattern = re.compile(args.layers)
        keep_indices = [i for i, layer in enumerate(layers) if pattern.search(layer)]
        layers = [layers[i] for i in keep_indices]
        volume_keys = [volume_keys[i] for i in keep_indices]

    layer_data: Dict[str, LayerData] = {}
    selection_records = []
    if args.affinity_device == "auto":
        affinity_device = args.device if torch.cuda.is_available() else "cpu"
    else:
        affinity_device = args.affinity_device
    print(f"Affinity computation device: {affinity_device}", flush=True)
    traces = []

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    allowed = {
        "scalar_only",
        "pattern_only",
        "scalar_candidate_pattern_coverage",
    }
    unknown = set(strategies) - allowed
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    for layer_index, (layer_name, volume_key) in enumerate(zip(layers, volume_keys), start=1):
        print(f"\n[Offline selection {layer_index}/{len(layers)}] {layer_name}", flush=True)
        layer_metrics = (
            metrics[metrics["layer"] == layer_name]
            .sort_values("unit_index")
            .reset_index(drop=True)
        )
        patterns, valid = normalize_patterns(arrays[volume_key])
        print(f"  contribution volume: {arrays[volume_key].shape}", flush=True)
        affinity = full_functional_affinity(
            patterns,
            valid,
            args.affinity_chunk_size,
            compute_device=affinity_device,
            desc=f"{layer_name} affinity",
        )
        print("  affinity ready; selecting units...", flush=True)
        unit_cost = infer_unit_cost(layer_metrics)
        data = LayerData(
            layer=layer_name,
            unit_type=str(layer_metrics["unit_type"].iloc[0]),
            metrics=layer_metrics,
            affinity=affinity,
            unit_cost=unit_cost,
        )
        layer_data[layer_name] = data

        remove_count = max(
            1,
            min(
                len(layer_metrics) - 1,
                int(round(args.mask_ratio * len(layer_metrics))),
            ),
        )
        selected = select_units_for_layer(
            data,
            remove_count,
            args.candidate_multiplier,
            args.scalar_risk,
        )

        for strategy, (units, trace) in selected.items():
            if strategy not in strategies:
                continue
            for unit in units:
                selection_records.append(
                    {
                        "strategy": strategy,
                        "layer": layer_name,
                        "unit_type": data.unit_type,
                        "unit_index": int(unit),
                        "d_abs": float(layer_metrics.loc[unit, "d_abs"]),
                        "d_rel": float(layer_metrics.loc[unit, "d_rel"]),
                        "unit_cost": float(unit_cost[unit]),
                    }
                )
            trace = trace.copy()
            trace.insert(0, "layer", layer_name)
            trace.insert(0, "strategy", strategy)
            traces.append(trace)

    selections_df = pd.DataFrame(selection_records)
    selections_df.to_csv(output_dir / "selected_units.csv", index=False)
    if traces:
        pd.concat(traces, ignore_index=True).to_csv(
            output_dir / "greedy_selection_trace.csv", index=False
        )

    strategy_selections: Dict[str, Dict[str, List[int]]] = {}
    for strategy in strategies:
        strategy_selections[strategy] = {}
        table = selections_df[selections_df["strategy"] == strategy]
        for layer_name, layer_table in table.groupby("layer"):
            strategy_selections[strategy][layer_name] = (
                layer_table["unit_index"].astype(int).tolist()
            )

    print("\nOffline selections complete. Loading dataset/model for masking validation...", flush=True)
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
    model_specs = {spec.name: spec for spec in discover_unit_layers(model)}

    missing = sorted(set(layers) - set(model_specs))
    if missing:
        raise KeyError(f"Layers missing from model: {missing}")

    cache = collect_cache(loader, args.ablation_videos)
    per_video_frames = []
    strategy_rows = []

    for strategy in strategies:
        frame = evaluate_strategy(
            model=model,
            layer_specs=model_specs,
            selections=strategy_selections[strategy],
            cache=cache,
            device=device,
            target_mode=args.target_mode,
        )
        frame.insert(0, "strategy", strategy)
        per_video_frames.append(frame)

        global_coverage, layer_coverage = strategy_coverage(
            layer_data, strategy_selections[strategy]
        )
        selected_table = selections_df[selections_df["strategy"] == strategy]
        strategy_rows.append(
            {
                "strategy": strategy,
                "masked_units": int(len(selected_table)),
                "masked_cost": float(selected_table["unit_cost"].sum()),
                "mean_target_logit_drop": float(frame["target_logit_drop"].mean()),
                "median_target_logit_drop": float(frame["target_logit_drop"].median()),
                "mean_margin_drop": float(frame["margin_drop"].mean()),
                "prediction_change_rate": float(frame["prediction_changed"].mean()),
                "retained_pattern_coverage": global_coverage,
                "layer_coverage_json": json.dumps(
                    layer_coverage, ensure_ascii=False
                ),
            }
        )

    per_video = pd.concat(per_video_frames, ignore_index=True)
    per_video.to_csv(output_dir / "strategy_masking_per_video.csv", index=False)
    strategy_summary = pd.DataFrame(strategy_rows)
    strategy_summary.to_csv(output_dir / "strategy_comparison.csv", index=False)

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(strategy_summary["strategy"], strategy_summary["mean_target_logit_drop"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean target-logit drop")
    plt.title("Masked-subset damage under equal unit budgets")
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_logit_drop.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(
        strategy_summary["strategy"],
        strategy_summary["retained_pattern_coverage"],
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Retained contribution-pattern coverage")
    plt.title("Functional coverage after masking")
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_pattern_coverage.png", dpi=220)
    plt.close()

    summary = {
        "model_metadata": model_metadata,
        "chosen_classes": chosen_classes,
        "selected_clips": selected_indices,
        "method": {
            "scalar_only": "lowest scalar risk",
            "pattern_only": "minimum greedy contribution-pattern coverage loss",
            "scalar_candidate_pattern_coverage": (
                "low scalar-risk candidate pool followed by minimum coverage loss"
            ),
        },
        "strategy_results": strategy_rows,
        "run_config": vars(args),
        "decision_rule": [
            "The hybrid strategy should retain more pattern coverage than scalar-only.",
            "The hybrid strategy should have no larger target-logit or margin drop than scalar-only.",
            "Pattern-only may preserve coverage but can mask high-risk units; it is an ablation, not the recommended method.",
            "Do not integrate into physical pruning if the hybrid advantage is inconsistent across layers or videos.",
        ],
        "limitations": [
            "This probe masks units but does not rebuild the physical network.",
            "Equal unit ratios are used unless parameter-count columns exist in unit_metrics.csv.",
            "The current contribution patterns come from only the saved calibration videos.",
            "The probe validates subset selection, not post-finetuning accuracy.",
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_lines = [
        "Contribution-Pattern Coverage Pruning Probe",
        "=" * 88,
        "",
        "This probe compares three masking strategies under the same per-layer unit budget.",
        "",
    ]
    for row in strategy_rows:
        report_lines += [
            f"[{row['strategy']}]",
            f"masked_units: {row['masked_units']}",
            f"masked_cost: {row['masked_cost']}",
            f"mean_target_logit_drop: {row['mean_target_logit_drop']}",
            f"mean_margin_drop: {row['mean_margin_drop']}",
            f"prediction_change_rate: {row['prediction_change_rate']}",
            f"retained_pattern_coverage: {row['retained_pattern_coverage']}",
            "",
        ]
    report_lines += [
        "Interpretation:",
        "- scalar_only is the baseline.",
        "- pattern_only tests whether coverage alone is sufficient.",
        "- scalar_candidate_pattern_coverage is the proposed two-stage rule.",
        "- Continue only if the hybrid improves retained coverage without increasing damage.",
    ]
    (output_dir / "COVERAGE_PRUNING_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Coverage pruning probe complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

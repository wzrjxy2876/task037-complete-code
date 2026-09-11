#!/usr/bin/env python3
"""Task041 collective temporal coverage pruning oracle.

This module is diagnostic-only. It reuses the Task040 fixed-cardinality
temporal interventions and the Task037 Video Swin/UCF101 model, checkpoint,
pruning-unit discovery, and temporary whole-unit masking semantics. It never
physically prunes weights and never fine-tunes the model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import random
import sys
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


TASK041_DOMAINS = ("271", "297", "269", "400", "415", "102", "103", "113", "76")
SPANS = (1, 2, 4, 8, 16)
PROFILE_COLUMNS = tuple(f"g_span_{span}" for span in SPANS)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task041 collective temporal coverage pruning oracle"
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profiles_csv", required=True)
    parser.add_argument("--domain_mapping_csv", required=True)
    parser.add_argument("--pooled_summary_csv", required=True)
    parser.add_argument("--replaceability_csv", required=True)
    parser.add_argument("--raw_records_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val_list", default="")
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--videos_per_class", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eps", type=float, default=EPS)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_importable(project_root: Path) -> None:
    for candidate in (
        project_root,
        project_root / "src" / "lgfr_runtime",
        project_root / "src",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    return device


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def float_value(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def compute_collective_coverage(
    profile_rows: Sequence[Mapping[str, str]],
    domains: Sequence[str] = TASK041_DOMAINS,
    eps: float = EPS,
) -> list[dict[str, Any]]:
    """Compute M_G, leave-one-out backup, delta, and R_MCTC in float64.

    The input g values are already generated Task040 fixed-cardinality span
    profiles. No profile normalization, clipping, or replacement interpretation
    is applied here.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    requested = set(domains)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for raw in profile_rows:
        domain = str(raw["domain_id"])
        if domain in requested:
            grouped[domain].append(dict(raw))
    missing = [domain for domain in domains if domain not in grouped]
    if missing:
        raise ValueError(f"missing requested Task041 domains: {missing}")
    result: list[dict[str, Any]] = []
    for domain in domains:
        rows = sorted(
            grouped[domain],
            key=lambda row: int(row["task037_global_index"]),
        )
        if len(rows) < 2:
            raise ValueError(f"domain {domain} needs at least two tested units")
        matrix = np.asarray(
            [[float_value(row, key) for key in PROFILE_COLUMNS] for row in rows],
            dtype=np.float64,
        )
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError(f"invalid nonnegative g profile in domain {domain}")
        envelope = np.max(matrix, axis=0).astype(np.float64, copy=False)
        for index, row in enumerate(rows):
            backup = np.max(
                np.delete(matrix, index, axis=0), axis=0
            ).astype(np.float64, copy=False)
            delta = (envelope - backup) / (envelope + np.float64(eps))
            if not np.isfinite(delta).all():
                raise ValueError("non-finite coverage loss")
            r_mctc = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            out: dict[str, Any] = {
                "domain_id": domain,
                "candidate_task040_global_index": int(row["unit_global_index"]),
                "candidate_task037_global_index": int(row["task037_global_index"]),
                "candidate_layer_name": row["layer_name"],
                "candidate_unit_type": row["unit_type"],
                "candidate_unit_index": int(row["unit_index"]),
                "candidate_stage": int(row["stage"]),
                "full_frozen_domain_size": "",
                "tested_unit_count": len(rows),
                "tested_head_count": sum(
                    item["unit_type"] == "head" for item in rows
                ),
                "tested_neuron_count": sum(
                    item["unit_type"] == "neuron" for item in rows
                ),
                "mixed_type": len({item["unit_type"] for item in rows}) > 1,
                "coverage_risk": r_mctc,
                "R_MCTC": r_mctc,
            }
            for span, value in zip(SPANS, matrix[index]):
                out[f"g_span_{span}"] = float(value)
            for span, value in zip(SPANS, envelope):
                out[f"M_G_span_{span}"] = float(value)
            for span, value in zip(SPANS, backup):
                out[f"backup_M_minus_i_span_{span}"] = float(value)
            for span, value in zip(SPANS, delta):
                out[f"delta_span_{span}"] = float(value)
            result.append(out)
    return result


def select_low_high(
    coverage_rows: Sequence[Mapping[str, Any]],
    domains: Sequence[str] = TASK041_DOMAINS,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        grouped[str(row["domain_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for domain in domains:
        rows = sorted(
            grouped[domain],
            key=lambda row: (
                float(row["R_MCTC"]),
                int(row["candidate_task037_global_index"]),
            ),
        )
        if len(rows) < 2:
            raise ValueError(f"domain {domain} has fewer than two candidates")
        for role, row in (("low", rows[0]), ("high", rows[-1])):
            selected.append({**dict(row), "candidate_role": role})
    return selected


def attach_domain_sizes(
    coverage_rows: Sequence[dict[str, Any]],
    mapping_rows: Sequence[Mapping[str, str]],
) -> None:
    # The D3 mapping contains the authoritative full-domain size for each
    # frozen BMS domain, while D1 added extra tested members not present in the
    # compact n03 mapping. Join the size by domain, not by candidate row.
    by_domain: dict[str, int] = {}
    for mapping in mapping_rows:
        domain = str(mapping["domain_id"])
        size = int(mapping["full_domain_size"])
        previous = by_domain.get(domain)
        if previous is not None and previous != size:
            raise ValueError(f"inconsistent frozen size for domain {domain}")
        by_domain[domain] = size
    for row in coverage_rows:
        domain = str(row["domain_id"])
        if domain not in by_domain:
            raise ValueError(f"no frozen BMS size for domain {domain}")
        row["full_frozen_domain_size"] = by_domain[domain]


def rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return correlation(rankdata(x), rankdata(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            dx = a[i] - a[j]
            dy = b[i] - b[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


def wilcoxon_signed_rank(values: Sequence[float]) -> dict[str, Any]:
    differences = np.asarray(values, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0.0]
    if len(differences) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "all_zero"}
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy.stats.wilcoxon",
            "n_nonzero": int(len(differences)),
        }
    except Exception as exc:
        return {
            "statistic": None,
            "p_value": None,
            "method": "unavailable",
            "n_nonzero": int(len(differences)),
            "error": str(exc),
        }


def unwrap_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"cannot extract logits from {type(output)!r}")


def make_batch_loader(
    ctfrs: Any,
    project_root: Path,
    val_list: str,
    frame_root: str,
    num_workers: int,
    seed: int,
    num_classes: int,
    videos_per_class: int,
    batch_size: int,
) -> tuple[DataLoader, list[int], list[int]]:
    base_loader, selected_indices, chosen_classes = ctfrs.build_balanced_loader(
        project_root=project_root,
        val_list=val_list,
        frame_root=frame_root,
        num_classes=num_classes,
        videos_per_class=videos_per_class,
        num_workers=num_workers,
        seed=seed,
    )
    loader = DataLoader(
        base_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=getattr(ctfrs, "_seed_worker", None),
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, selected_indices, chosen_classes


def video_manifest_rows(loader: DataLoader, selected_indices: Sequence[int]) -> list[dict[str, Any]]:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    rows = []
    for order, index in enumerate(selected_indices):
        directory, duration, label = dataset.clips[int(index)]
        rows.append(
            {
                "video_index": order,
                "dataset_index": int(index),
                "video_id": str(directory),
                "duration": int(duration),
                "label": int(label),
            }
        )
    return rows


def metric_tensors(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"expected [B,C] logits, got {tuple(logits.shape)}")
    with torch.no_grad():
        ce = F.cross_entropy(logits, targets, reduction="none")
        top1 = logits.argmax(dim=1).eq(targets)
        k = min(5, logits.shape[1])
        top5 = logits.topk(k=k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)
    return {
        "true_logit": logits.gather(1, targets.unsqueeze(1)).squeeze(1),
        "ce": ce,
        "top1": top1,
        "top5": top5,
        "pred": logits.argmax(dim=1),
    }


def collect_baseline(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    records: list[dict[str, Any]] = []
    first_batch: list[torch.Tensor] = []
    video_order = 0
    with torch.no_grad():
        for videos, targets, dataset_indices in loader:
            videos = videos.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = unwrap_logits(model(videos))
            values = metric_tensors(logits, targets)
            if not first_batch:
                first_batch = [videos.detach().clone(), logits.detach().clone()]
            for i in range(int(targets.shape[0])):
                records.append(
                    {
                        "video_index": video_order,
                        "dataset_index": int(dataset_indices[i].item()),
                        "label": int(targets[i].item()),
                        "baseline_true_logit": float(values["true_logit"][i].item()),
                        "baseline_ce": float(values["ce"][i].item()),
                        "baseline_top1": bool(values["top1"][i].item()),
                        "baseline_top5": bool(values["top5"][i].item()),
                        "baseline_pred": int(values["pred"][i].item()),
                    }
                )
                video_order += 1
    expected = len(loader.dataset)
    if len(records) != expected:
        raise AssertionError(f"baseline record count {len(records)} != {expected}")
    return records, first_batch


def collect_masked_damage(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    spec: Any,
    unit_index: int,
    baseline: Sequence[Mapping[str, Any]],
    temporary_mask: Any,
) -> dict[str, Any]:
    drops: list[float] = []
    ce_increases: list[float] = []
    flips: list[float] = []
    top1_changes: list[float] = []
    top5_changes: list[float] = []
    order = 0
    with temporary_mask(spec, unit_index):
        with torch.no_grad():
            for videos, targets, dataset_indices in loader:
                videos = videos.float().to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()
                logits = unwrap_logits(model(videos))
                values = metric_tensors(logits, targets)
                for i in range(int(targets.shape[0])):
                    reference = baseline[order]
                    observed_index = int(dataset_indices[i].item())
                    if observed_index != int(reference["dataset_index"]):
                        raise AssertionError(
                            "candidate loaders did not use the same 30 videos"
                        )
                    drops.append(
                        float(reference["baseline_true_logit"])
                        - float(values["true_logit"][i].item())
                    )
                    ce_increases.append(
                        float(values["ce"][i].item()) - float(reference["baseline_ce"])
                    )
                    flips.append(
                        float(
                            int(values["pred"][i].item())
                            != int(reference["baseline_pred"])
                        )
                    )
                    top1_changes.append(
                        float(values["top1"][i].item())
                        - float(reference["baseline_top1"])
                    )
                    top5_changes.append(
                        float(values["top5"][i].item())
                        - float(reference["baseline_top5"])
                    )
                    order += 1
    if order != len(baseline):
        raise AssertionError("masked record count differs from baseline")
    baseline_top1 = np.asarray(
        [r["baseline_top1"] for r in baseline], dtype=np.float64
    )
    baseline_top5 = np.asarray(
        [r["baseline_top5"] for r in baseline], dtype=np.float64
    )
    changes1 = np.asarray(top1_changes, dtype=np.float64)
    changes5 = np.asarray(top5_changes, dtype=np.float64)
    return {
        "true_class_logit_drop": float(np.mean(drops, dtype=np.float64)),
        "cross_entropy_increase": float(np.mean(ce_increases, dtype=np.float64)),
        "prediction_flip_rate": float(np.mean(flips, dtype=np.float64)),
        "top1_accuracy_change": float(np.mean(changes1, dtype=np.float64)),
        "top5_accuracy_change": float(np.mean(changes5, dtype=np.float64)),
        "baseline_top1_accuracy": float(np.mean(baseline_top1, dtype=np.float64)),
        "baseline_top5_accuracy": float(np.mean(baseline_top5, dtype=np.float64)),
        "masked_top1_accuracy": float(np.mean(baseline_top1 + changes1, dtype=np.float64)),
        "masked_top5_accuracy": float(np.mean(baseline_top5 + changes5, dtype=np.float64)),
        "n_videos": len(baseline),
    }


def add_mask_metrics(
    coverage_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    masking_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for raw in masking_rows:
        key = (str(raw["candidate_layer_name"]), int(raw["candidate_unit_index"]))
        coverage = coverage_by_key[key]
        rows.append({**dict(coverage), **dict(raw)})
    return rows


def domain_rank_analysis(
    masking_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_domain: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in masking_rows:
        by_domain[str(row["domain_id"])].append(
            (
                float(row["R_MCTC"]),
                float(row["true_class_logit_drop"]),
                float(row["cross_entropy_increase"]),
                float(row["prediction_flip_rate"]),
            )
        )
    output: dict[str, Any] = {}
    for domain, values in sorted(by_domain.items(), key=lambda item: int(item[0])):
        if len(values) < 3:
            continue
        arr = np.asarray(values, dtype=np.float64)
        output[domain] = {
            "n_units": int(len(values)),
            "spearman_R_MCTC_vs_true_class_logit_drop": spearman(arr[:, 0], arr[:, 1]),
            "kendall_R_MCTC_vs_true_class_logit_drop": kendall_tau_b(
                arr[:, 0], arr[:, 1]
            ),
            "spearman_R_MCTC_vs_cross_entropy_increase": spearman(
                arr[:, 0], arr[:, 2]
            ),
            "kendall_R_MCTC_vs_cross_entropy_increase": kendall_tau_b(
                arr[:, 0], arr[:, 2]
            ),
            "spearman_R_MCTC_vs_prediction_flip_rate": spearman(
                arr[:, 0], arr[:, 3]
            ),
            "kendall_R_MCTC_vs_prediction_flip_rate": kendall_tau_b(
                arr[:, 0], arr[:, 3]
            ),
        }
    means = list(output.values())
    if means:
        for key in (
            "spearman_R_MCTC_vs_true_class_logit_drop",
            "kendall_R_MCTC_vs_true_class_logit_drop",
            "spearman_R_MCTC_vs_cross_entropy_increase",
            "kendall_R_MCTC_vs_cross_entropy_increase",
            "spearman_R_MCTC_vs_prediction_flip_rate",
            "kendall_R_MCTC_vs_prediction_flip_rate",
        ):
            output[f"domain_balanced_mean_{key}"] = float(
                np.mean([value[key] for value in means], dtype=np.float64)
            )
    return output


def build_ordering_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    masking_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["candidate_layer_name"]), int(row["candidate_unit_index"])): row
        for row in masking_rows
    }
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected_rows:
        grouped[str(row["domain_id"])][str(row["candidate_role"])] = row
    output = []
    metrics = (
        "true_class_logit_drop",
        "cross_entropy_increase",
        "prediction_flip_rate",
        "top1_accuracy_change",
        "top5_accuracy_change",
    )
    for domain in TASK041_DOMAINS:
        low = grouped[domain]["low"]
        high = grouped[domain]["high"]
        low_damage = by_key[
            (str(low["candidate_layer_name"]), int(low["candidate_unit_index"]))
        ]
        high_damage = by_key[
            (str(high["candidate_layer_name"]), int(high["candidate_unit_index"]))
        ]
        row: dict[str, Any] = {
            "domain_id": domain,
            "low_task037_global_index": int(low["candidate_task037_global_index"]),
            "high_task037_global_index": int(high["candidate_task037_global_index"]),
            "low_R_MCTC": float(low["R_MCTC"]),
            "high_R_MCTC": float(high["R_MCTC"]),
            "delta_R_MCTC_high_minus_low": float(high["R_MCTC"]) - float(low["R_MCTC"]),
            "low_unit_type": low["candidate_unit_type"],
            "high_unit_type": high["candidate_unit_type"],
            "low_layer_name": low["candidate_layer_name"],
            "high_layer_name": high["candidate_layer_name"],
            "low_unit_index": int(low["candidate_unit_index"]),
            "high_unit_index": int(high["candidate_unit_index"]),
            "low_high_same_unit": bool(
                int(low["candidate_task037_global_index"])
                == int(high["candidate_task037_global_index"])
            ),
            "mixed_type": bool(low["mixed_type"]),
        }
        for metric in metrics:
            low_value = float(low_damage[metric])
            high_value = float(high_damage[metric])
            row[f"low_{metric}"] = low_value
            row[f"high_{metric}"] = high_value
            row[f"high_minus_low_{metric}"] = high_value - low_value
            row[f"high_higher_{metric}"] = high_value > low_value
            row[f"low_equal_high_{metric}"] = high_value == low_value
        output.append(row)
    return output


def derive_baselines_from_raw(
    raw_rows: Sequence[Mapping[str, str]],
) -> dict[int, dict[str, float]]:
    """Reproduce the frozen Task040 pooled metrics for D1-added units.

    Task040 D1 raw records contain three videos and 16 fixed-cardinality
    interventions per span. The formulas below are the historical C-orthogonal
    and pooled-metric definitions; they do not infer or alter BMS assignments.
    """
    grouped: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for raw in raw_rows:
        grouped[int(raw["unit_global_index"])].append(raw)
    output: dict[int, dict[str, float]] = {}
    for unit_id, rows in grouped.items():
        span_values: dict[int, dict[str, float]] = {}
        for span in SPANS:
            selected = [row for row in rows if int(row["block_size"]) == span]
            if not selected:
                raise ValueError(f"raw baseline has no span {span} for unit {unit_id}")
            a = np.asarray([float(row["z_true_original"]) for row in selected], dtype=np.float64)
            b = np.asarray([float(row["z_true_original_masked"]) for row in selected], dtype=np.float64)
            c = np.asarray([float(row["z_true_intervened"]) for row in selected], dtype=np.float64)
            d = np.asarray([float(row["z_true_intervened_masked"]) for row in selected], dtype=np.float64)
            tau = np.asarray([float(row["tau"]) for row in selected], dtype=np.float64)
            c_struct = a - b + c - d
            c_temp = a + b - c - d
            c_interaction = a - b - c + d
            ss_struct = float(np.sum(c_struct * c_struct, dtype=np.float64))
            ss_temp = float(np.sum(c_temp * c_temp, dtype=np.float64))
            ss_interaction = float(np.sum(c_interaction * c_interaction, dtype=np.float64))
            total = ss_struct + ss_temp + ss_interaction
            span_values[span] = {
                "G_RMS": float(np.sqrt(np.mean(c_interaction * c_interaction, dtype=np.float64))),
                "HTOR": float(np.sqrt(np.mean(tau * tau, dtype=np.float64))),
                "PTR": float(ss_interaction / (total + np.float64(EPS))),
            }
        unique_videos = {}
        for row in rows:
            unique_videos[int(row["video_index"])] = float(row["d_original"])
        output[unit_id] = {
            "mean_abs_d_original": float(
                np.mean(np.abs(np.asarray(list(unique_videos.values()), dtype=np.float64)))
            ),
            "G_RMS": float(
                np.sqrt(np.mean([span_values[span]["G_RMS"] ** 2 for span in SPANS], dtype=np.float64))
            ),
            "HTOR": float(
                np.sqrt(np.mean([span_values[span]["HTOR"] ** 2 for span in SPANS], dtype=np.float64))
            ),
            "PTR": float(
                np.mean([span_values[span]["PTR"] for span in SPANS], dtype=np.float64)
            ),
        }
    return output


def baseline_comparison_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, str]],
    replaceability_rows: Sequence[Mapping[str, str]],
    raw_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    pooled_by_key = {int(row["unit_global_index"]): row for row in pooled_rows}
    raw_baselines = derive_baselines_from_raw(raw_rows)
    replace_by_key = {
        int(row["candidate_task037_global_index"]): row
        for row in replaceability_rows
    }
    output = []
    for candidate in selected_rows:
        task040_index = int(candidate["candidate_task040_global_index"])
        task037_index = int(candidate["candidate_task037_global_index"])
        pooled = pooled_by_key.get(task040_index)
        derived = raw_baselines.get(task040_index)
        replace = replace_by_key.get(task037_index)
        if pooled is None and derived is None:
            raise ValueError(
                f"missing pooled/raw baseline for Task040 unit {task040_index}"
            )
        if replace is None:
            raise ValueError(
                f"missing replaceability baseline for Task037 unit {task037_index}"
            )
        baseline = derived if pooled is None else {
            "mean_abs_d_original": float(pooled["mean_abs_d_original"]),
            "G_RMS": float(pooled["G_RMS"]),
            "HTOR": float(pooled["HTOR"]),
            "PTR": float(pooled["PTR"]),
        }
        output.append(
            {
                "domain_id": candidate["domain_id"],
                "candidate_role": candidate["candidate_role"],
                "candidate_task037_global_index": task037_index,
                "candidate_task040_global_index": task040_index,
                "layer_name": candidate["candidate_layer_name"],
                "unit_type": candidate["candidate_unit_type"],
                "unit_index": candidate["candidate_unit_index"],
                "mean_abs_d_original": baseline["mean_abs_d_original"],
                "G_RMS": baseline["G_RMS"],
                "old_HTOR": baseline["HTOR"],
                "PTR": baseline["PTR"],
                "corrected_pairwise_best_E": float(replace["best_E"]),
                "R_MCTC": float(candidate["R_MCTC"]),
                "coverage_risk": float(candidate["coverage_risk"]),
                "baseline_source": "task040_n03_pooled" if pooled is not None else "task040_d1_raw_reproduced",
            }
        )
    return output

def mixed_domain_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    masking_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_keys = {
        (
            str(row["candidate_layer_name"]),
            int(row["candidate_unit_index"]),
        ): str(row["candidate_role"])
        for row in selected_rows
        if str(row["domain_id"]) in ("271", "297")
    }
    output = []
    for row in masking_rows:
        if str(row["domain_id"]) not in ("271", "297"):
            continue
        key = (str(row["candidate_layer_name"]), int(row["candidate_unit_index"]))
        output.append(
            {
                "domain_id": row["domain_id"],
                "candidate_role": selected_keys.get(key, "all_domain_units"),
                "candidate_task037_global_index": row["candidate_task037_global_index"],
                "candidate_task040_global_index": row["candidate_task040_global_index"],
                "layer_name": row["candidate_layer_name"],
                "unit_type": row["candidate_unit_type"],
                "unit_index": row["candidate_unit_index"],
                "R_MCTC": row["R_MCTC"],
                "coverage_risk": row["coverage_risk"],
                "true_class_logit_drop": row["true_class_logit_drop"],
                "cross_entropy_increase": row["cross_entropy_increase"],
                "prediction_flip_rate": row["prediction_flip_rate"],
                "top1_accuracy_change": row["top1_accuracy_change"],
                "top5_accuracy_change": row["top5_accuracy_change"],
                "backup_is_not_full_replacement": True,
                "analysis_note": (
                    "Leave-one-out backup is a domain envelope diagnostic; "
                    "it is not a full replacement intervention."
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    profiles_csv = Path(args.profiles_csv).expanduser().resolve()
    mapping_csv = Path(args.domain_mapping_csv).expanduser().resolve()
    pooled_csv = Path(args.pooled_summary_csv).expanduser().resolve()
    replace_csv = Path(args.replaceability_csv).expanduser().resolve()
    raw_csv = Path(args.raw_records_csv).expanduser().resolve()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader configuration")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for path in (profiles_csv, mapping_csv, pooled_csv, replace_csv, raw_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_importable(project_root)
    set_seed(args.seed)
    device = resolve_device(args.device)

    coverage_rows = compute_collective_coverage(
        read_csv(profiles_csv), domains=TASK041_DOMAINS, eps=args.eps
    )
    attach_domain_sizes(coverage_rows, read_csv(mapping_csv))
    selected_rows = select_low_high(coverage_rows)

    ctfrs = importlib.import_module("probe_ctfrs_dynamic_function")
    adapter = importlib.import_module("ucf101_videoswin_probe_adapter_v2")
    task040_probe = importlib.import_module("task040_htor_probe")
    core = importlib.import_module("task040_htor_core")
    model, adapter_metadata = adapter.build_model_for_probe(
        checkpoint=str(checkpoint), device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_specs = ctfrs.discover_unit_layers(model)
    spec_by_key = {(spec.name, spec.unit_type): spec for spec in all_specs}
    identity = task040_probe.make_checkpoint_identity(
        model, checkpoint, adapter_metadata, all_specs
    )
    identity.update(
        {
            "task": "task041",
            "device": str(device),
            "dtype": "torch.float32",
            "amp": False,
            "git_branch": task040_probe.git_value(
                project_root, "rev-parse", "--abbrev-ref", "HEAD"
            ),
            "git_commit": task040_probe.git_value(project_root, "rev-parse", "HEAD"),
            "task040_semantics_source": {
                "intervention_core": "task040_htor_core.py",
                "temporary_mask": "task040_htor_probe.py",
                "unit_discovery": "probe_ctfrs_dynamic_function.py",
                "model_adapter": "ucf101_videoswin_probe_adapter_v2.py",
            },
        }
    )
    write_json(output_dir / "task041_checkpoint_identity.json", identity)

    loader, selected_indices, selected_classes = make_batch_loader(
        ctfrs,
        project_root,
        args.val_list,
        args.frame_root,
        args.num_workers,
        args.seed,
        args.num_classes,
        args.videos_per_class,
        args.batch_size,
    )
    manifest = video_manifest_rows(loader, selected_indices)
    write_csv(output_dir / "task041_video_manifest.csv", manifest)

    baseline, first_batch = collect_baseline(model, loader, device)
    expected_videos = args.num_classes * args.videos_per_class
    if len(baseline) != expected_videos or expected_videos != 30:
        raise AssertionError("Task041 requires exactly 10 classes x 3 videos")
    first_videos, first_logits = first_batch
    actual_t = int(first_videos.shape[2])
    if actual_t != 32:
        raise RuntimeError(f"Task041 requires T=32, received T={actual_t}")
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(actual_t)
    if len(interventions) != 80:
        raise AssertionError("Task041 requires 80 fixed-cardinality interventions")
    core.verify_intervention_identity(interventions, actual_t)
    write_json(
        output_dir / "task041_intervention_manifest.json",
        {
            "task": "task041",
            "source": "task040_htor_core.enumerate_fixed_cardinality_temporal_pairs",
            "actual_T": actual_t,
            "spans": list(SPANS),
            "pairs_per_span": actual_t // 2,
            "num_interventions": len(interventions),
            "two_frame_swap": True,
            "interventions": [asdict(item) for item in interventions],
        },
    )
    identity["baseline_model_sanity"] = {
        "actual_T": actual_t,
        "num_videos": len(baseline),
        "selected_classes": [int(value) for value in selected_classes],
        "num_interventions": len(interventions),
        "intervention_spans": list(SPANS),
        "interventions_per_span": {
            str(span): sum(item.block_size == span for item in interventions)
            for span in SPANS
        },
        "FP32": True,
        "AMP": False,
    }
    write_json(output_dir / "task041_checkpoint_identity.json", identity)

    coverage_by_key = {
        (str(row["candidate_layer_name"]), int(row["candidate_unit_index"])): row
        for row in coverage_rows
    }
    selected_key_roles = {
        (
            str(row["candidate_layer_name"]),
            int(row["candidate_unit_index"]),
        ): str(row["candidate_role"])
        for row in selected_rows
    }
    all_candidates = sorted(
        coverage_rows,
        key=lambda row: (
            int(row["domain_id"]),
            int(row["candidate_task037_global_index"]),
        ),
    )
    masking_rows: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm
        iterator: Iterable[Mapping[str, Any]] = tqdm(
            all_candidates, desc="Task041 masking", ncols=100
        )
    except ImportError:
        iterator = all_candidates
    for candidate in iterator:
        key = (
            str(candidate["candidate_layer_name"]),
            str(candidate["candidate_unit_type"]),
        )
        spec = spec_by_key.get(key)
        if spec is None:
            raise ValueError(f"model pruning layer not found: {key}")
        actual = collect_masked_damage(
            model,
            loader,
            device,
            spec,
            int(candidate["candidate_unit_index"]),
            baseline,
            task040_probe.temporary_unit_mask,
        )
        masking_rows.append(
            {
                "domain_id": candidate["domain_id"],
                "candidate_role": selected_key_roles.get(
                    (
                        str(candidate["candidate_layer_name"]),
                        int(candidate["candidate_unit_index"]),
                    ),
                    "all_domain_units",
                ),
                "candidate_task037_global_index": candidate[
                    "candidate_task037_global_index"
                ],
                "candidate_task040_global_index": candidate[
                    "candidate_task040_global_index"
                ],
                "candidate_layer_name": candidate["candidate_layer_name"],
                "candidate_unit_type": candidate["candidate_unit_type"],
                "candidate_unit_index": candidate["candidate_unit_index"],
                **actual,
                "mask_mode": "temporary_whole_unit_mask",
                "physical_pruning": False,
                "finetuning": False,
            }
        )

    with torch.no_grad():
        restored_logits = unwrap_logits(model(first_videos))
    mask_restore_exact = bool(torch.equal(restored_logits, first_logits))
    if not mask_restore_exact:
        raise RuntimeError("temporary masking did not restore exact model output")

    enriched_masking = add_mask_metrics(coverage_by_key, masking_rows)
    ordering_rows = build_ordering_rows(selected_rows, enriched_masking)
    baseline_rows = baseline_comparison_rows(
        selected_rows,
        read_csv(pooled_csv),
        read_csv(replace_csv),
        read_csv(raw_csv),
    )
    mixed_rows = mixed_domain_rows(selected_rows, enriched_masking)
    rank_stats = domain_rank_analysis(enriched_masking)

    write_csv(output_dir / "task041_oracle_candidates.csv", selected_rows)
    write_csv(output_dir / "task041_masking_damage.csv", enriched_masking)
    write_csv(output_dir / "task041_domain_ordering.csv", ordering_rows)
    write_csv(output_dir / "task041_baseline_comparison.csv", baseline_rows)
    write_csv(output_dir / "task041_mixed_domain_analysis.csv", mixed_rows)

    metric_names = (
        "true_class_logit_drop",
        "cross_entropy_increase",
        "prediction_flip_rate",
        "top1_accuracy_change",
        "top5_accuracy_change",
    )
    aggregate: dict[str, Any] = {}
    for metric in metric_names:
        differences = [
            float(row[f"high_minus_low_{metric}"]) for row in ordering_rows
        ]
        aggregate[metric] = {
            "mean_high_minus_low": float(np.mean(differences, dtype=np.float64)),
            "median_high_minus_low": float(np.median(differences)),
            "high_greater_count": int(sum(value > 0 for value in differences)),
            "low_greater_count": int(sum(value < 0 for value in differences)),
            "equal_count": int(sum(value == 0 for value in differences)),
            "wilcoxon_signed_rank": wilcoxon_signed_rank(differences),
        }

    summary = {
        "task": "task041_collective_temporal_coverage_pruning_oracle",
        "domains": list(TASK041_DOMAINS),
        "mandatory_mixed_domains": ["271", "297"],
        "same_type_domains": ["269", "400", "415", "102", "103", "113", "76"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "seed": int(args.seed),
        "num_classes": int(args.num_classes),
        "videos_per_class": int(args.videos_per_class),
        "num_videos": len(baseline),
        "selected_classes": [int(value) for value in selected_classes],
        "selected_video_dataset_indices": [int(value) for value in selected_indices],
        "actual_T": actual_t,
        "spans": list(SPANS),
        "pairs_per_span": actual_t // 2,
        "num_interventions": len(interventions),
        "tested_unit_count": len(coverage_rows),
        "oracle_candidate_count": len(selected_rows),
        "masking_damage_count": len(enriched_masking),
        "R_MCTC_definition": (
            "sqrt(mean_s delta_i(s)^2), delta_i(s)=(M_G(s)-"
            "M_G_minus_i(s))/(M_G(s)+eps), float64, no clipping or normalization"
        ),
        "candidate_selection": {
            "primary_selector": "R_MCTC within each frozen BMS domain",
            "tie_break": "ascending Task037 global index",
            "non_selectors": [
                "mean_abs_d_original",
                "G_RMS",
                "old_HTOR",
                "PTR",
                "corrected_pairwise_best_E",
                "Contribution Field",
                "F3",
            ],
        },
        "input_artifacts": {
            "profiles_csv": str(profiles_csv),
            "domain_mapping_csv": str(mapping_csv),
            "pooled_summary_csv": str(pooled_csv),
            "replaceability_csv": str(replace_csv),
            "raw_records_csv": str(raw_csv),
        },
        "coverage_formula": {
            "M_G": "max_j g_j(s)",
            "M_G_minus_i": "max_{j != i} g_j(s)",
            "g_profile": "sqrt(mean C_interaction^2) from Task040 fixed-cardinality spans",
            "eps": float(args.eps),
            "profile_normalization": False,
        },
        "masking_metrics": list(metric_names),
        "masking_semantics": {
            "temporary_whole_unit_mask": True,
            "physical_pruning": False,
            "finetuning": False,
            "same_30_videos_for_every_candidate": True,
            "same_task040_unit_discovery": True,
            "same_videoswin_ucf101_semantics": True,
        },
        "mask_restore_exact": mask_restore_exact,
        "within_domain_monotonicity": rank_stats,
        "low_high_ordering": aggregate,
        "baseline_metrics": [
            "mean_abs_d_original",
            "G_RMS",
            "old_HTOR",
            "PTR",
            "corrected_pairwise_best_E",
            "R_MCTC",
        ],
        "no_full_pruning": True,
        "no_task042": True,
        "git_branch": identity["git_branch"],
        "git_commit": identity["git_commit"],
    }
    write_json(output_dir / "task041_summary.json", summary)
    report_lines = [
        "# Task041 Collective Temporal Coverage Pruning Oracle",
        "",
        "This is a diagnostic-only oracle. It performs temporary whole-unit masking",
        "on the existing Video Swin/UCF101 model and does not physically prune or",
        "fine-tune the model.",
        "",
        "## Frozen protocol",
        "",
        f"- Checkpoint: {checkpoint}",
        f"- Checkpoint SHA256: {identity['checkpoint_sha256']}",
        f"- Domains: {', '.join(TASK041_DOMAINS)}",
        f"- Validation subset: {len(baseline)} videos ({args.num_classes} classes x {args.videos_per_class})",
        f"- Temporal length: T={actual_t}; spans: {', '.join(map(str, SPANS))}; 16 pairs/span; 80 interventions",
        f"- Tested units: {len(coverage_rows)}; low/high oracle candidates: {len(selected_rows)}",
        "",
        "## Collective coverage score",
        "",
        "For each BMS domain, the frozen Task040 profile is",
        "g_i(s)=sqrt(mean(C_interaction^2)). The envelope is",
        "M_G(s)=max_j g_j(s), the backup is the max over j != i, and",
        "delta_i(s)=(M_G-M_G_minus_i)/(M_G+eps) in float64. R_MCTC is the",
        "five-span RMS of delta. No profile normalization or clipping is used.",
        "",
        "## Real masking oracle",
        "",
        "All tested units in the nine requested domains were masked temporarily",
        "so within-domain Spearman/Kendall analyses have at least three units where",
        "available. Domain low/high comparisons use the lowest/highest R_MCTC with",
        "Task037 global-index ascending tie-breaks. The leave-one-out backup is not",
        "treated as a full replacement intervention.",
        "",
        "## Outputs",
        "",
        "task041_oracle_candidates.csv: low/high candidates and five g/delta values.",
        "task041_masking_damage.csv: temporary whole-unit masking metrics.",
        "task041_domain_ordering.csv: within-domain low/high ordering and Wilcoxon inputs.",
        "task041_baseline_comparison.csv: same candidates compared with frozen baselines.",
        "task041_mixed_domain_analysis.csv: separate domain 271/297 head/FFN analysis.",
        "task041_summary.json: machine-readable protocol and results.",
        "",
        f"Mask restoration exact: {mask_restore_exact}",
        "",
        "Task041 stops after this oracle report; no Task042, full sparsity run,",
        "physical pruning, or fine-tuning is launched.",
    ]
    (output_dir / "task041_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

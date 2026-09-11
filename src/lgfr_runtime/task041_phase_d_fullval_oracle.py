#!/usr/bin/env python3
"""Task041 Phase D full-validation masking oracle resolution audit.

This diagnostic-only module freezes the existing Task041 N=30 scores and
candidate identities, then evaluates the same 29 temporary whole-unit masks on
the complete UCF101 validation split. It never recomputes temporal profiles or
interventions, never changes BMS assignments, and never physically prunes or
fine-tunes the model.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


EXPECTED_CHECKPOINT_SHA256 = (
    "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
)
SPANS = (1, 2, 4, 8, 16)
MIXED_DOMAINS = ("271", "297")
SAME_TYPE_DOMAINS = ("269", "400", "415", "102", "103", "113", "76")
ALL_DOMAINS = MIXED_DOMAINS + SAME_TYPE_DOMAINS
EXPECTED_VALID_CLIPS = 3783
EXPECTED_UNITS = 29
EXPECTED_BOOTSTRAP_RESAMPLES = 1000
EXPECTED_BOOTSTRAP_SEED = 3407
METRIC_NAMES = (
    "true_class_logit_drop",
    "cross_entropy_increase",
    "prediction_flip_rate",
    "top1_accuracy_change",
    "top5_accuracy_change",
)
DAMAGE_METRICS = ("true_class_logit_drop", "cross_entropy_increase")
SCORE_NAMES = (
    "mean_abs_d_original",
    "G_RMS",
    "old_HTOR",
    "PTR",
    "corrected_pairwise_best_E",
    "R_MCTC",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task041 Phase D full-validation resolution audit"
    )
    parser.add_argument("--phase", choices=("baseline", "mask", "finalize"), required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task041_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--frame_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)
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
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(project_root: Path, *arguments: str) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return 0.0 if denominator == 0.0 else float(np.dot(left, right) / denominator)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return correlation(rankdata(x), rankdata(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
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
    return 0.0 if denominator == 0.0 else (concordant - discordant) / denominator


def wilcoxon_signed_rank(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    array = array[array != 0.0]
    if len(array) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "all_zero",
            "n_nonzero": 0,
        }
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            array,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy.stats.wilcoxon",
            "n_nonzero": int(len(array)),
        }
    except Exception as exc:
        return {
            "statistic": None,
            "p_value": None,
            "method": "unavailable",
            "n_nonzero": int(len(array)),
            "error": str(exc),
        }


def candidate_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int(row["candidate_task037_global_index"]),
        int(row["candidate_task040_global_index"]),
    )


def unit_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["candidate_task037_global_index"]),
        int(row["candidate_task040_global_index"]),
        str(row["candidate_layer_name"]),
        str(row["candidate_unit_type"]),
        int(row["candidate_unit_index"]),
        str(row["domain_id"]),
    )


def load_frozen_units(task041_dir: Path) -> list[dict[str, Any]]:
    path = task041_dir / "task041_masking_damage.csv"
    rows = read_csv(path)
    required = {
        "domain_id",
        "candidate_task037_global_index",
        "candidate_task040_global_index",
        "candidate_layer_name",
        "candidate_unit_type",
        "candidate_unit_index",
        "R_MCTC",
        "coverage_risk",
        *[f"g_span_{span}" for span in SPANS],
        *[f"delta_span_{span}" for span in SPANS],
    }
    if not rows:
        raise ValueError("Task041 masking damage is empty")
    if not required.issubset(rows[0]):
        raise ValueError(
            "Task041 masking damage lacks frozen identity/score columns: "
            f"{sorted(required.difference(rows[0]))}"
        )
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        key = candidate_key(raw)
        normalized = {
            "domain_id": str(raw["domain_id"]),
            "candidate_task037_global_index": key[0],
            "candidate_task040_global_index": key[1],
            "candidate_layer_name": str(raw["candidate_layer_name"]),
            "candidate_unit_type": str(raw["candidate_unit_type"]),
            "candidate_unit_index": int(raw["candidate_unit_index"]),
            "R_MCTC": finite_float(raw["R_MCTC"], "R_MCTC"),
            "coverage_risk": finite_float(raw["coverage_risk"], "coverage_risk"),
        }
        for span in SPANS:
            normalized[f"g_span_{span}"] = finite_float(
                raw[f"g_span_{span}"], f"g_span_{span}"
            )
            normalized[f"delta_span_{span}"] = finite_float(
                raw[f"delta_span_{span}"], f"delta_span_{span}"
            )
        previous = by_key.get(key)
        if previous is not None and unit_identity(previous) != unit_identity(normalized):
            raise ValueError(f"conflicting Task041 identity for candidate {key}")
        by_key[key] = normalized
    if len(by_key) != EXPECTED_UNITS:
        raise ValueError(
            f"Task041 must contain exactly {EXPECTED_UNITS} units, got {len(by_key)}"
        )
    units = sorted(by_key.values(), key=lambda row: (int(row["domain_id"]), candidate_key(row)))
    domains = {str(row["domain_id"]) for row in units}
    if domains != set(ALL_DOMAINS):
        raise ValueError(f"Task041 domains changed: {sorted(domains)}")
    return units


def load_low_high(
    task041_dir: Path,
    units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[int, int]]]:
    path = task041_dir / "task041_oracle_candidates.csv"
    rows = read_csv(path)
    by_key = {candidate_key(row): row for row in units}
    selected: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for raw in rows:
        role = str(raw.get("candidate_role", ""))
        if role not in ("low", "high"):
            raise ValueError(f"invalid Task041 role: {role!r}")
        key = candidate_key(raw)
        if key not in by_key:
            raise ValueError(f"low/high candidate is not among frozen 29 units: {key}")
        frozen = by_key[key]
        if float(raw["R_MCTC"]) != float(frozen["R_MCTC"]):
            raise ValueError(f"R_MCTC changed for candidate {key}")
        for span in SPANS:
            if float(raw[f"g_span_{span}"]) != float(frozen[f"g_span_{span}"]):
                raise ValueError(f"g profile changed for candidate {key}")
            if float(raw[f"delta_span_{span}"]) != float(
                frozen[f"delta_span_{span}"]
            ):
                raise ValueError(f"delta profile changed for candidate {key}")
        domain = str(raw["domain_id"])
        if role in selected[domain]:
            raise ValueError(f"duplicate {role} candidate in domain {domain}")
        selected[domain][role] = key
    if set(selected) != set(ALL_DOMAINS):
        raise ValueError("Task041 low/high candidate domains changed")
    if any(set(value) != {"low", "high"} for value in selected.values()):
        raise ValueError("every Task041 domain must have exactly low and high candidates")
    return dict(selected)


def load_baseline(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    required = {
        "dataset_index",
        "label",
        "true_class_logit",
        "cross_entropy",
        "top1_correct",
        "top5_correct",
        "predicted_class",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("full-validation baseline has missing required columns")
    normalized = []
    for raw in rows:
        normalized.append(
            {
                "dataset_index": int(raw["dataset_index"]),
                "label": int(raw["label"]),
                "true_class_logit": finite_float(
                    raw["true_class_logit"], "true_class_logit"
                ),
                "cross_entropy": finite_float(raw["cross_entropy"], "cross_entropy"),
                "top1_correct": bool(int(raw["top1_correct"])),
                "top5_correct": bool(int(raw["top5_correct"])),
                "predicted_class": int(raw["predicted_class"]),
            }
        )
    normalized.sort(key=lambda row: row["dataset_index"])
    indices = [row["dataset_index"] for row in normalized]
    if len(normalized) != EXPECTED_VALID_CLIPS or len(set(indices)) != len(indices):
        raise ValueError(
            f"baseline must contain {EXPECTED_VALID_CLIPS} unique clips"
        )
    return normalized


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


def metric_tensors(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"expected [B,C] logits, got {tuple(logits.shape)}")
    ce = F.cross_entropy(logits, targets, reduction="none")
    true_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    top1 = logits.argmax(dim=1).eq(targets)
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(
        targets.unsqueeze(1)
    ).any(dim=1)
    return {
        "true_class_logit": true_logit,
        "cross_entropy": ce,
        "top1_correct": top1,
        "top5_correct": top5,
        "predicted_class": logits.argmax(dim=1),
    }


def install_dataset_module(frame_root: str) -> Any:
    import types

    previous_utils = sys.modules.get("utils")
    shim_installed = previous_utils is None
    if shim_installed:
        shim = types.ModuleType("utils")
        shim.UCF_DATA_ROOT = frame_root
        sys.modules["utils"] = shim
    try:
        module = importlib.import_module("dataset.ucf101")
    finally:
        if shim_installed:
            sys.modules.pop("utils", None)
    module.UCF_DATA_ROOT = frame_root
    return module


def build_full_loader(
    project_root: Path,
    val_list: str,
    frame_root: str,
    num_workers: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, int]:
    dataset_module = install_dataset_module(frame_root)
    spatial_transform, temporal_transform = dataset_module.test_transform()
    dataset = dataset_module.attack_ucf101(
        val_list,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
    )
    invalid = []
    for index, (directory, duration, label) in enumerate(dataset.clips):
        frame_indices = list(range(1, int(duration) + 1))
        sampled = (
            temporal_transform(frame_indices)
            if temporal_transform is not None
            else frame_indices
        )
        reason = ""
        if not os.path.isdir(directory):
            reason = "video directory does not exist"
        elif not sampled:
            reason = "temporal transform returned no frames"
        else:
            missing = [
                value
                for value in sampled
                if not os.path.isfile(
                    os.path.join(directory, f"image_{int(value):05d}.jpg")
                )
            ]
            if missing:
                reason = f"missing sampled frame {missing[0]}"
        if reason:
            invalid.append((index, directory, reason))
    if invalid:
        raise RuntimeError(
            f"full validation has {len(invalid)} invalid clips; "
            f"first={invalid[0]}"
        )
    if len(dataset) != EXPECTED_VALID_CLIPS:
        raise RuntimeError(
            f"expected {EXPECTED_VALID_CLIPS} validation clips, got {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, len(dataset)


def load_model(
    project_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], list[Any]]:
    ensure_importable(project_root)
    adapter = importlib.import_module("ucf101_videoswin_probe_adapter_v2")
    ctfrs = importlib.import_module("probe_ctfrs_dynamic_function")
    task040 = importlib.import_module("task040_htor_probe")
    model, adapter_metadata = adapter.build_model_for_probe(
        checkpoint=str(checkpoint), device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    specs = ctfrs.discover_unit_layers(model)
    identity = task040.make_checkpoint_identity(
        model, checkpoint, adapter_metadata, specs
    )
    identity.update(
        {
            "task": "task041_phase_d",
            "device": str(device),
            "dtype": "torch.float32",
            "amp": False,
            "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
            "git_branch": git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit": git_value(project_root, "rev-parse", "HEAD"),
        }
    )
    if identity["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA256 does not match Task041 authority")
    if identity["missing_keys"] or identity["unexpected_keys"]:
        raise RuntimeError(
            "checkpoint identity gate failed: missing/unexpected keys are non-empty"
        )
    if identity["classifier_head"]["status"] != "loaded":
        raise RuntimeError("classifier head was not loaded")
    return model, identity, specs


def collect_baseline(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    fields = (
        "dataset_index",
        "label",
        "true_class_logit",
        "cross_entropy",
        "top1_correct",
        "top5_correct",
        "predicted_class",
    )
    rows = []
    seen = set()
    first_videos = None
    first_logits = None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for videos, targets, dataset_indices in loader:
                videos = videos.float().to(device, non_blocking=True)
                targets = targets.long().to(device, non_blocking=True)
                logits = unwrap_logits(model(videos))
                values = metric_tensors(logits, targets)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("baseline produced non-finite logits")
                if first_videos is None:
                    first_videos = videos.detach().clone()
                    first_logits = logits.detach().clone()
                for index in range(int(targets.shape[0])):
                    dataset_index = int(dataset_indices[index].item())
                    if dataset_index in seen:
                        raise RuntimeError(
                            f"duplicate baseline dataset_index {dataset_index}"
                        )
                    seen.add(dataset_index)
                    row = {
                        "dataset_index": dataset_index,
                        "label": int(targets[index].item()),
                        "true_class_logit": float(
                            values["true_class_logit"][index].item()
                        ),
                        "cross_entropy": float(values["cross_entropy"][index].item()),
                        "top1_correct": int(values["top1_correct"][index].item()),
                        "top5_correct": int(values["top5_correct"][index].item()),
                        "predicted_class": int(values["predicted_class"][index].item()),
                    }
                    writer.writerow(row)
                    rows.append(row)
    if len(rows) != EXPECTED_VALID_CLIPS:
        raise RuntimeError(f"baseline rows={len(rows)} != {EXPECTED_VALID_CLIPS}")
    rows.sort(key=lambda row: row["dataset_index"])
    if first_videos is None or first_logits is None:
        raise RuntimeError("baseline loader returned no batches")
    return rows, first_videos, first_logits


def validate_baseline_identity(
    output_dir: Path,
    checkpoint: Path,
    val_list: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_dir / "task041_phase_d_baseline_identity.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("stored baseline checkpoint SHA mismatch")
    if stored.get("val_list_sha256") != sha256_file(val_list):
        raise RuntimeError("stored baseline validation-list SHA mismatch")
    if stored.get("git_commit") != identity.get("git_commit"):
        raise RuntimeError("baseline and masking code commits differ")
    if stored.get("valid_clips") != EXPECTED_VALID_CLIPS:
        raise RuntimeError("stored baseline valid clip count mismatch")
    return stored


def phase_baseline(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    val_list = Path(args.val_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and (output_dir / "task041_fullval_baseline_per_sample.csv").exists():
        raise FileExistsError("Phase D baseline already exists; refusing overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_importable(project_root)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, identity, _ = load_model(project_root, checkpoint, device)
    write_json(output_dir / "task041_phase_d_checkpoint_identity.json", identity)
    loader, valid_count = build_full_loader(
        project_root,
        str(val_list),
        args.frame_root,
        args.num_workers,
        args.batch_size,
        args.seed,
    )
    baseline_path = output_dir / "task041_fullval_baseline_per_sample.csv"
    rows, _, _ = collect_baseline(model, loader, device, baseline_path)
    baseline_identity = {
        "task": "task041_phase_d_full_validation_baseline",
        "checkpoint_absolute_path": str(checkpoint),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "val_list_absolute_path": str(val_list),
        "val_list_sha256": sha256_file(val_list),
        "frame_root": str(Path(args.frame_root).resolve()),
        "valid_clips": valid_count,
        "baseline_rows": len(rows),
        "same_videoswin_ucf101_preprocessing": True,
        "same_classification_head_semantics": True,
        "git_branch": identity["git_branch"],
        "git_commit": identity["git_commit"],
        "device": str(device),
        "dtype": "torch.float32",
        "amp": False,
    }
    write_json(output_dir / "task041_phase_d_baseline_identity.json", baseline_identity)
    return {
        "phase": "baseline",
        "valid_clips": valid_count,
        "baseline_rows": len(rows),
        "checkpoint_sha256": identity["checkpoint_sha256"],
    }


def phase_mask(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    project_root = Path(args.project_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    val_list = Path(args.val_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    task041_dir = Path(args.task041_output_dir).resolve()
    baseline_path = output_dir / "task041_fullval_baseline_per_sample.csv"
    baseline = load_baseline(baseline_path)
    set_seed(args.seed + args.shard_index)
    device = resolve_device(args.device)
    model, identity, specs = load_model(project_root, checkpoint, device)
    baseline_identity = validate_baseline_identity(
        output_dir, checkpoint, val_list, identity
    )
    units = load_frozen_units(task041_dir)
    selected = units[args.shard_index :: args.num_shards]
    if not selected:
        raise ValueError("mask shard received no candidates")
    loader, valid_count = build_full_loader(
        project_root,
        str(val_list),
        args.frame_root,
        args.num_workers,
        args.batch_size,
        args.seed,
    )
    if valid_count != len(baseline):
        raise RuntimeError("mask validation count differs from baseline")
    spec_by_key = {
        (str(spec.name), str(spec.unit_type)): spec for spec in specs
    }
    task040 = importlib.import_module("task040_htor_probe")
    first_batch = next(iter(loader))
    with torch.no_grad():
        first_videos = first_batch[0].float().to(device, non_blocking=True)
        first_logits = unwrap_logits(model(first_videos)).detach().clone()
    shard_dir = output_dir / f"shard_{args.shard_index}"
    shard_dir.mkdir(parents=True, exist_ok=False)
    sample_fields = (
        "candidate_task037_global_index",
        "candidate_task040_global_index",
        "domain_id",
        "candidate_layer_name",
        "candidate_unit_type",
        "candidate_unit_index",
        "dataset_index",
        "label",
        "masked_true_class_logit",
        "masked_cross_entropy",
        "masked_top1_correct",
        "masked_top5_correct",
        "masked_predicted_class",
    )
    restore_rows = []
    total_rows = 0
    shard_path = shard_dir / "task041_fullval_masking_per_sample.csv"
    with shard_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        for candidate in selected:
            spec_key = (
                str(candidate["candidate_layer_name"]),
                str(candidate["candidate_unit_type"]),
            )
            if spec_key not in spec_by_key:
                raise RuntimeError(f"model layer missing: {spec_key}")
            spec = spec_by_key[spec_key]
            observed = 0
            seen = set()
            with task040.temporary_unit_mask(
                spec, int(candidate["candidate_unit_index"])
            ):
                with torch.no_grad():
                    for videos, targets, dataset_indices in loader:
                        videos = videos.float().to(device, non_blocking=True)
                        targets = targets.long().to(device, non_blocking=True)
                        logits = unwrap_logits(model(videos))
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("masked inference produced non-finite logits")
                        values = metric_tensors(logits, targets)
                        for index in range(int(targets.shape[0])):
                            dataset_index = int(dataset_indices[index].item())
                            if dataset_index in seen:
                                raise RuntimeError(
                                    f"duplicate masked dataset_index {dataset_index}"
                                )
                            if dataset_index != baseline[observed]["dataset_index"]:
                                raise RuntimeError(
                                    "masking did not use the exact baseline clip order"
                                )
                            writer.writerow(
                                {
                                    "candidate_task037_global_index": candidate[
                                        "candidate_task037_global_index"
                                    ],
                                    "candidate_task040_global_index": candidate[
                                        "candidate_task040_global_index"
                                    ],
                                    "domain_id": candidate["domain_id"],
                                    "candidate_layer_name": candidate[
                                        "candidate_layer_name"
                                    ],
                                    "candidate_unit_type": candidate[
                                        "candidate_unit_type"
                                    ],
                                    "candidate_unit_index": candidate[
                                        "candidate_unit_index"
                                    ],
                                    "dataset_index": dataset_index,
                                    "label": int(targets[index].item()),
                                    "masked_true_class_logit": float(
                                        values["true_class_logit"][index].item()
                                    ),
                                    "masked_cross_entropy": float(
                                        values["cross_entropy"][index].item()
                                    ),
                                    "masked_top1_correct": int(
                                        values["top1_correct"][index].item()
                                    ),
                                    "masked_top5_correct": int(
                                        values["top5_correct"][index].item()
                                    ),
                                    "masked_predicted_class": int(
                                        values["predicted_class"][index].item()
                                    ),
                                }
                            )
                            seen.add(dataset_index)
                            observed += 1
                            total_rows += 1
            with torch.no_grad():
                restored_logits = unwrap_logits(model(first_videos))
            restored = bool(torch.equal(restored_logits, first_logits))
            restore_rows.append(
                {
                    "candidate_task037_global_index": candidate[
                        "candidate_task037_global_index"
                    ],
                    "candidate_task040_global_index": candidate[
                        "candidate_task040_global_index"
                    ],
                    "layer_name": candidate["candidate_layer_name"],
                    "unit_type": candidate["candidate_unit_type"],
                    "unit_index": candidate["candidate_unit_index"],
                    "mask_restored_exact": restored,
                    "sample_count": observed,
                }
            )
            if observed != len(baseline) or len(seen) != len(baseline):
                raise RuntimeError("masked candidate row count is not 3783")
            if not restored:
                raise RuntimeError("temporary mask failed exact restoration")
    if total_rows != len(selected) * len(baseline):
        raise RuntimeError("mask shard row count mismatch")
    meta = {
        "task": "task041_phase_d_mask_shard",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "candidate_count": len(selected),
        "sample_count": len(baseline),
        "row_count": total_rows,
        "candidate_keys": [list(candidate_key(row)) for row in selected],
        "mask_restore": restore_rows,
        "all_mask_restored_exact": all(
            bool(row["mask_restored_exact"]) for row in restore_rows
        ),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "baseline_git_commit": baseline_identity["git_commit"],
        "git_commit": identity["git_commit"],
        "device": str(device),
    }
    write_json(shard_dir / "task041_fullval_mask_shard_meta.json", meta)
    return {
        "phase": "mask",
        "shard_index": args.shard_index,
        "candidate_count": len(selected),
        "row_count": total_rows,
        "all_mask_restored_exact": meta["all_mask_restored_exact"],
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.RandomState,
    resamples: int = EXPECTED_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    indices = rng.randint(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1, dtype=np.float64)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def merge_fullval_rows(
    output_dir: Path,
    baseline: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    num_shards: int,
) -> list[dict[str, Any]]:
    baseline_by_index = {
        int(row["dataset_index"]): row for row in baseline
    }
    unit_by_key = {candidate_key(row): row for row in units}
    merged = []
    seen = set()
    for shard_index in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_index}"
        path = shard_dir / "task041_fullval_masking_per_sample.csv"
        meta_path = shard_dir / "task041_fullval_mask_shard_meta.json"
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing mask shard {shard_index}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not bool(meta["all_mask_restored_exact"]):
            raise RuntimeError(f"mask restore gate failed for shard {shard_index}")
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                key = (
                    int(raw["candidate_task037_global_index"]),
                    int(raw["candidate_task040_global_index"]),
                )
                dataset_index = int(raw["dataset_index"])
                pair_key = (key, dataset_index)
                if pair_key in seen:
                    raise RuntimeError(f"duplicate candidate x dataset_index {pair_key}")
                if key not in unit_by_key:
                    raise RuntimeError(f"unknown frozen candidate in shard: {key}")
                if dataset_index not in baseline_by_index:
                    raise RuntimeError(f"unknown validation dataset index {dataset_index}")
                baseline_row = baseline_by_index[dataset_index]
                if int(raw["label"]) != int(baseline_row["label"]):
                    raise RuntimeError("masked label differs from baseline label")
                for field in (
                    "masked_true_class_logit",
                    "masked_cross_entropy",
                ):
                    finite_float(raw[field], field)
                merged.append(
                    {
                        **raw,
                        "candidate_task037_global_index": key[0],
                        "candidate_task040_global_index": key[1],
                        "dataset_index": dataset_index,
                        "label": int(raw["label"]),
                        "masked_true_class_logit": finite_float(
                            raw["masked_true_class_logit"], "masked_true_class_logit"
                        ),
                        "masked_cross_entropy": finite_float(
                            raw["masked_cross_entropy"], "masked_cross_entropy"
                        ),
                        "masked_top1_correct": int(raw["masked_top1_correct"]),
                        "masked_top5_correct": int(raw["masked_top5_correct"]),
                        "masked_predicted_class": int(raw["masked_predicted_class"]),
                    }
                )
                seen.add(pair_key)
    expected = EXPECTED_UNITS * len(baseline)
    if len(merged) != expected:
        raise RuntimeError(f"merged masking rows={len(merged)} != {expected}")
    return merged


def damage_rows(
    merged: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    rng: np.random.RandomState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_index = {int(row["dataset_index"]): row for row in baseline}
    unit_by_key = {candidate_key(row): row for row in units}
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in merged:
        grouped[candidate_key(row)].append(row)
    damage = []
    ci_rows = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["dataset_index"]))
        if len(rows) != len(baseline):
            raise RuntimeError(f"candidate {key} has wrong sample count")
        logit_drop = np.asarray(
            [
                float(baseline_by_index[int(row["dataset_index"])]["true_class_logit"])
                - float(row["masked_true_class_logit"])
                for row in rows
            ],
            dtype=np.float64,
        )
        ce_increase = np.asarray(
            [
                float(row["masked_cross_entropy"])
                - float(baseline_by_index[int(row["dataset_index"])]["cross_entropy"])
                for row in rows
            ],
            dtype=np.float64,
        )
        flip = np.asarray(
            [
                int(row["masked_predicted_class"])
                != int(baseline_by_index[int(row["dataset_index"])]["predicted_class"])
                for row in rows
            ],
            dtype=np.float64,
        )
        top1_change = np.asarray(
            [
                int(row["masked_top1_correct"])
                - int(baseline_by_index[int(row["dataset_index"])]["top1_correct"])
                for row in rows
            ],
            dtype=np.float64,
        )
        top5_change = np.asarray(
            [
                int(row["masked_top5_correct"])
                - int(baseline_by_index[int(row["dataset_index"])]["top5_correct"])
                for row in rows
            ],
            dtype=np.float64,
        )
        candidate = unit_by_key[key]
        logit_low, logit_high = bootstrap_mean_ci(logit_drop, rng)
        ce_low, ce_high = bootstrap_mean_ci(ce_increase, rng)
        base_top1 = float(
            np.mean(
                [int(baseline_by_index[int(row["dataset_index"])]["top1_correct"]) for row in rows],
                dtype=np.float64,
            )
        )
        base_top5 = float(
            np.mean(
                [int(baseline_by_index[int(row["dataset_index"])]["top5_correct"]) for row in rows],
                dtype=np.float64,
            )
        )
        damage.append(
            {
                **dict(candidate),
                "mean_true_class_logit_drop": float(np.mean(logit_drop, dtype=np.float64)),
                "median_true_class_logit_drop": float(np.median(logit_drop)),
                "mean_cross_entropy_increase": float(np.mean(ce_increase, dtype=np.float64)),
                "median_cross_entropy_increase": float(np.median(ce_increase)),
                "prediction_flip_rate": float(np.mean(flip, dtype=np.float64)),
                "top1_accuracy_change": float(np.mean(top1_change, dtype=np.float64)),
                "top5_accuracy_change": float(np.mean(top5_change, dtype=np.float64)),
                "baseline_top1_accuracy": base_top1,
                "baseline_top5_accuracy": base_top5,
                "masked_top1_accuracy": float(base_top1 + np.mean(top1_change)),
                "masked_top5_accuracy": float(base_top5 + np.mean(top5_change)),
                "n_samples": len(rows),
                "mask_restored_exact": True,
                "damage_uses_signed_differences": True,
            }
        )
        ci_rows.append(
            {
                "candidate_task037_global_index": key[0],
                "candidate_task040_global_index": key[1],
                "domain_id": candidate["domain_id"],
                "layer_name": candidate["candidate_layer_name"],
                "unit_type": candidate["candidate_unit_type"],
                "unit_index": candidate["candidate_unit_index"],
                "mean_true_class_logit_drop": float(np.mean(logit_drop, dtype=np.float64)),
                "logit_drop_ci95_low": logit_low,
                "logit_drop_ci95_high": logit_high,
                "mean_cross_entropy_increase": float(np.mean(ce_increase, dtype=np.float64)),
                "ce_increase_ci95_low": ce_low,
                "ce_increase_ci95_high": ce_high,
                "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
                "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
                "bootstrap_offline_after_inference": True,
                "n_samples": len(rows),
            }
        )
    if len(damage) != EXPECTED_UNITS:
        raise RuntimeError("unit damage count is not 29")
    return damage, ci_rows


def build_domain_ordering(
    damage: Sequence[Mapping[str, Any]],
    low_high: Mapping[str, Mapping[str, tuple[int, int]]],
) -> list[dict[str, Any]]:
    damage_by_key = {candidate_key(row): row for row in damage}
    output = []
    for domain in ALL_DOMAINS:
        low_key = low_high[domain]["low"]
        high_key = low_high[domain]["high"]
        low = damage_by_key[low_key]
        high = damage_by_key[high_key]
        row: dict[str, Any] = {
            "domain_id": domain,
            "domain_group": "mixed" if domain in MIXED_DOMAINS else "same_type",
            "low_task037_global_index": low_key[0],
            "high_task037_global_index": high_key[0],
            "low_R_MCTC": float(low["R_MCTC"]),
            "high_R_MCTC": float(high["R_MCTC"]),
            "delta_R_MCTC_high_minus_low": float(high["R_MCTC"]) - float(low["R_MCTC"]),
            "low_unit_type": low["candidate_unit_type"],
            "high_unit_type": high["candidate_unit_type"],
        }
        damage_field_by_metric = {
            "true_class_logit_drop": "mean_true_class_logit_drop",
            "cross_entropy_increase": "mean_cross_entropy_increase",
            "prediction_flip_rate": "prediction_flip_rate",
            "top1_accuracy_change": "top1_accuracy_change",
            "top5_accuracy_change": "top5_accuracy_change",
        }
        for metric in METRIC_NAMES:
            damage_field = damage_field_by_metric[metric]
            low_value = float(low[damage_field])
            high_value = float(high[damage_field])
            row[f"low_{metric}"] = low_value
            row[f"high_{metric}"] = high_value
            row[f"high_minus_low_{metric}"] = high_value - low_value
            row[f"high_greater_{metric}"] = high_value > low_value
            row[f"low_greater_{metric}"] = low_value > high_value
            row[f"tie_{metric}"] = low_value == high_value
        output.append(row)
    return output


def group_ordering_summary(
    ordering: Sequence[Mapping[str, Any]],
    group: str,
) -> dict[str, Any]:
    rows = [row for row in ordering if str(row["domain_group"]) == group]
    result: dict[str, Any] = {
        "group": group,
        "domains": [str(row["domain_id"]) for row in rows],
        "domain_count": len(rows),
    }
    for metric in METRIC_NAMES:
        differences = np.asarray(
            [float(row[f"high_minus_low_{metric}"]) for row in rows],
            dtype=np.float64,
        )
        result[f"{metric}_high_greater_count"] = int(np.sum(differences > 0))
        result[f"{metric}_low_greater_count"] = int(np.sum(differences < 0))
        result[f"{metric}_tie_count"] = int(np.sum(differences == 0))
        result[f"{metric}_mean_high_minus_low"] = float(
            np.mean(differences, dtype=np.float64)
        )
        result[f"{metric}_wilcoxon_signed_rank"] = wilcoxon_signed_rank(differences)
    return result


def within_domain_stats(
    damage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in damage:
        grouped[str(row["domain_id"])].append(row)
    output: dict[str, Any] = {}
    for domain in ALL_DOMAINS:
        rows = grouped[domain]
        if len(rows) < 3:
            continue
        score = [float(row["R_MCTC"]) for row in rows]
        output[domain] = {
            "n_units": len(rows),
            "spearman_R_MCTC_vs_mean_logit_drop": spearman(
                score, [float(row["mean_true_class_logit_drop"]) for row in rows]
            ),
            "kendall_R_MCTC_vs_mean_logit_drop": kendall_tau_b(
                score, [float(row["mean_true_class_logit_drop"]) for row in rows]
            ),
            "spearman_R_MCTC_vs_mean_CE_increase": spearman(
                score, [float(row["mean_cross_entropy_increase"]) for row in rows]
            ),
            "kendall_R_MCTC_vs_mean_CE_increase": kendall_tau_b(
                score, [float(row["mean_cross_entropy_increase"]) for row in rows]
            ),
            "spearman_R_MCTC_vs_flip_rate": spearman(
                score, [float(row["prediction_flip_rate"]) for row in rows]
            ),
            "kendall_R_MCTC_vs_flip_rate": kendall_tau_b(
                score, [float(row["prediction_flip_rate"]) for row in rows]
            ),
        }
    values = list(output.values())
    if values:
        for key in (
            "spearman_R_MCTC_vs_mean_logit_drop",
            "kendall_R_MCTC_vs_mean_logit_drop",
            "spearman_R_MCTC_vs_mean_CE_increase",
            "kendall_R_MCTC_vs_mean_CE_increase",
            "spearman_R_MCTC_vs_flip_rate",
            "kendall_R_MCTC_vs_flip_rate",
        ):
            output[f"domain_balanced_mean_{key}"] = float(
                np.mean([row[key] for row in values], dtype=np.float64)
            )
    return output


def load_existing_baseline_scores(task041_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(task041_dir / "task041_baseline_comparison.csv")
    required = {"domain_id", "candidate_task037_global_index", "candidate_task040_global_index", *SCORE_NAMES[:-1]}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Task041 baseline comparison lacks frozen score columns")
    return rows


def baseline_relationship(
    score_rows: Sequence[Mapping[str, Any]],
    damage: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    damage_by_key = {candidate_key(row): row for row in damage}
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    score_field = {
        "mean_abs_d_original": "mean_abs_d_original",
        "G_RMS": "G_RMS",
        "old_HTOR": "old_HTOR",
        "PTR": "PTR",
        "corrected_pairwise_best_E": "corrected_pairwise_best_E",
        "R_MCTC": "R_MCTC",
    }
    for raw in score_rows:
        key = candidate_key(raw)
        if key not in damage_by_key:
            raise ValueError(f"baseline comparison candidate not in damage: {key}")
        for criterion, field in score_field.items():
            for metric in DAMAGE_METRICS:
                damage_field = f"mean_{metric}"
                grouped[(criterion, metric)].append(
                    (
                        float(raw[field]),
                        float(damage_by_key[key][damage_field]),
                    )
                )
    # There are exactly two frozen low/high candidates per domain.  The
    # domain-balanced values below are computed per domain, then averaged, so
    # head/FFN units are never pooled as the primary result.
    by_criterion_metric: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for raw in score_rows:
        domain = str(raw["domain_id"])
        key = candidate_key(raw)
        damage_row = damage_by_key[key]
        for criterion, field in score_field.items():
            for metric in DAMAGE_METRICS:
                by_criterion_metric[(criterion, metric)].append(
                    (
                        domain,
                        float(raw[field]),
                        float(damage_row[f"mean_{metric}"]),
                    )
                )
    output = []
    for criterion in score_field:
        for metric in DAMAGE_METRICS:
            domain_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for domain, score, observed in by_criterion_metric[(criterion, metric)]:
                domain_values[domain].append((score, observed))
            correct = incorrect = ties = 0
            domain_spearman = []
            domain_kendall = []
            for domain in sorted(domain_values, key=int):
                values = domain_values[domain]
                if len(values) != 2:
                    raise ValueError(
                        f"baseline criterion {criterion} domain {domain} does not have two candidates"
                    )
                scores = [values[0][0], values[1][0]]
                observed = [values[0][1], values[1][1]]
                score_sign = sign(scores[0] - scores[1])
                damage_sign = sign(observed[0] - observed[1])
                if score_sign == 0 or damage_sign == 0:
                    ties += 1
                elif score_sign == damage_sign:
                    correct += 1
                else:
                    incorrect += 1
                domain_spearman.append(spearman(scores, observed))
                domain_kendall.append(kendall_tau_b(scores, observed))
            output.append(
                {
                    "criterion": criterion,
                    "damage_metric": metric,
                    "scope": "existing_task041_low_high_candidates_n18",
                    "domain_count": len(domain_values),
                    "ordering_correct_count": correct,
                    "ordering_incorrect_count": incorrect,
                    "ordering_tie_count": ties,
                    "within_domain_ordering_correctness": (
                        correct / len(domain_values) if domain_values else 0.0
                    ),
                    "domain_balanced_spearman": float(
                        np.mean(domain_spearman, dtype=np.float64)
                    ),
                    "domain_balanced_kendall": float(
                        np.mean(domain_kendall, dtype=np.float64)
                    ),
                }
            )
    return output


def compare_n30_ordering(task041_dir: Path) -> dict[str, Any]:
    path = task041_dir / "task041_domain_ordering.csv"
    if not path.is_file():
        return {}
    rows = read_csv(path)
    result = {}
    for metric in ("true_class_logit_drop", "cross_entropy_increase"):
        differences = np.asarray(
            [float(row[f"high_minus_low_{metric}"]) for row in rows],
            dtype=np.float64,
        )
        result[metric] = {
            "high_greater_count": int(np.sum(differences > 0)),
            "low_greater_count": int(np.sum(differences < 0)),
            "tie_count": int(np.sum(differences == 0)),
        }
    return result


def resolution_evidence(
    baseline: Sequence[Mapping[str, Any]],
    damage: Sequence[Mapping[str, Any]],
    n30: Mapping[str, Any],
) -> dict[str, Any]:
    full_flip_nonzero = int(
        sum(float(row["prediction_flip_rate"]) > 0 for row in damage)
    )
    full_top1_nonzero = int(
        sum(float(row["top1_accuracy_change"]) != 0 for row in damage)
    )
    full_top5_nonzero = int(
        sum(float(row["top5_accuracy_change"]) != 0 for row in damage)
    )
    return {
        "full_validation_sample_count": len(baseline),
        "full_validation_candidate_count": len(damage),
        "full_validation_units_with_nonzero_flip_rate": full_flip_nonzero,
        "full_validation_units_with_nonzero_top1_change": full_top1_nonzero,
        "full_validation_units_with_nonzero_top5_change": full_top5_nonzero,
        "n30_reference_ordering": dict(n30),
        "full_validation_signed_logit_drop_std_across_units": float(
            np.std(
                [float(row["mean_true_class_logit_drop"]) for row in damage],
                dtype=np.float64,
            )
        ),
        "full_validation_signed_CE_increase_std_across_units": float(
            np.std(
                [float(row["mean_cross_entropy_increase"]) for row in damage],
                dtype=np.float64,
            )
        ),
    }


def decision_labels(
    ordering: Sequence[Mapping[str, Any]],
    baseline_relationship_rows: Sequence[Mapping[str, Any]],
    n30: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    def count(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[int, int, int]:
        differences = [
            float(row[f"high_minus_low_{metric}"]) for row in rows
        ]
        return (
            sum(value > 0 for value in differences),
            sum(value < 0 for value in differences),
            sum(value == 0 for value in differences),
        )

    all_logit = count(ordering, "true_class_logit_drop")
    all_ce = count(ordering, "cross_entropy_increase")
    same_rows = [row for row in ordering if row["domain_group"] == "same_type"]
    same_logit = count(same_rows, "true_class_logit_drop")
    same_ce = count(same_rows, "cross_entropy_increase")
    rel = {
        (row["criterion"], row["damage_metric"]): row
        for row in baseline_relationship_rows
    }
    mctc_logit = rel[("R_MCTC", "true_class_logit_drop")]
    mctc_ce = rel[("R_MCTC", "cross_entropy_increase")]
    best_logit = max(
        float(row["domain_balanced_spearman"])
        for row in baseline_relationship_rows
        if row["damage_metric"] == "true_class_logit_drop"
    )
    best_ce = max(
        float(row["domain_balanced_spearman"])
        for row in baseline_relationship_rows
        if row["damage_metric"] == "cross_entropy_increase"
    )
    full_resolution_improved = bool(
        resolution["full_validation_units_with_nonzero_flip_rate"] > 0
        or resolution["full_validation_units_with_nonzero_top1_change"] > 0
        or resolution["full_validation_units_with_nonzero_top5_change"] > 0
    )
    n30_logit = n30.get("true_class_logit_drop", {})
    n30_ce = n30.get("cross_entropy_increase", {})
    n30_both_weak = (
        int(n30_logit.get("high_greater_count", 0)) <= 4
        and int(n30_ce.get("high_greater_count", 0)) <= 4
    )
    full_both_majority = all_logit[0] >= 5 and all_ce[0] >= 5
    same_type_support = same_logit[0] >= 4 and same_ce[0] >= 4
    mctc_better_than_all_frozen = (
        float(mctc_logit["domain_balanced_spearman"]) >= best_logit
        and float(mctc_ce["domain_balanced_spearman"]) >= best_ce
        and float(mctc_logit["within_domain_ordering_correctness"]) >= max(
            float(row["within_domain_ordering_correctness"])
            for row in baseline_relationship_rows
            if row["damage_metric"] == "true_class_logit_drop"
        )
        and float(mctc_ce["within_domain_ordering_correctness"]) >= max(
            float(row["within_domain_ordering_correctness"])
            for row in baseline_relationship_rows
            if row["damage_metric"] == "cross_entropy_increase"
        )
    )
    if full_resolution_improved and full_both_majority and n30_both_weak:
        answer_a = "YES"
    elif full_resolution_improved:
        answer_a = "NO"
    else:
        answer_a = "UNRESOLVED"
    answer_b = "YES" if full_both_majority else "NO"
    answer_c = "YES" if same_type_support else "NO"
    answer_d = "YES" if mctc_better_than_all_frozen else "NO"
    if answer_c == "NO":
        answer_e = "REJECTED AS A STANDALONE SELECTOR"
    elif answer_b == "YES" and answer_d == "YES":
        answer_e = "RETAINED FOR FURTHER VALIDATION"
    else:
        answer_e = "UNRESOLVED"
    return {
        "A_full_validation_removes_N30_noise_explanation": answer_a,
        "B_frozen_N3_R_MCTC_predicts_full_validation_damage": answer_b,
        "C_relation_holds_in_same_type_domains": answer_c,
        "D_MCTC_better_than_frozen_baselines": answer_d,
        "E_selector_decision": answer_e,
        "decision_rule": {
            "majority": "high-risk > low-risk in at least 5 of 9 domains",
            "same_type_majority": "high-risk > low-risk in at least 4 of 7 same-type domains",
            "A": "requires increased resolution and both logit/CE majority improvement over weak N=30 ordering",
            "D": "R_MCTC must tie or exceed every frozen baseline on both scope-limited domain-balanced Spearman and ordering correctness",
        },
        "evidence": {
            "full_logit_ordering": {
                "high_greater": all_logit[0],
                "low_greater": all_logit[1],
                "ties": all_logit[2],
            },
            "full_CE_ordering": {
                "high_greater": all_ce[0],
                "low_greater": all_ce[1],
                "ties": all_ce[2],
            },
            "same_type_logit_ordering": {
                "high_greater": same_logit[0],
                "low_greater": same_logit[1],
                "ties": same_logit[2],
            },
            "same_type_CE_ordering": {
                "high_greater": same_ce[0],
                "low_greater": same_ce[1],
                "ties": same_ce[2],
            },
            "R_MCTC_domain_balanced_spearman_logit": float(
                mctc_logit["domain_balanced_spearman"]
            ),
            "R_MCTC_domain_balanced_spearman_CE": float(
                mctc_ce["domain_balanced_spearman"]
            ),
        },
    }


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    ordering: Sequence[Mapping[str, Any]],
    relationship: Sequence[Mapping[str, Any]],
    mixed: Mapping[str, Any],
    same: Mapping[str, Any],
) -> None:
    decisions = summary["decisions"]
    lines = [
        "# Task041 Phase D — Full-Validation Masking Oracle Resolution Audit",
        "",
        "This audit freezes the existing Task041 N=3 temporal-coverage scores and",
        "the exact 29 tested units. It evaluates only temporary whole-unit masks on",
        "the complete UCF101 validation split. No temporal intervention/profile was",
        "recomputed, and no pruning or fine-tuning was performed.",
        "",
        "## Gates",
        "",
        f"- Valid clips: {summary['valid_clips']} / {EXPECTED_VALID_CLIPS}",
        f"- Checkpoint SHA256: {summary['checkpoint_sha256']}",
        f"- Exact frozen units: {summary['frozen_unit_count']} / {EXPECTED_UNITS}",
        f"- Full-validation masking rows: {summary['masking_per_sample_rows']}",
        f"- Mask restoration exact after every candidate: {summary['all_mask_restored_exact']}",
        f"- All numeric candidate outputs finite: {summary['all_candidate_outputs_finite']}",
        f"- Duplicate candidate x dataset_index: {summary['duplicate_candidate_dataset_index']}",
        "",
        "## Same-type versus mixed-type",
        "",
        f"- Mixed domains 271/297: logit high>low {mixed['true_class_logit_drop_high_greater_count']}/{mixed['domain_count']}; CE high>low {mixed['cross_entropy_increase_high_greater_count']}/{mixed['domain_count']}.",
        f"- Same-type domains 269/400/415/102/103/113/76: logit high>low {same['true_class_logit_drop_high_greater_count']}/{same['domain_count']}; CE high>low {same['cross_entropy_increase_high_greater_count']}/{same['domain_count']}.",
        "",
        "## Frozen-score monotonicity",
        "",
        "The full unit table reports per-domain Spearman and Kendall correlations",
        "between frozen N=3 R_MCTC and signed full-validation damage. No score sign",
        "was flipped after observing the result.",
        "",
        "## Baseline scope",
        "",
        "Baseline comparison uses the same existing Task041 low/high candidate table",
        "(18 candidates, two per domain); it does not invent baseline scores for the",
        "11 non-low/high units. Domain-balanced statistics are not global head/FFN pools.",
        "",
        "## Decision",
        "",
        f"- A. Does full validation remove the N=30 oracle noise explanation? **{decisions['A_full_validation_removes_N30_noise_explanation']}**",
        f"- B. Does frozen N=3 R_MCTC predict full-validation masking damage? **{decisions['B_frozen_N3_R_MCTC_predicts_full_validation_damage']}**",
        f"- C. Does the relation hold in same-type domains? **{decisions['C_relation_holds_in_same_type_domains']}**",
        f"- D. Is MCTC better than frozen baselines? **{decisions['D_MCTC_better_than_frozen_baselines']}**",
        f"- E. Selector decision: **{decisions['E_selector_decision']}**",
        "",
        "No production pruning is approved by this audit. Task041 stops here; no",
        "N=9 temporal calibration, 50% pruning, fine-tuning, or Task042 is launched.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_finalize(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    val_list = Path(args.val_list).resolve()
    task041_dir = Path(args.task041_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    required_outputs = (
        "task041_fullval_unit_damage.csv",
        "task041_fullval_domain_ordering.csv",
        "task041_fullval_baseline_comparison.csv",
        "task041_fullval_same_vs_mixed.csv",
        "task041_fullval_bootstrap_ci.csv",
        "task041_phase_d_summary.json",
        "task041_phase_d_report.md",
    )
    if any((output_dir / name).exists() for name in required_outputs):
        raise FileExistsError("Phase D finalized outputs exist; refusing overwrite")
    ensure_importable(project_root)
    units = load_frozen_units(task041_dir)
    low_high = load_low_high(task041_dir, units)
    baseline = load_baseline(output_dir / "task041_fullval_baseline_per_sample.csv")
    identity = json.loads(
        (output_dir / "task041_phase_d_checkpoint_identity.json").read_text(
            encoding="utf-8"
        )
    )
    if identity.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("finalize checkpoint SHA gate failed")
    merged = merge_fullval_rows(output_dir, baseline, units, args.num_shards)
    rng = np.random.RandomState(EXPECTED_BOOTSTRAP_SEED)
    damage, ci_rows = damage_rows(merged, baseline, units, rng)
    ordering = build_domain_ordering(damage, low_high)
    relationship = baseline_relationship(
        load_existing_baseline_scores(task041_dir), damage
    )
    rank_stats = within_domain_stats(damage)
    mixed = group_ordering_summary(ordering, "mixed")
    same = group_ordering_summary(ordering, "same_type")
    n30 = compare_n30_ordering(task041_dir)
    resolution = resolution_evidence(baseline, damage, n30)
    decisions = decision_labels(ordering, relationship, n30, resolution)
    write_csv(output_dir / "task041_fullval_masking_per_sample.csv", merged)
    write_csv(output_dir / "task041_fullval_unit_damage.csv", damage)
    write_csv(output_dir / "task041_fullval_bootstrap_ci.csv", ci_rows)
    write_csv(output_dir / "task041_fullval_domain_ordering.csv", ordering)
    write_csv(output_dir / "task041_fullval_baseline_comparison.csv", relationship)
    write_csv(
        output_dir / "task041_fullval_same_vs_mixed.csv",
        [mixed, same],
    )
    all_finite = True
    for row in damage:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                all_finite = False
    summary = {
        "task": "task041_phase_d_full_validation_masking_oracle_resolution_audit",
        "phase": "D",
        "checkpoint_absolute_path": str(checkpoint),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "val_list_absolute_path": str(val_list),
        "val_list_sha256": sha256_file(val_list),
        "frame_root": str(Path(args.frame_root).resolve()),
        "valid_clips": len(baseline),
        "frozen_unit_count": len(units),
        "frozen_domains": list(ALL_DOMAINS),
        "mixed_domains": list(MIXED_DOMAINS),
        "same_type_domains": list(SAME_TYPE_DOMAINS),
        "masking_per_sample_rows": len(merged),
        "expected_masking_per_sample_rows": EXPECTED_UNITS * EXPECTED_VALID_CLIPS,
        "duplicate_candidate_dataset_index": False,
        "all_candidate_outputs_finite": all_finite,
        "all_mask_restored_exact": all(
            bool(json.loads(
                (output_dir / f"shard_{index}" / "task041_fullval_mask_shard_meta.json"
                ).read_text(encoding="utf-8")
            )["all_mask_restored_exact"])
            for index in range(args.num_shards)
        ),
        "same_preprocessing": True,
        "same_checkpoint_loading_semantics": True,
        "same_classification_head_semantics": True,
        "no_temporal_intervention_rerun": True,
        "no_profile_recomputation": True,
        "no_bms_change": True,
        "no_physical_pruning": True,
        "no_finetuning": True,
        "bootstrap": {
            "seed": EXPECTED_BOOTSTRAP_SEED,
            "resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
            "offline_after_inference": True,
            "metrics": [
                "mean_true_class_logit_drop",
                "mean_cross_entropy_increase",
            ],
        },
        "full_validation_ordering": {
            "all_domains": group_ordering_summary(ordering, "mixed"),
            "same_type_and_mixed_rows": [mixed, same],
            "all_domain_rows": len(ordering),
        },
        "within_domain_monotonicity": rank_stats,
        "baseline_relationship_scope": "existing_task041_low_high_candidates_n18",
        "decisions": decisions,
        "input_artifacts": {
            "task041_output_dir": str(task041_dir),
            "frozen_masking_damage": str(task041_dir / "task041_masking_damage.csv"),
            "frozen_oracle_candidates": str(task041_dir / "task041_oracle_candidates.csv"),
            "frozen_baseline_comparison": str(
                task041_dir / "task041_baseline_comparison.csv"
            ),
        },
        "git_branch": identity.get("git_branch", "unknown"),
        "git_commit": identity.get("git_commit", "unknown"),
    }
    write_json(output_dir / "task041_phase_d_summary.json", summary)
    write_report(
        output_dir / "task041_phase_d_report.md",
        summary,
        ordering,
        relationship,
        mixed,
        same,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader configuration")
    if args.phase == "baseline":
        result = phase_baseline(args)
    elif args.phase == "mask":
        result = phase_mask(args)
    else:
        result = phase_finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

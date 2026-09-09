#!/usr/bin/env python3
"""Evidence-only Task011 configuration, sample, checkpoint, and git audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


UNKNOWN = "UNKNOWN"
CONFIG_FIELDS = (
    "checkpoint",
    "dataset_root",
    "split",
    "number_of_classes",
    "classifier_head",
    "input_size",
    "clip_length",
    "frame_sampling",
    "validation_preprocessing",
    "calibration_preprocessing",
    "calib_batches",
    "calib_batch_size",
    "sigma",
    "bms_tol",
    "bms_max_iters",
    "sink_merge_tol",
    "min_keep_ratio",
    "alpha",
    "gamma_decay",
    "target_sparsity",
    "pruning_budget_definition",
    "group_scoring",
    "dependency_handling",
    "protect_first_layer_logic",
    "minimum_heads",
    "minimum_neurons",
    "validation_batch_size",
    "data_parallel_setup",
    "random_seed",
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cli_arguments(command: object) -> dict[str, object]:
    if isinstance(command, list):
        tokens = [str(value) for value in command]
    elif isinstance(command, str):
        tokens = shlex.split(command)
    else:
        return {}
    result: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if "=" in name:
            name, value = name.split("=", 1)
            result[name] = value
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[name] = tokens[index + 1]
            index += 2
        else:
            result[name] = True
            index += 1
    return result


def _value(*candidates: object) -> object:
    for candidate in candidates:
        if candidate is not None and candidate != "":
            return candidate
    return UNKNOWN


def _nested(source: dict, *path: str) -> object:
    current: object = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _dataset_source_facts(path: Path) -> dict:
    if not path.is_file():
        return {}
    source = path.read_text(encoding="utf-8")
    input_match = re.search(r"input_size\s*=\s*(\d+)", source)
    clip_match = re.search(r"LoopPadding\((\d+)\)", source)
    normalize_match = re.search(r"default_mean\s*=\s*(\[[^\n]+\]).*?default_std\s*=\s*(\[[^\n]+\])", source, re.S)
    return {
        "input_size": int(input_match.group(1)) if input_match else None,
        "clip_length": int(clip_match.group(1)) if clip_match else None,
        "frame_sampling": "LoopPadding" if "LoopPadding" in source else None,
        "crop": "CornerCrop(center)" if "CornerCrop(input_size, 'c')" in source else None,
        "normalization": (
            f"mean={normalize_match.group(1)}, std={normalize_match.group(2)}"
            if normalize_match else None
        ),
        "shuffle": "shuffle=True" if "shuffle=True" in source else None,
        "drop_last": "drop_last=True" if "drop_last=True" in source else None,
    }


def task010_configuration(metadata: dict, controlled: dict, dataset_facts: dict) -> dict:
    cli = _cli_arguments(metadata.get("command_line"))
    control = controlled.get("configuration", {}) if controlled else {}
    checkpoint = controlled.get("checkpoint", {}) if controlled else {}
    validation = controlled.get("validation_dataset", {}) if controlled else {}
    validation_split = _value(metadata.get("validation_split"))
    dataset_root = (
        str(Path(str(validation_split)).parent)
        if validation_split != UNKNOWN else UNKNOWN
    )
    preprocessing = ", ".join(
        str(value)
        for value in (
            dataset_facts.get("crop"),
            dataset_facts.get("normalization"),
            dataset_facts.get("frame_sampling"),
            dataset_facts.get("shuffle"),
            dataset_facts.get("drop_last"),
        )
        if value
    ) or UNKNOWN
    return {
        "checkpoint": _value(
            metadata.get("checkpoint_path"), checkpoint.get("path")
        ),
        "dataset_root": dataset_root,
        "split": validation_split,
        "number_of_classes": _value(validation.get("num_unique_labels")),
        "classifier_head": _value(
            checkpoint.get("model_classifier_output_dimension")
        ),
        "input_size": _value(dataset_facts.get("input_size")),
        "clip_length": _value(dataset_facts.get("clip_length")),
        "frame_sampling": _value(dataset_facts.get("frame_sampling")),
        "validation_preprocessing": preprocessing,
        "calibration_preprocessing": preprocessing,
        "calib_batches": _value(
            metadata.get("calibration_batches"), cli.get("calib_batches"), control.get("calib_batches")
        ),
        "calib_batch_size": _value(
            metadata.get("calibration_batch_size"), cli.get("calib_batch_size"), control.get("calib_batch_size")
        ),
        "sigma": _value(metadata.get("sigma"), cli.get("sigma"), control.get("sigma")),
        "bms_tol": "1e-4 (MC.py mean_shift_clustering literal)",
        "bms_max_iters": "60 (MC.py mean_shift_clustering literal)",
        "sink_merge_tol": "0.01 (P rounded to two decimals in MC.py)",
        "min_keep_ratio": _value(
            metadata.get("min_keep_ratio"), cli.get("min_keep_ratio"), control.get("min_keep_ratio")
        ),
        "alpha": _value(
            metadata.get("importance_alpha"), cli.get("importance_alpha"), control.get("importance_alpha")
        ),
        "gamma_decay": _value(
            metadata.get("gamma_decay"), cli.get("gamma_decay"), control.get("gamma_decay")
        ),
        "target_sparsity": _value(
            metadata.get("target_sparsity"), cli.get("sparsity"), control.get("target_sparsity")
        ),
        "pruning_budget_definition": "target = model parameter numel x target sparsity; estimated unit costs",
        "group_scoring": "existing manifold consistency x static strength",
        "dependency_handling": "logical keep_heads/keep_neurons registry; no physical compaction",
        "protect_first_layer_logic": "none identified in current BMS path",
        "minimum_heads": "max(1, int(num_heads x min_keep_ratio))",
        "minimum_neurons": "max(1, int(num_neurons x min_keep_ratio))",
        "validation_batch_size": _value(
            metadata.get("training_batch_size"), cli.get("batch_size"), control.get("validation_batch_size")
        ),
        "data_parallel_setup": _value(
            metadata.get("training_devices"), control.get("gpu_names")
        ),
        "random_seed": _value(metadata.get("seed"), cli.get("seed"), control.get("seed")),
    }


def historical_configuration(metadata: dict, log_path: Path | None) -> dict:
    command = metadata.get("command_line", metadata.get("command", []))
    cli = _cli_arguments(command)
    log_text = ""
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

    def regex(pattern: str) -> object:
        match = re.search(pattern, log_text, re.I)
        return match.group(1) if match else None

    result = {field: UNKNOWN for field in CONFIG_FIELDS}
    mappings = {
        "checkpoint": (metadata.get("checkpoint_path"), metadata.get("checkpoint")),
        "split": (metadata.get("validation_split"), metadata.get("val_split")),
        "number_of_classes": (metadata.get("number_of_classes"), metadata.get("num_classes")),
        "classifier_head": (metadata.get("classifier_output_dimension"),),
        "input_size": (metadata.get("input_size"), cli.get("input_size")),
        "clip_length": (metadata.get("clip_length"), cli.get("clip_length")),
        "calib_batches": (metadata.get("calibration_batches"), cli.get("calib_batches")),
        "calib_batch_size": (metadata.get("calibration_batch_size"), cli.get("calib_batch_size")),
        "sigma": (metadata.get("sigma"), cli.get("sigma"), regex(r"sigma\s*[=:]\s*([0-9.eE+-]+)")),
        "min_keep_ratio": (metadata.get("min_keep_ratio"), cli.get("min_keep_ratio")),
        "alpha": (metadata.get("importance_alpha"), cli.get("importance_alpha")),
        "gamma_decay": (metadata.get("gamma_decay"), cli.get("gamma_decay")),
        "target_sparsity": (metadata.get("target_sparsity"), cli.get("sparsity")),
        "validation_batch_size": (metadata.get("validation_batch_size"), cli.get("batch_size")),
        "random_seed": (metadata.get("seed"), cli.get("seed")),
    }
    for field, candidates in mappings.items():
        result[field] = _value(*candidates)
    if result["split"] != UNKNOWN:
        result["dataset_root"] = str(Path(str(result["split"])).parent)
    for field in CONFIG_FIELDS:
        if metadata.get(field) not in (None, ""):
            result[field] = metadata[field]
    return result


def _comparison_status(historical: object, current: object) -> str:
    if historical == UNKNOWN or current == UNKNOWN:
        return "UNKNOWN"
    return "IDENTICAL" if str(historical) == str(current) else "DIFFERENT"


def write_config_diff(
    path: Path, historical: dict, current: dict, historical_sources: list[str]
) -> None:
    unknown = [field for field in CONFIG_FIELDS if historical.get(field, UNKNOWN) == UNKNOWN]
    lines = ["# Historical main experiment vs Task010 Old3D configuration", ""]
    if unknown:
        lines.extend(
            [
                "**HISTORICAL CONFIGURATION NOT FULLY RECOVERED**",
                "",
                "Unknown historical fields: " + ", ".join(unknown) + ".",
                "",
            ]
        )
    lines.extend(
        [
            "Historical evidence sources: " + (", ".join(historical_sources) or "none found"),
            "",
            "| Field | Historical | Task010 Old3D | Classification |",
            "|---|---|---|---|",
        ]
    )
    for field in CONFIG_FIELDS:
        old = historical.get(field, UNKNOWN)
        new = current.get(field, UNKNOWN)
        lines.append(
            f"| {field} | {str(old).replace('|', '/')} | "
            f"{str(new).replace('|', '/')} | {_comparison_status(old, new)} |"
        )
    lines.extend(
        [
            "",
            "Unknown fields are not filled from remembered experiment values. "
            "A DIFFERENT label records evidence; it does not identify causality by itself.",
            "",
        ]
    )
    _atomic_text(path, "\n".join(lines))


def _read_probe_samples(path: Path) -> dict:
    if not path.is_file():
        return {"status": "UNAVAILABLE", "path": str(path)}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    canonical = [
        {
            "order": index,
            "sample_index": int(row["sample_index"]),
            "class_id": int(row["class_id"]),
            "video_identifier": str(row.get("video_identifier", "")),
        }
        for index, row in enumerate(rows)
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "available",
        "path": str(path.resolve()),
        "num_videos": len(canonical),
        "sample_order": canonical,
        "sample_list_sha256_canonical": digest,
    }


def sample_comparison(
    task009_samples: Path, controlled: dict, task009_metadata: dict
) -> dict:
    task009 = _read_probe_samples(task009_samples)
    task009["experiment_context"] = {
        "random_seed": _value(task009_metadata.get("seed")),
        "split": _value(
            task009_metadata.get("validation_split"),
            task009_metadata.get("split"),
        ),
        "clip_sampling_policy": _value(
            task009_metadata.get("clip_sampling_policy"),
            task009_metadata.get("temporal_sampling"),
        ),
        "spatial_preprocessing": _value(
            task009_metadata.get("spatial_preprocessing"),
            task009_metadata.get("preprocessing"),
        ),
    }
    task010 = controlled.get("calibration_samples", {}) if controlled else {}
    if task009.get("status") != "available" or not task010:
        same = "UNKNOWN"
    else:
        left = [
            (row["sample_index"], row["class_id"], row.get("video_identifier", ""))
            for row in task009["sample_order"]
        ]
        right = [
            (row["sample_index"], row["class_id"], row.get("video_identifier", ""))
            for row in task010.get("sample_order", [])
        ]
        same = "YES" if left == right else "NO"
    return {
        "same_ordered_calibration_samples": same,
        "task009": task009,
        "task010": task010 or {"status": "UNAVAILABLE until controlled run completes"},
        "interpretation": (
            "Different sample sets invalidate an exact saved-value equality conclusion, "
            "but do not imply a formula implementation bug."
        ),
    }


def _checkpoint_entry(metadata: dict, fallback_path: object = None) -> dict:
    checkpoint = metadata.get("checkpoint", metadata) if metadata else {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    path_value = _value(checkpoint.get("path"), checkpoint.get("checkpoint"), fallback_path)
    result = dict(checkpoint)
    result.setdefault("path", path_value)
    path = Path(str(path_value)).expanduser() if path_value != UNKNOWN else None
    if path is not None and path.is_file():
        result.setdefault("file_size_bytes", path.stat().st_size)
        result.setdefault("sha256", _sha256_file(path))
    else:
        result.setdefault("file_size_bytes", UNKNOWN)
        result.setdefault("sha256", UNKNOWN)
    return result


def checkpoint_comparison(
    task009_metadata: dict,
    task010_metadata: dict,
    controlled: dict,
    historical_metadata: dict,
) -> dict:
    task009 = _checkpoint_entry(
        task009_metadata, task009_metadata.get("checkpoint")
    )
    task010_source = controlled if controlled else {}
    task010 = _checkpoint_entry(
        task010_source,
        task010_metadata.get("checkpoint_path"),
    )
    historical = _checkpoint_entry(
        historical_metadata, historical_metadata.get("checkpoint_path")
    )
    sha009 = task009.get("sha256", UNKNOWN)
    sha010 = task010.get("sha256", UNKNOWN)
    same = "UNKNOWN" if UNKNOWN in (sha009, sha010) else ("YES" if sha009 == sha010 else "NO")
    return {
        "task009_vs_task010_same_checkpoint": same,
        "task009": task009,
        "task010": task010,
        "historical": historical,
    }


def git_diff_summary(path: Path, historical_ref: str, current_ref: str) -> None:
    lines = ["# Pruning-pipeline git diff summary", ""]
    if not historical_ref:
        lines.extend(
            [
                "**HISTORICAL CONFIGURATION NOT FULLY RECOVERED**",
                "",
                "No historical commit/branch was supplied. No revision is silently "
                "treated as the source of the approximately 26% result.",
                "",
            ]
        )
        _atomic_text(path, "\n".join(lines))
        return
    verify = subprocess.run(
        ["git", "rev-parse", "--verify", historical_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        lines.append(f"Historical ref `{historical_ref}` is unavailable in this clone.")
        _atomic_text(path, "\n".join(lines) + "\n")
        return
    result = subprocess.run(
        ["git", "diff", "--name-status", historical_ref, current_ref],
        check=True,
        capture_output=True,
        text=True,
    )
    categories = {
        "DESCRIPTOR ONLY": [],
        "BMS": [],
        "BUDGET": [],
        "REGISTRY": [],
        "MASK APPLICATION": [],
        "MODEL FORWARD": [],
        "DATASET": [],
        "VALIDATION": [],
        "TRAINING": [],
        "OTHER": [],
    }

    def content_categories(filename: str) -> set[str]:
        """Locate cross-cutting changes inside the two main pipeline files."""
        if filename not in {"MC.py", "ucf101_videoswin_my.py"}:
            return set()
        detail = subprocess.run(
            ["git", "diff", "--unified=0", historical_ref, current_ref, "--", filename],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.lower()
        patterns = {
            "DESCRIPTOR ONLY": (
                "descriptor", "temporal_dynamicity", "spatiotemporal", "d_abs", "d_rel"
            ),
            "BMS": ("mean_shift", "manifold", "group_score", "sigma"),
            "BUDGET": (
                "estimate_unit_cost", "target_sparsity", "target_red", "total_reduction"
            ),
            "REGISTRY": ("registry", "group_pruning", "coverage_pruning"),
            "MASK APPLICATION": ("apply_pruning_masks", "keep_heads", "keep_neurons"),
            "MODEL FORWARD": (
                "def forward", "windowattention3d", "class mlp", "swintransformer3d"
            ),
            "DATASET": ("get_dataset", "train_rgb_split", "val_rgb_split"),
            "VALIDATION": ("validate_rgb", "pre_ft_top", "top1", "top5"),
            "TRAINING": ("optimizer", "scheduler", "run_one_epoch", "gradscaler"),
        }
        located = {
            category
            for category, keywords in patterns.items()
            if any(keyword in detail for keyword in keywords)
        }
        if len(located) > 1:
            located.discard("DESCRIPTOR ONLY")
        return located

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        filename = parts[-1]
        located = content_categories(filename)
        if located:
            for category in sorted(located):
                categories[category].append(line + " (content-keyword locator)")
            continue
        lower = filename.lower()
        if "temporal" in lower or "descriptor" in lower or "tdd" in lower:
            category = "DESCRIPTOR ONLY"
        elif filename == "MC.py":
            category = "OTHER"
        elif "dataset" in lower:
            category = "DATASET"
        elif "ucf101_videoswin" in lower:
            category = "VALIDATION"
        elif "train" in lower:
            category = "TRAINING"
        else:
            category = "OTHER"
        categories[category].append(line)
    lines.append(f"Compared `{historical_ref}` against `{current_ref}`.")
    lines.append("")
    for category, entries in categories.items():
        lines.extend([f"## {category}", ""])
        lines.extend([f"- `{entry}`" for entry in entries] or ["- No file-level change classified."])
        lines.append("")
    lines.append(
        "File-level classification is a locator, not proof of causal behavior. "
        "Review the actual diff before any revert or fix."
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    task009_metadata = _read_json(Path(args.task009_metadata))
    task010_metadata = _read_json(Path(args.task010_metadata))
    controlled = _read_json(Path(args.task010_controlled))
    historical_metadata_path = Path(args.historical_metadata) if args.historical_metadata else None
    historical_log_path = Path(args.historical_log) if args.historical_log else None
    historical_metadata = (
        _read_json(historical_metadata_path) if historical_metadata_path else {}
    )
    dataset_facts = _dataset_source_facts(Path(args.dataset_source))
    current_config = task010_configuration(task010_metadata, controlled, dataset_facts)
    historical_config = historical_configuration(historical_metadata, historical_log_path)
    sources = []
    if historical_metadata_path and historical_metadata_path.is_file():
        sources.append(str(historical_metadata_path.resolve()))
    if historical_log_path and historical_log_path.is_file():
        sources.append(str(historical_log_path.resolve()))
    write_config_diff(
        output_dir / "historical_vs_task010_config_diff.md",
        historical_config,
        current_config,
        sources,
    )
    _atomic_json(
        output_dir / "sample_set_comparison.json",
        sample_comparison(
            Path(args.task009_samples), controlled, task009_metadata
        ),
    )
    _atomic_json(
        output_dir / "checkpoint_comparison.json",
        checkpoint_comparison(
            task009_metadata, task010_metadata, controlled, historical_metadata
        ),
    )
    git_diff_summary(
        output_dir / "pruning_pipeline_git_diff_summary.md",
        args.historical_ref,
        args.current_ref,
    )
    print(f"Task011 configuration audit written to {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task011 pruning-run configuration audit")
    parser.add_argument(
        "--task009_metadata", default="video_descriptor_validation/run_metadata.json"
    )
    parser.add_argument(
        "--task009_samples",
        default="/home/jixinye25/jxy_work1/work1-pruning/descriptor_ablation_validation/probe_samples.csv",
    )
    parser.add_argument(
        "--task010_metadata",
        default="tdd_pruning_validation/old3d/seed3407/run_metadata.json",
    )
    parser.add_argument(
        "--task010_controlled",
        default="task011_diagnosis/controlled_runs/old3d.json",
    )
    parser.add_argument("--historical_metadata", default="")
    parser.add_argument("--historical_log", default="")
    parser.add_argument("--historical_ref", default="")
    parser.add_argument("--current_ref", default="HEAD")
    parser.add_argument("--dataset_source", default="dataset/ucf101.py")
    parser.add_argument("--output_dir", default="task011_diagnosis")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

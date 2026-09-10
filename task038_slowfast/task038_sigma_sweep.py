"""Controlled Task038 BMS sigma sweep diagnostics.

This wrapper orchestrates the existing authoritative BMS, F3 selector,
logical-pruning, and pre-finetune validation implementations. It never
launches fine-tuning and never writes into the official result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

from .slowfast_model_task038 import slowfast_16x8_resnet101_kinetics400
from .slowfast_unit_adapter import Unit, UnitInventory, build_inventory


SIGMAS: tuple[float, ...] = (
    0.100, 0.095, 0.090, 0.085, 0.080, 0.075,
    0.070, 0.065, 0.060, 0.055, 0.050,
)
TARGET_REMAINING_RATIO = 0.50
MIN_KEEP_RATIO = 0.10
SWEEP_ROOT = Path(
    "/data/jixinye25/work1/output/"
    "task038_slowfast_functional_coverage_migration/"
    "sigma_sweep_n09_remain50"
)
REFERENCE_OUTPUT = Path(
    "/data/jixinye25/work1/output/"
    "task038_slowfast_functional_coverage_migration/"
    "n09_remain50_exact"
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def _sigma_name(sigma: float) -> str:
    return f"sigma_{sigma:.3f}".replace(".", "p")


def _verify_upstream(
    reference_output: Path,
    checkpoint: Path,
    sweep_root: Path,
    *,
    write_manifest: bool = True,
) -> dict[str, Any]:
    preflight_path = reference_output / "preflight.json"
    inventory_path = reference_output / "structure" / "unit_inventory.json"
    descriptors_path = reference_output / "descriptors"
    contribution_path = reference_output / "contribution"
    fields_path = contribution_path / "fields"
    field_manifest_path = contribution_path / "field_manifest.json"
    sample_identity_path = contribution_path / "sample_identity.json"

    required = (
        preflight_path,
        inventory_path,
        descriptors_path,
        contribution_path,
        fields_path,
        field_manifest_path,
        sample_identity_path,
        checkpoint,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing authoritative upstream artifacts: " + ", ".join(missing))

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = preflight["checkpoint"]["sha256"]
    actual_checkpoint_sha = _sha256(checkpoint)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("checkpoint identity differs from official Task038 output")

    field_manifest = json.loads(field_manifest_path.read_text(encoding="utf-8"))
    sample_count = int(field_manifest["sample_count"])
    if sample_count != 9:
        raise RuntimeError(f"official Contribution Field sample_count is {sample_count}, expected 9")
    sample_identity = json.loads(sample_identity_path.read_text(encoding="utf-8"))
    if len(sample_identity) != 9:
        raise RuntimeError("official N=9 sample identity manifest is not length 9")

    structure_sha = _sha256(inventory_path)
    descriptor_identity = _tree_identity(descriptors_path)
    contribution_identity = _tree_identity(fields_path)
    field_manifest_sha = _sha256(field_manifest_path)
    sample_identity_sha = _sha256(sample_identity_path)
    code_root = Path(__file__).resolve().parent
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    identity = {
        "schema": "task038_sigma_sweep_upstream_reuse_manifest_v1",
        "reuse_mode": "read_only_symlink_to_valid_task038_artifacts",
        "reference_output": str(reference_output),
        "checkpoint": {"path": str(checkpoint), "sha256": actual_checkpoint_sha},
        "architecture": preflight["architecture"],
        "unit_count": int(preflight["unit_count"]),
        "inventory_sha256": structure_sha,
        "descriptor_artifact": {
            "path": str(descriptors_path),
            **descriptor_identity,
        },
        "contribution_field_archive": {
            "path": str(fields_path),
            **contribution_identity,
        },
        "n09_sample_manifest": {
            "path": str(sample_identity_path),
            "sha256": sample_identity_sha,
            "sample_count": sample_count,
        },
        "contribution_field_manifest": {
            "path": str(field_manifest_path),
            "sha256": field_manifest_sha,
            "sample_count": sample_count,
        },
        "descriptor_code": {
            "path": str(code_root / "slowfast_descriptor_adapter.py"),
            "sha256": _sha256(code_root / "slowfast_descriptor_adapter.py"),
        },
        "current_code_commit": current_commit,
        "bms_frozen_parameters": {
            "tol": 1e-4,
            "max_iters": 100,
            "sink_merge_tol": 0.01,
        },
        "target_remaining_ratio": TARGET_REMAINING_RATIO,
        "sample_identity": sample_identity,
        "science_unchanged_by_sweep": True,
    }
    if write_manifest:
        _json(sweep_root / "upstream_reuse_manifest.json", identity)
    return identity


def _prepare_sigma_dir(
    sigma_dir: Path,
    descriptors: Path,
    contribution: Path,
) -> None:
    if sigma_dir.exists() and any(sigma_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty sigma output: {sigma_dir}"
        )
    sigma_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (
        ("descriptors", descriptors),
        ("contribution", contribution),
    ):
        destination = sigma_dir / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.symlink_to(source, target_is_directory=True)


def _run_cli(
    mode: str,
    sigma_dir: Path,
    checkpoint: Path,
    device: str,
    sigma: float,
    cuda_visible_devices: str,
) -> None:
    if mode not in {"bms", "selection", "logical", "preft"}:
        raise ValueError(f"sweep mode is not allowed: {mode}")
    command = [
        sys.executable,
        "-u",
        "-m",
        "task038_slowfast.task038_cli",
        "--mode",
        mode,
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--target-remaining-ratio",
        f"{TARGET_REMAINING_RATIO:.2f}",
        "--output_dir",
        str(sigma_dir),
    ]
    if mode == "bms":
        command.extend(["--bms-sigma", f"{sigma:.3f}"])
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "TASK038_TARGET_REMAINING_RATIO": f"{TARGET_REMAINING_RATIO:.2f}",
        }
    )
    print(
        f"\n=== Task038 sigma={sigma:.3f} mode={mode} "
        f"(Python; no fine-tuning) ===",
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)


def _unit_labels(unit: Unit) -> tuple[str, str, str]:
    pathway = {"fast": "Fast", "slow": "Slow", "lateral": "Lateral"}[unit.pathway]
    conv = unit.conv_position
    stage = unit.stage if unit.stage.startswith("res") else "lateral"
    return pathway, conv, stage


def _basic_stats(units: Iterable[Unit], pruned: set[int]) -> dict[str, Any]:
    selected = list(units)
    removed = sum(unit.global_index in pruned for unit in selected)
    total = len(selected)
    remaining = total - removed
    return {
        "total": total,
        "removed": removed,
        "remaining": remaining,
        "removed_ratio": removed / total if total else 0.0,
    }


def _write_bms_products(
    sigma: float,
    sigma_dir: Path,
    inventory: UnitInventory,
) -> dict[str, Any]:
    raw_path = sigma_dir / "bms" / "bms_domains.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    parameters = raw.get("parameters", {})
    if abs(float(parameters.get("sigma", -1.0)) - sigma) > 1e-12:
        raise RuntimeError("BMS output sigma does not match requested sweep sigma")
    domains = [list(map(int, domain)) for domain in raw["domains"]]
    sizes = [len(domain) for domain in domains]
    if sum(sizes) != inventory.num_units:
        raise RuntimeError("BMS domain sizes do not cover the inventory")
    if sorted(index for domain in domains for index in domain) != list(range(inventory.num_units)):
        raise RuntimeError("BMS domains are not an exact partition")

    _json(
        sigma_dir / "bms" / "domains.json",
        {
            "schema": "task038_bms_sigma_domains_v1",
            "sigma": sigma,
            "num_units": inventory.num_units,
            "num_domains": len(domains),
            "domains": domains,
        },
    )
    with (sigma_dir / "bms" / "domain_members.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sigma", "domain_id", "global_index"])
        for domain_id, members in enumerate(domains):
            for global_index in members:
                writer.writerow([f"{sigma:.3f}", domain_id, global_index])

    ranked = sorted(range(len(domains)), key=lambda domain_id: (-sizes[domain_id], domain_id))
    top10 = ranked[:10]
    summary = {
        "sigma": sigma,
        "num_units": inventory.num_units,
        "num_domains": len(domains),
        "mean_domain_size": statistics.fmean(sizes),
        "median_domain_size": statistics.median(sizes),
        "std_domain_size": statistics.pstdev(sizes),
        "min_domain_size": min(sizes),
        "max_domain_size": max(sizes),
        "singleton_domain_count": sum(size == 1 for size in sizes),
        "singleton_domain_ratio": sum(size == 1 for size in sizes) / len(sizes),
        "domains_size_ge_10": sum(size >= 10 for size in sizes),
        "domains_size_ge_50": sum(size >= 50 for size in sizes),
        "domains_size_ge_100": sum(size >= 100 for size in sizes),
        "domains_size_ge_500": sum(size >= 500 for size in sizes),
        "domains_size_ge_1000": sum(size >= 1000 for size in sizes),
        "domains_size_ge_2000": sum(size >= 2000 for size in sizes),
        "domains_size_ge_5000": sum(size >= 5000 for size in sizes),
        "top10_domain_sizes": [sizes[domain_id] for domain_id in top10],
        "largest_domain_id": ranked[0],
        "largest_domain_size": sizes[ranked[0]],
    }
    _json(sigma_dir / "bms" / "domain_summary.json", summary)

    composition_rows = []
    unit_by_index = {unit.global_index: unit for unit in inventory.units}
    for rank, domain_id in enumerate(top10, start=1):
        members = [unit_by_index[index] for index in domains[domain_id]]
        pathways = {"Fast": 0, "Slow": 0, "Lateral": 0}
        convs = {"conv1": 0, "conv2": 0, "conv3": 0, "lateral": 0}
        stages = {"res2": 0, "res3": 0, "res4": 0, "res5": 0, "lateral": 0}
        for unit in members:
            pathway, conv, stage = _unit_labels(unit)
            pathways[pathway] += 1
            convs[conv] += 1
            stages[stage] += 1
        composition_rows.append(
            {
                "sigma": sigma,
                "domain_rank": rank,
                "domain_id": domain_id,
                "domain_size": len(members),
                "fast_count": pathways["Fast"],
                "slow_count": pathways["Slow"],
                "lateral_count": pathways["Lateral"],
                "conv1_count": convs["conv1"],
                "conv2_count": convs["conv2"],
                "conv3_count": convs["conv3"],
                "lateral_conv_count": convs["lateral"],
                "res2_count": stages["res2"],
                "res3_count": stages["res3"],
                "res4_count": stages["res4"],
                "res5_count": stages["res5"],
            }
        )
    composition_fields = list(composition_rows[0]) if composition_rows else []
    with (sigma_dir / "bms" / "top10_domain_composition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=composition_fields)
        writer.writeheader()
        writer.writerows(composition_rows)
    return summary


def _read_registry(sigma_dir: Path) -> dict[str, Any]:
    selection = sigma_dir / "selection"
    registry = json.loads((selection / "f3_registry.json").read_text(encoding="utf-8"))
    shutil.copyfile(selection / "functional_selection_trace.csv", selection / "f3_trace.csv")
    shutil.copyfile(selection / "f3_registry.json", selection / "final_registry.json")
    target = json.loads((selection / "remaining_parameter_target.json").read_text(encoding="utf-8"))
    return {"registry": registry, "target": target}


def _write_pruning_diagnostics(
    sigma: float,
    sigma_dir: Path,
    inventory: UnitInventory,
    registry: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_dir = sigma_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    pruned = {int(index) for index in registry["pruned_global_indices"]}
    units = inventory.units
    categories = {
        "by_pathway": {
            "Fast": lambda unit: unit.pathway == "fast",
            "Slow": lambda unit: unit.pathway == "slow",
            "Lateral": lambda unit: unit.pathway == "lateral",
        },
        "by_conv_type": {
            "Conv1": lambda unit: unit.conv_position == "conv1",
            "Conv2": lambda unit: unit.conv_position == "conv2",
            "Conv3": lambda unit: unit.conv_position == "conv3",
            "Lateral": lambda unit: unit.conv_position == "lateral",
        },
        "by_stage": {
            stage: lambda unit, stage=stage: unit.stage == stage
            for stage in ("res2", "res3", "res4", "res5")
        },
    }
    distribution = {
        "sigma": sigma,
        "total_candidate_units": len(units),
        "removed_units": len(pruned),
        "remaining_units": len(units) - len(pruned),
        "removed_unit_ratio": len(pruned) / len(units),
    }
    for group_name, group in categories.items():
        distribution[group_name] = {
            name: _basic_stats(
                (unit for unit in units if predicate(unit)),
                pruned,
            )
            for name, predicate in group.items()
        }
    combinations = {}
    for pathway in ("fast", "slow"):
        for stage in ("res2", "res3", "res4", "res5"):
            combinations[f"{pathway}_{stage}"] = _basic_stats(
                (
                    unit
                    for unit in units
                    if unit.pathway == pathway and unit.stage == stage
                ),
                pruned,
            )
    distribution["by_pathway_stage"] = combinations

    layer_rows = []
    for layer_name, entry in sorted(registry["layers"].items()):
        total = int(entry["total"])
        removed = int(entry["pruned_count"])
        remaining = int(entry["keep_count"])
        keep_ratio = remaining / total if total else 0.0
        layer_rows.append(
            {
                "sigma": sigma,
                "layer_name": layer_name,
                "total": total,
                "removed": removed,
                "remaining": remaining,
                "removed_ratio": removed / total if total else 0.0,
                "remaining_ratio": keep_ratio,
                "at_min_keep_boundary": int(
                    removed == int(total * (1.0 - MIN_KEEP_RATIO))
                ),
                "remaining_le_15pct": int(keep_ratio <= 0.15),
                "remaining_le_20pct": int(keep_ratio <= 0.20),
                "remaining_le_30pct": int(keep_ratio <= 0.30),
                "remaining_le_40pct": int(keep_ratio <= 0.40),
                "remaining_le_50pct": int(keep_ratio <= 0.50),
            }
        )
    layer_fields = list(layer_rows[0]) if layer_rows else []
    with (diagnostic_dir / "layer_pruning_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_fields)
        writer.writeheader()
        writer.writerows(layer_rows)
    aggressive_layers = sorted(
        layer_rows,
        key=lambda row: (row["remaining_ratio"], -row["removed"], row["layer_name"]),
    )[:20]
    distribution["layer_count"] = len(layer_rows)
    distribution["layers_at_min_keep_boundary"] = sum(
        row["at_min_keep_boundary"] for row in layer_rows
    )
    distribution["layers_le_15pct_remaining"] = sum(
        row["remaining_le_15pct"] for row in layer_rows
    )
    distribution["layers_le_20pct_remaining"] = sum(
        row["remaining_le_20pct"] for row in layer_rows
    )
    distribution["layers_le_30pct_remaining"] = sum(
        row["remaining_le_30pct"] for row in layer_rows
    )
    distribution["layers_le_40pct_remaining"] = sum(
        row["remaining_le_40pct"] for row in layer_rows
    )
    distribution["layers_le_50pct_remaining"] = sum(
        row["remaining_le_50pct"] for row in layer_rows
    )
    distribution["top20_most_aggressively_pruned_layers"] = aggressive_layers
    _json(diagnostic_dir / "pruning_distribution.json", distribution)

    unit_by_index = {unit.global_index: unit for unit in units}
    domains = json.loads((sigma_dir / "bms" / "domains.json").read_text(encoding="utf-8"))["domains"]
    domain_rows = []
    for domain_id, members in enumerate(domains):
        domain_units = [unit_by_index[index] for index in members]
        removed_units = [unit for unit in domain_units if unit.global_index in pruned]
        row = {
            "sigma": sigma,
            "domain_id": domain_id,
            "domain_size": len(domain_units),
            "removed_count": len(removed_units),
            "remaining_count": len(domain_units) - len(removed_units),
            "removed_ratio": len(removed_units) / len(domain_units),
        }
        for label in ("Fast", "Slow", "Lateral"):
            row[label.lower() + "_count"] = sum(
                _unit_labels(unit)[0] == label for unit in domain_units
            )
        for label in ("conv1", "conv2", "conv3", "lateral"):
            column = "lateral_conv_count" if label == "lateral" else label + "_count"
            row[column] = sum(unit.conv_position == label for unit in domain_units)
        for stage in ("res2", "res3", "res4", "res5"):
            row[stage + "_count"] = sum(
                _unit_labels(unit)[2] == stage for unit in domain_units
            )
        row["lateral_stage_count"] = sum(
            _unit_labels(unit)[2] == "lateral" for unit in domain_units
        )
        domain_rows.append(row)
    domain_fields = list(domain_rows[0]) if domain_rows else []
    with (diagnostic_dir / "domain_pruning_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=domain_fields)
        writer.writeheader()
        writer.writerows(domain_rows)
    _json(
        diagnostic_dir / "top20_domains_by_removed_count.json",
        {
            "sigma": sigma,
            "domains": sorted(
                domain_rows,
                key=lambda row: (-row["removed_count"], row["domain_id"]),
            )[:20],
        },
    )
    return distribution


def _write_parameter_statistics(sigma: float, sigma_dir: Path) -> dict[str, Any]:
    summary = json.loads(
        (sigma_dir / "parameter_accounting" / "parameter_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (sigma_dir / "parameter_accounting" / "module_parameter_shapes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    breakdown = {}
    for row in rows:
        name = row["module_name"]
        module_type = row["module_type"]
        if name.startswith("fast_"):
            category = "Fast"
        elif name.startswith("slow_"):
            category = "Slow"
        elif name.startswith("lateral_"):
            category = "Lateral"
        elif module_type == "Linear" or name == "fc":
            category = "FC"
        else:
            category = "other_fixed_modules"
        bucket = breakdown.setdefault(
            category, {"original": 0, "remaining": 0, "removed": 0}
        )
        bucket["original"] += int(row["original_parameters"])
        bucket["remaining"] += int(row["remaining_parameters"])
        bucket["removed"] += int(row["removed_parameters"])
    payload = {
        "sigma": sigma,
        "P_original": summary["original_trainable_parameters"],
        "P_structural_remaining": summary["structural_equivalent_remaining_parameters"],
        "P_structural_removed": summary["structural_equivalent_removed_parameters"],
        "remaining_parameter_ratio": summary["remaining_parameter_ratio"],
        "parameter_pruning_ratio": summary["parameter_pruning_ratio"],
        "target_remaining_ratio": TARGET_REMAINING_RATIO,
        "absolute_target_error_parameters": abs(
            summary["structural_equivalent_remaining_parameters"]
            - summary["original_trainable_parameters"] * TARGET_REMAINING_RATIO
        ),
        "absolute_target_error_percentage_points": abs(
            summary["remaining_parameter_ratio"] - TARGET_REMAINING_RATIO
        ) * 100.0,
        "breakdown": breakdown,
        "accounting_checks": {
            "unaccounted_trainable_parameters": summary["unaccounted_trainable_parameters"],
            "double_counted_parameters": summary["double_counted_parameters"],
        },
    }
    _json(sigma_dir / "diagnostics" / "parameter_statistics.json", payload)
    return payload


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x)
        * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def _write_global_outputs(sweep_root: Path, rows: list[dict[str, Any]]) -> None:
    required_fields = [
        "sigma", "num_domains", "mean_domain_size", "median_domain_size",
        "max_domain_size", "singleton_domain_count", "largest_domain_size",
        "top10_domain_mean_size", "removed_units", "removed_unit_ratio",
        "conv1_removed_ratio", "conv2_removed_ratio", "conv3_removed_ratio",
        "fast_removed_ratio", "slow_removed_ratio", "lateral_removed_ratio",
        "res2_removed_ratio", "res3_removed_ratio", "res4_removed_ratio",
        "res5_removed_ratio", "fast_res4_removed_ratio", "slow_res4_removed_ratio",
        "layers_at_min_keep", "layers_le_20pct_remaining", "layers_le_30pct_remaining",
        "P_original", "P_structural_remaining", "remaining_parameter_ratio",
        "parameter_pruning_ratio", "target_error_pp", "preft_top1", "preft_top5",
        "preft_ce", "preft_samples",
    ]
    with (sweep_root / "sigma_sweep_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=required_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in required_fields} for row in rows
        )

    tables = {
        "sigma_sweep_by_preft_top1.csv": sorted(
            rows, key=lambda row: (-row["preft_top1"], row["sigma"])
        ),
        "sigma_sweep_by_max_domain_size.csv": sorted(
            rows, key=lambda row: (row["max_domain_size"], row["sigma"])
        ),
        "sigma_sweep_by_conv3_pruning.csv": sorted(
            rows, key=lambda row: (row["conv3_removed_ratio"], row["sigma"])
        ),
    }
    for filename, table_rows in tables.items():
        with (sweep_root / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=required_fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in required_fields}
                for row in table_rows
            )

    pair_specs = {
        "sigma__num_domains": ("sigma", "num_domains"),
        "sigma__max_domain_size": ("sigma", "max_domain_size"),
        "sigma__preft_top1": ("sigma", "preft_top1"),
        "max_domain_size__preft_top1": ("max_domain_size", "preft_top1"),
        "conv3_removed_ratio__preft_top1": ("conv3_removed_ratio", "preft_top1"),
        "fast_removed_ratio__preft_top1": ("fast_removed_ratio", "preft_top1"),
        "fast_res4_removed_ratio__preft_top1": (
            "fast_res4_removed_ratio",
            "preft_top1",
        ),
    }
    correlations = {
        "schema": "task038_sigma_sweep_correlation_v1",
        "n": len(rows),
        "pairs": {},
    }
    for name, (x_key, y_key) in pair_specs.items():
        x_values = [float(row[x_key]) for row in rows]
        y_values = [float(row[y_key]) for row in rows]
        correlations["pairs"][name] = {
            "x": x_key,
            "y": y_key,
            "pearson": _pearson(x_values, y_values),
            "spearman": _pearson(_ranks(x_values), _ranks(y_values)),
        }
    _json(sweep_root / "sigma_sweep_analysis.json", correlations)

    by_top1 = sorted(rows, key=lambda row: (-row["preft_top1"], row["sigma"]))
    reference = next(
        row for row in rows if abs(row["sigma"] - 0.100) < 1e-12
    )
    report_lines = [
        "# Task038 BMS sigma sweep diagnostic",
        "",
        "This is a diagnostic ablation. It does not change the official sigma and it launches no fine-tuning.",
        "",
        "| sigma | domains | median domain | max domain | Conv3 removed % | Fast removed % | Fast-res4 removed % | remaining parameter % | preFT Top1 | preFT Top5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['sigma']:.3f} | {row['num_domains']} | "
            f"{row['median_domain_size']:.2f} | {row['max_domain_size']} | "
            f"{100*row['conv3_removed_ratio']:.3f} | "
            f"{100*row['fast_removed_ratio']:.3f} | "
            f"{100*row['fast_res4_removed_ratio']:.3f} | "
            f"{100*row['remaining_parameter_ratio']:.6f} | "
            f"{100*row['preft_top1']:.3f} | {100*row['preft_top5']:.3f} |"
        )
    report_lines.extend(["", "## Top three by pre-FT Top1", ""])
    for index, row in enumerate(by_top1[:3], start=1):
        report_lines.append(
            f"{index}. sigma={row['sigma']:.3f}, Top1={row['preft_top1']:.8f}, "
            f"Top5={row['preft_top5']:.8f}"
        )
    smallest_max = min(
        rows, key=lambda row: (row["max_domain_size"], row["sigma"])
    )
    lowest_conv3 = min(
        rows, key=lambda row: (row["conv3_removed_ratio"], row["sigma"])
    )
    lowest_fast_res4 = min(
        rows, key=lambda row: (row["fast_res4_removed_ratio"], row["sigma"])
    )
    report_lines.extend(
        [
            "",
            "## Criterion winners",
            "",
            f"- Smallest maximum domain: sigma={smallest_max['sigma']:.3f}.",
            f"- Lowest Conv3 removal ratio: sigma={lowest_conv3['sigma']:.3f}.",
            f"- Lowest Fast-res4 removal ratio: sigma={lowest_fast_res4['sigma']:.3f}.",
            "",
            "## Reference sigma=0.100",
            "",
            f"Reference preFT Top1={reference['preft_top1']:.8f}, "
            f"Top5={reference['preft_top5']:.8f}, "
            f"max_domain_size={reference['max_domain_size']}, "
            f"num_domains={reference['num_domains']}.",
            "",
            "## Empirical monotonicity checks",
            "",
        ]
    )
    checks = [
        ("domain count", [row["num_domains"] for row in rows]),
        ("maximum domain size", [row["max_domain_size"] for row in rows]),
        ("Conv3 removal ratio", [row["conv3_removed_ratio"] for row in rows]),
        ("Fast-res4 removal ratio", [row["fast_res4_removed_ratio"] for row in rows]),
        ("pre-FT Top1", [row["preft_top1"] for row in rows]),
    ]
    for label, values in checks:
        increasing = all(a <= b for a, b in zip(values, values[1:]))
        decreasing = all(a >= b for a, b in zip(values, values[1:]))
        report_lines.append(
            f"- {label}: increasing={increasing}, decreasing={decreasing}; "
            "correlation is descriptive only, not causation."
        )
    (sweep_root / "sigma_sweep_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def _summary_row(
    sigma: float,
    sigma_dir: Path,
    bms_summary: dict[str, Any],
    registry_info: dict[str, Any],
    pruning: dict[str, Any],
    parameter: dict[str, Any],
) -> dict[str, Any]:
    target = registry_info["target"]
    registry = registry_info["registry"]
    preft = json.loads(
        (sigma_dir / "preft_validation.json").read_text(encoding="utf-8")
    )
    by_conv = pruning["by_conv_type"]
    by_pathway = pruning["by_pathway"]
    by_stage = pruning["by_stage"]
    top10_mean = statistics.fmean(bms_summary["top10_domain_sizes"])
    return {
        "sigma": sigma,
        "num_domains": bms_summary["num_domains"],
        "mean_domain_size": bms_summary["mean_domain_size"],
        "median_domain_size": bms_summary["median_domain_size"],
        "max_domain_size": bms_summary["max_domain_size"],
        "singleton_domain_count": bms_summary["singleton_domain_count"],
        "largest_domain_size": bms_summary["largest_domain_size"],
        "top10_domain_mean_size": top10_mean,
        "removed_units": pruning["removed_units"],
        "removed_unit_ratio": pruning["removed_unit_ratio"],
        "conv1_removed_ratio": by_conv["Conv1"]["removed_ratio"],
        "conv2_removed_ratio": by_conv["Conv2"]["removed_ratio"],
        "conv3_removed_ratio": by_conv["Conv3"]["removed_ratio"],
        "fast_removed_ratio": by_pathway["Fast"]["removed_ratio"],
        "slow_removed_ratio": by_pathway["Slow"]["removed_ratio"],
        "lateral_removed_ratio": by_pathway["Lateral"]["removed_ratio"],
        "res2_removed_ratio": by_stage["res2"]["removed_ratio"],
        "res3_removed_ratio": by_stage["res3"]["removed_ratio"],
        "res4_removed_ratio": by_stage["res4"]["removed_ratio"],
        "res5_removed_ratio": by_stage["res5"]["removed_ratio"],
        "fast_res4_removed_ratio": pruning["by_pathway_stage"]["fast_res4"]["removed_ratio"],
        "slow_res4_removed_ratio": pruning["by_pathway_stage"]["slow_res4"]["removed_ratio"],
        "layers_at_min_keep": pruning["layers_at_min_keep_boundary"],
        "layers_le_20pct_remaining": pruning["layers_le_20pct_remaining"],
        "layers_le_30pct_remaining": pruning["layers_le_30pct_remaining"],
        "P_original": parameter["P_original"],
        "P_structural_remaining": parameter["P_structural_remaining"],
        "remaining_parameter_ratio": parameter["remaining_parameter_ratio"],
        "parameter_pruning_ratio": parameter["parameter_pruning_ratio"],
        "target_error_pp": parameter["absolute_target_error_percentage_points"],
        "preft_top1": float(preft["top1"]),
        "preft_top5": float(preft["top5"]),
        "preft_ce": float(preft["mean_cross_entropy"]),
        "preft_samples": int(preft["sample_count"]),
        "selected_final_prefix_step": registry["selected_final_prefix_step"],
        "target_remaining_ratio": target["target_remaining_ratio"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        help="physical GPU list exposed to each child process",
    )
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--reference-output", type=Path, default=REFERENCE_OUTPUT)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=None,
        help="optional subset of the frozen 11-value sigma grid",
    )
    parser.add_argument(
        "--read-only-upstream",
        action="store_true",
        help="verify upstream without writing the shared reuse manifest",
    )
    args = parser.parse_args()

    selected_sigmas = SIGMAS if args.sigmas is None else tuple(args.sigmas)
    if not selected_sigmas:
        raise ValueError("--sigmas must contain at least one value")
    if len(set(round(sigma, 6) for sigma in selected_sigmas)) != len(selected_sigmas):
        raise ValueError("--sigmas must not contain duplicates")
    invalid_sigmas = [
        sigma for sigma in selected_sigmas
        if not any(abs(sigma - allowed) <= 1e-12 for allowed in SIGMAS)
    ]
    if invalid_sigmas:
        raise ValueError(f"sigma values outside frozen grid: {invalid_sigmas}")

    sweep_root = args.sweep_root.resolve()
    reference_output = args.reference_output.resolve()
    checkpoint = Path(args.checkpoint).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    if (sweep_root / "sigma_sweep_summary.csv").exists():
        raise FileExistsError(f"refusing to overwrite completed sweep: {sweep_root}")
    print("=== Task038 controlled BMS sigma sweep ===", flush=True)
    print(
        "sigmas=" + ",".join(f"{sigma:.3f}" for sigma in selected_sigmas),
        flush=True,
    )
    print(f"target_remaining_ratio={TARGET_REMAINING_RATIO:.2f}", flush=True)
    print("fine_tuning_launched=False", flush=True)
    identity = _verify_upstream(
        reference_output,
        checkpoint,
        sweep_root,
        write_manifest=not args.read_only_upstream,
    )
    print(
        "upstream identities verified: "
        f"checkpoint={identity['checkpoint']['sha256']} "
        f"inventory={identity['inventory_sha256']} "
        f"sample_count={identity['n09_sample_manifest']['sample_count']}",
        flush=True,
    )

    model = slowfast_16x8_resnet101_kinetics400(101)
    inventory = build_inventory(model, MIN_KEEP_RATIO)
    rows = []
    for sigma in selected_sigmas:
        sigma_dir = sweep_root / _sigma_name(sigma)
        _prepare_sigma_dir(
            sigma_dir,
            reference_output / "descriptors",
            reference_output / "contribution",
        )
        _run_cli("bms", sigma_dir, checkpoint, args.device, sigma, args.cuda_visible_devices)
        bms_summary = _write_bms_products(sigma, sigma_dir, inventory)
        _run_cli("selection", sigma_dir, checkpoint, args.device, sigma, args.cuda_visible_devices)
        registry_info = _read_registry(sigma_dir)
        _run_cli("logical", sigma_dir, checkpoint, args.device, sigma, args.cuda_visible_devices)
        _run_cli("preft", sigma_dir, checkpoint, args.device, sigma, args.cuda_visible_devices)
        pruning = _write_pruning_diagnostics(
            sigma, sigma_dir, inventory, registry_info["registry"]
        )
        parameter = _write_parameter_statistics(sigma, sigma_dir)
        row = _summary_row(
            sigma, sigma_dir, bms_summary, registry_info, pruning, parameter
        )
        rows.append(row)
        _json(sigma_dir / "diagnostic_summary.json", row)
        print(
            f"=== sigma={sigma:.3f} complete: domains={row['num_domains']} "
            f"removed_units={row['removed_units']} "
            f"preft_top1={row['preft_top1']:.8f} ===",
            flush=True,
        )
    if len(selected_sigmas) == len(SIGMAS):
        if len(rows) != len(SIGMAS):
            raise RuntimeError("sigma sweep did not produce exactly eleven rows")
        _write_global_outputs(sweep_root, rows)
        print("=== Task038 sigma sweep complete: eleven sigmas ===", flush=True)
    else:
        print(
            "=== Task038 sigma worker complete: "
            f"{len(rows)} sigma(s); global aggregation deferred ===",
            flush=True,
        )


if __name__ == "__main__":
    main()

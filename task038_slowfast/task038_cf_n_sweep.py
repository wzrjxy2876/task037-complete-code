"""Task038 Contribution Field sample-size ablation at fixed sigma=0.050.

This module is intentionally a thin orchestration layer.  It reuses the
authoritative Task038 probe, archive, BMS, F3 selector, logical pruning, and
pre-finetune validation code.  It never launches fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .slowfast_contribution_probe import probe_contribution_fields
from .slowfast_finetune import balanced_n9_indices, data_paths, set_seed
from .slowfast_functional_archive import ContributionFieldArchive, build_functional_similarity
from .slowfast_model_task038 import slowfast_16x8_resnet101_kinetics400
from .task038_sigma_sweep import (
    _read_registry,
    _summary_row,
    _write_parameter_statistics,
    _write_pruning_diagnostics,
)
from .slowfast_unit_adapter import UnitInventory, build_inventory


N_VALUES: tuple[int, ...] = (9, 18, 27, 36, 45)
SIGMA = 0.050
TARGET_REMAINING_RATIO = 0.50
MIN_KEEP_RATIO = 0.10
SEED = 3407
VIDEOS_PER_CLASS = 3
POOLED_SHAPE = (16, 7, 7)
OUTPUT_ROOT = Path(
    "/data/jixinye25/work1/output/"
    "task038_slowfast_functional_coverage_migration/"
    "cf_n_sweep_sigma0p050_remain50"
)
REFERENCE_OUTPUT = Path(
    "/data/jixinye25/work1/output/"
    "task038_slowfast_functional_coverage_migration/n09_remain50_exact"
)
REFERENCE_SIGMA_OUTPUT = Path(
    "/data/jixinye25/work1/output/"
    "task038_slowfast_functional_coverage_migration/"
    "sigma_sweep_n09_remain50/sigma_0p050"
)
DEFAULT_CHECKPOINT = Path(
    "/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt"
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": str(row["split"]),
        "split_index": int(row["split_index"]),
        "video_id": str(row["video_id"]),
        "label": int(row["label"]),
    }


def _class_name(video_id: str) -> str:
    stem = video_id[2:] if video_id.startswith("v_") else video_id
    return stem.split("_g", 1)[0]


def _read_split_by_class(split_path: str | Path) -> dict[int, list[tuple[int, str]]]:
    by_class: dict[int, list[tuple[int, str]]] = {}
    with Path(split_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 3:
                continue
            by_class.setdefault(int(fields[2]), []).append((index, fields[0]))
    return by_class


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    fields = [
        "global_sample_position",
        "class_rank",
        "class_name",
        "class_index",
        "within_class_rank",
        "video_path",
        "label",
        "source_split",
        "seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _build_nested_manifests(root: Path, reference_contribution: Path) -> dict[str, Any]:
    """Build one deterministic master set while preserving the old N=9 prefix."""
    _, val_split, _ = data_paths()
    existing_path = reference_contribution / "sample_identity.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    expected = balanced_n9_indices(val_split, seed=SEED)
    existing_identity = [_identity(row) for row in existing]
    if existing_identity != expected:
        raise RuntimeError(
            "authoritative N=9 samples do not match balanced_n9_indices; "
            "refusing to create a replacement baseline"
        )
    if len(existing_identity) != 9:
        raise RuntimeError("authoritative N=9 identity is not length 9")

    by_class = _read_split_by_class(val_split)
    eligible = sorted(label for label, rows in by_class.items() if len(rows) >= VIDEOS_PER_CLASS)
    original_classes = [int(row["label"]) for row in existing_identity[::VIDEOS_PER_CLASS]]
    if len(original_classes) != 3 or len(set(original_classes)) != 3:
        raise RuntimeError("authoritative N=9 is not three balanced classes")
    remaining_classes = [label for label in eligible if label not in original_classes]
    if len(remaining_classes) < 12:
        raise RuntimeError("fewer than 12 eligible extension classes")

    rng = random.Random(SEED)
    added_classes = sorted(rng.sample(remaining_classes, 12))
    extension: list[dict[str, Any]] = []
    for label in added_classes:
        chosen = sorted(rng.sample(by_class[label], VIDEOS_PER_CLASS), key=lambda item: item[0])
        extension.extend(
            {
                "split": str(Path(val_split).resolve()),
                "split_index": int(index),
                "video_id": video_id,
                "label": int(label),
            }
            for index, video_id in chosen
        )
    master_identity = existing_identity + extension
    if len(master_identity) != 45:
        raise RuntimeError("master manifest is not length 45")

    class_order = original_classes + added_classes
    class_rank = {label: rank for rank, label in enumerate(class_order, start=1)}
    within_class: dict[int, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    for position, item in enumerate(master_identity, start=1):
        label = int(item["label"])
        within_class[label] = within_class.get(label, 0) + 1
        manifest_rows.append(
            {
                "global_sample_position": position,
                "class_rank": class_rank[label],
                "class_name": _class_name(str(item["video_id"])),
                "class_index": label,
                "within_class_rank": within_class[label],
                "video_path": str(item["video_id"]),
                "label": label,
                "source_split": str(item["split"]),
                "seed": SEED,
            }
        )
    if any(within_class[label] != VIDEOS_PER_CLASS for label in class_order):
        raise RuntimeError("master manifest does not contain exactly three videos per class")

    for left, right in zip((9, 18, 27, 36), (18, 27, 36, 45)):
        if master_identity[:left] != master_identity[:right][:left]:
            raise RuntimeError(f"nested manifest prefix gate failed for {left}->{right}")

    manifest_hashes: dict[str, str] = {}
    manifest_hashes["master_n45_sample_manifest.csv"] = _write_manifest(
        root / "master_n45_sample_manifest.csv", manifest_rows
    )
    for n in N_VALUES:
        manifest_hashes[f"n{n:03d}_manifest.csv"] = _write_manifest(
            root / f"n{n:03d}_manifest.csv", manifest_rows[:n]
        )

    metadata = {
        "schema": "task038_cf_nested_manifest_v1",
        "seed": SEED,
        "videos_per_class": VIDEOS_PER_CLASS,
        "n_values": list(N_VALUES),
        "class_order": class_order,
        "original_n09_classes": original_classes,
        "added_class_indices": added_classes,
        "selection_algorithm": (
            "preserve authoritative balanced_n9_indices; Random(seed) sample 12 "
            "remaining eligible labels; sort labels; Random(seed) sample three "
            "videos per added label; sort by split index"
        ),
        "authoritative_n09_sample_identity_sha256": _sha256(existing_path),
        "manifest_sha256": manifest_hashes,
        "nested_prefix_identity": True,
        "master_sample_identity": master_identity,
    }
    _json(root / "manifest_metadata.json", metadata)
    return metadata


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _write_prefix_archive(
    destination: Path,
    source_fields: Path,
    source_manifest: dict[str, Any],
    sample_identity: Sequence[dict[str, Any]],
    sample_count: int,
    *,
    preserve_manifest_bytes: bool = False,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if preserve_manifest_bytes:
        shutil.copyfile(source_fields / "field_manifest.json", destination / "field_manifest.json")
    else:
        entries = []
        for source in source_manifest["entries"]:
            item = dict(source)
            shape = list(item["shape"])
            shape[0] = 45
            item["shape"] = shape
            item["source_sample_count"] = 45
            entries.append(item)
        _json(
            destination / "field_manifest.json",
            {
                "schema": "task038_signed_contribution_field_prefix_v1",
                "field_semantics": source_manifest["field_semantics"],
                "sample_count": sample_count,
                "feature_dimension": sample_count * int(np.prod(POOLED_SHAPE)),
                "source_sample_count": 45,
                "pooled_shape": list(POOLED_SHAPE),
                "unit_count": source_manifest["unit_count"],
                "entries": entries,
                "sample_identity": list(sample_identity),
                "no_per_video_normalization": True,
                "prefix_of_master_archive": True,
            },
        )
    _json(destination / "sample_identity.json", list(sample_identity))
    for entry in source_manifest["entries"]:
        _link(source_fields / entry["fields_path"], destination / entry["fields_path"])
        _link(source_fields / entry["valid_path"], destination / entry["valid_path"])


def _build_master_archive(
    master_root: Path,
    n09_fields: Path,
    extension_fields: Path,
    n09_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    master_identity: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fields_root = master_root / "fields"
    fields_root.mkdir(parents=True, exist_ok=True)
    extension_by_layer = {entry["layer_name"]: entry for entry in extension_manifest["entries"]}
    entries: list[dict[str, Any]] = []
    for source in n09_manifest["entries"]:
        layer = source["layer_name"]
        extension = extension_by_layer.get(layer)
        if extension is None:
            raise RuntimeError(f"extension archive missing layer {layer}")
        fields9 = np.load(n09_fields / source["fields_path"], mmap_mode="r", allow_pickle=False)
        fields36 = np.load(extension_fields / extension["fields_path"], mmap_mode="r", allow_pickle=False)
        valid9 = np.load(n09_fields / source["valid_path"], mmap_mode="r", allow_pickle=False)
        valid36 = np.load(extension_fields / extension["valid_path"], mmap_mode="r", allow_pickle=False)
        if fields9.shape[1:] != fields36.shape[1:] or fields9.shape[0] != 9 or fields36.shape[0] != 36:
            raise RuntimeError(f"field shape mismatch while joining {layer}")
        target = np.lib.format.open_memmap(
            fields_root / source["fields_path"], mode="w+", dtype=np.float32,
            shape=(45,) + tuple(fields9.shape[1:]),
        )
        target[:9] = np.asarray(fields9, dtype=np.float32)
        target[9:] = np.asarray(fields36, dtype=np.float32)
        if not np.array_equal(target[:9], np.asarray(fields9, dtype=np.float32)):
            raise RuntimeError(f"N=9 prefix identity changed while joining {layer}")
        target.flush()
        valid_target = np.lib.format.open_memmap(
            fields_root / source["valid_path"], mode="w+", dtype=np.bool_,
            shape=(45, fields9.shape[1]),
        )
        valid_target[:9] = np.asarray(valid9, dtype=np.bool_)
        valid_target[9:] = np.asarray(valid36, dtype=np.bool_)
        valid_target.flush()
        entries.append(
            {
                **source,
                "shape": [45] + list(source["shape"])[1:],
                "source_sample_count": 45,
            }
        )
    manifest = {
        "schema": "task038_signed_contribution_field_master_v1",
        "field_semantics": n09_manifest["field_semantics"],
        "sample_count": 45,
        "feature_dimension": 45 * int(np.prod(POOLED_SHAPE)),
        "pooled_shape": list(POOLED_SHAPE),
        "unit_count": n09_manifest["unit_count"],
        "entries": entries,
        "sample_identity": list(master_identity),
        "no_per_video_normalization": True,
        "master_prefix_identity_verified": True,
    }
    _json(fields_root / "field_manifest.json", manifest)
    _json(fields_root / "sample_identity.json", list(master_identity))
    return manifest


def _prepare_shared(root: Path, checkpoint: Path, device: str) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty N-sweep root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    shared = root / "shared"
    shared.mkdir()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    preflight = json.loads((REFERENCE_OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != preflight["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint identity differs from authoritative Task038 output")

    metadata = _build_nested_manifests(root, REFERENCE_OUTPUT / "contribution")
    _link(REFERENCE_OUTPUT / "descriptors", shared / "descriptors")
    _link(REFERENCE_SIGMA_OUTPUT / "bms", shared / "bms")
    bms_payload = json.loads((REFERENCE_SIGMA_OUTPUT / "bms" / "bms_domains.json").read_text())
    bms_parameters = bms_payload.get("parameters", {})
    if abs(float(bms_parameters.get("sigma", -1.0)) - SIGMA) > 1e-12:
        raise RuntimeError("shared BMS sigma is not exactly 0.050")
    if len(bms_payload["domains"]) != 1676:
        raise RuntimeError("shared BMS domain count is not the verified sigma=0.050 reference")

    n09_manifest = json.loads((REFERENCE_SIGMA_OUTPUT / "contribution" / "fields" / "field_manifest.json").read_text())
    if int(n09_manifest["sample_count"]) != 9:
        raise RuntimeError("reference Contribution Field is not N=9")
    master_identity = metadata["master_sample_identity"]
    extension_identity = master_identity[9:]
    extension_root = shared / "contribution_extension_n036"
    model = slowfast_16x8_resnet101_kinetics400(101)
    inventory = build_inventory(model, MIN_KEEP_RATIO)
    set_seed(SEED)
    print("=== Task038 N-sweep prepare: computing only the 36 new videos ===", flush=True)
    extension_manifest = probe_contribution_fields(
        model,
        checkpoint,
        inventory,
        torch.device(device),
        extension_root,
        SEED,
        sample_identity=extension_identity,
    )
    master_root = shared / "contribution_master_n045"
    master_manifest = _build_master_archive(
        master_root,
        REFERENCE_SIGMA_OUTPUT / "contribution" / "fields",
        extension_root / "fields",
        n09_manifest,
        extension_manifest,
        master_identity,
    )

    archive_sources: dict[int, Path] = {9: REFERENCE_SIGMA_OUTPUT / "contribution" / "fields"}
    for n in (18, 27, 36, 45):
        archive_sources[n] = master_root / "fields"
    for n in N_VALUES:
        out = root / f"n{n:03d}"
        out.mkdir()
        _link(shared / "descriptors", out / "descriptors")
        _link(shared / "bms", out / "bms")
        contribution = out / "contribution" / "fields"
        _write_prefix_archive(
            contribution,
            archive_sources[n],
            n09_manifest if n == 9 else master_manifest,
            master_identity[:n],
            n,
            preserve_manifest_bytes=(n == 9),
        )
    descriptor_identity = _tree_sha256(REFERENCE_OUTPUT / "descriptors")
    shared_identity = {
        "schema": "task038_cf_n_sweep_shared_identity_v1",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "descriptor": {"path": str(shared / "descriptors"), **descriptor_identity},
        "bms": {
            "path": str(shared / "bms" / "bms_domains.json"),
            "sha256": _sha256(REFERENCE_SIGMA_OUTPUT / "bms" / "bms_domains.json"),
            "sigma": SIGMA,
            "tol": 1e-4,
            "max_iters": 100,
            "sink_merge_tol": 0.01,
            "num_domains": len(bms_payload["domains"]),
            "max_domain_size": max(map(len, bms_payload["domains"])),
        },
        "n09_reference": {
            "sample_identity_sha256": metadata["authoritative_n09_sample_identity_sha256"],
            "field_manifest_sha256": _sha256(REFERENCE_SIGMA_OUTPUT / "contribution" / "fields" / "field_manifest.json"),
            "fields_root": str(REFERENCE_SIGMA_OUTPUT / "contribution" / "fields"),
        },
        "master_contribution_field": {
            "fields_root": str(master_root / "fields"),
            "field_manifest_sha256": _sha256(master_root / "fields" / "field_manifest.json"),
            "sample_count": 45,
            "normalization": "raw signed pooled float32; one L2 after cross-video concatenation",
        },
        "n_values": list(N_VALUES),
        "sigma": SIGMA,
        "target_remaining_ratio": TARGET_REMAINING_RATIO,
        "fine_tuning_launched": False,
    }
    _json(shared / "shared_identity.json", shared_identity)
    _json(root / "READY.json", {"status": "ready", "shared_identity": shared_identity})
    print("=== Task038 N-sweep shared preparation complete ===", flush=True)


def _run_cli(mode: str, output: Path, checkpoint: Path, device: str, physical_gpu: str) -> None:
    if mode not in {"selection", "logical", "preft"}:
        raise ValueError(mode)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": physical_gpu,
            "TASK038_TARGET_REMAINING_RATIO": f"{TARGET_REMAINING_RATIO:.2f}",
        }
    )
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
        str(output),
    ]
    print(f"=== N={output.name[1:4]} mode={mode} gpu={physical_gpu} (Python; no fine-tuning) ===", flush=True)
    subprocess.run(command, check=True, env=env)


def _similarity_stats(output: Path, inventory: UnitInventory, device: str) -> dict[str, Any]:
    domains = json.loads((output / "bms" / "bms_domains.json").read_text()) ["domains"]
    archive = ContributionFieldArchive(output / "contribution" / "fields", inventory)
    torch_device = torch.device(device)
    positive_chunks: list[np.ndarray] = []
    pair_count = 0
    zero_count = 0
    ge05 = ge08 = ge09 = 0
    for members in domains:
        vectors, valid = archive.load_vectors(members, torch_device)
        similarity = build_functional_similarity(vectors, valid)
        size = len(members)
        if size < 2:
            del vectors, valid, similarity
            continue
        upper = similarity[torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)]
        values = upper.detach().float().cpu().numpy()
        pair_count += int(values.size)
        zero_count += int(np.count_nonzero(values == 0.0))
        ge05 += int(np.count_nonzero(values >= 0.5))
        ge08 += int(np.count_nonzero(values >= 0.8))
        ge09 += int(np.count_nonzero(values >= 0.9))
        positive = values[values > 0.0]
        if positive.size:
            positive_chunks.append(positive)
        del vectors, valid, similarity, upper, values
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    positive = np.concatenate(positive_chunks) if positive_chunks else np.zeros(0, dtype=np.float32)
    return {
        "similarity_pair_count": pair_count,
        "mean_positive_similarity": float(positive.mean()) if positive.size else 0.0,
        "median_positive_similarity": float(np.median(positive)) if positive.size else 0.0,
        "p90_positive_similarity": float(np.quantile(positive, 0.90)) if positive.size else 0.0,
        "p95_positive_similarity": float(np.quantile(positive, 0.95)) if positive.size else 0.0,
        "fraction_similarity_zero": zero_count / pair_count if pair_count else 0.0,
        "fraction_similarity_ge_0p5": ge05 / pair_count if pair_count else 0.0,
        "fraction_similarity_ge_0p8": ge08 / pair_count if pair_count else 0.0,
        "fraction_similarity_ge_0p9": ge09 / pair_count if pair_count else 0.0,
    }


def _install_n09_reuse(output: Path) -> None:
    for name in ("selection", "parameter_accounting"):
        _link(REFERENCE_SIGMA_OUTPUT / name, output / name)
    for name in ("logical_pruning.json", "preft_validation.json"):
        _link(REFERENCE_SIGMA_OUTPUT / name, output / name)


def _make_result_summary(
    n: int,
    output: Path,
    inventory: UnitInventory,
    device: str,
    *,
    reused_n09: bool,
) -> dict[str, Any]:
    registry_info = _read_registry(output)
    pruning = _write_pruning_diagnostics(SIGMA, output, inventory, registry_info["registry"])
    parameter = _write_parameter_statistics(SIGMA, output)
    bms_summary = json.loads((output / "bms" / "domain_summary.json").read_text())
    base = _summary_row(SIGMA, output, bms_summary, registry_info, pruning, parameter)
    sim = _similarity_stats(output, inventory, device)
    manifest_path = OUTPUT_ROOT / f"n{n:03d}_manifest.csv"
    shared_identity = json.loads((OUTPUT_ROOT / "shared" / "shared_identity.json").read_text())
    result = {
        **base,
        **sim,
        "N": n,
        "num_classes": n // VIDEOS_PER_CLASS,
        "videos_per_class": VIDEOS_PER_CLASS,
        "sigma": SIGMA,
        "sample_manifest_sha256": _sha256(manifest_path),
        "descriptor_sha256": shared_identity["descriptor"]["sha256"],
        "bms_domain_sha256": shared_identity["bms"]["sha256"],
        "contribution_field_sha256": _sha256(output / "contribution" / "fields" / "field_manifest.json"),
        "master_contribution_field_sha256": shared_identity["master_contribution_field"]["field_manifest_sha256"],
        "P_original": parameter["P_original"],
        "P_structural_remaining": parameter["P_structural_remaining"],
        "remaining_parameter_ratio": parameter["remaining_parameter_ratio"],
        "parameter_pruning_ratio": parameter["parameter_pruning_ratio"],
        "remaining_units": inventory.num_units - pruning["removed_units"],
        "layers_at_min_keep": pruning["layers_at_min_keep_boundary"],
        "preft_sample_count": base["preft_samples"],
        "reused_n09_reference": reused_n09,
        "fine_tuning_launched": False,
        "target_remaining_ratio": TARGET_REMAINING_RATIO,
    }
    _json(output / "result_summary.json", result)
    return result


def _run_one(n: int, checkpoint: Path, device: str, physical_gpu: str) -> dict[str, Any]:
    if n not in N_VALUES:
        raise ValueError(f"N outside frozen grid: {n}")
    output = OUTPUT_ROOT / f"n{n:03d}"
    status = output / "status.json"
    _json(status, {"status": "RUNNING", "N": n, "gpu": physical_gpu, "fine_tuning_launched": False})
    try:
        if n == 9:
            _install_n09_reuse(output)
            result = _make_result_summary(n, output, build_inventory(slowfast_16x8_resnet101_kinetics400(101), MIN_KEEP_RATIO), device, reused_n09=True)
        else:
            model = slowfast_16x8_resnet101_kinetics400(101)
            inventory = build_inventory(model, MIN_KEEP_RATIO)
            _run_cli("selection", output, checkpoint, device, physical_gpu)
            _run_cli("logical", output, checkpoint, device, physical_gpu)
            _run_cli("preft", output, checkpoint, device, physical_gpu)
            result = _make_result_summary(n, output, inventory, device, reused_n09=False)
        _json(status, {"status": "DONE", "N": n, "gpu": physical_gpu, "result_summary": str(output / "result_summary.json"), "fine_tuning_launched": False})
        print(f"=== N={n} complete: domains={result['num_domains']} removed_units={result['removed_units']} preft_top1={result['preft_top1']:.8f} ===", flush=True)
        return result
    except Exception as exc:
        _json(status, {"status": "FAILED", "N": n, "gpu": physical_gpu, "error": repr(exc), "fine_tuning_launched": False})
        raise


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _registry_sets(output: Path, inventory: UnitInventory) -> dict[str, set[int]]:
    registry = json.loads((output / "selection" / "f3_registry.json").read_text())
    pruned = {int(value) for value in registry["pruned_global_indices"]}
    result: dict[str, set[int]] = {"overall": pruned}
    predicates = {
        "conv1": lambda unit: unit.conv_position == "conv1",
        "conv2": lambda unit: unit.conv_position == "conv2",
        "conv3": lambda unit: unit.conv_position == "conv3",
        "fast": lambda unit: unit.pathway == "fast",
        "slow": lambda unit: unit.pathway == "slow",
        "lateral": lambda unit: unit.pathway == "lateral",
    }
    for name, predicate in predicates.items():
        result[name] = {unit.global_index for unit in inventory.units if predicate(unit) and unit.global_index in pruned}
    return result


def _sequence(output: Path) -> list[int]:
    path = output / "selection" / "f3_selection_sequence.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for key in ("global_index", "selected_global_index", "unit_global_index"):
        if rows and key in rows[0]:
            return [int(row[key]) for row in rows]
    raise RuntimeError(f"F3 sequence has no global-index column: {path}")


def _pair_stability(left_n: int, right_n: int, sets: dict[int, dict[str, set[int]]]) -> dict[str, Any]:
    left, right = sets[left_n], sets[right_n]
    return {
        "N_a": left_n,
        "N_b": right_n,
        "overall_jaccard": _jaccard(left["overall"], right["overall"]),
        "conv1_jaccard": _jaccard(left["conv1"], right["conv1"]),
        "conv2_jaccard": _jaccard(left["conv2"], right["conv2"]),
        "conv3_jaccard": _jaccard(left["conv3"], right["conv3"]),
        "fast_jaccard": _jaccard(left["fast"], right["fast"]),
        "slow_jaccard": _jaccard(left["slow"], right["slow"]),
        "lateral_jaccard": _jaccard(left["lateral"], right["lateral"]),
    }


def _correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    def pearson(a: Sequence[float], b: Sequence[float]) -> float:
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        numerator = sum((u - ma) * (v - mb) for u, v in zip(a, b))
        denominator = math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))
        return numerator / denominator if denominator else 0.0

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + 1 + end) / 2.0
            for index in order[start:end]:
                result[index] = rank
            start = end
        return result

    return {"pearson": pearson(x, y), "spearman": pearson(ranks(x), ranks(y))}


def _aggregate(root: Path) -> dict[str, Any]:
    ready = json.loads((root / "READY.json").read_text())
    if ready["status"] != "ready":
        raise RuntimeError("shared preparation is not ready")
    summaries = {n: json.loads((root / f"n{n:03d}" / "result_summary.json").read_text()) for n in N_VALUES}
    statuses = {n: json.loads((root / f"n{n:03d}" / "status.json").read_text()) for n in N_VALUES}
    if any(status["status"] != "DONE" for status in statuses.values()):
        raise RuntimeError("aggregation requires all five N workers to be DONE")
    model = slowfast_16x8_resnet101_kinetics400(101)
    inventory = build_inventory(model, MIN_KEEP_RATIO)
    sets = {n: _registry_sets(root / f"n{n:03d}", inventory) for n in N_VALUES}
    for n in N_VALUES:
        if summaries[n]["sigma"] != SIGMA or summaries[n]["target_remaining_ratio"] != TARGET_REMAINING_RATIO:
            raise RuntimeError(f"frozen setting mismatch at N={n}")
        if summaries[n]["descriptor_sha256"] != summaries[9]["descriptor_sha256"]:
            raise RuntimeError("descriptor artifact differs across N")
        if summaries[n]["bms_domain_sha256"] != summaries[9]["bms_domain_sha256"]:
            raise RuntimeError("BMS domain artifact differs across N")
    if summaries[9]["sample_manifest_sha256"] != json.loads((root / "manifest_metadata.json").read_text())["manifest_sha256"]["n009_manifest.csv"]:
        raise RuntimeError("N=9 manifest hash changed")

    pairs = [(9, 18), (18, 27), (27, 36), (36, 45), (9, 45), (18, 45), (27, 45), (36, 45)]
    stability = [_pair_stability(a, b, sets) for a, b in pairs]
    stability_by_pair = {(row["N_a"], row["N_b"]): row for row in stability}
    for n in N_VALUES:
        summaries[n]["jaccard_to_n45"] = stability_by_pair.get((n, 45), {"overall_jaccard": 1.0})["overall_jaccard"]
    summary_fields = [
        "N", "num_classes", "sigma", "mean_positive_similarity", "median_positive_similarity",
        "p90_positive_similarity", "removed_units", "removed_unit_ratio", "conv1_removed_ratio",
        "conv2_removed_ratio", "conv3_removed_ratio", "fast_removed_ratio", "slow_removed_ratio",
        "lateral_removed_ratio", "fast_res4_removed_ratio", "slow_res4_removed_ratio",
        "remaining_parameter_ratio", "preft_top1", "preft_top5", "preft_ce", "jaccard_to_n45",
    ]
    with (root / "cf_n_sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: summaries[n][field] for field in summary_fields} for n in N_VALUES)
    stability_fields = [
        "N_a", "N_b", "overall_jaccard", "conv1_jaccard", "conv2_jaccard", "conv3_jaccard",
        "fast_jaccard", "slow_jaccard", "lateral_jaccard",
    ]
    with (root / "cf_n_selection_stability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stability_fields)
        writer.writeheader()
        writer.writerows(stability)
    sorted_rows = sorted((summaries[n] for n in N_VALUES), key=lambda row: (-row["preft_top1"], row["N"]))
    with (root / "cf_n_sweep_by_preft_top1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in summary_fields} for row in sorted_rows)

    sequences = {n: _sequence(root / f"n{n:03d}") for n in N_VALUES}
    prefix_rows = []
    for a, b in pairs:
        common = min(len(sequences[a]), len(sequences[b]))
        for index in range(common):
            if sequences[a][index] != sequences[b][index]:
                common = index
                break
        for limit in (1000, 5000, 10000):
            k = min(limit, len(sequences[a]), len(sequences[b]))
            if k == 0:
                continue
            prefix_rows.append({
                "N_a": a,
                "N_b": b,
                "requested_prefix": limit,
                "valid_prefix": k,
                "ordered_common_prefix": common,
                "ordered_prefix_ratio": common / k,
                "set_prefix_jaccard": _jaccard(set(sequences[a][:k]), set(sequences[b][:k])),
            })
    _json(root / "cf_n_prefix_stability.json", {"rows": prefix_rows})
    correlations = {}
    ns = [float(n) for n in N_VALUES]
    for key in ("preft_top1", "conv3_removed_ratio", "fast_removed_ratio", "fast_res4_removed_ratio", "mean_positive_similarity", "jaccard_to_n45"):
        correlations[key] = _correlation(ns, [float(summaries[n][key]) for n in N_VALUES])
    _json(root / "cf_n_correlation_analysis.json", {"n": 5, "descriptive_only": True, "pairs": correlations})
    _json(root / "cf_n_selection_type_stability.json", {"pairs": stability})

    manifest_metadata = json.loads((root / "manifest_metadata.json").read_text())
    lines = [
        "# Task038 Contribution Field N sweep at sigma=0.050",
        "",
        "Diagnostic only. No fine-tuning or optimizer training was launched.",
        "",
        "| N | classes | Conv3 removed % | Fast removed % | Fast-res4 removed % | remaining parameter % | preFT Top1 | preFT Top5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in N_VALUES:
        row = summaries[n]
        lines.append(
            f"| {n} | {row['num_classes']} | {100*row['conv3_removed_ratio']:.3f} | "
            f"{100*row['fast_removed_ratio']:.3f} | {100*row['fast_res4_removed_ratio']:.3f} | "
            f"{100*row['remaining_parameter_ratio']:.6f} | {100*row['preft_top1']:.3f} | {100*row['preft_top5']:.3f} |"
        )
    lines.extend(["", "## Nested manifest gates", "", f"- Class order: {manifest_metadata['class_order']}", "- N=9 is the preserved authoritative prefix: yes", "- N9 ⊂ N18 ⊂ N27 ⊂ N36 ⊂ N45: exact prefix identity verified", "", "## Consecutive and N=45 Jaccard", ""])
    for row in stability:
        lines.append(f"- J({row['N_a']},{row['N_b']}) = {row['overall_jaccard']:.8f}")
    lines.extend(["", "## Descriptive trends", ""])
    for key, label in (("preft_top1", "preFT Top1"), ("conv3_removed_ratio", "Conv3 removal ratio"), ("fast_removed_ratio", "Fast removal ratio"), ("fast_res4_removed_ratio", "Fast-res4 removal ratio"), ("mean_positive_similarity", "mean positive functional similarity")):
        values = [summaries[n][key] for n in N_VALUES]
        increasing = all(a <= b for a, b in zip(values, values[1:]))
        decreasing = all(a >= b for a, b in zip(values, values[1:]))
        lines.append(f"- {label}: increasing={increasing}, decreasing={decreasing}; correlation is descriptive only.")
    lines.extend(["", "## PreFT ranking", ""])
    for index, row in enumerate(sorted_rows[:3], start=1):
        lines.append(f"{index}. N={row['N']}, Top1={row['preft_top1']:.8f}, Top5={row['preft_top5']:.8f}")
    (root / "cf_n_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(root / "AGGREGATED.json", {"status": "complete", "N_values": list(N_VALUES), "fine_tuning_launched": False, "summary": str(root / "cf_n_sweep_summary.csv")})
    print("=== Task038 N-sweep aggregation complete: five N values ===", flush=True)
    return {"summaries": summaries, "stability": stability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "worker", "aggregate", "all"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase in {"prepare", "all"}:
        _prepare_shared(root, args.checkpoint.resolve(), args.device)
    if args.phase in {"worker", "all"}:
        values = N_VALUES if args.n_values is None else tuple(args.n_values)
        if not values or any(value not in N_VALUES for value in values):
            raise ValueError(f"worker N values must be drawn from {N_VALUES}")
        for value in values:
            _run_one(value, args.checkpoint.resolve(), args.device, args.cuda_visible_devices)
    if args.phase in {"aggregate", "all"}:
        _aggregate(root)


if __name__ == "__main__":
    main()

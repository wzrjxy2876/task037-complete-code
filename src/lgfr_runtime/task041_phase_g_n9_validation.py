#!/usr/bin/env python3
"""Task041 Phase G: N=9 BCTR calibration-stability validation only.

Reuses the frozen Phase-F N=3 signed interactions and Task040 validated
Video-Swin whole-unit mask / fixed-cardinality intervention implementations.
No pruning, finetuning, or changes to scientific definitions are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task041_phase_f_bctr_audit as phase_f

PROJECT_DEFAULT = "/home/jixinye25/jxy_work1/task040_htor_d39a947"
CHECKPOINT_DEFAULT = "/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt"
FRAME_ROOT = "/data/jixinye25/UCF101_Frame"
VAL_LIST = "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
OUTPUT_DEFAULT = "/data/jixinye25/work1/output/task041_phase_g_bctr_n9_validation"
WORK_DEFAULT = "/tmp/task041_phase_g_bctr_n9_validation_work"
TASK040_OUT = Path("/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis")
PHASEF_OUT = Path("/data/jixinye25/work1/output/task041_phase_f_bctr_audit")
PHASED_OUT = Path("/data/jixinye25/work1/output/task041_phase_d_fullval_resolution_audit")
EXPECTED_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DOMAINS = ("271", "297", "269", "400", "415", "102", "103", "113", "76")
SAME_TYPE = ("269", "400", "415", "102", "103", "113", "76")
MIXED = ("271", "297")
SPANS = (1, 2, 4, 8, 16)
N_UNITS = 29
N3_ROWS = 6960
NEW_ROWS = 13920
N9_ROWS = 20880
OUTPUT_FILES = (
    "task041_phase_g_n9_video_manifest.csv",
    "task041_phase_g_new_raw_records.csv",
    "task041_phase_g_all_n9_signatures.csv",
    "task041_phase_g_n9_bounded_reconstruction.csv",
    "task041_phase_g_n9_unit_BCTR.csv",
    "task041_phase_g_n3_vs_n9_stability.csv",
    "task041_phase_g_same_type_ce_oracle.csv",
    "task041_phase_g_secondary_damage_metrics.csv",
    "task041_phase_g_signed_ablation.csv",
    "task041_phase_g_collective_vs_pairwise.csv",
    "task041_phase_g_mixed_cross_type.csv",
    "task041_phase_g_summary.json",
    "task041_phase_g_report.md",
)


def say(message: str) -> None:
    print("[Task041 Phase G] " + message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Phase-G gate failed: " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def norm_int(value: Any) -> int:
    return int(float(str(value).strip()))


def video_base(path: str) -> str:
    return Path(path.replace("\\", "/")).name


def video_key(row: Mapping[str, Any]) -> tuple[int, str, int, int]:
    return (norm_int(row["dataset_index"]), video_base(str(row["video_id"])),
            norm_int(row["duration"]), norm_int(row["label"]))


def interaction(a: Any, b: Any, c: Any, d: Any) -> float:
    # Preserve Task041's raw float32 values, then perform frozen a-b-c+d in float64.
    return float(np.float64(a) - np.float64(b) - np.float64(c) + np.float64(d))


def validate_n9_subset(n9: Sequence[Mapping[str, Any]], n3: Sequence[Mapping[str, Any]]) -> None:
    require(len(n9) == 9, "authoritative Task040 N=9 manifest must contain 9 videos")
    require(len(n3) == 3, "authoritative Task040 N=3 manifest must contain 3 videos")
    n9_keys = [video_key(row) for row in n9]
    n3_keys = [video_key(row) for row in n3]
    require(len(set(n9_keys)) == 9, "duplicate N=9 canonical video identity")
    require(len(set(n3_keys)) == 3, "duplicate N=3 canonical video identity")
    require(set(n3_keys).issubset(set(n9_keys)),
            "Phase-F N=3 videos are not the exact dataset/video subset of Task040 N=9")
    class_counts: dict[int, int] = defaultdict(int)
    classes_by_name: dict[str, int] = {}
    for row in n9:
        cls = video_base(str(row["video_id"])).split("_")[1]
        label = norm_int(row["label"])
        if cls in classes_by_name:
            require(classes_by_name[cls] == label, "class name maps to inconsistent labels")
        classes_by_name[cls] = label
        class_counts[label] += 1
    require(len(class_counts) == 3 and sorted(class_counts.values()) == [3, 3, 3],
            "N=9 manifest must be exactly 3 classes x 3 videos/class")
    require(len(classes_by_name) == 3, "N=9 manifest must contain exactly 3 canonical classes")


def n9_gate(high_correct: int, domain_mean_rho: float | None,
            low_stable: int, high_stable: int) -> tuple[bool, dict[str, bool]]:
    checks = {
        "A_high_R_has_larger_CE_in_at_least_5_of_7": high_correct >= 5,
        "B_domain_balanced_Spearman_is_positive": domain_mean_rho is not None and domain_mean_rho > 0.0,
        "C_N3_to_N9_low_identity_stable_in_at_least_5_of_7": low_stable >= 5,
        "D_N3_to_N9_high_identity_stable_in_at_least_5_of_7": high_stable >= 5,
    }
    return all(checks.values()), checks


def input_paths(checkpoint: Path) -> dict[str, Path]:
    return {
        "n9_manifest": TASK040_OUT / "n09_exact/task040_video_manifest.csv",
        "n3_manifest": TASK040_OUT / "n03_fixed_span/task040_video_manifest.csv",
        "intervention_manifest": TASK040_OUT / "n03_fixed_span/task040_intervention_manifest.json",
        "checkpoint_identity": TASK040_OUT / "n09_exact/task040_checkpoint_identity.json",
        "phasef_units": PHASEF_OUT / "task041_phase_f_unit_BCTR.csv",
        "phasef_signatures": PHASEF_OUT / "task041_phase_f_frame_relation_signatures.csv",
        "fullval_damage": PHASED_OUT / "task041_fullval_unit_damage.csv",
        "checkpoint": checkpoint,
        "val_list": Path(VAL_LIST),
    }


def load_frozen_and_phasef(paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], list[dict[str, str]]]:
    frozen, _ = phase_f.load_frozen(paths["phasef_units"])
    units = {phase_f.int_string(row["candidate_task037_global_index"]): row
             for row in read_csv(paths["phasef_units"])}
    require(len(units) == N_UNITS and set(units) == set(frozen), "Phase-F unit file is not the frozen 29-unit set")
    signatures = read_csv(paths["phasef_signatures"])
    require(len(signatures) == N3_ROWS, "Phase-F raw signed signature count is not exactly 6,960")
    seen_units = set()
    for row in signatures:
        uid = phase_f.int_string(row["candidate_task037_global_index"])
        require(uid in frozen, "Phase-F signature contains a non-frozen unit")
        expected = frozen[uid]
        require(phase_f.freeze_key(row) == phase_f.freeze_key(expected), "Phase-F signature identity differs from frozen unit")
        require(phase_f.domain_string(row["domain_id"]) == expected["domain_id"], "Phase-F domain identity mismatch")
        span = norm_int(row["span"])
        require(span in SPANS and norm_int(row["intervention_level"]) == int(math.log2(span)),
                "Phase-F signed signature has an unexpected temporal span")
        require(norm_int(row["block_size"]) == span and norm_int(row["pair_index"]) in range(16),
                "Phase-F signed signature intervention identity is invalid")
        require(math.isfinite(float(row["C_signed"])), "Phase-F signed interaction is non-finite")
        seen_units.add(uid)
    require(seen_units == set(frozen), "Phase-F signatures do not cover all 29 frozen units")
    fullval = phase_f.load_fullval_damage(paths["fullval_damage"], frozen)
    require(len(fullval) == N_UNITS, "full-validation oracle does not cover exactly 29 units")
    for row in fullval.values():
        require(norm_int(row["n_samples"]) == 3783, "full-validation artifact is not the frozen 3,783-clip oracle")
        require(phase_f.mask_restoration_is_exact(row), "full-validation mask restoration is not exact")
    return frozen, units, signatures


def validate_interventions(path: Path) -> list[dict[str, Any]]:
    import task040_htor_core as core
    saved = read_json(path)
    require(norm_int(saved.get("actual_T", saved.get("temporal_length", -1))) == 32,
            "Phase-C intervention manifest is not T=32")
    expected = json.loads(json.dumps([asdict(item) for item in core.enumerate_fixed_cardinality_temporal_pairs(32)]))
    require(saved.get("interventions") == expected,
            "Phase-C intervention list differs from the validated fixed-cardinality enumeration")
    require(len(expected) == 80, "fixed-cardinality manifest must contain 80 interventions")
    for span in SPANS:
        group = [item for item in expected if norm_int(item["block_size"]) == span]
        require(len(group) == 16, "each temporal span must contain exactly 16 pair swaps")
        require(all(norm_int(item["left_end"]) - norm_int(item["left_start"]) == 1 and
                    norm_int(item["right_end"]) - norm_int(item["right_start"]) == 1 for item in group),
                "an intervention is not an exact two-single-frame swap")
    return expected


def prepare(args: argparse.Namespace) -> None:
    paths = input_paths(Path(args.checkpoint))
    for key, path in paths.items():
        require(path.is_file(), "missing authoritative input " + key + ": " + str(path))
    require(sha256(paths["checkpoint"]) == EXPECTED_SHA, "checkpoint SHA256 differs from the frozen Task040 checkpoint")
    identity = read_json(paths["checkpoint_identity"])
    require(identity.get("checkpoint_sha256") == EXPECTED_SHA, "Task040 Phase-B checkpoint identity SHA mismatch")
    n9, n3 = read_csv(paths["n9_manifest"]), read_csv(paths["n3_manifest"])
    validate_n9_subset(n9, n3)
    require([norm_int(row["video_index"]) for row in n9] == list(range(9)),
            "N=9 manifest video_index must be canonical 0..8")
    frozen, units, phasef_rows = load_frozen_and_phasef(paths)
    interventions = validate_interventions(paths["intervention_manifest"])

    n3_by_index = {norm_int(row["video_index"]): row for row in n3}
    seen_video_index = set()
    for row in phasef_rows:
        old_index = norm_int(row["video_index"])
        require(old_index in n3_by_index, "Phase-F signature video index is outside the N=3 manifest")
        require(str(row["video_id"]) == str(n3_by_index[old_index]["video_id"]),
                "Phase-F N=3 signature video identity differs from the authoritative manifest")
        seen_video_index.add(old_index)
    require(seen_video_index == set(n3_by_index), "Phase-F signatures do not contain all N=3 videos")
    phasef_dims: dict[tuple[str, int], set[tuple[int, int, int]]] = defaultdict(set)
    for row in phasef_rows:
        uid, span = phase_f.int_string(row["candidate_task037_global_index"]), norm_int(row["span"])
        phasef_dims[(uid, span)].add((norm_int(row["video_index"]), norm_int(row["intervention_level"]),
                                     norm_int(row["pair_index"])))
    for uid in frozen:
        for span in SPANS:
            expected_dims = {(v, int(math.log2(span)), pair) for v in range(3) for pair in range(16)}
            require(phasef_dims[(uid, span)] == expected_dims,
                    "Phase-F N=3 dimension identity is incomplete or inconsistent")

    raw_lines = Path(VAL_LIST).read_text(encoding="utf-8").splitlines()
    nonempty = [line for line in raw_lines if line.strip()]
    by_dataset = {}
    for row in n9:
        dataset_index = norm_int(row["dataset_index"])
        require(0 <= dataset_index < len(nonempty), "N=9 dataset_index is outside authoritative validation list")
        parts = nonempty[dataset_index].split()
        require(len(parts) >= 3, "unexpected UCF101 validation-list row schema")
        require(parts[0] == video_base(str(row["video_id"])) and
                norm_int(parts[1]) == norm_int(row["duration"]) and
                norm_int(parts[2]) == norm_int(row["label"]),
                "Task040 N=9 manifest does not match the authoritative validation-list row")
        by_dataset[dataset_index] = nonempty[dataset_index]
    require(len(by_dataset) == 9, "duplicate N=9 dataset indices")
    n3_keys = {video_key(row) for row in n3}
    missing_by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in n9:
        if video_key(row) not in n3_keys:
            missing_by_class[norm_int(row["label"])].append(row)
    require(len(missing_by_class) == 3 and all(len(rows) == 2 for rows in missing_by_class.values()),
            "six missing N=9 videos are not balanced as two per class")
    shards: dict[str, list[dict[str, str]]] = {"gpu0": [], "gpu1": []}
    for label in sorted(missing_by_class):
        ordered = sorted(missing_by_class[label], key=lambda row: norm_int(row["video_index"]))
        shards["gpu0"].append(ordered[0])
        shards["gpu1"].append(ordered[1])

    output, work = Path(args.output_dir), Path(args.work_dir)
    require(not work.exists(), "temporary work directory already exists; refusing to overwrite: " + str(work))
    if output.exists():
        require(output.is_dir(), "output path exists and is not a directory")
        require({p.name for p in output.iterdir()} <= {OUTPUT_FILES[0]},
                "Phase-G output directory is non-empty; refusing to overwrite")
    else:
        output.mkdir(parents=True)
    work.mkdir(parents=True)

    shard_for_index = {norm_int(row["video_index"]): worker
                       for worker, rows in shards.items() for row in rows}
    out_manifest = []
    for row in n9:
        out_manifest.append({
            "video_index": norm_int(row["video_index"]),
            "dataset_index": norm_int(row["dataset_index"]),
            "video_id": row["video_id"],
            "canonical_video_id": video_base(row["video_id"]),
            "duration": norm_int(row["duration"]),
            "label": norm_int(row["label"]),
            "phase_f_n3_member": video_key(row) in n3_keys,
            "new_video_worker": shard_for_index.get(norm_int(row["video_index"]), "reused_phase_f"),
        })
    manifest_path = output / OUTPUT_FILES[0]
    if manifest_path.exists():
        require(read_csv(manifest_path) == [{key: str(value) for key, value in row.items()} for row in out_manifest],
                "existing N=9 manifest differs from freshly verified authority")
    else:
        write_csv(manifest_path, out_manifest)

    work_manifest = {
        "task": "Task041 Phase G N=9 BCTR stability validation only",
        "branch_required": "task_041_collective_temporal_coverage_pruning_oracle",
        "checkpoint_sha256": EXPECTED_SHA,
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
        "n9_manifest": out_manifest,
        "frozen_units": [units[uid] for uid in sorted(units, key=int)],
        "interventions": interventions,
        "expected_phase_f_rows": N3_ROWS,
        "expected_new_rows": NEW_ROWS,
        "expected_final_rows": N9_ROWS,
        "shards": {},
    }
    for worker, rows in shards.items():
        list_path = work / (worker + "_val_list.txt")
        with list_path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(by_dataset[norm_int(row["dataset_index"])] for row in rows) + "\n")
        work_manifest["shards"][worker] = {
            "physical_gpu": 0 if worker == "gpu0" else 1,
            "n9_video_indices": [norm_int(row["video_index"]) for row in rows],
            "video_basenames": [video_base(row["video_id"]) for row in rows],
            "dataset_indices": [norm_int(row["dataset_index"]) for row in rows],
            "val_list": str(list_path),
            "expected_raw_rows": N_UNITS * 3 * 80,
        }
    write_json_new(work / "phase_g_prepared.json", work_manifest)
    say("verified N=9 manifest written before any GPU inference: " + str(manifest_path))
    say("exact N=3 subset, 29 frozen units, 80 interventions/video, checkpoint SHA, and 3+3 disjoint-video shards validated")


def candidate_expressions(frozen: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [re.escape(str(item["candidate_layer_name"])) + ":" +
            str(item["candidate_unit_type"]) + ":" + str(item["candidate_unit_index"])
            for _, item in sorted(frozen.items(), key=lambda pair: int(pair[0]))]


def load_model_and_units(project_root: Path, checkpoint: Path, device: Any,
                         frozen: Mapping[str, Mapping[str, Any]]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import task040_htor_probe as probe
    import probe_ctfrs_dynamic_function as ctfrs
    adapter = importlib.import_module("ucf101_videoswin_probe_adapter_v2")
    model, adapter_meta = adapter.build_model_for_probe(checkpoint=str(checkpoint), device=device)
    model.eval()
    model.requires_grad_(False)
    specs = ctfrs.discover_unit_layers(model)
    identity = probe.make_checkpoint_identity(model, checkpoint, adapter_meta, specs)
    require(identity["checkpoint_sha256"] == EXPECTED_SHA, "loaded checkpoint identity SHA mismatch")
    require(not identity["missing_keys"] and not identity["unexpected_keys"] and
            not identity["shape_mismatches"], "checkpoint has missing/unexpected/mismatched tensors")
    require(identity["classifier_head"]["status"] == "loaded", "400-dimensional classifier was not loaded")
    require(identity["loaded_tensor_count"] == 353 and identity["loaded_parameter_count"] == 53504614,
            "loaded checkpoint tensor/parameter count differs from Task040 Phase B")
    require(identity["model_parameter_count"] == 49816678 and identity["model_trainable_parameter_count"] == 0,
            "model parameter identity differs from Task040 Phase B")
    require(identity["discovered_pruning_layer_count"] == 48 and
            identity["discovered_attention_head_count"] == 282 and
            identity["discovered_ffn_neuron_count"] == 36096,
            "pruning-unit discovery differs from Task040 Phase B")
    selected = probe.select_units(specs, ".*", candidate_expressions(frozen), 1, ctfrs)
    require(len(selected) == N_UNITS, "Task040 explicit unit discovery did not return 29 units")
    for unit in selected:
        uid = str(unit.global_index)
        require(uid in frozen, "Task040 selected a unit outside the frozen Task041 identity set")
        item = frozen[uid]
        observed = (str(unit.layer_name), str(unit.unit_type), str(unit.unit_index), str(unit.spec.stage))
        expected = (str(item["candidate_layer_name"]), str(item["candidate_unit_type"]),
                    str(item["candidate_unit_index"]), str(item["candidate_stage"]))
        require(observed == expected, "Task040 global index/layer/type/unit/stage mismatch for " + uid)
    require({unit.unit_type for unit in selected} == {"head", "neuron"},
            "frozen selection must contain both attention heads and FFN neurons")
    return probe, ctfrs, model, identity


def preflight(args: argparse.Namespace) -> None:
    import torch
    project_root, checkpoint = Path(args.project_root).resolve(), Path(args.checkpoint).resolve()
    work, output = Path(args.work_dir), Path(args.output_dir)
    config = read_json(work / "phase_g_prepared.json")
    require((output / OUTPUT_FILES[0]).is_file(), "verified N=9 manifest is absent")
    frozen = {str(item["candidate_task037_global_index"]): item for item in config["frozen_units"]}
    say("CPU-only checkpoint/unit identity preflight; no GPU inference")
    _, ctfrs, model, identity = load_model_and_units(project_root, checkpoint, torch.device("cpu"), frozen)
    authoritative = read_json(TASK040_OUT / "n09_exact/task040_checkpoint_identity.json")
    for key in ("checkpoint_sha256", "loaded_tensor_count", "loaded_parameter_count",
                "model_parameter_count", "discovered_pruning_layer_count",
                "discovered_attention_head_count", "discovered_ffn_neuron_count"):
        require(identity.get(key) == authoritative.get(key), "CPU preflight differs from Task040 Phase-B field " + key)
    require(authoritative.get("baseline_model_sanity", {}).get("output_shape") == [1, 400],
            "authoritative Task040 classifier output is not 400-dimensional")
    shard_checks = {}
    for worker, shard in config["shards"].items():
        loader, _, chosen = ctfrs.build_balanced_loader(
            project_root=project_root, val_list=shard["val_list"], frame_root=FRAME_ROOT,
            num_classes=3, videos_per_class=1, num_workers=0, seed=3407 + int(shard["physical_gpu"]))
        require(len(chosen) == 3, "preflight did not select all 3 shard classes")
        dataset = loader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        seen = []
        for batch in loader:
            _, labels, dataset_indices = batch[:3]
            index = int(dataset_indices[0].item())
            seen.append((video_base(str(dataset.clips[index][0])), int(labels[0].item())))
        expected = sorted((video_base(row["video_id"]), norm_int(row["label"]))
                          for row in config["n9_manifest"]
                          if norm_int(row["video_index"]) in shard["n9_video_indices"])
        require(sorted(seen) == expected, worker + " preflight loader identity differs from exact shard")
        shard_checks[worker] = {"observed": seen, "videos": len(seen), "frame_paths_validated": True}
    write_json_new(work / "phase_g_preflight.json", {
        "checkpoint_identity": identity, "classifier_output_shape_from_authoritative_phase_b": [1, 400],
        "dtype": "float32", "amp": False, "selected_unit_count": len(frozen),
        "exact_unit_identity_match": True, "shard_input_validation": shard_checks,
        "gpu_inference_performed": False,
    })
    say("CPU preflight passed: 29/29 units, classifier=400, 6/6 new videos and sampled frames validated")
    del model


RAW_FIELDS = (
    "candidate_task037_global_index", "candidate_task040_global_index", "candidate_layer_name",
    "candidate_unit_type", "candidate_unit_index", "candidate_stage", "domain_id",
    "physical_gpu", "video_index", "dataset_index", "video_id", "canonical_video_id",
    "label", "span", "dimension_index", "intervention_level", "block_size", "pair_index",
    "a_unmasked_original", "b_masked_original", "c_unmasked_intervened", "d_masked_intervened",
    "C_signed", "mask_restored_exact",
)


def worker(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    import task040_htor_core as core
    project_root, checkpoint = Path(args.project_root).resolve(), Path(args.checkpoint).resolve()
    work = Path(args.work_dir)
    config = read_json(work / "phase_g_prepared.json")
    require((work / "phase_g_preflight.json").is_file(), "CPU preflight must pass before GPU worker")
    worker_name = "gpu" + str(args.gpu)
    shard = config["shards"][worker_name]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu),
            "worker must be isolated with CUDA_VISIBLE_DEVICES matching its physical GPU")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "worker must see exactly one CUDA device")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(3407 + int(args.gpu))
    torch.cuda.manual_seed_all(3407 + int(args.gpu))
    device = torch.device("cuda:0")
    frozen = {str(item["candidate_task037_global_index"]): item for item in config["frozen_units"]}
    probe, ctfrs, model, identity = load_model_and_units(project_root, checkpoint, device, frozen)
    selected = probe.select_units(ctfrs.discover_unit_layers(model), ".*",
                                  candidate_expressions(frozen), 1, ctfrs)
    selected.sort(key=lambda unit: unit.global_index)
    require(len(selected) == N_UNITS, "GPU worker unit set differs from preflight")
    loader, _, _ = ctfrs.build_balanced_loader(
        project_root=project_root, val_list=shard["val_list"], frame_root=FRAME_ROOT,
        num_classes=3, videos_per_class=1, num_workers=2, seed=3407 + int(args.gpu))
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    n9_by_base = {row["canonical_video_id"]: row for row in config["n9_manifest"]}
    expected_bases = set(shard["video_basenames"])
    out_path, done_path = work / (worker_name + "_new_raw_records.csv"), work / (worker_name + "_done.json")
    require(not out_path.exists() and not done_path.exists(), "worker output already exists; refusing to overwrite")
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(32)
    core.verify_intervention_identity(interventions, 32)
    require(len(interventions) == 80, "worker intervention count is not 80")
    seen_bases, total_rows = set(), 0
    say(worker_name + " starting on physical GPU " + str(args.gpu) + " with FP32, AMP=False")
    with out_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for batch in tqdm(loader, total=3, desc="Phase G " + worker_name + " videos", ncols=100):
            videos, targets, dataset_indices = batch[:3]
            videos = videos.float().to(device, non_blocking=True)
            label, local_index = int(targets[0].item()), int(dataset_indices[0].item())
            base = video_base(str(dataset.clips[local_index][0]))
            require(base in expected_bases and base not in seen_bases, "loader produced an unexpected/duplicate shard video")
            canonical = n9_by_base[base]
            require(label == norm_int(canonical["label"]), "video label differs from N=9 manifest")
            seen_bases.add(base)
            video_index, dataset_index = norm_int(canonical["video_index"]), norm_int(canonical["dataset_index"])
            with torch.inference_mode():
                base_logits = probe.unwrap_logits(model(videos))
                require(tuple(base_logits.shape) == (1, 400), "model output is not the frozen 400-class output")
                require(bool(torch.isfinite(base_logits).all().item()), "non-finite unmasked baseline logits")
                a = float(base_logits[0, label].detach().cpu().item())
                intervention_batches, c_by_intervention = [], {}
                clip = videos[0]
                for start in range(0, len(interventions), 2):
                    group = interventions[start:start + 2]
                    values = core.apply_temporal_interventions(clip, group, time_dim=1)
                    logits = probe.unwrap_logits(model(values))
                    require(bool(torch.isfinite(logits).all().item()), "non-finite unmasked intervention logits")
                    group_refs = []
                    for j, intervention in enumerate(group):
                        key = (int(intervention.level), int(intervention.block_size), int(intervention.pair_index))
                        c_by_intervention[key] = float(logits[j, label].detach().cpu().item())
                        group_refs.append(intervention)
                    intervention_batches.append((group_refs, values))
            require(len(c_by_intervention) == 80, "unmasked intervention cache is incomplete")
            for unit in tqdm(selected, total=N_UNITS, desc=base[-28:], leave=False, ncols=100):
                uid, identity_row = str(unit.global_index), frozen[str(unit.global_index)]
                with torch.inference_mode():
                    with probe.temporary_unit_mask(unit.spec, unit.unit_index):
                        masked_logits = probe.unwrap_logits(model(videos))
                        b = float(masked_logits[0, label].detach().cpu().item())
                        d_by_intervention = {}
                        for group, values in intervention_batches:
                            logits = probe.unwrap_logits(model(values))
                            for j, intervention in enumerate(group):
                                key = (int(intervention.level), int(intervention.block_size), int(intervention.pair_index))
                                d_by_intervention[key] = float(logits[j, label].detach().cpu().item())
                require(probe.true_class_logit(model, videos, label) == a,
                        "temporary whole-unit mask restoration failed for candidate " + uid)
                require(len(d_by_intervention) == 80, "masked intervention outputs are incomplete")
                unit_rows = []
                for intervention in interventions:
                    span, pair = int(intervention.block_size), int(intervention.pair_index)
                    key = (int(intervention.level), span, pair)
                    c, d = c_by_intervention[key], d_by_intervention[key]
                    value = interaction(a, b, c, d)
                    require(math.isfinite(value), "non-finite signed interaction")
                    unit_rows.append({
                        "candidate_task037_global_index": uid,
                        "candidate_task040_global_index": identity_row["candidate_task040_global_index"],
                        "candidate_layer_name": identity_row["candidate_layer_name"],
                        "candidate_unit_type": identity_row["candidate_unit_type"],
                        "candidate_unit_index": identity_row["candidate_unit_index"],
                        "candidate_stage": identity_row["candidate_stage"],
                        "domain_id": identity_row["domain_id"], "physical_gpu": args.gpu,
                        "video_index": video_index, "dataset_index": dataset_index,
                        "video_id": canonical["video_id"], "canonical_video_id": base,
                        "label": label, "span": span,
                        "dimension_index": video_index * 16 + pair,
                        "intervention_level": int(intervention.level), "block_size": span,
                        "pair_index": pair, "a_unmasked_original": a, "b_masked_original": b,
                        "c_unmasked_intervened": c, "d_masked_intervened": d,
                        "C_signed": value, "mask_restored_exact": True,
                    })
                writer.writerows(unit_rows)
                handle.flush()
                total_rows += len(unit_rows)
            say(worker_name + " completed video " + base + "; exact-restored candidates=" + str(len(selected)))
    require(seen_bases == expected_bases, "worker did not process its exact 3-video shard")
    require(total_rows == N_UNITS * 3 * 80, "worker raw row count is not 6,960")
    write_json_new(done_path, {
        "worker": worker_name, "physical_gpu": args.gpu, "rows": total_rows,
        "videos": sorted(seen_bases), "candidate_count": len(selected),
        "all_masks_restored_exactly": True, "amp": False, "dtype": "float32",
        "checkpoint_sha256": identity["checkpoint_sha256"],
    })
    say(worker_name + " COMPLETE: " + str(total_rows) + " raw records")


def signature_row(unit: Mapping[str, Any], span: int, video_index: int, video_id: str,
                  pair: int, value: float) -> dict[str, Any]:
    return {
        **{key: unit[key] for key in (
            "candidate_task037_global_index", "candidate_task040_global_index",
            "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
            "candidate_stage", "domain_id")},
        "span": span, "dimension_index": video_index * 16 + pair,
        "video_index": video_index, "video_id": video_id,
        "intervention_level": int(math.log2(span)), "block_size": span, "pair_index": pair,
        "C_signed": value, "representation": "signed_a_minus_b_minus_c_plus_d",
    }


def domain_members(frozen: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for uid, row in frozen.items():
        result[str(row["domain_id"])].append(uid)
    for domain in result:
        result[domain].sort(key=int)
    return result


def build_store(rows: Sequence[Mapping[str, Any]], frozen: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[int, dict[tuple[int, str, int, int, int], float]]], dict[int, list[tuple[int, str, int, int, int]]]]:
    store = {uid: {span: {} for span in SPANS} for uid in frozen}
    for row in rows:
        uid, span = str(row["candidate_task037_global_index"]), norm_int(row["span"])
        key = phase_f.dimension_key(row)
        require(uid in store and span in SPANS, "signature outside frozen units/spans")
        require(key not in store[uid][span], "duplicate signature dimension")
        store[uid][span][key] = float(row["C_signed"])
    orders = {}
    for span in SPANS:
        reference = None
        for uid in sorted(frozen, key=int):
            keys = sorted(store[uid][span], key=lambda key: (key[0], key[1], key[2], key[3], key[4]))
            require(len(keys) in (48, 144), "signature dimension count is neither N=3 nor N=9")
            if reference is None:
                reference = keys
            else:
                require(keys == reference, "canonical video/intervention order differs across units")
        require(reference is not None, "no dimension order constructed")
        orders[span] = reference
    return store, orders


def fit_reconstructions(store: Mapping[str, Mapping[int, Mapping[Any, float]]],
                        orders: Mapping[int, Sequence[Any]],
                        frozen: Mapping[str, Mapping[str, Any]],
                        transform: str = "signed",
                        cross_type_only: bool = False) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    members = domain_members(frozen)
    collective: dict[str, dict[int, float]] = {uid: {} for uid in frozen}
    pairwise: dict[str, dict[int, float]] = {uid: {} for uid in frozen}
    fit_rows, pair_rows = [], []
    for uid in sorted(frozen, key=int):
        item = frozen[uid]
        peers = [peer for peer in members[item["domain_id"]] if peer != uid and
                 (not cross_type_only or frozen[peer]["candidate_unit_type"] != item["candidate_unit_type"])]
        require(bool(peers), "no eligible peers for candidate " + uid)
        for span in SPANS:
            y = phase_f.vector_for(store, orders, uid, span, transform=transform)
            x = np.column_stack([phase_f.vector_for(store, orders, peer, span, transform=transform)
                                 for peer in peers])
            fit = phase_f.bounded_least_squares(y, x)
            require(bool(fit["success"]), "bounded collective least-squares solver failed")
            r = phase_f.residual_ratio(y, fit["reconstruction"])
            collective[uid][span] = r
            coeffs = [{"candidate_task037_global_index": peer, "alpha": float(alpha)}
                      for peer, alpha in zip(peers, fit["alpha"])]
            fit_rows.append({
                **item, "span": span, "representation": transform, "dimension_count": int(len(y)),
                "candidate_norm": float(np.linalg.norm(y)),
                "reconstruction_norm": float(np.linalg.norm(fit["reconstruction"])),
                "residual_norm": float(np.linalg.norm(y - fit["reconstruction"])),
                "residual_ratio": r, "coefficients_json": json.dumps(coeffs, separators=(",", ":")),
                "solver": "scipy.optimize.lsq_linear(method=bvls)", "solver_success": bool(fit["success"]),
                "solver_status": fit["status"], "solver_optimality": fit["optimality"],
                "solver_cost": fit["cost"], "solver_nit": fit["nit"],
                "bounds": "[0,1]", "dtype": "float64", "ridge": False,
                "lambda": "", "sum_alpha_constraint": False, "cross_type_only": bool(cross_type_only),
            })
            choices = []
            for peer in peers:
                x_peer = phase_f.vector_for(store, orders, peer, span, transform=transform)
                alpha, _, pair_r = phase_f.pairwise_fit(y, x_peer)
                choices.append((pair_r, int(peer), peer, alpha))
            best = min(choices, key=lambda val: (val[0], val[1]))
            pairwise[uid][span] = float(best[0])
            pair_rows.append({
                **item, "span": span, "best_single_substitute_task037_global_index": best[2],
                "best_single_substitute_unit_type": frozen[best[2]]["candidate_unit_type"],
                "alpha": float(best[3]), "residual_ratio": float(best[0]),
                "bounds": "[0,1]", "dtype": "float64",
                "tie_break": "ascending Task037 global_index",
            })
    return collective, pairwise, fit_rows, pair_rows


def aggregate_span_residuals(values: Mapping[str, Mapping[int, float]]) -> dict[str, float]:
    return {uid: float(math.sqrt(np.mean([values[uid][span] ** 2 for span in SPANS]))
            ) for uid in values}


def metric_summary(score_rows: Mapping[str, float], damage_rows: Mapping[str, Mapping[str, Any]],
                   frozen: Mapping[str, Mapping[str, Any]], domain: str, metric: str) -> tuple[float | None, float | None]:
    uids = sorted([uid for uid in frozen if frozen[uid]["domain_id"] == domain], key=int)
    return phase_f.rank_statistics([score_rows[uid] for uid in uids],
                                   [float(damage_rows[uid][metric]) for uid in uids])


def stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    require(len(arr) > 0 and bool(np.isfinite(arr).all()), "empty/non-finite reconstructability distribution")
    return {
        "min": float(np.min(arr)), "q25": float(np.quantile(arr, .25)),
        "median": float(np.median(arr)), "mean": float(np.mean(arr)),
        "q75": float(np.quantile(arr, .75)), "max": float(np.max(arr)),
        "strictly_below_zero_reconstruction_baseline_count": int(np.sum(arr < 1.0)),
        "count": int(len(arr)),
    }


def finalize(args: argparse.Namespace) -> None:
    work, output = Path(args.work_dir), Path(args.output_dir)
    config = read_json(work / "phase_g_prepared.json")
    require((work / "phase_g_preflight.json").is_file(), "CPU preflight is missing")
    for path in output.iterdir():
        require(path.name == OUTPUT_FILES[0], "unexpected pre-existing file in Phase-G output directory")
    for worker_name in ("gpu0", "gpu1"):
        done = read_json(work / (worker_name + "_done.json"))
        require(done.get("all_masks_restored_exactly") is True and done.get("rows") == 6960,
                worker_name + " completion/restoration marker is invalid")
        require(done.get("checkpoint_sha256") == EXPECTED_SHA, worker_name + " checkpoint differs")
    frozen, _, phasef_rows = load_frozen_and_phasef(input_paths(Path(args.checkpoint)))
    n9_manifest = read_csv(output / OUTPUT_FILES[0])
    n9_by_index = {norm_int(row["video_index"]): row for row in n9_manifest}
    raw_new, worker_videos = [], {}
    for worker_name in ("gpu0", "gpu1"):
        expected_videos = set(config["shards"][worker_name]["video_basenames"])
        rows = read_csv(work / (worker_name + "_new_raw_records.csv"))
        require(len(rows) == 6960, worker_name + " raw record count is not exactly 6,960")
        observed_videos = {row["canonical_video_id"] for row in rows}
        require(observed_videos == expected_videos, worker_name + " raw videos differ from the deterministic shard")
        worker_videos[worker_name] = observed_videos
        raw_new.extend(rows)
    require(worker_videos["gpu0"].isdisjoint(worker_videos["gpu1"]), "worker shards overlap")
    require(len(raw_new) == NEW_ROWS, "new raw row count is not exactly 13,920")
    seen = set()
    for row in raw_new:
        uid = phase_f.int_string(row["candidate_task037_global_index"])
        require(uid in frozen and phase_f.freeze_key(row) == phase_f.freeze_key(frozen[uid]),
                "new raw unit identity differs from the frozen 29")
        require(row["mask_restored_exact"].lower() == "true", "a candidate mask restoration is false")
        video = n9_by_index[norm_int(row["video_index"])]
        require(video["video_id"] == row["video_id"] and video["canonical_video_id"] == row["canonical_video_id"],
                "new raw video identity differs from verified N=9 manifest")
        span, pair = norm_int(row["span"]), norm_int(row["pair_index"])
        require(span in SPANS and pair in range(16) and norm_int(row["block_size"]) == span,
                "new raw intervention identity invalid")
        require(norm_int(row["intervention_level"]) == int(math.log2(span)) and
                norm_int(row["dimension_index"]) == norm_int(row["video_index"]) * 16 + pair,
                "new raw canonical dimension index invalid")
        require(interaction(row["a_unmasked_original"], row["b_masked_original"],
                            row["c_unmasked_intervened"], row["d_masked_intervened"]) == float(row["C_signed"]),
                "raw C_signed is not exact float64 a-b-c+d")
        key = (uid, span, norm_int(row["video_index"]), pair)
        require(key not in seen, "duplicate new raw unit/video/intervention record")
        seen.add(key)

    n3 = read_csv(input_paths(Path(args.checkpoint))["n3_manifest"])
    n3_index_to_n9 = {}
    n9_by_video = {row["video_id"]: row for row in n9_manifest}
    for row in n3:
        require(row["video_id"] in n9_by_video, "N=3 manifest is no longer a subset of verified N=9")
        n3_index_to_n9[norm_int(row["video_index"])] = norm_int(n9_by_video[row["video_id"]]["video_index"])
    signature_rows = []
    for row in phasef_rows:
        uid = phase_f.int_string(row["candidate_task037_global_index"])
        old_i, pair, span = norm_int(row["video_index"]), norm_int(row["pair_index"]), norm_int(row["span"])
        require(norm_int(row["dimension_index"]) == old_i * 16 + pair,
                "Phase-F N=3 dimension index does not match canonical N=3 ordering")
        signature_rows.append(signature_row(frozen[uid], span, n3_index_to_n9[old_i],
                                             row["video_id"], pair, float(row["C_signed"])))
    for row in raw_new:
        uid = phase_f.int_string(row["candidate_task037_global_index"])
        signature_rows.append(signature_row(frozen[uid], norm_int(row["span"]),
                                             norm_int(row["video_index"]), row["video_id"],
                                             norm_int(row["pair_index"]), float(row["C_signed"])))
    require(len(signature_rows) == N9_ROWS, "combined N=9 signature count is not exactly 20,880")
    signature_rows.sort(key=lambda row: (int(row["candidate_task037_global_index"]),
                                         int(row["span"]), int(row["video_index"]), int(row["pair_index"])))
    n9_store, n9_orders = build_store(signature_rows, frozen)
    require(all(len(n9_store[uid][span]) == 144 for uid in frozen for span in SPANS),
            "N=9 must have exactly 144 dimensions per unit/span")
    n3_signature_rows = [
        signature_row(frozen[phase_f.int_string(row["candidate_task037_global_index"])],
                      norm_int(row["span"]), n3_index_to_n9[norm_int(row["video_index"])],
                      row["video_id"], norm_int(row["pair_index"]), float(row["C_signed"]))
        for row in phasef_rows
    ]
    n3_store, n3_orders = build_store(n3_signature_rows, frozen)
    require(all(len(n3_store[uid][span]) == 48 for uid in frozen for span in SPANS),
            "reused Phase-F N=3 signatures are not exactly 48-D")

    n3_collective, _, _, _ = fit_reconstructions(n3_store, n3_orders, frozen)
    n3_scores = aggregate_span_residuals(n3_collective)
    saved_n3_rows = read_csv(PHASEF_OUT / "task041_phase_f_unit_BCTR.csv")
    saved_n3 = {phase_f.int_string(row["candidate_task037_global_index"]): float(row["R_BCTR"])
                for row in saved_n3_rows}
    require(set(saved_n3) == set(frozen), "saved Phase-F N=3 BCTR identities differ")
    for uid in frozen:
        require(math.isclose(n3_scores[uid], saved_n3[uid], rel_tol=1e-10, abs_tol=1e-12),
                "replayed N=3 BCTR differs from frozen Phase F for candidate " + uid)

    n9_span, pair_span, recon_rows, pair_rows = fit_reconstructions(n9_store, n9_orders, frozen)
    n9_scores, pair_scores = aggregate_span_residuals(n9_span), aggregate_span_residuals(pair_span)
    n3_stability_rows = []
    for domain in DOMAINS:
        ids = sorted([uid for uid in frozen if frozen[uid]["domain_id"] == domain], key=int)
        score3, score9 = {uid: n3_scores[uid] for uid in ids}, {uid: n9_scores[uid] for uid in ids}
        rho, tau = phase_f.rank_statistics([score3[u] for u in ids], [score9[u] for u in ids])
        low3, high3 = min(ids, key=lambda u: (score3[u], int(u))), min(ids, key=lambda u: (-score3[u], int(u)))
        low9, high9 = min(ids, key=lambda u: (score9[u], int(u))), min(ids, key=lambda u: (-score9[u], int(u)))
        order3, order9 = sorted(ids, key=lambda u: (score3[u], int(u))), sorted(ids, key=lambda u: (score9[u], int(u)))
        n3_stability_rows.append({
            "domain_id": domain, "scope": "same_type" if domain in SAME_TYPE else "mixed",
            "unit_count": len(ids), "spearman_N3_vs_N9": rho, "kendall_tau_b_N3_vs_N9": tau,
            "exact_order_match": order3 == order9, "N3_low_candidate": low3, "N9_low_candidate": low9,
            "low_candidate_stable": low3 == low9, "N3_high_candidate": high3, "N9_high_candidate": high9,
            "high_candidate_stable": high3 == high9, "N3_order_task037_indices": json.dumps(order3),
            "N9_order_task037_indices": json.dumps(order9),
        })

    damage = phase_f.load_fullval_damage(input_paths(Path(args.checkpoint))["fullval_damage"], frozen)
    damage_metrics = ("mean_cross_entropy_increase", "mean_true_class_logit_drop", "prediction_flip_rate")
    for row in damage.values():
        for metric in damage_metrics:
            require(math.isfinite(float(row[metric])), "non-finite frozen full-validation outcome " + metric)
    ce_rows, secondary_rows = [], []
    for domain in DOMAINS:
        ids = sorted([uid for uid in frozen if frozen[uid]["domain_id"] == domain], key=int)
        local = [{"candidate_task037_global_index": uid, "score": n9_scores[uid]} for uid in ids]
        low, high = phase_f.choose_low_high(local, "score")
        lo, hi = str(low["candidate_task037_global_index"]), str(high["candidate_task037_global_index"])
        ce_rho, ce_tau = metric_summary(n9_scores, damage, frozen, domain, damage_metrics[0])
        for metric in damage_metrics:
            rho, tau = metric_summary(n9_scores, damage, frozen, domain, metric)
            secondary_rows.append({
                "domain_id": domain, "scope": "same_type" if domain in SAME_TYPE else "mixed",
                "metric": metric, "within_domain_spearman_R_BCTR_N9": rho,
                "within_domain_kendall_tau_b_R_BCTR_N9": tau,
                "N9_low_candidate": lo, "N9_high_candidate": hi,
                "low_damage": float(damage[lo][metric]), "high_damage": float(damage[hi][metric]),
                "ordering": "high_gt_low" if float(damage[hi][metric]) > float(damage[lo][metric])
                            else "reverse" if float(damage[hi][metric]) < float(damage[lo][metric]) else "tie",
            })
        if domain in SAME_TYPE:
            stab = next(row for row in n3_stability_rows if row["domain_id"] == domain)
            ce_rows.append({
                "domain_id": domain, "unit_count": len(ids), "within_domain_spearman_R_BCTR_N9_vs_CE": ce_rho,
                "within_domain_kendall_tau_b_R_BCTR_N9_vs_CE": ce_tau,
                "N9_low_candidate": lo, "N9_high_candidate": hi,
                "low_candidate_CE_increase": float(damage[lo][damage_metrics[0]]),
                "high_candidate_CE_increase": float(damage[hi][damage_metrics[0]]),
                "CE_ordering": "high_gt_low" if float(damage[hi][damage_metrics[0]]) > float(damage[lo][damage_metrics[0]])
                               else "reverse" if float(damage[hi][damage_metrics[0]]) < float(damage[lo][damage_metrics[0]]) else "tie",
                "N3_low_candidate": stab["N3_low_candidate"], "N3_high_candidate": stab["N3_high_candidate"],
                "low_identity_stable": stab["low_candidate_stable"],
                "high_identity_stable": stab["high_candidate_stable"],
            })
    domain_mean_rho = phase_f.safe_mean([row["within_domain_spearman_R_BCTR_N9_vs_CE"] for row in ce_rows])
    domain_mean_tau = phase_f.safe_mean([row["within_domain_kendall_tau_b_R_BCTR_N9_vs_CE"] for row in ce_rows])
    high_correct = sum(row["CE_ordering"] == "high_gt_low" for row in ce_rows)
    low_stable = sum(row["low_identity_stable"] is True for row in ce_rows)
    high_stable = sum(row["high_identity_stable"] is True for row in ce_rows)
    passed, gate_checks = n9_gate(high_correct, domain_mean_rho, low_stable, high_stable)
    decision = "BCTR_STABLE_ENOUGH_FOR_SMALL_PRUNING_PILOT" if passed else "BCTR_REMAINS_WEAK_OR_UNRESOLVED"

    ablation_rows, ablation_means = [], {}
    for transform in ("signed", "absolute", "squared"):
        transformed_span, _, _, _ = fit_reconstructions(n9_store, n9_orders, frozen, transform=transform)
        transformed = aggregate_span_residuals(transformed_span)
        domain_rhos, domain_taus = [], []
        for domain in SAME_TYPE:
            rho, tau = metric_summary(transformed, damage, frozen, domain, damage_metrics[0])
            domain_rhos.append(rho)
            domain_taus.append(tau)
            ablation_rows.append({"representation": transform, "domain_id": domain,
                                  "scope": "same_type", "CE_spearman": rho, "CE_kendall_tau_b": tau})
        mean_rho, mean_tau = phase_f.safe_mean(domain_rhos), phase_f.safe_mean(domain_taus)
        ablation_means[transform] = mean_rho
        ablation_rows.append({"representation": transform, "domain_id": "DOMAIN_BALANCED_MEAN",
                              "scope": "same_type", "CE_spearman": mean_rho, "CE_kendall_tau_b": mean_tau})
    signed_not_less_informative = (
        ablation_means["signed"] is not None and ablation_means["absolute"] is not None and
        ablation_means["squared"] is not None and
        ablation_means["signed"] >= max(ablation_means["absolute"], ablation_means["squared"])
    )

    pair_lookup = {(row["candidate_task037_global_index"], norm_int(row["span"])): row for row in pair_rows}
    collective_pair_rows = []
    for row in recon_rows:
        uid, span = str(row["candidate_task037_global_index"]), norm_int(row["span"])
        pair = pair_lookup[(uid, span)]
        delta = float(pair["residual_ratio"]) - float(row["residual_ratio"])
        collective_pair_rows.append({
            "candidate_task037_global_index": uid, "domain_id": row["domain_id"],
            "unit_type": row["candidate_unit_type"], "span": span,
            "R_collective_span": float(row["residual_ratio"]), "R_best_pairwise_span": float(pair["residual_ratio"]),
            "pairwise_minus_collective": delta, "strict_collective_improvement": delta > 0.0,
            "best_single_substitute_task037_global_index": pair["best_single_substitute_task037_global_index"],
        })
    strict_unit_span = sum(row["strict_collective_improvement"] for row in collective_pair_rows)
    strict_unit = sum(n9_scores[uid] < pair_scores[uid] for uid in frozen)
    differences = [float(row["pairwise_minus_collective"]) for row in collective_pair_rows]
    span_aggregate, largest_by_span = {}, {}
    for span in SPANS:
        rows = [row for row in collective_pair_rows if row["span"] == span]
        best = max(rows, key=lambda row: row["pairwise_minus_collective"])
        span_aggregate[str(span)] = {
            "mean_pairwise_minus_collective": float(np.mean([row["pairwise_minus_collective"] for row in rows])),
            "median_pairwise_minus_collective": float(np.median([row["pairwise_minus_collective"] for row in rows])),
            "strict_improvement_count": sum(row["strict_collective_improvement"] for row in rows),
        }
        largest_by_span[str(span)] = {"candidate_task037_global_index": best["candidate_task037_global_index"],
                                      "improvement": best["pairwise_minus_collective"]}

    cross3, _, _, _ = fit_reconstructions(n3_store, n3_orders, frozen, cross_type_only=True)
    cross9, _, _, _ = fit_reconstructions(n9_store, n9_orders, frozen, cross_type_only=True)
    cross3_score, cross9_score = aggregate_span_residuals(cross3), aggregate_span_residuals(cross9)
    mixed_rows = []
    for uid in sorted(frozen, key=int):
        if frozen[uid]["domain_id"] in MIXED:
            mixed_rows.append({
                **frozen[uid], "N3_cross_type_R_BCTR": cross3_score[uid],
                "N9_cross_type_R_BCTR": cross9_score[uid], "N9_minus_N3": cross9_score[uid] - cross3_score[uid],
                "strict_residual_decrease": cross9_score[uid] < cross3_score[uid],
                "N3_span_residuals_json": json.dumps(cross3[uid], sort_keys=True),
                "N9_span_residuals_json": json.dumps(cross9[uid], sort_keys=True),
            })

    reconstruction_stats = {
        "all_units": stats([n9_scores[uid] for uid in sorted(frozen, key=int)]),
        "attention_heads": stats([n9_scores[uid] for uid in frozen if frozen[uid]["candidate_unit_type"] == "head"]),
        "ffn_neurons": stats([n9_scores[uid] for uid in frozen if frozen[uid]["candidate_unit_type"] == "neuron"]),
        "same_type_domains": stats([n9_scores[uid] for uid in frozen if frozen[uid]["domain_id"] in SAME_TYPE]),
        "mixed_domains": stats([n9_scores[uid] for uid in frozen if frozen[uid]["domain_id"] in MIXED]),
    }
    unit_rows = []
    for uid in sorted(frozen, key=int):
        unit_rows.append({
            **frozen[uid], "R_BCTR_N3": n3_scores[uid], "R_BCTR_N9": n9_scores[uid],
            "R_pair_N9": pair_scores[uid], "pairwise_minus_collective_N9": pair_scores[uid] - n9_scores[uid],
            **{"r_span_" + str(span): n9_span[uid][span] for span in SPANS},
            **{metric: float(damage[uid][metric]) for metric in damage_metrics},
            "fullval_n_samples": norm_int(damage[uid]["n_samples"]),
            "fullval_mask_restored_exact": phase_f.mask_restoration_is_exact(damage[uid]),
        })
    signature_rows.sort(key=lambda row: (int(row["candidate_task037_global_index"]),
                                         int(row["span"]), int(row["video_index"]), int(row["pair_index"])))
    output_rows = {
        OUTPUT_FILES[1]: sorted(raw_new, key=lambda row: (
            int(row["candidate_task037_global_index"]), int(row["span"]),
            int(row["video_index"]), int(row["pair_index"]))),
        OUTPUT_FILES[2]: signature_rows, OUTPUT_FILES[3]: recon_rows,
        OUTPUT_FILES[4]: unit_rows, OUTPUT_FILES[5]: n3_stability_rows,
        OUTPUT_FILES[6]: ce_rows, OUTPUT_FILES[7]: secondary_rows,
        OUTPUT_FILES[8]: ablation_rows, OUTPUT_FILES[9]: collective_pair_rows,
        OUTPUT_FILES[10]: mixed_rows,
    }
    for name, rows in output_rows.items():
        write_csv(output / name, rows)

    summary = {
        "task": "Task041 Phase G — N=9 BCTR calibration stability validation only",
        "decision": decision,
        "decision_rule_note": "Only the predeclared A-gate authorizes A. No numerical B-vs-C rejection threshold was specified; a failed A-gate is conservatively B, not an invented C threshold.",
        "branch_required": "task_041_collective_temporal_coverage_pruning_oracle",
        "checkpoint_sha256": EXPECTED_SHA, "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "amp": False, "dtype": "float32", "gpu_ids_used": [0, 1],
        "sample_count": {"N3_reused_rows": N3_ROWS, "N9_new_rows": NEW_ROWS, "N9_final_rows": N9_ROWS},
        "interventions": {"T": 32, "spans": list(SPANS), "pairs_per_span": 16, "total_per_video": 80},
        "frozen_units": N_UNITS, "domains": list(DOMAINS),
        "n3_to_n9_stability": {"same_type_low_stable_count": low_stable,
                               "same_type_high_stable_count": high_stable,
                               "same_type_domains": [row for row in n3_stability_rows if row["scope"] == "same_type"],
                               "mixed_domains": [row for row in n3_stability_rows if row["scope"] == "mixed"]},
        "primary_CE_gate": {
            "same_type_domains_high_R_higher_CE": high_correct, "same_type_domain_count": 7,
            "domain_balanced_mean_spearman": domain_mean_rho,
            "domain_balanced_mean_kendall_tau_b": domain_mean_tau,
            "low_identity_stable_count": low_stable, "high_identity_stable_count": high_stable,
            "checks": gate_checks, "passed": passed, "per_domain": ce_rows},
        "secondary_damage_metrics": secondary_rows,
        "signed_ablation_domain_balanced_CE_spearman": ablation_means,
        "signed_at_least_as_informative_as_absolute_and_squared": signed_not_less_informative,
        "collective_vs_pairwise": {
            "mean_pairwise_minus_collective_span_residual": float(np.mean(differences)),
            "median_pairwise_minus_collective_span_residual": float(np.median(differences)),
            "strict_collective_improvement_unit_span_count": int(strict_unit_span),
            "unit_span_count": len(collective_pair_rows),
            "strict_collective_improvement_unit_count": int(strict_unit),
            "unit_count": N_UNITS, "span_summary": span_aggregate,
            "largest_improvement_by_span": largest_by_span},
        "reconstructability": reconstruction_stats, "mixed_cross_type_only": mixed_rows,
        "mask_restoration_exact_for_every_new_candidate": True,
        "fullval_oracle_rerun": False, "pruning_or_finetuning_performed": False,
        "task042_created": False,
    }
    write_json_new(output / OUTPUT_FILES[11], summary)
    with (output / OUTPUT_FILES[12]).open("x", encoding="utf-8") as handle:
        handle.write(make_report(summary))
    require({path.name for path in output.iterdir()} == set(OUTPUT_FILES),
            "final output directory does not contain exactly the 13 required Phase-G files")
    say("FINAL: " + decision + "; output=" + str(output))


def make_report(summary: Mapping[str, Any]) -> str:
    gate, recon = summary["primary_CE_gate"], summary["reconstructability"]
    collective, checks = summary["collective_vs_pairwise"], summary["primary_CE_gate"]["checks"]
    lines = [
        "# Task041 Phase G — N=9 BCTR calibration stability", "",
        "## Decision", "", "**" + str(summary["decision"]) + "**", "",
        "This is calibration-stability validation only. It does not authorize pruning, physical removal, or finetuning. Task042 was not created.", "",
        "## Frozen protocol and integrity", "",
        "- Video Swin Transformer / UCF101, authoritative checkpoint SHA256 " + str(summary["checkpoint_sha256"]) + ".",
        "- FP32, AMP disabled; independent single-GPU workers on GPU 0 and GPU 1.",
        "- Exactly 29 frozen Task041 units in domains " + ", ".join(summary["domains"]) + ".",
        "- T=32; spans 1, 2, 4, 8, 16; 16 deterministic two-single-frame swaps/span.",
        "- Frozen signed interaction C=a-b-c+d; bounded BVLS coefficients in [0,1]; no ridge, sum constraint, type coefficient, or score fusion.",
        "- Reused 6,960 Phase-F N=3 rows exactly; acquired 13,920 rows for the six missing videos; final N=9 signatures: 20,880.",
        "- Every new temporary mask was restored exactly. Full-validation masking/bootstrap was not rerun.", "",
        "## Predeclared same-type CE gate", "",
        "| Check | Observed | Pass |", "|---|---:|:---:|",
        "| High-R candidate has larger CE in at least 5/7 domains | " + str(gate["same_type_domains_high_R_higher_CE"]) + "/7 | " + str(checks["A_high_R_has_larger_CE_in_at_least_5_of_7"]) + " |",
        "| Domain-balanced Spearman > 0 | " + str(gate["domain_balanced_mean_spearman"]) + " | " + str(checks["B_domain_balanced_Spearman_is_positive"]) + " |",
        "| N3→N9 low identity stable in at least 5/7 | " + str(gate["low_identity_stable_count"]) + "/7 | " + str(checks["C_N3_to_N9_low_identity_stable_in_at_least_5_of_7"]) + " |",
        "| N3→N9 high identity stable in at least 5/7 | " + str(gate["high_identity_stable_count"]) + "/7 | " + str(checks["D_N3_to_N9_high_identity_stable_in_at_least_5_of_7"]) + " |", "",
        "Domain-balanced CE Spearman: " + str(gate["domain_balanced_mean_spearman"]) +
        "; Kendall tau-b: " + str(gate["domain_balanced_mean_kendall_tau_b"]) + ".", "",
        "## Required secondary diagnostics", "",
        "Logit-drop and prediction-flip results are included unchanged in task041_phase_g_secondary_damage_metrics.csv and the summary; they are not substituted for CE.", "",
        "Signed-ablation domain-balanced CE Spearman: " + json.dumps(summary["signed_ablation_domain_balanced_CE_spearman"], sort_keys=True) + ".",
        "Signed representation is at least as informative as both diagnostic transforms: " +
        str(summary["signed_at_least_as_informative_as_absolute_and_squared"]) + ".", "",
        "Collective versus pairwise is reported as pairwise minus collective (positive favors collective): mean " +
        str(collective["mean_pairwise_minus_collective_span_residual"]) + ", median " +
        str(collective["median_pairwise_minus_collective_span_residual"]) + "; strict unit-span improvements " +
        str(collective["strict_collective_improvement_unit_span_count"]) + "/" +
        str(collective["unit_span_count"]) + ", strict unit-level improvements " +
        str(collective["strict_collective_improvement_unit_count"]) + "/" + str(collective["unit_count"]) + ".", "",
        "## Reconstructability — are units well reconstructed?", "",
        "N=9 unit-level R_BCTR distribution: " + json.dumps(recon["all_units"], sort_keys=True) + ".",
        "R=1 is the exact zero-reconstruction baseline; no extra pass/fail cutoff was invented. The median and quartiles above show whether frozen peers explain these signatures or whether ranking is merely relative among weakly reconstructed signatures. Counts below 1 indicate strict improvement over the zero baseline.", "",
        "Head distribution: " + json.dumps(recon["attention_heads"], sort_keys=True) + ".",
        "FFN distribution: " + json.dumps(recon["ffn_neurons"], sort_keys=True) + ".",
        "Same-type domains: " + json.dumps(recon["same_type_domains"], sort_keys=True) + ".",
        "Mixed domains: " + json.dumps(recon["mixed_domains"], sort_keys=True) + ".", "",
        "## Mixed cross-type-only audit", "",
        "Domains 271/297 are secondary. Candidate-level N=3 and N=9 opposite-type-only residuals are in task041_phase_g_mixed_cross_type.csv; no threshold was introduced and no cross-type replacement claim is made from the gate alone.", "",
        "## Scope stop", "",
        "No full pruning, 50% sparsity, physical pruning, finetuning, or Task042 was performed. Frozen D_abs, D_rel, D_st, BMS, intervention design, BCTR formula, and full-validation damage remain unchanged.", "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task041 Phase G N=9 calibration-stability-only runner")
    parser.add_argument("--project-root", default=PROJECT_DEFAULT)
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--work-dir", default=WORK_DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("preflight")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

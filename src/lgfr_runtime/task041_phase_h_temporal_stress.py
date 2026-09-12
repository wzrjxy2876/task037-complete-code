#!/usr/bin/env python3
"""Task041 Phase H: post-BMS frame-relation temporal stress diagnosis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

BRANCH = "task_041_collective_temporal_coverage_pruning_oracle"
BASE_HEAD = "18001ffc17c4a8469157644118387d076ce796a0"
PROJECT_DEFAULT = "/home/jixinye25/jxy_work1/swintrans_task035"
CHECKPOINT_DEFAULT = "/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt"
FRAME_ROOT = "/data/jixinye25/UCF101_Frame/frames"
VAL_LIST = "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
OUTPUT_DEFAULT = "/data/jixinye25/work1/output/task041_phase_h_temporal_stress_ranking"
WORK_DEFAULT = "/tmp/task041_phase_h_temporal_stress_work"
TASK040_OUT = Path("/data/jixinye25/work1/output/task040_hierarchical_temporal_responsibility_diagnosis")
PHASEF_OUT = Path("/data/jixinye25/work1/output/task041_phase_f_bctr_audit")
PHASEG_OUT = Path("/data/jixinye25/work1/output/task041_phase_g_bctr_n9_validation")
PHASED_OUT = Path("/data/jixinye25/work1/output/task041_phase_d_fullval_resolution_audit")
EXPECTED_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
DOMAINS = ("271", "297", "269", "400", "415", "102", "103", "113", "76")
SAME_TYPE = ("269", "400", "415", "102", "103", "113", "76")
MIXED = ("271", "297")
SPANS = (1, 2, 4, 8, 16)
OUTPUT_FILES = (
    "task041_phase_h_ce_temporal_records.csv",
    "task041_phase_h_unit_temporal_winrate.csv",
    "task041_phase_h_per_span_winrate.csv",
    "task041_phase_h_original_vs_temporal.csv",
    "task041_phase_h_same_type_oracle.csv",
    "task041_phase_h_mixed_domain.csv",
    "task041_phase_h_subset_stability.csv",
    "task041_phase_h_baseline_comparison.csv",
    "task041_phase_h_summary.json",
    "task041_phase_h_report.md",
)
RAW_FIELDS = (
    "candidate_task037_global_index", "candidate_task040_global_index",
    "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
    "candidate_stage", "domain_id", "physical_gpu", "video_index", "dataset_index",
    "canonical_video_id", "label", "condition_kind", "condition_id", "span",
    "intervention_level", "pair_index", "unmasked_cross_entropy",
    "masked_cross_entropy", "signed_ce_damage", "mask_restored_exact",
)
BASELINE_FIELDS = {
    "mean_abs_d_original": "baseline_mean_abs_d_original",
    "G_RMS": "baseline_G_RMS",
    "PTR": "baseline_PTR",
    "corrected_pairwise_best_E": "baseline_corrected_pairwise_best_E",
    "R_MCTC": "baseline_R_MCTC",
}


def say(text: str) -> None:
    print("[Task041 Phase H] " + text, flush=True)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("Phase-H gate failed: " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]],
                  fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_int(value: Any) -> int:
    return int(float(str(value).strip()))


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), label + " is non-finite")
    return result


def video_base(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def runtime_modules():
    runtime = str(Path(__file__).resolve().parent)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import task041_phase_f_bctr_audit as phase_f
    import task041_phase_g_n9_validation as phase_g
    return phase_f, phase_g


def average(values: Iterable[float | None]) -> float | None:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(items) / len(items) if items else None


def pairwise_win(candidate_damage: float, peer_damage: float) -> float:
    if candidate_damage < peer_damage:
        return 1.0
    if candidate_damage == peer_damage:
        return 0.5
    return 0.0


def group_win_rates(
    values: Mapping[str, Mapping[Hashable, float]],
    domains: Mapping[str, str],
    selected_units: Iterable[str] | None = None,
    condition_predicate=None,
) -> dict[str, float]:
    """Mean pairwise wins over common conditions and only same-domain peers."""
    selected = set(values) if selected_units is None else set(selected_units)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(selected, key=int):
        require(uid in values and uid in domains, "win-rate input misses unit " + uid)
        by_domain[str(domains[uid])].append(uid)
    result = {}
    for domain, members in by_domain.items():
        require(len(members) >= 2, "domain " + domain + " has fewer than two units")
        common = set(values[members[0]])
        if condition_predicate:
            common = {key for key in common if condition_predicate(key)}
        require(bool(common), "domain " + domain + " has no selected conditions")
        for uid in members[1:]:
            keys = set(values[uid])
            if condition_predicate:
                keys = {key for key in keys if condition_predicate(key)}
            require(keys == common, "condition set mismatch inside domain " + domain)
        for uid in members:
            wins = [
                pairwise_win(float(values[uid][key]), float(values[peer][key]))
                for key in sorted(common, key=repr)
                for peer in members if peer != uid
            ]
            result[uid] = sum(wins) / len(wins)
    return result


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and float(values[ordered[j]]) == float(values[ordered[i]]):
            j += 1
        rank = ((i + 1) + j) / 2.0
        for p in range(i, j):
            ranks[ordered[p]] = rank
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    vx, vy = sum(v * v for v in dx), sum(v * v for v in dy)
    if vx == 0.0 or vy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tied_x) *
                      (concordant + discordant + tied_y))
    return (concordant - discordant) / denom if denom else None


def ordered_extremes(scores: Mapping[str, float]) -> tuple[str, str]:
    ordered = sorted(scores, key=lambda uid: (float(scores[uid]), int(uid)))
    require(len(ordered) >= 2, "low/high ordering needs at least two candidates")
    return ordered[0], ordered[-1]


def safest_uid(win_rates: Mapping[str, float]) -> str:
    require(bool(win_rates), "cannot choose safest unit from an empty set")
    return min(win_rates, key=lambda uid: (-float(win_rates[uid]), int(uid)))


def fullval_safest_uid(damage: Mapping[str, float]) -> str:
    require(bool(damage), "cannot choose full-validation safest from an empty set")
    return min(damage, key=lambda uid: (float(damage[uid]), int(uid)))


def ordering_label(low_damage: float, high_damage: float) -> str:
    if high_damage > low_damage:
        return "correct"
    if high_damage == low_damage:
        return "tie"
    return "reverse"


def class_balanced_n6_subsets(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for row in manifest:
        by_label[norm_int(row["label"])].append(norm_int(row["video_index"]))
    require(len(by_label) == 3 and all(len(v) == 3 for v in by_label.values()),
            "N=9 manifest must be 3 classes x 3 videos")
    choices = [list(itertools.combinations(sorted(by_label[label]), 2))
               for label in sorted(by_label)]
    result = [
        {"subset_id": "n6_%02d" % index,
         "video_indices": sorted(v for pair in selected for v in pair)}
        for index, selected in enumerate(itertools.product(*choices), start=1)
    ]
    require(len(result) == 27, "expected all 27 class-balanced N=6 subsets")
    return result


def temporal_decision_gate(
    temporal: Mapping[str, float | None],
    original: Mapping[str, float | None],
) -> tuple[str, dict[str, Any]]:
    """A uses exactly the user-declared positive-association/comparison gate.

    No independent post-hoc boundary is defined for C, so a failed A gate is
    reported conservatively as B rather than inventing a rejection threshold.
    """
    primary = ("spearman", "kendall", "safest_accuracy")
    rho = temporal.get("spearman")
    positive = rho is not None and float(rho) > 0.0
    improved = [
        key for key in primary
        if temporal.get(key) is not None and original.get(key) is not None
        and float(temporal[key]) > float(original[key])
    ]
    comparative = False
    for key in improved:
        remaining = [other for other in primary if other != key]
        if not remaining or any(
            temporal.get(other) is not None and original.get(other) is not None
            and float(temporal[other]) >= float(original[other])
            for other in remaining
        ):
            comparative = True
            break
    passed = positive and comparative
    return (
        "TEMPORAL_STRESS_SELECTION_PROMISING" if passed
        else "TEMPORAL_STRESS_ADDS_NO_CLEAR_VALUE",
        {
            "positive_same_type_domain_balanced_ce_spearman": positive,
            "better_than_original_on_at_least_one_primary_measure": bool(improved),
            "not_worse_on_all_remaining_primary_measures": comparative,
            "improved_primary_measures": improved,
            "gate_passed": passed,
            "rejection_threshold_added": False,
        },
    )


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _runtime_paths() -> dict[str, Path]:
    return {
        "n9_manifest": PHASEG_OUT / "task041_phase_g_n9_video_manifest.csv",
        "phase_f_units": PHASEF_OUT / "task041_phase_f_unit_BCTR.csv",
        "phase_f_summary": PHASEF_OUT / "task041_phase_f_summary.json",
        "phase_g_units": PHASEG_OUT / "task041_phase_g_n9_unit_BCTR.csv",
        "phase_g_summary": PHASEG_OUT / "task041_phase_g_summary.json",
        "fullval_damage": PHASED_OUT / "task041_fullval_unit_damage.csv",
        "checkpoint_identity": TASK040_OUT / "n09_exact/task040_checkpoint_identity.json",
        "interventions": TASK040_OUT / "n03_fixed_span/task040_intervention_manifest.json",
        "checkpoint": Path(CHECKPOINT_DEFAULT),
        "val_list": Path(VAL_LIST),
    }


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def prepare(args: argparse.Namespace) -> None:
    phase_f, phase_g = runtime_modules()
    paths = _runtime_paths()
    for name, path in paths.items():
        require(path.is_file(), "missing authoritative input %s: %s" % (name, path))
    checkpoint = Path(args.checkpoint).resolve()
    require(checkpoint == paths["checkpoint"].resolve(),
            "checkpoint path is not the frozen Task041 path")
    require(sha256_file(checkpoint) == EXPECTED_SHA,
            "checkpoint SHA256 differs from the Task041 authority")
    output, work = Path(args.output_dir), Path(args.work_dir)
    require(not output.exists(), "Phase-H output already exists; refusing overwrite")
    require(not work.exists(), "Phase-H work directory already exists; refusing overwrite")
    repo = Path(__file__).resolve().parents[2]
    branch = git_value(repo, "rev-parse", "--abbrev-ref", "HEAD")
    require(branch == BRANCH, "not on frozen Task041 branch: " + branch)

    manifest = phase_g.read_csv(paths["n9_manifest"])
    require(len(manifest) == 9, "Phase-G manifest must contain exactly nine videos")
    require([norm_int(row["video_index"]) for row in manifest] == list(range(9)),
            "Phase-G manifest must be ordered video_index 0..8")
    labels: dict[int, int] = defaultdict(int)
    for row in manifest:
        labels[norm_int(row["label"])] += 1
        require(video_base(row["video_id"]) == row.get(
            "canonical_video_id", video_base(row["video_id"])),
            "manifest canonical video identity mismatch")
    require(len(labels) == 3 and sorted(labels.values()) == [3, 3, 3],
            "Phase-G manifest must be class-balanced 3x3")
    n3 = sorted(norm_int(row["video_index"]) for row in manifest
                if _bool(row.get("phase_f_n3_member", "false")))
    require(len(n3) == 3 and len({norm_int(row["label"]) for row in manifest
                                  if norm_int(row["video_index"]) in n3}) == 3,
            "manifest does not contain the exact class-balanced Phase-F N=3 subset")
    n6 = class_balanced_n6_subsets(manifest)

    phase_f_summary = json.loads(
        paths["phase_f_summary"].read_text(encoding="utf-8")
    )
    require(phase_f_summary.get("decision_gate", {}).get("decision")
            == "RETAINED FOR N=9 VALIDATION",
            "Phase-F frozen decision differs; refusing to reinterpret it")

    frozen, _ = phase_f.load_frozen(paths["phase_f_units"])
    f_rows = phase_g.read_csv(paths["phase_f_units"])
    g_rows = phase_g.read_csv(paths["phase_g_units"])
    f_by_uid = {phase_f.int_string(row["candidate_task037_global_index"]): row for row in f_rows}
    g_by_uid = {phase_f.int_string(row["candidate_task037_global_index"]): row for row in g_rows}
    fullval = phase_f.load_fullval_damage(paths["fullval_damage"], frozen)
    require(len(frozen) == 29 and set(frozen) == set(f_by_uid) == set(g_by_uid) == set(fullval),
            "Phase F/G/full-validation files must join exactly 29 frozen units")
    require({row["domain_id"] for row in frozen.values()} == set(DOMAINS),
            "frozen BMS domain set differs from the Phase-H authority")
    units = []
    for uid in sorted(frozen, key=int):
        identity = frozen[uid]
        drow, frow, grow = fullval[uid], f_by_uid[uid], g_by_uid[uid]
        for item, label in ((frow, "Phase-F"), (grow, "Phase-G"), (drow, "full-validation")):
            observed = (
                phase_f.int_string(item["candidate_task040_global_index"]),
                str(item.get("candidate_layer_name", item.get("layer_name", ""))),
                str(item.get("candidate_unit_type", item.get("unit_type", ""))),
                phase_f.int_string(item.get("candidate_unit_index", item.get("unit_index"))),
                phase_f.domain_string(item["domain_id"]),
            )
            expected = (
                identity["candidate_task040_global_index"], identity["candidate_layer_name"],
                identity["candidate_unit_type"], identity["candidate_unit_index"],
                identity["domain_id"],
            )
            require(observed == expected, "%s identity mismatch for unit %s" % (label, uid))
            if label != "full-validation":
                require(phase_f.int_string(item["candidate_stage"])
                        == identity["candidate_stage"],
                        "%s stage mismatch for unit %s" % (label, uid))
        require(norm_int(drow.get("n_samples", -1)) == 3783
                and phase_f.mask_restoration_is_exact(drow),
                "full-validation artifact is not the exact restored 3,783-clip oracle")
        require(norm_int(grow.get("fullval_n_samples", -1)) == 3783
                and _bool(grow.get("fullval_mask_restored_exact", "false")),
                "Phase-G row is not linked to the frozen full-validation oracle")
        damage = finite_float(drow["mean_cross_entropy_increase"], "full-validation CE")
        require(math.isclose(damage, finite_float(grow["mean_cross_entropy_increase"],
                                                  "Phase-G full-validation CE"),
                             rel_tol=0.0, abs_tol=1e-12),
                "Phase-D/G full-validation CE mismatch for " + uid)
        baseline_scores = {}
        for name, column in BASELINE_FIELDS.items():
            raw_score = str(frow.get(column, "")).strip()
            # Phase F deliberately has no frozen baseline value for 11/29
            # units. Preserve those missing values; never impute or recompute.
            baseline_scores[name] = (
                None if not raw_score else finite_float(raw_score, "baseline " + name)
            )
        units.append({
            **identity,
            "fullval_mean_cross_entropy_increase": damage,
            "fullval_n_samples": 3783,
            "R_BCTR_N9": finite_float(grow["R_BCTR_N9"], "R_BCTR_N9"),
            "baseline_scores": baseline_scores,
        })

    interventions = phase_g.validate_interventions(paths["interventions"])
    require(len(interventions) == 80, "Phase-G fixed-cardinality interventions must total 80")
    for span in SPANS:
        group = [q for q in interventions if norm_int(q["block_size"]) == span]
        require(len(group) == 16 and all(
            norm_int(q["left_end"]) - norm_int(q["left_start"]) == 1
            and norm_int(q["right_end"]) - norm_int(q["right_start"]) == 1
            for q in group
        ), "span intervention manifest does not encode 16 exact two-frame swaps")

    source_lines = [line for line in paths["val_list"].read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    exact_lines = []
    for row in manifest:
        index = norm_int(row["dataset_index"])
        require(0 <= index < len(source_lines), "manifest dataset_index outside validation list")
        line = source_lines[index]
        parts = line.split()
        require(len(parts) >= 3 and parts[0] == video_base(row["video_id"])
                and norm_int(parts[1]) == norm_int(row["duration"])
                and norm_int(parts[2]) == norm_int(row["label"]),
                "Phase-G manifest differs from authoritative validation-list row")
        exact_lines.append(line)
    phase_g_summary = json.loads(paths["phase_g_summary"].read_text(encoding="utf-8"))
    require(phase_g_summary.get("decision") == "BCTR_REMAINS_WEAK_OR_UNRESOLVED",
            "Phase-G frozen decision differs; refusing to reinterpret it")
    repo_head = git_value(repo, "rev-parse", "HEAD")
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    work.mkdir(parents=True, exist_ok=False)
    val_path = work / "phase_h_exact_n9_val_list.txt"
    with val_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(exact_lines) + "\n")
    config = {
        "task": "Task041 Phase H post-BMS frame-relation temporal stress ranking diagnosis",
        "branch_required": BRANCH, "base_head_before_phase_h": BASE_HEAD,
        "prepared_head": repo_head,
        "phase_g_decision": phase_g_summary["decision"],
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": EXPECTED_SHA,
        "project_root": str(Path(args.project_root).resolve()),
        "frame_root": FRAME_ROOT, "output_dir": str(output.resolve()),
        "work_dir": str(work.resolve()), "manifest": manifest, "frozen_units": units,
        "phase_f_n3_video_indices": n3, "n6_subsets": n6,
        "exact_n9_val_list": str(val_path), "interventions": interventions,
        "input_sha256": input_hashes,
        "raw_rows_expected": 29 * 9 * 81,
        "unmasked_cache_rows_expected": 9 * 81,
        "dtype": "float32", "amp": False,
        "mask_restore_policy": (
            "Use Task040 temporary_unit_mask; verify its exact pre-hook table and all "
            "parameter/buffer version counters are restored after every candidate/video, "
            "without repeating candidate-independent unmasked inference."
        ),
    }
    write_json_new(work / "phase_h_prepared.json", config)
    say("Prepared the frozen 29 units, nine manifest videos and 80 exact interventions; no inference.")


def _config(work: Path) -> dict[str, Any]:
    path = work / "phase_h_prepared.json"
    require(path.is_file(), "run prepare before this phase")
    return json.loads(path.read_text(encoding="utf-8"))


def _loader(ctfrs: Any, config: Mapping[str, Any], workers: int):
    return ctfrs.build_balanced_loader(
        project_root=Path(config["project_root"]),
        val_list=str(config["exact_n9_val_list"]),
        frame_root=str(config["frame_root"]),
        num_classes=3, videos_per_class=3, num_workers=workers, seed=3407,
    )


def preflight(args: argparse.Namespace) -> None:
    import torch
    _, phase_g = runtime_modules()
    work = Path(args.work_dir)
    config = _config(work)
    require(not Path(config["output_dir"]).exists(), "output path appeared; refusing overwrite")
    require(sha256_file(Path(config["checkpoint_path"])) == EXPECTED_SHA,
            "checkpoint SHA changed since prepare")
    repo = Path(__file__).resolve().parents[2]
    require(git_value(repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH
            and git_value(repo, "rev-parse", "HEAD") == config["prepared_head"],
            "server checkout differs from the prepared Task041 commit")
    frozen = {str(row["candidate_task037_global_index"]): row
              for row in config["frozen_units"]}
    say("CPU checkpoint/dataset preflight only; no model forward or GPU inference.")
    _, ctfrs, model, identity = phase_g.load_model_and_units(
        Path(config["project_root"]), Path(config["checkpoint_path"]),
        torch.device("cpu"), frozen,
    )
    authority = json.loads(
        (TASK040_OUT / "n09_exact/task040_checkpoint_identity.json").read_text(encoding="utf-8")
    )
    for key in ("checkpoint_sha256", "loaded_tensor_count", "loaded_parameter_count",
                "model_parameter_count", "discovered_pruning_layer_count",
                "discovered_attention_head_count", "discovered_ffn_neuron_count"):
        require(identity.get(key) == authority.get(key), "Task040 Phase-B identity mismatch: " + key)
    require(not identity["missing_keys"] and not identity["unexpected_keys"]
            and identity["classifier_head"]["status"] == "loaded",
            "checkpoint has missing/unexpected keys or classifier is not loaded")
    loader, selected, _ = _loader(ctfrs, config, workers=0)
    require(len(selected) == 9, "exact video list did not select all nine videos")
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    observed = []
    for batch in loader:
        videos, labels, local_indices = batch[:3]
        local = int(local_indices[0].item())
        require(int(videos.shape[2]) == 32, "test transform did not yield exactly T=32")
        observed.append((video_base(str(dataset.clips[local][0])), int(labels[0].item())))
    expected = sorted((str(row["canonical_video_id"]), norm_int(row["label"]))
                      for row in config["manifest"])
    require(sorted(observed) == expected and len(observed) == 9,
            "loaded frames or labels differ from frozen Phase-G N=9 manifest")
    write_json_new(work / "phase_h_preflight.json", {
        "checkpoint_identity": identity, "classifier_output_shape": [1, 400],
        "exact_frozen_unit_identity": True, "manifest_videos": observed,
        "sampled_frames_per_video": 32, "gpu_inference_performed": False,
    })
    del model
    say("Preflight passed: checkpoint, 400-class head, 29 units and exact nine videos.")


def _cuda_device(torch: Any, gpu: int) -> Any:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu),
            "CUDA_VISIBLE_DEVICES must isolate the requested physical GPU")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "each Phase-H process must see exactly one CUDA device")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3407 + gpu)
    torch.cuda.manual_seed_all(3407 + gpu)
    return torch.device("cuda:0")


def _load_cuda_model(config: Mapping[str, Any], gpu: int):
    import torch
    _, phase_g = runtime_modules()
    device = _cuda_device(torch, gpu)
    frozen = {str(row["candidate_task037_global_index"]): row
              for row in config["frozen_units"]}
    probe, ctfrs, model, identity = phase_g.load_model_and_units(
        Path(config["project_root"]), Path(config["checkpoint_path"]), device, frozen
    )
    require(identity["checkpoint_sha256"] == EXPECTED_SHA
            and not identity["missing_keys"] and not identity["unexpected_keys"]
            and identity["classifier_head"]["status"] == "loaded",
            "GPU model differs from the authoritative checkpoint identity")
    model.eval()
    model.requires_grad_(False)
    return torch, probe, ctfrs, phase_g, model, identity, device, frozen


def _intervention_batches(core: Any, clip: Any, interventions: Sequence[Any]):
    result = []
    for start in range(0, len(interventions), 2):
        group = interventions[start:start + 2]
        result.append((group, core.apply_temporal_interventions(clip, group, time_dim=1)))
    require(sum(len(group) for group, _ in result) == 80,
            "intervention batches must contain all 80 fixed conditions")
    return result


def _ce_values(torch: Any, logits: Any, labels: Any) -> list[float]:
    import torch.nn.functional as F
    require(logits.ndim == 2 and int(logits.shape[1]) == 400,
            "full classifier output must have 400 logits")
    require(bool(torch.isfinite(logits).all().item()), "non-finite classifier logits")
    target = labels.to(dtype=torch.long, device=logits.device).reshape(-1)
    require(int(target.shape[0]) == int(logits.shape[0]), "CE target/logit batch mismatch")
    values = F.cross_entropy(logits, target, reduction="none")
    require(bool(torch.isfinite(values).all().item()), "non-finite cross entropy")
    return [float(value) for value in values.detach().cpu().tolist()]


def cache_unmasked(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    import task040_htor_core as core

    work = Path(args.work_dir)
    config = _config(work)
    require((work / "phase_h_preflight.json").is_file(),
            "CPU preflight must pass before GPU inference")
    cache_path = work / "phase_h_unmasked_ce_cache.csv"
    started_path = work / "phase_h_unmasked_started.json"
    done_path = work / "phase_h_unmasked_done.json"
    require(not cache_path.exists() and not started_path.exists() and not done_path.exists(),
            "unmasked inference already started/completed; refusing duplicate baseline inference")
    write_json_new(started_path, {
        "phase": "unmasked_ce_cache", "physical_gpu": 0,
        "started_once": True, "checkpoint_sha256": EXPECTED_SHA,
    })
    torch, probe, ctfrs, phase_g, model, identity, device, _ = _load_cuda_model(config, 0)
    loader, selected, _ = _loader(ctfrs, config, workers=2)
    require(len(selected) == 9, "unmasked loader did not contain all frozen N=9 videos")
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(32)
    phase_g.validate_interventions(
        TASK040_OUT / "n03_fixed_span/task040_intervention_manifest.json"
    )
    require(len(interventions) == 80, "unmasked intervention count differs from Phase G")
    manifest = {str(row["canonical_video_id"]): row for row in config["manifest"]}
    observed, rows = set(), []
    model.eval()
    model.requires_grad_(False)
    say("GPU0 caches full-logit-derived CE once for original and all 80 temporal conditions/video.")
    for batch in tqdm(loader, total=9, desc="Phase H unmasked videos", ncols=110):
        videos, labels, local_indices = batch[:3]
        videos = videos.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        local = int(local_indices[0].item())
        name = video_base(str(dataset.clips[local][0]))
        require(name in manifest and name not in observed,
                "unmasked loader produced unexpected or duplicate video")
        entry = manifest[name]
        label = int(labels[0].item())
        require(label == norm_int(entry["label"]), "unmasked video label differs from manifest")
        require(videos.shape[0] == 1 and int(videos.shape[2]) == 32,
                "unmasked input shape differs from exact T=32 protocol")
        video_index = norm_int(entry["video_index"])
        with torch.inference_mode():
            logits = probe.unwrap_logits(model(videos))
            ce = _ce_values(torch, logits, labels)[0]
            rows.append(_cache_row(video_index, name, label, "original", 0, -1, ce))
            groups = _intervention_batches(core, videos[0], interventions)
            for group, values in tqdm(groups, total=40, desc="unmasked frame pairs",
                                     leave=False, ncols=100):
                outputs = probe.unwrap_logits(model(values))
                ces = _ce_values(torch, outputs, labels.expand(len(group)))
                for intervention, value in zip(group, ces):
                    rows.append(_cache_row(
                        video_index, name, label, "temporal",
                        int(intervention.block_size), int(intervention.pair_index), value,
                    ))
        observed.add(name)
    require(observed == set(manifest) and len(rows) == 9 * 81,
            "unmasked inference failed exact N=9 x 81 cache coverage")
    cache_fields = (
        "video_index", "canonical_video_id", "label", "condition_kind",
        "condition_id", "span", "intervention_level", "pair_index",
        "unmasked_cross_entropy",
    )
    write_csv_new(cache_path, rows, cache_fields)
    write_json_new(done_path, {
        "physical_gpu": 0, "video_count": 9, "unmasked_condition_count": len(rows),
        "computed_once": True, "dtype": "float32", "amp": False,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "cache_sha256": sha256_file(cache_path),
    })
    say("Unmasked CE cache complete: 729 conditions, computed once.")
    del model


def _cache_row(video_index: int, name: str, label: int, kind: str,
               span: int, pair: int, ce: float) -> dict[str, Any]:
    return {
        "video_index": video_index, "canonical_video_id": name, "label": label,
        "condition_kind": kind,
        "condition_id": "original" if kind == "original" else "s%d_p%02d" % (span, pair),
        "span": span, "intervention_level": int(math.log2(span)) if span else -1,
        "pair_index": pair, "unmasked_cross_entropy": ce,
    }


def _hook_snapshot(module: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(key), id(hook)) for key, hook in module._forward_pre_hooks.items())


def _state_versions(model: Any) -> tuple[tuple[int, int], ...]:
    tensors = list(model.parameters()) + list(model.buffers())
    return tuple((id(tensor), int(tensor._version)) for tensor in tensors)


def _cache_key(row: Mapping[str, Any]) -> tuple[int, str, int, int]:
    return (norm_int(row["video_index"]), str(row["condition_kind"]),
            norm_int(row["span"]), norm_int(row["pair_index"]))


def _load_cache(path: Path) -> dict[tuple[int, str, int, int], dict[str, str]]:
    result = {}
    for row in read_csv(path):
        key = _cache_key(row)
        require(key not in result, "duplicate unmasked CE cache key")
        result[key] = row
    require(len(result) == 9 * 81, "unmasked CE cache must have exactly 729 conditions")
    return result


def _unit_objects(probe: Any, ctfrs: Any, phase_g: Any, model: Any,
                  frozen: Mapping[str, Mapping[str, Any]],
                  requested: set[str]) -> dict[str, Any]:
    selected = probe.select_units(
        ctfrs.discover_unit_layers(model), ".*",
        phase_g.candidate_expressions(frozen), 1, ctfrs,
    )
    task040_map = phase_g.frozen_by_task040_index(frozen)
    result = {}
    for unit in selected:
        task040_uid = str(unit.global_index)
        require(task040_uid in task040_map,
                "Task040 discovery returned a unit outside the frozen 29")
        uid, frozen_row = task040_map[task040_uid]
        if uid not in requested:
            continue
        observed = (str(unit.global_index), str(unit.layer_name), str(unit.unit_type),
                    str(unit.unit_index), str(unit.spec.stage))
        expected = (str(frozen_row["candidate_task040_global_index"]),
                    str(frozen_row["candidate_layer_name"]),
                    str(frozen_row["candidate_unit_type"]),
                    str(frozen_row["candidate_unit_index"]),
                    str(frozen_row["candidate_stage"]))
        require(observed == expected, "Task037/Task040 unit identity mismatch: " + uid)
        result[uid] = unit
    require(set(result) == requested, "GPU unit shard differs from frozen identity set")
    return result


def _raw_row(unit: Mapping[str, Any], gpu: int, video_index: int,
             manifest: Mapping[str, Any], kind: str, span: int, pair: int,
             unmasked_ce: float, masked_ce: float) -> dict[str, Any]:
    return {
        "candidate_task037_global_index": unit["candidate_task037_global_index"],
        "candidate_task040_global_index": unit["candidate_task040_global_index"],
        "candidate_layer_name": unit["candidate_layer_name"],
        "candidate_unit_type": unit["candidate_unit_type"],
        "candidate_unit_index": unit["candidate_unit_index"],
        "candidate_stage": unit["candidate_stage"], "domain_id": unit["domain_id"],
        "physical_gpu": gpu, "video_index": video_index,
        "dataset_index": manifest["dataset_index"],
        "canonical_video_id": manifest["canonical_video_id"],
        "label": manifest["label"], "condition_kind": kind,
        "condition_id": ("original" if kind == "original"
                         else "s%d_p%02d" % (span, pair)),
        "span": span, "intervention_level": int(math.log2(span)) if span else -1,
        "pair_index": pair, "unmasked_cross_entropy": unmasked_ce,
        "masked_cross_entropy": masked_ce,
        "signed_ce_damage": masked_ce - unmasked_ce,
        "mask_restored_exact": True,
    }


def worker(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    import task040_htor_core as core

    work = Path(args.work_dir)
    config = _config(work)
    require((work / "phase_h_preflight.json").is_file()
            and (work / "phase_h_unmasked_done.json").is_file(),
            "preflight and the single unmasked cache must finish first")
    gpu = int(args.gpu)
    require(gpu in (0, 1), "only physical GPU 0 and 1 are authorized")
    name = "gpu%d" % gpu
    raw_path = work / ("phase_h_%s_masked_records.csv" % name)
    done_path = work / ("phase_h_%s_done.json" % name)
    require(not raw_path.exists() and not done_path.exists(),
            "candidate shard already started/completed; refusing duplicate inference")
    cache_path = work / "phase_h_unmasked_ce_cache.csv"
    done_cache = json.loads((work / "phase_h_unmasked_done.json").read_text(encoding="utf-8"))
    require(sha256_file(cache_path) == done_cache["cache_sha256"],
            "candidate-independent unmasked CE cache SHA mismatch")
    cache = _load_cache(cache_path)
    torch, probe, ctfrs, phase_g, model, identity, device, frozen = _load_cuda_model(config, gpu)
    all_ids = sorted((str(row["candidate_task037_global_index"])
                      for row in config["frozen_units"]), key=int)
    shard_ids = all_ids[gpu::2]
    units = _unit_objects(probe, ctfrs, phase_g, model, frozen, set(shard_ids))
    loader, selected, _ = _loader(ctfrs, config, workers=2)
    require(len(selected) == 9, "masked worker did not load all nine manifest videos")
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    interventions = core.enumerate_fixed_cardinality_temporal_pairs(32)
    require(len(interventions) == 80, "masked worker intervention count differs from Phase G")
    manifest = {str(row["canonical_video_id"]): row for row in config["manifest"]}
    unit_by_uid = {str(row["candidate_task037_global_index"]): row
                   for row in config["frozen_units"]}
    observed, total_rows = set(), 0
    model.eval()
    model.requires_grad_(False)
    say("%s begins: %d candidates x the exact nine videos; FP32, AMP=False."
        % (name, len(shard_ids)))
    with raw_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for batch in tqdm(loader, total=9, desc="Phase H %s videos" % name, ncols=110):
            videos, labels, local_indices = batch[:3]
            videos = videos.float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            local = int(local_indices[0].item())
            video_name = video_base(str(dataset.clips[local][0]))
            require(video_name in manifest and video_name not in observed,
                    "masked loader returned unexpected or repeated video")
            video = manifest[video_name]
            video_index, label = norm_int(video["video_index"]), int(labels[0].item())
            require(label == norm_int(video["label"]) and videos.shape[0] == 1
                    and int(videos.shape[2]) == 32,
                    "masked video does not match exact manifest/T=32")
            temporal_batches = _intervention_batches(core, videos[0], interventions)
            for uid in tqdm(shard_ids, total=len(shard_ids), desc=name + " unit shard",
                            leave=False, ncols=100):
                unit = units[uid]
                frozen_row = unit_by_uid[uid]
                hooks_before = _hook_snapshot(unit.spec.hook_module)
                versions_before = _state_versions(model)
                rows = []
                with torch.inference_mode():
                    with probe.temporary_unit_mask(unit.spec, unit.unit_index):
                        logits = probe.unwrap_logits(model(videos))
                        masked_ce = _ce_values(torch, logits, labels)[0]
                        key = (video_index, "original", 0, -1)
                        require(key in cache, "cached unmasked original CE is missing")
                        unmasked_ce = finite_float(cache[key]["unmasked_cross_entropy"],
                                                   "cached original CE")
                        rows.append(_raw_row(frozen_row, gpu, video_index, video,
                                             "original", 0, -1, unmasked_ce, masked_ce))
                        count = 0
                        for group, values in tqdm(
                            temporal_batches, total=40, desc=uid + " frame-pair CE",
                            leave=False, ncols=90,
                        ):
                            outputs = probe.unwrap_logits(model(values))
                            ce_values = _ce_values(torch, outputs, labels.expand(len(group)))
                            for intervention, value in zip(group, ce_values):
                                span, pair = int(intervention.block_size), int(intervention.pair_index)
                                ckey = (video_index, "temporal", span, pair)
                                require(ckey in cache, "cached unmasked temporal CE is missing")
                                baseline_ce = finite_float(
                                    cache[ckey]["unmasked_cross_entropy"], "cached temporal CE")
                                rows.append(_raw_row(frozen_row, gpu, video_index, video,
                                                     "temporal", span, pair,
                                                     baseline_ce, value))
                                count += 1
                        require(count == 80, "masked candidate lacks a temporal intervention")
                # temporary_unit_mask is the validated Task040 whole-unit input
                # hook: its finally block removes the exact hook. Verify that
                # hook table and all model parameter/buffer versions are unchanged.
                restored = (
                    _hook_snapshot(unit.spec.hook_module) == hooks_before
                    and _state_versions(model) == versions_before
                )
                require(restored, "temporary whole-unit mask was not restored: " + uid)
                for row in rows:
                    row["mask_restored_exact"] = True
                writer.writerows(rows)
                handle.flush()
                total_rows += len(rows)
            observed.add(video_name)
    expected_rows = len(shard_ids) * 9 * 81
    require(observed == set(manifest) and total_rows == expected_rows,
            "masked candidate shard has incomplete videos/records")
    write_json_new(done_path, {
        "worker": name, "physical_gpu": gpu, "candidate_count": len(shard_ids),
        "candidate_ids": shard_ids, "video_count": 9, "row_count": total_rows,
        "all_masks_restored_exact": True, "dtype": "float32", "amp": False,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "unmasked_cache_reused": True,
    })
    say("%s COMPLETE: %d signed CE records; exact temporary-mask restoration verified."
        % (name, total_rows))
    del model


def _row_score(domain: str, members: Sequence[str], scores: Mapping[str, float],
               damage: Mapping[str, float], method: str) -> dict[str, Any]:
    ids = sorted(members, key=int)
    x, y = [float(scores[uid]) for uid in ids], [float(damage[uid]) for uid in ids]
    rho, tau = spearman(x, y), kendall_tau_b(x, y)
    low_uid, high_uid = ordered_extremes({uid: scores[uid] for uid in ids})
    safe, truth = safest_uid({uid: scores[uid] for uid in ids}), fullval_safest_uid(
        {uid: damage[uid] for uid in ids}
    )
    low_damage, high_damage = float(damage[low_uid]), float(damage[high_uid])
    return {
        "row_type": "domain", "domain_id": domain,
        "scope": "same_type" if domain in SAME_TYPE else "mixed",
        "method": method, "unit_count": len(ids),
        "spearman": rho, "kendall_tau_b": tau,
        "safest_candidate_task037_global_index": safe,
        "fullval_safest_task037_global_index": truth,
        "safest_identity_match": safe == truth,
        "low_risk_task037_global_index": low_uid,
        "high_risk_task037_global_index": high_uid,
        "low_risk_fullval_ce": low_damage, "high_risk_fullval_ce": high_damage,
        "high_minus_low_fullval_ce": high_damage - low_damage,
        "low_high_ordering": ordering_label(low_damage, high_damage),
    }


def _baseline_domain_row(domain: str, members: Sequence[str],
                         scores: Mapping[str, float | None],
                         damage: Mapping[str, float], method: str) -> dict[str, Any]:
    """Compare a frozen baseline only where it was already recorded."""
    available = [uid for uid in members if scores.get(uid) is not None]
    if len(available) < 2:
        return {
            "row_type": "domain", "domain_id": domain,
            "scope": "same_type" if domain in SAME_TYPE else "mixed",
            "method": method, "unit_count": len(available),
            "missing_baseline_count": len(members) - len(available),
            "spearman": None, "kendall_tau_b": None,
            "safest_candidate_task037_global_index": None,
            "fullval_safest_task037_global_index": fullval_safest_uid(
                {uid: damage[uid] for uid in members}),
            "safest_identity_match": None,
            "low_risk_task037_global_index": None,
            "high_risk_task037_global_index": None,
            "low_risk_fullval_ce": None, "high_risk_fullval_ce": None,
            "high_minus_low_fullval_ce": None, "low_high_ordering": "unavailable",
        }
    return {
        **_row_score(domain, available,
                     {uid: float(scores[uid]) for uid in available}, damage, method),
        "missing_baseline_count": len(members) - len(available),
    }


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "candidate_task037_global_index", "candidate_task040_global_index",
        "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
        "candidate_stage", "domain_id",
    )}


def _summary_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "spearman": average(row["spearman"] for row in rows),
        "kendall": average(row["kendall_tau_b"] for row in rows),
        "safest_accuracy": average(
            1.0 if row["safest_identity_match"] else 0.0 for row in rows
        ),
        "low_high_correct": sum(row["low_high_ordering"] == "correct" for row in rows),
        "low_high_reverse": sum(row["low_high_ordering"] == "reverse" for row in rows),
        "low_high_tie": sum(row["low_high_ordering"] == "tie" for row in rows),
        "domain_count": len(rows),
    }


def finalize(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    config = _config(work)
    output = Path(config["output_dir"])
    require(not output.exists(), "Phase-H output already exists; refusing overwrite")
    for marker in ("phase_h_unmasked_done.json", "phase_h_gpu0_done.json",
                   "phase_h_gpu1_done.json"):
        require((work / marker).is_file(), "missing completed worker marker " + marker)
    cache_path = work / "phase_h_unmasked_ce_cache.csv"
    cache_done = json.loads((work / "phase_h_unmasked_done.json").read_text(encoding="utf-8"))
    require(cache_done.get("computed_once") is True
            and norm_int(cache_done.get("physical_gpu", -1)) == 0
            and cache_done.get("dtype") == "float32"
            and cache_done.get("amp") is False
            and cache_done.get("checkpoint_sha256") == EXPECTED_SHA
            and sha256_file(cache_path) == cache_done.get("cache_sha256"),
            "single unmasked CE cache is not intact")
    cache: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for row in read_csv(cache_path):
        key = _cache_key(row)
        require(key not in cache, "duplicate unmasked CE condition")
        cache[key] = row
    require(len(cache) == 9 * 81, "unmasked CE cache must contain exactly 729 conditions")

    config_units = {
        str(row["candidate_task037_global_index"]): row
        for row in config["frozen_units"]
    }
    domains = {uid: str(row["domain_id"]) for uid, row in config_units.items()}
    manifest = {norm_int(row["video_index"]): row for row in config["manifest"]}
    raw, worker_ids = [], []
    all_unit_ids = sorted(config_units, key=int)
    for gpu, name in ((0, "gpu0"), (1, "gpu1")):
        done = json.loads((work / ("phase_h_%s_done.json" % name)).read_text(encoding="utf-8"))
        require(done.get("all_masks_restored_exact") is True,
                name + " did not verify exact temporary-mask restoration")
        require(norm_int(done.get("physical_gpu", -1)) == gpu,
                name + " completion marker has the wrong physical GPU")
        require(done.get("checkpoint_sha256") == EXPECTED_SHA
                and done.get("dtype") == "float32" and done.get("amp") is False,
                name + " checkpoint/dtype/AMP identity differs from the frozen protocol")
        ids = [str(uid) for uid in done["candidate_ids"]]
        require(ids == all_unit_ids[gpu::2],
                name + " candidate list differs from the deterministic alternating shard")
        require(norm_int(done.get("candidate_count", -1)) == len(ids),
                name + " completion marker candidate count is inconsistent")
        require(set(ids).isdisjoint(worker_ids), "GPU candidate shards overlap")
        worker_ids.extend(ids)
        rows = read_csv(work / ("phase_h_%s_masked_records.csv" % name))
        require(len(rows) == norm_int(done["row_count"]),
                name + " row count differs from done marker")
        require(all(norm_int(row["physical_gpu"]) == gpu
                    and str(norm_int(row["candidate_task037_global_index"])) in ids
                    for row in rows),
                name + " raw records do not match the declared GPU/unit shard")
        raw.extend(rows)
    require(set(worker_ids) == set(config_units) and len(worker_ids) == 29,
            "GPU0/GPU1 do not cover exactly the frozen 29 candidates")
    require(len(raw) == 29 * 9 * 81, "masked CE records must total 29*9*81")

    original: dict[str, dict[int, float]] = {uid: {} for uid in config_units}
    temporal: dict[str, dict[tuple[int, int, int], float]] = {
        uid: {} for uid in config_units
    }
    seen = set()
    output_records = []
    for row in raw:
        uid = str(norm_int(row["candidate_task037_global_index"]))
        require(uid in config_units, "raw CE record is outside the frozen 29 units")
        identity = config_units[uid]
        for key in (
            "candidate_task040_global_index", "candidate_layer_name",
            "candidate_unit_type", "candidate_unit_index", "candidate_stage", "domain_id",
        ):
            require(str(row[key]) == str(identity[key]),
                    "raw CE unit identity mismatch for %s (%s)" % (uid, key))
        require(str(row["mask_restored_exact"]).strip().lower() == "true",
                "raw CE row lacks exact mask restoration flag")
        video_index = norm_int(row["video_index"])
        kind, span, pair = str(row["condition_kind"]), norm_int(row["span"]), norm_int(row["pair_index"])
        key = (video_index, kind, span, pair)
        record_key = (uid,) + key
        require(record_key not in seen, "duplicate candidate-condition CE record")
        seen.add(record_key)
        require(key in cache, "candidate CE record lacks unique unmasked cache row")
        base_row = cache[key]
        require(str(row["canonical_video_id"]) == str(manifest[video_index]["canonical_video_id"])
                and norm_int(row["label"]) == norm_int(manifest[video_index]["label"]),
                "raw CE video/label differs from frozen Phase-G manifest")
        require(norm_int(row["dataset_index"]) == norm_int(manifest[video_index]["dataset_index"]),
                "raw CE dataset index differs from frozen Phase-G manifest")
        expected_condition_id = (
            "original" if kind == "original" else "s%d_p%02d" % (span, pair)
        )
        expected_level = int(math.log2(span)) if span else -1
        require(str(row["condition_id"]) == expected_condition_id
                and norm_int(row["intervention_level"]) == expected_level,
                "raw CE condition identity differs from the frozen intervention")
        unmasked = finite_float(base_row["unmasked_cross_entropy"], "cached CE")
        recorded_unmasked = finite_float(row["unmasked_cross_entropy"], "recorded unmasked CE")
        masked = finite_float(row["masked_cross_entropy"], "masked CE")
        delta = finite_float(row["signed_ce_damage"], "signed CE damage")
        require(math.isclose(unmasked, recorded_unmasked, rel_tol=0.0, abs_tol=1e-12),
                "candidate row did not reuse the cached unmasked CE")
        require(math.isclose(delta, masked - unmasked, rel_tol=0.0, abs_tol=1e-12),
                "signed CE damage is not masked CE minus unmasked CE")
        if kind == "original":
            require((span, pair) == (0, -1), "original CE condition has temporal indices")
            original[uid][video_index] = delta
        else:
            require(kind == "temporal" and span in SPANS and 0 <= pair < 16,
                    "invalid temporal CE condition")
            temporal[uid][(video_index, span, pair)] = delta
        output_records.append(row)
    require(len(seen) == 29 * 9 * 81, "masked CE identity set is incomplete")
    for uid in config_units:
        require(len(original[uid]) == 9 and len(temporal[uid]) == 9 * 80,
                "unit has incomplete original/temporal CE grid: " + uid)

    original_values = {
        uid: {("original", video): damage for video, damage in by_video.items()}
        for uid, by_video in original.items()
    }
    temporal_values = {
        uid: {("temporal", video, span, pair): damage
              for (video, span, pair), damage in conditions.items()}
        for uid, conditions in temporal.items()
    }
    W_original = group_win_rates(original_values, domains)
    W_temporal = group_win_rates(temporal_values, domains)
    W_span = {
        span: group_win_rates(
            temporal_values, domains,
            condition_predicate=lambda key, s=span:
                len(key) == 4 and key[0] == "temporal" and key[2] == s,
        )
        for span in SPANS
    }
    fullval = {uid: float(row["fullval_mean_cross_entropy_increase"])
               for uid, row in config_units.items()}
    uids = sorted(config_units, key=int)
    unit_rows, span_rows = [], []
    span_safest: dict[str, dict[str, str]] = defaultdict(dict)
    for uid in uids:
        unit = config_units[uid]
        per_span = {str(span): W_span[span][uid] for span in SPANS}
        unit_rows.append({
            **_identity(unit),
            "W_original": W_original[uid], "W_temporal_N9": W_temporal[uid],
            "R_original": 1.0 - W_original[uid],
            "R_temporal": 1.0 - W_temporal[uid],
            "fullval_mean_cross_entropy_increase": fullval[uid],
            "fullval_n_samples": 3783,
            "per_span_W_json": json.dumps(per_span, sort_keys=True),
            **{"W_span_%d" % span: W_span[span][uid] for span in SPANS},
        })
        for span in SPANS:
            span_rows.append({
                **_identity(unit), "span": span,
                "W_temporal_span": W_span[span][uid],
                "R_temporal_span": 1.0 - W_span[span][uid],
                "fullval_mean_cross_entropy_increase": fullval[uid],
                "video_count": 9, "interventions_per_video": 16,
            })

    per_domain = {}
    domain_method_rows = []
    same_type_rows = []
    detailed_cases = []
    for domain in DOMAINS:
        members = [uid for uid in uids if domains[uid] == domain]
        damage = {uid: fullval[uid] for uid in members}
        temporal_risk = {uid: 1.0 - W_temporal[uid] for uid in members}
        original_risk = {uid: 1.0 - W_original[uid] for uid in members}
        trow = _row_score(domain, members, temporal_risk, damage, "R_temporal")
        orow = _row_score(domain, members, original_risk, damage, "R_original")
        domain_method_rows.extend((orow, trow))
        temporal_safe = safest_uid({uid: W_temporal[uid] for uid in members})
        original_safe = safest_uid({uid: W_original[uid] for uid in members})
        actual_safe = fullval_safest_uid(damage)
        span_safe = {}
        for span in SPANS:
            span_safe[str(span)] = safest_uid({uid: W_span[span][uid] for uid in members})
        details = []
        for role, uid in (("original_safest", original_safe),
                          ("temporal_safest", temporal_safe),
                          ("fullval_safest", actual_safe)):
            details.append({
                "role": role, **_identity(config_units[uid]),
                "W_original": W_original[uid], "W_temporal": W_temporal[uid],
                "per_span_W": {str(span): W_span[span][uid] for span in SPANS},
                "fullval_mean_cross_entropy_increase": fullval[uid],
            })
        correction = original_safe != actual_safe and temporal_safe == actual_safe
        regression = original_safe == actual_safe and temporal_safe != actual_safe
        row = {
            "row_type": "domain", "domain_id": domain,
            "scope": "same_type" if domain in SAME_TYPE else "mixed",
            "unit_count": len(members),
            "temporal_spearman": trow["spearman"],
            "temporal_kendall_tau_b": trow["kendall_tau_b"],
            "temporal_safest_task037_global_index": temporal_safe,
            "fullval_safest_task037_global_index": actual_safe,
            "temporal_safest_identity_match": temporal_safe == actual_safe,
            "original_safest_task037_global_index": original_safe,
            "original_safest_identity_match": original_safe == actual_safe,
            "original_spearman": orow["spearman"],
            "original_kendall_tau_b": orow["kendall_tau_b"],
            "temporal_low_risk_uid": trow["low_risk_task037_global_index"],
            "temporal_high_risk_uid": trow["high_risk_task037_global_index"],
            "temporal_low_high_ordering": trow["low_high_ordering"],
            "temporal_high_minus_low_fullval_ce": trow["high_minus_low_fullval_ce"],
            "original_low_high_ordering": orow["low_high_ordering"],
            "original_high_minus_low_fullval_ce": orow["high_minus_low_fullval_ce"],
            "per_span_safest_uid_json": json.dumps(span_safe, sort_keys=True),
            "distinct_per_span_safest_candidate_count": len(set(span_safe.values())),
            "temporal_correction_case": correction,
            "temporal_regression_case": regression,
            "candidate_details_json": json.dumps(details, sort_keys=True),
        }
        if domain in SAME_TYPE:
            same_type_rows.append(row)
        if correction or regression:
            detailed_cases.append(row)
        per_domain[domain] = {
            "temporal": trow, "original": orow,
            "temporal_safest_uid": temporal_safe,
            "original_safest_uid": original_safe,
            "fullval_safest_uid": actual_safe,
            "correction_case": correction, "regression_case": regression,
            "span_safest_uids": span_safe,
            "span_safest_changed": len(set(span_safe.values())) > 1,
        }

    same_t = [row for row in domain_method_rows
              if row["scope"] == "same_type" and row["method"] == "R_temporal"]
    same_o = [row for row in domain_method_rows
              if row["scope"] == "same_type" and row["method"] == "R_original"]
    temporal_summary = _summary_for(same_t)
    original_summary = _summary_for(same_o)
    decision, gate = temporal_decision_gate(temporal_summary, original_summary)
    same_type_rows.append({
        "row_type": "domain_balanced", "domain_id": "ALL", "scope": "same_type",
        "unit_count": sum(int(row["unit_count"]) for row in same_t),
        "temporal_spearman": temporal_summary["spearman"],
        "temporal_kendall_tau_b": temporal_summary["kendall"],
        "temporal_safest_identity_accuracy": temporal_summary["safest_accuracy"],
        "temporal_safest_correct_count": sum(bool(row["safest_identity_match"]) for row in same_t),
        "temporal_low_high_correct_count": temporal_summary["low_high_correct"],
        "temporal_low_high_reverse_count": temporal_summary["low_high_reverse"],
        "temporal_low_high_tie_count": temporal_summary["low_high_tie"],
        "domain_count": temporal_summary["domain_count"],
    })

    original_vs_temporal = [
        {"row_type": "domain", **row} for row in domain_method_rows
    ]
    for method, summary in (("R_original", original_summary),
                            ("R_temporal", temporal_summary)):
        original_vs_temporal.append({
            "row_type": "domain_balanced", "scope": "same_type",
            "domain_id": "ALL", "method": method,
            "spearman": summary["spearman"], "kendall_tau_b": summary["kendall"],
            "safest_identity_accuracy": summary["safest_accuracy"],
            "low_high_correct": summary["low_high_correct"],
            "low_high_reverse": summary["low_high_reverse"],
            "low_high_tie": summary["low_high_tie"],
            "domain_count": summary["domain_count"],
        })
    for row in detailed_cases:
        original_vs_temporal.append({
            "row_type": ("temporal_correction_case" if row["temporal_correction_case"]
                         else "temporal_regression_case"),
            **row,
        })

    # Mixed-domain candidate/type summaries are descriptive only.
    mixed_rows = []
    for domain in MIXED:
        members = [uid for uid in uids if domains[uid] == domain]
        for uid in members:
            mixed_rows.append({
                "row_type": "candidate", "domain_id": domain, **_identity(config_units[uid]),
                "W_original": W_original[uid], "W_temporal": W_temporal[uid],
                "fullval_mean_cross_entropy_increase": fullval[uid],
                **{"W_span_%d" % span: W_span[span][uid] for span in SPANS},
            })
        for unit_type in ("head", "neuron"):
            typed = [uid for uid in members
                     if str(config_units[uid]["candidate_unit_type"]) == unit_type]
            if typed:
                mixed_rows.append({
                    "row_type": "domain_type_mean", "domain_id": domain,
                    "candidate_unit_type": unit_type, "candidate_count": len(typed),
                    "mean_W_original": average(W_original[uid] for uid in typed),
                    "mean_W_temporal": average(W_temporal[uid] for uid in typed),
                    **{"mean_W_span_%d" % span:
                       average(W_span[span][uid] for uid in typed) for span in SPANS},
                    "mean_fullval_mean_cross_entropy_increase":
                        average(fullval[uid] for uid in typed),
                })

    # Reuse exactly the Phase-H records for the Phase-F N=3 subset and all 27
    # class-balanced N=6 subsets; there is no further inference.
    subset_defs = [
        {"subset_id": "n3_phase_f",
         "video_indices": config["phase_f_n3_video_indices"]}
    ] + list(config["n6_subsets"]) + [
        {"subset_id": "n9_full", "video_indices": list(range(9))}
    ]
    subset_rows = []
    for subset in subset_defs:
        selected_videos = {norm_int(v) for v in subset["video_indices"]}
        selected_values = {
            uid: {
                ("temporal", video, span, pair): value
                for (video, span, pair), value in temporal[uid].items()
                if video in selected_videos
            }
            for uid in uids
        }
        subset_w = group_win_rates(selected_values, domains)
        for domain in DOMAINS:
            members = [uid for uid in uids if domains[uid] == domain]
            risk_subset = {uid: 1.0 - subset_w[uid] for uid in members}
            risk_n9 = {uid: 1.0 - W_temporal[uid] for uid in members}
            ce = {uid: fullval[uid] for uid in members}
            safe_subset = safest_uid({uid: subset_w[uid] for uid in members})
            safe_n9 = safest_uid({uid: W_temporal[uid] for uid in members})
            safe_ce = fullval_safest_uid(ce)
            subset_rows.append({
                "subset_id": subset["subset_id"], "video_count": len(selected_videos),
                "video_indices_json": json.dumps(sorted(selected_videos)),
                "domain_id": domain,
                "scope": "same_type" if domain in SAME_TYPE else "mixed",
                "unit_count": len(members),
                "spearman_risk_subset_vs_n9": spearman(
                    [risk_subset[uid] for uid in members],
                    [risk_n9[uid] for uid in members]),
                "kendall_risk_subset_vs_n9": kendall_tau_b(
                    [risk_subset[uid] for uid in members],
                    [risk_n9[uid] for uid in members]),
                "spearman_risk_subset_vs_fullval_ce": spearman(
                    [risk_subset[uid] for uid in members],
                    [ce[uid] for uid in members]),
                "safest_uid_subset": safe_subset, "safest_uid_n9": safe_n9,
                "safest_uid_fullval": safe_ce,
                "safest_identity_match_n9": safe_subset == safe_n9,
                "safest_identity_match_fullval": safe_subset == safe_ce,
            })

    # Frozen baselines retain their original values and are compared only to
    # the same existing full-validation CE oracle, within each BMS domain.
    score_maps: dict[str, dict[str, float | None]] = {
        "R_original": original_risk,
        "R_temporal": {uid: 1.0 - W_temporal[uid] for uid in uids},
        "R_BCTR_N9": {uid: float(config_units[uid]["R_BCTR_N9"]) for uid in uids},
    }
    for criterion, _field in BASELINE_FIELDS.items():
        score_maps[criterion] = {
            uid: config_units[uid]["baseline_scores"][criterion] for uid in uids
        }
    baseline_rows = []
    for scope, selected_domains in (("same_type", SAME_TYPE), ("mixed", MIXED)):
        for criterion, scores in score_maps.items():
            group = []
            for domain in selected_domains:
                members = [uid for uid in uids if domains[uid] == domain]
                row = _baseline_domain_row(
                    domain, members, scores,
                    {uid: fullval[uid] for uid in members}, criterion,
                )
                row["scope"] = scope
                baseline_rows.append(row)
                if row["spearman"] is not None:
                    group.append(row)
            baseline_rows.append({
                "row_type": "domain_balanced", "scope": scope,
                "domain_id": "ALL", "method": criterion,
                "unit_count": sum(len([u for u in uids if domains[u] == d])
                                  for d in selected_domains),
                "spearman": average(row["spearman"] for row in group),
                "kendall_tau_b": average(row["kendall_tau_b"] for row in group),
                "safest_identity_accuracy": average(
                    1.0 if row["safest_identity_match"] else 0.0 for row in group),
                "low_high_correct": sum(row["low_high_ordering"] == "correct" for row in group),
                "low_high_reverse": sum(row["low_high_ordering"] == "reverse" for row in group),
                "low_high_tie": sum(row["low_high_ordering"] == "tie" for row in group),
                "domain_count": len(selected_domains),
                "valid_domain_count": len(group),
            })

    subset_summary = {}
    for subset_id in ("n3_phase_f", "n9_full"):
        rows = [row for row in subset_rows
                if row["subset_id"] == subset_id and row["scope"] == "same_type"]
        subset_summary[subset_id] = {
            "same_type_domain_count": len(rows),
            "mean_spearman_risk_vs_n9": average(
                row["spearman_risk_subset_vs_n9"] for row in rows),
            "safest_identity_matches_n9": sum(
                row["safest_identity_match_n9"] for row in rows),
            "safest_identity_matches_fullval": sum(
                row["safest_identity_match_fullval"] for row in rows),
        }
    n6_rows = [row for row in subset_rows
               if row["subset_id"].startswith("n6_") and row["scope"] == "same_type"]
    subset_summary["n6_all_27_class_balanced_subsets"] = {
        "domain_subset_rows": len(n6_rows),
        "mean_spearman_risk_vs_n9": average(
            row["spearman_risk_subset_vs_n9"] for row in n6_rows),
        "safest_identity_matches_n9": sum(row["safest_identity_match_n9"] for row in n6_rows),
        "safest_identity_matches_fullval": sum(
            row["safest_identity_match_fullval"] for row in n6_rows),
    }
    correction_domains = [d for d in SAME_TYPE if per_domain[d]["correction_case"]]
    regression_domains = [d for d in SAME_TYPE if per_domain[d]["regression_case"]]
    span_change_domains = [d for d in SAME_TYPE
                           if per_domain[d]["span_safest_changed"]]
    span_safest_by_domain = {
        d: per_domain[d]["span_safest_uids"] for d in DOMAINS
    }
    summary = {
        "task": "Task041 Phase H post-BMS frame-relation temporal stress ranking diagnosis",
        "decision": decision,
        "phase_f_decision_preserved": "RETAINED FOR N=9 VALIDATION",
        "phase_g_decision_preserved": "BCTR_REMAINS_WEAK_OR_UNRESOLVED",
        "branch": BRANCH,
        "base_head_before_phase_h": BASE_HEAD,
        "code_head": git_value(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        "checkpoint_path": config["checkpoint_path"], "checkpoint_sha256": EXPECTED_SHA,
        "dtype": "float32", "amp": False, "gpu_ids_used": [0, 1],
        "frozen_unit_count": 29, "domains": list(DOMAINS),
        "n9_video_count": 9, "phase_f_n3_video_indices": config["phase_f_n3_video_indices"],
        "n6_class_balanced_subset_count": 27,
        "temporal_conditions_per_video": 80,
        "unmasked_ce_conditions_computed_once": 9 * 81,
        "masked_ce_record_count": len(raw),
        "signed_ce_damage": "masked_cross_entropy - unmasked_cross_entropy",
        "absolute_or_squared_damage_used": False,
        "full_validation_oracle_rerun": False,
        "mask_restoration_exact_for_every_candidate_video": all(
            row["mask_restored_exact"].strip().lower() == "true" for row in raw),
        "same_type_temporal_vs_fullval_ce": temporal_summary,
        "same_type_original_vs_fullval_ce": original_summary,
        "temporal_per_domain_spearman": {
            row["domain_id"]: row["spearman"] for row in same_t
        },
        "temporal_per_domain_kendall_tau_b": {
            row["domain_id"]: row["kendall_tau_b"] for row in same_t
        },
        "original_per_domain_spearman": {
            row["domain_id"]: row["spearman"] for row in same_o
        },
        "original_per_domain_kendall_tau_b": {
            row["domain_id"]: row["kendall_tau_b"] for row in same_o
        },
        "temporal_correction_domains": correction_domains,
        "temporal_regression_domains": regression_domains,
        "same_type_domains_where_safest_changes_across_spans": span_change_domains,
        "same_type_subset_stability": subset_summary,
        "decision_gate": gate,
        "new_success_threshold_added": False,
        "task042_created": False, "pruning_or_physical_removal_performed": False,
        "finetuning_performed": False,
    }
    report = _report(summary, same_type_rows, span_safest_by_domain,
                     detailed_cases, subset_summary, baseline_rows)

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv_new(output / OUTPUT_FILES[0], output_records, RAW_FIELDS)
    write_csv_new(output / OUTPUT_FILES[1], unit_rows)
    write_csv_new(output / OUTPUT_FILES[2], span_rows)
    write_csv_new(output / OUTPUT_FILES[3], original_vs_temporal)
    write_csv_new(output / OUTPUT_FILES[4], same_type_rows)
    write_csv_new(output / OUTPUT_FILES[5], mixed_rows)
    write_csv_new(output / OUTPUT_FILES[6], subset_rows)
    write_csv_new(output / OUTPUT_FILES[7], baseline_rows)
    write_json_new(output / OUTPUT_FILES[8], summary)
    with (output / OUTPUT_FILES[9]).open("x", encoding="utf-8") as handle:
        handle.write(report)
    require({path.name for path in output.iterdir()} == set(OUTPUT_FILES),
            "Phase-H output must contain exactly the ten required artifacts")
    say("FINAL: %s; output=%s" % (decision, output))


def _report(summary: Mapping[str, Any],
            same_type_rows: Sequence[Mapping[str, Any]],
            span_safest: Mapping[str, Mapping[str, str]],
            detail_cases: Sequence[Mapping[str, Any]],
            subset_summary: Mapping[str, Any],
            baseline_rows: Sequence[Mapping[str, Any]]) -> str:
    temporal = summary["same_type_temporal_vs_fullval_ce"]
    original = summary["same_type_original_vs_fullval_ce"]
    lines = [
        "# Task041 Phase H — Post-BMS Frame-Relation Temporal Stress Diagnosis",
        "",
        "## Decision",
        "",
        "**%s**" % summary["decision"],
        "",
        "Phase F remains RETAINED FOR N=9 VALIDATION; the later Phase-G result remains "
        "BCTR_REMAINS_WEAK_OR_UNRESOLVED. Phase H ranks actual signed CE deletion damage "
        "under fixed frame-pair stress conditions; BCTR is not the selector.",
        "",
        "## Frozen protocol and integrity",
        "",
        "- Video Swin / UCF101; checkpoint SHA256 %s; authoritative 400-output classifier."
        % summary["checkpoint_sha256"],
        "- FP32, AMP=False; GPU 0/1 independent single-device workers.",
        "- Exact Phase-G manifest: 9 videos, 3 classes x 3 videos; exact 29 frozen Task041 units.",
        "- T=32; spans 1/2/4/8/16; 16 deterministic two-single-frame swaps per span.",
        "- Candidate-independent unmasked CE cache: 729 conditions computed once, then reused.",
        "- Masked CE records: %d; signed damage = masked CE - unmasked CE; negative values retained."
        % summary["masked_ce_record_count"],
        "- Exact temporary-mask restoration was verified after every candidate/video. The full-validation oracle was reused, not rerun.",
        "- No pruning, physical removal, persistent mask, finetuning, or Task042.",
        "",
        "## Primary same-type result: temporal risk vs full-validation CE",
        "",
        "| Domain | Spearman | Kendall tau-b | Temporal safest | Full-val safest | Match |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in same_type_rows:
        if row.get("row_type") != "domain":
            continue
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            row["domain_id"], row.get("temporal_spearman"),
            row.get("temporal_kendall_tau_b"),
            row.get("temporal_safest_task037_global_index"),
            row.get("fullval_safest_task037_global_index"),
            "yes" if row.get("temporal_safest_identity_match") else "no",
        ))
    lines.extend([
        "",
        "Domain-balanced Spearman: **%s**; Kendall tau-b: **%s**; safest identity accuracy: **%s**."
        % (temporal["spearman"], temporal["kendall"], temporal["safest_accuracy"]),
        "Low/high CE ordering correct/reverse/tie: **%d/%d/%d**."
        % (temporal["low_high_correct"], temporal["low_high_reverse"], temporal["low_high_tie"]),
        "",
        "## Central ablation: original-only vs temporal stress",
        "",
        "| Method | Spearman | Kendall tau-b | Safest identity accuracy | low/high correct/reverse/tie |",
        "|---|---:|---:|---:|---:|",
        "| Original-only | %s | %s | %s | %d/%d/%d |"
        % (original["spearman"], original["kendall"], original["safest_accuracy"],
           original["low_high_correct"], original["low_high_reverse"], original["low_high_tie"]),
        "| Temporal stress | %s | %s | %s | %d/%d/%d |"
        % (temporal["spearman"], temporal["kendall"], temporal["safest_accuracy"],
           temporal["low_high_correct"], temporal["low_high_reverse"], temporal["low_high_tie"]),
        "",
        "The predeclared A decision requires positive temporal domain-balanced CE association and improvement "
        "over original-only on at least one primary measure while not being worse on all remaining measures. "
        "No post-hoc success threshold was introduced; the gate details are in the summary JSON.",
        "",
        "## Temporal correction / regression cases",
        "",
    ])
    if detail_cases:
        for row in detail_cases:
            kind = ("correction" if row["temporal_correction_case"] else "regression")
            lines.append("- %s in domain %s: original safest %s, temporal safest %s, full-validation safest %s. "
                         % (
                             kind, row["domain_id"],
                             row["original_safest_task037_global_index"],
                             row["temporal_safest_task037_global_index"],
                             row["fullval_safest_task037_global_index"]))
            for candidate in json.loads(str(row["candidate_details_json"])):
                span_text = ", ".join(
                    "span%s=%.6f" % (span, candidate["per_span_W"][str(span)])
                    for span in SPANS
                )
                lines.append(
                    "  - %s: unit %s (%s, %s, stage %s), W_original=%.6f, "
                    "W_temporal=%.6f, %s, full-validation CE damage=%.9g."
                    % (candidate["role"], candidate["candidate_task037_global_index"],
                       candidate["candidate_layer_name"], candidate["candidate_unit_type"],
                       candidate["candidate_stage"], candidate["W_original"],
                       candidate["W_temporal"], span_text,
                       candidate["fullval_mean_cross_entropy_increase"])
                )
    else:
        lines.append("No same-type correction or regression cases occurred.")
    lines.extend([
        "",
        "## Span-specific safest candidate",
        "",
        "| Domain | span 1 | span 2 | span 4 | span 8 | span 16 | distinct IDs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for domain in SAME_TYPE:
        ids = [span_safest[domain][str(span)] for span in SPANS]
        lines.append("| %s | %s | %s | %s | %s | %s | %d |"
                     % (domain, *ids, len(set(ids))))
    lines.extend([
        "",
        "## N=3 / N=6 / N=9 stability",
        "",
        "N=3 is the exact Phase-F subset. All 27 class-balanced N=6 subsets were tested from the frozen N=9 "
        "manifest. The same CE records were reused; no additional inference occurred.",
        "",
        "| subset | same-type rows | mean Spearman vs N=9 | safest matches N=9 | safest matches full-val |",
        "|---|---:|---:|---:|---:|",
    ])
    for subset, row in subset_summary.items():
        count = row.get("same_type_domain_count", row.get("domain_subset_rows"))
        rho = row.get("mean_spearman_risk_vs_n9")
        lines.append("| %s | %s | %s | %s | %s |" % (
            subset, count, rho, row["safest_identity_matches_n9"],
            row["safest_identity_matches_fullval"]))
    lines.extend([
        "",
        "## Mixed domains 271 / 297",
        "",
        "Candidate head/neuron W_temporal and per-span rates are reported descriptively in the mixed-domain CSV; "
        "they do not determine the primary gate.",
        "",
        "## Frozen baseline comparison",
        "",
        "All historical criteria were compared within frozen domains against the same existing full-validation CE "
        "oracle. No baseline was modified.",
        "",
        "| Criterion | scope | units / valid domains | mean Spearman | mean Kendall | safest accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in baseline_rows:
        if row.get("row_type") == "domain_balanced":
            lines.append("| %s | %s | %s / %s | %s | %s | %s |" % (
                row.get("method"), row.get("scope"), row.get("unit_count"),
                row.get("valid_domain_count"), row.get("spearman"),
                row.get("kendall_tau_b"), row.get("safest_identity_accuracy")))
    lines.extend([
        "",
        "## Scope stop",
        "",
        "This is a diagnosis only. It does not authorize pruning, 50% sparsity, physical removal, finetuning, "
        "or automatic selector redesign. The frozen descriptors, BMS and earlier Phase-F/G conclusions remain unchanged.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task041 Phase-H temporal CE stress diagnosis")
    parser.add_argument("phase", choices=("prepare", "preflight", "cache_unmasked",
                                         "worker", "finalize"))
    parser.add_argument("--project-root", default=PROJECT_DEFAULT)
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--work-dir", default=WORK_DEFAULT)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.phase == "worker":
        require(args.gpu is not None, "worker requires --gpu 0 or --gpu 1")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "preflight":
        preflight(args)
    elif args.phase == "cache_unmasked":
        cache_unmasked(args)
    elif args.phase == "worker":
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()

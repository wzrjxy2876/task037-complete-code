"""Task020 task-level importance diagnosis from the exact Task019 28% state.

This module is analysis-only.  It never changes a pruning score, selector,
registry application rule, validation function, descriptor, BMS domain,
Contribution Field, parameter cost, or minimum-retention constraint.

Every individual intervention starts from the same exact 28% keep-index state.
The intervention is undone immediately after its fixed-sample forward pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SEED = 3407
CODE_VERSION = "task020_fixed28_task_importance_v3"
DIAGNOSTIC_CLASSES = 10
VIDEOS_PER_CLASS = 10
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
PRODUCTION_FILES = (
    "functional_competition_pruning.py",
    "MC.py",
    "ucf101_videoswin_my.py",
)
TASK019_COMMIT = "3c053f11fe6e344a5e7b60875ebe195cd30a1eee"
EXPECTED_TASK019_PRODUCTION_BLOBS = {
    "functional_competition_pruning.py": "608f3ddb1b2f56d6967b2da257ca63d6d5b6f8e1",
    "MC.py": "d67f05e19f8bfa4757fd46dcf006c4711e3d0240",
    "ucf101_videoswin_my.py": "7afbba9391ec01c1481f6ea3e3876d41e512345b",
}
GROUP_ATTENTION = "attention_counterfactual_heads"
GROUP_REPLACEMENT = "replacement_ffn"
GROUP_ORDINARY = "ordinary_high_sparsity_deleted_ffn"
GROUP_CONTROL = "matched_surviving_ffn_control"
METRIC_BASES = ("true_logit_drop", "margin_drop", "ce_increase", "kl")
METRIC_FIELDS = (
    "mean_true_logit_drop", "median_true_logit_drop",
    "mean_margin_drop", "median_margin_drop",
    "mean_ce_increase", "median_ce_increase",
    "mean_kl", "median_kl",
    "prediction_flip_rate",
    "correct_to_wrong_flip_rate",
    "top1_probability_drop",
    "top5_probability_mass_drop",
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[int]) -> str:
    array = np.asarray([int(value) for value in values], dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def production_source_sha256(repo_root: Path) -> dict[str, str]:
    return {name: sha256_file(Path(repo_root) / name) for name in PRODUCTION_FILES}


def git_blob_sha(ref: str, path: str, repo_root: Path | None = None) -> str:
    """Return the committed Git blob ID for ``ref:path``."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    return subprocess.check_output(
        ["git", "rev-parse", f"{ref}:{path}"], cwd=root, text=True,
    ).strip()


def git_worktree_clean_blob_sha(path: str, repo_root: Path | None = None) -> str:
    """Hash working-tree bytes using Git's path-specific clean-filter rules."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    return subprocess.check_output(
        ["git", "hash-object", f"--path={path}", path], cwd=root, text=True,
    ).strip()


def git_diff_clean(repo_root: Path | None = None, *, cached: bool = False) -> bool:
    """Check that protected paths have no unstaged or staged changes."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    command = ["git", "diff", "--quiet"]
    if cached:
        command.append("--cached")
    command.extend(["HEAD", "--", *PRODUCTION_FILES])
    return subprocess.run(command, cwd=root, check=False).returncode == 0


def git_committed_diff_clean(
    repo_root: Path | None = None, *, from_ref: str = TASK019_COMMIT, to_ref: str = "HEAD",
) -> bool:
    """Check that protected paths are unchanged between two committed refs."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    command = ["git", "diff", "--quiet", from_ref, to_ref, "--", *PRODUCTION_FILES]
    return subprocess.run(command, cwd=root, check=False).returncode == 0


def production_source_git_identity(
    repo_root: Path | None = None,
    *, task019_commit: str = TASK019_COMMIT,
) -> dict[str, dict[str, str]]:
    """Require Task019 blob, HEAD blob and filtered worktree blob equality."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    result: dict[str, dict[str, str]] = {}
    for path in PRODUCTION_FILES:
        task019_blob = git_blob_sha(task019_commit, path, root)
        head_blob = git_blob_sha("HEAD", path, root)
        worktree_blob = git_worktree_clean_blob_sha(path, root)
        if not (task019_blob == head_blob == worktree_blob):
            raise RuntimeError(
                f"Protected production Git identity mismatch for {path}: "
                f"Task019={task019_blob}, HEAD={head_blob}, worktree={worktree_blob}"
            )
        expected = EXPECTED_TASK019_PRODUCTION_BLOBS.get(path)
        if expected is not None and task019_blob != expected:
            raise RuntimeError(
                f"Unexpected Task019 protected blob for {path}: {task019_blob} != {expected}"
            )
        result[path] = {
            "task019_blob_sha": task019_blob,
            "head_blob_sha": head_blob,
            "worktree_blob_sha": worktree_blob,
        }
    if not git_diff_clean(root) or not git_diff_clean(root, cached=True):
        raise RuntimeError("Protected production files have staged or unstaged changes")
    if not git_committed_diff_clean(root, from_ref=task019_commit, to_ref="HEAD"):
        raise RuntimeError("Protected production files differ between Task019 and HEAD")
    return result


def production_source_git_identity_matches(
    identities: Mapping[str, Mapping[str, str]],
) -> bool:
    """Pure predicate used by regression tests for simulated identity changes."""
    return set(identities) == set(PRODUCTION_FILES) and all(
        values.get("task019_blob_sha") == values.get("head_blob_sha") == values.get("worktree_blob_sha")
        for values in identities.values()
    )


DEPENDENCY_FIELDS = (
    "artifact_identity_sha256", "cohort_sha256", "sample_identity_sha256",
    "baseline_output_sha256", "task020_experiment_identity_sha256", "code_version",
)


def experiment_dependency_fields(output_dir: Path) -> dict[str, str]:
    root = Path(output_dir)
    payload = {
        "artifact_identity_sha256": sha256_file(root / "artifact_identity.json"),
        "cohort_sha256": sha256_file(root / "task020_cohorts.csv"),
        "sample_identity_sha256": sha256_file(root / "diagnostic_samples.csv"),
        "baseline_output_sha256": sha256_file(root / "baseline_28_outputs.npz"),
        "code_version": CODE_VERSION,
    }
    payload["task020_experiment_identity_sha256"] = canonical_json_sha256(payload)
    return payload


def cache_identity_matches(cache: Mapping[str, object], expected: Mapping[str, str]) -> bool:
    return cache.get("status") == "PASS" and all(
        str(cache.get(name, "")) == str(expected[name]) for name in DEPENDENCY_FIELDS
    )


def baseline_cache_valid(output_dir: Path) -> bool:
    root = Path(output_dir)
    required = ("artifact_identity.json", "task020_cohorts.csv", "diagnostic_samples.csv",
                "baseline_28_outputs.npz", "baseline_28_cache.json")
    if any(not (root / name).is_file() for name in required):
        return False
    try:
        return cache_identity_matches(read_json(root / "baseline_28_cache.json"),
                                      experiment_dependency_fields(root))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def result_cache_valid(path: Path, output_dir: Path, **required: object) -> bool:
    path = Path(path)
    if not path.is_file() or not baseline_cache_valid(output_dir):
        return False
    try:
        value = read_json(path)
        return cache_identity_matches(value, experiment_dependency_fields(output_dir)) and all(
            value.get(name) == expected for name, expected in required.items()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _as_int(row: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return int(float(value))
    raise KeyError(names[0])


def _stage(layer: str) -> int:
    parts = str(layer).split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        return int(parts[1])
    raise ValueError(f"Cannot infer Video Swin stage from {layer!r}")


def _block(layer: str) -> int:
    parts = str(layer).split(".")
    if len(parts) >= 4 and parts[0] == "layers" and parts[2] == "blocks":
        return int(parts[3])
    raise ValueError(f"Cannot infer Video Swin block from {layer!r}")


def derive_counterfactual_sets(task019_root: Path) -> tuple[list[int], list[int]]:
    """Read exact Attention and replacement-FFN sets; never infer by count."""
    root = Path(task019_root)
    causal = read_csv(root / "causal_selection_trace.csv")
    attention = [
        _as_int(row, "global_index") for row in causal
        if str(row.get("variant")) == "dynamic_28_30"
        and str(row.get("unit_type")) == TYPE_ATTENTION
    ]
    replacement_rows = read_csv(root / "attention_replacement_trace.csv")
    replacements = [
        _as_int(row, "selected_ffn_global_index") for row in replacement_rows
    ]
    dynamic_indices = {
        _as_int(row, "global_index") for row in causal
        if str(row.get("variant")) == "dynamic_28_30"
    }
    counterfactual_rows = [
        row for row in causal
        if str(row.get("variant")) == "no_new_attention_dynamic_30"
    ]
    if any(str(row.get("unit_type")) != TYPE_FFN for row in counterfactual_rows):
        raise RuntimeError("Task019 no-new-Attention trace contains an Attention unit")
    reconstructed = [
        _as_int(row, "global_index") for row in counterfactual_rows
        if _as_int(row, "global_index") not in dynamic_indices
    ]
    if len(attention) != 3 or len(set(attention)) != 3:
        raise RuntimeError(
            f"Task019 counterfactual must identify the recorded exact three heads, got {len(attention)}"
        )
    if not replacements or len(replacements) != len(set(replacements)):
        raise RuntimeError("Task019 replacement FFN set is empty or duplicated")
    if reconstructed != replacements:
        raise RuntimeError("Task019 replacement table differs from causal trace reconstruction")
    return attention, replacements


def derive_original_dynamic_ffn(task019_root: Path, replacements: Sequence[int]) -> list[int]:
    """Return exact FFNs removed by the original dynamic 28%->30% interval."""
    replacement_set = {int(value) for value in replacements}
    rows = read_csv(Path(task019_root) / "causal_selection_trace.csv")
    values = [
        _as_int(row, "global_index") for row in rows
        if str(row.get("variant")) == "dynamic_28_30"
        and str(row.get("unit_type")) == TYPE_FFN
        and _as_int(row, "global_index") not in replacement_set
    ]
    if not values or len(values) != len(set(values)):
        raise RuntimeError("Original dynamic 28%-30% FFN set is empty or duplicated")
    return values


def deterministic_stratified_indices(
    labels: Sequence[int],
    *,
    classes: int = DIAGNOSTIC_CLASSES,
    videos_per_class: int = VIDEOS_PER_CLASS,
    seed: int = SEED,
) -> list[int]:
    """Choose a reproducible balanced validation subset.

    The input axis is dataset index ``[N]``.  The output is a sorted list of
    ``classes * videos_per_class`` distinct indices.
    """
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    eligible = sorted(label for label, values in grouped.items() if len(values) >= videos_per_class)
    if len(eligible) < classes:
        raise RuntimeError(
            f"Only {len(eligible)} classes contain {videos_per_class} validation videos"
        )
    rng = random.Random(seed)
    chosen_classes = sorted(rng.sample(eligible, classes))
    chosen: list[int] = []
    for label in chosen_classes:
        local = list(grouped[label])
        rng.shuffle(local)
        chosen.extend(sorted(local[:videos_per_class]))
    if len(chosen) != len(set(chosen)):
        raise RuntimeError("Diagnostic subset contains duplicate dataset indices")
    return sorted(chosen)


def task_metrics(
    baseline_logits: np.ndarray,
    ablated_logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute per-unit task effects on aligned ``[N,C]`` logits."""
    base = np.asarray(baseline_logits, dtype=np.float64)
    ablated = np.asarray(ablated_logits, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int64)
    if base.shape != ablated.shape or base.ndim != 2:
        raise ValueError("Baseline and ablated logits must have equal shape [N,C]")
    if target.shape != (base.shape[0],):
        raise ValueError("Targets must have shape [N]")
    if not np.isfinite(base).all() or not np.isfinite(ablated).all():
        raise ValueError("Task020 logits must be finite")
    rows = np.arange(base.shape[0])
    true_base = base[rows, target]
    true_ablated = ablated[rows, target]
    masked_base = base.copy()
    masked_ablated = ablated.copy()
    masked_base[rows, target] = -np.inf
    masked_ablated[rows, target] = -np.inf
    margin_base = true_base - masked_base.max(axis=1)
    margin_ablated = true_ablated - masked_ablated.max(axis=1)

    def log_softmax(values: np.ndarray) -> np.ndarray:
        maximum = values.max(axis=1, keepdims=True)
        shifted = values - maximum
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    logp_base, logp_ablated = log_softmax(base), log_softmax(ablated)
    p_base, p_ablated = np.exp(logp_base), np.exp(logp_ablated)
    pred_base, pred_ablated = base.argmax(axis=1), ablated.argmax(axis=1)
    base_correct = pred_base == target
    k = min(5, base.shape[1])
    top5 = np.argpartition(base, -k, axis=1)[:, -k:]
    ablated_top5 = np.argpartition(ablated, -k, axis=1)[:, -k:]
    top5_base = np.take_along_axis(p_base, top5, axis=1).sum(axis=1)
    top5_ablated = np.take_along_axis(p_ablated, top5, axis=1).sum(axis=1)
    sample_values = {
        "true_logit_drop": true_base - true_ablated,
        "margin_drop": margin_base - margin_ablated,
        "ce_increase": -logp_ablated[rows, target] + logp_base[rows, target],
        "kl": (p_base * (logp_base - logp_ablated)).sum(axis=1),
        "prediction_flip": pred_base != pred_ablated,
        "correct_to_wrong_flip": base_correct & (pred_ablated != target),
        "top1_probability_drop": p_base[rows, pred_base] - p_ablated[rows, pred_base],
        "top5_probability_mass_drop": top5_base - top5_ablated,
    }
    summary = {
        **{
            f"{stat}_{name}": float(getattr(np, stat)(sample_values[name]))
            for name in METRIC_BASES for stat in ("mean", "median")
        },
        "prediction_flip_rate": float(sample_values["prediction_flip"].mean()),
        "correct_to_wrong_flip_rate": float(sample_values["correct_to_wrong_flip"].mean()),
        "top1_probability_drop": float(sample_values["top1_probability_drop"].mean()),
        "top5_probability_mass_drop": float(sample_values["top5_probability_mass_drop"].mean()),
        "baseline_top1": float((pred_base == target).mean()),
        "ablated_top1": float((pred_ablated == target).mean()),
        "baseline_top5": float(np.any(top5 == target[:, None], axis=1).mean()),
        "ablated_top5": float(np.any(ablated_top5 == target[:, None], axis=1).mean()),
    }
    per_sample = []
    for offset in range(base.shape[0]):
        per_sample.append(
            {
                "sample_offset": offset,
                "target": int(target[offset]),
                "base_prediction": int(pred_base[offset]),
                "ablated_prediction": int(pred_ablated[offset]),
                **{
                    key: (bool(value[offset]) if value.dtype == np.bool_ else float(value[offset]))
                    for key, value in sample_values.items()
                },
            }
        )
    return summary, per_sample


def parameter_normalized_metrics(metrics: Mapping[str, float], parameter_cost: int) -> dict[str, float]:
    """Return diagnostic task effects per exact structural parameter cost."""
    cost = int(parameter_cost)
    if cost <= 0:
        raise ValueError("parameter_cost must be positive")
    return {
        "true_logit_drop_per_parameter": float(metrics["mean_true_logit_drop"]) / cost,
        "margin_drop_per_parameter": float(metrics["mean_margin_drop"]) / cost,
        "ce_increase_per_parameter": float(metrics["mean_ce_increase"]) / cost,
        "kl_per_parameter": float(metrics["mean_kl"]) / cost,
    }


def summed_parameter_cost(rows: Sequence[Mapping[str, object]]) -> int:
    """Sum exact artifact parameter costs without estimating unit granularity."""
    return sum(_as_int(row, "parameter_cost") for row in rows)


def validate_ffn_control_sets(
    replacement: Iterable[int],
    ordinary_deleted: Iterable[int],
    matched_surviving: Iterable[int],
    original_dynamic_interval: Iterable[int],
    removed_by_30: Iterable[int],
) -> None:
    """Enforce the scientific source and disjointness definitions of C/D."""
    groups = [
        {int(value) for value in replacement},
        {int(value) for value in ordinary_deleted},
        {int(value) for value in matched_surviving},
    ]
    if any(groups[left] & groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Task020 FFN diagnostic groups overlap")
    if not groups[1].issubset({int(value) for value in original_dynamic_interval}):
        raise RuntimeError("Ordinary deleted FFN controls are not from original dynamic 28%-30%")
    if groups[2] & {int(value) for value in removed_by_30}:
        raise RuntimeError("Matched surviving FFN control does not survive original dynamic 30%")


def expected_worker_global_indices(
    cohorts: Sequence[Mapping[str, object]], worker: int, workers: int
) -> list[int]:
    if workers <= 0 or not 0 <= worker < workers:
        raise ValueError("worker must lie in [0, workers)")
    values = [_as_int(row, "global_index") for offset, row in enumerate(cohorts)
              if offset % workers == worker]
    if len(values) != len(set(values)):
        raise RuntimeError("Task020 worker assignment contains duplicate global indices")
    return values


def validate_attention_sample_counts(
    rows: Sequence[Mapping[str, object]], attention_indices: Iterable[int], sample_count: int
) -> None:
    expected = {int(value): int(sample_count) for value in attention_indices}
    observed = {index: 0 for index in expected}
    for row in rows:
        index = _as_int(row, "global_index")
        if index not in observed:
            raise RuntimeError(f"Unexpected Attention per-sample output: {index}")
        observed[index] += 1
    if observed != expected:
        raise RuntimeError(f"Attention per-sample row counts differ: {observed} != {expected}")


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _rankdata(left), _rankdata(right)
    if x.size != y.size or x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def deterministic_control_match(
    replacements: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """One-to-one deterministic nearest matching without a tunable weight.

    Matching is lexicographic: exact layer, exact stage, absolute cost difference,
    standardized descriptor/energy/domain-size distance, then global index.
    """
    numeric = ("D_abs", "D_rel", "D_dyn", "functional_energy", "domain_size")
    pool = [dict(row) for row in candidates]
    if len(pool) < len(replacements):
        raise RuntimeError("Not enough eligible FFNs for deterministic one-to-one matching")
    combined = list(replacements) + pool
    matrix = np.asarray([[float(row[name]) for name in numeric] for row in combined])
    scale = np.std(matrix, axis=0)
    scale[scale == 0] = 1.0
    matched = []
    used: set[int] = set()
    for source in replacements:
        def key(candidate: Mapping[str, object]):
            index = int(candidate["global_index"])
            delta = np.asarray(
                [float(candidate[name]) - float(source[name]) for name in numeric]
            ) / scale
            return (
                str(candidate["layer"]) != str(source["layer"]),
                _stage(str(candidate["layer"])) != _stage(str(source["layer"])),
                abs(float(candidate["parameter_cost"]) - float(source["parameter_cost"])),
                float(np.linalg.norm(delta)),
                float(candidate.get("functional_rank_at_0", math.inf)),
                index,
            )
        available = [row for row in pool if int(row["global_index"]) not in used]
        chosen = min(available, key=key)
        used.add(int(chosen["global_index"]))
        matched.append({**chosen, "matched_to_global_index": int(source["global_index"])})
    return matched


@dataclass(frozen=True)
class UnitSpec:
    cohort: str
    global_index: int
    layer: str
    unit_type: str
    unit_index: int
    domain_id: int


@contextmanager
def additional_unit_ablation(model, unit: UnitSpec):
    modules = dict(model.named_modules())
    if unit.layer not in modules:
        raise KeyError(f"Missing Task020 layer: {unit.layer}")
    module = modules[unit.layer]
    if unit.unit_type == TYPE_ATTENTION:
        attribute, universe_size = "keep_heads", int(module.num_heads)
    elif unit.unit_type == TYPE_FFN:
        attribute, universe_size = "keep_neurons", int(module.original_hidden_features)
    else:
        raise ValueError(f"Unknown Task020 unit type: {unit.unit_type}")
    original = getattr(module, attribute)
    before = list(range(universe_size)) if original is None else list(original)
    if unit.unit_index not in before:
        raise RuntimeError(
            f"Task020 unit {unit.global_index} is not retained at the exact 28% state"
        )
    setattr(module, attribute, [value for value in before if int(value) != unit.unit_index])
    try:
        yield
    finally:
        setattr(module, attribute, original)


def verify_identity(
    *,
    task014_root: Path,
    task015_root: Path,
    task016_root: Path,
    task017_root: Path,
    task018_root: Path,
    task019_root: Path,
    output_dir: Path,
) -> None:
    import task019_dynamic_ranking_causal_ablation as task019

    roots = [Path(value) for value in (task014_root, task015_root, task016_root, task017_root, task018_root, task019_root)]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Task020 input roots: {missing}")
    completion = read_json(Path(task019_root) / "task019_completion.json")
    required = (
        "artifact_identity_pass",
        "dynamic_28_30_reproduction_pass",
        "frozen_26_validation_complete",
        "frozen_30_validation_complete",
        "no_new_attention_30_validation_complete",
        "budget_semantics_unchanged",
        "analysis_complete",
    )
    if completion.get("status") != "PASS" or not all(completion.get(key) is True for key in required):
        raise RuntimeError("Task019 completion gate is incomplete")
    if completion.get("production_pruning_code_modified") is not False:
        raise RuntimeError("Task019 reports modified production pruning code")
    completion15 = read_json(Path(task015_root) / "task015_completion.json")
    if completion15.get("status") != "passed" or completion15.get("replay_selected_indices_exact") is not True:
        raise RuntimeError("Task015 completion/replay identity is incomplete")
    energy_path = Path(task015_root) / "functional_energy.npy"
    energy_identity_path = Path(task015_root) / "functional_energy.json"
    energy_identity = read_json(energy_identity_path)
    energy = np.load(energy_path, mmap_mode="r", allow_pickle=False)
    if energy.shape != (EXPECTED_UNITS,) or energy.dtype != np.float32 or not np.isfinite(energy).all():
        raise RuntimeError("Task015 functional-energy artifact is incompatible")
    del energy
    from task015_attention_ffn_diagnosis import _mapping_paths
    _, _, mapping_audit_path = _mapping_paths(Path(task014_root))
    mapping_audit = read_json(mapping_audit_path)
    if energy_identity.get("mapping_sha256") != mapping_audit.get("mapping_sha256"):
        raise RuntimeError("Task015 Contribution Field mapping differs from Task014")
    identity19 = read_json(Path(task019_root) / "artifact_identity.json")
    if identity19.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task019 artifact identity did not pass")
    if int(identity19.get("descriptor_units", -1)) != EXPECTED_UNITS:
        raise RuntimeError("Task019 unit mapping is not the expected 36,378 units")
    if int(identity19.get("bms_domains", -1)) != EXPECTED_DOMAINS:
        raise RuntimeError("Task019 BMS-domain identity is not the expected 423 domains")
    repo_root = Path(__file__).resolve().parent
    source_commit_identity = production_source_git_identity(repo_root)
    current_hashes = production_source_sha256(repo_root)
    checkpoint = Path(str(identity19["checkpoint"]))
    stat = checkpoint.stat()
    if stat.st_size != int(identity19["checkpoint_size_bytes"]) or stat.st_mtime_ns != int(identity19["checkpoint_mtime_ns"]):
        raise RuntimeError("Checkpoint size/timestamp differs from Task019 identity")
    if sha256_file(checkpoint) != str(identity19["checkpoint_sha256"]):
        raise RuntimeError("Checkpoint SHA256 differs from Task019 identity")
    attention, replacements = derive_counterfactual_sets(Path(task019_root))
    registry28 = read_json(Path(task018_root) / "registries/s28.json")
    expected_prefix_sha = identity19["task018_prefix_sequence_sha256"]["s28"]
    if registry28.get("summary", {}).get("sequence_prefix_sha256") != expected_prefix_sha:
        raise RuntimeError("Task020 exact 28% prefix identity failed")
    evidence_paths = (
        Path(task019_root) / "attention_counterfactual.csv",
        Path(task019_root) / "attention_replacement_trace.csv",
        Path(task019_root) / "causal_selection_trace.csv",
        Path(task019_root) / "task019_completion.json",
        Path(task018_root) / "registries/s28.json",
        Path(task015_root) / "task015_completion.json",
        energy_identity_path,
        energy_path,
        mapping_audit_path,
    )
    payload = {
        "status": "PASS",
        "code_version": CODE_VERSION,
        "artifact_identity_pass": True,
        "task019_completion_pass": True,
        "checkpoint": identity19["checkpoint"],
        "checkpoint_sha256": identity19["checkpoint_sha256"],
        "checkpoint_size_bytes": identity19["checkpoint_size_bytes"],
        "checkpoint_mtime_ns": identity19["checkpoint_mtime_ns"],
        "parameters_before": identity19["parameters_before"],
        "validation_split": identity19["validation_split"],
        "validation_batch_size": identity19["validation_batch_size"],
        "amp_enabled": identity19["amp_enabled"],
        "descriptor_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS,
        "attention_global_indices": attention,
        "replacement_ffn_global_indices": replacements,
        "attention_sequence_sha256": sequence_sha256(attention),
        "replacement_ffn_sequence_sha256": sequence_sha256(replacements),
        "task018_s28_sequence_sha256": expected_prefix_sha,
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in evidence_paths},
        "production_source_sha256": current_hashes,
        "production_source_git_blob_sha": source_commit_identity,
        "production_pruning_code_modified": False,
        "diagnostic_classes": DIAGNOSTIC_CLASSES,
        "videos_per_class": VIDEOS_PER_CLASS,
        "seed": SEED,
        "task014_root": str(Path(task014_root).resolve()),
        "task015_root": str(Path(task015_root).resolve()),
        "task016_root": str(Path(task016_root).resolve()),
        "task017_root": str(Path(task017_root).resolve()),
        "task018_root": str(Path(task018_root).resolve()),
        "task019_root": str(Path(task019_root).resolve()),
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)


def _load_fresh_28_model(identity: Mapping[str, object]):
    import torch
    from ucf101_videoswin_my import SwinTransformer3D, set_seed
    from task018_high_sparsity_transition import _apply_registry

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each Task020 worker must see exactly one physical GPU as cuda:0")
    torch.cuda.set_device(0)
    set_seed(SEED)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.0,
        qkv_bias=True, patch_norm=True, drop_path_rate=0.2, use_checkpoint=True,
    ).to(torch.device("cuda:0"))
    checkpoint = Path(str(identity["checkpoint"]))
    stat = checkpoint.stat()
    if stat.st_size != int(identity["checkpoint_size_bytes"]) or stat.st_mtime_ns != int(identity["checkpoint_mtime_ns"]):
        raise RuntimeError("Task020 checkpoint changed after identity gate")
    payload = torch.load(checkpoint, map_location="cuda:0")
    state = payload.get("state_dict", payload)
    normalized = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
    model.load_state_dict(normalized, strict=False)
    registry28 = read_json(Path(str(identity["task018_root"])) / "registries/s28.json")
    _apply_registry(model, registry28["registry"])
    model.eval()
    return model


def _load_model_and_loader(identity: Mapping[str, object], sample_indices: Sequence[int] | None = None):
    from torch.utils.data import DataLoader, Subset
    from ucf101_videoswin_my import get_dataset

    model = _load_fresh_28_model(identity)
    full_loader = get_dataset(str(identity["validation_split"]), int(identity["validation_batch_size"]))
    dataset = full_loader.dataset
    if sample_indices is None:
        labels = [int(value[2]) for value in dataset.clips]
        sample_indices = deterministic_stratified_indices(labels)
    subset = Subset(dataset, [int(value) for value in sample_indices])
    loader = DataLoader(
        subset,
        batch_size=int(identity["validation_batch_size"]),
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return model, loader, dataset


def _forward_logits(model, loader, amp_enabled: bool):
    import torch

    logits, targets, indices = [], [], []
    with torch.inference_mode():
        for inputs, target, index in loader:
            inputs = inputs.float().to("cuda:0", non_blocking=True)
            target = target.to("cuda:0", non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
                output, _ = model(inputs)
            logits.append(output.float().cpu().numpy())
            targets.append(target.cpu().numpy())
            indices.append(np.asarray(index))
    return (
        np.concatenate(logits, axis=0),
        np.concatenate(targets, axis=0).astype(np.int64),
        np.concatenate(indices, axis=0).astype(np.int64),
    )


def _materialize_batches(loader):
    """Decode once and retain fixed ``[B,C,T,H,W]`` batches on cuda:0.

    The input axes are batch, RGB channel, time, height and width.  Keeping the
    tensors on the worker's 24GB GPU removes repeated decoding and host-to-GPU
    transfer from every unit ablation.  The cache is process-local.
    """
    return [
        (
            inputs.float().contiguous().to("cuda:0", non_blocking=True),
            target.contiguous().to("cuda:0", non_blocking=True),
            np.asarray(index),
        )
        for inputs, target, index in loader
    ]


def _verify_worker_baseline(
    model, batches, amp_enabled: bool, expected_logits, expected_targets, expected_indices
) -> None:
    logits, targets, indices = _forward_logits(model, batches, amp_enabled)
    if not np.array_equal(targets, expected_targets) or not np.array_equal(indices, expected_indices):
        raise RuntimeError("Task020 worker baseline sample identity drifted")
    if not np.allclose(logits, expected_logits, rtol=1e-5, atol=1e-5):
        maximum = float(np.max(np.abs(logits - expected_logits)))
        raise RuntimeError(f"Task020 worker baseline logits differ from cache; max_abs={maximum}")


def cache_baseline(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Run Task020 identity first")
    model, loader, dataset = _load_model_and_loader(identity)
    logits, targets, indices = _forward_logits(model, loader, bool(identity["amp_enabled"]))
    expected = deterministic_stratified_indices([int(value[2]) for value in dataset.clips])
    if indices.tolist() != expected:
        raise RuntimeError("Task020 diagnostic sample order changed")
    sample_rows = [
        {
            "sample_offset": offset,
            "dataset_index": int(index),
            "target": int(targets[offset]),
            "clip_path": str(dataset.clips[int(index)][0]),
        }
        for offset, index in enumerate(indices)
    ]
    atomic_csv(output_dir / "diagnostic_samples.csv", tuple(sample_rows[0]), sample_rows)
    sample_sha = sha256_file(output_dir / "diagnostic_samples.csv")
    baseline_path = output_dir / "baseline_28_outputs.npz"
    temporary = baseline_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, logits=logits.astype(np.float32),
                            targets=targets, sample_indices=indices)
    temporary.replace(baseline_path)
    dependencies = experiment_dependency_fields(output_dir)
    atomic_json(
        output_dir / "baseline_28_cache.json",
        {
            "status": "PASS", "code_version": CODE_VERSION, "baseline_28_cached": True,
            "sample_count": len(indices), "sample_identity_sha256": sample_sha,
            "baseline_prediction_accuracy": float((logits.argmax(axis=1) == targets).mean()),
            **dependencies,
        },
    )


def _unit_metadata(identity: Mapping[str, object]) -> tuple[dict[int, dict], dict[int, int]]:
    from task015_attention_ffn_diagnosis import (
        _descriptor_path, _load_descriptors, _load_units, _mapping_paths,
        infer_parameter_costs,
    )
    from task018_high_sparsity_transition import _load_snapshot

    task014_root = Path(str(identity["task014_root"]))
    task015_root = Path(str(identity["task015_root"]))
    task016_root = Path(str(identity["task016_root"]))
    task017_root = Path(str(identity["task017_root"]))
    unit_path, layer_path, _ = _mapping_paths(task014_root)
    units = _load_units(unit_path)
    costs, _ = infer_parameter_costs(units, read_csv(layer_path))
    descriptor_path = _descriptor_path(task014_root, task016_root / "domain_total/s30")
    descriptors = _load_descriptors(descriptor_path, units)
    energy = np.load(task015_root / "functional_energy.npy", mmap_mode="r", allow_pickle=False)
    snapshot_path = task017_root / "replay/domain_total/candidate_snapshots.npz"
    snapshot0 = _load_snapshot(snapshot_path, 0.0)
    snapshot28 = _load_snapshot(snapshot_path, 0.28)
    domain_by_global = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
    domain_by_global[snapshot0["global_index"]] = snapshot0["domain_id"]
    sizes = np.bincount(domain_by_global, minlength=EXPECTED_DOMAINS)
    rank0_order = np.argsort(snapshot0["delta_total"], kind="mergesort")
    rank0 = {
        int(snapshot0["global_index"][offset]): rank + 1
        for rank, offset in enumerate(rank0_order)
    }
    rank28_order = np.lexsort((snapshot28["global_index"], snapshot28["delta_total"]))
    rank28 = {
        int(snapshot28["global_index"][offset]): rank + 1
        for rank, offset in enumerate(rank28_order)
    }
    p28 = {
        int(index): {
            "delta_average_at_28": float(snapshot28["delta_average"][offset]),
            "delta_total_at_28": float(snapshot28["delta_total"][offset]),
            "functional_rank_at_28": int(rank28[int(index)]),
            "best_substitute_similarity_at_28": float(snapshot28["best_similarity"][offset]),
            "domain_coverage_at_28": float(snapshot28["coverage"][offset]),
            "domain_valid_demand_count": int(snapshot28["active_demand_count"][offset]),
        }
        for offset, index in enumerate(snapshot28["global_index"])
    }
    metadata = {}
    for unit in units:
        index = int(unit.global_index)
        domain = int(domain_by_global[index])
        metadata[index] = {
            "global_index": index, "layer": unit.layer,
            "unit_type": unit.unit_type, "unit_index": int(unit.unit_index),
            "stage": _stage(unit.layer), "block": _block(unit.layer),
            "domain_id": domain, "parameter_cost": int(costs[index]),
            "D_abs": float(descriptors[index, 0]), "D_rel": float(descriptors[index, 1]),
            "D_dyn": float(descriptors[index, 2]),
            "functional_energy": float(energy[index]), "domain_size": int(sizes[domain]),
            "functional_rank_at_0": int(rank0[index]),
            **p28.get(index, {}),
        }
    return metadata, {index: int(costs[index]) for index in range(EXPECTED_UNITS)}


def prepare_cohorts(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    metadata, _ = _unit_metadata(identity)
    attention = [int(value) for value in identity["attention_global_indices"]]
    replacement = [int(value) for value in identity["replacement_ffn_global_indices"]]
    registry28 = read_json(Path(str(identity["task018_root"])) / "registries/s28.json")["registry"]
    removed28 = set()
    for layer, entry in registry28.items():
        unit_type = str(entry["unit_type"])
        indices = {int(value) for value in entry["indices"]}
        removed28.update(
            index for index, row in metadata.items()
            if row["layer"] == layer and row["unit_type"] == unit_type and int(row["unit_index"]) in indices
        )
    original_interval = {
        _as_int(row, "global_index")
        for row in read_csv(Path(str(identity["task019_root"])) / "causal_selection_trace.csv")
        if str(row.get("variant")) == "dynamic_28_30"
    }
    removed30 = original_interval | removed28
    replacement_rows = [metadata[index] for index in replacement]
    ordinary_indices = derive_original_dynamic_ffn(Path(str(identity["task019_root"])), replacement)
    ordinary_pool = [metadata[index] for index in ordinary_indices]
    if any("delta_total_at_28" not in row for row in ordinary_pool):
        raise RuntimeError("Original dynamic 28%-30% FFN is absent from the exact 28% state")
    ordinary = deterministic_control_match(replacement_rows, ordinary_pool)
    surviving_pool = [
        row for index, row in metadata.items()
        if row["unit_type"] == TYPE_FFN and index not in removed30
        and index not in replacement and "delta_total_at_28" in row
    ]
    surviving = deterministic_control_match(replacement_rows, surviving_pool)
    cohort_rows = []
    for cohort, indices in ((GROUP_ATTENTION, attention), (GROUP_REPLACEMENT, replacement)):
        cohort_rows.extend({"cohort": cohort, **metadata[index], "matched_to_global_index": ""} for index in indices)
    cohort_rows.extend({"cohort": GROUP_ORDINARY, **row} for row in ordinary)
    cohort_rows.extend({"cohort": GROUP_CONTROL, **row} for row in surviving)
    if any(int(row["global_index"]) in removed28 for row in cohort_rows):
        raise RuntimeError("Task020 cohort contains a unit absent from the exact 28% state")
    group_sets = {
        group: {int(row["global_index"]) for row in cohort_rows if row["cohort"] == group}
        for group in (GROUP_REPLACEMENT, GROUP_ORDINARY, GROUP_CONTROL)
    }
    validate_ffn_control_sets(
        group_sets[GROUP_REPLACEMENT], group_sets[GROUP_ORDINARY],
        group_sets[GROUP_CONTROL], original_interval, removed30,
    )
    atomic_csv(output_dir / "task020_cohorts.csv", tuple(cohort_rows[0]), cohort_rows)

    def matching_diagnostics(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        by_source = {int(row["matched_to_global_index"]): row for row in values}
        result = []
        for source in replacement_rows:
            control = by_source[int(source["global_index"])]
            result.append({
                "replacement_global_index": int(source["global_index"]),
                "control_global_index": int(control["global_index"]),
                "stage_match": int(source["stage"]) == int(control["stage"]),
                "layer_match": str(source["layer"]) == str(control["layer"]),
                "parameter_cost_difference": int(control["parameter_cost"]) - int(source["parameter_cost"]),
                **{f"{name}_difference": float(control[name]) - float(source[name])
                   for name in ("D_abs", "D_rel", "D_dyn", "functional_energy", "domain_size")},
            })
        return result

    ordinary_diagnostics = matching_diagnostics(ordinary)
    surviving_diagnostics = matching_diagnostics(surviving)
    atomic_csv(output_dir / "ordinary_deleted_ffn_matching_diagnostics.csv",
               tuple(ordinary_diagnostics[0]), ordinary_diagnostics)
    atomic_csv(output_dir / "matched_surviving_ffn_matching_diagnostics.csv",
               tuple(surviving_diagnostics[0]), surviving_diagnostics)

    attention_rows = [metadata[index] for index in attention]
    attention_cost = summed_parameter_cost(attention_rows)
    replacement_cost = summed_parameter_cost(replacement_rows)
    counterfactual = read_csv(Path(str(identity["task019_root"])) / "attention_counterfactual.csv")
    original_summary = next(row for row in counterfactual if row.get("mode") == "original_dynamic_30")
    expected_attention_cost = _as_int(original_summary, "incremental_attention_parameter_cost")
    replacement_trace = read_csv(Path(str(identity["task019_root"])) / "attention_replacement_trace.csv")
    expected_replacement_cost = summed_parameter_cost(replacement_trace)
    expected_difference = expected_replacement_cost - expected_attention_cost
    observed_difference = replacement_cost - attention_cost
    if attention_cost != expected_attention_cost or replacement_cost != expected_replacement_cost or observed_difference != expected_difference:
        raise RuntimeError("Task020 matched-budget identity differs from exact Task019 artifacts")
    budget_rows = []
    for group, values, cost in (("three_attention_heads", attention_rows, attention_cost),
                                ("replacement_ffn_set", replacement_rows, replacement_cost)):
        budget_rows.append({
            "group": group, "num_units": len(values), "parameter_cost": cost,
            "parameter_cost_difference_vs_attention": cost - attention_cost,
            "parameter_cost_ratio_vs_attention": cost / float(attention_cost),
            "sequence_sha256": sequence_sha256(int(row["global_index"]) for row in values),
        })
    atomic_csv(output_dir / "joint_budget_identity.csv", tuple(budget_rows[0]), budget_rows)
    atomic_json(output_dir / "joint_budget_identity.json", {
        "status": "PASS", "code_version": CODE_VERSION, "joint_parameter_budget_verified": True,
        "attention_parameter_cost": attention_cost, "replacement_parameter_cost": replacement_cost,
        "expected_parameter_cost_difference": expected_difference,
        "observed_parameter_cost_difference": observed_difference,
    })
    atomic_json(
        output_dir / "cohort_reconstruction.json",
        {
            "status": "PASS", "code_version": CODE_VERSION, "cohorts_reconstructed": True,
            "attention_count": len(attention), "replacement_ffn_count": len(replacement),
            "ordinary_deleted_ffn_count": len(ordinary),
            "matched_surviving_ffn_count": len(surviving),
            "control_source_policy": "exact original 28%-30% deleted FFN plus original-30%-surviving matched FFN",
            "pre20_removed_control_rejected": True,
            "cohort_sequence_sha256": sequence_sha256(int(row["global_index"]) for row in cohort_rows),
            "cohort_sha256": sha256_file(output_dir / "task020_cohorts.csv"),
        },
    )


def run_worker(output_dir: Path, worker: int, workers: int) -> None:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    cache = read_json(output_dir / "baseline_28_cache.json")
    if cache.get("baseline_28_cached") is not True or not baseline_cache_valid(output_dir):
        raise RuntimeError("Task020 baseline cache is incomplete or stale")
    with np.load(output_dir / "baseline_28_outputs.npz", allow_pickle=False) as payload:
        base_logits = payload["logits"].astype(np.float32, copy=True)
        targets = payload["targets"].astype(np.int64, copy=True)
        sample_indices = payload["sample_indices"].astype(np.int64, copy=True)
    cohorts = read_csv(output_dir / "task020_cohorts.csv")
    expected_indices = expected_worker_global_indices(cohorts, worker, workers)
    expected_set = set(expected_indices)
    selected = [row for row in cohorts if _as_int(row, "global_index") in expected_set]
    selected.sort(key=lambda row: expected_indices.index(_as_int(row, "global_index")))
    model, loader, _dataset = _load_model_and_loader(identity, sample_indices.tolist())
    batches = _materialize_batches(loader)
    _verify_worker_baseline(
        model, batches, bool(identity["amp_enabled"]), base_logits, targets, sample_indices
    )
    rows, attention_samples = [], []
    started = time.perf_counter()
    for ordinal, row in enumerate(selected, start=1):
        unit = UnitSpec(
            cohort=str(row["cohort"]), global_index=_as_int(row, "global_index"),
            layer=str(row["layer"]), unit_type=str(row["unit_type"]),
            unit_index=_as_int(row, "unit_index"), domain_id=_as_int(row, "domain_id"),
        )
        with additional_unit_ablation(model, unit):
            ablated, observed_targets, observed_indices = _forward_logits(
                model, batches, bool(identity["amp_enabled"])
            )
        if not np.array_equal(observed_targets, targets) or not np.array_equal(observed_indices, sample_indices):
            raise RuntimeError("Task020 worker sample identity drifted")
        metrics, samples = task_metrics(base_logits, ablated, targets)
        normalized = parameter_normalized_metrics(metrics, _as_int(row, "parameter_cost"))
        rows.append({**dict(row), **metrics, **normalized, "worker": worker, "worker_ordinal": ordinal})
        if unit.cohort == GROUP_ATTENTION:
            attention_samples.extend(
                {
                    "global_index": unit.global_index,
                    "video_id": int(sample_indices[int(sample["sample_offset"])]),
                    "label": int(sample["target"]),
                    "baseline_correct": int(sample["base_prediction"]) == int(sample["target"]),
                    "ablated_correct": int(sample["ablated_prediction"]) == int(sample["target"]),
                    "true_logit_drop": float(sample["true_logit_drop"]),
                    "margin_drop": float(sample["margin_drop"]),
                    "ce_increase": float(sample["ce_increase"]),
                    "kl_divergence": float(sample["kl"]),
                    "prediction_flipped": bool(sample["prediction_flip"]),
                }
                for sample in samples
            )
    fields = tuple(rows[0]) if rows else tuple(cohorts[0]) + METRIC_FIELDS + ("worker", "worker_ordinal")
    atomic_csv(output_dir / "workers" / f"worker{worker}_unit_task_importance.csv", fields, rows)
    sample_fields = tuple(attention_samples[0]) if attention_samples else (
        "global_index", "video_id", "label", "baseline_correct", "ablated_correct",
        "true_logit_drop", "margin_drop", "ce_increase", "kl_divergence", "prediction_flipped"
    )
    atomic_csv(output_dir / "workers" / f"worker{worker}_attention_task_effect_per_sample.csv", sample_fields, attention_samples)
    observed_indices = [int(row["global_index"]) for row in rows]
    if observed_indices != expected_indices or len(observed_indices) != len(set(observed_indices)):
        raise RuntimeError("Task020 worker output differs from its exact assigned cohort subset")
    dependencies = experiment_dependency_fields(output_dir)
    atomic_json(output_dir / "workers" / f"worker{worker}_completion.json", {
        "status": "PASS", **dependencies, "worker": worker, "workers": workers,
        "units_complete": len(rows), "elapsed_seconds": time.perf_counter() - started,
        "sample_count": len(sample_indices), "device": "cuda:0",
        "cohort_sha256": dependencies["cohort_sha256"],
        "expected_global_index_sequence_sha256": sequence_sha256(expected_indices),
        "observed_global_index_sequence_sha256": sequence_sha256(observed_indices),
    })


def run_joint(output_dir: Path, cohort: str) -> None:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if not baseline_cache_valid(output_dir):
        raise RuntimeError("Task020 joint ablation refuses stale baseline dependencies")
    with np.load(output_dir / "baseline_28_outputs.npz", allow_pickle=False) as payload:
        base_logits = payload["logits"].astype(np.float32, copy=True)
        targets = payload["targets"].astype(np.int64, copy=True)
        indices = payload["sample_indices"].astype(np.int64, copy=True)
    rows = [row for row in read_csv(output_dir / "task020_cohorts.csv") if row["cohort"] == cohort]
    if cohort not in {GROUP_ATTENTION, GROUP_REPLACEMENT} or not rows:
        raise ValueError(cohort)
    units = [UnitSpec(cohort, _as_int(row, "global_index"), str(row["layer"]), str(row["unit_type"]), _as_int(row, "unit_index"), _as_int(row, "domain_id")) for row in rows]
    model, loader, _ = _load_model_and_loader(identity, indices.tolist())
    batches = _materialize_batches(loader)
    _verify_worker_baseline(
        model, batches, bool(identity["amp_enabled"]), base_logits, targets, indices
    )
    contexts = []
    try:
        for unit in units:
            context = additional_unit_ablation(model, unit)
            context.__enter__()
            contexts.append(context)
        ablated, observed_targets, observed_indices = _forward_logits(model, batches, bool(identity["amp_enabled"]))
    finally:
        for context in reversed(contexts):
            context.__exit__(None, None, None)
    if not np.array_equal(observed_targets, targets) or not np.array_equal(observed_indices, indices):
        raise RuntimeError("Task020 joint sample identity drifted")
    metrics, _ = task_metrics(base_logits, ablated, targets)
    dependencies = experiment_dependency_fields(output_dir)
    cohort_sha = sequence_sha256(unit.global_index for unit in units)
    atomic_json(output_dir / "joint" / f"{cohort}.json", {
        "status": "PASS", **dependencies, "cohort": cohort, "unit_count": len(units),
        "cohort_sequence_sha256": cohort_sha, "sequence_sha256": cohort_sha, **metrics,
    })


def _full_validation_units(output_dir: Path, variant: str) -> list[UnitSpec]:
    rows = read_csv(Path(output_dir) / "task020_cohorts.csv")
    attention = [row for row in rows if row["cohort"] == GROUP_ATTENTION]
    replacement = [row for row in rows if row["cohort"] == GROUP_REPLACEMENT]
    if variant == "baseline_28":
        selected = []
    elif variant == "attention_joint":
        selected = attention
    elif variant == "replacement_ffn_joint":
        selected = replacement
    elif variant.startswith("attention_"):
        index = int(variant.split("_", 1)[1])
        selected = [row for row in attention if int(row["global_index"]) == index]
        if len(selected) != 1:
            raise ValueError(f"Unknown Task020 Attention full-validation variant: {variant}")
    else:
        raise ValueError(f"Unknown Task020 full-validation variant: {variant}")
    return [
        UnitSpec(str(row["cohort"]), _as_int(row, "global_index"), str(row["layer"]),
                 str(row["unit_type"]), _as_int(row, "unit_index"), _as_int(row, "domain_id"))
        for row in selected
    ]


def full_validation_expected_identity(output_dir: Path, variant: str) -> dict[str, object]:
    root = Path(output_dir)
    identity = read_json(root / "artifact_identity.json")
    units = _full_validation_units(root, variant)
    return {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "s28_prefix_sha256": identity["task018_s28_sequence_sha256"],
        "intervention_sequence_sha256": sequence_sha256(unit.global_index for unit in units),
        "production_source_sha256": identity["production_source_sha256"],
        "code_version": CODE_VERSION,
    }


def full_validation_cache_valid(output_dir: Path, variant: str) -> bool:
    path = Path(output_dir) / "full_validation" / f"{variant}.json"
    if not path.is_file():
        return False
    try:
        value = read_json(path)
        expected = full_validation_expected_identity(output_dir, variant)
        return value.get("status") == "PASS" and all(value.get(key) == expected_value
            for key, expected_value in expected.items())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_full_validation(output_dir: Path, variant: str) -> None:
    import torch
    from ucf101_videoswin_my import AverageMeter, get_dataset, validate_rgb

    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task020 full validation requires passed artifact identity")
    source_commit_identity = production_source_git_identity(Path(__file__).resolve().parent)
    recorded_identity = identity.get("production_source_git_blob_sha")
    if recorded_identity is not None and source_commit_identity != recorded_identity:
        raise RuntimeError("Protected production Git identity changed before Task020 full validation")
    units = _full_validation_units(output_dir, variant)
    cohort_rows = {int(row["global_index"]): row for row in read_csv(output_dir / "task020_cohorts.csv")}
    model = _load_fresh_28_model(identity)
    loader = get_dataset(str(identity["validation_split"]), int(identity["validation_batch_size"]))
    contexts = []
    try:
        for unit in units:
            context = additional_unit_ablation(model, unit); context.__enter__(); contexts.append(context)
        top1, top5 = AverageMeter(), AverageMeter()
        started = time.perf_counter()
        validate_rgb(loader, model, top1, top5, use_amp=bool(identity["amp_enabled"]))
        elapsed = time.perf_counter() - started
    finally:
        for context in reversed(contexts): context.__exit__(None, None, None)
    expected = full_validation_expected_identity(output_dir, variant)
    attention_count = sum(unit.unit_type == TYPE_ATTENTION for unit in units)
    ffn_count = sum(unit.unit_type == TYPE_FFN for unit in units)
    atomic_json(output_dir / "full_validation" / f"{variant}.json", {
        "status": "PASS", "variant": variant, "num_additional_units": len(units),
        "attention_units": attention_count, "ffn_units": ffn_count,
        "additional_parameter_cost": sum(int(cohort_rows[unit.global_index]["parameter_cost"]) for unit in units),
        "top1": float(top1.avg), "top5": float(top5.avg),
        "validation_samples": len(loader.dataset), "validation_split": identity["validation_split"],
        "amp_enabled": bool(identity["amp_enabled"]), "validation_time_seconds": elapsed,
        "fresh_original_checkpoint": True, "fresh_exact_28_state": True,
        "cumulative_intervention_reuse": False, "selector_search_executed": False,
        "fine_tuning_executed": False, "device": "cuda:0", **expected,
    })
    del model, loader
    torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("identity")
    for name in ("task014", "task015", "task016", "task017", "task018", "task019"):
        identity.add_argument(f"--{name}-root", type=Path, required=True)
    identity.add_argument("--output-dir", type=Path, required=True)
    for name in ("prepare", "baseline"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--worker", type=int, required=True)
    worker.add_argument("--workers", type=int, default=2)
    joint = sub.add_parser("joint")
    joint.add_argument("--output-dir", type=Path, required=True)
    joint.add_argument("--cohort", choices=(GROUP_ATTENTION, GROUP_REPLACEMENT), required=True)
    full = sub.add_parser("full-validate")
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument("--variant", required=True)
    check = sub.add_parser("check-cache")
    check.add_argument("--output-dir", type=Path, required=True)
    check.add_argument("--kind", choices=("baseline", "worker", "joint", "full"), required=True)
    check.add_argument("--worker", type=int)
    check.add_argument("--cohort", choices=(GROUP_ATTENTION, GROUP_REPLACEMENT))
    check.add_argument("--variant")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "identity":
        verify_identity(
            task014_root=args.task014_root, task015_root=args.task015_root,
            task016_root=args.task016_root, task017_root=args.task017_root,
            task018_root=args.task018_root, task019_root=args.task019_root,
            output_dir=args.output_dir,
        )
    elif args.command == "prepare":
        prepare_cohorts(args.output_dir)
    elif args.command == "baseline":
        cache_baseline(args.output_dir)
    elif args.command == "worker":
        if not 0 <= args.worker < args.workers:
            raise ValueError("worker must lie in [0, workers)")
        run_worker(args.output_dir, args.worker, args.workers)
    elif args.command == "joint":
        run_joint(args.output_dir, args.cohort)
    elif args.command == "full-validate":
        run_full_validation(args.output_dir, args.variant)
    else:
        output_dir = Path(args.output_dir)
        if args.kind == "baseline":
            valid = baseline_cache_valid(output_dir)
        elif args.kind == "worker":
            if args.worker is None: raise ValueError("--worker is required")
            valid = result_cache_valid(output_dir / "workers" / f"worker{args.worker}_completion.json",
                                       output_dir, worker=args.worker, workers=2)
        elif args.kind == "joint":
            if args.cohort is None: raise ValueError("--cohort is required")
            units = [row for row in read_csv(output_dir / "task020_cohorts.csv") if row["cohort"] == args.cohort]
            valid = result_cache_valid(output_dir / "joint" / f"{args.cohort}.json", output_dir,
                cohort=args.cohort,
                cohort_sequence_sha256=sequence_sha256(_as_int(row, "global_index") for row in units))
        else:
            if args.variant is None: raise ValueError("--variant is required")
            valid = full_validation_cache_valid(output_dir, args.variant)
        return 0 if valid else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

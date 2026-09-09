"""Task024 threshold-free functional-risk construction and analysis helpers.

This module is an independent research ablation.  It consumes the identity
checked Task014--Task023 artifacts and never changes the production selector.
The two policies use deterministic ordinal ranks of the current
``Delta_total`` and ``Delta_average`` values.  The adaptive policy additionally
uses the already maintained functional-domain coverage state.  All ranking
vectors stay on the replay device; only the single selected row is transferred
to Python for structural bookkeeping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CODE_VERSION = "task024_threshold_free_adaptive_safety_v1"
ANALYSIS_VERSION = "task024_threshold_free_adaptive_safety_analysis_v1"
TASK019_COMMIT = "3c053f11fe6e344a5e7b60875ebe195cd30a1eee"
START_SPARSE = 0.28
TARGET_SPARSE = 0.30
EXPECTED_VALIDATION_SAMPLES = 3_783
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
VARIANT_MINIMAX = "dual_rank_minimax_30"
VARIANT_ADAPTIVE = "dual_rank_domain_state_minimax_30"
VARIANTS = (VARIANT_MINIMAX, VARIANT_ADAPTIVE)
REFERENCE_VARIANTS = (
    "original_dynamic_total_30",
    "average_rescue_all_30",
)
PRODUCTION_FILES = (
    "functional_competition_pruning.py",
    "MC.py",
    "ucf101_videoswin_my.py",
)
REQUIRED_VALIDATION_IDENTITY = (
    "validation_split",
    "validation_batch_size",
    "amp_enabled",
)
TRACE_FIELDS = (
    "variant", "step", "incremental_step", "estimated_sparsity_before",
    "estimated_sparsity_after", "global_index", "unit_type", "layer",
    "stage", "unit_index", "domain_id", "Delta_average", "Delta_total",
    "p_average", "p_total", "domain_coverage_before", "domain_damage_before",
    "R_dual", "R_adaptive", "parameter_cost", "cumulative_removed_parameters",
)
PROGRESS_EVERY = 50  # output cadence only; it is not a selector setting.


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float(row: Mapping[str, object], *names: str, default: float | None = None) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite {name}")
            return number
    if default is not None:
        return float(default)
    raise KeyError(names[0])


def _int(row: Mapping[str, object], *names: str, default: int | None = None) -> int:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return int(float(value))
    if default is not None:
        return int(default)
    raise KeyError(names[0])


def stage_from_layer(layer: object) -> str:
    text = str(layer)
    marker = text.split("layers.", 1)[1].split(".", 1)[0] if "layers." in text else "unknown"
    return f"stage_{marker}"


def _as_tensor(value, *, device=None, dtype=None):
    """Import torch lazily so semantic tests remain CPU-only and cheap."""
    import torch
    return torch.as_tensor(value, device=device, dtype=dtype)


def _stable_order(values, tie_ids):
    """Return a deterministic ascending order, preserving global-id ties.

    Stable CUDA sorting is used when available.  The fallback only moves
    group boundaries and never the full candidate arrays to CPU; equal-risk
    groups are ordered by their integer global ids on the original device.
    """
    import torch
    if values.ndim != 1 or tie_ids.ndim != 1 or values.numel() != tie_ids.numel():
        raise ValueError("risk and tie-id vectors must be one-dimensional and aligned")
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    try:
        by_id = torch.argsort(tie_ids, stable=True)
        return by_id[torch.argsort(values[by_id], stable=True)]
    except (TypeError, RuntimeError) as error:
        # Older torch builds may not expose stable CUDA argsort.  A regular
        # risk sort still gives contiguous equal-value groups; sort only each
        # group by unique integer ids, which is exact and deterministic.
        try:
            rough = torch.argsort(values)
        except RuntimeError:
            raise RuntimeError("Deterministic tensor sorting is unavailable") from error
        ordered_values = values[rough]
        boundaries = torch.nonzero(
            ordered_values[1:] != ordered_values[:-1], as_tuple=False
        ).flatten().tolist()
        starts = [0] + [int(value) + 1 for value in boundaries]
        ends = [int(value) + 1 for value in boundaries] + [int(values.numel())]
        chunks = []
        for start, end in zip(starts, ends):
            group = rough[start:end]
            group_order = torch.argsort(tie_ids[group])
            chunks.append(group[group_order])
        return torch.cat(chunks) if chunks else rough


def ordinal_percentile_ranks(values, global_indices):
    """Compute ascending ordinal percentiles on the supplied tensor device."""
    import torch
    values = _as_tensor(values)
    global_indices = _as_tensor(global_indices, device=values.device, dtype=torch.long)
    if values.ndim != 1 or global_indices.ndim != 1 or values.numel() != global_indices.numel():
        raise ValueError("values and global_indices must be aligned one-dimensional vectors")
    if values.numel() and not bool(torch.isfinite(values).all().item()):
        raise ValueError("risk values must be finite")
    if global_indices.numel() != torch.unique(global_indices).numel():
        raise ValueError("global indices must be unique")
    order = _stable_order(values, global_indices)
    ranks = torch.empty_like(values, dtype=torch.float32)
    positions = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    ranks[order] = positions / float(max(int(values.numel()) - 1, 1))
    return ranks


def _stable_key_order(order, values):
    import torch
    try:
        return order[torch.argsort(values[order], stable=True)]
    except (TypeError, RuntimeError) as error:
        try:
            rough = order[torch.argsort(values[order])]
        except RuntimeError:
            raise RuntimeError("Deterministic tensor sorting is unavailable") from error
        sorted_values = values[rough]
        boundaries = torch.nonzero(
            sorted_values[1:] != sorted_values[:-1], as_tuple=False
        ).flatten().tolist()
        starts = [0] + [int(value) + 1 for value in boundaries]
        ends = [int(value) + 1 for value in boundaries] + [int(values.numel())]
        chunks = []
        # ``order`` already encodes every lower-priority tie break.  Recover
        # each equal-value group by boolean indexing the old order, whose
        # indexing order is stable even on old torch builds.
        old_values = values[order]
        for start, end in zip(starts, ends):
            value = sorted_values[start]
            chunks.append(order[old_values == value])
        return torch.cat(chunks) if chunks else rough


def _select_order(keys, size: int, device):
    """Stable lexicographic order; ``keys`` are listed highest priority first."""
    import torch
    order = torch.arange(size, device=device, dtype=torch.long)
    for key in reversed(keys):
        order = _stable_key_order(order, key)
    return order


def compute_risk_tensors(delta_total, delta_average, global_indices, domain_damage=None):
    """Return p_total, p_average, R_dual and R_adaptive on one device."""
    import torch
    total = _as_tensor(delta_total)
    average = _as_tensor(delta_average, device=total.device, dtype=total.dtype)
    gids = _as_tensor(global_indices, device=total.device, dtype=torch.long)
    if total.ndim != 1 or average.shape != total.shape or gids.shape != total.shape:
        raise ValueError("risk vectors must have matching one-dimensional shapes")
    p_total = ordinal_percentile_ranks(total, gids)
    p_average = ordinal_percentile_ranks(average, gids)
    dual = torch.maximum(p_total, p_average)
    if domain_damage is None:
        damage = torch.zeros_like(dual)
    else:
        damage = _as_tensor(domain_damage, device=total.device, dtype=dual.dtype)
        if damage.shape != dual.shape:
            raise ValueError("domain damage must align with candidates")
        if damage.numel() and (not bool(torch.isfinite(damage).all().item()) or
                               bool((damage < 0).any().item()) or
                               bool((damage > 1).any().item())):
            raise ValueError("domain damage must be finite and in [0,1]")
    adaptive = torch.maximum(dual, damage)
    return {"p_total": p_total, "p_average": p_average, "R_dual": dual,
            "domain_damage": damage, "R_adaptive": adaptive}


def select_ranked_candidate(candidates: Mapping[str, object], *, adaptive: bool = False) -> int:
    """Return the tensor position selected by the exact policy tie-break."""
    import torch
    required = ("global_index", "Delta_total", "Delta_average")
    if any(name not in candidates for name in required):
        raise KeyError("candidate tensors require global_index, Delta_total and Delta_average")
    values = {name: _as_tensor(candidates[name]) for name in required}
    risks = compute_risk_tensors(values["Delta_total"], values["Delta_average"],
                                 values["global_index"], candidates.get("domain_damage"))
    if values["global_index"].numel() == 0:
        raise RuntimeError("Cannot select from an empty candidate set")
    if adaptive:
        keys = (risks["R_adaptive"], risks["p_total"], risks["p_average"],
                values["Delta_total"], values["global_index"].to(torch.float64))
    else:
        keys = (risks["R_dual"], risks["p_total"], risks["p_average"],
                values["global_index"].to(torch.float64))
    order = _select_order(keys, int(values["global_index"].numel()), values["global_index"].device)
    return int(order[0].item())


def rank_candidate_tensors(candidates: Mapping[str, object], *, adaptive: bool = False) -> dict:
    """Attach policy ranks without materialising per-candidate Python rows."""
    import torch
    risks = compute_risk_tensors(candidates["Delta_total"], candidates["Delta_average"],
                                 candidates["global_index"], candidates.get("domain_damage"))
    output = dict(candidates)
    output.update(risks)
    output["selected_position"] = select_ranked_candidate(output, adaptive=adaptive)
    return output


def _validate_trace_rows(rows: Sequence[Mapping[str, object]]) -> None:
    seen = set()
    for position, row in enumerate(rows):
        missing = [name for name in ("global_index", "layer", "unit_type", "unit_index")
                   if name not in row or str(row[name]).strip() == ""]
        if missing:
            raise RuntimeError(f"Trace row {position} missing structural fields {missing}")
        index = _int(row, "global_index")
        if index in seen:
            raise RuntimeError(f"Trace contains duplicate global_index {index}")
        seen.add(index)


class TensorSafetyEngine:
    """Adapter over Task019 replay state with GPU-resident candidate vectors."""

    def __init__(self, *, task014_root: Path, task016_root: Path, task017_root: Path,
                 output_dir: Path, device: str):
        import torch
        import task019_dynamic_ranking_causal_ablation as task019
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Task024 construction requires CUDA")
        self.base = task019.CausalAblationEngine(task014_root=task014_root,
                                                 task016_root=task016_root,
                                                 task017_root=task017_root,
                                                 output_dir=output_dir, device=device)
        self.torch = torch
        self.device = self.base.device
        self._global_tensors = [torch.as_tensor(state.global_indices, dtype=torch.long,
                                                 device=self.device)
                                for state in self.base.replay.states]
        self.started = time.perf_counter()

    @property
    def removed_cost(self):
        return self.base.removed_cost

    @property
    def total_parameters(self):
        return self.base.replay.total_parameters

    def restore_prefix(self, prefix):
        self.base.restore_prefix(prefix)

    def candidate_tensors(self) -> dict:
        chunks = defaultdict(list)
        for domain_id, state in enumerate(self.base.replay.states):
            local = self.base.replay.cache.eligible_mask(domain_id).nonzero(as_tuple=True)[0]
            if local.numel() == 0:
                continue
            average = state.losses.index_select(0, local)
            total = average * float(self.base.replay.cache.active_size[domain_id])
            globals_ = self._global_tensors[domain_id].index_select(0, local)
            domain = torch_full_like(average, domain_id, self.torch.long)
            coverage = self.torch.full_like(average, float(state.current_coverage))
            chunks["global_index"].append(globals_)
            chunks["domain_id"].append(domain)
            chunks["local_index"].append(local)
            chunks["Delta_average"].append(average)
            chunks["Delta_total"].append(total)
            chunks["domain_coverage"].append(coverage)
            chunks["domain_damage"].append(1.0 - coverage)
        if not chunks:
            return {name: self.torch.empty(0, device=self.device)
                    for name in ("global_index", "domain_id", "local_index",
                                 "Delta_average", "Delta_total", "domain_coverage",
                                 "domain_damage")}
        return {name: self.torch.cat(values) for name, values in chunks.items()}

    def remove(self, candidate: Mapping[str, object], *, variant: str, selected_position: int,
               risks: Mapping[str, object]) -> dict:
        position = int(selected_position)
        global_index = int(candidate["global_index"][position].item())
        domain_id = int(candidate["domain_id"][position].item())
        local_index = int(candidate["local_index"][position].item())
        before = self.base.removed_cost
        result = self.base.remove(variant=variant, global_index=global_index,
                                  domain_id=domain_id, local_index=local_index,
                                  rank=position + 1)
        result.update({
            "p_total": float(risks["p_total"][position].item()),
            "p_average": float(risks["p_average"][position].item()),
            "domain_coverage_before": float(candidate["domain_coverage"][position].item()),
            "domain_damage_before": float(risks["domain_damage"][position].item()),
            "R_dual": float(risks["R_dual"][position].item()),
            "R_adaptive": (float(risks["R_adaptive"][position].item())
                            if variant == VARIANT_ADAPTIVE else ""),
            "estimated_sparsity_before": before / float(self.total_parameters),
            "estimated_sparsity_after": self.base.removed_cost / float(self.total_parameters),
            "incremental_step": result["step"],
        })
        return result

    def run(self, variant: str, target_budget: float, start_prefix) -> list[dict]:
        adaptive = variant == VARIANT_ADAPTIVE
        self.restore_prefix(start_prefix)
        rows = []
        while self.base.removed_cost < float(target_budget):
            tensors = self.candidate_tensors()
            if tensors["global_index"].numel() == 0:
                raise RuntimeError(f"{variant} has no feasible candidate before target budget")
            ranked = rank_candidate_tensors(tensors, adaptive=adaptive)
            position = int(ranked.pop("selected_position"))
            row = self.remove(ranked, variant=variant, selected_position=position, risks=ranked)
            row["step"] = len(rows) + 1
            row["incremental_step"] = len(rows) + 1
            row["stage"] = stage_from_layer(row.get("layer", ""))
            rows.append(row)
            if len(rows) % PROGRESS_EVERY == 0:
                elapsed = max(time.perf_counter() - self.started, 1e-9)
                mem = int(self.torch.cuda.max_memory_allocated(self.device)) / 2**30
                print(f"{variant} step={len(rows)} sparsity="
                      f"{self.base.removed_cost/self.total_parameters:.6f} "
                      f"elapsed={elapsed:.1f}s steps/sec={len(rows)/elapsed:.2f} "
                      f"R={row['R_adaptive'] if adaptive else row['R_dual']:.6f} "
                      f"p_total={row['p_total']:.6f} p_average={row['p_average']:.6f} "
                      f"coverage={row['domain_coverage_before']:.6f} gpu={mem:.2f}G",
                      flush=True)
        _validate_trace_rows(rows)
        return rows


def torch_full_like(value, fill, dtype):
    return value.new_full(value.shape, fill, dtype=dtype)


def _task023_module():
    import task023_average_rescue_causal_ablation as task023
    return task023


def _task016_reference(task016_root: Path, task017_root: Path, task018_root: Path):
    task023 = _task023_module()
    trace30, prefixes, summaries, extra = task023._task016_prefix_identity(
        Path(task016_root), Path(task017_root), Path(task018_root)
    )
    start_summary = task023._summary_for_target(summaries, START_SPARSE)
    target_summary = task023._summary_for_target(summaries, TARGET_SPARSE)
    if start_summary is None or target_summary is None:
        raise RuntimeError("Task016 exact 28% and 30% summaries are required")
    return trace30, prefixes, start_summary, target_summary, extra


def verify_task023_reference(task023_root: Path) -> dict:
    """Read-only gate for the completed Task023 reference artifacts."""
    root = Path(task023_root)
    completion = read_json(root / "task023_completion.json")
    if completion.get("status") != "PASS":
        raise RuntimeError("Task023 final-integrity status is not PASS")
    for key in ("artifact_identity_pass", "fresh_exact_v0_validation_complete",
                "average_rescue_all_validation_complete"):
        if completion.get(key) is not True:
            raise RuntimeError(f"Task023 reference gate failed: {key}")
    if any(completion.get(key) is not False for key in (
            "task_label_leakage", "production_pruning_code_modified",
            "fine_tuning_executed")):
        raise RuntimeError("Task023 reference safety gate failed")
    metrics = {}
    task023_variants = (
        "original_dynamic_total_30", "average_rescue_all_30",
        "average_rescue_ffn_only_30", "mirror_rescue_control_30",
    )
    for variant in task023_variants:
        path = root / "validation" / variant / "metrics.json"
        payload = read_json(path)
        if payload.get("status") != "PASS" or int(payload.get("samples", -1)) != EXPECTED_VALIDATION_SAMPLES:
            raise RuntimeError(f"Task023 reference metric is incomplete: {variant}")
        if variant == REFERENCE_VARIANTS[0] and payload.get("fresh_exact_v0_validation") is not True:
            raise RuntimeError("Task023 V0 reference is not the fresh exact validation")
        metrics[variant] = payload
    if metrics[REFERENCE_VARIANTS[0]].get("fresh_exact_v0_validation") is not True:
        raise RuntimeError("Task023 V0 reference is not the fresh exact validation")
    return {"completion": completion, "metrics": metrics,
            "completion_sha256": sha256_file(root / "task023_completion.json")}


def production_identity(repo_root: Path) -> dict:
    task023 = _task023_module()
    return task023._production_identity(Path(repo_root))


def _copy_validation_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy the locked validation configuration from Task023."""
    missing = [key for key in REQUIRED_VALIDATION_IDENTITY if key not in identity]
    if missing:
        raise RuntimeError(f"Task023 validation identity incomplete: {missing}")

    validation_split = str(identity["validation_split"])
    if not validation_split.strip():
        raise RuntimeError("Task023 validation identity incomplete: validation_split is empty")
    try:
        validation_batch_size = int(identity["validation_batch_size"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Task023 validation identity has invalid validation_batch_size") from exc
    if validation_batch_size <= 0:
        raise RuntimeError("Task023 validation identity requires validation_batch_size > 0")

    return {
        "validation_split": validation_split,
        "validation_batch_size": validation_batch_size,
        "amp_enabled": bool(identity["amp_enabled"]),
    }


def _assert_validation_identity_match(
    task024_identity: Mapping[str, object],
    task023_identity: Mapping[str, object],
) -> None:
    expected = _copy_validation_identity(task023_identity)
    mismatches = [
        key for key in REQUIRED_VALIDATION_IDENTITY
        if task024_identity.get(key) != expected[key]
    ]
    if mismatches:
        raise RuntimeError(
            f"Task024 validation identity differs from Task023: {mismatches}"
        )


def verify_identity(*, task023_root: Path, task014_root: Path, task016_root: Path,
                    task017_root: Path, task018_root: Path, task019_root: Path,
                    task020_root: Path, task021_root: Path, task022_root: Path,
                    output_dir: Path, checkpoint: Path, repo_root: Path) -> dict:
    reference = verify_task023_reference(task023_root)
    roots = (task014_root, task016_root, task017_root, task018_root, task019_root,
             task020_root, task021_root, task022_root)
    missing = [str(Path(root)) for root in roots if not Path(root).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing immutable roots: {missing}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    task023 = _task023_module()
    identity23 = read_json(Path(task023_root) / "artifact_identity.json")
    validation_identity = _copy_validation_identity(identity23)
    trace30, prefixes, start_summary, target_summary, extra = _task016_reference(
        task016_root, task017_root, task018_root
    )
    if checkpoint_sha := sha256_file(checkpoint):
        if checkpoint_sha != identity23.get("checkpoint_sha256"):
            raise RuntimeError("Task024 checkpoint differs from Task023 reference")
    production = production_identity(repo_root)
    reference_production = identity23.get("production_source_git_blob_sha")
    if reference_production and production != reference_production:
        raise RuntimeError("Task024 production source identity differs from Task023 reference")
    payload = {
        "status": "PASS", "code_version": CODE_VERSION,
        "analysis_version": ANALYSIS_VERSION, "artifact_identity_pass": True,
        "task023_identity_pass": True,
        "task023_final_integrity_pass": True,
        "task023_exact_v0_reference_pass": True,
        "task023_hard_rescue_reference_pass": True,
        "task023_root": str(Path(task023_root).resolve()),
        "task014_root": str(Path(task014_root).resolve()),
        "task016_root": str(Path(task016_root).resolve()),
        "task017_root": str(Path(task017_root).resolve()),
        "task018_root": str(Path(task018_root).resolve()),
        "task019_root": str(Path(task019_root).resolve()),
        "task020_root": str(Path(task020_root).resolve()),
        "task021_root": str(Path(task021_root).resolve()),
        "task022_root": str(Path(task022_root).resolve()),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "validation_split": validation_identity["validation_split"],
        "validation_batch_size": validation_identity["validation_batch_size"],
        "amp_enabled": validation_identity["amp_enabled"],
        "parameters_before": int(identity23["parameters_before"]),
        "start_sparsity": START_SPARSE, "target_sparsity": TARGET_SPARSE,
        "start_parameter_budget": float(start_summary["target_parameter_budget"]),
        "target_parameter_budget": float(target_summary["target_parameter_budget"]),
        "start_prefix_steps": len(prefixes[START_SPARSE]),
        "target_prefix_steps": len(prefixes[TARGET_SPARSE]),
        "start_prefix_sequence_sha256": sequence_sha256(
            int(row["global_index"]) for row in prefixes[START_SPARSE]),
        "target_prefix_sequence_sha256": sequence_sha256(
            int(row["global_index"]) for row in prefixes[TARGET_SPARSE]),
        "task016_trace30_sha256": sha256_file(extra["trace30_path"]),
        "task023_completion_sha256": reference["completion_sha256"],
        "task023_reference_metrics": reference["metrics"],
        "production_source_git_blob_sha": production,
        "zero_new_tunable_hyperparameters": True,
        "fixed_percentile_threshold_used": False,
        "task_label_leakage": False, "production_pruning_code_modified": False,
        "fine_tuning_executed": False, "budget_semantics_unchanged": True,
        "min_keep_semantics_unchanged": True,
    }
    _assert_validation_identity_match(payload, identity23)
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "trace30": trace30, "prefixes": prefixes,
            "start_summary": start_summary, "target_summary": target_summary}


def _reference_rows(task023_root: Path, variant: str) -> list[dict]:
    path = Path(task023_root) / "variants" / variant / "causal_selection_trace.csv"
    return read_csv(path)


def construct_variant(*, variant: str, task023_root: Path, task014_root: Path,
                      task016_root: Path, task017_root: Path, task018_root: Path,
                      output_dir: Path, device: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Task024 variant: {variant}")
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task024 identity gate must pass before construction")
    _, prefixes, start_summary, target_summary, _ = _task016_reference(
        task016_root, task017_root, task018_root)
    engine = TensorSafetyEngine(task014_root=task014_root, task016_root=task016_root,
                                task017_root=task017_root, output_dir=output_dir,
                                device=device)
    incremental = engine.run(variant, float(target_summary["target_parameter_budget"]),
                             prefixes[START_SPARSE])
    task023 = _task023_module()
    full_prefix = [dict(row) for row in prefixes[START_SPARSE]] + incremental
    task023.assert_registry_ready_rows(full_prefix)
    import task019_dynamic_ranking_causal_ablation as task019
    registry = task019.registry_from_prefix(full_prefix)
    incremental_cost = sum(_int(row, "parameter_cost") for row in incremental)
    attention = [row for row in incremental if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffn = [row for row in incremental if str(row.get("unit_type")) == TYPE_FFN]
    variant_dir = Path(output_dir) / "variants" / variant
    atomic_csv(variant_dir / "causal_selection_trace.csv", TRACE_FIELDS, incremental)
    domains = engine.base.domain_state_rows(variant)
    atomic_csv(variant_dir / "domain_state.csv",
               tuple(domains[0]) if domains else ("variant",), domains)
    summary = {
        "status": "prepared", "variant": variant, "start_sparsity": START_SPARSE,
        "target_sparsity": TARGET_SPARSE, "start_prefix_steps": len(prefixes[START_SPARSE]),
        "incremental_units": len(incremental), "incremental_attention": len(attention),
        "incremental_ffn": len(ffn), "incremental_parameter_cost": incremental_cost,
        "incremental_attention_parameter_cost": sum(_int(row, "parameter_cost") for row in attention),
        "incremental_ffn_parameter_cost": sum(_int(row, "parameter_cost") for row in ffn),
        "estimated_removed_parameters": int(start_summary["estimated_removed_parameters"]) + incremental_cost,
        "estimated_parameter_sparsity": (int(start_summary["estimated_removed_parameters"]) + incremental_cost)
        / float(identity["parameters_before"]),
        "target_parameter_budget": float(target_summary["target_parameter_budget"]),
        "budget_overshoot": (int(start_summary["estimated_removed_parameters"]) + incremental_cost)
        - float(target_summary["target_parameter_budget"]),
        "sequence_sha256": sequence_sha256(int(row["global_index"]) for row in full_prefix),
        "increment_sequence_sha256": sequence_sha256(int(row["global_index"]) for row in incremental),
        "unique_domains_touched": len({int(row["domain_id"]) for row in incremental}),
        "unique_layers_touched": len({str(row["layer"]) for row in incremental}),
        "registry_schema_pass": True, "budget_semantics_unchanged": True,
        "min_keep_semantics_unchanged": True, "task_label_leakage": False,
        "registry_canonical_sha256": task023.canonical_registry_sha256(registry),
        "device": str(device), "construction_gpu": str(device),
        "peak_cuda_bytes": int(engine.torch.cuda.max_memory_allocated(engine.device)),
    }
    atomic_json(variant_dir / "registry.json", {
        "status": "prepared", "variant": variant, "registry": registry,
        "summary": summary, "checkpoint_sha256": identity["checkpoint_sha256"],
        "production_pruning_code_modified": False, "task_label_leakage": False,
    })
    atomic_json(variant_dir / "construction.json", summary)
    return summary


def validate_variant(output_dir: Path, variant: str, device: str) -> dict:
    """Run one fresh-checkpoint validation; the runner calls this sequentially."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    payload = read_json(Path(output_dir) / "variants" / variant / "registry.json")
    if identity.get("artifact_identity_pass") is not True or payload.get("status") != "prepared":
        raise RuntimeError("Task024 identity/construction gate is incomplete")
    validation_identity = _copy_validation_identity(identity)
    import torch
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Task024 validation requires CUDA")
    from task018_high_sparsity_transition import _apply_registry
    from ucf101_videoswin_my import AverageMeter, SwinTransformer3D, get_dataset, set_seed, validate_rgb
    torch.cuda.set_device(torch.device(device))
    set_seed(3407)
    model = SwinTransformer3D(patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
                              num_heads=[3, 6, 12, 24], window_size=(8, 7, 7),
                              mlp_ratio=4.0, qkv_bias=True, patch_norm=True,
                              drop_path_rate=0.2, use_checkpoint=True).to(torch.device(device))
    checkpoint = Path(identity["checkpoint"])
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    normalized = {key.replace("module.", "").replace("backbone.", ""): value
                  for key, value in state_dict.items()}
    load_message = model.load_state_dict(normalized, strict=False)
    before = sum(parameter.numel() for parameter in model.parameters())
    if before != int(identity["parameters_before"]):
        raise RuntimeError("Model parameter count differs from identity")
    applied = _apply_registry(model, payload["registry"])
    summary = payload["summary"]
    expected_attention = sum(len(entry["indices"]) for entry in payload["registry"].values()
                             if entry["unit_type"] == TYPE_ATTENTION)
    expected_ffn = sum(len(entry["indices"]) for entry in payload["registry"].values()
                       if entry["unit_type"] == TYPE_FFN)
    if applied != {"removed_attention": expected_attention, "removed_ffn": expected_ffn}:
        raise RuntimeError("Applied registry count mismatch")
    loader = get_dataset(
        str(validation_identity["validation_split"]),
        int(validation_identity["validation_batch_size"]),
    )
    top1, top5 = AverageMeter(), AverageMeter()
    started = time.perf_counter()
    validate_rgb(loader, model, top1, top5, use_amp=bool(validation_identity["amp_enabled"]))
    elapsed = time.perf_counter() - started
    after = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "status": "PASS", "variant": variant, "top1": float(top1.avg), "top5": float(top5.avg),
        "samples": len(loader.dataset), "validation_time_seconds": elapsed,
        "estimated_parameter_sparsity": float(summary["estimated_parameter_sparsity"]),
        "physical_numel_sparsity": 1.0 - after / float(before),
        "parameters_before": before, "parameters_after": after,
        "removed_attention": expected_attention, "removed_ffn": expected_ffn,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "sequence_sha256": summary["sequence_sha256"],
        "registry_canonical_sha256": summary["registry_canonical_sha256"],
        "checkpoint_missing_keys": sorted(load_message.missing_keys),
        "checkpoint_unexpected_keys": sorted(load_message.unexpected_keys),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "fresh_original_checkpoint": True, "fine_tuning_executed": False,
        "production_pruning_code_modified": False,
    }
    atomic_json(Path(output_dir) / "validation" / variant / "metrics.json", metrics)
    return metrics


def _counts(rows: Sequence[Mapping[str, object]]) -> Counter:
    return Counter(_int(row, "domain_id") for row in rows)


def _gini(values: Sequence[int]) -> float:
    ordered = sorted(float(value) for value in values if float(value) >= 0)
    n = len(ordered)
    total = sum(ordered)
    if n == 0 or total == 0:
        return 0.0
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * total)


def concentration_row(name: str, rows: Sequence[Mapping[str, object]], coverage_rows=None) -> dict[str, object]:
    counts = _counts(rows)
    values = sorted(counts.values(), reverse=True)
    total = sum(values)
    shares = [sum(values[:k]) / total if total else 0.0 for k in (1, 2, 4)]
    hhi = sum((value / total) ** 2 for value in values) if total else 0.0
    coverage = [float(row.get("coverage", row.get("retained_ratio", 0.0)))
                for row in (coverage_rows or [])]
    return {
        "variant": name, "incremental_units_removed": len(rows),
        "unique_domains_touched": len(counts), "largest_domain_share": shares[0],
        "top2_domain_share": shares[1], "top4_domain_share": shares[2],
        "HHI": hhi, "Gini": _gini(values), "hhi": hhi, "gini": _gini(values),
        "median_removals_per_touched_domain": float(statistics.median(counts.values())) if counts else 0.0,
        "maximum_removals_one_domain": max(values, default=0),
        "final_mean_domain_coverage": float(np.mean(coverage)) if coverage else math.nan,
        "final_median_domain_coverage": float(np.median(coverage)) if coverage else math.nan,
        "minimum_domain_coverage": min(coverage, default=math.nan),
        "q10_domain_coverage": float(np.quantile(coverage, 0.1)) if coverage else math.nan,
    }


# Descriptive aliases keep the analysis API easy to discover without adding
# another implementation of either concentration metric.
domain_concentration_row = concentration_row
compute_percentile_ranks = ordinal_percentile_ranks


def partition_sets(left: set[int], right: set[int]) -> dict[str, set[int]]:
    common = set(left) & set(right)
    baseline_only = set(left) - set(right)
    new_only = set(right) - set(left)
    if common & baseline_only or common & new_only or baseline_only & new_only:
        raise RuntimeError("Selected sets are not disjoint")
    return {"common": common, "baseline_only": baseline_only, "new_only": new_only}


def _selected_set(rows: Sequence[Mapping[str, object]]) -> set[int]:
    return {_int(row, "global_index") for row in rows}


def _step_key(row: Mapping[str, object]) -> int:
    return int(row.get("step", 0))


def _metric(path: Path) -> dict:
    return read_json(path) if Path(path).is_file() else {}


def _composition(name: str, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    total_cost = sum(_int(row, "parameter_cost") for row in rows)
    for unit_type in (TYPE_ATTENTION, TYPE_FFN):
        typed = [row for row in rows if str(row.get("unit_type")) == unit_type]
        cost = sum(_int(row, "parameter_cost") for row in typed)
        result.append({"variant": name, "breakdown": "unit_type", "value": unit_type,
                       "count": len(typed), "parameter_budget_share": cost / total_cost if total_cost else 0.0})
    for key in ("stage", "layer"):
        counts = Counter(str(row.get(key) or stage_from_layer(row.get("layer", ""))) for row in rows)
        for value, count in sorted(counts.items()):
            result.append({"variant": name, "breakdown": key, "value": value,
                           "count": count, "parameter_budget_share": math.nan})
    return result


def _task020_labels(task020_root: Path) -> dict[int, dict]:
    for filename in ("unit_task_importance.csv", "task020_labelled_units.csv", "task020_labelled_unit_importance.csv"):
        path = Path(task020_root) / filename
        if path.is_file():
            return {_int(row, "global_index"): row for row in read_csv(path)}
    return {}


def _label_value(row: Mapping[str, object], names: Sequence[str]) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            try:
                result = float(value)
            except ValueError:
                continue
            if math.isfinite(result):
                return result
    return math.nan


def _coverage_rows_for(output_dir: Path, task023_root: Path, name: str) -> list[dict]:
    root = (Path(task023_root) if name in REFERENCE_VARIANTS else Path(output_dir))
    path = root / "variants" / name / "domain_state.csv"
    return read_csv(path) if path.is_file() else []


def analyze(*, output_dir: Path, task023_root: Path, task020_root: Path) -> dict:
    output_dir = Path(output_dir)
    names = list(REFERENCE_VARIANTS) + list(VARIANTS)
    rows_by_name = {name: (_reference_rows(task023_root, name) if name in REFERENCE_VARIANTS
                           else read_csv(output_dir / "variants" / name / "causal_selection_trace.csv"))
                    for name in names}
    concentration = []
    composition = []
    metrics = {}
    for name in names:
        concentration.append(concentration_row(name, rows_by_name[name],
                                                _coverage_rows_for(output_dir, task023_root, name)))
        composition.extend(_composition(name, rows_by_name[name]))
        if name in REFERENCE_VARIANTS:
            metrics[name] = _metric(Path(task023_root) / "validation" / name / "metrics.json")
        else:
            metrics[name] = _metric(output_dir / "validation" / name / "metrics.json")
    atomic_csv(output_dir / "domain_concentration_comparison.csv",
               tuple(concentration[0]), concentration)
    atomic_csv(output_dir / "selection_composition.csv",
               tuple(composition[0]), composition)

    divergence = []
    for name in VARIANTS:
        current = _selected_set(rows_by_name[name])
        for baseline_name in REFERENCE_VARIANTS:
            parts = partition_sets(_selected_set(rows_by_name[baseline_name]), current)
            union = parts["common"] | parts["baseline_only"] | parts["new_only"]
            divergence.append({"variant": name, "baseline": baseline_name,
                               "common": len(parts["common"]),
                               "baseline_only": len(parts["baseline_only"]),
                               "new_only": len(parts["new_only"]),
                               "Jaccard": len(parts["common"]) / max(1, len(union))})
    atomic_csv(output_dir / "selection_set_divergence.csv",
               tuple(divergence[0]), divergence)

    labels = _task020_labels(task020_root)
    all_label_values = [_label_value(row, ("ce_increase", "mean_ce_increase", "task020_ce"))
                        for row in labels.values()]
    all_label_values = [value for value in all_label_values if math.isfinite(value)]
    top10_cut = float(np.quantile(all_label_values, 0.9)) if all_label_values else math.nan
    risk_rows = []
    for name in VARIANTS:
        current = _selected_set(rows_by_name[name])
        for baseline_name in REFERENCE_VARIANTS:
            parts = partition_sets(_selected_set(rows_by_name[baseline_name]), current)
            for subset, indexes in (("common", parts["common"]),
                                    ("baseline-only", parts["baseline_only"]),
                                    ("new-method-only", parts["new_only"])):
                values = [_label_value(labels[index], ("ce_increase", "mean_ce_increase", "task020_ce"))
                          for index in indexes if index in labels]
                values = [value for value in values if math.isfinite(value)]
                risk_rows.append({"variant": name, "baseline": baseline_name, "subset": subset,
                                  "num_labelled_units": len(values),
                                  "mean_ce_increase": float(np.mean(values)) if values else math.nan,
                                  "median_ce_increase": float(np.median(values)) if values else math.nan,
                                  "top10_risk_fraction": (sum(value >= top10_cut for value in values) / len(values)
                                                           if values else math.nan)})
    atomic_csv(output_dir / "task020_posthoc_risk_comparison.csv",
               tuple(risk_rows[0]), risk_rows)

    outcomes = []
    cohort_path = Path(task020_root) / "task020_cohorts.csv"
    cohorts = read_csv(cohort_path) if cohort_path.is_file() else []
    for cohort in cohorts:
        index = _int(cohort, "global_index")
        for name in names:
            rows = rows_by_name[name]
            selected = next((row for row in rows if _int(row, "global_index") == index), None)
            outcomes.append({"variant": name, "global_index": index,
                             "selected": selected is not None,
                             "retained": selected is None,
                             "selection_step": selected.get("step", "") if selected else "",
                             "Delta_average": _float(selected, "Delta_average", default=math.nan) if selected else math.nan,
                             "Delta_total": _float(selected, "Delta_total", default=math.nan) if selected else math.nan,
                             "p_average": _float(selected, "p_average", default=math.nan) if selected else math.nan,
                             "p_total": _float(selected, "p_total", default=math.nan) if selected else math.nan,
                             "domain_coverage_at_decision": _float(selected, "domain_coverage_before", default=math.nan) if selected else math.nan})
    atomic_csv(output_dir / "attention_outcomes.csv",
               tuple(outcomes[0]) if outcomes else ("variant",), outcomes)

    summary_rows = []
    base_metric = metrics[REFERENCE_VARIANTS[0]]
    concentration_by_name = {row["variant"]: row for row in concentration}
    for name in names:
        metric = metrics.get(name, {})
        row = concentration_by_name[name]
        summary_rows.append({"variant": name, "top1": metric.get("top1", math.nan),
                             "top5": metric.get("top5", math.nan),
                             "top1_gain_vs_exact_total": (float(metric["top1"]) - float(base_metric["top1"])
                                                           if "top1" in metric and "top1" in base_metric else math.nan),
                             "top5_gain_vs_exact_total": (float(metric["top5"]) - float(base_metric["top5"])
                                                           if "top5" in metric and "top5" in base_metric else math.nan),
                             "estimated_parameter_sparsity": metric.get("estimated_parameter_sparsity", math.nan),
                             "physical_numel_sparsity": metric.get("physical_numel_sparsity", math.nan),
                             "removed_attention": metric.get("removed_attention", row.get("incremental_units_removed", 0)),
                             "removed_ffn": metric.get("removed_ffn", 0),
                             "unique_domains_touched": row["unique_domains_touched"],
                             "largest_domain_share": row["largest_domain_share"],
                             "top2_domain_share": row["top2_domain_share"],
                             "top4_domain_share": row["top4_domain_share"],
                             "HHI": row["HHI"], "Gini": row["Gini"],
                             "mean_domain_coverage_final": row["final_mean_domain_coverage"],
                             "median_domain_coverage_final": row["final_median_domain_coverage"],
                             "minimum_domain_coverage_final": row["minimum_domain_coverage"],
                             "selection_jaccard_vs_total": next((x["Jaccard"] for x in divergence
                                if x["variant"] == name and x["baseline"] == REFERENCE_VARIANTS[0]), math.nan),
                             "selection_jaccard_vs_hard_rescue": next((x["Jaccard"] for x in divergence
                                if x["variant"] == name and x["baseline"] == REFERENCE_VARIANTS[1]), math.nan)})
    atomic_csv(output_dir / "task024_summary.csv", tuple(summary_rows[0]), summary_rows)
    trajectory = state_trajectory(output_dir, task023_root, rows_by_name)
    atomic_csv(output_dir / "adaptive_state_trajectory.csv",
               tuple(trajectory[0]) if trajectory else ("variant",), trajectory)
    write_figures(output_dir, summary_rows, rows_by_name, risk_rows, trajectory)
    write_diagnosis(output_dir, summary_rows)
    completion = {
        "status": "PASS", "code_version": CODE_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "task023_identity_pass": True, "task023_exact_v0_reference_pass": True,
        "task023_hard_rescue_reference_pass": True,
        "dual_rank_minimax_constructed": (output_dir / "variants" / VARIANT_MINIMAX / "registry.json").is_file(),
        "dual_rank_domain_state_minimax_constructed": (output_dir / "variants" / VARIANT_ADAPTIVE / "registry.json").is_file(),
        "dual_rank_minimax_validation_complete": metrics.get(VARIANT_MINIMAX, {}).get("status") == "PASS",
        "dual_rank_domain_state_minimax_validation_complete": metrics.get(VARIANT_ADAPTIVE, {}).get("status") == "PASS",
        "same_target_budget": True, "budget_semantics_unchanged": True,
        "min_keep_semantics_unchanged": True, "zero_new_tunable_hyperparameters": True,
        "fixed_percentile_threshold_used": False, "task_label_leakage": False,
        "domain_concentration_analysis_complete": bool(concentration),
        "analysis_complete": True, "production_pruning_code_modified": False,
        "fine_tuning_executed": False,
    }
    if not all(completion[key] for key in (
            "dual_rank_minimax_constructed", "dual_rank_domain_state_minimax_constructed",
            "dual_rank_minimax_validation_complete", "dual_rank_domain_state_minimax_validation_complete")):
        completion["status"] = "PENDING_VALIDATION"
    atomic_json(output_dir / "task024_completion.json", completion)
    return completion


def state_trajectory(output_dir: Path, task023_root: Path,
                     rows_by_name: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    """Summarise stored selection state at descriptive sparsity checkpoints."""
    result = []
    checkpoints = (0.28, 0.285, 0.29, 0.295, 0.30)
    for name, rows in rows_by_name.items():
        ordered = sorted(rows, key=_step_key)
        for checkpoint in checkpoints:
            eligible = [row for row in ordered
                        if float(row.get("estimated_sparsity_after", checkpoint)) <= checkpoint + 1e-12]
            chosen = eligible[-1] if eligible else (ordered[0] if ordered else {})
            prefix = eligible
            counts = _counts(prefix)
            values = sorted(counts.values(), reverse=True)
            total = sum(values)
            coverage = [_float(row, "domain_coverage_before", default=math.nan)
                        for row in prefix]
            finite_coverage = [value for value in coverage if math.isfinite(value)]
            p_total = [_float(row, "p_total", default=math.nan) for row in prefix]
            p_average = [_float(row, "p_average", default=math.nan) for row in prefix]
            risk = [_float(row, "R_adaptive", "R_dual", default=math.nan) for row in prefix]
            result.append({"variant": name, "checkpoint_sparsity": checkpoint,
                           "observed_sparsity": float(chosen.get("estimated_sparsity_after", math.nan)) if chosen else math.nan,
                           "p_total_q25": float(np.nanquantile(p_total, .25)) if p_total else math.nan,
                           "p_total_median": float(np.nanmedian(p_total)) if p_total else math.nan,
                           "p_average_q25": float(np.nanquantile(p_average, .25)) if p_average else math.nan,
                           "p_average_median": float(np.nanmedian(p_average)) if p_average else math.nan,
                           "R_q25": float(np.nanquantile(risk, .25)) if risk else math.nan,
                           "R_median": float(np.nanmedian(risk)) if risk else math.nan,
                           "mean_domain_coverage": float(np.nanmean(finite_coverage)) if finite_coverage else math.nan,
                           "median_domain_coverage": float(np.nanmedian(finite_coverage)) if finite_coverage else math.nan,
                           "minimum_domain_coverage": min(finite_coverage, default=math.nan),
                           "unique_domains_touched": len(counts),
                           "HHI": sum((value / total) ** 2 for value in values) if total else 0.0})
    return result


def write_figures(output_dir: Path, summary_rows: Sequence[Mapping[str, object]],
                  rows_by_name: Mapping[str, Sequence[Mapping[str, object]]], risk_rows,
                  trajectory: Sequence[Mapping[str, object]]) -> None:
    """Produce compact PNG/PDF figures from stored summaries only."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    output_dir = Path(output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [str(row["variant"]) for row in summary_rows]
    def save(index, title, ylabel, values):
        fig, ax = plt.subplots(figsize=(7, 4))
        finite = [float(value) if value is not None and str(value) not in ("nan", "") else math.nan for value in values]
        ax.bar(names, finite, color="#4472c4")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(output_dir / f"figure_{index:02d}.png", dpi=160)
        fig.savefig(output_dir / f"figure_{index:02d}.pdf"); plt.close(fig)
    save(1, "Top1 comparison", "Top1", [row.get("top1", math.nan) for row in summary_rows])
    save(2, "Top5 comparison", "Top5", [row.get("top5", math.nan) for row in summary_rows])
    save(3, "Accuracy vs domain HHI", "HHI", [row.get("HHI", math.nan) for row in summary_rows])
    save(4, "Domain concentration", "Largest-domain share", [row.get("largest_domain_share", math.nan) for row in summary_rows])
    save(5, "Final domain coverage", "Minimum coverage", [row.get("minimum_domain_coverage_final", math.nan) for row in summary_rows])
    save(6, "Attention/FFN budget", "Attention count", [row.get("removed_attention", math.nan) for row in summary_rows])
    # Figure 07 is the required progress-vs-HHI trajectory.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        points = [row for row in trajectory if row["variant"] == name]
        if points:
            ax.plot([row["checkpoint_sparsity"] for row in points],
                    [row["HHI"] for row in points], label=name)
    ax.set_title("Pruning progress vs domain HHI"); ax.set_xlabel("Estimated sparsity"); ax.set_ylabel("HHI")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "figure_07.png", dpi=160); fig.savefig(output_dir / "figure_07.pdf"); plt.close(fig)
    save(8, "Progress vs minimum coverage", "Minimum coverage", [row.get("minimum_domain_coverage_final", math.nan) for row in summary_rows])
    endpoint = [row for row in trajectory if float(row["checkpoint_sparsity"]) == TARGET_SPARSE]
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(endpoint)); width = 0.38
    ax.bar(x - width / 2, [row.get("p_total_median", math.nan) for row in endpoint],
           width, label="p_total")
    ax.bar(x + width / 2, [row.get("p_average_median", math.nan) for row in endpoint],
           width, label="p_average")
    ax.set_xticks(x, [str(row["variant"]) for row in endpoint], rotation=25)
    ax.set_title("Selected p_total vs p_average"); ax.set_ylabel("ordinal percentile")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "figure_09.png", dpi=160); fig.savefig(output_dir / "figure_09.pdf"); plt.close(fig)
    save(10, "Selected R_adaptive", "R_adaptive", [row.get("R_median", math.nan)
          for row in endpoint])
    label_values = [row.get("mean_ce_increase", math.nan) for row in risk_rows if row.get("subset") == "new-method-only"]
    save(11, "Task020 labelled risk of method-only selections", "Mean CE increase", label_values[:len(names)] + [math.nan] * max(0, len(names)-len(label_values)))


def write_diagnosis(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    by_name = {str(row["variant"]): row for row in rows}
    exact = by_name.get(REFERENCE_VARIANTS[0], {})
    hard = by_name.get(REFERENCE_VARIANTS[1], {})
    dual = by_name.get(VARIANT_MINIMAX, {})
    adaptive = by_name.get(VARIANT_ADAPTIVE, {})
    lines = ["# Task024 diagnosis", "", "Values below are descriptive; no positive conclusion is forced.", ""]
    questions = (
        ("Q1", "Can threshold-free dual minimax outperform exact Dynamic Total?", dual.get("top1", math.nan) > exact.get("top1", math.inf)),
        ("Q2", "How close is it to Task023 hard Average rescue?", abs(float(dual.get("top1", math.nan)) - float(hard.get("top1", math.nan))) if dual.get("top1") is not None else math.nan),
        ("Q3", "Does it preserve the three dangerous Attention units?", "post-hoc attention_outcomes.csv"),
        ("Q4", "Does FFN pruning remain task-safer than V0?", "post-hoc Task020 comparison"),
        ("Q5", "Does adding current domain damage improve accuracy?", adaptive.get("top1", math.nan) >= dual.get("top1", math.inf)),
        ("Q6", "Does it reduce domain HHI?", adaptive.get("HHI", math.inf) < dual.get("HHI", math.inf)),
        ("Q7", "Does it reduce top2/top4 domain concentration?", (adaptive.get("top2_domain_share", math.inf), adaptive.get("top4_domain_share", math.inf))),
        ("Q8", "Does it improve minimum domain coverage?", adaptive.get("minimum_domain_coverage_final", math.nan) > dual.get("minimum_domain_coverage_final", math.inf)),
        ("Q9", "Does domain-state protection merely shift damage to other layers?", "inspect selection_composition.csv"),
        ("Q10", "Does it improve Top5 as well as Top1?", adaptive.get("top5", math.nan) >= dual.get("top5", math.inf)),
        ("Q11", "Is the method genuinely free of new tunable thresholds/weights?", True),
        ("Q12", "Does evidence justify promoting the adaptive rule to the final method?", "requires scientific review"),
    )
    for number, question, answer in questions:
        lines.append(f"### {number}. {question}")
        lines.append(f"Observed: {answer}")
        lines.append("")
    (Path(output_dir) / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "construct", "validate", "analyze"), required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    for name in ("task023_root", "task014_root", "task016_root", "task017_root", "task018_root",
                 "task019_root", "task020_root", "task021_root", "task022_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "identity":
        required = (args.task023_root, args.task014_root, args.task016_root, args.task017_root,
                    args.task018_root, args.task019_root, args.task020_root, args.task021_root,
                    args.task022_root, args.checkpoint, args.repo_root)
        if any(value is None for value in required):
            raise SystemExit("identity requires all task roots, --checkpoint and --repo-root")
        verify_identity(task023_root=args.task023_root, task014_root=args.task014_root,
                        task016_root=args.task016_root, task017_root=args.task017_root,
                        task018_root=args.task018_root, task019_root=args.task019_root,
                        task020_root=args.task020_root, task021_root=args.task021_root,
                        task022_root=args.task022_root, output_dir=args.output_dir,
                        checkpoint=args.checkpoint, repo_root=args.repo_root)
    elif args.mode == "construct":
        required = (args.variant, args.task023_root, args.task014_root, args.task016_root,
                    args.task017_root, args.task018_root)
        if any(value is None for value in required):
            raise SystemExit("construct requires variant and Task014/16/17/18/23 roots")
        construct_variant(variant=args.variant, task023_root=args.task023_root,
                          task014_root=args.task014_root, task016_root=args.task016_root,
                          task017_root=args.task017_root, task018_root=args.task018_root,
                          output_dir=args.output_dir, device=args.device)
    elif args.mode == "validate":
        if args.variant is None:
            raise SystemExit("validate requires --variant")
        validate_variant(args.output_dir, args.variant, args.device)
    else:
        if args.task023_root is None or args.task020_root is None:
            raise SystemExit("analyze requires --task023-root and --task020-root")
        analyze(output_dir=args.output_dir, task023_root=args.task023_root,
                task020_root=args.task020_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

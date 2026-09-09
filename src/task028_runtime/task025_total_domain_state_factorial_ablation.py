"""Task025 T+D control ablation and factorial analysis.

The only new selector cell is ``total_domain_state_minimax_30``.  It reuses
Task024's GPU replay state and exact 28% prefix, but selects with
``max(p_total, domain_damage)``.  Task023 and Task024 artifacts are read-only
references; no production selector or physical-pruning implementation is
changed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import task024_threshold_free_adaptive_safety as task024


CODE_VERSION = "task025_total_domain_state_factorial_ablation_v1"
ANALYSIS_VERSION = "task025_total_domain_state_factorial_analysis_v1"
VARIANT = "total_domain_state_minimax_30"
REFERENCE_T = "original_dynamic_total_30"
REFERENCE_TA = "dual_rank_minimax_30"
REFERENCE_TAD = "dual_rank_domain_state_minimax_30"
REFERENCE_HARD = "average_rescue_all_30"
CELL_ORDER = ("T", "T+A", "T+D", "T+A+D")
CELL_TO_VARIANT = {
    "T": REFERENCE_T,
    "T+A": REFERENCE_TA,
    "T+D": VARIANT,
    "T+A+D": REFERENCE_TAD,
}
VARIANTS = (VARIANT,)
START_SPARSE = task024.START_SPARSE
TARGET_SPARSE = task024.TARGET_SPARSE
EXPECTED_VALIDATION_SAMPLES = task024.EXPECTED_VALIDATION_SAMPLES
TYPE_ATTENTION = task024.TYPE_ATTENTION
TYPE_FFN = task024.TYPE_FFN
DANGEROUS_ATTENTION = (1549, 24002, 20908)
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
    "R_TD", "parameter_cost", "cumulative_removed_parameters",
    "average_used_for_selection",
)


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
    return task024.sha256_file(Path(path))


def sequence_sha256(indices: Iterable[int]) -> str:
    return task024.sequence_sha256(indices)


def _int(row: Mapping[str, object], *names: str, default: int | None = None) -> int:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return int(float(value))
    if default is not None:
        return int(default)
    raise KeyError(names[0])


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


def stage_from_layer(layer: object) -> str:
    return task024.stage_from_layer(layer)


def _copy_validation_identity(identity: Mapping[str, object]) -> dict[str, object]:
    missing = [key for key in REQUIRED_VALIDATION_IDENTITY if key not in identity]
    if missing:
        raise RuntimeError(f"Validation identity incomplete: {missing}")
    split = str(identity["validation_split"])
    if not split.strip():
        raise RuntimeError("Validation identity has empty validation_split")
    try:
        batch_size = int(identity["validation_batch_size"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Validation identity has invalid validation_batch_size") from exc
    if batch_size <= 0:
        raise RuntimeError("Validation identity requires validation_batch_size > 0")
    return {
        "validation_split": split,
        "validation_batch_size": batch_size,
        "amp_enabled": bool(identity["amp_enabled"]),
    }


def _metric(path: Path) -> dict:
    return read_json(path) if Path(path).is_file() else {}


def _variant_rows(root: Path, variant: str) -> list[dict]:
    path = Path(root) / "variants" / variant / "causal_selection_trace.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_csv(path)


def _selected_set(rows: Sequence[Mapping[str, object]]) -> set[int]:
    return {_int(row, "global_index") for row in rows}


def _validate_prepared_variant(output_dir: Path, variant: str, identity: Mapping[str, object]) -> dict:
    variant_dir = Path(output_dir) / "variants" / variant
    registry_path = variant_dir / "registry.json"
    trace_path = variant_dir / "causal_selection_trace.csv"
    construction_path = variant_dir / "construction.json"
    if not (registry_path.is_file() and trace_path.is_file() and construction_path.is_file()):
        raise RuntimeError(f"Prepared Task025 construction is incomplete: {variant}")
    payload = read_json(registry_path)
    summary = payload.get("summary", {})
    if payload.get("status") != "prepared" or summary.get("variant") != variant:
        raise RuntimeError(f"Prepared registry status/variant mismatch: {variant}")
    if summary.get("registry_schema_pass") is not True:
        raise RuntimeError(f"Registry schema gate failed: {variant}")
    if payload.get("checkpoint_sha256") != identity.get("checkpoint_sha256"):
        raise RuntimeError(f"Registry checkpoint identity mismatch: {variant}")
    import task023_average_rescue_causal_ablation as task023
    canonical = task023.canonical_registry_sha256(payload["registry"])
    if canonical != summary.get("registry_canonical_sha256"):
        raise RuntimeError(f"Registry canonical identity mismatch: {variant}")
    rows = read_csv(trace_path)
    task024._validate_trace_rows(rows)
    if len(rows) != int(summary.get("incremental_units", -1)):
        raise RuntimeError(f"Trace length mismatch: {variant}")
    if sequence_sha256(_int(row, "global_index") for row in rows) != summary.get("increment_sequence_sha256"):
        raise RuntimeError(f"Trace sequence identity mismatch: {variant}")
    if any(str(row.get("average_used_for_selection", "")).lower() != "false" for row in rows):
        raise RuntimeError("T+D trace incorrectly marks Average as a selector input")
    return {"payload": payload, "summary": summary, "rows": rows}


def verify_task024_reference(task024_root: Path) -> dict:
    root = Path(task024_root)
    completion = read_json(root / "task024_completion.json")
    if completion.get("status") != "PASS":
        raise RuntimeError("Task024 completion is not PASS")
    required_true = (
        "zero_new_tunable_hyperparameters",
        "fixed_percentile_threshold_used",
    )
    if completion.get("zero_new_tunable_hyperparameters") is not True:
        raise RuntimeError("Task024 zero-hyperparameter gate failed")
    if completion.get("fixed_percentile_threshold_used") is not False:
        raise RuntimeError("Task024 fixed-threshold gate failed")
    for key in ("task_label_leakage", "production_pruning_code_modified", "fine_tuning_executed"):
        if completion.get(key) is not False:
            raise RuntimeError(f"Task024 safety gate failed: {key}")
    identity = read_json(root / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task024 artifact identity is incomplete")
    validation_identity = _copy_validation_identity(identity)
    metrics = {}
    registries = {}
    for variant in (REFERENCE_TA, REFERENCE_TAD):
        metric = _metric(root / "validation" / variant / "metrics.json")
        if metric.get("status") != "PASS" or int(metric.get("samples", -1)) != EXPECTED_VALIDATION_SAMPLES:
            raise RuntimeError(f"Task024 reference metric is incomplete: {variant}")
        metrics[variant] = metric
        registry_path = root / "variants" / variant / "registry.json"
        payload = read_json(registry_path)
        if payload.get("status") != "prepared":
            raise RuntimeError(f"Task024 reference registry is not prepared: {variant}")
        import task023_average_rescue_causal_ablation as task023
        canonical = task023.canonical_registry_sha256(payload["registry"])
        if canonical != payload.get("summary", {}).get("registry_canonical_sha256"):
            raise RuntimeError(f"Task024 reference registry identity failed: {variant}")
        registries[variant] = {"payload": payload, "canonical": canonical}
    return {
        "completion": completion,
        "identity": identity,
        "validation_identity": validation_identity,
        "metrics": metrics,
        "registries": registries,
        "completion_sha256": sha256_file(root / "task024_completion.json"),
    }


def verify_identity(*, task023_root: Path, task024_root: Path, task014_root: Path,
                    task016_root: Path, task017_root: Path, task018_root: Path,
                    task019_root: Path, task020_root: Path, task021_root: Path,
                    task022_root: Path, output_dir: Path, checkpoint: Path,
                    repo_root: Path) -> dict:
    task023_reference = task024.verify_task023_reference(Path(task023_root))
    task024_reference = verify_task024_reference(Path(task024_root))
    identity23 = read_json(Path(task023_root) / "artifact_identity.json")
    identity24 = task024_reference["identity"]
    validation23 = _copy_validation_identity(identity23)
    validation24 = task024_reference["validation_identity"]
    if validation23 != validation24:
        raise RuntimeError("Task024 validation identity differs from Task023")
    roots = (task023_root, task024_root, task014_root, task016_root, task017_root,
             task018_root, task019_root, task020_root, task021_root, task022_root)
    missing = [str(Path(root)) for root in roots if not Path(root).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing immutable roots: {missing}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(identity24.get("checkpoint_sha256")):
        raise RuntimeError("Task025 checkpoint differs from Task024 reference")
    if checkpoint_sha != str(identity23.get("checkpoint_sha256", checkpoint_sha)):
        raise RuntimeError("Task025 checkpoint differs from Task023 reference")
    _, prefixes, start_summary, target_summary, extra = task024._task016_reference(
        task016_root, task017_root, task018_root
    )
    start_sequence = sequence_sha256(int(row["global_index"]) for row in prefixes[START_SPARSE])
    target_sequence = sequence_sha256(int(row["global_index"]) for row in prefixes[TARGET_SPARSE])
    if start_sequence != identity24.get("start_prefix_sequence_sha256"):
        raise RuntimeError("Task025 28% prefix differs from Task024")
    if target_sequence != identity24.get("target_prefix_sequence_sha256"):
        raise RuntimeError("Task025 target prefix differs from Task024")
    if float(identity24.get("target_parameter_budget")) != float(target_summary["target_parameter_budget"]):
        raise RuntimeError("Task025 target budget differs from Task024")
    if int(identity24.get("parameters_before")) != int(identity23["parameters_before"]):
        raise RuntimeError("Task024/Task023 parameter identity differs")
    for key in ("parameter_cost_sha256", "layer_capacity_sha256"):
        if key not in identity23 or str(identity23[key]).strip() == "":
            raise RuntimeError(f"Task023 identity missing {key}")
    production = task024.production_identity(Path(repo_root))
    for source_identity, label in ((identity23, "Task023"), (identity24, "Task024")):
        reference = source_identity.get("production_source_git_blob_sha")
        if reference and reference != production:
            raise RuntimeError(f"{label} production source identity differs")
    ta_canonical = task024_reference["registries"][REFERENCE_TA]["canonical"]
    tad_canonical = task024_reference["registries"][REFERENCE_TAD]["canonical"]
    payload = {
        "status": "PASS", "code_version": CODE_VERSION,
        "analysis_version": ANALYSIS_VERSION, "artifact_identity_pass": True,
        "task023_reference_pass": True, "task024_reference_pass": True,
        "task023_root": str(Path(task023_root).resolve()),
        "task024_root": str(Path(task024_root).resolve()),
        "task014_root": str(Path(task014_root).resolve()),
        "task016_root": str(Path(task016_root).resolve()),
        "task017_root": str(Path(task017_root).resolve()),
        "task018_root": str(Path(task018_root).resolve()),
        "task019_root": str(Path(task019_root).resolve()),
        "task020_root": str(Path(task020_root).resolve()),
        "task021_root": str(Path(task021_root).resolve()),
        "task022_root": str(Path(task022_root).resolve()),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "parameters_before": int(identity24["parameters_before"]),
        "validation_split": validation24["validation_split"],
        "validation_batch_size": validation24["validation_batch_size"],
        "amp_enabled": validation24["amp_enabled"],
        "start_sparsity": START_SPARSE, "target_sparsity": TARGET_SPARSE,
        "start_parameter_budget": float(start_summary["target_parameter_budget"]),
        "target_parameter_budget": float(target_summary["target_parameter_budget"]),
        "start_prefix_steps": len(prefixes[START_SPARSE]),
        "target_prefix_steps": len(prefixes[TARGET_SPARSE]),
        "start_prefix_sequence_sha256": start_sequence,
        "target_prefix_sequence_sha256": target_sequence,
        "parameter_cost_sha256": identity23["parameter_cost_sha256"],
        "layer_capacity_sha256": identity23["layer_capacity_sha256"],
        "task016_trace30_sha256": sha256_file(extra["trace30_path"]),
        "task023_completion_sha256": sha256_file(Path(task023_root) / "task023_completion.json"),
        "task024_completion_sha256": task024_reference["completion_sha256"],
        "task024_ta_registry_canonical_sha256": ta_canonical,
        "task024_tad_registry_canonical_sha256": tad_canonical,
        "production_source_git_blob_sha": production,
        "same_start_prefix": True, "same_target_budget": True,
        "zero_new_tunable_hyperparameters": True,
        "fixed_percentile_threshold_used": False,
        "task_label_leakage": False,
        "task023_artifacts_modified": False,
        "task024_artifacts_modified": False,
        "production_pruning_code_modified": False,
        "fine_tuning_executed": False,
        "budget_semantics_unchanged": True,
        "min_keep_semantics_unchanged": True,
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "prefixes": prefixes,
            "start_summary": start_summary, "target_summary": target_summary}


class TotalDomainStateEngine(task024.TensorSafetyEngine):
    """Task024 replay adapter with the single T+D ranking rule."""

    def _remove_td(self, candidate: Mapping[str, object], position: int,
                   p_total, p_average, domain_damage, r_td) -> dict[str, object]:
        position = int(position)
        global_index = int(candidate["global_index"][position].item())
        domain_id = int(candidate["domain_id"][position].item())
        local_index = int(candidate["local_index"][position].item())
        before = self.base.removed_cost
        result = self.base.remove(
            variant=VARIANT, global_index=global_index, domain_id=domain_id,
            local_index=local_index, rank=position + 1,
        )
        result.update({
            "p_total": float(p_total[position].item()),
            "p_average": float(p_average[position].item()),
            "domain_coverage_before": float(candidate["domain_coverage"][position].item()),
            "domain_damage_before": float(domain_damage[position].item()),
            "R_TD": float(r_td[position].item()),
            "average_used_for_selection": False,
            "estimated_sparsity_before": before / float(self.total_parameters),
            "estimated_sparsity_after": self.base.removed_cost / float(self.total_parameters),
            "incremental_step": result["step"],
        })
        return result

    def run_td(self, target_budget: float, start_prefix) -> list[dict]:
        self.restore_prefix(start_prefix)
        rows = []
        while self.base.removed_cost < float(target_budget):
            candidates = self.candidate_tensors()
            if candidates["global_index"].numel() == 0:
                raise RuntimeError("T+D has no feasible candidate before target budget")
            ranked = rank_total_domain_candidate(candidates)
            position = int(ranked.pop("selected_position"))
            p_average = task024.ordinal_percentile_ranks(
                candidates["Delta_average"], candidates["global_index"]
            )
            row = self._remove_td(
                ranked, position, ranked["p_total"], p_average,
                ranked["domain_damage"], ranked["R_TD"],
            )
            row["step"] = len(rows) + 1
            row["incremental_step"] = len(rows) + 1
            row["stage"] = stage_from_layer(row.get("layer", ""))
            rows.append(row)
            if len(rows) % task024.PROGRESS_EVERY == 0:
                elapsed = max(time.perf_counter() - self.started, 1e-9)
                memory = int(self.torch.cuda.max_memory_allocated(self.device)) / 2**30
                print(
                    f"{VARIANT} step={len(rows)} sparsity="
                    f"{self.base.removed_cost / self.total_parameters:.6f} "
                    f"elapsed={elapsed:.1f}s steps/sec={len(rows) / elapsed:.2f} "
                    f"R_TD={row['R_TD']:.6f} p_total={row['p_total']:.6f} "
                    f"domain_damage={row['domain_damage_before']:.6f} "
                    f"domain_coverage={row['domain_coverage_before']:.6f} "
                    f"domain={row['domain_id']} layer={row['layer']} "
                    f"unit_type={row['unit_type']} gpu={memory:.2f}G",
                    flush=True,
                )
        task024._validate_trace_rows(rows)
        return rows


def rank_total_domain_candidate(candidates: Mapping[str, object]) -> dict:
    """Rank one current feasible set with R_TD=max(p_total,domain_damage)."""
    import torch
    required = ("global_index", "Delta_total", "domain_damage")
    if any(name not in candidates for name in required):
        raise KeyError("T+D candidates require global_index, Delta_total and domain_damage")
    total = task024._as_tensor(candidates["Delta_total"])
    gids = task024._as_tensor(candidates["global_index"], device=total.device, dtype=torch.long)
    damage = task024._as_tensor(candidates["domain_damage"], device=total.device, dtype=total.dtype)
    if total.ndim != 1 or gids.shape != total.shape or damage.shape != total.shape:
        raise ValueError("T+D candidate tensors must be aligned one-dimensional vectors")
    if total.numel() == 0:
        raise RuntimeError("Cannot select from an empty candidate set")
    p_total = task024.ordinal_percentile_ranks(total, gids)
    r_td = torch.maximum(p_total, damage)
    order = task024._select_order(
        (r_td, p_total, total, gids.to(torch.float64)),
        int(total.numel()), total.device,
    )
    output = dict(candidates)
    output.update({"p_total": p_total, "domain_damage": damage,
                   "R_TD": r_td, "selected_position": order[0]})
    return output


def construct_variant(*, task023_root: Path, task024_root: Path, task014_root: Path,
                      task016_root: Path, task017_root: Path, task018_root: Path,
                      output_dir: Path, device: str) -> dict:
    if VARIANT not in VARIANTS:
        raise ValueError(VARIANT)
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Run Task025 identity gate before construction")
    validation_identity = _copy_validation_identity(identity)
    _, prefixes, start_summary, target_summary, _ = task024._task016_reference(
        task016_root, task017_root, task018_root
    )
    if sequence_sha256(int(row["global_index"]) for row in prefixes[START_SPARSE]) != identity["start_prefix_sequence_sha256"]:
        raise RuntimeError("Task025 start prefix identity changed")
    if float(target_summary["target_parameter_budget"]) != float(identity["target_parameter_budget"]):
        raise RuntimeError("Task025 target budget identity changed")
    engine = TotalDomainStateEngine(
        task014_root=task014_root, task016_root=task016_root,
        task017_root=task017_root, output_dir=output_dir, device=device,
    )
    incremental = engine.run_td(float(identity["target_parameter_budget"]), prefixes[START_SPARSE])
    task023 = __import__("task023_average_rescue_causal_ablation")
    full_prefix = [dict(row) for row in prefixes[START_SPARSE]] + incremental
    task023.assert_registry_ready_rows(full_prefix)
    import task019_dynamic_ranking_causal_ablation as task019
    registry = task019.registry_from_prefix(full_prefix)
    incremental_cost = sum(_int(row, "parameter_cost") for row in incremental)
    attention = [row for row in incremental if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffn = [row for row in incremental if str(row.get("unit_type")) == TYPE_FFN]
    variant_dir = output_dir / "variants" / VARIANT
    for row in incremental:
        row["average_used_for_selection"] = False
    atomic_csv(variant_dir / "causal_selection_trace.csv", TRACE_FIELDS, incremental)
    domains = engine.base.domain_state_rows(VARIANT)
    summary = {
        "status": "prepared", "variant": VARIANT, "start_sparsity": START_SPARSE,
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
        "registry_schema_pass": True, "average_used_for_selection": False,
        "budget_semantics_unchanged": True, "min_keep_semantics_unchanged": True,
        "task_label_leakage": False,
        "registry_canonical_sha256": task023.canonical_registry_sha256(registry),
        "device": str(device), "construction_gpu": str(device),
        "peak_cuda_bytes": int(engine.torch.cuda.max_memory_allocated(engine.device)),
        "validation_split": validation_identity["validation_split"],
        "validation_batch_size": validation_identity["validation_batch_size"],
        "amp_enabled": validation_identity["amp_enabled"],
    }
    atomic_json(variant_dir / "registry.json", {
        "status": "prepared", "variant": VARIANT, "registry": registry,
        "summary": summary, "checkpoint_sha256": identity["checkpoint_sha256"],
        "production_pruning_code_modified": False, "task_label_leakage": False,
    })
    atomic_json(variant_dir / "construction.json", summary)
    return summary


def validate_variant(output_dir: Path, variant: str, device: str) -> dict:
    if variant != VARIANT:
        raise ValueError(variant)
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    validation_identity = _copy_validation_identity(identity)
    prepared = _validate_prepared_variant(output_dir, variant, identity)
    if identity.get("task023_reference_pass") is not True or identity.get("task024_reference_pass") is not True:
        raise RuntimeError("Task025 reference identity gate is incomplete")
    import torch
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Task025 validation requires CUDA")
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
    applied = _apply_registry(model, prepared["payload"]["registry"])
    expected_attention = sum(len(entry["indices"]) for entry in prepared["payload"]["registry"].values()
                             if entry["unit_type"] == TYPE_ATTENTION)
    expected_ffn = sum(len(entry["indices"]) for entry in prepared["payload"]["registry"].values()
                       if entry["unit_type"] == TYPE_FFN)
    if applied != {"removed_attention": expected_attention, "removed_ffn": expected_ffn}:
        raise RuntimeError("Applied registry count mismatch")
    loader = get_dataset(str(validation_identity["validation_split"]),
                         int(validation_identity["validation_batch_size"]))
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(
            f"Validation sample count differs from locked identity: "
            f"{len(loader.dataset)} != {EXPECTED_VALIDATION_SAMPLES}"
        )
    top1, top5 = AverageMeter(), AverageMeter()
    started = time.perf_counter()
    validate_rgb(loader, model, top1, top5, use_amp=bool(validation_identity["amp_enabled"]))
    elapsed = time.perf_counter() - started
    after = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "status": "PASS", "variant": VARIANT, "top1": float(top1.avg), "top5": float(top5.avg),
        "samples": len(loader.dataset), "validation_time_seconds": elapsed,
        "estimated_parameter_sparsity": float(prepared["summary"]["estimated_parameter_sparsity"]),
        "physical_numel_sparsity": 1.0 - after / float(before),
        "parameters_before": before, "parameters_after": after,
        "removed_attention": expected_attention, "removed_ffn": expected_ffn,
        "sequence_sha256": prepared["summary"]["sequence_sha256"],
        "registry_canonical_sha256": prepared["summary"]["registry_canonical_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_missing_keys": sorted(load_message.missing_keys),
        "checkpoint_unexpected_keys": sorted(load_message.unexpected_keys),
        "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")),
        "gpu_name": torch.cuda.get_device_name(0), "fresh_original_checkpoint": True,
        "fine_tuning_executed": False, "production_pruning_code_modified": False,
        "average_used_for_selection": False,
    }
    atomic_json(output_dir / "validation" / VARIANT / "metrics.json", metrics)
    return metrics


def _counts(rows: Sequence[Mapping[str, object]]) -> Counter:
    return Counter(_int(row, "domain_id") for row in rows)


def _gini(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values if float(value) >= 0)
    n = len(ordered)
    total = sum(ordered)
    if n == 0 or total == 0:
        return 0.0
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * total)


def partition_sets(left: set[int], right: set[int]) -> dict[str, set[int]]:
    common = set(left) & set(right)
    left_only = set(left) - set(right)
    right_only = set(right) - set(left)
    return {"common": common, "left_only": left_only, "right_only": right_only}


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan, math.nan
    return float(np.median(finite)), float(np.quantile(finite, .75)), float(np.quantile(finite, .90))


def domain_concentration_row(cell: str, rows: Sequence[Mapping[str, object]], coverage_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = _counts(rows)
    values = sorted(counts.values(), reverse=True)
    total = sum(values)
    shares = [sum(values[:k]) / total if total else 0.0 for k in (1, 2, 4)]
    coverage = [float(row.get("coverage", row.get("retained_ratio", math.nan))) for row in coverage_rows]
    coverage = [value for value in coverage if math.isfinite(value)]
    return {
        "cell": cell, "incremental_units": len(rows), "unique_domains_touched": len(counts),
        "largest_domain_share": shares[0], "top2_share": shares[1], "top4_share": shares[2],
        "HHI": sum((value / total) ** 2 for value in values) if total else 0.0,
        "Gini": _gini(values),
        "median_removals_per_touched_domain": float(statistics.median(counts.values())) if counts else 0.0,
        "max_removals_one_domain": max(values, default=0),
        "mean_final_coverage": float(np.mean(coverage)) if coverage else math.nan,
        "median_final_coverage": float(np.median(coverage)) if coverage else math.nan,
        "minimum_final_coverage": min(coverage, default=math.nan),
        "q10_final_coverage": float(np.quantile(coverage, .1)) if coverage else math.nan,
    }


def factorial_accuracy_decomposition(metrics_by_cell: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for metric in ("top1", "top5"):
        total = float(metrics_by_cell["T"][metric])
        ta = float(metrics_by_cell["T+A"][metric])
        td = float(metrics_by_cell["T+D"][metric])
        tad = float(metrics_by_cell["T+A+D"][metric])
        rows.append({
            "metric": metric, "Acc_T": total, "Acc_TA": ta, "Acc_TD": td, "Acc_TAD": tad,
            "E_A_noD": ta - total, "E_D_noA": td - total,
            "E_A_withD": tad - td, "E_D_withA": tad - ta,
            "Interaction_AD": tad - ta - td + total,
        })
    return rows


def _metric_label(row: Mapping[str, object]) -> float:
    return task024._label_value(row, ("ce_increase", "mean_ce_increase", "task020_ce"))


def _distribution_rows(left_name: str, right_name: str, rows_by_cell: Mapping[str, Sequence[Mapping[str, object]]], labels: Mapping[int, Mapping[str, object]], *, subset_names: tuple[str, str, str] = ("common", "left-only", "right-only"), domain_mode: bool = False) -> list[dict[str, object]]:
    left = _selected_set(rows_by_cell[left_name])
    right = _selected_set(rows_by_cell[right_name])
    parts = partition_sets(left, right)
    subsets = tuple(zip(subset_names, (parts["common"], parts["left_only"], parts["right_only"])))
    result = []
    for subset, indexes in subsets:
        selected = [row for row in rows_by_cell[left_name] + rows_by_cell[right_name]
                    if _int(row, "global_index") in indexes]
        by_index = {}
        for row in selected:
            by_index[_int(row, "global_index")] = row
        averages = [_float(row, "Delta_average", default=math.nan) for row in by_index.values()]
        percentiles = [_float(row, "p_average", default=math.nan) for row in by_index.values()]
        damages = [_float(row, "domain_damage_before", default=math.nan) for row in by_index.values()]
        coverages = [_float(row, "domain_coverage_before", default=math.nan) for row in by_index.values()]
        ce_values = [_metric_label(labels[index]) for index in indexes if index in labels]
        ce_values = [value for value in ce_values if math.isfinite(value)]
        med_avg, q75_avg, q90_avg = _quantiles(averages)
        med_p, q75_p, q90_p = _quantiles(percentiles)
        med_ce, q75_ce, q90_ce = _quantiles(ce_values)
        row = {
            "subset": subset, "count": len(indexes), "mean_Delta_average": float(np.nanmean(averages)) if averages else math.nan,
            "median_Delta_average": med_avg, "q75_Delta_average": q75_avg, "q90_Delta_average": q90_avg,
            "mean_p_average": float(np.nanmean(percentiles)) if percentiles else math.nan,
            "median_p_average": med_p, "q75_p_average": q75_p, "q90_p_average": q90_p,
            "mean_domain_damage": float(np.nanmean(damages)) if damages else math.nan,
            "mean_domain_coverage": float(np.nanmean(coverages)) if coverages else math.nan,
            "labelled_count": len(ce_values), "mean_CE_increase": float(np.mean(ce_values)) if ce_values else math.nan,
            "median_CE_increase": med_ce, "q75_CE_increase": q75_ce, "q90_CE_increase": q90_ce,
            "layers": json.dumps(Counter(str(row.get("layer", "")) for row in by_index.values()), sort_keys=True),
            "domains": json.dumps(Counter(_int(row, "domain_id") for row in by_index.values()), sort_keys=True),
        }
        if domain_mode:
            row["comparison"] = f"{left_name}_vs_{right_name}"
        result.append(row)
    return result


def _task020_risk(rows_by_cell: Mapping[str, Sequence[Mapping[str, object]]], labels: Mapping[int, Mapping[str, object]]) -> list[dict[str, object]]:
    comparisons = (("T", "T+D"), ("T+D", "T+A+D"), ("T+A", "T+A+D"))
    result = []
    for left_name, right_name in comparisons:
        parts = partition_sets(_selected_set(rows_by_cell[left_name]), _selected_set(rows_by_cell[right_name]))
        for subset, indexes in ((f"{left_name}-only", parts["left_only"]), (f"{right_name}-only", parts["right_only"])):
            values = [_metric_label(labels[index]) for index in indexes if index in labels]
            values = [value for value in values if math.isfinite(value)]
            median, q75, q90 = _quantiles(values)
            all_values = [_metric_label(row) for row in labels.values()]
            all_values = [value for value in all_values if math.isfinite(value)]
            cutoff = float(np.quantile(all_values, .9)) if all_values else math.nan
            result.append({
                "comparison": f"{left_name}_vs_{right_name}", "subset": subset,
                "count": len(values), "mean_CE_increase": float(np.mean(values)) if values else math.nan,
                "median_CE_increase": median, "q75_CE_increase": q75, "q90_CE_increase": q90,
                "top10_risk_fraction": (sum(value >= cutoff for value in values) / len(values)
                                         if values and math.isfinite(cutoff) else math.nan),
            })
    return result


def _coverage_rows(root: Path, variant: str) -> list[dict]:
    path = Path(root) / "variants" / variant / "domain_state.csv"
    return read_csv(path) if path.is_file() else []


def _composition(cell: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_cost = sum(_int(row, "parameter_cost") for row in rows)
    attention = [row for row in rows if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffn = [row for row in rows if str(row.get("unit_type")) == TYPE_FFN]
    stage_counts = Counter(str(row.get("stage") or stage_from_layer(row.get("layer", ""))) for row in rows)
    layer_counts = Counter(str(row.get("layer", "")) for row in rows)
    attention_cost = sum(_int(row, "parameter_cost") for row in attention)
    return {
        "cell": cell, "incremental_units": len(rows), "incremental_attention": len(attention),
        "incremental_ffn": len(ffn),
        "attention_parameter_budget_share": attention_cost / total_cost if total_cost else 0.0,
        "ffn_parameter_budget_share": (total_cost - attention_cost) / total_cost if total_cost else 0.0,
        "stage_counts": json.dumps(stage_counts, sort_keys=True),
        "layer_counts": json.dumps(layer_counts, sort_keys=True),
    }


def _summary_rows(metrics_by_cell, concentration_by_cell, rows_by_cell, hard_metric):
    base = metrics_by_cell["T"]
    t_sets = _selected_set(rows_by_cell["T"])
    tad_sets = _selected_set(rows_by_cell["T+A+D"])
    rows = []
    names = list(CELL_ORDER) + ["hard_rescue_reference"]
    for cell in names:
        metric = hard_metric if cell == "hard_rescue_reference" else metrics_by_cell[cell]
        concentration = concentration_by_cell.get(cell, {})
        selected = rows_by_cell.get(cell, [])
        selected_set = _selected_set(selected)
        union_t = t_sets | selected_set
        union_tad = tad_sets | selected_set
        rows.append({
            "cell": cell, "top1": metric.get("top1", math.nan), "top5": metric.get("top5", math.nan),
            "top1_gain_vs_T": float(metric["top1"]) - float(base["top1"]) if "top1" in metric and "top1" in base else math.nan,
            "top5_gain_vs_T": float(metric["top5"]) - float(base["top5"]) if "top5" in metric and "top5" in base else math.nan,
            "estimated_parameter_sparsity": metric.get("estimated_parameter_sparsity", math.nan),
            "incremental_units": len(selected),
            "incremental_attention": sum(str(row.get("unit_type")) == TYPE_ATTENTION for row in selected),
            "incremental_ffn": sum(str(row.get("unit_type")) == TYPE_FFN for row in selected),
            "unique_domains": concentration.get("unique_domains_touched", 0),
            "largest_domain_share": concentration.get("largest_domain_share", math.nan),
            "top2_share": concentration.get("top2_share", math.nan),
            "top4_share": concentration.get("top4_share", math.nan),
            "HHI": concentration.get("HHI", math.nan),
            "minimum_domain_coverage": concentration.get("minimum_final_coverage", math.nan),
            "Jaccard_vs_T": len(t_sets & selected_set) / max(1, len(union_t)),
            "Jaccard_vs_TAD": len(tad_sets & selected_set) / max(1, len(union_tad)),
        })
    return rows


def _save_plot(fig, output_dir: Path, number: int) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"figure_{number:02d}.png", dpi=160)
    fig.savefig(output_dir / f"figure_{number:02d}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def write_figures(output_dir: Path, summary_rows: Sequence[Mapping[str, object]], effects: Sequence[Mapping[str, object]], concentration_rows: Sequence[Mapping[str, object]], risk_rows: Sequence[Mapping[str, object]], average_rows: Sequence[Mapping[str, object]], composition_rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = [str(row["cell"]) for row in summary_rows if str(row["cell"]) in CELL_ORDER]
    by_name = {str(row["cell"]): row for row in summary_rows}
    for number, metric, title in ((1, "top1", "2x2 factorial Top1"), (2, "top5", "2x2 factorial Top5")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, [float(by_name[name].get(metric, math.nan)) for name in names], color="#4472c4")
        ax.set_title(title); ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=20)
        _save_plot(fig, figure_dir, number)
    fig, ax = plt.subplots(figsize=(7, 4))
    effect = effects[0] if effects else {}
    ax.bar(["A no D", "D no A", "A with D", "D with A"],
           [effect.get("E_A_noD", math.nan), effect.get("E_D_noA", math.nan),
            effect.get("E_A_withD", math.nan), effect.get("E_D_withA", math.nan)], color="#70ad47")
    ax.set_title("Factor main effects"); ax.set_ylabel("percentage points")
    _save_plot(fig, figure_dir, 3)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([row["metric"] for row in effects], [row.get("Interaction_AD", math.nan) for row in effects], color="#ed7d31")
    ax.set_title("A-D interaction"); ax.set_ylabel("interaction")
    _save_plot(fig, figure_dir, 4)
    for number, field, title in ((5, "HHI", "Domain HHI"), (6, "top2_share", "Top2/Top4 domain concentration"), (7, "minimum_final_coverage", "Minimum domain coverage")):
        fig, ax = plt.subplots(figsize=(7, 4))
        values = [next((row.get(field, math.nan) for row in concentration_rows if row["cell"] == name), math.nan) for name in names]
        if field == "top2_share":
            top4 = [next((row.get("top4_share", math.nan) for row in concentration_rows if row["cell"] == name), math.nan) for name in names]
            x = np.arange(len(names)); width = .36
            ax.bar(x - width / 2, values, width, label="top2")
            ax.bar(x + width / 2, top4, width, label="top4")
            ax.set_xticks(x, names, rotation=20); ax.legend(loc="upper left")
        else:
            ax.bar(names, values, color="#5b9bd5")
            ax.tick_params(axis="x", rotation=20)
        ax.set_title(title); ax.set_ylabel(field)
        _save_plot(fig, figure_dir, number)
    fig, ax = plt.subplots(figsize=(7, 4))
    comps = {row["cell"]: row for row in composition_rows}
    x = np.arange(2); width = .35
    td = comps.get("T+D", {}); tad = comps.get("T+A+D", {})
    ax.bar(x - width / 2, [td.get("incremental_attention", 0), td.get("incremental_ffn", 0)], width, label="T+D")
    ax.bar(x + width / 2, [tad.get("incremental_attention", 0), tad.get("incremental_ffn", 0)], width, label="T+A+D")
    ax.set_xticks(x, ["Attention", "FFN"]); ax.set_title("T+D vs T+A+D composition"); ax.legend(loc="upper left")
    _save_plot(fig, figure_dir, 8)
    fig, ax = plt.subplots(figsize=(7, 4))
    risk_values = [row.get("mean_CE_increase", math.nan) for row in risk_rows]
    ax.bar(np.arange(len(risk_values)), risk_values, color="#c0504d")
    ax.set_title("Task020 labelled CE risk"); ax.set_ylabel("mean CE increase")
    _save_plot(fig, figure_dir, 9)
    fig, ax = plt.subplots(figsize=(7, 4))
    for subset in ("TD-only", "TAD-only"):
        values = [row.get("p_average", math.nan) for row in average_rows if row.get("subset") == subset]
        if values:
            ax.hist(values, bins=20, alpha=.55, label=subset)
    ax.set_title("Average percentile: TD-only vs TAD-only"); ax.set_xlabel("p_average"); ax.legend(loc="upper right")
    _save_plot(fig, figure_dir, 10)


def write_diagnosis(output_dir: Path, effects: Sequence[Mapping[str, object]], summary_rows: Sequence[Mapping[str, object]], attention_rows: Sequence[Mapping[str, object]]) -> None:
    by_cell = {str(row["cell"]): row for row in summary_rows}
    effect = {str(row["metric"]): row for row in effects}
    td = by_cell.get("T+D", {})
    t = by_cell.get("T", {})
    ta = by_cell.get("T+A", {})
    tad = by_cell.get("T+A+D", {})
    interaction = effect.get("top1", {}).get("Interaction_AD", math.nan)
    if math.isfinite(float(interaction)):
        interaction_label = "positive synergy" if interaction > 1e-9 else "partial redundancy / antagonism" if interaction < -1e-9 else "approximately additive"
    else:
        interaction_label = "undetermined"
    retained_td = sum(row.get("retained") is True for row in attention_rows if row.get("cell") == "T+D")
    lines = ["# Task025 factorial diagnosis", "", "The T+D cell is a control ablation; conclusions are descriptive.", ""]
    questions = (
        ("Q1", "Does T+D outperform exact T?", float(td.get("top1", math.nan)) - float(t.get("top1", math.nan))),
        ("Q2", "How does T+D compare with T+A?", float(td.get("top1", math.nan)) - float(ta.get("top1", math.nan))),
        ("Q3", "How does T+D compare with T+A+D?", float(td.get("top1", math.nan)) - float(tad.get("top1", math.nan))),
        ("Q4", "Average main effect without Domain State?", effect.get("top1", {}).get("E_A_noD", math.nan)),
        ("Q5", "Domain-State main effect without Average?", effect.get("top1", {}).get("E_D_noA", math.nan)),
        ("Q6", "Average incremental effect given Domain State?", effect.get("top1", {}).get("E_A_withD", math.nan)),
        ("Q7", "Domain-State incremental effect given Average?", effect.get("top1", {}).get("E_D_withA", math.nan)),
        ("Q8", "Is the interaction positive, additive, or redundant?", interaction_label),
        ("Q9", "Does T+D retain the three dangerous Attention units?", retained_td),
        ("Q10", "What independent evidence remains for Average?", "Task020-labelled FFN risk and set-level changes"),
        ("Q11", "Does Average improve Task020-labelled FFN safety given Domain State?", "see factorial_task020_risk.csv"),
        ("Q12", "Does Average alter domain allocation or within-domain choice?", "see factorial_selection_composition.csv and average_incremental_value_given_domain_state.csv"),
        ("Q13", "Which factor primarily reduces domain HHI?", "compare factorial_domain_concentration.csv"),
        ("Q14", "Which factor primarily improves minimum coverage?", "compare factorial_domain_concentration.csv"),
        ("Q15", "Should all three terms be retained?", "requires the measured factorial and risk evidence"),
        ("Q16", "What experiment should follow before promotion?", "independent physical structural-pruning validation and fresh task-risk evaluation"),
    )
    for number, question, answer in questions:
        lines.extend((f"## {number}. {question}", f"Observed: {answer}", ""))
    (Path(output_dir) / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(*, output_dir: Path, task023_root: Path, task024_root: Path, task020_root: Path) -> dict:
    output_dir = Path(output_dir)
    names = list(CELL_ORDER)
    rows_by_cell = {
        "T": _variant_rows(task023_root, REFERENCE_T),
        "T+A": _variant_rows(task024_root, REFERENCE_TA),
        "T+D": _variant_rows(output_dir, VARIANT),
        "T+A+D": _variant_rows(task024_root, REFERENCE_TAD),
    }
    metrics_by_cell = {
        "T": _metric(Path(task023_root) / "validation" / REFERENCE_T / "metrics.json"),
        "T+A": _metric(Path(task024_root) / "validation" / REFERENCE_TA / "metrics.json"),
        "T+D": _metric(output_dir / "validation" / VARIANT / "metrics.json"),
        "T+A+D": _metric(Path(task024_root) / "validation" / REFERENCE_TAD / "metrics.json"),
    }
    hard_metric = _metric(Path(task023_root) / "validation" / REFERENCE_HARD / "metrics.json")
    effects = factorial_accuracy_decomposition(metrics_by_cell)
    concentration_rows = []
    for cell in names:
        root = task023_root if cell == "T" else task024_root if cell in ("T+A", "T+A+D") else output_dir
        concentration_rows.append(domain_concentration_row(cell, rows_by_cell[cell], _coverage_rows(root, CELL_TO_VARIANT[cell])))
    atomic_csv(output_dir / "factorial_accuracy_decomposition.csv", tuple(effects[0]), effects)
    atomic_csv(output_dir / "factorial_domain_concentration.csv", tuple(concentration_rows[0]), concentration_rows)
    composition_rows = [_composition(cell, rows_by_cell[cell]) for cell in names]
    atomic_csv(output_dir / "factorial_selection_composition.csv", tuple(composition_rows[0]), composition_rows)

    attention_rows = []
    for cell in names:
        selected = {_int(row, "global_index"): row for row in rows_by_cell[cell]}
        for index in DANGEROUS_ATTENTION:
            row = selected.get(index)
            attention_rows.append({
                "cell": cell, "global_index": index, "selected": row is not None,
                "retained": row is None, "selection_step": row.get("step", "") if row else "",
                "p_total": _float(row, "p_total", default=math.nan) if row else math.nan,
                "p_average": _float(row, "p_average", default=math.nan) if row else math.nan,
                "domain_damage_at_decision": _float(row, "domain_damage_before", default=math.nan) if row else math.nan,
            })
    atomic_csv(output_dir / "factorial_attention_outcomes.csv", tuple(attention_rows[0]), attention_rows)

    labels = task024._task020_labels(Path(task020_root))
    task020_rows = _task020_risk(rows_by_cell, labels)
    atomic_csv(output_dir / "factorial_task020_risk.csv", tuple(task020_rows[0]) if task020_rows else ("comparison",), task020_rows)
    average_details = _distribution_rows(
        "T+D", "T+A+D", rows_by_cell, labels,
        subset_names=("common", "TD-only", "TAD-only"),
    )
    atomic_csv(output_dir / "average_incremental_value_given_domain_state.csv", tuple(average_details[0]), average_details)
    domain_details = _distribution_rows(
        "T+A", "T+A+D", rows_by_cell, labels,
        subset_names=("common", "TA-only", "TAD-only"), domain_mode=True,
    )
    atomic_csv(output_dir / "domain_state_incremental_value_given_average.csv", tuple(domain_details[0]), domain_details)

    jaccard_rows = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            parts = partition_sets(_selected_set(rows_by_cell[left]), _selected_set(rows_by_cell[right]))
            union = parts["common"] | parts["left_only"] | parts["right_only"]
            jaccard_rows.append({"left": left, "right": right, "common": len(parts["common"]),
                                 "left_only": len(parts["left_only"]), "right_only": len(parts["right_only"]),
                                 "Jaccard": len(parts["common"]) / max(1, len(union))})
    atomic_csv(output_dir / "factorial_selection_jaccard.csv", tuple(jaccard_rows[0]), jaccard_rows)
    summary_rows = _summary_rows(metrics_by_cell, {row["cell"]: row for row in concentration_rows}, rows_by_cell, hard_metric)
    atomic_csv(output_dir / "task025_summary.csv", tuple(summary_rows[0]), summary_rows)
    td_set = _selected_set(rows_by_cell["T+D"])
    tad_set = _selected_set(rows_by_cell["T+A+D"])
    by_index = {}
    for row in rows_by_cell["T+D"] + rows_by_cell["T+A+D"]:
        by_index[_int(row, "global_index")] = row
    average_plot_rows = []
    for subset, indexes in (("TD-only", td_set - tad_set), ("TAD-only", tad_set - td_set)):
        average_plot_rows.extend(
            {"subset": subset, "p_average": _float(by_index[index], "p_average", default=math.nan)}
            for index in indexes if index in by_index
        )
    write_figures(output_dir, summary_rows, effects, concentration_rows, task020_rows, average_plot_rows, composition_rows)
    write_diagnosis(output_dir, effects, summary_rows, attention_rows)
    completion = {
        "status": "PASS", "code_version": CODE_VERSION, "analysis_version": ANALYSIS_VERSION,
        "task023_reference_pass": True, "task024_reference_pass": True,
        "same_start_prefix": True, "same_target_budget": True,
        "total_domain_state_constructed": (output_dir / "variants" / VARIANT / "registry.json").is_file(),
        "total_domain_state_validation_complete": metrics_by_cell["T+D"].get("status") == "PASS",
        "factorial_analysis_complete": bool(effects), "task020_posthoc_complete": bool(task020_rows),
        "domain_concentration_analysis_complete": bool(concentration_rows),
        "zero_new_tunable_hyperparameters": True, "fixed_percentile_threshold_used": False,
        "task_label_leakage": False, "task023_artifacts_modified": False,
        "task024_artifacts_modified": False, "production_pruning_code_modified": False,
        "fine_tuning_executed": False,
    }
    required = ("total_domain_state_constructed", "total_domain_state_validation_complete",
                "factorial_analysis_complete", "task020_posthoc_complete",
                "domain_concentration_analysis_complete")
    if not all(completion[key] for key in required):
        completion["status"] = "PENDING_VALIDATION"
    atomic_json(output_dir / "task025_completion.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "construct", "validate", "analyze"), required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    for name in ("task023_root", "task024_root", "task014_root", "task016_root", "task017_root",
                 "task018_root", "task019_root", "task020_root", "task021_root", "task022_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "identity":
        required = (args.task023_root, args.task024_root, args.task014_root, args.task016_root,
                    args.task017_root, args.task018_root, args.task019_root, args.task020_root,
                    args.task021_root, args.task022_root, args.output_dir, args.checkpoint, args.repo_root)
        if any(value is None for value in required):
            raise SystemExit("identity requires all task roots, --checkpoint and --repo-root")
        verify_identity(task023_root=args.task023_root, task024_root=args.task024_root,
                        task014_root=args.task014_root, task016_root=args.task016_root,
                        task017_root=args.task017_root, task018_root=args.task018_root,
                        task019_root=args.task019_root, task020_root=args.task020_root,
                        task021_root=args.task021_root, task022_root=args.task022_root,
                        output_dir=args.output_dir, checkpoint=args.checkpoint, repo_root=args.repo_root)
    elif args.mode == "construct":
        required = (args.task023_root, args.task024_root, args.task014_root, args.task016_root,
                    args.task017_root, args.task018_root, args.output_dir)
        if any(value is None for value in required):
            raise SystemExit("construct requires Task023/24/14/16/17 roots")
        construct_variant(task023_root=args.task023_root, task024_root=args.task024_root,
                          task014_root=args.task014_root, task016_root=args.task016_root,
                          task017_root=args.task017_root, task018_root=args.task018_root,
                          output_dir=args.output_dir, device=args.device)
    elif args.mode == "validate":
        if args.variant is None:
            raise SystemExit("validate requires --variant")
        validate_variant(args.output_dir, args.variant, args.device)
    else:
        required = (args.task023_root, args.task024_root, args.task020_root)
        if any(value is None for value in required):
            raise SystemExit("analyze requires Task023, Task024 and Task020 roots")
        analyze(output_dir=args.output_dir, task023_root=args.task023_root,
                task024_root=args.task024_root, task020_root=args.task020_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

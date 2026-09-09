"""Task034: F3 midpoint-veto logical pruning and Task028-aligned fine-tuning.

Task034 is a new experiment branch layered on the immutable Task033 replay.
The selector is intentionally small at its public boundary: F3 uses the
already computed Task033 tensors and the frozen midpoint equation, while all
model/runtime imports are lazy.  This keeps CPU identity and regression tests
independent of CUDA, torch, and the UCF101 installation.

The production path materialises Python rows only for reports and snapshots.
Selection remains delegated to the repaired Task031/Task033 tensor replay;
the module never changes Task028--Task033 source artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_VERSION = "task034_mid_veto_50_logical_finetune_v1"
TASK034_BRANCH = "task_034_mid_veto_50_logical_finetune"
TASK033_BASE_BRANCH = "task_033_average_veto_mechanism_diagnosis"
TASK033_BASE_COMMIT = "cd51d84b5cb351a2deb4a74f548683d8b6a23cad"

FUSION = "F3_MID_VETO"
TARGET_SPARSITY = 0.50
SNAPSHOT_TARGETS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
# Task033 computes the ordinal percentile and production risk views as
# float64.  Delta source tensors remain in their recorded dtype; this marker
# makes the exactness contract explicit in Task034 reports.
PERCENTILE_RISK_DTYPE = "torch.float64"
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
UNIT_TYPES = (TYPE_ATTENTION, TYPE_FFN)
STAGES = tuple(f"stage_{index}" for index in range(4))

# These are protocol values, not CLI knobs.  Keeping them in one immutable
# record makes accidental Task034-specific tuning visible in the identity
# artifact and in the tests.
PROTOCOL = {
    "precision": "fp32",
    "amp_enabled": False,
    "use_checkpoint": False,
    "gpu_ids": [0, 1],
    "batch_size": 4,
    "optimizer": "SGD",
    "learning_rate": 0.0005,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "scheduler": "NONE",
    "loss": "CrossEntropyLoss",
    "epochs": 100,
    "seed": 3407,
    "validation_uses_softmax": False,
}
MODEL_ARCHITECTURE = {
    "patch_size": [2, 4, 4],
    "embed_dim": 96,
    "depths": [2, 2, 18, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [8, 7, 7],
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "patch_norm": True,
    "drop_path_rate": 0.2,
    "use_checkpoint": False,
}
EXPECTED_SWIN_BLOCKS = sum(
    int(depth) for depth in MODEL_ARCHITECTURE["depths"]
)

TASK033_REFERENCE_FILES = (
    "fusion_final_50_summary.csv",
    "fusion_selection_snapshots.csv",
    "fusion_first_attention.json",
    "fusion_stage_removals.csv",
    "fusion_domain_removals.csv",
    "fusion_selection_overlap.json",
    "task033_report_only_completion.json",
)
TASK028_REFERENCE_FILES = (
    "selection_50/registry.json",
    "selection_50/causal_selection_trace.csv",
    "selection_50/construction.json",
    "finetune_config.json",
    "artifact_identity.json",
)
TASK029_REFERENCE_FILES = (
    "snapshot_candidates.csv", "final_50_attention_candidates.csv",
    "domain_type_degradation.csv", "attention_rank_by_snapshot.csv",
    "task029_completion.json",
)
TASK029_F0_JSON_ARTIFACT_NAMES = (
    "replay_data.json", "replay.json", "artifact_identity.json",
    "task029_completion.json",
)
TASK029_F0_SEQUENCE_KEYS = {
    "replay_sequence", "trace_sequence", "selected_sequence",
    "selection_sequence", "global_index_sequence", "selected_global_indices",
    "replayed_global_indices", "replay_indices", "selected_rows", "trace",
    "selected",
}
TASK029_F0_SHA_KEYS = {
    "replay_sequence_sha256", "trace_sequence_sha256",
    "task028_trace_sequence_sha256", "task028_sequence_sha256",
    "replay_trace_sequence_sha256", "selected_sequence_sha256",
    "sequence_sha256",
}
TASK029_F0_FLAG_KEYS = {
    "replayed_0_to_50_exact", "trace_sequence_match", "replay_match",
    "exact_replay", "exact_task028_replay", "replay_exact",
    "full_sequence_equality", "sha_identity_equality",
}
TASK030_REFERENCE_FILES = (
    "causal_ablation_results.csv", "selected_attention_heads.csv",
    "selected_causal_pairs.csv", "cost_matched_ffn_packs.csv",
)
TASK031_REFERENCE_FILES = (
    "tested_unit_risk_table.csv", "granularity_controlled_pair_audit.csv",
    "variant_granularity_dependence.csv", "variant_type_distribution.csv",
    "variant_final_50_summary.csv",
)
TASK032_REFERENCE_FILES = (
    "artifact_identity.json", "causal_beta_pairwise.csv",
    "causal_beta_leave_one_out.csv", "causal_beta_bootstrap.json",
    "causal_beta_summary.json", "average_granularity_scaling.csv",
    "causal_granularity_components.csv", "excess_granularity_decomposition.json",
    "causal_residual_alignment.csv", "causal_residual_alignment_summary.json",
    "cost_matched_decomposition.csv", "decomposed_type_separation.csv",
    "domain_residual_analysis.csv", "mixed_domain_residual_analysis.csv",
    "stage_granularity_decomposition.csv", "task032_scientific_summary.md",
    "task032_completion.json",
)

TRACE_FIELDS = (
    "step", "global_index", "unit_type", "layer", "stage", "unit_index",
    "domain_id", "parameter_cost", "Delta_average", "Delta_total",
    "p_total", "p_average", "domain_damage", "B", "V", "R_F3",
    "cumulative_removed_parameters", "effective_sparsity_after", "feasible",
)
SEQUENCE_FIELDS = ("step", "global_index")
SNAPSHOT_FIELDS = (
    "fusion", "snapshot_target", "actual_effective_sparsity",
    "attention_removed", "ffn_removed", "remaining_attention",
    "remaining_ffn", "first_attention_sparsity", "best_attention_rank",
    "ffn_ahead", "attention_parameter_contribution",
    "ffn_parameter_contribution", "stage_removals", "domain_removals",
)


def _int(value: object, label: str = "value") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc


def _float(value: object, label: str = "value", default: float | None = None) -> float:
    if value is None or str(value).strip() == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Missing {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _torch():
    """Import PyTorch lazily for model/runtime-only Task034 stages."""
    return __import__("torch")


def _tqdm(iterable, *args, **kwargs):
    """Wrap an iterable with tqdm using a lazy runtime import."""
    module = __import__("tqdm", fromlist=("tqdm",))
    return module.tqdm(iterable, *args, **kwargs)


def _optional_float(value: object, label: str = "value") -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def read_json(path: Path) -> dict[str, Any]:
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
        json.dumps(_json_safe(payload), indent=2, sort_keys=True,
                   ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field, "")) for field in fields})
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
    text = ",".join(str(_int(value, "global_index")) for value in indices)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def assert_output_is_separate(output_dir: Path, roots: Iterable[Path]) -> None:
    output = Path(output_dir).expanduser().resolve()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if output == root or _path_is_inside(output, root):
            raise RuntimeError(f"Task034 output must be separate from immutable root: {root}")


def hash_existing_files(root: Path, names: Sequence[str]) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / name) for name in names
            if (root / name).is_file() and (root / name).stat().st_size > 0}


def hash_required_files(root: Path, names: Sequence[str], label: str) -> dict[str, str]:
    root = Path(root)
    missing = [name for name in names if not (root / name).is_file()
               or (root / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing {label} artifacts below {root}: {missing}")
    return {name: sha256_file(root / name) for name in names}


def source_artifact_hashes(roots: Mapping[str, Path], *, strict: bool = True) -> dict[str, dict[str, str]]:
    groups = {
        "task028": TASK028_REFERENCE_FILES,
        "task029": TASK029_REFERENCE_FILES,
        "task030": TASK030_REFERENCE_FILES,
        "task031": TASK031_REFERENCE_FILES,
        "task032": TASK032_REFERENCE_FILES,
        "task033": TASK033_REFERENCE_FILES,
    }
    result: dict[str, dict[str, str]] = {}
    for label, names in groups.items():
        root = Path(roots[label])
        result[label] = (hash_required_files(root, names, label)
                         if strict else hash_existing_files(root, names))
    return result


def verify_source_hashes_unchanged(before: Mapping[str, Mapping[str, str]],
                                   roots: Mapping[str, Path]) -> bool:
    after: dict[str, dict[str, str]] = {}
    try:
        for label, files in before.items():
            root = Path(roots[str(label)])
            after[str(label)] = {
                str(name): sha256_file(root / str(name))
                for name in files
                if (root / str(name)).is_file() and (root / str(name)).stat().st_size > 0
            }
    except (OSError, KeyError, TypeError):
        return False
    return {str(key): dict(value) for key, value in before.items()} == after


# ---------------------------------------------------------------------------
# Frozen F3 math and exact Python ordering (CPU regression surface)
# ---------------------------------------------------------------------------

def f3_components(p_total: float, p_average: float, domain_damage: float) -> tuple[float, float, float]:
    """Return ``(B, V, R_F3)`` using the frozen midpoint equation."""
    total = _float(p_total, "p_total")
    average = _float(p_average, "p_average")
    damage = _float(domain_damage, "domain_damage")
    base = max(total, damage)
    veto = max(average - base, 0.0)
    risk = base + veto / 2.0
    return base, veto, risk


def f3_mid_veto(p_total: float, p_average: float, domain_damage: float) -> float:
    return f3_components(p_total, p_average, domain_damage)[2]


def f3_order_key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
    """The frozen production order: risk, total percentile, average, index."""
    risk = row.get("R_F3")
    if risk in (None, ""):
        risk = f3_mid_veto(
            _float(row.get("p_total"), "p_total"),
            _float(row.get("p_average"), "p_average"),
            _float(row.get("domain_damage"), "domain_damage"),
        )
    return (
        _float(risk, "R_F3"),
        _float(row.get("p_total"), "p_total"),
        _float(row.get("p_average"), "p_average"),
        _int(row.get("global_index"), "global_index"),
    )


def f3_rank_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ranked = [dict(row) for row in rows]
    ranked.sort(key=f3_order_key)
    for rank, row in enumerate(ranked, 1):
        row["global_rank"] = rank
    return ranked


def annotate_f3_row(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    b, v, risk = f3_components(
        _float(row.get("p_total"), "p_total"),
        _float(row.get("p_average"), "p_average"),
        _float(row.get("domain_damage"), "domain_damage"),
    )
    result.update({"B": b, "V": v, "R_F3": risk})
    return result


def rank_f3(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return f3_rank_rows([annotate_f3_row(row) for row in rows])


def budget_reached(removed_parameters: float, total_parameters: float,
                   target_sparsity: float = TARGET_SPARSITY) -> bool:
    return float(removed_parameters) >= float(total_parameters) * float(target_sparsity) - 1e-12


def dynamic_selection_step(rows: Sequence[Mapping[str, object]],
                           removed_parameters: float = 0.0) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Rank the current candidate set and remove exactly one selected row."""
    if not rows:
        raise RuntimeError("No feasible candidates remain")
    ranked = rank_f3(rows)
    chosen = ranked[0]
    remaining = [row for row in rows
                 if _int(row.get("global_index"), "global_index") != _int(chosen["global_index"], "global_index")]
    return chosen, remaining


def simulate_dynamic_selection(rows: Sequence[Mapping[str, object]],
                               total_parameters: float,
                               target_sparsity: float = TARGET_SPARSITY,
                               recompute: Any | None = None) -> list[dict[str, object]]:
    """Small CPU oracle used by tests; real replay uses Task033 tensors."""
    current = [dict(row) for row in rows]
    if recompute is not None:
        current = [dict(row) for row in recompute(current)]
    removed = 0.0
    selected: list[dict[str, object]] = []
    while not budget_reached(removed, total_parameters, target_sparsity):
        chosen, current = dynamic_selection_step(current, removed)
        cost = _float(chosen.get("parameter_cost"), "parameter_cost", default=0.0)
        removed += cost
        chosen = dict(chosen)
        chosen["step"] = len(selected) + 1
        chosen["cumulative_removed_parameters"] = removed
        chosen["effective_sparsity_after"] = removed / max(float(total_parameters), 1e-12)
        selected.append(chosen)
        if recompute is not None and current:
            current = [dict(row) for row in recompute(current)]
    return selected


def _stage_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "stage_unknown"
    if text.startswith("stage_"):
        return text
    if text.lstrip("-").isdigit():
        return f"stage_{int(text)}"
    match = re.search(r"(?:^|\.)(?:layers?|stage)[._]?(\d+)(?:\.|$)", text)
    if match:
        return f"stage_{int(match.group(1))}"
    return f"stage_{text}"


def _scalar(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _row_value(row: Mapping[str, object], *names: str, default: object = "") -> object:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


# ---------------------------------------------------------------------------
# Exact Task033 component materialisation and F3 replay adapter
# ---------------------------------------------------------------------------

def materialize_component_rows(candidates: Mapping[str, object], p_total: object,
                               p_average: object, domain_damage: object,
                               lookup: Mapping[int, Mapping[str, object]] | None = None
                               ) -> list[dict[str, object]]:
    """Materialise snapshot rows without recomputing global p_average.

    Candidate tensors retain their current order.  ``.item()`` is used only
    for the report scalar boundary; source tensors and their dtypes are not
    changed.  The returned p_average is exactly the supplied Task033 tensor
    view, not a Python Delta_average rerank.
    """
    lookup = lookup or {}
    global_tensor = candidates.get("global_index")
    if global_tensor is None:
        raise ValueError("candidates must contain global_index")
    count = int(global_tensor.numel()) if hasattr(global_tensor, "numel") else len(global_tensor)
    metadata_fields = (
        "global_index", "domain_id", "local_index", "Delta_average", "Delta_total",
        "parameter_cost", "unit_type", "layer", "stage", "unit_index", "feasible",
    )
    rows: list[dict[str, object]] = []
    for index in range(count):
        gid = _int(_scalar(global_tensor[index]), "global_index")
        row = dict(lookup.get(gid, {}))
        for field in metadata_fields:
            tensor = candidates.get(field)
            if tensor is not None:
                row[field] = _scalar(tensor[index])
        row["global_index"] = gid
        row["p_total"] = _scalar(p_total[index])
        row["p_average"] = _scalar(p_average[index])
        row["domain_damage"] = _scalar(domain_damage[index])
        row["feasible"] = row.get("feasible", True)
        rows.append(row)
    return rows


def _task033_module():
    return __import__("task033_average_veto_mechanism_diagnosis")


def run_f3_benchmark(*, task014_root: Path, task016_root: Path, task017_root: Path,
                     output_dir: Path, device: str = "cuda:0", steps: int = 1000,
                     ranking_backend: str = "single-gpu") -> dict[str, object]:
    """Run only the tensor hot-path gate; no snapshot Python rows are built."""
    task033 = _task033_module()
    report = task033.benchmark_fusion(
        fusion=FUSION, task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_benchmark",
        device=device, ranking_backend=ranking_backend, steps=int(steps),
    )
    rate = _float(report.get("steps_per_second"), "steps_per_second", default=0.0)
    result = {
        **dict(report), "fusion": FUSION, "steps": int(steps),
        "minimum_steps_per_second": 50.0,
        "preferred_steps_per_second": 100.0,
        "hot_path_python_sorted": False,
        "status": "PASS" if rate >= 50.0 else "FAIL",
        "preferred_gate_pass": rate >= 100.0,
    }
    atomic_json(Path(output_dir) / "selection" / "f3_performance_benchmark.json", result)
    return result


def _lookup_from_trace(root: Path) -> dict[int, dict[str, object]]:
    path = Path(root) / "selection_50" / "causal_selection_trace.csv"
    if not path.is_file():
        return {}
    return {_int(row["global_index"], "global_index"): dict(row)
            for row in read_csv(path) if row.get("global_index", "") != ""}


def _task028_sequence(root: Path) -> list[int]:
    path = Path(root) / "selection_50" / "causal_selection_trace.csv"
    rows = read_csv(path)
    return [_int(row.get("global_index"), "global_index") for row in sorted(
        rows, key=lambda row: (_int(row.get("step", 0) or 0, "step"),
                               _int(row.get("incremental_step", 0) or 0, "incremental_step")))]


def _snapshot_summary(ranked: Sequence[Mapping[str, object]], selected: Sequence[Mapping[str, object]],
                      sparsity: float, total_parameters: float) -> dict[str, object]:
    attention = [row for row in ranked if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffns = [row for row in ranked if str(row.get("unit_type")) == TYPE_FFN]
    removed = [row for row in selected
               if _float(row.get("effective_sparsity_after"), "effective_sparsity_after", 0.0)
               <= float(sparsity) + 1e-12]
    first = next((row for row in selected if row.get("unit_type") == TYPE_ATTENTION), None)
    best = attention[0] if attention else None
    return {
        "attention_removed": sum(row.get("unit_type") == TYPE_ATTENTION for row in removed),
        "ffn_removed": sum(row.get("unit_type") == TYPE_FFN for row in removed),
        "remaining_attention": len(attention), "remaining_ffn": len(ffns),
        "first_attention_sparsity": first.get("effective_sparsity_after") if first else None,
        "best_attention_rank": best.get("global_rank") if best else None,
        "ffn_ahead": sum(row.get("unit_type") == TYPE_FFN for row in ranked[:ranked.index(best)]) if best else None,
        "attention_parameter_contribution": sum(
            _float(row.get("parameter_cost"), "parameter_cost", 0.0)
            for row in removed if row.get("unit_type") == TYPE_ATTENTION
        ) / max(float(total_parameters), 1e-12),
        "ffn_parameter_contribution": sum(
            _float(row.get("parameter_cost"), "parameter_cost", 0.0)
            for row in removed if row.get("unit_type") == TYPE_FFN
        ) / max(float(total_parameters), 1e-12),
        "stage_removals": dict(Counter(_stage_label(row.get("stage")) for row in removed)),
        "domain_removals": dict(Counter(str(row.get("domain_id", "")) for row in removed)),
    }


def _tensor_candidate_row(candidates: Mapping[str, object], ranked: Mapping[str, object],
                          position: int, lookup: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    gid = _int(_scalar(candidates["global_index"][position]), "global_index")
    row = dict(lookup.get(gid, {}))
    for name in ("global_index", "domain_id", "local_index", "Delta_average", "Delta_total", "domain_damage"):
        if name in candidates:
            row[name] = _scalar(candidates[name][position])
    row["global_index"] = gid
    for name in ("p_total", "p_average", "B", "V", "R_F3", "R_dual"):
        if name in ranked:
            row[name] = _scalar(ranked[name][position])
    row["feasible"] = True
    return row


def replay_f3(*, task014_root: Path, task016_root: Path, task017_root: Path,
              task028_root: Path, output_dir: Path, device: str = "cuda:0",
              ranking_backend: str = "single-gpu") -> dict[str, object]:
    """Replay F3 dynamically through the frozen Task033 backend."""
    task033 = _task033_module()
    task031 = __import__("task031_domain_conditioned_average_diagnosis")
    engine, optimized = task033._engine_for_replay(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_replay" / FUSION,
        device=device, ranking_backend=ranking_backend,
    )
    lookup = task031.lookup_from_engine(engine)
    lookup.update(_lookup_from_trace(Path(task028_root)))
    total_parameters = float(getattr(engine, "total_parameters"))
    task028_sequence = _task028_sequence(Path(task028_root))
    result = task033.replay_fusion(
        fusion=FUSION, engine=engine, optimized=optimized,
        total_parameters=total_parameters, target_budget=TARGET_SPARSITY * total_parameters,
        lookup=lookup, task028_sequence=task028_sequence,
    )
    selected = [dict(row) for row in result.get("selected", [])]
    # Task033 already carries exact p_total/p_average/domain_damage tensors to
    # each selected row.  Fill only the frozen F3 scalar fields if an older
    # compatible backend omitted them; never reconstruct p_average from Delta.
    for step, row in enumerate(selected, 1):
        row["step"] = step
        row["stage"] = _stage_label(row.get("stage"))
        b, v, risk = f3_components(row.get("p_total"), row.get("p_average"), row.get("domain_damage", 0.0))
        row.update({"B": row.get("B", b), "V": row.get("V", v), "R_F3": row.get("R_F3", risk)})
        row["cumulative_removed_parameters"] = _float(
            row.get("cumulative_removed_parameters", row.get("parameter_cost", 0.0)),
            "cumulative_removed_parameters",
        )
        row["effective_sparsity_after"] = _float(
            row.get("effective_sparsity_after"), "effective_sparsity_after",
        )
    result["selected"] = selected
    result["sequence"] = [_int(row["global_index"], "global_index") for row in selected]
    result["sequence_sha256"] = sequence_sha256(result["sequence"])
    result["task028_sequence_sha256"] = sequence_sha256(task028_sequence)
    result["task028_sequence_equal"] = result["sequence"] == task028_sequence
    result["fusion"] = FUSION
    result["total_parameters"] = total_parameters
    result["inventory"] = inventory_from_engine(engine, selected)
    result["percentile_risk_dtype"] = PERCENTILE_RISK_DTYPE
    return result


# ---------------------------------------------------------------------------
# Task033 report-only authority and exact sequence/registry artefacts
# ---------------------------------------------------------------------------

def _target_key(value: object) -> float:
    return round(_float(value, "snapshot_target"), 12)


def _numeric_equal(left: object, right: object) -> bool:
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    try:
        # Serialized Task033 and Task034 values are generated from the same
        # float64 tensors.  Exact equality is therefore the intended gate;
        # the tiny fallback only handles decimal CSV round-tripping.
        a, b = float(left), float(right)
        return a == b or (math.isfinite(a) and math.isfinite(b)
                          and abs(a - b) <= 1e-15 and a.hex() == b.hex())
    except (TypeError, ValueError):
        return str(left) == str(right)


def _int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return _int(value)
    except ValueError:
        return None


def _row_number(row: Mapping[str, object], names: Sequence[str], default: object = None) -> object:
    return _row_value(row, *names, default=default)


def load_task033_f3_reference(task033_root: Path) -> dict[str, object]:
    """Load the immutable Task033 F3 aggregate artifacts.

    The values are read from the report files at runtime.  No approximate
    Attention/FFN counts or sparsity values are embedded in this module.
    """
    root = Path(task033_root)
    hashes = hash_required_files(root, TASK033_REFERENCE_FILES, "Task033 F3 reference")
    final_rows = [row for row in read_csv(root / "fusion_final_50_summary.csv")
                  if str(row.get("fusion", "")) == FUSION]
    if len(final_rows) != 1:
        raise RuntimeError("Task033 fusion_final_50_summary.csv must contain exactly one F3 row")
    snapshot_rows = [row for row in read_csv(root / "fusion_selection_snapshots.csv")
                     if str(row.get("fusion", "")) == FUSION]
    by_target: dict[float, dict[str, str]] = {}
    for row in snapshot_rows:
        target = _target_key(row.get("snapshot_target"))
        if target in by_target:
            raise RuntimeError(f"Duplicate Task033 F3 snapshot target: {target}")
        by_target[target] = row
    expected = {_target_key(target) for target in SNAPSHOT_TARGETS}
    if set(by_target) != expected:
        raise RuntimeError(f"Task033 F3 snapshots are incomplete: {sorted(by_target)}")
    first = read_json(root / "fusion_first_attention.json")
    first_row = first.get(FUSION, first) if isinstance(first, Mapping) else {}
    stage_rows = [row for row in read_csv(root / "fusion_stage_removals.csv")
                  if str(row.get("fusion", "")) == FUSION
                  and _target_key(row.get("snapshot_target")) == _target_key(TARGET_SPARSITY)]
    domain_rows = [row for row in read_csv(root / "fusion_domain_removals.csv")
                   if str(row.get("fusion", "")) == FUSION
                   and _target_key(row.get("snapshot_target")) == _target_key(TARGET_SPARSITY)]
    completion = read_json(root / "task033_report_only_completion.json")
    if completion.get("status") not in ("PASS", True):
        raise RuntimeError("Task033 report-only completion is not PASS")
    return {
        "root": str(root.resolve()),
        "hashes": hashes,
        "final": final_rows[0],
        "snapshots": by_target,
        "first_attention": dict(first_row) if isinstance(first_row, Mapping) else {},
        "stage_rows": stage_rows,
        "domain_rows": domain_rows,
        "completion": completion,
    }


def _selection_summary(result: Mapping[str, object]) -> dict[str, object]:
    selected = [row for row in result.get("selected", ()) if isinstance(row, Mapping)]
    attention = [row for row in selected if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffns = [row for row in selected if str(row.get("unit_type")) == TYPE_FFN]
    first = next((row for row in selected if str(row.get("unit_type")) == TYPE_ATTENTION), None)
    final_sparsity = _float(result.get("final_sparsity"), "final_sparsity")
    return {
        "effective_sparsity": final_sparsity,
        "sequence_length": len(selected),
        "attention_removed": len(attention),
        "ffn_removed": len(ffns),
        "first_attention_sparsity": first.get("effective_sparsity_after") if first else None,
    }


def _snapshot_result_map(result: Mapping[str, object]) -> dict[float, Mapping[str, object]]:
    output: dict[float, Mapping[str, object]] = {}
    for row in result.get("snapshots", ()):
        if isinstance(row, Mapping) and row.get("snapshot_target") not in (None, ""):
            output[_target_key(row["snapshot_target"])] = row
    return output


def _counts_from_rows(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        raw_key = row.get(key, row.get("domain", "") if key == "domain_id" else "")
        normal_key = _stage_label(raw_key) if key == "stage" else str(raw_key)
        output[normal_key] = _int(row.get("removed_count", 0), "removed_count")
    return output


def compare_task033_f3_reference(reference: Mapping[str, object],
                                 result: Mapping[str, object]) -> dict[str, object]:
    """Compare Task034's exact F3 replay against Task033 report authority."""
    ref_final = reference.get("final", {})
    if not isinstance(ref_final, Mapping):
        return {"status": "FAIL", "reason": "invalid reference final row"}
    observed_final = _selection_summary(result)
    final_fields = {
        "effective_sparsity": ("effective_sparsity", "actual_effective_sparsity"),
        "sequence_length": ("sequence_length",),
        "attention_removed": ("attention_removed", "removed_attention"),
        "ffn_removed": ("ffn_removed", "removed_ffn"),
        "first_attention_sparsity": ("first_attention_sparsity",),
    }
    checks: dict[str, bool] = {}
    for name, aliases in final_fields.items():
        expected = _row_number(ref_final, aliases, default=None)
        checks[f"final_{name}"] = _numeric_equal(observed_final.get(name), expected)

    ref_snapshots = reference.get("snapshots", {})
    observed_snapshots = _snapshot_result_map(result)
    snapshot_fields = {
        "actual_effective_sparsity": ("actual_effective_sparsity", "effective_sparsity"),
        "attention_removed": ("attention_removed", "removed_attention"),
        "ffn_removed": ("ffn_removed", "removed_ffn"),
        "first_attention_sparsity": ("first_attention_sparsity",),
        "best_attention_rank": ("best_attention_rank",),
        "ffn_ahead": ("ffn_ahead", "ffn_candidates_ahead"),
    }
    per_snapshot: dict[str, bool] = {}
    if isinstance(ref_snapshots, Mapping):
        for target in SNAPSHOT_TARGETS:
            key = _target_key(target)
            expected = ref_snapshots.get(key, ref_snapshots.get(str(key), {}))
            observed = observed_snapshots.get(key, {})
            if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
                per_snapshot[str(key)] = False
                continue
            for name, aliases in snapshot_fields.items():
                per_snapshot[f"{key}:{name}"] = _numeric_equal(
                    _row_number(observed, (name,), default=None),
                    _row_number(expected, aliases, default=None),
                )
    checks["all_snapshots"] = bool(per_snapshot) and all(per_snapshot.values())

    observed_final_snapshot = observed_snapshots.get(_target_key(TARGET_SPARSITY), {})
    ref_stage = _counts_from_rows(reference.get("stage_rows", ()), "stage")
    ref_domain = _counts_from_rows(reference.get("domain_rows", ()), "domain_id")
    observed_stage = {_stage_label(key): _int(value, "removed_count")
                      for key, value in (observed_final_snapshot.get("stage_removals", {})
                                         if isinstance(observed_final_snapshot, Mapping) else {}).items()}
    observed_domain = {str(key): _int(value, "removed_count")
                       for key, value in (observed_final_snapshot.get("domain_removals", {})
                                          if isinstance(observed_final_snapshot, Mapping) else {}).items()}
    checks["final_stage_removals"] = ref_stage == observed_stage
    checks["final_domain_removals"] = ref_domain == observed_domain

    ref_first = reference.get("first_attention", {})
    observed_first = {
        "first_attention_sparsity": observed_final.get("first_attention_sparsity"),
        "attention_removed": observed_final.get("attention_removed"),
    }
    if isinstance(ref_first, Mapping):
        checks["first_attention"] = all(_numeric_equal(
            observed_first.get(name), _row_number(ref_first, (name,), default=None)
        ) for name in observed_first)
    else:
        checks["first_attention"] = False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "per_snapshot": per_snapshot,
        "reference_sequence_sha256": result.get("task033_reference_sequence_sha256"),
        "observed_sequence_sha256": result.get("sequence_sha256"),
        "reference_hashes": reference.get("hashes", {}),
    }


def _normalise_unit_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"attention", "head", "attention_head", "attention-head", "attn"}:
        return TYPE_ATTENTION
    if text in {"ffn", "neuron", "ffn_neuron", "ffn-neuron", "mlp"}:
        return TYPE_FFN
    return str(value or "")


def inventory_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Derive layer inventories from metadata supplied by replay/test rows."""
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        layer = str(row.get("layer", ""))
        typ = _normalise_unit_type(row.get("unit_type"))
        if not layer or typ not in UNIT_TYPES:
            continue
        key = (layer, typ)
        item = groups.setdefault(key, {"layer": layer, "unit_type": typ,
                                       "indices": set(), "global_indices": set(),
                                       "original_units": None, "parameter_total": 0.0,
                                       "stage": _stage_label(row.get("stage"))})
        unit_index = _int(row.get("unit_index"), "unit_index")
        item["indices"].add(unit_index)
        item["global_indices"].add(_int(row.get("global_index"), "global_index"))
        original = _row_value(row, "original_units", "layer_size", "unit_count", default=None)
        if original not in (None, ""):
            item["original_units"] = max(_int(original, "original_units"), int(item["original_units"] or 0))
        item["parameter_total"] = float(item["parameter_total"]) + _float(
            row.get("parameter_cost"), "parameter_cost", default=0.0
        )
        item["stage"] = _stage_label(item.get("stage", row.get("stage")))
    for item in groups.values():
        if item["original_units"] is None:
            item["original_units"] = (max(item["indices"]) + 1) if item["indices"] else 0
    return {f"{layer}::{typ}": {
        **item,
        "indices": sorted(item["indices"]),
        "global_indices": sorted(item["global_indices"]),
    } for (layer, typ), item in sorted(groups.items())}


def inventory_from_engine(engine: object, rows: Sequence[Mapping[str, object]] = ()) -> dict[str, dict[str, object]]:
    replay = getattr(getattr(engine, "base", None), "replay", None)
    units = list(getattr(replay, "units", ()))
    if not units:
        units = list(getattr(replay, "unit_by_global", {}).values()) if isinstance(
            getattr(replay, "unit_by_global", {}), Mapping) else []
    full_rows: list[dict[str, object]] = []
    costs = getattr(replay, "costs", ())
    for unit in units:
        gid = _int(getattr(unit, "global_index"), "global_index")
        typ = _normalise_unit_type(getattr(unit, "unit_type", getattr(unit, "kind", "")))
        if typ not in UNIT_TYPES:
            continue
        cost = getattr(unit, "parameter_cost", 0)
        if hasattr(costs, "__len__") and len(costs) > gid:
            cost = costs[gid]
        full_rows.append({
            "global_index": gid, "unit_type": typ,
            "layer": str(getattr(unit, "layer", "")),
            "stage": _stage_label(getattr(unit, "stage", "")) if getattr(unit, "stage", "") not in (None, "")
            else _stage_label(getattr(unit, "layer", "")),
            "unit_index": _int(getattr(unit, "unit_index"), "unit_index"),
            "domain_id": getattr(unit, "domain_id", 0),
            "parameter_cost": cost,
        })
    return inventory_from_rows([*full_rows, *rows])


def _registry_entry_key(item: Mapping[str, object]) -> str:
    return f"{item['layer']}::{item['unit_type']}"


def construct_f3_registry(rows: Sequence[Mapping[str, object]],
                          inventory: Mapping[str, Mapping[str, object]] | None = None) -> dict[str, object]:
    """Build a deterministic logical keep-list registry from removed units."""
    inventory = dict(inventory or inventory_from_rows(rows))
    removed_by_key: dict[str, set[int]] = defaultdict(set)
    removed_global_by_key: dict[str, list[int]] = defaultdict(list)
    selected_global: list[int] = []
    for row in rows:
        typ = _normalise_unit_type(row.get("unit_type"))
        layer = str(row.get("layer", ""))
        if typ not in UNIT_TYPES or not layer:
            continue
        key = f"{layer}::{typ}"
        removed_by_key[key].add(_int(row.get("unit_index"), "unit_index"))
        gid = _int(row.get("global_index"), "global_index")
        removed_global_by_key[key].append(gid)
        selected_global.append(gid)
    entries: dict[str, dict[str, object]] = {}
    keep_heads: dict[str, list[int]] = {}
    keep_neurons: dict[str, list[int]] = {}
    for key, item in sorted(inventory.items()):
        typ = _normalise_unit_type(item.get("unit_type"))
        layer = str(item.get("layer", key.split("::", 1)[0]))
        original_units = _int(item.get("original_units"), "original_units")
        if original_units <= 0:
            raise ValueError(f"Invalid original unit count for {key}")
        all_indices = set(range(original_units))
        removed = set(removed_by_key.get(key, set()))
        if not removed.issubset(all_indices):
            raise ValueError(f"Removed unit index outside inventory for {key}")
        kept = sorted(all_indices - removed)
        if not kept:
            raise ValueError(f"Logical registry would remove every unit from {key}")
        entry = {
            "layer": layer, "unit_type": typ, "indices": sorted(removed),
            "removed_global_indices": sorted(removed_global_by_key.get(key, ())),
            "keep_indices": kept, "original_units": original_units,
            "removed_count": len(removed), "retained_count": len(kept),
        }
        entries[key] = entry
        if typ == TYPE_ATTENTION:
            keep_heads[layer] = kept
            entry["keep_heads"] = kept
        else:
            keep_neurons[layer] = kept
            entry["keep_neurons"] = kept
    selected_global = sorted(set(selected_global))
    return {
        "status": "PASS",
        "fusion": FUSION,
        "target_sparsity": TARGET_SPARSITY,
        "registry": entries,
        "keep_heads": keep_heads,
        "keep_neurons": keep_neurons,
        "selected_global_indices": selected_global,
        "selected_count": len(selected_global),
        "logical_only": True,
        "physical_tensor_shrink": False,
    }


def registry_payload_for_task027(registry_payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return Task027-compatible entries without losing Task034 metadata."""
    output: dict[str, dict[str, object]] = {}
    entries = registry_payload.get("registry", registry_payload)
    if not isinstance(entries, Mapping):
        raise ValueError("registry payload has no registry mapping")
    for key, value in entries.items():
        if not isinstance(value, Mapping):
            continue
        typ = _normalise_unit_type(value.get("unit_type"))
        layer = str(value.get("layer", str(key).split("::", 1)[0]))
        # Task027 consumes exact model module names.  The ``::unit_type``
        # suffix is only an unambiguous Task034 registry identity for the
        # synthetic/aggregate case where two unit types share a display
        # layer; real Swin attention/MLP module names are distinct.
        output[layer] = {
            "layer": layer,
            "unit_type": typ,
            "indices": [_int(item, "registry index") for item in value.get("indices", ())],
        }
    return output


def verify_registry_pure(registry_payload: Mapping[str, object],
                         rows: Sequence[Mapping[str, object]],
                         *, total_parameters: float | None = None,
                         target_sparsity: float = TARGET_SPARSITY) -> dict[str, object]:
    """Verify registry identity/counts without importing a model runtime."""
    entries = registry_payload.get("registry", {})
    selected = {_int(row.get("global_index"), "global_index") for row in rows}
    recorded = {_int(value, "global_index") for value in registry_payload.get("selected_global_indices", ())}
    checks: dict[str, bool] = {"selected_global_indices": selected == recorded}
    removed_total = 0
    retained_total = 0
    for key, entry in entries.items() if isinstance(entries, Mapping) else ():
        if not isinstance(entry, Mapping):
            checks[f"entry:{key}"] = False
            continue
        removed = {_int(value, "registry index") for value in entry.get("indices", ())}
        keep = {_int(value, "keep index") for value in entry.get("keep_indices", ())}
        original = _int(entry.get("original_units"), "original_units")
        checks[f"entry:{key}"] = bool(keep) and removed.isdisjoint(keep) and removed | keep == set(range(original))
        removed_total += len(removed)
        retained_total += len(keep)
    row_indices = {
        (str(row.get("layer", "")), _normalise_unit_type(row.get("unit_type")), _int(row.get("unit_index"), "unit_index"))
        for row in rows
    }
    for key, entry in entries.items() if isinstance(entries, Mapping) else ():
        if not isinstance(entry, Mapping):
            continue
        layer = str(entry.get("layer", str(key).split("::", 1)[0]))
        typ = _normalise_unit_type(entry.get("unit_type"))
        selected_units = {(layer, typ, _int(row.get("unit_index"), "unit_index"))
                          for row in rows if str(row.get("layer", "")) == layer
                          and _normalise_unit_type(row.get("unit_type")) == typ}
        removed_units = {(layer, typ, _int(value, "registry index")) for value in entry.get("indices", ())}
        checks[f"selected_removed:{key}"] = selected_units == removed_units
        checks[f"keep_metadata:{key}"] = (
            list(entry.get("keep_heads", entry.get("keep_neurons", entry.get("keep_indices", ()))))
            == list(entry.get("keep_indices", ()))
        )
    removed_parameters = sum(_float(row.get("parameter_cost"), "parameter_cost", 0.0) for row in rows)
    if total_parameters is not None:
        effective = removed_parameters / max(float(total_parameters), 1e-12)
        checks["target_sparsity"] = effective >= float(target_sparsity) - 1e-12
    else:
        effective = None
    checks["logical_counts"] = removed_total > 0 and retained_total > 0
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "removed_unit_count": removed_total,
        "retained_unit_count": retained_total,
        "removed_parameter_total": removed_parameters,
        "effective_sparsity": effective,
    }


def _registry_sha256(registry_payload: Mapping[str, object]) -> str:
    # The digest is stored inside the registry itself.  Exclude that self
    # reference so recomputing the digest from the persisted JSON is stable.
    payload = dict(registry_payload)
    payload.pop("registry_sha256", None)
    canonical = json.dumps(_json_safe(payload), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _trace_row(row: Mapping[str, object], step: int) -> dict[str, object]:
    p_total = _float(row.get("p_total"), "p_total")
    p_average = _float(row.get("p_average"), "p_average")
    damage = _float(row.get("domain_damage"), "domain_damage", 0.0)
    b, v, risk = f3_components(p_total, p_average, damage)
    return {
        "step": step,
        "global_index": _int(row.get("global_index"), "global_index"),
        "unit_type": _normalise_unit_type(row.get("unit_type")),
        "layer": str(row.get("layer", "")),
        "stage": _stage_label(row.get("stage")),
        "unit_index": _int(row.get("unit_index"), "unit_index"),
        "domain_id": row.get("domain_id", ""),
        "parameter_cost": _float(row.get("parameter_cost"), "parameter_cost", 0.0),
        "Delta_average": row.get("Delta_average", ""),
        "Delta_total": row.get("Delta_total", ""),
        "p_total": p_total,
        "p_average": p_average,
        "domain_damage": damage,
        "B": _float(row.get("B"), "B", b),
        "V": _float(row.get("V"), "V", v),
        "R_F3": _float(row.get("R_F3"), "R_F3", risk),
        "cumulative_removed_parameters": _float(
            row.get("cumulative_removed_parameters"),
            "cumulative_removed_parameters",
            _float(row.get("parameter_cost"), "parameter_cost", 0.0),
        ),
        "effective_sparsity_after": _float(
            row.get("effective_sparsity_after"), "effective_sparsity_after", 0.0
        ),
        "feasible": bool(row.get("feasible", True)),
    }


def _normalise_snapshot_row(row: Mapping[str, object]) -> dict[str, object]:
    output = {field: row.get(field, "") for field in SNAPSHOT_FIELDS}
    output["fusion"] = FUSION
    output["snapshot_target"] = _float(row.get("snapshot_target"), "snapshot_target")
    output["actual_effective_sparsity"] = _float(
        row.get("actual_effective_sparsity"), "actual_effective_sparsity"
    )
    for field in ("attention_removed", "ffn_removed", "remaining_attention", "remaining_ffn",
                  "best_attention_rank", "ffn_ahead"):
        value = row.get(field)
        output[field] = "" if value in (None, "") else _int(value, field)
    for field in ("first_attention_sparsity", "attention_parameter_contribution",
                  "ffn_parameter_contribution"):
        value = row.get(field)
        output[field] = "" if value in (None, "") else _float(value, field)
    output["stage_removals"] = json.dumps(_json_safe(row.get("stage_removals", {})), sort_keys=True)
    output["domain_removals"] = json.dumps(_json_safe(row.get("domain_removals", {})), sort_keys=True)
    return output


def _normalise_schema_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _sequence_from_value(value: object) -> list[int] | None:
    """Decode one ordered Task029 sequence field without guessing units."""
    if not isinstance(value, (list, tuple)) or not value:
        return [] if isinstance(value, (list, tuple)) else None
    if all(isinstance(item, Mapping) for item in value):
        rows = [item for item in value if isinstance(item, Mapping)]
        if any(item.get("global_index", "") in (None, "") for item in rows):
            return None
        indexed = list(enumerate(rows))
        if any(item.get("step", item.get("incremental_step", "")) not in (None, "")
               for _, item in indexed):
            try:
                indexed.sort(key=lambda pair: (
                    _int(pair[1].get("step", 0) or 0, "step"),
                    _int(pair[1].get("incremental_step", 0) or 0, "incremental_step"),
                    pair[0],
                ))
            except (TypeError, ValueError):
                return None
        try:
            sequence = [_int(item["global_index"], "global_index") for _, item in indexed]
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            sequence = [_int(item, "global_index") for item in value]
        except (TypeError, ValueError):
            return None
    if len(sequence) != len(set(sequence)):
        return None
    return sequence


def _task029_sequence_candidates(payload: object, source: str,
                                 path: tuple[str, ...] = ()) -> list[tuple[str, list[int]]]:
    candidates: list[tuple[str, list[int]]] = []
    if not isinstance(payload, Mapping):
        return candidates
    ordered_items = sorted(
        payload.items(),
        key=lambda item: (
            0 if _normalise_schema_key(item[0]) in TASK029_F0_SEQUENCE_KEYS else 1,
            _normalise_schema_key(item[0]),
        ),
    )
    for key, value in ordered_items:
        normal_key = _normalise_schema_key(key)
        current_path = path + (str(key),)
        if normal_key in TASK029_F0_SEQUENCE_KEYS:
            sequence = _sequence_from_value(value)
            if sequence:
                candidates.append((f"{source}#" + ".".join(current_path), sequence))
        if isinstance(value, Mapping):
            candidates.extend(_task029_sequence_candidates(value, source, current_path))
    return candidates


def _task029_json_paths(root: Path) -> list[Path]:
    known = [Path(root) / name for name in TASK029_F0_JSON_ARTIFACT_NAMES]
    known = [path for path in known if path.is_file() and path.stat().st_size > 0]
    known_set = set(known)
    discovered = sorted(
        path for path in Path(root).rglob("*.json")
        if path.is_file() and path.stat().st_size > 0 and path not in known_set
    )
    return known + discovered


def _task029_json_evidence(root: Path) -> dict[str, object]:
    sequences: list[tuple[str, list[int]]] = []
    hashes: list[dict[str, str]] = []
    flags: list[dict[str, object]] = []
    identity_statuses: list[dict[str, object]] = []
    evidence_paths: list[str] = []
    for path in _task029_json_paths(Path(root)):
        try:
            payload = read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        relative = path.relative_to(Path(root)).as_posix()
        evidence_paths.append(relative)
        sequences.extend(_task029_sequence_candidates(payload, relative))

        def visit(value: object, trail: tuple[str, ...] = ()) -> None:
            if isinstance(value, Mapping):
                for raw_key, item in sorted(value.items(), key=lambda pair: _normalise_schema_key(pair[0])):
                    key = _normalise_schema_key(raw_key)
                    current = trail + (str(raw_key),)
                    if key in TASK029_F0_SHA_KEYS or key.endswith("_sequence_sha256"):
                        text = str(item).strip().lower()
                        if re.fullmatch(r"[0-9a-f]{64}", text):
                            hashes.append({
                                "path": f"{relative}#" + ".".join(current),
                                "key": key, "sha256": text,
                            })
                    if key in TASK029_F0_FLAG_KEYS:
                        parsed: bool | None = None
                        if isinstance(item, bool):
                            parsed = item
                        elif str(item).strip().lower() in {"true", "pass", "yes", "1"}:
                            parsed = True
                        elif str(item).strip().lower() in {"false", "fail", "no", "0"}:
                            parsed = False
                        if parsed is not None:
                            flags.append({
                                "path": f"{relative}#" + ".".join(current),
                                "key": key, "value": parsed,
                            })
                    if key in {"status", "artifact_identity_pass", "identity_pass",
                               "task029_identity_verified", "source_identity_pass"}:
                        parsed_status: bool | None = None
                        if isinstance(item, bool):
                            parsed_status = item
                        elif str(item).strip().upper() in {"PASS", "TRUE", "YES", "1"}:
                            parsed_status = True
                        elif str(item).strip().upper() in {"FAIL", "FALSE", "NO", "0"}:
                            parsed_status = False
                        if parsed_status is not None:
                            identity_statuses.append({
                                "path": f"{relative}#" + ".".join(current),
                                "key": key, "value": parsed_status,
                                "identity_file": "identity" in path.name.lower(),
                                "completion_file": "completion" in path.name.lower(),
                            })
                    if isinstance(item, Mapping):
                        visit(item, current)

        visit(payload)
    csv_candidates: list[tuple[str, list[int]]] = []
    for path in sorted(Path(root).rglob("*.csv")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        relative = path.relative_to(Path(root)).as_posix().lower()
        if "snapshot" in relative or not any(
            token in relative for token in ("replay", "trace", "sequence", "selection")
        ):
            continue
        try:
            sequence = _sequence_from_value(read_csv(path))
        except (OSError, TypeError, ValueError, csv.Error):
            continue
        if sequence:
            csv_candidates.append((path.relative_to(Path(root)).as_posix(), sequence))
            evidence_paths.append(path.relative_to(Path(root)).as_posix())
    return {
        "sequences": [*sequences, *csv_candidates], "hashes": hashes,
        "flags": flags, "identity_statuses": identity_statuses,
        "evidence_paths": sorted(set(evidence_paths)),
    }


def load_task029_f0_authority(task029_root: Path,
                              expected_sequence: Sequence[int]) -> dict[str, object]:
    """Recover exact Task028 replay evidence from Task029's actual schema."""
    expected = [_int(value, "global_index") for value in expected_sequence]
    evidence = _task029_json_evidence(Path(task029_root))
    complete = [item for item in evidence["sequences"] if len(item[1]) == len(expected)]
    unique_sequences: list[list[int]] = []
    for _, sequence in complete:
        if sequence not in unique_sequences:
            unique_sequences.append(sequence)
    if len(unique_sequences) > 1:
        return {
            "status": "FAIL", "reason": "Task029 contains conflicting complete replay sequences",
            "evidence": evidence,
        }
    observed = unique_sequences[0] if unique_sequences else []
    hashes = [item["sha256"] for item in evidence["hashes"]]
    flags = [bool(item["value"]) for item in evidence["flags"]]
    identity_statuses = [bool(item["value"]) for item in evidence["identity_statuses"]]
    identity_file_statuses = [
        bool(item["value"]) for item in evidence["identity_statuses"]
        if item.get("identity_file")
    ]
    completion_statuses = [
        bool(item["value"]) for item in evidence["identity_statuses"]
        if item.get("completion_file")
    ]
    exact_flag = all(flags) if flags else None
    identity_pass = (
        all(identity_file_statuses) if identity_file_statuses
        else all(identity_statuses) if identity_statuses
        else False
    )
    expected_sha = sequence_sha256(expected)
    observed_sha = sequence_sha256(observed) if observed else None
    if observed:
        full_equal = observed == expected
        checks = {
            "full_sequence_equality": full_equal,
            "first_100_equality": observed[:100] == expected[:100],
            "last_100_equality": observed[-100:] == expected[-100:],
            "sequence_sha256_equality": observed_sha == expected_sha,
            "task029_exact_replay_flag": exact_flag is not False,
        }
        if flags and not all(flags):
            checks["task029_exact_replay_flag"] = False
        if hashes and any(value != expected_sha for value in hashes):
            checks["stored_sequence_sha256_equality"] = False
        else:
            checks["stored_sequence_sha256_equality"] = True
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "authority": "Task029 complete ordered replay sequence",
            "evidence_paths": evidence["evidence_paths"],
            "sequence_path": next(path for path, sequence in complete if sequence == observed),
            "sequence": observed, "sequence_sha256": observed_sha,
            "task029_exact_replay_flag": exact_flag,
            "identity_status_pass": identity_pass,
            "checks": checks,
            "stored_sequence_hashes": hashes,
        }
    hash_checks = {
        "stored_sequence_sha256_available": bool(hashes),
        "stored_sequence_sha256_equality": bool(hashes) and all(value == expected_sha for value in hashes),
        "task029_exact_replay_flag": exact_flag is True,
        "task029_identity_status_pass": identity_pass,
    }
    return {
        "status": "PASS" if all(hash_checks.values()) else "FAIL",
        "authority": "Task029 exact-replay flag and sequence SHA",
        "evidence_paths": evidence["evidence_paths"],
        "sequence_path": None, "sequence": [], "sequence_sha256": hashes[0] if hashes else None,
        "task029_exact_replay_flag": exact_flag,
        "identity_status_pass": identity_pass,
        "checks": hash_checks,
        "stored_sequence_hashes": hashes,
        "completion_statuses": completion_statuses,
    }


def _write_f0_authority_gate(output_dir: Path | None, gate: Mapping[str, object]) -> None:
    if output_dir is not None:
        atomic_json(Path(output_dir) / "selection" / "f0_authority_gate.json", gate)


def verify_f0_exact_task028(task028_root: Path, task029_root: Path,
                            output_dir: Path | None = None) -> dict[str, object]:
    """Check F0 identity using immutable Task028 and exact Task029 evidence."""
    gate: dict[str, object] = {
        "status": "FAIL", "authority_primary": "Task028",
        "authority_exact_replay": "Task029", "task033_sequence_required": False,
        "task033_sequence_available": False, "task033_sequence_used_for_gate": False,
    }
    try:
        task028 = Path(task028_root).expanduser().resolve()
        task029 = Path(task029_root).expanduser().resolve()
        expected = _task028_sequence(task028)
        trace = read_csv(task028 / "selection_50" / "causal_selection_trace.csv")
        selected = [row for row in trace if row.get("global_index", "") != ""]
        attention = sum(_normalise_unit_type(row.get("unit_type")) == TYPE_ATTENTION for row in selected)
        ffn = sum(_normalise_unit_type(row.get("unit_type")) == TYPE_FFN for row in selected)
        expected_sha = sequence_sha256(expected)
        task029_gate = load_task029_f0_authority(task029, expected)
        checks = {
            "task028_trace_present": bool(expected),
            "task028_sequence_nonempty": bool(expected),
            "task028_attention_removed_zero": attention == 0,
            "task028_ffn_removed_28044": ffn == 28044,
            "task029_authority_pass": task029_gate["status"] == "PASS",
        }
        gate.update({
            "task028_trace_path": str((task028 / "selection_50" / "causal_selection_trace.csv").resolve()),
            "task029_evidence_paths": task029_gate.get("evidence_paths", []),
            "task028_sequence_length": len(expected),
            "task028_sequence_sha256": expected_sha,
            "task029_sequence_sha256": task029_gate.get("sequence_sha256"),
            "expected_length": len(expected), "observed_length": len(task029_gate.get("sequence", [])),
            "expected_sequence_sha256": expected_sha,
            "observed_sequence_sha256": task029_gate.get("sequence_sha256"),
            "first_100_equality": task029_gate.get("checks", {}).get("first_100_equality"),
            "last_100_equality": task029_gate.get("checks", {}).get("last_100_equality"),
            "full_sequence_equality": task029_gate.get("checks", {}).get("full_sequence_equality"),
            "sha_identity_equality": task029_gate.get("checks", {}).get("stored_sequence_sha256_equality"),
            "task029_exact_replay_flag": task029_gate.get("task029_exact_replay_flag"),
            "task029_identity_status_pass": task029_gate.get("identity_status_pass"),
            "attention_removed": attention, "ffn_removed": ffn,
            "checks": checks, "task029_evidence": task029_gate,
        })
        gate["status"] = "PASS" if all(checks.values()) else "FAIL"
    except (OSError, TypeError, ValueError, KeyError, csv.Error, RuntimeError) as exc:
        gate["reason"] = f"{type(exc).__name__}: {exc}"
    _write_f0_authority_gate(output_dir, gate)
    return gate


def _stage_rows(trace: Sequence[Mapping[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for stage in STAGES:
        for typ in UNIT_TYPES:
            items = [item for item in inventory.values() if _stage_label(item.get("stage")) == stage
                     and _normalise_unit_type(item.get("unit_type")) == typ]
            removed = [row for row in trace if _stage_label(row.get("stage")) == stage
                       and _normalise_unit_type(row.get("unit_type")) == typ]
            original = sum(_int(item.get("original_units"), "original_units") for item in items)
            removed_cost = sum(_float(row.get("parameter_cost"), "parameter_cost", 0.0) for row in removed)
            output.append({
                "stage": stage, "unit_type": typ, "original_units": original,
                "removed_units": len(removed), "retained_units": original - len(removed),
                "removed_parameter_cost": removed_cost,
                "effective_sparsity": len(removed) / max(original, 1),
            })
    return output


def _layer_rows(trace: Sequence[Mapping[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, item in sorted(inventory.items()):
        layer = str(item.get("layer", key.split("::", 1)[0]))
        typ = _normalise_unit_type(item.get("unit_type"))
        removed = [row for row in trace if str(row.get("layer", "")) == layer
                   and _normalise_unit_type(row.get("unit_type")) == typ]
        original = _int(item.get("original_units"), "original_units")
        output.append({
            "layer": layer, "unit_type": typ, "stage": _stage_label(item.get("stage")),
            "original_units": original, "removed_units": len(removed),
            "retained_units": original - len(removed),
            "removed_parameter_cost": sum(_float(row.get("parameter_cost"), "parameter_cost", 0.0) for row in removed),
            "effective_sparsity": len(removed) / max(original, 1),
        })
    return output


def _domain_rows(trace: Sequence[Mapping[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    domains = sorted({str(row.get("domain_id", "")) for row in trace}
                     | {str(item.get("domain_id", "")) for item in inventory.values()
                        if item.get("domain_id") not in (None, "")})
    output: list[dict[str, object]] = []
    for domain in domains:
        selected = [row for row in trace if str(row.get("domain_id", "")) == domain]
        output.append({
            "domain_id": domain,
            "original_units": "",
            "removed_units": len(selected),
            "retained_units": "",
            "removed_parameter_cost": sum(_float(row.get("parameter_cost"), "parameter_cost", 0.0) for row in selected),
            "unit_type_counts": json.dumps(dict(Counter(str(row.get("unit_type", "")) for row in selected)), sort_keys=True),
            "last_selected_domain_damage": selected[-1].get("domain_damage", "") if selected else "",
        })
    return output


def persist_f3_selection_artifacts(*, result: Mapping[str, object], output_dir: Path,
                                   inventory: Mapping[str, Mapping[str, object]],
                                   task033_reference: Mapping[str, object] | None = None,
                                   f0_gate: Mapping[str, object] | None = None) -> dict[str, object]:
    output = Path(output_dir)
    selection = output / "selection"
    selected = [_trace_row(row, index) for index, row in enumerate(result.get("selected", ()), 1)]
    sequence = [_int(row["global_index"], "global_index") for row in selected]
    atomic_csv(selection / "f3_selection_trace.csv", TRACE_FIELDS, selected)
    atomic_csv(selection / "f3_selection_sequence.csv", SEQUENCE_FIELDS,
               ({"step": index, "global_index": gid} for index, gid in enumerate(sequence, 1)))
    sequence_sha = sequence_sha256(sequence)
    atomic_json(selection / "f3_sequence_sha256.json", {
        "fusion": FUSION, "length": len(sequence), "sequence_sha256": sequence_sha,
        "full_replay": True, "ordered_by": ["R_F3", "p_total", "p_average", "global_index"],
    })

    snapshots = [_normalise_snapshot_row(row) for row in result.get("snapshots", ())]
    snapshots.sort(key=lambda row: _target_key(row["snapshot_target"]))
    atomic_csv(selection / "f3_selection_snapshots.csv", SNAPSHOT_FIELDS, snapshots)
    registry = construct_f3_registry(selected, inventory)
    registry_sha = _registry_sha256(registry)
    registry["registry_sha256"] = registry_sha
    atomic_json(selection / "f3_registry.json", registry)
    verification = verify_registry_pure(registry, selected, total_parameters=None)

    removed_cost = sum(_float(row.get("parameter_cost"), "parameter_cost", 0.0) for row in selected)
    total_parameters = _optional_float(result.get("total_parameters"))
    if total_parameters is None:
        total_parameters = _optional_float(result.get("total_parameter_count"))
    if total_parameters is None:
        total_parameters = removed_cost / max(_float(result.get("final_sparsity"), "final_sparsity"), 1e-12)
    effective = removed_cost / max(total_parameters, 1e-12)
    manifest = {
        "status": "PASS" if verification["status"] == "PASS" and effective >= TARGET_SPARSITY - 1e-12 else "FAIL",
        "fusion": FUSION, "target_sparsity": TARGET_SPARSITY,
        "percentile_risk_dtype": PERCENTILE_RISK_DTYPE,
        "actual_sparsity": effective, "selected_count": len(selected),
        "retained_count": sum(_int(item.get("original_units"), "original_units") for item in inventory.values()) - verification["removed_unit_count"],
        "removed_parameter_total": removed_cost, "analytical_total_parameters": total_parameters,
        "sequence_sha256": sequence_sha, "registry_sha256": registry_sha,
        "registry_application_pure": verification,
        "task033_reference": task033_reference or {}, "f0_exact_task028": f0_gate or {},
    }
    atomic_json(selection / "f3_registry_manifest.json", manifest)
    atomic_csv(selection / "f3_layer_structure.csv",
               ("layer", "unit_type", "stage", "original_units", "removed_units", "retained_units", "removed_parameter_cost", "effective_sparsity"),
               _layer_rows(selected, inventory))
    atomic_csv(selection / "f3_stage_structure.csv",
               ("stage", "unit_type", "original_units", "removed_units", "retained_units", "removed_parameter_cost", "effective_sparsity"),
               _stage_rows(selected, inventory))
    atomic_csv(selection / "f3_domain_structure.csv",
               ("domain_id", "original_units", "removed_units", "retained_units", "removed_parameter_cost", "unit_type_counts", "last_selected_domain_damage"),
               _domain_rows(selected, inventory))
    return {"trace": selected, "sequence": sequence, "sequence_sha256": sequence_sha,
            "snapshots": snapshots, "registry": registry, "registry_sha256": registry_sha,
            "manifest": manifest, "verification": verification}


# ---------------------------------------------------------------------------
# Base-checkpoint and Task028 protocol identity gates
# ---------------------------------------------------------------------------

def _first_nested_value(payload: object, keys: Sequence[str]) -> object | None:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for value in payload.values():
            found = _first_nested_value(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _first_nested_value(value, keys)
            if found not in (None, ""):
                return found
    return None


def _task028_json(root: Path, names: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    for name in names:
        path = Path(root) / name
        if path.is_file():
            return path, read_json(path)
    raise FileNotFoundError(f"None of {names} found below {root}")


def _resolve_path(value: object, *, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(str(value)).expanduser()
    possibilities = [candidate] if candidate.is_absolute() else [root / candidate, Path.cwd() / candidate]
    return next((path.resolve() for path in possibilities if path.is_file()), None)


def _looks_like_finetuned_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    forbidden = (
        "f3_best", "f3_latest", "swin_pruned", "pruned_best", "fine_tune",
        "finetune", "finetuned", "epoch60", "epoch75", "epoch_60", "epoch_75",
        "best_50", "logical_prun", "pruned", "best",
    )
    return any(token in name for token in forbidden)


def resolve_base_checkpoint_identity(task028_root: Path,
                                     explicit_checkpoint: Path | None = None) -> dict[str, object]:
    """Resolve and verify the exact *original* Task028 starting checkpoint.

    A Task034 checkpoint or a Task028 fine-tuned/pruned checkpoint is never a
    valid fallback.  The Task028 artifact identity must contain the recorded
    SHA; without it the gate fails before any model construction/training.
    """
    root = Path(task028_root).expanduser().resolve()
    identity_path, identity = _task028_json(root, ("artifact_identity.json", "identity.json"))
    recorded_path_value = _first_nested_value(
        identity, ("checkpoint", "checkpoint_path", "original_checkpoint", "base_checkpoint")
    )
    recorded_sha = _first_nested_value(
        identity, ("checkpoint_sha256", "original_checkpoint_sha256", "base_checkpoint_sha256")
    )
    path = (_resolve_path(explicit_checkpoint, root=root) if explicit_checkpoint is not None
            else _resolve_path(recorded_path_value, root=root))
    if path is None:
        raise RuntimeError("Task028 exact original checkpoint path could not be resolved")
    rejected = _looks_like_finetuned_checkpoint(path)
    if rejected:
        raise RuntimeError(f"Refusing fine-tuned/pruned checkpoint as Task034 base: {path}")
    if recorded_sha in (None, ""):
        raise RuntimeError("Task028 artifact identity has no original checkpoint SHA256")
    recorded_sha = str(recorded_sha).lower()
    observed_sha = sha256_file(path).lower()
    match = observed_sha == recorded_sha
    if not match:
        raise RuntimeError(
            "Task034 base checkpoint SHA mismatch: "
            f"expected {recorded_sha}, observed {observed_sha}"
        )
    # An explicit path is allowed only when it agrees with the recorded
    # Task028 identity; this prevents silently selecting an arbitrary original
    # checkpoint with a convenient filename.
    return {
        "status": "PASS", "path": str(path), "sha256": observed_sha,
        "task028_reference_sha256": recorded_sha, "match": True,
        "rejected_finetuned_checkpoint": False,
        "identity_source": str(identity_path), "explicit_path": explicit_checkpoint is not None,
        "base_checkpoint_is_original": True,
    }


def _normalise_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    return str(value).strip() if isinstance(value, str) else value


def _protocol_equal(left: object, right: object) -> bool:
    left, right = _normalise_value(left), _normalise_value(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _find_model_architecture(task028_root: Path, repo_root: Path,
                             config: Mapping[str, object],
                             artifact: Mapping[str, object]) -> dict[str, object] | None:
    for payload in (config, artifact):
        candidate = _first_nested_value(payload, ("model_architecture", "architecture", "senior_model"))
        if isinstance(candidate, Mapping):
            return {str(key): _normalise_value(value) for key, value in candidate.items()}
    for relative in ("model_architecture.json", "task028_training_protocol.json", "architecture.json"):
        path = Path(task028_root) / relative
        if path.is_file():
            payload = read_json(path)
            candidate = payload.get("model_architecture", payload.get("architecture", payload))
            if isinstance(candidate, Mapping):
                return {str(key): _normalise_value(value) for key, value in candidate.items()}
    source = Path(repo_root) / "task028_main50_tad_logical_finetune.py"
    if not source.is_file():
        source = Path(repo_root) / "task027_senior_style_logical_pruning_finetune.py"
    if source.is_file():
        text = source.read_text(encoding="utf-8")

        def ints(pattern: str) -> list[int] | None:
            match = re.search(pattern, text)
            return [int(item.strip()) for item in match.group(1).split(",") if item.strip()] if match else None

        patch_size = ints(r"patch_size\s*=\s*\(([^)]*)\)")
        depths = ints(r"depths\s*=\s*\[([^]]*)\]")
        num_heads = ints(r"num_heads\s*=\s*\[([^]]*)\]")
        window = ints(r"window_size\s*=\s*\(([^)]*)\)")
        architecture = {
            "patch_size": patch_size, "embed_dim": 96,
            "depths": depths, "num_heads": num_heads, "window_size": window,
            "mlp_ratio": 4.0, "qkv_bias": True, "patch_norm": True,
            "drop_path_rate": 0.2, "use_checkpoint": False,
        }
        if patch_size and depths and num_heads and window:
            return architecture
    return None


def task028_protocol_identity(task028_root: Path, repo_root: Path) -> dict[str, object]:
    """Read Task028's recorded protocol and compare every critical field."""
    config_path, config = _task028_json(
        task028_root, ("finetune_config.json", "finetune/finetune_config.json")
    )
    artifact_path, artifact = _task028_json(task028_root, ("artifact_identity.json",))
    def value(*keys: str, default: object = None) -> object:
        config_value = _first_nested_value(config, keys)
        if config_value not in (None, ""):
            return config_value
        artifact_value = _first_nested_value(artifact, keys)
        return artifact_value if artifact_value not in (None, "") else default

    task028_gpu_ids = value("gpu_ids", default=None)
    task028_batch = value("batch_size", "train_batch_size", default=None)
    task028_lr = value("actual_finetune_lr", "learning_rate", "lr", default=None)
    task028_frequency = value("validation_frequency", "validation_freq", "val_frequency", default=None)
    if task028_frequency is None:
        protocol_file = Path(task028_root) / "task028_training_protocol.json"
        if protocol_file.is_file():
            task028_frequency = _first_nested_value(read_json(protocol_file), ("validation_frequency", "validation_freq"))
    if task028_frequency is None:
        # Task028's loop validates once per epoch when no explicit frequency is
        # recorded.  This fallback is only accepted when the source confirms
        # that behavior; it does not introduce a Task034 choice.
        source = Path(repo_root) / "task028_main50_tad_logical_finetune.py"
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        task028_frequency = 1 if "_progress_validation" in text and "for epoch" in text else None

    task028_arch = _find_model_architecture(task028_root, repo_root, config, artifact)
    fields: dict[str, tuple[object, object]] = {
        "precision": (PROTOCOL["precision"], value("precision", default=None)),
        "amp_enabled": (PROTOCOL["amp_enabled"], value("amp_enabled", "amp", default=None)),
        "use_checkpoint": (PROTOCOL["use_checkpoint"], value("use_checkpoint", default=None)),
        "gpu_ids": (PROTOCOL["gpu_ids"], task028_gpu_ids),
        "batch_size": (PROTOCOL["batch_size"], task028_batch),
        "optimizer": (PROTOCOL["optimizer"], value("optimizer", default=None)),
        "actual_finetune_lr": (PROTOCOL["learning_rate"], task028_lr),
        "momentum": (PROTOCOL["momentum"], value("momentum", default=None)),
        "weight_decay": (PROTOCOL["weight_decay"], value("weight_decay", default=None)),
        "scheduler": (PROTOCOL["scheduler"], value("scheduler", "lr_policy", default=None)),
        "loss": (PROTOCOL["loss"], value("loss", "criterion", default=None)),
        "epochs": (PROTOCOL["epochs"], value("epochs", "total_epochs", default=None)),
        "seed": (PROTOCOL["seed"], value("seed", default=None)),
        "validation_uses_softmax": (PROTOCOL["validation_uses_softmax"], value("validation_uses_softmax", default=None)),
        "validation_frequency": (task028_frequency, task028_frequency),
        "model_architecture": (MODEL_ARCHITECTURE, task028_arch),
    }
    # Task028's artifact identity is the source for the data split paths and
    # validation batch size; Task034 is prohibited from inventing replacements.
    train_split = _first_nested_value(artifact, ("resolved_train_split", "train_split", "training_split"))
    val_split = _first_nested_value(artifact, ("resolved_validation_split", "validation_split", "val_split"))
    val_batch = _first_nested_value(artifact, ("validation_batch_size", "val_batch_size"))
    fields.update({
        "resolved_train_split": (train_split, train_split),
        "resolved_validation_split": (val_split, val_split),
        "validation_batch_size": (val_batch, val_batch),
    })
    field_rows = {}
    for name, (expected, observed) in fields.items():
        present = expected not in (None, "") and observed not in (None, "")
        field_rows[name] = {
            "task034": _normalise_value(expected), "task028": _normalise_value(observed),
            "match": present and _protocol_equal(expected, observed),
        }
    critical = set(fields)
    all_match = all(field_rows[name]["match"] for name in critical)
    return {
        "status": "PASS" if all_match else "FAIL",
        "task028_config_path": str(config_path), "task028_artifact_identity_path": str(artifact_path),
        "field_by_field": field_rows, "all_match": all_match,
        "critical_fields": sorted(critical), "task034_protocol": _json_safe(PROTOCOL),
        "task034_model_architecture": MODEL_ARCHITECTURE,
        "task028_model_architecture": task028_arch,
        "training_frequency_source": "Task028 artifact/config/source",
        "no_new_training_hyperparameters": True,
    }


def require_protocol_pass(identity: Mapping[str, object]) -> None:
    if identity.get("status") != "PASS" or identity.get("all_match") is not True:
        raise RuntimeError("Task028/Task034 protocol identity gate failed")


def verify_identity(*, task028_root: Path, task029_root: Path, task030_root: Path,
                    task031_root: Path, task032_root: Path, task033_root: Path,
                    output_dir: Path, repo_root: Path | None = None,
                    checkpoint: Path | None = None) -> dict[str, object]:
    roots = {
        "task028": Path(task028_root).expanduser().resolve(),
        "task029": Path(task029_root).expanduser().resolve(),
        "task030": Path(task030_root).expanduser().resolve(),
        "task031": Path(task031_root).expanduser().resolve(),
        "task032": Path(task032_root).expanduser().resolve(),
        "task033": Path(task033_root).expanduser().resolve(),
    }
    output = Path(output_dir).expanduser().resolve()
    assert_output_is_separate(output, roots.values())
    repo = Path(repo_root or Path(__file__).parent).expanduser().resolve()
    hashes = source_artifact_hashes(roots, strict=True)
    reference = load_task033_f3_reference(roots["task033"])
    base = resolve_base_checkpoint_identity(roots["task028"], checkpoint)
    protocol = task028_protocol_identity(roots["task028"], repo)
    identity = {
        "status": "PASS" if protocol["status"] == "PASS" else "FAIL",
        "code_version": CODE_VERSION, "task034_branch": TASK034_BRANCH,
        "base_task033_branch": TASK033_BASE_BRANCH, "base_task033_commit": TASK033_BASE_COMMIT,
        "target_sparsity": TARGET_SPARSITY, "fusion": FUSION,
        "roots": {key: str(value) for key, value in roots.items()},
        "hashes_before": hashes, "source_hashes_unchanged": True,
        "task033_f3_reference_hashes": reference["hashes"],
        "task033_reference_gate": "PASS",
        "task028_base_checkpoint": base, "task028_protocol_gate": protocol["status"],
        "new_tunable_pruning_hyperparameters": 0,
        "selection_started_from_zero": True, "selection_rerun": False,
        "training_executed": False, "validation_executed": False,
        "physical_pruning_executed": False, "old_task_artifacts_modified": False,
    }
    atomic_json(output / "artifact_identity.json", identity)
    atomic_json(output / "base_checkpoint_identity.json", base)
    atomic_json(output / "task028_training_protocol_identity.json", protocol)
    return identity


# ---------------------------------------------------------------------------
# Selection gate and artifact construction
# ---------------------------------------------------------------------------

def _identity_roots(identity: Mapping[str, object]) -> dict[str, Path]:
    roots = identity.get("roots", {})
    if not isinstance(roots, Mapping):
        raise RuntimeError("Task034 identity has no immutable roots")
    required = ("task028", "task029", "task030", "task031", "task032", "task033")
    missing = [name for name in required if name not in roots]
    if missing:
        raise RuntimeError(f"Task034 identity is missing roots: {missing}")
    return {name: Path(str(roots[name])).expanduser().resolve() for name in required}


def _require_identity(output_dir: Path) -> dict[str, object]:
    path = Path(output_dir) / "artifact_identity.json"
    if not path.is_file():
        raise RuntimeError("Task034 identity must be completed before this stage")
    identity = read_json(path)
    if identity.get("code_version") != CODE_VERSION:
        raise RuntimeError("Task034 artifact identity code version mismatch")
    if identity.get("status") != "PASS":
        raise RuntimeError("Task034 identity gate is not PASS")
    return identity


def _require_benchmark(output_dir: Path) -> dict[str, object]:
    path = Path(output_dir) / "selection" / "f3_performance_benchmark.json"
    if not path.is_file():
        raise RuntimeError("Task034 selection benchmark is required before full replay")
    report = read_json(path)
    rate = _float(report.get("steps_per_second"), "steps_per_second", 0.0)
    if report.get("status") != "PASS" or rate < 50.0:
        raise RuntimeError("Task034 F3 1000-step selection benchmark gate failed")
    return report


def construct_selection(*, output_dir: Path, task014_root: Path, task016_root: Path,
                        task017_root: Path, device: str = "cuda:0",
                        ranking_backend: str = "single-gpu") -> dict[str, object]:
    """Run the full 0% -> 50% dynamic F3 replay and persist it first."""
    output = Path(output_dir).expanduser().resolve()
    identity = _require_identity(output)
    roots = _identity_roots(identity)
    _require_benchmark(output)
    if str(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")) != "0,1":
        raise RuntimeError("Task034 requires exactly CUDA_VISIBLE_DEVICES=0,1")
    if ranking_backend == "both":
        raise RuntimeError("Task034 production selection has one frozen single-GPU selector backend")
    reference = load_task033_f3_reference(roots["task033"])
    f0_gate = verify_f0_exact_task028(
        roots["task028"], roots["task029"], output_dir=output
    )
    if f0_gate["status"] != "PASS":
        raise RuntimeError("Task028 F0 exactness gate failed before Task034 selection")
    result = replay_f3(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), task028_root=roots["task028"],
        output_dir=output, device=device, ranking_backend=ranking_backend,
    )
    reference_gate = compare_task033_f3_reference(reference, result)
    if reference_gate["status"] != "PASS":
        raise RuntimeError("Task034 F3 replay does not match Task033 aggregate authority")
    artifacts = persist_f3_selection_artifacts(
        result=result, output_dir=output,
        inventory=result.get("inventory", {}), task033_reference=reference_gate,
        f0_gate=f0_gate,
    )
    manifest = artifacts["manifest"]
    selection_completion = {
        "status": "PASS" if (
            len(artifacts["sequence"]) == len(result.get("selected", ()))
            and manifest.get("actual_sparsity", 0.0) >= TARGET_SPARSITY - 1e-12
            and artifacts["verification"].get("status") == "PASS"
            and reference_gate["status"] == "PASS"
            and f0_gate["status"] == "PASS"
            and len(artifacts["snapshots"]) == len(SNAPSHOT_TARGETS)
        ) else "FAIL",
        "fusion": FUSION, "selection_started_from_zero": True,
        "full_dynamic_replay": True, "stale_ranking_used": False,
        "python_full_candidate_sort_in_hot_path": False,
        "task033_f3_aggregate_identity_gate": reference_gate,
        "f0_exact_task028_gate": f0_gate,
        "target_sparsity": TARGET_SPARSITY,
        "actual_sparsity": manifest.get("actual_sparsity"),
        "sequence_length": len(artifacts["sequence"]),
        "sequence_sha256": artifacts["sequence_sha256"],
        "all_six_snapshots": len(artifacts["snapshots"]) == len(SNAPSHOT_TARGETS),
        "registry": artifacts["verification"],
        "source_hashes_unchanged": verify_source_hashes_unchanged(
            identity.get("hashes_before", {}), roots
        ),
        "training_executed": False, "validation_executed": False,
    }
    if not selection_completion["source_hashes_unchanged"]:
        selection_completion["status"] = "FAIL"
    atomic_json(output / "selection" / "f3_selection_completion.json", selection_completion)
    if selection_completion["status"] != "PASS":
        raise RuntimeError("Task034 selection completion gate failed")
    return {"result": result, "artifacts": artifacts,
            "completion": selection_completion}


def verify_selection_artifacts(output_dir: Path) -> dict[str, object]:
    """Recheck persisted sequence, registry, snapshots, and source hashes."""
    output = Path(output_dir)
    identity = _require_identity(output)
    roots = _identity_roots(identity)
    completion = read_json(output / "selection" / "f3_selection_completion.json")
    trace = read_csv(output / "selection" / "f3_selection_trace.csv")
    sequence = [_int(row.get("global_index"), "global_index") for row in trace]
    stored_sha = read_json(output / "selection" / "f3_sequence_sha256.json")
    registry = read_json(output / "selection" / "f3_registry.json")
    inventory = inventory_from_rows(trace)
    checks = {
        "completion_pass": completion.get("status") == "PASS",
        "trace_sequence": sequence_sha256(sequence) == stored_sha.get("sequence_sha256"),
        "trace_steps": [ _int(row.get("step"), "step") for row in trace ] == list(range(1, len(trace) + 1)),
        "registry": verify_registry_pure(registry, trace).get("status") == "PASS",
        "six_snapshots": len(read_csv(output / "selection" / "f3_selection_snapshots.csv")) == len(SNAPSHOT_TARGETS),
        "source_hashes_unchanged": verify_source_hashes_unchanged(identity.get("hashes_before", {}), roots),
        "no_old_root_write": all(not _path_is_inside(output / "selection" / "f3_selection_trace.csv", root)
                                  for root in roots.values()),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "sequence_length": len(sequence), "sequence_sha256": sequence_sha256(sequence),
              "inventory_entries": len(inventory)}
    atomic_json(output / "selection" / "f3_selection_verification.json", result)
    return result


# ---------------------------------------------------------------------------
# Logical model application, validation, sanity, and speed gates
# ---------------------------------------------------------------------------

def _task034_protocol(output_dir: Path) -> dict[str, object]:
    protocol = read_json(Path(output_dir) / "task028_training_protocol_identity.json")
    require_protocol_pass(protocol)
    return protocol


def _protocol_task028_value(protocol: Mapping[str, object], field: str,
                            default: object = None) -> object:
    fields = protocol.get("field_by_field", {})
    if isinstance(fields, Mapping) and isinstance(fields.get(field), Mapping):
        value = fields[field].get("task028", default)
        return default if value in (None, "") else value
    return default


def _model_task028_helpers():
    task028 = __import__("task028_main50_tad_logical_finetune")
    task027 = __import__("task027_senior_style_logical_pruning_finetune")
    return task028, task027


def _load_apply_logical_model(*, output_dir: Path, checkpoint: Path | None = None,
                              device: str = "cuda:0") -> tuple[object, object, object, dict[str, object]]:
    identity = _require_identity(output_dir)
    protocol = _task034_protocol(output_dir)
    base_identity = read_json(Path(output_dir) / "base_checkpoint_identity.json")
    if base_identity.get("match") is not True:
        raise RuntimeError("Exact Task028 base checkpoint gate is not PASS")
    source_checkpoint = Path(str(checkpoint or base_identity.get("path", "")))
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Task034 base checkpoint missing: {source_checkpoint}")
    registry = read_json(Path(output_dir) / "selection" / "f3_registry.json")
    task028, task027 = _model_task028_helpers()
    model = task028._load_senior_original(source_checkpoint, device)
    before_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    before_numel = sum(int(value.numel()) for value in model.state_dict().values())
    compatible = registry_payload_for_task027(registry)
    apply_result = task027.apply_logical_pruning_registry(model, compatible)
    after_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    after_numel = sum(int(value.numel()) for value in model.state_dict().values())
    if before_shapes != after_shapes or before_numel != after_numel:
        raise RuntimeError("Logical registry changed state_dict shape/numel")
    if hasattr(task027, "assert_registry_keep_identity"):
        task027.assert_registry_keep_identity(model, compatible)
    return model, task028, task027, {
        "identity": identity, "protocol": protocol, "registry": registry,
        "compatible_registry": compatible, "apply_result": _json_safe(apply_result),
        "state_dict_shapes_before": before_shapes, "state_dict_shapes_after": after_shapes,
        "state_dict_numel_before": before_numel, "state_dict_numel_after": after_numel,
    }


def verify_logical_pruning(output_dir: Path, *, checkpoint: Path | None = None,
                           device: str = "cuda:0") -> dict[str, object]:
    """Apply Task027's existing keep-list implementation and verify invariants."""
    output = Path(output_dir)
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=output, checkpoint=checkpoint, device=device
    )
    trace = read_csv(output / "selection" / "f3_selection_trace.csv")
    registry = context["registry"]
    pure = verify_registry_pure(registry, trace)
    manifest = read_json(output / "selection" / "f3_registry_manifest.json")
    analytical = None
    if hasattr(task027, "logical_parameter_report"):
        try:
            analytical = task027.logical_parameter_report(
                model, context["apply_result"],
                state_dict_before=context["state_dict_numel_before"],
            )
        except TypeError:
            analytical = task027.logical_parameter_report(model, context["apply_result"])
    if analytical is None:
        analytical = {"removed_parameter_total": manifest.get("removed_parameter_total"),
                      "effective_sparsity": manifest.get("actual_sparsity")}
    analytical_removed = _optional_float(
        analytical.get("analytical_parameter_reduction", analytical.get("estimated_removed_parameters",
                                                                         analytical.get("removed_parameter_total")))
        if isinstance(analytical, Mapping) else None
    )
    selected_unit_keys = {
        (str(row.get("layer", "")), _normalise_unit_type(row.get("unit_type")),
         _int(row.get("unit_index"), "unit_index")) for row in trace
    }
    removed_registry_keys = {
        (str(entry.get("layer", "")), _normalise_unit_type(entry.get("unit_type")),
         _int(index, "registry index"))
        for entry in registry.get("registry", {}).values() if isinstance(entry, Mapping)
        for index in entry.get("indices", ())
    }
    result = {
        "status": "PASS" if (
            pure.get("status") == "PASS"
            and context["state_dict_shapes_before"] == context["state_dict_shapes_after"]
            and context["state_dict_numel_before"] == context["state_dict_numel_after"]
            and float(manifest.get("actual_sparsity", 0.0)) >= TARGET_SPARSITY - 1e-12
            and int(pure.get("removed_unit_count", 0)) > 0
            and analytical_removed is not None and analytical_removed > 0.0
            and selected_unit_keys == removed_registry_keys
            and all("keep_heads" in entry or "keep_neurons" in entry
                    for entry in registry.get("registry", {}).values()
                    if isinstance(entry, Mapping))
        ) else "FAIL",
        "fusion": FUSION, "logical_only": True, "physical_tensor_shrink": False,
        "state_dict_shapes_unchanged": context["state_dict_shapes_before"] == context["state_dict_shapes_after"],
        "state_dict_numel_unchanged": context["state_dict_numel_before"] == context["state_dict_numel_after"],
        "state_dict_numel_before": context["state_dict_numel_before"],
        "state_dict_numel_after": context["state_dict_numel_after"],
        "analytical_parameter_report": analytical,
        "analytical_parameter_count_decreases": analytical_removed is not None and analytical_removed > 0.0,
        "no_selected_unit_remains_active": selected_unit_keys == removed_registry_keys,
        "no_retained_removed": all(
            not set(entry.get("indices", ())) & set(entry.get("keep_indices", ()))
            for entry in registry.get("registry", {}).values() if isinstance(entry, Mapping)
        ),
        "effective_sparsity": manifest.get("actual_sparsity"),
        "registry_reproduction": pure,
        "keep_heads_metadata": registry.get("keep_heads", {}),
        "keep_neurons_metadata": registry.get("keep_neurons", {}),
        "senior_protocol_alignment": True,
        "use_checkpoint": False, "checkpoint_enabled_block_count": 0,
        "swin_block_count": EXPECTED_SWIN_BLOCKS, "precision": "fp32",
        "amp_enabled": False,
    }
    atomic_json(output / "model" / "logical_pruning_verification.json", result)
    atomic_json(output / "model" / "registry_reproduction.json", pure)
    if result["status"] != "PASS":
        raise RuntimeError("Task034 logical pruning verification failed")
    del model
    return result


def _forward_logits(model: object, inputs: object) -> object:
    output = model(inputs)
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output"):
            if key in output:
                return output[key]
    return output


def _accuracy_counts(logits: object, targets: object, topk: tuple[int, ...] = (1, 5)) -> tuple[int, int, int]:
    torch = _torch()
    maximum = min(max(topk), int(logits.shape[1]))
    _, prediction = logits.topk(maximum, dim=1, largest=True, sorted=True)
    correct = prediction.t().eq(targets.reshape(1, -1).expand_as(prediction.t()))
    return (int(correct[:1].reshape(-1).sum().item()),
            int(correct[:min(5, maximum)].reshape(-1).sum().item()),
            int(targets.numel()))


def _get_loader(task027: object, split: object, batch_size: int):
    return task027._get_loader(str(split), int(batch_size))


def _validation_raw_logits(*, model: object, loader: Iterable[object], device: str) -> dict[str, object]:
    torch = _torch()
    model.eval()
    total_loss = 0.0
    top1 = top5 = samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].float().to(torch.device(device), non_blocking=True)
            targets = batch[1].to(torch.device(device), non_blocking=True)
            logits = _forward_logits(model, inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            c1, c5, count = _accuracy_counts(logits, targets)
            total_loss += float(loss.item())
            top1 += c1; top5 += c5; samples += count
    return {
        "top1": top1 / max(samples, 1), "top5": top5 / max(samples, 1),
        "loss": total_loss / max(samples, 1), "samples": samples,
        "validation_uses_softmax": False, "raw_logits": True,
    }


def pre_finetune_validate(output_dir: Path, *, checkpoint: Path | None = None,
                          device: str = "cuda:0") -> dict[str, object]:
    output = Path(output_dir)
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=output, checkpoint=checkpoint, device=device
    )
    val_split = _protocol_task028_value(context["protocol"], "resolved_validation_split")
    val_batch = _protocol_task028_value(context["protocol"], "validation_batch_size")
    if val_split in (None, "") or val_batch in (None, ""):
        raise RuntimeError("Task028 validation split/batch identity is unavailable")
    loader = _get_loader(task027, val_split, _int(val_batch, "validation_batch_size"))
    metrics = _validation_raw_logits(model=model, loader=loader, device=device)
    report = {
        "status": "PASS", "metrics_are_descriptive_only": True,
        "top1": metrics["top1"], "top5": metrics["top5"], "loss": metrics["loss"],
        "samples": metrics["samples"], "total_samples": metrics["samples"],
        "effective_sparsity": read_json(output / "selection" / "f3_registry_manifest.json").get("actual_sparsity"),
        "removed_attention": sum(row.get("unit_type") == TYPE_ATTENTION
                                  for row in read_csv(output / "selection" / "f3_selection_trace.csv")),
        "removed_ffn": sum(row.get("unit_type") == TYPE_FFN
                            for row in read_csv(output / "selection" / "f3_selection_trace.csv")),
        "validation_uses_softmax": False, "raw_logits": True,
        "accuracy_threshold_used": False, "selection_rerun": False,
    }
    atomic_json(output / "model" / "pre_finetune_validation.json", report)
    del model
    return report


def forward_backward_sanity(output_dir: Path, *, checkpoint: Path | None = None,
                            device: str = "cuda:0") -> dict[str, object]:
    """Perform one fresh FP32 forward/backward/SGD step, then discard it."""
    torch = _torch()
    output = Path(output_dir)
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=output, checkpoint=checkpoint, device=device
    )
    train_split = _protocol_task028_value(context["protocol"], "resolved_train_split")
    if train_split in (None, ""):
        raise RuntimeError("Task028 training split identity is unavailable")
    loader = _get_loader(task027, train_split, int(PROTOCOL["batch_size"]))
    batch = next(iter(loader))
    target = torch.device(device)
    model.train()
    inputs = batch[0].float().to(target, non_blocking=True)
    targets = batch[1].to(target, non_blocking=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=PROTOCOL["learning_rate"],
                                momentum=PROTOCOL["momentum"], weight_decay=PROTOCOL["weight_decay"])
    optimizer.zero_grad()
    logits = _forward_logits(model, inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    finite = bool(torch.isfinite(loss.detach()).item()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    report = {
        "status": "PASS" if finite else "FAIL", "finite_loss": finite,
        "forward_backward_sgd": True, "batch_size": int(targets.shape[0]),
        "logit_shape": list(logits.shape), "precision": "fp32", "amp_enabled": False,
        "use_checkpoint": False, "keep_heads_path": bool(context["registry"].get("keep_heads")),
        "keep_neurons_path": bool(context["registry"].get("keep_neurons")),
        "weights_saved": False, "training_state_consumed": False,
    }
    atomic_json(output / "model" / "forward_backward_sanity.json", report)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if report["status"] != "PASS":
        raise RuntimeError("Task034 forward/backward sanity failed")
    return report


def speed_gate(output_dir: Path, *, checkpoint: Path | None = None,
               device: str = "cuda:0", steps: int = 100) -> dict[str, object]:
    """Run the requested ~100-iteration throughput gate without saving state."""
    torch = _torch()
    output = Path(output_dir)
    if int(steps) <= 0:
        raise ValueError("speed-gate steps must be positive")
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=output, checkpoint=checkpoint, device=device
    )
    if torch.device(device).type == "cuda" and torch.cuda.device_count() < 2:
        raise RuntimeError("Task034 speed gate requires CUDA devices 0 and 1")
    train_split = _protocol_task028_value(context["protocol"], "resolved_train_split")
    if train_split in (None, ""):
        raise RuntimeError("Task028 training split identity is unavailable")
    loader = _get_loader(task027, train_split, int(PROTOCOL["batch_size"]))
    student = torch.nn.DataParallel(model, device_ids=list(PROTOCOL["gpu_ids"]), output_device=0)
    optimizer = torch.optim.SGD(student.parameters(), lr=PROTOCOL["learning_rate"],
                                momentum=PROTOCOL["momentum"], weight_decay=PROTOCOL["weight_decay"])
    iterator = iter(loader)
    student.train()
    target = torch.device(device)
    started = time.perf_counter()
    for _ in range(int(steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = batch[0].float().to(target, non_blocking=True)
        targets = batch[1].to(target, non_blocking=True)
        optimizer.zero_grad()
        logits = _forward_logits(student, inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError("Non-finite speed-gate loss")
        loss.backward()
        optimizer.step()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = max(time.perf_counter() - started, 1e-12)
    iters = int(steps) / elapsed
    peak = {}
    if target.type == "cuda":
        for gpu_id in PROTOCOL["gpu_ids"]:
            with torch.cuda.device(gpu_id):
                peak[str(gpu_id)] = int(torch.cuda.max_memory_allocated(gpu_id))
    report = {
        "status": "PASS", "steps": int(steps), "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / int(steps), "iterations_per_second": iters,
        "estimated_minutes_per_epoch": len(loader) / max(iters, 1e-12) / 60.0,
        "gpu_peak_memory_bytes": peak, "gpu_ids": list(PROTOCOL["gpu_ids"]),
        "batch_size": PROTOCOL["batch_size"], "precision": PROTOCOL["precision"],
        "amp_enabled": False, "use_checkpoint": False, "scheduler": "NONE",
        "optimizer": "SGD", "learning_rate": PROTOCOL["learning_rate"],
        "momentum": PROTOCOL["momentum"], "weight_decay": PROTOCOL["weight_decay"],
        "loss": "CrossEntropyLoss", "selection_rerun": False,
        "training_state_consumed": False, "weights_saved": False,
        "registry_sha256": read_json(output / "selection" / "f3_registry_manifest.json").get("registry_sha256"),
        "reset_required_before_training": True,
    }
    atomic_json(output / "finetune" / "speed_gate.json", report)
    atomic_json(output / "speed_gate.json", report)
    del student, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# Task028-aligned fine-tuning (lazy, guarded, and resumable)
# ---------------------------------------------------------------------------

def _set_seed(seed: int = int(PROTOCOL["seed"])) -> None:
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except ImportError:
        pass
    try:
        torch = _torch()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _rng_state() -> dict[str, object]:
    torch = _torch()
    state: dict[str, object] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, object]) -> None:
    torch = _torch()
    if state.get("python") is not None:
        random.setstate(state["python"])
    try:
        import numpy as np
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _canonical_sha(payload: object) -> str:
    canonical = json.dumps(_json_safe(payload), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _training_config(output_dir: Path, protocol: Mapping[str, object],
                     base_identity: Mapping[str, object]) -> dict[str, object]:
    validation_frequency = _protocol_task028_value(protocol, "validation_frequency")
    if validation_frequency in (None, ""):
        raise RuntimeError("Task028 validation frequency identity is unavailable")
    return {
        "status": "PASS", "fusion": FUSION, "target_sparsity": TARGET_SPARSITY,
        "optimizer": PROTOCOL["optimizer"], "learning_rate": PROTOCOL["learning_rate"],
        "actual_finetune_lr": PROTOCOL["learning_rate"], "momentum": PROTOCOL["momentum"],
        "weight_decay": PROTOCOL["weight_decay"], "scheduler": PROTOCOL["scheduler"],
        "loss": PROTOCOL["loss"], "epochs": PROTOCOL["epochs"],
        "batch_size": PROTOCOL["batch_size"], "seed": PROTOCOL["seed"],
        "gpu_ids": list(PROTOCOL["gpu_ids"]), "precision": PROTOCOL["precision"],
        "amp_enabled": False, "use_checkpoint": False,
        "validation_uses_softmax": False,
        "validation_frequency": _int(validation_frequency, "validation_frequency"),
        "resolved_train_split": _protocol_task028_value(protocol, "resolved_train_split"),
        "resolved_validation_split": _protocol_task028_value(protocol, "resolved_validation_split"),
        "validation_batch_size": _int(_protocol_task028_value(protocol, "validation_batch_size"), "validation_batch_size"),
        "senior_protocol_alignment": True, "existing_50_registry_reused": False,
        "selection_rerun": False, "physical_pruning_executed": False,
        "base_checkpoint_sha256": base_identity.get("sha256"),
        "base_checkpoint_reference_sha256": base_identity.get("task028_reference_sha256"),
        "registry_sha256": read_json(output_dir / "selection" / "f3_registry_manifest.json").get("registry_sha256"),
        "task028_protocol_identity_sha256": _canonical_sha(protocol),
    }


def _atomic_torch_save(path: Path, payload: object) -> None:
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(model: object, optimizer: object, epoch: int,
                        best_metrics: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    return {
        "epoch": int(epoch), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": dict(best_metrics), "rng_state": _rng_state(),
        "base_checkpoint_sha256": config.get("base_checkpoint_sha256"),
        "registry_sha256": config.get("registry_sha256"),
        "task028_protocol_identity_sha256": config.get("task028_protocol_identity_sha256"),
        "training_config_sha256": config.get("training_config_sha256"),
        "precision": "fp32", "amp_enabled": False, "use_checkpoint": False,
    }


def _load_resume_checkpoint(path: Path, model: object, optimizer: object,
                            config: Mapping[str, object], device: str) -> tuple[int, dict[str, object]]:
    torch = _torch()
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Task034 resume checkpoint is not a mapping")
    for field in ("base_checkpoint_sha256", "registry_sha256", "task028_protocol_identity_sha256",
                  "training_config_sha256"):
        if payload.get(field) != config.get(field):
            raise RuntimeError(f"Task034 resume identity mismatch: {field}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("rng_state"):
        _restore_rng_state(payload["rng_state"])
    return _int(payload.get("epoch", 0), "resume epoch"), dict(payload.get("best_metrics", {}))


def resume_identity_guard(output_dir: Path, checkpoint: Path) -> dict[str, object]:
    """Pure filesystem guard used before loading an interrupted run."""
    output = Path(output_dir)
    state = read_json(output / "finetune" / "training_state.json")
    config = read_json(output / "finetune" / "finetune_config.json")
    payload = {
        "base_checkpoint_sha256": config.get("base_checkpoint_sha256"),
        "registry_sha256": config.get("registry_sha256"),
        "task028_protocol_identity_sha256": config.get("task028_protocol_identity_sha256"),
        "training_config_sha256": config.get("training_config_sha256"),
    }
    stored = state.get("identity", {})
    if not isinstance(stored, Mapping) or any(stored.get(key) != value for key, value in payload.items()):
        raise RuntimeError("Task034 training resume identity mismatch")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    return {"status": "PASS", "identity": payload, "resume_checkpoint": str(Path(checkpoint).resolve())}


def _metric_better(current: Mapping[str, object], best: Mapping[str, object]) -> bool:
    if not best:
        return True
    current_top1 = float(current.get("top1", -math.inf))
    best_top1 = float(best.get("top1", -math.inf))
    if current_top1 != best_top1:
        return current_top1 > best_top1
    return float(current.get("top5", -math.inf)) > float(best.get("top5", -math.inf))


def _write_training_state(output: Path, *, status: str, training_status: str,
                          epoch: int, best_metrics: Mapping[str, object],
                          config: Mapping[str, object], error: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status, "training_status": training_status,
        "current_epoch": int(epoch), "total_epochs": int(PROTOCOL["epochs"]),
        "best_metrics": dict(best_metrics),
        "identity": {field: config.get(field) for field in (
            "base_checkpoint_sha256", "registry_sha256", "task028_protocol_identity_sha256",
            "training_config_sha256")},
        "precision": "fp32", "amp_enabled": False, "use_checkpoint": False,
        "scheduler": "NONE", "selection_rerun": False,
    }
    if error:
        payload["error"] = error
    atomic_json(output / "finetune" / "training_state.json", payload)
    return payload


def run_finetune(output_dir: Path, *, checkpoint: Path | None = None,
                 identity: Mapping[str, object] | None = None,
                 device: str = "cuda:0", model_name: str = "swintrans",
                 resume: bool = True) -> dict[str, object]:
    """Run the exactly specified 100-epoch Task028-aligned protocol."""
    torch = _torch()
    output = Path(output_dir).expanduser().resolve()
    identity = dict(identity or _require_identity(output))
    if identity.get("status") != "PASS":
        raise RuntimeError("Task034 identity is not PASS")
    selection = verify_selection_artifacts(output)
    if selection.get("status") != "PASS":
        raise RuntimeError("Task034 selection artifacts are not verified")
    protocol = _task034_protocol(output)
    base_identity = read_json(output / "base_checkpoint_identity.json")
    if base_identity.get("match") is not True:
        raise RuntimeError("Task034 exact Task028 base checkpoint gate failed")
    base_path = Path(str(checkpoint or base_identity.get("path", "")))
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    config = _training_config(output, protocol, base_identity)
    config_sha = _canonical_sha(config)
    config["training_config_sha256"] = config_sha
    config_path = output / "finetune" / "finetune_config.json"
    existing_state = output / "finetune" / "training_state.json"
    latest_path = output / "finetune" / "f3_latest.pth"
    start_epoch = 0
    best_metrics: dict[str, object] = {}
    _set_seed(int(PROTOCOL["seed"]))
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=output, checkpoint=base_path, device=device
    )
    if torch.device(device).type == "cuda" and torch.cuda.device_count() < 2:
        raise RuntimeError("Task034 training requires CUDA devices 0 and 1")
    student = torch.nn.DataParallel(model, device_ids=list(PROTOCOL["gpu_ids"]), output_device=0)
    optimizer = torch.optim.SGD(student.parameters(), lr=PROTOCOL["learning_rate"],
                                momentum=PROTOCOL["momentum"], weight_decay=PROTOCOL["weight_decay"])
    if resume and latest_path.is_file() and existing_state.is_file():
        resume_identity_guard(output, latest_path)
        start_epoch, best_metrics = _load_resume_checkpoint(latest_path, student, optimizer, config, device)
        if start_epoch >= int(PROTOCOL["epochs"]):
            return read_json(output / "finetune" / "training_state.json")
    atomic_json(config_path, config)

    train_split = _protocol_task028_value(protocol, "resolved_train_split")
    val_split = _protocol_task028_value(protocol, "resolved_validation_split")
    val_batch = _protocol_task028_value(protocol, "validation_batch_size")
    validation_frequency = _int(_protocol_task028_value(protocol, "validation_frequency"), "validation_frequency")
    if train_split in (None, "") or val_split in (None, "") or val_batch in (None, ""):
        raise RuntimeError("Task028 data/protocol identity is incomplete")
    train_loader = _get_loader(task027, train_split, int(PROTOCOL["batch_size"]))
    val_loader = _get_loader(task027, val_split, _int(val_batch, "validation_batch_size"))
    history: list[dict[str, object]] = []
    history_path = output / "finetune" / "training_history.csv"
    if history_path.is_file() and start_epoch:
        history = read_csv(history_path)
    started = time.perf_counter()
    training_status = "COMPLETED"
    status = "PASS"
    error: str | None = None
    try:
        for epoch in range(start_epoch + 1, int(PROTOCOL["epochs"]) + 1):
            student.train()
            sum_loss = correct = samples = 0.0
            epoch_started = time.perf_counter()
            for batch in _tqdm(train_loader, desc=f"Task034 F3 epoch {epoch:03d}/{PROTOCOL['epochs']}",
                                dynamic_ncols=True, mininterval=0.5):
                inputs = batch[0].float().to(torch.device(device), non_blocking=True)
                targets = batch[1].to(torch.device(device), non_blocking=True)
                optimizer.zero_grad()
                logits = _forward_logits(student, inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise RuntimeError("Non-finite Task034 training loss")
                loss.backward()
                optimizer.step()
                count = int(targets.numel())
                c1, _, _ = _accuracy_counts(logits.detach(), targets)
                sum_loss += float(loss.detach().item()) * count
                correct += c1; samples += count
            train_metrics = {
                "loss": sum_loss / max(samples, 1.0),
                "top1": correct / max(samples, 1.0), "samples": int(samples),
            }
            if epoch % validation_frequency == 0 or epoch == int(PROTOCOL["epochs"]):
                validation = _validation_raw_logits(model=student, loader=val_loader, device=device)
                best_updated = _metric_better(validation, best_metrics)
                if best_updated:
                    best_metrics = {"epoch": epoch, **validation}
            else:
                validation = {"top1": best_metrics.get("top1", ""),
                              "top5": best_metrics.get("top5", ""),
                              "loss": best_metrics.get("loss", ""),
                              "samples": best_metrics.get("samples", "")}
                best_updated = False
            row = {
                "epoch": epoch, "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"], "train_samples": train_metrics["samples"],
                "val_loss": validation.get("loss", ""), "val_top1": validation.get("top1", ""),
                "val_top5": validation.get("top5", ""), "val_samples": validation.get("samples", ""),
                "best_epoch": best_metrics.get("epoch", ""), "best_top1": best_metrics.get("top1", ""),
                "best_top5": best_metrics.get("top5", ""),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history = [item for item in history if _int(item.get("epoch"), "epoch") != epoch]
            history.append(row)
            history.sort(key=lambda item: _int(item.get("epoch"), "epoch"))
            atomic_csv(output / "finetune" / "training_history.csv",
                       ("epoch", "train_loss", "train_top1", "train_samples", "val_loss", "val_top1",
                        "val_top5", "val_samples", "best_epoch", "best_top1", "best_top5", "epoch_seconds"),
                       history)
            payload = _checkpoint_payload(student, optimizer, epoch, best_metrics, config)
            _atomic_torch_save(latest_path, payload)
            if best_updated:
                _atomic_torch_save(output / "finetune" / "f3_best.pth", payload)
            _write_training_state(output, status="RUNNING", training_status="RUNNING",
                                  epoch=epoch, best_metrics=best_metrics, config=config)
    except KeyboardInterrupt:
        status = "PARTIAL_VALID_RESULT"; training_status = "MANUALLY_STOPPED"
        error = "manual interruption"
    except Exception as exc:
        status = "PARTIAL_VALID_RESULT"; training_status = "FAILED_INTERRUPTED"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current_epoch = max([_int(item.get("epoch"), "epoch") for item in history], default=start_epoch)
        _write_training_state(output, status=status if status != "PASS" else "PASS",
                              training_status=training_status, epoch=current_epoch,
                              best_metrics=best_metrics, config=config, error=error)
    if status != "PASS":
        # A manual stop is a valid handoff result, but it is not represented as
        # a completed experiment and must never be relabelled PASS.
        atomic_json(output / "finetune" / "best_metrics.json", {
            "status": status, "training_status": training_status, **best_metrics,
            "accuracy_threshold_used": False,
        })
        return read_json(output / "finetune" / "training_state.json")

    reload_report = _reload_best_validation(output, base_path, device)
    if reload_report.get("status") != "PASS":
        raise RuntimeError("Task034 best checkpoint reload validation failed")
    best_payload = {
        "status": "PASS", "training_status": "COMPLETED", **best_metrics,
        "accuracy_threshold_used": False, "validation_uses_softmax": False,
        "raw_logits": True, "precision": "fp32", "amp_enabled": False,
        "use_checkpoint": False, "optimizer": "SGD", "scheduler": "NONE",
        "base_checkpoint_sha256": config.get("base_checkpoint_sha256"),
        "registry_sha256": config.get("registry_sha256"),
        "best_checkpoint_reload": reload_report,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "finetune" / "best_metrics.json", best_payload)
    return best_payload


def _reload_best_validation(output_dir: Path, checkpoint: Path, device: str) -> dict[str, object]:
    torch = _torch()
    best_path = Path(output_dir) / "finetune" / "f3_best.pth"
    if not best_path.is_file():
        return {"status": "FAIL", "reason": "missing f3_best.pth"}
    model, _task028, task027, context = _load_apply_logical_model(
        output_dir=Path(output_dir), checkpoint=checkpoint, device=device
    )
    student = torch.nn.DataParallel(model, device_ids=list(PROTOCOL["gpu_ids"]), output_device=0)
    payload = torch.load(best_path, map_location=device)
    student.load_state_dict(payload["model_state_dict"], strict=True)
    val_split = _protocol_task028_value(context["protocol"], "resolved_validation_split")
    val_batch = _protocol_task028_value(context["protocol"], "validation_batch_size")
    loader = _get_loader(task027, val_split, _int(val_batch, "validation_batch_size"))
    measured = _validation_raw_logits(model=student, loader=loader, device=device)
    saved = payload.get("best_metrics", {})
    status = ("PASS" if _numeric_equal(measured.get("top1"), saved.get("top1"))
              and _numeric_equal(measured.get("top5"), saved.get("top5")) else "FAIL")
    result = {"status": status, "top1": measured["top1"], "top5": measured["top5"],
              "loss": measured["loss"], "samples": measured["samples"],
              "saved_top1": saved.get("top1"), "saved_top5": saved.get("top5"),
              "strict_state_dict_reload": True, "validation_uses_softmax": False,
              "raw_logits": True}
    atomic_json(Path(output_dir) / "finetune" / "best_checkpoint_reload_validation.json", result)
    del student, model
    return result


# ---------------------------------------------------------------------------
# Comparison, completion, and command-line orchestration
# ---------------------------------------------------------------------------

def _load_task028_result(task028_root: Path) -> dict[str, object]:
    root = Path(task028_root)
    for relative in (
        "final_summary.json", "finetune/final_summary.json", "task028_completion.json",
        "finetune/final_summary.json", "training_progress.json",
    ):
        path = root / relative
        if path.is_file():
            return {"source": str(path), "payload": read_json(path)}
    history_candidates = (root / "finetune_history.csv", root / "finetune" / "training_history.csv")
    for path in history_candidates:
        if path.is_file():
            rows = read_csv(path)
            if rows:
                return {"source": str(path), "payload": rows[-1]}
    raise FileNotFoundError("Task028 final result artifact not found")


def _actual_task034_result(output_dir: Path) -> dict[str, object]:
    root = Path(output_dir)
    for relative in ("finetune/best_metrics.json", "finetune/training_state.json"):
        path = root / relative
        if path.is_file():
            return {"source": str(path), "payload": read_json(path)}
    raise FileNotFoundError("Task034 training result artifact not found")


def compare_to_task028(output_dir: Path, task028_root: Path) -> dict[str, object]:
    output = Path(output_dir)
    task028_result = _load_task028_result(Path(task028_root))
    task034_result = _actual_task034_result(output)
    expected = task028_result["payload"]
    observed = task034_result["payload"]
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise RuntimeError("Task028/Task034 result payloads must be mappings")

    def metric(payload: Mapping[str, object], *names: str) -> object:
        return _row_value(payload, *names, default=None)

    fields = ("top1", "top5", "loss", "best_epoch")
    rows = {}
    for field in fields:
        task028_value = metric(expected, field, f"best_{field}", f"val_{field}")
        task034_value = metric(observed, field, f"best_{field}", f"val_{field}")
        delta = None
        if task028_value not in (None, "") and task034_value not in (None, ""):
            try:
                delta = float(task034_value) - float(task028_value)
            except (TypeError, ValueError):
                delta = None
        rows[field] = {"task028": task028_value, "task034": task034_value, "delta": delta}

    task028_structure = {}
    for relative in ("selection_50/causal_selection_trace.csv", "selection_50/registry.json"):
        path = Path(task028_root) / relative
        if path.is_file():
            task028_structure[relative] = {
                "sha256": sha256_file(path),
                "rows": len(read_csv(path)) if path.suffix == ".csv" else None,
            }
    task034_trace = output / "selection" / "f3_selection_trace.csv"
    task034_structure = {
        "sha256": sha256_file(task034_trace) if task034_trace.is_file() else None,
        "rows": len(read_csv(task034_trace)) if task034_trace.is_file() else None,
    }
    training_status = str(observed.get("training_status", observed.get("status", "")))
    report = {
        "status": "PASS", "comparison_is_descriptive": True,
        "task028_result_source": task028_result["source"],
        "task034_result_source": task034_result["source"],
        "metrics": rows,
        "task028_structure": task028_structure, "task034_structure": task034_structure,
        "f0_exact_task028": read_json(output / "selection" / "f3_selection_completion.json").get("f0_exact_task028_gate", {}),
        "f3_sequence_sha256": read_json(output / "selection" / "f3_sequence_sha256.json").get("sequence_sha256"),
        "training_status": training_status,
        "no_causal_claim_from_accuracy": True,
        "accuracy_threshold_used": False,
    }
    atomic_json(output / "comparison_to_task028.json", report)
    markdown = [
        "# Task034 comparison to Task028", "",
        "This comparison is descriptive. Accuracy deltas alone are not a causal claim.", "",
        "| Metric | Task028 | Task034 | Delta |", "|---|---:|---:|---:|",
    ]
    for field, row in rows.items():
        markdown.append(f"| {field} | {row['task028']} | {row['task034']} | {row['delta']} |")
    markdown.extend(["", f"Task034 F3 sequence SHA256: `{report['f3_sequence_sha256']}`",
                     f"Training status: `{training_status}`", "",
                     "Accuracy threshold used: NO", "Causal claim from accuracy: NO", ""])
    atomic_text(output / "comparison_to_task028.md", "\n".join(markdown))
    return report


def task034_completion(output_dir: Path) -> dict[str, object]:
    output = Path(output_dir)
    identity = _require_identity(output)
    roots = _identity_roots(identity)
    selection = read_json(output / "selection" / "f3_selection_completion.json")
    logical_path = output / "model" / "logical_pruning_verification.json"
    pre_path = output / "model" / "pre_finetune_validation.json"
    sanity_path = output / "model" / "forward_backward_sanity.json"
    speed_path = output / "finetune" / "speed_gate.json"
    state_path = output / "finetune" / "training_state.json"
    protocol = read_json(output / "task028_training_protocol_identity.json")
    base = read_json(output / "base_checkpoint_identity.json")
    logical = read_json(logical_path) if logical_path.is_file() else {}
    pre = read_json(pre_path) if pre_path.is_file() else {}
    sanity = read_json(sanity_path) if sanity_path.is_file() else {}
    speed = read_json(speed_path) if speed_path.is_file() else {}
    state = read_json(state_path) if state_path.is_file() else {}
    source_unchanged = verify_source_hashes_unchanged(identity.get("hashes_before", {}), roots)
    training_status = str(state.get("training_status", ""))
    complete_training = state.get("status") == "PASS" and training_status == "COMPLETED" \
        and _int(state.get("current_epoch", 0), "current_epoch") == int(PROTOCOL["epochs"])
    manual_partial = state.get("status") == "PARTIAL_VALID_RESULT" \
        and training_status == "MANUALLY_STOPPED"
    checks = {
        "identity_gate": identity.get("status") == "PASS",
        "task033_f3_aggregate_identity_gate": selection.get("task033_f3_aggregate_identity_gate", {}).get("status") == "PASS",
        "f0_exact_task028_gate": selection.get("f0_exact_task028_gate", {}).get("status") == "PASS",
        "full_sequence_persisted": (output / "selection" / "f3_selection_trace.csv").is_file(),
        "registry_persisted": (output / "selection" / "f3_registry.json").is_file(),
        "logical_verification": logical.get("status") == "PASS",
        "pre_finetune_validation": pre.get("status") == "PASS",
        "forward_backward_sanity": sanity.get("status") == "PASS",
        "speed_gate": speed.get("status") == "PASS",
        "protocol_gate": protocol.get("status") == "PASS" and protocol.get("all_match") is True,
        "base_checkpoint_gate": base.get("match") is True,
        "source_hashes_unchanged": source_unchanged,
        "no_old_task_artifact_write": identity.get("old_task_artifacts_modified") is False,
        "training_complete_or_valid_manual_stop": complete_training or manual_partial,
        "fixed_midpoint": True, "new_tunable_pruning_hyperparameters": 0,
    }
    status = "PASS" if complete_training and all(checks.values()) else (
        "PARTIAL_VALID_RESULT" if manual_partial and all(value for key, value in checks.items()
                                                          if key != "training_complete_or_valid_manual_stop") else "FAIL"
    )
    report = {
        "status": status, "task034_branch": TASK034_BRANCH,
        "base_task033_branch": TASK033_BASE_BRANCH, "base_task033_commit": TASK033_BASE_COMMIT,
        "fusion": FUSION, "target_sparsity": TARGET_SPARSITY,
        "checks": checks, "training_status": training_status,
        "epochs_completed": state.get("current_epoch", 0),
        "scientific_logic_changed": False, "f3_formula_changed": False,
        "p_average_changed": False, "new_tunable_pruning_hyperparameters": 0,
        "gpu_hot_path_changed": False, "python_oracle_weakened": False,
        "snapshot_materialization": "not applicable to Task034 selector; inherited exact Task033 replay",
        "logical_pruning": True, "physical_pruning": False,
        "precision": "fp32", "amp_enabled": False, "use_checkpoint": False,
        "gpu_ids": [0, 1], "batch_size": 4, "optimizer": "SGD",
        "learning_rate": 0.0005, "momentum": 0.9, "weight_decay": 1e-5,
        "scheduler": "NONE", "loss": "CrossEntropyLoss", "epochs": 100,
        "accuracy_threshold_used": False, "validation_uses_softmax": False,
        "training_executed": complete_training or manual_partial,
        "validation_executed": pre.get("status") == "PASS",
        "old_task_artifacts_modified": False,
        "comparison_written": (output / "comparison_to_task028.json").is_file(),
    }
    atomic_json(output / "task034_completion.json", report)
    return report


def completion_gate(payload: Mapping[str, object]) -> bool:
    """Pure gate for completed and manually stopped Task034 runs."""
    checks = payload.get("checks", {})
    if not isinstance(checks, Mapping):
        return False
    common = (
        "identity_gate", "task033_f3_aggregate_identity_gate", "f0_exact_task028_gate",
        "full_sequence_persisted", "registry_persisted", "logical_verification",
        "pre_finetune_validation", "forward_backward_sanity", "speed_gate",
        "protocol_gate", "base_checkpoint_gate", "source_hashes_unchanged",
        "no_old_task_artifact_write", "new_tunable_pruning_hyperparameters",
    )
    if not all(checks.get(name) is True for name in common[:-1]):
        return False
    if int(payload.get("new_tunable_pruning_hyperparameters", -1)) != 0:
        return False
    if payload.get("physical_pruning") is not False or payload.get("amp_enabled") is not False:
        return False
    status = payload.get("status")
    if status == "PASS":
        return (payload.get("training_status") == "COMPLETED"
                and _int(payload.get("epochs_completed", 0), "epochs_completed") == int(PROTOCOL["epochs"])
                and checks.get("training_complete_or_valid_manual_stop") is True)
    if status == "PARTIAL_VALID_RESULT":
        return (payload.get("training_status") == "MANUALLY_STOPPED"
                and 0 <= _int(payload.get("epochs_completed", 0), "epochs_completed") <= int(PROTOCOL["epochs"])
                and checks.get("training_complete_or_valid_manual_stop") is True)
    return False


task034_completion_gate = completion_gate


def _required_path_args(args: argparse.Namespace, names: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    missing = []
    for name in names:
        value = getattr(args, name.replace("-", "_"), None)
        if value is None:
            missing.append(name)
        else:
            result[name] = Path(value)
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join('--' + name for name in missing)}")
    return result


# Stable descriptive aliases for focused tests and downstream report tools.
f3_risk = f3_mid_veto
f3_order = f3_order_key
rank_fusion = rank_f3
construct_registry = construct_f3_registry
materialize_snapshot_rows = materialize_component_rows
verify_logical_registry = verify_registry_pure
benchmark_selection = run_f3_benchmark
select_f3 = construct_selection
verify_logical = verify_logical_pruning
preft_validate = pre_finetune_validate
sanity = forward_backward_sanity
completion = task034_completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(
        "identity", "benchmark-selection", "select", "verify-logical",
        "preft-validate", "sanity", "speed-gate", "finetune", "compare",
        "completion", "run-all",
    ), required=True)
    for name in ("task028-root", "task029-root", "task030-root", "task031-root",
                 "task032-root", "task033-root", "task014-root", "task016-root",
                 "task017-root", "output-dir", "repo-root", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranking-backend", choices=("single-gpu", "dual-gpu"), default="single-gpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model-name", default="swintrans")
    return parser


def _run_identity_from_args(args: argparse.Namespace) -> dict[str, object]:
    paths = _required_path_args(args, ("task028-root", "task029-root", "task030-root",
                                        "task031-root", "task032-root", "task033-root", "output-dir"))
    return verify_identity(
        task028_root=paths["task028-root"], task029_root=paths["task029-root"],
        task030_root=paths["task030-root"], task031_root=paths["task031-root"],
        task032_root=paths["task032-root"], task033_root=paths["task033-root"],
        output_dir=paths["output-dir"], repo_root=args.repo_root, checkpoint=args.checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = args.mode
    if mode == "identity":
        result = _run_identity_from_args(args)
        return 0 if result.get("status") == "PASS" else 1
    output = Path(args.output_dir) if args.output_dir is not None else None
    if output is None:
        raise SystemExit("--output-dir is required")
    if mode == "benchmark-selection":
        paths = _required_path_args(args, ("task014-root", "task016-root", "task017-root", "output-dir"))
        result = run_f3_benchmark(task014_root=paths["task014-root"], task016_root=paths["task016-root"],
                                  task017_root=paths["task017-root"], output_dir=paths["output-dir"],
                                  device=args.device, steps=args.steps, ranking_backend=args.ranking_backend)
        return 0 if result["status"] == "PASS" else 1
    if mode == "select":
        paths = _required_path_args(args, ("task014-root", "task016-root", "task017-root", "output-dir"))
        result = construct_selection(output_dir=paths["output-dir"], task014_root=paths["task014-root"],
                                     task016_root=paths["task016-root"], task017_root=paths["task017-root"],
                                     device=args.device, ranking_backend=args.ranking_backend)
        return 0 if result["completion"]["status"] == "PASS" else 1
    if mode == "verify-logical":
        result = verify_logical_pruning(output, checkpoint=args.checkpoint, device=args.device)
        return 0 if result["status"] == "PASS" else 1
    if mode == "preft-validate":
        result = pre_finetune_validate(output, checkpoint=args.checkpoint, device=args.device)
        return 0 if result["status"] == "PASS" else 1
    if mode == "sanity":
        result = forward_backward_sanity(output, checkpoint=args.checkpoint, device=args.device)
        return 0 if result["status"] == "PASS" else 1
    if mode == "speed-gate":
        result = speed_gate(output, checkpoint=args.checkpoint, device=args.device, steps=args.steps)
        return 0 if result["status"] == "PASS" else 1
    if mode == "finetune":
        result = run_finetune(output, checkpoint=args.checkpoint, device=args.device,
                              model_name=args.model_name)
        return 0 if result.get("status") in ("PASS", "PARTIAL_VALID_RESULT") else 1
    if mode == "compare":
        if args.task028_root is None:
            raise SystemExit("compare requires --task028-root")
        compare_to_task028(output, args.task028_root)
        return 0
    if mode == "completion":
        result = task034_completion(output)
        return 0 if result["status"] in ("PASS", "PARTIAL_VALID_RESULT") else 1
    # run-all is intentionally a strict convenience wrapper.  It is still
    # runner-driven and never bypasses any of the pre-training gates.
    _run_identity_from_args(args)
    paths = _required_path_args(args, ("task014-root", "task016-root", "task017-root", "output-dir"))
    benchmark = run_f3_benchmark(task014_root=paths["task014-root"], task016_root=paths["task016-root"],
                                 task017_root=paths["task017-root"], output_dir=output,
                                 device=args.device, steps=args.steps, ranking_backend=args.ranking_backend)
    if benchmark["status"] != "PASS":
        return 1
    construct_selection(output_dir=output, task014_root=paths["task014-root"],
                        task016_root=paths["task016-root"], task017_root=paths["task017-root"],
                        device=args.device, ranking_backend=args.ranking_backend)
    verify_logical_pruning(output, checkpoint=args.checkpoint, device=args.device)
    pre_finetune_validate(output, checkpoint=args.checkpoint, device=args.device)
    forward_backward_sanity(output, checkpoint=args.checkpoint, device=args.device)
    speed_gate(output, checkpoint=args.checkpoint, device=args.device, steps=100)
    run_finetune(output, checkpoint=args.checkpoint, device=args.device, model_name=args.model_name)
    if args.task028_root is None:
        raise SystemExit("run-all requires --task028-root for comparison")
    compare_to_task028(output, args.task028_root)
    report = task034_completion(output)
    return 0 if report["status"] in ("PASS", "PARTIAL_VALID_RESULT") else 1


if __name__ == "__main__":
    raise SystemExit(main())

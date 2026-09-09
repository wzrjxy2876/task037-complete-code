"""GPU-first offline diagnosis of Task016's 20%-to-30% collapse.

Task017 is read-only with respect to pruning.  It reconstructs saved removal
sets, exactly replays the saved 30% traces with the unchanged Task014/Task016
selector state, and writes diagnostic evidence.  It never creates a registry,
applies pruning, validates a model, or changes a pruning score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


TARGETS = (0.10, 0.20, 0.30)
SNAPSHOTS = (0.00, 0.10, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
TAGS = {0.10: "s10", 0.20: "s20", 0.30: "s30"}
MODES = ("domain_average", "domain_total")
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
EXPECTED_UNITS = 36_378
EXPECTED_DOMAINS = 423
TRACE_TOLERANCE = 1e-7
CHECKPOINT_INTERVAL = 500
SUBSTITUTE_THRESHOLDS = (0.1, 0.3, 0.5)


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(value) for value in indices], separators=(",", ":")).encode()
    ).hexdigest()


def set_sha256(indices: Iterable[int]) -> str:
    return sequence_sha256(sorted({int(value) for value in indices}))


def removed_indices(rows: Sequence[Mapping[str, object]]) -> list[int]:
    result = [int(row["global_index"]) for row in rows]
    if len(result) != len(set(result)):
        raise ValueError("A pruning trace removes the same global index twice")
    return result


def reconstruct_removed_sets(
    traces: Mapping[tuple[str, float], Sequence[Mapping[str, object]]]
) -> dict[tuple[str, float], set[int]]:
    expected = {(mode, target) for mode in MODES for target in TARGETS}
    if set(traces) != expected:
        raise ValueError("Task017 requires average/total traces at 10%, 20%, 30%")
    return {key: set(removed_indices(rows)) for key, rows in traces.items()}


def verify_nested_removal_sets(
    sets: Mapping[tuple[str, float], set[int]], mode: str
) -> bool:
    if mode not in MODES:
        raise ValueError(mode)
    return (
        sets[(mode, 0.10)] <= sets[(mode, 0.20)]
        and sets[(mode, 0.20)] <= sets[(mode, 0.30)]
    )


def incremental_layer_statistics(
    *,
    units: Sequence[object],
    removed_at_20: set[int],
    removed_at_30: set[int],
    costs: Sequence[int],
) -> list[dict[str, object]]:
    if len(units) != len(costs):
        raise ValueError("units and costs must have identical lengths")
    counts = Counter(str(unit.layer) for unit in units)
    removed20 = Counter(str(units[index].layer) for index in removed_at_20)
    removed30 = Counter(str(units[index].layer) for index in removed_at_30)
    extra_cost = Counter()
    for index in removed_at_30 - removed_at_20:
        extra_cost[str(units[index].layer)] += int(costs[index])
    rows = []
    for layer in sorted(counts):
        extra = removed30[layer] - removed20[layer]
        rows.append(
            {
                "layer": layer,
                "units_before": counts[layer],
                "removed_at_20": removed20[layer],
                "additional_removed_20_to_30": extra,
                "remaining_at_30": counts[layer] - removed30[layer],
                "incremental_removal_ratio": extra / counts[layer],
                "cumulative_removal_ratio": removed30[layer] / counts[layer],
                "parameter_cost_extra": extra_cost[layer],
            }
        )
    return rows


def domain_depletion_statistics(
    *,
    domain_members: Mapping[int, Sequence[int]],
    valid_mask: Sequence[bool],
    removed: set[int],
) -> list[dict[str, object]]:
    valid = np.asarray(valid_mask, dtype=np.bool_)  # Bool[N], global unit axis
    rows = []
    for domain_id, members_value in sorted(domain_members.items()):
        members = np.asarray(members_value, dtype=np.int64)  # [G], global indices
        retained = np.asarray(
            [index not in removed for index in members], dtype=np.bool_
        )  # Bool[G]
        active = valid[members]  # Bool[G]
        rows.append(
            {
                "domain_id": int(domain_id),
                "initial_units": int(members.size),
                "valid_demand_count": int(active.sum()),
                "retained_units": int(retained.sum()),
                "retained_active_units": int((retained & active).sum()),
                "removed_units": int((~retained).sum()),
                "retained_ratio": float(retained.mean()) if members.size else 1.0,
                "active_retained_ratio": (
                    float((retained & active).sum() / active.sum())
                    if active.any()
                    else 1.0
                ),
            }
        )
    return rows


def herfindahl_hirschman(values: Sequence[int | float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError("HHI requires a non-negative one-dimensional array")
    total = float(array.sum())
    return float(np.square(array / total).sum()) if total else 0.0


def gini_coefficient(values: Sequence[int | float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError("Gini requires a non-negative one-dimensional array")
    if array.size == 0 or float(array.sum()) == 0.0:
        return 0.0
    ordered = np.sort(array)
    positions = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(positions * ordered) / (ordered.size * ordered.sum()))
        - (ordered.size + 1.0) / ordered.size
    )


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def coverage_change(coverage_20: float, coverage_30: float) -> dict[str, float]:
    if not math.isfinite(coverage_20) or not math.isfinite(coverage_30):
        raise ValueError("Coverage must be finite")
    return {
        "coverage_20": float(coverage_20),
        "coverage_30": float(coverage_30),
        "coverage_drop_20_to_30": float(coverage_20 - coverage_30),
    }


def substitute_count_extraction(
    similarities: Sequence[float], thresholds: Sequence[float] = SUBSTITUTE_THRESHOLDS
) -> dict[str, float | int]:
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Substitute similarities must be a finite 1D array")
    ordered = np.sort(values)[::-1]
    output: dict[str, float | int] = {
        "best_remaining_similarity": float(ordered[0]) if ordered.size else 0.0,
        "second_best_remaining_similarity": (
            float(ordered[1]) if ordered.size > 1 else 0.0
        ),
    }
    for threshold in thresholds:
        output[f"substitutes_ge_{threshold:g}"] = int((values >= threshold).sum())
    return output


def rank_candidates(
    global_indices: Sequence[int], scores: Sequence[float]
) -> dict[int, tuple[int, float]]:
    indices = np.asarray(global_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if indices.shape != values.shape or indices.ndim != 1:
        raise ValueError("Candidate indices and scores must be aligned 1D arrays")
    if len(indices) != len(set(indices.tolist())) or not np.isfinite(values).all():
        raise ValueError("Candidate indices must be unique and scores finite")
    order = np.lexsort((indices, values))
    denominator = max(len(order) - 1, 1)
    return {
        int(indices[position]): (rank + 1, rank / denominator)
        for rank, position in enumerate(order)
    }


def track_dynamic_ranks(
    snapshots: Mapping[float, tuple[Sequence[int], Sequence[float]]],
    tracked_indices: Iterable[int],
) -> list[dict[str, object]]:
    tracked = sorted({int(value) for value in tracked_indices})
    rows = []
    for progress, (indices, scores) in sorted(snapshots.items()):
        ranks = rank_candidates(indices, scores)
        for index in tracked:
            rank = ranks.get(index)
            rows.append(
                {
                    "progress": float(progress),
                    "global_index": index,
                    "available": rank is not None,
                    "rank": rank[0] if rank else "",
                    "percentile": rank[1] if rank else "",
                }
            )
    return rows


IDENTITY_KEYS = (
    "checkpoint_sha256",
    "descriptor_variant",
    "descriptor_sha256",
    "seed",
    "sigma",
    "contribution_mapping_sha256",
    "calibration_samples_sha256",
    "min_keep_ratio",
    "parameters_before",
    "descriptor_units",
    "mapped_units",
    "bms_domains",
)


def validate_artifact_identity(
    records: Sequence[Mapping[str, object]],
    *,
    expected_units: int = EXPECTED_UNITS,
    expected_domains: int = EXPECTED_DOMAINS,
) -> dict[str, object]:
    if not records:
        raise ValueError("No run identities were supplied")
    reference = {key: records[0].get(key) for key in IDENTITY_KEYS}
    if (
        int(reference["descriptor_units"]) != expected_units
        or int(reference["mapped_units"]) != expected_units
        or int(reference["bms_domains"]) != expected_domains
    ):
        raise ValueError("Task017 identity has unexpected unit/domain counts")
    mismatches = []
    for position, record in enumerate(records[1:], start=1):
        changed = [
            key for key in IDENTITY_KEYS if record.get(key) != reference.get(key)
        ]
        if changed:
            mismatches.append({"record": position, "changed_keys": changed})
    if mismatches:
        raise ValueError(f"Task017 artifact identity mismatch: {mismatches}")
    return {"status": "passed", "reference": reference, "run_count": len(records)}


def stage_from_layer(layer: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(layer))
    if match is None:
        raise ValueError(f"Cannot parse stage from {layer!r}")
    return int(match.group(1))


def block_from_layer(layer: str) -> int:
    match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", str(layer))
    if match is None:
        raise ValueError(f"Cannot parse block from {layer!r}")
    return int(match.group(1))


def quantile_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "q10": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q90": math.nan,
            "mean": math.nan,
        }
    q10, q25, median, q75, q90 = np.quantile(
        array, (0.10, 0.25, 0.50, 0.75, 0.90)
    )
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "mean": float(array.mean()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        return math.nan
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _run_dir(task014_root: Path, task016_root: Path, mode: str, target: float) -> Path:
    if mode == "domain_average":
        return Path(task014_root) / "functional" / TAGS[target]
    if mode == "domain_total":
        return Path(task016_root) / "domain_total" / TAGS[target]
    raise ValueError(mode)


def verify_artifacts(
    task014_root: Path, task015_root: Path, task016_root: Path, output_dir: Path
) -> dict[str, object]:
    from task015_attention_ffn_diagnosis import _descriptor_path, _mapping_paths

    task014_root, task015_root = Path(task014_root), Path(task015_root)
    task016_root, output_dir = Path(task016_root), Path(output_dir)
    unit_path, _, mapping_path = _mapping_paths(task014_root)
    mapping = read_json(mapping_path)
    if (
        int(mapping.get("descriptor_units", -1)) != EXPECTED_UNITS
        or int(mapping.get("mapped_units", -1)) != EXPECTED_UNITS
        or int(mapping.get("missing", -1)) != 0
        or int(mapping.get("duplicate", -1)) != 0
    ):
        raise ValueError("Task014 mapping evidence is not a complete 36,378-unit map")
    records = []
    runs = []
    domain_hash = None
    for mode in MODES:
        for target in TARGETS:
            run = _run_dir(task014_root, task016_root, mode, target)
            metadata = read_json(run / "run_metadata.json")
            metrics = read_json(run / "final_metrics.json")
            expected_status = (
                "task014_prune_only_complete"
                if mode == "domain_average"
                else "task016_prune_only_complete"
            )
            if metrics.get("status") != expected_status:
                raise ValueError(f"Incomplete run: {run}")
            if mode == "domain_total" and metadata.get("functional_score") != mode:
                raise ValueError(f"Wrong functional score identity: {run}")
            domains = read_csv(run / "functional_domain_summary.csv")
            if len(domains) != EXPECTED_DOMAINS:
                raise ValueError(f"BMS domains={len(domains)} in {run}")
            digest = hashlib.sha256()
            for row in domains:
                digest.update(
                    "\t".join(
                        row[key]
                        for key in (
                            "domain_id",
                            "initial_size",
                            "attention_count",
                            "mlp_count",
                            "active_functional_count",
                            "null_functional_count",
                        )
                    ).encode()
                )
                digest.update(b"\n")
            current_domain_hash = digest.hexdigest()
            if domain_hash is None:
                domain_hash = current_domain_hash
            elif current_domain_hash != domain_hash:
                raise ValueError(f"BMS domain membership differs in {run}")
            records.append(
                {
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "descriptor_variant": metadata.get("descriptor_variant"),
                    "descriptor_sha256": sha256_file(
                        _descriptor_path(task014_root, run)
                    ),
                    "seed": int(metadata.get("seed", -1)),
                    "sigma": float(metadata.get("sigma", math.nan)),
                    "contribution_mapping_sha256": metadata.get(
                        "contribution_mapping_sha256"
                    ),
                    "calibration_samples_sha256": metadata.get(
                        "calibration_samples_sha256"
                    ),
                    "min_keep_ratio": float(
                        metadata.get("min_keep_ratio", math.nan)
                    ),
                    "parameters_before": int(metrics.get("parameters_before", -1)),
                    "descriptor_units": int(mapping["descriptor_units"]),
                    "mapped_units": int(mapping["mapped_units"]),
                    "bms_domains": len(domains),
                }
            )
            runs.append(
                {
                    "score_mode": mode,
                    "target": target,
                    "run_dir": str(run.resolve()),
                    "trace_sha256": sha256_file(
                        run / "functional_selection_trace.csv"
                    ),
                    "metadata_sha256": sha256_file(run / "run_metadata.json"),
                    "metrics_sha256": sha256_file(run / "final_metrics.json"),
                }
            )
    identity = validate_artifact_identity(records)
    if records[0]["descriptor_variant"] != "dynamic3d":
        raise ValueError("Task017 requires the unchanged Dynamic3D descriptor")
    if not math.isclose(float(records[0]["sigma"]), 0.1, abs_tol=1e-12):
        raise ValueError("Task017 requires unchanged sigma=0.1")
    energy = task015_root / "functional_energy.npy"
    energy_evidence = task015_root / "functional_energy.json"
    for path in (unit_path, energy, energy_evidence):
        if not path.is_file():
            raise FileNotFoundError(path)
    energy_identity = read_json(energy_evidence)
    if (
        energy_identity.get("mapping_sha256") != mapping.get("mapping_sha256")
        or energy_identity.get("shape") != [EXPECTED_UNITS]
        or energy_identity.get("definition")
        != "L2 of signed pooled concatenated [9*16*7*7] vector"
    ):
        raise ValueError("Task015 functional energy identity does not match Task014")
    energy_array = np.load(energy, mmap_mode="r", allow_pickle=False)
    if energy_array.shape != (EXPECTED_UNITS,) or not np.isfinite(energy_array).all():
        raise ValueError("Task015 functional energy cache is invalid")
    del energy_array
    payload = {
        **identity,
        "artifact_identity_pass": True,
        "bms_domain_sha256": domain_hash,
        "mapping_evidence_sha256": sha256_file(mapping_path),
        "unit_mapping_sha256": sha256_file(unit_path),
        "functional_energy_sha256": sha256_file(energy),
        "functional_energy_identity": energy_identity,
        "runs": runs,
    }
    atomic_json(output_dir / "artifact_identity.json", payload)
    return payload


class HighSparsityReplay:
    """Exact 30% trace replay using Task016's domain-cached CUDA selector.

    Every domain owns ``A_k[G,G]`` on one CUDA device.  Candidate snapshots
    concatenate feasible vectors ``[F]`` over all domains; no global ``N x N``
    matrix is constructed.  Parameter costs are read only after the saved
    functional candidate has been verified.
    """

    def __init__(
        self,
        *,
        task014_root: Path,
        task016_root: Path,
        output_dir: Path,
        mode: str,
        device: str,
    ) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Task017 replay requires PyTorch") from error
        if torch.device(device).type != "cuda":
            raise ValueError("Task017 replay requires CUDA")
        from task015_attention_ffn_diagnosis import (
            _descriptor_path,
            _layer_capacities,
            _load_descriptors,
            _load_units,
            _mapping_paths,
            infer_parameter_costs,
            rebuild_bms_domains,
        )

        self.torch = torch
        self.mode = mode
        self.device = torch.device(device)
        self.task014_root = Path(task014_root)
        self.task016_root = Path(task016_root)
        self.output_dir = Path(output_dir)
        self.run_dir = _run_dir(
            self.task014_root, self.task016_root, mode, 0.30
        )
        unit_path, layer_path, mapping_path = _mapping_paths(self.task014_root)
        self.units = _load_units(unit_path)
        self.unit_by_global = {unit.global_index: unit for unit in self.units}
        self.mapping_path = mapping_path
        descriptor_path = _descriptor_path(self.task014_root, self.run_dir)
        self.descriptor_path = descriptor_path
        self.descriptors = _load_descriptors(descriptor_path, self.units)
        self.groups = rebuild_bms_domains(self.descriptors, str(self.device))
        self.costs, self.layer_costs = infer_parameter_costs(
            self.units, read_csv(layer_path)
        )
        self.capacities = _layer_capacities(self.units)
        self.expected_trace = read_csv(
            self.run_dir / "functional_selection_trace.csv"
        )
        self.metrics = read_json(self.run_dir / "final_metrics.json")
        self.total_parameters = int(self.metrics["parameters_before"])
        self.vector_path = (
            self.task014_root / "field_cache" / "aligned_function_fields.npy"
        )
        self.mask_path = (
            self.task014_root
            / "field_cache"
            / "aligned_function_valid_mask.npy"
        )
        for path in (self.vector_path, self.mask_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.global_to_domain = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        self.global_to_local = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        for domain_id, members in enumerate(self.groups):
            values = np.asarray(members, dtype=np.int64)
            self.global_to_domain[values] = domain_id
            self.global_to_local[values] = np.arange(len(values), dtype=np.int64)
        if np.any(self.global_to_domain < 0):
            raise ValueError("Task017 BMS lookup is incomplete")
        self.states = []
        self.cache = None
        self.snapshots: dict[str, np.ndarray] = {}
        self.domain_rows: list[dict[str, object]] = []
        self.risk_rows: list[dict[str, object]] = []
        self.removed_cost = 0
        self.step = 0
        self.next_snapshot = 0
        self.interrupt_requested = False

    @property
    def worker_dir(self) -> Path:
        return self.output_dir / "replay" / self.mode

    @property
    def checkpoint_json(self) -> Path:
        return self.output_dir / "replay_checkpoints" / f"{self.mode}.json"

    @property
    def checkpoint_npz(self) -> Path:
        return self.output_dir / "replay_checkpoints" / f"{self.mode}.npz"

    def _checkpoint_identity(self) -> dict[str, object]:
        return {
            "format": "task017_high_sparsity_replay_v1",
            "mode": self.mode,
            "trace_sha256": sha256_file(
                self.run_dir / "functional_selection_trace.csv"
            ),
            "descriptor_sha256": sha256_file(self.descriptor_path),
            "mapping_sha256": sha256_file(self.mapping_path),
            "vector_size": self.vector_path.stat().st_size,
            "vector_mtime_ns": self.vector_path.stat().st_mtime_ns,
            "mask_size": self.mask_path.stat().st_size,
            "mask_mtime_ns": self.mask_path.stat().st_mtime_ns,
            "score_formula": self.mode,
        }

    def _initialize_states(self) -> None:
        from functional_competition_pruning import (
            DomainState,
            build_functional_similarity,
        )

        vectors = np.load(self.vector_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)
        if vectors.shape[0] != EXPECTED_UNITS or valid.shape != (EXPECTED_UNITS,):
            raise ValueError("Task017 normalized field cache has an invalid shape")
        states = []
        for domain_id, members in enumerate(self.groups):
            cpu_vectors = self.torch.from_numpy(
                np.asarray(vectors[members], dtype=np.float32).copy()
            )  # [G,9*16*7*7], axes are unit and signed field coordinate
            cpu_valid = self.torch.from_numpy(
                np.asarray(valid[members], dtype=np.bool_).copy()
            )  # Bool[G], fixed non-null demand mask
            domain_vectors = cpu_vectors.to(
                self.device, dtype=self.torch.float32, non_blocking=True
            )
            domain_valid = cpu_valid.to(
                self.device, dtype=self.torch.bool, non_blocking=True
            )
            similarity = build_functional_similarity(
                domain_vectors, domain_valid
            )  # [G,G], signed-cosine affinity clamped to [0,1]
            states.append(DomainState(domain_id, members, similarity, domain_valid))
            del cpu_vectors, cpu_valid, domain_vectors, domain_valid
        self.states = states
        del vectors, valid

    def _initialize_cache(self) -> None:
        from functional_competition_pruning import _FunctionalDomainCandidateCache

        self.cache = _FunctionalDomainCandidateCache(
            self.states, self.unit_by_global, self.capacities
        )

    def _restore_checkpoint(self) -> bool:
        if not self.checkpoint_json.exists() and not self.checkpoint_npz.exists():
            return False
        if not self.checkpoint_json.is_file() or not self.checkpoint_npz.is_file():
            raise RuntimeError("Task017 checkpoint pair is incomplete")
        metadata = read_json(self.checkpoint_json)
        if metadata.get("identity") != self._checkpoint_identity():
            raise RuntimeError("Stale Task017 checkpoint rejected")
        with np.load(self.checkpoint_npz, allow_pickle=False) as payload:
            retained = payload["retained"].astype(np.bool_)
            self.snapshots = {
                key[len("snapshot__") :]: payload[key].copy()
                for key in payload.files
                if key.startswith("snapshot__")
            }
        if retained.shape != (EXPECTED_UNITS,):
            raise RuntimeError("Task017 checkpoint retained mask has wrong shape")
        self.step = int(metadata["step"])
        self.removed_cost = int(metadata["removed_cost"])
        self.next_snapshot = int(metadata["next_snapshot"])
        self.risk_rows = list(metadata.get("risk_rows", []))
        self.domain_rows = list(metadata.get("domain_rows", []))
        expected_prefix = removed_indices(self.expected_trace[: self.step])
        expected_retained = np.ones(EXPECTED_UNITS, dtype=np.bool_)
        expected_retained[expected_prefix] = False
        if not np.array_equal(retained, expected_retained):
            raise RuntimeError("Task017 checkpoint disagrees with trace prefix")
        from functional_competition_pruning import marginal_coverage_losses

        for domain_id, state in enumerate(self.states):
            globals_ = np.asarray(state.global_indices, dtype=np.int64)
            local_retained = retained[globals_]
            state.retained.copy_(
                self.torch.as_tensor(
                    local_retained, dtype=self.torch.bool, device=self.device
                )
            )
            state.version = int((~local_retained).sum())
            if local_retained.any():
                state.losses, coverage = marginal_coverage_losses(
                    state.similarity, state.retained, state.valid_function_mask
                )
                state.current_coverage = float(coverage.item())
            else:
                state.losses = self.torch.full_like(state.losses, self.torch.inf)
                state.current_coverage = (
                    1.0
                    if not bool(state.valid_function_mask.any().item())
                    else 0.0
                )
        self._initialize_cache()
        pruned = Counter(self.units[index].layer for index in expected_prefix)
        for layer, count in pruned.items():
            layer_id = self.cache.layer_to_id[layer]
            self.cache.pruned[layer_id] = count
        full = self.cache.pruned >= self.cache.capacities
        for feasible in self.cache.feasible_by_device.values():
            full_mask = self.torch.as_tensor(
                full, dtype=self.torch.bool, device=feasible.device
            )
            feasible[full_mask] = False
        self.cache.refresh_all()
        expected_cost = int(self.costs[expected_prefix].sum()) if expected_prefix else 0
        if expected_cost != self.removed_cost or len(self.risk_rows) != self.step:
            raise RuntimeError("Task017 checkpoint counters are inconsistent")
        return True

    def _save_checkpoint(self) -> None:
        retained = np.ones(EXPECTED_UNITS, dtype=np.bool_)
        for state in self.states:
            globals_ = np.asarray(state.global_indices, dtype=np.int64)
            retained[globals_] = state.retained.detach().cpu().numpy()
        arrays = {"retained": retained}
        arrays.update({f"snapshot__{key}": value for key, value in self.snapshots.items()})
        atomic_npz(self.checkpoint_npz, arrays)
        atomic_json(
            self.checkpoint_json,
            {
                "status": "incomplete",
                "identity": self._checkpoint_identity(),
                "step": self.step,
                "removed_cost": self.removed_cost,
                "next_snapshot": self.next_snapshot,
                "risk_rows": self.risk_rows,
                "domain_rows": self.domain_rows,
            },
        )

    def _substitutes(self, state, local_index: int) -> dict[str, float | int]:
        remaining = state.retained & state.valid_function_mask
        remaining = remaining.clone()
        remaining[local_index] = False
        local = remaining.nonzero(as_tuple=True)[0]  # [R]
        if local.numel() == 0:
            return substitute_count_extraction([])
        values = state.similarity[local_index].index_select(0, local)  # [R]
        packet = values.detach().cpu().numpy().astype(np.float64, copy=False)
        return substitute_count_extraction(packet)

    def _domain_neighborhood(self, state) -> dict[str, float]:
        demand = state.valid_function_mask.nonzero(as_tuple=True)[0]  # [D]
        reps = (state.retained & state.valid_function_mask).nonzero(
            as_tuple=True
        )[0]  # [R]
        if demand.numel() == 0 or reps.numel() == 0:
            return {
                "mean_positive_neighbors": 0.0,
                "mean_best_remaining_similarity": 0.0,
                "median_best_remaining_similarity": 0.0,
            }
        matrix = state.similarity.index_select(0, demand).index_select(1, reps).clone()
        same = demand[:, None] == reps[None, :]
        matrix.masked_fill_(same, 0.0)
        positive = (matrix > 0.0).sum(dim=1).to(self.torch.float32)
        best = matrix.max(dim=1).values
        packet = self.torch.cat((positive, best)).detach().cpu().numpy()
        split = demand.numel()
        best_cpu = packet[split:]
        return {
            "mean_positive_neighbors": float(packet[:split].mean()),
            "mean_best_remaining_similarity": float(best_cpu.mean()),
            "median_best_remaining_similarity": float(np.median(best_cpu)),
        }

    def _collect_snapshot(self, label: float) -> None:
        if self.cache is None:
            raise RuntimeError("Task017 candidate cache is not initialized")
        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        actual = self.removed_cost / self.total_parameters
        for domain_id, state in enumerate(self.states):
            eligible = self.cache.eligible_mask(domain_id)
            local = eligible.nonzero(as_tuple=True)[0]  # [F_k]
            if local.numel():
                average = state.losses.index_select(0, local)  # [F_k]
                total = average * float(self.cache.active_size[domain_id])
                similarities = state.similarity.index_select(0, local).clone()
                row = self.torch.arange(local.numel(), device=self.device)
                similarities[row, local] = 0.0
                remaining_active = state.retained & state.valid_function_mask
                similarities.masked_fill_(~remaining_active.unsqueeze(0), 0.0)
                best = similarities.max(dim=1).values
                packet = self.torch.stack((average, total, best), dim=0).detach().cpu().numpy()
                local_cpu = local.detach().cpu().numpy().astype(np.int64, copy=False)
                globals_ = np.asarray(state.global_indices, dtype=np.int64)[local_cpu]
                arrays["global_index"].append(globals_)
                arrays["domain_id"].append(
                    np.full(local_cpu.size, domain_id, dtype=np.int64)
                )
                arrays["unit_type"].append(
                    np.asarray(
                        [
                            0
                            if self.unit_by_global[int(index)].unit_type
                            == TYPE_ATTENTION
                            else 1
                            for index in globals_
                        ],
                        dtype=np.int8,
                    )
                )
                arrays["delta_average"].append(packet[0].astype(np.float64))
                arrays["delta_total"].append(packet[1].astype(np.float64))
                arrays["best_similarity"].append(packet[2].astype(np.float64))
                arrays["active_demand_count"].append(
                    np.full(
                        local_cpu.size,
                        int(self.cache.active_size[domain_id]),
                        dtype=np.int64,
                    )
                )
                arrays["retained_ratio"].append(
                    np.full(
                        local_cpu.size,
                        state.retained_count / state.initial_size,
                        dtype=np.float64,
                    )
                )
                arrays["coverage"].append(
                    np.full(local_cpu.size, state.current_coverage, dtype=np.float64)
                )
            retained = state.retained_count
            active_retained = int(
                (state.retained & state.valid_function_mask).sum().item()
            )
            valid_count = int(state.valid_function_mask.sum().item())
            self.domain_rows.append(
                {
                    "score_mode": self.mode,
                    "snapshot_target": label,
                    "snapshot_estimated_sparsity": actual,
                    "domain_id": domain_id,
                    "initial_units": state.initial_size,
                    "valid_demand_count": valid_count,
                    "retained_units": retained,
                    "retained_active_units": active_retained,
                    "removed_units": state.initial_size - retained,
                    "retained_ratio": retained / state.initial_size,
                    "active_retained_ratio": (
                        active_retained / valid_count if valid_count else 1.0
                    ),
                    "coverage": state.current_coverage,
                    "coverage_drop_from_initial": (
                        state.initial_coverage - state.current_coverage
                    ),
                    **self._domain_neighborhood(state),
                }
            )
        prefix = f"p{int(round(label * 100)):02d}_"
        for key in (
            "global_index",
            "domain_id",
            "unit_type",
            "delta_average",
            "delta_total",
            "best_similarity",
            "active_demand_count",
            "retained_ratio",
            "coverage",
        ):
            self.snapshots[prefix + key] = np.concatenate(arrays[key])
        self.snapshots[prefix + "estimated_sparsity"] = np.asarray(
            [actual], dtype=np.float64
        )

    def _record_due_snapshots(self) -> None:
        actual = self.removed_cost / self.total_parameters
        while self.next_snapshot < len(SNAPSHOTS) and (
            actual + 1e-15 >= SNAPSHOTS[self.next_snapshot]
        ):
            self._collect_snapshot(SNAPSHOTS[self.next_snapshot])
            self.next_snapshot += 1

    def run(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self._initialize_states()
        resumed = self._restore_checkpoint()
        if not resumed:
            self._initialize_cache()
            self.cache.refresh_all()
        if self.cache is None:
            raise RuntimeError("Task017 candidate cache initialization failed")
        self._record_due_snapshots()
        try:
            from tqdm import tqdm

            progress = tqdm(
                total=len(self.expected_trace),
                initial=self.step,
                desc=f"Task017 {self.mode}",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        except ModuleNotFoundError:
            progress = None
        previous_handler = signal.getsignal(signal.SIGINT)

        def request_interrupt(_signum, _frame) -> None:
            self.interrupt_requested = True

        signal.signal(signal.SIGINT, request_interrupt)
        started = time.perf_counter()
        try:
            while self.step < len(self.expected_trace):
                expected = self.expected_trace[self.step]
                selected = self.cache.best(0, self.mode)
                if selected is None:
                    raise RuntimeError("Task017 replay exhausted feasible candidates")
                score, global_index, domain_id, local_index = selected
                expected_global = int(expected["global_index"])
                expected_domain = int(expected["domain_id"])
                expected_score = float(
                    expected.get(
                        "functional_score",
                        expected.get("marginal_functional_loss", "nan"),
                    )
                )
                expected_cost = int(expected["parameter_cost"])
                if not (
                    global_index == expected_global
                    and domain_id == expected_domain
                    and int(self.global_to_local[global_index]) == local_index
                    and int(expected["step"]) == self.step + 1
                    and expected_cost == int(self.costs[global_index])
                    and math.isclose(
                        score,
                        expected_score,
                        rel_tol=0.0,
                        abs_tol=TRACE_TOLERANCE,
                    )
                ):
                    atomic_json(
                        self.worker_dir.with_name(self.worker_dir.name + "_mismatch.json"),
                        {
                            "status": "failed",
                            "step": self.step + 1,
                            "expected": dict(expected),
                            "actual": {
                                "score": score,
                                "global_index": global_index,
                                "domain_id": domain_id,
                                "local_index": local_index,
                            },
                        },
                    )
                    raise RuntimeError(f"Task017 exact replay mismatch at step {self.step + 1}")
                state = self.states[domain_id]
                coverage_before = state.current_coverage
                retained_before = state.retained_count
                active_before = int(
                    (state.retained & state.valid_function_mask).sum().item()
                )
                delta_average = float(state.losses[local_index].item())
                delta_total = delta_average * float(self.cache.active_size[domain_id])
                substitutes = self._substitutes(state, local_index)
                budget_before = self.removed_cost
                update = state.remove(local_index)
                self.removed_cost += expected_cost
                became_full, affected = self.cache.mark_removed(
                    self.units[global_index].layer
                )
                if became_full:
                    for affected_domain in affected:
                        self.cache.refresh(affected_domain)
                else:
                    self.cache.refresh(domain_id)
                self.risk_rows.append(
                    {
                        "score_mode": self.mode,
                        "step": self.step + 1,
                        "global_index": global_index,
                        "domain_id": domain_id,
                        "layer": self.units[global_index].layer,
                        "unit_type": self.units[global_index].unit_type,
                        "selection_sparsity_before": budget_before
                        / self.total_parameters,
                        "selection_sparsity_after": self.removed_cost
                        / self.total_parameters,
                        "functional_score": score,
                        "delta_average": delta_average,
                        "delta_total": delta_total,
                        "parameter_cost": expected_cost,
                        "domain_retained_units_before": retained_before,
                        "domain_retained_ratio_before": retained_before
                        / state.initial_size,
                        "domain_active_retained_ratio_before": (
                            active_before / self.cache.active_size[domain_id]
                            if self.cache.active_size[domain_id]
                            else 1.0
                        ),
                        "domain_coverage_before": coverage_before,
                        "domain_coverage_after": update["coverage_after"],
                        "coverage_drop": coverage_before
                        - float(update["coverage_after"]),
                        "previous_removals_same_domain": state.version - 1,
                        **substitutes,
                    }
                )
                self.step += 1
                self._record_due_snapshots()
                if progress is not None:
                    progress.update(1)
                    if time.perf_counter() - started > 2.0:
                        progress.set_postfix(
                            sparsity=f"{self.removed_cost/self.total_parameters:.2%}",
                            gpu=f"{self.torch.cuda.max_memory_allocated(self.device)/2**30:.2f}G",
                        )
                if self.step % CHECKPOINT_INTERVAL == 0 or self.interrupt_requested:
                    self._save_checkpoint()
                if self.interrupt_requested:
                    raise KeyboardInterrupt
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if progress is not None:
                progress.close()
        self._record_due_snapshots()
        if self.next_snapshot != len(SNAPSHOTS):
            raise RuntimeError("Task017 did not reach every diagnostic snapshot")
        expected_indices = removed_indices(self.expected_trace)
        actual_indices = [int(row["global_index"]) for row in self.risk_rows]
        exact = actual_indices == expected_indices
        if not exact:
            raise RuntimeError("Task017 replay sequence differs from saved trace")
        target_budget = 0.30 * self.total_parameters
        if self.removed_cost < target_budget:
            raise RuntimeError("Task017 saved trace did not reach its target budget")
        if expected_indices:
            before_final = self.removed_cost - int(self.costs[expected_indices[-1]])
            if before_final >= target_budget:
                raise RuntimeError("Task017 saved trace continued after budget stop")
        if self.worker_dir.exists():
            shutil.rmtree(self.worker_dir)
        self.worker_dir.mkdir(parents=True)
        atomic_csv(
            self.worker_dir / "incremental_selection_risk_full.csv",
            tuple(self.risk_rows[0]),
            self.risk_rows,
        )
        atomic_csv(
            self.worker_dir / "domain_snapshots.csv",
            tuple(self.domain_rows[0]),
            self.domain_rows,
        )
        atomic_npz(self.worker_dir / "candidate_snapshots.npz", self.snapshots)
        atomic_json(
            self.worker_dir / "replay_evidence.json",
            {
                "status": "passed",
                "score_mode": self.mode,
                "trace_replay_exact": exact,
                "expected_sequence_sha256": sequence_sha256(expected_indices),
                "actual_sequence_sha256": sequence_sha256(actual_indices),
                "steps": len(actual_indices),
                "removed_parameter_cost": self.removed_cost,
                "target_parameter_budget": target_budget,
                "budget_overshoot": self.removed_cost - target_budget,
                "estimated_parameter_sparsity": self.removed_cost
                / self.total_parameters,
                "device": str(self.device),
                "gpu_name": self.torch.cuda.get_device_name(self.device),
                "peak_cuda_bytes": int(
                    self.torch.cuda.max_memory_allocated(self.device)
                ),
                "model_forward_executed": False,
                "pruning_applied": False,
                "validation_executed": False,
            },
        )
        for path in (self.checkpoint_json, self.checkpoint_npz):
            if path.exists():
                path.unlink()


def _snapshot(path: Path, progress: float) -> dict[str, np.ndarray]:
    prefix = f"p{int(round(progress * 100)):02d}_"
    with np.load(path, allow_pickle=False) as payload:
        result = {
            key[len(prefix) :]: payload[key].copy()
            for key in payload.files
            if key.startswith(prefix)
        }
    required = {
        "global_index",
        "domain_id",
        "unit_type",
        "delta_average",
        "delta_total",
        "best_similarity",
        "active_demand_count",
        "retained_ratio",
        "coverage",
        "estimated_sparsity",
    }
    if set(result) != required:
        raise ValueError(f"Incomplete Task017 snapshot {progress}: {set(result)}")
    return result


def _concentration_row(
    mode: str,
    extra_trace: Sequence[Mapping[str, object]],
    key: str,
    top: int,
    universe_size: int,
) -> dict[str, object]:
    sequence = [str(row[key]) for row in extra_trace]
    counts = Counter(sequence)
    ordered = sorted(counts.values(), reverse=True)
    if len(ordered) > universe_size:
        raise ValueError("Concentration universe is smaller than touched groups")
    full_counts = ordered + [0] * (universe_size - len(ordered))
    total = len(sequence)
    same_pairs = sum(
        left == right for left, right in zip(sequence[:-1], sequence[1:])
    )
    maximum_run = 0
    current_run = 0
    previous = None
    for value in sequence:
        current_run = current_run + 1 if value == previous else 1
        maximum_run = max(maximum_run, current_run)
        previous = value
    return {
        "score_mode": mode,
        "axis": key,
        "extra_units": total,
        "unique_groups_touched": len(counts),
        "consecutive_same_group_pairs": same_pairs,
        "maximum_consecutive_run": maximum_run,
        "maximum_removals_one_group": max(ordered, default=0),
        f"top{top}_share": sum(ordered[:top]) / total if total else 0.0,
        "hhi": herfindahl_hirschman(ordered),
        "gini": gini_coefficient(full_counts),
    }


def _domain_row_map(
    rows: Sequence[Mapping[str, str]], progress: float
) -> dict[int, dict[str, str]]:
    selected = {
        int(row["domain_id"]): dict(row)
        for row in rows
        if math.isclose(
            float(row["snapshot_target"]), progress, rel_tol=0.0, abs_tol=1e-12
        )
    }
    if len(selected) != EXPECTED_DOMAINS:
        raise ValueError(
            f"Domain snapshot {progress:.0%} has {len(selected)} domains"
        )
    return selected


def _metric(metrics: Mapping[str, object], *names: str) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(names)


def analyze(
    task014_root: Path,
    task015_root: Path,
    task016_root: Path,
    output_dir: Path,
) -> None:
    from task015_attention_ffn_diagnosis import (
        _descriptor_path,
        _load_descriptors,
        _load_units,
        _mapping_paths,
        infer_parameter_costs,
    )

    task014_root, task015_root = Path(task014_root), Path(task015_root)
    task016_root, output_dir = Path(task016_root), Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task017 artifact identity gate has not passed")
    replay_evidence = {
        mode: read_json(output_dir / "replay" / mode / "replay_evidence.json")
        for mode in MODES
    }
    if not all(value.get("trace_replay_exact") is True for value in replay_evidence.values()):
        raise RuntimeError("Task017 exact replay gate has not passed")
    unit_path, layer_path, _ = _mapping_paths(task014_root)
    units = _load_units(unit_path)
    costs, _ = infer_parameter_costs(units, read_csv(layer_path))
    descriptor_path = _descriptor_path(
        task014_root, _run_dir(task014_root, task016_root, "domain_average", 0.30)
    )
    descriptors = _load_descriptors(descriptor_path, units)  # [N,3]
    energy = np.load(
        task015_root / "functional_energy.npy", mmap_mode="r", allow_pickle=False
    )
    if energy.shape != (EXPECTED_UNITS,):
        raise ValueError("Task015 energy vector has the wrong shape")
    valid = np.load(
        task014_root / "field_cache" / "aligned_function_valid_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if valid.shape != (EXPECTED_UNITS,):
        raise ValueError("Task014 valid-function mask has the wrong shape")

    traces = {
        (mode, target): read_csv(
            _run_dir(task014_root, task016_root, mode, target)
            / "functional_selection_trace.csv"
        )
        for mode in MODES
        for target in TARGETS
    }
    for mode in MODES:
        expected_sequence = sequence_sha256(
            removed_indices(traces[(mode, 0.30)])
        )
        if replay_evidence[mode].get("expected_sequence_sha256") != expected_sequence:
            raise RuntimeError(f"Stale Task017 {mode} replay evidence")
    sets = reconstruct_removed_sets(traces)
    nested = {mode: verify_nested_removal_sets(sets, mode) for mode in MODES}
    sequence_prefix_nested = {}
    for mode in MODES:
        sequence10 = removed_indices(traces[(mode, 0.10)])
        sequence20 = removed_indices(traces[(mode, 0.20)])
        sequence30 = removed_indices(traces[(mode, 0.30)])
        sequence_prefix_nested[mode] = (
            sequence20[: len(sequence10)] == sequence10
            and sequence30[: len(sequence20)] == sequence20
        )
    set_rows = []
    for mode in MODES:
        for target in TARGETS:
            rows = traces[(mode, target)]
            indices = removed_indices(rows)
            set_rows.append(
                {
                    "score_mode": mode,
                    "target": target,
                    "removed_units": len(indices),
                    "attention_removed": sum(
                        units[index].unit_type == TYPE_ATTENTION for index in indices
                    ),
                    "ffn_removed": sum(
                        units[index].unit_type == TYPE_FFN for index in indices
                    ),
                    "removed_parameter_cost": int(costs[indices].sum()),
                    "set_sha256": set_sha256(indices),
                    "sequence_sha256": sequence_sha256(indices),
                    "sequence_prefix_nested": sequence_prefix_nested[mode],
                }
            )
    atomic_csv(output_dir / "removed_set_identity.csv", tuple(set_rows[0]), set_rows)
    if not all(nested.values()) or not all(sequence_prefix_nested.values()):
        atomic_json(
            output_dir / "task017_completion.json",
            {
                "artifact_identity_pass": True,
                "removed_set_reconstruction_pass": True,
                "average_sets_nested": nested["domain_average"],
                "total_sets_nested": nested["domain_total"],
                "average_sequence_prefix_nested": sequence_prefix_nested[
                    "domain_average"
                ],
                "total_sequence_prefix_nested": sequence_prefix_nested[
                    "domain_total"
                ],
                "analysis_complete": False,
            },
        )
        raise RuntimeError("Task017 removal sets or ordered trace prefixes are not nested")

    a_extra = sets[("domain_average", 0.30)] - sets[("domain_average", 0.20)]
    t_extra = sets[("domain_total", 0.30)] - sets[("domain_total", 0.20)]
    named_sets = {
        "Average_extra": a_extra,
        "Total_extra": t_extra,
        "Common_extra": a_extra & t_extra,
        "Average_only_extra": a_extra - t_extra,
        "Total_only_extra": t_extra - a_extra,
    }
    parameters_before = int(
        read_json(
            _run_dir(task014_root, task016_root, "domain_average", 0.30)
            / "final_metrics.json"
        )["parameters_before"]
    )
    extra_summary = []
    for name, indices in named_sets.items():
        cost = int(costs[sorted(indices)].sum()) if indices else 0
        extra_summary.append(
            {
                "set_name": name,
                "unit_count": len(indices),
                "attention_count": sum(
                    units[index].unit_type == TYPE_ATTENTION for index in indices
                ),
                "ffn_count": sum(
                    units[index].unit_type == TYPE_FFN for index in indices
                ),
                "parameter_cost": cost,
                "fraction_of_incremental_10_percent_budget": cost
                / (0.10 * parameters_before),
                "set_sha256": set_sha256(indices),
            }
        )
    atomic_csv(
        output_dir / "extra_pruning_set_summary.csv",
        tuple(extra_summary[0]),
        extra_summary,
    )

    snapshots = {
        (mode, progress): _snapshot(
            output_dir / "replay" / mode / "candidate_snapshots.npz", progress
        )
        for mode in MODES
        for progress in SNAPSHOTS
    }
    domain_snapshot_rows = {
        mode: read_csv(output_dir / "replay" / mode / "domain_snapshots.csv")
        for mode in MODES
    }
    risk_rows = {
        mode: read_csv(
            output_dir / "replay" / mode / "incremental_selection_risk_full.csv"
        )
        for mode in MODES
    }
    p00 = snapshots[("domain_average", 0.0)]
    if len(p00["global_index"]) != EXPECTED_UNITS:
        raise ValueError("Task017 initial snapshot does not cover all pruning units")
    domain_by_global = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
    domain_by_global[p00["global_index"]] = p00["domain_id"]
    initial_average = np.full(EXPECTED_UNITS, np.nan, dtype=np.float64)
    initial_total = np.full(EXPECTED_UNITS, np.nan, dtype=np.float64)
    initial_average[p00["global_index"]] = p00["delta_average"]
    initial_total[p00["global_index"]] = p00["delta_total"]
    if np.any(domain_by_global < 0) or not np.isfinite(initial_average).all():
        raise ValueError("Task017 initial candidate identity is incomplete")
    initial_domain = _domain_row_map(domain_snapshot_rows["domain_average"], 0.0)
    domain_members: dict[int, list[int]] = defaultdict(list)
    for index, domain_id in enumerate(domain_by_global):
        domain_members[int(domain_id)].append(index)
    if len(domain_members) != EXPECTED_DOMAINS:
        raise ValueError("Task017 domain membership lookup is incomplete")

    trace_by_index = {
        mode: {int(row["global_index"]): row for row in traces[(mode, 0.30)]}
        for mode in MODES
    }
    risk_by_index = {
        mode: {int(row["global_index"]): row for row in risk_rows[mode]}
        for mode in MODES
    }
    extra_unit_rows = []
    for index in sorted(a_extra | t_extra):
        average_trace = trace_by_index["domain_average"].get(index, {})
        total_trace = trace_by_index["domain_total"].get(index, {})
        average_risk = risk_by_index["domain_average"].get(index, {})
        total_risk = risk_by_index["domain_total"].get(index, {})
        domain_id = int(domain_by_global[index])
        extra_unit_rows.append(
            {
                "global_index": index,
                "layer": units[index].layer,
                "stage": stage_from_layer(units[index].layer),
                "block": block_from_layer(units[index].layer),
                "unit_type": units[index].unit_type,
                "unit_index": units[index].unit_index,
                "in_average_extra": index in a_extra,
                "in_total_extra": index in t_extra,
                "D_abs": float(descriptors[index, 0]),
                "D_rel": float(descriptors[index, 1]),
                "D_dyn": float(descriptors[index, 2]),
                "functional_energy": float(energy[index]),
                "domain_id": domain_id,
                "domain_initial_size": int(initial_domain[domain_id]["initial_units"]),
                "domain_valid_demand_count": int(
                    initial_domain[domain_id]["valid_demand_count"]
                ),
                "initial_delta_average": float(initial_average[index]),
                "initial_delta_total": float(initial_total[index]),
                "selection_step_average": average_trace.get("step", ""),
                "selection_step_total": total_trace.get("step", ""),
                "selection_sparsity_average": average_risk.get(
                    "selection_sparsity_after", ""
                ),
                "selection_sparsity_total": total_risk.get(
                    "selection_sparsity_after", ""
                ),
                "delta_average_at_selection": average_risk.get(
                    "delta_average", ""
                ),
                "delta_total_at_selection": total_risk.get("delta_total", ""),
                "parameter_cost": int(costs[index]),
                "coverage_before_average": average_risk.get(
                    "domain_coverage_before", ""
                ),
                "coverage_after_average": average_risk.get(
                    "domain_coverage_after", ""
                ),
                "coverage_drop_average": average_risk.get("coverage_drop", ""),
                "coverage_before_total": total_risk.get(
                    "domain_coverage_before", ""
                ),
                "coverage_after_total": total_risk.get(
                    "domain_coverage_after", ""
                ),
                "coverage_drop_total": total_risk.get("coverage_drop", ""),
                "primary_coverage_mode": (
                    "domain_total" if index in t_extra else "domain_average"
                ),
                "coverage_before": (
                    total_risk.get("domain_coverage_before", "")
                    if index in t_extra
                    else average_risk.get("domain_coverage_before", "")
                ),
                "coverage_after": (
                    total_risk.get("domain_coverage_after", "")
                    if index in t_extra
                    else average_risk.get("domain_coverage_after", "")
                ),
                "coverage_drop": (
                    total_risk.get("coverage_drop", "")
                    if index in t_extra
                    else average_risk.get("coverage_drop", "")
                ),
            }
        )
    atomic_csv(
        output_dir / "extra_pruning_units.csv",
        tuple(extra_unit_rows[0]),
        extra_unit_rows,
    )

    type_rows = []
    for mode, indices in (("domain_average", a_extra), ("domain_total", t_extra)):
        for unit_type in (TYPE_ATTENTION, TYPE_FFN):
            typed = [index for index in indices if units[index].unit_type == unit_type]
            type_rows.append(
                {
                    "score_mode": mode,
                    "unit_type": unit_type,
                    "extra_units": len(typed),
                    "extra_parameter_cost": int(costs[typed].sum()) if typed else 0,
                    "parameter_fraction_within_extra": (
                        float(costs[typed].sum() / costs[sorted(indices)].sum())
                        if indices
                        else 0.0
                    ),
                }
            )
    atomic_csv(output_dir / "extra_type_composition.csv", tuple(type_rows[0]), type_rows)

    layer_rows_by_mode = {}
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        rows = incremental_layer_statistics(
            units=units,
            removed_at_20=sets[(mode, 0.20)],
            removed_at_30=sets[(mode, 0.30)],
            costs=costs,
        )
        layer_rows_by_mode[mode] = {row["layer"]: row for row in rows}
    layer_rows = []
    for mode in MODES:
        other = "domain_total" if mode == "domain_average" else "domain_average"
        for row in layer_rows_by_mode[mode].values():
            comparison = layer_rows_by_mode[other][row["layer"]]
            layer_rows.append(
                {
                    "score_mode": mode,
                    **row,
                    "other_mode_additional_removed": comparison[
                        "additional_removed_20_to_30"
                    ],
                    "extra_count_difference_vs_other": row[
                        "additional_removed_20_to_30"
                    ]
                    - comparison["additional_removed_20_to_30"],
                    "more_concentrated_than_other": row[
                        "incremental_removal_ratio"
                    ]
                    > comparison["incremental_removal_ratio"],
                }
            )
    layer_rows.sort(
        key=lambda row: (
            row["score_mode"],
            -int(row["additional_removed_20_to_30"]),
            -float(row["incremental_removal_ratio"]),
        )
    )
    atomic_csv(output_dir / "extra_layerwise_pruning.csv", tuple(layer_rows[0]), layer_rows)

    stage_rows = []
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        for stage in range(4):
            for unit_type in (TYPE_ATTENTION, TYPE_FFN):
                members = [
                    index
                    for index, unit in enumerate(units)
                    if stage_from_layer(unit.layer) == stage
                    and unit.unit_type == unit_type
                ]
                member_set = set(members)
                added = [index for index in extra if index in member_set]
                cost = int(costs[added].sum()) if added else 0
                stage_rows.append(
                    {
                        "score_mode": mode,
                        "stage": stage,
                        "unit_type": unit_type,
                        "additional_units_removed": len(added),
                        "additional_parameters_removed": cost,
                        "fraction_of_extra_10_percent_budget": cost
                        / (0.10 * parameters_before),
                        "remaining_units_at_30": len(
                            set(members) - sets[(mode, 0.30)]
                        ),
                    }
                )
    atomic_csv(output_dir / "extra_stagewise_pruning.csv", tuple(stage_rows[0]), stage_rows)

    trajectory = []
    for mode in MODES:
        for progress in (0.0, 0.10, 0.20, 0.30):
            for row in _domain_row_map(domain_snapshot_rows[mode], progress).values():
                trajectory.append(dict(row))
    atomic_csv(
        output_dir / "domain_depletion_trajectory.csv",
        tuple(trajectory[0]),
        trajectory,
    )

    depletion_rows = []
    domain_maps = {
        (mode, progress): _domain_row_map(domain_snapshot_rows[mode], progress)
        for mode in MODES
        for progress in (0.20, 0.30)
    }
    for domain_id in range(EXPECTED_DOMAINS):
        row: dict[str, object] = {"domain_id": domain_id}
        for mode, prefix in (
            ("domain_average", "average"),
            ("domain_total", "total"),
        ):
            before = domain_maps[(mode, 0.20)][domain_id]
            after = domain_maps[(mode, 0.30)][domain_id]
            row.update(
                {
                    f"{prefix}_extra_removed_20_to_30": int(after["removed_units"])
                    - int(before["removed_units"]),
                    f"{prefix}_retained_ratio_20": float(before["retained_ratio"]),
                    f"{prefix}_retained_ratio_30": float(after["retained_ratio"]),
                    f"{prefix}_retained_ratio_drop": float(before["retained_ratio"])
                    - float(after["retained_ratio"]),
                    f"{prefix}_active_retained_ratio_20": float(
                        before["active_retained_ratio"]
                    ),
                    f"{prefix}_active_retained_ratio_30": float(
                        after["active_retained_ratio"]
                    ),
                }
            )
        row["total_minus_average_extra_removed"] = (
            int(row["total_extra_removed_20_to_30"])
            - int(row["average_extra_removed_20_to_30"])
        )
        row["total_more_depleted_than_average"] = float(
            row["total_retained_ratio_30"]
        ) < float(row["average_retained_ratio_30"])
        depletion_rows.append(row)
    atomic_csv(
        output_dir / "domain_depletion_20_30.csv",
        tuple(depletion_rows[0]),
        depletion_rows,
    )

    bin_rows = []
    for mode in MODES:
        for progress in (0.20, 0.30):
            current = domain_maps[(mode, progress)]
            for threshold in (0.75, 0.50, 0.25, 0.10):
                ids = [
                    domain_id
                    for domain_id, row in current.items()
                    if float(row["retained_ratio"]) <= threshold
                ]
                members = [index for domain_id in ids for index in domain_members[domain_id]]
                bin_rows.append(
                    {
                        "score_mode": mode,
                        "progress": progress,
                        "retained_ratio_le": threshold,
                        "domain_count": len(ids),
                        "initial_unit_count": len(members),
                        "attention_count": sum(
                            units[index].unit_type == TYPE_ATTENTION for index in members
                        ),
                        "ffn_count": sum(
                            units[index].unit_type == TYPE_FFN for index in members
                        ),
                        "diagnostic_bin_only": True,
                    }
                )
    atomic_csv(output_dir / "domain_depletion_bins.csv", tuple(bin_rows[0]), bin_rows)

    coverage_rows = []
    for domain_id in range(EXPECTED_DOMAINS):
        row = {"domain_id": domain_id}
        for mode, prefix in (
            ("domain_average", "average"),
            ("domain_total", "total"),
        ):
            before = domain_maps[(mode, 0.20)][domain_id]
            after = domain_maps[(mode, 0.30)][domain_id]
            row.update(
                {
                    f"{prefix}_coverage_20": float(before["coverage"]),
                    f"{prefix}_coverage_30": float(after["coverage"]),
                    f"{prefix}_coverage_drop_from_initial_20": float(
                        before["coverage_drop_from_initial"]
                    ),
                    f"{prefix}_coverage_drop_from_initial_30": float(
                        after["coverage_drop_from_initial"]
                    ),
                    f"{prefix}_coverage_drop_20_to_30": float(before["coverage"])
                    - float(after["coverage"]),
                }
            )
        coverage_rows.append(row)
    atomic_csv(
        output_dir / "domain_coverage_20_30.csv",
        tuple(coverage_rows[0]),
        coverage_rows,
    )
    coverage_summary_rows = []
    for mode, prefix in (("domain_average", "average"), ("domain_total", "total")):
        for progress in (0.20, 0.30):
            values = [
                float(row[f"{prefix}_coverage_{int(progress * 100)}"])
                for row in coverage_rows
            ]
            stats = quantile_summary(values)
            coverage_summary_rows.append(
                {
                    "score_mode": mode,
                    "progress": progress,
                    **stats,
                    "domains_with_coverage_drop_gt_tolerance": (
                        sum(
                            float(row[f"{prefix}_coverage_drop_20_to_30"])
                            > TRACE_TOLERANCE
                            for row in coverage_rows
                        )
                        if math.isclose(progress, 0.30, abs_tol=1e-12)
                        else 0
                    ),
                    "coverage_drop_tolerance": TRACE_TOLERANCE,
                }
            )
    atomic_csv(
        output_dir / "domain_coverage_summary.csv",
        tuple(coverage_summary_rows[0]),
        coverage_summary_rows,
    )
    worst_rows = []
    for row in sorted(
        coverage_rows,
        key=lambda value: float(value["total_coverage_drop_20_to_30"]),
        reverse=True,
    )[:20]:
        domain_id = int(row["domain_id"])
        members = domain_members[domain_id]
        total_extra_domain = [index for index in t_extra if domain_by_global[index] == domain_id]
        worst_rows.append(
            {
                "domain_id": domain_id,
                "domain_size": len(members),
                "valid_demand_count": int(initial_domain[domain_id]["valid_demand_count"]),
                "attention_count": sum(
                    units[index].unit_type == TYPE_ATTENTION for index in members
                ),
                "ffn_count": sum(units[index].unit_type == TYPE_FFN for index in members),
                "layers": "|".join(sorted({units[index].layer for index in members})),
                "total_coverage_20": row["total_coverage_20"],
                "total_coverage_30": row["total_coverage_30"],
                "total_coverage_drop_20_to_30": row[
                    "total_coverage_drop_20_to_30"
                ],
                "average_coverage_20": row["average_coverage_20"],
                "average_coverage_30": row["average_coverage_30"],
                "average_coverage_drop_20_to_30": row[
                    "average_coverage_drop_20_to_30"
                ],
                "extra_removed_units_total": len(total_extra_domain),
                "extra_removed_parameter_cost_total": int(
                    costs[total_extra_domain].sum()
                )
                if total_extra_domain
                else 0,
            }
        )
    atomic_csv(
        output_dir / "worst_total_domain_coverage_drop.csv",
        tuple(worst_rows[0]),
        worst_rows,
    )

    extra_trace = {}
    domain_concentration = []
    layer_concentration = []
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        rows = [
            {
                **risk_by_index[mode][index],
                "domain_id": int(domain_by_global[index]),
                "layer": units[index].layer,
            }
            for index in removed_indices(traces[(mode, 0.30)])
            if index in extra
        ]
        extra_trace[mode] = rows
        domain_concentration.append(
            _concentration_row(mode, rows, "domain_id", 10, EXPECTED_DOMAINS)
        )
        layer_concentration.append(
            _concentration_row(mode, rows, "layer", 5, 48)
        )
    atomic_csv(
        output_dir / "extra_domain_concentration.csv",
        tuple(domain_concentration[0]),
        domain_concentration,
    )
    atomic_csv(
        output_dir / "extra_layer_concentration.csv",
        tuple(layer_concentration[0]),
        layer_concentration,
    )

    score_rows = []
    correlation_rows = []
    for mode in MODES:
        for progress in (0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
            snap = snapshots[(mode, progress)]
            for code, unit_type in ((0, TYPE_ATTENTION), (1, TYPE_FFN)):
                mask = snap["unit_type"] == code
                average_stats = quantile_summary(snap["delta_average"][mask])
                total_stats = quantile_summary(snap["delta_total"][mask])
                size_stats = quantile_summary(
                    snap["active_demand_count"][mask].astype(np.float64)
                )
                score_rows.append(
                    {
                        "score_mode": mode,
                        "progress": progress,
                        "unit_type": unit_type,
                        **{f"delta_average_{key}": value for key, value in average_stats.items()},
                        **{f"delta_total_{key}": value for key, value in total_stats.items()},
                        "active_domain_size_q25": size_stats["q25"],
                        "active_domain_size_median": size_stats["median"],
                        "active_domain_size_q75": size_stats["q75"],
                    }
                )
            correlation_rows.append(
                {
                    "score_mode": mode,
                    "progress": progress,
                    "candidate_count": len(snap["global_index"]),
                    "spearman_active_demand_vs_delta_average": spearman(
                        snap["active_demand_count"], snap["delta_average"]
                    ),
                    "spearman_active_demand_vs_delta_total": spearman(
                        snap["active_demand_count"], snap["delta_total"]
                    ),
                    "spearman_retained_ratio_vs_delta_total": spearman(
                        snap["retained_ratio"], snap["delta_total"]
                    ),
                    "spearman_coverage_vs_delta_total": spearman(
                        snap["coverage"], snap["delta_total"]
                    ),
                }
            )
    atomic_csv(output_dir / "score_evolution_20_30.csv", tuple(score_rows[0]), score_rows)
    atomic_csv(
        output_dir / "dynamic_score_correlation.csv",
        tuple(correlation_rows[0]),
        correlation_rows,
    )

    incremental_risk = []
    substitute_rows = []
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        for row in extra_trace[mode]:
            payload = dict(row)
            payload["in_exact_20_to_30_set_difference"] = True
            incremental_risk.append(payload)
            substitute_rows.append(
                {
                    key: payload[key]
                    for key in (
                        "score_mode",
                        "step",
                        "global_index",
                        "domain_id",
                        "layer",
                        "unit_type",
                        "selection_sparsity_before",
                        "domain_retained_ratio_before",
                        "domain_coverage_before",
                        "previous_removals_same_domain",
                        "best_remaining_similarity",
                        "second_best_remaining_similarity",
                        "substitutes_ge_0.1",
                        "substitutes_ge_0.3",
                        "substitutes_ge_0.5",
                    )
                }
            )
    atomic_csv(
        output_dir / "incremental_selection_risk.csv",
        tuple(incremental_risk[0]),
        incremental_risk,
    )
    atomic_csv(
        output_dir / "substitute_availability_20_30.csv",
        tuple(substitute_rows[0]),
        substitute_rows,
    )

    neighborhood_rows = []
    for domain_id in range(EXPECTED_DOMAINS):
        row = {"domain_id": domain_id}
        for mode, prefix in (("domain_average", "average"), ("domain_total", "total")):
            before = domain_maps[(mode, 0.20)][domain_id]
            after = domain_maps[(mode, 0.30)][domain_id]
            for field in (
                "mean_positive_neighbors",
                "mean_best_remaining_similarity",
                "median_best_remaining_similarity",
            ):
                row[f"{prefix}_{field}_20"] = float(before[field])
                row[f"{prefix}_{field}_30"] = float(after[field])
                row[f"{prefix}_{field}_change"] = float(after[field]) - float(
                    before[field]
                )
        neighborhood_rows.append(row)
    atomic_csv(
        output_dir / "functional_neighborhood_shrinkage.csv",
        tuple(neighborhood_rows[0]),
        neighborhood_rows,
    )

    total_risk = risk_by_index["domain_total"]
    p30_total = snapshots[("domain_total", 0.30)]
    p30_best = {
        int(index): float(value)
        for index, value in zip(
            p30_total["global_index"], p30_total["best_similarity"]
        )
    }
    attention_rows = []
    for index, unit in enumerate(units):
        if unit.unit_type != TYPE_ATTENTION:
            continue
        if index in sets[("domain_total", 0.20)]:
            cohort = "removed_before_or_at_20"
        elif index in t_extra:
            cohort = "total_extra_20_to_30"
        else:
            cohort = "surviving_at_30"
        risk = total_risk.get(index, {})
        attention_rows.append(
            {
                "global_index": index,
                "cohort": cohort,
                "stage": stage_from_layer(unit.layer),
                "block": block_from_layer(unit.layer),
                "layer": unit.layer,
                "head_index": unit.unit_index,
                "D_abs": float(descriptors[index, 0]),
                "D_rel": float(descriptors[index, 1]),
                "D_dyn": float(descriptors[index, 2]),
                "functional_energy": float(energy[index]),
                "best_substitute_similarity": risk.get(
                    "best_remaining_similarity", p30_best.get(index, "")
                ),
                "domain_coverage_before_removal": risk.get(
                    "domain_coverage_before", ""
                ),
                "parameter_cost": int(costs[index]),
                "selection_sparsity": risk.get("selection_sparsity_after", ""),
            }
        )
    atomic_csv(
        output_dir / "attention_high_sparsity_comparison.csv",
        tuple(attention_rows[0]),
        attention_rows,
    )

    ffn_rows = []
    by_ffn: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        if unit.unit_type != TYPE_FFN:
            continue
        if index in sets[("domain_total", 0.20)]:
            cohort = "removed_before_or_at_20"
        elif index in t_extra:
            cohort = "total_extra_20_to_30"
        else:
            cohort = "surviving_at_30"
        by_ffn[(unit.layer, stage_from_layer(unit.layer), cohort)].append(index)
    for (layer, stage, cohort), indices in sorted(by_ffn.items()):
        ffn_rows.append(
            {
                "layer": layer,
                "stage": stage,
                "selection_interval": cohort,
                "unit_count": len(indices),
                "D_abs_mean": float(descriptors[indices, 0].mean()),
                "D_abs_median": float(np.median(descriptors[indices, 0])),
                "D_rel_mean": float(descriptors[indices, 1].mean()),
                "D_rel_median": float(np.median(descriptors[indices, 1])),
                "D_dyn_mean": float(descriptors[indices, 2].mean()),
                "D_dyn_median": float(np.median(descriptors[indices, 2])),
                "functional_energy_mean": float(np.asarray(energy[indices]).mean()),
                "functional_energy_median": float(np.median(energy[indices])),
                "parameter_cost": int(costs[indices].sum()),
            }
        )
    atomic_csv(
        output_dir / "ffn_high_sparsity_comparison.csv",
        tuple(ffn_rows[0]),
        ffn_rows,
    )

    budget_rows = []
    intervals = ((0.20, 0.22), (0.22, 0.24), (0.24, 0.26), (0.26, 0.28), (0.28, 0.30))
    for mode in MODES:
        for lower, upper in intervals:
            rows = [
                row
                for row in risk_rows[mode]
                if lower <= float(row["selection_sparsity_before"]) < upper
            ]
            attention_cost = sum(
                int(row["parameter_cost"])
                for row in rows
                if row["unit_type"] == TYPE_ATTENTION
            )
            ffn_cost = sum(
                int(row["parameter_cost"])
                for row in rows
                if row["unit_type"] == TYPE_FFN
            )
            total_cost = attention_cost + ffn_cost
            budget_rows.append(
                {
                    "score_mode": mode,
                    "interval_start": lower,
                    "interval_end": upper,
                    "attention_removed_parameter_cost": attention_cost,
                    "ffn_removed_parameter_cost": ffn_cost,
                    "attention_share": attention_cost / total_cost if total_cost else 0.0,
                    "ffn_share": ffn_cost / total_cost if total_cost else 0.0,
                    "units_removed": len(rows),
                    "actual_interval_parameter_cost": total_cost,
                }
            )
    atomic_csv(
        output_dir / "incremental_budget_composition.csv",
        tuple(budget_rows[0]),
        budget_rows,
    )

    divergence_rows = []
    for label, left, right in (
        ("removed_20", sets[("domain_average", 0.20)], sets[("domain_total", 0.20)]),
        ("removed_30", sets[("domain_average", 0.30)], sets[("domain_total", 0.30)]),
        ("extra_20_to_30", a_extra, t_extra),
    ):
        divergence_rows.append(
            {
                "comparison": label,
                "jaccard": jaccard_similarity(left, right),
                "common_removed_units": len(left & right),
                "average_only_removed_units": len(left - right),
                "total_only_removed_units": len(right - left),
                "union_units": len(left | right),
            }
        )
    atomic_csv(
        output_dir / "selection_set_divergence.csv",
        tuple(divergence_rows[0]),
        divergence_rows,
    )

    descriptor_rows = []
    for name, indices_set in named_sets.items():
        indices = np.asarray(sorted(indices_set), dtype=np.int64)
        quantities = {
            "D_abs": descriptors[indices, 0] if indices.size else np.asarray([]),
            "D_rel": descriptors[indices, 1] if indices.size else np.asarray([]),
            "D_dyn": descriptors[indices, 2] if indices.size else np.asarray([]),
            "functional_energy": np.asarray(energy[indices]) if indices.size else np.asarray([]),
            "domain_size": np.asarray(
                [len(domain_members[int(domain_by_global[index])]) for index in indices],
                dtype=np.float64,
            ),
            "parameter_cost": costs[indices].astype(np.float64),
        }
        for quantity, values in quantities.items():
            stats = quantile_summary(values)
            descriptor_rows.append(
                {"set_name": name, "quantity": quantity, **stats}
            )
    atomic_csv(
        output_dir / "extra_set_descriptor_statistics.csv",
        tuple(descriptor_rows[0]),
        descriptor_rows,
    )

    future_rank_rows = []
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        snap = snapshots[(mode, 0.20)]
        average_ranks = rank_candidates(snap["global_index"], snap["delta_average"])
        total_ranks = rank_candidates(snap["global_index"], snap["delta_total"])
        for index in sorted(extra):
            if index not in average_ranks or index not in total_ranks:
                raise RuntimeError(f"Future extra unit {index} is not feasible at 20%")
            future_rank_rows.append(
                {
                    "score_mode": mode,
                    "global_index": index,
                    "rank_average_at_20": average_ranks[index][0],
                    "rank_total_at_20": total_ranks[index][0],
                    "percentile_average_at_20": average_ranks[index][1],
                    "percentile_total_at_20": total_ranks[index][1],
                }
            )
    atomic_csv(
        output_dir / "future_extra_unit_rank_at_20.csv",
        tuple(future_rank_rows[0]),
        future_rank_rows,
    )
    dynamic_rank_inputs = {
        progress: (
            snapshots[("domain_total", progress)]["global_index"],
            snapshots[("domain_total", progress)]["delta_total"],
        )
        for progress in (0.20, 0.22, 0.24, 0.26, 0.28)
    }
    dynamic_ranks = track_dynamic_ranks(dynamic_rank_inputs, t_extra)
    atomic_csv(
        output_dir / "total_extra_dynamic_rank_shift.csv",
        tuple(dynamic_ranks[0]),
        dynamic_ranks,
    )

    metrics = {
        (mode, target): read_json(
            _run_dir(task014_root, task016_root, mode, target) / "final_metrics.json"
        )
        for mode in MODES
        for target in TARGETS
    }
    concentration_by_mode = {row["score_mode"]: row for row in domain_concentration}
    layer_concentration_by_mode = {row["score_mode"]: row for row in layer_concentration}
    collapse_rows = []
    for mode, extra in (("domain_average", a_extra), ("domain_total", t_extra)):
        before_domains = domain_maps[(mode, 0.20)]
        after_domains = domain_maps[(mode, 0.30)]
        extra_risk = extra_trace[mode]
        attention_cost = sum(
            int(costs[index]) for index in extra if units[index].unit_type == TYPE_ATTENTION
        )
        ffn_cost = sum(
            int(costs[index]) for index in extra if units[index].unit_type == TYPE_FFN
        )
        total_cost = attention_cost + ffn_cost
        top1_20 = _metric(metrics[(mode, 0.20)], "pre_ft_top1")
        top1_30 = _metric(metrics[(mode, 0.30)], "pre_ft_top1")
        collapse_rows.append(
            {
                "score_mode": mode,
                "top1_20": top1_20,
                "top1_30": top1_30,
                "top1_drop_20_to_30": top1_20 - top1_30,
                "extra_attention_units": sum(
                    units[index].unit_type == TYPE_ATTENTION for index in extra
                ),
                "extra_ffn_units": sum(units[index].unit_type == TYPE_FFN for index in extra),
                "extra_attention_parameter_share": attention_cost / total_cost,
                "extra_ffn_parameter_share": ffn_cost / total_cost,
                "unique_domains_touched": concentration_by_mode[mode][
                    "unique_groups_touched"
                ],
                "top10_domain_share": concentration_by_mode[mode]["top10_share"],
                "domain_removal_hhi": concentration_by_mode[mode]["hhi"],
                "unique_layers_touched": layer_concentration_by_mode[mode][
                    "unique_groups_touched"
                ],
                "top5_layer_share": layer_concentration_by_mode[mode]["top5_share"],
                "layer_removal_hhi": layer_concentration_by_mode[mode]["hhi"],
                "median_domain_retained_ratio_20": float(
                    np.median([float(row["retained_ratio"]) for row in before_domains.values()])
                ),
                "median_domain_retained_ratio_30": float(
                    np.median([float(row["retained_ratio"]) for row in after_domains.values()])
                ),
                "mean_domain_coverage_20": float(
                    np.mean([float(row["coverage"]) for row in before_domains.values()])
                ),
                "mean_domain_coverage_30": float(
                    np.mean([float(row["coverage"]) for row in after_domains.values()])
                ),
                "min_domain_coverage_30": min(
                    float(row["coverage"]) for row in after_domains.values()
                ),
                "median_best_substitute_similarity_extra": float(
                    np.median(
                        [float(row["best_remaining_similarity"]) for row in extra_risk]
                    )
                )
                if extra_risk
                else 0.0,
            }
        )
    atomic_csv(output_dir / "collapse_summary.csv", tuple(collapse_rows[0]), collapse_rows)

    average_summary = next(
        row for row in collapse_rows if row["score_mode"] == "domain_average"
    )
    total_summary = next(
        row for row in collapse_rows if row["score_mode"] == "domain_total"
    )
    corr_total_20 = next(
        row
        for row in correlation_rows
        if row["score_mode"] == "domain_total"
        and math.isclose(float(row["progress"]), 0.20, abs_tol=1e-12)
    )
    corr_total_30 = next(
        row
        for row in correlation_rows
        if row["score_mode"] == "domain_total"
        and math.isclose(float(row["progress"]), 0.30, abs_tol=1e-12)
    )
    rank_medians = {}
    for progress in (0.20, 0.28):
        values = [
            float(row["percentile"])
            for row in dynamic_ranks
            if row["available"]
            and math.isclose(float(row["progress"]), progress, abs_tol=1e-12)
        ]
        rank_medians[progress] = float(np.median(values)) if values else math.nan
    root_cause_rows = [
        {
            "category": "A_type_imbalance",
            "supported_directionally": float(
                total_summary["extra_attention_parameter_share"]
            )
            > float(average_summary["extra_attention_parameter_share"]),
            "evidence": (
                f"attention parameter share {average_summary['extra_attention_parameter_share']}"
                f" -> {total_summary['extra_attention_parameter_share']}"
            ),
        },
        {
            "category": "B_layer_concentration",
            "supported_directionally": float(total_summary["top5_layer_share"])
            > float(average_summary["top5_layer_share"]),
            "evidence": (
                f"top5 layer share {average_summary['top5_layer_share']}"
                f" -> {total_summary['top5_layer_share']}"
            ),
        },
        {
            "category": "C_domain_depletion",
            "supported_directionally": (
                float(total_summary["top10_domain_share"])
                > float(average_summary["top10_domain_share"])
                or float(total_summary["median_domain_retained_ratio_30"])
                < float(average_summary["median_domain_retained_ratio_30"])
            ),
            "evidence": (
                f"top10 share {average_summary['top10_domain_share']} -> "
                f"{total_summary['top10_domain_share']}; median retained30 "
                f"{average_summary['median_domain_retained_ratio_30']} -> "
                f"{total_summary['median_domain_retained_ratio_30']}"
            ),
        },
        {
            "category": "D_substitute_exhaustion",
            "supported_directionally": float(
                total_summary["median_best_substitute_similarity_extra"]
            )
            < float(average_summary["median_best_substitute_similarity_extra"]),
            "evidence": (
                f"median best substitute "
                f"{average_summary['median_best_substitute_similarity_extra']} -> "
                f"{total_summary['median_best_substitute_similarity_extra']}"
            ),
        },
        {
            "category": "E_dynamic_ranking_amplification",
            "supported_directionally": (
                math.isfinite(rank_medians[0.20])
                and math.isfinite(rank_medians[0.28])
                and rank_medians[0.28] < rank_medians[0.20]
            ),
            "evidence": (
                f"median available T_extra total-rank percentile "
                f"{rank_medians[0.20]} -> {rank_medians[0.28]}"
            ),
        },
        {
            "category": "F_score_scale_instability",
            "supported_directionally": abs(
                float(corr_total_30["spearman_active_demand_vs_delta_total"])
            )
            > abs(float(corr_total_20["spearman_active_demand_vs_delta_total"])),
            "evidence": (
                f"abs Spearman(active demand, total) "
                f"{abs(float(corr_total_20['spearman_active_demand_vs_delta_total']))}"
                f" -> {abs(float(corr_total_30['spearman_active_demand_vs_delta_total']))}"
            ),
        },
    ]
    supported = [
        row["category"] for row in root_cause_rows if row["supported_directionally"]
    ]
    classification = (
        "G_combination"
        if len(supported) > 1
        else (supported[0] if supported else "inconclusive")
    )
    for row in root_cause_rows:
        row["overall_classification"] = classification
        row["diagnostic_only"] = True
    atomic_csv(
        output_dir / "root_cause_evidence.csv",
        tuple(root_cause_rows[0]),
        root_cause_rows,
    )

    atomic_csv(
        output_dir / "high_sparsity_accuracy_curve.csv",
        ("status", "reason", "model_validation_executed"),
        [
            {
                "status": "NOT_RUN",
                "reason": (
                    "Optional micro-ablation requires explicit safe registry/model "
                    "reconstruction evidence; offline diagnosis does not require it"
                ),
                "model_validation_executed": False,
            }
        ],
    )

    _write_figures(
        output_dir=output_dir,
        collapse_rows=collapse_rows,
        type_rows=type_rows,
        layer_rows=layer_rows,
        domain_maps=domain_maps,
        coverage_rows=coverage_rows,
        domain_concentration=domain_concentration,
        extra_trace=extra_trace,
        dynamic_ranks=dynamic_ranks,
    )
    _write_diagnosis(
        output_dir=output_dir,
        collapse_rows=collapse_rows,
        divergence_rows=divergence_rows,
        correlation_rows=correlation_rows,
        neighborhood_rows=neighborhood_rows,
        root_cause_rows=root_cause_rows,
        classification=classification,
    )
    atomic_json(
        output_dir / "task017_completion.json",
        {
            "artifact_identity_pass": True,
            "removed_set_reconstruction_pass": True,
            "average_sets_nested": True,
            "total_sets_nested": True,
            "average_sequence_prefix_nested": True,
            "total_sequence_prefix_nested": True,
            "analysis_complete": True,
            "replay_used": True,
            "trace_replay_exact": True,
            "micro_ablation": "NOT_RUN",
            "pruning_formula_modified": False,
            "model_forward_executed": False,
            "validation_executed": False,
        },
    )


def _write_figures(
    *,
    output_dir: Path,
    collapse_rows: Sequence[Mapping[str, object]],
    type_rows: Sequence[Mapping[str, object]],
    layer_rows: Sequence[Mapping[str, object]],
    domain_maps: Mapping[tuple[str, float], Mapping[int, Mapping[str, str]]],
    coverage_rows: Sequence[Mapping[str, object]],
    domain_concentration: Sequence[Mapping[str, object]],
    extra_trace: Mapping[str, Sequence[Mapping[str, object]]],
    dynamic_ranks: Sequence[Mapping[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/task017-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"domain_average": "#0072B2", "domain_total": "#D55E00"}

    def save(figure, name: str) -> None:
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    for row in collapse_rows:
        axis.plot(
            [20, 30],
            [float(row["top1_20"]), float(row["top1_30"])],
            marker="o",
            label=row["score_mode"],
            color=colors[str(row["score_mode"])],
        )
    axis.set_xlabel("Target parameter budget (%)")
    axis.set_ylabel("Pre-finetune Top-1 (%)")
    axis.set_xticks([20, 30])
    axis.grid(alpha=0.25)
    axis.legend()
    save(figure, "figure1_top1_20_to_30")

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    modes = list(MODES)
    attention = [
        next(
            float(row["extra_parameter_cost"])
            for row in type_rows
            if row["score_mode"] == mode and row["unit_type"] == TYPE_ATTENTION
        )
        for mode in modes
    ]
    ffn = [
        next(
            float(row["extra_parameter_cost"])
            for row in type_rows
            if row["score_mode"] == mode and row["unit_type"] == TYPE_FFN
        )
        for mode in modes
    ]
    positions = np.arange(2)
    axis.bar(positions, attention, label="Attention", color="#CC79A7")
    axis.bar(positions, ffn, bottom=attention, label="FFN", color="#56B4E9")
    axis.set_xticks(positions, modes)
    axis.set_ylabel("20%-30% estimated parameter cost")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save(figure, "figure2_extra_parameter_composition")

    total_layer = {
        str(row["layer"]): int(row["additional_removed_20_to_30"])
        for row in layer_rows
        if row["score_mode"] == "domain_total"
    }
    average_layer = {
        str(row["layer"]): int(row["additional_removed_20_to_30"])
        for row in layer_rows
        if row["score_mode"] == "domain_average"
    }
    layers = sorted(
        total_layer,
        key=lambda layer: max(total_layer[layer], average_layer[layer]),
        reverse=True,
    )[:20]
    figure, axis = plt.subplots(figsize=(10.0, 6.0))
    positions = np.arange(len(layers))
    width = 0.4
    axis.barh(
        positions - width / 2,
        [average_layer[layer] for layer in layers],
        height=width,
        label="domain_average",
    )
    axis.barh(
        positions + width / 2,
        [total_layer[layer] for layer in layers],
        height=width,
        label="domain_total",
    )
    axis.set_yticks(positions, layers)
    axis.invert_yaxis()
    axis.set_xlabel("Additional units removed (20%-30%)")
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    save(figure, "figure3_layerwise_extra_removals")

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for axis, mode in zip(axes, MODES):
        axis.boxplot(
            [
                [
                    float(row["retained_ratio"])
                    for row in domain_maps[(mode, progress)].values()
                ]
                for progress in (0.20, 0.30)
            ],
            labels=["20%", "30%"],
            showfliers=False,
        )
        axis.set_title(mode)
        axis.set_xlabel("Target parameter budget")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Domain retained ratio")
    save(figure, "figure4_domain_retained_ratio")

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.boxplot(
        [
            [float(row[f"{prefix}_coverage_drop_20_to_30"]) for row in coverage_rows]
            for prefix in ("average", "total")
        ],
        labels=["domain_average", "domain_total"],
        showfliers=False,
    )
    axis.set_ylabel("Domain coverage drop (20%-30%)")
    axis.grid(axis="y", alpha=0.25)
    save(figure, "figure5_domain_coverage_change")

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for offset, mode in enumerate(MODES):
        counts = Counter(int(row["domain_id"]) for row in extra_trace[mode])
        ordered = sorted(counts.values(), reverse=True)[:20]
        axis.plot(
            np.arange(1, len(ordered) + 1),
            ordered,
            marker="o",
            markersize=3,
            label=mode,
            color=colors[mode],
        )
    axis.set_xlabel("Domain rank by extra removals")
    axis.set_ylabel("Additional removals")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, "figure6_extra_domain_concentration")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for mode in MODES:
        rows = extra_trace[mode]
        axis.scatter(
            [100 * float(row["selection_sparsity_before"]) for row in rows],
            [float(row["functional_score"]) for row in rows],
            s=7,
            alpha=0.35,
            label=mode,
            color=colors[mode],
        )
    axis.set_xlabel("Estimated parameter sparsity before selection (%)")
    axis.set_ylabel("Mode-specific functional score")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, "figure7_selection_score_progress")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for mode in MODES:
        rows = extra_trace[mode]
        axis.scatter(
            [100 * float(row["selection_sparsity_before"]) for row in rows],
            [float(row["domain_retained_ratio_before"]) for row in rows],
            s=7,
            alpha=0.35,
            label=mode,
            color=colors[mode],
        )
    axis.set_xlabel("Estimated parameter sparsity before selection (%)")
    axis.set_ylabel("Selected domain retained ratio")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, "figure8_selected_domain_retained_ratio")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for mode in MODES:
        rows = extra_trace[mode]
        axis.scatter(
            [100 * float(row["selection_sparsity_before"]) for row in rows],
            [float(row["best_remaining_similarity"]) for row in rows],
            s=7,
            alpha=0.35,
            label=mode,
            color=colors[mode],
        )
    axis.set_xlabel("Estimated parameter sparsity before selection (%)")
    axis.set_ylabel("Best remaining substitute similarity")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, "figure9_best_substitute_similarity")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    available = [row for row in dynamic_ranks if row["available"]]
    for progress in (0.20, 0.22, 0.24, 0.26, 0.28):
        values = [
            float(row["percentile"])
            for row in available
            if math.isclose(float(row["progress"]), progress, abs_tol=1e-12)
        ]
        if values:
            axis.errorbar(
                100 * progress,
                np.median(values),
                yerr=[
                    [np.median(values) - np.quantile(values, 0.25)],
                    [np.quantile(values, 0.75) - np.median(values)],
                ],
                fmt="o",
                color=colors["domain_total"],
            )
    axis.set_xlabel("Estimated parameter sparsity (%)")
    axis.set_ylabel("T_extra total-score rank percentile")
    axis.grid(alpha=0.25)
    save(figure, "figure10_total_extra_dynamic_rank")


def _write_diagnosis(
    *,
    output_dir: Path,
    collapse_rows: Sequence[Mapping[str, object]],
    divergence_rows: Sequence[Mapping[str, object]],
    correlation_rows: Sequence[Mapping[str, object]],
    neighborhood_rows: Sequence[Mapping[str, object]],
    root_cause_rows: Sequence[Mapping[str, object]],
    classification: str,
) -> None:
    summary = {str(row["score_mode"]): row for row in collapse_rows}
    average, total = summary["domain_average"], summary["domain_total"]
    extra_jaccard = next(
        row for row in divergence_rows if row["comparison"] == "extra_20_to_30"
    )
    corr20 = next(
        row
        for row in correlation_rows
        if row["score_mode"] == "domain_total"
        and math.isclose(float(row["progress"]), 0.20, abs_tol=1e-12)
    )
    corr30 = next(
        row
        for row in correlation_rows
        if row["score_mode"] == "domain_total"
        and math.isclose(float(row["progress"]), 0.30, abs_tol=1e-12)
    )
    depleted_total = sum(
        float(row["total_mean_best_remaining_similarity_change"])
        < float(row["average_mean_best_remaining_similarity_change"])
        for row in neighborhood_rows
    )
    supported_categories = [
        str(row["category"])
        for row in root_cause_rows
        if row["supported_directionally"]
    ]
    report = f"""# Task017 high-sparsity collapse diagnosis

All statements below are generated from identity-matched Task014/Task016
artifacts and exact read-only GPU replay. Task017 did not alter or execute a
pruning formula, registry, model forward, validation, or fine-tuning.

## Q1. What differs between A_extra and T_extra?

The incremental-set Jaccard similarity is `{float(extra_jaccard['jaccard']):.6g}`.
The sets contain `{extra_jaccard['common_removed_units']}` common units,
`{extra_jaccard['average_only_removed_units']}` average-only units, and
`{extra_jaccard['total_only_removed_units']}` total-only units. Detailed type,
descriptor, layer, stage and domain evidence is in the CSV artifacts.

## Q2. Is the collapse primarily additional Attention pruning?

The incremental Attention parameter share is
`{float(average['extra_attention_parameter_share']):.4%}` for domain_average and
`{float(total['extra_attention_parameter_share']):.4%}` for domain_total.
This composition is evidence, but Task017 does not infer causality from unit
type counts alone; layer, domain, coverage and substitute evidence must agree.

## Q3. Are removals concentrated in particular layers?

The top-five layer share changes from `{float(average['top5_layer_share']):.4%}`
to `{float(total['top5_layer_share']):.4%}`; layer HHI changes from
`{float(average['layer_removal_hhi']):.6g}` to
`{float(total['layer_removal_hhi']):.6g}`.

## Q4. Are removals concentrated in particular BMS domains?

The top-ten domain share changes from `{float(average['top10_domain_share']):.4%}`
to `{float(total['top10_domain_share']):.4%}`; domain HHI changes from
`{float(average['domain_removal_hhi']):.6g}` to
`{float(total['domain_removal_hhi']):.6g}`.

## Q5. Does total mode over-deplete competition domains?

Median retained ratio at 30% is
`{float(average['median_domain_retained_ratio_30']):.6g}` for average and
`{float(total['median_domain_retained_ratio_30']):.6g}` for total. Domain-level
differences are reported without turning retained-ratio bins into method
hyperparameters.

## Q6. Does coverage deteriorate disproportionately?

Mean domain coverage changes from
`{float(average['mean_domain_coverage_20']):.6g}` to
`{float(average['mean_domain_coverage_30']):.6g}` for average and from
`{float(total['mean_domain_coverage_20']):.6g}` to
`{float(total['mean_domain_coverage_30']):.6g}` for total. The minimum total
coverage at 30% is `{float(total['min_domain_coverage_30']):.6g}`.

## Q7. Do later units have fewer functional substitutes?

Median best remaining similarity in the extra interval is
`{float(average['median_best_substitute_similarity_extra']):.6g}` for average and
`{float(total['median_best_substitute_similarity_extra']):.6g}` for total.
Across `{depleted_total}` domains, the total-mode 20%-to-30% neighborhood change
is more negative than the average-mode change.

## Q8. Do units become attractive dynamically?

`total_extra_dynamic_rank_shift.csv` distinguishes units ranked highly at 20%
from units whose rank rises after partner deletion. This evidence is based on
exact 20/22/24/26/28% replay snapshots, not a static initial ranking.

## Q9. Does Delta_total develop a high-sparsity scaling problem?

Spearman(active demand count, Delta_total) changes from
`{float(corr20['spearman_active_demand_vs_delta_total']):.6g}` at 20% to
`{float(corr30['spearman_active_demand_vs_delta_total']):.6g}` at 30%.
Spearman(retained ratio, Delta_total) changes from
`{float(corr20['spearman_retained_ratio_vs_delta_total']):.6g}` to
`{float(corr30['spearman_retained_ratio_vs_delta_total']):.6g}`. These are
diagnostics only and were never used for selection.

## Q10. Where does behavior begin to diverge?

The saved accuracy evidence provides endpoints at 20% and 30%. Task017 reports
selection-state changes at 22/24/26/28%, but without optional intermediate
model validation it does not claim an exact accuracy-collapse onset.

## Q11. Most strongly supported explanation

The measured Top-1 drop is `{float(average['top1_drop_20_to_30']):.4g}` points for
domain_average and `{float(total['top1_drop_20_to_30']):.4g}` points for
domain_total. The directional evidence classification is `{classification}`;
supported categories are `{', '.join(supported_categories) or 'none'}`.
This classification summarizes measured directions rather than introducing a
new score. Task017 does not state that Attention alone caused the collapse
unless the type, layer, domain, coverage and substitute tables jointly support
that causal interpretation.

## Q12. What should be tested next?

Use these diagnostics to define a separate, explicitly reviewed future task.
Task017 does not implement a new score, quota, penalty, threshold or selector.

## Parameter-count caveat

The x-axis is `target_parameter_budget` / `estimated_parameter_sparsity`.
Task017 does not relabel these values as physical sparsity when saved
`physical_numel_sparsity` is zero. Optional 22/24/26/28% micro-ablation is
`NOT_RUN`; no model validation was executed.
"""
    (output_dir / "diagnosis.md").write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def roots(current: argparse.ArgumentParser) -> None:
        current.add_argument("--task014-root", type=Path, required=True)
        current.add_argument("--task015-root", type=Path, required=True)
        current.add_argument("--task016-root", type=Path, required=True)
        current.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    roots(verify)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--task014-root", type=Path, required=True)
    replay.add_argument("--task016-root", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--mode", choices=MODES, required=True)
    replay.add_argument("--device", default="cuda:0")
    aggregate_parser = subparsers.add_parser("analyze")
    roots(aggregate_parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "verify":
        verify_artifacts(
            args.task014_root,
            args.task015_root,
            args.task016_root,
            args.output_dir,
        )
    elif args.command == "replay":
        HighSparsityReplay(
            task014_root=args.task014_root,
            task016_root=args.task016_root,
            output_dir=args.output_dir,
            mode=args.mode,
            device=args.device,
        ).run()
    elif args.command == "analyze":
        analyze(
            args.task014_root,
            args.task015_root,
            args.task016_root,
            args.output_dir,
        )
    else:
        raise RuntimeError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GPU-first diagnostic replay for Task015 Attention/FFN competition.

This module never creates a pruning registry and never applies or validates a
pruned model.  It reconstructs the unchanged Task014 Dynamic3D/BMS domains,
replays the unchanged marginal-coverage selector, and refuses to publish
diagnostic artifacts unless every selected global index exactly matches the
saved Task014 functional trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Lightweight CSV/statistics tests do not need it.
    torch = None


TARGETS = (0.10, 0.20, 0.30)
SNAPSHOT_PROGRESS = (0.0, 0.05, 0.10, 0.20, 0.30)
SEED = 3407
SIGMA = 0.1
MIN_KEEP_RATIO = 0.1
EXPECTED_UNITS = 36_378
EXPECTED_LAYERS = 48
EXPECTED_DOMAINS = 423
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
EPS = 1e-8
TRACE_LOSS_ABS_TOLERANCE = 1e-7
CHECKPOINT_INTERVAL_STEPS = 500
HEARTBEAT_INTERVAL_SECONDS = 60.0
PROFILE_BUCKETS = (
    "state_update",
    "main_heap",
    "typed_heap",
    "snapshot_collection",
    "trace_verification",
    "csv_json_writing",
)
CANDIDATE_ALL = 0
CANDIDATE_ATTENTION = 1
CANDIDATE_FFN = 2
CANDIDATE_KINDS = 3


@dataclass(frozen=True)
class UnitRecord:
    global_index: int
    layer_index: int
    layer: str
    unit_type: str
    unit_index: int
    cache_key: str


class ReplayInterrupted(RuntimeError):
    """Raised after a consistent Task015 checkpoint has been saved."""


class ReplayProfiler:
    """Low-overhead cumulative timer; detailed buckets are opt-in."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.seconds = {name: 0.0 for name in PROFILE_BUCKETS}

    @contextmanager
    def measure(self, bucket: str):
        if bucket not in self.seconds:
            raise KeyError(bucket)
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[bucket] += time.perf_counter() - started


class ReplayProgress:
    """Rate-limited tqdm display with a deterministic heartbeat fallback."""

    def __init__(self, tag: str, total: int, initial: int = 0):
        self.tag = str(tag)
        self.total = int(total)
        self.current = int(initial)
        self.initial = int(initial)
        self.started = time.perf_counter()
        self.last_heartbeat = self.started
        self.last_postfix = self.started
        self.bar = None
        try:
            from tqdm import tqdm

            self.bar = tqdm(
                total=self.total,
                initial=self.current,
                desc=f"Task015 replay {self.tag}",
                mininterval=1.0,
                maxinterval=2.0,
                dynamic_ncols=True,
                unit="step",
            )
        except ModuleNotFoundError:
            self.bar = None

    def update(
        self,
        *,
        step: int,
        sparsity: float,
        selected_type: str,
        attention_loss: float,
        ffn_loss: float,
    ) -> None:
        step = int(step)
        increment = step - self.current
        self.current = step
        if self.bar is not None:
            now = time.perf_counter()
            if now - self.last_postfix >= 1.0 or step == self.total:
                self.bar.set_postfix(
                    sparsity=f"{sparsity:.2%}",
                    selected="attn" if selected_type == TYPE_ATTENTION else "ffn",
                    attn=f"{attention_loss:.3g}",
                    ffn=f"{ffn_loss:.3g}",
                    refresh=False,
                )
                self.last_postfix = now
            if increment:
                self.bar.update(increment)
            return
        now = time.perf_counter()
        if (
            step % CHECKPOINT_INTERVAL_STEPS
            and now - self.last_heartbeat < HEARTBEAT_INTERVAL_SECONDS
            and step != self.total
        ):
            return
        elapsed = max(now - self.started, 1e-9)
        completed = max(step - self.initial, 1)
        speed = completed / elapsed
        remaining = max(self.total - step, 0) / max(speed, 1e-12)
        print(
            f"[Task015 replay {self.tag}] step={step}/{self.total} "
            f"sparsity={sparsity:.2%} elapsed={_duration(elapsed)} "
            f"ETA={_duration(remaining)} speed={speed:.2f} steps/s "
            f"selected_type={selected_type} "
            f"best_attn_delta={attention_loss:.7g} "
            f"best_ffn_delta={ffn_loss:.7g}",
            flush=True,
        )
        self.last_heartbeat = now

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _git_commit() -> str:
    """Return the checked-out commit without exposing repository credentials."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write an uncompressed NPZ to minimize checkpoint CPU work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_checkpoint_identity(
    saved: Mapping[str, object],
    current: Mapping[str, object],
    checkpoint_path: Path | str,
) -> None:
    """Reject stale replay state instead of silently changing evidence."""
    if dict(saved) != dict(current):
        raise RuntimeError(
            f"Stale Task015 checkpoint rejected: {checkpoint_path}. "
            "Use --no-resume only when intentionally starting over."
        )


def cached_global_best(
    losses: np.ndarray,
    global_indices: np.ndarray,
    local_indices: np.ndarray,
) -> tuple[float, int, int, int] | None:
    """Return ``(loss, global, domain, local)`` with Task014 tie behavior.

    All inputs have shape ``[K]`` (cached domain candidate).  The minimum is
    ordered lexicographically by ``(float32 loss, global_index)`` exactly as
    the Task014 heap, while avoiding per-unit heap entries.
    """
    values = np.asarray(losses, dtype=np.float32)
    globals_ = np.asarray(global_indices, dtype=np.int64)
    locals_ = np.asarray(local_indices, dtype=np.int64)
    if values.ndim != 1 or globals_.shape != values.shape or locals_.shape != values.shape:
        raise ValueError("Cached candidate arrays must have identical shape [K]")
    eligible = np.isfinite(values) & (globals_ >= 0) & (locals_ >= 0)
    if not eligible.any():
        return None
    best_loss = values[eligible].min()
    tied = eligible & (values == best_loss)
    tie_globals = np.where(tied, globals_, np.iinfo(np.int64).max)
    domain_id = int(tie_globals.argmin())
    return (
        float(values[domain_id]),
        int(globals_[domain_id]),
        domain_id,
        int(locals_[domain_id]),
    )


def _require_torch():
    if torch is None:
        raise RuntimeError("Task015 diagnostic replay requires PyTorch")
    return torch


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tag(target: float) -> str:
    return f"s{int(round(100.0 * float(target))):02d}"


def _stage(layer: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(layer))
    if match is None:
        raise ValueError(f"Cannot identify stage from layer {layer!r}")
    return int(match.group(1))


def quantile_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Return the Task015 fixed quantiles for finite one-dimensional values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantile_summary requires a non-empty 1D array")
    if not np.isfinite(array).all():
        raise ValueError("quantile_summary values must be finite")
    quantiles = np.quantile(array, QUANTILES)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def split_candidate_indices(unit_types: Sequence[str]) -> dict[str, np.ndarray]:
    values = np.asarray(unit_types, dtype=object)
    unexpected = sorted(set(values.tolist()) - {TYPE_ATTENTION, TYPE_FFN})
    if unexpected:
        raise ValueError(f"Unexpected pruning-unit types: {unexpected}")
    return {
        TYPE_ATTENTION: np.flatnonzero(values == TYPE_ATTENTION),
        TYPE_FFN: np.flatnonzero(values == TYPE_FFN),
    }


def loss_per_parameter(delta: Sequence[float], cost: Sequence[int]) -> np.ndarray:
    losses = np.asarray(delta, dtype=np.float64)
    costs = np.asarray(cost, dtype=np.float64)
    if losses.shape != costs.shape:
        raise ValueError("delta and cost must have identical shapes")
    if np.any(costs <= 0) or not np.isfinite(losses).all():
        raise ValueError("cost must be positive and delta must be finite")
    return losses / costs


def unnormalized_coverage_loss(
    delta: Sequence[float], active_domain_size: Sequence[int]
) -> np.ndarray:
    losses = np.asarray(delta, dtype=np.float64)
    sizes = np.asarray(active_domain_size, dtype=np.float64)
    if losses.shape != sizes.shape:
        raise ValueError("delta and active_domain_size must have identical shapes")
    if np.any(sizes < 0) or not np.isfinite(losses).all():
        raise ValueError("domain sizes must be non-negative and delta finite")
    return losses * sizes


def identify_mixed_domains(
    domain_ids: Sequence[int], unit_types: Sequence[str]
) -> set[int]:
    domains = np.asarray(domain_ids, dtype=np.int64)
    types = np.asarray(unit_types, dtype=object)
    if domains.shape != types.shape:
        raise ValueError("domain_ids and unit_types must have identical shapes")
    result = set()
    for domain_id in np.unique(domains):
        members = set(types[domains == domain_id].tolist())
        if TYPE_ATTENTION in members and TYPE_FFN in members:
            result.add(int(domain_id))
    return result


def cross_type_similarity_rows(
    similarity: np.ndarray,
    global_indices: Sequence[int],
    unit_types: Sequence[str],
    source_type: str,
) -> list[dict[str, object]]:
    """Extract same/cross maxima from ``A[G,G]`` while excluding self."""
    matrix = np.asarray(similarity, dtype=np.float64)
    indices = np.asarray(global_indices, dtype=np.int64)
    types = np.asarray(unit_types, dtype=object)
    if matrix.shape != (len(indices), len(indices)) or types.shape != indices.shape:
        raise ValueError("similarity, global_indices and unit_types disagree")
    if source_type not in {TYPE_ATTENTION, TYPE_FFN}:
        raise ValueError(f"Unexpected source type: {source_type}")
    cross_type = TYPE_FFN if source_type == TYPE_ATTENTION else TYPE_ATTENTION
    rows = []
    for local in np.flatnonzero(types == source_type):
        same = np.flatnonzero(types == source_type)
        same = same[same != local]
        cross = np.flatnonzero(types == cross_type)

        def best(candidates: np.ndarray) -> tuple[float, int]:
            if candidates.size == 0:
                return 0.0, -1
            values = matrix[local, candidates]
            maximum = float(values.max())
            tied = candidates[values == maximum]
            chosen = int(tied[np.argmin(indices[tied])])
            return maximum, int(indices[chosen])

        same_value, same_index = best(same)
        cross_value, cross_index = best(cross)
        rows.append(
            {
                "global_index": int(indices[local]),
                "best_same_type_similarity": same_value,
                "best_cross_type_similarity": cross_value,
                "best_same_type_global_index": same_index,
                "best_cross_type_global_index": cross_index,
            }
        )
    return rows


def _load_units(mapping_csv: Path) -> list[UnitRecord]:
    rows = _read_csv(mapping_csv)
    units = [
        UnitRecord(
            global_index=int(row["global_index"]),
            layer_index=int(row["layer_index"]),
            layer=row["layer"],
            unit_type=row["unit_type"],
            unit_index=int(row["unit_index"]),
            cache_key=row["cache_key"],
        )
        for row in rows
    ]
    if len(units) != EXPECTED_UNITS:
        raise ValueError(f"Mapped units={len(units)}, expected {EXPECTED_UNITS}")
    if [row.global_index for row in units] != list(range(EXPECTED_UNITS)):
        raise ValueError("Contribution mapping is not contiguous in global index")
    return units


def _mapping_paths(task014_root: Path) -> tuple[Path, Path, Path]:
    candidates = [
        Path(task014_root) / "preflight",
        Path(task014_root) / "functional" / "s10",
    ]
    for directory in candidates:
        unit_path = directory / "contribution_unit_mapping.csv"
        layer_path = directory / "contribution_layer_mapping.csv"
        audit_path = directory / "contribution_mapping_audit.json"
        if unit_path.is_file() and layer_path.is_file() and audit_path.is_file():
            return unit_path, layer_path, audit_path
    raise FileNotFoundError("Task014 mapping evidence is missing")


def _functional_run_dir(task014_root: Path, target: float) -> Path:
    run_dir = Path(task014_root) / "functional" / _tag(target)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir


def _descriptor_path(task014_root: Path, run_dir: Path) -> Path:
    candidates = (
        run_dir / "model_output" / "dynamic3d" / f"seed{SEED}" / "descriptor_statistics.csv",
        run_dir / "dynamic3d" / f"seed{SEED}" / "descriptor_statistics.csv",
        Path(task014_root) / "dynamic3d" / f"seed{SEED}" / "descriptor_statistics.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No saved Task014 Dynamic3D descriptor_statistics.csv was found"
    )


def _load_descriptors(path: Path, units: Sequence[UnitRecord]) -> np.ndarray:
    rows = _read_csv(path)
    if len(rows) != len(units):
        raise ValueError(f"Descriptor rows={len(rows)}, mapped units={len(units)}")
    descriptors = np.empty((len(rows), 3), dtype=np.float32)  # [N, descriptor]
    for index, (row, unit) in enumerate(zip(rows, units)):
        if (
            int(row["global_index"]) != index
            or row["layer"] != unit.layer
            or row["unit_type"] != unit.unit_type
            or int(row["unit_index"]) != unit.unit_index
        ):
            raise ValueError(f"Descriptor/mapping mismatch at global index {index}")
        descriptors[index] = (
            float(row["D_abs"]),
            float(row["D_rel"]),
            float(row["D_dyn"]),
        )
    if not np.isfinite(descriptors).all():
        raise ValueError("Dynamic3D descriptors contain NaN or infinity")
    return descriptors


def rebuild_bms_domains(descriptors: np.ndarray, device: str) -> list[list[int]]:
    """Reproduce unchanged Task014 standardization and BMS on CUDA.

    ``V`` has shape ``[N,3]`` (unit, Dynamic3D dimension).  Each BMS chunk has
    distance/kernel shape ``[Q,N]`` with ``Q<=1024``; no global pair of
    distance and kernel matrices is retained.
    """
    module = _require_torch()
    cuda = module.device(device)
    if cuda.type != "cuda":
        raise ValueError("Task015 BMS replay is GPU-only; use cuda:0")
    values = module.as_tensor(descriptors, dtype=module.float32, device=cuda)
    normalized = (values - values.mean(dim=0, keepdim=True)) / (
        values.std(dim=0, keepdim=True) + EPS
    )  # [N,3]
    positions = normalized.clone()
    count = positions.shape[0]
    for _ in range(60):
        updated = module.empty_like(positions)
        for start in range(0, count, min(1024, count)):
            stop = min(start + min(1024, count), count)
            distances = module.cdist(positions[start:stop], positions)  # [Q,N]
            kernel = module.exp(-(distances ** 2) / (2.0 * SIGMA ** 2))  # [Q,N]
            updated[start:stop] = (kernel @ positions) / (
                kernel.sum(dim=1, keepdim=True) + EPS
            )
            del distances, kernel
        maximum_movement = module.linalg.vector_norm(
            updated - positions, ord=2, dim=1
        ).max()
        positions = updated
        if float(maximum_movement.item()) < 1e-4:
            break
    rounded = module.round(positions * 100.0) / 100.0  # [N,3]
    _, inverse = module.unique(rounded, dim=0, return_inverse=True)
    groups = [
        (inverse == domain_id).nonzero(as_tuple=True)[0].cpu().tolist()
        for domain_id in range(int(inverse.max().item()) + 1)
    ]
    groups = [sorted(int(index) for index in group) for group in groups if group]
    if len(groups) != EXPECTED_DOMAINS:
        raise ValueError(
            f"Replayed BMS domains={len(groups)}, expected {EXPECTED_DOMAINS}"
        )
    if sorted(index for group in groups for index in group) != list(
        range(EXPECTED_UNITS)
    ):
        raise ValueError("Replayed BMS domains are not a 36,378-unit partition")
    return groups


def infer_parameter_costs(
    units: Sequence[UnitRecord], layer_rows: Sequence[Mapping[str, str]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce Task014 ``estimate_unit_cost`` for the fixed Video Swin.

    MLP hidden width is present in the mapping.  For each block, model width
    ``d=hidden/4`` and head width ``d/h`` recover the exact qkv-bias=True model
    costs: FFN ``2d+1`` and Attention ``(4d+3)*(d/h)``.
    """
    counts: dict[str, int] = {}
    for row in layer_rows:
        layer = str(row["layer"])
        count = int(row.get("unit_count", 0))
        if count <= 0:
            raise ValueError(f"Invalid unit count for {layer}")
        counts[layer] = count
    by_layer_cost: dict[str, int] = {}
    for layer, count in counts.items():
        if layer.endswith(".mlp"):
            if count % 4:
                raise ValueError(f"MLP hidden width is not 4*d: {layer} U={count}")
            model_width = count // 4
            by_layer_cost[layer] = 2 * model_width + 1
        elif layer.endswith(".attn"):
            mlp_layer = layer[:-5] + ".mlp"
            if mlp_layer not in counts or counts[mlp_layer] % 4:
                raise ValueError(f"Missing paired MLP width for {layer}")
            model_width = counts[mlp_layer] // 4
            if model_width % count:
                raise ValueError(f"Attention head width is not integral for {layer}")
            head_width = model_width // count
            by_layer_cost[layer] = (4 * model_width + 3) * head_width
        else:
            raise ValueError(f"Unexpected pruning layer: {layer}")
    costs = np.asarray([by_layer_cost[unit.layer] for unit in units], dtype=np.int64)
    if np.any(costs <= 0):
        raise ValueError("Every Task015 unit cost must be positive")
    return costs, by_layer_cost


def _layer_capacities(units: Sequence[UnitRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit.layer] = counts.get(unit.layer, 0) + 1
    return {
        layer: count - max(1, int(count * MIN_KEEP_RATIO))
        for layer, count in counts.items()
    }


def prepare_caches(
    task014_root: Path, contribution_npz: Path, output_dir: Path, device: str
) -> None:
    """Validate/reuse Task014 vectors and build pooled energy ``e[N]`` on GPU."""
    module = _require_torch()
    from functional_competition_pruning import (
        ALIGNED_FIELD_SHAPE,
        EXPECTED_NPZ_ARRAYS,
        EXPECTED_VIDEO_FIELDS,
        LayerWiseContributionFieldArchive,
    )
    import torch.nn.functional as functional

    if module.device(device).type != "cuda":
        raise ValueError("Task015 cache preparation requires a CUDA device")
    unit_path, layer_path, audit_path = _mapping_paths(task014_root)
    units = _load_units(unit_path)
    layer_rows = _read_csv(layer_path)
    layer_names = [row["layer"] for row in layer_rows]
    layer_types = {row["layer"]: row["unit_type"] for row in layer_rows}
    unit_info = [{"layer": unit.layer, "idx": unit.unit_index} for unit in units]
    archive = LayerWiseContributionFieldArchive(contribution_npz)
    audit = archive.audit_descriptor_mapping(
        unit_info,
        layer_names,
        layer_types,
        expected_total_units=EXPECTED_UNITS,
        expected_layer_count=EXPECTED_LAYERS,
        expected_array_count=EXPECTED_NPZ_ARRAYS,
        expected_video_fields=EXPECTED_VIDEO_FIELDS,
    )
    saved_audit = _read_json(audit_path)
    if audit.mapping_sha256 != saved_audit.get("mapping_sha256"):
        raise ValueError("Task015 source mapping does not match Task014 evidence")
    cache_dir = Path(task014_root) / "field_cache"
    vector_path, mask_path = archive.prepare_vector_memmap(
        audit, cache_dir, module.device(device)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    energy_path = output_dir / "functional_energy.npy"
    identity_path = output_dir / "functional_energy.json"
    source_stat = Path(contribution_npz).stat()
    identity = {
        "format_version": "task015_pooled_signed_energy_v1",
        "source_npz": str(Path(contribution_npz).resolve()),
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "mapping_sha256": audit.mapping_sha256,
        "shape": [EXPECTED_UNITS],
        "dtype": "float32",
        "definition": "L2 of signed pooled concatenated [9*16*7*7] vector",
    }
    if energy_path.is_file() and identity_path.is_file():
        saved = _read_json(identity_path)
        cached = np.load(energy_path, mmap_mode="r", allow_pickle=False)
        compatible = (
            saved == identity
            and cached.shape == (EXPECTED_UNITS,)
            and cached.dtype == np.float32
            and np.isfinite(cached).all()
        )
        del cached
        if compatible:
            _atomic_json(
                output_dir / "cache_preparation.json",
                {
                    "status": "passed",
                    "vector_cache": str(vector_path),
                    "valid_mask_cache": str(mask_path),
                    "energy_cache": str(energy_path),
                    "energy_cache_reused": True,
                    "device": str(device),
                },
            )
            return

    temporary = energy_path.with_suffix(".npy.tmp")
    if temporary.exists():
        temporary.unlink()
    energy = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(EXPECTED_UNITS,)
    )
    try:
        for spec in audit.layers:
            source = archive._load_layer_array(spec)  # [S,U,T,H,W], one layer
            bytes_per_unit = int(np.prod(source.shape[:1] + source.shape[2:])) * 4
            chunk_units = max(1, min(256, (384 * 1024 * 1024) // bytes_per_unit))
            for start in range(0, spec.unit_count, chunk_units):
                stop = min(start + chunk_units, spec.unit_count)
                cpu = module.from_numpy(
                    np.ascontiguousarray(source[:, start:stop], dtype=np.float32)
                )  # [S,B,T,H,W]
                fields = cpu.permute(1, 0, 2, 3, 4).contiguous().to(
                    device=device, dtype=module.float32, non_blocking=True
                )  # [B,S,T,H,W]
                pooled = functional.adaptive_avg_pool3d(
                    fields.reshape(
                        (stop - start) * fields.shape[1],
                        1,
                        fields.shape[2],
                        fields.shape[3],
                        fields.shape[4],
                    ),
                    ALIGNED_FIELD_SHAPE,
                ).reshape(stop - start, fields.shape[1], *ALIGNED_FIELD_SHAPE)
                norms = module.linalg.vector_norm(
                    pooled.reshape(stop - start, -1), ord=2, dim=1
                )  # [B]
                energy[spec.global_start + start:spec.global_start + stop] = (
                    norms.detach().cpu().numpy()
                )
                del cpu, fields, pooled, norms
            del source
            energy.flush()
    except Exception:
        del energy
        if temporary.exists():
            temporary.unlink()
        raise
    del energy
    temporary.replace(energy_path)
    _atomic_json(identity_path, identity)
    _atomic_json(
        output_dir / "cache_preparation.json",
        {
            "status": "passed",
            "vector_cache": str(vector_path),
            "valid_mask_cache": str(mask_path),
            "energy_cache": str(energy_path),
            "energy_cache_reused": False,
            "device": str(device),
        },
    )


class DomainCandidateIndex:
    """GPU-backed per-domain candidate cache for exact Task014 ordering.

    For ``K`` BMS domains, ``loss/global/local`` each have shape ``[3,K]``.
    Axis 0 is all units, Attention, and FFN respectively.  Each domain keeps
    its ``losses[G]`` and masks on CUDA; a refresh transfers only seven scalar
    values (three losses, three local indices, one coverage) to CPU.
    """

    def __init__(
        self,
        *,
        states,
        global_indices: Sequence[np.ndarray],
        layer_ids,
        attention_masks,
        ffn_masks,
        layer_feasible,
    ):
        self.states = list(states)
        self.global_indices = [
            np.asarray(values, dtype=np.int64) for values in global_indices
        ]
        self.layer_ids = list(layer_ids)
        self.attention_masks = list(attention_masks)
        self.ffn_masks = list(ffn_masks)
        self.layer_feasible = layer_feasible
        domains = len(self.states)
        if not (
            len(self.global_indices)
            == len(self.layer_ids)
            == len(self.attention_masks)
            == len(self.ffn_masks)
            == domains
        ):
            raise ValueError("Per-domain candidate metadata length mismatch")
        self.loss = np.full((CANDIDATE_KINDS, domains), np.inf, dtype=np.float32)
        self.global_index = np.full(
            (CANDIDATE_KINDS, domains), -1, dtype=np.int64
        )
        self.local_index = np.full(
            (CANDIDATE_KINDS, domains), -1, dtype=np.int64
        )

    def refresh(self, domain_id: int, coverage=None) -> None:
        module = _require_torch()
        state = self.states[domain_id]
        feasible = self.layer_feasible.index_select(
            0, self.layer_ids[domain_id]
        )  # Bool[G]
        eligible = state.retained & feasible  # Bool[G]
        masks = module.stack(
            (
                eligible,
                eligible & self.attention_masks[domain_id],
                eligible & self.ffn_masks[domain_id],
            ),
            dim=0,
        )  # Bool[3,G]
        candidate_losses = state.losses.unsqueeze(0).expand(
            CANDIDATE_KINDS, -1
        ).masked_fill(~masks, module.inf)  # [3,G]
        values, locals_ = candidate_losses.min(dim=1)  # [3], [3]
        if coverage is None:
            coverage = candidate_losses.new_tensor(float(state.current_coverage))
        packet = module.cat(
            (values.to(module.float32), locals_.to(module.float32), coverage.reshape(1))
        ).detach().cpu().numpy()  # float32[7], one CUDA synchronization
        values_cpu = packet[:CANDIDATE_KINDS].astype(np.float32, copy=False)
        locals_cpu = packet[CANDIDATE_KINDS:2 * CANDIDATE_KINDS].astype(
            np.int64, copy=False
        )
        globals_cpu = np.full(CANDIDATE_KINDS, -1, dtype=np.int64)
        for kind in range(CANDIDATE_KINDS):
            if np.isfinite(values_cpu[kind]):
                globals_cpu[kind] = self.global_indices[domain_id][locals_cpu[kind]]
            else:
                locals_cpu[kind] = -1
        self.loss[:, domain_id] = values_cpu
        self.global_index[:, domain_id] = globals_cpu
        self.local_index[:, domain_id] = locals_cpu
        state.current_coverage = float(packet[-1])

    def refresh_all(self) -> None:
        for domain_id in range(len(self.states)):
            self.refresh(domain_id)

    def best(self, kind: int) -> tuple[float, int, int, int] | None:
        if kind < 0 or kind >= CANDIDATE_KINDS:
            raise IndexError(kind)
        return cached_global_best(
            self.loss[kind], self.global_index[kind], self.local_index[kind]
        )


class DiagnosticReplay:
    """Exact Task014 selector replay plus read-only candidate observation."""

    def __init__(
        self,
        *,
        task014_root: Path,
        output_dir: Path,
        target: float,
        device: str,
        resume: bool = True,
        keep_checkpoint: bool = False,
        profile: bool = False,
    ):
        self.constructed_at = time.perf_counter()
        module = _require_torch()
        if module.device(device).type != "cuda":
            raise ValueError("Task015 replay requires cuda:0")
        self.task014_root = Path(task014_root)
        self.output_dir = Path(output_dir)
        self.target = float(target)
        if self.target not in TARGETS:
            raise ValueError(f"Unsupported Task015 target: {self.target}")
        self.device = module.device(device)
        self.resume = bool(resume)
        self.keep_checkpoint = bool(keep_checkpoint)
        self.profiler = ReplayProfiler(profile)
        self.run_dir = _functional_run_dir(self.task014_root, self.target)
        unit_path, layer_path, _ = _mapping_paths(self.task014_root)
        self.units = _load_units(unit_path)
        self.unit_by_global = {unit.global_index: unit for unit in self.units}
        self.layer_rows = _read_csv(layer_path)
        self.descriptor_path = _descriptor_path(self.task014_root, self.run_dir)
        self.descriptors = _load_descriptors(self.descriptor_path, self.units)
        self.groups = rebuild_bms_domains(self.descriptors, str(self.device))
        self.costs, self.layer_costs = infer_parameter_costs(
            self.units, self.layer_rows
        )
        self.capacities = _layer_capacities(self.units)
        self.metrics = _read_json(self.run_dir / "final_metrics.json")
        self.expected_trace = _read_csv(
            self.run_dir / "functional_selection_trace.csv"
        )
        selection_summary = _read_json(
            self.run_dir / "functional_selection_summary.json"
        )
        self.total_parameters = int(self.metrics["parameters_before"])
        self.target_budget = float(selection_summary["target_parameter_budget"])
        expected_budget = self.total_parameters * self.target
        if not math.isclose(
            self.target_budget, expected_budget, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("Task014 target budget evidence is inconsistent")
        self.vector_path = self.task014_root / "field_cache" / "aligned_function_fields.npy"
        self.mask_path = self.task014_root / "field_cache" / "aligned_function_valid_mask.npy"
        self.energy_path = self.output_dir / "functional_energy.npy"
        for path in (self.vector_path, self.mask_path, self.energy_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.states = []
        self.layer_names = sorted(self.capacities)
        self.layer_to_id = {
            layer: index for index, layer in enumerate(self.layer_names)
        }
        if len(self.layer_names) != EXPECTED_LAYERS:
            raise ValueError(
                f"Pruning layers={len(self.layer_names)}, expected {EXPECTED_LAYERS}"
            )
        self.capacity_array = np.asarray(
            [self.capacities[layer] for layer in self.layer_names], dtype=np.int64
        )  # [L]
        self.global_layer_id = np.asarray(
            [self.layer_to_id[unit.layer] for unit in self.units], dtype=np.int64
        )  # [N]
        self.global_to_domain = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        self.global_to_local = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        self.domain_globals: list[np.ndarray] = []
        self.domain_layer_ids = []
        self.domain_attention_masks = []
        self.domain_ffn_masks = []
        self.layer_domains: list[set[int]] = [set() for _ in self.layer_names]
        for domain_id, members in enumerate(self.groups):
            globals_ = np.asarray(members, dtype=np.int64)
            locals_ = np.arange(len(members), dtype=np.int64)
            self.global_to_domain[globals_] = domain_id
            self.global_to_local[globals_] = locals_
            layer_ids = self.global_layer_id[globals_]
            types = np.asarray(
                [self.units[int(index)].unit_type for index in globals_], dtype=object
            )
            self.domain_globals.append(globals_)
            self.domain_layer_ids.append(
                module.as_tensor(layer_ids, dtype=module.long, device=self.device)
            )
            self.domain_attention_masks.append(
                module.as_tensor(
                    types == TYPE_ATTENTION, dtype=module.bool, device=self.device
                )
            )
            self.domain_ffn_masks.append(
                module.as_tensor(types == TYPE_FFN, dtype=module.bool, device=self.device)
            )
            for layer_id in np.unique(layer_ids):
                self.layer_domains[int(layer_id)].add(domain_id)
        if np.any(self.global_to_domain < 0) or np.any(self.global_to_local < 0):
            raise ValueError("BMS domain lookup does not cover all mapped units")
        self.layer_feasible = module.as_tensor(
            self.capacity_array > 0, dtype=module.bool, device=self.device
        )  # Bool[L]
        self.candidates: DomainCandidateIndex | None = None
        self.domain_retained_counts = np.asarray(
            [len(group) for group in self.groups], dtype=np.int64
        )  # [K]
        self.retained_cpu = np.ones(EXPECTED_UNITS, dtype=np.bool_)  # Bool[N]
        self._interrupt_requested = False

    def _initialize_states(self) -> None:
        from functional_competition_pruning import (
            DomainState,
            build_functional_similarity,
        )

        vectors = np.load(self.vector_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)
        if vectors.shape[0] != EXPECTED_UNITS or valid.shape != (EXPECTED_UNITS,):
            raise ValueError("Task014 vector cache has an unexpected shape")
        states = []
        for domain_id, members in enumerate(self.groups):
            cpu_vectors = torch.from_numpy(
                np.asarray(vectors[members], dtype=np.float32).copy()
            )  # [G,9*16*7*7]
            cpu_valid = torch.from_numpy(
                np.asarray(valid[members], dtype=np.bool_).copy()
            )  # Bool[G]
            domain_vectors = cpu_vectors.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            domain_valid = cpu_valid.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            similarity = build_functional_similarity(
                domain_vectors, domain_valid
            )  # [G,G]
            states.append(DomainState(domain_id, members, similarity, domain_valid))
            del cpu_vectors, cpu_valid, domain_vectors, domain_valid
        self.states = states
        del vectors, valid

    def _validate_cost_evidence(self) -> None:
        for row in self.expected_trace:
            index = int(row["global_index"])
            if int(row["parameter_cost"]) != int(self.costs[index]):
                raise ValueError(f"Parameter cost mismatch at global index {index}")
        for mode in ("bms", "functional"):
            for target in TARGETS:
                layer_path = (
                    self.task014_root / mode / _tag(target) / "layer_pruning.csv"
                )
                if not layer_path.is_file():
                    continue
                for row in _read_csv(layer_path):
                    removed = int(row["removed_units"])
                    if removed:
                        observed = int(row["removed_parameter_cost"]) // removed
                        expected = self.layer_costs[row["layer"]]
                        if observed != expected:
                            raise ValueError(
                                f"Layer cost mismatch for {row['layer']}: "
                                f"{observed} vs {expected}"
                            )

    def _legacy_type_best(
        self, state, unit_type: str, feasible
    ) -> tuple[float, int, int] | None:
        local = [
            position
            for position, global_index in enumerate(state.global_indices)
            if bool(state.retained[position].item())
            and self.unit_by_global[global_index].unit_type == unit_type
            and feasible(global_index)
        ]
        if not local:
            return None
        local_tensor = torch.tensor(local, dtype=torch.long, device=self.device)
        losses = state.losses.index_select(0, local_tensor)
        value, position = losses.min(dim=0)
        chosen_local = local[int(position.item())]
        return (
            float(value.item()),
            state.global_indices[chosen_local],
            chosen_local,
        )

    def _candidate_arrays(self) -> dict[str, np.ndarray]:
        """Collect a full feasible snapshot with one GPU transfer per domain."""
        globals_: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        domain_sizes: list[np.ndarray] = []
        best_similarities: list[np.ndarray] = []
        for domain_id, state in enumerate(self.states):
            feasible = self.layer_feasible.index_select(
                0, self.domain_layer_ids[domain_id]
            )  # Bool[G]
            tensor = (state.retained & feasible).nonzero(as_tuple=True)[0]  # [F_k]
            if tensor.numel() == 0:
                continue
            losses = state.losses.index_select(0, tensor)
            if not torch.isfinite(losses).all():
                raise ValueError("Feasible candidate has non-finite marginal loss")
            similarities = state.similarity.index_select(0, tensor).clone()
            similarities[
                torch.arange(tensor.numel(), device=self.device), tensor
            ] = 0.0
            best = similarities.max(dim=1).values
            packet = torch.stack((losses, best), dim=0).detach().cpu().numpy()
            local = tensor.detach().cpu().numpy().astype(np.int64, copy=False)
            globals_.append(self.domain_globals[domain_id][local])
            deltas.append(packet[0].astype(np.float64))
            active_size = int(state.valid_function_mask.sum().item())
            domain_sizes.append(np.full(local.size, active_size, dtype=np.int64))
            best_similarities.append(packet[1].astype(np.float64))
            del tensor, losses, similarities, best
        global_indices = np.concatenate(globals_)
        delta = np.concatenate(deltas)
        active_domain_size = np.concatenate(domain_sizes)
        best_similarity = np.concatenate(best_similarities)
        types = np.asarray(
            [self.unit_by_global[int(index)].unit_type for index in global_indices],
            dtype="U16",
        )
        costs = self.costs[global_indices]
        energy = np.load(self.energy_path, mmap_mode="r", allow_pickle=False)[
            global_indices
        ].astype(np.float64)
        result = {
            "global_index": global_indices,
            "unit_type": types,
            "delta": delta,
            "cost": costs,
            "rho": loss_per_parameter(delta, costs),
            "active_domain_size": active_domain_size,
            "unnormalized_loss": unnormalized_coverage_loss(
                delta, active_domain_size
            ),
            "best_similarity": best_similarity,
            "functional_energy": energy,
        }
        result.update(
            {
                "D_abs": self.descriptors[global_indices, 0].astype(np.float64),
                "D_rel": self.descriptors[global_indices, 1].astype(np.float64),
                "D_dyn": self.descriptors[global_indices, 2].astype(np.float64),
                "domain_id": self.global_to_domain[global_indices].astype(np.int64),
            }
        )
        return result

    def _static_domain_artifacts(self, worker_dir: Path) -> None:
        composition = []
        attention_similarity = []
        ffn_similarity = []
        mixed_losses = []
        for state in self.states:
            records = [self.unit_by_global[index] for index in state.global_indices]
            types = [record.unit_type for record in records]
            attention_count = types.count(TYPE_ATTENTION)
            ffn_count = types.count(TYPE_FFN)
            mixed = bool(attention_count and ffn_count)
            null_count = int((~state.valid_function_mask).sum().item())
            composition.append(
                {
                    "domain_id": state.domain_id,
                    "domain_size": state.initial_size,
                    "attention_count": attention_count,
                    "ffn_count": ffn_count,
                    "attention_fraction": attention_count / state.initial_size,
                    "ffn_fraction": ffn_count / state.initial_size,
                    "is_attention_only": bool(attention_count and not ffn_count),
                    "is_ffn_only": bool(ffn_count and not attention_count),
                    "is_mixed": mixed,
                    "num_layers": len({record.layer for record in records}),
                    "num_stages": len({_stage(record.layer) for record in records}),
                    "initial_active_count": state.initial_size - null_count,
                    "null_count": null_count,
                }
            )
            if not mixed:
                continue
            matrix = state.similarity.detach().cpu().numpy()
            globals_ = [record.global_index for record in records]
            for source_type, target in (
                (TYPE_ATTENTION, attention_similarity),
                (TYPE_FFN, ffn_similarity),
            ):
                for row in cross_type_similarity_rows(
                    matrix, globals_, types, source_type
                ):
                    record = self.unit_by_global[int(row["global_index"])]
                    target.append(
                        {
                            "domain_id": state.domain_id,
                            "global_index": row["global_index"],
                            "layer": record.layer,
                            **row,
                        }
                    )
            local_attention = np.flatnonzero(np.asarray(types) == TYPE_ATTENTION)
            local_ffn = np.flatnonzero(np.asarray(types) == TYPE_FFN)
            losses = state.losses.detach().cpu().numpy()
            mixed_losses.append(
                {
                    "domain_id": state.domain_id,
                    "attention_count": len(local_attention),
                    "ffn_count": len(local_ffn),
                    "median_delta_attention": float(np.median(losses[local_attention])),
                    "median_delta_ffn": float(np.median(losses[local_ffn])),
                    "min_delta_attention": float(losses[local_attention].min()),
                    "min_delta_ffn": float(losses[local_ffn].min()),
                }
            )
        _atomic_csv(worker_dir / "domain_composition.csv", tuple(composition[0]), composition)
        _atomic_csv(
            worker_dir / "mixed_attention_similarity.csv",
            (
                "domain_id", "global_index", "layer",
                "best_same_type_similarity", "best_cross_type_similarity",
                "best_same_type_global_index", "best_cross_type_global_index",
            ),
            attention_similarity,
        )
        _atomic_csv(
            worker_dir / "mixed_ffn_similarity.csv",
            (
                "domain_id", "global_index", "layer",
                "best_same_type_similarity", "best_cross_type_similarity",
                "best_same_type_global_index", "best_cross_type_global_index",
            ),
            ffn_similarity,
        )
        _atomic_csv(
            worker_dir / "mixed_domain_losses.csv",
            (
                "domain_id", "attention_count", "ffn_count",
                "median_delta_attention", "median_delta_ffn",
                "min_delta_attention", "min_delta_ffn",
            ),
            mixed_losses,
        )

    def _initialize_candidate_index(self) -> None:
        self.candidates = DomainCandidateIndex(
            states=self.states,
            global_indices=self.domain_globals,
            layer_ids=self.domain_layer_ids,
            attention_masks=self.domain_attention_masks,
            ffn_masks=self.domain_ffn_masks,
            layer_feasible=self.layer_feasible,
        )
        self.candidates.refresh_all()

    def _remove_selected(
        self, domain_id: int, local_index: int, selected_loss: float
    ):
        """Apply one known trace deletion and update only ``losses[G]``."""
        from functional_competition_pruning import marginal_coverage_losses

        state = self.states[domain_id]
        if not self.retained_cpu[state.global_indices[local_index]]:
            raise ValueError("Trace candidate has already been removed")
        state.retained[local_index] = False
        state.version += 1
        state.removed_losses.append(float(selected_loss))
        self.domain_retained_counts[domain_id] -= 1
        self.retained_cpu[state.global_indices[local_index]] = False
        if self.domain_retained_counts[domain_id] > 0:
            state.losses, coverage = marginal_coverage_losses(
                state.similarity, state.retained, state.valid_function_mask
            )
        else:
            state.losses = torch.full_like(state.losses, torch.inf)
            coverage = state.similarity.new_tensor(
                1.0 if not bool(state.valid_function_mask.any().item()) else 0.0
            )
        return coverage

    def _refresh_after_removal(
        self,
        *,
        domain_id: int,
        layer_id: int,
        pruned_by_layer: np.ndarray,
        coverage,
    ) -> None:
        if self.candidates is None:
            raise RuntimeError("Candidate cache is not initialized")
        reached_capacity = (
            pruned_by_layer[layer_id] == self.capacity_array[layer_id]
        )
        if reached_capacity:
            self.layer_feasible[layer_id] = False
            affected = sorted(self.layer_domains[layer_id])
        else:
            affected = [domain_id]
        for current_domain in affected:
            self.candidates.refresh(
                current_domain,
                coverage if current_domain == domain_id else None,
            )

    def _checkpoint_paths(self) -> tuple[Path, Path]:
        base = self.output_dir / "replay_checkpoints" / f"{_tag(self.target)}_checkpoint"
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def _checkpoint_identity(self) -> dict[str, object]:
        unit_path, _, audit_path = _mapping_paths(self.task014_root)
        energy_identity = _read_json(self.output_dir / "functional_energy.json")
        vector_stat = self.vector_path.stat()
        mask_stat = self.mask_path.stat()
        return {
            "format_version": "task015_exact_trace_checkpoint_v2",
            "task014_root": str(self.task014_root.resolve()),
            "target_sparsity": self.target,
            "trace_sha256": _sha256_file(
                self.run_dir / "functional_selection_trace.csv"
            ),
            "selection_summary_sha256": _sha256_file(
                self.run_dir / "functional_selection_summary.json"
            ),
            "final_metrics_sha256": _sha256_file(
                self.run_dir / "final_metrics.json"
            ),
            "descriptor_sha256": _sha256_file(self.descriptor_path),
            "mapping_csv_sha256": _sha256_file(unit_path),
            "mapping_evidence_sha256": _sha256_file(audit_path),
            "bms_membership_sha256": hashlib.sha256(
                json.dumps(self.groups, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "contribution_field_cache_identity": energy_identity,
            "normalized_vector_cache_identity": {
                "path": str(self.vector_path.resolve()),
                "size_bytes": int(vector_stat.st_size),
                "mtime_ns": int(vector_stat.st_mtime_ns),
            },
            "valid_mask_cache_identity": {
                "path": str(self.mask_path.resolve()),
                "size_bytes": int(mask_stat.st_size),
                "mtime_ns": int(mask_stat.st_mtime_ns),
            },
            "sigma": SIGMA,
            "seed": SEED,
            "min_keep_ratio": MIN_KEEP_RATIO,
            "task015_git_commit": _git_commit(),
            "task015_source_sha256": _sha256_file(Path(__file__)),
        }

    def _save_checkpoint(
        self,
        *,
        step: int,
        removed_cost: int,
        pruned_by_layer: np.ndarray,
        replay_rows: list[dict[str, object]],
        snapshot_payload: dict[str, list[np.ndarray]],
        next_snapshot: int,
        worker_tmp: Path,
    ) -> None:
        if self.candidates is None:
            raise RuntimeError("Candidate cache is not initialized")
        npz_path, json_path = self._checkpoint_paths()
        generation = str(time.time_ns())
        retained = torch.cat(
            [state.retained for state in self.states], dim=0
        ).detach().cpu().numpy().astype(np.bool_, copy=False)  # Bool[N], domain order
        arrays: dict[str, np.ndarray] = {
            "generation": np.asarray(generation),
            "retained_by_domain_order": retained,
            "domain_versions": np.asarray(
                [state.version for state in self.states], dtype=np.int64
            ),
            "domain_best_loss": self.candidates.loss.copy(),
            "domain_best_global": self.candidates.global_index.copy(),
            "domain_best_local": self.candidates.local_index.copy(),
            "pruned_by_layer": np.asarray(pruned_by_layer, dtype=np.int64),
        }
        for key, value in snapshot_payload.items():
            arrays[f"snapshot__{key}"] = np.asarray(value[0])
        _atomic_npz(npz_path, arrays)
        _atomic_json(
            json_path,
            {
                "status": "incomplete",
                "generation": generation,
                "identity": self._checkpoint_identity(),
                "step": int(step),
                "removed_cost": int(removed_cost),
                "pruned_by_layer": {
                    layer: int(pruned_by_layer[index])
                    for index, layer in enumerate(self.layer_names)
                },
                "replay_rows": replay_rows,
                "next_snapshot": int(next_snapshot),
                "snapshot_keys": sorted(snapshot_payload),
                "worker_tmp": str(worker_tmp),
                "profile_seconds": self.profiler.seconds,
            },
        )

    def _load_checkpoint(self) -> dict[str, object] | None:
        npz_path, json_path = self._checkpoint_paths()
        if not npz_path.exists() and not json_path.exists():
            return None
        if not npz_path.is_file() or not json_path.is_file():
            raise RuntimeError("Task015 checkpoint pair is incomplete")
        metadata = _read_json(json_path)
        validate_checkpoint_identity(
            metadata.get("identity", {}), self._checkpoint_identity(), json_path
        )
        with np.load(npz_path, allow_pickle=False) as payload:
            generation = str(payload["generation"].item())
            if generation != metadata.get("generation"):
                raise RuntimeError("Task015 checkpoint JSON/NPZ generation mismatch")
            retained = payload["retained_by_domain_order"].astype(np.bool_)
            versions = payload["domain_versions"].astype(np.int64)
            saved_loss = payload["domain_best_loss"].astype(np.float32)
            saved_global = payload["domain_best_global"].astype(np.int64)
            saved_local = payload["domain_best_local"].astype(np.int64)
            saved_pruned = payload["pruned_by_layer"].astype(np.int64)
            snapshots = {
                key[len("snapshot__"):]: [payload[key].copy()]
                for key in payload.files
                if key.startswith("snapshot__")
            }
        if retained.shape != (EXPECTED_UNITS,) or versions.shape != (
            len(self.states),
        ):
            raise RuntimeError("Task015 checkpoint retained/version shape mismatch")
        if saved_pruned.shape != (len(self.layer_names),):
            raise RuntimeError("Task015 checkpoint layer-count shape mismatch")
        expected_candidate_shape = (CANDIDATE_KINDS, len(self.states))
        if (
            saved_loss.shape != expected_candidate_shape
            or saved_global.shape != expected_candidate_shape
            or saved_local.shape != expected_candidate_shape
        ):
            raise RuntimeError("Task015 checkpoint candidate-cache shape mismatch")
        if sorted(snapshots) != sorted(metadata.get("snapshot_keys", [])):
            raise RuntimeError("Task015 checkpoint snapshot-state mismatch")
        step = int(metadata["step"])
        replay_rows = list(metadata.get("replay_rows", []))
        if step != len(replay_rows) or step > len(self.expected_trace):
            raise RuntimeError("Task015 checkpoint replay-row count mismatch")
        expected_prefix = [
            int(row["global_index"]) for row in self.expected_trace[:step]
        ]
        observed_prefix = [
            int(row["selected_global_index"]) for row in replay_rows
        ]
        if observed_prefix != expected_prefix:
            raise RuntimeError("Task015 checkpoint trace prefix mismatch")
        expected_versions = np.zeros(len(self.states), dtype=np.int64)
        for row in self.expected_trace[:step]:
            expected_versions[int(row["domain_id"])] += 1
        if not np.array_equal(versions, expected_versions):
            raise RuntimeError("Task015 checkpoint domain versions disagree with trace")

        offset = 0
        self.retained_cpu.fill(True)
        for domain_id, state in enumerate(self.states):
            stop = offset + state.initial_size
            domain_retained = retained[offset:stop]
            state.retained.copy_(
                torch.as_tensor(
                    domain_retained, dtype=torch.bool, device=self.device
                )
            )
            state.version = int(versions[domain_id])
            state.removed_losses = [
                float(row["marginal_functional_loss"])
                for row in self.expected_trace[:step]
                if int(row["domain_id"]) == domain_id
            ]
            globals_ = self.domain_globals[domain_id]
            self.retained_cpu[globals_] = domain_retained
            self.domain_retained_counts[domain_id] = int(domain_retained.sum())
            if self.domain_retained_counts[domain_id]:
                from functional_competition_pruning import marginal_coverage_losses

                state.losses, coverage = marginal_coverage_losses(
                    state.similarity, state.retained, state.valid_function_mask
                )
                state.current_coverage = float(coverage.item())
            else:
                state.losses = torch.full_like(state.losses, torch.inf)
                state.current_coverage = (
                    1.0 if not bool(state.valid_function_mask.any().item()) else 0.0
                )
            offset = stop
        if offset != EXPECTED_UNITS:
            raise RuntimeError("Task015 checkpoint domain-order size mismatch")
        expected_pruned = np.zeros_like(saved_pruned)
        for index in expected_prefix:
            expected_pruned[self.global_layer_id[index]] += 1
        if not np.array_equal(saved_pruned, expected_pruned):
            raise RuntimeError("Task015 checkpoint layer counts disagree with trace")
        json_pruned = metadata.get("pruned_by_layer", {})
        if json_pruned != {
            layer: int(saved_pruned[index])
            for index, layer in enumerate(self.layer_names)
        }:
            raise RuntimeError("Task015 checkpoint JSON/NPZ layer counts disagree")
        expected_cost = int(self.costs[expected_prefix].sum()) if expected_prefix else 0
        if int(metadata["removed_cost"]) != expected_cost:
            raise RuntimeError("Task015 checkpoint parameter cost disagrees with trace")
        feasible = saved_pruned < self.capacity_array
        self.layer_feasible.copy_(
            torch.as_tensor(feasible, dtype=torch.bool, device=self.device)
        )
        self._initialize_candidate_index()
        if self.candidates is None:
            raise RuntimeError("Candidate cache initialization failed")
        if not (
            np.array_equal(self.candidates.loss, saved_loss)
            and np.array_equal(self.candidates.global_index, saved_global)
            and np.array_equal(self.candidates.local_index, saved_local)
        ):
            raise RuntimeError("Task015 checkpoint candidate cache is not reproducible")
        return {
            "step": step,
            "removed_cost": expected_cost,
            "pruned_by_layer": saved_pruned,
            "replay_rows": replay_rows,
            "snapshot_payload": snapshots,
            "next_snapshot": int(metadata["next_snapshot"]),
        }

    def _delete_checkpoint(self) -> None:
        for path in self._checkpoint_paths():
            if path.exists():
                path.unlink()

    def _write_mismatch(
        self,
        worker_tmp: Path,
        *,
        step: int,
        expected: Mapping[str, str],
        actual: Mapping[str, object],
        domain_id: int,
    ) -> None:
        state = self.states[domain_id]
        retained_local = state.retained.nonzero(as_tuple=True)[0]
        losses = state.losses.index_select(0, retained_local)
        local_cpu = retained_local.detach().cpu().numpy().astype(np.int64)
        loss_cpu = losses.detach().cpu().numpy().astype(np.float64)
        _atomic_json(
            worker_tmp / "replay_mismatch.json",
            {
                "status": "failed",
                "step": int(step),
                "expected": dict(expected),
                "actual": dict(actual),
                "domain_state": {
                    "domain_id": int(domain_id),
                    "version": int(state.version),
                    "retained_count": int(local_cpu.size),
                    "retained_global_indices": self.domain_globals[domain_id][
                        local_cpu
                    ].tolist(),
                    "retained_marginal_losses": loss_cpu.tolist(),
                },
            },
        )

    def run(self) -> None:
        module = _require_torch()
        wall_started = self.constructed_at
        module.cuda.reset_peak_memory_stats(self.device)
        self._validate_cost_evidence()
        self._initialize_states()
        worker_final = self.output_dir / "replay" / _tag(self.target)
        worker_tmp = worker_final.with_name(worker_final.name + ".tmp")
        checkpoint = self._load_checkpoint() if self.resume else None
        if checkpoint is None:
            if worker_tmp.exists():
                shutil.rmtree(worker_tmp)
            worker_tmp.mkdir(parents=True, exist_ok=False)
            pruned_by_layer = np.zeros(len(self.layer_names), dtype=np.int64)
            removed_cost = 0
            replay_rows: list[dict[str, object]] = []
            snapshot_payload: dict[str, list[np.ndarray]] = {}
            next_snapshot = 0
            self._initialize_candidate_index()
            checkpoint_resumed = False
        else:
            worker_tmp.mkdir(parents=True, exist_ok=True)
            pruned_by_layer = checkpoint["pruned_by_layer"]
            removed_cost = int(checkpoint["removed_cost"])
            replay_rows = checkpoint["replay_rows"]
            snapshot_payload = checkpoint["snapshot_payload"]
            next_snapshot = int(checkpoint["next_snapshot"])
            checkpoint_resumed = True
            print(
                f"[Task015] resuming {_tag(self.target)} from step "
                f"{len(replay_rows)}",
                flush=True,
            )
        if self.candidates is None:
            raise RuntimeError("Candidate cache is not initialized")

        if math.isclose(self.target, 0.30, rel_tol=0.0, abs_tol=1e-12) and not (
            worker_tmp / "domain_composition.csv"
        ).is_file():
            self._static_domain_artifacts(worker_tmp)
        thresholds = [value for value in SNAPSHOT_PROGRESS if value <= self.target]
        detailed_seconds = {
            "domain_update_seconds": 0.0,
            "global_candidate_seconds": 0.0,
            "type_candidate_seconds": 0.0,
            "snapshot_collection_seconds": 0.0,
            "output_write_seconds": 0.0,
            "checkpoint_write_seconds": 0.0,
        }

        def record_snapshots() -> None:
            nonlocal next_snapshot
            sparsity = removed_cost / self.total_parameters
            while next_snapshot < len(thresholds) and (
                sparsity + 1e-15 >= thresholds[next_snapshot]
            ):
                started = time.perf_counter()
                with self.profiler.measure("snapshot_collection"):
                    label = thresholds[next_snapshot]
                    arrays = self._candidate_arrays()
                    prefix = f"p{int(round(label * 100)):02d}_"
                    for name, value in arrays.items():
                        snapshot_payload[prefix + name] = [value]
                    snapshot_payload[prefix + "estimated_sparsity"] = [
                        np.asarray([sparsity], dtype=np.float64)
                    ]
                    next_snapshot += 1
                detailed_seconds["snapshot_collection_seconds"] += (
                    time.perf_counter() - started
                )

        record_snapshots()
        progress = ReplayProgress(
            _tag(self.target), len(self.expected_trace), len(replay_rows)
        )
        previous_signal = signal.getsignal(signal.SIGINT)

        def request_interrupt(_signum, _frame) -> None:
            if self._interrupt_requested:
                raise KeyboardInterrupt
            self._interrupt_requested = True
            print(
                "\n[Task015] interrupt requested; saving a consistent checkpoint...",
                flush=True,
            )

        signal.signal(signal.SIGINT, request_interrupt)
        try:
            for expected_position in range(len(replay_rows), len(self.expected_trace)):
                expected = self.expected_trace[expected_position]
                expected_global = int(expected["global_index"])
                expected_domain = int(expected["domain_id"])
                expected_local = int(self.global_to_local[expected_global])
                expected_loss = float(expected["marginal_functional_loss"])

                started = time.perf_counter()
                with self.profiler.measure("main_heap"):
                    global_best = self.candidates.best(CANDIDATE_ALL)
                detailed_seconds["global_candidate_seconds"] += (
                    time.perf_counter() - started
                )
                started = time.perf_counter()
                with self.profiler.measure("typed_heap"):
                    attention_best = self.candidates.best(CANDIDATE_ATTENTION)
                    ffn_best = self.candidates.best(CANDIDATE_FFN)
                detailed_seconds["type_candidate_seconds"] += (
                    time.perf_counter() - started
                )
                if global_best is None or attention_best is None or ffn_best is None:
                    raise RuntimeError("Task015 exhausted a required candidate cache")
                loss, global_index, domain_id, local_index = global_best
                actual = {
                    "global_index": global_index,
                    "domain_id": domain_id,
                    "local_index": local_index,
                    "marginal_functional_loss": loss,
                    "best_attention": attention_best,
                    "best_ffn": ffn_best,
                }
                layer_id = int(self.global_layer_id[expected_global])
                with self.profiler.measure("trace_verification"):
                    valid = (
                        expected_global == global_index
                        and expected_domain == domain_id
                        and expected_local == local_index
                        and int(expected["step"]) == expected_position + 1
                        and self.retained_cpu[expected_global]
                        and pruned_by_layer[layer_id] < self.capacity_array[layer_id]
                        and int(expected["parameter_cost"])
                        == int(self.costs[expected_global])
                        and int(expected["domain_retained_before"])
                        == int(self.domain_retained_counts[expected_domain])
                        and math.isclose(
                            expected_loss,
                            loss,
                            rel_tol=0.0,
                            abs_tol=TRACE_LOSS_ABS_TOLERANCE,
                        )
                    )
                if not valid:
                    self._write_mismatch(
                        worker_tmp,
                        step=expected_position + 1,
                        expected=expected,
                        actual=actual,
                        domain_id=expected_domain,
                    )
                    raise RuntimeError(
                        f"Task015 exact replay mismatch at step "
                        f"{expected_position + 1}; see replay_mismatch.json"
                    )

                selected = self.units[expected_global]
                budget_before = removed_cost
                cost = int(self.costs[expected_global])
                attention_loss, attention_index, _, _ = attention_best
                ffn_loss, ffn_index, _, _ = ffn_best
                replay_rows.append(
                    {
                        "step": expected_position + 1,
                        "estimated_sparsity": budget_before / self.total_parameters,
                        "best_attention_delta": attention_loss,
                        "best_attention_global_index": attention_index,
                        "best_attention_layer": self.units[attention_index].layer,
                        "best_ffn_delta": ffn_loss,
                        "best_ffn_global_index": ffn_index,
                        "best_ffn_layer": self.units[ffn_index].layer,
                        "delta_ratio_attention_over_ffn": attention_loss
                        / (ffn_loss + EPS),
                        "selected_type": selected.unit_type,
                        "selected_global_index": expected_global,
                    }
                )
                started = time.perf_counter()
                with self.profiler.measure("state_update"):
                    coverage = self._remove_selected(
                        expected_domain, expected_local, loss
                    )
                    removed_cost += cost
                    pruned_by_layer[layer_id] += 1
                    self._refresh_after_removal(
                        domain_id=expected_domain,
                        layer_id=layer_id,
                        pruned_by_layer=pruned_by_layer,
                        coverage=coverage,
                    )
                detailed_seconds["domain_update_seconds"] += (
                    time.perf_counter() - started
                )
                record_snapshots()
                progress.update(
                    step=expected_position + 1,
                    sparsity=removed_cost / self.total_parameters,
                    selected_type=selected.unit_type,
                    attention_loss=attention_loss,
                    ffn_loss=ffn_loss,
                )
                step = expected_position + 1
                if step % CHECKPOINT_INTERVAL_STEPS == 0 or self._interrupt_requested:
                    started = time.perf_counter()
                    self._save_checkpoint(
                        step=step,
                        removed_cost=removed_cost,
                        pruned_by_layer=pruned_by_layer,
                        replay_rows=replay_rows,
                        snapshot_payload=snapshot_payload,
                        next_snapshot=next_snapshot,
                        worker_tmp=worker_tmp,
                    )
                    detailed_seconds["checkpoint_write_seconds"] += (
                        time.perf_counter() - started
                    )
                if self._interrupt_requested:
                    raise ReplayInterrupted(
                        f"Task015 {_tag(self.target)} checkpointed at step {step}"
                    )
        finally:
            signal.signal(signal.SIGINT, previous_signal)
            progress.close()

        record_snapshots()
        expected_indices = [int(row["global_index"]) for row in self.expected_trace]
        actual_indices = [int(row["selected_global_index"]) for row in replay_rows]
        if actual_indices != expected_indices:
            raise ValueError("Task015 replay did not exactly reproduce Task014")
        expected_removed_cost = int(self.costs[expected_indices].sum())
        if removed_cost != expected_removed_cost or removed_cost < self.target_budget:
            raise ValueError("Task015 final parameter budget does not match Task014")
        if len(expected_indices) > 1:
            before_final = expected_removed_cost - int(self.costs[expected_indices[-1]])
            if before_final >= self.target_budget:
                raise ValueError("Task014 trace continued after reaching its budget")

        started = time.perf_counter()
        with self.profiler.measure("csv_json_writing"):
            _atomic_csv(
                worker_tmp / "best_candidate_type_competition.csv",
                tuple(replay_rows[0]),
                replay_rows,
            )
            np.savez_compressed(
                worker_tmp / "candidate_snapshots.npz",
                **{key: value[0] for key, value in snapshot_payload.items()},
            )
            detailed_seconds["output_write_seconds"] = (
                time.perf_counter() - started
            )
            wall_seconds = time.perf_counter() - wall_started
            performance = {
                "target_sparsity": self.target,
                "steps": len(replay_rows),
                "wall_time_seconds": wall_seconds,
                "steps_per_second": len(replay_rows) / max(wall_seconds, 1e-12),
                "gpu_device": str(self.device),
                "physical_cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "peak_cuda_memory_mib": module.cuda.max_memory_allocated(
                    self.device
                ) / (1024.0 ** 2),
                "checkpoint_resumed": checkpoint_resumed,
                **detailed_seconds,
                "internal_profiler_enabled": self.profiler.enabled,
                "internal_profile_seconds": self.profiler.seconds,
            }
            _atomic_json(worker_tmp / "performance_metrics.json", performance)
            _atomic_json(
                worker_tmp / "replay_verification.json",
                {
                    "status": "passed",
                    "engine": "optimized_trace_driven",
                    "target_sparsity": self.target,
                    "selected_indices_exact_match": True,
                    "marginal_loss_abs_tolerance": TRACE_LOSS_ABS_TOLERANCE,
                    "selection_steps": len(replay_rows),
                    "removed_parameter_cost": removed_cost,
                    "target_parameter_budget": self.target_budget,
                    "trace_sha256": _sha256_file(
                        self.run_dir / "functional_selection_trace.csv"
                    ),
                    "descriptor_statistics": str(self.descriptor_path.resolve()),
                    "descriptor_sha256": _sha256_file(self.descriptor_path),
                    "bms_domains": len(self.groups),
                    "bms_membership_sha256": hashlib.sha256(
                        json.dumps(self.groups, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "device": str(self.device),
                    "physical_cuda_visibility": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "pruning_registry_created": False,
                    "pruning_applied": False,
                    "validation_executed": False,
                },
            )
        if worker_final.exists():
            shutil.rmtree(worker_final)
        worker_tmp.replace(worker_final)
        if not self.keep_checkpoint:
            self._delete_checkpoint()

    def benchmark(self, steps: int, engine: str) -> dict[str, object]:
        """Benchmark an exact prefix without writing scientific artifacts."""
        if engine not in {"optimized", "legacy"}:
            raise ValueError(engine)
        limit = min(int(steps), len(self.expected_trace))
        if limit <= 0:
            raise ValueError("Benchmark steps must be positive")
        _require_torch().cuda.reset_peak_memory_stats(self.device)
        self._validate_cost_evidence()
        self._initialize_states()
        started = time.perf_counter()
        if engine == "optimized":
            rows = self._benchmark_optimized(limit)
        else:
            rows = self._benchmark_legacy(limit)
        wall = time.perf_counter() - started
        return {
            "engine": engine,
            "steps": limit,
            "wall_time_seconds": wall,
            "initialization_time_seconds": started - self.constructed_at,
            "total_wall_time_seconds": time.perf_counter() - self.constructed_at,
            "steps_per_second": limit / max(wall, 1e-12),
            "estimated_full_target_seconds": wall
            * len(self.expected_trace)
            / limit,
            "selected_indices": [row["global_index"] for row in rows],
            "rows": rows,
            "gpu_device": str(self.device),
            "peak_cuda_memory_mib": _require_torch().cuda.max_memory_allocated(
                self.device
            ) / (1024.0 ** 2),
            "scientific_outputs_written": False,
        }

    def _benchmark_optimized(self, limit: int) -> list[dict[str, object]]:
        self._initialize_candidate_index()
        if self.candidates is None:
            raise RuntimeError("Candidate cache is not initialized")
        pruned = np.zeros(len(self.layer_names), dtype=np.int64)
        rows = []
        removed_cost = 0
        for position, expected in enumerate(self.expected_trace[:limit]):
            best = self.candidates.best(CANDIDATE_ALL)
            attention = self.candidates.best(CANDIDATE_ATTENTION)
            ffn = self.candidates.best(CANDIDATE_FFN)
            if best is None or attention is None or ffn is None:
                raise RuntimeError("Benchmark candidate cache exhausted")
            loss, global_index, domain_id, local_index = best
            if global_index != int(expected["global_index"]):
                raise RuntimeError(
                    f"Optimized benchmark mismatch at step {position + 1}"
                )
            layer_id = int(self.global_layer_id[global_index])
            rows.append(
                {
                    "step": position + 1,
                    "global_index": global_index,
                    "domain_id": domain_id,
                    "marginal_loss": loss,
                    "best_attention_delta": attention[0],
                    "best_attention_global_index": attention[1],
                    "best_ffn_delta": ffn[0],
                    "best_ffn_global_index": ffn[1],
                    "removed_cost_before": removed_cost,
                }
            )
            coverage = self._remove_selected(domain_id, local_index, loss)
            removed_cost += int(self.costs[global_index])
            pruned[layer_id] += 1
            self._refresh_after_removal(
                domain_id=domain_id,
                layer_id=layer_id,
                pruned_by_layer=pruned,
                coverage=coverage,
            )
        return rows

    def _benchmark_legacy(self, limit: int) -> list[dict[str, object]]:
        import heapq

        pruned = np.zeros(len(self.layer_names), dtype=np.int64)
        main_heap: list[tuple[float, int, int, int, int]] = []
        typed_heaps: dict[str, list[tuple[float, int, int, int, int]]] = {
            TYPE_ATTENTION: [],
            TYPE_FFN: [],
        }

        def feasible(global_index: int) -> bool:
            layer_id = int(self.global_layer_id[global_index])
            return pruned[layer_id] < self.capacity_array[layer_id]

        def push_main(state) -> None:
            candidate = state.best_feasible_candidate(feasible)
            if candidate is not None:
                loss, global_index, local_index = candidate
                heapq.heappush(
                    main_heap,
                    (loss, global_index, state.domain_id, state.version, local_index),
                )

        def push_type(state, unit_type: str) -> None:
            candidate = self._legacy_type_best(state, unit_type, feasible)
            if candidate is not None:
                loss, global_index, local_index = candidate
                heapq.heappush(
                    typed_heaps[unit_type],
                    (loss, global_index, state.domain_id, state.version, local_index),
                )

        def peek_type(unit_type: str):
            heap = typed_heaps[unit_type]
            while heap:
                entry = heap[0]
                _, global_index, domain_id, version, local_index = entry
                state = self.states[domain_id]
                current = self._legacy_type_best(state, unit_type, feasible)
                if current is None:
                    heapq.heappop(heap)
                    continue
                current_loss, current_global, current_local = current
                if (
                    version != state.version
                    or global_index != current_global
                    or local_index != current_local
                ):
                    heapq.heapreplace(
                        heap,
                        (
                            current_loss,
                            current_global,
                            domain_id,
                            state.version,
                            current_local,
                        ),
                    )
                    continue
                return entry
            return None

        for state in self.states:
            push_main(state)
            push_type(state, TYPE_ATTENTION)
            push_type(state, TYPE_FFN)
        rows = []
        removed_cost = 0
        while len(rows) < limit:
            if not main_heap:
                raise RuntimeError("Legacy benchmark heap exhausted")
            _, global_index, domain_id, version, local_index = heapq.heappop(
                main_heap
            )
            state = self.states[domain_id]
            if version != state.version:
                continue
            current = state.best_feasible_candidate(feasible)
            if current is None:
                continue
            loss, current_global, current_local = current
            if current_global != global_index or current_local != local_index:
                heapq.heappush(
                    main_heap,
                    (
                        loss,
                        current_global,
                        domain_id,
                        state.version,
                        current_local,
                    ),
                )
                continue
            attention = peek_type(TYPE_ATTENTION)
            ffn = peek_type(TYPE_FFN)
            if attention is None or ffn is None:
                raise RuntimeError("Legacy typed candidate heap exhausted")
            expected = self.expected_trace[len(rows)]
            if global_index != int(expected["global_index"]):
                raise RuntimeError(f"Legacy benchmark mismatch at step {len(rows) + 1}")
            rows.append(
                {
                    "step": len(rows) + 1,
                    "global_index": global_index,
                    "domain_id": domain_id,
                    "marginal_loss": loss,
                    "best_attention_delta": attention[0],
                    "best_attention_global_index": attention[1],
                    "best_ffn_delta": ffn[0],
                    "best_ffn_global_index": ffn[1],
                    "removed_cost_before": removed_cost,
                }
            )
            layer_id = int(self.global_layer_id[global_index])
            state.remove(local_index)
            removed_cost += int(self.costs[global_index])
            self.retained_cpu[global_index] = False
            self.domain_retained_counts[domain_id] -= 1
            pruned[layer_id] += 1
            if self.domain_retained_counts[domain_id]:
                push_main(state)
                push_type(state, TYPE_ATTENTION)
                push_type(state, TYPE_FFN)
        return rows


def _snapshot_arrays(path: Path, progress: float) -> dict[str, np.ndarray]:
    prefix = f"p{int(round(progress * 100)):02d}_"
    with np.load(path, allow_pickle=False) as payload:
        names = [key[len(prefix):] for key in payload.files if key.startswith(prefix)]
        if not names:
            raise KeyError(f"Snapshot {progress:.0%} not found in {path}")
        return {name: payload[prefix + name] for name in names}


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    if method == "spearman":
        def ranks(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            result = np.empty(values.size, dtype=np.float64)
            start = 0
            while start < values.size:
                stop = start + 1
                while stop < values.size and values[order[stop]] == values[order[start]]:
                    stop += 1
                result[order[start:stop]] = 0.5 * (start + stop - 1)
                start = stop
            return result
        left, right = ranks(left), ranks(right)
    elif method != "pearson":
        raise ValueError(method)
    return float(np.corrcoef(left, right)[0, 1])


def _summary_rows(
    snapshots: dict[tuple[float, float], dict[str, np.ndarray]], key: str
) -> list[dict[str, object]]:
    rows = []
    for (target, progress), arrays in sorted(snapshots.items()):
        split = split_candidate_indices(arrays["unit_type"])
        for unit_type, indices in split.items():
            stats = quantile_summary(arrays[key][indices])
            rows.append(
                {
                    "target_sparsity": target,
                    "progress": progress,
                    "unit_type": unit_type,
                    "candidate_count": stats.pop("count"),
                    **stats,
                }
            )
    return rows


def _metric_group_rows(
    arrays: Mapping[str, np.ndarray], groups: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    metrics = {
        "D_abs": arrays["D_abs"],
        "D_rel": arrays["D_rel"],
        "D_dyn": arrays["D_dyn"],
        "functional_energy": arrays["functional_energy"],
        "initial_marginal_loss": arrays["delta"],
        "parameter_cost": arrays["cost"],
        "active_domain_size": arrays["active_domain_size"],
        "best_functional_similarity": arrays["best_similarity"],
    }
    rows = []
    for group_name, indices in groups.items():
        for metric, values in metrics.items():
            selected = np.asarray(values)[indices]
            if selected.size == 0:
                raise ValueError(f"Empty diagnostic group: {group_name}/{metric}")
            q25, median, q75 = np.quantile(selected.astype(np.float64), (0.25, 0.5, 0.75))
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "count": int(selected.size),
                    "mean": float(selected.mean()),
                    "median": float(median),
                    "q25": float(q25),
                    "q75": float(q75),
                }
            )
    return rows


def _write_figures(
    output_dir: Path,
    snapshots: dict[tuple[float, float], dict[str, np.ndarray]],
    best_rows: list[dict[str, str]],
    attention_similarity: list[dict[str, str]],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source = {
        progress: snapshots[(0.30, progress)]
        for progress in (0.0, 0.10, 0.20, 0.30)
    }

    def save(figure, name: str) -> None:
        figure.tight_layout()
        figure.savefig(figure_dir / f"{name}.png", dpi=300)
        figure.savefig(figure_dir / f"{name}.pdf")
        plt.close(figure)

    def boxes(key: str, ylabel: str, name: str, log: bool = False) -> None:
        values, labels = [], []
        for progress, arrays in source.items():
            split = split_candidate_indices(arrays["unit_type"])
            for unit_type in (TYPE_ATTENTION, TYPE_FFN):
                values.append(arrays[key][split[unit_type]])
                labels.append(f"{int(progress * 100)}%\n{'Attn' if unit_type == TYPE_ATTENTION else 'FFN'}")
        figure, axis = plt.subplots(figsize=(10.0, 4.8))
        axis.boxplot(values, labels=labels, showfliers=False)
        axis.set_ylabel(ylabel)
        if log:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        save(figure, name)

    boxes("delta", "Marginal functional loss Δ", "figure1_raw_marginal_loss")
    boxes("rho", "Diagnostic loss per parameter ρ", "figure2_loss_per_parameter")
    boxes("cost", "Parameter cost", "figure3_parameter_cost", log=True)
    boxes("active_domain_size", "Active BMS domain size", "figure4_domain_size")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    x = np.asarray([float(row["estimated_sparsity"]) for row in best_rows]) * 100.0
    axis.plot(x, [float(row["best_attention_delta"]) for row in best_rows], label="Attention")
    axis.plot(x, [float(row["best_ffn_delta"]) for row in best_rows], label="FFN")
    axis.set_xlabel("Estimated parameter sparsity (%)")
    axis.set_ylabel("Best feasible Δ")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, "figure5_best_candidate_over_progress")

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    same = [float(row["best_same_type_similarity"]) for row in attention_similarity]
    cross = [float(row["best_cross_type_similarity"]) for row in attention_similarity]
    axis.scatter(same, cross, s=16, alpha=0.65)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Best Attention→Attention similarity")
    axis.set_ylabel("Best Attention→FFN similarity")
    axis.grid(alpha=0.25)
    save(figure, "figure6_cross_type_similarity")

    arrays = source[0.0]
    split = split_candidate_indices(arrays["unit_type"])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for unit_type, label in ((TYPE_ATTENTION, "Attention"), (TYPE_FFN, "FFN")):
        indices = split[unit_type]
        if len(indices) > 10_000:
            rng = np.random.default_rng(SEED)
            indices = np.sort(rng.choice(indices, 10_000, replace=False))
        axis.scatter(
            arrays["active_domain_size"][indices], arrays["delta"][indices],
            s=8, alpha=0.35, label=label,
        )
    axis.set_xlabel("Active BMS domain size |Gₖ⁺|")
    axis.set_ylabel("Marginal functional loss Δ")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, "figure7_delta_vs_domain_size")
    boxes(
        "unnormalized_loss",
        "Diagnostic unnormalized loss Δ·|Gₖ⁺|",
        "figure8_unnormalized_coverage_loss",
    )


def aggregate(task014_root: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    snapshots: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    verifications = []
    performance_rows = []
    for target in TARGETS:
        worker_dir = output_dir / "replay" / _tag(target)
        verification = _read_json(worker_dir / "replay_verification.json")
        if verification.get("status") != "passed" or not verification.get(
            "selected_indices_exact_match"
        ):
            raise RuntimeError(f"Unverified Task015 replay: {worker_dir}")
        verifications.append(verification)
        performance_rows.append(_read_json(worker_dir / "performance_metrics.json"))
        for progress in SNAPSHOT_PROGRESS:
            if progress <= target:
                snapshots[(target, progress)] = _snapshot_arrays(
                    worker_dir / "candidate_snapshots.npz", progress
                )
    if len({row["descriptor_sha256"] for row in verifications}) != 1:
        raise RuntimeError("Task014 Dynamic3D descriptor tables differ across targets")
    if len({row["bms_membership_sha256"] for row in verifications}) != 1:
        raise RuntimeError("Task014 BMS memberships differ across targets")
    _atomic_json(
        output_dir / "performance_summary.json",
        {
            "status": "passed",
            "engine": "optimized_trace_driven",
            "targets": {
                _tag(float(row["target_sparsity"])): row
                for row in performance_rows
            },
            "total_wall_time_worker_seconds": sum(
                float(row["wall_time_seconds"]) for row in performance_rows
            ),
            "scientific_definition_changed": False,
        },
    )

    marginal_rows = _summary_rows(snapshots, "delta")
    rho_rows = _summary_rows(snapshots, "rho")
    unnormalized_rows = _summary_rows(snapshots, "unnormalized_loss")
    distribution_fields = (
        "target_sparsity", "progress", "unit_type", "candidate_count",
        "min", "q01", "q05", "q10", "q25", "median", "q75",
        "q90", "q95", "q99", "max", "mean", "std",
    )
    _atomic_csv(
        output_dir / "marginal_loss_distribution_by_progress.csv",
        distribution_fields,
        marginal_rows,
    )
    _atomic_csv(
        output_dir / "loss_per_parameter_distribution.csv",
        distribution_fields,
        rho_rows,
    )
    _atomic_csv(
        output_dir / "unnormalized_coverage_loss_by_type.csv",
        distribution_fields,
        unnormalized_rows,
    )

    initial = snapshots[(0.30, 0.0)]
    descriptors = _load_descriptors(
        _descriptor_path(Path(task014_root), _functional_run_dir(Path(task014_root), 0.30)),
        _load_units(_mapping_paths(Path(task014_root))[0]),
    )
    initial.update(
        {
            "D_abs": descriptors[initial["global_index"], 0],
            "D_rel": descriptors[initial["global_index"], 1],
            "D_dyn": descriptors[initial["global_index"], 2],
        }
    )
    split = split_candidate_indices(initial["unit_type"])
    cost_rows = []
    for unit_type, indices in split.items():
        stats = quantile_summary(initial["cost"][indices])
        cost_rows.append(
            {
                "unit_type": unit_type,
                "candidate_count": stats["count"],
                **{key: stats[key] for key in ("min", "q10", "q25", "median", "q75", "q90", "max", "mean")},
            }
        )
    _atomic_csv(
        output_dir / "parameter_cost_by_type.csv",
        (
            "unit_type", "candidate_count", "min", "q10", "q25",
            "median", "q75", "q90", "max", "mean",
        ),
        cost_rows,
    )

    imbalance_rows = []
    domain_loss_rows = []
    for (target, progress), arrays in sorted(snapshots.items()):
        current_split = split_candidate_indices(arrays["unit_type"])
        attn = current_split[TYPE_ATTENTION]
        ffn = current_split[TYPE_FFN]
        best_attention = float(arrays["delta"][attn].min())
        below = int(np.count_nonzero(arrays["delta"][ffn] < best_attention))
        empirical_cdf = below / len(ffn)
        probability = -math.expm1(
            len(ffn) * math.log1p(-min(empirical_cdf, 1.0 - 1e-15))
        ) if empirical_cdf else 0.0
        imbalance_rows.append(
            {
                "target_sparsity": target,
                "progress": progress,
                "feasible_attention_candidates": len(attn),
                "feasible_ffn_candidates": len(ffn),
                "ffn_over_attention_count_ratio": len(ffn) / len(attn),
                "exchangeable_probability_minimum_is_ffn": len(ffn) / (len(ffn) + len(attn)),
                "ffn_candidates_below_best_attention": below,
                "empirical_ffn_cdf_at_best_attention": empirical_cdf,
                "order_statistic_probability_ffn_min_below_attention": probability,
                "observed_ffn_min_below_attention": bool(below),
            }
        )
        for unit_type, indices in {
            TYPE_ATTENTION: attn,
            TYPE_FFN: ffn,
            "all_units": np.arange(len(arrays["delta"])),
        }.items():
            domain_stats = quantile_summary(arrays["active_domain_size"][indices])
            domain_loss_rows.append(
                {
                    "target_sparsity": target,
                    "progress": progress,
                    "unit_type": unit_type,
                    "candidate_count": len(indices),
                    "pearson_delta_vs_active_domain_size": _correlation(
                        arrays["delta"][indices], arrays["active_domain_size"][indices], "pearson"
                    ),
                    "spearman_delta_vs_active_domain_size": _correlation(
                        arrays["delta"][indices], arrays["active_domain_size"][indices], "spearman"
                    ),
                    "domain_size_q25": domain_stats["q25"],
                    "domain_size_median": domain_stats["median"],
                    "domain_size_q75": domain_stats["q75"],
                    "domain_size_q90": domain_stats["q90"],
                    "domain_size_mean": domain_stats["mean"],
                    "mean_delta_times_domain_size": float(arrays["unnormalized_loss"][indices].mean()),
                    "median_delta_times_domain_size": float(np.median(arrays["unnormalized_loss"][indices])),
                }
            )
    _atomic_csv(output_dir / "candidate_count_imbalance.csv", tuple(imbalance_rows[0]), imbalance_rows)
    _atomic_csv(output_dir / "domain_size_loss_analysis.csv", tuple(domain_loss_rows[0]), domain_loss_rows)

    source_worker = output_dir / "replay" / "s30"
    composition = _read_csv(source_worker / "domain_composition.csv")
    attention_similarity = _read_csv(source_worker / "mixed_attention_similarity.csv")
    ffn_similarity = _read_csv(source_worker / "mixed_ffn_similarity.csv")
    mixed_losses = _read_csv(source_worker / "mixed_domain_losses.csv")
    _atomic_csv(output_dir / "bms_domain_type_composition.csv", tuple(composition[0]), composition)
    _atomic_csv(
        output_dir / "mixed_domain_cross_type_similarity.csv",
        (
            "domain_id", "attention_global_index", "layer",
            "best_same_type_similarity", "best_ffn_similarity",
            "best_same_type_global_index", "best_ffn_global_index",
        ),
        (
            {
                "domain_id": row["domain_id"],
                "attention_global_index": row["global_index"],
                "layer": row["layer"],
                "best_same_type_similarity": row["best_same_type_similarity"],
                "best_ffn_similarity": row["best_cross_type_similarity"],
                "best_same_type_global_index": row["best_same_type_global_index"],
                "best_ffn_global_index": row["best_cross_type_global_index"],
            }
            for row in attention_similarity
        ),
    )
    _atomic_csv(
        output_dir / "mixed_domain_reverse_similarity.csv",
        (
            "domain_id", "ffn_global_index", "layer",
            "best_ffn_similarity", "best_attention_similarity",
            "best_ffn_global_index", "best_attention_global_index",
        ),
        (
            {
                "domain_id": row["domain_id"],
                "ffn_global_index": row["global_index"],
                "layer": row["layer"],
                "best_ffn_similarity": row["best_same_type_similarity"],
                "best_attention_similarity": row["best_cross_type_similarity"],
                "best_ffn_global_index": row["best_same_type_global_index"],
                "best_attention_global_index": row["best_cross_type_global_index"],
            }
            for row in ffn_similarity
        ),
    )
    _atomic_csv(
        output_dir / "mixed_domain_marginal_loss_by_type.csv",
        (
            "domain_id", "attention_count", "ffn_count",
            "median_delta_attention", "median_delta_ffn",
            "min_delta_attention", "min_delta_ffn",
        ),
        mixed_losses,
    )

    best_source = _read_csv(source_worker / "best_candidate_type_competition.csv")
    _atomic_csv(output_dir / "best_candidate_type_competition.csv", tuple(best_source[0]), best_source)

    selected_surviving_rows = []
    attention_selected_rows = []
    for target in TARGETS:
        trace = _read_csv(
            _functional_run_dir(Path(task014_root), target)
            / "functional_selection_trace.csv"
        )
        selected_global = {int(row["global_index"]) for row in trace}
        ffn_positions = split[TYPE_FFN]
        selected_ffn = np.asarray(
            [position for position in ffn_positions if int(initial["global_index"][position]) in selected_global],
            dtype=np.int64,
        )
        surviving_ffn = np.asarray(
            [position for position in ffn_positions if int(initial["global_index"][position]) not in selected_global],
            dtype=np.int64,
        )
        for row in _metric_group_rows(
            initial, {"selected_ffn": selected_ffn, "surviving_ffn": surviving_ffn}
        ):
            selected_surviving_rows.append({"target_sparsity": target, **row})
        for row in _metric_group_rows(
            initial, {"attention": split[TYPE_ATTENTION], "selected_ffn": selected_ffn}
        ):
            attention_selected_rows.append({"target_sparsity": target, **row})
    metric_fields = (
        "target_sparsity", "group", "metric", "count", "mean", "median", "q25", "q75"
    )
    _atomic_csv(output_dir / "selected_vs_surviving_ffn.csv", metric_fields, selected_surviving_rows)
    _atomic_csv(output_dir / "attention_vs_selected_ffn.csv", metric_fields, attention_selected_rows)

    mixed_count = sum(row["is_mixed"].lower() == "true" for row in composition)
    attention_only = sum(row["is_attention_only"].lower() == "true" for row in composition)
    ffn_only = sum(row["is_ffn_only"].lower() == "true" for row in composition)
    same_values = np.asarray(
        [float(row["best_same_type_similarity"]) for row in attention_similarity]
    )
    cross_values = np.asarray(
        [float(row["best_cross_type_similarity"]) for row in attention_similarity]
    )
    initial_delta = {
        unit_type: float(np.median(initial["delta"][indices]))
        for unit_type, indices in split.items()
    }
    initial_rho = {
        unit_type: float(np.median(initial["rho"][indices]))
        for unit_type, indices in split.items()
    }
    initial_unnormalized = {
        unit_type: float(np.median(initial["unnormalized_loss"][indices]))
        for unit_type, indices in split.items()
    }
    cost_medians = {
        unit_type: float(np.median(initial["cost"][indices]))
        for unit_type, indices in split.items()
    }
    raw_ratio = initial_delta[TYPE_ATTENTION] / (initial_delta[TYPE_FFN] + EPS)
    rho_ratio = initial_rho[TYPE_ATTENTION] / (initial_rho[TYPE_FFN] + EPS)
    unnormalized_ratio = initial_unnormalized[TYPE_ATTENTION] / (
        initial_unnormalized[TYPE_FFN] + EPS
    )
    cost_ratio = cost_medians[TYPE_ATTENTION] / cost_medians[TYPE_FFN]
    attention_domain_median = float(
        np.median(initial["active_domain_size"][split[TYPE_ATTENTION]])
    )
    ffn_domain_median = float(
        np.median(initial["active_domain_size"][split[TYPE_FFN]])
    )
    ffn_best_fraction = float(
        np.mean(
            [
                float(row["best_ffn_delta"])
                <= float(row["best_attention_delta"])
                for row in best_source
            ]
        )
    )
    selected_ffn_fraction = float(
        np.mean([row["selected_type"] == TYPE_FFN for row in best_source])
    )
    gap_reduced_by_cost = abs(math.log(max(rho_ratio, 1e-300))) < abs(
        math.log(max(raw_ratio, 1e-300))
    )
    gap_reduced_by_domain_size = abs(
        math.log(max(unnormalized_ratio, 1e-300))
    ) < abs(math.log(max(raw_ratio, 1e-300)))
    if same_values.size:
        cross_above_same = float(np.mean(cross_values > same_values))
        same_mean = float(same_values.mean())
        same_median = float(np.median(same_values))
        cross_mean = float(cross_values.mean())
        cross_median = float(np.median(cross_values))
    else:
        cross_above_same = same_mean = same_median = cross_mean = cross_median = 0.0
    diagnosis = {
        "status": "complete",
        "task014_replay_exact_for_all_targets": True,
        "raw_delta_median_attention_over_ffn": raw_ratio,
        "median_cost_attention_over_ffn": cost_ratio,
        "rho_median_attention_over_ffn": rho_ratio,
        "unnormalized_loss_median_attention_over_ffn": unnormalized_ratio,
        "cost_normalization_reduces_type_gap": gap_reduced_by_cost,
        "domain_unnormalization_reduces_type_gap": gap_reduced_by_domain_size,
        "initial_ffn_over_attention_candidate_ratio": len(split[TYPE_FFN]) / len(split[TYPE_ATTENTION]),
        "initial_attention_active_domain_size_median": attention_domain_median,
        "initial_ffn_active_domain_size_median": ffn_domain_median,
        "fraction_steps_best_ffn_not_above_best_attention": ffn_best_fraction,
        "fraction_steps_selected_type_ffn": selected_ffn_fraction,
        "attention_only_domains": attention_only,
        "ffn_only_domains": ffn_only,
        "mixed_domains": mixed_count,
        "fraction_attention_with_cross_similarity_above_same": cross_above_same,
        "mean_best_attention_to_attention_similarity": same_mean,
        "median_best_attention_to_attention_similarity": same_median,
        "mean_best_attention_to_ffn_similarity": cross_mean,
        "median_best_attention_to_ffn_similarity": cross_median,
        "selection_formula_changed": False,
        "pruning_executed": False,
        "validation_executed": False,
    }
    _atomic_json(output_dir / "diagnosis_summary.json", diagnosis)

    report = f"""# Task015 Attention–FFN competition diagnosis

All 10%, 20% and 30% diagnostic replays exactly reproduced the saved Task014
selected global-index sequences before these statistics were accepted. This
report is diagnostic only; it did not create a registry, apply pruning, run
validation, or change `argmin Delta_i(S)`.

## Quantitative answers

1. **Raw marginal loss.** At 0% progress, the Attention/FFN median raw Delta
   ratio is `{diagnosis['raw_delta_median_attention_over_ffn']:.6g}`; a ratio
   above one means the median Attention loss is larger.
2. **Unit parameter cost.** The median Attention/FFN unit-cost ratio is
   `{diagnosis['median_cost_attention_over_ffn']:.6g}`.
3. **Loss per parameter.** The median Attention/FFN diagnostic rho ratio is
   `{diagnosis['rho_median_attention_over_ffn']:.6g}`. This quantity was never
   used for selection. Relative to the raw ratio, cost normalization
   `{'reduces' if diagnosis['cost_normalization_reduces_type_gap'] else 'does not reduce'}`
   the multiplicative type gap.
4. **Candidate count.** Initially there are
   `{diagnosis['initial_ffn_over_attention_candidate_ratio']:.6g}` FFN
   candidates per Attention candidate; the progress-resolved order-statistic
   evidence is in `candidate_count_imbalance.csv`.
5. **BMS composition.** The 423 domains contain `{attention_only}`
   Attention-only, `{ffn_only}` FFN-only and `{mixed_count}` mixed domains.
6. **Cross-type substitution.** In mixed domains, the fraction of Attention
   heads whose best FFN similarity exceeds their best non-self Attention
   similarity is
   `{diagnosis['fraction_attention_with_cross_similarity_above_same']:.6g}`.
   Mean/median same-type similarities are
   `{diagnosis['mean_best_attention_to_attention_similarity']:.6g}` /
   `{diagnosis['median_best_attention_to_attention_similarity']:.6g}`; the
   cross-type values are
   `{diagnosis['mean_best_attention_to_ffn_similarity']:.6g}` /
   `{diagnosis['median_best_attention_to_ffn_similarity']:.6g}`.
7. **Domain-size normalization.** Type- and progress-resolved Pearson,
   Spearman and domain-size distributions are reported in
   `domain_size_loss_analysis.csv`. Initial median active domain sizes are
   `{diagnosis['initial_attention_active_domain_size_median']:.6g}` for
   Attention and `{diagnosis['initial_ffn_active_domain_size_median']:.6g}`
   for FFN.
8. **Unnormalized loss.** The initial Attention/FFN median ratio for
   `Delta*|G_k^+|` is
   `{diagnosis['unnormalized_loss_median_attention_over_ffn']:.6g}`; removing
   the domain average
   `{'reduces' if diagnosis['domain_unnormalization_reduces_type_gap'] else 'does not reduce'}`
   the multiplicative type gap.
9. **Why zero Attention heads were selected.** The measured explanation must
   start with the direct selector evidence: FFN's best feasible Delta was no
   larger than Attention's at
   `{diagnosis['fraction_steps_best_ffn_not_above_best_attention']:.6%}` of
   30%-replay steps, and FFN supplied
   `{diagnosis['fraction_steps_selected_type_ffn']:.6%}` of selected units.
   The reason that inequality persists must be read jointly from raw Delta
   separation, the cost-normalized diagnostic,
   the `{diagnosis['initial_ffn_over_attention_candidate_ratio']:.6g}`:1
   candidate-count imbalance, BMS type composition,
   cross-type substitution and domain-size normalization. The evidence table
   below separates direct functional evidence from scale/granularity evidence;
   it does not prescribe a new selector.

## Decision table

| Evidence | Supports genuine Attention importance | Supports scale/granularity bias |
|---|---:|---:|
| Raw Delta separation | ratio `{diagnosis['raw_delta_median_attention_over_ffn']:.6g}` | raw units have unequal structural granularity |
| Delta/cost separation | residual ratio `{diagnosis['rho_median_attention_over_ffn']:.6g}` | reduction relative to raw ratio indicates cost scale |
| Candidate count | observed global minima by type in trace | FFN/Attention count ratio `{diagnosis['initial_ffn_over_attention_candidate_ratio']:.6g}` |
| Domain size | correlations reported by type | domain averaging can reduce loss in large domains |
| Delta*domain_size | ratio `{diagnosis['unnormalized_loss_median_attention_over_ffn']:.6g}` | reduction relative to raw ratio indicates averaging scale |
| Cross-type similarity | low cross-type substitution supports irreplaceability | high cross-type substitution supports mixed-domain comparability |
| Selected FFN statistics | selected/surviving table identifies low-loss FFN | pool size and layer/domain concentration can dominate minima |

The phrase “functional importance” here is conditional on the current
functional calibration set and Task014 representation; it is not a claim of
global functionlessness or universal importance.
"""
    (output_dir / "diagnosis.md").write_text(report, encoding="utf-8")
    _write_figures(output_dir, snapshots, best_source, attention_similarity)
    _atomic_json(
        output_dir / "task015_completion.json",
        {
            "status": "passed",
            "verified_targets": list(TARGETS),
            "replay_required": True,
            "replay_selected_indices_exact": True,
            "task014_results_sufficient_without_calibration_or_validation": True,
            "gpu_workers": [
                {
                    "logical_device": verification["device"],
                    "physical_cuda_visibility": verification[
                        "physical_cuda_visibility"
                    ],
                }
                for verification in verifications
            ],
            "selector_modified": False,
            "pruning_executed": False,
            "validation_executed": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-caches")
    prepare.add_argument("--task014-root", type=Path, required=True)
    prepare.add_argument("--contribution-npz", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--device", default="cuda:0")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--task014-root", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--target", type=float, required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--no-resume", action="store_true")
    replay.add_argument("--keep-checkpoint", action="store_true")
    replay.add_argument("--profile", action="store_true")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument(
        "--task014-root",
        type=Path,
        default=Path(os.environ.get("TASK014_OUTPUT_DIR", "task014_functional_pruning_signed_v3")),
    )
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TASK015_OUTPUT_DIR", "task015_attention_ffn_diagnosis")),
    )
    benchmark.add_argument("--target", type=float, default=0.10)
    benchmark.add_argument("--steps", type=int, default=1000)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument(
        "--engine",
        choices=("optimized", "legacy", "both"),
        default="optimized",
    )
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--task014-root", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare-caches":
        prepare_caches(
            args.task014_root, args.contribution_npz, args.output_dir, args.device
        )
    elif args.command == "replay":
        try:
            DiagnosticReplay(
                task014_root=args.task014_root,
                output_dir=args.output_dir,
                target=args.target,
                device=args.device,
                resume=not args.no_resume,
                keep_checkpoint=args.keep_checkpoint,
                profile=args.profile,
            ).run()
        except ReplayInterrupted as error:
            print(str(error), flush=True)
            return 130
    elif args.command == "benchmark":
        engines = ("optimized", "legacy") if args.engine == "both" else (args.engine,)
        results = {}
        for engine in engines:
            result = DiagnosticReplay(
                task014_root=args.task014_root,
                output_dir=args.output_dir,
                target=args.target,
                device=args.device,
                resume=False,
            ).benchmark(args.steps, engine)
            results[engine] = result
        if args.engine == "both":
            optimized_rows = results["optimized"]["rows"]
            legacy_rows = results["legacy"]["rows"]
            exact_fields = (
                "global_index",
                "domain_id",
                "best_attention_global_index",
                "best_ffn_global_index",
                "removed_cost_before",
            )
            numeric_fields = (
                "marginal_loss",
                "best_attention_delta",
                "best_ffn_delta",
            )
            exact = all(
                left[field] == right[field]
                for left, right in zip(optimized_rows, legacy_rows)
                for field in exact_fields
            )
            numeric = all(
                math.isclose(
                    float(left[field]),
                    float(right[field]),
                    rel_tol=0.0,
                    abs_tol=TRACE_LOSS_ABS_TOLERANCE,
                )
                for left, right in zip(optimized_rows, legacy_rows)
                for field in numeric_fields
            )
            if not exact or not numeric:
                raise RuntimeError("Optimized/legacy benchmark regression mismatch")
        printable = {}
        for engine, result in results.items():
            indices = result.pop("selected_indices")
            result.pop("rows")
            result["selected_sequence_sha256"] = hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            printable[engine] = result
        printable["optimized_legacy_equivalent"] = args.engine == "both"
        print(json.dumps(printable, indent=2, sort_keys=True))
    elif args.command == "aggregate":
        aggregate(args.task014_root, args.output_dir)
    else:
        raise RuntimeError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

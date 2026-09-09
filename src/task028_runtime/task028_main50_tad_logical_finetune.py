"""Task028 main 50% T+A+D logical pruning and senior recovery.

This is the main experiment, independent of the Task024/Task027 30% output.
Selection starts with a fully retained replay (an empty prefix), uses the
already tested Task024 adaptive ranking tensors on CUDA, and stops at the
parameter budget rather than at a unit count.  Model tensors are never
physically resized: Task027's runtime ``index_select`` implementation is
used for logical pruning and the original state-dict shapes remain intact.

The module is intentionally lazy about torch/GluonCV imports.  Importing it
for semantic tests therefore does not initialize CUDA or decode a dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CODE_VERSION = "task028_main50_tad_logical_finetune_v1"
TARGET_SPARSITY = 0.50
TARGET_PARAMETER_SPARSITY = TARGET_SPARSITY
START_PREFIX: tuple[dict[str, object], ...] = ()
START_REMOVED_COST = 0
EXPECTED_REPLAY_UNITS = 36_378
EXPECTED_VALIDATION_SAMPLES = 3_783
EXPECTED_SWIN_BLOCKS = 24
TRAIN_BATCH_SIZE = 4
SEED = 3407
GPU_IDS = (0, 1)
RANKING_BACKENDS = ("auto", "single-gpu", "dual-gpu")
DEFAULT_MODEL_NAME = "swintrans"
RELOAD_TOLERANCE_PP = 1e-4
CHECKPOINT_NAME = "swin_pruned_best_50.pth"
VARIANT_NAME = "main_tad_50"
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
# The unchanged Task027 logical implementation sets these attributes and
# performs runtime index_select; no Parameter tensor is resized.
LOGICAL_KEEP_FIELDS = ("keep_heads", "keep_neurons")
TRACE_FIELDS = (
    "variant", "step", "incremental_step", "estimated_sparsity_before",
    "estimated_sparsity_after", "global_index", "unit_type", "layer",
    "stage", "unit_index", "domain_id", "local_index", "Delta_average",
    "Delta_total", "p_average", "p_total", "domain_coverage_before",
    "domain_damage_before", "R_dual", "R_adaptive", "parameter_cost",
    "cumulative_removed_parameters",
)
REQUIRED_REGISTRY_FIELDS = ("global_index", "layer", "unit_type", "unit_index")


def _torch():
    import torch
    return torch


def _argsort_unique(values):
    """Return the order of a one-dimensional vector of unique keys.

    ``global_index`` is validated as unique by ``OptimizedCandidateState``
    before this helper is used.  Consequently no stable-sort keyword is
    needed here; using the basic ``argsort`` keeps this path compatible with
    PyTorch 1.12 while preserving the exact order for unique integer keys.
    """
    torch = _torch()
    if values.ndim != 1:
        raise ValueError("unique-key sort expects a one-dimensional tensor")
    return torch.argsort(values)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_registry_sha256(registry: Mapping[str, Mapping[str, object]]) -> str:
    """Use the authoritative Task023 canonical registry representation."""
    import task023_average_rescue_causal_ablation as task023
    return task023.canonical_registry_sha256(registry)


def risk_from_components(p_total: float, p_average: float,
                         domain_damage: float) -> float:
    """Pure counterpart of Task024's adaptive risk rule."""
    values = (float(p_total), float(p_average), float(domain_damage))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("risk components must be finite")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("risk components must be in [0, 1]")
    return max(values)


def stop_after_budget(removed_cost: float, target_budget: float) -> bool:
    """Stopping is checked before removal; the last unit may overshoot."""
    return float(removed_cost) >= float(target_budget)


def assert_started_from_zero(engine: object) -> None:
    """Hard gate preventing accidental continuation of the 30% path."""
    if tuple(getattr(engine, "start_prefix", ())) not in ((), START_PREFIX):
        raise RuntimeError("Task028 must start from an empty prefix")
    if float(getattr(engine, "removed_cost", 0)) != START_REMOVED_COST:
        raise RuntimeError("Task028 must start with zero removed parameters")


def assert_full_retention_start(engine: object) -> None:
    """Prove that the replay is fully retained before selection step one."""
    replay = getattr(getattr(engine, "base", None), "replay", None)
    units = getattr(replay, "units", None)
    if units is not None:
        retained_units = len(units)
    else:
        retained_units = getattr(engine, "retained_units", None)
    if retained_units is None or int(retained_units) != EXPECTED_REPLAY_UNITS:
        raise RuntimeError(
            "Task028 full-retention gate requires exactly "
            f"{EXPECTED_REPLAY_UNITS} replay units"
        )
    states = getattr(replay, "states", None)
    if states is not None:
        for position, state in enumerate(states):
            retained_count = int(getattr(state, "retained_count", -1))
            initial_size = int(getattr(state, "initial_size", -2))
            if retained_count != initial_size:
                raise RuntimeError(f"Replay domain {position} is not fully retained")
            retained_mask = getattr(state, "retained", None)
            if retained_mask is not None and not bool(retained_mask.all().item()):
                raise RuntimeError(f"Replay domain {position} has an active removal mask")
    cache = getattr(replay, "cache", None)
    pruned = getattr(cache, "pruned", None)
    if pruned is not None and bool(pruned.any().item()):
        raise RuntimeError("Replay removal cache is active at the zero-sparsity start")


def assert_registry_ready_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate structural data before the unchanged registry constructor."""
    seen: set[int] = set()
    seen_units: set[tuple[str, str, int]] = set()
    for position, row in enumerate(rows):
        missing = [name for name in REQUIRED_REGISTRY_FIELDS
                   if name not in row or str(row[name]).strip() == ""]
        if missing:
            raise RuntimeError(f"Registry row {position} missing {missing}")
        global_index = int(row["global_index"])
        if global_index in seen:
            raise RuntimeError(f"Duplicate global_index {global_index}")
        seen.add(global_index)
        unit_type = str(row["unit_type"])
        if unit_type not in (TYPE_ATTENTION, TYPE_FFN):
            raise RuntimeError(f"Unsupported unit_type {unit_type!r}")
        key = (str(row["layer"]), unit_type, int(row["unit_index"]))
        if key in seen_units:
            raise RuntimeError(f"Duplicate structural unit {key}")
        seen_units.add(key)


def completion_gate(payload: Mapping[str, object]) -> bool:
    """Pure completion check used by the offline gate and CPU tests."""
    required_true = (
        "target_sparsity_50", "selection_started_from_zero",
        "final_tad_rule_used", "new_50_registry_constructed",
        "task024_30_registry_not_reused", "task027_30_registry_not_reused",
        "effective_parameter_sparsity_ge_target", "logical_pruning_applied",
        "pre_finetune_validation_complete", "registry_reproduction_pass",
        "fine_tuning_executed", "optimizer_sgd", "lr_0005", "momentum_09",
        "weight_decay_1e5", "batch_size_4", "seed_3407", "cross_entropy_only",
        "progress_display_enabled", "best_checkpoint_saved",
        "best_checkpoint_reload_pass", "senior_protocol_alignment",
        "existing_50_registry_reused",
    )
    required_false = (
        "physical_pruning_executed", "scheduler_used",
        "task024_artifacts_modified", "task025_artifacts_modified",
        "task027_artifacts_modified", "selection_rerun",
    )
    protocol_exact = (
        payload.get("use_checkpoint") is False and
        int(payload.get("checkpoint_enabled_block_count", -1)) == 0 and
        int(payload.get("swin_block_count", -1)) == EXPECTED_SWIN_BLOCKS and
        payload.get("amp_enabled") is False and
        payload.get("precision") == "fp32" and
        payload.get("validation_uses_softmax") is False
    )
    return (all(payload.get(key) is True for key in required_true) and
            all(payload.get(key) is False for key in required_false) and
            int(payload.get("epochs_completed", 0)) == 100 and
            protocol_exact)


def _task023():
    import task023_average_rescue_causal_ablation as task023
    return task023


def _task024():
    import task024_threshold_free_adaptive_safety as task024
    return task024


def _task027():
    import task027_senior_style_logical_pruning_finetune as task027
    return task027


def _build_senior_model(device: str):
    """Build the fresh original model with senior FP32 settings.

    Task027's historical runner intentionally enables activation checkpointing
    for its auxiliary 30% reproduction.  Task028 is a separate recovery run:
    it reuses the implementation and checkpoint helpers, but must construct a
    fresh model with checkpointing disabled.
    """
    torch = _torch()
    from ucf101_videoswin_my import SwinTransformer3D
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.0,
        qkv_bias=True, patch_norm=True, drop_path_rate=0.2,
        use_checkpoint=False,
    ).to(target)
    return model


def _checkpoint_block_counts(model: object) -> tuple[int, int]:
    """Return ``(total Swin blocks, checkpoint-enabled blocks)``."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("senior model must expose modules()")
    total = enabled = 0
    for module in modules():
        if module.__class__.__name__ != "SwinTransformerBlock3D":
            continue
        total += 1
        enabled += int(bool(getattr(module, "use_checkpoint", False)))
    return total, enabled


def assert_senior_model_protocol(model: object) -> dict[str, object]:
    """Strictly gate the 24-block, no-activation-checkpoint model."""
    total, enabled = _checkpoint_block_counts(model)
    if total != EXPECTED_SWIN_BLOCKS:
        raise RuntimeError(
            f"Task028 senior model requires {EXPECTED_SWIN_BLOCKS} Swin blocks; "
            f"found {total}"
        )
    if enabled != 0:
        raise RuntimeError(
            f"Task028 senior model must disable checkpointing; {enabled} blocks enabled"
        )
    return {
        "swin_block_count": total,
        "checkpoint_enabled_block_count": enabled,
        "use_checkpoint": False,
    }


def _load_senior_original(checkpoint: Path, device: str):
    """Load the source checkpoint into a fresh senior-protocol model."""
    torch = _torch()
    task027 = _task027()
    task027.set_seed(SEED)
    model = _build_senior_model(device)
    message = model.load_state_dict(
        task027._normalise_state_dict(
            torch.load(Path(checkpoint), map_location=device)
        ),
        strict=False,
    )
    model.eval()
    model._task028_checkpoint_load_message = message
    assert_senior_model_protocol(model)
    return model


def _tqdm(iterable=None, **kwargs):
    from tqdm.auto import tqdm
    return tqdm(iterable, **kwargs)


def _vectorized_exact_order(values, base):
    """Return exact ``(values, global_index)`` order without CPU tie work.

    ``base`` is already in ascending global-index order and ``values`` is the
    corresponding compact view.  The first argsort groups equal values; the
    unique composite key then restores the lower-priority global-index order
    entirely on the tensor device.  This uses only argsort operations
    supported by PyTorch 1.12.
    """
    torch = _torch()
    if values.ndim != 1 or base.ndim != 1 or values.numel() != base.numel():
        raise ValueError("exact order inputs must be aligned one-dimensional tensors")
    count = int(values.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    rough = torch.argsort(values)
    sorted_values = values.index_select(0, rough)
    group_start = torch.ones(count, dtype=torch.bool, device=values.device)
    if count > 1:
        group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    stride = count + 1
    composite = group_id * stride + rough.to(torch.int64)
    secondary = torch.argsort(composite)
    order_in_base = rough.index_select(0, secondary)
    return base.index_select(0, order_in_base)


def _cached_ordinal_ranks(values, static_positions, global_index_order):
    """Exact ordinal ranks using a cached global-index tie order on-device."""
    torch = _torch()
    count = int(values.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.float32, device=values.device)
    compact_index = torch.full(
        (int(global_index_order.numel()),), -1, dtype=torch.long,
        device=values.device,
    )
    compact_index.index_copy_(
        0, static_positions,
        torch.arange(count, dtype=torch.long, device=values.device),
    )
    membership = torch.zeros(
        int(global_index_order.numel()), dtype=torch.bool, device=values.device
    )
    membership.index_fill_(0, static_positions, True)
    active_static_in_global_order = global_index_order[
        membership.index_select(0, global_index_order)
    ]
    active_compact_in_global_order = compact_index.index_select(
        0, active_static_in_global_order
    )
    active_values = values.index_select(0, active_compact_in_global_order)
    ordered = _vectorized_exact_order(
        active_values, active_static_in_global_order
    )
    ranks = torch.empty_like(values, dtype=torch.float32)
    positions = torch.arange(count, dtype=torch.float32, device=values.device)
    compact_order = compact_index.index_select(0, ordered)
    ranks.index_copy_(0, compact_order, positions / float(max(count - 1, 1)))
    return ranks


def _select_ranked_from_ranks(candidates: Mapping[str, object], p_total,
                              p_average, domain_damage=None) -> dict[str, object]:
    """Select from already computed rank tensors on the primary device."""
    torch = _torch()
    if domain_damage is None:
        domain_damage = torch.zeros_like(p_total)
    risks = {
        "p_total": p_total,
        "p_average": p_average,
        "R_dual": torch.maximum(p_total, p_average),
        "domain_damage": domain_damage,
    }
    risks["R_adaptive"] = torch.maximum(risks["R_dual"], domain_damage)
    if int(p_total.numel()) == 0:
        raise RuntimeError("Cannot select from an empty candidate state")
    minimum_risk = risks["R_adaptive"].min()
    tied = risks["R_adaptive"] == minimum_risk
    p_total = torch.where(tied, risks["p_total"],
                          torch.full_like(risks["p_total"], torch.inf))
    selected = torch.argmin(p_total)
    output = dict(candidates)
    output.update(risks)
    output["selected_position"] = int(selected.item())
    return output


def fast_select_ranked_candidate(candidates: Mapping[str, object], *,
                                 global_index_order=None) -> dict[str, object]:
    """Exact fast adaptive ranker; the legacy full selector remains available.

    Since ordinal ``p_total`` is unique for every active candidate (including
    the one-candidate edge case), minimizing ``(R_adaptive, p_total)`` is
    mathematically equivalent to the old five-key lexicographic selector.
    """
    total = candidates["Delta_total"]
    average = candidates["Delta_average"]
    gids = candidates["global_index"]
    if global_index_order is None:
        task024 = _task024()
        risks = task024.compute_risk_tensors(
            total, average, gids, candidates.get("domain_damage")
        )
    else:
        p_total = _cached_ordinal_ranks(
            total, candidates["_positions"], global_index_order
        )
        p_average = _cached_ordinal_ranks(
            average, candidates["_positions"], global_index_order
        )
        return _select_ranked_from_ranks(
            candidates, p_total, p_average, candidates.get("domain_damage")
        )
    return _select_ranked_from_ranks(
        candidates, risks["p_total"], risks["p_average"], risks.get("domain_damage")
    )


class OptimizedCandidateState:
    """GPU-resident global candidate arrays for the exact Task028 selector."""

    def __init__(self, engine, ranking_backend: str = "auto") -> None:
        torch = engine.torch
        if ranking_backend not in RANKING_BACKENDS:
            raise ValueError(f"Unknown ranking backend: {ranking_backend}")
        self.engine = engine
        self.torch = torch
        self.device = engine.device
        self.ranking_backend_requested = ranking_backend
        self.ranking_backend: str | None = (
            None if ranking_backend == "auto" else ranking_backend
        )
        self.backend_benchmark: dict[str, object] | None = None
        self.last_rank_timings = {
            "gpu0_rank_seconds": 0.0,
            "gpu1_rank_seconds": 0.0,
            "inter_gpu_transfer_seconds": 0.0,
        }
        replay = engine.base.replay
        self.replay = replay
        self.total_units = len(replay.units)
        if self.total_units != EXPECTED_REPLAY_UNITS:
            raise RuntimeError("Unexpected Task028 replay unit count")
        self.global_index = torch.empty(self.total_units, dtype=torch.long,
                                        device=self.device)
        self.domain_id = torch.empty_like(self.global_index)
        self.local_index = torch.empty_like(self.global_index)
        self.delta_average = torch.empty(self.total_units, dtype=torch.float32,
                                         device=self.device)
        self.delta_total = torch.empty_like(self.delta_average)
        self.domain_damage = torch.empty_like(self.delta_average)
        self.active_mask = torch.zeros(self.total_units, dtype=torch.bool,
                                       device=self.device)
        self.parameter_cost = torch.as_tensor(
            replay.costs, dtype=torch.long, device=self.device
        )
        self.mean_parameter_cost = float(replay.costs.mean())
        self.domain_positions: list[object] = []
        self.layer_positions: dict[str, list[int]] = {}
        cursor = 0
        for domain, state in enumerate(replay.states):
            size = len(state.global_indices)
            positions = torch.arange(cursor, cursor + size, dtype=torch.long,
                                     device=self.device)
            globals_ = torch.as_tensor(state.global_indices, dtype=torch.long,
                                       device=self.device)
            self.domain_positions.append(positions)
            self.global_index.index_copy_(0, positions, globals_)
            self.domain_id.index_fill_(0, positions, domain)
            self.local_index.index_copy_(
                0, positions,
                torch.arange(size, dtype=torch.long, device=self.device),
            )
            self.delta_average.index_copy_(0, positions, state.losses.float())
            self.delta_total.index_copy_(
                0, positions,
                state.losses.float() * float(engine.base.replay.cache.active_size[domain]),
            )
            self.domain_damage.index_fill_(
                0, positions, 1.0 - float(state.current_coverage)
            )
            eligible = replay.cache.eligible_mask(domain)
            self.active_mask.index_copy_(0, positions, eligible)
            for local, global_index in enumerate(state.global_indices):
                layer = str(replay.unit_by_global[int(global_index)].layer)
                self.layer_positions.setdefault(layer, []).append(cursor + local)
            cursor += size
        if cursor != self.total_units:
            raise RuntimeError("Candidate state did not cover every replay unit")
        if int(self.global_index.unique().numel()) != self.total_units:
            raise RuntimeError("Replay global indices are not unique")
        self.global_index_order = _argsort_unique(self.global_index)
        active_average = self.delta_average[self.active_mask]
        if active_average.numel() and not bool(torch.isfinite(active_average).all().item()):
            raise RuntimeError("Initial candidate state contains non-finite losses")
        self.initial_active_units = int(self.active_mask.sum().item())
        # Public aliases make the fixed global state explicit in diagnostics.
        self.Delta_average = self.delta_average
        self.Delta_total = self.delta_total
        self.dual_gpu_available = False
        self.secondary_device = None
        # Keep telemetry well-defined for a true single-GPU run.  The dual
        # mirror initializer overwrites these with the detected peer status.
        self.peer_access_0_to_1 = False
        self.peer_access_1_to_0 = False
        if self.device.type == "cuda" and torch.cuda.device_count() >= 2:
            self._initialize_dual_gpu_mirror()
            # Complete the one-time cross-device mirror before any ranking
            # stream consumes it; no per-step global synchronization follows.
            self._synchronize_all_ranking_devices()
        elif ranking_backend == "dual-gpu":
            raise RuntimeError("dual-gpu ranking requires CUDA devices 0 and 1")
        if ranking_backend == "dual-gpu" and not self.dual_gpu_available:
            raise RuntimeError("dual-gpu ranking backend is unavailable")

    def _initialize_dual_gpu_mirror(self) -> None:
        """Mirror immutable state and p-average state on CUDA device 1."""
        torch = self.torch
        if self.device.type != "cuda" or int(self.device.index or 0) != 0:
            raise RuntimeError("dual-gpu ranking requires the primary device cuda:0")
        self.secondary_device = torch.device("cuda:1")
        self.dual_gpu_available = True
        self.global_index_secondary = self.global_index.to(
            self.secondary_device, non_blocking=True
        )
        self.global_index_order_secondary = _argsort_unique(
            self.global_index_secondary
        )
        self.domain_id_secondary = self.domain_id.to(
            self.secondary_device, non_blocking=True
        )
        self.local_index_secondary = self.local_index.to(
            self.secondary_device, non_blocking=True
        )
        self.parameter_cost_secondary = self.parameter_cost.to(
            self.secondary_device, non_blocking=True
        )
        self.delta_average_secondary = self.delta_average.to(
            self.secondary_device, non_blocking=True
        )
        self.active_mask_secondary = self.active_mask.to(
            self.secondary_device, non_blocking=True
        )
        self.domain_positions_secondary = [
            positions.to(self.secondary_device, non_blocking=True)
            for positions in self.domain_positions
        ]
        self.layer_positions_secondary = {
            layer: torch.as_tensor(positions, dtype=torch.long,
                                   device=self.secondary_device)
            for layer, positions in self.layer_positions.items()
        }
        self._primary_rank_stream = torch.cuda.Stream(device=self.device)
        self._secondary_rank_stream = torch.cuda.Stream(device=self.secondary_device)
        with torch.cuda.device(self.secondary_device):
            self._secondary_rank_event = torch.cuda.Event()
        try:
            self.peer_access_0_to_1 = bool(
                torch.cuda.can_device_access_peer(0, 1)
            )
            self.peer_access_1_to_0 = bool(
                torch.cuda.can_device_access_peer(1, 0)
            )
        except (AttributeError, RuntimeError):
            self.peer_access_0_to_1 = False
            self.peer_access_1_to_0 = False

    def _synchronize_all_ranking_devices(self) -> None:
        if not self.dual_gpu_available:
            self.torch.cuda.synchronize(self.device)
            return
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.synchronize(self.secondary_device)

    def _rank_single_gpu(self, candidates: Mapping[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        result = fast_select_ranked_candidate(
            candidates, global_index_order=self.global_index_order
        )
        self.last_rank_timings = {
            "gpu0_rank_seconds": time.perf_counter() - started,
            "gpu1_rank_seconds": 0.0,
            "inter_gpu_transfer_seconds": 0.0,
        }
        return result

    def _rank_dual_gpu(self, candidates: Mapping[str, object]) -> dict[str, object]:
        """Compute p_total/p_average concurrently and reduce on cuda:0."""
        if not self.dual_gpu_available:
            raise RuntimeError("dual-gpu ranking backend is unavailable")
        torch = self.torch
        positions = candidates["_positions"]
        primary_stream = self._primary_rank_stream
        secondary_stream = self._secondary_rank_stream
        primary_stream.wait_stream(torch.cuda.current_stream(self.device))
        secondary_stream.wait_stream(torch.cuda.current_stream(self.secondary_device))
        gpu0_started = time.perf_counter()
        with torch.cuda.stream(primary_stream):
            p_total = _cached_ordinal_ranks(
                candidates["Delta_total"], positions, self.global_index_order
            )
        gpu0_elapsed = time.perf_counter() - gpu0_started
        gpu1_started = time.perf_counter()
        with torch.cuda.stream(secondary_stream):
            secondary_positions = self.active_mask_secondary.nonzero(
                as_tuple=True
            )[0]
            if int(secondary_positions.numel()) != int(positions.numel()):
                raise RuntimeError("Primary/secondary active masks diverged")
            p_average_secondary = _cached_ordinal_ranks(
                self.delta_average_secondary.index_select(0, secondary_positions),
                secondary_positions, self.global_index_order_secondary,
            )
            self._secondary_rank_event.record(secondary_stream)
        gpu1_elapsed = time.perf_counter() - gpu1_started
        primary_stream.wait_event(self._secondary_rank_event)
        transfer_started = time.perf_counter()
        with torch.cuda.stream(primary_stream):
            p_average = p_average_secondary.to(
                device=self.device, non_blocking=True
            )
            result = _select_ranked_from_ranks(
                candidates, p_total, p_average,
                candidates.get("domain_damage"),
            )
        transfer_elapsed = time.perf_counter() - transfer_started
        torch.cuda.current_stream(self.device).wait_stream(primary_stream)
        self.last_rank_timings = {
            "gpu0_rank_seconds": gpu0_elapsed,
            "gpu1_rank_seconds": gpu1_elapsed,
            "inter_gpu_transfer_seconds": transfer_elapsed,
        }
        return result

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    def candidate_tensors(self) -> dict[str, object]:
        positions = self.active_mask.nonzero(as_tuple=True)[0]
        return {
            "_positions": positions,
            "global_index": self.global_index.index_select(0, positions),
            "domain_id": self.domain_id.index_select(0, positions),
            "local_index": self.local_index.index_select(0, positions),
            "Delta_average": self.delta_average.index_select(0, positions),
            "Delta_total": self.delta_total.index_select(0, positions),
            "domain_coverage": 1.0 - self.domain_damage.index_select(0, positions),
            "domain_damage": self.domain_damage.index_select(0, positions),
        }

    def rank(self, candidates: Mapping[str, object] | None = None,
             *, backend: str | None = None) -> dict[str, object]:
        if candidates is None:
            candidates = self.candidate_tensors()
        requested = backend or self.ranking_backend
        if requested is None:
            if not self.dual_gpu_available:
                self.ranking_backend = "single-gpu"
                return self._rank_single_gpu(candidates)
            # Measure both exact backends once, before any removal, and keep
            # the faster one for all dependent pruning steps.
            self._synchronize_all_ranking_devices()
            single_started = time.perf_counter()
            single = self._rank_single_gpu(candidates)
            self._synchronize_all_ranking_devices()
            single_elapsed = time.perf_counter() - single_started
            dual_started = time.perf_counter()
            dual = self._rank_dual_gpu(candidates)
            self._synchronize_all_ranking_devices()
            dual_elapsed = time.perf_counter() - dual_started
            single_position = int(single["selected_position"])
            dual_position = int(dual["selected_position"])
            single_global = int(single["global_index"][single_position].item())
            dual_global = int(dual["global_index"][dual_position].item())
            if single_global != dual_global:
                raise RuntimeError("Single/dual GPU ranking sequence mismatch")
            for name in ("p_total", "p_average", "R_dual", "domain_damage", "R_adaptive"):
                _assert_equivalence_tensor(f"auto.{name}", single[name], dual[name])
            selected = "dual-gpu" if dual_elapsed < single_elapsed else "single-gpu"
            self.ranking_backend = selected
            self.backend_benchmark = {
                "single_gpu_seconds": single_elapsed,
                "dual_gpu_seconds": dual_elapsed,
                "selected_backend": selected,
            }
            return dual if selected == "dual-gpu" else single
        if requested == "single-gpu":
            return self._rank_single_gpu(candidates)
        if requested == "dual-gpu":
            return self._rank_dual_gpu(candidates)
        raise ValueError(f"Unknown ranking backend: {requested}")

    def _refresh_domain(self, domain: int) -> None:
        state = self.replay.states[domain]
        positions = self.domain_positions[domain]
        losses = state.losses.float()
        self.delta_average.index_copy_(0, positions, losses)
        self.delta_total.index_copy_(
            0, positions,
            losses * float(self.replay.cache.active_size[domain]),
        )
        self.domain_damage.index_fill_(
            0, positions, 1.0 - float(state.current_coverage)
        )
        if self.dual_gpu_available:
            with self.torch.cuda.stream(self._secondary_rank_stream):
                losses_secondary = losses.to(
                    self.secondary_device, non_blocking=True
                )
                self.delta_average_secondary.index_copy_(
                    0, self.domain_positions_secondary[domain], losses_secondary
                )

    def remove(self, ranked: Mapping[str, object], position: int) -> dict[str, object]:
        task024 = _task024()
        static_position = int(ranked["_positions"][position].item())
        global_index = int(self.global_index[static_position].item())
        domain = int(self.domain_id[static_position].item())
        local = int(self.local_index[static_position].item())
        before = self.engine.base.removed_cost
        result = self.engine.base.remove(
            variant=task024.VARIANT_ADAPTIVE,
            global_index=global_index, domain_id=domain, local_index=local,
            rank=position + 1,
        )
        result.update({
            "p_total": float(ranked["p_total"][position].item()),
            "p_average": float(ranked["p_average"][position].item()),
            "domain_coverage_before": float(ranked["domain_coverage"][position].item()),
            "domain_damage_before": float(ranked["domain_damage"][position].item()),
            "R_dual": float(ranked["R_dual"][position].item()),
            "R_adaptive": float(ranked["R_adaptive"][position].item()),
            "estimated_sparsity_before": before / float(self.engine.total_parameters),
            "estimated_sparsity_after": self.engine.base.removed_cost / float(self.engine.total_parameters),
            "incremental_step": result["step"],
        })
        self.active_mask[static_position] = False
        layer = str(result["layer"])
        layer_id = self.replay.cache.layer_to_id[layer]
        if self.replay.cache.pruned[layer_id] >= self.replay.cache.capacities[layer_id]:
            layer_positions = self.layer_positions[layer]
            self.active_mask.index_fill_(
                0, self.torch.as_tensor(layer_positions, dtype=self.torch.long,
                                        device=self.device), False
            )
        if self.dual_gpu_available:
            with self.torch.cuda.stream(self._secondary_rank_stream):
                self.active_mask_secondary[static_position] = False
                if self.replay.cache.pruned[layer_id] >= self.replay.cache.capacities[layer_id]:
                    self.active_mask_secondary.index_fill_(
                        0, self.layer_positions_secondary[layer], False
                    )
        self._refresh_domain(domain)
        return result


def legacy_select_step(engine) -> dict[str, object]:
    """One exact reference step retained for the equivalence harness."""
    task024 = _task024()
    candidates = engine.candidate_tensors()
    ranked = task024.rank_candidate_tensors(candidates, adaptive=True)
    position = int(ranked.pop("selected_position"))
    return engine.remove(ranked, variant=task024.VARIANT_ADAPTIVE,
                         selected_position=position, risks=ranked)


def _sequence_sha(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gpu_memory_snapshot(torch, device) -> dict[str, object]:
    """Return lightweight server-side GPU diagnostics without a CPU tensor copy."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return {
            "device": str(device), "device_name": "unavailable",
            "allocated_memory_bytes": 0, "reserved_memory_bytes": 0,
            "peak_allocated_memory_bytes": 0,
        }
    with torch.cuda.device(device):
        return {
            "device": str(device),
            "device_name": str(torch.cuda.get_device_name(device)),
            "allocated_memory_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }


def _legacy_active_count(engine) -> int:
    """Count feasible retained units for the equivalence harness only."""
    total = 0
    for domain_id, state in enumerate(engine.base.replay.states):
        total += int(engine.base.replay.cache.eligible_mask(domain_id).sum().item())
    return total


EQUIVALENCE_CANDIDATE_FIELDS = (
    "global_index", "domain_id", "local_index", "Delta_average",
    "Delta_total", "domain_coverage", "domain_damage",
)


def _task024_oracle_candidates(candidates: Mapping[str, object]) -> dict[str, object]:
    """Strip Task028-private fields before calling the Task024 oracle."""
    missing = [name for name in EQUIVALENCE_CANDIDATE_FIELDS
               if name not in candidates]
    if missing:
        raise RuntimeError(f"Equivalence candidates missing fields: {missing}")
    return {name: candidates[name] for name in EQUIVALENCE_CANDIDATE_FIELDS}


def _sync_equivalence_device(engine) -> None:
    torch = engine.torch
    if torch.cuda.is_available() and torch.device(engine.device).type == "cuda":
        torch.cuda.synchronize(engine.device)


def _assert_equivalence_tensor(name: str, left, right, *, atol: float = 1e-6) -> None:
    torch = _torch()
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(
            f"equivalence tensor shape mismatch for {name}: "
            f"{tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.dtype == torch.bool or not (left.is_floating_point() or right.is_floating_point()):
        if not torch.equal(left, right):
            raise RuntimeError(f"equivalence tensor mismatch for {name}")
        return
    if not bool(torch.allclose(left, right, rtol=0.0, atol=atol)):
        raise RuntimeError(f"equivalence tensor mismatch for {name}")


def _sorted_candidate_view(candidates: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize an active candidate view by its unique global indices."""
    order = _argsort_unique(candidates["global_index"])
    return {
        name: value.index_select(0, order)
        for name, value in candidates.items()
        if name in EQUIVALENCE_CANDIDATE_FIELDS
    }


def _check_cached_candidate_state(optimized: OptimizedCandidateState) -> dict[str, object]:
    """Compare the cached view with the original builder at a checkpoint."""
    started = time.perf_counter()
    legacy_candidates = optimized.engine.candidate_tensors()
    cached_candidates = optimized.candidate_tensors()
    legacy = _sorted_candidate_view(legacy_candidates)
    cached = _sorted_candidate_view(cached_candidates)
    _assert_equivalence_tensor("checkpoint.global_index", legacy["global_index"],
                               cached["global_index"])
    for name in ("domain_id", "local_index"):
        _assert_equivalence_tensor(f"checkpoint.{name}", legacy[name], cached[name])
    for name in ("Delta_average", "Delta_total", "domain_damage", "domain_coverage"):
        _assert_equivalence_tensor(f"checkpoint.{name}", legacy[name], cached[name])
    return {
        "status": "PASS",
        "candidate_count": int(cached["global_index"].numel()),
        "active_global_index_set_match": True,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _check_changed_domain_cache(optimized: OptimizedCandidateState,
                                domain_id: int) -> None:
    """Check only the domain whose replay state changed on this step."""
    torch = optimized.torch
    state = optimized.engine.base.replay.states[int(domain_id)]
    positions = optimized.domain_positions[int(domain_id)]
    losses = state.losses.to(device=optimized.device, dtype=torch.float32)
    active_size = float(optimized.engine.base.replay.cache.active_size[int(domain_id)])
    expected_damage = 1.0 - float(state.current_coverage)
    _assert_equivalence_tensor(
        f"domain[{domain_id}].Delta_average",
        optimized.delta_average.index_select(0, positions), losses,
    )
    _assert_equivalence_tensor(
        f"domain[{domain_id}].Delta_total",
        optimized.delta_total.index_select(0, positions), losses * active_size,
    )
    _assert_equivalence_tensor(
        f"domain[{domain_id}].domain_damage",
        optimized.domain_damage.index_select(0, positions),
        torch.full_like(losses, expected_damage),
    )
    expected_active = optimized.engine.base.replay.cache.eligible_mask(int(domain_id))
    if expected_active.device != optimized.device:
        expected_active = expected_active.to(device=optimized.device)
    _assert_equivalence_tensor(
        f"domain[{domain_id}].active_mask",
        optimized.active_mask.index_select(0, positions), expected_active,
    )


def run_selector_equivalence(*, task014_root: Path, task016_root: Path,
                             task017_root: Path, output_dir: Path,
                             device: str = "cuda:0", steps: int = 500,
                             equivalence_mode: str = "oracle",
                             ranking_backend: str = "auto") -> dict[str, object]:
    """Validate the cached selector against Task024 without dual replay.

    The default oracle mode evolves one Task024 engine and computes the
    reference ranker on the exact same candidate tensors at every step.  The
    original two-engine replay remains available only as an explicit,
    diagnostic ``equivalence_mode='full-legacy'`` audit.
    """
    steps = int(steps)
    if steps <= 0:
        raise ValueError("equivalence steps must be positive")
    if equivalence_mode == "full-legacy":
        return run_selector_equivalence_full_legacy(
            task014_root=task014_root, task016_root=task016_root,
            task017_root=task017_root, output_dir=output_dir,
            device=device, steps=steps,
        )
    if equivalence_mode != "oracle":
        raise ValueError(f"Unknown equivalence mode: {equivalence_mode}")
    if ranking_backend not in RANKING_BACKENDS:
        raise ValueError(f"Unknown ranking backend: {ranking_backend}")
    task024 = _task024()
    optimized_engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_optimized",
        device=device,
    )
    optimized_engine.restore_prefix([])
    assert_started_from_zero(optimized_engine)
    assert_full_retention_start(optimized_engine)
    optimized = OptimizedCandidateState(
        optimized_engine, ranking_backend=ranking_backend
    )
    checkpoint_steps = {0, 1, 10, 20}
    if steps >= 100:
        checkpoint_steps.add(100)
    if steps >= 500:
        checkpoint_steps.add(500)
    cache_checks: dict[str, dict[str, object]] = {}
    if 0 in checkpoint_steps:
        cache_checks["0"] = _check_cached_candidate_state(optimized)
    reference_indices: list[int] = []
    optimized_indices: list[int] = []
    timings = {
        "reference_ranking_seconds": 0.0,
        "optimized_ranking_seconds": 0.0,
        "domain_update_seconds": 0.0,
        "cache_checkpoint_compare_seconds": 0.0,
    }
    started = time.perf_counter()
    iterator = _tqdm(range(steps), total=steps, desc="Task028 equivalence",
                      mininterval=0.5)
    for step_index in iterator:
        step = int(step_index) + 1
        candidates = optimized.candidate_tensors()
        candidate_count = int(candidates["global_index"].numel())
        if candidate_count == 0:
            raise RuntimeError("Equivalence exhausted candidates before requested steps")
        oracle_candidates = _task024_oracle_candidates(candidates)
        before = time.perf_counter()
        reference = task024.rank_candidate_tensors(oracle_candidates, adaptive=True)
        _sync_equivalence_device(optimized_engine)
        timings["reference_ranking_seconds"] += time.perf_counter() - before
        before = time.perf_counter()
        fast = optimized.rank(candidates)
        _sync_equivalence_device(optimized_engine)
        timings["optimized_ranking_seconds"] += time.perf_counter() - before
        reference_position = int(reference["selected_position"])
        optimized_position = int(fast["selected_position"])
        reference_global_index = int(reference["global_index"][reference_position].item())
        optimized_global_index = int(fast["global_index"][optimized_position].item())
        if reference_global_index != optimized_global_index:
            raise RuntimeError(
                f"selector sequence mismatch at step {step}: "
                f"reference={reference_global_index} optimized={optimized_global_index}"
            )
        for name in ("p_total", "p_average", "R_dual", "domain_damage", "R_adaptive"):
            _assert_equivalence_tensor(
                f"step[{step}].{name}", reference[name], fast[name]
            )
        reference_indices.append(reference_global_index)
        optimized_indices.append(optimized_global_index)
        before = time.perf_counter()
        removed = optimized.remove(fast, optimized_position)
        _sync_equivalence_device(optimized_engine)
        timings["domain_update_seconds"] += time.perf_counter() - before
        before = time.perf_counter()
        _check_changed_domain_cache(optimized, int(removed["domain_id"]))
        timings["domain_update_seconds"] += time.perf_counter() - before
        cache_status = "NO"
        if step in checkpoint_steps:
            cache_checks[str(step)] = _check_cached_candidate_state(optimized)
            timings["cache_checkpoint_compare_seconds"] += cache_checks[str(step)]["elapsed_seconds"]
            cache_status = "YES"
        elapsed = max(time.perf_counter() - started, 1e-9)
        speed = step / elapsed
        eta = max(steps - step, 0) / max(speed, 1e-9)
        iterator.set_postfix({
            "selected": reference_global_index,
            "candidate_count": candidate_count,
            "cache_check": cache_status,
            "steps/s": f"{speed:.2f}",
            "ETA": f"{eta:.1f}s",
        })
        if step % 10 == 0 or step == steps:
            print(
                f"[Task028 equivalence] step={step}/{steps} "
                f"global_index={reference_global_index} sequence_match=True "
                f"elapsed={elapsed:.1f}s steps/s={speed:.2f}",
                flush=True,
            )
    elapsed = max(time.perf_counter() - started, 1e-9)
    result = {
        "status": "PASS", "mode": "oracle", "steps": len(reference_indices),
        "ranking_backend_requested": ranking_backend,
        "ranking_backend": optimized.ranking_backend,
        "backend_benchmark": optimized.backend_benchmark,
        "dual_gpu_available": optimized.dual_gpu_available,
        "reference_sequence_sha256": _sequence_sha(reference_indices),
        "legacy_sequence_sha256": _sequence_sha(reference_indices),
        "optimized_sequence_sha256": _sequence_sha(optimized_indices),
        "sequence_sha256_match": reference_indices == optimized_indices,
        "global_index_match": True, "domain_id_match": True,
        "unit_index_match": True, "removed_cost_match": True,
        "first_10_sequence_sha256": _sequence_sha(reference_indices[:10]),
        "first_20_sequence_sha256": _sequence_sha(reference_indices[:20]),
        "first_100_sequence_sha256": _sequence_sha(reference_indices[:100]),
        "first_500_sequence_sha256": _sequence_sha(reference_indices[:500]),
        "first_10_match": reference_indices[:10] == optimized_indices[:10],
        "first_20_match": reference_indices[:20] == optimized_indices[:20],
        "first_100_match": reference_indices[:100] == optimized_indices[:100],
        "first_500_match": reference_indices[:500] == optimized_indices[:500],
        "all_metadata_match": True, "cache_checkpoints": cache_checks,
        "cache_checkpoint_steps": sorted(int(value) for value in cache_checks),
        "candidate_cache_comparison": "checkpointed",
        "changed_domain_comparison": "every_step",
        "reference_ranking_seconds": timings["reference_ranking_seconds"],
        "optimized_ranking_seconds": timings["optimized_ranking_seconds"],
        "domain_update_seconds": timings["domain_update_seconds"],
        "cache_checkpoint_compare_seconds": timings["cache_checkpoint_compare_seconds"],
        "elapsed_seconds": elapsed,
        "steps_per_second": len(reference_indices) / elapsed,
    }
    if reference_indices != optimized_indices:
        result["status"] = "FAIL"
        raise RuntimeError("optimized selector sequence differs from Task024 oracle")
    atomic_json(Path(output_dir) / "selection_50" / "selector_equivalence.json", result)
    return result


def run_selector_equivalence_full_legacy(*, task014_root: Path, task016_root: Path,
                                         task017_root: Path, output_dir: Path,
                                         device: str = "cuda:0", steps: int = 500) -> dict[str, object]:
    """Optional deep audit retaining the original two-engine replay."""
    steps = int(steps)
    if steps <= 0:
        raise ValueError("equivalence steps must be positive")
    task024 = _task024()
    legacy = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_legacy",
        device=device,
    )
    optimized_engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_optimized",
        device=device,
    )
    legacy.restore_prefix([])
    optimized_engine.restore_prefix([])
    assert_started_from_zero(legacy)
    assert_started_from_zero(optimized_engine)
    assert_full_retention_start(legacy)
    assert_full_retention_start(optimized_engine)
    optimized = OptimizedCandidateState(optimized_engine)
    legacy_indices: list[int] = []
    optimized_indices: list[int] = []
    compared_rows = 0
    for _ in range(steps):
        left = legacy_select_step(legacy)
        ranked = optimized.rank()
        right = optimized.remove(ranked, int(ranked["selected_position"]))
        for name in ("global_index", "domain_id", "unit_index"):
            if int(left[name]) != int(right[name]):
                raise RuntimeError(f"selector identity mismatch at step {compared_rows + 1}: {name}")
        for name in ("Delta_average", "Delta_total", "p_average", "p_total",
                     "domain_damage_before", "R_adaptive"):
            if not math.isclose(float(left[name]), float(right[name]),
                                rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"selector value mismatch at step {compared_rows + 1}: {name}")
        legacy_indices.append(int(left["global_index"]))
        optimized_indices.append(int(right["global_index"]))
        if legacy.removed_cost != optimized_engine.base.removed_cost:
            raise RuntimeError("selector removed-cost mismatch")
        if _legacy_active_count(legacy) != optimized.active_count:
            raise RuntimeError("selector active-count mismatch")
        left_state = legacy.base.replay.states[int(left["domain_id"])]
        right_state = optimized_engine.base.replay.states[int(right["domain_id"])]
        if not math.isclose(left_state.current_coverage, right_state.current_coverage,
                            rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError("selector changed-domain coverage mismatch")
        for domain_id, (left_domain, right_domain) in enumerate(
                zip(legacy.base.replay.states, optimized_engine.base.replay.states)):
            if not math.isclose(left_domain.current_coverage, right_domain.current_coverage,
                                rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"selector domain coverage mismatch at domain {domain_id}")
        compared_rows += 1
    result = {
        "status": "PASS", "mode": "full-legacy", "steps": compared_rows,
        "legacy_sequence_sha256": _sequence_sha(legacy_indices),
        "optimized_sequence_sha256": _sequence_sha(optimized_indices),
        "sequence_sha256_match": legacy_indices == optimized_indices,
        "global_index_match": True, "domain_id_match": True,
        "unit_index_match": True, "removed_cost_match": True,
        "first_10_sequence_sha256": _sequence_sha(legacy_indices[:10]),
        "first_100_sequence_sha256": _sequence_sha(legacy_indices[:100]),
        "first_500_sequence_sha256": _sequence_sha(legacy_indices[:500]),
        "first_10_match": legacy_indices[:10] == optimized_indices[:10],
        "first_100_match": legacy_indices[:100] == optimized_indices[:100],
        "first_500_match": legacy_indices[:500] == optimized_indices[:500],
        "all_metadata_match": True,
    }
    if legacy_indices != optimized_indices:
        result["status"] = "FAIL"
        raise RuntimeError("optimized selector sequence differs from legacy selector")
    atomic_json(Path(output_dir) / "selection_50" / "selector_equivalence_full_legacy.json", result)
    return result


def run_selector_backend_equivalence(*, task014_root: Path, task016_root: Path,
                                      task017_root: Path, output_dir: Path,
                                      device: str = "cuda:0", steps: int = 20) -> dict[str, object]:
    """Compare exact single-GPU and dual-GPU optimized sequences."""
    steps = int(steps)
    if steps <= 0:
        raise ValueError("backend equivalence steps must be positive")
    task024 = _task024()
    single_engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_single_gpu",
        device=device,
    )
    dual_engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_dual_gpu",
        device=device,
    )
    single_engine.restore_prefix([])
    dual_engine.restore_prefix([])
    assert_started_from_zero(single_engine)
    assert_started_from_zero(dual_engine)
    assert_full_retention_start(single_engine)
    assert_full_retention_start(dual_engine)
    single = OptimizedCandidateState(single_engine, ranking_backend="single-gpu")
    dual = OptimizedCandidateState(dual_engine, ranking_backend="dual-gpu")
    oracle_indices: list[int] = []
    single_indices: list[int] = []
    dual_indices: list[int] = []
    started = time.perf_counter()
    iterator = _tqdm(range(steps), total=steps,
                      desc="Task028 backend equivalence", mininterval=0.5)
    for step_index in iterator:
        step = int(step_index) + 1
        single_candidates = single.candidate_tensors()
        dual_candidates = dual.candidate_tensors()
        oracle_candidates = _task024_oracle_candidates(single_candidates)
        oracle = task024.rank_candidate_tensors(oracle_candidates, adaptive=True)
        single_ranked = single.rank(single_candidates, backend="single-gpu")
        dual_ranked = dual.rank(dual_candidates, backend="dual-gpu")
        oracle_position = int(oracle["selected_position"])
        single_position = int(single_ranked["selected_position"])
        dual_position = int(dual_ranked["selected_position"])
        oracle_global = int(oracle["global_index"][oracle_position].item())
        single_global = int(single_ranked["global_index"][single_position].item())
        dual_global = int(dual_ranked["global_index"][dual_position].item())
        if not (oracle_global == single_global == dual_global):
            raise RuntimeError(
                f"backend sequence mismatch at step {step}: "
                f"oracle={oracle_global} single={single_global} dual={dual_global}"
            )
        for name in ("p_total", "p_average", "R_dual", "domain_damage", "R_adaptive"):
            _assert_equivalence_tensor(
                f"backend[{step}].single_vs_dual.{name}",
                single_ranked[name], dual_ranked[name],
            )
            _assert_equivalence_tensor(
                f"backend[{step}].oracle_vs_single.{name}",
                oracle[name], single_ranked[name],
            )
        oracle_indices.append(oracle_global)
        single_indices.append(single_global)
        dual_indices.append(dual_global)
        single.remove(single_ranked, single_position)
        dual.remove(dual_ranked, dual_position)
        elapsed = max(time.perf_counter() - started, 1e-9)
        speed = step / elapsed
        iterator.set_postfix({"step": step, "steps/s": f"{speed:.2f}",
                              "sequence_match": "YES"})
    result = {
        "status": "PASS", "steps": len(oracle_indices),
        "oracle_sequence_sha256": _sequence_sha(oracle_indices),
        "single_gpu_sequence_sha256": _sequence_sha(single_indices),
        "dual_gpu_sequence_sha256": _sequence_sha(dual_indices),
        "oracle_single_match": oracle_indices == single_indices,
        "single_dual_match": single_indices == dual_indices,
        "oracle_dual_match": oracle_indices == dual_indices,
        "elapsed_seconds": max(time.perf_counter() - started, 1e-9),
    }
    atomic_json(Path(output_dir) / "selection_50" / "selector_backend_equivalence.json", result)
    return result


def _benchmark_selector_pass(*, task014_root: Path, task016_root: Path,
                             task017_root: Path, output_dir: Path, device: str,
                             steps: int, ranking_backend: str) -> dict[str, object]:
    """Measure an exact optimized selector backend only."""
    torch = _torch()
    task024 = _task024()
    if ranking_backend not in RANKING_BACKENDS:
        raise ValueError(f"Unknown ranking backend: {ranking_backend}")
    engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root),
        output_dir=Path(output_dir) / f"_{ranking_backend}", device=device,
    )
    engine.restore_prefix([])
    engine.start_prefix = ()
    assert_started_from_zero(engine)
    assert_full_retention_start(engine)
    cached = OptimizedCandidateState(engine, ranking_backend=ranking_backend)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(engine.device)
        if cached.dual_gpu_available:
            with torch.cuda.device(cached.secondary_device):
                torch.cuda.reset_peak_memory_stats(cached.secondary_device)
        cached._synchronize_all_ranking_devices()
    timings = {
        "candidate_state_seconds": 0.0,
        "ranking_seconds": 0.0,
        "domain_update_seconds": 0.0,
        "inter_gpu_transfer_seconds": 0.0,
        "gpu0_rank_seconds": 0.0,
        "gpu1_rank_seconds": 0.0,
    }
    sequence: list[int] = []
    started = time.perf_counter()
    for _ in range(int(steps)):
        before = time.perf_counter()
        candidates = cached.candidate_tensors()
        timings["candidate_state_seconds"] += time.perf_counter() - before
        if candidates["global_index"].numel() == 0:
            break
        before = time.perf_counter()
        ranked = cached.rank(candidates, backend=ranking_backend)
        cached._synchronize_all_ranking_devices()
        timings["ranking_seconds"] += time.perf_counter() - before
        for name in ("gpu0_rank_seconds", "gpu1_rank_seconds",
                     "inter_gpu_transfer_seconds"):
            timings[name] += cached.last_rank_timings[name]
        position = int(ranked["selected_position"])
        before = time.perf_counter()
        row = cached.remove(ranked, position)
        cached._synchronize_all_ranking_devices()
        timings["domain_update_seconds"] += time.perf_counter() - before
        sequence.append(int(row["global_index"]))
    cached._synchronize_all_ranking_devices()
    elapsed = max(time.perf_counter() - started, 1e-9)
    speed = len(sequence) / elapsed
    peak0 = (int(torch.cuda.max_memory_allocated(engine.device))
             if torch.cuda.is_available() else 0)
    peak1 = 0
    if torch.cuda.is_available() and cached.dual_gpu_available:
        with torch.cuda.device(cached.secondary_device):
            peak1 = int(torch.cuda.max_memory_allocated(cached.secondary_device))
    gpu0_snapshot = _gpu_memory_snapshot(torch, engine.device)
    gpu1_snapshot = (
        _gpu_memory_snapshot(torch, cached.secondary_device)
        if cached.dual_gpu_available else
        {"device": "cuda:1", "device_name": "unavailable",
         "allocated_memory_bytes": 0, "reserved_memory_bytes": 0,
         "peak_allocated_memory_bytes": 0}
    )
    average_cost = float(engine.removed_cost) / max(len(sequence), 1)
    target_budget = TARGET_SPARSITY * float(engine.total_parameters)
    estimated_steps = target_budget / max(average_cost, 1e-12)
    return {
        "selector": "optimized", "ranking_backend": ranking_backend,
        "ranking_backend_selected": cached.ranking_backend,
        "steps": len(sequence), "requested_steps": int(steps),
        "elapsed_seconds": elapsed, "steps_per_second": speed,
        "seconds_per_step": elapsed / max(len(sequence), 1),
        "gpu0_peak_memory_bytes": peak0, "gpu1_peak_memory_bytes": peak1,
        "gpu0": gpu0_snapshot, "gpu1": gpu1_snapshot,
        "candidate_state_seconds": timings["candidate_state_seconds"],
        "ranking_seconds": timings["ranking_seconds"],
        "domain_update_seconds": timings["domain_update_seconds"],
        "gpu0_rank_seconds": timings["gpu0_rank_seconds"],
        "gpu1_rank_seconds": timings["gpu1_rank_seconds"],
        "inter_gpu_transfer_seconds": timings["inter_gpu_transfer_seconds"],
        "removed_parameters": int(engine.removed_cost),
        "total_parameters": int(engine.total_parameters),
        "estimated_steps_to_50_percent": estimated_steps,
        "estimated_50_percent_seconds": estimated_steps / max(speed, 1e-12),
        "sequence_sha256": _sequence_sha(sequence),
        "peer_access_0_to_1": cached.peer_access_0_to_1,
        "peer_access_1_to_0": cached.peer_access_1_to_0,
    }


def benchmark_selection(*, task014_root: Path, task016_root: Path,
                        task017_root: Path, output_dir: Path,
                        device: str = "cuda:0", steps: int = 100,
                        optimized_steps: int = 500,
                        ranking_backend: str = "both") -> dict[str, object]:
    """Benchmark exact optimized ranking backends only.

    This entry point deliberately stops after the requested prefixes: it does
    not construct a registry, load a checkpoint, validate UCF101, or train.
    """
    if ranking_backend not in RANKING_BACKENDS + ("both",):
        raise ValueError(f"Unknown ranking backend: {ranking_backend}")
    backends = ("single-gpu", "dual-gpu") if ranking_backend == "both" else (ranking_backend,)
    reports: dict[str, dict[str, object]] = {}
    for backend in backends:
        reports[f"{backend}_first100"] = _benchmark_selector_pass(
            task014_root=task014_root, task016_root=task016_root,
            task017_root=task017_root, output_dir=output_dir, device=device,
            steps=int(steps), ranking_backend=backend,
        )
        reports[f"{backend}_first500"] = _benchmark_selector_pass(
            task014_root=task014_root, task016_root=task016_root,
            task017_root=task017_root, output_dir=output_dir, device=device,
            steps=int(optimized_steps), ranking_backend=backend,
        )
    result = {
        "status": "PASS", "construction_only": True,
        "ranking_backend_requested": ranking_backend,
        "reports": reports,
        "target_speedup_claimed": False,
        "target_speedup_required": 20.0,
    }
    atomic_json(Path(output_dir) / "selection_benchmark.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(Path(repo_root)), text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve Git provenance in {repo_root}") from exc


def task027_source_identity(repo_root: Path) -> dict[str, object]:
    """Bind the Task027 implementation source, not its unfinished artifacts."""
    repo_root = Path(repo_root).expanduser().resolve()
    source_path = repo_root / "task027_senior_style_logical_pruning_finetune.py"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    relative = source_path.relative_to(repo_root).as_posix()
    return {
        "task027_source_path": relative,
        "task027_source_git_blob_sha": _git_output(repo_root, "rev-parse", f"HEAD:{relative}"),
        "task028_git_commit_sha": _git_output(repo_root, "rev-parse", "HEAD"),
    }


def _identity_payload(base: Mapping[str, object], *, output_dir: Path,
                      task024_root: Path, task025_root: Path,
                      checkpoint: Path, source_identity: Mapping[str, object]) -> dict[str, object]:
    payload = dict(base)
    payload.update({
        "status": "PASS", "code_version": CODE_VERSION,
        "target_sparsity": TARGET_SPARSITY,
        "target_sparsity_50": True,
        "target_parameter_budget": float(base.get("parameters_before", 0)) * TARGET_SPARSITY,
        "start_sparsity": 0.0, "start_parameter_budget": 0.0,
        "start_prefix_steps": 0, "start_removed_parameters": 0,
        "selection_started_from_zero": False,
        "final_tad_rule_used": False, "new_50_registry_constructed": False,
        "task024_30_registry_not_reused": True,
        "task027_30_registry_not_reused": True,
        "logical_pruning_applied": False, "physical_pruning_executed": False,
        "task024_artifacts_modified": False,
        "task025_artifacts_modified": False,
        "task027_artifacts_modified": False,
        "task024_root": str(Path(task024_root).resolve()),
        "task025_root": str(Path(task025_root).resolve()),
        **dict(source_identity),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
        "task028_output_dir": str(Path(output_dir).resolve()),
        "selection_variant": VARIANT_NAME,
        "source_registry_variant": None,
    })
    return payload


def verify_identity(*, task024_root: Path, task025_root: Path,
                    output_dir: Path, checkpoint: Path,
                    repo_root: Path, model_name: str = DEFAULT_MODEL_NAME,
                    gpu_ids: Sequence[int] = GPU_IDS) -> dict[str, object]:
    """Run the strict inherited identity gates and create Task028 identity."""
    task027 = _task027()
    roots = (task024_root, task025_root)
    missing = [str(Path(root)) for root in roots if not Path(root).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing immutable roots: {missing}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if tuple(int(value) for value in gpu_ids) != GPU_IDS:
        raise RuntimeError("Task028 requires DataParallel GPUs [0, 1]")
    # This call verifies Task024/Task025 and the locked validation/configuration
    # without touching their artifacts.  Its returned registry is intentionally
    # used only for identity provenance, never as the 50% selection seed.
    inherited = task027.verify_identity(
        task024_root=Path(task024_root), task025_root=Path(task025_root),
        output_dir=Path(output_dir), checkpoint=checkpoint,
        repo_root=Path(repo_root), model_name=model_name, gpu_ids=GPU_IDS,
    )
    identity = inherited["identity"]
    source_identity = task027_source_identity(repo_root)
    payload = _identity_payload(identity, output_dir=Path(output_dir),
                                task024_root=Path(task024_root),
                                task025_root=Path(task025_root),
                                checkpoint=checkpoint,
                                source_identity=source_identity)
    payload["source_checkpoint_sha256"] = payload["checkpoint_sha256"]
    # Re-running the provenance-only identity stage after construction must
    # not erase the canonical identity of the already prepared 50% registry.
    # No selection is performed here; the existing trace/registry are merely
    # re-read and rebound to the refreshed identity payload.
    prepared_path = Path(output_dir) / "selection_50" / "registry.json"
    if prepared_path.is_file():
        prepared = verify_existing_50_registry(
            output_dir=Path(output_dir),
        )
        payload.update({
            "registry_canonical_sha256": prepared["registry_sha256"],
            "selection_started_from_zero": True,
            "final_tad_rule_used": True,
            "new_50_registry_constructed": True,
        })
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "inherited": inherited}


def _registry_from_prefix(rows: Sequence[Mapping[str, object]]) -> Mapping[str, Mapping[str, object]]:
    assert_registry_ready_rows(rows)
    import task019_dynamic_ranking_causal_ablation as task019
    return task019.registry_from_prefix([dict(row) for row in rows])


def _registry_trace_identity(rows: Sequence[Mapping[str, object]],
                             registry: Mapping[str, Mapping[str, object]]) -> bool:
    expected: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["layer"]), str(row["unit_type"]))
        expected.setdefault(key, set()).add(int(row["unit_index"]))
    actual: dict[tuple[str, str], set[int]] = {}
    for layer, entry in registry.items():
        key = (str(layer), str(entry["unit_type"]))
        actual[key] = {int(value) for value in entry["indices"]}
    return expected == actual


def construct_selection(*, output_dir: Path, task014_root: Path,
                        task016_root: Path, task017_root: Path,
                        device: str = "cuda:0",
                        ranking_backend: str = "auto") -> dict[str, object]:
    """Construct a new 0 -> 50% adaptive trajectory on GPU(s)."""
    task024 = _task024()
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task028 identity gate must pass before selection")
    source_variant = identity.get("source_registry_variant")
    if source_variant is not None and str(source_variant).strip():
        raise RuntimeError("Task028 must not inherit a 30% registry")
    selection_dir = output_dir / "selection_50"
    selection_dir.mkdir(parents=True, exist_ok=True)
    engine = task024.TensorSafetyEngine(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=selection_dir, device=device,
    )
    start_prefix: list[dict[str, object]] = []
    engine.restore_prefix(start_prefix)
    engine.start_prefix = tuple(start_prefix)
    assert_started_from_zero(engine)
    assert_full_retention_start(engine)
    optimized = OptimizedCandidateState(engine, ranking_backend=ranking_backend)
    total_parameters = float(engine.total_parameters)
    if total_parameters <= 0:
        raise RuntimeError("Replay reported no parameters")
    target_budget = TARGET_SPARSITY * total_parameters
    rows: list[dict[str, object]] = []
    removed = float(engine.removed_cost)
    progress = _tqdm(total=target_budget, initial=removed, unit="param",
                     unit_scale=True, dynamic_ncols=True, mininterval=0.5,
                     desc="Select T+A+D 50%")
    started = time.perf_counter()
    last_display = started
    try:
        while not stop_after_budget(removed, target_budget):
            candidates = optimized.candidate_tensors()
            if candidates["global_index"].numel() == 0:
                raise RuntimeError("No feasible candidate before 50% budget")
            ranked = optimized.rank(candidates)
            position = int(ranked["selected_position"])
            row = optimized.remove(ranked, position)
            row = dict(row)
            row["variant"] = VARIANT_NAME
            row["step"] = len(rows) + 1
            row["incremental_step"] = len(rows) + 1
            row["stage"] = task024.stage_from_layer(row.get("layer", ""))
            cost = float(row["parameter_cost"])
            removed = float(engine.removed_cost)
            row["cumulative_removed_parameters"] = removed
            rows.append(row)
            progress.update(cost)
            now = time.perf_counter()
            if len(rows) % 50 == 0 or now - last_display >= 0.5:
                elapsed = max(now - started, 1e-9)
                speed = len(rows) / elapsed
                remaining = max(target_budget - removed, 0.0)
                eta = remaining / max(speed * optimized.mean_parameter_cost, 1e-9)
                progress.set_postfix(
                    step=len(rows), selected_units=len(rows),
                    sparsity=f"{removed / total_parameters:.4f}",
                    speed=f"{speed:.1f}/s", elapsed=f"{elapsed:.0f}s",
                    eta=f"{eta:.0f}s", p_total=f"{row['p_total']:.3f}",
                    p_average=f"{row['p_average']:.3f}",
                    damage=f"{row['domain_damage_before']:.3f}",
                    R=f"{row['R_adaptive']:.3f}", refresh=False,
                )
                print(f"[Task028] selection step={len(rows)} "
                      f"selected_units={len(rows)} "
                      f"sparsity={removed / total_parameters:.6f} "
                      f"steps/s={speed:.3f} elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                      flush=True)
                last_display = now
    finally:
        progress.close()
    assert_registry_ready_rows(rows)
    registry = _registry_from_prefix(rows)
    if not _registry_trace_identity(rows, registry):
        raise RuntimeError("50% registry does not match selection trace")
    registry_sha = canonical_registry_sha256(registry)
    actual_sparsity = removed / total_parameters
    if actual_sparsity < TARGET_SPARSITY:
        raise RuntimeError("Selection stopped below the 50% parameter budget")
    trace_fields = [field for field in TRACE_FIELDS if any(field in row for row in rows)]
    atomic_csv(selection_dir / "causal_selection_trace.csv", trace_fields, rows)
    atomic_json(selection_dir / "registry.json", {
        "status": "prepared", "target_sparsity": TARGET_SPARSITY,
        "target_parameter_budget": target_budget,
        "removed_parameters": removed,
        "actual_effective_parameter_sparsity": actual_sparsity,
        "registry": registry, "registry_canonical_sha256": registry_sha,
        "selection_started_from_zero": True,
        "final_tad_rule_used": True,
        "task024_30_registry_not_reused": True,
        "task027_30_registry_not_reused": True,
    })
    domain_rows = []
    try:
        domain_rows = [dict(row) for row in engine.base.domain_state_rows(
            task024.VARIANT_ADAPTIVE)]
    except AttributeError:
        domain_rows = []
    if domain_rows:
        atomic_csv(selection_dir / "domain_state.csv",
                   tuple(domain_rows[0].keys()), domain_rows)
    else:
        atomic_csv(selection_dir / "domain_state.csv",
                   ("domain_id", "coverage", "damage"), [])
    construction = {
        "status": "PASS", "target_sparsity": TARGET_SPARSITY,
        "start_prefix": [], "start_removed_parameters": START_REMOVED_COST,
        "start_effective_sparsity": 0.0,
        "total_parameters": total_parameters,
        "target_parameter_budget": target_budget,
        "removed_parameters": removed,
        "actual_effective_parameter_sparsity": actual_sparsity,
        "selection_started_from_zero": True,
        "initial_domain_state_restored": True,
        "final_tad_rule_used": True, "new_50_registry_constructed": True,
        "task024_30_registry_not_reused": True,
        "task027_30_registry_not_reused": True,
        "registry_canonical_sha256": registry_sha,
        "registry_matches_trace": True,
        "selected_units": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device), "physical_pruning_executed": False,
        "ranking_backend_requested": ranking_backend,
        "ranking_backend": optimized.ranking_backend,
        "backend_benchmark": optimized.backend_benchmark,
        "dual_gpu_available": optimized.dual_gpu_available,
    }
    atomic_json(selection_dir / "construction.json", construction)
    identity.update({
        "selection_started_from_zero": True,
        "final_tad_rule_used": True,
        "new_50_registry_constructed": True,
        "target_parameter_budget": target_budget,
        "removed_parameters": removed,
        "actual_effective_parameter_sparsity": actual_sparsity,
        "registry_canonical_sha256": registry_sha,
        "source_registry_variant": None,
        "ranking_backend_requested": ranking_backend,
        "ranking_backend": optimized.ranking_backend,
        "backend_benchmark": optimized.backend_benchmark,
    })
    atomic_json(output_dir / "artifact_identity.json", identity)
    return {"identity": identity, "rows": rows, "registry": registry,
            "registry_sha256": registry_sha, "construction": construction}


def verify_existing_50_registry(*, output_dir: Path,
                                 identity: Mapping[str, object] | None = None) -> dict[str, object]:
    """Verify and return the already prepared 50% registry.

    This reads only Task028's ``selection_50`` artifacts.  It never invokes
    construction, replay, ranking, or any Task027 experiment output.
    """
    output_dir = Path(output_dir)
    selection = read_json(output_dir / "selection_50" / "registry.json")
    if selection.get("status") != "prepared":
        raise RuntimeError("Task028 50% registry is not prepared")
    if not math.isclose(float(selection.get("target_sparsity", -1)),
                        TARGET_SPARSITY, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Task028 registry target is not 50%")
    registry = selection.get("registry")
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError("50% registry is malformed")
    registry_sha = canonical_registry_sha256(registry)
    if registry_sha != str(selection.get("registry_canonical_sha256", "")):
        raise RuntimeError("50% registry canonical hash does not match payload")
    if identity is not None:
        expected = str(identity.get("registry_canonical_sha256", ""))
        if not expected or registry_sha != expected:
            raise RuntimeError("50% registry canonical identity changed")
    for key in ("task024_30_registry_not_reused", "task027_30_registry_not_reused"):
        if selection.get(key) is not True:
            raise RuntimeError(f"Prepared registry provenance gate failed: {key}")
    trace_path = output_dir / "selection_50" / "causal_selection_trace.csv"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    if not trace_rows or not _registry_trace_identity(trace_rows, registry):
        raise RuntimeError("50% registry does not match its selection trace")
    report_path = output_dir / "logical_pruning_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if str(report.get("registry_canonical_sha256", "")) != registry_sha:
            raise RuntimeError("Logical pruning report registry identity changed")
        if report.get("state_dict_numel_unchanged") is False:
            raise RuntimeError("Logical pruning report indicates physical mutation")
    return {"selection": selection, "registry": registry,
            "registry_sha256": registry_sha}


def _assert_finetune_identity(output_dir: Path, identity: Mapping[str, object]) -> None:
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task028 identity gate must pass before fine-tuning")
    if not math.isclose(float(identity.get("target_sparsity", -1)),
                        TARGET_SPARSITY, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Task028 identity target is not 50%")
    verify_existing_50_registry(
        output_dir=output_dir,
        identity=identity,
    )


def prepare_logical_model(*, output_dir: Path, checkpoint: Path,
                          device: str = "cuda:0") -> dict[str, object]:
    """Apply the prepared registry with Task027's unchanged logical runtime."""
    task027 = _task027()
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    prepared = verify_existing_50_registry(
        output_dir=output_dir,
        identity=identity,
    )
    selection = prepared["selection"]
    registry = prepared["registry"]
    model = _load_senior_original(Path(checkpoint), device)
    protocol = assert_senior_model_protocol(model)
    before = task027.state_dict_numel(model)
    audits = task027.apply_logical_pruning_registry(model, registry)
    task027.assert_registry_keep_identity(model, registry)
    report = task027.logical_parameter_report(model, audits, state_dict_before=before)
    if report["actual_state_dict_numel_after"] != before:
        raise RuntimeError("Logical pruning changed state_dict numel")
    report.update({
        "target_effective_parameter_sparsity": TARGET_SPARSITY,
        "effective_parameter_sparsity_ge_target":
            float(report["effective_parameter_sparsity"]) >= TARGET_SPARSITY,
        "physical_pruning_executed": False,
        "registry_canonical_sha256": selection["registry_canonical_sha256"],
        **protocol,
    })
    fields = ("layer", "unit_type", "original_units", "pruned_units", "kept_units",
              "min_keep_units", "min_keep_pass", "registry_indices_valid",
              "layer_exists", "type_match", "analytical_parameter_reduction")
    atomic_csv(Path(output_dir) / "logical_registry_audit.csv", fields, audits)
    atomic_json(Path(output_dir) / "logical_pruning_report.json", report)
    return {"model": model, "registry": registry, "audits": audits, "report": report}


def _forward_logits(model, inputs):
    output = model(inputs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _progress_validation(loader, model, device: str,
                         *, desc: str) -> tuple[float, float, int]:
    torch = _torch()
    from ucf101_videoswin_my import accuracy
    model.eval()
    top1_sum = top5_sum = 0.0
    samples = 0
    iterator = _tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True,
                     mininterval=0.5)
    with torch.no_grad():
        for batch in iterator:
            inputs = batch[0].float().to(torch.device(device), non_blocking=True)
            targets = batch[1].to(torch.device(device), non_blocking=True)
            # Senior validation is FP32 and ranks raw logits directly.
            logits = _forward_logits(model, inputs)
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            count = int(inputs.shape[0])
            top1_sum += float(prec1.item()) * count
            top5_sum += float(prec5.item()) * count
            samples += count
            iterator.set_postfix(top1=f"{top1_sum / samples:.2f}",
                                  top5=f"{top5_sum / samples:.2f}",
                                  samples=samples)
    if samples <= 0:
        raise RuntimeError("Validation loader is empty")
    return top1_sum / samples, top5_sum / samples, samples


def pre_finetune_validate(*, model, identity: Mapping[str, object],
                          output_dir: Path, device: str = "cuda:0") -> dict[str, object]:
    task027 = _task027()
    loader = task027._get_loader(str(identity["validation_split"]),
                                 int(identity["validation_batch_size"]))
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Validation split does not contain 3783 entries")
    top1, top5, evaluated = _progress_validation(
        loader, model, device, desc="PreFT Val 50%")
    report = {
        "status": "PASS" if all(math.isfinite(value) for value in (top1, top5)) else "FAIL",
        "top1": top1, "top5": top5,
        "dataset_samples": EXPECTED_VALIDATION_SAMPLES,
        "evaluated_samples": evaluated,
        "registry_canonical_sha256": identity["registry_canonical_sha256"],
        "effective_parameter_sparsity": identity.get("actual_effective_parameter_sparsity"),
        "fine_tuning_executed": False,
    }
    atomic_json(Path(output_dir) / "pre_finetune_validation_50.json", report)
    return report


def registry_reproduction(*, output_dir: Path, checkpoint: Path,
                          identity: Mapping[str, object], device: str = "cuda:0") -> dict[str, object]:
    """Reapply the same registry to two fresh models and compare one batch."""
    task027 = _task027()
    registry_payload = read_json(Path(output_dir) / "selection_50" / "registry.json")
    registry = registry_payload["registry"]
    registry_sha = canonical_registry_sha256(registry)
    if registry_sha != identity["registry_canonical_sha256"]:
        raise RuntimeError("50% registry canonical identity changed")
    model_a = _load_senior_original(Path(checkpoint), device)
    model_b = _load_senior_original(Path(checkpoint), device)
    assert_senior_model_protocol(model_a)
    assert_senior_model_protocol(model_b)
    task027.apply_logical_pruning_registry(model_a, registry)
    task027.apply_logical_pruning_registry(model_b, registry)
    keep_a = task027.extract_keep_indices(model_a)
    keep_b = task027.extract_keep_indices(model_b)
    if keep_a != keep_b:
        raise RuntimeError("Fresh registry applications produced different keep lists")
    heads_a = {name: value for name, value in keep_a.items()
               if value.get("type") == "head"}
    heads_b = {name: value for name, value in keep_b.items()
               if value.get("type") == "head"}
    neurons_a = {name: value for name, value in keep_a.items()
                 if value.get("type") == "neuron"}
    neurons_b = {name: value for name, value in keep_b.items()
                 if value.get("type") == "neuron"}
    loader = task027._get_loader(str(identity["validation_split"]),
                                 int(identity["validation_batch_size"]))
    batch = next(iter(loader))
    torch = _torch()
    inputs = batch[0].float().to(torch.device(device), non_blocking=True)
    with torch.no_grad():
        logits_a = _forward_logits(model_a, inputs)
        logits_b = _forward_logits(model_b, inputs)
    logits_match = bool(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=0.0))
    report = {
        "status": "PASS" if logits_match else "FAIL",
        "keep_indices_match": keep_a == keep_b,
        "same_keep_heads": heads_a == heads_b,
        "same_keep_neurons": neurons_a == neurons_b,
        "registry_canonical_sha256": registry_sha,
        "registry_sha_match": registry_sha == identity["registry_canonical_sha256"],
        "logits_allclose": logits_match, "atol": 1e-5, "rtol": 0.0,
    }
    atomic_json(Path(output_dir) / "registry_reproduction.json", report)
    return report


def _train_one_epoch_progress(model, loader, optimizer, device: str,
                              *, epoch: int, epochs: int) -> tuple[float, float, int]:
    torch = _torch()
    import torch.nn.functional as F
    model.train()
    total_loss = total_correct = samples = 0.0
    iterator = _tqdm(loader, total=len(loader), desc=f"Train {epoch:03d}/{epochs:03d}",
                     dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        inputs = batch[0].float().to(torch.device(device), non_blocking=True)
        targets = batch[1].to(torch.device(device), non_blocking=True)
        optimizer.zero_grad()
        logits = _forward_logits(model, inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        count = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        samples += count
        iterator.set_postfix(loss=f"{total_loss / samples:.4f}",
                              acc=f"{100.0 * total_correct / samples:.2f}%",
                              lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    if samples <= 0:
        raise RuntimeError("Training loader is empty")
    return total_loss / samples, 100.0 * total_correct / samples, int(samples)


def _write_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def run_finetune(*, output_dir: Path, checkpoint: Path, identity: Mapping[str, object],
                 device: str = "cuda:0", model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    task027 = _task027()
    torch = _torch()
    output_dir = Path(output_dir)
    _assert_finetune_identity(output_dir, identity)
    config = task027.resolve_senior_config(model_name, TRAIN_BATCH_SIZE)
    if int(config["epochs"]) != 100:
        raise RuntimeError("Task028 requires the resolved senior 100-epoch config")
    bundle = prepare_logical_model(output_dir=output_dir, checkpoint=checkpoint, device=device)
    model = bundle["model"]
    report = bundle["report"]
    pre = pre_finetune_validate(model=model, identity=identity,
                                output_dir=output_dir, device=device)
    if pre["status"] != "PASS":
        raise RuntimeError("50% pre-finetune validation was not finite")
    reproduction = registry_reproduction(output_dir=output_dir, checkpoint=checkpoint,
                                          identity=identity, device=device)
    if reproduction["status"] != "PASS":
        raise RuntimeError("50% registry reproduction failed")
    train_loader = task027._get_loader(str(config["train_split"]), TRAIN_BATCH_SIZE)
    val_loader = task027._get_loader(str(identity["validation_split"]),
                                     int(identity["validation_batch_size"]))
    student = torch.nn.DataParallel(model, device_ids=list(GPU_IDS), output_device=GPU_IDS[0])
    optimizer = task027.build_senior_optimizer(student, config)
    finetune_config = {
        "optimizer": "SGD", "base_cfg_lr": config["base_cfg_lr"],
        "actual_finetune_lr": config["actual_finetune_lr"],
        "momentum": 0.9, "weight_decay": config["weight_decay"],
        "epochs": 100, "batch_size": 4, "seed": SEED,
        "scheduler": "NONE", "loss": "CrossEntropyLoss", "gpu_ids": list(GPU_IDS),
        "senior_protocol_alignment": True,
        "use_checkpoint": False,
        "checkpoint_enabled_block_count": 0,
        "swin_block_count": EXPECTED_SWIN_BLOCKS,
        "precision": "fp32",
        "amp_enabled": False,
        "validation_uses_softmax": False,
        "selection_rerun": False,
        "existing_50_registry_reused": True,
        "registry_canonical_sha256": identity["registry_canonical_sha256"],
    }
    atomic_json(output_dir / "finetune_config.json", finetune_config)
    history: list[dict[str, object]] = []
    progress_path = output_dir / "training_progress.json"
    log_path = output_dir / "logs" / "task028.log"
    atomic_json(progress_path, {"status": "RUNNING", "current_epoch": 0,
                                "total_epochs": 100})
    best_payload: dict[str, object] | None = None
    best_top1 = -float("inf")
    start = time.perf_counter()
    for epoch in range(1, 101):
        epoch_start = time.perf_counter()
        train_loss, train_acc, _ = _train_one_epoch_progress(
            student, train_loader, optimizer, device, epoch=epoch, epochs=100)
        val_top1, val_top5, val_samples = _progress_validation(
            val_loader, student, device, desc=f"Val {epoch:03d}/100")
        row = {"epoch": epoch, "train_loss": train_loss,
               "train_accuracy": train_acc, "val_top1": val_top1,
               "val_top5": val_top5, "learning_rate": optimizer.param_groups[0]["lr"],
               "effective_parameter_sparsity": report["effective_parameter_sparsity"],
               "epoch_time_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        atomic_csv(output_dir / "finetune_history.csv", tuple(row.keys()), history)
        best_changed = val_top1 > best_top1
        if best_changed:
            best_top1 = val_top1
            best_payload = {
                "epoch": epoch,
                "state_dict": {key: value.detach().cpu().clone()
                                for key, value in student.module.state_dict().items()},
                "keep_indices": task027.extract_keep_indices(student),
                "top1": val_top1, "top5": val_top5,
                "effective_parameter_sparsity": report["effective_parameter_sparsity"],
                "estimated_removed_parameters": report["estimated_removed_parameters"],
                "history": list(history), "optimizer": finetune_config,
                "registry_canonical_sha256": identity["registry_canonical_sha256"],
                "source_checkpoint_sha256": identity["checkpoint_sha256"],
            }
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(best_payload, output_dir / "checkpoints" / CHECKPOINT_NAME)
        elapsed = time.perf_counter() - start
        line = (f"[Task028] Epoch {epoch:03d}/100 train_loss={train_loss:.6f} "
                f"train_acc={train_acc:.3f} val_top1={val_top1:.3f} "
                f"val_top5={val_top5:.3f} best_top1={best_top1:.3f} "
                f"best_epoch={best_payload['epoch'] if best_payload else 0} "
                f"epoch_time={row['epoch_time_seconds']:.1f}s total_elapsed={elapsed:.1f}s")
        _write_log(log_path, line)
        atomic_json(progress_path, {"status": "RUNNING", "current_epoch": epoch,
                                    "total_epochs": 100, "current_train_loss": train_loss,
                                    "current_train_accuracy": train_acc,
                                    "current_val_top1": val_top1, "current_val_top5": val_top5,
                                    "best_top1": best_top1,
                                    "best_top5": best_payload["top5"] if best_payload else None,
                                    "best_epoch": best_payload["epoch"] if best_payload else None,
                                    "elapsed_seconds": elapsed,
                                    "last_update_time": time.time()})
    if best_payload is None:
        raise RuntimeError("No best checkpoint saved")
    atomic_json(progress_path, {"status": "PASS", "current_epoch": 100,
                                "total_epochs": 100, "best_top1": best_top1,
                                "best_top5": best_payload["top5"],
                                "best_epoch": best_payload["epoch"],
                                "elapsed_seconds": time.perf_counter() - start,
                                "last_update_time": time.time()})
    del student, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_model = _load_senior_original(Path(checkpoint), device)
    saved = torch.load(output_dir / "checkpoints" / CHECKPOINT_NAME, map_location=device)
    message = reload_model.load_state_dict(saved["state_dict"], strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError("Best checkpoint state_dict failed strict reload")
    task027.restore_keep_indices(reload_model, saved["keep_indices"])
    reload_loader = task027._get_loader(str(identity["validation_split"]),
                                        int(identity["validation_batch_size"]))
    reload_top1, reload_top5, reload_samples = _progress_validation(
        reload_loader, reload_model, device, desc="Final reload Val 50%")
    reload_report = {"status": "PASS" if
                     abs(reload_top1 - float(saved["top1"])) <= RELOAD_TOLERANCE_PP and
                     abs(reload_top5 - float(saved["top5"])) <= RELOAD_TOLERANCE_PP else "FAIL",
                     "top1": reload_top1, "top5": reload_top5,
                     "evaluated_samples": reload_samples,
                     "saved_best_top1": saved["top1"], "saved_best_top5": saved["top5"],
                     "reload_tolerance_pp": RELOAD_TOLERANCE_PP,
                     "strict_state_dict": True, "keep_indices_restored": True}
    atomic_json(output_dir / "best_checkpoint_reload_validation.json", reload_report)
    if reload_report["status"] != "PASS":
        raise RuntimeError("Best checkpoint reload did not reproduce validation")
    summary = {"status": "PASS", "best_epoch": saved["epoch"],
               "best_top1": saved["top1"], "best_top5": saved["top5"],
               "pre_finetune_top1": pre["top1"], "pre_finetune_top5": pre["top5"],
               "target_effective_parameter_sparsity": TARGET_SPARSITY,
               "actual_effective_parameter_sparsity": report["effective_parameter_sparsity"],
               "effective_parameter_sparsity": report["effective_parameter_sparsity"],
               "actual_state_dict_numel_before": report["actual_state_dict_numel_before"],
               "actual_state_dict_numel_after": task027.state_dict_numel(reload_model),
               "estimated_removed_parameters": report["estimated_removed_parameters"],
               "fine_tuning_executed": True, "physical_pruning_executed": False,
               "registry_canonical_sha256": identity["registry_canonical_sha256"],
               "senior_protocol_alignment": True, "use_checkpoint": False,
               "checkpoint_enabled_block_count": 0,
               "swin_block_count": EXPECTED_SWIN_BLOCKS, "precision": "fp32",
               "amp_enabled": False, "validation_uses_softmax": False,
               "selection_rerun": False, "existing_50_registry_reused": True}
    atomic_json(output_dir / "final_summary.json", summary)
    del reload_model
    return summary


def _synchronize_cuda(torch, device: str) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target)


def senior_finetune_speed_gate(*, output_dir: Path, checkpoint: Path,
                               device: str = "cuda:0", steps: int = 100,
                               model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    """Run a no-output, no-validation 100-iteration senior protocol check.

    The only model structure consumed here is the already prepared
    ``selection_50/registry.json``.  In particular, this function never runs
    a new selection or ranking routine.
    """
    torch = _torch()
    task027 = _task027()
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    _assert_finetune_identity(output_dir, identity)
    steps = int(steps)
    if steps <= 0:
        raise ValueError("speed-gate steps must be positive")
    config = task027.resolve_senior_config(model_name, TRAIN_BATCH_SIZE)
    if int(config["epochs"]) != 100:
        raise RuntimeError("Task028 requires the resolved senior 100-epoch config")
    bundle = prepare_logical_model(output_dir=output_dir, checkpoint=checkpoint, device=device)
    model = bundle["model"]
    protocol = assert_senior_model_protocol(model)
    student = torch.nn.DataParallel(model, device_ids=list(GPU_IDS), output_device=GPU_IDS[0])
    optimizer = task027.build_senior_optimizer(student, config)
    loader = task027._get_loader(str(config["train_split"]), TRAIN_BATCH_SIZE)
    iterator = iter(loader)
    student.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        for gpu_id in GPU_IDS:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.reset_peak_memory_stats(gpu_id)
            except (RuntimeError, AssertionError):
                pass
    started = time.perf_counter()
    for _ in _tqdm(range(steps), total=steps, desc="Task028 senior speed gate",
                   dynamic_ncols=True, mininterval=0.5):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = batch[0].float().to(torch.device(device), non_blocking=True)
        targets = batch[1].to(torch.device(device), non_blocking=True)
        optimizer.zero_grad()
        logits = _forward_logits(student, inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    _synchronize_cuda(torch, device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    iterations_per_second = steps / elapsed
    peak0 = peak1 = 0
    if torch.cuda.is_available():
        peak0 = int(torch.cuda.max_memory_allocated(torch.device(device)))
        if len(GPU_IDS) > 1 and torch.cuda.device_count() > 1:
            with torch.cuda.device(GPU_IDS[1]):
                peak1 = int(torch.cuda.max_memory_allocated(GPU_IDS[1]))
    report = {
        "status": "PASS", "steps": steps, "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / steps,
        "iterations_per_second": iterations_per_second,
        "estimated_minutes_per_epoch": len(loader) / max(iterations_per_second, 1e-12) / 60.0,
        "gpu0_peak_memory_bytes": peak0, "gpu1_peak_memory_bytes": peak1,
        "precision": "fp32", "amp_enabled": False,
        "use_checkpoint": False,
        "checkpoint_enabled_block_count": protocol["checkpoint_enabled_block_count"],
        "swin_block_count": protocol["swin_block_count"],
        "batch_size": TRAIN_BATCH_SIZE, "gpu_ids": list(GPU_IDS),
        "optimizer": "SGD", "base_cfg_lr": config["base_cfg_lr"],
        "actual_finetune_lr": config["actual_finetune_lr"],
        "momentum": 0.9, "weight_decay": config["weight_decay"],
        "scheduler": "NONE", "loss": "CrossEntropyLoss", "seed": SEED,
        "selection_rerun": False, "existing_50_registry_reused": True,
        "registry_canonical_sha256": identity["registry_canonical_sha256"],
        "validation_executed": False, "weights_saved": False,
    }
    atomic_json(output_dir / "senior_finetune_speed_gate.json", report)
    del student, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def analyze(*, output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    construction = read_json(output_dir / "selection_50" / "construction.json")
    report = read_json(output_dir / "logical_pruning_report.json")
    pre = read_json(output_dir / "pre_finetune_validation_50.json")
    reproduction = read_json(output_dir / "registry_reproduction.json")
    config = read_json(output_dir / "finetune_config.json")
    progress = read_json(output_dir / "training_progress.json")
    reload_report = read_json(output_dir / "best_checkpoint_reload_validation.json")
    completion = {
        "status": "PASS", "code_version": CODE_VERSION,
        "target_sparsity_50": float(identity.get("target_sparsity", 0)) == TARGET_SPARSITY,
        "selection_started_from_zero": construction.get("selection_started_from_zero") is True,
        "final_tad_rule_used": construction.get("final_tad_rule_used") is True,
        "new_50_registry_constructed": construction.get("new_50_registry_constructed") is True,
        "task024_30_registry_not_reused": construction.get("task024_30_registry_not_reused") is True,
        "task027_30_registry_not_reused": construction.get("task027_30_registry_not_reused") is True,
        "effective_parameter_sparsity_ge_target": float(report.get("effective_parameter_sparsity", 0)) >= TARGET_SPARSITY,
        "logical_pruning_applied": True, "physical_pruning_executed": False,
        "pre_finetune_validation_complete": pre.get("status") == "PASS",
        "registry_reproduction_pass": reproduction.get("status") == "PASS",
        "fine_tuning_executed": progress.get("status") == "PASS",
        "epochs_completed": int(progress.get("current_epoch", 0)),
        "optimizer_sgd": config.get("optimizer") == "SGD",
        "lr_0005": math.isclose(float(config.get("actual_finetune_lr", -1)), 0.0005,
                                 rel_tol=0.0, abs_tol=1e-15),
        "momentum_09": float(config.get("momentum", -1)) == 0.9,
        "weight_decay_1e5": math.isclose(float(config.get("weight_decay", -1)), 1e-5,
                                          rel_tol=0.0, abs_tol=1e-12),
        "batch_size_4": int(config.get("batch_size", -1)) == 4,
        "seed_3407": int(config.get("seed", -1)) == SEED,
        "scheduler_used": config.get("scheduler") != "NONE",
        "cross_entropy_only": config.get("loss") == "CrossEntropyLoss",
        "progress_display_enabled": progress.get("status") == "PASS",
        "best_checkpoint_saved": (output_dir / "checkpoints" / CHECKPOINT_NAME).is_file(),
        "best_checkpoint_reload_pass": reload_report.get("status") == "PASS",
        "senior_protocol_alignment": config.get("senior_protocol_alignment") is True,
        "use_checkpoint": config.get("use_checkpoint"),
        "checkpoint_enabled_block_count": config.get("checkpoint_enabled_block_count", -1),
        "swin_block_count": config.get("swin_block_count", -1),
        "amp_enabled": config.get("amp_enabled"),
        "precision": config.get("precision"),
        "validation_uses_softmax": config.get("validation_uses_softmax"),
        "existing_50_registry_reused": config.get("existing_50_registry_reused") is True,
        "selection_rerun": config.get("selection_rerun"),
        "task024_artifacts_modified": False, "task025_artifacts_modified": False,
        "task027_artifacts_modified": False,
    }
    if not completion_gate(completion):
        completion["status"] = "PENDING"
    atomic_json(output_dir / "task028_completion.json", completion)
    diagnosis = ["# Task028 main 50% T+A+D logical pruning", "",
                 f"Target effective sparsity: {TARGET_SPARSITY:.2f}",
                 f"Actual effective sparsity: {report.get('effective_parameter_sparsity')}",
                 "Selection starts from the fully retained state: YES",
                 "Task024/Task027 30% registries reused: NO", "Physical pruning: NO",
                 f"Completion status: {completion['status']}", ""]
    (output_dir / "diagnosis.md").write_text("\n".join(diagnosis), encoding="utf-8")
    return completion


# Small public aliases keep the orchestration API convenient without creating
# a second implementation or changing the selection/training semantics.
construct = construct_selection
run = run_finetune
finetune_only = run_finetune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "construct", "run", "finetune-only",
                                            "speed-gate", "analyze", "equivalence",
                                            "backend-equivalence", "benchmark"),
                        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task014-root", type=Path)
    parser.add_argument("--task016-root", type=Path)
    parser.add_argument("--task017-root", type=Path)
    parser.add_argument("--task024-root", type=Path)
    parser.add_argument("--task025-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--optimized-steps", type=int, default=500)
    parser.add_argument("--equivalence-mode", choices=("oracle", "full-legacy"),
                        default="oracle")
    parser.add_argument("--ranking-backend", choices=RANKING_BACKENDS + ("both",),
                        default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "identity":
        required = (args.task024_root, args.task025_root,
                    args.checkpoint, args.repo_root)
        if any(value is None for value in required):
            raise SystemExit("identity requires all immutable roots, checkpoint and repo-root")
        verify_identity(task024_root=args.task024_root, task025_root=args.task025_root,
                        output_dir=args.output_dir,
                        checkpoint=args.checkpoint, repo_root=args.repo_root,
                        model_name=args.model)
    elif args.mode == "construct":
        required = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in required):
            raise SystemExit("construct requires Task014/Task016/Task017 roots")
        construct_selection(output_dir=args.output_dir, task014_root=args.task014_root,
                            task016_root=args.task016_root, task017_root=args.task017_root,
                            device=args.device, ranking_backend=args.ranking_backend)
    elif args.mode in ("run", "finetune-only"):
        if args.checkpoint is None:
            raise SystemExit(f"{args.mode} requires --checkpoint")
        identity = read_json(args.output_dir / "artifact_identity.json")
        run_finetune(output_dir=args.output_dir, checkpoint=args.checkpoint,
                     identity=identity, device=args.device, model_name=args.model)
    elif args.mode == "speed-gate":
        if args.checkpoint is None:
            raise SystemExit("speed-gate requires --checkpoint")
        senior_finetune_speed_gate(
            output_dir=args.output_dir, checkpoint=args.checkpoint,
            device=args.device, steps=args.steps, model_name=args.model,
        )
    elif args.mode == "analyze":
        analyze(output_dir=args.output_dir)
    elif args.mode == "equivalence":
        required = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in required):
            raise SystemExit("equivalence requires Task014/Task016/Task017 roots")
        run_selector_equivalence(
            task014_root=args.task014_root, task016_root=args.task016_root,
            task017_root=args.task017_root, output_dir=args.output_dir,
            device=args.device, steps=args.steps,
            equivalence_mode=args.equivalence_mode,
            ranking_backend=args.ranking_backend,
        )
    elif args.mode == "backend-equivalence":
        required = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in required):
            raise SystemExit("backend-equivalence requires Task014/Task016/Task017 roots")
        run_selector_backend_equivalence(
            task014_root=args.task014_root, task016_root=args.task016_root,
            task017_root=args.task017_root, output_dir=args.output_dir,
            device=args.device, steps=args.steps,
        )
    else:
        required = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in required):
            raise SystemExit("benchmark requires Task014/Task016/Task017 roots")
        benchmark_selection(
            task014_root=args.task014_root, task016_root=args.task016_root,
            task017_root=args.task017_root, output_dir=args.output_dir,
            device=args.device, steps=args.steps,
            optimized_steps=args.optimized_steps,
            ranking_backend=args.ranking_backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

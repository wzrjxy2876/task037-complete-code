"""Task019 causal ablations for Task016/Task018 domain-total pruning.

This module is deliberately outside the production pruning path.  It consumes
the exact, identity-checked Task016--Task018 artifacts and constructs three
counterfactual registries:

* frozen_26: freeze the feasible domain-total ordering at the 24% state;
* frozen_30: freeze the feasible domain-total ordering at the 28% state;
* no_new_attention_dynamic_30: keep Task016 dynamic updates, but allow only
  FFN candidates after the exact 28% prefix.

Parameter cost never participates in candidate ranking.  A selected unit is
removed first, its exact Task014/Task016 cost is then accumulated, and an
interval stops once the absolute target parameter budget is reached or
exceeded.  Overshoot is therefore intentional and unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from task017_high_sparsity_diagnosis import (
    EXPECTED_DOMAINS,
    EXPECTED_UNITS,
    HighSparsityReplay,
    TYPE_ATTENTION,
    TYPE_FFN,
    sequence_sha256,
    sha256_file,
)
from task018_high_sparsity_transition import (
    _apply_registry,
    read_csv,
    read_json,
    reconstruct_all_prefixes,
    registry_from_prefix,
    sequence_prefix_sha256,
    target_tag,
    verify_saved_endpoint,
)


VARIANTS = {
    "frozen_26": (0.24, 0.26, "frozen"),
    "frozen_30": (0.28, 0.30, "frozen"),
    "no_new_attention_dynamic_30": (0.28, 0.30, "no_new_attention"),
}
PRODUCTION_FILES = (
    "functional_competition_pruning.py",
    "MC.py",
    "ucf101_videoswin_my.py",
)
TRACE_FIELDS = (
    "variant",
    "step",
    "incremental_step",
    "estimated_sparsity_before",
    "estimated_sparsity_after",
    "global_index",
    "unit_type",
    "layer",
    "unit_index",
    "domain_id",
    "functional_score",
    "Delta_average",
    "Delta_total",
    "rank_at_selection",
    "domain_retained_ratio_before",
    "domain_coverage_before",
    "domain_coverage_after",
    "best_substitute_similarity",
    "parameter_cost",
    "cumulative_removed_parameters",
)


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
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_mapping_sha256(values: Mapping[str, int]) -> str:
    payload = json.dumps(
        {str(key): int(value) for key, value in sorted(values.items())},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def production_source_sha256(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root)
    return {name: sha256_file(root / name) for name in PRODUCTION_FILES}


def ordered_selected_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    selected = [dict(row) for row in rows if str(row.get("selected", "true")).lower() != "false"]
    selected.sort(key=lambda row: int(row["step"]))
    expected_steps = list(range(1, len(selected) + 1))
    if [int(row["step"]) for row in selected] != expected_steps:
        raise ValueError("Selection trace steps are not contiguous")
    indices = [int(row["global_index"]) for row in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("Selection trace contains duplicate global indices")
    return selected


def frozen_candidate_order(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Sort once by functional score and deterministic global-index tie break."""
    prepared: list[dict[str, object]] = []
    seen: set[int] = set()
    for row in candidates:
        index = int(row["global_index"])
        score = float(row.get("Delta_total", row.get("functional_score", math.nan)))
        if index in seen or not math.isfinite(score):
            raise ValueError("Frozen candidates must be unique with finite scores")
        seen.add(index)
        prepared.append({**dict(row), "global_index": index, "Delta_total": score})
    return sorted(prepared, key=lambda row: (float(row["Delta_total"]), int(row["global_index"])))


def simulate_frozen_budget_stop(
    candidates: Sequence[Mapping[str, object]],
    *,
    removed_cost: int,
    target_budget: float,
    layer_removed: Mapping[str, int],
    layer_capacities: Mapping[str, int],
) -> list[dict[str, object]]:
    """Pure reference implementation of frozen ordering and budget semantics.

    This helper is used by regression tests.  Costs are read only after a
    candidate has been chosen from the immutable functional ordering.
    """
    cumulative = int(removed_cost)
    pruned = Counter({str(key): int(value) for key, value in layer_removed.items()})
    selected: list[dict[str, object]] = []
    for rank, row in enumerate(frozen_candidate_order(candidates), start=1):
        if cumulative >= target_budget:
            break
        layer = str(row["layer"])
        if pruned[layer] >= int(layer_capacities[layer]):
            continue
        chosen = dict(row)
        chosen["frozen_rank"] = rank
        selected.append(chosen)
        pruned[layer] += 1
        cumulative += int(row["parameter_cost"])
    if cumulative < target_budget:
        raise RuntimeError("Frozen ordering exhausted before reaching parameter budget")
    return selected


def assert_no_new_attention(rows: Sequence[Mapping[str, object]]) -> None:
    offending = [int(row["global_index"]) for row in rows if row["unit_type"] == TYPE_ATTENTION]
    if offending:
        raise RuntimeError(f"No-new-Attention intervention selected Attention: {offending[:5]}")


def simulate_no_new_attention_dynamic(
    candidate_provider: Callable[[Sequence[int]], Sequence[Mapping[str, object]]],
    *,
    removed_cost: int,
    target_budget: float,
    layer_removed: Mapping[str, int],
    layer_capacities: Mapping[str, int],
) -> tuple[list[dict[str, object]], int]:
    """Small pure oracle for no-new-Attention regression tests.

    ``candidate_provider`` is called again after every removal, representing
    unchanged dynamic functional recomputation.  Cost is intentionally absent
    from the ordering key.
    """
    cumulative = int(removed_cost)
    pruned = Counter({str(key): int(value) for key, value in layer_removed.items()})
    selected: list[dict[str, object]] = []
    selected_indices: list[int] = []
    while cumulative < target_budget:
        feasible = []
        for row in candidate_provider(tuple(selected_indices)):
            if row["unit_type"] != TYPE_FFN:
                continue
            layer = str(row["layer"])
            if pruned[layer] >= int(layer_capacities[layer]):
                continue
            score = float(row.get("Delta_total", row.get("functional_score", math.nan)))
            if not math.isfinite(score):
                raise ValueError("Dynamic functional scores must be finite")
            feasible.append(({**dict(row), "Delta_total": score}))
        if not feasible:
            raise RuntimeError("No-new-Attention oracle exhausted feasible FFN candidates")
        chosen = min(feasible, key=lambda row: (float(row["Delta_total"]), int(row["global_index"])))
        index = int(chosen["global_index"])
        if index in selected_indices:
            raise RuntimeError("Dynamic provider returned an already selected unit")
        selected.append(chosen)
        selected_indices.append(index)
        pruned[str(chosen["layer"])] += 1
        cumulative += int(chosen["parameter_cost"])
    assert_no_new_attention(selected)
    return selected, cumulative


def reproduce_interval_sequence(
    authoritative: Sequence[Mapping[str, object]],
    replay: Sequence[Mapping[str, object]],
    start_steps: int,
    target_steps: int,
) -> bool:
    if not (0 <= start_steps <= target_steps):
        raise ValueError("Invalid interval step bounds")
    expected = [int(row["global_index"]) for row in authoritative[start_steps:target_steps]]
    actual = [int(row["global_index"]) for row in replay[start_steps:target_steps]]
    return expected == actual


def _task016_dir(root: Path, target: float) -> Path:
    return Path(root) / "domain_total" / target_tag(target)


def _prefix_identity(
    task016_root: Path, task017_root: Path, task018_root: Path
) -> tuple[list[dict], dict[float, list[dict]], dict[float, dict], dict]:
    trace30_path = _task016_dir(task016_root, 0.30) / "functional_selection_trace.csv"
    trace30 = ordered_selected_rows(read_csv(trace30_path))
    metrics30 = read_json(_task016_dir(task016_root, 0.30) / "final_metrics.json")
    parameters_before = int(metrics30["parameters_before"])
    prefixes, summaries = reconstruct_all_prefixes(trace30, parameters_before)
    by_target = {
        float(summary["target_parameter_sparsity"]): summary for summary in summaries
    }

    replay_path = Path(task017_root) / "replay/domain_total/incremental_selection_risk_full.csv"
    replay = ordered_selected_rows(read_csv(replay_path))
    replay_evidence = read_json(
        Path(task017_root) / "replay/domain_total/replay_evidence.json"
    )
    sequence30 = [int(row["global_index"]) for row in prefixes[0.30]]
    if replay_evidence.get("trace_replay_exact") is not True:
        raise RuntimeError("Task017 trace_replay_exact gate did not pass")
    if replay_evidence.get("expected_sequence_sha256") != sequence_sha256(sequence30):
        raise RuntimeError("Task017 expected sequence differs from Task016")
    if replay_evidence.get("actual_sequence_sha256") != sequence_sha256(
        int(row["global_index"]) for row in replay
    ):
        raise RuntimeError("Task017 replay sequence evidence is stale")

    required = (0.24, 0.26, 0.28, 0.30)
    for target in required:
        registry = read_json(Path(task018_root) / "registries" / f"{target_tag(target)}.json")
        expected_sha = sequence_prefix_sha256(
            int(row["global_index"]) for row in prefixes[target]
        )
        if registry.get("summary", {}).get("sequence_prefix_sha256") != expected_sha:
            raise RuntimeError(f"Task018 {target:.0%} prefix SHA differs from Task016")
        if int(registry["summary"]["prefix_steps"]) != len(prefixes[target]):
            raise RuntimeError(f"Task018 {target:.0%} prefix length mismatch")
    return trace30, prefixes, by_target, {
        "replay": replay,
        "replay_evidence": replay_evidence,
        "trace30_path": trace30_path,
        "parameters_before": parameters_before,
    }


def verify_identity(
    *,
    task014_root: Path,
    task016_root: Path,
    task017_root: Path,
    task018_root: Path,
    output_dir: Path,
    checkpoint: Path,
) -> None:
    """Strictly bind Task019 to the already validated Task016--Task018 run."""
    roots = [Path(value) for value in (task014_root, task016_root, task017_root, task018_root)]
    if any(not root.is_dir() for root in roots):
        raise FileNotFoundError(f"Missing Task019 input root: {[str(p) for p in roots if not p.is_dir()]}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    completion17 = read_json(Path(task017_root) / "task017_completion.json")
    completion18 = read_json(Path(task018_root) / "task018_completion.json")
    if (
        completion17.get("trace_replay_exact") is not True
        or completion17.get("analysis_complete") is not True
        or completion17.get("artifact_identity_pass") is not True
    ):
        raise RuntimeError("Task017 completion gate is incomplete")
    if completion18.get("status") != "PASS" or completion18.get("analysis_complete") is not True:
        raise RuntimeError("Task018 completion gate is incomplete")
    identity17 = read_json(Path(task017_root) / "artifact_identity.json")
    identity18 = read_json(Path(task018_root) / "artifact_identity.json")
    if identity17.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task017 artifact identity did not pass")
    if identity18.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task018 artifact identity did not pass")

    trace30, prefixes, summaries, extra = _prefix_identity(
        Path(task016_root), Path(task017_root), Path(task018_root)
    )
    reference17 = identity17.get("reference", {})
    for name, observed, expected in (
        ("descriptor_units", reference17.get("descriptor_units"), EXPECTED_UNITS),
        ("mapped_units", reference17.get("mapped_units"), EXPECTED_UNITS),
        ("bms_domains", reference17.get("bms_domains"), EXPECTED_DOMAINS),
    ):
        if int(observed) != expected:
            raise RuntimeError(f"Task019 identity mismatch: {name}={observed}")
    if int(identity18["parameters_before"]) != int(extra["parameters_before"]):
        raise RuntimeError("Task018 original parameter count differs from Task016")

    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != identity18["checkpoint_sha256"]:
        raise RuntimeError("Task019 checkpoint differs from Task018")
    if checkpoint_sha != reference17["checkpoint_sha256"]:
        raise RuntimeError("Task019 checkpoint differs from Task017")

    from task015_attention_ffn_diagnosis import (
        _layer_capacities,
        _load_units,
        _mapping_paths,
        infer_parameter_costs,
    )

    unit_path, layer_path, _ = _mapping_paths(Path(task014_root))
    units = _load_units(unit_path)
    costs, _ = infer_parameter_costs(units, read_csv(layer_path))
    capacities = _layer_capacities(units)
    for row in trace30:
        index = int(row["global_index"])
        if int(row["parameter_cost"]) != int(costs[index]):
            raise RuntimeError(f"Task016 parameter cost mismatch at unit {index}")
    if sum(
        len(value["indices"])
        for value in registry_from_prefix(prefixes[0.30]).values()
    ) <= 0:
        raise RuntimeError("Task018 registry reconstruction is empty")

    s24_steps = len(prefixes[0.24])
    s26_steps = len(prefixes[0.26])
    s28_steps = len(prefixes[0.28])
    s30_steps = len(prefixes[0.30])
    reproduction_24_26 = reproduce_interval_sequence(
        trace30, extra["replay"], s24_steps, s26_steps
    )
    reproduction_28_30 = reproduce_interval_sequence(
        trace30, extra["replay"], s28_steps, s30_steps
    )
    if not reproduction_24_26 or not reproduction_28_30:
        raise RuntimeError("Task017 dynamic replay does not reproduce Task016 intervals")

    repo_root = Path(__file__).resolve().parent
    stat = checkpoint.stat()
    payload = {
        "status": "PASS",
        "artifact_identity_pass": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "task014_root": str(Path(task014_root).resolve()),
        "task016_root": str(Path(task016_root).resolve()),
        "task017_root": str(Path(task017_root).resolve()),
        "task018_root": str(Path(task018_root).resolve()),
        "descriptor_units": EXPECTED_UNITS,
        "mapped_units": EXPECTED_UNITS,
        "bms_domains": EXPECTED_DOMAINS,
        "parameters_before": int(extra["parameters_before"]),
        "parameter_cost_sha256": hashlib.sha256(costs.tobytes()).hexdigest(),
        "layer_min_keep_capacity_sha256": canonical_mapping_sha256(capacities),
        "source_trace_sha256": sha256_file(extra["trace30_path"]),
        "task016_s30_sequence_sha256": sequence_sha256(
            int(row["global_index"]) for row in prefixes[0.30]
        ),
        "task017_replay_sequence_sha256": extra["replay_evidence"]["actual_sequence_sha256"],
        "task018_prefix_sequence_sha256": {
            target_tag(target): summaries[target]["sequence_prefix_sha256"]
            for target in (0.24, 0.26, 0.28, 0.30)
        },
        "dynamic_24_26_reproduction_pass": reproduction_24_26,
        "dynamic_28_30_reproduction_pass": reproduction_28_30,
        "validation_split": identity18["validation_split"],
        "validation_batch_size": int(identity18["validation_batch_size"]),
        "amp_enabled": bool(identity18["amp_enabled"]),
        "production_source_sha256": production_source_sha256(repo_root),
        "production_pruning_code_modified": False,
        "budget_semantics": "select_remove_then_accumulate_stop_at_or_above_target_overshoot_allowed",
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)


class CausalAblationEngine:
    """GPU domain-state engine using the unchanged Task016 functional math."""

    def __init__(
        self,
        *,
        task014_root: Path,
        task016_root: Path,
        task017_root: Path,
        output_dir: Path,
        device: str,
    ) -> None:
        # Deliberately bypass HighSparsityReplay.__init__: that constructor
        # rebuilds BMS for Task017's independent replay audit.  Task019 must
        # reuse the already identity-checked Task017 domain membership instead
        # of extracting descriptors or running BMS again.
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Task019 construction requires PyTorch") from error
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Task019 construction requires CUDA")
        from task015_attention_ffn_diagnosis import (
            _layer_capacities,
            _load_units,
            _mapping_paths,
            infer_parameter_costs,
        )

        task014_root, task016_root = Path(task014_root), Path(task016_root)
        unit_path, layer_path, mapping_path = _mapping_paths(task014_root)
        units = _load_units(unit_path)
        snapshot_path = Path(task017_root) / "replay/domain_total/candidate_snapshots.npz"
        with np.load(snapshot_path, allow_pickle=False) as payload:
            globals_ = payload["p00_global_index"].astype(np.int64, copy=True)
            domains = payload["p00_domain_id"].astype(np.int64, copy=True)
        if (
            globals_.shape != (EXPECTED_UNITS,)
            or domains.shape != (EXPECTED_UNITS,)
            or set(globals_.tolist()) != set(range(EXPECTED_UNITS))
            or set(domains.tolist()) != set(range(EXPECTED_DOMAINS))
        ):
            raise ValueError("Task017 cached BMS membership is incomplete")
        groups = [globals_[domains == domain_id].tolist() for domain_id in range(EXPECTED_DOMAINS)]
        if any(not group for group in groups):
            raise ValueError("Task017 cached BMS membership contains an empty domain")

        replay = object.__new__(HighSparsityReplay)
        replay.torch = torch
        replay.mode = "domain_total"
        replay.device = torch.device(device)
        replay.task014_root = task014_root
        replay.task016_root = task016_root
        replay.output_dir = Path(output_dir) / "_task019_internal"
        replay.run_dir = _task016_dir(task016_root, 0.30)
        replay.units = units
        replay.unit_by_global = {unit.global_index: unit for unit in units}
        replay.mapping_path = mapping_path
        replay.groups = groups
        replay.costs, replay.layer_costs = infer_parameter_costs(units, read_csv(layer_path))
        replay.capacities = _layer_capacities(units)
        replay.metrics = read_json(replay.run_dir / "final_metrics.json")
        replay.total_parameters = int(replay.metrics["parameters_before"])
        replay.vector_path = task014_root / "field_cache/aligned_function_fields.npy"
        replay.mask_path = task014_root / "field_cache/aligned_function_valid_mask.npy"
        for path in (replay.vector_path, replay.mask_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        replay.global_to_domain = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        replay.global_to_local = np.full(EXPECTED_UNITS, -1, dtype=np.int64)
        for domain_id, members in enumerate(groups):
            values = np.asarray(members, dtype=np.int64)
            replay.global_to_domain[values] = domain_id
            replay.global_to_local[values] = np.arange(len(values), dtype=np.int64)
        replay.states = []
        replay.cache = None
        self.replay = replay
        self.task017_root = Path(task017_root)
        self.output_dir = Path(output_dir)
        self.torch = self.replay.torch
        self.device = self.replay.device
        self.replay._initialize_states()
        self.removed_cost = 0
        self.steps = 0

    def restore_prefix(self, prefix: Sequence[Mapping[str, object]]) -> None:
        """Restore an exact prefix without re-running thousands of selector steps."""
        from functional_competition_pruning import marginal_coverage_losses

        removed = {int(row["global_index"]) for row in prefix}
        if len(removed) != len(prefix):
            raise ValueError("Start prefix contains duplicate units")
        for state in self.replay.states:
            globals_ = np.asarray(state.global_indices, dtype=np.int64)
            retained = np.asarray([int(value) not in removed for value in globals_], dtype=np.bool_)
            state.retained.copy_(
                self.torch.as_tensor(retained, dtype=self.torch.bool, device=self.device)
            )
            state.version = int((~retained).sum())
            if retained.any():
                state.losses, coverage = marginal_coverage_losses(
                    state.similarity, state.retained, state.valid_function_mask
                )
                state.current_coverage = float(coverage.item())
            else:
                state.losses = self.torch.full_like(state.losses, self.torch.inf)
                state.current_coverage = (
                    1.0 if not bool(state.valid_function_mask.any().item()) else 0.0
                )
        self.replay._initialize_cache()
        self.replay.cache.pruned[:] = 0
        layer_removed = Counter(self.replay.units[index].layer for index in removed)
        for layer, count in layer_removed.items():
            layer_id = self.replay.cache.layer_to_id[layer]
            if count > self.replay.cache.capacities[layer_id]:
                raise RuntimeError(f"Start prefix exceeds min-keep at {layer}")
            self.replay.cache.pruned[layer_id] = count
        full = self.replay.cache.pruned >= self.replay.cache.capacities
        for feasible in self.replay.cache.feasible_by_device.values():
            feasible.copy_(
                self.torch.as_tensor(~full, dtype=self.torch.bool, device=feasible.device)
            )
        self.replay.cache.refresh_all()
        indices = [int(row["global_index"]) for row in prefix]
        self.removed_cost = int(self.replay.costs[indices].sum()) if indices else 0
        self.steps = len(prefix)

    def all_candidates(self) -> list[dict[str, object]]:
        """Collect the start-state candidate vector once for frozen ranking."""
        rows: list[dict[str, object]] = []
        for domain_id, state in enumerate(self.replay.states):
            local = self.replay.cache.eligible_mask(domain_id).nonzero(as_tuple=True)[0]
            if local.numel() == 0:
                continue
            average = state.losses.index_select(0, local)
            total = average * float(self.replay.cache.active_size[domain_id])
            similarities = state.similarity.index_select(0, local).clone()
            row_index = self.torch.arange(local.numel(), device=self.device)
            similarities[row_index, local] = 0.0
            similarities.masked_fill_(
                ~(state.retained & state.valid_function_mask).unsqueeze(0), 0.0
            )
            best = similarities.max(dim=1).values
            packet = self.torch.stack((average, total, best), dim=1).detach().cpu().numpy()
            local_cpu = local.detach().cpu().numpy().astype(np.int64, copy=False)
            for offset, local_index in enumerate(local_cpu):
                global_index = int(state.global_indices[int(local_index)])
                unit = self.replay.units[global_index]
                rows.append(
                    {
                        "global_index": global_index,
                        "domain_id": domain_id,
                        "local_index": int(local_index),
                        "layer": unit.layer,
                        "unit_type": unit.unit_type,
                        "unit_index": unit.unit_index,
                        "Delta_average": float(packet[offset, 0]),
                        "Delta_total": float(packet[offset, 1]),
                        "best_substitute_similarity": float(packet[offset, 2]),
                        "domain_retained_ratio_before": state.retained_count / state.initial_size,
                        "domain_coverage_before": state.current_coverage,
                        "parameter_cost": int(self.replay.costs[global_index]),
                    }
                )
        return rows

    def _selection_context(self, global_index: int, domain_id: int, local_index: int) -> dict:
        state = self.replay.states[domain_id]
        average = float(state.losses[local_index].item())
        total = average * float(self.replay.cache.active_size[domain_id])
        substitutes = self.replay._substitutes(state, local_index)
        return {
            "Delta_average": average,
            "Delta_total": total,
            "domain_retained_ratio_before": state.retained_count / state.initial_size,
            "domain_coverage_before": state.current_coverage,
            "best_substitute_similarity": float(substitutes["best_remaining_similarity"]),
        }

    def remove(
        self,
        *,
        variant: str,
        global_index: int,
        domain_id: int,
        local_index: int,
        rank: int,
        frozen: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        state = self.replay.states[domain_id]
        if not bool(state.retained[local_index].item()):
            raise RuntimeError(f"Task019 attempted duplicate removal {global_index}")
        context = self._selection_context(global_index, domain_id, local_index)
        before = self.removed_cost
        unit = self.replay.units[global_index]
        cost = int(self.replay.costs[global_index])
        update = state.remove(local_index)
        became_full, affected = self.replay.cache.mark_removed(unit.layer)
        if became_full:
            for value in affected:
                self.replay.cache.refresh(value)
        else:
            self.replay.cache.refresh(domain_id)
        self.removed_cost += cost
        self.steps += 1
        functional_score = (
            float(frozen["Delta_total"]) if frozen is not None else context["Delta_total"]
        )
        return {
            "variant": variant,
            "step": self.steps,
            "incremental_step": 0,
            "estimated_sparsity_before": before / self.replay.total_parameters,
            "estimated_sparsity_after": self.removed_cost / self.replay.total_parameters,
            "global_index": global_index,
            "unit_type": unit.unit_type,
            "layer": unit.layer,
            "unit_index": unit.unit_index,
            "domain_id": domain_id,
            "functional_score": functional_score,
            "Delta_average": (
                float(frozen["Delta_average"]) if frozen is not None else context["Delta_average"]
            ),
            "Delta_total": functional_score,
            "rank_at_selection": rank,
            "domain_retained_ratio_before": context["domain_retained_ratio_before"],
            "domain_coverage_before": context["domain_coverage_before"],
            "domain_coverage_after": float(update["coverage_after"]),
            "best_substitute_similarity": context["best_substitute_similarity"],
            "parameter_cost": cost,
            "cumulative_removed_parameters": self.removed_cost,
        }

    def run_frozen(self, variant: str, target_budget: float) -> list[dict[str, object]]:
        candidates = frozen_candidate_order(self.all_candidates())
        rows: list[dict[str, object]] = []
        for frozen_rank, candidate in enumerate(candidates, start=1):
            if self.removed_cost >= target_budget:
                break
            global_index = int(candidate["global_index"])
            domain_id = int(candidate["domain_id"])
            local_index = int(candidate["local_index"])
            state = self.replay.states[domain_id]
            if not bool(state.retained[local_index].item()):
                continue
            layer_id = self.replay.cache.layer_to_id[str(candidate["layer"])]
            if self.replay.cache.pruned[layer_id] >= self.replay.cache.capacities[layer_id]:
                continue
            row = self.remove(
                variant=variant,
                global_index=global_index,
                domain_id=domain_id,
                local_index=local_index,
                rank=frozen_rank,
                frozen=candidate,
            )
            row["incremental_step"] = len(rows) + 1
            rows.append(row)
        if self.removed_cost < target_budget:
            raise RuntimeError("Frozen Task019 variant exhausted before budget endpoint")
        return rows

    def run_no_new_attention(
        self, variant: str, target_budget: float
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        while self.removed_cost < target_budget:
            selected = self.replay.cache.best(2, "domain_total")  # kind=2 is FFN only
            if selected is None:
                raise RuntimeError("No-new-Attention variant exhausted feasible FFN units")
            _score, global_index, domain_id, local_index = selected
            row = self.remove(
                variant=variant,
                global_index=global_index,
                domain_id=domain_id,
                local_index=local_index,
                rank=1,
            )
            row["incremental_step"] = len(rows) + 1
            rows.append(row)
        assert_no_new_attention(rows)
        return rows

    def domain_state_rows(self, variant: str) -> list[dict[str, object]]:
        rows = []
        for domain_id, state in enumerate(self.replay.states):
            neighborhood = self.replay._domain_neighborhood(state)
            rows.append(
                {
                    "variant": variant,
                    "domain_id": domain_id,
                    "initial_units": state.initial_size,
                    "retained_units": state.retained_count,
                    "retained_ratio": state.retained_count / state.initial_size,
                    "coverage": state.current_coverage,
                    "coverage_drop": state.initial_coverage - state.current_coverage,
                    **neighborhood,
                }
            )
        return rows


def _variant_summary(
    *,
    variant: str,
    start: float,
    target: float,
    start_prefix: Sequence[Mapping[str, object]],
    full_rows: Sequence[Mapping[str, object]],
    incremental: Sequence[Mapping[str, object]],
    total_parameters: int,
    domain_rows: Sequence[Mapping[str, object]],
    device: str,
    peak_cuda_bytes: int,
) -> dict[str, object]:
    attention = [row for row in incremental if row["unit_type"] == TYPE_ATTENTION]
    ffn = [row for row in incremental if row["unit_type"] == TYPE_FFN]
    incremental_cost = sum(int(row["parameter_cost"]) for row in incremental)
    attention_cost = sum(int(row["parameter_cost"]) for row in attention)
    removed_cost = sum(int(row["parameter_cost"]) for row in full_rows)
    target_budget = target * total_parameters
    return {
        "status": "prepared",
        "variant": variant,
        "start_sparsity": start,
        "target_sparsity": target,
        "prefix_steps_total": len(full_rows),
        "start_prefix_steps": len(start_prefix),
        "incremental_units": len(incremental),
        "incremental_attention": len(attention),
        "incremental_ffn": len(ffn),
        "incremental_parameter_cost": incremental_cost,
        "incremental_attention_parameter_cost": attention_cost,
        "incremental_ffn_parameter_cost": incremental_cost - attention_cost,
        "attention_parameter_share": attention_cost / incremental_cost if incremental_cost else 0.0,
        "estimated_removed_parameters": removed_cost,
        "estimated_parameter_sparsity": removed_cost / total_parameters,
        "target_parameter_budget": target_budget,
        "budget_overshoot": removed_cost - target_budget,
        "sequence_sha256": sequence_sha256(int(row["global_index"]) for row in full_rows),
        "increment_sequence_sha256": sequence_sha256(
            int(row["global_index"]) for row in incremental
        ),
        "unique_domains_touched_incremental": len({int(row["domain_id"]) for row in incremental}),
        "unique_layers_touched_incremental": len({str(row["layer"]) for row in incremental}),
        "mean_domain_coverage_final": float(np.mean([float(row["coverage"]) for row in domain_rows])),
        "median_domain_retained_ratio_final": float(
            np.median([float(row["retained_ratio"]) for row in domain_rows])
        ),
        "median_best_substitute_similarity_final": float(
            np.median([float(row["median_best_remaining_similarity"]) for row in domain_rows])
        ),
        "no_new_attention_added_attention_heads": len(attention),
        "budget_semantics_unchanged": True,
        "device": device,
        "peak_cuda_bytes": int(peak_cuda_bytes),
    }


def construct_variant(
    *,
    variant: str,
    task014_root: Path,
    task016_root: Path,
    task017_root: Path,
    task018_root: Path,
    output_dir: Path,
    device: str,
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Task019 variant: {variant}")
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Run Task019 identity gate before construction")
    if production_source_sha256(Path(__file__).resolve().parent) != identity["production_source_sha256"]:
        raise RuntimeError("Protected production source changed after Task019 identity gate")
    start, target, mode = VARIANTS[variant]
    trace30 = ordered_selected_rows(
        read_csv(_task016_dir(Path(task016_root), 0.30) / "functional_selection_trace.csv")
    )
    prefixes, summaries = reconstruct_all_prefixes(trace30, int(identity["parameters_before"]))
    summary_by_target = {
        float(row["target_parameter_sparsity"]): row for row in summaries
    }
    start_prefix = prefixes[start]
    target_budget = float(summary_by_target[target]["target_parameter_budget"])

    engine = CausalAblationEngine(
        task014_root=task014_root,
        task016_root=task016_root,
        task017_root=task017_root,
        output_dir=output_dir,
        device=device,
    )
    engine.torch.cuda.reset_peak_memory_stats(engine.device)
    engine.restore_prefix(start_prefix)
    if engine.removed_cost != int(summary_by_target[start]["estimated_removed_parameters"]):
        raise RuntimeError("Restored start-prefix cost differs from Task018")
    if mode == "frozen":
        incremental = engine.run_frozen(variant, target_budget)
    else:
        incremental = engine.run_no_new_attention(variant, target_budget)
    full_rows = [dict(row) for row in start_prefix] + [dict(row) for row in incremental]
    domain_rows = engine.domain_state_rows(variant)
    summary = _variant_summary(
        variant=variant,
        start=start,
        target=target,
        start_prefix=start_prefix,
        full_rows=full_rows,
        incremental=incremental,
        total_parameters=int(identity["parameters_before"]),
        domain_rows=domain_rows,
        device=str(engine.device),
        peak_cuda_bytes=int(engine.torch.cuda.max_memory_allocated(engine.device)),
    )
    if mode == "no_new_attention" and summary["incremental_attention"] != 0:
        raise RuntimeError("No-new-Attention construction violated intervention")
    registry = registry_from_prefix(full_rows)
    variant_dir = output_dir / "variants" / variant
    atomic_csv(variant_dir / "causal_selection_trace.csv", TRACE_FIELDS, incremental)
    atomic_csv(variant_dir / "domain_state.csv", tuple(domain_rows[0]), domain_rows)
    atomic_json(
        variant_dir / "registry.json",
        {
            "status": "prepared",
            "summary": summary,
            "registry": registry,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "production_pruning_code_modified": False,
        },
    )
    atomic_json(variant_dir / "construction.json", summary)


def validate_variant(output_dir: Path, variant: str, device: str) -> None:
    """Validate one prepared registry from a fresh original checkpoint."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    payload = read_json(output_dir / "variants" / variant / "registry.json")
    if identity.get("artifact_identity_pass") is not True or payload.get("status") != "prepared":
        raise RuntimeError("Task019 identity/construction gate is incomplete")
    if production_source_sha256(Path(__file__).resolve().parent) != identity["production_source_sha256"]:
        raise RuntimeError("Protected production source changed before validation")
    checkpoint = Path(identity["checkpoint"])
    stat = checkpoint.stat()
    if stat.st_size != int(identity["checkpoint_size_bytes"]) or stat.st_mtime_ns != int(
        identity["checkpoint_mtime_ns"]
    ):
        raise RuntimeError("Task019 checkpoint changed after identity audit")

    import torch
    import torch.nn as nn
    from ucf101_videoswin_my import (
        AverageMeter,
        SwinTransformer3D,
        get_dataset,
        set_seed,
        validate_rgb,
    )

    if not torch.cuda.is_available() or device != "cuda:0":
        raise RuntimeError("Each isolated Task019 validation worker requires logical cuda:0")
    torch.cuda.set_device(0)
    set_seed(3407)
    torch.cuda.reset_peak_memory_stats(0)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        drop_path_rate=0.2,
        use_checkpoint=True,
    ).to(torch.device(device))
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    normalized = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }
    load_message = model.load_state_dict(normalized, strict=False)
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    if parameters_before != int(identity["parameters_before"]):
        raise RuntimeError("Task019 model parameter count differs from identity")
    applied = _apply_registry(model, payload["registry"])
    summary = payload["summary"]
    expected_attention = sum(
        len(entry["indices"])
        for entry in payload["registry"].values()
        if entry["unit_type"] == TYPE_ATTENTION
    )
    expected_ffn = sum(
        len(entry["indices"])
        for entry in payload["registry"].values()
        if entry["unit_type"] == TYPE_FFN
    )
    if applied != {"removed_attention": expected_attention, "removed_ffn": expected_ffn}:
        raise RuntimeError("Task019 applied registry count mismatch")

    loader = get_dataset(str(identity["validation_split"]), int(identity["validation_batch_size"]))
    # The shell gives each validation worker exactly one physical 3090.  This
    # avoids competing DataParallel replicas and keeps all model compute on GPU.
    evaluation_model = model
    top1, top5 = AverageMeter(), AverageMeter()
    started = time.perf_counter()
    validate_rgb(
        loader,
        evaluation_model,
        top1,
        top5,
        use_amp=bool(identity["amp_enabled"]),
    )
    elapsed = time.perf_counter() - started
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    report = model.get_detailed_pruning_report()
    if not math.isclose(
        float(report["sparsity"]),
        float(summary["estimated_parameter_sparsity"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Task019 registry application disagrees with budget estimate")
    metrics = {
        "status": "PASS",
        "variant": variant,
        "top1": float(top1.avg),
        "top5": float(top5.avg),
        "samples": len(loader.dataset),
        "validation_time_seconds": elapsed,
        "estimated_sparsity": float(summary["estimated_parameter_sparsity"]),
        "estimated_removed_parameters": int(summary["estimated_removed_parameters"]),
        "budget_overshoot": float(summary["budget_overshoot"]),
        "physical_numel_sparsity": 1.0 - parameters_after / float(parameters_before),
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "sequence_sha256": summary["sequence_sha256"],
        "checkpoint_missing_keys": sorted(load_message.missing_keys),
        "checkpoint_unexpected_keys": sorted(load_message.unexpected_keys),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
        "fresh_original_checkpoint": True,
        "fine_tuning_executed": False,
        "production_pruning_code_modified": False,
    }
    atomic_json(output_dir / "validation" / variant / "metrics.json", metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    identity = subparsers.add_parser("identity")
    for command in (identity,):
        command.add_argument("--task014-root", type=Path, required=True)
        command.add_argument("--task016-root", type=Path, required=True)
        command.add_argument("--task017-root", type=Path, required=True)
        command.add_argument("--task018-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
    construct = subparsers.add_parser("construct")
    construct.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    construct.add_argument("--task014-root", type=Path, required=True)
    construct.add_argument("--task016-root", type=Path, required=True)
    construct.add_argument("--task017-root", type=Path, required=True)
    construct.add_argument("--task018-root", type=Path, required=True)
    construct.add_argument("--output-dir", type=Path, required=True)
    construct.add_argument("--device", default="cuda:0")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    validate.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "identity":
        verify_identity(
            task014_root=args.task014_root,
            task016_root=args.task016_root,
            task017_root=args.task017_root,
            task018_root=args.task018_root,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
        )
    elif args.command == "construct":
        construct_variant(
            variant=args.variant,
            task014_root=args.task014_root,
            task016_root=args.task016_root,
            task017_root=args.task017_root,
            task018_root=args.task018_root,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        validate_variant(args.output_dir, args.variant, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

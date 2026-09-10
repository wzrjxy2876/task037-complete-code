"""Global BMS and dynamic F3 selector for Task038."""
from __future__ import annotations

import csv
import hashlib
import math
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np

from .slowfast_functional_archive import (
    ContributionFieldArchive,
    DomainState,
    domain_total_losses,
    marginal_coverage_losses,
    validate_partition,
)
from .slowfast_unit_adapter import Unit, UnitInventory
from .slowfast_parameter_accounting import (
    count_structural_parameters,
    write_parameter_accounting,
)


def standardize_descriptors(descriptor: torch.Tensor) -> torch.Tensor:
    if descriptor.ndim != 2 or descriptor.shape[1] != 3:
        raise ValueError("Dynamic3D descriptors must have shape [N,3]")
    return (descriptor.float() - descriptor.float().mean(0, keepdim=True)) / (
        descriptor.float().std(0, keepdim=True) + 1e-8
    )


def _merge_sinks(sinks: torch.Tensor, tol: float) -> list[list[int]]:
    groups: list[list[int]] = []
    representatives: list[torch.Tensor] = []
    for index in range(sinks.shape[0]):
        point = sinks[index]
        found = None
        for gid, rep in enumerate(representatives):
            if float(torch.linalg.norm(point - rep).item()) <= tol:
                found = gid
                break
        if found is None:
            representatives.append(point.clone())
            groups.append([index])
        else:
            groups[found].append(index)
    return groups


def global_bms_domains(
    descriptor: torch.Tensor,
    device: torch.device,
    sigma: float = 0.1,
    tol: float = 1e-4,
    max_iters: int = 100,
    sink_merge_tol: float = 0.01,
) -> tuple[list[list[int]], torch.Tensor]:
    if sigma <= 0.0 or max_iters != 100 or sink_merge_tol != 0.01:
        raise ValueError("Task038 BMS parameters are frozen")
    positions = standardize_descriptors(descriptor.to(device))
    count = int(positions.shape[0])
    for _ in range(max_iters):
        updated = torch.empty_like(positions)
        for start in range(0, count, 1024):
            stop = min(start + 1024, count)
            distances = torch.cdist(positions[start:stop], positions)
            weights = torch.exp(-(distances.square()) / (2.0 * sigma * sigma))
            updated[start:stop] = weights @ positions / weights.sum(1, keepdim=True).clamp_min(1e-8)
        movement = torch.linalg.norm(updated - positions, dim=1).max()
        positions = updated
        if float(movement.item()) < tol:
            break
    groups = validate_partition(_merge_sinks(positions, sink_merge_tol), count)
    return groups, positions


def _stable_order(values: torch.Tensor, tie_ids: torch.Tensor) -> torch.Tensor:
    # NumPy lexsort is the exact stable implementation on this PyTorch 1.12
    # runtime; it is used only for rank ordering, never for field arithmetic.
    value_cpu = values.detach().cpu().numpy()
    tie_cpu = tie_ids.detach().cpu().numpy()
    order_cpu = np.lexsort((tie_cpu, value_cpu))
    return torch.from_numpy(order_cpu.astype(np.int64, copy=False)).to(values.device)


def ordinal_percentile(
    values: torch.Tensor, global_indices: torch.Tensor
) -> torch.Tensor:
    if values.ndim != 1 or global_indices.shape != values.shape:
        raise ValueError("rank tensors must be aligned")
    if values.numel() == 0:
        return torch.empty_like(values, dtype=torch.float64)
    order = _stable_order(values, global_indices)
    ranks = torch.empty(
        values.numel(), dtype=torch.float64, device=values.device
    )
    positions = torch.arange(
        values.numel(), dtype=torch.float64, device=values.device
    )
    ranks[order] = positions / float(max(values.numel() - 1, 1))
    return ranks


def f3_order(
    r_f3: torch.Tensor,
    p_total: torch.Tensor,
    p_average: torch.Tensor,
    global_indices: torch.Tensor,
) -> torch.Tensor:
    """Return local positions sorted by the exact four frozen F3 keys."""
    arrays = [
        torch.as_tensor(x, dtype=torch.float64).detach().cpu().numpy()
        for x in (global_indices, p_average, p_total, r_f3)
    ]
    order = np.lexsort((arrays[0], arrays[1], arrays[2], arrays[3]))
    return torch.from_numpy(order.astype(np.int64, copy=False)).to(
        device=r_f3.device
    )


def f3_components(
    delta_total: torch.Tensor,
    delta_average: torch.Tensor,
    domain_damage: torch.Tensor,
    global_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Raw deltas are intentionally float32.  Only ordinal ranks and risk
    # arithmetic are promoted to float64 for exact tie-boundary behavior.
    p_total = ordinal_percentile(delta_total, global_indices)
    p_average = ordinal_percentile(delta_average, global_indices)
    damage64 = torch.as_tensor(
        domain_damage, dtype=torch.float64, device=p_total.device
    )
    b = torch.maximum(p_total, damage64)
    rescue = torch.maximum(p_average - b, torch.zeros_like(b))
    return {
        "p_total": p_total,
        "p_average": p_average,
        "B": b,
        "V": rescue,
        "R_F3": b + rescue / 2.0,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def closest_prefix_choice(current_remaining: int, next_remaining: int, target_parameters: float) -> bool:
    """Return whether the next adjacent F3 prefix is closer to the target."""
    current_error = abs(float(current_remaining) - float(target_parameters))
    next_error = abs(float(next_remaining) - float(target_parameters))
    return next_error < current_error or (
        next_error == current_error and next_remaining <= target_parameters
    )


def select_f3(
    archive: ContributionFieldArchive,
    inventory: UnitInventory,
    domains: Sequence[Sequence[int]],
    target_remaining_ratio: float,
    output_dir: str | Path,
    preferred_device: torch.device,
    max_steps: int | None = None,
    structural_model: Any | None = None,
    dependency_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run frozen F3 and stop at the closest structural-parameter prefix.

    F3 chooses the next unit.  The structural counter is consulted only after
    that next unit has been selected, and only to choose between adjacent
    prefixes at the target crossing.
    """
    if not 0.0 < float(target_remaining_ratio) <= 1.0:
        raise ValueError("target_remaining_ratio must be in (0,1]")
    if structural_model is None:
        from .slowfast_model_task038 import slowfast_16x8_resnet101_kinetics400
        structural_model = slowfast_16x8_resnet101_kinetics400(101)
    domains = validate_partition(domains, inventory.num_units)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive.materialize_aligned()
    aligned_np = np.load(archive.root / "aligned_vectors.npy", mmap_mode="r", allow_pickle=False)
    valid_np = np.load(archive.root / "aligned_valid.npy", mmap_mode="r", allow_pickle=False)
    aligned_cpu = torch.from_numpy(np.array(aligned_np, dtype=np.float32, copy=True))
    valid_cpu = torch.from_numpy(np.array(valid_np, dtype=np.bool_, copy=True))
    all_vectors = aligned_cpu.to(device=preferred_device, dtype=torch.float32)
    all_valid = valid_cpu.to(device=preferred_device, dtype=torch.bool)
    all_norms = torch.linalg.norm(all_vectors, dim=1)
    # Normalize in place to avoid a second N*unit*feature allocation.  This
    # is numerically equivalent to the previous float32 where/division path;
    # invalid rows are exact zero in the archive and are masked explicitly.
    all_vectors.div_(all_norms.clamp_min(1e-12)[:, None])
    all_vectors.masked_fill_(~all_valid[:, None], 0.0)
    del all_norms
    states: list[DomainState] = []
    for domain_id, members in enumerate(domains):
        member_tensor = torch.tensor(members, dtype=torch.long, device=preferred_device)
        states.append(DomainState.create(
            domain_id, members, all_vectors.index_select(0, member_tensor),
            all_valid.index_select(0, member_tensor)
        ))
    del aligned_np, valid_np, aligned_cpu, valid_cpu, all_vectors, all_valid
    by_global: dict[int, tuple[int, int]] = {
        gid: (did, local)
        for did, state in enumerate(states)
        for local, gid in enumerate(state.members)
    }
    max_prunable: dict[str, int] = {}
    layer_widths: dict[str, int] = {}
    for layer in {u.module_name for u in inventory.units}:
        width = next(u.out_channels for u in inventory.units if u.module_name == layer)
        layer_widths[layer] = int(width)
        max_prunable[layer] = int(width * (1.0 - inventory.min_keep_ratio))
    current_pruned: set[int] = set()
    layer_counts = {layer: 0 for layer in max_prunable}
    trace: list[dict[str, Any]] = []
    layer_names = sorted(max_prunable)
    layer_number = {name: index for index, name in enumerate(layer_names)}
    unit_layer_ids = torch.tensor(
        [layer_number[u.module_name] for u in inventory.units],
        dtype=torch.long, device=preferred_device
    )
    layer_caps = torch.tensor(
        [max_prunable[name] for name in layer_names],
        dtype=torch.long, device=preferred_device
    )
    member_tensors = [
        torch.tensor(state.members, dtype=torch.long, device=preferred_device)
        for state in states
    ]
    loss_global = torch.full(
        (inventory.num_units,), float("inf"), dtype=torch.float32, device=preferred_device
    )
    retained_global = torch.zeros(
        inventory.num_units, dtype=torch.bool, device=preferred_device
    )
    damage_global = torch.zeros_like(loss_global)
    active_count_global = torch.zeros_like(loss_global)
    for domain_id, state in enumerate(states):
        members = member_tensors[domain_id]
        active_count = float(state.valid.sum().item())
        damage_value = 1.0 - float(state.coverage.item())
        active_count_global.index_fill_(0, members, active_count)
        damage_global.index_fill_(0, members, damage_value)
        loss_global.index_copy_(0, members, state.losses)
        retained_global.index_copy_(0, members, state.retained)
    layer_count_tensor = torch.zeros(
        len(layer_names), dtype=torch.long, device=preferred_device
    )

    def accounting_for(counts: Mapping[str, int]) -> dict[str, Any]:
        # The counter needs only the propagated keep widths.  The final JSON
        # registry is still built from exact global indices after selection.
        keep_state = {
            "layers": {
                name: {"keep_count": layer_widths[name] - int(counts[name])}
                for name in layer_names
            }
        }
        return count_structural_parameters(structural_model, keep_state, dependency_graph)

    current_accounting = accounting_for(layer_counts)
    original_parameters = int(current_accounting["original_trainable_parameters"])
    target_parameters = float(original_parameters) * float(target_remaining_ratio)
    crossing: dict[str, Any] | None = None
    stop_reason = "closest_structural_equivalent_parameter_prefix"

    def apply_candidate(gid: int, did: int, local: int) -> None:
        state = states[did]
        state.remove(local)
        members = member_tensors[did]
        loss_global.index_copy_(0, members, state.losses)
        retained_global.index_copy_(0, members, state.retained)
        damage_global.index_fill_(0, members, 1.0 - float(state.coverage.item()))
        current_pruned.add(gid)
        unit = inventory.units[gid]
        layer_counts[unit.module_name] += 1
        layer_count_tensor[layer_number[unit.module_name]] += 1

    while True:
        if max_steps is not None and len(trace) >= max_steps:
            stop_reason = "max_steps"
            break
        feasible = retained_global & (
            layer_count_tensor.index_select(0, unit_layer_ids) < layer_caps.index_select(0, unit_layer_ids)
        )
        gids = feasible.nonzero(as_tuple=True)[0]
        if gids.numel() == 0:
            stop_reason = "no_feasible_candidate"
            break
        average = loss_global.index_select(0, gids)
        total = average * active_count_global.index_select(0, gids)
        damage = damage_global.index_select(0, gids)
        components = f3_components(total, average, damage, gids)
        order_t = f3_order(components["R_F3"], components["p_total"], components["p_average"], gids)
        pos = int(order_t[0].item())
        gid = int(gids[pos].item())
        did, local = by_global[gid]
        unit = inventory.units[gid]
        next_counts = dict(layer_counts)
        next_counts[unit.module_name] += 1
        next_accounting = accounting_for(next_counts)
        current_remaining = int(current_accounting["structural_equivalent_remaining_parameters"])
        next_remaining = int(next_accounting["structural_equivalent_remaining_parameters"])
        current_error = abs(float(current_remaining) - target_parameters)
        next_error = abs(float(next_remaining) - target_parameters)
        is_crossing = current_remaining >= target_parameters and next_remaining <= target_parameters
        if is_crossing:
            choose_next = closest_prefix_choice(current_remaining, next_remaining, target_parameters)
            crossing = {
                "before_step": len(trace),
                "after_step": len(trace) + 1,
                "before_remaining_parameters": current_remaining,
                "before_remaining_ratio": current_remaining / original_parameters,
                "before_error_parameters": current_error,
                "after_remaining_parameters": next_remaining,
                "after_remaining_ratio": next_remaining / original_parameters,
                "after_error_parameters": next_error,
                "selected_after": choose_next,
            }
            if not choose_next:
                break
        cost = int(unit.parameter_cost)
        avg_value = float(average[pos].item())
        total_value = float(total[pos].item())
        damage_value = float(damage[pos].item())
        apply_candidate(gid, did, local)
        current_accounting = next_accounting
        trace.append({
            "step": len(trace) + 1,
            "global_index": gid,
            "layer_name": unit.layer_name,
            "local_channel_index": unit.local_channel_index,
            "pathway": unit.pathway,
            "stage": unit.stage,
            "block": unit.block,
            "conv_position": unit.conv_position,
            "domain_id": did,
            "domain_size": len(states[did].members),
            "domain_active_size": int(states[did].valid.sum().item()),
            "delta_average": avg_value,
            "delta_total": total_value,
            "domain_damage": damage_value,
            "p_total": float(components["p_total"][pos].item()),
            "p_average": float(components["p_average"][pos].item()),
            "B": float(components["B"][pos].item()),
            "V": float(components["V"][pos].item()),
            "R_F3": float(components["R_F3"][pos].item()),
            "legacy_local_parameter_cost": cost,
            "structural_equivalent_remaining_parameters": next_remaining,
            "structural_equivalent_remaining_ratio": next_remaining / original_parameters,
            "selected": True,
        })
        if is_crossing:
            break

    registry_layers: dict[str, dict[str, Any]] = {}
    for layer in sorted(max_prunable):
        units = [u for u in inventory.units if u.module_name == layer]
        pruned = sorted(u.local_channel_index for u in units if u.global_index in current_pruned)
        keep = sorted(set(range(units[0].out_channels)) - set(pruned))
        registry_layers[layer] = {
            "total": units[0].out_channels, "pruned": pruned, "keep": keep,
            "pruned_count": len(pruned), "keep_count": len(keep),
        }
    final_remaining = int(current_accounting["structural_equivalent_remaining_parameters"])
    final_ratio = final_remaining / original_parameters
    legacy_sum = sum(int(row["legacy_local_parameter_cost"]) for row in trace)
    registry = {
        "schema": "task038_f3_registry_v2",
        "selection_method": "dynamic global F3 fixed BMS",
        "target_remaining_ratio": float(target_remaining_ratio),
        "target_parameters_real": target_parameters,
        "selected_final_prefix_step": len(trace),
        "structural_equivalent_remaining_parameters": final_remaining,
        "structural_equivalent_removed_parameters": original_parameters - final_remaining,
        "remaining_parameter_ratio": final_ratio,
        "parameter_pruning_ratio": 1.0 - final_ratio,
        "legacy_local_parameter_cost_sum": legacy_sum,
        "layers": registry_layers,
        "pruned_global_indices": sorted(current_pruned),
        "trace_length": len(trace),
        "stop_reason": stop_reason,
    }
    trace_path = out / "functional_selection_trace.csv"
    fields = list(trace[0]) if trace else ["step", "global_index"]
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(trace)
    seq_path = out / "f3_selection_sequence.csv"
    with seq_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "global_index", "legacy_local_parameter_cost"])
        writer.writeheader()
        writer.writerows({"step": r["step"], "global_index": r["global_index"],
                          "legacy_local_parameter_cost": r["legacy_local_parameter_cost"]} for r in trace)
    sequence_sha = hashlib.sha256(seq_path.read_bytes()).hexdigest()
    (out / "f3_selection_sequence.sha256").write_text(sequence_sha + "\n", encoding="utf-8")
    _write_json(out / "f3_registry.json", registry)
    _write_json(out / "f3_registry_manifest.json", {"sha256": hashlib.sha256((out / "f3_registry.json").read_bytes()).hexdigest(), "sequence_sha256": sequence_sha, "schema": registry["schema"]})
    structures = {"stage": {}, "pathway": {}, "domain": {}}
    for unit in inventory.units:
        is_pruned = unit.global_index in current_pruned
        for key, value in (("stage", unit.stage), ("pathway", unit.pathway)):
            bucket = structures[key].setdefault(value, {"total": 0, "pruned": 0, "legacy_local_parameter_cost": 0})
            bucket["total"] += 1; bucket["pruned"] += int(is_pruned)
            bucket["legacy_local_parameter_cost"] += int(is_pruned) * unit.parameter_cost
    for did, members in enumerate(domains):
        structures["domain"][str(did)] = {"total": len(members), "pruned": sum(x in current_pruned for x in members)}
    _write_json(out / "f3_stage_structure.json", structures["stage"])
    _write_json(out / "f3_pathway_structure.json", structures["pathway"])
    _write_json(out / "f3_domain_structure.json", structures["domain"])
    _write_json(out / "f3_layer_structure.json", registry_layers)
    target_payload = {
        "P_original": original_parameters,
        "target_remaining_ratio": float(target_remaining_ratio),
        "P_target_real": target_parameters,
        "P_target_floor": math.floor(target_parameters),
        "P_target_ceil": math.ceil(target_parameters),
        "prefix_before_crossing_step": None if crossing is None else crossing["before_step"],
        "prefix_before_remaining_parameters": None if crossing is None else crossing["before_remaining_parameters"],
        "prefix_before_remaining_ratio": None if crossing is None else crossing["before_remaining_ratio"],
        "prefix_before_error_parameters": None if crossing is None else crossing["before_error_parameters"],
        "prefix_after_crossing_step": None if crossing is None else crossing["after_step"],
        "prefix_after_remaining_parameters": None if crossing is None else crossing["after_remaining_parameters"],
        "prefix_after_remaining_ratio": None if crossing is None else crossing["after_remaining_ratio"],
        "prefix_after_error_parameters": None if crossing is None else crossing["after_error_parameters"],
        "selected_final_prefix_step": len(trace),
        "final_remaining_parameters": final_remaining,
        "final_removed_parameters": original_parameters - final_remaining,
        "final_remaining_ratio": final_ratio,
        "final_parameter_pruning_ratio": 1.0 - final_ratio,
        "absolute_target_error_parameters": abs(float(final_remaining) - target_parameters),
        "absolute_target_error_ratio": abs(final_ratio - float(target_remaining_ratio)),
        "absolute_target_error_percentage_points": abs(final_ratio - float(target_remaining_ratio)) * 100.0,
        "stop_reason": stop_reason,
    }
    _write_json(out / "remaining_parameter_target.json", target_payload)
    write_parameter_accounting(current_accounting, out / "parameter_accounting")
    return {"registry": registry, "trace": trace, "sequence_sha256": sequence_sha, "states": states, "accounting": current_accounting, "target": target_payload}

def reference_f3_prefix(
    archive: ContributionFieldArchive,
    inventory: UnitInventory,
    domains: Sequence[Sequence[int]],
    max_steps: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Direct replay oracle; recomputes every domain loss after each removal."""
    domains = validate_partition(domains, inventory.num_units)
    states: list[DomainState] = []
    for did, members in enumerate(domains):
        vectors, valid = archive.load_vectors(members, device)
        states.append(DomainState.create(did, members, vectors, valid))
    by_global = {
        gid: (did, local)
        for did, state in enumerate(states)
        for local, gid in enumerate(state.members)
    }
    layer_names = sorted({u.module_name for u in inventory.units})
    layer_number = {name: i for i, name in enumerate(layer_names)}
    layer_caps = {
        name: int(next(
            u.out_channels for u in inventory.units if u.module_name == name
        ) * (1.0 - inventory.min_keep_ratio))
        for name in layer_names
    }
    layer_counts = {name: 0 for name in layer_names}
    retained_global = torch.zeros(
        inventory.num_units, dtype=torch.bool, device=device
    )
    loss_global = torch.full(
        (inventory.num_units,), float("inf"), dtype=torch.float32, device=device
    )
    active_global = torch.zeros(
        inventory.num_units, dtype=torch.float32, device=device
    )
    damage_global = torch.zeros_like(loss_global)
    for state in states:
        members = torch.tensor(state.members, dtype=torch.long, device=device)
        retained_global.index_copy_(0, members, state.retained)
        loss_global.index_copy_(0, members, state.losses)
        active = float(state.valid.sum().item())
        active_global.index_fill_(0, members, active)
        damage_global.index_fill_(0, members, 1.0 - float(state.coverage.item()))

    trace: list[dict[str, Any]] = []
    for step in range(int(max_steps)):
        unit_layer_ids = torch.tensor(
            [layer_number[u.module_name] for u in inventory.units],
            dtype=torch.long, device=device
        )
        caps = torch.tensor(
            [layer_caps[name] for name in layer_names],
            dtype=torch.long, device=device
        )
        counts = torch.tensor(
            [layer_counts[u.module_name] for u in inventory.units],
            dtype=torch.long, device=device
        )
        feasible = retained_global & (
            counts < caps.index_select(0, unit_layer_ids)
        )
        gids = feasible.nonzero(as_tuple=True)[0]
        if gids.numel() == 0:
            break
        average = loss_global.index_select(0, gids)
        total = average * active_global.index_select(0, gids)
        components = f3_components(
            total, average, damage_global.index_select(0, gids), gids
        )
        pos = int(f3_order(
            components["R_F3"], components["p_total"],
            components["p_average"], gids
        )[0].item())
        gid = int(gids[pos].item())
        did, local = by_global[gid]
        state = states[did]
        avg_value = float(average[pos].item())
        total_value = float(total[pos].item())
        damage_value = float(damage_global[gid].item())
        state.retained[local] = False
        if bool(state.retained.any()):
            state.losses, state.coverage = marginal_coverage_losses(
                state.similarity, state.retained, state.valid
            )
        else:
            state.losses.fill_(float("inf"))
            state.coverage = state.coverage.new_tensor(
                0.0 if bool(state.valid.any()) else 1.0
            )
        members = torch.tensor(state.members, dtype=torch.long, device=device)
        loss_global.index_copy_(0, members, state.losses)
        retained_global.index_copy_(0, members, state.retained)
        damage_global.index_fill_(
            0, members, 1.0 - float(state.coverage.item())
        )
        unit = inventory.units[gid]
        layer_counts[unit.module_name] += 1
        trace.append({
            "step": step + 1,
            "global_index": gid,
            "delta_average": avg_value,
            "delta_total": total_value,
            "domain_damage": damage_value,
            "p_total": float(components["p_total"][pos].item()),
            "p_average": float(components["p_average"][pos].item()),
            "R_F3": float(components["R_F3"][pos].item()),
        })
    return trace

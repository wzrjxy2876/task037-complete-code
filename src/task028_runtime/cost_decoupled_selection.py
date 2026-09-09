"""Pure selection helpers for Task013 cost-decoupling diagnostics.

The production BMS implementation owns descriptor construction, grouping and
structural feasibility.  These helpers make the one changed principle small
and independently testable: raw redundancy determines ordering; parameter
cost only accumulates toward a global budget.
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence


SELECTION_COST_MODES = ("coupled", "decoupled")


def validate_selection_cost_mode(mode: str) -> str:
    """Validate and return one of the two Task013 modes."""
    if mode not in SELECTION_COST_MODES:
        raise ValueError(
            f"selection_cost_mode must be one of {SELECTION_COST_MODES}, got {mode!r}"
        )
    return mode


def effective_selection_score(
    raw_pruning_score: float, parameter_cost: int, mode: str
) -> float:
    """Return the scalar actually used to rank a candidate.

    Source inspection found that Task012's coupled path also sorts by the raw
    score; cost couples through whole-group admission and the stopping
    boundary, not through a scalar score transform.  Keeping this function
    explicit prevents a fictitious score/cost formula from entering Task013.
    """
    validate_selection_cost_mode(mode)
    if int(parameter_cost) <= 0:
        raise ValueError("parameter_cost must be positive")
    return float(raw_pruning_score)


def rank_candidates(
    candidates: Sequence[Mapping[str, object]], mode: str = "decoupled"
) -> list[dict[str, object]]:
    """Return stable ascending raw-score order without reading cost in the key.

    Each candidate is one pruning unit and must contain
    ``raw_pruning_score`` and a positive ``parameter_cost``.  The input axis is
    the original candidate order; stable sorting preserves it for ties.
    """
    validate_selection_cost_mode(mode)
    ranked = []
    for original_rank, candidate in enumerate(candidates):
        row = dict(candidate)
        raw = float(row["raw_pruning_score"])
        cost = int(row["parameter_cost"])
        row["effective_selection_score"] = effective_selection_score(raw, cost, mode)
        row["_original_rank"] = original_rank
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            float(row["effective_selection_score"]),
            int(row["_original_rank"]),
        )
    )
    for selection_rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = selection_rank
        row.pop("_original_rank", None)
    return ranked


def select_decoupled_candidates(
    candidates: Sequence[Mapping[str, object]],
    target_budget: int,
    max_prunable_by_layer: Mapping[str, int],
) -> tuple[list[dict[str, object]], int]:
    """Select unit candidates by raw score until parameter budget is reached.

    This small unit-level reference is used for synthetic proof tests.  The
    production selector applies the same principle to existing BMS groups.
    ``max_prunable_by_layer`` contains an integer capacity for each layer; no
    unit-type quota is accepted or implemented.
    """
    budget = int(target_budget)
    if budget <= 0:
        raise ValueError("target_budget must be positive")
    capacities = {str(layer): int(value) for layer, value in max_prunable_by_layer.items()}
    if any(value < 0 for value in capacities.values()):
        raise ValueError("layer capacities must be non-negative")

    removed_by_layer = {layer: 0 for layer in capacities}
    removed_cost = 0
    trace = []
    for row in rank_candidates(candidates, mode="decoupled"):
        layer = str(row["layer"])
        if layer not in capacities:
            raise KeyError(f"missing layer capacity for {layer!r}")
        selected = removed_by_layer[layer] < capacities[layer] and removed_cost < budget
        before = removed_cost
        if selected:
            removed_by_layer[layer] += 1
            removed_cost += int(row["parameter_cost"])
        item = dict(row)
        item.update(
            {
                "selected": bool(selected),
                "removed_cost_before": before,
                "removed_cost_after": removed_cost,
            }
        )
        trace.append(item)
        if removed_cost >= budget:
            break
    return trace, removed_cost


def selection_trace_snapshots(
    candidates: Iterable[Mapping[str, object]], target_budget: float
) -> list[dict[str, object]]:
    """Sample the next ranked unit initially and at each 10% budget boundary.

    Input rows are full ranked candidate records with final ``selected`` state.
    Output rows use exact progress values ``0.0, 0.1, ..., 1.0`` whenever a
    next candidate exists.  Cost is used only to locate progress boundaries;
    it never changes candidate order.
    """
    budget = float(target_budget)
    if budget <= 0.0:
        raise ValueError("target_budget must be positive")
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: int(row["selection_rank"]),
    )
    if not ordered:
        return []

    removed_before = []
    cumulative = 0.0
    for row in ordered:
        removed_before.append(cumulative)
        if bool(row.get("selected")):
            cost = int(row["parameter_cost"])
            if cost <= 0:
                raise ValueError("selected candidate cost must be positive")
            cumulative += cost

    snapshots = []
    for tenth in range(11):
        threshold = budget * tenth / 10.0
        next_index = None
        for index, before in enumerate(removed_before):
            if before + 1e-12 >= threshold:
                next_index = index
                break
        if next_index is None:
            next_index = len(ordered) - 1
        row = dict(ordered[next_index])
        row["budget_progress"] = tenth / 10.0
        snapshots.append(row)
    return snapshots


def mark_selected(
    ranked_candidates: Sequence[Mapping[str, object]],
    registry: Mapping[str, Iterable[int]],
) -> list[dict[str, object]]:
    """Attach final registry membership to full ranked candidate evidence."""
    selected = {
        str(layer): {int(index) for index in indices}
        for layer, indices in registry.items()
    }
    rows = []
    for candidate in ranked_candidates:
        row: MutableMapping[str, object] = dict(candidate)
        layer = str(row["layer"])
        row["selected"] = int(row["unit_index"]) in selected.get(layer, set())
        rows.append(dict(row))
    return rows

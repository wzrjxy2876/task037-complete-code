"""Pure HTOR temporal-intervention mathematics.

This module deliberately has no model, dataset, checkpoint, or Contribution
Field dependency.  Temporal indices are zero based and ``*_end`` fields in a
manifest are exclusive Python slice bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TemporalIntervention:
    """One adjacent equal-block swap at one hierarchy level."""

    temporal_length: int
    level: int
    block_size: int
    pair_index: int
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    permutation: tuple[int, ...]


def validate_temporal_length(temporal_length: int) -> int:
    """Validate the frozen Task040 power-of-two temporal rule."""

    if isinstance(temporal_length, bool):
        raise ValueError("T must be an integer >= 2; boolean values are invalid")
    t = int(temporal_length)
    if t != temporal_length or t < 2:
        raise ValueError(
            f"Task040 requires T to be a positive power of two with T >= 2; got T={temporal_length!r}"
        )
    if t & (t - 1):
        raise ValueError(
            f"Task040 requires T to be a positive power of two; got T={t}"
        )
    return t


def build_temporal_permutation(
    temporal_length: int, intervention: TemporalIntervention
) -> tuple[int, ...]:
    """Return the deterministic permutation encoded by ``intervention``."""

    t = validate_temporal_length(temporal_length)
    if intervention.temporal_length != t:
        raise ValueError("intervention temporal length does not match T")
    permutation = tuple(int(index) for index in intervention.permutation)
    if len(permutation) != t or sorted(permutation) != list(range(t)):
        raise ValueError("intervention permutation is not a bijection over [0, T)")
    return permutation


def enumerate_hierarchical_interventions(
    temporal_length: int,
) -> list[TemporalIntervention]:
    """Enumerate all adjacent block swaps from fine to coarse temporal scale.

    At level ``l`` the block size is ``2**l``.  Every level covers the full
    clip with disjoint adjacent pairs, and the total number is ``T - 1``.
    """

    t = validate_temporal_length(temporal_length)
    interventions: list[TemporalIntervention] = []
    for level in range(t.bit_length() - 1):
        block_size = 1 << level
        pair_count = t // (2 * block_size)
        for pair_index in range(pair_count):
            left_start = pair_index * 2 * block_size
            left_end = left_start + block_size
            right_start = left_end
            right_end = right_start + block_size
            permutation = list(range(t))
            permutation[left_start:right_end] = (
                permutation[right_start:right_end]
                + permutation[left_start:left_end]
            )
            interventions.append(
                TemporalIntervention(
                    temporal_length=t,
                    level=level,
                    block_size=block_size,
                    pair_index=pair_index,
                    left_start=left_start,
                    left_end=left_end,
                    right_start=right_start,
                    right_end=right_end,
                    permutation=tuple(permutation),
                )
            )
    if len(interventions) != t - 1:
        raise AssertionError("hierarchical intervention count must equal T - 1")
    return interventions


def apply_temporal_intervention(
    x: torch.Tensor,
    intervention: TemporalIntervention,
    time_dim: int,
) -> torch.Tensor:
    """Apply one intervention using tensor indexing without copying frames in Python."""

    permutation = build_temporal_permutation(
        x.shape[time_dim], intervention
    )
    indices = torch.tensor(permutation, dtype=torch.long, device=x.device)
    return torch.index_select(x, dim=time_dim, index=indices)


def apply_temporal_interventions(
    x: torch.Tensor,
    interventions: Sequence[TemporalIntervention],
    time_dim: int,
) -> torch.Tensor:
    """Apply a batch of interventions in one tensor operation.

    ``x`` is one clip without a batch dimension.  The returned tensor has a
    leading intervention dimension, followed by the original dimensions.
    """

    if not interventions:
        raise ValueError("at least one intervention is required")
    t = validate_temporal_length(int(x.shape[time_dim]))
    permutations = torch.tensor(
        [build_temporal_permutation(t, item) for item in interventions],
        dtype=torch.long,
        device=x.device,
    )
    moved = x.movedim(time_dim, -1)
    expanded = moved.unsqueeze(0).expand(len(interventions), *moved.shape)
    index_shape = (len(interventions),) + (1,) * (moved.ndim - 1) + (t,)
    indices = permutations.view(index_shape).expand_as(expanded)
    result = torch.gather(expanded, dim=-1, index=indices)
    return result.movedim(-1, time_dim + 1)


def compute_tau(
    d_original: torch.Tensor | float,
    d_intervened: torch.Tensor | float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the frozen float64 temporal-order responsibility quantity."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    original = torch.as_tensor(d_original, dtype=torch.float64)
    intervened = torch.as_tensor(d_intervened, dtype=torch.float64)
    tau = torch.abs(original - intervened) / (
        torch.abs(original) + torch.abs(intervened) + float(eps)
    )
    return tau.clamp_(0.0, 1.0)


def compute_level_rms(tau_values: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Compute one level's RMS over videos and interventions."""

    values = torch.as_tensor(tau_values, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("a hierarchy level must contain at least one tau value")
    return torch.sqrt(torch.mean(torch.square(values)))


def compute_htor(level_rms: Mapping[int, torch.Tensor | float]) -> torch.Tensor:
    """Compute HTOR with equal weight for every hierarchy level."""

    if not level_rms:
        raise ValueError("at least one hierarchy level is required")
    ordered = [level_rms[level] for level in sorted(level_rms)]
    values = torch.as_tensor(ordered, dtype=torch.float64)
    return torch.sqrt(torch.mean(torch.square(values)))


def verify_intervention_identity(
    interventions: Sequence[TemporalIntervention], temporal_length: int
) -> None:
    """Fail if a manifest loses, duplicates, or reorders an invalid index."""

    t = validate_temporal_length(temporal_length)
    permutations = []
    for intervention in interventions:
        permutation = build_temporal_permutation(t, intervention)
        if len(set(permutation)) != t:
            raise ValueError("intervention duplicates a sampled frame")
        if sorted(permutation) != list(range(t)):
            raise ValueError("intervention drops a sampled frame")
        permutations.append(permutation)
    if len(set(permutations)) != len(permutations):
        raise ValueError("duplicate temporal interventions were generated")


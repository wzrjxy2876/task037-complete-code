"""Coverage-aware repair for a D_rel-selected pruning set.

Contribution Fields are used only to repair the set of removed units.  They
never replace, extend, or blend with the three-dimensional descriptor and do
not define a second importance score.
"""

from typing import Iterable

import torch
import torch.nn.functional as F


_DEFAULT_CHUNK_SIZE = 1024


def _prepare_fields(fields: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized flattened Contribution Fields.

    ``fields`` has shape ``[N,T,H,W]``: ``N`` pruning units, ``T`` temporal
    positions, and ``H/W`` spatial positions.  Flattening is performed only
    in memory for cosine similarity, producing ``[N,D]`` with
    ``D = T * H * W``; the stored fields remain four-dimensional.
    """
    contribution_fields = torch.as_tensor(fields).detach()
    if contribution_fields.ndim != 4:
        raise ValueError(
            "fields must have shape [N,T,H,W], "
            f"but received {tuple(contribution_fields.shape)}"
        )
    if any(size == 0 for size in contribution_fields.shape[1:]):
        raise ValueError("fields contains an empty temporal or spatial axis")
    if contribution_fields.shape[0] == 0:
        flattened_size = 1
        for size in contribution_fields.shape[1:]:
            flattened_size *= size
        return contribution_fields.to(dtype=torch.float32).reshape(
            0, flattened_size
        )

    contribution_fields = contribution_fields.to(dtype=torch.float32)
    if not torch.isfinite(contribution_fields).all():
        raise ValueError("fields contains NaN or infinity")

    flattened = contribution_fields.reshape(
        contribution_fields.shape[0], -1
    )  # [N,D]
    return F.normalize(flattened, p=2, dim=1)


def _validated_indices(
    indices: Iterable[int], num_units: int, name: str
) -> torch.Tensor:
    values = [int(index) for index in indices]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate indices")
    if any(index < 0 or index >= num_units for index in values):
        raise IndexError(f"{name} contains an index outside [0, {num_units})")
    return torch.tensor(sorted(values), dtype=torch.long)


def _validated_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


def _scalar_remove_indices(scores: torch.Tensor, remove_num: int) -> torch.Tensor:
    """Return the ``remove_num`` lowest-score indices with stable tie breaks."""
    if isinstance(remove_num, bool) or int(remove_num) != remove_num:
        raise ValueError("remove_num must be an integer")
    remove_num = int(remove_num)
    if remove_num < 0 or remove_num > scores.numel():
        raise ValueError(
            f"remove_num must be in [0, {scores.numel()}], got {remove_num}"
        )

    score_values = scores.detach().cpu().tolist()
    order = sorted(
        range(scores.numel()), key=lambda index: (score_values[index], index)
    )
    return torch.tensor(order[:remove_num], dtype=torch.long)  # [R]


def _chunked_best_similarity(
    normalized_fields: torch.Tensor,
    reference_indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Return each unit's best cosine match using bounded matrices.

    ``normalized_fields`` is ``[N,D]`` and ``reference_indices`` is ``[K]``.
    At most ``chunk_size * chunk_size`` cosine values are materialized at
    once, so this function can run on either CPU or CUDA without an ``N×K``
    allocation.
    """
    num_units = normalized_fields.shape[0]
    if num_units == 0:
        return normalized_fields.new_empty((0,))
    if reference_indices.numel() == 0:
        return normalized_fields.new_zeros((num_units,))

    device = normalized_fields.device
    references = reference_indices.to(device=device)
    best_values = normalized_fields.new_empty((num_units,))  # [N]

    for query_start in range(0, num_units, chunk_size):
        query_end = min(query_start + chunk_size, num_units)
        query = normalized_fields[query_start:query_end]  # [Bq,D]
        query_best = normalized_fields.new_full(
            (query_end - query_start,), -torch.inf
        )  # [Bq]

        for ref_start in range(0, references.numel(), chunk_size):
            ref_end = min(ref_start + chunk_size, references.numel())
            ref_indices = references[ref_start:ref_end]  # [Br]
            similarities = query @ normalized_fields[ref_indices].T  # [Bq,Br]
            similarities = similarities.clamp(min=-1.0, max=1.0)
            query_best = torch.maximum(
                query_best, similarities.max(dim=1).values
            )

        best_values[query_start:query_end] = query_best

    return best_values


def _coverage_from_normalized(
    normalized_fields: torch.Tensor,
    keep_indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute mean best-match coverage from normalized ``[N,D]`` fields."""
    if normalized_fields.shape[0] == 0 or keep_indices.numel() == 0:
        return normalized_fields.new_tensor(0.0)
    best_values = _chunked_best_similarity(
        normalized_fields, keep_indices, chunk_size
    )  # [N]
    return best_values.mean()


def functional_coverage(
    fields: torch.Tensor,
    keep_indices: Iterable[int],
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ``Coverage(K) = mean_i max_(j in K) cos(F_i, F_j)``.

    ``fields`` is ``[N,T,H,W]`` and ``K`` contains retained-unit indices.
    The returned scalar is created on the same device as ``fields``.  Query
    and reference units are chunked, while the complete temporal and spatial
    content of every field is retained.
    """
    chunk_size = _validated_chunk_size(chunk_size)
    normalized_fields = _prepare_fields(fields)  # [N,D]
    keep = _validated_indices(
        keep_indices, normalized_fields.shape[0], "keep_indices"
    )  # [K]
    return _coverage_from_normalized(normalized_fields, keep, chunk_size)


def greedy_coverage_repair(
    scalar_scores: torch.Tensor,
    fields: torch.Tensor,
    remove_num: int,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Repair the D_rel-selected removal set without changing its size.

    ``scalar_scores`` is ``[N]`` and contains only D_rel. ``fields`` is
    ``[N,T,H,W]``.  The procedure is:

    1. select the ``remove_num`` lowest D_rel values as ``initial_remove``;
    2. compute the initial Functional Coverage;
    3. restore the removed unit with the largest temporary coverage gain;
    4. delete the lowest-D_rel unit from the original keep set;
    5. accept the equal-budget swap only if final coverage improves.

    The returned CPU tensor is ``[R]`` with ``R == remove_num``. Contribution
    Fields modify only membership of the removal set; no importance score is
    computed from coverage.
    """
    chunk_size = _validated_chunk_size(chunk_size)
    scores = torch.as_tensor(scalar_scores).detach().to(
        dtype=torch.float32
    ).reshape(-1)  # [N]
    if not torch.isfinite(scores).all():
        raise ValueError("scalar_scores contains NaN or infinity")

    normalized_fields = _prepare_fields(fields)  # [N,D]
    num_units = normalized_fields.shape[0]
    if scores.numel() != num_units:
        raise ValueError(
            f"scalar_scores has {scores.numel()} units but fields has "
            f"{num_units}"
        )

    initial_remove = _scalar_remove_indices(scores, remove_num)  # [R]
    remove_count = initial_remove.numel()
    if remove_count == 0:
        return initial_remove
    if remove_count >= num_units:
        raise ValueError("coverage repair requires at least one retained unit")

    remove_set = set(initial_remove.tolist())
    initial_keep = torch.tensor(
        [index for index in range(num_units) if index not in remove_set],
        dtype=torch.long,
    )  # [K]

    device = normalized_fields.device
    remove_device = initial_remove.to(device=device)  # [R]
    best_initial = _chunked_best_similarity(
        normalized_fields, initial_keep, chunk_size
    )  # [N]
    initial_coverage = best_initial.mean()

    # Evaluate restoring every removed unit.  The accumulator is [R], while
    # each temporary cosine matrix is at most [chunk_size, chunk_size].
    restored_sums = normalized_fields.new_zeros((remove_count,))  # [R]
    for query_start in range(0, num_units, chunk_size):
        query_end = min(query_start + chunk_size, num_units)
        query = normalized_fields[query_start:query_end]  # [Bq,D]
        base_best = best_initial[query_start:query_end, None]  # [Bq,1]

        for remove_start in range(0, remove_count, chunk_size):
            remove_end = min(remove_start + chunk_size, remove_count)
            candidates = remove_device[remove_start:remove_end]  # [Br]
            similarities = query @ normalized_fields[candidates].T  # [Bq,Br]
            similarities = similarities.clamp(min=-1.0, max=1.0)
            restored_sums[remove_start:remove_end] += torch.maximum(
                base_best, similarities
            ).sum(dim=0)

    restored_coverages = restored_sums / num_units  # [R]
    restore_position = int(torch.argmax(restored_coverages).item())
    restore_index = int(initial_remove[restore_position].item())

    # The replacement must come from the original keep set.  Allowing the
    # just-restored low-D_rel unit here would immediately undo every repair.
    score_values = scores.cpu().tolist()
    replacement_index = min(
        initial_keep.tolist(), key=lambda index: (score_values[index], index)
    )
    candidate_keep_values = [
        index for index in initial_keep.tolist() if index != replacement_index
    ]
    candidate_keep_values.append(restore_index)
    candidate_keep = torch.tensor(
        sorted(candidate_keep_values), dtype=torch.long
    )  # [K]
    repaired_coverage = _coverage_from_normalized(
        normalized_fields, candidate_keep, chunk_size
    )

    if repaired_coverage > initial_coverage:
        repaired_remove = (remove_set - {restore_index}) | {replacement_index}
    else:
        repaired_remove = remove_set

    result = torch.tensor(sorted(repaired_remove), dtype=torch.long)  # [R]
    if result.numel() != remove_count:
        raise RuntimeError("coverage repair changed the removal count")
    return result


def coverage_aware_selection(
    scalar_scores: torch.Tensor,
    fields: torch.Tensor,
    remove_num: int,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compatibility wrapper for the complete equal-budget repair."""
    return greedy_coverage_repair(
        scalar_scores, fields, remove_num, chunk_size=chunk_size
    )

from __future__ import annotations

"""Numerically safe LG-FRF similarity and functional redundancy utilities."""

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


NUMERICAL_EPS = 1e-8
REFERENCE_TIME_BINS = 8
MAX_ACTIVATION_SAMPLES = 2048
BMS_TOLERANCE = 1e-4
BMS_MAX_ITERATIONS = 60
DIAGNOSTIC_RANDOM_SEED = 3407
DIAGNOSTIC_SAMPLE_COUNT = 200000
TOP_CANDIDATE_RATIO = 0.2


def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
    if logger is not None:
        logger(message)


def assert_finite(
    name: str,
    tensor: torch.Tensor,
    bounds: Optional[Tuple[float, float]] = None,
) -> None:
    """Raise immediately when a statistic is invalid or outside its contract."""
    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    if bounds is not None:
        lower, upper = bounds
        actual_min = float(tensor.min())
        actual_max = float(tensor.max())
        if actual_min < lower - 1e-6 or actual_max > upper + 1e-6:
            raise ValueError(
                f"{name} range [{actual_min:.6g}, {actual_max:.6g}] "
                f"violates [{lower}, {upper}]"
            )


def tensor_summary(name: str, tensor: torch.Tensor) -> str:
    assert_finite(name, tensor)
    return (
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={float(tensor.min()):.6f}, max={float(tensor.max()):.6f}, "
        f"mean={float(tensor.float().mean()):.6f}"
    )


def robust_scale_01(x: torch.Tensor) -> torch.Tensor:
    """Scale one layer statistic to [0, 1] after 5/95 percentile clipping."""
    if x.ndim != 1:
        raise ValueError(f"robust_scale_01 expects [U], got {tuple(x.shape)}")
    assert_finite("robust_scale_input", x)
    x_float = x.float()
    if x_float.numel() < 2:
        return torch.zeros_like(x_float)
    lower = torch.quantile(x_float, 0.05)
    upper = torch.quantile(x_float, 0.95)
    clipped = x_float.clamp(min=lower, max=upper)
    value_min = clipped.min()
    value_max = clipped.max()
    if float(value_max - value_min) <= NUMERICAL_EPS:
        scaled = torch.zeros_like(clipped)
    else:
        scaled = (clipped - value_min) / (value_max - value_min)
    scaled = scaled.clamp_(0.0, 1.0)
    assert_finite("robust_scale_output", scaled, (0.0, 1.0))
    return scaled


def safe_standardize_descriptors(
    descriptors: torch.Tensor,
    logger: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Standardize each of the three descriptor coordinates without NaNs."""
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError(
            f"descriptors must have shape [N, 3], got {tuple(descriptors.shape)}"
        )
    assert_finite("descriptors_raw", descriptors)
    _log(logger, tensor_summary("descriptors_raw", descriptors))
    mean = descriptors.mean(dim=0, keepdim=True)
    std = descriptors.std(dim=0, unbiased=False, keepdim=True)
    stable_std = torch.where(std <= NUMERICAL_EPS, torch.ones_like(std), std)
    normalized = (descriptors - mean) / stable_std
    normalized[:, std.squeeze(0) <= NUMERICAL_EPS] = 0.0
    assert_finite("descriptors_normalized", normalized)
    _log(logger, tensor_summary("descriptors_normalized", normalized))
    return normalized


def normalize_frame_probabilities(response: torch.Tensor) -> torch.Tensor:
    """Convert [B, U, T, H, W] responses into per-frame spatial probabilities."""
    if response.ndim != 5:
        raise ValueError(f"response must be [B,U,T,H,W], got {tuple(response.shape)}")
    magnitude = response.float().abs()
    energy = magnitude.sum(dim=(-2, -1), keepdim=True)
    probabilities = torch.where(
        energy > NUMERICAL_EPS,
        magnitude / energy.clamp_min(NUMERICAL_EPS),
        torch.zeros_like(magnitude),
    )
    assert_finite("frame_probabilities", probabilities, (0.0, 1.0))
    return probabilities


def compute_frame_overlap(probabilities: torch.Tensor) -> torch.Tensor:
    """Return cosine overlap matrices [B, U, T, T]."""
    if probabilities.ndim != 5:
        raise ValueError(
            f"probabilities must be [B,U,T,H,W], got {tuple(probabilities.shape)}"
        )
    batch, units, frames, height, width = probabilities.shape
    flat = probabilities.reshape(batch, units, frames, height * width)
    norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    normalized = torch.where(
        norm > NUMERICAL_EPS,
        flat / norm.clamp_min(NUMERICAL_EPS),
        torch.zeros_like(flat),
    )
    overlap = torch.matmul(normalized, normalized.transpose(-1, -2)).clamp_(0.0, 1.0)
    assert_finite("frame_overlap", overlap, (0.0, 1.0))
    return overlap


def compute_frame_centroid(probabilities: torch.Tensor) -> torch.Tensor:
    """Return normalized spatial centroids [B, U, T, 2] as (x, y)."""
    if probabilities.ndim != 5:
        raise ValueError(
            f"probabilities must be [B,U,T,H,W], got {tuple(probabilities.shape)}"
        )
    height, width = probabilities.shape[-2:]
    x_coords = torch.linspace(
        0.0, 1.0, steps=width, device=probabilities.device, dtype=probabilities.dtype
    )
    y_coords = torch.linspace(
        0.0, 1.0, steps=height, device=probabilities.device, dtype=probabilities.dtype
    )
    centroid_x = (probabilities * x_coords.view(1, 1, 1, 1, width)).sum(
        dim=(-2, -1)
    )
    centroid_y = (probabilities * y_coords.view(1, 1, 1, height, 1)).sum(
        dim=(-2, -1)
    )
    centroid = torch.stack((centroid_x, centroid_y), dim=-1).clamp_(0.0, 1.0)
    assert_finite("frame_centroid", centroid, (0.0, 1.0))
    return centroid


def compute_motion_retention(probabilities: torch.Tensor) -> torch.Tensor:
    """Return normalized displacement retention matrices [B, U, T, T]."""
    centroid = compute_frame_centroid(probabilities)
    displacement = torch.cdist(centroid, centroid) / math.sqrt(2.0)
    retention = (1.0 - displacement).clamp_(0.0, 1.0)
    assert_finite("motion_retention", retention, (0.0, 1.0))
    return retention


def resize_relation_matrix(
    relation: torch.Tensor, target_t: int = REFERENCE_TIME_BINS
) -> torch.Tensor:
    """Resize [..., T, T] relation matrices to [..., target_t, target_t]."""
    if relation.ndim < 3 or relation.shape[-1] != relation.shape[-2]:
        raise ValueError(f"relation must end in [T,T], got {tuple(relation.shape)}")
    original_shape = relation.shape
    if original_shape[-1] == target_t:
        resized = relation
    else:
        flat = relation.reshape(-1, 1, original_shape[-2], original_shape[-1])
        flat = F.interpolate(
            flat.float(),
            size=(target_t, target_t),
            mode="bilinear",
            align_corners=False,
        )
        resized = flat.reshape(*original_shape[:-2], target_t, target_t)
    resized = resized.clamp_(0.0, 1.0)
    assert_finite("resized_relation", resized, (0.0, 1.0))
    return resized


def build_local_global_masks(
    time_bins: int = REFERENCE_TIME_BINS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exhaustive, disjoint non-diagonal local/global frame-pair masks."""
    indices = torch.arange(time_bins)
    distance = (indices[:, None] - indices[None, :]).abs()
    quarter = max(1, time_bins // 4)
    local_mask = (distance > 0) & (distance <= quarter)
    global_mask = distance > quarter
    if torch.any(local_mask & global_mask):
        raise AssertionError("local and global masks overlap")
    off_diagonal = ~torch.eye(time_bins, dtype=torch.bool)
    if not torch.equal(local_mask | global_mask, off_diagonal):
        raise AssertionError("local/global masks do not cover all non-diagonal pairs")
    return local_mask, global_mask


def build_lag_matrix(
    time_bins: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return deterministic frame lag magnitudes as Tensor[T,T]."""
    indices = torch.arange(time_bins, dtype=torch.float32, device=device)
    return (indices[:, None] - indices[None, :]).abs()


def build_displacement_rate(
    motion_retention_relation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert retention Tensor[N,T,T] to displacement and per-lag rate."""
    if motion_retention_relation.ndim != 3:
        raise ValueError("motion retention must have shape [N,T,T]")
    if motion_retention_relation.shape[-1] != motion_retention_relation.shape[-2]:
        raise ValueError("motion retention matrices must be square")
    assert_finite(
        "motion_retention_relation", motion_retention_relation, (0.0, 1.0)
    )
    displacement = (1.0 - motion_retention_relation.float()).clamp_(0.0, 1.0)
    lag = build_lag_matrix(
        motion_retention_relation.shape[-1],
        device=motion_retention_relation.device,
    ).clamp_min_(1.0)
    displacement_rate = (displacement / lag).clamp_(0.0, 1.0)
    displacement = torch.nan_to_num(displacement, nan=0.0, posinf=1.0, neginf=0.0)
    displacement_rate = torch.nan_to_num(
        displacement_rate, nan=0.0, posinf=1.0, neginf=0.0
    )
    assert_finite("displacement_relation", displacement, (0.0, 1.0))
    assert_finite("displacement_rate_relation", displacement_rate, (0.0, 1.0))
    return displacement, displacement_rate


def center_relation_features(features: torch.Tensor) -> torch.Tensor:
    """Remove each unit's own baseline from relation features Tensor[N,L]."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,L], got {tuple(features.shape)}")
    centered = features.float() - features.float().mean(dim=1, keepdim=True)
    assert_finite("centered_relation_features", centered)
    return centered


def summarize_centered_features(
    features: torch.Tensor, eps: float = NUMERICAL_EPS
) -> Dict[str, Any]:
    """Summarize centered relation features Tensor[N,L]."""
    centered = center_relation_features(features)
    norms = torch.linalg.vector_norm(centered, dim=1)
    return {
        "shape": list(centered.shape),
        "mean_absolute_value": float(centered.abs().mean()),
        "std": float(centered.std(unbiased=False)),
        "zero_norm_unit_count": int((norms < eps).sum()),
        "finite_ratio": float(torch.isfinite(centered).float().mean()),
    }


def _pairwise_centered_positive_cosine(
    features: torch.Tensor,
    chunk_size: int = 256,
    eps: float = NUMERICAL_EPS,
    storage_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return centered positive cosine similarity Tensor[N,N] on CPU.

    Two near-constant rows receive similarity 1 only when their original
    constant levels are approximately equal. A single zero-norm row, two
    unequal constant rows, and every negative correlation receive 0.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N,L], got {tuple(features.shape)}")
    original = features.float()
    assert_finite("centered_cosine_input", original)
    centered = center_relation_features(original)
    norms = torch.linalg.vector_norm(centered, dim=1)
    means = original.mean(dim=1)
    zero_norm = norms < eps
    units = original.shape[0]
    similarity = torch.empty(
        (units, units), dtype=storage_dtype, device="cpu"
    )

    for start in range(0, units, chunk_size):
        end = min(start + chunk_size, units)
        numerator = centered[start:end] @ centered.transpose(0, 1)
        denominator = norms[start:end, None] * norms[None, :]
        normal_pair = (~zero_norm[start:end, None]) & (~zero_norm[None, :])
        chunk = torch.zeros_like(numerator, dtype=torch.float32)
        chunk[normal_pair] = (
            numerator[normal_pair] / denominator[normal_pair].clamp_min(eps)
        )
        both_constant = zero_norm[start:end, None] & zero_norm[None, :]
        same_constant = torch.isclose(
            means[start:end, None],
            means[None, :],
            rtol=1e-5,
            atol=1e-6,
        )
        chunk[both_constant & same_constant] = 1.0
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=0.0)
        chunk = chunk.clamp_(-1.0, 1.0).clamp_(0.0, 1.0)
        similarity[start:end].copy_(
            chunk.to(device="cpu", dtype=storage_dtype)
        )

    similarity.fill_diagonal_(1.0)
    assert_finite("centered_positive_cosine", similarity, (0.0, 1.0))
    if not torch.allclose(
        similarity, similarity.transpose(0, 1), atol=2e-3, rtol=0.0
    ):
        raise AssertionError("centered positive cosine is not symmetric")
    return similarity


def summarize_matrix_distribution(
    matrix: torch.Tensor,
    sample_count: int = DIAGNOSTIC_SAMPLE_COUNT,
    seed: int = DIAGNOSTIC_RANDOM_SEED,
) -> Dict[str, Any]:
    """Return reproducible distribution diagnostics without a full float copy."""
    assert_finite("summary_matrix", matrix)
    cpu = matrix.detach().cpu()
    flat = cpu.reshape(-1)
    total = flat.numel()
    running_sum = 0.0
    running_square_sum = 0.0
    running_min = float("inf")
    running_max = float("-inf")
    reduction_chunk = 1000000
    for start in range(0, total, reduction_chunk):
        values = flat[start:start + reduction_chunk].float()
        running_sum += float(values.sum())
        running_square_sum += float(values.square().sum())
        running_min = min(running_min, float(values.min()))
        running_max = max(running_max, float(values.max()))
    mean = running_sum / max(total, 1)
    variance = max(running_square_sum / max(total, 1) - mean * mean, 0.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if total <= sample_count:
        sample = flat.float()
    else:
        sample_indices = torch.randint(
            0, total, (sample_count,), generator=generator
        )
        sample = flat[sample_indices].float()
    quantiles = torch.quantile(
        sample, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    )
    summary = {
        "shape": list(cpu.shape),
        "min": running_min,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": running_max,
        "mean": mean,
        "std": math.sqrt(variance),
    }
    if cpu.ndim == 2 and cpu.shape[0] == cpu.shape[1]:
        units = cpu.shape[0]
        diagonal = torch.diagonal(cpu).float()
        summary["diagonal_mean"] = float(diagonal.mean())
        if units > 1:
            pair_samples = min(sample_count, units * (units - 1))
            row_indices = torch.randint(
                0, units, (pair_samples,), generator=generator
            )
            column_indices = torch.randint(
                0, units - 1, (pair_samples,), generator=generator
            )
            column_indices += (column_indices >= row_indices).long()
            off_diagonal = cpu[row_indices, column_indices].float()
            summary["off_diagonal_mean"] = float(off_diagonal.mean())
            summary["off_diagonal_std"] = float(
                off_diagonal.std(unbiased=False)
            )
        else:
            summary["off_diagonal_mean"] = 0.0
            summary["off_diagonal_std"] = 0.0
    return summary


def _geometric_mean_matrices(
    first: torch.Tensor,
    second: torch.Tensor,
    chunk_size: int,
    storage_dtype: torch.dtype,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine two similarity matrices Tensor[N,N] without full float copies."""
    if first.shape != second.shape:
        raise ValueError("geometric-mean matrices must have identical shape")
    if output is None:
        output = torch.empty(first.shape, dtype=storage_dtype, device="cpu")
    elif output.shape != first.shape or output.dtype != storage_dtype:
        raise ValueError("geometric-mean output has incompatible shape or dtype")
    for start in range(0, first.shape[0], chunk_size):
        end = min(start + chunk_size, first.shape[0])
        chunk = torch.sqrt(
            (
                first[start:end].float()
                * second[start:end].float()
            ).clamp_min(0.0)
        )
        output[start:end].copy_(chunk.to(storage_dtype))
    return output


def compute_function_similarity(
    overlap_relation: torch.Tensor,
    motion_retention_relation: torch.Tensor,
    local_mask: Optional[torch.Tensor] = None,
    global_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
    storage_dtype: torch.dtype = torch.float32,
    return_details: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute centered S_loc, S_glo, and S_func for relation Tensor[N,T,T]."""
    if overlap_relation.shape != motion_retention_relation.shape:
        raise ValueError("overlap and motion relation fields must have identical shapes")
    if overlap_relation.ndim != 3:
        raise ValueError(
            f"relation fields must be [N,T,T], got {tuple(overlap_relation.shape)}"
        )
    units, time_a, time_b = overlap_relation.shape
    if time_a != time_b:
        raise ValueError("relation fields must be square")
    if local_mask is None or global_mask is None:
        local_mask, global_mask = build_local_global_masks(time_a)
    local_mask = local_mask.to(overlap_relation.device)
    global_mask = global_mask.to(overlap_relation.device)

    displacement, displacement_rate = build_displacement_rate(
        motion_retention_relation
    )
    feature_sets = {
        "overlap_local_centered": overlap_relation[:, local_mask],
        "overlap_global_centered": overlap_relation[:, global_mask],
        "motion_local_centered": displacement_rate[:, local_mask],
        "motion_global_centered": displacement_rate[:, global_mask],
    }
    feature_summaries = {
        name: summarize_centered_features(features)
        for name, features in feature_sets.items()
    }
    similarity_summaries = {}
    s_o_local = _pairwise_centered_positive_cosine(
        feature_sets["overlap_local_centered"],
        chunk_size=chunk_size,
        storage_dtype=storage_dtype,
    )
    similarity_summaries["S_O_loc"] = summarize_matrix_distribution(s_o_local)
    s_m_local = _pairwise_centered_positive_cosine(
        feature_sets["motion_local_centered"],
        chunk_size=chunk_size,
        storage_dtype=storage_dtype,
    )
    similarity_summaries["S_M_loc"] = summarize_matrix_distribution(s_m_local)
    s_local = _geometric_mean_matrices(
        s_o_local, s_m_local, chunk_size, storage_dtype, output=s_o_local
    )
    del s_m_local

    s_o_global = _pairwise_centered_positive_cosine(
        feature_sets["overlap_global_centered"],
        chunk_size=chunk_size,
        storage_dtype=storage_dtype,
    )
    similarity_summaries["S_O_glo"] = summarize_matrix_distribution(s_o_global)
    s_m_global = _pairwise_centered_positive_cosine(
        feature_sets["motion_global_centered"],
        chunk_size=chunk_size,
        storage_dtype=storage_dtype,
    )
    similarity_summaries["S_M_glo"] = summarize_matrix_distribution(s_m_global)
    s_global = _geometric_mean_matrices(
        s_o_global, s_m_global, chunk_size, storage_dtype, output=s_o_global
    )
    del s_m_global

    s_function = _geometric_mean_matrices(
        s_local, s_global, chunk_size, storage_dtype
    )
    for name, value in (
        ("S_loc", s_local),
        ("S_glo", s_global),
        ("S_func", s_function),
    ):
        value.copy_(
            torch.nan_to_num(
                value.float(), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0).to(storage_dtype)
        )
        value.fill_diagonal_(1.0)
        assert_finite(name, value, (0.0, 1.0))
        if not torch.allclose(value, value.transpose(0, 1), atol=2e-3, rtol=0.0):
            raise AssertionError(f"{name} is not symmetric")
        similarity_summaries[name] = summarize_matrix_distribution(value)

    if not return_details:
        return s_local, s_global, s_function
    details = {
        "displacement_relation": displacement,
        "displacement_rate_relation": displacement_rate,
        "feature_summaries": feature_summaries,
        "similarity_summaries": similarity_summaries,
    }
    return s_local, s_global, s_function, details


def compute_intra_cluster_functional_redundancy(
    groups: Sequence[Sequence[int]],
    function_similarity: torch.Tensor,
) -> torch.Tensor:
    """Return per-unit LG-FRF redundancy ``[N]`` inside descriptor clusters.

    ``function_similarity`` is the existing centered positive-cosine
    ``S_func [N,N]``. For a unit ``i`` in descriptor cluster ``C``, the output
    is ``mean(S_func[i,j] for j in C if j != i)``. A singleton has no redundant
    peer and therefore receives the neutral nonredundant value ``0.0``.
    """
    if function_similarity.ndim != 2:
        raise ValueError("function_similarity must have shape [N,N]")
    units_a, units_b = function_similarity.shape
    if units_a < 1 or units_a != units_b:
        raise ValueError("function_similarity must be a nonempty square matrix")
    if not groups:
        raise ValueError("descriptor groups must be nonempty")
    flattened = [int(member) for group in groups for member in group]
    if sorted(flattened) != list(range(units_a)) or len(set(flattened)) != units_a:
        raise ValueError("descriptor groups must cover every unit exactly once")
    assert_finite("functional_redundancy.S_func", function_similarity, (0.0, 1.0))

    redundancy = torch.zeros(
        units_a,
        dtype=torch.float32,
        device=function_similarity.device,
    )
    for members in groups:
        if not members:
            raise ValueError("descriptor groups must not contain empty clusters")
        if len(members) == 1:
            redundancy[int(members[0])] = 0.0
            continue
        member_index = torch.as_tensor(
            members,
            dtype=torch.long,
            device=function_similarity.device,
        )
        within_cluster = function_similarity[member_index][:, member_index].float()
        diagonal = torch.diagonal(within_cluster)
        redundancy[member_index] = (
            within_cluster.sum(dim=1) - diagonal
        ) / float(len(members) - 1)
    redundancy = torch.nan_to_num(
        redundancy, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp_(0.0, 1.0)
    assert_finite("functional_redundancy.values", redundancy, (0.0, 1.0))
    return redundancy


def compute_function_uniqueness(
    functional_redundancy: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Return parameter-free LGFR uniqueness as ``1 - redundancy``."""
    redundancy = torch.as_tensor(functional_redundancy, dtype=torch.float32)
    if redundancy.ndim != 1 or redundancy.numel() < 1:
        raise ValueError("functional_redundancy must have shape [N]")
    assert_finite("functional_redundancy.values", redundancy, (0.0, 1.0))
    uniqueness = (1.0 - redundancy).clamp_(0.0, 1.0)
    assert_finite("functional_uniqueness.values", uniqueness, (0.0, 1.0))
    return uniqueness


def _aligned_unit_vector(
    values: torch.Tensor | Sequence[float],
    unit_count: int,
    name: str,
) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if vector.shape != (unit_count,):
        raise ValueError(f"{name} must have shape [N]")
    if not torch.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return vector


def select_descriptor_cluster_representatives(
    groups: Sequence[Sequence[int]],
    descriptor_importance: torch.Tensor | Sequence[float],
    functional_uniqueness: torch.Tensor | Sequence[float],
) -> list[int]:
    """Select one LGFR representative from each descriptor cluster.

    The candidate pool is the fixed top 20 percent by descriptor importance.
    The representative is the most functionally unique candidate in that pool.
    LGFR never orders pruning candidates and introduces no user parameter.
    """
    if not groups:
        raise ValueError("descriptor groups must be nonempty")
    unit_count = sum(len(group) for group in groups)
    flattened = [int(member) for group in groups for member in group]
    if sorted(flattened) != list(range(unit_count)):
        raise ValueError("descriptor groups must cover every unit exactly once")
    importance = _aligned_unit_vector(
        descriptor_importance, unit_count, "descriptor_importance"
    )
    uniqueness = _aligned_unit_vector(
        functional_uniqueness, unit_count, "functional_uniqueness"
    )

    representatives = []
    for members in groups:
        members = [int(member) for member in members]
        candidate_count = max(
            1,
            int(math.ceil(TOP_CANDIDATE_RATIO * len(members))),
        )
        top_importance_candidates = sorted(
            members,
            key=lambda index: (-float(importance[index]), index),
        )[:candidate_count]
        representative = max(
            top_importance_candidates,
            key=lambda index: (float(uniqueness[index]), -index),
        )
        representatives.append(representative)
    return representatives


def summarize_functional_redundancy(
    groups: Sequence[Sequence[int]],
    functional_redundancy: torch.Tensor | Sequence[float],
    representatives: Sequence[int],
) -> Dict[str, Any]:
    """Summarize descriptor clusters and their LG-FRF redundancy ``[N]``."""
    unit_count = sum(len(group) for group in groups)
    redundancy = _aligned_unit_vector(
        functional_redundancy, unit_count, "functional_redundancy"
    )
    if len(representatives) != len(groups):
        raise ValueError("representatives must contain one unit per cluster")
    sizes = torch.as_tensor([len(group) for group in groups], dtype=torch.float64)
    return {
        "cluster_count": len(groups),
        "average_cluster_size": float(sizes.mean()),
        "average_functional_redundancy": float(redundancy.mean()),
        "representative_count": len(representatives),
        "protected_units": [int(index) for index in representatives],
        "global_candidate_count": unit_count - len(representatives),
    }


def descriptor_kernel(points: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distance = torch.cdist(points, points)
    kernel = torch.exp(-(distance.square()) / (2.0 * sigma * sigma))
    kernel.fill_diagonal_(1.0)
    assert_finite("descriptor_kernel", kernel, (0.0, 1.0))
    return kernel


def joint_kernel(
    points: torch.Tensor,
    sigma: float,
    function_similarity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    descriptor = descriptor_kernel(points, sigma)
    similarity = function_similarity.to(
        device=descriptor.device, dtype=descriptor.dtype
    )
    if similarity.shape != descriptor.shape:
        raise ValueError("function similarity shape does not match descriptor kernel")
    joint = descriptor * similarity
    joint.fill_diagonal_(1.0)
    assert_finite("joint_kernel", joint, (0.0, 1.0))
    return descriptor, joint


def mean_shift_bms(
    descriptors_normalized: torch.Tensor,
    sigma: float,
    function_similarity: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
    max_iterations: int = BMS_MAX_ITERATIONS,
    tolerance: float = BMS_TOLERANCE,
) -> Tuple[list, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run original or function-guided BMS with a row-chunked kernel."""
    if descriptors_normalized.ndim != 2 or descriptors_normalized.shape[1] != 3:
        raise ValueError("descriptors_normalized must be [N,3]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    assert_finite("bms_input", descriptors_normalized)
    points = descriptors_normalized.clone()
    units = points.shape[0]
    if function_similarity is not None:
        if tuple(function_similarity.shape) != (units, units):
            raise ValueError("function_similarity must be [N,N]")
        assert_finite("function_similarity", function_similarity, (0.0, 1.0))

    initial_desc_min = float("inf")
    initial_desc_max = float("-inf")
    initial_desc_sum = 0.0
    initial_joint_min = float("inf")
    initial_joint_max = float("-inf")
    initial_joint_sum = 0.0
    initial_abs_difference_sum = 0.0
    initial_abs_difference_max = 0.0
    initial_descriptor_abs_sum = 0.0
    initial_joint_lower_count = 0
    initial_joint_less_half_count = 0
    initial_count = 0
    iterations = 0
    max_movement_value = 0.0

    for step in range(max_iterations):
        updated = torch.empty_like(points)
        for start in range(0, units, chunk_size):
            end = min(start + chunk_size, units)
            distance = torch.cdist(points[start:end], points)
            desc = torch.exp(-(distance.square()) / (2.0 * sigma * sigma))
            if function_similarity is None:
                kernel = desc
            else:
                relation = function_similarity[start:end].to(
                    device=points.device, dtype=points.dtype
                )
                kernel = desc * relation
            row_indices = torch.arange(end - start, device=points.device)
            kernel[row_indices, torch.arange(start, end, device=points.device)] = 1.0
            if not torch.isfinite(kernel).all():
                raise FloatingPointError("joint BMS kernel contains NaN or Inf")
            updated[start:end] = (kernel @ points) / (
                kernel.sum(dim=1, keepdim=True) + NUMERICAL_EPS
            )
            if step == 0:
                count = desc.numel()
                absolute_difference = (kernel - desc).abs()
                initial_desc_min = min(initial_desc_min, float(desc.min()))
                initial_desc_max = max(initial_desc_max, float(desc.max()))
                initial_desc_sum += float(desc.sum())
                initial_joint_min = min(initial_joint_min, float(kernel.min()))
                initial_joint_max = max(initial_joint_max, float(kernel.max()))
                initial_joint_sum += float(kernel.sum())
                initial_abs_difference_sum += float(absolute_difference.sum())
                initial_abs_difference_max = max(
                    initial_abs_difference_max, float(absolute_difference.max())
                )
                initial_descriptor_abs_sum += float(desc.abs().sum())
                initial_joint_lower_count += int(
                    (kernel < desc - NUMERICAL_EPS).sum()
                )
                initial_joint_less_half_count += int(
                    (kernel < 0.5 * desc).sum()
                )
                initial_count += count
        movement = torch.linalg.vector_norm(updated - points, dim=1)
        max_movement_value = float(movement.max())
        points = updated
        iterations = step + 1
        if max_movement_value < tolerance:
            break

    rounded = torch.round(points * 100.0) / 100.0
    _, inverse = torch.unique(rounded, dim=0, return_inverse=True)
    groups = []
    for group_index in range(int(inverse.max()) + 1):
        members = (inverse == group_index).nonzero(as_tuple=True)[0].tolist()
        if members:
            groups.append(members)
    trajectories = points - descriptors_normalized
    stats = {
        "iterations": iterations,
        "max_movement": max_movement_value,
        "descriptor_kernel_min": initial_desc_min,
        "descriptor_kernel_max": initial_desc_max,
        "descriptor_kernel_mean": initial_desc_sum / max(initial_count, 1),
        "joint_kernel_min": initial_joint_min,
        "joint_kernel_max": initial_joint_max,
        "joint_kernel_mean": initial_joint_sum / max(initial_count, 1),
        "mean_abs_kernel_difference": (
            initial_abs_difference_sum / max(initial_count, 1)
        ),
        "max_abs_kernel_difference": initial_abs_difference_max,
        "relative_kernel_difference": (
            initial_abs_difference_sum
            / (initial_descriptor_abs_sum + NUMERICAL_EPS)
        ),
        "ratio_joint_lower_than_descriptor": (
            initial_joint_lower_count / max(initial_count, 1)
        ),
        "ratio_joint_less_than_half_descriptor": (
            initial_joint_less_half_count / max(initial_count, 1)
        ),
    }
    return groups, trajectories, points, stats


def groups_to_labels(groups: list, unit_count: int) -> torch.Tensor:
    labels = torch.full((unit_count,), -1, dtype=torch.long)
    for group_index, members in enumerate(groups):
        labels[members] = group_index
    if torch.any(labels < 0):
        raise AssertionError("at least one unit has no cluster label")
    return labels


def _contingency_table(
    first_labels: torch.Tensor, second_labels: torch.Tensor
) -> torch.Tensor:
    """Build cluster contingency counts for two label vectors Tensor[N]."""
    first = first_labels.detach().cpu().long()
    second = second_labels.detach().cpu().long()
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("cluster label vectors must have identical shape [N]")
    _, first_inverse = torch.unique(first, sorted=True, return_inverse=True)
    _, second_inverse = torch.unique(second, sorted=True, return_inverse=True)
    table = torch.zeros(
        int(first_inverse.max()) + 1,
        int(second_inverse.max()) + 1,
        dtype=torch.float64,
    )
    table.index_put_(
        (first_inverse, second_inverse),
        torch.ones_like(first_inverse, dtype=torch.float64),
        accumulate=True,
    )
    return table


def adjusted_rand_index(
    first_labels: torch.Tensor, second_labels: torch.Tensor
) -> float:
    """Compute Adjusted Rand Index without an sklearn dependency."""
    table = _contingency_table(first_labels, second_labels)
    sample_count = int(table.sum())
    if sample_count < 2:
        return 1.0

    def combination_two(values):
        return (values * (values - 1.0) * 0.5).sum()

    sum_cells = float(combination_two(table))
    sum_first = float(combination_two(table.sum(dim=1)))
    sum_second = float(combination_two(table.sum(dim=0)))
    total_pairs = sample_count * (sample_count - 1.0) * 0.5
    expected = sum_first * sum_second / max(total_pairs, NUMERICAL_EPS)
    maximum = 0.5 * (sum_first + sum_second)
    denominator = maximum - expected
    if abs(denominator) <= NUMERICAL_EPS:
        return 1.0 if abs(sum_cells - maximum) <= NUMERICAL_EPS else 0.0
    return float((sum_cells - expected) / denominator)


def normalized_mutual_information(
    first_labels: torch.Tensor, second_labels: torch.Tensor
) -> float:
    """Compute geometric-mean normalized mutual information."""
    table = _contingency_table(first_labels, second_labels)
    total = float(table.sum())
    joint = table / max(total, NUMERICAL_EPS)
    first_probability = joint.sum(dim=1)
    second_probability = joint.sum(dim=0)
    expected = first_probability[:, None] * second_probability[None, :]
    positive = joint > 0
    mutual_information = float(
        (joint[positive] * torch.log(joint[positive] / expected[positive])).sum()
    )
    first_entropy = float(
        -(first_probability[first_probability > 0]
          * torch.log(first_probability[first_probability > 0])).sum()
    )
    second_entropy = float(
        -(second_probability[second_probability > 0]
          * torch.log(second_probability[second_probability > 0])).sum()
    )
    denominator = math.sqrt(first_entropy * second_entropy)
    if denominator <= NUMERICAL_EPS:
        return 1.0 if first_entropy <= NUMERICAL_EPS and second_entropy <= NUMERICAL_EPS else 0.0
    return float(mutual_information / denominator)


def _hungarian_min_cost(cost: torch.Tensor) -> list:
    """Return exact row-to-column assignment for a square cost matrix."""
    square = cost.detach().cpu().double()
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("Hungarian cost matrix must be square")
    size = square.shape[0]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column_zero = 0
        min_values = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column_zero] = True
            row_zero = p[column_zero]
            delta = float("inf")
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = (
                    float(square[row_zero - 1, column - 1])
                    - u[row_zero]
                    - v[column]
                )
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column_zero
                if min_values[column] < delta:
                    delta = min_values[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column_zero = next_column
            if p[column_zero] == 0:
                break
        while True:
            next_column = way[column_zero]
            p[column_zero] = p[next_column]
            column_zero = next_column
            if column_zero == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column] > 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def hungarian_match_clusters(
    original_labels: torch.Tensor, function_labels: torch.Tensor
) -> Dict[int, int]:
    """Match each functional cluster to one original cluster by maximum overlap."""
    table = _contingency_table(original_labels, function_labels)
    original_count, function_count = table.shape
    size = max(original_count, function_count)
    rewards = torch.zeros((size, size), dtype=torch.float64)
    rewards[:function_count, :original_count] = table.transpose(0, 1)
    maximum = float(rewards.max()) if rewards.numel() else 0.0
    assignment = _hungarian_min_cost(maximum - rewards)
    return {
        function_id: (
            assignment[function_id]
            if assignment[function_id] < original_count
            else -1
        )
        for function_id in range(function_count)
    }


def compute_pairwise_disagreement(
    original_labels: torch.Tensor,
    function_labels: torch.Tensor,
    sample_count: int = DIAGNOSTIC_SAMPLE_COUNT,
    seed: int = DIAGNOSTIC_RANDOM_SEED,
) -> Dict[str, Any]:
    """Compare co-clustering decisions on deterministic unit pairs."""
    original = original_labels.detach().cpu().long()
    functional = function_labels.detach().cpu().long()
    units = original.numel()
    if units < 2:
        return {
            "sampled_pair_count": 0,
            "original_same_function_different": 0,
            "original_different_function_same": 0,
            "pairwise_disagreement_ratio": 0.0,
        }
    total_pairs = units * (units - 1) // 2
    if total_pairs <= sample_count:
        pairs = torch.triu_indices(units, units, offset=1)
        first, second = pairs[0], pairs[1]
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        first = torch.randint(0, units, (sample_count,), generator=generator)
        second = torch.randint(0, units - 1, (sample_count,), generator=generator)
        second += (second >= first).long()
    original_same = original[first] == original[second]
    function_same = functional[first] == functional[second]
    split_count = int((original_same & ~function_same).sum())
    merge_count = int((~original_same & function_same).sum())
    pair_count = first.numel()
    return {
        "sampled_pair_count": pair_count,
        "original_same_function_different": split_count,
        "original_different_function_same": merge_count,
        "pairwise_disagreement_ratio": (
            (split_count + merge_count) / max(pair_count, 1)
        ),
    }


def summarize_cluster_sizes(labels: torch.Tensor) -> Dict[str, Any]:
    """Summarize cluster-size distribution for labels Tensor[N]."""
    counts = torch.bincount(labels.detach().cpu().long()).float()
    quantiles = torch.quantile(
        counts, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    )
    sorted_counts = torch.sort(counts, descending=True).values
    total = float(sorted_counts.sum())
    return {
        "cluster_count": int(counts.numel()),
        "size_min": float(quantiles[0]),
        "size_q25": float(quantiles[1]),
        "size_median": float(quantiles[2]),
        "size_q75": float(quantiles[3]),
        "size_max": float(quantiles[4]),
        "maximum_cluster_ratio": float(sorted_counts[0] / total),
        "top5_coverage": float(sorted_counts[:5].sum() / total),
        "top10_coverage": float(sorted_counts[:10].sum() / total),
    }


def compute_cluster_function_separation(
    labels: torch.Tensor,
    function_similarity: torch.Tensor,
    row_chunk_size: int = 256,
) -> Dict[str, Any]:
    """Compute per-cluster intra/inter S_func separation without large copies."""
    labels_cpu = labels.detach().cpu().long()
    similarity_cpu = function_similarity.detach().cpu()
    units = labels_cpu.numel()
    per_cluster = []
    for cluster_id in range(int(labels_cpu.max()) + 1):
        members = (labels_cpu == cluster_id).nonzero(as_tuple=True)[0]
        member_count = members.numel()
        intra_sum = 0.0
        inter_sum = 0.0
        for start in range(0, member_count, row_chunk_size):
            row_indices = members[start:start + row_chunk_size]
            rows = similarity_cpu[row_indices].float()
            member_values = rows[:, members]
            intra_sum += float(member_values.sum()) - len(row_indices)
            inter_sum += float(rows.sum()) - float(member_values.sum())
        intra_count = member_count * max(member_count - 1, 0)
        inter_count = member_count * max(units - member_count, 0)
        mean_intra = intra_sum / max(intra_count, 1)
        mean_inter = inter_sum / max(inter_count, 1)
        if member_count <= 1:
            mean_intra = 1.0
        separation = mean_intra - mean_inter
        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "size": member_count,
                "mean_intra_S_func": mean_intra,
                "mean_inter_S_func": mean_inter,
                "function_separation": separation,
            }
        )
    separations = torch.tensor(
        [entry["function_separation"] for entry in per_cluster],
        dtype=torch.float32,
    )
    return {
        "per_cluster": per_cluster,
        "mean_separation": float(separations.mean()),
        "median_separation": float(separations.median()),
        "positive_separation_ratio": float((separations > 0).float().mean()),
    }


def compare_cluster_assignments(
    original_labels: torch.Tensor,
    function_labels: torch.Tensor,
    function_similarity: torch.Tensor,
) -> Dict[str, Any]:
    """Return label-invariant cluster differences and functional separation."""
    matching = hungarian_match_clusters(original_labels, function_labels)
    aligned = torch.tensor(
        [matching.get(int(label), -1) for label in function_labels.tolist()],
        dtype=torch.long,
    )
    original_cpu = original_labels.detach().cpu().long()
    original_separation = compute_cluster_function_separation(
        original_cpu, function_similarity
    )
    function_separation = compute_cluster_function_separation(
        function_labels, function_similarity
    )
    metrics = {
        "original_cluster_count": int(original_cpu.max()) + 1,
        "function_cluster_count": int(function_labels.max()) + 1,
        "ARI": adjusted_rand_index(original_cpu, function_labels),
        "NMI": normalized_mutual_information(original_cpu, function_labels),
        "aligned_label_change_ratio": float((aligned != original_cpu).float().mean()),
        "pairwise": compute_pairwise_disagreement(
            original_cpu, function_labels
        ),
        "original_cluster_sizes": summarize_cluster_sizes(original_cpu),
        "function_cluster_sizes": summarize_cluster_sizes(function_labels),
        "original_separation": original_separation,
        "function_separation": function_separation,
        "function_to_original_matching": matching,
    }
    return metrics

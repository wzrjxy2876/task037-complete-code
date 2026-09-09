"""Video-specific temporal dynamicity descriptor utilities.

The production helper in this module has no state, learned coefficient, or
trainable parameter.  It operates on signed pruning-unit responses and is
shared by Attention Heads and FFN neurons.
"""

from __future__ import annotations

from typing import Final

import torch


EPSILON: Final[float] = 1e-8
DESCRIPTOR_VARIANTS: Final[tuple[str, ...]] = (
    "abs_rel",
    "old3d",
    "dynamic3d",
)


def compute_temporal_dynamicity(
    response: torch.Tensor,
    eps: float = EPSILON,
    unit_chunk_size: int = 256,
) -> torch.Tensor:
    """Return per-video, per-unit temporal dynamicity.

    Args:
        response: Signed response ``A`` with logical shape ``[B,U,T,H,W,D]``.
            ``B`` is video batch, ``U`` is pruning unit, ``T`` is time,
            ``H/W`` are spatial axes, and ``D`` is the retained feature axis
            (Attention ``head_dim`` or one for an FFN neuron).
        eps: Numerical stability constant used only in the final ratio.
        unit_chunk_size: Memory implementation constant.  It does not change
            the descriptor definition and is intentionally not a CLI option.

    Returns:
        Tensor ``D_dyn`` with shape ``[B,U]`` and values in ``[0,1]``.
    """
    if not isinstance(response, torch.Tensor):
        raise TypeError("response must be a torch.Tensor")
    if response.ndim != 6:
        raise ValueError(
            "response must have logical shape [B,U,T,H,W,D], got "
            f"{tuple(response.shape)}"
        )
    if any(int(size) <= 0 for size in response.shape):
        raise ValueError(f"response axes must be non-empty: {tuple(response.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if unit_chunk_size <= 0:
        raise ValueError("unit_chunk_size must be positive")

    # Keep the signed response through temporal decomposition.  float32 avoids
    # half-precision cancellation in A - mean_t(A) on CUDA.
    signed = torch.nan_to_num(
        response.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    _, unit_count, _, _, _, _ = signed.shape
    score_chunks: list[torch.Tensor] = []

    for start in range(0, unit_count, unit_chunk_size):
        end = min(start + unit_chunk_size, unit_count)
        block = signed[:, start:end]  # [B,Q,T,H,W,D], Q <= unit_chunk_size
        temporal_mean = block.mean(dim=2, keepdim=True)  # [B,Q,1,H,W,D]
        residual = block - temporal_mean  # [B,Q,T,H,W,D]

        # L2 reduction happens only after the signed residual is formed.
        dynamic_mag = residual.norm(p=2, dim=-1).mean(dim=(2, 3, 4))  # [B,Q]
        # T=1 here is equivalent to broadcasting the stable response over time
        # and then averaging across T/H/W.
        stable_mag = temporal_mean.norm(p=2, dim=-1).mean(dim=(2, 3, 4))
        ratio = dynamic_mag / (dynamic_mag + stable_mag + eps)
        score_chunks.append(
            torch.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        )

    return torch.cat(score_chunks, dim=1)


def assemble_descriptor_variant(
    d_abs: torch.Tensor,
    d_rel: torch.Tensor,
    d_old: torch.Tensor | None,
    d_dyn: torch.Tensor | None,
    variant: str,
) -> torch.Tensor:
    """Assemble the controlled descriptor matrix ``V in R^[N,d]``.

    ``d=2`` for ``abs_rel`` and ``d=3`` for ``old3d``/``dynamic3d``.
    No scaling, learned weight, or BMS operation is performed here.
    """
    if variant not in DESCRIPTOR_VARIANTS:
        raise ValueError(
            f"descriptor variant must be one of {DESCRIPTOR_VARIANTS}, got {variant!r}"
        )
    if d_abs.ndim != 1 or d_rel.ndim != 1 or d_abs.shape != d_rel.shape:
        raise ValueError("D_abs and D_rel must be matching vectors [N]")
    if d_abs.numel() == 0:
        raise ValueError("descriptor vectors must be non-empty")

    columns = [d_abs, d_rel]
    if variant == "old3d":
        if d_old is None or d_old.shape != d_abs.shape:
            raise ValueError("old3d requires D_old with shape [N]")
        columns.append(d_old)
    elif variant == "dynamic3d":
        if d_dyn is None or d_dyn.shape != d_abs.shape:
            raise ValueError("dynamic3d requires D_dyn with shape [N]")
        columns.append(d_dyn)

    descriptor = torch.stack(columns, dim=1)
    if not torch.isfinite(descriptor).all():
        raise ValueError("descriptor contains NaN or infinity")
    return descriptor


def descriptor_variant_label(variant: str) -> str:
    """Return the stable user-facing name for one controlled variant."""
    labels = {
        "abs_rel": "AbsRel (2D ablation)",
        "old3d": "Old3D",
        "dynamic3d": "Dynamic3D",
    }
    try:
        return labels[variant]
    except KeyError as exc:
        raise ValueError(f"unknown descriptor variant {variant!r}") from exc

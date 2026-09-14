"""Differentiable Task042 Phase E.0 temporal-relation preservation primitives.

This module contains non-trainable structural output gates and the frozen
Task042 Pearson relation-distance analogue used by the feasibility audit.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


EPS_SENSITIVITY = 1e-12
EPS_NORMALIZATION = 1e-8
EPS_PEARSON = 1e-12


class PhaseE0Error(ValueError):
    pass


def _gate_value(value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise PhaseE0Error("structural gate must be in [0, 1]")
    return value


class GateState:
    """Plain Python gate state; no tensor or trainable model parameter."""

    def __init__(self, candidate_ids: Iterable[int]):
        self.candidate_ids = tuple(int(x) for x in candidate_ids)
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise PhaseE0Error("duplicate candidate gate identity")
        self.values: Dict[int, float] = {uid: 1.0 for uid in self.candidate_ids}

    def reset(self) -> None:
        for uid in self.candidate_ids:
            self.values[uid] = 1.0

    def set_one(self, uid: int, value: float) -> None:
        uid = int(uid)
        if uid not in self.values:
            raise PhaseE0Error("gate target is outside frozen candidate set")
        self.reset()
        self.values[uid] = _gate_value(value)

    @contextmanager
    def temporary(self, uid: int, value: float):
        self.set_one(uid, value)
        try:
            yield self.values
        finally:
            self.reset()


def scale_attention_probabilities(attention, gates: Mapping[int, float]):
    """Scale selected attention heads before A@V, equivalent to H'_i=g_i H_i."""
    if not gates:
        return attention
    if attention.ndim != 4:
        raise PhaseE0Error("attention probabilities must be [windows, heads, q, k]")
    factors = attention.new_ones((int(attention.shape[1]),))
    for head, value in gates.items():
        head = int(head)
        if not 0 <= head < int(attention.shape[1]):
            raise PhaseE0Error("attention head gate index is out of range")
        factors[head] = _gate_value(value)
    return attention * factors.view(1, -1, 1, 1)


def scale_ffn_activations(activation, gates: Mapping[int, float]):
    """Scale selected post-GELU/pre-fc2 neurons, H'_i=g_i H_i."""
    if not gates:
        return activation
    if activation.ndim < 2:
        raise PhaseE0Error("FFN activation must have a feature dimension")
    factors = activation.new_ones((int(activation.shape[-1]),))
    for neuron, value in gates.items():
        neuron = int(neuron)
        if not 0 <= neuron < int(activation.shape[-1]):
            raise PhaseE0Error("FFN neuron gate index is out of range")
        factors[neuron] = _gate_value(value)
    return activation * factors.view(*([1] * (activation.ndim - 1)), -1)


def relative_sensitivity(h_base, h_intervened, eps: float = EPS_SENSITIVITY):
    """Task042 relative activation sensitivity, kept differentiable for student."""
    if tuple(h_base.shape) != tuple(h_intervened.shape):
        raise PhaseE0Error("baseline and intervention activation shapes differ")
    return ((h_intervened - h_base).reshape(-1).norm(p=2) /
            (h_base.reshape(-1).norm(p=2) + float(eps)))


def differentiable_normalize(values, eps: float = EPS_NORMALIZATION):
    """Differentiate through within-unit frame-pair z normalization."""
    if values.ndim < 1 or int(values.shape[-1]) < 2:
        raise PhaseE0Error("a sensitivity signature needs at least two frame pairs")
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered / (variance + float(eps)).sqrt()


def stable_pearson(xs, ys, eps: float = EPS_PEARSON):
    """Pearson correlation with epsilon only in the norm product denominator."""
    if tuple(xs.shape) != tuple(ys.shape) or int(xs.shape[-1]) < 2:
        raise PhaseE0Error("Pearson inputs must have matching pair signatures")
    x = xs - xs.mean(dim=-1, keepdim=True)
    y = ys - ys.mean(dim=-1, keepdim=True)
    numerator = (x * y).sum(dim=-1)
    denominator = (x.square().sum(dim=-1) * y.square().sum(dim=-1) + float(eps)).sqrt()
    return (numerator / denominator).clamp(-1.0, 1.0)


def relation_distance(xs, ys, eps: float = EPS_PEARSON):
    """Frozen Task042 d=(1-rho)/2 on differentiably normalized signatures."""
    rho = stable_pearson(differentiable_normalize(xs),
                         differentiable_normalize(ys), eps=eps)
    return (1.0 - rho) * 0.5


def relation_loss(student_sensitivities, teacher_sensitivities,
                  domain_ids: Sequence[str], alive: Sequence[bool],
                  unit_ids: Optional[Sequence[int]] = None,
                  eps_normalization: float = EPS_NORMALIZATION,
                  eps_pearson: float = EPS_PEARSON):
    """Mean squared same-domain distance drift over videos and live unit pairs.

    Inputs have shape [videos, units, selected frame-pair interventions].
    Teacher sensitivities are treated as constants by the caller.
    """
    if tuple(student_sensitivities.shape) != tuple(teacher_sensitivities.shape):
        raise PhaseE0Error("teacher/student sensitivity tensors differ in shape")
    if student_sensitivities.ndim != 3:
        raise PhaseE0Error("sensitivities must have [videos, units, pairs] shape")
    units = int(student_sensitivities.shape[1])
    if len(domain_ids) != units or len(alive) != units:
        raise PhaseE0Error("domain/alive metadata do not match sensitivity units")
    if unit_ids is None:
        unit_ids = list(range(units))
    if len(unit_ids) != units:
        raise PhaseE0Error("unit IDs do not match sensitivity units")
    alive = tuple(bool(x) for x in alive)
    pair_indices = [(i, j) for i in range(units) for j in range(i + 1, units)
                    if alive[i] and alive[j] and str(domain_ids[i]) == str(domain_ids[j])]
    if not pair_indices:
        raise PhaseE0Error("no same-domain alive unit pairs remain")

    losses, details = [], []
    video_count = int(student_sensitivities.shape[0])
    for video in range(video_count):
        for i, j in pair_indices:
            d_student = relation_distance(student_sensitivities[video, i],
                                          student_sensitivities[video, j],
                                          eps=eps_pearson)
            d_teacher = relation_distance(teacher_sensitivities[video, i],
                                          teacher_sensitivities[video, j],
                                          eps=eps_pearson)
            losses.append((d_student - d_teacher).square())
            details.append({
                "video_position": video,
                "unit_i": int(unit_ids[i]), "unit_j": int(unit_ids[j]),
                "domain_id": str(domain_ids[i]),
                "d_student": d_student,
                "d_teacher": d_teacher,
                "squared_error": (d_student - d_teacher).square(),
                "absolute_drift": (d_student - d_teacher).abs(),
            })
    return sum(losses) / float(len(losses)), details

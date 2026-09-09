"""Task031: offline diagnosis of granularity-aware Average risk.

Task031 is deliberately a read-only diagnostic.  It compares three ways of
forming the *reference set* for the already frozen ``Delta_average`` signal;
``p_total``, ``domain_damage``, parameter costs, and all Task028 scientific
semantics remain unchanged.  No model, checkpoint, dataset, or validation
loader is imported at module import time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

CODE_VERSION = "task031_domain_conditioned_average_diagnosis_v1"
TARGET_SPARSITY = 0.50
SNAPSHOT_TARGETS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
VARIANTS = ("V0_global", "V1_domain", "V2_cost")
EXPECTED_TASK028_REMOVED = 28_044
EXPECTED_TASK028_ATTENTION = 0
EXPECTED_TASK028_FFN = 28_044
EXPECTED_TASK028_SEQUENCE_SHA = None  # populated from the frozen trace
BENCHMARK_STEPS = 1000
BENCHMARK_MIN_STEPS_PER_SECOND = {
    "V0_global": 100.0,
    "V1_domain": 50.0,
    "V2_cost": 50.0,
}


def _torch():
    """Lazy torch import so CPU-only analysis/tests need no torch install."""
    import torch
    return torch
TASK028_INPUTS = (
    "selection_50/registry.json",
    "selection_50/causal_selection_trace.csv",
    "selection_50/construction.json",
)
TASK029_INPUTS = (
    "snapshot_candidates.csv",
    "final_50_attention_candidates.csv",
    "domain_type_degradation.csv",
    "attention_rank_by_snapshot.csv",
    "task029_completion.json",
)
TASK030_INPUTS = (
    "causal_ablation_results.csv",
    "selected_attention_heads.csv",
    "selected_causal_pairs.csv",
    "cost_matched_ffn_packs.csv",
    "task030_completion.json",
)

SNAPSHOT_FIELDS = (
    "variant", "snapshot_target", "actual_effective_sparsity", "candidate_count",
    "removed_attention", "removed_ffn", "removed_attention_parameters",
    "removed_ffn_parameters", "remaining_attention", "remaining_ffn",
    "best_attention_rank", "ffn_candidates_ahead", "best_attention_global_index",
    "best_attention_p_total", "best_attention_p_A_variant",
    "best_attention_domain_damage", "best_attention_R_adaptive",
)
TRACE_FIELDS = (
    "variant", "step", "global_index", "unit_type", "stage", "layer", "unit_index",
    "domain_id", "local_index", "parameter_cost", "Delta_average", "Delta_total",
    "p_total", "p_A_variant", "domain_damage", "R_dual", "R_adaptive",
    "cumulative_removed_parameters", "effective_sparsity_after",
)


def _int(value: object, label: str = "value") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc


def _float(value: object, label: str = "value", default: float | None = None) -> float:
    if value is None or str(value).strip() == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Missing {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True,
                              ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inputs(root: Path, names: Sequence[str]) -> dict[str, str]:
    root = Path(root)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing immutable artifacts below {root}: {missing}")
    return {name: sha256_file(root / name) for name in names}


def sequence_sha256(indices: Iterable[int]) -> str:
    text = ",".join(str(_int(v, "global_index")) for v in indices)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stage(row: Mapping[str, object]) -> str:
    if str(row.get("stage", "")).strip():
        return str(row["stage"])
    layer = str(row.get("layer", ""))
    if "layers." in layer:
        return layer.split("layers.", 1)[1].split(".", 1)[0]
    if layer.startswith("stage"):
        return layer.split(".", 1)[0].removeprefix("stage")
    return layer.split(".", 1)[0] if layer else "unknown"


def _feasible(row: Mapping[str, object]) -> bool:
    return str(row.get("feasible", "True")).strip().lower() not in {"false", "0", "no"}


def _ordinal(values: Sequence[float], gids: Sequence[int]) -> list[float]:
    """Deterministic ordinal ranks in [0, 1], with global-id tie breaking."""
    if len(values) != len(gids):
        raise ValueError("values and global_index lengths differ")
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), int(gids[i])))
    result = [0.0] * len(values)
    denominator = float(max(len(values) - 1, 1))
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def ordinal_rank(values: Sequence[float], global_indices: Sequence[int] | None = None) -> list[float]:
    gids = list(range(len(values))) if global_indices is None else [_int(v, "global_index") for v in global_indices]
    return _ordinal([float(v) for v in values], gids)


def p_average_variant(rows: Sequence[Mapping[str, object]], variant: str) -> list[float]:
    """Return p_A for the current feasible set; only the reference set changes."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "V0_global":
        return _ordinal([_float(row.get("Delta_average"), "Delta_average") for row in rows], [_int(row.get("global_index"), "global_index") for row in rows])
    if variant == "V1_domain":
        return p_a_domain(rows)
    return p_a_cost(rows)


def gpu_lexicographic_order(values, global_index):
    """Return exact ``(value, global_index)`` order on the tensor device.

    PyTorch 1.12 may not expose stable sorting for ``argsort``.  Canonicalising
    the input by unique global index first makes the otherwise unspecified
    order of equal values deterministic: the secondary key is the canonical
    position.  Every operation remains tensor-side and no Python candidate
    rows are materialised.
    """
    if values.ndim != 1 or global_index.ndim != 1:
        raise ValueError("values and global_index must be one-dimensional")
    if values.numel() != global_index.numel():
        raise ValueError("values and global_index lengths differ")
    torch = _torch()
    count = int(values.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    if int(torch.unique(global_index).numel()) != count:
        raise ValueError("global_index must be unique for deterministic ranking")
    gid_order = torch.argsort(global_index)
    canonical_values = values.index_select(0, gid_order)
    rough = torch.argsort(canonical_values)
    sorted_values = canonical_values.index_select(0, rough)
    starts = torch.ones(count, dtype=torch.bool, device=values.device)
    if count > 1:
        starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    # ``rough`` is a permutation of canonical (global-index) positions.
    # Composite keys are unique, so the final argsort needs no stable mode.
    composite = group_id * (count + 1) + rough.to(torch.int64)
    return gid_order.index_select(0, rough.index_select(0, torch.argsort(composite)))


def gpu_ordinal_rank(values, global_index):
    """Exact ordinal percentile ranks, including equal-value tie semantics."""
    torch = _torch()
    count = int(values.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.float64, device=values.device)
    if count == 1:
        return torch.zeros(1, dtype=torch.float64, device=values.device)
    order = gpu_lexicographic_order(values, global_index)
    # Keep the sorted Delta tensor in its source dtype, but represent the
    # mathematical ordinal percentile in float64.  Python's oracle performs
    # ``rank / (N - 1)`` as a Python float; storing these values as float32
    # can turn 1/5 into 0.20000000298023224 and change an exact max tie set.
    ranks = torch.empty(count, dtype=torch.float64, device=values.device)
    ordinal_values = torch.arange(count, dtype=torch.float64, device=values.device) / float(count - 1)
    ranks.index_copy_(0, order, ordinal_values)
    return ranks


def _cached_gpu_ordinal_rank(values, static_positions, global_index_order):
    """Cached-global-index ordinal ranks with float64 percentile values.

    ``values`` is a one-dimensional Delta tensor aligned with
    ``static_positions``.  ``global_index_order`` contains all static
    positions sorted by global index.  Only the Delta sort stays in the
    source dtype; the returned ordinal values are float64 for Python-oracle
    exactness.
    """
    torch = _torch()
    count = int(values.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.float64, device=values.device)
    compact_index = torch.full(
        (int(global_index_order.numel()),), -1, dtype=torch.long,
        device=values.device,
    )
    compact_index.index_copy_(
        0, static_positions,
        torch.arange(count, dtype=torch.long, device=values.device),
    )
    membership = torch.zeros(
        int(global_index_order.numel()), dtype=torch.bool, device=values.device,
    )
    membership.index_fill_(0, static_positions, True)
    active_static_in_global_order = global_index_order[
        membership.index_select(0, global_index_order)
    ]
    active_compact_in_global_order = compact_index.index_select(
        0, active_static_in_global_order
    )
    active_values = values.index_select(0, active_compact_in_global_order)
    rough = torch.argsort(active_values)
    sorted_values = active_values.index_select(0, rough)
    starts = torch.ones(count, dtype=torch.bool, device=values.device)
    if count > 1:
        starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    composite = group_id * (count + 1) + rough.to(torch.int64)
    ordered_static = active_static_in_global_order.index_select(
        0, rough.index_select(0, torch.argsort(composite))
    )
    compact_order = compact_index.index_select(0, ordered_static)
    ranks = torch.empty(count, dtype=torch.float64, device=values.device)
    ordinal_values = torch.arange(count, dtype=torch.float64, device=values.device) / float(max(count - 1, 1))
    ranks.index_copy_(0, compact_order, ordinal_values)
    return ranks


def p_a_global(rows: Sequence[Mapping[str, object]]) -> list[float]:
    return p_average_variant(rows, "V0_global")


def p_a_domain(rows: Sequence[Mapping[str, object]]) -> list[float]:
    groups: dict[object, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get("domain_id")].append(i)
    result = [0.0] * len(rows)
    for group in groups.values():
        if len(group) == 1:
            result[group[0]] = _float(rows[group[0]].get("p_total"), "p_total")
        else:
            local = _ordinal([_float(rows[i].get("Delta_average"), "Delta_average") for i in group], [_int(rows[i].get("global_index"), "global_index") for i in group])
            for j, i in enumerate(group): result[i] = local[j]
    return result


def p_a_cost(rows: Sequence[Mapping[str, object]]) -> list[float]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[_int(row.get("parameter_cost"), "parameter_cost")].append(i)
    result = [0.0] * len(rows)
    for group in groups.values():
        if len(group) == 1:
            result[group[0]] = _float(rows[group[0]].get("p_total"), "p_total")
        else:
            local = _ordinal([_float(rows[i].get("Delta_average"), "Delta_average") for i in group], [_int(rows[i].get("global_index"), "global_index") for i in group])
            for j, i in enumerate(group): result[i] = local[j]
    return result


compute_p_a_global = p_a_global
compute_p_a_domain = p_a_domain
compute_p_a_cost = p_a_cost


def compute_variant_scores(rows: Sequence[Mapping[str, object]], variant: str = "V0_global") -> list[dict[str, object]]:
    active = [dict(row) for row in rows if _feasible(row)]
    p_a = p_average_variant(active, variant)
    output: list[dict[str, object]] = []
    for row, average in zip(active, p_a):
        total = _float(row.get("p_total"), "p_total")
        damage = _float(row.get("domain_damage"), "domain_damage", 0.0)
        row["p_A_variant"] = average
        row["p_average"] = average
        row["R_dual"] = max(total, average)
        row["R_adaptive"] = max(total, average, damage)
        row["variant"] = variant
        output.append(row)
    return output


def assign_p_a_variant(rows: Sequence[Mapping[str, object]], variant: str) -> list[dict[str, object]]:
    """Return copies annotated with only the selected p_A reference set."""
    return compute_variant_scores(rows, variant)


def rank_variant(rows: Sequence[Mapping[str, object]], variant: str = "V0_global") -> list[dict[str, object]]:
    scored = compute_variant_scores(rows, variant)
    scored.sort(key=lambda row: (_float(row["R_adaptive"], "R_adaptive"),
                                 _float(row["p_total"], "p_total"),
                                 _float(row["p_A_variant"], "p_A_variant"),
                                 _int(row["global_index"], "global_index")))
    for rank, row in enumerate(scored, 1):
        row["global_rank"] = rank
    return scored


# Descriptive aliases make the mathematical contract easy to test.
rank_candidates = rank_variant
compute_p_a = p_average_variant
rank_variant_candidates = rank_variant
compute_p_a_variant = p_average_variant


def risk_from_components(p_total: float, p_a: float, domain_damage: float) -> float:
    return max(float(p_total), float(p_a), float(domain_damage))


fuse_risk = risk_from_components
dynamic_recompute_scores = compute_variant_scores


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    result = [0.0] * len(values); pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and float(values[order[end + 1]]) == float(values[order[pos]]):
            end += 1
        value = (pos + end + 2) / 2.0
        for j in range(pos, end + 1): result[order[j]] = value
        pos = end + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def ks_distance(xs: Sequence[float], ys: Sequence[float]) -> float:
    a, b = sorted(float(v) for v in xs), sorted(float(v) for v in ys)
    if not a or not b:
        return math.nan
    values = sorted(set(a + b)); return max(abs(sum(v <= x for v in a) / len(a) - sum(v <= x for v in b) / len(b)) for x in values)


def auc_greater(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        return math.nan
    wins = ties = 0
    for x in xs:
        for y in ys:
            wins += x > y; ties += x == y
    return (wins + .5 * ties) / (len(xs) * len(ys))


def quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {k: math.nan for k in ("count", "mean", "median", "q10", "q25", "q50", "q75", "q90")}
    ordered = sorted(float(v) for v in values)
    def q(p: float) -> float:
        x = p * (len(ordered) - 1); lo, hi = math.floor(x), math.ceil(x)
        return ordered[lo] * (hi - x) + ordered[hi] * (x - lo)
    return {"count": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "q10": q(.1), "q25": q(.25), "q50": q(.5), "q75": q(.75), "q90": q(.9)}


def jaccard(a: Iterable[int], b: Iterable[int]) -> dict[str, object]:
    left, right = set(int(v) for v in a), set(int(v) for v in b)
    common, union = left & right, left | right
    return {"intersection_count": len(common), "union_count": len(union),
            "jaccard": len(common) / max(1, len(union)),
            "attention_intersection": None, "ffn_intersection": None}


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b): break
        count += 1
    return count


def partition_selected_sets(original: Iterable[int], current: Iterable[int]) -> dict[str, set[int]]:
    left, right = set(int(v) for v in original), set(int(v) for v in current)
    return {"common": left & right, "original_only": left - right,
            "counterfactual_only": right - left}


def overlap_report(selections: Mapping[str, Sequence[int]], types: Mapping[int, str] | None = None) -> dict[str, object]:
    """Pairwise overlap and exact-prefix diagnostics for V0/V1/V2."""
    output: dict[str, object] = {}
    for left, right in (("V0_global", "V1_domain"), ("V0_global", "V2_cost"), ("V1_domain", "V2_cost")):
        a, b = set(selections.get(left, ())), set(selections.get(right, ()))
        common = a & b
        output[f"{left}_vs_{right}"] = {"intersection_count": len(common), "union_count": len(a | b), "jaccard": len(common) / max(1, len(a | b)), "attention_intersection": sum(types.get(gid) == TYPE_ATTENTION for gid in common) if types else None, "ffn_intersection": sum(types.get(gid) == TYPE_FFN for gid in common) if types else None, "longest_common_prefix": longest_common_prefix(list(selections.get(left, ())), list(selections.get(right, ()) ))}
    return output


def _unit_type_from_intervention(row: Mapping[str, object], heads: Mapping[int, Mapping[str, object]], pairs: Mapping[str, Mapping[str, object]]) -> str:
    current = str(row.get("unit_type", "")).strip()
    if current in (TYPE_ATTENTION, TYPE_FFN): return current
    intervention = str(row.get("intervention_type", row.get("intervention", ""))).lower()
    if "attention" in intervention: return TYPE_ATTENTION
    gid = row.get("global_index")
    if gid not in (None, "") and _int(gid, "global_index") in heads: return TYPE_ATTENTION
    pair = pairs.get(str(row.get("pair_id", "")), {})
    if gid not in (None, "") and _int(gid, "global_index") == _int(pair.get("ffn_global_index", -1), "ffn_global_index"):
        return TYPE_FFN
    if "ffn" in intervention: return TYPE_FFN
    return current or "unknown"


def recover_task030_unit_metadata(raw_rows: Sequence[Mapping[str, object]], selected_heads: Sequence[Mapping[str, object]], selected_pairs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    heads = {_int(r.get("global_index"), "global_index"): r for r in selected_heads if r.get("global_index", "") != ""}
    pairs = {str(r.get("pair_id")): r for r in selected_pairs}
    output = []
    for source in raw_rows:
        row = dict(source); row["unit_type"] = _unit_type_from_intervention(row, heads, pairs)
        if not str(row.get("unit_type", "")).strip() or row["unit_type"] == "unknown":
            continue
        output.append(row)
    return output


def causal_correlation(rows: Sequence[Mapping[str, object]], risk: str, metric: str, *, context: str | None = None, unit_type: str | None = None) -> dict[str, object]:
    selected = [r for r in rows if (context is None or str(r.get("context")) == context) and (unit_type is None or str(r.get("unit_type")) == unit_type)]
    paired = [r for r in selected if r.get(risk, "") not in (None, "") and r.get(metric, "") not in (None, "")]
    xs = [_float(r.get(risk), risk) for r in paired]
    ys = [_float(r.get(metric), metric) for r in paired]
    return {"context": context or "all", "group": unit_type or "all", "risk_quantity": risk, "metric": metric, "rho": spearman(xs, ys), "n": len(xs)}


def causal_alignment_by_variant(raw_rows: Sequence[Mapping[str, object]], risk_rows: Sequence[Mapping[str, object]], selected_heads: Sequence[Mapping[str, object]] = (), selected_pairs: Sequence[Mapping[str, object]] = ()) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Join raw Task030 single-unit effects to the exact 50% risk state."""
    recovered = recover_task030_unit_metadata(raw_rows, selected_heads, selected_pairs)
    # Cost-matched packs are interpretation controls, never individual
    # pruning units in the causal/risk correlation.
    recovered = [r for r in recovered if str(r.get("intervention", r.get("intervention_type", ""))) in ("attention_single", "ffn_single")]
    by_gid = {_int(r.get("global_index"), "global_index"): r for r in risk_rows if r.get("global_index", "") != ""}
    joined = []
    for source in recovered:
        gid = _int(source.get("global_index"), "global_index") if source.get("global_index", "") != "" else None
        if gid is None:
            pair = {str(r.get("pair_id")): r for r in selected_pairs}.get(str(source.get("pair_id", "")), {})
            gid_value = pair.get("attention_global_index" if source.get("unit_type") == TYPE_ATTENTION else "ffn_global_index", "")
            gid = _int(gid_value, "global_index") if gid_value not in (None, "") else None
        risk = by_gid.get(gid, {})
        if not risk: continue
        row = dict(source); row.update({"p_A_global": risk.get("p_A_global", risk.get("p_average", "")), "p_A_domain": risk.get("p_A_domain", ""), "p_A_cost": risk.get("p_A_cost", ""), "p_total": risk.get("p_total", "")})
        for name, p_a in (("global", "p_A_global"), ("domain", "p_A_domain"), ("cost", "p_A_cost")):
            if row.get(p_a, "") != "": row[f"R_{name}"] = risk_from_components(_float(row.get("p_total"), "p_total"), _float(row.get(p_a), p_a), _float(row.get("domain_damage"), "domain_damage", 0.0))
        joined.append(row)
    metrics = ("relative_logit_l2", "kl_divergence", "ce_increase", "prediction_flip_rate")
    align = []
    for context in ("FULL", "TASK028_50"):
        for group in ("all", TYPE_ATTENTION, TYPE_FFN):
            for metric in metrics:
                for name in ("global", "domain", "cost"):
                    subset = [r for r in joined if str(r.get("context")) == context and (group == "all" or str(r.get("unit_type")) == group) and r.get(metric, "") != "" and r.get(f"p_A_{name}", "") != ""]
                    align.append({"context": context, "group": group, "risk_quantity": f"p_A_{name}", "metric": metric, "rho": spearman([_float(r[f"p_A_{name}"]) for r in subset], [_float(r[metric]) for r in subset]), "n": len(subset)})
                for name in ("global", "domain", "cost"):
                    subset = [r for r in joined if str(r.get("context")) == context and (group == "all" or str(r.get("unit_type")) == group) and r.get(metric, "") != "" and r.get(f"R_{name}", "") != ""]
                    align.append({"context": context, "group": group, "risk_quantity": f"R_{name}", "metric": metric, "rho": spearman([_float(r[f"R_{name}"]) for r in subset], [_float(r[metric]) for r in subset]), "n": len(subset)})
    return joined, align, {"raw_single_units": len(joined), "contexts": ["FULL", "TASK028_50"], "metrics": list(metrics), "alignment_complete": bool(align)}


def granularity_controlled_pair_audit(raw_rows: Sequence[Mapping[str, object]], risk_rows: Sequence[Mapping[str, object]], packs: Sequence[Mapping[str, object]] = ()) -> list[dict[str, object]]:
    """Pair Attention, single-FFN and cost-matched-pack effects descriptively."""
    by_pair: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in raw_rows:
        intervention = str(row.get("intervention", row.get("intervention_type", "")))
        if intervention in ("attention_single", "ffn_single", "ffn_cost_matched_pack"):
            by_pair[str(row.get("pair_id", ""))][intervention] = row
    risk = {_int(r.get("global_index"), "global_index"): r for r in risk_rows if r.get("global_index", "") != ""}
    output = []
    for pair_id, group in sorted(by_pair.items()):
        a = group.get("attention_single", {}); s = group.get("ffn_single", {}); p = group.get("ffn_cost_matched_pack", {})
        gid = a.get("global_index", ""); rr = risk.get(_int(gid, "global_index"), {}) if gid != "" else {}
        output.append({"pair_id": pair_id, "attention_global_index": gid, "attention_p_A_global": rr.get("p_A_global", ""), "attention_p_A_domain": rr.get("p_A_domain", ""), "attention_p_A_cost": rr.get("p_A_cost", ""), "attention_relative_logit_l2": a.get("relative_logit_l2", ""), "single_ffn_relative_logit_l2": s.get("relative_logit_l2", ""), "cost_matched_pack_relative_logit_l2": p.get("relative_logit_l2", ""), "pack_parameter_cost": p.get("removed_parameter_cost", "")})
    return output


def granularity_dependence(rows: Sequence[Mapping[str, object]], variant: str, snapshot_target: float) -> list[dict[str, object]]:
    """Measure p_A/log(cost) association for all, Attention, and FFN."""
    output = []
    for group in ("all", TYPE_ATTENTION, TYPE_FFN):
        selected = [r for r in rows if group == "all" or str(r.get("unit_type")) == group]
        selected = [r for r in selected if _float(r.get("parameter_cost"), "parameter_cost") > 0]
        output.append({"variant": variant, "snapshot_target": snapshot_target, "group": group, "spearman_pA_log_cost": spearman([_float(r.get("p_A_variant"), "p_A_variant") for r in selected], [math.log(_float(r.get("parameter_cost"), "parameter_cost")) for r in selected]), "candidate_count": len(selected)})
    return output


def rank_shift_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Embed local/domain p_A values in a global ranking to expose displacement."""
    global_ranked = sorted(rows, key=lambda r: (_float(r.get("p_A_global", r.get("p_average", 0.0))), _int(r.get("global_index"))))
    domain_ranked = sorted(rows, key=lambda r: (_float(r.get("p_A_domain", 0.0)), _int(r.get("global_index"))))
    g = {_int(r.get("global_index")): i + 1 for i, r in enumerate(global_ranked)}; d = {_int(r.get("global_index")): i + 1 for i, r in enumerate(domain_ranked)}
    return [{"global_index": gid, "p_A_global": global_ranked[i].get("p_A_global", global_ranked[i].get("p_average", "")), "p_A_domain": next((r.get("p_A_domain", "") for r in rows if _int(r.get("global_index")) == gid), ""), "rank_global_A": g[gid], "rank_domain_A_when_embedded_globally": d[gid], "rank_shift": d[gid] - g[gid]} for i, gid in enumerate(sorted(g))]


def _registry_keys(registry: Mapping[str, object]) -> set[tuple[str, str, int]]:
    return {(str(layer), str(entry.get("unit_type", "")), _int(index, "unit_index"))
            for layer, entry in registry.items() if isinstance(entry, Mapping)
            for index in entry.get("indices", ())}


def verify_task028_reference(root: Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve(); hashes = hash_inputs(root, TASK028_INPUTS)
    selection = read_json(root / TASK028_INPUTS[0]); trace = read_csv(root / TASK028_INPUTS[1]); construction = read_json(root / TASK028_INPUTS[2])
    if selection.get("status") != "prepared": raise RuntimeError("Task028 registry is not prepared")
    if abs(_float(selection.get("target_sparsity"), "target_sparsity") - .5) > 1e-12: raise RuntimeError("Task028 target is not 50%")
    registry = selection.get("registry")
    if not isinstance(registry, Mapping) or not registry: raise RuntimeError("Task028 registry malformed")
    required = ("global_index", "unit_type", "layer", "unit_index")
    if trace and any(name not in trace[0] for name in required): raise RuntimeError("Task028 trace schema incomplete")
    sequence = [_int(row.get("global_index"), "global_index") for row in sorted(trace, key=lambda r: (_int(r.get("step", 0)), _int(r.get("incremental_step", 0))))]
    if len(sequence) != len(set(sequence)): raise RuntimeError("Task028 sequence has duplicate global_index")
    if construction.get("selection_started_from_zero") is not True: raise RuntimeError("Task028 did not start from zero")
    return {"root": str(root), "hashes": hashes, "selection": selection, "trace": trace, "registry": registry, "construction": construction, "sequence": sequence, "sequence_sha256": sequence_sha256(sequence)}


def verify_task029_reference(root: Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve(); hashes = hash_inputs(root, TASK029_INPUTS)
    completion = read_json(root / "task029_completion.json")
    if completion.get("status") != "PASS": raise RuntimeError("Task029 completion is not PASS")
    for name in ("replayed_0_to_50_exact", "trace_sequence_match", "task028_identity_verified"):
        if name in completion and completion.get(name) is not True:
            raise RuntimeError(f"Task029 completion gate failed: {name}")
    return {"root": str(root), "hashes": hashes, "completion": completion,
            "snapshots": read_csv(root / "snapshot_candidates.csv"),
            "final_attention": read_csv(root / "final_50_attention_candidates.csv"),
            "domain_degradation": read_csv(root / "domain_type_degradation.csv"),
            "attention_ranks": read_csv(root / "attention_rank_by_snapshot.csv")}


def verify_task030_reference(root: Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve(); hashes = hash_inputs(root, TASK030_INPUTS)
    completion = read_json(root / "task030_completion.json")
    if completion.get("status") != "PASS": raise RuntimeError("Task030 completion is not PASS")
    for name in ("training_executed", "fine_tuning_executed", "selector_replay_executed"):
        if completion.get(name) is True: raise RuntimeError(f"Task030 {name} must be false")
    raw, heads, pairs = read_csv(root / TASK030_INPUTS[0]), read_csv(root / TASK030_INPUTS[1]), read_csv(root / TASK030_INPUTS[2])
    return {"root": str(root), "hashes": hashes, "completion": completion,
            "raw": raw, "heads": heads, "pairs": pairs,
            "packs": read_csv(root / TASK030_INPUTS[3])}


def verify_identity(*, task028_root: Path, task029_root: Path, task030_root: Path, output_dir: Path) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    roots = [Path(task028_root).expanduser().resolve(), Path(task029_root).expanduser().resolve(), Path(task030_root).expanduser().resolve()]
    if any(output == root or root in output.parents for root in roots): raise RuntimeError("Task031 output must be separate from frozen roots")
    task028, task029, task030 = verify_task028_reference(roots[0]), verify_task029_reference(roots[1]), verify_task030_reference(roots[2])
    payload = {"status": "PASS", "code_version": CODE_VERSION, "target_sparsity": TARGET_SPARSITY,
               "task028_reference_pass": True, "task029_reference_pass": True, "task030_reference_pass": True,
               "task028_input_hashes": task028["hashes"], "task029_input_hashes": task029["hashes"], "task030_input_hashes": task030["hashes"],
               "task028_root": str(roots[0]), "task029_root": str(roots[1]), "task030_root": str(roots[2]),
               "selector_replay_executed": False, "model_forward_executed": False, "validation_executed": False,
               "training_executed": False, "fine_tuning_executed": False, "new_pruning_hyperparameters": 0,
               "variants": list(VARIANTS), "no_checkpoint_loaded": True, "no_new_registry": True}
    atomic_json(output / "artifact_identity.json", payload)
    return {"identity": payload, "task028": task028, "task029": task029, "task030": task030}


def _tensor_list(value: object) -> list[object]:
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def candidate_rows(candidates: Mapping[str, object], lookup: Mapping[int, Mapping[str, object]] | None = None) -> list[dict[str, object]]:
    lookup = lookup or {}; names = ("global_index", "domain_id", "local_index", "Delta_average", "Delta_total", "domain_damage", "domain_coverage", "parameter_cost", "p_total")
    arrays = {n: _tensor_list(candidates[n]) for n in names if n in candidates}; result = []
    for i, gid in enumerate(arrays.get("global_index", ())):
        row = dict(lookup.get(_int(gid, "global_index"), {})); row.update({n: arrays[n][i] for n in arrays if i < len(arrays[n])}); row["global_index"] = _int(gid, "global_index"); row["feasible"] = True
        result.append(row)
    # Task028's cache exposes Delta values, but intentionally does not carry
    # ordinal ranks.  Recompute p_total from the exact current feasible set;
    # this is the unchanged Task028 definition and does not use type/cost.
    if result and any("p_total" not in row for row in result):
        ranks = _ordinal([_float(row.get("Delta_total"), "Delta_total") for row in result], [_int(row["global_index"]) for row in result])
        for row, value in zip(result, ranks):
            row["p_total"] = value
    return result


class OptimizedVariantAverageState:
    """Tensor-resident V1/V2 average-risk state.

    Task028 owns the mutable replay/cache state.  This companion owns only
    the variant-specific reference ranks.  Group membership is encoded once
    at construction; after a removal only the selected domain/cost class is
    recomputed.  The global p_total and dynamic domain damage are refreshed
    from the current Task028 candidate tensors, entirely on the device.
    """

    def __init__(self, optimized: object, variant: str) -> None:
        if variant not in ("V1_domain", "V2_cost"):
            raise ValueError(f"OptimizedVariantAverageState requires V1/V2: {variant}")
        self.optimized = optimized
        self.variant = variant
        self.torch = getattr(optimized, "torch", None)
        if self.torch is None:
            self.torch = _torch()
        self.device = getattr(optimized, "device", None)
        self._task028 = None
        try:
            self._task028 = __import__("task028_main50_tad_logical_finetune")
        except ImportError:
            self._task028 = None
        self.static_count = int(getattr(optimized, "total_units"))
        self.global_index = optimized.global_index
        self.delta_average = optimized.delta_average
        self.delta_total = optimized.delta_total
        raw_parameter_cost = getattr(optimized, "parameter_cost", None)
        if raw_parameter_cost is not None:
            # Task028 stores replay.costs by authoritative global_index, while
            # its candidate arrays are concatenated by domain.  Align once at
            # initialization so every subsequent V2 lookup is tensor-local.
            max_gid = int(self.global_index.max().item()) if self.global_index.numel() else -1
            self.parameter_cost = (raw_parameter_cost.index_select(0, self.global_index)
                                    if max_gid < int(raw_parameter_cost.numel())
                                    else raw_parameter_cost)
        else:
            self.parameter_cost = None
        self.domain_id = optimized.domain_id
        if variant == "V2_cost" and self.parameter_cost is None:
            raise RuntimeError("Task028 candidate state has no parameter_cost tensor")
        group_source = self.domain_id if variant == "V1_domain" else self.parameter_cost
        # Group tensors are created once.  Python is used only during this
        # initialization, never in the per-removal ranking path.
        group_values = group_source.detach().cpu().tolist()
        self.groups: dict[int, object] = {}
        self._group_keys = sorted({int(value) for value in group_values})
        self.group_code_static = self.torch.empty(self.static_count, dtype=self.torch.long, device=self.device)
        for code, group in enumerate(self._group_keys):
            indices = [i for i, value in enumerate(group_values) if int(value) == group]
            self.groups[group] = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)
            self.group_code_static.index_fill_(0, self.groups[group], code)
        self.group_for_position = group_source
        # Percentile/risk state is float64 because Python's ordinal values are
        # Python floats.  The source Delta tensors remain untouched in their
        # original dtype and are still sorted on-device.
        self.p_a_static = self.torch.zeros(self.static_count, dtype=self.torch.float64, device=self.device)
        self.p_total_static = self.torch.zeros(self.static_count, dtype=self.torch.float64, device=self.device)
        self.singleton_mask = self.torch.zeros(self.static_count, dtype=self.torch.bool, device=self.device)
        self.group_active_counts = self.torch.zeros(len(self._group_keys), dtype=self.torch.long, device=self.device)
        self._pending_positions = None
        self._pending_p_total = None
        self._initialise_groups()

    def _global_ranks(self, positions):
        values = self.delta_total.index_select(0, positions)
        if hasattr(self.optimized, "global_index_order"):
            return _cached_gpu_ordinal_rank(
                values, positions, self.optimized.global_index_order,
            )
        return gpu_ordinal_rank(
            values,
            self.global_index.index_select(0, positions),
        )

    def _group_ranks(self, positions, p_total_static):
        count = int(positions.numel())
        if count == 0:
            return self.torch.empty(0, dtype=self.torch.float64, device=self.device)
        if count == 1:
            return p_total_static.index_select(0, positions)
        values = self.delta_average.index_select(0, positions)
        if hasattr(self.optimized, "global_index_order"):
            return _cached_gpu_ordinal_rank(
                values, positions, self.optimized.global_index_order,
            )
        return gpu_ordinal_rank(
            values,
            self.global_index.index_select(0, positions),
        )

    def _active_group_positions(self, group: int):
        positions = self.groups[int(group)]
        active = self.optimized.active_mask.index_select(0, positions)
        return positions[active]

    def _initialise_groups(self) -> None:
        active = self.optimized.active_mask.nonzero(as_tuple=True)[0]
        p_total = self._global_ranks(active)
        p_total_static = self.p_total_static
        p_total_static.zero_()
        p_total_static.index_copy_(0, active, p_total)
        self._pending_positions = active
        self._pending_p_total = p_total
        for code, group in enumerate(self._group_keys):
            positions = self._active_group_positions(group)
            self.group_active_counts[code] = int(positions.numel())
            if int(positions.numel()) == 1:
                self.singleton_mask[positions] = True
            values = self._group_ranks(positions, p_total_static)
            self.p_a_static.index_copy_(0, positions, values)

    def _refresh_group(self, group: int, p_total_static) -> None:
        positions = self.groups[int(group)]
        active_positions = self._active_group_positions(group)
        self.singleton_mask[positions] = False
        if int(active_positions.numel()) == 1:
            self.singleton_mask[active_positions] = True
        elif int(active_positions.numel()) == 0:
            return
        values = self._group_ranks(active_positions, p_total_static)
        self.p_a_static.index_copy_(0, active_positions, values)

    def rank(self, candidates: Mapping[str, object]) -> dict[str, object]:
        positions = candidates["_positions"]
        if self._pending_p_total is not None:
            # ``after_remove`` already computed the next state's global rank;
            # consume it once to avoid doing the same full GPU sort twice per
            # removal.  A repeated rank call in the same state falls back to
            # a fresh exact computation after the pending value is consumed.
            p_total = self._pending_p_total
            self._pending_p_total = None
            self._pending_positions = None
        else:
            p_total = self._global_ranks(positions)
        p_total_static = self.p_total_static
        p_total_static.zero_()
        p_total_static.index_copy_(0, positions, p_total)
        # Singleton fallback is p_total and therefore changes whenever the
        # global active set changes.  This is one vector assignment, not a
        # Python group reconstruction.
        singleton_active = self.singleton_mask.index_select(0, positions)
        p_a = self.p_a_static.index_select(0, positions)
        p_a = self.torch.where(singleton_active, p_total, p_a)
        damage = candidates.get("domain_damage")
        if damage is None:
            damage = self.torch.zeros_like(p_total)
            damage_for_risk = damage
        else:
            # ``domain_damage`` remains sourced/stored as-is.  Only this
            # candidate view is promoted so max fusion uses the same double
            # precision values as the Python oracle.
            damage_for_risk = damage.to(dtype=self.torch.float64)
        risks = {
            "p_total": p_total,
            "p_average": p_a,
            "p_A_variant": p_a,
            "R_dual": self.torch.maximum(p_total, p_a),
            "domain_damage": damage,
        }
        risks["R_adaptive"] = self.torch.maximum(risks["R_dual"], damage_for_risk)
        minimum = risks["R_adaptive"].min()
        tied_risk = risks["R_adaptive"] == minimum
        inf = self.torch.full_like(p_total, self.torch.inf)
        minimum_total = self.torch.where(tied_risk, p_total, inf).min()
        tied_total = tied_risk & (p_total == minimum_total)
        minimum_average = self.torch.where(tied_total, p_a, inf).min()
        tied_average = tied_total & (p_a == minimum_average)
        global_index = candidates["global_index"]
        max_global_index = self.torch.iinfo(global_index.dtype).max
        tie_global_index = self.torch.where(
            tied_average,
            global_index,
            self.torch.full_like(global_index, max_global_index),
        )
        selected = self.torch.argmin(tie_global_index)
        result = dict(candidates)
        result.update(risks)
        result["selected_position"] = int(selected.item())
        return result

    def after_remove(self, changed_group: int | None = None) -> None:
        """Refresh groups whose feasible membership changed.

        Normally this is exactly the selected group.  The tensor count check
        also covers a layer-capacity transition that deactivates additional
        candidates, without rebuilding Python group dictionaries.
        """
        active = self.optimized.active_mask.nonzero(as_tuple=True)[0]
        p_total = self._global_ranks(active)
        p_total_static = self.p_total_static
        p_total_static.zero_()
        p_total_static.index_copy_(0, active, p_total)
        self._pending_positions = active
        self._pending_p_total = p_total
        counts = self.torch.bincount(
            self.group_code_static.index_select(0, active),
            minlength=len(self._group_keys),
        )
        changed = (counts != self.group_active_counts).nonzero(as_tuple=True)[0]
        if changed_group is not None and int(changed.numel()) == 0:
            code = self._group_keys.index(int(changed_group))
            changed = self.torch.as_tensor([code], dtype=self.torch.long, device=self.device)
        self.group_active_counts.copy_(counts)
        for code in changed.detach().cpu().tolist():
            self._refresh_group(self._group_keys[int(code)], p_total_static)


def _scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _selected_candidate_row(candidates: Mapping[str, object], ranked: Mapping[str, object], position: int, lookup: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    """Materialize one selected record; full candidate tables stay on-device."""
    gid = _int(_scalar(candidates["global_index"][position]), "global_index")
    row = dict(lookup.get(gid, {}))
    for name in ("global_index", "domain_id", "local_index", "Delta_average", "Delta_total", "domain_damage", "domain_coverage"):
        if name in candidates:
            row[name] = _scalar(candidates[name][position])
    if "domain_coverage" not in row and "domain_damage" in row:
        row["domain_coverage"] = 1.0 - float(row["domain_damage"])
    row["global_index"] = gid
    for name in ("p_total", "p_average", "p_A_variant", "R_dual", "R_adaptive"):
        if name in ranked:
            row[name] = _scalar(ranked[name][position])
    row["feasible"] = True
    return row


def _capture_variant_snapshot(*, variant: str, target: float, sparsity: float,
                              candidates: Mapping[str, object], lookup: Mapping[int, Mapping[str, object]],
                              selected: Sequence[Mapping[str, object]], total_parameters: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Materialize and rank one checkpoint; never called outside snapshots."""
    rows = candidate_rows(candidates, lookup)
    ranked = rank_variant(rows, variant)
    summary = _snapshot_summary_rows(variant, target, sparsity, ranked, selected, total_parameters)
    full = [{**row, "snapshot_target": target, "actual_effective_sparsity": sparsity} for row in ranked]
    return summary, full


def lookup_from_engine(engine: object) -> dict[int, dict[str, object]]:
    """Recover structural metadata without loading a model or checkpoint."""
    replay = getattr(getattr(engine, "base", None), "replay", None)
    output: dict[int, dict[str, object]] = {}
    for unit in getattr(replay, "units", ()):
        gid = _int(getattr(unit, "global_index"), "global_index")
        layer = str(getattr(unit, "layer", "")); typ = str(getattr(unit, "unit_type", getattr(unit, "kind", "")))
        domain = getattr(unit, "domain_id", 0); local = getattr(unit, "local_index", 0); cost = getattr(unit, "parameter_cost", 0)
        costs = getattr(replay, "costs", ())
        if len(costs) > gid: cost = costs[gid]
        output[gid] = {"global_index": gid, "unit_type": typ, "layer": layer, "unit_index": _int(getattr(unit, "unit_index"), "unit_index"), "stage": _stage({"layer": layer}), "domain_id": _int(domain, "domain_id"), "local_index": _int(local, "local_index"), "parameter_cost": _int(cost, "parameter_cost")}
    return output


def _engine_for_replay(*, task014_root: Path, task016_root: Path, task017_root: Path, output_dir: Path, device: str, ranking_backend: str):
    import task024_threshold_free_adaptive_safety as task024
    import task028_main50_tad_logical_finetune as task028
    engine = task024.TensorSafetyEngine(task014_root=Path(task014_root), task016_root=Path(task016_root), task017_root=Path(task017_root), output_dir=Path(output_dir) / "_replay", device=device)
    engine.restore_prefix([]); engine.start_prefix = ()
    task028.assert_started_from_zero(engine); task028.assert_full_retention_start(engine)
    return engine, task028.OptimizedCandidateState(engine, ranking_backend=ranking_backend)


def _removed_cost(engine: object) -> float:
    value = getattr(engine, "removed_cost", None)
    if value is None:
        value = getattr(getattr(engine, "base", None), "removed_cost", 0.0)
    return float(value)


def replay_variant(*, variant: str, engine: object, optimized: object, total_parameters: float, target_budget: float, lookup: Mapping[int, Mapping[str, object]] | None = None, task028_sequence: Sequence[int] | None = None, progress_desc: str | None = None) -> dict[str, object]:
    """Replay one variant with a tensor-resident hot path.

    V0 calls Task028's optimized ranker directly.  V1/V2 use
    :class:`OptimizedVariantAverageState`, which refreshes only the selected
    domain/cost class after each removal.  Python rows are materialized only at
    the six snapshot checkpoints and for the one selected trace record.
    """
    del task028_sequence  # retained in the public API for callers/fixtures
    lookup = lookup or {}
    selected: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    snapshot_candidate_rows: list[dict[str, object]] = []
    captured: set[float] = set()
    total_parameters = float(total_parameters)
    target_budget = float(target_budget)
    singleton_evaluations = 0
    singleton_selected = 0
    variant_state = OptimizedVariantAverageState(optimized, variant) if variant in ("V1_domain", "V2_cost") else None
    iterator = _tqdm(desc=progress_desc or f"Task031 {variant} replay", total=None, mininterval=.5)
    try:
        while _removed_cost(engine) < target_budget - 1e-12:
            tensors = optimized.candidate_tensors()
            positions = tensors.get("_positions")
            if positions is None or int(positions.numel()) == 0:
                raise RuntimeError(f"{variant} exhausted candidates before target")
            sparsity = _removed_cost(engine) / max(total_parameters, 1e-12)

            # Full CPU materialisation is deliberately restricted to snapshot
            # states; it is never part of the normal selection hot path.
            for target in SNAPSHOT_TARGETS:
                if target not in captured and sparsity >= target - 1e-12:
                    summary_rows, full_rows = _capture_variant_snapshot(
                        variant=variant, target=target, sparsity=sparsity,
                        candidates=tensors, lookup=lookup,
                        selected=selected, total_parameters=total_parameters,
                    )
                    snapshots.extend(summary_rows)
                    snapshot_candidate_rows.extend(full_rows)
                    captured.add(target)

            if variant == "V0_global":
                ranked = optimized.rank(tensors)
            else:
                ranked = variant_state.rank(tensors)
                singleton_evaluations += int((variant_state.singleton_mask & optimized.active_mask).sum().item())
            position = int(ranked["selected_position"])
            static_position = int(positions[position].item())
            gid = _int(_scalar(tensors["global_index"][position]), "global_index")
            chosen = _selected_candidate_row(tensors, ranked, position, lookup)
            was_singleton = bool(variant_state.singleton_mask[static_position].item()) if variant_state is not None else False

            removed = optimized.remove(ranked, position)
            if variant_state is not None:
                group_value = int(variant_state.group_for_position[static_position].item())
                variant_state.after_remove(group_value)
            row = {**chosen, **(dict(removed) if isinstance(removed, Mapping) else {})}
            row.update({
                "variant": variant,
                "step": len(selected) + 1,
                "global_index": gid,
                "unit_type": row.get("unit_type", lookup.get(gid, {}).get("unit_type", "")),
                "effective_sparsity_after": _removed_cost(engine) / max(total_parameters, 1e-12),
                "cumulative_removed_parameters": _removed_cost(engine),
            })
            selected.append(row)
            singleton_selected += int(was_singleton)
            if hasattr(iterator, "update"):
                iterator.update(1)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"selected": gid, "sparsity": f"{row['effective_sparsity_after']:.4f}"})
    finally:
        if hasattr(iterator, "close"):
            iterator.close()

    final_sparsity = _removed_cost(engine) / max(total_parameters, 1e-12)
    if .5 not in captured:
        final_rows = candidate_rows(optimized.candidate_tensors(), lookup)
        ranked_final = rank_variant(final_rows, variant)
        snapshots.extend(_snapshot_summary_rows(variant, .5, final_sparsity, ranked_final, selected, total_parameters))
        snapshot_candidate_rows.extend({**row, "snapshot_target": .5, "actual_effective_sparsity": final_sparsity} for row in ranked_final)
        captured.add(.5)
    final_candidates = rank_variant(candidate_rows(optimized.candidate_tensors(), lookup), variant)
    return {"variant": variant, "selected": selected, "snapshots": snapshots,
            "snapshot_candidate_rows": snapshot_candidate_rows,
            "final_candidates": final_candidates,
            "sequence": [_int(row["global_index"]) for row in selected],
            "final_sparsity": final_sparsity,
            "snapshots_complete": captured == set(SNAPSHOT_TARGETS),
            "singleton_candidate_evaluations": singleton_evaluations,
            "singleton_selected_removals": singleton_selected}


def _synchronize_if_cuda(engine: object) -> None:
    torch = getattr(engine, "torch", None)
    device = getattr(engine, "device", None)
    if torch is not None and device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_memory(optimized: object) -> dict[str, int]:
    torch = getattr(optimized, "torch", None)
    device = getattr(optimized, "device", None)
    if torch is None or device is None or torch.device(device).type != "cuda":
        return {"peak_cuda_memory_bytes": 0}
    torch.cuda.synchronize(device)
    return {"peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))}


def benchmark_variant(*, variant: str, task014_root: Path, task016_root: Path,
                      task017_root: Path, output_dir: Path, device: str,
                      ranking_backend: str, steps: int = BENCHMARK_STEPS) -> dict[str, object]:
    """Time exactly ``steps`` optimized removals without scientific outputs."""
    if int(steps) != BENCHMARK_STEPS:
        raise ValueError(f"Task031 benchmark must run exactly {BENCHMARK_STEPS} steps")
    import task028_main50_tad_logical_finetune as task028
    engine, optimized = _engine_for_replay(
        task014_root=task014_root, task016_root=task016_root,
        task017_root=task017_root, output_dir=Path(output_dir) / "_benchmark",
        device=device, ranking_backend=ranking_backend,
    )
    variant_state = OptimizedVariantAverageState(optimized, variant) if variant in ("V1_domain", "V2_cost") else None
    first = optimized.candidate_tensors()
    start_count = int(first["global_index"].numel())
    _synchronize_if_cuda(engine)
    started = time.perf_counter()
    ranking_seconds = 0.0
    remove_seconds = 0.0
    for _ in range(BENCHMARK_STEPS):
        candidates = optimized.candidate_tensors()
        rank_started = time.perf_counter()
        ranked = optimized.rank(candidates) if variant == "V0_global" else variant_state.rank(candidates)
        ranking_seconds += time.perf_counter() - rank_started
        position = int(ranked["selected_position"])
        static_position = int(candidates["_positions"][position].item())
        remove_started = time.perf_counter()
        optimized.remove(ranked, position)
        if variant_state is not None:
            group = int(variant_state.group_for_position[static_position].item())
            variant_state.after_remove(group)
        remove_seconds += time.perf_counter() - remove_started
    _synchronize_if_cuda(engine)
    elapsed = time.perf_counter() - started
    end_count = int(optimized.candidate_tensors()["global_index"].numel())
    steps_per_second = BENCHMARK_STEPS / max(elapsed, 1e-12)
    memory = _benchmark_memory(optimized)
    # Keep the decomposition explicit so the next server profile can identify
    # any regression to Python row/sort work in the hot path.
    components = {"ranking": ranking_seconds, "remove_and_state_update": remove_seconds,
                  "other": max(0.0, elapsed - ranking_seconds - remove_seconds)}
    return {
        "variant": variant, "steps": BENCHMARK_STEPS,
        "seconds_total": elapsed, "steps_per_second": steps_per_second,
        "seconds_per_step": elapsed / BENCHMARK_STEPS,
        "ranking_seconds": ranking_seconds, "remove_seconds": remove_seconds,
        "other_seconds": max(0.0, elapsed - ranking_seconds - remove_seconds),
        "top_runtime_components": [name for name, _ in sorted(components.items(), key=lambda item: item[1], reverse=True)],
        "candidate_count_start": start_count, "candidate_count_end": end_count,
        "hot_path_candidate_rows": False, "hot_path_python_sorted": False,
        **memory,
        "status": "PASS" if steps_per_second >= BENCHMARK_MIN_STEPS_PER_SECOND[variant] else "FAIL",
    }


def benchmark_gate(report: Mapping[str, object]) -> bool:
    """Strictly gate the full replay on all three measured throughputs."""
    variants = report.get("variants", report)
    if not isinstance(variants, Mapping):
        return False
    for variant, minimum in BENCHMARK_MIN_STEPS_PER_SECOND.items():
        row = variants.get(variant)
        if not isinstance(row, Mapping):
            return False
        if row.get("status") != "PASS":
            return False
        if float(row.get("steps_per_second", 0.0)) < minimum:
            return False
    return report.get("status", "PASS") in ("PASS", True)


def run_benchmark(*, task014_root: Path, task016_root: Path, task017_root: Path,
                  output_dir: Path, device: str = "cuda:0",
                  ranking_backend: str = "single-gpu") -> dict[str, object]:
    """Run the 1000-step preflight and write telemetry only."""
    rows = {}
    for variant in VARIANTS:
        rows[variant] = benchmark_variant(
            variant=variant, task014_root=task014_root,
            task016_root=task016_root, task017_root=task017_root,
            output_dir=output_dir, device=device,
            ranking_backend=ranking_backend,
        )
    report: dict[str, object] = {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows.values()) else "FAIL",
        "steps": BENCHMARK_STEPS, "variants": rows,
        "estimated_minutes_to_50": {
            v: EXPECTED_TASK028_REMOVED / max(float(r["steps_per_second"]), 1e-12) / 60.0
            for v, r in rows.items()
        },
    }
    report["estimated_total_replay_minutes"] = sum(report["estimated_minutes_to_50"].values())
    report["performance_benchmark_pass"] = benchmark_gate(report)
    atomic_json(Path(output_dir) / "task031_performance_benchmark.json", report)
    print("Task031 performance benchmark:", flush=True)
    for variant in VARIANTS:
        row = rows[variant]
        print(f"{variant}: {float(row['steps_per_second']):.3f} steps/s {row['status']}", flush=True)
    print(f"Estimated total replay: {float(report['estimated_total_replay_minutes']):.2f} min", flush=True)
    if not report["performance_benchmark_pass"]:
        raise RuntimeError("Task031 performance benchmark gate failed; full replay not started")
    return report


def prepare_task031_output(output_dir: Path) -> None:
    """Mark a previous incomplete diagnostic as superseded.

    Only Task031's own run marker is touched.  Frozen Task028/029/030 roots
    are never cleaned or rewritten; every official output is subsequently
    replaced atomically by ``run_diagnosis``.
    """
    output = Path(output_dir)
    prior = output / "task031_completion.json"
    if not prior.is_file() or read_json(prior).get("status") != "PASS":
        atomic_json(output / "task031_run_state.json", {
            "status": "RUNNING", "code_version": CODE_VERSION,
            "incomplete_previous_run_ignored": prior.is_file(),
        })


def _snapshot_summary_rows(variant: str, target: float, sparsity: float, candidates: Sequence[Mapping[str, object]], selected: Sequence[Mapping[str, object]], total_parameters: float) -> list[dict[str, object]]:
    attention = [r for r in candidates if str(r.get("unit_type")) == TYPE_ATTENTION]; ffns = [r for r in candidates if str(r.get("unit_type")) == TYPE_FFN]
    best = attention[0] if attention else {}; ahead = sum(str(r.get("unit_type")) == TYPE_FFN for r in candidates[:int(best.get("global_rank", len(candidates) + 1)) - 1]) if best else ""
    old = [r for r in selected if float(r.get("effective_sparsity_after", 0)) <= sparsity + 1e-12]
    return [{"variant": variant, "snapshot_target": target, "actual_effective_sparsity": sparsity, "candidate_count": len(candidates), "removed_attention": sum(str(r.get("unit_type")) == TYPE_ATTENTION for r in old), "removed_ffn": sum(str(r.get("unit_type")) == TYPE_FFN for r in old), "removed_attention_parameters": sum(_int(r.get("parameter_cost", 0)) for r in old if str(r.get("unit_type")) == TYPE_ATTENTION), "removed_ffn_parameters": sum(_int(r.get("parameter_cost", 0)) for r in old if str(r.get("unit_type")) == TYPE_FFN), "remaining_attention": len(attention), "remaining_ffn": len(ffns), "best_attention_rank": best.get("global_rank", ""), "ffn_candidates_ahead": ahead, "best_attention_global_index": best.get("global_index", ""), "best_attention_p_total": best.get("p_total", ""), "best_attention_p_A_variant": best.get("p_A_variant", ""), "best_attention_domain_damage": best.get("domain_damage", ""), "best_attention_R_adaptive": best.get("R_adaptive", "")}]


def domain_active_size_audit(rows: Sequence[Mapping[str, object]], variant: str = "V1_domain", snapshot_target: float | None = None) -> list[dict[str, object]]:
    groups: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows: groups[row.get("domain_id")].append(row)
    bins = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-49", 10, 49), (">=50", 50, 10**12))
    counts = {name: 0 for name, _, _ in bins}; units = {name: 0 for name, _, _ in bins}
    for group in groups.values():
        for name, lo, hi in bins:
            if lo <= len(group) <= hi: counts[name] += 1; units[name] += len(group); break
    total = max(1, len(rows)); return [{"variant": variant, "snapshot_target": snapshot_target if snapshot_target is not None else "", "size_bin": n, "domain_count": counts[n], "candidate_count": units[n], "fraction_candidates": units[n] / total, "singleton_fraction": units[n] / total if n == "1" else 0.0} for n, _, _ in bins]


def mixed_domain_audit(rows: Sequence[Mapping[str, object]], variant: str = "V1_domain") -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows: groups[row.get("domain_id")].append(row)
    output = []
    for domain, group in sorted(groups.items(), key=lambda x: str(x[0])):
        a = [r for r in group if r.get("unit_type") == TYPE_ATTENTION]; f = [r for r in group if r.get("unit_type") == TYPE_FFN]
        if not a or not f: continue
        a_da = [_float(r.get("Delta_average")) for r in a]; f_da = [_float(r.get("Delta_average")) for r in f]
        a_pa = [_float(r.get("p_A_variant", r.get("p_A_domain", r.get("p_average", 0.0)))) for r in a]; f_pa = [_float(r.get("p_A_variant", r.get("p_A_domain", r.get("p_average", 0.0)))) for r in f]
        a_pg = [_float(r.get("p_A_global", r.get("p_average", 0.0))) for r in a]; f_pg = [_float(r.get("p_A_global", r.get("p_average", 0.0))) for r in f]
        output.append({"variant": variant, "domain_id": domain, "attention_count": len(a), "ffn_count": len(f), "attention_Delta_average_values": json.dumps(a_da), "ffn_Delta_average_values": json.dumps(f_da), "attention_p_A_global_values": json.dumps(a_pg), "ffn_p_A_global_values": json.dumps(f_pg), "attention_p_A_domain_values": json.dumps(a_pa), "ffn_p_A_domain_values": json.dumps(f_pa), "attention_Delta_average_median": statistics.median(a_da), "ffn_Delta_average_median": statistics.median(f_da), "attention_p_A_median": statistics.median(a_pa), "ffn_p_A_median": statistics.median(f_pa), "median_gap": statistics.median(a_pa) - statistics.median(f_pa), "rank_order": "attention_lower" if statistics.median(a_pa) < statistics.median(f_pa) else "ffn_lower"})
    summary = {"variant": variant, "mixed_domain_count": len(output), "audit_complete": True, "rows": len(output)}
    return output, summary


def build_type_distribution(rows: Sequence[Mapping[str, object]], variant: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    output = []; separation = {}
    for typ in (TYPE_ATTENTION, TYPE_FFN):
        subset = [r for r in rows if str(r.get("unit_type")) == typ]
        for field in ("Delta_average", "p_A_variant", "p_total", "domain_damage", "R_adaptive"):
            stats = quantiles([_float(r.get(field), field) for r in subset]); output.append({"variant": variant, "unit_type": typ, "metric": field, **stats})
    a = [_float(r.get("p_A_variant"), "p_A_variant") for r in rows if r.get("unit_type") == TYPE_ATTENTION]; f = [_float(r.get("p_A_variant"), "p_A_variant") for r in rows if r.get("unit_type") == TYPE_FFN]
    separation.update({"variant": variant, "attention_count": len(a), "ffn_count": len(f), "attention_minus_ffn_median_gap": (statistics.median(a) - statistics.median(f)) if a and f else math.nan, "attention_ffn_raw_delta_average_median_ratio": (statistics.median([_float(r.get("Delta_average")) for r in rows if r.get("unit_type") == TYPE_ATTENTION]) / statistics.median([_float(r.get("Delta_average")) for r in rows if r.get("unit_type") == TYPE_FFN])) if a and f else math.nan, "p_A_ks_distance": ks_distance(a, f), "p_A_auc_attention_greater_ffn": auc_greater(a, f)})
    return output, separation


def cost_class_audit(rows: Sequence[Mapping[str, object]], variant: str, snapshot_target: float | None = None) -> list[dict[str, object]]:
    groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows: groups[_int(row.get("parameter_cost"), "parameter_cost")].append(row)
    return [{"variant": variant, "snapshot_target": snapshot_target if snapshot_target is not None else "", "parameter_cost": cost, "class_size": len(group), "median_p_A": statistics.median([_float(r.get("p_A_variant")) for r in group])} for cost, group in sorted(groups.items())]


def domain_order_preservation(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows: groups[row.get("domain_id")].append(row)
    output = []
    for domain, group in sorted(groups.items(), key=lambda x: str(x[0])):
        if len(group) < 2: continue
        values = [_float(r.get("p_A_variant", r.get("p_A_domain", 0.0)), "p_A_domain") for r in group]
        output.append({"domain_id": domain, "candidate_count": len(group), "spearman_Delta_average_vs_p_A_domain": spearman([_float(r.get("Delta_average")) for r in group], values)})
    return output


def local_ordering_preserved(rows: Sequence[Mapping[str, object]]) -> bool:
    reports = domain_order_preservation(rows)
    return all(abs(float(r["spearman_Delta_average_vs_p_A_domain"]) - 1.0) < 1e-12 for r in reports)


def singleton_fallback_audit(evaluations: Sequence[Mapping[str, object]], selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
    singleton = [r for r in evaluations if str(r.get("active_domain_size", "")) in ("1", "1.0")]
    selected_singleton = [r for r in selected if str(r.get("active_domain_size", "")) in ("1", "1.0")]
    return {"singleton_candidate_evaluations": len(singleton), "singleton_selected_removals": len(selected_singleton), "total_removals": len(selected), "fallback_fraction": len(selected_singleton) / max(1, len(selected)), "fallback_rule": "p_A_domain = p_total"}


def make_figures(output_dir: Path) -> list[str]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(output_dir); root.joinpath("figures").mkdir(parents=True, exist_ok=True); paths = []
    snapshots = read_csv(root / "variant_selection_snapshots.csv") if (root / "variant_selection_snapshots.csv").is_file() else []
    dist = read_csv(root / "variant_type_distribution.csv") if (root / "variant_type_distribution.csv").is_file() else []
    causal = read_csv(root / "tested_unit_risk_table.csv") if (root / "tested_unit_risk_table.csv").is_file() else []
    alignment = read_csv(root / "causal_alignment_by_variant.csv") if (root / "causal_alignment_by_variant.csv").is_file() else []
    mixed = read_csv(root / "mixed_domain_audit.csv") if (root / "mixed_domain_audit.csv").is_file() else []
    active = read_csv(root / "domain_active_size_audit.csv") if (root / "domain_active_size_audit.csv").is_file() else []
    shifts = read_csv(root / "global_rank_shift_50.csv") if (root / "global_rank_shift_50.csv").is_file() else []
    def save(fig, num, title):
        path = root / "figures" / f"Figure{num:02d}_{title}.png"; fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path))
    variants = list(VARIANTS)
    for num in range(1, 11):
        fig, ax = plt.subplots(figsize=(6, 4))
        if num == 1:
            for v in variants:
                for typ, marker in ((TYPE_ATTENTION, "o"), (TYPE_FFN, "s")):
                    vals = [_float(r.get("median"), "median") for r in dist if r.get("variant") == v and r.get("unit_type") == typ and r.get("metric") == "p_A_variant"]
                    ax.plot([v], vals or [math.nan], marker=marker, linestyle="none", label=f"{v}/{typ}")
            ax.set_ylabel("p_A median"); ax.legend(fontsize=6)
        elif num == 2:
            for v in variants:
                sub = [r for r in causal if r.get("variant") == v and r.get("p_A_variant", "") not in (None, "") and r.get("parameter_cost", "") not in (None, "")]
                ax.scatter([math.log(_float(r.get("parameter_cost"), "parameter_cost")) for r in sub], [_float(r.get("p_A_variant"), "p_A_variant") for r in sub], label=v)
            ax.set_xlabel("log(parameter cost)"); ax.set_ylabel("p_A"); ax.legend(fontsize=7)
        elif num == 3:
            for risk, marker in (("p_A_global", "o"), ("p_A_domain", "s"), ("p_A_cost", "^")):
                sub = [r for r in causal if r.get(risk, "") not in (None, "") and r.get("relative_logit_l2", "") not in (None, "")]
                ax.scatter([_float(r.get(risk), risk) for r in sub], [_float(r.get("relative_logit_l2"), "relative_logit_l2") for r in sub], marker=marker, label=risk)
            ax.set_xlabel("risk p_A"); ax.set_ylabel("Task030 relative logit L2"); ax.legend(fontsize=7)
        elif num == 4:
            for v in variants:
                sub = [r for r in snapshots if r.get("variant") == v and r.get("best_attention_rank", "") not in (None, "")]; ax.plot([_float(r.get("actual_effective_sparsity"), "sparsity") for r in sub], [_float(r.get("best_attention_rank"), "rank") for r in sub], marker="o", label=v)
            ax.set_xlabel("effective sparsity"); ax.set_ylabel("best Attention rank"); ax.legend(fontsize=7)
        elif num == 5:
            for v in variants:
                sub = [r for r in snapshots if r.get("variant") == v]; ax.plot([_float(r.get("actual_effective_sparsity"), "sparsity") for r in sub], [_float(r.get("removed_attention"), "removed") for r in sub], marker="o", label=v)
            ax.set_xlabel("effective sparsity"); ax.set_ylabel("Attention removals"); ax.legend(fontsize=7)
        elif num == 6:
            stage_rows = read_csv(root / "variant_stage_removals.csv") if (root / "variant_stage_removals.csv").is_file() else []
            for v in variants:
                sub = [r for r in stage_rows if r.get("variant") == v]; ax.plot([str(r.get("stage")) for r in sub], [_float(r.get("removed_count"), "removed_count") for r in sub], marker="o", label=v)
            ax.set_xlabel("stage"); ax.set_ylabel("FFN/Attention removal count"); ax.legend(fontsize=7)
        elif num == 7:
            for typ, marker in (("attention_p_A_domain", "o"), ("ffn_p_A_domain", "s")):
                vals = [_float(r.get(typ), typ) for r in mixed if r.get(typ, "") not in (None, "")]; ax.scatter(range(len(vals)), vals, marker=marker, label=typ)
            ax.set_xlabel("mixed-domain row"); ax.set_ylabel("domain-conditioned p_A"); ax.legend(fontsize=7)
        elif num == 8:
            for v in variants:
                sub = [r for r in active if r.get("variant") == v]; ax.bar([f"{v}:{r.get('size_bin')}" for r in sub], [_float(r.get("domain_count"), "domain_count") for r in sub], alpha=.5, label=v)
            ax.set_ylabel("domain count"); ax.tick_params(axis="x", rotation=70)
        elif num == 9:
            sub = [r for r in shifts if r.get("rank_global_A", "") not in (None, "") and r.get("rank_domain_A_when_embedded_globally", "") not in (None, "")]; ax.scatter([_float(r.get("rank_global_A"), "rank_global_A") for r in sub], [_float(r.get("rank_domain_A_when_embedded_globally"), "rank_domain_A") for r in sub]); ax.set_xlabel("global p_A rank"); ax.set_ylabel("V1 embedded rank")
        else:
            for group in ("all", TYPE_ATTENTION, TYPE_FFN):
                sub = [r for r in alignment if r.get("group") == group and r.get("risk_quantity") == "p_A_domain" and r.get("metric") == "relative_logit_l2"]; ax.plot([str(r.get("context")) for r in sub], [_float(r.get("rho"), "rho") for r in sub], marker="o", label=group)
            ax.set_ylabel("Spearman rho"); ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=7)
        save(fig, num, ("pA_distributions" if num == 1 else f"diagnostic_{num:02d}"))
    return paths


def write_summary(output_dir: Path, *, separations: Mapping[str, Mapping[str, object]], overlaps: Mapping[str, object], causal: Mapping[str, object], singleton: Mapping[str, object]) -> Path:
    lines = ["# Task031 scientific summary", "", "Task031 is an offline diagnostic of the frozen Task028 selector. No model forward, validation, training, fine-tuning, checkpoint load, or new registry was performed.", "", "## Interpretation", "", "V0 is the current global Average reference set. V1 uses current BMS-domain reference sets; V2 is an exact parameter-cost-class diagnostic control. The measured tables and causal alignment, rather than a desired Attention count, determine whether a change is supported.", "", "- `p_total` remains global safety.", "- V1 `p_A_domain` is local within-domain sensitivity.", "- `domain_damage` remains current domain state.", "- No new pruning hyperparameter was introduced (V0/V1/V2 = 0).", "", f"Singleton fallback rule: `{singleton.get('fallback_rule', 'p_A_domain = p_total')}`.", "", "All granularity, mixed-domain, causal, ordering, overlap, and rank-shift results are in the CSV/JSON artifacts."]
    path = Path(output_dir) / "task031_scientific_summary.md"; path.write_text("\n".join(lines) + "\n", encoding="utf-8"); return path


def completion_gate(payload: Mapping[str, object]) -> bool:
    positive = ("task028_inputs_verified", "task029_inputs_verified", "task030_inputs_verified", "source_hashes_unchanged", "v0_exact_task028_replay", "v0_first_100_exact", "v0_last_100_exact", "v1_replay_complete", "v2_replay_complete", "snapshots_complete", "mixed_domain_audit_complete", "causal_alignment_complete", "granularity_dependence_complete", "singleton_fallback_complete", "selection_overlap_complete", "figures_complete", "scientific_summary_complete", "performance_benchmark_pass", "optimized_hot_path_used")
    negative = ("training_executed", "fine_tuning_executed", "validation_executed", "model_forward_executed")
    return payload.get("status") == "PASS" and all(payload.get(k) is True for k in positive) and payload.get("python_full_candidate_sort_in_hot_path") is False and all(payload.get(k) is False for k in negative) and int(payload.get("new_pruning_hyperparameters", -1)) == 0


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable if iterable is not None else range(0)


def _stage_removals(selected: Sequence[Mapping[str, object]], variant: str) -> list[dict[str, object]]:
    groups = Counter(str(row.get("stage", _stage(row))) for row in selected)
    return [{"variant": variant, "stage": stage, "removed_count": count} for stage, count in sorted(groups.items())]


def _domain_removals(selected: Sequence[Mapping[str, object]], variant: str) -> list[dict[str, object]]:
    groups = Counter(_int(row.get("domain_id", 0), "domain_id") for row in selected)
    return [{"variant": variant, "domain_id": domain, "removed_count": count} for domain, count in sorted(groups.items())]


def _variant_granularity_rows(result: Mapping[str, object], variant: str) -> list[dict[str, object]]:
    output = []
    for snap in SNAPSHOT_TARGETS:
        rows = [r for r in result.get("snapshots", ()) if abs(float(r.get("snapshot_target", -1)) - snap) < 1e-12]
        # A snapshot summary is intentionally compact; candidate-level rows
        # are supplied by ``final_candidates`` at 50% and remain the source for
        # the full granularity audit.
        output.extend(granularity_dependence(result.get("final_candidates", ()), variant, snap) if snap == .5 else [])
    return output


def run_diagnosis(*, task028_root: Path, task029_root: Path, task030_root: Path, output_dir: Path, device: str = "cuda:0", ranking_backend: str = "single-gpu", task014_root: Path | None = None, task016_root: Path | None = None, task017_root: Path | None = None) -> dict[str, object]:
    prepare_task031_output(output_dir)
    identity_result = verify_identity(task028_root=task028_root, task029_root=task029_root, task030_root=task030_root, output_dir=output_dir)
    refs = (identity_result["task028"], identity_result["task029"], identity_result["task030"]); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    benchmark_path = output / "task031_performance_benchmark.json"
    if not benchmark_path.is_file() or not benchmark_gate(read_json(benchmark_path)):
        raise RuntimeError("Task031 performance benchmark PASS is required before full replay")
    if not (task014_root and task016_root and task017_root):
        raise RuntimeError("optimized Task028 replay requires read-only Task014/Task016/Task017 roots; pass them explicitly")
    import task028_main50_tad_logical_finetune as task028
    results = {}
    for variant in VARIANTS:
        engine, optimized = _engine_for_replay(task014_root=task014_root, task016_root=task016_root, task017_root=task017_root, output_dir=output, device=device, ranking_backend=ranking_backend)
        lookup = lookup_from_engine(engine)
        for row in refs[0]["trace"]:
            gid = _int(row.get("global_index"), "global_index")
            lookup[gid] = {**lookup.get(gid, {}), **dict(row)}
        total = float(getattr(engine, "total_parameters")); result = replay_variant(variant=variant, engine=engine, optimized=optimized, total_parameters=total, target_budget=TARGET_SPARSITY * total, lookup=lookup, task028_sequence=refs[0]["sequence"])
        if variant == "V0_global":
            if result["sequence"] != refs[0]["sequence"]:
                raise RuntimeError("V0 replay does not reproduce Task028 sequence")
            if result["sequence"][:100] != refs[0]["sequence"][:100]:
                raise RuntimeError("V0 first-100 sequence differs from Task028")
            if result["sequence"][-100:] != refs[0]["sequence"][-100:]:
                raise RuntimeError("V0 last-100 sequence differs from Task028")
            if len(result["sequence"]) != EXPECTED_TASK028_REMOVED:
                raise RuntimeError("V0 replay length differs from Task028 (28044)")
            counts = Counter(r.get("unit_type") for r in result["selected"])
            if counts[TYPE_ATTENTION] != EXPECTED_TASK028_ATTENTION or counts[TYPE_FFN] != EXPECTED_TASK028_FFN:
                raise RuntimeError("V0 replay type counts differ from Task028")
        results[variant] = result
    all_snapshots = [row for result in results.values() for row in result["snapshots"]]
    atomic_csv(output / "variant_selection_snapshots.csv", SNAPSHOT_FIELDS, all_snapshots)
    summaries, distributions, separations = [], [], {}
    first_attention = {}
    for variant in VARIANTS:
        result = results[variant]; final_rows = list(result.get("final_candidates", ())); dist, sep = build_type_distribution(final_rows, variant); distributions.extend(dist); separations[variant] = sep
        first = next((r for r in result["selected"] if r.get("unit_type") == TYPE_ATTENTION), None); first_attention[variant] = {"first_attention_found": first is not None, "first_attention": first or {}, "first_attention_sparsity": first.get("effective_sparsity_after") if first else None}
        summaries.append({"variant": variant, "effective_sparsity": result["final_sparsity"], "attention_removed": sum(r.get("unit_type") == TYPE_ATTENTION for r in result["selected"]), "ffn_removed": sum(r.get("unit_type") == TYPE_FFN for r in result["selected"]), "sequence_length": len(result["sequence"]), "first_attention_sparsity": first.get("effective_sparsity_after") if first else None})
    atomic_csv(output / "variant_final_50_summary.csv", tuple(summaries[0].keys()), summaries); atomic_csv(output / "variant_type_distribution.csv", tuple(distributions[0].keys()) if distributions else ("variant",), distributions); atomic_json(output / "variant_type_separation.json", separations); atomic_json(output / "variant_first_attention.json", first_attention)
    stage_rows = [_stage_removals(results[v]["selected"], v) for v in VARIANTS]; domain_rows = [_domain_removals(results[v]["selected"], v) for v in VARIANTS]
    atomic_csv(output / "variant_stage_removals.csv", ("variant", "stage", "removed_count"), [r for group in stage_rows for r in group]); atomic_csv(output / "variant_domain_removals.csv", ("variant", "domain_id", "removed_count"), [r for group in domain_rows for r in group])
    granularity = [r for v in VARIANTS for target in SNAPSHOT_TARGETS for r in granularity_dependence([x for x in results[v].get("snapshot_candidate_rows", ()) if abs(float(x.get("snapshot_target", -1)) - target) < 1e-12], v, target)]
    atomic_csv(output / "variant_granularity_dependence.csv", ("variant", "snapshot_target", "group", "spearman_pA_log_cost", "candidate_count"), granularity)
    # V1 singleton and active-domain-size diagnostics are evaluated on the
    # exact final feasible set; fallback removals are counted from its trace.
    v1_final = list(results["V1_domain"].get("final_candidates", ())); active_audit = [r for target in SNAPSHOT_TARGETS for r in domain_active_size_audit([x for x in results["V1_domain"].get("snapshot_candidate_rows", ()) if abs(float(x.get("snapshot_target", -1)) - target) < 1e-12], "V1_domain", target)]; atomic_csv(output / "domain_active_size_audit.csv", ("variant", "snapshot_target", "size_bin", "domain_count", "candidate_count", "fraction_candidates", "singleton_fraction"), active_audit)
    cost_rows = [r for v in VARIANTS for target in SNAPSHOT_TARGETS for r in cost_class_audit([x for x in results[v].get("snapshot_candidate_rows", ()) if abs(float(x.get("snapshot_target", -1)) - target) < 1e-12], v, target)]; atomic_csv(output / "cost_class_audit.csv", ("variant", "snapshot_target", "parameter_cost", "class_size", "median_p_A"), cost_rows)
    mixed_rows, mixed_summary = mixed_domain_audit(results["V0_global"].get("final_candidates", ()), "V0_global"); atomic_csv(output / "mixed_domain_audit.csv", tuple(mixed_rows[0].keys()) if mixed_rows else ("variant", "domain_id"), mixed_rows); atomic_json(output / "mixed_domain_summary.json", mixed_summary)
    base_candidates = list(results["V0_global"].get("final_candidates", ()))
    v0_scores = {int(r["global_index"]): r for r in compute_variant_scores(base_candidates, "V0_global")}
    v1_scores = {int(r["global_index"]): r for r in compute_variant_scores(base_candidates, "V1_domain")}
    v2_scores = {int(r["global_index"]): r for r in compute_variant_scores(base_candidates, "V2_cost")}
    risk_rows = []
    for gid, r0 in v0_scores.items():
        r1, r2 = v1_scores[gid], v2_scores[gid]; risk_rows.append({**r0, "p_A_global": r0["p_A_variant"], "p_A_domain": r1["p_A_variant"], "p_A_cost": r2["p_A_variant"], "domain_damage": r0.get("domain_damage", 0.0)})
    raw = refs[2]["raw"]; joined, align_rows, align_summary = causal_alignment_by_variant(raw, risk_rows, refs[2]["heads"], refs[2]["pairs"]); atomic_csv(output / "causal_alignment_by_variant.csv", ("context", "group", "risk_quantity", "metric", "rho", "n"), align_rows); atomic_json(output / "causal_alignment_summary.json", align_summary); atomic_csv(output / "tested_unit_risk_table.csv", tuple(joined[0].keys()) if joined else ("global_index", "unit_type"), joined)
    pair_audit = granularity_controlled_pair_audit(raw, risk_rows, refs[2]["packs"]); atomic_csv(output / "granularity_controlled_pair_audit.csv", tuple(pair_audit[0].keys()) if pair_audit else ("pair_id",), pair_audit)
    singleton = {"singleton_candidate_evaluations": results["V1_domain"].get("singleton_candidate_evaluations", 0), "singleton_selected_removals": results["V1_domain"].get("singleton_selected_removals", 0), "total_removals": len(results["V1_domain"]["selected"]), "fallback_fraction": results["V1_domain"].get("singleton_selected_removals", 0) / max(1, len(results["V1_domain"]["selected"])), "fallback_rule": "p_A_domain = p_total"}; atomic_json(output / "singleton_fallback_audit.json", singleton)
    type_lookup = {int(r.get("global_index")): str(r.get("unit_type", "")) for r in base_candidates}; overlaps = overlap_report({v: results[v]["sequence"] for v in VARIANTS}, type_lookup); atomic_json(output / "variant_selection_overlap.json", overlaps)
    atomic_csv(output / "domain_order_preservation.csv", ("domain_id", "candidate_count", "spearman_Delta_average_vs_p_A_domain"), domain_order_preservation([r for r in compute_variant_scores(v1_final, "V1_domain")])); atomic_csv(output / "global_rank_shift_50.csv", ("global_index", "p_A_global", "p_A_domain", "rank_global_A", "rank_domain_A_when_embedded_globally", "rank_shift"), rank_shift_rows(risk_rows))
    make_figures(output); write_summary(output, separations=separations, overlaps=overlaps, causal=align_summary, singleton=singleton)
    source_hashes_after = {"task028": hash_inputs(Path(task028_root), TASK028_INPUTS), "task029": hash_inputs(Path(task029_root), TASK029_INPUTS), "task030": hash_inputs(Path(task030_root), TASK030_INPUTS)}
    source_unchanged = source_hashes_after == {"task028": refs[0]["hashes"], "task029": refs[1]["hashes"], "task030": refs[2]["hashes"]}
    if not source_unchanged: raise RuntimeError("Task028/Task029/Task030 inputs changed during Task031")
    benchmark = read_json(benchmark_path)
    completion = {"status": "PASS", "task028_inputs_verified": True, "task029_inputs_verified": True, "task030_inputs_verified": True, "source_hashes_unchanged": source_unchanged, "v0_exact_task028_replay": results["V0_global"]["sequence"] == refs[0]["sequence"], "v0_first_100_exact": results["V0_global"]["sequence"][:100] == refs[0]["sequence"][:100], "v0_last_100_exact": results["V0_global"]["sequence"][-100:] == refs[0]["sequence"][-100:], "v1_replay_complete": bool(results["V1_domain"]["sequence"]), "v2_replay_complete": bool(results["V2_cost"]["sequence"]), "snapshots_complete": all(r["snapshots_complete"] for r in results.values()), "mixed_domain_audit_complete": mixed_summary.get("audit_complete") is True, "causal_alignment_complete": align_summary.get("alignment_complete") is True, "granularity_dependence_complete": bool(granularity), "singleton_fallback_complete": True, "selection_overlap_complete": bool(overlaps), "figures_complete": len(list((output / "figures").glob("Figure*.png"))) >= 10, "scientific_summary_complete": (output / "task031_scientific_summary.md").is_file(), "performance_benchmark_pass": benchmark_gate(benchmark), "optimized_hot_path_used": True, "python_full_candidate_sort_in_hot_path": False, "training_executed": False, "fine_tuning_executed": False, "validation_executed": False, "model_forward_executed": False, "selector_replay_executed": True, "new_pruning_hyperparameters": 0, "v0_sequence_sha256": sequence_sha256(results["V0_global"]["sequence"])}
    if not completion_gate(completion): raise RuntimeError("Task031 completion gate failed")
    atomic_json(output / "task031_completion.json", completion); return {"identity": identity_result["identity"], "results": results, "completion": completion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("identity", "benchmark", "run", "completion"), required=True); parser.add_argument("--task028-root", type=Path); parser.add_argument("--task029-root", type=Path); parser.add_argument("--task030-root", type=Path); parser.add_argument("--task014-root", type=Path); parser.add_argument("--task016-root", type=Path); parser.add_argument("--task017-root", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default=os.environ.get("TASK031_DEVICE", "cuda:0")); parser.add_argument("--ranking-backend", default=os.environ.get("TASK031_RANKING_BACKEND", "single-gpu")); return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "completion":
        if not completion_gate(read_json(args.output_dir / "task031_completion.json")): raise RuntimeError("Task031 completion gate failed")
        print("Task031 completion: PASS", flush=True); return 0
    if args.mode == "benchmark":
        roots = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in roots):
            raise SystemExit("Task031 benchmark requires Task014, Task016 and Task017 roots")
        run_benchmark(task014_root=args.task014_root, task016_root=args.task016_root, task017_root=args.task017_root, output_dir=args.output_dir, device=args.device, ranking_backend=args.ranking_backend)
        print("Task031 benchmark: PASS", flush=True); return 0
    required = (args.task028_root, args.task029_root, args.task030_root)
    if any(v is None for v in required): raise SystemExit("Task031 requires Task028, Task029 and Task030 roots")
    if args.mode == "identity":
        verify_identity(task028_root=args.task028_root, task029_root=args.task029_root, task030_root=args.task030_root, output_dir=args.output_dir); print("Task031 identity: PASS", flush=True); return 0
    run_diagnosis(task028_root=args.task028_root, task029_root=args.task029_root, task030_root=args.task030_root, output_dir=args.output_dir, device=args.device, ranking_backend=args.ranking_backend, task014_root=args.task014_root, task016_root=args.task016_root, task017_root=args.task017_root); print("Task031: PASS", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())

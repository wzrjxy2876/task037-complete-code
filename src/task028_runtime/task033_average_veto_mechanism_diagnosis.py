"""Task033: diagnose whether the frozen Average hard veto is over-protective.

This module is deliberately diagnostic-only.  It reconstructs the frozen
Task028/Task031 risk components, measures the additive Average veto margin,
and compares four *counterfactual fusion operators*.  The operators are not
registered pruning methods and this module never performs a model forward,
validation, training, or fine-tuning.

The selector replay, when the optional immutable Task014/016/017 roots are
available, delegates candidate state and removal to the optimized Task028
backend.  Python rows are used only for reports and snapshot boundaries; the
per-removal comparison is tensor-side.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_VERSION = "task033_average_veto_mechanism_diagnosis_v1"
SEED = 3407
BOOTSTRAP_REPLICATES = 10_000
BENCHMARK_STEPS = 1_000
BENCHMARK_MIN_STEPS_PER_SECOND = {
    "F0_HARD": 100.0,
    "F1_NO_VETO": 50.0,
    "F2_CONSENSUS": 50.0,
    "F3_MID_VETO": 50.0,
}
TARGET_SPARSITY = 0.50
SNAPSHOT_TARGETS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
CONTEXTS = ("FULL", "TASK028_50")
FUSIONS = ("F0_HARD", "F1_NO_VETO", "F2_CONSENSUS", "F3_MID_VETO")
SINGLE_INTERVENTIONS = ("attention_single", "ffn_single")
CAUSAL_METRICS = (
    ("relative_logit_l2", ("relative_logit_l2", "relative_logit_L2", "relative_logit_l2_effect")),
    ("kl", ("kl", "kl_divergence", "KL", "KL_divergence")),
    ("ce_increase", ("ce_increase", "CE_increase", "cross_entropy_increase")),
    ("prediction_flip_rate", ("prediction_flip_rate", "flip_rate", "prediction_flip")),
)

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
)
TASK031_INPUTS = (
    "tested_unit_risk_table.csv",
    "granularity_controlled_pair_audit.csv",
    "variant_granularity_dependence.csv",
    "variant_type_distribution.csv",
    "variant_final_50_summary.csv",
)
TASK032_INPUTS = (
    "artifact_identity.json",
    "causal_beta_pairwise.csv",
    "causal_beta_leave_one_out.csv",
    "causal_beta_bootstrap.json",
    "causal_beta_summary.json",
    "average_granularity_scaling.csv",
    "causal_granularity_components.csv",
    "excess_granularity_decomposition.json",
    "causal_residual_alignment.csv",
    "causal_residual_alignment_summary.json",
    "cost_matched_decomposition.csv",
    "decomposed_type_separation.csv",
    "domain_residual_analysis.csv",
    "mixed_domain_residual_analysis.csv",
    "stage_granularity_decomposition.csv",
    "task032_scientific_summary.md",
    "task032_completion.json",
)

TASK033_REPLAY_ARTIFACTS = (
    "task033_performance_benchmark.json",
    "veto_margin_by_snapshot.csv",
    "veto_margin_summary.json",
    "veto_causal_alignment.csv",
    "veto_causal_alignment_summary.json",
    "pairwise_veto_audit.csv",
    "pairwise_veto_accuracy.json",
    "fusion_causal_alignment.csv",
    "fusion_causal_alignment_summary.json",
    "fusion_pairwise_decisions.csv",
    "fusion_pairwise_accuracy.json",
    "veto_cost_matched_pack_audit.csv",
    "fusion_stage_removals.csv",
    "fusion_domain_removals.csv",
    "fusion_first_attention.json",
    "fusion_final_50_summary.csv",
    "fusion_selection_overlap.json",
    "fusion_selection_snapshots.csv",
    "f0_veto_decision_changes.csv",
    "f0_veto_decision_summary.json",
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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    text = ",".join(str(_int(value, "global_index")) for value in indices)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _output_is_separate(output: Path, roots: Sequence[Path]) -> None:
    output = Path(output).resolve()
    for root in roots:
        root = Path(root).resolve()
        if output == root or root in output.parents:
            raise RuntimeError("Task033 output must be separate from immutable roots")


def _ordinal(values: Sequence[float], gids: Sequence[int]) -> list[float]:
    if len(values) != len(gids):
        raise ValueError("values and global_index lengths differ")
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), int(gids[index])))
    denominator = float(max(len(values) - 1, 1))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def ordinal_rank(values: Sequence[float], global_indices: Sequence[int] | None = None) -> list[float]:
    gids = list(range(len(values))) if global_indices is None else [_int(value, "global_index") for value in global_indices]
    return _ordinal([float(value) for value in values], gids)


# ---------------------------------------------------------------------------
# Frozen component and fusion definitions
# ---------------------------------------------------------------------------

def base_risk(p_total: float, domain_damage: float) -> float:
    """Return B=max(p_total, domain_damage), without altering either input."""
    return max(float(p_total), float(domain_damage))


def hard_risk(p_total: float, p_average: float, domain_damage: float) -> float:
    return max(base_risk(p_total, domain_damage), float(p_average))


def veto_margin(p_total: float, p_average: float, domain_damage: float) -> float:
    return max(float(p_average) - base_risk(p_total, domain_damage), 0.0)


def veto_active(p_total: float, p_average: float, domain_damage: float) -> bool:
    return float(p_average) > base_risk(p_total, domain_damage)


def f0_hard(p_total: float, p_average: float, domain_damage: float) -> float:
    return hard_risk(p_total, p_average, domain_damage)


def f1_no_veto(p_total: float, p_average: float, domain_damage: float) -> float:
    del p_average
    return base_risk(p_total, domain_damage)


def f2_consensus(p_total: float, p_average: float, domain_damage: float) -> float:
    values = sorted((float(p_total), float(p_average), float(domain_damage)))
    return values[1]


def f3_mid_veto(p_total: float, p_average: float, domain_damage: float) -> float:
    base = base_risk(p_total, domain_damage)
    return max(base, (base + float(p_average)) / 2.0)


FUSION_FUNCTIONS = {
    "F0_HARD": f0_hard,
    "F1_NO_VETO": f1_no_veto,
    "F2_CONSENSUS": f2_consensus,
    "F3_MID_VETO": f3_mid_veto,
}


def fusion_risk(fusion: str, p_total: float, p_average: float, domain_damage: float) -> float:
    try:
        function = FUSION_FUNCTIONS[str(fusion)]
    except KeyError as exc:
        raise ValueError(f"Unknown fusion: {fusion}") from exc
    return float(function(p_total, p_average, domain_damage))


def component_row(row: Mapping[str, object], p_average_key: str | None = None) -> dict[str, object]:
    """Return a copy annotated with the four frozen components."""
    result = dict(row)
    p_total = _float(row.get("p_total"), "p_total")
    if p_average_key is None:
        p_average_key = next((key for key in ("p_average", "p_A_global", "p_A_variant", "p_A")
                              if row.get(key, "") not in (None, "")), "p_average")
    p_average = _float(row.get(p_average_key), p_average_key)
    damage = _float(row.get("domain_damage"), "domain_damage", 0.0)
    b = base_risk(p_total, damage)
    result.update({
        "p_total": p_total,
        "p_average": p_average,
        "domain_damage": damage,
        "B": b,
        "V": max(p_average - b, 0.0),
        "veto_active": p_average > b,
        "uniquely_p_average_dominated": p_average > p_total and p_average > damage,
        "R_hard": max(b, p_average),
        "R_F0": f0_hard(p_total, p_average, damage),
        "R_F1": f1_no_veto(p_total, p_average, damage),
        "R_F2": f2_consensus(p_total, p_average, damage),
        "R_F3": f3_mid_veto(p_total, p_average, damage),
    })
    return result


def annotate_components(rows: Sequence[Mapping[str, object]], p_average_key: str | None = None) -> list[dict[str, object]]:
    return [component_row(row, p_average_key) for row in rows]


def _risk_key(row: Mapping[str, object], fusion: str) -> tuple[float, float, float, int]:
    if fusion == "F0_HARD":
        return (float(row["R_F0"]), float(row["p_total"]), float(row["p_average"]), _int(row["global_index"]))
    if fusion == "F1_NO_VETO":
        return (float(row["R_F1"]), float(row["p_average"]), float(row["p_total"]), _int(row["global_index"]))
    if fusion == "F2_CONSENSUS":
        return (float(row["R_F2"]), float(row["p_total"]), float(row["p_average"]), _int(row["global_index"]))
    if fusion == "F3_MID_VETO":
        return (float(row["R_F3"]), float(row["p_total"]), float(row["p_average"]), _int(row["global_index"]))
    raise ValueError(f"Unknown fusion: {fusion}")


def rank_fusion(rows: Sequence[Mapping[str, object]], fusion: str) -> list[dict[str, object]]:
    """Exact Python oracle for report boundaries and focused tests only."""
    ranked = annotate_components(rows)
    ranked.sort(key=lambda row: _risk_key(row, fusion))
    for rank, row in enumerate(ranked, 1):
        row["global_rank"] = rank
    return ranked


# Descriptive aliases are useful to downstream diagnostic tests.
compute_base_risk = base_risk
compute_veto_margin = veto_margin
compute_hard_risk = hard_risk
rank_candidates = rank_fusion


# ---------------------------------------------------------------------------
# Descriptive statistics, causal alignment, and pairwise audits
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, object]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else math.nan,
        "median": statistics.median(values) if values else math.nan,
        "q10": _percentile(values, .10), "q25": _percentile(values, .25),
        "q50": _percentile(values, .50), "q75": _percentile(values, .75),
        "q90": _percentile(values, .90),
    }


def _rank_with_ties(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and float(values[order[end + 1]]) == float(values[order[position]]):
            end += 1
        average_rank = (position + end + 2) / 2.0
        for index in range(position, end + 1):
            result[order[index]] = average_rank
        position = end + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx, ry = _rank_with_ties(xs), _rank_with_ties(ys)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in rx) * sum((y - mean_y) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def _stage(row: Mapping[str, object]) -> str:
    if str(row.get("stage", "")).strip():
        return str(row["stage"])
    layer = str(row.get("layer", ""))
    if "layers." in layer:
        return layer.split("layers.", 1)[1].split(".", 1)[0]
    return layer.split(".", 1)[0] if layer else "unknown"


def _first_value(row: Mapping[str, object], keys: Sequence[str], default: object = "") -> object:
    for key in keys:
        if row.get(key, "") not in (None, ""):
            return row[key]
    return default


def _metric(row: Mapping[str, object], metric: str) -> float | None:
    aliases = dict(CAUSAL_METRICS)[metric]
    value = _first_value(row, aliases)
    if value in (None, ""):
        return None
    try:
        return _float(value, metric)
    except ValueError:
        return None


def _normalise_context(value: object) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"TASK028_50", "50", "TASK02850", "EXACT50", "TASK028-50"}:
        return "TASK028_50"
    return "FULL" if text in {"", "FULL", "ALL"} else text


def _normalise_type(row: Mapping[str, object], heads: Mapping[int, Mapping[str, object]], pairs: Mapping[str, Mapping[str, object]]) -> str:
    current = str(row.get("unit_type", "")).strip()
    if current in (TYPE_ATTENTION, TYPE_FFN):
        return current
    intervention = str(_first_value(row, ("intervention", "intervention_type"))).lower()
    if "attention" in intervention:
        return TYPE_ATTENTION
    gid_value = _first_value(row, ("global_index",))
    if gid_value != "":
        try:
            gid = _int(gid_value, "global_index")
            if gid in heads:
                return TYPE_ATTENTION
            pair = pairs.get(str(row.get("pair_id", "")), {})
            if gid == _int(pair.get("ffn_global_index", -1), "ffn_global_index"):
                return TYPE_FFN
        except ValueError:
            pass
    if "ffn" in intervention:
        return TYPE_FFN
    return current or "unknown"


def recover_single_interventions(raw_rows: Sequence[Mapping[str, object]], selected_heads: Sequence[Mapping[str, object]], selected_pairs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    heads = {_int(row.get("global_index"), "global_index"): row for row in selected_heads if row.get("global_index", "") != ""}
    pairs = {str(row.get("pair_id")): row for row in selected_pairs}
    recovered = []
    for source in raw_rows:
        row = dict(source)
        row["context"] = _normalise_context(row.get("context"))
        row["unit_type"] = _normalise_type(row, heads, pairs)
        intervention = str(_first_value(row, ("intervention", "intervention_type"))).strip()
        if intervention not in SINGLE_INTERVENTIONS or row["unit_type"] not in (TYPE_ATTENTION, TYPE_FFN):
            continue
        gid = _first_value(row, ("global_index",))
        if gid in (None, ""):
            pair = pairs.get(str(row.get("pair_id", "")), {})
            gid = pair.get("attention_global_index" if row["unit_type"] == TYPE_ATTENTION else "ffn_global_index", "")
        if gid in (None, ""):
            continue
        row["global_index"] = _int(gid, "global_index")
        recovered.append(row)
    return recovered


def _prepare_candidates(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = [dict(row) for row in rows if str(row.get("feasible", "True")).strip().lower() not in {"false", "0", "no"}]
    if not result:
        return []
    gids = [_int(row.get("global_index"), "global_index") for row in result]
    if any(row.get("p_total", "") in (None, "") for row in result):
        values = [_float(row.get("Delta_total"), "Delta_total") for row in result]
        for row, value in zip(result, _ordinal(values, gids)):
            row["p_total"] = value
    if any(_first_value(row, ("p_average", "p_A_global", "p_A_variant", "p_A")) == "" for row in result):
        values = [_float(row.get("Delta_average"), "Delta_average") for row in result]
        for row, value in zip(result, _ordinal(values, gids)):
            row["p_average"] = value
    for row in result:
        if row.get("p_average", "") == "":
            row["p_average"] = _first_value(row, ("p_A_global", "p_A_variant", "p_A"))
        if row.get("domain_damage", "") in (None, ""):
            row["domain_damage"] = 0.0
    return [component_row(row) for row in result]


def _join_causal_rows(causal_rows: Sequence[Mapping[str, object]], risk_by_gid: Mapping[int, Mapping[str, object]]) -> list[dict[str, object]]:
    joined = []
    for source in causal_rows:
        gid = _int(source.get("global_index"), "global_index")
        risk = risk_by_gid.get(gid)
        if risk is None:
            continue
        row = dict(source)
        row.update({key: risk.get(key, "") for key in ("p_total", "p_average", "domain_damage", "B", "V", "R_hard", "R_F0", "R_F1", "R_F2", "R_F3")})
        joined.append(row)
    return joined


def causal_alignment(joined: Sequence[Mapping[str, object]], risk_names: Sequence[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        context_rows = [row for row in joined if _normalise_context(row.get("context")) == context]
        for group in ("all", TYPE_ATTENTION, TYPE_FFN):
            group_rows = [row for row in context_rows if group == "all" or row.get("unit_type") == group]
            for metric, _ in CAUSAL_METRICS:
                metrics = [(_float(row[name], name), _metric(row, metric)) for row in group_rows for name in risk_names if row.get(name, "") not in (None, "")]
                for name in risk_names:
                    pairs = [(_float(row[name], name), effect) for row in group_rows if row.get(name, "") not in (None, "") for effect in [_metric(row, metric)] if effect is not None]
                    rows.append({"context": context, "group": group, "risk_quantity": name, "metric": metric,
                                 "rho": spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs]), "n": len(pairs)})
    summary = {"contexts": list(CONTEXTS), "groups": ["all", TYPE_ATTENTION, TYPE_FFN], "risk_quantities": list(risk_names),
               "metrics": [name for name, _ in CAUSAL_METRICS], "row_count": len(rows), "alignment_complete": bool(rows)}
    return rows, summary


def _decision(value_a: float, value_f: float) -> int:
    if value_a < value_f:
        return -1
    if value_f < value_a:
        return 1
    return 0


def _pair_records(joined: Sequence[Mapping[str, object]], risk_names: Sequence[str]) -> list[dict[str, object]]:
    by_pair: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in joined:
        by_pair[(_normalise_context(row.get("context")), str(row.get("pair_id", "")))][str(row.get("unit_type"))] = row
    output = []
    for (context, pair_id), members in sorted(by_pair.items()):
        attention, ffn = members.get(TYPE_ATTENTION), members.get(TYPE_FFN)
        if attention is None or ffn is None:
            continue
        record: dict[str, object] = {"context": context, "pair_id": pair_id}
        for typ, prefix, row in ((TYPE_ATTENTION, "attention", attention), (TYPE_FFN, "ffn", ffn)):
            for name in risk_names:
                record[f"{prefix}_{name}"] = row.get(name, "")
            record[f"{prefix}_causal_impact"] = _metric(row, "relative_logit_l2")
            record[f"{prefix}_causal_kl"] = _metric(row, "kl")
            record[f"{prefix}_causal_ce_increase"] = _metric(row, "ce_increase")
            record[f"{prefix}_causal_flip_rate"] = _metric(row, "prediction_flip_rate")
            record[f"{prefix}_global_index"] = row.get("global_index", "")
        output.append(record)
    return output


def _pair_decision_summary(records: Sequence[Mapping[str, object]], risk_name: str) -> dict[str, object]:
    correct = incorrect = ties = 0
    for row in records:
        risk_decision = _decision(_float(row.get(f"attention_{risk_name}"), risk_name), _float(row.get(f"ffn_{risk_name}"), risk_name))
        actual_a, actual_f = row.get("attention_causal_impact"), row.get("ffn_causal_impact")
        if actual_a is None or actual_f is None:
            continue
        actual_decision = _decision(float(actual_a), float(actual_f))
        if risk_decision == 0 or actual_decision == 0:
            ties += 1
        elif risk_decision == actual_decision:
            correct += 1
        else:
            incorrect += 1
    denominator = correct + incorrect
    return {"correct": correct, "incorrect": incorrect, "tie": ties,
            "accuracy": correct / denominator if denominator else math.nan,
            "n": correct + incorrect + ties}


def _bootstrap_pair_accuracy(records: Sequence[Mapping[str, object]], risk_name: str, seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, object]:
    rng = random.Random(seed)
    samples: list[float] = []
    records = list(records)
    for _ in range(replicates):
        sample = [records[rng.randrange(len(records))] for _ in records] if records else []
        summary = _pair_decision_summary(sample, risk_name) if sample else {"accuracy": math.nan}
        if isinstance(summary.get("accuracy"), (int, float)) and math.isfinite(float(summary["accuracy"])):
            samples.append(float(summary["accuracy"]))
    return {"seed": seed, "replicates": replicates, "n_pairs": len(records),
            "accuracy": _pair_decision_summary(records, risk_name).get("accuracy", math.nan),
            "ci_95": [_percentile(samples, .025), _percentile(samples, .975)] if samples else [math.nan, math.nan]}


def pairwise_audit(joined: Sequence[Mapping[str, object]], risk_name: str = "R_hard") -> tuple[list[dict[str, object]], dict[str, object]]:
    risk_names = ("p_total", "p_average", "domain_damage", "B", "V", "R_hard", "R_F0", "R_F1", "R_F2", "R_F3")
    records = _pair_records(
        [component_row(row) if "R_hard" not in row else row for row in joined],
        risk_names,
    )
    output = []
    for row in records:
        risk_decision = _decision(_float(row[f"attention_{risk_name}"], risk_name), _float(row[f"ffn_{risk_name}"], risk_name))
        actual_decision = _decision(float(row["attention_causal_impact"]), float(row["ffn_causal_impact"])) if row["attention_causal_impact"] is not None and row["ffn_causal_impact"] is not None else 0
        enriched = dict(row)
        enriched.update({"risk_quantity": risk_name,
                         "risk_safer_member": "attention" if risk_decision == -1 else "ffn" if risk_decision == 1 else "tie",
                         "actual_lower_causal_member": "attention" if actual_decision == -1 else "ffn" if actual_decision == 1 else "tie",
                         "hard_veto_correct": risk_decision != 0 and actual_decision != 0 and risk_decision == actual_decision,
                         "hard_veto_tie": risk_decision == 0 or actual_decision == 0})
        output.append(enriched)
    accuracy = {}
    for context in CONTEXTS:
        subset = [row for row in output if row["context"] == context]
        point = _pair_decision_summary(subset, risk_name)
        accuracy[context] = {**point, **_bootstrap_pair_accuracy(subset, risk_name)}
    accuracy["risk_quantity"] = risk_name
    accuracy["seed"] = SEED
    accuracy["replicates"] = BOOTSTRAP_REPLICATES
    return output, accuracy


def fusion_pairwise_audit(joined: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_records = _pair_records(
        [component_row(row) if "R_F0" not in row else row for row in joined],
        ("R_F0", "R_F1", "R_F2", "R_F3"),
    )
    output = []
    summaries: dict[str, object] = {}
    for fusion in FUSIONS:
        risk = f"R_{fusion.split('_', 1)[0]}" if fusion != "F0_HARD" else "R_F0"
        # F1/F2/F3 names already match R_F1/R_F2/R_F3.
        records = []
        for row in base_records:
            actual_decision = _decision(float(row["attention_causal_impact"]), float(row["ffn_causal_impact"])) if row["attention_causal_impact"] is not None and row["ffn_causal_impact"] is not None else 0
            risk_decision = _decision(_float(row[f"attention_{risk}"], risk), _float(row[f"ffn_{risk}"], risk))
            record = {"context": row["context"], "pair_id": row["pair_id"], "fusion": fusion,
                      "risk_safer_member": "attention" if risk_decision == -1 else "ffn" if risk_decision == 1 else "tie",
                      "actual_lower_causal_member": "attention" if actual_decision == -1 else "ffn" if actual_decision == 1 else "tie",
                      "correct": risk_decision != 0 and actual_decision != 0 and risk_decision == actual_decision,
                      "tie": risk_decision == 0 or actual_decision == 0,
                      "risk_decision": risk_decision, "actual_decision": actual_decision}
            records.append(record)
            output.append(record)
        summaries[fusion] = {}
        for context in CONTEXTS:
            subset = [row for row in records if row["context"] == context]
            correct = sum(bool(row["correct"]) for row in subset)
            incorrect = sum(not row["correct"] and not row["tie"] for row in subset)
            ties = sum(bool(row["tie"]) for row in subset)
            rng = random.Random(SEED)
            samples = []
            for _ in range(BOOTSTRAP_REPLICATES):
                sample = [subset[rng.randrange(len(subset))] for _ in subset] if subset else []
                denom = sum(not item["tie"] for item in sample)
                samples.append(sum(bool(item["correct"]) for item in sample) / denom if denom else math.nan)
            samples = [value for value in samples if math.isfinite(value)]
            summaries[fusion][context] = {"correct": correct, "incorrect": incorrect, "tie": ties,
                                          "accuracy": correct / (correct + incorrect) if correct + incorrect else math.nan,
                                          "n_pairs": len(subset), "seed": SEED, "replicates": BOOTSTRAP_REPLICATES,
                                          "ci_95": [_percentile(samples, .025), _percentile(samples, .975)] if samples else [math.nan, math.nan]}
    return output, {"fusions": summaries, "seed": SEED, "replicates": BOOTSTRAP_REPLICATES}


# ---------------------------------------------------------------------------
# Snapshot, cost-matched, overlap, and F0 veto-event diagnostics
# ---------------------------------------------------------------------------

def veto_snapshot_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in _prepare_candidates(rows):
        target = _float(row.get("snapshot_target"), "snapshot_target", default=math.nan)
        result = dict(row)
        result.update({"snapshot_target": target, "unit_type": row.get("unit_type", "unknown"), "stage": _stage(row),
                       "veto_active": bool(row["veto_active"]), "uniquely_p_average_dominated": bool(row["uniquely_p_average_dominated"])})
        output.append(result)
    return output


def summarize_veto_snapshots(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    snapshots: dict[str, object] = {}
    for target in SNAPSHOT_TARGETS:
        target_rows = [row for row in rows if abs(_float(row.get("snapshot_target"), "snapshot_target") - target) < 1e-12]
        by_type: dict[str, object] = {}
        for typ in (TYPE_ATTENTION, TYPE_FFN):
            subset = [row for row in target_rows if str(row.get("unit_type")) == typ]
            by_type[typ] = {
                "B": distribution([_float(row["B"], "B") for row in subset]),
                "p_average": distribution([_float(row["p_average"], "p_average") for row in subset]),
                "V": distribution([_float(row["V"], "V") for row in subset]),
                "R_hard": distribution([_float(row["R_hard"], "R_hard") for row in subset]),
                "veto_active_fraction": sum(bool(row["veto_active"]) for row in subset) / len(subset) if subset else math.nan,
                "uniquely_p_average_dominated_fraction": sum(bool(row["uniquely_p_average_dominated"]) for row in subset) / len(subset) if subset else math.nan,
                "spearman_V_log_parameter_cost": spearman(
                    [_float(row["V"], "V") for row in subset if _float(row.get("parameter_cost"), "parameter_cost", default=0.0) > 0],
                    [math.log(_float(row["parameter_cost"], "parameter_cost")) for row in subset if _float(row.get("parameter_cost"), "parameter_cost", default=0.0) > 0],
                ),
                "spearman_V_stage": spearman(
                    [_float(row["V"], "V") for row in subset],
                    [float(_stage(row)) if _stage(row).replace(".", "", 1).isdigit() else float(index) for index, row in enumerate(subset)],
                ),
                "spearman_V_Delta_average": spearman(
                    [_float(row["V"], "V") for row in subset if row.get("Delta_average", "") not in (None, "")],
                    [_float(row["Delta_average"], "Delta_average") for row in subset if row.get("Delta_average", "") not in (None, "")],
                ),
            }
        snapshots[str(target)] = {"snapshot_target": target, "groups": by_type, "count": len(target_rows)}
    return {"snapshot_targets": list(SNAPSHOT_TARGETS), "snapshots": snapshots}


def cost_matched_pack_audit(joined: Sequence[Mapping[str, object]], raw_rows: Sequence[Mapping[str, object]], packs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    singles = {(str(row.get("context")), str(row.get("pair_id", "")), str(row.get("unit_type"))): row for row in joined}
    pack_rows = {(str(row.get("context", "FULL")), str(row.get("pair_id", ""))): row for row in raw_rows
                 if str(_first_value(row, ("intervention", "intervention_type"))) == "ffn_cost_matched_pack"}
    output = []
    for context, pair_id, typ in sorted(singles):
        if typ != TYPE_ATTENTION:
            continue
        attention = singles[(context, pair_id, typ)]
        pack = pack_rows.get((context, pair_id), {})
        attention_damage = _metric(attention, "relative_logit_l2")
        pack_damage = _metric(pack, "relative_logit_l2")
        output.append({"context": context, "pair_id": pair_id, "attention_global_index": attention.get("global_index", ""),
                      "V": attention.get("V", ""), "attention_causal_damage": attention_damage,
                      "ffn_pack_causal_damage": pack_damage, "attention_parameter_cost": attention.get("parameter_cost", ""),
                      "pack_parameter_cost": _first_value(pack, ("removed_parameter_cost", "parameter_cost")),
                      "attention_pack_causal_ratio": attention_damage / pack_damage if attention_damage is not None and pack_damage not in (None, 0) else math.nan})
    return output


def selection_overlap(selections: Mapping[str, Sequence[int]], types: Mapping[int, str] | None = None) -> dict[str, object]:
    output: dict[str, object] = {}
    for left, right in (("F0_HARD", "F1_NO_VETO"), ("F0_HARD", "F2_CONSENSUS"),
                        ("F0_HARD", "F3_MID_VETO"), ("F1_NO_VETO", "F3_MID_VETO"),
                        ("F2_CONSENSUS", "F3_MID_VETO")):
        a, b = set(int(value) for value in selections.get(left, ())), set(int(value) for value in selections.get(right, ()))
        common, union = a & b, a | b
        output[f"{left}_vs_{right}"] = {
            "intersection_count": len(common), "union_count": len(union),
            "jaccard": len(common) / max(1, len(union)),
            "longest_common_exact_prefix": longest_common_prefix(selections.get(left, ()), selections.get(right, ())),
            "attention_intersection": sum(types.get(gid) == TYPE_ATTENTION for gid in common) if types else None,
            "ffn_intersection": sum(types.get(gid) == TYPE_FFN for gid in common) if types else None,
        }
    return output


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def _cpu_veto_event_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ranked_base = sorted(rows, key=lambda row: (float(row["B"]), float(row["p_average"]), float(row["p_total"]), _int(row["global_index"])))
    ranked_hard = sorted(rows, key=lambda row: _risk_key(row, "F0_HARD"))
    if not ranked_base or not ranked_hard or int(ranked_base[0]["global_index"]) == int(ranked_hard[0]["global_index"]):
        return []
    hard, base = ranked_hard[0], ranked_base[0]
    return [{"step": "", "sparsity": "", "hard_selected_global_index": hard["global_index"],
             "base_selected_global_index": base["global_index"], "hard_selected_type": hard.get("unit_type", ""),
             "base_selected_type": base.get("unit_type", ""), "hard_selected_B": hard["B"],
             "base_selected_B": base["B"], "hard_selected_p_average": hard["p_average"],
             "base_selected_p_average": base["p_average"], "hard_selected_V": hard["V"],
             "base_selected_V": base["V"], "hard_selected_R_hard": hard["R_hard"],
             "base_selected_R_hard": base["R_hard"], "hard_selected_parameter_cost": hard.get("parameter_cost", ""),
             "base_selected_parameter_cost": base.get("parameter_cost", ""),
             "hard_selected_domain_id": hard.get("domain_id", ""), "base_selected_domain_id": base.get("domain_id", ""),
             "hard_selected_stage": _stage(hard), "base_selected_stage": _stage(base)}]


# ---------------------------------------------------------------------------
# Exact optimized selector replay
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _gpu_multi_key_order(keys: Sequence[object]):
    """Stable lexicographic tensor order without materialising Python rows."""
    if not keys:
        raise ValueError("at least one tensor key is required")
    torch = _torch()
    count = int(keys[0].numel())
    order = torch.arange(count, dtype=torch.long, device=keys[0].device)
    for key in reversed(keys):
        current = key.index_select(0, order)
        rough = torch.argsort(current)
        sorted_values = current.index_select(0, rough)
        starts = torch.ones(count, dtype=torch.bool, device=current.device)
        if count > 1:
            starts[1:] = sorted_values[1:] != sorted_values[:-1]
        group_id = torch.cumsum(starts.to(torch.int64), dim=0) - 1
        prior_position = torch.arange(count, dtype=torch.long, device=current.device)
        composite = group_id * (count + 1) + prior_position.index_select(0, rough).to(torch.int64)
        order = order.index_select(0, rough.index_select(0, torch.argsort(composite)))
    return order


def _exact_global_components(tensor_candidates: Mapping[str, object], optimized: object, task031: object, base_ranked: Mapping[str, object]):
    torch = getattr(optimized, "torch", None) or _torch()
    positions = tensor_candidates["_positions"]
    delta_total = tensor_candidates.get("Delta_total")
    delta_average = tensor_candidates.get("Delta_average")
    if delta_total is None or delta_average is None:
        p_total = base_ranked["p_total"].to(dtype=torch.float64)
        p_average = base_ranked.get("p_average", base_ranked.get("p_A_variant")).to(dtype=torch.float64)
    else:
        if hasattr(optimized, "global_index_order"):
            p_total = task031._cached_gpu_ordinal_rank(delta_total, positions, optimized.global_index_order)
            p_average = task031._cached_gpu_ordinal_rank(delta_average, positions, optimized.global_index_order)
        else:
            p_total = task031.gpu_ordinal_rank(delta_total, tensor_candidates["global_index"])
            p_average = task031.gpu_ordinal_rank(delta_average, tensor_candidates["global_index"])
    damage = tensor_candidates.get("domain_damage")
    if damage is None:
        damage = torch.zeros_like(p_total)
    damage_view = damage.to(dtype=torch.float64)
    return p_total, p_average, damage, damage_view


def _materialize_component_rows(candidates: Mapping[str, object], p_total: object,
                                p_average: object, damage: object,
                                lookup: Mapping[int, Mapping[str, object]]) -> list[dict[str, object]]:
    """Materialize snapshot rows using the already-computed exact tensors.

    This helper is intentionally a boundary operation.  It preserves the
    candidate tensor order, copies structural metadata from the candidate or
    immutable replay lookup, and attaches the current Task033 component views.
    In particular, it never calls a Python percentile/reranker to recreate
    ``p_total`` or global ``p_average``.
    """
    metadata_fields = (
        "global_index", "domain_id", "local_index", "Delta_average",
        "Delta_total", "parameter_cost", "unit_type", "layer", "stage",
        "unit_index", "feasible", "domain_coverage",
    )
    count = int(candidates["global_index"].numel())

    def scalar(value: object) -> object:
        return value.item() if hasattr(value, "item") else value

    rows: list[dict[str, object]] = []
    for index in range(count):
        gid = _int(scalar(candidates["global_index"][index]), "global_index")
        row = dict(lookup.get(gid, {}))
        for name in metadata_fields:
            if name in candidates:
                row[name] = scalar(candidates[name][index])
        row["global_index"] = gid
        row["p_total"] = scalar(p_total[index])
        row["p_average"] = scalar(p_average[index])
        row["domain_damage"] = scalar(damage[index])
        row["feasible"] = row.get("feasible", True)
        rows.append(row)
    return rows


def _gpu_fusion_rank(tensor_candidates: Mapping[str, object], p_total, p_average, damage, damage_view, fusion: str) -> dict[str, object]:
    torch = getattr(p_total, "new_empty", None)
    del torch
    base = __import__("torch")
    b = base.maximum(p_total, damage_view)
    if fusion == "F0_HARD":
        risk = base.maximum(b, p_average)
        keys = (risk, p_total, p_average, tensor_candidates["global_index"].to(dtype=base.int64))
    elif fusion == "F1_NO_VETO":
        risk = b
        keys = (risk, p_average, p_total, tensor_candidates["global_index"].to(dtype=base.int64))
    elif fusion == "F2_CONSENSUS":
        risk = base.sort(base.stack((p_total, p_average, damage_view), dim=1), dim=1).values[:, 1]
        keys = (risk, p_total, p_average, tensor_candidates["global_index"].to(dtype=base.int64))
    elif fusion == "F3_MID_VETO":
        risk = base.maximum(b, (b + p_average) / 2.0)
        keys = (risk, p_total, p_average, tensor_candidates["global_index"].to(dtype=base.int64))
    else:
        raise ValueError(f"Unknown fusion: {fusion}")
    order = _gpu_multi_key_order(keys)
    result = dict(tensor_candidates)
    result.update({"p_total": p_total, "p_average": p_average, "p_A_variant": p_average,
                   "domain_damage": damage, "B": b, "V": base.maximum(p_average - b, base.zeros_like(b)),
                   "R_F0": base.maximum(b, p_average), "R_F1": b,
                   "R_F2": base.sort(base.stack((p_total, p_average, damage_view), dim=1), dim=1).values[:, 1],
                   "R_F3": base.maximum(b, (b + p_average) / 2.0),
                   "R_hard": base.maximum(b, p_average), "R_dual": base.maximum(p_total, p_average),
                   "selected_position": int(order[0].item())})
    return result


def _removed_cost(engine: object) -> float:
    value = getattr(engine, "removed_cost", None)
    if value is None:
        value = getattr(getattr(engine, "base", None), "removed_cost", 0.0)
    return float(value)


def _engine_for_replay(*, task014_root: Path, task016_root: Path, task017_root: Path, output_dir: Path, device: str, ranking_backend: str):
    # The import is intentionally local: CPU diagnostics do not load the
    # selector/runtime, while replay uses the frozen Task031/Task028 backend.
    task031 = __import__("task031_domain_conditioned_average_diagnosis")
    return task031._engine_for_replay(task014_root=task014_root, task016_root=task016_root,
                                      task017_root=task017_root, output_dir=output_dir,
                                      device=device, ranking_backend=ranking_backend)


def replay_fusion(*, fusion: str, engine: object, optimized: object, total_parameters: float,
                  target_budget: float, lookup: Mapping[int, Mapping[str, object]] | None = None,
                  task028_sequence: Sequence[int] = ()) -> dict[str, object]:
    task031 = __import__("task031_domain_conditioned_average_diagnosis")
    lookup = lookup or {}
    selected: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    captured: set[float] = set()
    while _removed_cost(engine) < float(target_budget) - 1e-12:
        candidates = optimized.candidate_tensors()
        positions = candidates.get("_positions")
        if positions is None or int(positions.numel()) == 0:
            raise RuntimeError(f"{fusion} exhausted candidates before target")
        sparsity = _removed_cost(engine) / max(float(total_parameters), 1e-12)
        base_ranked = optimized.rank(candidates)
        p_total, p_average, damage, damage_view = _exact_global_components(candidates, optimized, task031, base_ranked)
        if fusion == "F0_HARD":
            ranked = base_ranked
            selected_position = int(ranked["selected_position"])
            # Audit uses the exact current components but does not alter F0's
            # authoritative Task028 selection.
            audit_rank = _gpu_fusion_rank(candidates, p_total, p_average, damage, damage_view, "F0_HARD")
        else:
            ranked = _gpu_fusion_rank(candidates, p_total, p_average, damage, damage_view, fusion)
            selected_position = int(ranked["selected_position"])
            audit_rank = ranked if fusion == "F0_HARD" else _gpu_fusion_rank(candidates, p_total, p_average, damage, damage_view, "F0_HARD")
        # Snapshot CPU materialisation is explicitly outside the hot path.
        for target in SNAPSHOT_TARGETS:
            if target not in captured and sparsity >= target - 1e-12:
                rows = _materialize_component_rows(candidates, p_total, p_average, damage, lookup)
                ranked_rows = rank_fusion(rows, fusion)
                snapshots.append({"fusion": fusion, "snapshot_target": target,
                                  "actual_effective_sparsity": sparsity,
                                  **_snapshot_summary(ranked_rows, selected, sparsity, total_parameters)})
                captured.add(target)
        if fusion == "F0_HARD":
            base_rank = _gpu_fusion_rank(candidates, p_total, p_average, damage, damage_view, "F1_NO_VETO")
            base_position = int(base_rank["selected_position"])
            hard_gid = _int(candidates["global_index"][selected_position].item(), "global_index")
            base_gid = _int(candidates["global_index"][base_position].item(), "global_index")
            if hard_gid != base_gid:
                hard_row = _tensor_candidate_row(candidates, audit_rank, selected_position, lookup)
                base_row = _tensor_candidate_row(candidates, audit_rank, base_position, lookup)
                event_rows.append({"step": len(selected) + 1, "sparsity": sparsity,
                                   "hard_selected_global_index": hard_gid, "base_selected_global_index": base_gid,
                                   "hard_selected_type": hard_row.get("unit_type", ""), "base_selected_type": base_row.get("unit_type", ""),
                                   "hard_selected_B": hard_row.get("B", ""), "base_selected_B": base_row.get("B", ""),
                                   "hard_selected_p_average": hard_row.get("p_average", ""), "base_selected_p_average": base_row.get("p_average", ""),
                                   "hard_selected_V": hard_row.get("V", ""), "base_selected_V": base_row.get("V", ""),
                                   "hard_selected_R_hard": hard_row.get("R_hard", ""), "base_selected_R_hard": base_row.get("R_hard", ""),
                                   "hard_selected_parameter_cost": hard_row.get("parameter_cost", lookup.get(hard_gid, {}).get("parameter_cost", "")),
                                   "base_selected_parameter_cost": base_row.get("parameter_cost", lookup.get(base_gid, {}).get("parameter_cost", "")),
                                   "hard_selected_domain_id": hard_row.get("domain_id", ""), "base_selected_domain_id": base_row.get("domain_id", ""),
                                   "hard_selected_stage": _stage(hard_row), "base_selected_stage": _stage(base_row)})
        chosen = _tensor_candidate_row(candidates, ranked, selected_position, lookup)
        remove_ranked = dict(base_ranked)
        remove_ranked["selected_position"] = selected_position
        removed = optimized.remove(remove_ranked, selected_position)
        row = {**chosen, **(dict(removed) if isinstance(removed, Mapping) else {})}
        row.update({"fusion": fusion, "step": len(selected) + 1, "global_index": chosen["global_index"],
                    "effective_sparsity_after": _removed_cost(engine) / max(float(total_parameters), 1e-12),
                    "cumulative_removed_parameters": _removed_cost(engine)})
        selected.append(row)
    final_sparsity = _removed_cost(engine) / max(float(total_parameters), 1e-12)
    if .5 not in captured:
        candidates = optimized.candidate_tensors()
        p_total, p_average, damage, _ = _exact_global_components(
            candidates, optimized, task031, optimized.rank(candidates),
        )
        rows = rank_fusion(_materialize_component_rows(candidates, p_total, p_average, damage, lookup), fusion)
        snapshots.append({"fusion": fusion, "snapshot_target": .5, "actual_effective_sparsity": final_sparsity,
                          **_snapshot_summary(rows, selected, final_sparsity, total_parameters)})
    return {"fusion": fusion, "sequence": [_int(row["global_index"], "global_index") for row in selected],
            "selected": selected, "snapshots": snapshots, "event_rows": event_rows,
            "final_sparsity": final_sparsity, "snapshots_complete": captured | {.5} == set(SNAPSHOT_TARGETS),
            "task028_sequence": list(task028_sequence)}


def benchmark_fusion(*, fusion: str, task014_root: Path, task016_root: Path,
                     task017_root: Path, output_dir: Path, device: str = "cuda:0",
                     ranking_backend: str = "single-gpu", steps: int = BENCHMARK_STEPS) -> dict[str, object]:
    """Measure the tensor-only selector hot path before the full replay."""
    task031 = __import__("task031_domain_conditioned_average_diagnosis")
    engine, optimized = _engine_for_replay(
        task014_root=Path(task014_root), task016_root=Path(task016_root),
        task017_root=Path(task017_root), output_dir=Path(output_dir) / "_benchmark" / fusion,
        device=device, ranking_backend=ranking_backend,
    )
    started = time.perf_counter()
    rank_seconds = 0.0
    remove_seconds = 0.0
    for _ in range(int(steps)):
        candidates = optimized.candidate_tensors()
        rank_started = time.perf_counter()
        base_ranked = optimized.rank(candidates)
        p_total, p_average, damage, damage_view = _exact_global_components(candidates, optimized, task031, base_ranked)
        ranked = base_ranked if fusion == "F0_HARD" else _gpu_fusion_rank(candidates, p_total, p_average, damage, damage_view, fusion)
        rank_seconds += time.perf_counter() - rank_started
        position = int(ranked["selected_position"])
        remove_started = time.perf_counter()
        remove_ranked = dict(base_ranked)
        remove_ranked["selected_position"] = position
        optimized.remove(remove_ranked, position)
        remove_seconds += time.perf_counter() - remove_started
    torch = getattr(engine, "torch", None)
    if torch is not None and str(getattr(engine, "device", "")).startswith("cuda"):
        torch.cuda.synchronize(getattr(engine, "device", None))
    elapsed = time.perf_counter() - started
    rate = int(steps) / max(elapsed, 1e-12)
    return {"fusion": fusion, "steps": int(steps), "seconds_total": elapsed,
            "steps_per_second": rate, "ranking_seconds": rank_seconds,
            "remove_seconds": remove_seconds, "hot_path_candidate_rows": False,
            "hot_path_python_sorted": False,
            "status": "PASS" if rate >= BENCHMARK_MIN_STEPS_PER_SECOND[fusion] else "FAIL"}


def benchmark_gate(report: Mapping[str, object]) -> bool:
    variants = report.get("fusions", report)
    if not isinstance(variants, Mapping):
        return False
    for fusion, minimum in BENCHMARK_MIN_STEPS_PER_SECOND.items():
        row = variants.get(fusion)
        if not isinstance(row, Mapping) or row.get("status") != "PASS" or float(row.get("steps_per_second", 0.0)) < minimum:
            return False
    return report.get("status", "PASS") in ("PASS", True)


def run_benchmark(*, task014_root: Path, task016_root: Path, task017_root: Path,
                  output_dir: Path, device: str = "cuda:0",
                  ranking_backend: str = "single-gpu", steps: int = BENCHMARK_STEPS) -> dict[str, object]:
    started = time.perf_counter()
    print(f"Task033 benchmark: start, steps={int(steps)}", flush=True)
    rows = {fusion: benchmark_fusion(
        fusion=fusion, task014_root=task014_root, task016_root=task016_root,
        task017_root=task017_root, output_dir=Path(output_dir), device=device,
        ranking_backend=ranking_backend, steps=steps,
    ) for fusion in FUSIONS}
    report = {"status": "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL",
              "steps": int(steps), "fusions": rows,
              "minimum_steps_per_second": BENCHMARK_MIN_STEPS_PER_SECOND}
    atomic_json(Path(output_dir) / "task033_performance_benchmark.json", report)
    print(f"Task033 benchmark: complete, seconds={time.perf_counter() - started:.3f}", flush=True)
    return report


def _tensor_candidate_row(candidates: Mapping[str, object], ranked: Mapping[str, object], position: int, lookup: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    gid = _int(candidates["global_index"][position].item(), "global_index")
    row = dict(lookup.get(gid, {}))
    for name in ("global_index", "domain_id", "local_index", "Delta_average", "Delta_total", "domain_damage"):
        if name in candidates:
            value = candidates[name][position]
            row[name] = value.item() if hasattr(value, "item") else value
    row["global_index"] = gid
    for name in ("p_total", "p_average", "p_A_variant", "B", "V", "R_hard", "R_F0", "R_F1", "R_F2", "R_F3", "R_dual"):
        if name in ranked:
            value = ranked[name][position]
            row[name] = value.item() if hasattr(value, "item") else value
    row["feasible"] = True
    return row


def _snapshot_summary(ranked: Sequence[Mapping[str, object]], selected: Sequence[Mapping[str, object]], sparsity: float, total_parameters: float) -> dict[str, object]:
    attention = [row for row in ranked if str(row.get("unit_type")) == TYPE_ATTENTION]
    ffns = [row for row in ranked if str(row.get("unit_type")) == TYPE_FFN]
    first = next((row for row in selected if row.get("unit_type") == TYPE_ATTENTION), None)
    best = attention[0] if attention else {}
    removed = [row for row in selected if float(row.get("effective_sparsity_after", 0.0)) <= sparsity + 1e-12]
    return {"attention_removed": sum(row.get("unit_type") == TYPE_ATTENTION for row in removed),
            "ffn_removed": sum(row.get("unit_type") == TYPE_FFN for row in removed),
            "attention_parameter_contribution": sum(_float(row.get("parameter_cost"), "parameter_cost", default=0.0) for row in removed if row.get("unit_type") == TYPE_ATTENTION) / max(float(total_parameters), 1e-12),
            "ffn_parameter_contribution": sum(_float(row.get("parameter_cost"), "parameter_cost", default=0.0) for row in removed if row.get("unit_type") == TYPE_FFN) / max(float(total_parameters), 1e-12),
            "remaining_attention": len(attention), "remaining_ffn": len(ffns),
            "first_attention_sparsity": first.get("effective_sparsity_after") if first else None,
            "best_attention_rank": best.get("global_rank", 1) if best else None,
            "ffn_ahead": sum(row.get("unit_type") == TYPE_FFN for row in ranked[:ranked.index(best)]) if best else None,
            "veto_active_fraction": sum(float(row.get("V", 0.0)) > 0.0 for row in ranked) / len(ranked) if ranked else math.nan,
            "dominant_component_counts": {name: sum(float(row.get(name, 0.0)) >= max(float(row.get("p_total", 0.0)), float(row.get("p_average", 0.0)), float(row.get("domain_damage", 0.0))) for row in ranked) for name in ("p_total", "p_average", "domain_damage")},
            "stage_removals": dict(Counter(_stage(row) for row in removed)),
            "domain_removals": dict(Counter(str(row.get("domain_id", "")) for row in removed))}


# ---------------------------------------------------------------------------
# Files, figures, summary, and completion gate
# ---------------------------------------------------------------------------

def _optional_float(value: object, label: str = "value") -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fusion_risk_name(fusion: str) -> str:
    return "R_F0" if fusion == "F0_HARD" else f"R_{fusion.split('_', 1)[0]}"


def _stage_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "stage_unknown"
    if text.startswith("stage_"):
        return text
    if text.lstrip("-").isdigit():
        return f"stage_{int(text)}"
    return f"stage_{text}"


def _mapping_value(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _prepare_figure_data(*, veto_rows: Sequence[Mapping[str, object]], causal_rows: Sequence[Mapping[str, object]],
                         fusion_rows: Sequence[Mapping[str, object]], snapshots: Sequence[Mapping[str, object]],
                         events: Sequence[Mapping[str, object]], overlaps: Mapping[str, object],
                         pack_rows: Sequence[Mapping[str, object]], pair_accuracy: Mapping[str, object],
                         stage_rows: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    """Build compact figure inputs with one pass over the large tables."""
    veto_values: dict[tuple[float, str], list[float]] = defaultdict(list)
    cost_points: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in veto_rows:
        target = _optional_float(row.get("snapshot_target"), "snapshot_target")
        value = _optional_float(row.get("V"), "V")
        typ = str(row.get("unit_type", "unknown"))
        if target is not None and value is not None:
            veto_values[(target, typ)].append(value)
            cost = _optional_float(row.get("parameter_cost"), "parameter_cost")
            if cost is not None and cost > 0:
                cost_points[typ]["x"].append(math.log(cost))
                cost_points[typ]["y"].append(value)
    veto_summary = []
    for target in SNAPSHOT_TARGETS:
        for typ in (TYPE_ATTENTION, TYPE_FFN):
            values = veto_values.get((target, typ), [])
            stats = distribution(values)
            veto_summary.append({"snapshot_target": target, "unit_type": typ,
                                 "median": stats["median"], "q25": stats["q25"],
                                 "q75": stats["q75"], "count": stats["count"]})

    causal_points: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    fusion_points: dict[str, dict[str, list[float]]] = {fusion: {"x": [], "y": []} for fusion in FUSIONS}
    for row in causal_rows:
        impact = _metric(row, "relative_logit_l2")
        margin = _optional_float(row.get("V"), "V")
        typ = str(row.get("unit_type", "unknown"))
        if impact is not None and margin is not None:
            causal_points[typ]["x"].append(margin)
            causal_points[typ]["y"].append(impact)
        if impact is None:
            continue
        for fusion in FUSIONS:
            risk = _optional_float(row.get(_fusion_risk_name(fusion)), _fusion_risk_name(fusion))
            if risk is not None:
                fusion_points[fusion]["x"].append(risk)
                fusion_points[fusion]["y"].append(impact)

    event_points = {"x": [], "y": []}
    for index, row in enumerate(events, 1):
        sparsity = _optional_float(row.get("sparsity"), "sparsity")
        if sparsity is not None:
            event_points["x"].append(sparsity)
            event_points["y"].append(float(index))
    pack_points = {"x": [], "y": []}
    for row in pack_rows:
        margin = _optional_float(row.get("V"), "V")
        ratio = _optional_float(row.get("attention_pack_causal_ratio"), "ratio")
        if margin is not None and ratio is not None:
            pack_points["x"].append(margin)
            pack_points["y"].append(ratio)
    return {"veto_summary": veto_summary, "cost_points": dict(cost_points),
            "causal_points": dict(causal_points), "fusion_points": fusion_points,
            "event_points": event_points, "pack_points": pack_points,
            "snapshots": snapshots, "fusion_rows": fusion_rows, "events": events,
            "overlaps": overlaps, "pair_accuracy": pair_accuracy,
            "stage_rows": stage_rows}


def _write_figures(output: Path, veto_rows: Sequence[Mapping[str, object]], causal_rows: Sequence[Mapping[str, object]], fusion_rows: Sequence[Mapping[str, object]], snapshots: Sequence[Mapping[str, object]], events: Sequence[Mapping[str, object]], overlaps: Mapping[str, object], pack_rows: Sequence[Mapping[str, object]], pair_accuracy: Mapping[str, object], stage_rows: Sequence[Mapping[str, object]] = ()) -> list[str]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    prepared = _prepare_figure_data(
        veto_rows=veto_rows, causal_rows=causal_rows, fusion_rows=fusion_rows,
        snapshots=snapshots, events=events, overlaps=overlaps,
        pack_rows=pack_rows, pair_accuracy=pair_accuracy, stage_rows=stage_rows,
    )

    def save(number: int, title: str, xlabel: str, ylabel: str, draw) -> str:
        started = time.perf_counter()
        print(f"Task033 Figure{number:02d}: start", flush=True)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        draw(ax)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
        path = figures / f"Figure{number:02d}.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        print(f"Task033 Figure{number:02d}: complete, seconds={time.perf_counter() - started:.3f}", flush=True)
        return str(path)

    paths = [
        save(1, "Veto margin by type and snapshot", "snapshot target", "V", lambda ax: _plot_veto_summary(ax, prepared["veto_summary"])),
        save(2, "Veto margin versus log(parameter cost)", "log(parameter cost)", "V", lambda ax: _plot_cost(ax, prepared["cost_points"])),
        save(3, "Veto margin versus causal relative-logit-L2", "V", "relative-logit-L2", lambda ax: _plot_causal(ax, prepared["causal_points"])),
        save(4, "Fusion risk versus causal relative-logit-L2", "fusion risk", "relative-logit-L2", lambda ax: _plot_fusion_risk(ax, prepared["fusion_points"])),
        save(5, "Fusion pairwise causal accuracy", "fusion / context", "accuracy", lambda ax: _plot_pair_accuracy(ax, pair_accuracy)),
        save(6, "Best Attention rank versus sparsity", "sparsity", "best Attention rank", lambda ax: _plot_snapshots(ax, snapshots, "best_attention_rank")),
        save(7, "Attention removals versus sparsity", "sparsity", "removed Attention", lambda ax: _plot_snapshots(ax, snapshots, "attention_removed")),
        save(8, "Stage-wise FFN removals at exact 50%", "stage", "removed FFN", lambda ax: _plot_stage(ax, snapshots, stage_rows)),
        save(9, "F0 veto-induced decision changes", "sparsity", "event count", lambda ax: _plot_event_points(ax, prepared["event_points"])),
        save(10, "F0/F1/F2/F3 exact-50 selection overlap", "comparison", "Jaccard", lambda ax: _plot_overlap(ax, overlaps)),
        save(11, "Attention veto margin versus pack causal ratio", "V", "Attention / pack causal ratio", lambda ax: _plot_pack_points(ax, prepared["pack_points"])),
        save(12, "Causal fidelity versus selection-change tradeoff", "selection change from F0", "causal alignment rho", lambda ax: _plot_tradeoff(ax, fusion_rows, overlaps)),
    ]
    return paths


def _plot_veto_summary(ax, rows: Sequence[Mapping[str, object]]) -> None:
    colors = {TYPE_ATTENTION: "C0", TYPE_FFN: "C1"}
    for typ in (TYPE_ATTENTION, TYPE_FFN):
        subset = [row for row in rows if row.get("unit_type") == typ and row.get("count", 0)]
        if not subset:
            continue
        x = [float(row["snapshot_target"]) for row in subset]
        median = [float(row["median"]) for row in subset]
        q25 = [float(row["q25"]) for row in subset]
        q75 = [float(row["q75"]) for row in subset]
        ax.plot(x, median, marker="o", color=colors[typ], label=typ)
        ax.fill_between(x, q25, q75, color=colors[typ], alpha=.2)
    ax.legend(fontsize=7)


def _plot_cost(ax, points: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
    for typ in (TYPE_ATTENTION, TYPE_FFN):
        point = points.get(typ, {})
        if point.get("x"):
            ax.scatter(point["x"], point["y"], s=5, alpha=.25, label=typ)
    ax.legend(fontsize=7)


def _plot_causal(ax, points: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
    for typ in (TYPE_ATTENTION, TYPE_FFN):
        point = points.get(typ, {})
        if point.get("x"):
            ax.scatter(point["x"], point["y"], s=14, alpha=.65, label=typ)
    ax.legend(fontsize=7)


def _plot_fusion_risk(ax, points: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
    for fusion in FUSIONS:
        point = points.get(fusion, {})
        if point.get("x"):
            ax.scatter(point["x"], point["y"], s=12, alpha=.55, label=fusion)
    ax.legend(fontsize=7)


def _plot_pair_accuracy(ax, payload: Mapping[str, object]) -> None:
    fusions = payload.get("fusions", {})
    offsets = {"FULL": -.18, "TASK028_50": .18}
    colors = {"FULL": "C0", "TASK028_50": "C1"}
    for context in CONTEXTS:
        xs, values, errors = [], [], []
        for index, fusion in enumerate(FUSIONS):
            item = fusions.get(fusion, {}).get(context, {}) if isinstance(fusions, Mapping) and isinstance(fusions.get(fusion, {}), Mapping) else {}
            value = _optional_float(item.get("accuracy") if isinstance(item, Mapping) else None, "accuracy")
            if value is not None:
                xs.append(index + offsets.get(context, 0.0))
                values.append(value)
                interval = item.get("ci_95") if isinstance(item, Mapping) else None
                if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes)) and len(interval) == 2:
                    lower = _optional_float(interval[0], "ci_95")
                    upper = _optional_float(interval[1], "ci_95")
                    errors.append([max(0.0, value - lower), max(0.0, upper - value)]
                                  if lower is not None and upper is not None else [0.0, 0.0])
                else:
                    errors.append([0.0, 0.0])
        if xs:
            yerr = list(zip(*errors)) if any(any(error) for error in errors) else None
            ax.bar(xs, values, width=.32, label=context, color=colors.get(context), yerr=yerr, capsize=2)
    ax.set_xticks(range(len(FUSIONS)), FUSIONS, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.legend(fontsize=7)


def _plot_snapshots(ax, rows: Sequence[Mapping[str, object]], field: str) -> None:
    for fusion in FUSIONS:
        subset = []
        for row in rows:
            if row.get("fusion") != fusion:
                continue
            target = _optional_float(row.get("snapshot_target"), "snapshot_target")
            value = _optional_float(row.get(field), field)
            if target is not None and value is not None:
                subset.append((target, value))
        if subset:
            subset.sort()
            ax.plot([item[0] for item in subset], [item[1] for item in subset], marker="o", label=fusion)
    ax.legend(fontsize=7)


def _plot_stage(ax, snapshots: Sequence[Mapping[str, object]], stage_rows: Sequence[Mapping[str, object]] = ()) -> None:
    counts: dict[tuple[str, str], int] = Counter()
    source_rows = stage_rows
    if source_rows:
        for row in source_rows:
            target = _optional_float(row.get("snapshot_target"), "snapshot_target")
            count = _optional_float(row.get("removed_count"), "removed_count")
            if row.get("fusion") in FUSIONS and target is not None and abs(target - .5) < 1e-12 and count is not None:
                counts[(str(row["fusion"]), _stage_label(row.get("stage")))] += int(count)
    else:
        for row in snapshots:
            target = _optional_float(row.get("snapshot_target"), "snapshot_target")
            if row.get("fusion") not in FUSIONS or target is None or abs(target - .5) >= 1e-12:
                continue
            for stage, count in _mapping_value(row.get("stage_removals", {})).items():
                value = _optional_float(count, "removed_count")
                if value is not None:
                    counts[(str(row["fusion"]), _stage_label(stage))] += int(value)
    stages = sorted({stage for _, stage in counts}, key=lambda value: (value != "stage_unknown", value))
    if not stages:
        return
    width = .8 / len(FUSIONS)
    for index, fusion in enumerate(FUSIONS):
        values = [counts.get((fusion, stage), 0) for stage in stages]
        ax.bar([position + index * width for position in range(len(stages))], values, width=width, label=fusion)
    ax.set_xticks([position + width * (len(FUSIONS) - 1) / 2 for position in range(len(stages))], stages)
    ax.legend(fontsize=7)


def _plot_event_points(ax, points: Mapping[str, Sequence[float]]) -> None:
    if points.get("x"):
        ax.scatter(points["x"], points["y"], s=12)


def _plot_overlap(ax, overlaps: Mapping[str, object]) -> None:
    labels, values = [], []
    for key, value in overlaps.items():
        if not isinstance(value, Mapping):
            continue
        jaccard = _optional_float(value.get("jaccard"), "jaccard")
        if jaccard is not None:
            labels.append(key.replace("_vs_", "\nvs\n")); values.append(jaccard)
    if values:
        ax.bar(range(len(values)), values); ax.set_xticks(range(len(values)), labels, rotation=25, ha="right"); ax.set_ylim(0, 1)


def _plot_pack_points(ax, points: Mapping[str, Sequence[float]]) -> None:
    if points.get("x"):
        ax.scatter(points["x"], points["y"], s=14)


def _plot_tradeoff(ax, fusion_rows: Sequence[Mapping[str, object]], overlaps: Mapping[str, object]) -> None:
    alignment: dict[tuple[str, str], float] = {}
    for row in fusion_rows:
        if row.get("group") != "all" or row.get("metric") != "relative_logit_l2":
            continue
        context = _normalise_context(row.get("context"))
        value = _optional_float(row.get("rho"), "rho")
        if value is not None:
            alignment[(context, str(row.get("risk_quantity")))] = value
    for context in CONTEXTS:
        xs, ys, labels = [], [], []
        for fusion in FUSIONS:
            risk_name = _fusion_risk_name(fusion)
            if fusion == "F0_HARD":
                change = 0.0
            else:
                comparison = overlaps.get(f"F0_HARD_vs_{fusion}", {})
                jaccard = _optional_float(comparison.get("jaccard") if isinstance(comparison, Mapping) else None, "jaccard")
                if jaccard is None:
                    continue
                change = 1.0 - jaccard
            rho = alignment.get((context, risk_name))
            if rho is None:
                continue
            xs.append(change); ys.append(rho); labels.append(fusion)
        if xs:
            ax.scatter(xs, ys, s=28, label=context)
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.legend(fontsize=7)


def _fmt(value: object) -> str:
    try:
        number = float(value)
        return "NA" if not math.isfinite(number) else f"{number:.6g}"
    except (TypeError, ValueError):
        return "NA"


def _mapping_path(payload: Mapping[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _veto_group_value(veto_summary: Mapping[str, object], snapshot_rows: Sequence[Mapping[str, object]], target: float, typ: str, metric: str, field: str | None = None) -> object:
    target_keys = (str(target), f"{target:g}")
    group: object = None
    for key in target_keys:
        group = _mapping_path(veto_summary, "snapshots", key, "groups", typ)
        if isinstance(group, Mapping):
            break
    if isinstance(group, Mapping):
        value = group.get(metric)
        if field is not None and isinstance(value, Mapping):
            value = value.get(field)
        if _optional_float(value, metric) is not None:
            return value
    values = []
    for row in snapshot_rows:
        row_target = _optional_float(row.get("snapshot_target"), "snapshot_target")
        if row_target is None or abs(row_target - target) >= 1e-12 or str(row.get("unit_type")) != typ:
            continue
        value = row.get(metric)
        if field is not None and isinstance(value, Mapping):
            value = value.get(field)
        number = _optional_float(value, metric)
        if number is not None:
            values.append(number)
    if metric == "veto_active_fraction":
        active = [bool(row.get("veto_active")) for row in snapshot_rows
                  if (_optional_float(row.get("snapshot_target"), "snapshot_target") is not None
                      and abs(float(row["snapshot_target"]) - target) < 1e-12
                      and str(row.get("unit_type")) == typ)]
        return sum(active) / len(active) if active else math.nan
    if metric == "V" and field is not None:
        return distribution(values).get(field, math.nan)
    return math.nan


def _alignment_value(rows: Sequence[Mapping[str, object]], context: str, group: str, risk_quantity: str) -> float:
    for row in rows:
        if (_normalise_context(row.get("context")) == context and str(row.get("group")) == group
                and str(row.get("risk_quantity")) == risk_quantity
                and str(row.get("metric")) == "relative_logit_l2"):
            value = _optional_float(row.get("rho"), "rho")
            if value is not None:
                return value
    return math.nan


def _pair_context_text(payload: Mapping[str, object], fusion: str, context: str) -> str:
    fusions = payload.get("fusions", {})
    if isinstance(fusions, Mapping) and fusion in fusions and isinstance(fusions.get(fusion), Mapping):
        item = fusions.get(fusion, {}).get(context, {})
    else:
        # The F0 veto audit is persisted as {context: metrics}, while the
        # fusion audit is persisted as {fusions: {fusion: {context: metrics}}}.
        item = payload.get(context, {})
    if not isinstance(item, Mapping):
        item = {}
    return (f"correct={item.get('correct', 'NA')}, incorrect={item.get('incorrect', 'NA')}, "
            f"ties={item.get('tie', item.get('ties', 'NA'))}, accuracy={_fmt(item.get('accuracy'))}")


def _mean_metric(values: Iterable[object]) -> float:
    finite = [number for value in values if (number := _optional_float(value)) is not None]
    return statistics.fmean(finite) if finite else math.nan


def _final50_text(rows: Sequence[Mapping[str, object]], fusion: str) -> str:
    row = next((item for item in rows if item.get("fusion") == fusion), {})
    return (f"effective sparsity={_fmt(row.get('effective_sparsity'))}; "
            f"Attention removed={_fmt(row.get('attention_removed'))}; "
            f"FFN removed={_fmt(row.get('ffn_removed'))}; "
            f"first Attention sparsity={_fmt(row.get('first_attention_sparsity'))}")


def _interpretation(pair_accuracy: Mapping[str, object], fusion_rows: Sequence[Mapping[str, object]]) -> list[str]:
    scores: dict[str, tuple[float, float]] = {}
    for fusion in FUSIONS:
        accuracy = _mean_metric(_mapping_path(pair_accuracy, "fusions", fusion, context, "accuracy") for context in CONTEXTS)
        rho = _mean_metric(_alignment_value(fusion_rows, context, "all", _fusion_risk_name(fusion)) for context in CONTEXTS)
        if math.isfinite(accuracy) or math.isfinite(rho):
            scores[fusion] = (accuracy, rho)
    if "F0_HARD" not in scores:
        return ["Evidence-based interpretation: evidence insufficient because F0 comparison metrics are unavailable."]
    f0_accuracy, f0_rho = scores["F0_HARD"]
    alternatives = {fusion: value for fusion, value in scores.items() if fusion != "F0_HARD"}
    if alternatives and math.isfinite(f0_accuracy) and math.isfinite(f0_rho):
        dominates = all((not math.isfinite(value[0]) or value[0] <= f0_accuracy)
                        and (not math.isfinite(value[1]) or value[1] <= f0_rho)
                        for value in alternatives.values())
        better_alternative = any(math.isfinite(value[0]) and math.isfinite(value[1])
                                 and value[0] > f0_accuracy and value[1] > f0_rho
                                 for value in alternatives.values())
        if dominates:
            primary = "Evidence favors retaining the hard veto."
        elif better_alternative:
            primary = "The hard veto appears over-protective relative to at least one counterfactual on both measured criteria."
        else:
            primary = "Evidence is insufficient to prefer the hard veto or a counterfactual on both measured criteria."
    else:
        primary = "Evidence is insufficient because one of the aggregate comparison metrics is unavailable."
    lines = [primary]
    for fusion, label in (("F1_NO_VETO", "No-veto"), ("F2_CONSENSUS", "Consensus")):
        if fusion in scores and "F0_HARD" in scores:
            acc, rho = scores[fusion]
            if (math.isfinite(acc) and math.isfinite(f0_accuracy) and acc < f0_accuracy
                    and (not math.isfinite(rho) or not math.isfinite(f0_rho) or rho <= f0_rho)):
                lines.append(f"{label} appears too permissive relative to F0 on the stored pairwise/alignment evidence.")
    if "F3_MID_VETO" in scores:
        acc, rho = scores["F3_MID_VETO"]
        if ((math.isfinite(acc) and math.isfinite(f0_accuracy) and acc >= f0_accuracy)
                or (math.isfinite(rho) and math.isfinite(f0_rho) and rho >= f0_rho)):
            lines.append("The fixed midpoint has evidence that merits model-level validation; this diagnostic does not perform that validation.")
    return lines


def write_summary(output: Path, veto_summary: Mapping[str, object], causal_summary: Mapping[str, object], pair_accuracy: Mapping[str, object], fusion_accuracy: Mapping[str, object], event_summary: Mapping[str, object], replay: Mapping[str, object], task032_status: object, *, veto_alignment_rows: Sequence[Mapping[str, object]] = (), fusion_alignment_rows: Sequence[Mapping[str, object]] = (), final50_rows: Sequence[Mapping[str, object]] = (), snapshot_rows: Sequence[Mapping[str, object]] = (), overlaps: Mapping[str, object] | None = None) -> Path:
    overlaps = overlaps or {}
    target = .5
    lines = [
        "# Task033 scientific summary", "",
        "Task033 is a diagnosis-only comparison of frozen risk components. It does not change Average estimation, p_total, domain_damage, feasibility, budget, min-keep, or Task028 semantics. No model forward, validation, training, fine-tuning, or official registry was created.", "",
        "## Required scientific answers", "",
        "1. **Average veto margin at 50%.**",
        f"   Attention: median V={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_ATTENTION, 'V', 'median'))}, veto-active fraction={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_ATTENTION, 'veto_active_fraction'))}; FFN: median V={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_FFN, 'V', 'median'))}, veto-active fraction={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_FFN, 'veto_active_fraction'))}.",
        "2. **Veto margin versus causal relative-logit-L2.**",
        *[f"   {context}: all rho={_fmt(_alignment_value(veto_alignment_rows, context, 'all', 'V'))}, Attention rho={_fmt(_alignment_value(veto_alignment_rows, context, TYPE_ATTENTION, 'V'))}, FFN rho={_fmt(_alignment_value(veto_alignment_rows, context, TYPE_FFN, 'V'))}." for context in CONTEXTS],
        "3. **Cost, type, and stage evidence.**",
        f"   At 50%, V versus log(parameter cost) rho: Attention={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_ATTENTION, 'spearman_V_log_parameter_cost'))}, FFN={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_FFN, 'spearman_V_log_parameter_cost'))}; V versus stage rho: Attention={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_ATTENTION, 'spearman_V_stage'))}, FFN={_fmt(_veto_group_value(veto_summary, snapshot_rows, target, TYPE_FFN, 'spearman_V_stage'))}.",
        "4. **F0 veto-induced decision changes.**",
        f"   Total={event_summary.get('total_veto_induced_decision_changes', 'NA')}; Attention protected={event_summary.get('attention_protected_by_veto', 'NA')}; FFN protected={event_summary.get('ffn_protected_by_veto', 'NA')}.",
        "5. **F0 pairwise causal decision accuracy.**",
        *[f"   {context}: {_pair_context_text(pair_accuracy, 'F0_HARD', context)}." for context in CONTEXTS],
        "6. **F1 causal fidelity versus F0.**",
        *[f"   {context}: F0 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F0'))}; F1 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F1'))}." for context in CONTEXTS],
        "7. **F2 causal fidelity versus F0.**",
        *[f"   {context}: F0 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F0'))}; F2 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F2'))}." for context in CONTEXTS],
        "8. **F3 causal fidelity versus F1/F0.**",
        *[f"   {context}: F0 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F0'))}; F1 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F1'))}; F3 rho={_fmt(_alignment_value(fusion_alignment_rows, context, 'all', 'R_F3'))}." for context in CONTEXTS],
        "9. **Fusion pairwise causal decision accuracies.**",
        *[f"   {fusion}: FULL ({_pair_context_text(fusion_accuracy, fusion, 'FULL')}); TASK028_50 ({_pair_context_text(fusion_accuracy, fusion, 'TASK028_50')})." for fusion in FUSIONS],
        "10. **Exact-50 selection composition.**",
        *[f"   {fusion}: {_final50_text(final50_rows, fusion)}." for fusion in FUSIONS],
        "11. **Selection stability versus F0.**",
        *[f"   {key}: Jaccard={_fmt(value.get('jaccard'))}, exact-prefix={value.get('longest_common_exact_prefix', 'NA')}." for key, value in overlaps.items() if isinstance(value, Mapping)],
        "12. **Evidence-based interpretation.**",
        *[f"   {line}" for line in _interpretation(fusion_accuracy, fusion_alignment_rows)],
        "", "## Scope and integrity", "",
        "F0 uses the frozen Task028 optimized ranker. F1/F2/F3 use the same exact dynamic p_total, global p_average, domain_damage, feasibility, and budget; only the diagnostic fusion operator changes.",
        f"Task032 input status recorded: `{task032_status}`. Selector replay executed in this report: `{bool(replay.get('executed'))}`.",
        "F3 uses arithmetic `(B + p_average) / 2`; new tunable pruning hyperparameters = 0.",
    ]
    path = output / "task033_scientific_summary.md"
    atomic_text(path, "\n".join(lines) + "\n")
    return path


def completion_gate(payload: Mapping[str, object]) -> bool:
    required_true = (
        "task028_inputs_verified", "task029_inputs_verified", "task030_inputs_verified", "task031_inputs_verified", "task032_inputs_verified",
        "source_hashes_unchanged", "f0_exact_task028_replay", "f1_replay_complete", "f2_replay_complete", "f3_replay_complete",
        "veto_margin_complete", "task030_causal_alignment_complete", "pairwise_causal_audit_complete", "fusion_causal_alignment_complete",
        "fusion_pairwise_audit_complete", "f0_veto_decision_audit_complete", "selection_overlap_complete", "figures_complete",
        "scientific_summary_complete",
    )
    required_false = ("model_forward_executed", "validation_executed", "training_executed", "fine_tuning_executed",
                      "new_official_registry", "source_writes", "python_full_candidate_sort_in_hot_path")
    return (payload.get("status") == "PASS" and all(payload.get(key) is True for key in required_true)
            and all(payload.get(key) is False for key in required_false)
            and int(payload.get("new_tunable_pruning_hyperparameters", -1)) == 0)


def report_only_completion_gate(payload: Mapping[str, object]) -> bool:
    required_true = (
        "report_only", "report_artifacts_complete", "all_fusion_final_summaries",
        "all_fusion_snapshots", "causal_artifacts_complete", "pairwise_artifacts_complete",
        "veto_event_artifacts_complete", "selection_overlap_complete", "figures_complete",
        "scientific_summary_complete", "source_hashes_unchanged",
        "task033_replay_artifacts_unchanged",
    )
    required_false = ("selector_replay_executed", "gpu_used", "model_forward_executed",
                      "training_executed", "validation_executed", "fine_tuning_executed")
    hyperparameters = payload.get("new_pruning_hyperparameters",
                                  payload.get("new_tunable_pruning_hyperparameters", -1))
    return (payload.get("status") == "PASS"
            and all(payload.get(key) is True for key in required_true)
            and all(payload.get(key) is False for key in required_false)
            and int(hyperparameters) == 0)


def _report_artifact_hashes(output: Path) -> dict[str, str]:
    output = Path(output)
    return {name: sha256_file(output / name) for name in TASK033_REPLAY_ARTIFACTS}


def _require_report_artifacts(output: Path) -> dict[str, object]:
    output = Path(output)
    missing = [name for name in TASK033_REPLAY_ARTIFACTS
               if not (output / name).is_file() or (output / name).stat().st_size == 0]
    if missing:
        raise RuntimeError("Report-only requires non-empty Task033 replay artifacts: " + ", ".join(missing))
    final_rows = read_csv(output / "fusion_final_50_summary.csv")
    final_fusions = {str(row.get("fusion", "")) for row in final_rows}
    if len(final_rows) != len(FUSIONS) or final_fusions != set(FUSIONS):
        raise RuntimeError(f"fusion_final_50_summary.csv must contain exactly {FUSIONS}, got {sorted(final_fusions)}")
    snapshot_rows = read_csv(output / "fusion_selection_snapshots.csv")
    snapshot_keys = set()
    for row in snapshot_rows:
        target = _optional_float(row.get("snapshot_target"), "snapshot_target")
        if row.get("fusion") in FUSIONS and target is not None:
            snapshot_keys.add((str(row["fusion"]), round(target, 12)))
    expected_keys = {(fusion, round(target, 12)) for fusion in FUSIONS for target in SNAPSHOT_TARGETS}
    if len(snapshot_rows) != len(expected_keys) or snapshot_keys != expected_keys:
        raise RuntimeError("fusion_selection_snapshots.csv is missing one or more fusion/snapshot targets")
    return {
        "benchmark": read_json(output / "task033_performance_benchmark.json"),
        "veto_rows": read_csv(output / "veto_margin_by_snapshot.csv"),
        "veto_summary": read_json(output / "veto_margin_summary.json"),
        "veto_alignment_rows": read_csv(output / "veto_causal_alignment.csv"),
        "veto_alignment_summary": read_json(output / "veto_causal_alignment_summary.json"),
        "pair_rows": read_csv(output / "pairwise_veto_audit.csv"),
        "pair_accuracy": read_json(output / "pairwise_veto_accuracy.json"),
        "fusion_alignment_rows": read_csv(output / "fusion_causal_alignment.csv"),
        "fusion_alignment_summary": read_json(output / "fusion_causal_alignment_summary.json"),
        "fusion_decisions": read_csv(output / "fusion_pairwise_decisions.csv"),
        "fusion_accuracy": read_json(output / "fusion_pairwise_accuracy.json"),
        "pack_rows": read_csv(output / "veto_cost_matched_pack_audit.csv"),
        "stage_rows": read_csv(output / "fusion_stage_removals.csv"),
        "domain_rows": read_csv(output / "fusion_domain_removals.csv"),
        "first_attention": read_json(output / "fusion_first_attention.json"),
        "final_rows": final_rows,
        "overlaps": read_json(output / "fusion_selection_overlap.json"),
        "snapshots": snapshot_rows,
        "events": read_csv(output / "f0_veto_decision_changes.csv"),
        "event_summary": read_json(output / "f0_veto_decision_summary.json"),
    }


def _causal_rows_from_pair_audit(pair_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Recover compact raw causal plotting rows from persisted pair audits."""
    output: list[dict[str, object]] = []
    for row in pair_rows:
        for prefix, typ in (("attention", TYPE_ATTENTION), ("ffn", TYPE_FFN)):
            impact = _first_value(row, (f"{prefix}_causal_impact",))
            if _optional_float(impact, "relative_logit_l2") is None:
                continue
            output.append({
                "context": row.get("context", ""), "unit_type": typ,
                "V": row.get(f"{prefix}_V", ""),
                "R_F0": row.get(f"{prefix}_R_F0", ""),
                "R_F1": row.get(f"{prefix}_R_F1", ""),
                "R_F2": row.get(f"{prefix}_R_F2", ""),
                "R_F3": row.get(f"{prefix}_R_F3", ""),
                "relative_logit_l2": impact,
            })
    return output


def _sequence_from_csv(path: Path) -> list[int]:
    rows = read_csv(path)
    if not rows or not any(row.get("global_index", "") != "" for row in rows):
        return []
    def step_key(row: Mapping[str, object]) -> tuple[int, int]:
        return (_int(row.get("step", 0) or 0, "step"), _int(row.get("incremental_step", 0) or 0, "incremental_step"))
    ordered = sorted(rows, key=step_key)
    return [_int(row.get("global_index"), "global_index") for row in ordered if row.get("global_index", "") != ""]


def _recover_f0_sequence(output: Path) -> tuple[list[int], str | None]:
    output = Path(output)
    candidates = [output / "fusion_sequences" / "F0_HARD.csv", output / "f0_sequence.csv"]
    for root in (output / "fusion_sequences", output / "_replay"):
        if root.is_dir():
            candidates.extend(sorted(path for path in root.rglob("*.csv")
                                     if "f0" in path.name.lower() and "veto_decision" not in path.name.lower()))
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file() or path.stat().st_size == 0:
            continue
        seen.add(path)
        try:
            sequence = _sequence_from_csv(path)
        except (OSError, ValueError, csv.Error):
            continue
        if sequence:
            return sequence, str(path)
    return [], None


def _persist_replay_manifest(*, output: Path, replay: Mapping[str, object], final_rows: Sequence[Mapping[str, object]],
                             benchmark: Mapping[str, object], source_hashes: Mapping[str, object],
                             task028_sequence: Sequence[int]) -> dict[str, object]:
    output = Path(output)
    sequence_dir = output / "fusion_sequences"
    fusions = replay.get("fusions", {}) if isinstance(replay, Mapping) else {}
    sequence_info: dict[str, object] = {}
    final_by_fusion = {str(row.get("fusion")): row for row in final_rows}
    for fusion in FUSIONS:
        result = fusions.get(fusion, {}) if isinstance(fusions, Mapping) else {}
        selected = result.get("selected", []) if isinstance(result, Mapping) else []
        sequence_rows = []
        for step, row in enumerate(selected, 1):
            if not isinstance(row, Mapping):
                continue
            sequence_rows.append({"step": step, "global_index": row.get("global_index", ""),
                                  "effective_sparsity_after": row.get("effective_sparsity_after", "")})
        atomic_csv(sequence_dir / f"{fusion}.csv", ("step", "global_index", "effective_sparsity_after"), sequence_rows)
        summary = final_by_fusion.get(fusion, {})
        sequence = [row["global_index"] for row in sequence_rows]
        sequence_info[fusion] = {
            "length": len(sequence), "sequence_sha256": sequence_sha256(sequence),
            "final_sparsity": summary.get("effective_sparsity", result.get("final_sparsity", "")),
            "attention_removed": summary.get("attention_removed", ""),
            "ffn_removed": summary.get("ffn_removed", ""),
            "first_attention_sparsity": summary.get("first_attention_sparsity", ""),
            "all_snapshots_complete": bool(result.get("snapshots_complete", False)) if isinstance(result, Mapping) else False,
        }
    f0_result = sequence_info.get("F0_HARD", {})
    f0_sequence_sha = f0_result.get("sequence_sha256") if isinstance(f0_result, Mapping) else None
    manifest = {
        "code_version": CODE_VERSION, "timestamp": datetime.now(timezone.utc).isoformat(),
        "replay_artifacts_persisted_before_plotting": True,
        "selector_replay_executed": True, "gpu_used": True,
        "sequences": sequence_info, "benchmark_status": benchmark.get("status"),
        "source_hashes": source_hashes, "f0_task028_expected_sequence_sha256": sequence_sha256(task028_sequence),
        "f0_observed_sequence_sha256": f0_sequence_sha,
        "f0_exact_task028_sequence_equality": bool(f0_sequence_sha == sequence_sha256(task028_sequence)
                                                   and f0_result.get("length") == len(task028_sequence)),
    }
    atomic_json(output / "task033_replay_manifest.json", manifest)
    return manifest


def run_report_only(*, task028_root: Path, task029_root: Path, task030_root: Path, task031_root: Path,
                    task032_root: Path, output_dir: Path) -> dict[str, object]:
    """Regenerate report-layer outputs without loading the selector/runtime."""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = Path(output_dir).expanduser().resolve()
    roots = tuple(Path(value).expanduser().resolve() for value in (task028_root, task029_root, task030_root, task031_root, task032_root))
    _output_is_separate(output, roots)
    print("Task033 report-only: validating existing artifacts", flush=True)
    data = _require_report_artifacts(output)
    replay_hashes_before = _report_artifact_hashes(output)
    source_hashes = {
        "task028": hash_inputs(roots[0], TASK028_INPUTS), "task029": hash_inputs(roots[1], TASK029_INPUTS),
        "task030": hash_inputs(roots[2], TASK030_INPUTS), "task031": hash_inputs(roots[3], TASK031_INPUTS),
        "task032": hash_inputs(roots[4], TASK032_INPUTS),
    }
    prior_identity = read_json(output / "artifact_identity.json") if (output / "artifact_identity.json").is_file() else {}
    original_hashes = prior_identity.get("hashes_before", {}) if isinstance(prior_identity, Mapping) else {}
    source_unchanged = bool(original_hashes) and source_hashes == original_hashes
    print("Task033 report-only: generating figures", flush=True)
    causal_rows = _causal_rows_from_pair_audit(data["pair_rows"])
    figures = _write_figures(output, data["veto_rows"], causal_rows, data["fusion_alignment_rows"],
                             data["snapshots"], data["events"], data["overlaps"], data["pack_rows"],
                             data["fusion_accuracy"], data["stage_rows"])
    print("Task033 figures complete", flush=True)
    task032_status = read_json(roots[4] / "task032_completion.json").get("status", "unknown")
    summary_path = write_summary(output, data["veto_summary"], data["veto_alignment_summary"],
                                 data["pair_accuracy"], data["fusion_accuracy"], data["event_summary"],
                                 {"executed": False}, task032_status,
                                 veto_alignment_rows=data["veto_alignment_rows"],
                                 fusion_alignment_rows=data["fusion_alignment_rows"],
                                 final50_rows=data["final_rows"], snapshot_rows=data["veto_rows"],
                                 overlaps=data["overlaps"])
    print("Task033 scientific summary complete", flush=True)
    replay_hashes_after = _report_artifact_hashes(output)
    replay_unchanged = replay_hashes_before == replay_hashes_after
    task028_sequence = _task028_sequence(roots[0])
    recovered_f0, recovery_path = _recover_f0_sequence(output)
    f0_recoverable = bool(recovered_f0)
    f0_exact = f0_recoverable and recovered_f0 == task028_sequence
    report_identity = {
        "code_version": CODE_VERSION, "report_only": True,
        "source_hashes_current": source_hashes, "source_hashes_before": original_hashes,
        "source_hashes_unchanged": source_unchanged,
        "task033_replay_artifact_hashes_before": replay_hashes_before,
        "task033_replay_artifact_hashes_after": replay_hashes_after,
        "task033_replay_artifacts_unchanged": replay_unchanged,
        "selector_replay_executed": False, "gpu_used": False,
        "f0_exact_task028_replay_recoverable": f0_recoverable,
        "f0_exact_task028_replay": f0_exact if f0_recoverable else "UNVERIFIED_FROM_INTERRUPTED_RUN",
        "f0_sequence_recovery_path": recovery_path,
    }
    atomic_json(output / "task033_report_identity.json", report_identity)
    final_fusions = {str(row.get("fusion")) for row in data["final_rows"]}
    snapshot_keys = {(str(row.get("fusion")), round(float(row.get("snapshot_target")), 12)) for row in data["snapshots"]}
    expected_keys = {(fusion, round(target, 12)) for fusion in FUSIONS for target in SNAPSHOT_TARGETS}
    report_completion = {
        "status": "PASS" if source_unchanged and replay_unchanged and len(figures) >= 12 and summary_path.is_file() else "FAIL",
        "report_only": True, "report_recovery_pass": True, "report_artifacts_complete": True,
        "all_fusion_final_summaries": len(data["final_rows"]) == len(FUSIONS) and final_fusions == set(FUSIONS),
        "all_fusion_snapshots": len(data["snapshots"]) == len(expected_keys) and snapshot_keys == expected_keys,
        "causal_artifacts_complete": bool(data["veto_alignment_rows"]) and bool(data["fusion_alignment_rows"]),
        "pairwise_artifacts_complete": bool(data["pair_rows"]) and bool(data["fusion_decisions"]),
        "veto_event_artifacts_complete": bool(data["events"]) or bool(data["event_summary"]),
        "selection_overlap_complete": bool(data["overlaps"]),
        "figures_complete": len(figures) >= 12,
        "scientific_summary_complete": summary_path.is_file(),
        "source_hashes_unchanged": source_unchanged,
        "task033_replay_artifacts_unchanged": replay_unchanged,
        "f0_exact_task028_replay_recoverable": f0_recoverable,
        "f0_exact_task028_replay": f0_exact if f0_recoverable else "UNVERIFIED_FROM_INTERRUPTED_RUN",
        "f0_task028_expected_sequence_sha256": sequence_sha256(task028_sequence),
        "f0_observed_sequence_sha256": sequence_sha256(recovered_f0) if f0_recoverable else None,
        "selector_replay_executed": False, "gpu_used": False, "model_forward_executed": False,
        "training_executed": False, "validation_executed": False, "fine_tuning_executed": False,
        "new_pruning_hyperparameters": 0, "benchmark_status": data["benchmark"].get("status"),
        "new_tunable_pruning_hyperparameters": 0,
    }
    atomic_json(output / "task033_report_only_completion.json", report_completion)
    print("Task033 report-only: PASS" if report_only_completion_gate(report_completion) else "Task033 report-only: FAIL", flush=True)
    return {"report_identity": report_identity, "completion": report_completion, "figures": figures}


def _load_exact50(task029_root: Path, task031_root: Path) -> list[dict[str, object]]:
    snapshot = read_csv(Path(task029_root) / "snapshot_candidates.csv")
    exact = [row for row in snapshot if abs(_float(row.get("snapshot_target"), "snapshot_target", default=-1.0) - .5) < 1e-12]
    risk = read_csv(Path(task031_root) / "tested_unit_risk_table.csv")
    by_gid = {_int(row.get("global_index"), "global_index"): row for row in exact if row.get("global_index", "") != ""}
    for row in risk:
        if row.get("global_index", "") == "":
            continue
        gid = _int(row.get("global_index"), "global_index")
        by_gid[gid] = {**by_gid.get(gid, {}), **row}
    return _prepare_candidates(list(by_gid.values()))


def _load_snapshot_rows(task029_root: Path) -> list[dict[str, object]]:
    return veto_snapshot_rows(read_csv(Path(task029_root) / "snapshot_candidates.csv"))


def verify_identity(*, task028_root: Path, task029_root: Path, task030_root: Path, task031_root: Path, task032_root: Path, output_dir: Path) -> dict[str, object]:
    roots = tuple(Path(value).expanduser().resolve() for value in (task028_root, task029_root, task030_root, task031_root, task032_root))
    _output_is_separate(Path(output_dir), roots)
    hashes = {
        "task028": hash_inputs(roots[0], TASK028_INPUTS), "task029": hash_inputs(roots[1], TASK029_INPUTS),
        "task030": hash_inputs(roots[2], TASK030_INPUTS), "task031": hash_inputs(roots[3], TASK031_INPUTS),
        "task032": hash_inputs(roots[4], TASK032_INPUTS),
    }
    payload = {"status": "PASS", "code_version": CODE_VERSION, "hashes_before": hashes,
               "model_forward_executed": False, "validation_executed": False, "training_executed": False,
               "fine_tuning_executed": False, "selector_replay_executed": False, "source_writes": False,
               "new_official_registry": False, "new_tunable_pruning_hyperparameters": 0,
               "fusions": list(FUSIONS), "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return payload


def _task028_sequence(root: Path) -> list[int]:
    trace = read_csv(Path(root) / TASK028_INPUTS[1])
    return [_int(row.get("global_index"), "global_index") for row in sorted(trace, key=lambda row: (_int(row.get("step", 0), "step"), _int(row.get("incremental_step", 0), "incremental_step")))]


def _lookup_from_trace(root: Path) -> dict[int, dict[str, object]]:
    result = {}
    for row in read_csv(Path(root) / TASK028_INPUTS[1]):
        if row.get("global_index", "") != "":
            result[_int(row["global_index"], "global_index")] = dict(row)
    return result


def run_diagnosis(*, task028_root: Path, task029_root: Path, task030_root: Path, task031_root: Path, task032_root: Path, output_dir: Path, task014_root: Path | None = None, task016_root: Path | None = None, task017_root: Path | None = None, device: str = "cuda:0", ranking_backend: str = "single-gpu") -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = tuple(Path(value).expanduser().resolve() for value in (task028_root, task029_root, task030_root, task031_root, task032_root))
    identity = verify_identity(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], task032_root=roots[4], output_dir=output)
    before = identity["hashes_before"]
    snapshot_rows = _load_snapshot_rows(roots[1])
    veto_summary = summarize_veto_snapshots(snapshot_rows)
    atomic_csv(output / "veto_margin_by_snapshot.csv", tuple(sorted(set().union(*(row.keys() for row in snapshot_rows))) if snapshot_rows else ("snapshot_target", "unit_type", "B", "p_average", "V", "R_hard")), snapshot_rows)
    atomic_json(output / "veto_margin_summary.json", veto_summary)
    exact50 = _load_exact50(roots[1], roots[3])
    raw_causal = read_csv(roots[2] / TASK030_INPUTS[0])
    heads = read_csv(roots[2] / TASK030_INPUTS[1])
    pairs = read_csv(roots[2] / TASK030_INPUTS[2])
    packs = read_csv(roots[2] / TASK030_INPUTS[3])
    recovered = recover_single_interventions(raw_causal, heads, pairs)
    joined = _join_causal_rows(recovered, {_int(row["global_index"], "global_index"): row for row in exact50})
    causal_rows, causal_summary = causal_alignment(joined, ("p_total", "p_average", "B", "V", "R_hard"))
    atomic_csv(output / "veto_causal_alignment.csv", ("context", "group", "risk_quantity", "metric", "rho", "n"), causal_rows)
    atomic_json(output / "veto_causal_alignment_summary.json", causal_summary)
    pair_rows, pair_accuracy = pairwise_audit(joined)
    atomic_csv(output / "pairwise_veto_audit.csv", tuple(sorted(set().union(*(row.keys() for row in pair_rows))) if pair_rows else ("context", "pair_id")), pair_rows)
    atomic_json(output / "pairwise_veto_accuracy.json", pair_accuracy)
    fusion_causal_rows, fusion_causal_summary = causal_alignment(joined, ("R_F0", "R_F1", "R_F2", "R_F3"))
    atomic_csv(output / "fusion_causal_alignment.csv", ("context", "group", "risk_quantity", "metric", "rho", "n"), fusion_causal_rows)
    atomic_json(output / "fusion_causal_alignment_summary.json", fusion_causal_summary)
    fusion_decisions, fusion_accuracy = fusion_pairwise_audit(joined)
    atomic_csv(output / "fusion_pairwise_decisions.csv", tuple(sorted(set().union(*(row.keys() for row in fusion_decisions))) if fusion_decisions else ("context", "pair_id", "fusion")), fusion_decisions)
    atomic_json(output / "fusion_pairwise_accuracy.json", fusion_accuracy)
    pack_audit = cost_matched_pack_audit(joined, raw_causal, packs)
    atomic_csv(output / "veto_cost_matched_pack_audit.csv", tuple(sorted(set().union(*(row.keys() for row in pack_audit))) if pack_audit else ("context", "pair_id", "V")), pack_audit)

    replay: dict[str, object] = {"executed": False, "fusions": {}}
    overlaps: dict[str, object] = {}
    all_snapshots: list[dict[str, object]] = []
    all_final: list[dict[str, object]] = []
    all_stage: list[dict[str, object]] = []
    all_domain: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    event_summary: dict[str, object] = {"total_veto_induced_decision_changes": 0, "attention_protected_by_veto": 0, "ffn_protected_by_veto": 0, "stage_distribution": {}, "domain_distribution": {}}
    benchmark: dict[str, object] = {"status": "NOT_RUN"}
    if task014_root and task016_root and task017_root:
        benchmark_path = output / "task033_performance_benchmark.json"
        if benchmark_path.is_file():
            benchmark = read_json(benchmark_path)
        else:
            benchmark = run_benchmark(task014_root=Path(task014_root), task016_root=Path(task016_root), task017_root=Path(task017_root), output_dir=output, device=device, ranking_backend=ranking_backend)
        if not benchmark_gate(benchmark):
            raise RuntimeError("Task033 performance benchmark gate failed; full replay not started")
        print("Task033 benchmark complete", flush=True)
        task028_sequence = _task028_sequence(roots[0])
        selections: dict[str, Sequence[int]] = {}
        type_lookup = {int(row.get("global_index")): str(row.get("unit_type", "")) for row in read_csv(roots[1] / "snapshot_candidates.csv") if row.get("global_index", "") != ""}
        for fusion in FUSIONS:
            started = time.perf_counter()
            print(f"Task033 replay {fusion}: start", flush=True)
            engine, optimized = _engine_for_replay(task014_root=Path(task014_root), task016_root=Path(task016_root), task017_root=Path(task017_root), output_dir=output / "_replay" / fusion, device=device, ranking_backend=ranking_backend)
            task031 = __import__("task031_domain_conditioned_average_diagnosis")
            # The trace covers selected units only.  Structural metadata for
            # the still-feasible tail must come from the immutable replay
            # object, as in Task031; otherwise snapshot type/cost counts would
            # silently lose unselected candidates.
            lookup = task031.lookup_from_engine(engine)
            lookup.update(_lookup_from_trace(roots[0]))
            total = float(getattr(engine, "total_parameters"))
            result = replay_fusion(fusion=fusion, engine=engine, optimized=optimized, total_parameters=total, target_budget=TARGET_SPARSITY * total, lookup=lookup, task028_sequence=task028_sequence)
            replay["fusions"][fusion] = result
            selections[fusion] = result["sequence"]
            all_snapshots.extend(result["snapshots"])
            event_rows.extend(result["event_rows"])
            for snapshot in result["snapshots"]:
                for stage, count in snapshot.get("stage_removals", {}).items():
                    all_stage.append({"fusion": fusion, "snapshot_target": snapshot["snapshot_target"], "stage": stage, "removed_count": count})
                for domain, count in snapshot.get("domain_removals", {}).items():
                    all_domain.append({"fusion": fusion, "snapshot_target": snapshot["snapshot_target"], "domain_id": domain, "removed_count": count})
            all_final.append({"fusion": fusion, "effective_sparsity": result["final_sparsity"], "sequence_length": len(result["sequence"]), "first_attention_sparsity": next((row.get("effective_sparsity_after") for row in result["selected"] if row.get("unit_type") == TYPE_ATTENTION), None), "attention_removed": sum(row.get("unit_type") == TYPE_ATTENTION for row in result["selected"]), "ffn_removed": sum(row.get("unit_type") == TYPE_FFN for row in result["selected"])})
            print(f"Task033 replay {fusion}: complete, seconds={time.perf_counter() - started:.3f}", flush=True)
        overlaps = selection_overlap(selections, type_lookup)
        hard_events = event_rows
        event_summary = {"total_veto_induced_decision_changes": len(hard_events),
                         "attention_protected_by_veto": sum(row.get("hard_selected_type") == TYPE_ATTENTION and row.get("base_selected_type") != TYPE_ATTENTION for row in hard_events),
                         "ffn_protected_by_veto": sum(row.get("hard_selected_type") == TYPE_FFN and row.get("base_selected_type") != TYPE_FFN for row in hard_events),
                         "stage_distribution": dict(Counter(str(row.get("hard_selected_stage", "")) for row in hard_events)),
                         "domain_distribution": dict(Counter(str(row.get("hard_selected_domain_id", "")) for row in hard_events))}
        replay["executed"] = True
    else:
        overlaps = {"status": "NOT_RUN", "reason": "immutable Task014/016/017 roots were not supplied"}
    atomic_csv(output / "fusion_selection_snapshots.csv", tuple(sorted(set().union(*(row.keys() for row in all_snapshots))) if all_snapshots else ("fusion", "snapshot_target")), all_snapshots)
    atomic_csv(output / "fusion_final_50_summary.csv", tuple(sorted(set().union(*(row.keys() for row in all_final))) if all_final else ("fusion", "effective_sparsity")), all_final)
    atomic_json(output / "fusion_first_attention.json", {row["fusion"]: {"first_attention_sparsity": row.get("first_attention_sparsity"), "attention_removed": row.get("attention_removed")} for row in all_final})
    atomic_csv(output / "fusion_stage_removals.csv", ("fusion", "snapshot_target", "stage", "removed_count"), all_stage)
    atomic_csv(output / "fusion_domain_removals.csv", ("fusion", "snapshot_target", "domain_id", "removed_count"), all_domain)
    atomic_json(output / "fusion_selection_overlap.json", overlaps)
    atomic_csv(output / "f0_veto_decision_changes.csv", tuple(sorted(set().union(*(row.keys() for row in event_rows))) if event_rows else ("step", "sparsity", "hard_selected_global_index", "base_selected_global_index")), event_rows)
    atomic_json(output / "f0_veto_decision_summary.json", event_summary)
    if replay.get("executed"):
        _persist_replay_manifest(output=output, replay=replay, final_rows=all_final,
                                 benchmark=benchmark, source_hashes=before,
                                 task028_sequence=task028_sequence)
        print("Task033 replay artifacts persisted", flush=True)
    figures = _write_figures(output, snapshot_rows, joined, fusion_causal_rows, all_snapshots, event_rows, overlaps, pack_audit, fusion_accuracy, all_stage)
    print("Task033 figures complete", flush=True)
    summary_path = write_summary(output, veto_summary, causal_summary, pair_accuracy, fusion_accuracy, event_summary, replay, read_json(roots[4] / "task032_completion.json").get("status", "unknown"), veto_alignment_rows=causal_rows, fusion_alignment_rows=fusion_causal_rows, final50_rows=all_final, snapshot_rows=snapshot_rows, overlaps=overlaps)
    print("Task033 scientific summary complete", flush=True)
    after = {"task028": hash_inputs(roots[0], TASK028_INPUTS), "task029": hash_inputs(roots[1], TASK029_INPUTS), "task030": hash_inputs(roots[2], TASK030_INPUTS), "task031": hash_inputs(roots[3], TASK031_INPUTS), "task032": hash_inputs(roots[4], TASK032_INPUTS)}
    unchanged = before == after
    replay_results = replay.get("fusions", {}) if isinstance(replay, Mapping) else {}
    f0_result = replay_results.get("F0_HARD", {}) if isinstance(replay_results, Mapping) else {}
    completion = {
        "status": "PASS" if replay.get("executed") and unchanged and benchmark_gate(benchmark) and len(figures) >= 12 else "NOT_RUN" if not replay.get("executed") else "FAIL",
        "task028_inputs_verified": True, "task029_inputs_verified": True, "task030_inputs_verified": True,
        "task031_inputs_verified": True, "task032_inputs_verified": True, "source_hashes_unchanged": unchanged,
        "f0_exact_task028_replay": bool(replay.get("executed")) and f0_result.get("sequence") == _task028_sequence(roots[0]),
        "f1_replay_complete": bool(replay_results.get("F1_NO_VETO")), "f2_replay_complete": bool(replay_results.get("F2_CONSENSUS")),
        "f3_replay_complete": bool(replay_results.get("F3_MID_VETO")), "veto_margin_complete": bool(snapshot_rows),
        "task030_causal_alignment_complete": bool(causal_rows), "pairwise_causal_audit_complete": bool(pair_rows),
        "fusion_causal_alignment_complete": bool(fusion_causal_rows), "fusion_pairwise_audit_complete": bool(fusion_decisions),
        "f0_veto_decision_audit_complete": bool(replay.get("executed")), "selection_overlap_complete": bool(overlaps),
        "figures_complete": len(figures) >= 12, "scientific_summary_complete": summary_path.is_file(),
        "performance_benchmark_pass": benchmark_gate(benchmark),
        "model_forward_executed": False, "validation_executed": False, "training_executed": False,
        "fine_tuning_executed": False, "new_official_registry": False, "source_writes": False,
        "python_full_candidate_sort_in_hot_path": False, "new_tunable_pruning_hyperparameters": 0,
        "selector_replay_executed": bool(replay.get("executed")), "figure_paths": figures,
        "f0_sequence_sha256": sequence_sha256(f0_result.get("sequence", ())) if f0_result else None,
    }
    atomic_json(output / "artifact_identity.json", {**identity, "hashes_after": after, "source_hashes_unchanged": unchanged})
    atomic_json(output / "task033_performance_benchmark.json", benchmark)
    atomic_json(output / "task033_completion.json", completion)
    print("Task033 completion written", flush=True)
    return {"identity": identity, "veto_summary": veto_summary, "causal_summary": causal_summary,
            "pair_accuracy": pair_accuracy, "fusion_accuracy": fusion_accuracy, "replay": replay,
            "overlaps": overlaps, "completion": completion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "benchmark", "run", "completion", "report-only", "report-completion"), required=True)
    parser.add_argument("--task028-root", type=Path)
    parser.add_argument("--task029-root", type=Path)
    parser.add_argument("--task030-root", type=Path)
    parser.add_argument("--task031-root", type=Path)
    parser.add_argument("--task032-root", type=Path)
    parser.add_argument("--task014-root", type=Path)
    parser.add_argument("--task016-root", type=Path)
    parser.add_argument("--task017-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranking-backend", default="single-gpu")
    return parser


def _required(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    names = ("task028_root", "task029_root", "task030_root", "task031_root", "task032_root")
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    return tuple(getattr(args, name) for name in names)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "completion":
        if not completion_gate(read_json(args.output_dir / "task033_completion.json")):
            raise RuntimeError("Task033 completion gate failed")
        print("Task033 completion: PASS", flush=True)
        return 0
    if args.mode == "report-completion":
        if not report_only_completion_gate(read_json(args.output_dir / "task033_report_only_completion.json")):
            raise RuntimeError("Task033 report-only completion gate failed")
        print("Task033 report-only recovery: PASS", flush=True)
        return 0
    roots = _required(args)
    if args.mode == "identity":
        verify_identity(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], task032_root=roots[4], output_dir=args.output_dir)
        print("Task033 identity: PASS", flush=True)
        return 0
    if args.mode == "benchmark":
        technical = (args.task014_root, args.task016_root, args.task017_root)
        if any(value is None for value in technical):
            raise SystemExit("benchmark mode requires --task014-root, --task016-root, and --task017-root")
        verify_identity(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], task032_root=roots[4], output_dir=args.output_dir)
        report = run_benchmark(task014_root=args.task014_root, task016_root=args.task016_root, task017_root=args.task017_root, output_dir=args.output_dir, device=args.device, ranking_backend=args.ranking_backend)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not benchmark_gate(report):
            return 1
        return 0
    if args.mode == "report-only":
        result = run_report_only(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], task032_root=roots[4], output_dir=args.output_dir)
        return 0 if report_only_completion_gate(result["completion"]) else 1
    result = run_diagnosis(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], task032_root=roots[4], output_dir=args.output_dir, task014_root=args.task014_root, task016_root=args.task016_root, task017_root=args.task017_root, device=args.device, ranking_backend=args.ranking_backend)
    print(json.dumps({"status": result["completion"]["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task032: causal decomposition of Average granularity dependence.

This module is an offline, CPU-only diagnosis.  It consumes the immutable
CSV/JSON artifacts produced by Task028--Task031, computes causal and Average
granularity exponents, and writes all new products below ``output_dir``.
It intentionally does not import a model, a selector, a dataset, or a
checkpoint runtime.  In particular, it never replays a pruning trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CODE_VERSION = "task032_average_causal_granularity_decomposition_v1"
SEED = 3407
BOOTSTRAP_REPLICATES = 10_000
CONTEXT_FULL = "FULL"
CONTEXT_TASK028_50 = "TASK028_50"
CONTEXTS = (CONTEXT_FULL, CONTEXT_TASK028_50)
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"
SNAPSHOT_TARGETS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)

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

PRIMARY_METRIC = "relative_logit_l2"
UNIT_TYPES = (TYPE_ATTENTION, TYPE_FFN)
FIGURE_COUNT = 12


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _output_is_separate(output_dir: Path, roots: Sequence[Path]) -> None:
    output = Path(output_dir).expanduser().resolve()
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if output == resolved or resolved in output.parents:
            raise RuntimeError(f"Task032 output must be outside immutable root: {resolved}")


def _stage(row: Mapping[str, object]) -> str:
    value = str(row.get("stage", "")).strip()
    if value:
        return value
    layer = str(row.get("layer", ""))
    if "layers." in layer:
        return layer.split("layers.", 1)[1].split(".", 1)[0]
    if layer.startswith("stage"):
        return layer.split(".", 1)[0].removeprefix("stage")
    return layer.split(".", 1)[0] if layer else "unknown"


def _feasible(row: Mapping[str, object]) -> bool:
    return str(row.get("feasible", "True")).strip().lower() not in {"false", "0", "no"}


def _first(row: Mapping[str, object], names: Sequence[str], default: object = None) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _unit_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"attention", "attention_head", "attn", "head", "attention_single"}:
        return TYPE_ATTENTION
    if text in {"ffn", "ffn_neuron", "neuron", "ffn_single", "ffn_cost_matched_pack"}:
        return TYPE_FFN
    return str(value or "").strip()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"n": 0, **{name: math.nan for name in ("mean", "median", "std", "q25", "q75", "min", "max")}}
    return {
        "n": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "median": statistics.median(cleaned),
        "std": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
        "q25": _percentile(cleaned, 0.25),
        "q75": _percentile(cleaned, 0.75),
        "min": min(cleaned),
        "max": max(cleaned),
    }


def machine_safe_epsilon(values: Sequence[float], dtype: str = "float64") -> float:
    """Return a reported machine-scale offset, never a research hyperparameter."""
    if dtype in {"float32", "single"}:
        dtype_epsilon = 1.1920928955078125e-7
    elif dtype in {"float16", "half"}:
        dtype_epsilon = 0.0009765625
    else:
        dtype_epsilon = 2.220446049250313e-16
    positive = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    observed_min = min(positive) if positive else 1.0
    return dtype_epsilon * max(1.0, observed_min)


def _positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _is_true(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def causal_beta_pair(attention_parameter_cost: float, single_ffn_parameter_cost: float,
                     attention_relative_logit_l2: float,
                     single_ffn_relative_logit_l2: float) -> float:
    """Compute log(I_ratio)/log(C_ratio) only for valid positive ratios."""
    values = (attention_parameter_cost, single_ffn_parameter_cost,
              attention_relative_logit_l2, single_ffn_relative_logit_l2)
    if not all(_positive(value) for value in values):
        raise ValueError("causal beta requires positive finite costs and metrics")
    cost_ratio = float(attention_parameter_cost) / float(single_ffn_parameter_cost)
    impact_ratio = float(attention_relative_logit_l2) / float(single_ffn_relative_logit_l2)
    if not _positive(cost_ratio) or not _positive(impact_ratio):
        raise ValueError("causal beta requires positive ratios")
    log_cost = math.log(cost_ratio)
    if log_cost == 0.0:
        raise ValueError("causal beta is undefined when the cost ratio is one")
    return math.log(impact_ratio) / log_cost


def _row_cost(row: Mapping[str, object]) -> float:
    value = _first(row, ("removed_parameter_cost", "parameter_cost"))
    return _float(value, "parameter_cost")


def _row_metric(row: Mapping[str, object], name: str = PRIMARY_METRIC) -> float:
    return _float(row.get(name), name)


def recover_task030_unit_metadata(raw_results: Sequence[Mapping[str, object]],
                                  selected_attention_heads: Sequence[Mapping[str, object]],
                                  selected_causal_pairs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Fill blank historical Task030 labels from immutable identity metadata."""
    attention_ids: set[int] = set()
    pair_types: dict[int, str] = {}
    for row in selected_attention_heads:
        if str(_first(row, ("global_index",), "")).strip():
            attention_ids.add(_int(_first(row, ("global_index",)), "attention global_index"))
    for row in selected_causal_pairs:
        attention = _first(row, ("attention_global_index",))
        ffn = _first(row, ("ffn_global_index",))
        if attention is not None and str(attention).strip():
            pair_types[_int(attention, "attention global_index")] = TYPE_ATTENTION
            attention_ids.add(_int(attention, "attention global_index"))
        if ffn is not None and str(ffn).strip():
            pair_types[_int(ffn, "ffn global_index")] = TYPE_FFN

    recovered: list[dict[str, object]] = []
    for source in raw_results:
        row = dict(source)
        current = _unit_type(row.get("unit_type"))
        intervention = str(row.get("intervention", "")).strip().lower()
        gid_value = _first(row, ("global_index",))
        gid = _int(gid_value, "global_index") if gid_value is not None and str(gid_value).strip() else None
        if current not in UNIT_TYPES:
            if intervention == "attention_single" or (gid is not None and gid in attention_ids):
                current = TYPE_ATTENTION
            elif intervention in {"ffn_single", "ffn_cost_matched_pack"} or (gid is not None and gid in pair_types):
                current = TYPE_FFN
        if current in UNIT_TYPES:
            row["unit_type"] = current
        row["stage"] = _stage(row)
        recovered.append(row)
    return recovered


def _pair_costs(selected_causal_pairs: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], tuple[float, float]]:
    costs: dict[tuple[str, str], tuple[float, float]] = {}
    for row in selected_causal_pairs:
        pair_id = str(_first(row, ("pair_id",), ""))
        attention = _first(row, ("attention_parameter_cost",))
        ffn = _first(row, ("ffn_parameter_cost", "single_ffn_parameter_cost"))
        if attention is None or ffn is None:
            continue
        costs[(str(_first(row, ("context",), "*")), pair_id)] = (_float(attention, "attention_parameter_cost"), _float(ffn, "ffn_parameter_cost"))
        costs[("*", pair_id)] = costs[(str(_first(row, ("context",), "*")), pair_id)]
    return costs


def pairwise_causal_records(raw_results: Sequence[Mapping[str, object]],
                            selected_causal_pairs: Sequence[Mapping[str, object]] = ()) -> list[dict[str, object]]:
    """Return one row per context/pair with raw causal ratios and beta."""
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in raw_results:
        context = str(row.get("context", ""))
        pair_id = str(row.get("pair_id", ""))
        grouped[(context, pair_id)][str(row.get("intervention", ""))] = row
    pair_cost = _pair_costs(selected_causal_pairs)
    output: list[dict[str, object]] = []
    for (context, pair_id), group in sorted(grouped.items()):
        attention = group.get("attention_single")
        ffn = group.get("ffn_single")
        if attention is None or ffn is None:
            continue
        configured = pair_cost.get((context, pair_id), pair_cost.get(("*", pair_id)))
        attention_cost = configured[0] if configured else _row_cost(attention)
        ffn_cost = configured[1] if configured else _row_cost(ffn)
        attention_metric = _row_metric(attention)
        ffn_metric = _row_metric(ffn)
        row: dict[str, object] = {
            "context": context,
            "pair_id": pair_id,
            "attention_global_index": _first(attention, ("global_index",), ""),
            "ffn_global_index": _first(ffn, ("global_index",), ""),
            "attention_parameter_cost": attention_cost,
            "single_ffn_parameter_cost": ffn_cost,
            "attention_relative_logit_l2": attention_metric,
            "single_ffn_relative_logit_l2": ffn_metric,
            "C_ratio": attention_cost / ffn_cost if _positive(ffn_cost) else math.nan,
            "I_ratio": attention_metric / ffn_metric if _positive(ffn_metric) else math.nan,
        }
        try:
            row["beta_causal_pair"] = causal_beta_pair(attention_cost, ffn_cost, attention_metric, ffn_metric)
            row["valid_positive_ratios"] = True
            row["invalid_reason"] = ""
        except ValueError as exc:
            row["beta_causal_pair"] = math.nan
            row["valid_positive_ratios"] = False
            row["invalid_reason"] = str(exc)
        output.append(row)
    return output


def causal_beta_statistics(pair_rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    values = [float(row["beta_causal_pair"]) for row in pair_rows
              if _is_true(row.get("valid_positive_ratios", True))
              and math.isfinite(float(row.get("beta_causal_pair")))]
    return summary_stats(values)


def theil_sen_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("x and y lengths differ")
    slopes = []
    for left in range(len(x_values)):
        for right in range(left + 1, len(x_values)):
            dx = float(x_values[right]) - float(x_values[left])
            if dx != 0.0:
                slopes.append((float(y_values[right]) - float(y_values[left])) / dx)
    return statistics.median(slopes) if slopes else math.nan


def ols_loglog(values_x: Sequence[float], values_y: Sequence[float],
               eps_y: float | None = None) -> dict[str, float | int]:
    pairs = [(float(x), float(y)) for x, y in zip(values_x, values_y)
             if math.isfinite(float(x)) and math.isfinite(float(y)) and float(x) > 0.0]
    if eps_y is None:
        eps_y = machine_safe_epsilon([y for _, y in pairs])
    x = [math.log(value) for value, _ in pairs]
    y = [math.log(value + eps_y) for _, value in pairs]
    if len(x) < 2:
        return {"n": len(x), "slope": math.nan, "intercept": math.nan, "eps_y": eps_y}
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator if denominator else math.nan
    intercept = mean_y - slope * mean_x if math.isfinite(slope) else math.nan
    return {"n": len(x), "slope": slope, "intercept": intercept, "eps_y": eps_y}


def causal_regression_estimates(pair_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in pair_rows if _is_true(row.get("valid_positive_ratios"), default=False)]
    filtered = [row for row in valid if _float(row["C_ratio"], "C_ratio") != 1.0]
    x = [math.log(_float(row["C_ratio"], "C_ratio")) for row in filtered]
    impacts = [_float(row["I_ratio"], "I_ratio") for row in filtered]
    eps_metric = machine_safe_epsilon(impacts)
    y = [math.log(value + eps_metric) for value in impacts]
    sen = theil_sen_slope(x, y) if len(x) >= 2 else math.nan
    ols = ols_loglog([_float(row["C_ratio"], "C_ratio") for row in filtered], impacts, eps_y=eps_metric)
    return {
        "n": len(valid),
        "median_pairwise_beta": causal_beta_statistics(valid).get("median", math.nan),
        "theil_sen_slope": sen,
        "ols_loglog_slope": ols["slope"],
        "ols_loglog_intercept": ols["intercept"],
        "eps_metric": eps_metric,
    }


def paired_bootstrap_median(values: Sequence[float], replicates: int = BOOTSTRAP_REPLICATES,
                            seed: int = SEED) -> dict[str, object]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"seed": seed, "replicates": replicates, "n": 0, "median": math.nan,
                "ci_95": [math.nan, math.nan], "samples": []}
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(int(replicates)):
        resample = [cleaned[generator.randrange(len(cleaned))] for _ in cleaned]
        samples.append(float(statistics.median(resample)))
    return {
        "seed": seed,
        "replicates": int(replicates),
        "n": len(cleaned),
        "median": float(statistics.median(cleaned)),
        "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "samples": samples,
    }


# Descriptive aliases keep the mathematical helpers easy to discover for
# downstream diagnostics without creating a second implementation.
pairwise_beta = causal_beta_pair
pairwise_causal_beta = causal_beta_pair
compute_causal_beta = causal_beta_pair
bootstrap_median_beta = paired_bootstrap_median


def leave_one_pair_out(pair_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in pair_rows if row.get("valid_positive_ratios") is True]
    output = []
    for held_out in valid:
        training = [row for row in valid if row is not held_out and str(row.get("pair_id")) != str(held_out.get("pair_id"))]
        betas = [float(row["beta_causal_pair"]) for row in training]
        x = [math.log(_float(row["C_ratio"], "C_ratio")) for row in training if _float(row["C_ratio"], "C_ratio") != 1.0]
        y = [math.log(_float(row["I_ratio"], "I_ratio")) for row in training if _float(row["C_ratio"], "C_ratio") != 1.0]
        output.append({
            "context": held_out.get("context", ""),
            "held_out_pair_id": held_out.get("pair_id", ""),
            "n_training_pairs": len(training),
            "median_beta_causal_excluding_pair": statistics.median(betas) if betas else math.nan,
            "theil_sen_beta_causal_excluding_pair": theil_sen_slope(x, y) if len(x) >= 2 else math.nan,
            "ols_beta_causal_excluding_pair": ols_loglog(
                [_float(row["C_ratio"], "C_ratio") for row in training],
                [_float(row["I_ratio"], "I_ratio") for row in training],
            )["slope"] if training else math.nan,
        })
    return output


def _normalise_candidate(source: Mapping[str, object], metadata: Mapping[int, Mapping[str, object]] | None = None) -> dict[str, object]:
    row = dict(source)
    gid_value = _first(row, ("global_index",))
    gid = _int(gid_value, "global_index")
    extra = dict((metadata or {}).get(gid, {}))
    output = {**extra, **row}
    output["global_index"] = gid
    output["unit_type"] = _unit_type(_first(row, ("unit_type",), extra.get("unit_type", "")))
    output["stage"] = _stage(output)
    output["domain_id"] = _int(_first(output, ("domain_id",), 0), "domain_id")
    output["parameter_cost"] = _float(_first(output, ("parameter_cost",), 0), "parameter_cost")
    output["Delta_average"] = _float(_first(output, ("Delta_average",), 0.0), "Delta_average")
    output["Delta_total"] = _float(_first(output, ("Delta_total",), 0.0), "Delta_total")
    p_average = _first(output, ("p_A_global", "p_average", "p_A_variant", "p_A"), math.nan)
    output["p_average"] = _float(p_average, "p_average", default=math.nan)
    output["p_A_global"] = output["p_average"]
    output["feasible"] = _feasible(output)
    return output


def load_exact50_candidates(task031_root: Path, task029_root: Path) -> list[dict[str, object]]:
    primary = read_csv(Path(task031_root) / "tested_unit_risk_table.csv")
    snapshot = read_csv(Path(task029_root) / "snapshot_candidates.csv")
    metadata: dict[int, Mapping[str, object]] = {}
    for raw in snapshot:
        if str(raw.get("global_index", "")).strip():
            metadata[_int(raw["global_index"], "global_index")] = raw
    tested = {_int(row["global_index"], "global_index"): row for row in primary
              if str(row.get("global_index", "")).strip()}
    snapshot_50 = []
    for raw in snapshot:
        try:
            if abs(_float(raw.get("snapshot_target"), "snapshot_target") - .5) >= 1e-12:
                continue
            gid = _int(raw["global_index"], "global_index")
            # Task031's tested table is intentionally only the causal probe
            # subset in historical runs.  The complete exact-50 candidate
            # state remains Task029's raw snapshot; overlay the Task031 risk
            # fields for the probe units without discarding other candidates.
            merged = dict(raw)
            merged.update(tested.get(gid, {}))
            snapshot_50.append(merged)
        except (KeyError, TypeError, ValueError):
            continue
    source_rows = snapshot_50 if snapshot_50 else primary
    rows = [_normalise_candidate(row, metadata) for row in source_rows]
    return [row for row in rows if _feasible(row)]


def load_snapshots(task029_root: Path, exact50: Sequence[Mapping[str, object]]) -> dict[float, list[dict[str, object]]]:
    source = read_csv(Path(task029_root) / "snapshot_candidates.csv")
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for raw in source:
        try:
            target = _float(raw.get("snapshot_target"), "snapshot_target")
            grouped[target].append(_normalise_candidate(raw))
        except (TypeError, ValueError):
            continue
    grouped[0.5] = [dict(row) for row in exact50]
    return {target: [row for row in rows if _feasible(row)] for target, rows in grouped.items()}


def average_granularity_scaling(snapshot_rows: Mapping[float, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    output = []
    for target in sorted(snapshot_rows):
        rows = [row for row in snapshot_rows[target] if _feasible(row)]
        costs = [_float(row.get("parameter_cost"), "parameter_cost") for row in rows]
        averages = [_float(row.get("Delta_average"), "Delta_average") for row in rows]
        eps = machine_safe_epsilon(averages)
        regression = ols_loglog(costs, averages, eps_y=eps)
        valid_x = [math.log(cost) for cost, average in zip(costs, averages)
                   if cost > 0 and math.isfinite(average) and average + eps > 0]
        valid_y = [math.log(average + eps) for cost, average in zip(costs, averages)
                   if cost > 0 and math.isfinite(average) and average + eps > 0]
        by_type: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            if str(row.get("unit_type")) in UNIT_TYPES:
                by_type[str(row["unit_type"])].append(row)
        attn, ffn = by_type.get(TYPE_ATTENTION, []), by_type.get(TYPE_FFN, [])
        attn_a = statistics.median([_float(row["Delta_average"], "Delta_average") for row in attn]) if attn else math.nan
        ffn_a = statistics.median([_float(row["Delta_average"], "Delta_average") for row in ffn]) if ffn else math.nan
        attn_c = statistics.median([_float(row["parameter_cost"], "parameter_cost") for row in attn]) if attn else math.nan
        ffn_c = statistics.median([_float(row["parameter_cost"], "parameter_cost") for row in ffn]) if ffn else math.nan
        beta_type = math.nan
        if all(_positive(value) for value in (attn_a, ffn_a, attn_c, ffn_c)) and attn_c != ffn_c:
            beta_type = math.log(attn_a / ffn_a) / math.log(attn_c / ffn_c)
        output.append({
            "snapshot_target": target,
            "candidate_count": len(rows),
            "positive_metric_count": sum(value > 0 for value in averages),
            "eps_A": eps,
            "ols_beta_A": regression["slope"],
            "ols_intercept_A": regression["intercept"],
            "theil_sen_beta_A": theil_sen_slope(valid_x, valid_y) if len(valid_x) >= 2 else math.nan,
            "beta_A_type": beta_type,
            "attention_median_Delta_average": attn_a,
            "ffn_median_Delta_average": ffn_a,
            "attention_median_parameter_cost": attn_c,
            "ffn_median_parameter_cost": ffn_c,
        })
    return output


def excess_granularity_decomposition(scaling_rows: Sequence[Mapping[str, object]],
                                     causal_contexts: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    exact = next((row for row in scaling_rows if abs(_float(row.get("snapshot_target"), "snapshot_target") - 0.5) < 1e-12), {})
    beta_a_type = _float(exact.get("beta_A_type"), "beta_A_type", default=math.nan)
    beta_a_ols = _float(exact.get("ols_beta_A"), "ols_beta_A", default=math.nan)
    result: dict[str, object] = {"status": "PASS", "diagnostic_only": True, "snapshot_target": 0.5,
                                 "contexts": {}, "snapshot_rows": []}
    for context, causal in causal_contexts.items():
        median_beta = _float(causal.get("median_pairwise_beta"), "median_pairwise_beta", default=math.nan)
        result["contexts"][context] = {
            "beta_A_type": beta_a_type,
            "median_beta_causal": median_beta,
            "beta_excess_type_minus_causal": beta_a_type - median_beta if math.isfinite(beta_a_type) and math.isfinite(median_beta) else math.nan,
            "beta_A_ols": beta_a_ols,
            "beta_excess_ols_minus_causal": beta_a_ols - median_beta if math.isfinite(beta_a_ols) and math.isfinite(median_beta) else math.nan,
            "causal_beta_bootstrap_ci_95": causal.get("bootstrap_ci_95", [math.nan, math.nan]),
        }
    for row in scaling_rows:
        target = _float(row.get("snapshot_target"), "snapshot_target")
        for context, causal in causal_contexts.items():
            beta_causal = _float(causal.get("median_pairwise_beta"), "median_pairwise_beta", default=math.nan)
            beta_a = _float(row.get("beta_A_type"), "beta_A_type", default=math.nan)
            result["snapshot_rows"].append({
                "context": context, "snapshot_target": target,
                "beta_A_type": beta_a, "median_beta_causal": beta_causal,
                "beta_excess": beta_a - beta_causal if math.isfinite(beta_a) and math.isfinite(beta_causal) else math.nan,
            })
    return result


def beta_excess(beta_A: float, beta_causal: float) -> float:
    """Return the diagnostic difference beta_A - beta_causal."""
    return float(beta_A) - float(beta_causal)


def residual_decomposition(candidates: Sequence[Mapping[str, object]], beta_causal: float,
                           intercept_A: float, eps_A: float) -> list[dict[str, object]]:
    output = []
    for source in candidates:
        row = dict(source)
        cost = _float(row.get("parameter_cost"), "parameter_cost")
        average = _float(row.get("Delta_average"), "Delta_average")
        if cost <= 0:
            continue
        log_observed = math.log(average + eps_A)
        log_predicted = intercept_A + beta_causal * math.log(cost)
        log_residual = log_observed - beta_causal * math.log(cost)
        row.update({
            "log_Delta_average_plus_eps_A": log_observed,
            "log_A_granularity_predicted": log_predicted,
            "log_A_residual_component": log_residual,
            "log_A_residual_after_intercept": log_observed - log_predicted,
            "A_causal_granularity_component": math.exp(log_predicted),
            # The formula requested by Part G omits the fitted intercept.  A
            # multiplicative component must remove that constant as well so
            # that causal_component * residual_component reconstructs
            # Delta_average + eps_A exactly.
            "A_residual_component": math.exp(log_observed - log_predicted),
            "A_residual_formula_component": math.exp(log_residual),
            "causal_beta_used": beta_causal,
            "eps_A": eps_A,
        })
        output.append(row)
    return output


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and float(values[order[end + 1]]) == float(values[order[position]]):
            end += 1
        rank = (position + end + 2) / 2.0
        for index in range(position, end + 1):
            ranks[order[index]] = rank
        position = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def ks_distance(xs: Sequence[float], ys: Sequence[float]) -> float:
    left, right = sorted(float(x) for x in xs), sorted(float(y) for y in ys)
    if not left or not right:
        return math.nan
    values = sorted(set(left + right))
    return max(abs(sum(x <= value for x in left) / len(left) - sum(y <= value for y in right) / len(right)) for value in values)


def auc_greater(attention: Sequence[float], ffn: Sequence[float]) -> float:
    if not attention or not ffn:
        return math.nan
    wins = ties = 0
    for left in attention:
        for right in ffn:
            wins += left > right
            ties += left == right
    return (wins + 0.5 * ties) / (len(attention) * len(ffn))


def causal_residual_alignment(pair_rows: Sequence[Mapping[str, object]],
                              raw_results: Sequence[Mapping[str, object]],
                              candidates: Sequence[Mapping[str, object]],
                              eps_A: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_gid = {_int(row["global_index"], "global_index"): row for row in candidates}
    alignment: list[dict[str, object]] = []
    pair_loo = []
    for context in CONTEXTS:
        context_pairs = [row for row in pair_rows if str(row.get("context")) == context]
        pair_loo.extend(leave_one_pair_out(context_pairs))
    loo_by_context = defaultdict(dict)
    for row in pair_loo:
        loo_by_context[str(row.get("context"))][str(row.get("held_out_pair_id"))] = row
    for causal in pair_rows:
        context, pair_id = str(causal.get("context")), str(causal.get("pair_id"))
        beta_row = loo_by_context.get(context, {}).get(pair_id, {})
        beta = _float(beta_row.get("median_beta_causal_excluding_pair"), "loo beta", default=math.nan)
        for intervention, typ, gid_key in (("attention_single", TYPE_ATTENTION, "attention_global_index"),
                                            ("ffn_single", TYPE_FFN, "ffn_global_index")):
            raw = next((row for row in raw_results if str(row.get("context")) == context and str(row.get("pair_id")) == pair_id and str(row.get("intervention")) == intervention), None)
            if raw is None:
                continue
            gid = _int(causal[gid_key], "global_index")
            candidate = by_gid.get(gid)
            if candidate is None or not math.isfinite(beta) or _float(candidate.get("parameter_cost"), "parameter_cost") <= 0:
                continue
            cost = _float(candidate["parameter_cost"], "parameter_cost")
            average = _float(candidate["Delta_average"], "Delta_average")
            alignment.append({
                "context": context, "pair_id": pair_id,
                "held_out_pair_id": pair_id, "unit_type": typ, "global_index": gid,
                "parameter_cost": cost, "relative_logit_l2": _row_metric(raw),
                "Delta_average": average, "p_average": _float(candidate.get("p_A_global"), "p_average", default=math.nan),
                "raw_log_Delta_average": math.log(average + eps_A),
                "leave_one_pair_out_residual": math.log(average + eps_A) - beta * math.log(cost),
                "beta_causal_excluding_pair": beta,
                "n_training_pairs": beta_row.get("n_training_pairs", 0),
            })
    summaries = []
    for context in CONTEXTS:
        rows = [row for row in alignment if row["context"] == context]
        for group, subset in (("all", rows), (TYPE_ATTENTION, [row for row in rows if row["unit_type"] == TYPE_ATTENTION]),
                              (TYPE_FFN, [row for row in rows if row["unit_type"] == TYPE_FFN])):
            summaries.append({
                "context": context, "group": group, "n": len(subset),
                "spearman_Delta_average": spearman([_float(row["Delta_average"], "Delta_average") for row in subset], [_float(row["relative_logit_l2"], PRIMARY_METRIC) for row in subset]),
                "spearman_p_average": spearman([_float(row["p_average"], "p_average") for row in subset], [_float(row["relative_logit_l2"], PRIMARY_METRIC) for row in subset]),
                "spearman_leave_one_pair_out_residual": spearman([_float(row["leave_one_pair_out_residual"], "residual") for row in subset], [_float(row["relative_logit_l2"], PRIMARY_METRIC) for row in subset]),
            })
    return alignment, summaries


def _pack_members(pack: Mapping[str, object]) -> list[int]:
    raw = _first(pack, ("ffn_global_indices",), "[]")
    try:
        value = json.loads(str(raw)) if isinstance(raw, str) else raw
        return [_int(item, "pack global_index") for item in (value or ())]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def cost_matched_decomposition(pair_rows: Sequence[Mapping[str, object]],
                               raw_results: Sequence[Mapping[str, object]],
                               packs: Sequence[Mapping[str, object]],
                               components: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_gid = {_int(row["global_index"], "global_index"): row for row in components}
    pack_by_id = {str(row.get("pair_id", "")): row for row in packs}
    output = []
    for pair in pair_rows:
        context, pair_id = str(pair.get("context")), str(pair.get("pair_id"))
        attn = next((row for row in raw_results if str(row.get("context")) == context and str(row.get("pair_id")) == pair_id and str(row.get("intervention")) == "attention_single"), None)
        pack = next((row for row in raw_results if str(row.get("context")) == context and str(row.get("pair_id")) == pair_id and str(row.get("intervention")) == "ffn_cost_matched_pack"), None)
        if attn is None or pack is None:
            continue
        attn_gid = _int(pair["attention_global_index"], "attention_global_index")
        candidate = by_gid.get(attn_gid)
        if candidate is None:
            continue
        pack_definition = pack_by_id.get(pair_id, {})
        member_rows = [by_gid[gid] for gid in _pack_members(pack_definition) if gid in by_gid]
        pack_average = statistics.fmean([_float(row["Delta_average"], "Delta_average") for row in member_rows]) if member_rows else math.nan
        causal_component = _float(candidate.get("A_causal_granularity_component"), "causal component", default=math.nan)
        residual_component = _float(candidate.get("A_residual_component"), "residual component", default=math.nan)
        attention_average = _float(candidate.get("Delta_average"), "Delta_average")
        output.append({
            "context": context, "pair_id": pair_id,
            "attention_global_index": attn_gid,
            "ffn_pack_size": len(member_rows),
            "attention_causal_relative_logit_l2": _row_metric(attn),
            "ffn_pack_causal_relative_logit_l2": _row_metric(pack),
            "causal_impact_ratio_attention_over_pack": _row_metric(attn) / _row_metric(pack) if _positive(_row_metric(pack)) else math.nan,
            "attention_Delta_average": attention_average,
            "ffn_pack_mean_Delta_average": pack_average,
            "attention_A_causal_granularity_component": causal_component,
            "attention_A_residual_component": residual_component,
            "raw_attention_minus_pack_average": attention_average - pack_average if math.isfinite(pack_average) else math.nan,
            "component_explains_raw_gap": (causal_component - pack_average) if math.isfinite(causal_component) and math.isfinite(pack_average) else math.nan,
        })
    return output


def type_separation(rows: Sequence[Mapping[str, object]], metrics: Mapping[str, str]) -> list[dict[str, object]]:
    output = []
    for name, field in metrics.items():
        attention = [_float(row.get(field), field) for row in rows if row.get("unit_type") == TYPE_ATTENTION and math.isfinite(_float(row.get(field), field))]
        ffn = [_float(row.get(field), field) for row in rows if row.get("unit_type") == TYPE_FFN and math.isfinite(_float(row.get(field), field))]
        output.append({
            "metric": name, "field": field,
            "attention_count": len(attention), "ffn_count": len(ffn),
            "attention_median": statistics.median(attention) if attention else math.nan,
            "ffn_median": statistics.median(ffn) if ffn else math.nan,
            "median_gap_attention_minus_ffn": (statistics.median(attention) - statistics.median(ffn)) if attention and ffn else math.nan,
            "ks_distance": ks_distance(attention, ffn),
            "auc_separability_attention_greater": auc_greater(attention, ffn),
        })
    return output


def domain_residual_analysis(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    domains: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        domains[_int(row.get("domain_id", 0), "domain_id")].append(row)
    detail, mixed = [], []
    for domain_id, members in sorted(domains.items()):
        kinds = {row.get("unit_type") for row in members if row.get("unit_type") in UNIT_TYPES}
        domain_kind = "mixed" if kinds == set(UNIT_TYPES) else "pure_attention" if kinds == {TYPE_ATTENTION} else "pure_ffn" if kinds == {TYPE_FFN} else "unknown"
        for typ in UNIT_TYPES:
            values = [_float(row["log_A_residual_component"], "residual") for row in members if row.get("unit_type") == typ and "log_A_residual_component" in row]
            detail.append({"domain_id": domain_id, "domain_kind": domain_kind, "unit_type": typ,
                           "count": len(values), "residual_mean": statistics.fmean(values) if values else math.nan,
                           "residual_median": statistics.median(values) if values else math.nan,
                           "residual_q25": _percentile(values, .25), "residual_q75": _percentile(values, .75)})
        if domain_kind == "mixed":
            attn = [_float(row["log_A_residual_component"], "residual") for row in members if row.get("unit_type") == TYPE_ATTENTION]
            ffn = [_float(row["log_A_residual_component"], "residual") for row in members if row.get("unit_type") == TYPE_FFN]
            mixed.append({"domain_id": domain_id, "attention_count": len(attn), "ffn_count": len(ffn),
                          "attention_residual_median": statistics.median(attn) if attn else math.nan,
                          "ffn_residual_median": statistics.median(ffn) if ffn else math.nan,
                          "median_gap_attention_minus_ffn": statistics.median(attn) - statistics.median(ffn) if attn and ffn else math.nan,
                          "ks_distance": ks_distance(attn, ffn), "auc_separability_attention_greater": auc_greater(attn, ffn)})
    return detail, mixed


mixed_domain_analysis = domain_residual_analysis


def stage_granularity_decomposition(rows: Sequence[Mapping[str, object]], eps_A: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("unit_type") in UNIT_TYPES:
            grouped[(_stage(row), str(row["unit_type"]))].append(row)
    output = []
    for (stage, typ), members in sorted(grouped.items()):
        costs = [_float(row["parameter_cost"], "parameter_cost") for row in members]
        averages = [_float(row["Delta_average"], "Delta_average") for row in members]
        fit = ols_loglog(costs, averages, eps_y=eps_A)
        residuals = [_float(row["log_A_residual_component"], "residual") for row in members if "log_A_residual_component" in row]
        output.append({"stage": stage, "unit_type": typ, "count": len(members),
                       "beta_A_ols": fit["slope"], "beta_A_ols_intercept": fit["intercept"],
                       "residual_mean": statistics.fmean(residuals) if residuals else math.nan,
                       "residual_median": statistics.median(residuals) if residuals else math.nan,
                       "residual_q25": _percentile(residuals, .25), "residual_q75": _percentile(residuals, .75),
                       "eps_A": eps_A})
    return output


def _identity_payload(task028_root: Path, task029_root: Path, task030_root: Path, task031_root: Path) -> dict[str, object]:
    return {
        "status": "PASS", "code_version": CODE_VERSION,
        "task028_hashes": hash_inputs(task028_root, TASK028_INPUTS),
        "task029_hashes": hash_inputs(task029_root, TASK029_INPUTS),
        "task030_hashes": hash_inputs(task030_root, TASK030_INPUTS),
        "task031_hashes": hash_inputs(task031_root, TASK031_INPUTS),
        "contexts": list(CONTEXTS), "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "model_forward_executed": False, "training_executed": False,
        "fine_tuning_executed": False, "validation_executed": False,
        "selector_replay_executed": False, "new_pruning_hyperparameters": 0,
        "source_writes": False,
    }


def verify_identity(*, task028_root: Path, task029_root: Path, task030_root: Path,
                    task031_root: Path, output_dir: Path) -> dict[str, object]:
    roots = tuple(Path(value).expanduser().resolve() for value in (task028_root, task029_root, task030_root, task031_root))
    _output_is_separate(Path(output_dir), roots)
    payload = _identity_payload(*roots)
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return payload


def _causal_context_summary(pair_rows: Sequence[Mapping[str, object]], bootstrap: Mapping[str, object]) -> dict[str, object]:
    estimates = causal_regression_estimates(pair_rows)
    estimates["pairwise_stats"] = causal_beta_statistics(pair_rows)
    estimates["bootstrap_median"] = bootstrap.get("median", math.nan)
    estimates["bootstrap_ci_95"] = bootstrap.get("ci_95", [math.nan, math.nan])
    return estimates


def _write_causal_outputs(output: Path, pair_rows: Sequence[Mapping[str, object]],
                          leave_rows: Sequence[Mapping[str, object]],
                          bootstrap_by_context: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    fields = tuple(pair_rows[0].keys()) if pair_rows else ("context", "pair_id")
    atomic_csv(output / "causal_beta_pairwise.csv", fields, pair_rows)
    atomic_csv(output / "causal_beta_leave_one_out.csv", tuple(leave_rows[0].keys()) if leave_rows else ("context", "held_out_pair_id"), leave_rows)
    bootstrap_payload = {"seed": SEED, "replicates": BOOTSTRAP_REPLICATES, "contexts": {}}
    summaries = {}
    for context in CONTEXTS:
        rows = [row for row in pair_rows if str(row.get("context")) == context]
        bootstrap = dict(bootstrap_by_context.get(context, {}))
        bootstrap["eps_metric"] = machine_safe_epsilon(
            [_float(row.get("I_ratio"), "I_ratio") for row in rows if row.get("valid_positive_ratios") is True]
        )
        bootstrap_payload["contexts"][context] = bootstrap
        summaries[context] = _causal_context_summary(rows, bootstrap)
    atomic_json(output / "causal_beta_bootstrap.json", bootstrap_payload)
    atomic_json(output / "causal_beta_summary.json", summaries)
    return summaries


def _maybe_task031_comparators(task031_root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    path = Path(task031_root) / "causal_alignment_by_variant.csv"
    if path.is_file():
        result["causal_alignment_by_variant"] = read_csv(path)
    summary = Path(task031_root) / "causal_alignment_summary.json"
    if summary.is_file():
        result["causal_alignment_summary"] = read_json(summary)
    return result


def _fmt(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
        return "NA" if not math.isfinite(number) else f"{number:.{digits}g}"
    except (TypeError, ValueError):
        return "NA"


def write_scientific_summary(output: Path, causal: Mapping[str, Mapping[str, object]],
                             scaling: Sequence[Mapping[str, object]], excess: Mapping[str, object],
                             alignment: Sequence[Mapping[str, object]], mixed: Sequence[Mapping[str, object]],
                             comparators: Mapping[str, object]) -> Path:
    primary_scaling = next((row for row in scaling if abs(_float(row.get("snapshot_target"), "snapshot_target") - .5) < 1e-12), {})
    lines = [
        "# Task032 scientific summary", "",
        "Task032 is a diagnosis-only, CPU/statistical decomposition of Average granularity dependence. "
        "Task028, Task029, Task030, and Task031 inputs were read-only; no model forward, selector replay, "
        "training, fine-tuning, validation, or new pruning method was run.", "",
        "## Required scientific answers", "",
        "1. **Measured causal granularity scaling exponent beta_causal.**",
    ]
    for context in CONTEXTS:
        row = causal.get(context, {})
        lines.append(f"   - {context}: median pairwise beta = `{_fmt(row.get('median_pairwise_beta'))}`; "
                     f"Theil-Sen = `{_fmt(row.get('theil_sen_slope'))}`; OLS = `{_fmt(row.get('ols_loglog_slope'))}`.")
    lines += [
        "2. **Stability across eight pairs.** Stability is described by the pairwise q25/q75/std below; "
        "no clipping or [0,1] constraint was imposed.",
    ]
    for context in CONTEXTS:
        stats = causal.get(context, {}).get("pairwise_stats", {})
        lines.append(f"   - {context}: n={stats.get('n', 0)}, std=`{_fmt(stats.get('std'))}`, "
                     f"q25=`{_fmt(stats.get('q25'))}`, q75=`{_fmt(stats.get('q75'))}`.")
    lines += [
        "3. **FULL versus TASK028_50.** The two context estimates are reported separately; their difference is "
        "descriptive and is not used as a selector parameter.",
        f"   - Median difference TASK028_50 minus FULL: `{_fmt(_float(causal.get(CONTEXT_TASK028_50, {}).get('median_pairwise_beta'), 'beta', default=math.nan) - _float(causal.get(CONTEXT_FULL, {}).get('median_pairwise_beta'), 'beta', default=math.nan))}`.",
        f"4. **Observed Average granularity exponent beta_A.** At exact 50%, type-median beta_A = `{_fmt(primary_scaling.get('beta_A_type'))}`, OLS beta_A = `{_fmt(primary_scaling.get('ols_beta_A'))}`, and Theil-Sen beta_A = `{_fmt(primary_scaling.get('theil_sen_beta_A'))}`.",
        "5. **Is beta_A substantially larger?** This is assessed by the sign and magnitude of beta_excess, with bootstrap uncertainty shown for beta_causal; it is not declared from a tolerance alone.",
    ]
    for context, row in (excess.get("contexts", {}) if isinstance(excess.get("contexts"), Mapping) else {}).items():
        lines.append(f"   - {context}: beta_excess(type minus causal) = `{_fmt(row.get('beta_excess_type_minus_causal'))}`; "
                     f"causal bootstrap CI = `{row.get('causal_beta_bootstrap_ci_95', [])}`.")
    lines += [
        "6. **How large is beta_excess?** See `excess_granularity_decomposition.json` and the context rows above; it is a diagnostic quantity only, not a coefficient.",
        "7. **Causal alignment after causal scaling.** The leave-one-pair-out correlations are in `causal_residual_alignment_summary.json`; the held-out construction avoids using the same pair to estimate its correction.",
    ]
    for row in alignment:
        if row.get("group") == "all":
            lines.append(f"   - {row.get('context')}: residual Spearman = `{_fmt(row.get('spearman_leave_one_pair_out_residual'))}`, n={row.get('n')}; raw Delta Spearman = `{_fmt(row.get('spearman_Delta_average'))}`.")
    if comparators:
        lines.append("8. **Compared with Task031 domain-conditioned Average.** Task031 comparator artifacts were available and are preserved for inspection; this report does not promote a new selector or claim superiority from a single correlation.")
        lines.append("9. **Compared with Task031 cost-conditioned Average.** The same cautious comparison applies; anti-correlation is not silently treated as corrected by this diagnosis.")
    else:
        lines.append("8. **Compared with Task031 domain-conditioned Average.** The optional Task031 causal-alignment comparator was not present in the required raw input subset; evidence is insufficient for a direct superiority claim.")
        lines.append("9. **Compared with Task031 cost-conditioned Average.** The optional Task031 comparator was not present; no claim about avoiding its anti-correlation is made.")
    lines += [
        f"10. **Mixed domains.** `{len(mixed)}` mixed-domain rows were analyzed in `mixed_domain_residual_analysis.csv`; within-domain Attention/FFN residual gaps are reported without pooling domains.",
        "11. **Is correction justified?** This task does not automatically recommend a new method. The defensible conclusion is determined by the measured beta_excess, bootstrap interval, held-out residual alignment, and mixed-domain rows; if those do not agree, the evidence is insufficient.",
        "",
        "## Integrity and scope",
        "",
        "- `beta_excess = beta_A - beta_causal` is diagnostic only.",
        "- Raw Delta_average, causal impact, and source domain_damage artifacts were not rewritten.",
        "- No official beta hyperparameter, quota, type-specific coefficient, or registry was introduced.",
        "- Reproducibility seed: `3407`; bootstrap replicates: `10000`.",
    ]
    path = output / "task032_scientific_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_figures(output: Path, pair_rows: Sequence[Mapping[str, object]],
                 bootstrap_by_context: Mapping[str, Mapping[str, object]],
                 scaling: Sequence[Mapping[str, object]], components: Sequence[Mapping[str, object]],
                 alignment: Sequence[Mapping[str, object]], pack_rows: Sequence[Mapping[str, object]],
                 mixed_rows: Sequence[Mapping[str, object]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Task032 figures require matplotlib") from exc
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, number: int, title: str, xlabel: str, ylabel: str) -> None:
        fig.tight_layout()
        fig.suptitle(title, y=1.02)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        path = figure_dir / f"Figure{number:02d}_{title.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    valid_pairs = [row for row in pair_rows if row.get("valid_positive_ratios") is True]
    fig, ax = plt.subplots(figsize=(6, 4))
    for typ, marker in (("Attention", "o"), ("single FFN", "s")):
        x, y = [], []
        for row in valid_pairs:
            x.append(math.log(_float(row["attention_parameter_cost" if typ == "Attention" else "single_ffn_parameter_cost"], "cost")))
            y.append(math.log(_float(row["attention_relative_logit_l2" if typ == "Attention" else "single_ffn_relative_logit_l2"], PRIMARY_METRIC)))
        ax.scatter(x, y, marker=marker, label=typ)
    ax.legend(); save(fig, 1, "log cost vs causal impact", "log parameter cost", "log relative-logit-L2")

    fig, ax = plt.subplots(figsize=(6, 4))
    for context in CONTEXTS:
        values = [_float(row["beta_causal_pair"], "beta") for row in valid_pairs if row.get("context") == context]
        ax.scatter([context] * len(values), values, label=context)
    ax.set_ylim(auto=True); save(fig, 2, "pairwise causal beta", "context", "beta causal pair")

    fig, ax = plt.subplots(figsize=(6, 4))
    for context in CONTEXTS:
        values = bootstrap_by_context.get(context, {}).get("samples", [])
        if values:
            ax.hist(values, bins=min(30, max(5, len(set(values)))), alpha=.55, label=context)
    ax.legend(); save(fig, 3, "bootstrap causal beta", "bootstrap beta", "count")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([math.log(_float(row["parameter_cost"], "cost")) for row in components],
               [_float(row["Delta_average"], "Delta_average") for row in components], s=8)
    save(fig, 4, "log cost vs Average", "log parameter cost", "Delta_average")

    fig, ax = plt.subplots(figsize=(6, 4))
    targets = [_float(row["snapshot_target"], "snapshot_target") for row in scaling]
    ax.plot(targets, [_float(row["beta_A_type"], "beta_A_type", default=math.nan) for row in scaling], marker="o", label="beta_A type")
    ax.plot(targets, [_float(row["ols_beta_A"], "ols_beta_A", default=math.nan) for row in scaling], marker="s", label="beta_A OLS")
    ax.legend(); save(fig, 5, "Average exponent across snapshots", "snapshot target", "beta_A")

    fig, ax = plt.subplots(figsize=(6, 4))
    for context in CONTEXTS:
        rows = [row for row in []]
        beta_causal = statistics.median([_float(row["beta_causal_pair"], "beta") for row in valid_pairs if row.get("context") == context]) if any(row.get("context") == context for row in valid_pairs) else math.nan
        x = targets
        y = [_float(row["beta_A_type"], "beta_A_type", default=math.nan) - beta_causal for row in scaling]
        ax.plot(x, y, marker="o", label=context)
    ax.axhline(0, color="black", linewidth=.8); ax.legend(); save(fig, 6, "excess granularity exponent", "snapshot target", "beta excess")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([_float(row["raw_log_Delta_average"], "raw log") for row in alignment], [_float(row["leave_one_pair_out_residual"], "residual") for row in alignment], s=14)
    save(fig, 7, "raw Average vs causal residual", "raw log Delta_average", "leave-one-pair-out residual")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([_float(row["p_average"], "p_average") for row in alignment], [_float(row["relative_logit_l2"], PRIMARY_METRIC) for row in alignment], s=14)
    save(fig, 8, "causal impact vs p average", "global p_average", "relative-logit-L2")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([_float(row["leave_one_pair_out_residual"], "residual") for row in alignment], [_float(row["relative_logit_l2"], PRIMARY_METRIC) for row in alignment], s=14)
    save(fig, 9, "causal impact vs held-out residual", "leave-one-pair-out residual", "relative-logit-L2")

    fig, ax = plt.subplots(figsize=(6, 4))
    for typ, marker in ((TYPE_ATTENTION, "o"), (TYPE_FFN, "s")):
        values = [_float(row["log_A_residual_component"], "residual") for row in components if row.get("unit_type") == typ]
        ax.boxplot(values, positions=[1 if typ == TYPE_ATTENTION else 2], manage_ticks=False)
    ax.set_xticks([1, 2], [TYPE_ATTENTION, TYPE_FFN])
    save(fig, 10, "Attention FFN residual distributions", "unit type", "log residual component")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([_float(row["attention_causal_relative_logit_l2"], PRIMARY_METRIC) for row in pack_rows], [_float(row["attention_A_causal_granularity_component"], "component") for row in pack_rows], label="causal component")
    ax.scatter([_float(row["ffn_pack_causal_relative_logit_l2"], PRIMARY_METRIC) for row in pack_rows], [_float(row["ffn_pack_mean_Delta_average"], "pack average") for row in pack_rows], label="FFN pack average")
    ax.legend(); save(fig, 11, "cost matched pack decomposition", "causal impact", "Average component")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([_float(row["attention_residual_median"], "attention residual") for row in mixed_rows], [_float(row["ffn_residual_median"], "ffn residual") for row in mixed_rows], s=18)
    ax.axline((0, 0), slope=1, color="black", linewidth=.8); save(fig, 12, "mixed domain residual comparison", "Attention residual median", "FFN residual median")
    return paths


def completion_gate(payload: Mapping[str, object]) -> bool:
    required_true = (
        "task028_inputs_verified", "task029_inputs_verified", "task030_inputs_verified", "task031_inputs_verified",
        "source_hashes_unchanged", "attention_metadata_recovered", "eight_pairs_per_context",
        "attention_single_eight_per_context", "ffn_single_eight_per_context", "cost_matched_packs_eight",
        "causal_beta_complete", "bootstrap_complete", "leave_one_pair_out_complete", "average_scaling_complete",
        "excess_decomposition_complete", "causal_residual_alignment_complete", "cost_matched_decomposition_complete",
        "type_separation_complete", "mixed_domain_analysis_complete", "stage_analysis_complete", "figures_complete",
        "scientific_summary_complete",
    )
    required_false = ("model_forward_executed", "training_executed", "fine_tuning_executed", "validation_executed",
                      "selector_replay_executed", "source_writes")
    return (payload.get("status") == "PASS" and all(payload.get(key) is True for key in required_true)
            and all(payload.get(key) is False for key in required_false)
            and int(payload.get("new_pruning_hyperparameters", -1)) == 0)


def run_diagnosis(*, task028_root: Path, task029_root: Path, task030_root: Path,
                  task031_root: Path, output_dir: Path) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = tuple(Path(value).expanduser().resolve() for value in (task028_root, task029_root, task030_root, task031_root))
    _output_is_separate(output, roots)
    before = {
        "task028": hash_inputs(roots[0], TASK028_INPUTS),
        "task029": hash_inputs(roots[1], TASK029_INPUTS),
        "task030": hash_inputs(roots[2], TASK030_INPUTS),
        "task031": hash_inputs(roots[3], TASK031_INPUTS),
    }
    atomic_json(output / "artifact_identity.json", {"status": "PASS", "code_version": CODE_VERSION, "hashes_before": before,
                                                      "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                                                      "model_forward_executed": False, "selector_replay_executed": False,
                                                      "training_executed": False, "fine_tuning_executed": False,
                                                      "validation_executed": False, "source_writes": False,
                                                      "new_pruning_hyperparameters": 0})

    raw_results = read_csv(roots[2] / "causal_ablation_results.csv")
    heads = read_csv(roots[2] / "selected_attention_heads.csv")
    pairs = read_csv(roots[2] / "selected_causal_pairs.csv")
    packs = read_csv(roots[2] / "cost_matched_ffn_packs.csv")
    recovered_raw = recover_task030_unit_metadata(raw_results, heads, pairs)
    pair_rows = pairwise_causal_records(recovered_raw, pairs)
    leave_rows = [row for context in CONTEXTS for row in leave_one_pair_out([r for r in pair_rows if r.get("context") == context])]
    bootstrap_by_context = {}
    for context in CONTEXTS:
        values = [float(row["beta_causal_pair"]) for row in pair_rows if row.get("context") == context and row.get("valid_positive_ratios") is True]
        bootstrap_by_context[context] = paired_bootstrap_median(values)
    causal_summaries = _write_causal_outputs(output, pair_rows, leave_rows, bootstrap_by_context)
    exact50 = load_exact50_candidates(roots[3], roots[1])
    snapshots = load_snapshots(roots[1], exact50)
    scaling = average_granularity_scaling(snapshots)
    atomic_csv(output / "average_granularity_scaling.csv", tuple(scaling[0].keys()) if scaling else ("snapshot_target",), scaling)
    exact_scaling = next((row for row in scaling if abs(_float(row.get("snapshot_target"), "snapshot_target") - .5) < 1e-12), {})
    beta_causal_50 = _float(causal_summaries.get(CONTEXT_TASK028_50, {}).get("median_pairwise_beta"), "beta_causal", default=math.nan)
    eps_A = _float(exact_scaling.get("eps_A"), "eps_A", default=machine_safe_epsilon([_float(row["Delta_average"], "Delta_average") for row in exact50]))
    intercept_A = _float(exact_scaling.get("ols_intercept_A"), "intercept_A", default=math.nan)
    components = residual_decomposition(exact50, beta_causal_50, intercept_A, eps_A)
    atomic_csv(output / "causal_granularity_components.csv", tuple(components[0].keys()) if components else ("global_index",), components)
    causal_contexts = {}
    for context in CONTEXTS:
        causal_contexts[context] = {**causal_summaries.get(context, {}), "bootstrap_ci_95": bootstrap_by_context[context].get("ci_95", [math.nan, math.nan])}
    excess = excess_granularity_decomposition(scaling, causal_contexts)
    atomic_json(output / "excess_granularity_decomposition.json", excess)
    alignment, alignment_summary = causal_residual_alignment(pair_rows, recovered_raw, exact50, eps_A)
    atomic_csv(output / "causal_residual_alignment.csv", tuple(alignment[0].keys()) if alignment else ("context", "pair_id"), alignment)
    atomic_json(output / "causal_residual_alignment_summary.json", {"alignment_complete": bool(alignment_summary), "rows": alignment_summary})
    pack_rows = cost_matched_decomposition(pair_rows, recovered_raw, packs, components)
    atomic_csv(output / "cost_matched_decomposition.csv", tuple(pack_rows[0].keys()) if pack_rows else ("context", "pair_id"), pack_rows)
    separation = type_separation(components, {
        "raw_global_p_average": "p_A_global", "raw_log_Delta_average": "log_Delta_average_plus_eps_A",
        "causal_granularity_component": "log_A_granularity_predicted", "residual_component": "log_A_residual_component",
    })
    atomic_csv(output / "decomposed_type_separation.csv", tuple(separation[0].keys()) if separation else ("metric",), separation)
    domain_detail, mixed = domain_residual_analysis(components)
    atomic_csv(output / "domain_residual_analysis.csv", tuple(domain_detail[0].keys()) if domain_detail else ("domain_id",), domain_detail)
    atomic_csv(output / "mixed_domain_residual_analysis.csv", tuple(mixed[0].keys()) if mixed else ("domain_id",), mixed)
    stages = stage_granularity_decomposition(components, eps_A)
    atomic_csv(output / "stage_granularity_decomposition.csv", tuple(stages[0].keys()) if stages else ("stage", "unit_type"), stages)
    comparators = _maybe_task031_comparators(roots[3])
    write_scientific_summary(output, causal_summaries, scaling, excess, alignment_summary, mixed, comparators)
    figures = make_figures(output, pair_rows, bootstrap_by_context, scaling, components, alignment, pack_rows, mixed)

    after = {
        "task028": hash_inputs(roots[0], TASK028_INPUTS),
        "task029": hash_inputs(roots[1], TASK029_INPUTS),
        "task030": hash_inputs(roots[2], TASK030_INPUTS),
        "task031": hash_inputs(roots[3], TASK031_INPUTS),
    }
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("Task028/Task029/Task030/Task031 inputs changed during Task032")
    atomic_json(output / "artifact_identity.json", {
        "status": "PASS", "code_version": CODE_VERSION,
        "hashes_before": before, "hashes_after": after,
        "source_hashes_unchanged": unchanged, "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "model_forward_executed": False, "training_executed": False,
        "fine_tuning_executed": False, "validation_executed": False,
        "selector_replay_executed": False, "source_writes": False,
        "new_pruning_hyperparameters": 0,
    })
    counts = {(str(row.get("context")), str(row.get("intervention"))) for row in recovered_raw}
    intervention_pairs = defaultdict(set)
    for row in recovered_raw:
        intervention_pairs[(str(row.get("context")), str(row.get("intervention")))].add(str(row.get("pair_id")))
    eight_pairs = all(len({str(row.get("pair_id")) for row in pair_rows if row.get("context") == context}) == 8 for context in CONTEXTS)
    attention_single_eight = all(len(intervention_pairs[(context, "attention_single")]) == 8 for context in CONTEXTS)
    ffn_single_eight = all(len(intervention_pairs[(context, "ffn_single")]) == 8 for context in CONTEXTS)
    cost_packs_eight = len({str(row.get("pair_id")) for row in packs}) == 8
    completion = {
        "status": "PASS", "task028_inputs_verified": True, "task029_inputs_verified": True,
        "task030_inputs_verified": True, "task031_inputs_verified": True,
        "source_hashes_unchanged": unchanged, "attention_metadata_recovered": bool(recovered_raw),
        "eight_pairs_per_context": eight_pairs, "attention_single_eight_per_context": attention_single_eight,
        "ffn_single_eight_per_context": ffn_single_eight, "cost_matched_packs_eight": cost_packs_eight,
        "causal_beta_complete": len(pair_rows) == 16 and all(
            sum(row.get("valid_positive_ratios") is True for row in pair_rows if row.get("context") == context) == 8
            for context in CONTEXTS
        ),
        "bootstrap_complete": all(item.get("replicates") == BOOTSTRAP_REPLICATES and item.get("n") == 8 for item in bootstrap_by_context.values()),
        "leave_one_pair_out_complete": len(leave_rows) == 16,
        "average_scaling_complete": bool(scaling), "excess_decomposition_complete": bool(excess),
        "causal_residual_alignment_complete": bool(alignment_summary),
        "cost_matched_decomposition_complete": cost_packs_eight and len(pack_rows) == 16, "type_separation_complete": len(separation) == 4,
        "mixed_domain_analysis_complete": True, "stage_analysis_complete": True,
        "figures_complete": len(figures) == FIGURE_COUNT,
        "scientific_summary_complete": (output / "task032_scientific_summary.md").is_file(),
        "model_forward_executed": False, "training_executed": False, "fine_tuning_executed": False,
        "validation_executed": False, "selector_replay_executed": False, "source_writes": False,
        "new_pruning_hyperparameters": 0, "raw_result_row_count": len(recovered_raw),
        "valid_causal_pair_count": sum(row.get("valid_positive_ratios") is True for row in pair_rows),
        "figure_paths": figures, "intervention_kinds": sorted(counts),
    }
    if not completion_gate(completion):
        completion["status"] = "FAIL"
        atomic_json(output / "task032_completion.json", completion)
        raise RuntimeError("Task032 completion gate failed")
    atomic_json(output / "task032_completion.json", completion)
    return {"causal": causal_summaries, "scaling": scaling, "excess": excess,
            "alignment": alignment_summary, "completion": completion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "run", "completion"), required=True)
    parser.add_argument("--task028-root", type=Path)
    parser.add_argument("--task029-root", type=Path)
    parser.add_argument("--task030-root", type=Path)
    parser.add_argument("--task031-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _required(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    names = ("task028_root", "task029_root", "task030_root", "task031_root")
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    return tuple(getattr(args, name) for name in names)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "completion":
        completion = read_json(args.output_dir / "task032_completion.json")
        if not completion_gate(completion):
            raise RuntimeError("Task032 completion gate failed")
        print("Task032 completion: PASS", flush=True)
        return 0
    roots = _required(args)
    if args.mode == "identity":
        verify_identity(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], output_dir=args.output_dir)
        print("Task032 identity: PASS", flush=True)
        return 0
    result = run_diagnosis(task028_root=roots[0], task029_root=roots[1], task030_root=roots[2], task031_root=roots[3], output_dir=args.output_dir)
    print(json.dumps({"status": result["completion"]["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

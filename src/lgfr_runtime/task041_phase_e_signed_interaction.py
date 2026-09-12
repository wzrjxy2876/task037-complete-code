#!/usr/bin/env python3
"""Offline Task041 Phase E signed-interaction failure diagnosis; no model runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SPANS = (1, 2, 4, 8, 16)
SAME_TYPE_DOMAINS = ("269", "400", "415", "102", "103", "113", "76")
MIXED_DOMAINS = ("271", "297")
SPECIAL_DOMAINS = ("102", "103", "113", "269")
DAMAGE = ("mean_true_class_logit_drop", "mean_cross_entropy_increase", "prediction_flip_rate")
DIAGNOSTICS = ("mean_sign_balance", "cancellation_ratio", "signed_profile_norm", "RMS_profile_norm", "R_MCTC")
BASELINES = ("mean_abs_d_original", "G_RMS", "old_HTOR", "PTR", "corrected_pairwise_best_E", "R_MCTC")
EPS = 1e-12
N_UNITS, N_VIDEOS, N_PAIRS = 29, 3, 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase_c_raw", required=True)
    p.add_argument("--phase_d1_raw", required=True)
    p.add_argument("--task041_output_dir", required=True)
    p.add_argument("--phase_d_output_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "no rows"}]
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite " + name)
    return result


def unit_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (str(row["candidate_task040_global_index"]), str(row["candidate_layer_name"]),
            str(row["candidate_unit_type"]), str(row["candidate_unit_index"]))


def unit_identity(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return (str(row["candidate_task037_global_index"]), str(row["candidate_task040_global_index"]),
            str(row["candidate_layer_name"]), str(row["candidate_unit_type"]),
            str(row["candidate_unit_index"]), str(row["domain_id"]))


def read_unique(path: Path, required: Iterable[str], key: Sequence[str]) -> List[Dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("empty required input: " + str(path))
    missing = set(required).difference(rows[0])
    if missing:
        raise ValueError("missing input fields: " + repr(sorted(missing)))
    keys = [tuple(row[k] for k in key) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate rows in " + str(path))
    return rows


def load_frozen(path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    rows = read_unique(
        path / "task041_masking_damage.csv",
        ("domain_id", "candidate_task037_global_index", "candidate_task040_global_index",
         "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
         "candidate_stage", "R_MCTC", *("g_span_" + str(s) for s in SPANS)),
        ("candidate_task037_global_index", "candidate_task040_global_index"),
    )
    if len(rows) != N_UNITS:
        raise ValueError("Task041 frozen table must contain exactly 29 units")
    result: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    seen37 = set()
    for raw in rows:
        row: Dict[str, Any] = dict(raw)
        row["candidate_task037_global_index"] = str(raw["candidate_task037_global_index"])
        row["candidate_task040_global_index"] = str(raw["candidate_task040_global_index"])
        row["candidate_unit_index"] = int(raw["candidate_unit_index"])
        row["candidate_stage"] = int(raw["candidate_stage"])
        row["domain_id"] = str(raw["domain_id"])
        row["R_MCTC"] = number(raw["R_MCTC"], "R_MCTC")
        for s in SPANS:
            row["g_span_" + str(s)] = number(raw["g_span_" + str(s)], "frozen g")
        k = unit_key(row)
        if k in result or row["candidate_task037_global_index"] in seen37:
            raise ValueError("Task041 unit mapping is not one-to-one")
        result[k] = row
        seen37.add(row["candidate_task037_global_index"])
    domains = {r["domain_id"] for r in result.values()}
    if domains != set(SAME_TYPE_DOMAINS).union(MIXED_DOMAINS):
        raise ValueError("Task041 frozen domain set mismatch")
    return result


def load_damage(path: Path, frozen: Mapping[Tuple[str, str, str, str], Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows = read_unique(
        path,
        ("domain_id", "candidate_task037_global_index", "candidate_task040_global_index",
         "candidate_layer_name", "candidate_unit_type", "candidate_unit_index", *DAMAGE),
        ("candidate_task037_global_index", "candidate_task040_global_index"),
    )
    if len(rows) != N_UNITS:
        raise ValueError("full-validation damage must contain exactly 29 units")
    out = {}
    for raw in rows:
        k = unit_key(raw)
        if k not in frozen or unit_identity(raw) != unit_identity(frozen[k]):
            raise ValueError("full-validation damage identity mismatch")
        row: Dict[str, Any] = dict(raw)
        row["candidate_task037_global_index"] = str(raw["candidate_task037_global_index"])
        for f in DAMAGE:
            row[f] = number(raw[f], f)
        out[row["candidate_task037_global_index"]] = row
    return out


def parse_raw(raw: Mapping[str, str], source: str,
              frozen: Mapping[Tuple[str, str, str, str], Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    k = (str(raw["unit_global_index"]), str(raw["layer_name"]),
         str(raw["unit_type"]), str(raw["unit_index"]))
    if k not in frozen:
        return None
    unit = frozen[k]
    if raw.get("stage", ""):
        if int(raw["stage"]) != int(unit["candidate_stage"]):
            raise ValueError("raw stage differs from frozen identity")
    else:
        m = re.search(r"layers\.(\d+)\.", str(raw["layer_name"]))
        if not m or int(m.group(1)) != int(unit["candidate_stage"]):
            raise ValueError("cannot verify Phase D.1 stage from layer path")
    span, pair, level = int(raw["block_size"]), int(raw["pair_index"]), int(raw["level"])
    if span not in SPANS or pair not in range(N_PAIRS) or level != int(round(math.log(span, 2))):
        raise ValueError("invalid span, level, or intervention index")
    a, b, c, d = (np.float64(number(raw[k0], k0)) for k0 in
                  ("z_true_original", "z_true_original_masked", "z_true_intervened", "z_true_intervened_masked"))
    signed = float(a - b - c + d)
    if raw.get("C_interaction", "") != "":
        if signed != number(raw["C_interaction"], "stored C_interaction"):
            raise ValueError("Phase C reconstructed C differs from stored C_interaction")
    return {
        "source": source, "candidate_task037_global_index": unit["candidate_task037_global_index"],
        "candidate_task040_global_index": unit["candidate_task040_global_index"],
        "candidate_layer_name": unit["candidate_layer_name"], "candidate_unit_type": unit["candidate_unit_type"],
        "candidate_unit_index": unit["candidate_unit_index"], "candidate_stage": unit["candidate_stage"],
        "domain_id": unit["domain_id"], "video_index": int(raw["video_index"]),
        "video_id": str(raw["video_id"]), "span": span, "pair_index": pair, "C_signed": signed,
    }


def reconstruct(phase_c: Path, phase_d1: Path,
                frozen: Mapping[Tuple[str, str, str, str], Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    source_units = {"phase_c": set(), "phase_d1": set()}
    metadata: Dict[str, Any] = {}
    seen = set()
    video_by_unit: Dict[str, Dict[int, str]] = defaultdict(dict)
    for source, path, has_saved_c in (("phase_c", phase_c, True), ("phase_d1", phase_d1, False)):
        rows = read_csv(path)
        required = {"video_index", "video_id", "unit_global_index", "layer_name", "unit_type", "unit_index",
                    "level", "block_size", "pair_index", "z_true_original", "z_true_original_masked",
                    "z_true_intervened", "z_true_intervened_masked"}
        if has_saved_c:
            required.add("C_interaction")
        if not rows or not required.issubset(rows[0]):
            raise ValueError("raw factorial input missing required stored logits/identity")
        n = 0
        for raw in rows:
            item = parse_raw(raw, source, frozen)
            if item is None:
                continue
            n += 1
            uid = item["candidate_task037_global_index"]
            source_units[source].add(uid)
            old = video_by_unit[uid].get(item["video_index"])
            if old is not None and old != item["video_id"]:
                raise ValueError("one video index maps to multiple video ids")
            video_by_unit[uid][item["video_index"]] = item["video_id"]
            key = (uid, item["video_index"], item["video_id"], item["span"], item["pair_index"])
            if key in seen:
                raise ValueError("duplicate candidate/video/span/intervention raw record")
            seen.add(key)
            records.append(item)
        metadata[source] = {"path": str(path.resolve()), "sha256": file_sha(path),
                            "raw_rows": len(rows), "selected_rows": n,
                            "selected_units": len(source_units[source])}
    if source_units["phase_c"].intersection(source_units["phase_d1"]):
        raise ValueError("Phase C and D.1 overlap in frozen candidate units")
    expected = {str(r["candidate_task037_global_index"]) for r in frozen.values()}
    if source_units["phase_c"].union(source_units["phase_d1"]) != expected:
        raise ValueError("Phase C + D.1 do not exactly partition the 29 frozen units")
    videos = {(r["video_index"], r["video_id"]) for r in records}
    if len(videos) != N_VIDEOS or len({v[0] for v in videos}) != N_VIDEOS:
        raise ValueError("raw inputs must use exactly the same three videos")
    per_unit_span: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        per_unit_span[(r["candidate_task037_global_index"], r["span"])].append(r)
    for unit in frozen.values():
        uid = str(unit["candidate_task037_global_index"])
        if len(video_by_unit[uid]) != N_VIDEOS:
            raise ValueError("candidate does not have exactly three videos")
        for span in SPANS:
            rows = per_unit_span[(uid, span)]
            keys = {(r["video_index"], r["pair_index"]) for r in rows}
            expected_keys = {(v, p) for v in video_by_unit[uid] for p in range(N_PAIRS)}
            if len(rows) != N_VIDEOS * N_PAIRS or keys != expected_keys:
                raise ValueError("candidate/span lacks exact 3 videos x 16 interventions")
    metadata["unit_partition"] = {
        "phase_c_unit_count": len(source_units["phase_c"]),
        "phase_d1_unit_count": len(source_units["phase_d1"]),
        "phase_c_task037_indices": sorted(source_units["phase_c"], key=int),
        "phase_d1_task037_indices": sorted(source_units["phase_d1"], key=int),
        "selected_interaction_rows": len(records),
        "expected_interaction_rows": N_UNITS * len(SPANS) * N_VIDEOS * N_PAIRS,
        "video_count": len(videos), "interventions_per_video_span": N_PAIRS,
    }
    if len(records) != N_UNITS * len(SPANS) * N_VIDEOS * N_PAIRS:
        raise ValueError("raw interaction total is not 29 x 5 x 3 x 16")
    return records, metadata


def make_span_statistics(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    identity = {}
    for r in records:
        uid, span = str(r["candidate_task037_global_index"]), int(r["span"])
        values[(uid, span)].append(float(r["C_signed"]))
        identity[uid] = r
    out = []
    for (uid, span), raw_values in sorted(values.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        x = np.asarray(raw_values, dtype=np.float64)
        mu = float(x.mean(dtype=np.float64))
        mean_abs = float(np.abs(x).mean(dtype=np.float64))
        rms = float(np.sqrt(np.square(x, dtype=np.float64).mean(dtype=np.float64)))
        it = identity[uid]
        out.append({
            "candidate_task037_global_index": uid,
            "candidate_task040_global_index": it["candidate_task040_global_index"],
            "candidate_layer_name": it["candidate_layer_name"], "candidate_unit_type": it["candidate_unit_type"],
            "candidate_unit_index": it["candidate_unit_index"], "candidate_stage": it["candidate_stage"],
            "domain_id": it["domain_id"], "span": span, "interaction_count": len(x),
            "mu": mu, "mean_abs": mean_abs, "rms": rms,
            "std_population": float(x.std(dtype=np.float64, ddof=0)),
            "positive_fraction": float(np.count_nonzero(x > 0) / len(x)),
            "negative_fraction": float(np.count_nonzero(x < 0) / len(x)),
            "zero_fraction": float(np.count_nonzero(x == 0) / len(x)),
            "sign_balance": float(abs(mu) / (mean_abs + EPS)),
        })
    if len(out) != N_UNITS * len(SPANS):
        raise ValueError("expected 145 per-unit/per-span rows")
    return out


def make_unit_summary(span_rows: Sequence[Mapping[str, Any]],
                      frozen: Mapping[Tuple[str, str, str, str], Mapping[str, Any]],
                      damage: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_uid: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for r in span_rows:
        by_uid[str(r["candidate_task037_global_index"])][int(r["span"])] = r
    frozen_uid = {str(r["candidate_task037_global_index"]): r for r in frozen.values()}
    out = []
    for uid in sorted(frozen_uid, key=int):
        u = frozen_uid[uid]
        if set(by_uid[uid]) != set(SPANS):
            raise ValueError("unit is missing one or more spans")
        mus = np.asarray([by_uid[uid][s]["mu"] for s in SPANS], dtype=np.float64)
        rmss = np.asarray([by_uid[uid][s]["rms"] for s in SPANS], dtype=np.float64)
        balances = np.asarray([by_uid[uid][s]["sign_balance"] for s in SPANS], dtype=np.float64)
        row: Dict[str, Any] = {
            "candidate_task037_global_index": uid,
            "candidate_task040_global_index": u["candidate_task040_global_index"],
            "candidate_layer_name": u["candidate_layer_name"], "candidate_unit_type": u["candidate_unit_type"],
            "candidate_unit_index": u["candidate_unit_index"], "candidate_stage": u["candidate_stage"],
            "domain_id": u["domain_id"], "R_MCTC": u["R_MCTC"],
            "mean_sign_balance": float(balances.mean()), "min_sign_balance": float(balances.min()),
            "max_sign_balance": float(balances.max()), "signed_profile_norm": float(np.linalg.norm(mus)),
            "RMS_profile_norm": float(np.linalg.norm(rmss)),
            "cancellation_ratio": float(np.linalg.norm(mus) / (np.linalg.norm(rmss) + EPS)),
            **{f: float(damage[uid][f]) for f in DAMAGE},
        }
        for s in SPANS:
            row["frozen_g_span_" + str(s)] = float(u["g_span_" + str(s)])
            for metric in ("mu", "mean_abs", "rms", "std_population", "positive_fraction",
                           "negative_fraction", "zero_fraction", "sign_balance"):
                row[metric + "_span_" + str(s)] = float(by_uid[uid][s][metric])
        out.append(row)
    if len(out) != N_UNITS:
        raise ValueError("unit summary count mismatch")
    return out


def midranks(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return None if den == 0 else float(np.dot(a, b) / den)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(midranks(x), midranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    con = dis = tx = ty = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            dx, dy = a[i] - a[j], b[i] - b[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                con += 1
            else:
                dis += 1
    den = math.sqrt((con + dis + tx) * (con + dis + ty))
    return None if den == 0 else float((con - dis) / den)


def make_correlations(units: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in units:
        grouped[str(row["domain_id"])].append(row)
    out = []
    for domain in sorted(grouped, key=int):
        for diagnostic in DIAGNOSTICS:
            for damage_metric in DAMAGE:
                rows = grouped[domain]
                out.append({
                    "scope": "within_domain", "domain_id": domain, "diagnostic": diagnostic,
                    "damage_metric": damage_metric, "unit_count": len(rows),
                    "spearman": spearman([r[diagnostic] for r in rows], [r[damage_metric] for r in rows]),
                    "kendall_tau_b": kendall_tau_b([r[diagnostic] for r in rows], [r[damage_metric] for r in rows]),
                })
    scopes = {
        "domain_balanced_all_9": sorted(grouped, key=int),
        "domain_balanced_same_type_7": list(SAME_TYPE_DOMAINS),
        "domain_balanced_mixed_2": list(MIXED_DOMAINS),
    }
    for scope, domains in scopes.items():
        for diagnostic in DIAGNOSTICS:
            for damage_metric in DAMAGE:
                pairs = []
                for domain in domains:
                    rows = grouped[domain]
                    rho = spearman([r[diagnostic] for r in rows], [r[damage_metric] for r in rows])
                    tau = kendall_tau_b([r[diagnostic] for r in rows], [r[damage_metric] for r in rows])
                    if rho is not None and tau is not None:
                        pairs.append((rho, tau))
                out.append({
                    "scope": scope, "domain_id": "", "diagnostic": diagnostic, "damage_metric": damage_metric,
                    "unit_count": sum(len(grouped[d]) for d in domains), "valid_domain_count": len(pairs),
                    "spearman": float(np.mean([p[0] for p in pairs])) if pairs else None,
                    "kendall_tau_b": float(np.mean([p[1] for p in pairs])) if pairs else None,
                    "mean_absolute_within_domain_spearman": float(np.mean([abs(p[0]) for p in pairs])) if pairs else None,
                    "mean_absolute_within_domain_kendall": float(np.mean([abs(p[1]) for p in pairs])) if pairs else None,
                })
    for diagnostic in DIAGNOSTICS:
        for damage_metric in DAMAGE:
            out.append({
                "scope": "global_secondary", "domain_id": "", "diagnostic": diagnostic,
                "damage_metric": damage_metric, "unit_count": len(units),
                "spearman": spearman([r[diagnostic] for r in units], [r[damage_metric] for r in units]),
                "kendall_tau_b": kendall_tau_b([r[diagnostic] for r in units], [r[damage_metric] for r in units]),
            })
    same = [r for r in out if r["scope"] == "domain_balanced_same_type_7"]
    return out, same


def load_low_high(task041_dir: Path, units: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    by_uid = {str(r["candidate_task037_global_index"]): r for r in units}
    selected: Dict[str, Dict[str, str]] = defaultdict(dict)
    for raw in read_csv(task041_dir / "task041_oracle_candidates.csv"):
        domain, role = str(raw["domain_id"]), str(raw["candidate_role"])
        uid = str(raw["candidate_task037_global_index"])
        if role not in ("low", "high") or role in selected[domain] or uid not in by_uid:
            raise ValueError("invalid frozen low/high candidate")
        if unit_identity(raw) != unit_identity(by_uid[uid]) or float(raw["R_MCTC"]) != float(by_uid[uid]["R_MCTC"]):
            raise ValueError("frozen low/high identity/score changed")
        selected[domain][role] = uid
    expected = set(SAME_TYPE_DOMAINS).union(MIXED_DOMAINS)
    if set(selected) != expected or any(set(v) != {"low", "high"} for v in selected.values()):
        raise ValueError("expected one frozen low/high pair in each of nine domains")
    for domain, pair in selected.items():
        if float(by_uid[pair["high"]]["R_MCTC"]) <= float(by_uid[pair["low"]]["R_MCTC"]):
            raise ValueError("frozen candidate roles are not ordered by R_MCTC")
    return dict(selected)


def make_failure_cases(units: Sequence[Mapping[str, Any]], low_high: Mapping[str, Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_uid = {str(r["candidate_task037_global_index"]): r for r in units}
    pairs, out = [], []
    for domain in SAME_TYPE_DOMAINS:
        low, high = (by_uid[low_high[domain][role]] for role in ("low", "high"))
        delta = {m: float(high[m]) - float(low[m]) for m in DAMAGE}
        failed_logit, failed_ce = delta[DAMAGE[0]] <= 0, delta[DAMAGE[1]] <= 0
        pair = {
            "domain_id": domain, "low_task037_global_index": low["candidate_task037_global_index"],
            "high_task037_global_index": high["candidate_task037_global_index"],
            "low_R_MCTC": low["R_MCTC"], "high_R_MCTC": high["R_MCTC"],
            "logit_high_minus_low": delta[DAMAGE[0]], "ce_high_minus_low": delta[DAMAGE[1]],
            "flip_high_minus_low": delta[DAMAGE[2]],
            "RMS_norm_high_minus_low": high["RMS_profile_norm"] - low["RMS_profile_norm"],
            "signed_norm_high_minus_low": high["signed_profile_norm"] - low["signed_profile_norm"],
            "mean_sign_balance_high_minus_low": high["mean_sign_balance"] - low["mean_sign_balance"],
            "cancellation_ratio_high_minus_low": high["cancellation_ratio"] - low["cancellation_ratio"],
            "logit_ordering_failed": failed_logit, "CE_ordering_failed": failed_ce,
            "any_signed_damage_ordering_failed": failed_logit or failed_ce,
        }
        pairs.append(pair)
        if not pair["any_signed_damage_ordering_failed"]:
            continue
        for role, row in (("low", low), ("high", high)):
            item = dict(pair)
            item.update({"case_role": role, **{k: row[k] for k in (
                "candidate_task037_global_index", "candidate_task040_global_index", "candidate_layer_name",
                "candidate_unit_type", "candidate_unit_index", "candidate_stage", "R_MCTC",
                "mean_sign_balance", "cancellation_ratio", "signed_profile_norm", "RMS_profile_norm",
                *DAMAGE)}})
            for s in SPANS:
                item["frozen_g_span_" + str(s)] = row["frozen_g_span_" + str(s)]
                for f in ("mu", "rms", "mean_abs", "std_population", "sign_balance",
                          "positive_fraction", "negative_fraction", "zero_fraction"):
                    item[f + "_span_" + str(s)] = row[f + "_span_" + str(s)]
            out.append(item)
    return out, pairs


def baseline_flip(task041_dir: Path, units: Sequence[Mapping[str, Any]], low_high: Mapping[str, Mapping[str, str]]) -> List[Dict[str, Any]]:
    by_uid = {str(r["candidate_task037_global_index"]): r for r in units}
    score_rows = read_csv(task041_dir / "task041_baseline_comparison.csv")
    if len(score_rows) != 18:
        raise ValueError("baseline flip audit must use the existing 18 candidates")
    scores: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in score_rows:
        domain, role, uid = str(row["domain_id"]), str(row["candidate_role"]), str(row["candidate_task037_global_index"])
        if role not in ("low", "high") or (domain, role) in scores or uid != low_high[domain][role]:
            raise ValueError("baseline candidate roles/identities changed")
        unit = by_uid[uid]
        if (str(row["candidate_task040_global_index"]) != str(unit["candidate_task040_global_index"])
                or str(row["layer_name"]) != str(unit["candidate_layer_name"])
                or str(row["unit_type"]) != str(unit["candidate_unit_type"])
                or int(row["unit_index"]) != int(unit["candidate_unit_index"])
                or float(row["R_MCTC"]) != float(unit["R_MCTC"])):
            raise ValueError("frozen baseline comparison identity mismatch")
        scores[(domain, role)] = row
    out = []
    for criterion in BASELINES:
        correct = incorrect = ties = 0
        rhos, taus, pooled_x, pooled_y, details = [], [], [], [], []
        for domain in sorted(low_high, key=int):
            lo, hi = low_high[domain]["low"], low_high[domain]["high"]
            x0 = number(scores[(domain, "low")][criterion], criterion)
            x1 = number(scores[(domain, "high")][criterion], criterion)
            y0, y1 = float(by_uid[lo]["prediction_flip_rate"]), float(by_uid[hi]["prediction_flip_rate"])
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 or dy == 0:
                ties += 1
                order = "tie"
            elif dx * dy > 0:
                correct += 1
                order = "correct"
            else:
                incorrect += 1
                order = "incorrect"
            rho, tau = spearman([x0, x1], [y0, y1]), kendall_tau_b([x0, x1], [y0, y1])
            if rho is not None:
                rhos.append(rho)
            if tau is not None:
                taus.append(tau)
            pooled_x.extend([x0, x1]); pooled_y.extend([y0, y1])
            details.append({"domain_id": domain, "score_delta_high_minus_low": dx,
                            "flip_delta_high_minus_low": dy, "ordering": order,
                            "spearman": rho, "kendall_tau_b": tau})
        out.append({
            "criterion": criterion, "domain_count": 9, "ordering_correct_count": correct,
            "ordering_incorrect_count": incorrect, "ordering_tie_count": ties,
            "domain_balanced_spearman": float(np.mean(rhos)) if rhos else None,
            "domain_balanced_kendall_tau_b": float(np.mean(taus)) if taus else None,
            "global_secondary_spearman_n18": spearman(pooled_x, pooled_y),
            "global_secondary_kendall_tau_b_n18": kendall_tau_b(pooled_x, pooled_y),
            "domain_pair_details": json.dumps(details, sort_keys=True),
        })
    return out


def group_summary(units: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    dims = {
        "by_unit_type": lambda r: str(r["candidate_unit_type"]),
        "by_stage": lambda r: str(r["candidate_stage"]),
        "by_domain": lambda r: str(r["domain_id"]),
        "mixed_domain_by_unit_type": lambda r: (str(r["domain_id"]) + ":" + str(r["candidate_unit_type"])
                                                  if str(r["domain_id"]) in MIXED_DOMAINS else ""),
    }
    metrics = ("mean_sign_balance", "cancellation_ratio", "signed_profile_norm", "RMS_profile_norm", *DAMAGE)
    output = {}
    for name, keyfn in dims.items():
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in units:
            key = keyfn(row)
            if key:
                groups[key].append(row)
        table = []
        for key, rows in sorted(groups.items()):
            item: Dict[str, Any] = {"group": key, "unit_count": len(rows)}
            for metric in metrics:
                vals = [float(r[metric]) for r in rows]
                item["mean_" + metric] = float(np.mean(vals))
                item["median_" + metric] = float(np.median(vals))
            table.append(item)
        output[name] = table
    return output


def vec(row: Mapping[str, Any], metric: str, frozen: bool = False) -> str:
    prefix = "frozen_g_span_" if frozen else metric + "_span_"
    return "[" + ", ".join(format(float(row[prefix + str(s)]), ".4f") for s in SPANS) + "]"


def make_report(path: Path, summary: Mapping[str, Any], units: Sequence[Mapping[str, Any]],
                correlations: Sequence[Mapping[str, Any]], same: Sequence[Mapping[str, Any]],
                failures: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
                baseline: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    lines = [
        "# Task041 Phase E — Signed Temporal Interaction Failure Diagnosis", "",
        "Offline-only; each interaction was reconstructed from stored logits as a - b - c + d in float64. No model inference/GPU, intervention rerun, BMS recomputation, pruning, fine-tuning, MCTC redesign, or new pruning score.",
        "",
        "- Frozen Task041 identities: 29/29; full-validation damage joined exactly.",
        "- Selected factorial records: " + str(summary["raw_records"]["unit_partition"]["selected_interaction_rows"]) +
        " / " + str(summary["raw_records"]["unit_partition"]["expected_interaction_rows"]) +
        "; Phase C/D.1 unit partition " + str(summary["raw_records"]["unit_partition"]["phase_c_unit_count"]) +
        " / " + str(summary["raw_records"]["unit_partition"]["phase_d1_unit_count"]) + ".",
        "- Epsilon for sign balance: 1e-12; SD convention: population (ddof=0).",
        "",
        "## A. RMS and cancellation",
        "",
        "Across units, mean sign-balance mean/median/range = " +
        " / ".join(format(float(summary["cancellation"][k]), ".4f") for k in
                   ("mean_sign_balance_mean", "mean_sign_balance_median", "mean_sign_balance_min", "mean_sign_balance_max")) +
        "; cancellation-ratio mean/median/range = " +
        " / ".join(format(float(summary["cancellation"][k]), ".4f") for k in
                   ("cancellation_ratio_mean", "cancellation_ratio_median", "cancellation_ratio_min", "cancellation_ratio_max")) +
        ". Read these with the complete per-unit and per-span rows; sign_balance near zero reflects direction cancellation.",
        "",
        "## B. Descriptive type, stage, and domain audit",
        "",
        "| Type | n | mean sign balance | median sign balance | mean cancellation ratio | mean signed norm | mean RMS norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in groups["by_unit_type"]:
        lines.append("| " + str(r["group"]) + " | " + str(r["unit_count"]) + " | " +
                     " | ".join(format(float(r[k]), ".4f") for k in
                                ("mean_mean_sign_balance", "median_mean_sign_balance",
                                 "mean_cancellation_ratio", "mean_signed_profile_norm", "mean_RMS_profile_norm")) + " |")
    lines += [
        "",
        "Type/stage/domain rows are descriptive strata only; no type coefficient or pooled selector was fit. Full group summaries are in the JSON summary.",
        "",
        "## C–E. Same-type primary associations",
        "",
        "| Diagnostic | Outcome | domain-balanced Spearman | Kendall tau-b | mean absolute within-domain Spearman | valid domains |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in same:
        lines.append("| " + str(r["diagnostic"]) + " | " + str(r["damage_metric"]) + " | " +
                     str(r.get("spearman")) + " | " + str(r.get("kendall_tau_b")) + " | " +
                     str(r.get("mean_absolute_within_domain_spearman")) + " | " +
                     str(r.get("valid_domain_count")) + " |")
    lines += [
        "",
        "Raw signed correlation coefficients are retained; mean absolute within-domain coefficients are supplemental descriptive association strength, not a pruning score. Global pooled correlations are secondary in the correlation CSV only.",
        "",
        "### Same-type low/high MCTC failure cases",
        "",
        "| Domain | high-R RMS norm > low | high-R sign balance < low | high-R cancellation ratio < low | logit failed | CE failed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        if p["any_signed_damage_ordering_failed"]:
            lines.append("| " + str(p["domain_id"]) + " | " +
                         str(p["RMS_norm_high_minus_low"] > 0) + " | " +
                         str(p["mean_sign_balance_high_minus_low"] < 0) + " | " +
                         str(p["cancellation_ratio_high_minus_low"] < 0) + " | " +
                         str(p["logit_ordering_failed"]) + " | " + str(p["CE_ordering_failed"]) + " |")
    lines += [
        "",
        "Detailed low/high rows (both frozen units, five frozen g values, five signed means, five RMS values, five balance values, and full-validation damage) are in task041_mctc_failure_cases.csv.",
        "",
        "## F. Fixed-cardinality intervention",
        "",
        "The factorial measurements remain a diagnostic of temporal interaction: the raw stored logits preserve direction and permit cancellation analysis. This does not reinstate R_MCTC as a standalone pruning selector.",
        "",
        "## G. Next decision",
        "",
        "Recommendation: **" + str(summary["recommendation"]) + "**. No new method is created or launched by this diagnosis.",
        "",
        "## Complete reversal audit: domains 102 / 103 / 113 / 269",
        "",
        "Each vector is ordered by span [1,2,4,8,16]. Fractions use 48 interactions per span.",
        "",
        "| Domain | Task037 | Unit | R_MCTC | mu | raw RMS | frozen g | mean abs | SD | balance | positive | negative | zero | logit | CE | flip |",
        "|---:|---:|---|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in units:
        if str(r["domain_id"]) not in SPECIAL_DOMAINS:
            continue
        vectors = [vec(r, f) for f in ("mu", "rms")]
        vectors += [vec(r, "frozen_g", True)]
        vectors += [vec(r, f) for f in ("mean_abs", "std_population", "sign_balance",
                                        "positive_fraction", "negative_fraction", "zero_fraction")]
        lines.append("| " + str(r["domain_id"]) + " | " + str(r["candidate_task037_global_index"]) +
                     " | " + str(r["candidate_unit_type"]) + " " + str(r["candidate_unit_index"]) +
                     " | " + format(float(r["R_MCTC"]), ".4f") + " | " + " | ".join(vectors) +
                     " | " + " | ".join(format(float(r[f]), ".4f") for f in DAMAGE) + " |")
    lines += [
        "",
        "The observed signed/RMS contrasts can make discarded sign information a plausible explanation only where unit-level patterns align with the frozen reversal. They do not establish causality.",
        "",
        "## Mixed domains 271 / 297 (separate descriptive audit; not selector validation)",
        "",
        "| Domain | Task037 | Type/index | R_MCTC | balance | cancel ratio | signed norm | RMS norm | logit | CE | flip |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in units:
        if str(r["domain_id"]) in MIXED_DOMAINS:
            lines.append("| " + str(r["domain_id"]) + " | " + str(r["candidate_task037_global_index"]) +
                         " | " + str(r["candidate_unit_type"]) + "/" + str(r["candidate_unit_index"]) +
                         " | " + " | ".join(format(float(r[k]), ".4f") for k in
                            ("R_MCTC", "mean_sign_balance", "cancellation_ratio", "signed_profile_norm",
                             "RMS_profile_norm", *DAMAGE)) + " |")
    lines += [
        "",
        "## Baseline flip-rate comparison (frozen 18 low/high candidates)",
        "",
        "| Criterion | correct | incorrect | tie | domain-balanced rho | domain-balanced tau-b |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in baseline:
        lines.append("| " + str(r["criterion"]) + " | " + str(r["ordering_correct_count"]) +
                     " | " + str(r["ordering_incorrect_count"]) + " | " + str(r["ordering_tie_count"]) +
                     " | " + str(r["domain_balanced_spearman"]) + " | " +
                     str(r["domain_balanced_kendall_tau_b"]) + " |")
    lines += [
        "",
        "## Answers A–G",
        "",
        "A/B/C/D/E are read from the signed-cancellation distributions, descriptive type strata, same-type correlations and paired reversals above. The association evidence is descriptive (29 units; seven same-type domains) and must not be interpreted causally.",
        "",
        "All eight Phase E artifacts are newly written in the Phase E directory. Previous Task040/Task041 outputs are inputs only and were not overwritten.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    phase_c, phase_d1 = Path(args.phase_c_raw).resolve(), Path(args.phase_d1_raw).resolve()
    task041 = Path(args.task041_output_dir).resolve()
    phase_d = Path(args.phase_d_output_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    names = ("task041_signed_span_statistics.csv", "task041_signed_unit_summary.csv",
             "task041_signed_vs_fullval_damage.csv", "task041_signed_same_type_domains.csv",
             "task041_mctc_failure_cases.csv", "task041_baseline_flip_comparison.csv",
             "task041_phase_e_report.md", "task041_phase_e_summary.json")
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir / n).exists() for n in names):
        raise FileExistsError("refusing to overwrite Phase E output")
    frozen = load_frozen(task041)
    damage = load_damage(phase_d / "task041_fullval_unit_damage.csv", frozen)
    records, raw_meta = reconstruct(phase_c, phase_d1, frozen)
    span_rows = make_span_statistics(records)
    units = make_unit_summary(span_rows, frozen, damage)
    correlations, same = make_correlations(units)
    low_high = load_low_high(task041, units)
    failures, pairs = make_failure_cases(units, low_high)
    baseline = baseline_flip(task041, units, low_high)
    groups = group_summary(units)
    balances = [float(r["mean_sign_balance"]) for r in units]
    ratios = [float(r["cancellation_ratio"]) for r in units]
    summary: Dict[str, Any] = {
        "task": "task041_phase_e_signed_temporal_interaction_failure_diagnosis",
        "phase": "E", "offline_only": True, "model_inference": False, "gpu_used": False,
        "temporal_intervention_rerun": False, "bms_recomputed": False,
        "pruning": False, "finetuning": False, "mctc_redesigned": False,
        "new_pruning_score_created": False, "frozen_unit_count": len(units),
        "fullval_damage_join_exact": len(damage) == N_UNITS,
        "identity_fields_verified": ["Task037 global_index", "Task040 global_index",
                                      "layer_name", "unit_type", "unit_index", "domain_id"],
        "raw_records": raw_meta,
        "cancellation": {
            "epsilon": EPS, "std_convention": "population ddof=0",
            "mean_sign_balance_mean": float(np.mean(balances)),
            "mean_sign_balance_median": float(np.median(balances)),
            "mean_sign_balance_min": float(np.min(balances)),
            "mean_sign_balance_max": float(np.max(balances)),
            "cancellation_ratio_mean": float(np.mean(ratios)),
            "cancellation_ratio_median": float(np.median(ratios)),
            "cancellation_ratio_min": float(np.min(ratios)),
            "cancellation_ratio_max": float(np.max(ratios)),
        },
        "group_summaries": groups, "same_type_domain_balanced_correlations": same,
        "correlations": correlations, "same_type_failure_pairs": pairs,
        "same_type_failure_case_count": len(failures) // 2,
        "baseline_flip_comparison": baseline,
        "recommendation": "RETAIN MCTC ONLY AS DIAGNOSTIC",
        "recommendation_note": "Conservative default pending direct inspection of signed-vs-RMS same-type evidence; no new method is automatically generated.",
        "input_artifacts": {
            "phase_c_raw": str(phase_c), "phase_d1_raw": str(phase_d1),
            "task041_frozen_units": str(task041 / "task041_masking_damage.csv"),
            "task041_fullval_damage": str(phase_d / "task041_fullval_unit_damage.csv"),
        },
    }
    if len(units) != N_UNITS or len(span_rows) != N_UNITS * len(SPANS) or not summary["fullval_damage_join_exact"]:
        raise ValueError("Phase E identity/count gate failed")
    write_csv(outdir / names[0], span_rows)
    write_csv(outdir / names[1], units)
    write_csv(outdir / names[2], correlations)
    write_csv(outdir / names[3], same)
    write_csv(outdir / names[4], failures)
    write_csv(outdir / names[5], baseline)
    make_report(outdir / names[6], summary, units, correlations, same, failures, pairs, baseline, groups)
    write_json(outdir / names[7], summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

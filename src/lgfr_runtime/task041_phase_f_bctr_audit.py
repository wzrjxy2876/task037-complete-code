from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.optimize import lsq_linear
from scipy.stats import kendalltau, rankdata, spearmanr


SPANS = (1, 2, 4, 8, 16)
FROZEN_DOMAINS = ("271", "297", "269", "400", "415", "102", "103", "113", "76")
SAME_TYPE_DOMAINS = ("269", "400", "415", "102", "103", "113", "76")
MIXED_DOMAINS = ("271", "297")
EPS = 1e-12
N_UNITS = 29
BASELINE_CRITERIA = (
    "mean_abs_d_original",
    "G_RMS",
    "PTR",
    "corrected_pairwise_best_E",
    "R_MCTC",
    "R_BCTR",
)
DAMAGE_METRICS = (
    "mean_true_class_logit_drop",
    "mean_cross_entropy_increase",
    "prediction_flip_rate",
)
OUT_NAMES = (
    "task041_phase_f_frame_relation_signatures.csv",
    "task041_phase_f_bounded_reconstruction.csv",
    "task041_phase_f_unit_BCTR.csv",
    "task041_phase_f_same_type_oracle.csv",
    "task041_phase_f_mixed_domain_analysis.csv",
    "task041_phase_f_collective_vs_pairwise.csv",
    "task041_phase_f_signed_vs_magnitude_ablation.csv",
    "task041_phase_f_calibration_stability.csv",
    "task041_phase_f_baseline_comparison.csv",
    "task041_phase_f_summary.json",
    "task041_phase_f_report.md",
)


def progress(message: str) -> None:
    print("[Task041 Phase F] " + message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline BMS-conditioned frame-relation reconstruction audit."
    )
    parser.add_argument("--phase_c_raw", required=True)
    parser.add_argument("--phase_d1_raw", required=True)
    parser.add_argument("--frozen_units_csv", required=True)
    parser.add_argument("--fullval_damage_csv", required=True)
    parser.add_argument("--baseline_scores_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def int_string(value: Any) -> str:
    return str(int(float(str(value).strip())))


def domain_string(value: Any) -> str:
    return int_string(value)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def mask_restoration_is_exact(row: Mapping[str, Any]) -> bool:
    # The authoritative Phase D CSV uses mask_restored_exact. Accept the older
    # spelling only as an explicit schema alias; every supplied flag must be true.
    flags = [
        str(row[name]).strip().lower()
        for name in ("mask_restored_exact", "mask_restored_exactly")
        if name in row and str(row[name]).strip() != ""
    ]
    return bool(flags) and all(flag == "true" for flag in flags)


def freeze_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        int_string(row.get("candidate_task040_global_index", row.get("unit_global_index"))),
        str(row.get("candidate_layer_name", row.get("layer_name", ""))),
        str(row.get("candidate_unit_type", row.get("unit_type", ""))),
        int_string(row.get("candidate_unit_index", row.get("unit_index"))),
        int_string(row.get("candidate_stage", row.get("stage"))),
    )


def fullval_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        int_string(row["candidate_task040_global_index"]),
        str(row["candidate_layer_name"]),
        str(row["candidate_unit_type"]),
        int_string(row["candidate_unit_index"]),
    )


def load_frozen(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, str, str, str], str]]:
    rows = read_csv(path)
    by_uid: Dict[str, Dict[str, Any]] = {}
    raw_to_uid: Dict[Tuple[str, str, str, str, str], str] = {}
    for row in rows:
        uid = int_string(row["candidate_task037_global_index"])
        record = {
            "candidate_task037_global_index": uid,
            "candidate_task040_global_index": int_string(row["candidate_task040_global_index"]),
            "candidate_layer_name": str(row["candidate_layer_name"]),
            "candidate_unit_type": str(row["candidate_unit_type"]),
            "candidate_unit_index": int_string(row["candidate_unit_index"]),
            "candidate_stage": int_string(row["candidate_stage"]),
            "domain_id": domain_string(row["domain_id"]),
        }
        key = freeze_key(record)
        if uid in by_uid or key in raw_to_uid:
            raise ValueError("duplicate frozen Task041 unit identity")
        by_uid[uid] = record
        raw_to_uid[key] = uid
    domains = {r["domain_id"] for r in by_uid.values()}
    if len(by_uid) != N_UNITS or domains != set(FROZEN_DOMAINS):
        raise ValueError(
            "frozen-unit gate failed: units=" + str(len(by_uid)) +
            ", domains=" + repr(sorted(domains))
        )
    return by_uid, raw_to_uid


def load_fullval_damage(path: Path, frozen: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows = read_csv(path)
    result: Dict[str, Dict[str, Any]] = {}
    expected: Dict[Tuple[str, str, str, str], str] = {}
    for uid, item in frozen.items():
        key = (
            item["candidate_task040_global_index"],
            item["candidate_layer_name"],
            item["candidate_unit_type"],
            item["candidate_unit_index"],
        )
        expected[key] = uid
    for row in rows:
        key = fullval_key(row)
        if key not in expected:
            raise ValueError("full-validation row is not a frozen Task041 unit: " + repr(key))
        uid = expected[key]
        if uid in result:
            raise ValueError("duplicate full-validation damage identity: " + uid)
        if domain_string(row["domain_id"]) != frozen[uid]["domain_id"]:
            raise ValueError("full-validation domain differs from frozen identity: " + uid)
        if int_string(row["candidate_task037_global_index"]) != uid:
            raise ValueError("full-validation Task037 identity mismatch: " + uid)
        result[uid] = dict(row)
    if set(result) != set(frozen):
        raise ValueError("full-validation damage does not cover exactly the frozen 29 units")
    return result


def load_baseline_scores(
    path: Path, frozen: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    rows = read_csv(path)
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        uid = int_string(row["candidate_task037_global_index"])
        if uid not in frozen:
            raise ValueError("baseline candidate is outside the frozen Task041 29")
        item = frozen[uid]
        expected = (
            item["candidate_task040_global_index"],
            item["candidate_layer_name"],
            item["candidate_unit_type"],
            item["candidate_unit_index"],
        )
        actual = (
            int_string(row["candidate_task040_global_index"]),
            str(row.get("layer_name", row.get("candidate_layer_name", ""))),
            str(row.get("unit_type", row.get("candidate_unit_type", ""))),
            int_string(row.get("unit_index", row.get("candidate_unit_index"))),
        )
        if actual != expected or domain_string(row["domain_id"]) != item["domain_id"]:
            raise ValueError("baseline identity differs from frozen Task041 identity: " + uid)
        if uid in result:
            raise ValueError("duplicate baseline candidate: " + uid)
        result[uid] = dict(row)
    if len(result) != 18:
        raise ValueError("expected the existing frozen 18 low/high baseline candidates")
    return result


def dimension_key(row: Mapping[str, Any]) -> Tuple[int, str, int, int, int]:
    return (
        int(int_string(row["video_index"])),
        str(row["video_id"]),
        int(int_string(row["level"])),
        int(int_string(row["block_size"])),
        int(int_string(row["pair_index"])),
    )


def interaction_value(row: Mapping[str, Any]) -> float:
    a = np.float64(row["z_true_original"])
    b = np.float64(row["z_true_original_masked"])
    c = np.float64(row["z_true_intervened"])
    d = np.float64(row["z_true_intervened_masked"])
    return float(a - b - c + d)


def reconstruct_signatures(
    phase_c_path: Path,
    phase_d1_path: Path,
    frozen: Mapping[str, Mapping[str, Any]],
    raw_to_uid: Mapping[Tuple[str, str, str, str, str], str],
) -> Tuple[
    Dict[str, Dict[int, Dict[Tuple[int, str, int, int, int], float]]],
    Dict[int, List[Tuple[int, str, int, int, int]]],
    List[Tuple[int, str]],
    Dict[str, Any],
]:
    store: Dict[str, Dict[int, Dict[Tuple[int, str, int, int, int], float]]] = {
        uid: {span: {} for span in SPANS} for uid in frozen
    }
    phase_units: Dict[str, set[str]] = {"phase_c": set(), "phase_d1": set()}
    raw_counts: Dict[str, int] = {"phase_c": 0, "phase_d1": 0}
    selected_counts: Dict[str, int] = {"phase_c": 0, "phase_d1": 0}
    skipped_phase_c = 0
    c_exact_matches = 0
    all_video_keys: set[Tuple[int, str]] = set()

    for phase, path in (("phase_c", phase_c_path), ("phase_d1", phase_d1_path)):
        for row in read_csv(path):
            raw_counts[phase] += 1
            raw_key = freeze_key(row)
            uid = raw_to_uid.get(raw_key)
            if uid is None:
                if phase == "phase_c":
                    skipped_phase_c += 1
                    continue
                raise ValueError("D.1 raw unit is not in the frozen Task041 identity set")
            phase_units[phase].add(uid)
            selected_counts[phase] += 1
            span = int(int_string(row["block_size"]))
            level = int(int_string(row["level"]))
            pair_index = int(int_string(row["pair_index"]))
            if span not in SPANS or level != int(round(math.log2(span))):
                raise ValueError("unexpected span/level in Task040 raw records")
            if pair_index not in range(16):
                raise ValueError("unexpected pair index in Task040 raw records")
            dkey = dimension_key(row)
            all_video_keys.add((dkey[0], dkey[1]))
            bucket = store[uid][span]
            if dkey in bucket:
                raise ValueError("duplicate frame-relation dimension for unit/span")
            value = interaction_value(row)
            if phase == "phase_c":
                stored = finite_float(row.get("C_interaction"))
                if stored is None or value != stored:
                    raise ValueError("Phase C reconstructed a-b-c+d differs from stored C_interaction")
                c_exact_matches += 1
            bucket[dkey] = value

    if phase_units["phase_c"] & phase_units["phase_d1"]:
        raise ValueError("Phase C and D.1 raw unit partitions overlap")
    if phase_units["phase_c"] | phase_units["phase_d1"] != set(frozen):
        raise ValueError("Phase C/D.1 raw unit sets do not partition the frozen 29")
    if len(phase_units["phase_c"]) + len(phase_units["phase_d1"]) != N_UNITS:
        raise ValueError("unexpected number of selected Phase C + D.1 units")
    if selected_counts["phase_c"] + selected_counts["phase_d1"] != N_UNITS * len(SPANS) * 48:
        raise ValueError("selected raw record count is not 29*5*48")

    video_keys = sorted(all_video_keys, key=lambda x: (x[0], x[1]))
    if len(video_keys) != 3:
        raise ValueError("expected exactly three canonical video identities")
    orders: Dict[int, List[Tuple[int, str, int, int, int]]] = {}
    for span in SPANS:
        reference: Optional[List[Tuple[int, str, int, int, int]]] = None
        expected_by_video = {
            (video_index, video_id): set(range(16)) for video_index, video_id in video_keys
        }
        for uid in sorted(frozen, key=lambda x: int(x)):
            values = store[uid][span]
            keys = sorted(values, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            if len(keys) != 48:
                raise ValueError(
                    "unit/span must have exactly 48 dimensions: " + uid + "/" + str(span)
                )
            observed_by_video: Dict[Tuple[int, str], set[int]] = defaultdict(set)
            for key in keys:
                observed_by_video[(key[0], key[1])].add(key[4])
            if dict(observed_by_video) != expected_by_video:
                raise ValueError("video/intervention identity coverage differs across units")
            if reference is None:
                reference = keys
            elif keys != reference:
                raise ValueError("canonical 48-D frame/intervention order is not identical")
        if reference is None:
            raise ValueError("no canonical dimensions constructed")
        orders[span] = reference

    metadata = {
        "raw_record_count_phase_c_total": raw_counts["phase_c"],
        "raw_record_count_phase_d1_total": raw_counts["phase_d1"],
        "selected_record_count_phase_c": selected_counts["phase_c"],
        "selected_record_count_phase_d1": selected_counts["phase_d1"],
        "selected_interaction_rows": selected_counts["phase_c"] + selected_counts["phase_d1"],
        "expected_interaction_rows": N_UNITS * len(SPANS) * 48,
        "phase_c_frozen_unit_count": len(phase_units["phase_c"]),
        "phase_d1_frozen_unit_count": len(phase_units["phase_d1"]),
        "phase_c_unselected_rows_ignored": skipped_phase_c,
        "phase_c_C_interaction_exact_match_count": c_exact_matches,
        "canonical_video_count": len(video_keys),
        "canonical_video_order": [
            {"video_position_1based": i + 1, "video_index": v[0], "video_id": v[1]}
            for i, v in enumerate(video_keys)
        ],
        "dimension_count_per_unit_span": 48,
        "dimension_order_verified_for_all_unit_spans": True,
    }
    return store, orders, video_keys, metadata


def vector_for(
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
    uid: str,
    span: int,
    video_subset: Optional[set[Tuple[int, str]]] = None,
    transform: str = "signed",
) -> np.ndarray:
    keys = orders[span]
    if video_subset is not None:
        keys = [k for k in keys if (k[0], k[1]) in video_subset]
    values = np.asarray([store[uid][span][k] for k in keys], dtype=np.float64)
    if transform == "signed":
        return values
    if transform == "absolute":
        return np.abs(values)
    if transform == "squared":
        return np.square(values, dtype=np.float64)
    raise ValueError("unknown diagnostic representation: " + transform)


def bounded_least_squares(y: np.ndarray, predictors: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(predictors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("predictor matrix must have shape [dimensions, units]")
    if x.shape[1] == 0:
        reconstruction = np.zeros_like(y)
        return {
            "alpha": np.empty((0,), dtype=np.float64),
            "reconstruction": reconstruction,
            "status": 0,
            "success": True,
            "optimality": 0.0,
            "cost": float(0.5 * np.dot(y, y)),
            "message": "no eligible same-domain predictors",
            "nit": 0,
            "active_mask": [],
        }
    result = lsq_linear(
        x,
        y,
        bounds=(0.0, 1.0),
        method="bvls",
        tol=1e-12,
        max_iter=1000,
    )
    alpha = np.asarray(result.x, dtype=np.float64)
    reconstruction = x @ alpha
    return {
        "alpha": alpha,
        "reconstruction": reconstruction,
        "status": int(result.status),
        "success": bool(result.success),
        "optimality": float(result.optimality),
        "cost": float(result.cost),
        "message": str(result.message),
        "nit": int(result.nit or 0),
        "active_mask": [int(x) for x in result.active_mask],
    }


def residual_ratio(y: np.ndarray, reconstruction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    residual = y - np.asarray(reconstruction, dtype=np.float64)
    return float(np.linalg.norm(residual) / (np.linalg.norm(y) + EPS))


def pairwise_fit(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    denominator = float(np.dot(x, x))
    alpha = 0.0 if denominator == 0.0 else float(np.clip(np.dot(x, y) / denominator, 0.0, 1.0))
    reconstruction = alpha * x
    return alpha, float(np.linalg.norm(y)), residual_ratio(y, reconstruction)


def choose_low_high(
    rows: Sequence[Mapping[str, Any]], score_key: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not rows:
        raise ValueError("cannot choose low/high from an empty domain")
    low = min(
        rows,
        key=lambda r: (float(r[score_key]), int(str(r["candidate_task037_global_index"]))),
    )
    high = min(
        rows,
        key=lambda r: (-float(r[score_key]), int(str(r["candidate_task037_global_index"]))),
    )
    return low, high


def rank_statistics(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) < 2 or len(ya) != len(xa):
        return None, None
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return None, None
    rho = float(spearmanr(xa, ya).statistic)
    tau = float(kendalltau(xa, ya, variant="b").statistic)
    return (
        rho if math.isfinite(rho) else None,
        tau if math.isfinite(tau) else None,
    )


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def order_status(low_damage: float, high_damage: float) -> str:
    if high_damage > low_damage:
        return "correct"
    if high_damage < low_damage:
        return "reverse"
    return "tie"


def domain_score_oracle(
    units: Sequence[Mapping[str, Any]],
    score_key: str,
    damage_key: str,
    scope: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in units:
        grouped[str(row["domain_id"])].append(row)
    result: List[Dict[str, Any]] = []
    for domain in sorted(grouped, key=lambda d: (FROZEN_DOMAINS.index(d))):
        rows = grouped[domain]
        x = [float(r[score_key]) for r in rows]
        y = [float(r[damage_key]) for r in rows]
        rho, tau = rank_statistics(x, y)
        low, high = choose_low_high(rows, score_key)
        low_damage = float(low[damage_key])
        high_damage = float(high[damage_key])
        same_candidate = (
            str(low["candidate_task037_global_index"])
            == str(high["candidate_task037_global_index"])
        )
        status = "tie" if same_candidate else order_status(low_damage, high_damage)
        result.append({
            "row_level": "domain",
            "scope": scope,
            "domain_id": domain,
            "unit_count": len(rows),
            "score": score_key,
            "damage_metric": damage_key,
            "spearman": rho,
            "kendall_tau_b": tau,
            "low_task037_global_index": str(low["candidate_task037_global_index"]),
            "high_task037_global_index": str(high["candidate_task037_global_index"]),
            "low_score": float(low[score_key]),
            "high_score": float(high[score_key]),
            "low_damage": low_damage,
            "high_damage": high_damage,
            "high_minus_low_damage": high_damage - low_damage,
            "ordering": status,
            "low_high_same_candidate_due_score_tie": same_candidate,
        })
    return result


def aggregate_oracle(
    domain_rows: Sequence[Mapping[str, Any]], scope: str, score_key: str, damage_key: str
) -> Dict[str, Any]:
    selected = [r for r in domain_rows if r["scope"] == scope]
    counts = Counter(str(r["ordering"]) for r in selected)
    return {
        "row_level": "domain_balanced",
        "scope": scope,
        "domain_id": "ALL",
        "unit_count": sum(int(r["unit_count"]) for r in selected),
        "domain_count": len(selected),
        "score": score_key,
        "damage_metric": damage_key,
        "domain_balanced_spearman": safe_mean([r["spearman"] for r in selected]),
        "domain_balanced_kendall_tau_b": safe_mean([r["kendall_tau_b"] for r in selected]),
        "valid_spearman_domain_count": sum(r["spearman"] is not None for r in selected),
        "valid_kendall_domain_count": sum(r["kendall_tau_b"] is not None for r in selected),
        "correct_count": counts["correct"],
        "reverse_count": counts["reverse"],
        "tie_count": counts["tie"],
        "domain_count_expected": len(selected),
    }


def make_primary_oracle(
    units: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_domain: List[Dict[str, Any]] = []
    for domain in FROZEN_DOMAINS:
        subset = [r for r in units if str(r["domain_id"]) == domain]
        scope = "same_type" if domain in SAME_TYPE_DOMAINS else "mixed"
        for damage in DAMAGE_METRICS:
            per_domain.extend(domain_score_oracle(subset, "R_BCTR", damage, scope))
    rows = list(per_domain)
    summaries: Dict[str, Any] = {}
    for scope, domains in (("same_type", SAME_TYPE_DOMAINS), ("mixed", MIXED_DOMAINS)):
        for damage in DAMAGE_METRICS:
            selected = [
                r for r in per_domain
                if r["scope"] == scope and r["damage_metric"] == damage
            ]
            counts = Counter(str(r["ordering"]) for r in selected)
            summary = {
                "domain_balanced_spearman": safe_mean([r["spearman"] for r in selected]),
                "domain_balanced_kendall_tau_b": safe_mean([r["kendall_tau_b"] for r in selected]),
                "valid_spearman_domain_count": sum(r["spearman"] is not None for r in selected),
                "valid_kendall_domain_count": sum(r["kendall_tau_b"] is not None for r in selected),
                "correct_count": counts["correct"],
                "reverse_count": counts["reverse"],
                "tie_count": counts["tie"],
                "domain_count": len(selected),
            }
            summaries.setdefault(scope, {})[damage] = summary
            rows.append({
                "row_level": "domain_balanced",
                "scope": scope,
                "domain_id": "ALL",
                "unit_count": sum(int(r["unit_count"]) for r in selected),
                "score": "R_BCTR",
                "damage_metric": damage,
                **summary,
            })
    return rows, summaries


def make_signature_csv_rows(
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
    frozen: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for uid in sorted(frozen, key=lambda x: int(x)):
        base = frozen[uid]
        for span in SPANS:
            for dimension_index, key in enumerate(orders[span]):
                rows.append({
                    **base,
                    "span": span,
                    "dimension_index": dimension_index,
                    "video_index": key[0],
                    "video_id": key[1],
                    "intervention_level": key[2],
                    "block_size": key[3],
                    "pair_index": key[4],
                    "C_signed": float(store[uid][span][key]),
                    "representation": "signed_a_minus_b_minus_c_plus_d",
                })
    return rows


def run_signed_collective_and_pairwise(
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
    frozen: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[int, np.ndarray]], Dict[str, float]]:
    domain_members: Dict[str, List[str]] = defaultdict(list)
    for uid, row in frozen.items():
        domain_members[str(row["domain_id"])].append(uid)
    for domain in domain_members:
        domain_members[domain].sort(key=lambda x: int(x))

    recon_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    span_residuals: Dict[str, Dict[int, np.ndarray]] = {
        uid: {} for uid in frozen
    }
    pair_span_residuals: Dict[str, Dict[int, float]] = {
        uid: {} for uid in frozen
    }
    for unit_number, uid in enumerate(sorted(frozen, key=lambda x: int(x)), 1):
        domain = str(frozen[uid]["domain_id"])
        peers = [peer for peer in domain_members[domain] if peer != uid]
        if not peers:
            raise ValueError("BMS domain has no remaining tested same-domain peer")
        for span in SPANS:
            y = vector_for(store, orders, uid, span)
            matrix = np.column_stack([vector_for(store, orders, peer, span) for peer in peers])
            fit = bounded_least_squares(y, matrix)
            ratio = residual_ratio(y, fit["reconstruction"])
            coeffs = [
                {
                    "candidate_task037_global_index": peer,
                    "unit_type": frozen[peer]["candidate_unit_type"],
                    "alpha": float(alpha),
                }
                for peer, alpha in zip(peers, fit["alpha"])
            ]
            nonzero = [x for x in coeffs if float(x["alpha"]) > 0.0]
            span_residuals[uid][span] = np.asarray([ratio], dtype=np.float64)
            recon_rows.append({
                **frozen[uid],
                "span": span,
                "candidate_norm": float(np.linalg.norm(y)),
                "reconstruction_norm": float(np.linalg.norm(fit["reconstruction"])),
                "residual_norm": float(np.linalg.norm(y - fit["reconstruction"])),
                "residual_ratio": ratio,
                "coefficients_json": json.dumps(coeffs, ensure_ascii=False, separators=(",", ":")),
                "nonzero_coefficients_json": json.dumps(nonzero, ensure_ascii=False, separators=(",", ":")),
                "nonzero_coefficient_count": len(nonzero),
                "coefficient_sum": float(np.sum(fit["alpha"])),
                "solver": "scipy.optimize.lsq_linear(method=bvls)",
                "solver_status": fit["status"],
                "solver_success": fit["success"],
                "solver_optimality": fit["optimality"],
                "solver_cost": fit["cost"],
                "solver_nit": fit["nit"],
                "solver_active_mask_json": json.dumps(fit["active_mask"]),
                "solver_message": fit["message"],
                "bounds": "[0,1]",
                "dtype": "float64",
                "lambda": "",
                "ridge": False,
                "sum_alpha_constraint": False,
            })

            choices: List[Tuple[float, int, str, float, float, float]] = []
            for peer in peers:
                x = vector_for(store, orders, peer, span)
                alpha, candidate_norm, pair_ratio = pairwise_fit(y, x)
                choices.append((
                    pair_ratio, int(peer), peer, alpha, candidate_norm,
                    float(np.linalg.norm(alpha * x)),
                ))
            best = min(choices, key=lambda x: (x[0], x[1]))
            pair_span_residuals[uid][span] = float(best[0])
            pair_rows.append({
                **frozen[uid],
                "span": span,
                "best_single_substitute_task037_global_index": best[2],
                "best_single_substitute_unit_type": frozen[best[2]]["candidate_unit_type"],
                "alpha": best[3],
                "candidate_norm": best[4],
                "reconstruction_norm": best[5],
                "residual_ratio": best[0],
                "bounds": "[0,1]",
                "dtype": "float64",
                "tie_break": "ascending Task037 global_index",
            })
        if unit_number % 5 == 0 or unit_number == N_UNITS:
            progress("signed bounded reconstruction: " + str(unit_number) + "/" + str(N_UNITS))
    pair_summary = {
        uid: float(math.sqrt(np.mean([pair_span_residuals[uid][s] ** 2 for s in SPANS])))
        for uid in frozen
    }
    return recon_rows, pair_rows, span_residuals, pair_summary


def make_unit_rows(
    frozen: Mapping[str, Mapping[str, Any]],
    damage: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    span_residuals: Mapping[str, Mapping[int, np.ndarray]],
    pair_summary: Mapping[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for uid in sorted(frozen, key=lambda x: int(x)):
        residuals = {span: float(span_residuals[uid][span][0]) for span in SPANS}
        bctr = float(math.sqrt(np.mean([residuals[s] ** 2 for s in SPANS])))
        most_unique = max(SPANS, key=lambda s: (residuals[s], -SPANS.index(s)))
        item: Dict[str, Any] = {
            **frozen[uid],
            "R_BCTR": bctr,
            "R_pair": float(pair_summary[uid]),
            "pairwise_minus_collective": float(pair_summary[uid] - bctr),
            "most_unique_span": most_unique,
            "mean_true_class_logit_drop": float(damage[uid]["mean_true_class_logit_drop"]),
            "mean_cross_entropy_increase": float(damage[uid]["mean_cross_entropy_increase"]),
            "prediction_flip_rate": float(damage[uid]["prediction_flip_rate"]),
            "R_MCTC": finite_float(damage[uid].get("R_MCTC")),
            "fullval_samples": int(float(damage[uid]["n_samples"])),
            "mask_restored_exactly": str(damage[uid]["mask_restored_exactly"]).lower() == "true",
            "damage_uses_signed_differences": str(damage[uid]["damage_uses_signed_differences"]).lower() == "true",
            "baseline_candidate": uid in baseline,
        }
        for span in SPANS:
            item["r_span_" + str(span)] = residuals[span]
        for criterion in BASELINE_CRITERIA:
            if criterion != "R_BCTR":
                item["baseline_" + criterion] = finite_float(baseline.get(uid, {}).get(criterion))
        rows.append(item)

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain_id"])].append(row)
    for domain_rows in by_domain.values():
        ordered = sorted(
            domain_rows,
            key=lambda r: (float(r["R_BCTR"]), int(r["candidate_task037_global_index"])),
        )
        for rank, row in enumerate(ordered, 1):
            row["within_domain_R_BCTR_rank"] = rank
        ordered_pair = sorted(
            domain_rows,
            key=lambda r: (float(r["R_pair"]), int(r["candidate_task037_global_index"])),
        )
        for rank, row in enumerate(ordered_pair, 1):
            row["within_domain_R_pair_rank"] = rank
    return rows


def run_representation_ablation(
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
    frozen: Mapping[str, Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    damage_by_uid = {str(r["candidate_task037_global_index"]): r for r in unit_rows}
    output: List[Dict[str, Any]] = []
    scores: Dict[str, Dict[str, float]] = {variant: {} for variant in ("signed", "absolute", "squared")}
    domain_members: Dict[str, List[str]] = defaultdict(list)
    for uid, item in frozen.items():
        domain_members[str(item["domain_id"])].append(uid)
    for values in domain_members.values():
        values.sort(key=lambda x: int(x))

    for variant in ("signed", "absolute", "squared"):
        progress("signed-vs-magnitude ablation: " + variant)
        for unit_number, uid in enumerate(sorted(frozen, key=lambda x: int(x)), 1):
            domain = str(frozen[uid]["domain_id"])
            peers = [p for p in domain_members[domain] if p != uid]
            ratios: Dict[int, float] = {}
            for span in SPANS:
                y = vector_for(store, orders, uid, span, transform=variant)
                matrix = np.column_stack([
                    vector_for(store, orders, peer, span, transform=variant) for peer in peers
                ])
                fit = bounded_least_squares(y, matrix)
                ratios[span] = residual_ratio(y, fit["reconstruction"])
            value = float(math.sqrt(np.mean([ratios[s] ** 2 for s in SPANS])))
            scores[variant][uid] = value
            row = {
                **frozen[uid],
                "representation": variant,
                "BCTR_diagnostic": value,
                "mean_true_class_logit_drop": damage_by_uid[uid]["mean_true_class_logit_drop"],
                "mean_cross_entropy_increase": damage_by_uid[uid]["mean_cross_entropy_increase"],
                "prediction_flip_rate": damage_by_uid[uid]["prediction_flip_rate"],
                "interpretation": "diagnostic-only; not a selector",
            }
            for span in SPANS:
                row["r_span_" + str(span)] = ratios[span]
            output.append(row)
            if unit_number % 10 == 0 or unit_number == N_UNITS:
                progress(variant + " variant units: " + str(unit_number) + "/" + str(N_UNITS))
    return output, scores


def make_mixed_rows(
    frozen: Mapping[str, Mapping[str, Any]],
    signed_reconstruction_rows: Sequence[Mapping[str, Any]],
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    primary = {
        (str(r["candidate_task037_global_index"]), int(r["span"])): r
        for r in signed_reconstruction_rows
    }
    domain_members: Dict[str, List[str]] = defaultdict(list)
    for uid, item in frozen.items():
        domain_members[str(item["domain_id"])].append(uid)
    for values in domain_members.values():
        values.sort(key=lambda x: int(x))
    result: List[Dict[str, Any]] = []
    support_variation: Dict[str, int] = {}
    head_target_multi_ffn_spans = 0
    head_target_spans = 0
    cross_residual_by_target_type: Dict[str, List[float]] = {"head": [], "neuron": []}

    for uid, item in frozen.items():
        domain = str(item["domain_id"])
        if domain not in MIXED_DOMAINS:
            continue
        peers = [p for p in domain_members[domain] if p != uid]
        support_by_span: List[Tuple[str, ...]] = []
        for span in SPANS:
            primary_row = primary[(uid, span)]
            coeffs = json.loads(str(primary_row["coefficients_json"]))
            active = [a for a in coeffs if float(a["alpha"]) > 0.0]
            heads = [a for a in active if a["unit_type"] == "head"]
            neurons = [a for a in active if a["unit_type"] == "neuron"]
            support_by_span.append(tuple(sorted(str(a["candidate_task037_global_index"]) for a in active)))

            opposite_type = "neuron" if item["candidate_unit_type"] == "head" else "head"
            cross_peers = [p for p in peers if frozen[p]["candidate_unit_type"] == opposite_type]
            y = vector_for(store, orders, uid, span)
            if cross_peers:
                cross_x = np.column_stack([vector_for(store, orders, p, span) for p in cross_peers])
                cross_fit = bounded_least_squares(y, cross_x)
                cross_ratio = residual_ratio(y, cross_fit["reconstruction"])
                cross_active = [
                    {"candidate_task037_global_index": p, "alpha": float(alpha)}
                    for p, alpha in zip(cross_peers, cross_fit["alpha"])
                    if float(alpha) > 0.0
                ]
            else:
                cross_ratio = None
                cross_active = []
            cross_residual_by_target_type[str(item["candidate_unit_type"])].append(
                float(cross_ratio) if cross_ratio is not None else float("nan")
            )
            if item["candidate_unit_type"] == "head":
                head_target_spans += 1
                if len(neurons) >= 2:
                    head_target_multi_ffn_spans += 1
            result.append({
                **item,
                "span": span,
                "R_BCTR_span": float(primary_row["residual_ratio"]),
                "all_same_domain_coefficients_json": str(primary_row["coefficients_json"]),
                "nonzero_coefficients_json": json.dumps(active, ensure_ascii=False, separators=(",", ":")),
                "nonzero_head_units": len(heads),
                "nonzero_ffn_units": len(neurons),
                "alpha_sum_attention_heads": float(sum(float(a["alpha"]) for a in heads)),
                "alpha_sum_ffn_neurons": float(sum(float(a["alpha"]) for a in neurons)),
                "cross_type_only_predictor_type": opposite_type,
                "cross_type_only_residual_ratio": cross_ratio,
                "cross_type_only_nonzero_coefficients_json": json.dumps(
                    cross_active, ensure_ascii=False, separators=(",", ":")
                ),
                "nonzero_support_definition": "alpha > 0 exactly; raw coefficients retained",
            })
        support_variation[uid] = len(set(support_by_span))

    finite_head = [x for x in cross_residual_by_target_type["head"] if math.isfinite(x)]
    finite_neuron = [x for x in cross_residual_by_target_type["neuron"] if math.isfinite(x)]
    summary = {
        "head_target_span_count": head_target_spans,
        "head_target_spans_with_at_least_two_nonzero_ffn_substitutes": head_target_multi_ffn_spans,
        "head_target_spans_with_at_least_one_nonzero_ffn_substitute": sum(
            1 for r in result
            if r["candidate_unit_type"] == "head" and int(r["nonzero_ffn_units"]) >= 1
        ),
        "candidate_support_changes_across_spans": sum(v > 1 for v in support_variation.values()),
        "mixed_candidate_count_with_support_change": sum(v > 1 for v in support_variation.values()),
        "mixed_candidate_count": len(support_variation),
        "mean_cross_type_only_residual_when_target_is_head_ffn_predictors": (
            float(np.mean(finite_head)) if finite_head else None
        ),
        "mean_cross_type_only_residual_when_target_is_ffn_head_predictors": (
            float(np.mean(finite_neuron)) if finite_neuron else None
        ),
    }
    return result, summary


def make_collective_pairwise_rows(
    unit_rows: Sequence[Mapping[str, Any]],
    pairwise_span_rows: Sequence[Mapping[str, Any]],
    primary_span_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    primary = {
        (str(r["candidate_task037_global_index"]), int(r["span"])): r
        for r in primary_span_rows
    }
    pairwise = {
        (str(r["candidate_task037_global_index"]), int(r["span"])): r
        for r in pairwise_span_rows
    }
    result: List[Dict[str, Any]] = []
    for unit in unit_rows:
        uid = str(unit["candidate_task037_global_index"])
        for span in SPANS:
            collective = float(primary[(uid, span)]["residual_ratio"])
            pair = float(pairwise[(uid, span)]["residual_ratio"])
            result.append({
                **{k: unit[k] for k in (
                    "candidate_task037_global_index", "candidate_task040_global_index",
                    "candidate_layer_name", "candidate_unit_type", "candidate_unit_index",
                    "candidate_stage", "domain_id",
                )},
                "span": span,
                "collective_signed_residual_ratio": collective,
                "pairwise_signed_residual_ratio": pair,
                "pairwise_minus_collective": pair - collective,
                "pairwise_best_substitute_task037_global_index":
                    pairwise[(uid, span)]["best_single_substitute_task037_global_index"],
                "pairwise_best_alpha": pairwise[(uid, span)]["alpha"],
                "collective_le_pairwise_within_numeric_tolerance": collective <= pair + 1e-10,
            })
    return result


def make_calibration_rows(
    frozen: Mapping[str, Mapping[str, Any]],
    store: Mapping[str, Mapping[int, Mapping[Tuple[int, str, int, int, int], float]]],
    orders: Mapping[int, Sequence[Tuple[int, str, int, int, int]]],
    full_units: Sequence[Mapping[str, Any]],
    video_keys: Sequence[Tuple[int, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(video_keys) != 3:
        raise ValueError("leave-one-video-out requires exactly three videos")
    label_to_video = {i + 1: video for i, video in enumerate(video_keys)}
    subsets = (
        ((1, 2), "videos_1_2"),
        ((1, 3), "videos_1_3"),
        ((2, 3), "videos_2_3"),
    )
    domain_members: Dict[str, List[str]] = defaultdict(list)
    for uid, item in frozen.items():
        domain_members[str(item["domain_id"])].append(uid)
    for values in domain_members.values():
        values.sort(key=lambda x: int(x))
    full_score = {
        str(r["candidate_task037_global_index"]): float(r["R_BCTR"])
        for r in full_units
    }
    output: List[Dict[str, Any]] = []
    domain_summaries: List[Dict[str, Any]] = []
    all_correlations: List[Optional[float]] = []
    all_kendalls: List[Optional[float]] = []
    low_stable = high_stable = domain_subset_count = exact_order_count = 0

    for labels, subset_name in subsets:
        selected_videos = {label_to_video[i] for i in labels}
        subset_scores: Dict[str, float] = {}
        for unit_number, uid in enumerate(sorted(frozen, key=lambda x: int(x)), 1):
            domain = str(frozen[uid]["domain_id"])
            peers = [p for p in domain_members[domain] if p != uid]
            ratios = []
            for span in SPANS:
                y = vector_for(store, orders, uid, span, selected_videos)
                x = np.column_stack([
                    vector_for(store, orders, peer, span, selected_videos) for peer in peers
                ])
                fit = bounded_least_squares(y, x)
                ratios.append(residual_ratio(y, fit["reconstruction"]))
            subset_scores[uid] = float(math.sqrt(np.mean(np.square(ratios))))
            if unit_number % 10 == 0 or unit_number == N_UNITS:
                progress(subset_name + " calibration units: " + str(unit_number) + "/" + str(N_UNITS))

        for domain in FROZEN_DOMAINS:
            members = domain_members[domain]
            full_order = sorted(members, key=lambda u: (full_score[u], int(u)))
            subset_order = sorted(members, key=lambda u: (subset_scores[u], int(u)))
            rho, tau = rank_statistics(
                [full_score[u] for u in members],
                [subset_scores[u] for u in members],
            )
            full_low = min(members, key=lambda u: (full_score[u], int(u)))
            full_high = min(members, key=lambda u: (-full_score[u], int(u)))
            subset_low = min(members, key=lambda u: (subset_scores[u], int(u)))
            subset_high = min(members, key=lambda u: (-subset_scores[u], int(u)))
            low_same = full_low == subset_low
            high_same = full_high == subset_high
            exact_same = full_order == subset_order
            all_correlations.append(rho)
            all_kendalls.append(tau)
            low_stable += int(low_same)
            high_stable += int(high_same)
            exact_order_count += int(exact_same)
            domain_subset_count += 1
            domain_summaries.append({
                "domain_id": domain,
                "domain_group": "same_type" if domain in SAME_TYPE_DOMAINS else "mixed",
                "video_subset": subset_name,
                "video_positions_1based": json.dumps(labels),
                "unit_count": len(members),
                "ranking_spearman_full_vs_subset": rho,
                "ranking_kendall_tau_b_full_vs_subset": tau,
                "full_order_task037_json": json.dumps(full_order),
                "subset_order_task037_json": json.dumps(subset_order),
                "exact_order_match": exact_same,
                "full_low_task037_global_index": full_low,
                "subset_low_task037_global_index": subset_low,
                "low_candidate_stable": low_same,
                "full_high_task037_global_index": full_high,
                "subset_high_task037_global_index": subset_high,
                "high_candidate_stable": high_same,
            })
            rank_map = {uid: i + 1 for i, uid in enumerate(subset_order)}
            full_rank_map = {uid: i + 1 for i, uid in enumerate(full_order)}
            for uid in members:
                output.append({
                    **frozen[uid],
                    "video_subset": subset_name,
                    "video_positions_1based": json.dumps(labels),
                    "R_BCTR_all_3_videos": full_score[uid],
                    "R_BCTR_subset": subset_scores[uid],
                    "full_rank_within_domain": full_rank_map[uid],
                    "subset_rank_within_domain": rank_map[uid],
                    "rank_shift_subset_minus_full": rank_map[uid] - full_rank_map[uid],
                    "full_low_candidate": uid == full_low,
                    "subset_low_candidate": uid == subset_low,
                    "full_high_candidate": uid == full_high,
                    "subset_high_candidate": uid == subset_high,
                    "domain_spearman_full_vs_subset": rho,
                    "domain_kendall_tau_b_full_vs_subset": tau,
                    "domain_exact_order_match": exact_same,
                    "low_candidate_stable": low_same,
                    "high_candidate_stable": high_same,
                })

    same_type_domains = [r for r in domain_summaries if r["domain_id"] in SAME_TYPE_DOMAINS]
    summary = {
        "subset_count": len(subsets),
        "domain_subset_comparisons": domain_subset_count,
        "same_type_domain_subset_comparisons": len(same_type_domains),
        "same_type_mean_spearman": safe_mean([r["ranking_spearman_full_vs_subset"] for r in same_type_domains]),
        "same_type_mean_kendall_tau_b": safe_mean([r["ranking_kendall_tau_b_full_vs_subset"] for r in same_type_domains]),
        "same_type_low_candidate_stability_count": sum(bool(r["low_candidate_stable"]) for r in same_type_domains),
        "same_type_high_candidate_stability_count": sum(bool(r["high_candidate_stable"]) for r in same_type_domains),
        "same_type_exact_order_match_count": sum(bool(r["exact_order_match"]) for r in same_type_domains),
        "all_domain_mean_spearman": safe_mean(all_correlations),
        "all_domain_mean_kendall_tau_b": safe_mean(all_kendalls),
        "all_domain_low_candidate_stability_count": low_stable,
        "all_domain_high_candidate_stability_count": high_stable,
        "all_domain_exact_order_match_count": exact_order_count,
        "domain_summaries": domain_summaries,
    }
    return output, summary


def make_baseline_rows(
    unit_rows: Sequence[Mapping[str, Any]],
    baseline_scores: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    unit_by_uid = {str(r["candidate_task037_global_index"]): r for r in unit_rows}
    candidates: List[Dict[str, Any]] = []
    for uid, source in baseline_scores.items():
        row = dict(unit_by_uid[uid])
        for criterion in BASELINE_CRITERIA:
            if criterion == "R_BCTR":
                row[criterion] = float(unit_by_uid[uid]["R_BCTR"])
            else:
                row[criterion] = float(source[criterion])
        candidates.append(row)
    if len(candidates) != 18:
        raise ValueError("baseline comparison must use the same frozen 18 candidates")
    scopes = (
        ("same_type", tuple(SAME_TYPE_DOMAINS)),
        ("mixed", tuple(MIXED_DOMAINS)),
        ("all_9_secondary", tuple(FROZEN_DOMAINS)),
    )
    output: List[Dict[str, Any]] = []
    compact: Dict[str, Any] = {}
    for scope, domains in scopes:
        for criterion in BASELINE_CRITERIA:
            for damage in DAMAGE_METRICS:
                per_domain = []
                for domain in domains:
                    members = [r for r in candidates if str(r["domain_id"]) == domain]
                    if not members:
                        continue
                    rho, tau = rank_statistics(
                        [float(r[criterion]) for r in members],
                        [float(r[damage]) for r in members],
                    )
                    low, high = choose_low_high(members, criterion)
                    same_candidate = (
                        str(low["candidate_task037_global_index"])
                        == str(high["candidate_task037_global_index"])
                    )
                    status = (
                        "tie" if same_candidate
                        else order_status(float(low[damage]), float(high[damage]))
                    )
                    row = {
                        "row_level": "domain",
                        "scope": scope,
                        "domain_id": domain,
                        "candidate_count": len(members),
                        "criterion": criterion,
                        "damage_metric": damage,
                        "spearman": rho,
                        "kendall_tau_b": tau,
                        "ordering": status,
                        "low_task037_global_index": str(low["candidate_task037_global_index"]),
                        "high_task037_global_index": str(high["candidate_task037_global_index"]),
                        "low_score": float(low[criterion]),
                        "high_score": float(high[criterion]),
                        "low_damage": float(low[damage]),
                        "high_damage": float(high[damage]),
                        "high_minus_low_damage": float(high[damage]) - float(low[damage]),
                    }
                    per_domain.append(row)
                    output.append(row)
                counts = Counter(str(r["ordering"]) for r in per_domain)
                aggregate = {
                    "row_level": "domain_balanced",
                    "scope": scope,
                    "domain_id": "ALL",
                    "candidate_count": sum(int(r["candidate_count"]) for r in per_domain),
                    "domain_count": len(per_domain),
                    "criterion": criterion,
                    "damage_metric": damage,
                    "domain_balanced_spearman": safe_mean([r["spearman"] for r in per_domain]),
                    "domain_balanced_kendall_tau_b": safe_mean([r["kendall_tau_b"] for r in per_domain]),
                    "valid_spearman_domain_count": sum(r["spearman"] is not None for r in per_domain),
                    "valid_kendall_domain_count": sum(r["kendall_tau_b"] is not None for r in per_domain),
                    "correct_count": counts["correct"],
                    "reverse_count": counts["reverse"],
                    "tie_count": counts["tie"],
                }
                output.append(aggregate)
                compact.setdefault(scope, {}).setdefault(criterion, {})[damage] = aggregate
    return output, compact


def make_report(
    summary: Mapping[str, Any],
    primary_oracle: Sequence[Mapping[str, Any]],
    baseline_comp: Sequence[Mapping[str, Any]],
    mixed_summary: Mapping[str, Any],
    calibration: Mapping[str, Any],
    collective_pairwise: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    same = summary["same_type_oracle_summary"]
    gate = summary["decision_gate"]
    spans = summary["most_unique_span_counts"]
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        return format(float(value), ".3f")
    def primary_metric(outcome: str, scope: str = "same_type") -> Mapping[str, Any]:
        return same[scope][outcome]
    def base(scope: str, criterion: str, outcome: str) -> Mapping[str, Any]:
        return baseline_comp[scope][criterion][outcome]

    high_improvements = sorted(
        collective_pairwise,
        key=lambda r: (-float(r["pairwise_minus_collective"]), int(r["candidate_task037_global_index"]), int(r["span"])),
    )
    all_unit_improvements = summary["collective_pairwise_unit_summary"]
    mixed_head_multi = int(mixed_summary["head_target_spans_with_at_least_two_nonzero_ffn_substitutes"])
    mixed_head_total = int(mixed_summary["head_target_span_count"])
    mixed_head_any = int(mixed_summary["head_target_spans_with_at_least_one_nonzero_ffn_substitute"])
    mixed_head_cross = mixed_summary["mean_cross_type_only_residual_when_target_is_head_ffn_predictors"]
    mixed_neuron_cross = mixed_summary["mean_cross_type_only_residual_when_target_is_ffn_head_predictors"]
    if mixed_head_cross is None or mixed_neuron_cross is None:
        direction = "not estimable from the available opposite-type peers"
    elif mixed_head_cross < mixed_neuron_cross:
        direction = "FFN peers reconstruct head targets with lower mean normalized residual than head peers reconstruct neuron targets"
    elif mixed_neuron_cross < mixed_head_cross:
        direction = "head peers reconstruct neuron targets with lower mean normalized residual than FFN peers reconstruct head targets"
    else:
        direction = "the two cross-type directions have equal mean normalized residual"
    lines = [
        "# Task041 Phase F — BMS-conditioned Frame-relation Unit Selection Audit",
        "",
        "Offline CPU analysis only. No model inference, GPU, temporal-intervention rerun, BMS rerun, pruning, fine-tuning, or descriptor modification occurred.",
        "",
        "## Frozen inputs and exact reconstruction",
        "",
        "- Frozen units/domains: 29 units in the nine existing BMS domains; Phase C/D.1 partition: " +
        str(summary["raw_records"]["phase_c_frozen_unit_count"]) + " / " +
        str(summary["raw_records"]["phase_d1_frozen_unit_count"]) + ".",
        "- Selected raw interactions: " + str(summary["raw_records"]["selected_interaction_rows"]) +
        " / " + str(summary["raw_records"]["expected_interaction_rows"]) +
        "; Phase C stored C_interaction exact matches: " +
        str(summary["raw_records"]["phase_c_C_interaction_exact_match_count"]) + ".",
        "- Each unit/span has 48 dimensions (3 canonical videos x 16 interventions). Ordering is identical across all units and was checked exactly.",
        "- Primary signature is reconstructed in float64 as a-b-c+d. Primary fit is bounded least squares with 0 <= alpha <= 1, no ridge/lambda, and no sum(alpha)=1 constraint.",
        "- Full-validation damage was joined by the frozen identity; all 29 joins are exact.",
        "",
        "## Primary same-type oracle",
        "",
        "| Damage | domain-balanced rho | domain-balanced Kendall tau-b | high-BCTR greater-damage domains | reverse | tie |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in DAMAGE_METRICS:
        r = primary_metric(metric)
        lines.append(
            "| " + metric + " | " + fmt(r["domain_balanced_spearman"]) +
            " | " + fmt(r["domain_balanced_kendall_tau_b"]) +
            " | " + str(r["correct_count"]) + "/" + str(r["domain_count"]) +
            " | " + str(r["reverse_count"]) + " | " + str(r["tie_count"]) + " |"
        )
    lines += [
        "",
        "Low/high is selected independently inside each domain by R_BCTR; score ties are resolved by ascending Task037 global_index. Mixed domains are summarized separately and do not enter the same-type gate.",
        "",
        "## Q1. Full frame-pair pattern vs five-span RMS baseline",
        "",
        "The comparison is made on the same frozen 18 low/high baseline candidates, with within-domain ranks and same-type domains primary.",
        "",
        "| Damage | R_BCTR rho | G_RMS rho | R_BCTR correct/reverse/tie | G_RMS correct/reverse/tie |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in DAMAGE_METRICS:
        rb = base("same_type", "R_BCTR", metric)
        rms = base("same_type", "G_RMS", metric)
        lines.append(
            "| " + metric + " | " + fmt(rb["domain_balanced_spearman"]) +
            " | " + fmt(rms["domain_balanced_spearman"]) +
            " | " + str(rb["correct_count"]) + "/" + str(rb["reverse_count"]) + "/" + str(rb["tie_count"]) +
            " | " + str(rms["correct_count"]) + "/" + str(rms["reverse_count"]) + "/" + str(rms["tie_count"]) + " |"
        )
    lines += [
        "",
        "Q1 is answered by these matched-cohort values; no representation is promoted after observing results.",
        "",
        "## Q2. Collective vs pairwise reconstruction",
        "",
        "- Unit-level R_pair - R_BCTR mean/median: " +
        fmt(summary["collective_pairwise_unit_summary"]["mean_pairwise_minus_collective"]) +
        " / " + fmt(summary["collective_pairwise_unit_summary"]["median_pairwise_minus_collective"]) +
        "; collective residual is no greater than pairwise (within numeric tolerance) for " +
        str(summary["collective_pairwise_unit_summary"]["collective_le_pairwise_count"]) +
        "/" + str(N_UNITS) + " units.",
        "- Largest five pairwise-to-collective improvements are listed without a threshold; all candidate/span residuals are in task041_phase_f_collective_vs_pairwise.csv.",
    ]
    for row in high_improvements[:5]:
        lines.append(
            "- Task037 " + str(row["candidate_task037_global_index"]) +
            ", domain " + str(row["domain_id"]) + ", span " + str(row["span"]) +
            ": R_pair - R_collective = " + fmt(row["pairwise_minus_collective"]) + "."
        )
    lines += [
        "",
        "## Q3. Same-type damage prediction",
        "",
        "The primary seven-domain results above determine the gate. R_BCTR remains a within-domain diagnostic; it is not a cross-domain score and does not authorize pruning.",
        "",
        "## Q4. Mixed head/FFN domains",
        "",
        "- Head target spans with at least one nonzero FFN coefficient: " +
        str(mixed_head_any) + "/" + str(mixed_head_total) + "; with at least two: " +
        str(mixed_head_multi) + "/" + str(mixed_head_total) + ".",
        "- Mean opposite-type-only residual: head targets reconstructed from FFN peers = " +
        fmt(mixed_head_cross) + "; FFN-neuron targets reconstructed from head peers = " +
        fmt(mixed_neuron_cross) + ".",
        "- Directional comparison: " + direction + ".",
        "- Coefficient sums by type and exact nonzero supports for every mixed candidate/span are in task041_phase_f_mixed_domain_analysis.csv. These two mixed domains are not evidence for selector success.",
        "",
        "## Q5. Most unique temporal span",
        "",
        "| Span | Units for which this span has the largest residual |",
        "|---:|---:|",
    ]
    for span in SPANS:
        lines.append("| " + str(span) + " | " + str(spans.get(str(span), 0)) + " |")
    lines += [
        "",
        "Ties in argmax use the earliest span in [1,2,4,8,16]; this is descriptive only.",
        "",
        "## Q6. Leave-one-video-out stability",
        "",
        "- Same-type domain/subset comparisons: " +
        str(calibration["same_type_domain_subset_comparisons"]) +
        "; mean within-domain rank Spearman/Kendall: " +
        fmt(calibration["same_type_mean_spearman"]) + " / " +
        fmt(calibration["same_type_mean_kendall_tau_b"]) + ".",
        "- Same-type low/high identity stability: low " +
        str(calibration["same_type_low_candidate_stability_count"]) + "/" +
        str(calibration["same_type_domain_subset_comparisons"]) + "; high " +
        str(calibration["same_type_high_candidate_stability_count"]) + "/" +
        str(calibration["same_type_domain_subset_comparisons"]) + ".",
        "- Exact within-domain ordering match: " +
        str(calibration["same_type_exact_order_match_count"]) + "/" +
        str(calibration["same_type_domain_subset_comparisons"]) + ".",
        "- Per-domain and per-unit rankings for video subsets {1,2}, {1,3}, {2,3} are in task041_phase_f_calibration_stability.csv.",
        "",
        "## Q7. Decision gate",
        "",
        "Decision: **" + str(gate["decision"]) + "**.",
        "",
        str(gate["rationale"]),
        "",
        "This is not approval to prune. No N=9 inference/validation was started; follow-up requires a separate instruction.",
        "",
        "## Baselines",
        "",
        "The matched frozen 18-candidate comparison reports mean_abs_d_original, G_RMS, PTR, corrected_pairwise_best_E, R_MCTC, and R_BCTR against full-validation logit, CE, and flip damage. Same-type and mixed-domain summaries are separate; all-domain values are secondary.",
        "",
        "| Criterion | Damage | Same-type rho | Same-type tau-b | correct/reverse/tie |",
        "|---|---|---:|---:|---:|",
    ]
    for criterion in BASELINE_CRITERIA:
        for metric in DAMAGE_METRICS:
            r = base("same_type", criterion, metric)
            lines.append(
                "| " + criterion + " | " + metric + " | " +
                fmt(r["domain_balanced_spearman"]) + " | " +
                fmt(r["domain_balanced_kendall_tau_b"]) + " | " +
                str(r["correct_count"]) + "/" + str(r["reverse_count"]) + "/" +
                str(r["tie_count"]) + " |"
            )
    lines += [
        "",
        "## Scope and stopping condition",
        "",
        "- No cross-domain ranking, global temporal importance, type coefficient, lambda, ridge, learned weight, or new pruning score was created.",
        "- Existing BMS and D_abs/D_rel/D_st were read-only inputs. No source descriptor, mask, model state, or historical artifact was changed.",
        "- New outputs are confined to " + str(output_dir) + ".",
        "- Phase F ends here: no Task042, N=9, GPU, pruning, or fine-tuning.",
    ]
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.perf_counter()
    inputs = {
        "phase_c_raw": Path(args.phase_c_raw).resolve(),
        "phase_d1_raw": Path(args.phase_d1_raw).resolve(),
        "frozen_units_csv": Path(args.frozen_units_csv).resolve(),
        "fullval_damage_csv": Path(args.fullval_damage_csv).resolve(),
        "baseline_scores_csv": Path(args.baseline_scores_csv).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite or reuse Phase F output directory: " + str(output_dir))
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(name + " not found: " + str(path))

    progress("loading and validating exact frozen identities")
    frozen, raw_to_uid = load_frozen(inputs["frozen_units_csv"])
    damage = load_fullval_damage(inputs["fullval_damage_csv"], frozen)
    baseline = load_baseline_scores(inputs["baseline_scores_csv"], frozen)
    for uid, row in damage.items():
        if not mask_restoration_is_exact(row):
            raise ValueError("full-validation mask restoration flag is missing, false, or conflicting: " + uid)
    if any(int(float(r["n_samples"])) != 3783 for r in damage.values()):
        raise ValueError("full-validation oracle is not consistently 3783 samples")
    progress("reconstructing signed raw frame-relation signatures")
    store, orders, video_keys, raw_meta = reconstruct_signatures(
        inputs["phase_c_raw"], inputs["phase_d1_raw"], frozen, raw_to_uid
    )

    signatures = make_signature_csv_rows(store, orders, frozen)
    progress("solving signed bounded collective and pairwise fits")
    reconstruction_rows, pairwise_rows, span_residuals, pair_summary = (
        run_signed_collective_and_pairwise(store, orders, frozen)
    )
    unit_rows = make_unit_rows(frozen, damage, baseline, span_residuals, pair_summary)
    primary_oracle_rows, primary_summary = make_primary_oracle(unit_rows)

    domain_members: Dict[str, List[str]] = defaultdict(list)
    unit_by_uid = {str(r["candidate_task037_global_index"]): r for r in unit_rows}
    for uid, row in frozen.items():
        domain_members[str(row["domain_id"])].append(uid)
    pairwise_unit_values = {
        uid: float(unit_by_uid[uid]["R_pair"]) for uid in unit_by_uid
    }
    improvements = [
        float(pairwise_unit_values[uid] - float(unit_by_uid[uid]["R_BCTR"]))
        for uid in unit_by_uid
    ]
    collective_pairwise_rows = make_collective_pairwise_rows(
        unit_rows, pairwise_rows, reconstruction_rows
    )
    collective_pairwise_summary = {
        "mean_pairwise_minus_collective": float(np.mean(improvements)),
        "median_pairwise_minus_collective": float(np.median(improvements)),
        "collective_le_pairwise_count": sum(
            float(unit_by_uid[str(r["candidate_task037_global_index"])]["R_BCTR"])
            <= float(unit_by_uid[str(r["candidate_task037_global_index"])]["R_pair"]) + 1e-10
            for r in unit_rows
        ),
        "strict_pairwise_greater_count": sum(x > 0.0 for x in improvements),
        "unit_count": len(improvements),
        "unit_rows": [
            {
                "candidate_task037_global_index": uid,
                "domain_id": str(unit_by_uid[uid]["domain_id"]),
                "R_BCTR": float(unit_by_uid[uid]["R_BCTR"]),
                "R_pair": float(unit_by_uid[uid]["R_pair"]),
                "pairwise_minus_collective": float(unit_by_uid[uid]["R_pair"] - unit_by_uid[uid]["R_BCTR"]),
            }
            for uid in sorted(unit_by_uid, key=lambda x: int(x))
        ],
    }

    progress("solving absolute and squared diagnostic-only ablations")
    ablation_rows, ablation_scores = run_representation_ablation(
        store, orders, frozen, unit_rows
    )
    progress("analyzing mixed-domain coefficients and directional reconstruction")
    mixed_rows, mixed_summary = make_mixed_rows(
        frozen, reconstruction_rows, store, orders
    )
    progress("recomputing leave-one-video-out calibration rankings")
    calibration_rows, calibration_summary = make_calibration_rows(
        frozen, store, orders, unit_rows, video_keys
    )
    progress("comparing frozen baseline scores on the matched 18-candidate cohort")
    baseline_rows, baseline_summary = make_baseline_rows(unit_rows, baseline)

    ablation_oracle_summary: Dict[str, Any] = {}
    for variant, scores in ablation_scores.items():
        variant_units = []
        for uid, score in scores.items():
            row = dict(unit_by_uid[uid])
            row["BCTR_variant"] = float(score)
            variant_units.append(row)
        domain_rows = []
        for domain in FROZEN_DOMAINS:
            scope = "same_type" if domain in SAME_TYPE_DOMAINS else "mixed"
            members = [r for r in variant_units if str(r["domain_id"]) == domain]
            for metric in DAMAGE_METRICS:
                domain_rows.extend(domain_score_oracle(members, "BCTR_variant", metric, scope))
        grouped: Dict[str, Dict[str, Any]] = {}
        for scope, domains in (("same_type", SAME_TYPE_DOMAINS), ("mixed", MIXED_DOMAINS)):
            grouped[scope] = {}
            for metric in DAMAGE_METRICS:
                selected = [r for r in domain_rows if r["scope"] == scope and r["damage_metric"] == metric]
                grouped[scope][metric] = {
                    "domain_balanced_spearman": safe_mean([r["spearman"] for r in selected]),
                    "domain_balanced_kendall_tau_b": safe_mean([r["kendall_tau_b"] for r in selected]),
                    "valid_domain_count": sum(r["spearman"] is not None for r in selected),
                }
        ablation_oracle_summary[variant] = grouped

    # Add signed/absolute/squared within-domain oracle summaries as rows in the ablation artifact.
    for variant, grouped in ablation_oracle_summary.items():
        for scope in ("same_type", "mixed"):
            for metric in DAMAGE_METRICS:
                stats = grouped[scope][metric]
                ablation_rows.append({
                    "row_level": "domain_balanced_oracle",
                    "representation": variant,
                    "scope": scope,
                    "damage_metric": metric,
                    "domain_balanced_spearman": stats["domain_balanced_spearman"],
                    "domain_balanced_kendall_tau_b": stats["domain_balanced_kendall_tau_b"],
                    "valid_domain_count": stats["valid_domain_count"],
                    "interpretation": "diagnostic-only; compare all representations",
                })

    span_counts = Counter(str(r["most_unique_span"]) for r in unit_rows)
    same_type_summary = primary_summary["same_type"]
    logit_gate = same_type_summary["mean_true_class_logit_drop"]
    ce_gate = same_type_summary["mean_cross_entropy_increase"]
    qualifies_logit = (
        int(logit_gate["correct_count"]) >= 4
        and logit_gate["domain_balanced_spearman"] is not None
        and float(logit_gate["domain_balanced_spearman"]) > 0.0
    )
    qualifies_ce = (
        int(ce_gate["correct_count"]) >= 4
        and ce_gate["domain_balanced_spearman"] is not None
        and float(ce_gate["domain_balanced_spearman"]) > 0.0
    )
    if qualifies_logit or qualifies_ce:
        decision = "RETAINED FOR N=9 VALIDATION"
        rationale = (
            "The signed collective R_BCTR passed the predeclared same-type gate for " +
            ("logit damage" if qualifies_logit else "CE damage") +
            ": majority-correct low/high ordering in the seven same-type domains and positive domain-balanced Spearman. "
            "This permits only a separately authorized N=9 validation; it does not approve pruning."
        )
    else:
        decision = "WEAK / UNRESOLVED"
        rationale = (
            "The predeclared gate was not met: no signed BCTR outcome simultaneously has majority-correct low/high ordering in the seven same-type domains and positive domain-balanced Spearman. "
            "The seven-domain sample remains descriptive; do not promote the method or start N=9 from this result."
        )

    raw_meta.update({
        "fullval_damage_rows": len(damage),
        "baseline_candidate_rows": len(baseline),
        "fullval_samples_per_unit": 3783,
        "fullval_mask_restoration_exact_for_all": True,
    })
    summary: Dict[str, Any] = {
        "task": "task041_phase_f_bms_conditioned_frame_relation_unit_selection_audit",
        "phase": "F",
        "offline_only": True,
        "model_inference": False,
        "gpu_used": False,
        "temporal_intervention_rerun": False,
        "bms_recomputed": False,
        "pruning": False,
        "finetuning": False,
        "descriptor_modified": False,
        "frozen_unit_count": len(frozen),
        "frozen_domains": list(FROZEN_DOMAINS),
        "same_type_primary_domains": list(SAME_TYPE_DOMAINS),
        "mixed_secondary_domains": list(MIXED_DOMAINS),
        "identity_fields_verified": [
            "Task037 global_index", "Task040 global_index", "domain_id",
            "layer_name", "unit_type", "unit_index", "stage",
        ],
        "fullval_damage_join_exact": len(damage) == N_UNITS,
        "raw_records": raw_meta,
        "signature_order": {
            "dimensions": 48,
            "ordering": "video identity (canonical video_index, video_id), then intervention identity (level, block_size, pair_index)",
            "identical_for_every_unit_and_span": True,
            "video_order": raw_meta["canonical_video_order"],
        },
        "solver": {
            "method": "scipy.optimize.lsq_linear(method='bvls')",
            "scipy_version": scipy.__version__,
            "dtype": "float64",
            "bounds": [0.0, 1.0],
            "lambda": None,
            "ridge": False,
            "sum_alpha_equals_one": False,
            "epsilon": EPS,
            "residual_ratio_clipped": False,
            "coefficient_support_definition": "alpha > 0 exactly; no threshold",
        },
        "same_type_oracle_summary": primary_summary,
        "mixed_domain_summary": mixed_summary,
        "collective_pairwise_unit_summary": collective_pairwise_summary,
        "signed_vs_magnitude_domain_balanced_summary": ablation_oracle_summary,
        "calibration_stability_summary": calibration_summary,
        "baseline_comparison_summary": baseline_summary,
        "most_unique_span_counts": {str(s): int(span_counts.get(str(s), 0)) for s in SPANS},
        "decision_gate": {
            "decision": decision,
            "qualifies_logit": qualifies_logit,
            "qualifies_ce": qualifies_ce,
            "majority_correct_threshold_from_instruction": 4,
            "domain_balanced_association_metric": "Spearman",
            "rationale": rationale,
        },
        "output_files": list(OUT_NAMES),
        "input_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "elapsed_seconds": float(time.perf_counter() - start),
    }

    if len(unit_rows) != N_UNITS or len(signatures) != N_UNITS * len(SPANS) * 48:
        raise ValueError("Phase F output identity/signature count gate failed")
    if not summary["fullval_damage_join_exact"]:
        raise ValueError("full-validation identity gate failed")
    if any(not r["collective_le_pairwise_within_numeric_tolerance"] for r in collective_pairwise_rows):
        raise ValueError("collective fit exceeded its feasible pairwise submodel beyond numeric tolerance")

    output_dir.mkdir(parents=True, exist_ok=False)
    signature_fields = union_fieldnames(signatures)
    recon_fields = union_fieldnames(reconstruction_rows)
    unit_fields = union_fieldnames(unit_rows)
    same_fields = union_fieldnames(primary_oracle_rows)
    mixed_fields = union_fieldnames(mixed_rows) if mixed_rows else []
    pair_fields = union_fieldnames(collective_pairwise_rows)
    ablation_fields = union_fieldnames(ablation_rows)
    calibration_fields = union_fieldnames(calibration_rows)
    baseline_fields = union_fieldnames(baseline_rows)
    write_csv(output_dir / OUT_NAMES[0], signature_fields, signatures)
    write_csv(output_dir / OUT_NAMES[1], recon_fields, reconstruction_rows)
    write_csv(output_dir / OUT_NAMES[2], unit_fields, unit_rows)
    write_csv(output_dir / OUT_NAMES[3], same_fields, primary_oracle_rows)
    write_csv(output_dir / OUT_NAMES[4], mixed_fields, mixed_rows)
    write_csv(output_dir / OUT_NAMES[5], pair_fields, collective_pairwise_rows)
    write_csv(output_dir / OUT_NAMES[6], ablation_fields, ablation_rows)
    write_csv(output_dir / OUT_NAMES[7], calibration_fields, calibration_rows)
    write_csv(output_dir / OUT_NAMES[8], baseline_fields, baseline_rows)

    report = make_report(
        summary, primary_oracle_rows, baseline_summary, mixed_summary,
        calibration_summary, collective_pairwise_rows, output_dir,
    )
    (output_dir / OUT_NAMES[10]).write_text(report, encoding="utf-8")
    write_json(output_dir / OUT_NAMES[9], summary)
    progress("wrote exactly 11 new Phase F files")
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    compact = {
        "task": summary["task"],
        "output_files": len(summary["output_files"]),
        "frozen_units": summary["frozen_unit_count"],
        "selected_interaction_rows": summary["raw_records"]["selected_interaction_rows"],
        "decision": summary["decision_gate"]["decision"],
        "same_type_logit_rho": summary["same_type_oracle_summary"]["same_type"]["mean_true_class_logit_drop"]["domain_balanced_spearman"],
        "same_type_logit_correct_reverse_tie": [
            summary["same_type_oracle_summary"]["same_type"]["mean_true_class_logit_drop"][k]
            for k in ("correct_count", "reverse_count", "tie_count")
        ],
        "gpu_used": summary["gpu_used"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

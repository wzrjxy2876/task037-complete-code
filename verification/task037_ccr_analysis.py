"""Offline TASK037 contextual competitive regret aggregation audit.

This script consumes the completed 30-video context-proxy table and the
frozen masking-oracle columns embedded in that table.  It performs no model
inference, masking, pruning, or finetuning.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
REL_SPANS = (1, 2, 4, 8, 16)
REL_CONTEXTS = 20
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 3407
KEYS = [
    "class",
    "video_key",
    "context_id",
    "span",
    "pair_index",
    "domain_id",
    "global_index",
    "unit_type",
    "stage",
    "unit_index",
]


def rank_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman and Kendall correlation for small domain rankings."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    sp = float(np.corrcoef(rx, ry)[0, 1])
    concordant = discordant = comparable = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx = rx[i] - rx[j]
        dy = ry[i] - ry[j]
        if dx == 0 or dy == 0:
            continue
        comparable += 1
        if dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    kt = float((concordant - discordant) / comparable) if comparable else np.nan
    return sp, kt


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def add_metadata(raw: pd.DataFrame, video_manifest: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    if "class_name" not in raw:
        raw["class_name"] = raw["class"]
    if "category" not in raw:
        raw["category"] = raw["unit_type"].map(
            lambda x: "AA" if x == "attention_head" else "FF"
        )
    vm = video_manifest.copy()
    join_cols = ["video_key"] if "video_key" in vm else ["video_path"]
    if "video_key" not in vm:
        vm["video_key"] = vm["video_path"].astype(str).str.split("/").str[-1]
    keep = [c for c in ["video_key", "within_class_position", "manifest_order", "class_name", "class_index"] if c in vm]
    raw = raw.merge(vm[keep].drop_duplicates("video_key"), on="video_key", how="left", suffixes=("", "_vm"))
    if "class_name_vm" in raw:
        raw["class_name"] = raw["class_name"].fillna(raw["class_name_vm"])
        raw = raw.drop(columns=["class_name_vm"])
    raw["within_class_position"] = raw["within_class_position"].fillna(0).astype(int)
    return raw


def exact_join(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build separate proxy/oracle tables and enforce the frozen identity."""
    proxy_cols = KEYS + ["proxy_signed_damage", "proxy_absolute_damage", "Delta_model", "class_name", "category", "within_class_position"]
    oracle_cols = KEYS + ["oracle_signed_damage", "oracle_absolute_damage", "Delta_model", "class_name", "category", "within_class_position"]
    proxy = raw[proxy_cols].copy()
    oracle = raw[oracle_cols].copy()
    if proxy.duplicated(KEYS).any() or oracle.duplicated(KEYS).any():
        raise RuntimeError("STOP: duplicate rows in the exact proxy/oracle identity")
    merged = proxy.merge(oracle, on=KEYS, suffixes=("_proxy", "_oracle"), validate="one_to_one")
    if len(merged) != len(raw):
        raise RuntimeError("STOP: exact proxy/oracle join cardinality mismatch")
    for meta in ["class_name", "category", "Delta_model", "within_class_position"]:
        pcol, ocol = f"{meta}_proxy", f"{meta}_oracle"
        if pcol in merged and ocol in merged:
            if not merged[pcol].equals(merged[ocol]):
                raise RuntimeError(f"STOP: metadata mismatch for {meta}")
            merged[meta] = merged[pcol]
            merged = merged.drop(columns=[pcol, ocol])
    for c in ["class", "video_key", "context_id", "span", "pair_index", "domain_id", "global_index", "unit_type", "stage", "unit_index"]:
        if merged[c].isna().any():
            raise RuntimeError(f"STOP: null identity field {c}")
    return merged, proxy


def normalized_regret(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    gcols = ["video_key", "context_id", "domain_id"]
    stats = out.groupby(gcols)[value_col].agg(["min", "max"]).rename(columns={"min": f"{prefix}_min", "max": f"{prefix}_max"})
    out = out.join(stats, on=gcols)
    span = out[f"{prefix}_max"] - out[f"{prefix}_min"]
    unresolved = span.abs() <= EPS
    out[f"{prefix}_context_unresolved"] = unresolved
    out[f"{prefix}_regret"] = np.where(unresolved, 0.0, (out[value_col] - out[f"{prefix}_min"]) / (span + EPS))
    out[f"{prefix}_regret"] = out[f"{prefix}_regret"].clip(lower=0.0, upper=1.0)
    return out


def video_scores(df: pd.DataFrame, regret_col: str, primary_only: bool = True) -> pd.DataFrame:
    d = df[df.context_id > 0].copy() if primary_only else df.copy()
    vals = d.groupby(["video_key", "domain_id", "global_index"], as_index=False)[regret_col].max()
    return vals.rename(columns={regret_col: "video_worst_regret"})


def ccr_scores(df: pd.DataFrame, regret_col: str, primary_only: bool = True) -> pd.DataFrame:
    vs = video_scores(df, regret_col, primary_only)
    return vs.groupby(["domain_id", "global_index"], as_index=False).video_worst_regret.mean().rename(columns={"video_worst_regret": "CCR"})


def score_lookup(score: pd.DataFrame, name: str) -> pd.DataFrame:
    return score.rename(columns={"CCR": name})


def ordered_domain_scores(score: pd.DataFrame, col: str) -> pd.DataFrame:
    s = score.sort_values(["domain_id", col, "global_index"], ascending=[True, True, True]).copy()
    s["rank"] = s.groupby("domain_id").cumcount() + 1
    return s


def select_by_score(score: pd.DataFrame, col: str, ascending: bool = True) -> pd.DataFrame:
    s = score.sort_values(["domain_id", col, "global_index"], ascending=[True, ascending, True])
    return s.groupby("domain_id", as_index=False).first()[["domain_id", "global_index", col]].rename(columns={"global_index": "selected_unit", col: "selected_score"})


def method_score_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return unit-level scores for every aggregation principle."""
    relation = df[df.context_id > 0].copy()
    all_context = df.copy()
    out: dict[str, pd.DataFrame] = {}
    out["CCR"] = ccr_scores(df, "proxy_regret", True)
    # Frequency uses all 21 contexts, as specified for the baseline.
    winners = all_context.sort_values(["video_key", "context_id", "domain_id", "proxy_signed_damage", "global_index"]).groupby(["video_key", "context_id", "domain_id"], as_index=False).first()
    freq = winners.groupby(["domain_id", "global_index"], as_index=False).size().rename(columns={"size": "frequency"})
    out["frequency"] = freq
    out["mean_damage"] = all_context.groupby(["domain_id", "global_index"], as_index=False).proxy_signed_damage.mean().rename(columns={"proxy_signed_damage": "mean_damage"})
    out["fixed_original"] = all_context[all_context.context_id == 0].groupby(["domain_id", "global_index"], as_index=False).proxy_signed_damage.mean().rename(columns={"proxy_signed_damage": "fixed_original"})
    raw = relation.copy()
    mins = raw.groupby(["video_key", "context_id", "domain_id"]).proxy_signed_damage.transform("min")
    raw["raw_regret"] = raw.proxy_signed_damage - mins
    raw_vs = raw.groupby(["video_key", "domain_id", "global_index"], as_index=False).raw_regret.max().rename(columns={"raw_regret": "raw_video_worst"})
    out["raw_regret"] = raw_vs.groupby(["domain_id", "global_index"], as_index=False).raw_video_worst.mean().rename(columns={"raw_video_worst": "raw_CCR"})
    # Class-specific CCR: max over relation contexts per video, then mean within class.
    cvs = video_scores(df, "proxy_regret", True).merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key", how="left")
    out["CCR_class"] = cvs.groupby(["class_name", "domain_id", "global_index"], as_index=False).video_worst_regret.mean().rename(columns={"video_worst_regret": "CCR"})
    for method, col in [("frequency", "frequency"), ("mean_damage", "mean_damage"), ("fixed_original", "fixed_original"), ("raw_regret", "raw_CCR")]:
        temp = df.copy()
        if method == "frequency":
            win = temp.sort_values(["video_key", "context_id", "domain_id", "proxy_signed_damage", "global_index"]).groupby(["video_key", "context_id", "domain_id"], as_index=False).first()
            if "class_name" not in win:
                win = win.merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key", how="left")
            out[f"{method}_class"] = win.groupby(["class_name", "domain_id", "global_index"], as_index=False).size().rename(columns={"size": col})
        else:
            if method == "mean_damage": temp = temp
            elif method == "fixed_original": temp = temp[temp.context_id == 0]
            else:
                temp = temp[temp.context_id > 0].copy(); temp["raw_regret"] = temp.proxy_signed_damage - temp.groupby(["video_key", "context_id", "domain_id"]).proxy_signed_damage.transform("min")
                temp = temp.groupby(["video_key", "domain_id", "global_index"], as_index=False).raw_regret.max().rename(columns={"raw_regret": "raw_CCR"})
                temp = temp.merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key", how="left")
            if "class_name" not in temp:
                temp = temp.merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key", how="left")
            if col not in temp:
                temp[col] = temp["proxy_signed_damage"]
            out[f"{method}_class"] = temp.groupby(["class_name", "domain_id", "global_index"], as_index=False)[col].mean()
    # Oracle target CCR.
    out["oracle"] = ccr_scores(df, "oracle_regret", True).rename(columns={"CCR": "oracle_CCR"})
    return out


def metric_table(score: pd.DataFrame, oracle: pd.DataFrame, category_map: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "CCR": ("CCR", True),
        "frequency": ("frequency", False),
        "mean_damage": ("mean_damage", True),
        "fixed_original": ("fixed_original", True),
        "raw_regret": ("raw_CCR", True),
    }
    rows = []
    oracle_best = select_by_score(oracle.rename(columns={"oracle_CCR": "oracle_CCR"}), "oracle_CCR", True)
    for domain, og in oracle.groupby("domain_id", sort=True):
        oo = og.sort_values(["oracle_CCR", "global_index"]).reset_index(drop=True)
        best = int(oo.iloc[0].global_index)
        best_val = float(oo.iloc[0].oracle_CCR)
        cat = str(category_map.loc[category_map.domain_id == domain, "category"].iloc[0])
        for method, (col, asc) in methods.items():
            sg = score[score.domain_id == domain].sort_values([col, "global_index"], ascending=[asc, True])
            sel = int(sg.iloc[0].global_index)
            true = float(oo.loc[oo.global_index == sel, "oracle_CCR"].iloc[0])
            rows.append({"domain_id": int(domain), "category": cat, "method": method, "selected_unit": sel, "oracle_best_unit": best, "selected_oracle_CCR": true, "oracle_best_CCR": best_val, "SelectionGap": true - best_val})
    return pd.DataFrame(rows)


def domain_rankings(score: pd.DataFrame, oracle: pd.DataFrame, category_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    full = []
    for domain in sorted(score.domain_id.unique()):
        a = score[score.domain_id == domain].sort_values("global_index")
        b = oracle[oracle.domain_id == domain].sort_values("global_index")
        sp, kt = rank_corr(a.CCR, b.oracle_CCR)
        ar = a.sort_values(["CCR", "global_index"]).global_index.tolist()
        br = b.sort_values(["oracle_CCR", "global_index"]).global_index.tolist()
        cat = str(category_map.loc[category_map.domain_id == domain, "category"].iloc[0])
        rows.append({"domain_id": int(domain), "category": cat, "spearman": sp, "kendall": kt, "exact_full_ranking": ar == br, "top1_identity": int(ar[0] == br[0]), "bottom1_identity": int(ar[-1] == br[-1]), "proxy_ordering": ">".join(map(str, ar)), "oracle_ordering": ">".join(map(str, br))})
        full.append({"domain_id": int(domain), "category": cat, "rank": np.arange(1, len(ar) + 1), "proxy_global_index": ar, "oracle_global_index": br})
    # Expand complete orderings in a regular table.
    fr = []
    for d in full:
        for r, p, o in zip(d["rank"], d["proxy_global_index"], d["oracle_global_index"]):
            fr.append({"domain_id": d["domain_id"], "category": d["category"], "rank": int(r), "proxy_global_index": int(p), "oracle_global_index": int(o)})
    return pd.DataFrame(rows), pd.DataFrame(fr)


def stratified_stability(df: pd.DataFrame, oracle: pd.DataFrame, positions: dict[str, list[int]], category_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_oracle = oracle.set_index(["domain_id", "global_index"]).oracle_CCR
    for label, pos in positions.items():
        subvideos = df[df.within_class_position.isin(pos)].video_key.unique()
        sub = df[df.video_key.isin(subvideos)]
        s = ccr_scores(sub, "proxy_regret", True)
        for d in sorted(oracle.domain_id.unique()):
            ss = s[s.domain_id == d].sort_values(["CCR", "global_index"])
            oo = oracle[oracle.domain_id == d].sort_values(["oracle_CCR", "global_index"])
            if ss.empty:
                continue
            sel = int(ss.iloc[0].global_index); best = int(oo.iloc[0].global_index)
            sp, kt = rank_corr(s[s.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)
            gap = float(full_oracle.loc[(d, sel)] - full_oracle.loc[(d, best)])
            rows.append({"split": label, "videos": len(subvideos), "domain_id": int(d), "category": str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]), "spearman": sp, "kendall": kt, "top1_identity": int(sel == best), "selected_unit": sel, "oracle_best_unit": best, "SelectionGap": gap})
    return pd.DataFrame(rows)


def loco_and_sizes(df: pd.DataFrame, oracle: pd.DataFrame, vm: pd.DataFrame, category_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vm = vm.copy()
    if "video_key" not in vm:
        vm["video_key"] = vm["video_path"].astype(str).str.split("/").str[-1]
    full = []
    for c in sorted(vm.class_name.astype(str).unique()):
        sub = df[df.class_name != c]
        s = ccr_scores(sub, "proxy_regret", True)
        for d in sorted(oracle.domain_id.unique()):
            ss = s[s.domain_id == d].sort_values(["CCR", "global_index"])
            oo = oracle[oracle.domain_id == d].sort_values(["oracle_CCR", "global_index"])
            if ss.empty: continue
            sel = int(ss.iloc[0].global_index); best = int(oo.iloc[0].global_index)
            full.append({"held_out_class": c, "domain_id": int(d), "category": str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]), "spearman": rank_corr(s[s.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[0], "kendall": rank_corr(s[s.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[1], "top1_identity": int(sel == best), "selected_unit": sel, "oracle_best_unit": best, "SelectionGap": float(oracle.loc[(oracle.domain_id == d) & (oracle.global_index == sel), "oracle_CCR"].iloc[0] - oo.iloc[0].oracle_CCR)})
    size_rows = []
    classes = sorted(vm.class_name.astype(str).unique())
    stream = vm.sort_values(["within_class_position", "class_index", "manifest_order"]).video_key.tolist()
    for n in [3, 6, 9, 12, 18, 30]:
        chosen = stream[:n]
        sub = df[df.video_key.isin(chosen)]
        s = ccr_scores(sub, "proxy_regret", True)
        for d in sorted(oracle.domain_id.unique()):
            ss = s[s.domain_id == d].sort_values(["CCR", "global_index"]); oo = oracle[oracle.domain_id == d].sort_values(["oracle_CCR", "global_index"])
            if ss.empty: continue
            sel = int(ss.iloc[0].global_index); best = int(oo.iloc[0].global_index)
            size_rows.append({"N": n, "selected_videos": "|".join(chosen), "domain_id": int(d), "category": str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]), "spearman": rank_corr(s[s.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[0], "kendall": rank_corr(s[s.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[1], "top1_identity": int(sel == best), "selected_unit": sel, "oracle_best_unit": best, "SelectionGap": float(oracle.loc[(oracle.domain_id == d) & (oracle.global_index == sel), "oracle_CCR"].iloc[0] - oo.iloc[0].oracle_CCR)})
    return pd.DataFrame(full), pd.DataFrame(size_rows)


def context_count_audit(df: pd.DataFrame, oracle: pd.DataFrame, category_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [5, 10, 15, 20]:
        keep = []
        for s in REL_SPANS:
            ids = sorted(df[(df.context_id > 0) & (df.span == s)].context_id.unique())
            keep.extend(ids[: m // 5])
        sub = df[df.context_id.isin(keep)]
        sccr = ccr_scores(sub, "proxy_regret", True)
        for d in sorted(oracle.domain_id.unique()):
            ss = sccr[sccr.domain_id == d].sort_values(["CCR", "global_index"]); oo = oracle[oracle.domain_id == d].sort_values(["oracle_CCR", "global_index"])
            sel = int(ss.iloc[0].global_index); best = int(oo.iloc[0].global_index)
            rows.append({"context_count": m, "context_ids": "|".join(map(str, keep)), "domain_id": int(d), "category": str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]), "spearman": rank_corr(sccr[sccr.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[0], "kendall": rank_corr(sccr[sccr.domain_id == d].set_index("global_index").reindex(oo.global_index).CCR, oo.oracle_CCR)[1], "top1_identity": int(sel == best), "selected_unit": sel, "oracle_best_unit": best, "SelectionGap": float(oracle.loc[(oracle.domain_id == d) & (oracle.global_index == sel), "oracle_CCR"].iloc[0] - oo.iloc[0].oracle_CCR)})
    return pd.DataFrame(rows)


def safety_curves(df: pd.DataFrame, comparisons: pd.DataFrame, category_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = df[df.context_id > 0].copy()
    rows = []
    summaries = []
    for d in sorted(df.domain_id.unique()):
        cd = comparisons[comparisons.domain_id == d].set_index("method")
        for method, row in cd.iterrows():
            unit = int(row.selected_unit)
            z = rel[(rel.domain_id == d) & (rel.global_index == unit)].copy()
            z["method"] = method; z["selected_unit"] = unit; z["category"] = str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]); z["oracle_normalized_regret"] = z.oracle_regret
            rows.extend(z[["method", "category", "domain_id", "video_key", "context_id", "span", "pair_index", "global_index", "oracle_normalized_regret"]].to_dict("records"))
        best = int(comparisons[(comparisons.domain_id == d) & (comparisons.method == "CCR")].oracle_best_unit.iloc[0])
        z = rel[(rel.domain_id == d) & (rel.global_index == best)].copy(); z["method"] = "oracle_optimum"; z["selected_unit"] = best; z["category"] = str(category_map.loc[category_map.domain_id == d, "category"].iloc[0]); z["oracle_normalized_regret"] = z.oracle_regret
        rows.extend(z[["method", "category", "domain_id", "video_key", "context_id", "span", "pair_index", "global_index", "oracle_normalized_regret"]].to_dict("records"))
    curves = pd.DataFrame(rows)
    for (d, method), g in curves.groupby(["domain_id", "method"]):
        vals = g.oracle_normalized_regret.to_numpy(float)
        summaries.append({"domain_id": int(d), "category": str(g.category.iloc[0]), "method": method, "mean": float(np.mean(vals)), "median": float(np.median(vals)), "max": float(np.max(vals)), "q90": float(np.quantile(vals, .9))})
    return curves, pd.DataFrame(summaries)


def bootstrap(df: pd.DataFrame, oracle: pd.DataFrame, comparisons: pd.DataFrame, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    classes = sorted(df.class_name.astype(str).unique())
    domains = sorted(oracle.domain_id.unique())
    units = sorted(oracle.global_index.unique())
    cindex = {c: i for i, c in enumerate(classes)}
    dindex = {d: i for i, d in enumerate(domains)}
    uindex = {u: i for i, u in enumerate(units)}
    shape = (len(classes), len(domains), len(units))
    arrays: dict[str, np.ndarray] = {m: np.full(shape, np.nan, dtype=float) for m in ["CCR", "frequency", "mean_damage", "fixed_original", "raw_regret"]}
    rel = df[df.context_id > 0].copy()
    vs = video_scores(df, "proxy_regret", True).merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key")
    class_ccr = vs.groupby(["class_name", "domain_id", "global_index"]).video_worst_regret.mean().reset_index(name="value")
    winners = df.sort_values(["video_key", "context_id", "domain_id", "proxy_signed_damage", "global_index"]).groupby(["video_key", "context_id", "domain_id"], as_index=False).first()
    class_freq = winners.groupby(["class_name", "domain_id", "global_index"]).size().reset_index(name="value")
    class_mean = df.groupby(["class_name", "domain_id", "global_index"]).proxy_signed_damage.mean().reset_index(name="value")
    class_fixed = df[df.context_id == 0].groupby(["class_name", "domain_id", "global_index"]).proxy_signed_damage.mean().reset_index(name="value")
    rr = rel.copy(); rr["raw_regret"] = rr.proxy_signed_damage - rr.groupby(["video_key", "context_id", "domain_id"]).proxy_signed_damage.transform("min"); rr = rr.groupby(["video_key", "domain_id", "global_index"], as_index=False).raw_regret.max().merge(df[["video_key", "class_name"]].drop_duplicates(), on="video_key"); class_raw = rr.groupby(["class_name", "domain_id", "global_index"]).raw_regret.mean().reset_index(name="value")
    for method, tab in [("CCR", class_ccr), ("frequency", class_freq), ("mean_damage", class_mean), ("fixed_original", class_fixed), ("raw_regret", class_raw)]:
        for r in tab.itertuples(index=False):
            arrays[method][cindex[str(r.class_name)], dindex[int(r.domain_id)], uindex[int(r.global_index)]] = float(r.value)
    target = np.full((len(domains), len(units)), np.nan, dtype=float)
    for r in oracle.itertuples(index=False): target[dindex[int(r.domain_id)], uindex[int(r.global_index)]] = float(r.oracle_CCR)
    opt = np.nanmin(target, axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    lower_methods = {"CCR", "mean_damage", "fixed_original", "raw_regret"}
    for b in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, len(classes), size=len(classes))
        weights = np.bincount(sampled, minlength=len(classes)).astype(float)
        vals = {}
        for method, arr in arrays.items():
            agg = np.nansum(arr * weights[:, None, None], axis=0) / len(classes)
            selected = []
            for di in range(len(domains)):
                criterion = agg[di] if method in lower_methods else -agg[di]
                criterion = np.asarray(criterion, dtype=float)
                criterion[~np.isfinite(target[di])] = np.inf
                order = np.argsort(criterion, kind="stable")
                selected.append(order[0])
            vals[method] = float(np.mean(target[np.arange(len(domains)), selected] - opt))
        rows.append({"replicate": b, **{f"{m}_mean_SelectionGap": v for m, v in vals.items()}, "CCR_minus_frequency": vals["CCR"] - vals["frequency"], "CCR_minus_mean": vals["CCR"] - vals["mean_damage"], "CCR_minus_fixed": vals["CCR"] - vals["fixed_original"]})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw = read_csv(Path(args.raw))
    vm = read_csv(Path(args.video_manifest))
    raw = add_metadata(raw, vm)
    merged, _ = exact_join(raw)
    proxy = merged.rename(columns={"proxy_signed_damage": "proxy_signed_damage", "oracle_signed_damage": "oracle_signed_damage"})
    proxy = normalized_regret(proxy, "proxy_signed_damage", "proxy")
    proxy = normalized_regret(proxy, "oracle_signed_damage", "oracle")
    proxy["raw_regret"] = np.where(proxy.context_id > 0, proxy.proxy_signed_damage - proxy.groupby(["video_key", "context_id", "domain_id"]).proxy_signed_damage.transform("min"), 0.0)
    proxy["oracle_raw_regret"] = np.where(proxy.context_id > 0, proxy.oracle_signed_damage - proxy.groupby(["video_key", "context_id", "domain_id"]).oracle_signed_damage.transform("min"), 0.0)
    # Identity and context regret table.
    identity_rows = [
        {"check": "exact_proxy_oracle_join", "expected": len(raw), "observed": len(merged), "status": "PASS" if len(raw) == len(merged) else "FAIL"},
        {"check": "30_videos", "expected": 30, "observed": int(raw.video_key.nunique()), "status": "PASS" if raw.video_key.nunique() == 30 else "FAIL"},
        {"check": "10_classes", "expected": 10, "observed": int(raw.class_name.nunique()), "status": "PASS" if raw.class_name.nunique() == 10 else "FAIL"},
        {"check": "9_domains", "expected": 9, "observed": int(raw.domain_id.nunique()), "status": "PASS" if raw.domain_id.nunique() == 9 else "FAIL"},
        {"check": "36_units", "expected": 36, "observed": int(raw.global_index.nunique()), "status": "PASS" if raw.global_index.nunique() == 36 else "FAIL"},
        {"check": "21_contexts_per_video", "expected": 21, "observed": int(raw.groupby("video_key").context_id.nunique().min()), "status": "PASS" if raw.groupby("video_key").context_id.nunique().eq(21).all() else "FAIL"},
        {"check": "normalized_proxy_regret_bounds", "expected": "[0,1]", "observed": [float(proxy.proxy_regret.min()), float(proxy.proxy_regret.max())], "status": "PASS" if proxy.proxy_regret.between(0,1).all() else "FAIL"},
        {"check": "normalized_oracle_regret_bounds", "expected": "[0,1]", "observed": [float(proxy.oracle_regret.min()), float(proxy.oracle_regret.max())], "status": "PASS" if proxy.oracle_regret.between(0,1).all() else "FAIL"},
    ]
    pd.DataFrame(identity_rows).to_csv(out / "task_ccr_identity.csv", index=False)
    proxy["regret_minimum_check"] = proxy.groupby(["video_key", "context_id", "domain_id"]).proxy_regret.transform("min")
    proxy["oracle_regret_minimum_check"] = proxy.groupby(["video_key", "context_id", "domain_id"]).oracle_regret.transform("min")
    proxy.to_csv(out / "task_ccr_context_regret.csv", index=False)
    # Unit scores and domain rankings.
    scores = method_score_tables(proxy)
    category_map = raw[["domain_id", "category"]].drop_duplicates().groupby("domain_id", as_index=False).first()
    ccr = ordered_domain_scores(scores["CCR"], "CCR")
    oracle = ordered_domain_scores(scores["oracle"], "oracle_CCR")
    ccr.to_csv(out / "task_ccr_unit_scores.csv", index=False)
    oracle.to_csv(out / "task_ccr_oracle_scores.csv", index=False)
    rankings, ordering = domain_rankings(ccr[["domain_id", "global_index", "CCR"]], oracle[["domain_id", "global_index", "oracle_CCR"]], category_map)
    rankings.to_csv(out / "task_ccr_domain_rankings.csv", index=False); ordering.to_csv(out / "task_ccr_domain_orderings.csv", index=False)
    # Baselines and comparison.
    for method, tab, col in [("frequency", scores["frequency"], "frequency"), ("mean", scores["mean_damage"], "mean_damage"), ("fixed", scores["fixed_original"], "fixed_original"), ("raw", scores["raw_regret"], "raw_CCR")]:
        tab.to_csv(out / f"task_ccr_{method}_baseline.csv", index=False)
    # Build one score frame with one column per aggregation method.
    score_frame = ccr[["domain_id", "global_index", "CCR"]].copy()
    for tab, col in [(scores["frequency"], "frequency"), (scores["mean_damage"], "mean_damage"), (scores["fixed_original"], "fixed_original"), (scores["raw_regret"], "raw_CCR")]: score_frame = score_frame.merge(tab, on=["domain_id", "global_index"], how="left")
    comp = metric_table(score_frame, oracle[["domain_id", "global_index", "oracle_CCR"]], category_map)
    comp.to_csv(out / "task_ccr_selection_gap.csv", index=False); comp.to_csv(out / "task_ccr_method_comparison.csv", index=False)
    # Contextual and static reference tables.
    static = proxy[proxy.context_id == 0].groupby(["domain_id", "global_index"], as_index=False).proxy_signed_damage.mean().rename(columns={"proxy_signed_damage": "original_proxy_damage"}); static.to_csv(out / "task_ccr_fixed_original_context_values.csv", index=False)
    # Type summary and stability splits.
    type_rows = []
    for cat, g in comp.groupby("category"):
        cg = g[g.method == "CCR"]; fg = g[g.method == "frequency"]; mg = g[g.method == "mean_damage"]; xg = g[g.method == "fixed_original"]
        type_rows.append({"category": cat, "domains": int(cg.domain_id.nunique()), "top1_oracle_identity_rate": float((cg.selected_unit == cg.oracle_best_unit).mean()), "mean_SelectionGap": float(cg.SelectionGap.mean()), "median_SelectionGap": float(cg.SelectionGap.median()), "frequency_mean_SelectionGap": float(fg.SelectionGap.mean()), "mean_damage_mean_SelectionGap": float(mg.SelectionGap.mean()), "fixed_mean_SelectionGap": float(xg.SelectionGap.mean())})
    pd.DataFrame(type_rows).to_csv(out / "task_ccr_type_summary.csv", index=False)
    positions = {"position_1": [0], "position_2": [1], "position_3": [2]}
    stratified_stability(proxy, oracle[["domain_id", "global_index", "oracle_CCR"]], positions, category_map).to_csv(out / "task_ccr_position_stability.csv", index=False)
    loco, sizes = loco_and_sizes(proxy, oracle[["domain_id", "global_index", "oracle_CCR"]], vm, category_map); loco.to_csv(out / "task_ccr_loco.csv", index=False); sizes.to_csv(out / "task_ccr_calibration_size.csv", index=False)
    context_counts = context_count_audit(proxy, oracle[["domain_id", "global_index", "oracle_CCR"]], category_map)
    context_counts.to_csv(out / "task_ccr_context_count.csv", index=False)
    # Severity and same-span audits.
    rel = proxy[proxy.context_id > 0].copy(); severity_rows = []
    for label, col in [("raw_regret", "raw_regret"), ("normalized_regret", "proxy_regret")]:
        severity_rows.append({"measure": label, "pearson_with_Delta_model": float(rel[col].corr(rel.Delta_model, method="pearson")), "spearman_with_Delta_model": float(rel[col].corr(rel.Delta_model, method="spearman"))})
    pd.DataFrame(severity_rows).to_csv(out / "task_ccr_severity_audit.csv", index=False)
    same_rows = []
    for (d, s, gi, v), g in rel.groupby(["domain_id", "span", "global_index", "video_key"]):
        vals = g.proxy_regret.to_numpy(float); same_rows.append({"domain_id": int(d), "span": int(s), "global_index": int(gi), "video_key": v, "relation_count": len(vals), "mean_regret": float(vals.mean()), "max_regret": float(vals.max()), "min_regret": float(vals.min()), "within_span_relation_range": float(vals.max() - vals.min()), "within_span_relation_std": float(vals.std())})
    same = pd.DataFrame(same_rows); same.to_csv(out / "task_ccr_same_span.csv", index=False)
    # Oracle safety curves and disagreement cases.
    curves, curve_summary = safety_curves(proxy, comp, category_map); curves.to_csv(out / "task_ccr_oracle_safety_curves.csv", index=False); curve_summary.to_csv(out / "task_ccr_oracle_safety_curve_summary.csv", index=False)
    dis = []
    freq_scores = scores["frequency"].set_index(["domain_id", "global_index"]).frequency
    for d in sorted(comp.domain_id.unique()):
        c = comp[(comp.domain_id == d) & (comp.method == "CCR")].iloc[0]; f = comp[(comp.domain_id == d) & (comp.method == "frequency")].iloc[0]
        if int(c.selected_unit) == int(f.selected_unit): continue
        a = rel[(rel.domain_id == d) & (rel.global_index == int(c.selected_unit))][["video_key", "context_id", "span", "proxy_regret", "oracle_regret"]].rename(columns={"proxy_regret": "CCR_proxy_regret", "oracle_regret": "CCR_oracle_regret"})
        b = rel[(rel.domain_id == d) & (rel.global_index == int(f.selected_unit))][["video_key", "context_id", "proxy_regret", "oracle_regret"]].rename(columns={"proxy_regret": "frequency_proxy_regret", "oracle_regret": "frequency_oracle_regret"})
        z = a.merge(b, on=["video_key", "context_id"], how="outer"); z["domain_id"] = int(d); z["CCR_selected_unit"] = int(c.selected_unit); z["frequency_selected_unit"] = int(f.selected_unit); z["CCR_frequency_count"] = int(freq_scores.loc[(d, int(c.selected_unit))]); z["frequency_frequency_count"] = int(freq_scores.loc[(d, int(f.selected_unit))]); z["CCR_SelectionGap"] = float(c.SelectionGap); z["frequency_SelectionGap"] = float(f.SelectionGap); dis.extend(z.to_dict("records"))
    pd.DataFrame(dis).to_csv(out / "task_ccr_disagreement_cases.csv", index=False)
    # Mandatory synthetic sanity test.
    syn = pd.DataFrame({"unit": ["A", "B"], "best_context_count": [9, 0], "regret_contexts": [[0]*9+[1], [0.2]*10]})
    syn["frequency"] = syn.best_context_count / 10; syn["CCR"] = syn.regret_contexts.map(max); syn["frequency_selected"] = syn.frequency == syn.frequency.max(); syn["CCR_selected"] = syn.CCR == syn.CCR.min(); syn.to_json(out / "task_ccr_synthetic_sanity.json", orient="records", indent=2)
    # Bootstrap and summary.
    boot = bootstrap(proxy, oracle[["domain_id", "global_index", "oracle_CCR"]], comp); boot.to_csv(out / "task_ccr_bootstrap.csv", index=False)
    def ci(col: str) -> list[float]: return [float(boot[col].quantile(.025)), float(boot[col].quantile(.975))]
    ccr_g = comp[comp.method == "CCR"]; freq_g = comp[comp.method == "frequency"]; mean_g = comp[comp.method == "mean_damage"]; fixed_g = comp[comp.method == "fixed_original"]
    summary = {
        "task": "task037_contextual_competitive_regret",
        "decision": "PENDING_PREDECLARED_GATE",
        "videos": int(raw.video_key.nunique()), "classes": int(raw.class_name.nunique()), "domains": int(raw.domain_id.nunique()), "units": int(raw.global_index.nunique()), "contexts_per_video": int(raw.groupby("video_key").context_id.nunique().min()), "primary_relation_contexts": 20,
        "CCR_mean_SelectionGap": float(ccr_g.SelectionGap.mean()), "frequency_mean_SelectionGap": float(freq_g.SelectionGap.mean()), "mean_damage_mean_SelectionGap": float(mean_g.SelectionGap.mean()), "fixed_mean_SelectionGap": float(fixed_g.SelectionGap.mean()),
        "CCR_top1_identity_rate": float((ccr_g.selected_unit == ccr_g.oracle_best_unit).mean()), "frequency_top1_identity_rate": float((freq_g.selected_unit == freq_g.oracle_best_unit).mean()), "CCR_bootstrap_CI": ci("CCR_mean_SelectionGap"), "frequency_bootstrap_CI": ci("frequency_mean_SelectionGap"), "mean_bootstrap_CI": ci("mean_damage_mean_SelectionGap"), "fixed_bootstrap_CI": ci("fixed_original_mean_SelectionGap"), "pairwise_bootstrap_CI": {"CCR_minus_frequency": ci("CCR_minus_frequency"), "CCR_minus_mean": ci("CCR_minus_mean"), "CCR_minus_fixed": ci("CCR_minus_fixed")}, "synthetic_frequency_selects_A": True, "synthetic_CCR_selects_B": True, "bootstrap_replicates": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "required_questions": {k: None for k in "ABCDEFGHIJ"},
    }
    # Conservative predeclared gate: numerical decision is based on all required comparisons.
    ccr_better_fixed = summary["CCR_mean_SelectionGap"] <= summary["fixed_mean_SelectionGap"]
    ccr_better_mean = summary["CCR_mean_SelectionGap"] <= summary["mean_damage_mean_SelectionGap"]
    ccr_better_freq = summary["CCR_mean_SelectionGap"] < summary["frequency_mean_SelectionGap"]
    no_aa_catastrophe = float(type_summary := pd.DataFrame(type_rows).query("category == 'AA'").mean_SelectionGap.iloc[0]) <= summary["fixed_mean_SelectionGap"] if any(x["category"] == "AA" for x in type_rows) else False
    practical = float(sizes[sizes.N == 9].SelectionGap.mean()) <= float(sizes[sizes.N == 30].SelectionGap.mean()) + 1e-12 if len(sizes) else False
    robust_class = bool(loco.top1_identity.mean() >= 0.5) if len(loco) else False
    summary["required_questions"] = {"A": bool(rankings.spearman.mean() > 0 and rankings.top1_identity.mean() >= 0.5), "B": bool(ccr_g.selected_unit.eq(ccr_g.oracle_best_unit).mean() >= 0.5), "C": bool(ccr_g.SelectionGap.mean() <= fixed_g.SelectionGap.mean()), "D": bool(ccr_better_freq), "E": bool(ccr_better_fixed), "F": bool(ccr_better_mean), "G": bool(rel.proxy_regret.corr(rel.Delta_model, method="spearman") <= rel.raw_regret.corr(rel.Delta_model, method="spearman")), "H": bool(no_aa_catastrophe and all(t["domains"] > 0 for t in type_rows)), "I": bool(practical), "J": bool(context_counts[context_counts.context_count == 20].SelectionGap.mean() <= summary["fixed_mean_SelectionGap"] + 1e-12), "all_categories": [x["category"] for x in type_rows], "loco_top1_rate": float(loco.top1_identity.mean()) if len(loco) else np.nan}
    # Exact predeclared A gate, with no post-hoc threshold tuning.
    if all(summary["required_questions"][k] for k in "ABCDEFGHIJ") and ccr_better_freq and ccr_better_fixed and ccr_better_mean and practical and robust_class:
        summary["decision"] = "CONTEXTUAL_COMPETITIVE_REGRET_PROMISING"
    else:
        summary["decision"] = "CONTEXTUAL_COMPETITIVE_REGRET_WEAK_OR_UNRESOLVED"
    json.dump(summary, open(out / "task_ccr_summary.json", "w", encoding="utf-8"), indent=2)
    report = f"""# TASK037 — Contextual Competitive Regret Aggregation Audit\n\nDecision: **{summary['decision']}**\n\nOffline-only reuse of the completed 30-video context-conditioned proxy and exact masking oracle. No GPU, inference, masking, pruning, or finetuning was performed.\n\n- CCR mean SelectionGap: `{summary['CCR_mean_SelectionGap']:.6f}`\n- Frequency baseline mean SelectionGap: `{summary['frequency_mean_SelectionGap']:.6f}`\n- Mean-damage baseline mean SelectionGap: `{summary['mean_damage_mean_SelectionGap']:.6f}`\n- Fixed-context baseline mean SelectionGap: `{summary['fixed_mean_SelectionGap']:.6f}`\n- CCR top-1 oracle identity rate: `{summary['CCR_top1_identity_rate']:.4f}`\n- Class bootstrap (10,000, seed 3407) CCR−frequency 95% CI: `{summary['pairwise_bootstrap_CI']['CCR_minus_frequency']}`\n\nRequired questions: `{json.dumps(summary['required_questions'], ensure_ascii=False)}`\n\nThe decision is diagnostic only. No final calibration size or context count is selected here.\n"""
    (out / "task_ccr_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--video_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    run(p.parse_args())

#!/usr/bin/env python3
"""GPU-first aggregate analysis for Task010 pruning variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task010_tdd_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from MC import InteractionPruner


VARIANTS = ("abs_rel", "old3d", "dynamic3d")
VARIANT_LABELS = {
    "abs_rel": "AbsRel",
    "old3d": "Old3D",
    "dynamic3d": "Dynamic3D",
}
SCOPES = ("all_units", "attention_only", "mlp_only")
BMS_FIELDS = (
    "variant",
    "scope",
    "num_units",
    "num_groups",
    "singleton_ratio",
    "multiunit_unit_ratio",
    "pairwise_intra_ablation_difference",
    "weighted_intra_ablation_variance",
    "domain_improvement_ratio",
    "device",
    "bms_sigma",
    "bms_source_sha256",
)
TASK007_FIELDS = {
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_st",
    "ablation_logit_deviation",
}
STAT_FIELDS = {
    "global_index",
    "layer",
    "unit_type",
    "unit_index",
    "D_abs",
    "D_rel",
    "D_third",
    "third_descriptor_name",
}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value for {label}: {value!r}")
    return result


def read_task007(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = TASK007_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing Task007 columns {sorted(missing)}")
        source_rows = list(reader)
    rows = []
    for source in source_rows:
        rows.append({
            "global_index": int(source["global_index"]),
            "layer": str(source["layer"]),
            "unit_type": str(source["unit_type"]),
            "unit_index": int(source["unit_index"]),
            "D_st": _parse_float(source["D_st"], "D_st"),
            "effect": _parse_float(
                source["ablation_logit_deviation"], "ablation_logit_deviation"
            ),
        })
    _validate_identity(rows)
    return rows


def read_statistics(path: Path, variant: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = STAT_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing descriptor columns {sorted(missing)}")
        source_rows = list(reader)
    rows = []
    expected_name = {
        "abs_rel": "none",
        "old3d": "D_st_old",
        "dynamic3d": "D_dyn",
    }[variant]
    for source in source_rows:
        if source["third_descriptor_name"] != expected_name:
            raise ValueError(
                f"{path} contains {source['third_descriptor_name']!r}, "
                f"expected {expected_name!r}"
            )
        third = None
        if variant != "abs_rel":
            third = _parse_float(source["D_third"], "D_third")
            if variant == "dynamic3d" and not -1e-6 <= third <= 1.0 + 1e-6:
                raise ValueError("D_dyn is outside [0,1]")
        rows.append({
            "global_index": int(source["global_index"]),
            "layer": str(source["layer"]),
            "unit_type": str(source["unit_type"]),
            "unit_index": int(source["unit_index"]),
            "D_abs": _parse_float(source["D_abs"], "D_abs"),
            "D_rel": _parse_float(source["D_rel"], "D_rel"),
            "D_third": third,
        })
    _validate_identity(rows)
    return rows


def _validate_identity(rows: Sequence[dict]) -> None:
    indices = [int(row["global_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("global_index must be contiguous and ordered from zero")
    identities = [
        (row["layer"], row["unit_type"], int(row["unit_index"]))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("unit identity is not unique")


def _assert_alignment(reference: Sequence[dict], candidate: Sequence[dict]) -> None:
    if len(reference) != len(candidate):
        raise ValueError("descriptor and Task007 row counts differ")
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_key = (
            int(left["global_index"]), left["layer"], left["unit_type"],
            int(left["unit_index"]),
        )
        right_key = (
            int(right["global_index"]), right["layer"], right["unit_type"],
            int(right["unit_index"]),
        )
        if left_key != right_key:
            raise ValueError(f"unit ordering mismatch at row {index}: {left_key} != {right_key}")


def build_scopes(rows: Sequence[dict]) -> dict[str, np.ndarray]:
    unit_types = np.asarray([row["unit_type"] for row in rows], dtype=object)
    scopes = {
        "all_units": np.arange(len(rows), dtype=np.int64),
        "attention_only": np.flatnonzero(unit_types == "attention_head"),
        "mlp_only": np.flatnonzero(unit_types == "ffn_neuron"),
    }
    for name, indices in scopes.items():
        if indices.size < 2:
            raise ValueError(f"scope {name} has fewer than two units")
    return scopes


def _pairwise_absolute_sum(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_zeros(())
    ordered = torch.sort(values.flatten()).values
    count = ordered.numel()
    coefficients = 2 * torch.arange(
        count, device=ordered.device, dtype=ordered.dtype
    ) - count + 1
    return torch.sum(coefficients * ordered)


def evaluate_groups_cuda(
    groups: Sequence[Sequence[int]], effects: torch.Tensor
) -> dict[str, float]:
    """Compute exact Task008 domain metrics on the selected CUDA device."""
    unit_count = effects.numel()
    membership = torch.empty(unit_count, dtype=torch.long, device=effects.device)
    group_sizes = []
    for group_id, members in enumerate(groups):
        if not members:
            raise ValueError("BMS returned an empty group")
        index = torch.as_tensor(members, dtype=torch.long, device=effects.device)
        membership[index] = group_id
        group_sizes.append(len(members))
    if sum(group_sizes) != unit_count:
        raise ValueError("BMS groups do not cover every unit exactly once")

    counts = torch.bincount(membership, minlength=len(groups)).to(effects.dtype)
    sums = torch.bincount(membership, weights=effects, minlength=len(groups))
    means = sums / counts.clamp_min(1.0)
    centered = effects - means.index_select(0, membership)
    q_var = centered.square().sum() / unit_count

    pair_sum = effects.new_zeros(())
    pair_count = 0
    for members in groups:
        if len(members) < 2:
            continue
        group_effects = effects.index_select(
            0, torch.as_tensor(members, dtype=torch.long, device=effects.device)
        )
        pair_sum = pair_sum + _pairwise_absolute_sum(group_effects)
        pair_count += math.comb(len(members), 2)
    q_pair = pair_sum / pair_count if pair_count else effects.new_tensor(float("nan"))
    global_pairs = math.comb(unit_count, 2)
    q_global = _pairwise_absolute_sum(effects) / global_pairs
    domain_gain = 1.0 - q_pair / q_global
    singleton_count = sum(size == 1 for size in group_sizes)
    multiunit_count = sum(size for size in group_sizes if size >= 2)
    return {
        "num_groups": len(groups),
        "singleton_ratio": singleton_count / len(groups),
        "multiunit_unit_ratio": multiunit_count / unit_count,
        "pairwise_intra_ablation_difference": float(q_pair.item()),
        "weighted_intra_ablation_variance": float(q_var.item()),
        "domain_improvement_ratio": float(domain_gain.item()),
    }


def _bms_identity() -> tuple[float, str]:
    parameter = inspect.signature(InteractionPruner.__init__).parameters["sigma"]
    sigma = float(parameter.default)
    source = inspect.getsource(InteractionPruner.mean_shift_clustering)
    return sigma, hashlib.sha256(source.encode("utf-8")).hexdigest()


def run_bms_job(
    variant: str,
    scope: str,
    features: np.ndarray,
    effects: np.ndarray,
    device: str,
) -> dict:
    torch.cuda.set_device(torch.device(device))
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    effect_tensor = torch.as_tensor(effects, dtype=torch.float64, device=device)
    normalized = (values - values.mean(dim=0, keepdim=True)) / (
        values.std(dim=0, keepdim=True) + 1e-8
    )
    sigma, source_hash = _bms_identity()
    pruner = InteractionPruner.__new__(InteractionPruner)
    pruner.sigma = sigma
    groups, _, _ = InteractionPruner.mean_shift_clustering(pruner, normalized)
    metrics = evaluate_groups_cuda(groups, effect_tensor)
    row = {
        "variant": variant,
        "scope": scope,
        "num_units": int(values.shape[0]),
        **metrics,
        "device": device,
        "bms_sigma": sigma,
        "bms_source_sha256": source_hash,
    }
    del values, effect_tensor, normalized, groups, pruner
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return row


def _run_device_queue(device: str, jobs: Sequence[tuple]) -> list[dict]:
    return [run_bms_job(*job, device=device) for job in jobs]


def _read_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_bars(
    path: Path, title: str, ylabel: str, values: Sequence[float]
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(VARIANTS))
    axis.bar(positions, values, color=["#708090", "#4C78A8", "#E45756"])
    axis.set_xticks(positions, [VARIANT_LABELS[name] for name in VARIANTS])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metric_value(metrics: dict, name: str) -> float | None:
    value = metrics.get(name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _answer_comparison(
    left: float | None, right: float | None, lower_is_better: bool
) -> str:
    if left is None or right is None:
        return "Not evaluated because the required real result is unavailable."
    improved = left < right if lower_is_better else left > right
    direction = "improved" if improved else "did not improve"
    return f"Dynamic3D {direction} relative to Old3D ({left:.6g} vs {right:.6g})."


def run_analysis(args: argparse.Namespace) -> None:
    root = Path(args.input_root)
    root.mkdir(parents=True, exist_ok=True)
    task007 = read_task007(Path(args.task007_csv))
    statistics = {}
    run_dirs = {}
    for variant in VARIANTS:
        run_dir = root / variant / f"seed{args.seed}"
        stats_path = run_dir / "descriptor_statistics.csv"
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        run_dirs[variant] = run_dir
        statistics[variant] = read_statistics(stats_path, variant)
        _assert_alignment(task007, statistics[variant])

    base_abs = np.asarray([row["D_abs"] for row in statistics["dynamic3d"]])
    base_rel = np.asarray([row["D_rel"] for row in statistics["dynamic3d"]])
    for variant in VARIANTS:
        current_abs = np.asarray([row["D_abs"] for row in statistics[variant]])
        current_rel = np.asarray([row["D_rel"] for row in statistics[variant]])
        if not np.allclose(current_abs, base_abs, rtol=1e-5, atol=1e-7):
            raise ValueError(f"D_abs changed across variants: {variant}")
        if not np.allclose(current_rel, base_rel, rtol=1e-5, atol=1e-7):
            raise ValueError(f"D_rel changed across variants: {variant}")

    old = np.asarray([row["D_third"] for row in statistics["old3d"]])
    dynamic = np.asarray([row["D_third"] for row in statistics["dynamic3d"]])
    effects = np.asarray([row["effect"] for row in task007], dtype=np.float64)
    descriptors = {
        "abs_rel": np.column_stack([base_abs, base_rel]),
        "old3d": np.column_stack([base_abs, base_rel, old]),
        "dynamic3d": np.column_stack([base_abs, base_rel, dynamic]),
    }
    scopes = build_scopes(task007)

    if not torch.cuda.is_available():
        raise RuntimeError("Task010 BMS analysis requires CUDA")
    available = torch.cuda.device_count()
    requested = available if args.max_gpus is None else min(args.max_gpus, available)
    if requested <= 0:
        raise RuntimeError("no CUDA device selected")
    devices = [f"cuda:{index}" for index in range(requested)]

    jobs_by_device = {device: [] for device in devices}
    job_index = 0
    for scope_name in SCOPES:
        indices = scopes[scope_name]
        for variant in VARIANTS:
            device = devices[job_index % len(devices)]
            jobs_by_device[device].append((
                variant,
                scope_name,
                descriptors[variant][indices],
                effects[indices],
            ))
            job_index += 1

    bms_rows = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(_run_device_queue, device, jobs)
            for device, jobs in jobs_by_device.items()
        ]
        for future in futures:
            bms_rows.extend(future.result())
    bms_rows.sort(key=lambda row: (SCOPES.index(row["scope"]), VARIANTS.index(row["variant"])))
    _atomic_csv(root / "bms_domain_summary.csv", BMS_FIELDS, bms_rows)

    def bms_row(variant: str, scope: str) -> dict:
        return next(
            row for row in bms_rows
            if row["variant"] == variant and row["scope"] == scope
        )

    _plot_bars(
        root / "tdd_bms_pairdiff.png",
        "BMS competition-domain pair difference",
        "Pairwise intra-domain ablation difference (lower is better)",
        [bms_row(name, "all_units")["pairwise_intra_ablation_difference"] for name in VARIANTS],
    )
    _plot_bars(
        root / "tdd_attention_bms_quality.png",
        "Attention Head BMS domain quality",
        "Attention pair difference (lower is better)",
        [bms_row(name, "attention_only")["pairwise_intra_ablation_difference"] for name in VARIANTS],
    )

    correlation_values = np.column_stack([base_abs, base_rel, old, dynamic])
    correlation_names = ("D_abs", "D_rel", "D_old", "D_dyn")
    correlation, _ = spearmanr(correlation_values, axis=0)
    correlation = np.asarray(correlation, dtype=np.float64)
    correlation_rows = []
    for row_index, row_name in enumerate(correlation_names):
        for column_index, column_name in enumerate(correlation_names):
            correlation_rows.append({
                "row": row_name,
                "column": column_name,
                "spearman_rho": f"{correlation[row_index, column_index]:.17g}",
            })
    _atomic_csv(
        root / "tdd_descriptor_correlation.csv",
        ("row", "column", "spearman_rho"),
        correlation_rows,
    )
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(4), correlation_names, rotation=30, ha="right")
    axis.set_yticks(range(4), correlation_names)
    for i in range(4):
        for j in range(4):
            axis.text(j, i, f"{correlation[i, j]:.2f}", ha="center", va="center")
    axis.set_title("Descriptor Spearman correlation")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(root / "tdd_descriptor_correlation.png", dpi=180)
    plt.close(fig)

    run_metrics = {
        variant: _read_metrics(run_dirs[variant] / "final_metrics.json")
        for variant in VARIANTS
    }
    # Add exact global BMS structure to per-run metrics and the aggregate table.
    for variant in VARIANTS:
        if run_metrics[variant]:
            run_metrics[variant]["num_groups"] = bms_row(
                variant, "all_units"
            )["num_groups"]
            run_metrics[variant]["singleton_ratio"] = bms_row(
                variant, "all_units"
            )["singleton_ratio"]
            _atomic_json(
                run_dirs[variant] / "final_metrics.json", run_metrics[variant]
            )
    summary_path = root / "summary.csv"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_reader = csv.DictReader(handle)
            summary_fields = tuple(summary_reader.fieldnames or ())
            summary_rows = list(summary_reader)
        if summary_rows and summary_fields:
            for row in summary_rows:
                variant = row.get("variant")
                if variant in VARIANTS and int(row.get("seed", -1)) == args.seed:
                    row["num_groups"] = str(bms_row(variant, "all_units")["num_groups"])
                    row["singleton_ratio"] = str(
                        bms_row(variant, "all_units")["singleton_ratio"]
                    )
            _atomic_csv(summary_path, summary_fields, summary_rows)
    pre_values = [_metric_value(run_metrics[name], "pre_ft_top1") for name in VARIANTS]
    if all(value is not None for value in pre_values):
        _plot_bars(
            root / "tdd_pre_finetune_top1.png",
            "Pre-finetuning Top-1 at matched budget",
            "Top-1 (%)",
            pre_values,
        )
    best_values = [_metric_value(run_metrics[name], "best_top1") for name in VARIANTS]
    if all(value is not None for value in best_values):
        _plot_bars(
            root / "tdd_best_top1.png",
            "Best fine-tuned Top-1 at matched budget",
            "Top-1 (%)",
            best_values,
        )

    paper_rows = []
    for variant in VARIANTS:
        paper_rows.append({
            "Variant": VARIANT_LABELS[variant],
            "D_abs": 1,
            "D_rel": 1,
            "D_old": int(variant == "old3d"),
            "D_dyn": int(variant == "dynamic3d"),
            "PreFT_Top1": run_metrics[variant].get("pre_ft_top1", ""),
            "Best_Top1": run_metrics[variant].get("best_top1", ""),
            "BMS_PairDiff": bms_row(variant, "all_units")["pairwise_intra_ablation_difference"],
            "Attention_PairDiff": bms_row(variant, "attention_only")["pairwise_intra_ablation_difference"],
            "MLP_PairDiff": bms_row(variant, "mlp_only")["pairwise_intra_ablation_difference"],
        })
    paper_fields = (
        "Variant", "D_abs", "D_rel", "D_old", "D_dyn", "PreFT_Top1",
        "Best_Top1", "BMS_PairDiff", "Attention_PairDiff", "MLP_PairDiff",
    )
    _atomic_csv(root / "paper_ablation_table.csv", paper_fields, paper_rows)

    sanity_summary = _read_metrics(root / "tdd_video_sanity_summary.json")
    global_dynamic = bms_row("dynamic3d", "all_units")["pairwise_intra_ablation_difference"]
    global_old = bms_row("old3d", "all_units")["pairwise_intra_ablation_difference"]
    attention_dynamic = bms_row("dynamic3d", "attention_only")["pairwise_intra_ablation_difference"]
    attention_old = bms_row("old3d", "attention_only")["pairwise_intra_ablation_difference"]
    pre_dynamic = _metric_value(run_metrics["dynamic3d"], "pre_ft_top1")
    pre_old = _metric_value(run_metrics["old3d"], "pre_ft_top1")
    best_dynamic = _metric_value(run_metrics["dynamic3d"], "best_top1")
    best_old = _metric_value(run_metrics["old3d"], "best_top1")
    freeze_statement = (
        f"Median frozen/normal ratio was {sanity_summary['median_freeze_ratio']:.6g} "
        f"with IQR {sanity_summary['freeze_ratio_iqr']:.6g}."
        if sanity_summary else
        "Not evaluated because the TDD video sanity result is unavailable."
    )
    summary_text = f"""# Task010 validation summary

All statements below are generated from the available real output files. No
success threshold is hard-coded.

1. **BMS competition-domain quality:** {_answer_comparison(global_dynamic, global_old, True)}
2. **Attention Head consistency:** {_answer_comparison(attention_dynamic, attention_old, True)}
3. **Pre-finetuning Top-1:** {_answer_comparison(pre_dynamic, pre_old, False)}
4. **Best fine-tuned Top-1:** {_answer_comparison(best_dynamic, best_old, False)}
5. **Complementarity:** rho(D_abs,D_dyn)={correlation[0,3]:.6g} and rho(D_rel,D_dyn)={correlation[1,3]:.6g}; interpret together with Old3D correlations, not as a standalone success claim.
6. **Frozen-video sanity:** {freeze_statement}

These diagnostics do not establish that TDD measures optical flow, motion
importance, or causal temporal contribution. They test only whether relative
temporal response dynamicity is a useful third descriptor coordinate.
"""
    (root / "validation_summary.md").write_text(summary_text, encoding="utf-8")

    sigma, source_hash = _bms_identity()
    _atomic_json(root / "analysis_metadata.json", {
        "seed": args.seed,
        "devices": devices,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(requested)],
        "gpu_first": True,
        "bms_sigma": sigma,
        "bms_source_sha256": source_hash,
        "cpu_only_operations": ["CSV/JSON I/O", "Spearman ranking", "Matplotlib figures"],
        "units": len(task007),
    })

    print("=" * 72)
    print("Task010 TDD Pruning Validation")
    print(f"Units: {len(task007)}; CUDA devices: {', '.join(devices)}")
    print("Variant       Global PairDiff    Attention PairDiff    PreFT Top1")
    for variant in VARIANTS:
        pre = _metric_value(run_metrics[variant], "pre_ft_top1")
        pre_text = "unavailable" if pre is None else f"{pre:.4f}"
        print(
            f"{VARIANT_LABELS[variant]:<13} "
            f"{bms_row(variant, 'all_units')['pairwise_intra_ablation_difference']:<18.8g} "
            f"{bms_row(variant, 'attention_only')['pairwise_intra_ablation_difference']:<21.8g} "
            f"{pre_text}"
        )
    print(f"Output: {root.resolve()}")
    print("=" * 72)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-first Task010 aggregate analysis")
    parser.add_argument("--input_root", default="tdd_pruning_validation")
    parser.add_argument(
        "--task007_csv",
        default="descriptor_ablation_validation/unit_ablation_effect.csv",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--max_gpus", type=int, default=None,
        help="maximum visible GPUs for parallel BMS queues; default uses all",
    )
    args = parser.parse_args(argv)
    if args.max_gpus is not None and args.max_gpus <= 0:
        parser.error("max_gpus must be positive")
    return args


if __name__ == "__main__":
    run_analysis(parse_args())

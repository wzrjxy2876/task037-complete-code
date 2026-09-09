#!/usr/bin/env python3
"""Pure helpers for Task012 pruning-severity diagnostics.

This module intentionally has no Torch dependency.  It is shared by the
pruner, the offline analysis, the resume-aware shell scripts, and unit tests.
It does not define a pruning score or alter the existing parameter budget.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union


ALLOWED_DIAGNOSTIC_MIN_HEADS = (0, 2, 3)


def parse_sparsity_list(value: Union[str, Iterable[float]]) -> list[float]:
    """Parse an ordered, duplicate-free list of sparsities in ``(0, 1)``."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("sparsity list is empty")
        values = [float(token) for token in tokens]
    else:
        values = [float(item) for item in value]
        if not values:
            raise ValueError("sparsity list is empty")
    if any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in values):
        raise ValueError("every sparsity must be finite and lie in (0, 1)")
    if len(set(values)) != len(values):
        raise ValueError("sparsity list contains duplicates")
    if values != sorted(values):
        raise ValueError("sparsity list must be strictly increasing")
    return values


def minimum_keep_units(
    original_units: int,
    min_keep_ratio: float,
    unit_type: str,
    diagnostic_min_attention_heads: int = 0,
) -> int:
    """Return the existing minimum, optionally protected for Attention only.

    ``diagnostic_min_attention_heads=0`` is byte-for-byte equivalent to the
    Task011 rule: ``max(1, int(original_units * min_keep_ratio))``.
    """
    original_units = int(original_units)
    diagnostic = int(diagnostic_min_attention_heads)
    if original_units <= 0:
        raise ValueError("original_units must be positive")
    if not 0.0 < float(min_keep_ratio) <= 1.0:
        raise ValueError("min_keep_ratio must lie in (0, 1]")
    if diagnostic not in ALLOWED_DIAGNOSTIC_MIN_HEADS:
        raise ValueError(
            "diagnostic_min_attention_heads must be one of "
            f"{ALLOWED_DIAGNOSTIC_MIN_HEADS}"
        )
    baseline = max(1, int(original_units * float(min_keep_ratio)))
    if unit_type in {"head", "attention_head"} and diagnostic:
        return min(original_units, max(baseline, diagnostic))
    return baseline


def layer_at_min_keep(remaining_units: int, min_keep_units: int) -> bool:
    """Identify the existing minimum-retention boundary exactly."""
    return int(remaining_units) == int(min_keep_units)


def parameter_accounting(
    parameters_before: int,
    parameters_after: int,
    estimated_removed_parameters: int,
) -> dict[str, float | int]:
    """Keep physical tensor reduction separate from estimated pruning cost."""
    before = int(parameters_before)
    after = int(parameters_after)
    estimated = int(estimated_removed_parameters)
    if before <= 0:
        raise ValueError("parameters_before must be positive")
    if after < 0 or after > before:
        raise ValueError("parameters_after must lie in [0, parameters_before]")
    if estimated < 0:
        raise ValueError("estimated_removed_parameters must be non-negative")
    return {
        "parameters_before": before,
        "parameters_after": after,
        "estimated_removed_parameters": estimated,
        "physical_numel_sparsity": (before - after) / before,
        "estimated_budget_sparsity": estimated / before,
    }


def metadata_matches(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    float_keys: Sequence[str] = ("target_sparsity", "sigma", "min_keep_ratio"),
) -> bool:
    """Return true only when every requested resume key matches."""
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if key in float_keys:
            try:
                if not math.isclose(
                    float(actual_value), float(expected_value), rel_tol=0.0, abs_tol=1e-12
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(actual_value) != str(expected_value):
            return False
    return True


def consecutive_accuracy_drops(
    points: Sequence[tuple[float, float]],
) -> list[dict[str, float]]:
    """Compute positive Top-1 loss for each consecutive pair."""
    ordered = sorted((float(sparsity), float(accuracy)) for sparsity, accuracy in points)
    if len(ordered) < 2:
        return []
    if len({sparsity for sparsity, _ in ordered}) != len(ordered):
        raise ValueError("duplicate sparsity in accuracy points")
    return [
        {
            "from_sparsity": left[0],
            "to_sparsity": right[0],
            "top1_before": left[1],
            "top1_after": right[1],
            "incremental_top1_drop": left[1] - right[1],
        }
        for left, right in zip(ordered, ordered[1:])
    ]


def largest_incremental_drop(
    points: Sequence[tuple[float, float]],
) -> Optional[dict[str, float]]:
    """Return the largest observed consecutive loss without a hard threshold."""
    rows = consecutive_accuracy_drops(points)
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["incremental_top1_drop"],
            -row["to_sparsity"],
        ),
    )


def _resume_match(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "final_metrics.json"
    if not metadata_path.is_file() or not metrics_path.is_file():
        print("run is incomplete")
        return 1
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"run metadata is unreadable: {exc}")
        return 1
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    expected = {
        "git_commit": (
            git_result.stdout.strip() if git_result.returncode == 0 else "unavailable"
        ),
        "variant": args.variant,
        "target_sparsity": args.sparsity,
        "sigma": args.sigma,
        "min_keep_ratio": args.min_keep_ratio,
        "diagnostic_min_attention_heads": args.diagnostic_min_attention_heads,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "calibration_batch_size": args.calib_batch_size,
        "calibration_batches": args.calib_batches,
        "validation_batch_size": args.batch_size,
    }
    if not metadata_matches(metadata, expected):
        print("saved run configuration does not match")
        return 1
    if checkpoint.is_file():
        stat = checkpoint.stat()
        if int(metadata.get("checkpoint_size_bytes", -1)) != int(stat.st_size):
            print("checkpoint size changed")
            return 1
        if int(metadata.get("checkpoint_mtime_ns", -1)) != int(stat.st_mtime_ns):
            print("checkpoint timestamp changed")
            return 1
    if metrics.get("status") != "task012_prune_only_complete":
        print("saved final metrics are not complete")
        return 1
    print(f"matching completed run: {run_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resume = subparsers.add_parser("resume-match")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--variant", choices=("old3d", "dynamic3d"), required=True)
    resume.add_argument("--sparsity", type=float, required=True)
    resume.add_argument("--sigma", type=float, required=True)
    resume.add_argument("--min-keep-ratio", type=float, required=True)
    resume.add_argument(
        "--diagnostic-min-attention-heads",
        type=int,
        choices=ALLOWED_DIAGNOSTIC_MIN_HEADS,
        required=True,
    )
    resume.add_argument("--seed", type=int, required=True)
    resume.add_argument("--calib-batch-size", type=int, required=True)
    resume.add_argument("--calib-batches", type=int, required=True)
    resume.add_argument("--batch-size", type=int, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "resume-match":
        return _resume_match(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

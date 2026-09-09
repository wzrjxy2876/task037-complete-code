"""Resume validation for Task013 cost-mode experiments."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from cost_decoupled_selection import SELECTION_COST_MODES


def metadata_matches(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    float_keys: Sequence[str] = ("target_sparsity", "sigma", "min_keep_ratio"),
) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if key in float_keys:
            try:
                if not math.isclose(
                    float(actual[key]), float(expected_value),
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(actual[key]) != str(expected_value):
            return False
    return True


def resume_match(args: argparse.Namespace) -> int:
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
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    expected = {
        "git_commit": (
            git_result.stdout.strip() if git_result.returncode == 0 else "unavailable"
        ),
        "checkpoint": str(checkpoint),
        "descriptor_variant": args.variant,
        "selection_cost_mode": args.selection_cost_mode,
        "target_sparsity": args.sparsity,
        "sigma": args.sigma,
        "min_keep_ratio": args.min_keep_ratio,
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
    if metrics.get("status") != "task013_prune_only_complete":
        print("saved final metrics are not complete")
        return 1
    print(f"matching completed run: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resume = subparsers.add_parser("resume-match")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--variant", choices=("dynamic3d", "old3d"), required=True)
    resume.add_argument(
        "--selection-cost-mode", choices=SELECTION_COST_MODES, required=True
    )
    resume.add_argument("--sparsity", type=float, required=True)
    resume.add_argument("--sigma", type=float, required=True)
    resume.add_argument("--min-keep-ratio", type=float, required=True)
    resume.add_argument("--seed", type=int, required=True)
    resume.add_argument("--calib-batch-size", type=int, required=True)
    resume.add_argument("--calib-batches", type=int, required=True)
    resume.add_argument("--batch-size", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "resume-match":
        return resume_match(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

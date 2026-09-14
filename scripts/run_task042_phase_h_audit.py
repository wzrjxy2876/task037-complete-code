#!/usr/bin/env python3
"""Run the offline Task042 Phase-H continuous-coverage audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
_REPO_SRC = HERE.parents[1] / "src"
_LOCAL_SRC = HERE.parent / "src"
SOURCE_ROOT = _REPO_SRC if (_REPO_SRC / "lgfr_runtime").is_dir() else _LOCAL_SRC
sys.path.insert(0, str(SOURCE_ROOT))

from lgfr_runtime.task042_phase_h_continuous_coverage import run_phase_h  # noqa: E402


def current_head() -> str:
    try:
        root = HERE.parents[1] if (HERE.parents[1] / ".git").exists() else HERE.parents[2]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Task042 Phase-H audit; no inference/GPU/pruning.")
    parser.add_argument("--phase-g-dir", required=True, type=Path)
    parser.add_argument("--phase-f-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_phase_h(args.phase_g_dir, args.phase_f_dir, args.output_dir, current_head())
    print(f"decision={summary['decision']['label']}")
    print(f"subsets={summary['counts']['subset_coverage_rows']} leave_one_out={summary['counts']['leave_one_out_rows']}")
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the offline Task042 Phase-G functional-coverage audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgfr_runtime.task042_phase_g_functional_coverage import run_phase_g  # noqa: E402


def current_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Task042 Phase-G audit; no inference/GPU/performance oracle.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decision", choices=("A", "B", "C"), default="B")
    parser.add_argument(
        "--decision-reason",
        default="The evidence is unresolved or qualitatively ambiguous; under the preregistered rule, ambiguity is assigned to B.",
    )
    args = parser.parse_args()
    summary = run_phase_g(args.input_dir, args.output_dir, current_head(), args.decision, args.decision_reason)
    print(f"decision={summary['decision']['label']}")
    print(f"units={summary['counts']['units']} videos={summary['counts']['videos']} raw_rows={summary['counts']['raw_rows']}")
    print(f"multi_unit_domains={summary['counts']['multi_unit_domains']} atoms={summary['counts']['atoms']}")
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

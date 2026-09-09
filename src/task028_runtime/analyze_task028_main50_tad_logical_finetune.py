"""Offline analyzer for the Task028 main 50% experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import task028_main50_tad_logical_finetune as task028


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    task028.analyze(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Command-line entrypoint for the Task042 Phase-I functional-coverage pilot."""
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1] / "src" / "lgfr_runtime"
sys.path.insert(0, str(RUNTIME))
from task042_phase_i_coverage import main

if __name__ == "__main__":
    main()

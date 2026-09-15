#!/usr/bin/env python3
"""Repository entry point for the Task042 Phase-L diagnostic."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))

from task042_phase_l_interaction import main


if __name__ == "__main__":
    main()

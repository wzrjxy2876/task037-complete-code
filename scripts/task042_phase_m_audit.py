#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lgfr_runtime"))
from task042_phase_m_trajectory import main
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repository entry point for the integrated ARX critic/recovery delivery."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "integrations" / "cosmos-arx-rgb-cr"
sys.path.insert(0, str(DELIVERY))
sys.argv[0] = str(DELIVERY / "cr.py")
runpy.run_path(str(DELIVERY / "cr.py"), run_name="__main__")

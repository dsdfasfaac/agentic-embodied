#!/usr/bin/env python3
"""Continue an audited interrupted closing call through the frozen bundle suffix.

Source SHA and consumed budgets are verified before live motion. This command
retains its original CLI; orchestration lives in the deployment module.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.grasp_continuation_runner import main, run


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a frozen CandidateBundle against the ARX real gateway, with step evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.contracts import EXIT_CODES, RunnerLimits
from robots.arx.deployment.real_runner import RealBundleRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "python", "hardware-config", "hardware-sha256", "task",
                 "model-contract", "runtime-config", "runner-limits", "bundle",
                 "tool-catalog", "real-input-contract", "real-input-sha256",
                 "feature-provider", "feature-provider-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--kinematics-calibration")
    parser.add_argument("--grasp-config")
    parser.add_argument("--grasp-config-sha256")
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument("--listen-host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    args = vars(parser.parse_args())
    limits = RunnerLimits.model_validate_json(Path(args.pop("runner_limits")).read_text())
    args["catalog"] = args.pop("tool_catalog")
    result = RealBundleRunner(runner_limits=limits, **args).run()
    print(result.model_dump_json())
    raise SystemExit(EXIT_CODES[result.status])


if __name__ == "__main__":
    main()

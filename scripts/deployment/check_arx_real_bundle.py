#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Audit a frozen ARX CandidateBundle against a real-input contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.real_input import (
    LiveCapabilities,
    _json_object,
    preflight_real_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle",
        "task-manifest",
        "model-contract",
        "tool-catalog",
        "real-contract",
        "capabilities-snapshot",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()
    try:
        capabilities = LiveCapabilities.model_validate(
            _json_object(args.capabilities_snapshot)
        )
        report = preflight_real_bundle(
            bundle_path=args.bundle,
            task_manifest_path=args.task_manifest,
            model_contract_path=args.model_contract,
            tool_catalog_path=args.tool_catalog,
            real_contract_path=args.real_contract,
            live_capabilities=capabilities,
            expected_real_contract_sha256=args.expected_contract_sha256,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"eligible": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    report["audit_mode"] = "offline_capability_snapshot"
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

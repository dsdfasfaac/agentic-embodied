#!/usr/bin/env python3
"""Export the frozen ARX real-gateway tool catalog without opening hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.gateway.tools import default_registry


class _Execution:
    def prepare(self, args, context):
        raise RuntimeError("catalog-only handler")


class _Review:
    def inspect(self, args, context):
        raise RuntimeError("catalog-only handler")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-eef", action="store_true")
    args = parser.parse_args()
    catalog = default_registry(
        zeva=_Execution(), gripper=_Execution(),
        eef=_Execution() if args.with_eef else None, reentry=_Review(),
    ).describe()
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(catalog["catalog_sha256"])


if __name__ == "__main__":
    main()

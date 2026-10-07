#!/usr/bin/env python3
"""Record fork changes separately from differences with a current upstream.

Reads local Git objects only; never fetches, merges, edits code or opens hardware.
Fetch the desired upstream revision beforehand and pass an immutable local ref.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def tree(ref: str) -> dict[str, str]:
    result = {}
    for entry in git("ls-tree", "-r", ref).splitlines():
        metadata, path = entry.split("\t", 1)
        result[path] = metadata.split()[2]
    return result


def change(before: str | None, after: str | None) -> str:
    if before == after:
        return "unchanged"
    if before is None:
        return "added"
    if after is None:
        return "removed"
    return "modified"


def compare(local: str, upstream: str) -> dict:
    local_sha = git("rev-parse", f"{local}^{{commit}}")
    upstream_sha = git("rev-parse", f"{upstream}^{{commit}}")
    base_sha = git("merge-base", local_sha, upstream_sha)
    base, ours, theirs = tree(base_sha), tree(local_sha), tree(upstream_sha)
    files = []
    for path in sorted(base.keys() | ours.keys() | theirs.keys()):
        b, o, t = base.get(path), ours.get(path), theirs.get(path)
        if b == o == t:
            continue
        relation = (
            "same"
            if o == t
            else (
                "local_only"
                if t is None
                else "upstream_only" if o is None else "different"
            )
        )
        files.append(
            {
                "path": path,
                "local_change_since_base": change(b, o),
                "upstream_change_since_base": change(b, t),
                "snapshot_relation": relation,
                "base_blob": b,
                "local_blob": o,
                "upstream_blob": t,
            }
        )
    local_counts = Counter(row["local_change_since_base"] for row in files)
    return {
        "schema_version": "arx.zetta.source-comparison.v1",
        "local_commit": local_sha,
        "upstream_commit": upstream_sha,
        "common_base_commit": base_sha,
        "upstream_repository": "https://github.com/air-embodied-brain/Zetta-Embodiment",
        "interpretation": "Local changes are measured from the common base. "
        "Upstream-only paths are snapshot differences, not evidence "
        "that this fork deleted newer upstream integrations.",
        "local_change_counts": {
            k: v for k, v in sorted(local_counts.items()) if k != "unchanged"
        },
        "snapshot_counts": dict(
            sorted(Counter(row["snapshot_relation"] for row in files).items())
        ),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.local, args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "local_commit",
                    "upstream_commit",
                    "common_base_commit",
                    "local_change_counts",
                    "snapshot_counts",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

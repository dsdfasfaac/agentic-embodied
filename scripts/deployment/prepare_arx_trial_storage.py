#!/usr/bin/env python3
"""Put a new trial's small gateway journal on SSD; large artifacts stay on its data disk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(output: Path, fast_root: Path):
    output, fast_root = output.resolve(), fast_root.resolve()
    if output.exists():
        raise ValueError('trial output must be new; never replace an existing episode')
    identity = hashlib.sha256(str(output).encode()).hexdigest()[:20]
    fast = fast_root / (output.name + '-' + identity)
    if fast.exists():
        raise ValueError('fast journal directory already exists')
    fast.mkdir(parents=True)
    (output/'private').mkdir(parents=True)
    large = output/'sensor-artifacts'
    (large/'images').mkdir(parents=True)
    (large/'grasp-sensors').mkdir()
    (fast/'public').mkdir()
    (fast/'public/images').symlink_to(large/'images', target_is_directory=True)
    (fast/'grasp-sensors').symlink_to(large/'grasp-sensors', target_is_directory=True)
    (output/'private/gateway').symlink_to(fast, target_is_directory=True)
    result = {'schema_version': 'arx.real.storage_layout.v1', 'output': str(output),
              'gateway': str(fast), 'journal': str(fast/'journal.sqlite3'),
              'large_sensor_artifacts': str(large), 'durable_journal': 'SQLite FULL WAL',
              'note': 'Keep the small SSD journal directory; the trial gateway link resolves to it.'}
    (output/'private/storage-layout.json').write_text(json.dumps(result, indent=2)+'\n')
    (fast/'.prepared-storage.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def admit_gateway_output(output: Path):
    """Allow only an untouched prepared directory, never an existing episode."""
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        return
    marker = output/'.prepared-storage.json'
    try:
        layout = json.loads(marker.read_text())
        expected = {'.prepared-storage.json', 'public', 'grasp-sensors'}
        if (layout['schema_version'] != 'arx.real.storage_layout.v1'
                or Path(layout['gateway']).resolve() != output.resolve()
                or (Path(layout['output'])/'private/gateway').resolve() != output.resolve()
                or {p.name for p in output.iterdir()} != expected
                or {p.name for p in (output/'public').iterdir()} != {'images'}
                or any((output/'public/images').iterdir())
                or any((output/'grasp-sensors').iterdir())):
            raise ValueError('prepared storage is not empty or identity changed')
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError('gateway output already exists and is not unused prepared storage') from exc
    marker.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fast-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.fast_root)))


if __name__ == '__main__':
    main()

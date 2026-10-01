# Copyright (c) 2026 Zetta Contributors
"""Export public journal observations as videos without accessing simulator state."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .public import CAMERAS, atomic_write


def export_public_recording(journal, output: Path, *, fps=15):
    observations = [
        json.loads(row[0])
        for row in journal.db.execute(
            "SELECT payload FROM records WHERE kind='ObservationPublished' ORDER BY sequence"
        )
    ]
    atomic_write(
        output / "observations.jsonl",
        ("\n".join(json.dumps(o) for o in observations) + "\n").encode(),
    )
    privileged = [
        json.loads(row[0])
        for row in journal.db.execute(
            "SELECT payload FROM records WHERE kind='PrivilegedObservation' ORDER BY sequence"
        )
    ]
    if privileged:
        atomic_write(
            output / "privileged_observations.jsonl",
            ("\n".join(json.dumps(o, sort_keys=True) for o in privileged) + "\n").encode(),
        )
    writers = {}
    try:
        for name in (*CAMERAS, "three_view"):
            writers[name] = imageio.get_writer(
                output / (name + ".mp4"),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
            )
        for observation in observations:
            frames = [
                imageio.imread(
                    output
                    / "public"
                    / "images"
                    / (observation["cameras"][name]["content_id"] + ".png")
                )
                for name in CAMERAS
            ]
            for name, frame in zip(CAMERAS, frames):
                writers[name].append_data(frame)
            writers["three_view"].append_data(np.concatenate(frames, axis=1))
    finally:
        for writer in writers.values():
            writer.close()
    return len(observations)

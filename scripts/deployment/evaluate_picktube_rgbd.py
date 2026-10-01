#!/usr/bin/env python3
"""Read-only RGB localization check against dodo's raw PickTube episodes.

The saved *_depth JPEGs are colorized previews, not metric depth; this script
never treats their RGB pixels as millimetres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider


def evaluate(root: Path, *, frame_stride: int = 10) -> dict:
    episodes = sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    passed = []
    failed = []
    depth_preview_formats = {}
    tracked_frames = 0
    tracked_detected = 0
    for episode in episodes:
        frames = sorted((episode / "images/front_rgb").glob("*.jpg"))
        if not frames:
            failed.append({"episode": episode.name, "reason": "front RGB absent"})
            continue
        observer = PickTubeRgbdProvider()
        try:
            first_mask = None
            for index, frame_path in enumerate(frames):
                if index != 0 and index % frame_stride:
                    continue
                tracked_frames += 1
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
                try:
                    mask = observer._pink_component(rgb)
                    tracked_detected += 1
                    if index == 0:
                        first_mask = mask
                except ValueError:
                    if index == 0:
                        raise
            if first_mask is None:
                raise ValueError("initial front RGB decode failed")
            mask = first_mask
            yy, xx = np.nonzero(mask)
            passed.append({"episode": episode.name, "centre_px": [float(np.median(xx)),
                                                                   float(np.median(yy))],
                           "label_pixels": int(mask.sum())})
        except ValueError as exc:
            failed.append({"episode": episode.name, "reason": str(exc)})
        depth_file = episode / "images/front_depth" / frames[0].name
        if depth_file.exists():
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            key = f"{depth.dtype}:{depth.shape}" if depth is not None else "decode_failed"
            depth_preview_formats[key] = depth_preview_formats.get(key, 0) + 1
    return {
        "dataset": str(root), "episodes": len(episodes),
        "initial_front_rgb_detected": len(passed), "failures": failed,
        "tracked_frame_stride": frame_stride, "tracked_frames": tracked_frames,
        "tracked_detected": tracked_detected,
        "detections": passed, "saved_depth_formats": depth_preview_formats,
        "metric_depth_validated": False,
        "note": "Offline JPEG depth is colorized; live aligned z16 D405 frames are required for metre distances.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("frame stride must be positive")
    report = evaluate(args.dataset, frame_stride=args.frame_stride)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

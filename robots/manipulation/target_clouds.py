"""Expand an identified upright tube using only measured RGB-D surface points."""

from __future__ import annotations

import numpy as np


def upright_tube_mask(
    rgb,
    points_base,
    seed_base,
    seed_mask,
    *,
    radius_m=0.012,
    below_m=0.008,
    above_m=0.09,
    allow_disconnected=False,
):
    """Select the seed-connected metric column, excluding yellow rack pixels.

    points_base is HxWx3; invalid depth positions are NaN. Missing transparent
    surfaces are never filled. The physical upright prior is explicit.
    allow_disconnected admits other MEASURED surfaces in the same narrow metric
    column after pink-seed admission; depth holes are not filled or fabricated.
    """
    import cv2

    points = np.asarray(points_base)
    if points.shape != (*rgb.shape[:2], 3) or seed_mask.shape != rgb.shape[:2]:
        raise ValueError("tube cloud must align RGB, depth and seed mask")
    delta = points - seed_base
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rack = (
        (hsv[..., 0] >= 18)
        & (hsv[..., 0] <= 45)
        & (hsv[..., 1] >= 80)
        & (hsv[..., 2] >= 90)
    )
    roi = (
        np.isfinite(points).all(axis=-1)
        & (np.linalg.norm(delta[..., :2], axis=-1) <= radius_m)
        & (delta[..., 2] >= -below_m)
        & (delta[..., 2] <= above_m)
        & ~rack
    )
    _, labels = cv2.connectedComponents(roi.astype(np.uint8), 8)
    ids, counts = np.unique(labels[seed_mask & roi], return_counts=True)
    ids, counts = ids[ids != 0], counts[ids != 0]
    if not len(ids):
        raise ValueError("tube depth column is disconnected from pink identity seed")
    mask = roi if allow_disconnected else labels == ids[np.argmax(counts)]
    selected = points[mask]
    if len(selected) < 32 or np.ptp(selected[:, 2]) < 0.02:
        raise ValueError("insufficient measured upright tube surface extent")
    return mask

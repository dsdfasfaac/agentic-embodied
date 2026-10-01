"""RGB-only runtime critic for C2 single-finger / lost-grasp failures.

The critic intentionally has no simulator, contact, object-pose, joint-state, or
action inputs.  It observes the three uint8 RGB streams and emits evidence plus
proposals.  It never selects or executes recovery actions.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np


CAMERAS = ("front_rgb", "left_rgb", "right_rgb")


@dataclasses.dataclass(frozen=True, slots=True)
class ViewEvidence:
    pink_pixels: int
    pink_area_ratio: float
    target_visible: bool
    target_xy: tuple[float, float] | None
    target_rack_delta: tuple[float, float] | None
    baseline_delta_error: float | None
    dark_dynamic_near_target: float | None
    global_rgb_change: float


@dataclasses.dataclass(frozen=True, slots=True)
class CriticProposal:
    proposal_id: str
    kind: str
    evidence: dict[str, Any]
    candidate_actions: tuple[str, ...]
    expected_success_signal: str
    escalation_condition: str
    nominal_vla_reentry_state: str


def _pink_mask(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.int16, copy=False)
    r, g, b = (rgb[..., index] for index in range(3))
    return (r > 110) & (r > 1.35 * g) & (r > 1.08 * b) & ((r - g) > 35)


def _yellow_mask(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.int16, copy=False)
    r, g, b = (rgb[..., index] for index in range(3))
    return (r > 130) & (g > 90) & (r > 0.90 * g) & (r > 1.60 * b) & (g > 1.50 * b)


def _centroid(mask: np.ndarray) -> tuple[int, tuple[float, float] | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, None
    height, width = mask.shape
    return len(xs), (float(xs.mean() / width), float(ys.mean() / height))


class RgbC2Critic:
    """Detect a visibly attempted grasp that releases the target unsupported.

    The detector uses the pink target band, the yellow rack, and RGB background
    change around the target.  Numeric values are normalized by image size or
    reset-frame target area.  MuJoCo state and contacts are never available.
    """

    def __init__(
        self,
        reset_images: Mapping[str, np.ndarray],
        *,
        approach_area_ratio: float = 0.48,
        return_area_ratio: float = 0.62,
        fixture_delta_tolerance: float = 0.065,
        displaced_delta_threshold: float = 0.11,
        motion_tolerance: float = 0.018,
        reentry_dynamic_threshold: float = 0.32,
        persistence: int = 12,
        feature_cameras: tuple[str, ...] = CAMERAS,
    ) -> None:
        # The phase critic uses this class only for RGB features. Legacy
        # observe()/readiness() still require their default three-view input.
        self.feature_cameras = tuple(feature_cameras)
        if (not self.feature_cameras or len(set(self.feature_cameras)) != len(self.feature_cameras)
                or not set(self.feature_cameras) <= set(CAMERAS)):
            raise ValueError("feature_cameras must be a nonempty unique subset of CAMERAS")
        self._validate_camera_keys(reset_images)
        self.reset_images = {
            name: self._validate_image(reset_images[name]).copy() for name in self.feature_cameras
        }
        self.approach_area_ratio = float(approach_area_ratio)
        self.return_area_ratio = float(return_area_ratio)
        self.fixture_delta_tolerance = float(fixture_delta_tolerance)
        self.displaced_delta_threshold = float(displaced_delta_threshold)
        self.motion_tolerance = float(motion_tolerance)
        self.reentry_dynamic_threshold = float(reentry_dynamic_threshold)
        self.persistence = int(persistence)
        if not 0 < self.approach_area_ratio < self.return_area_ratio < 1:
            raise ValueError("area ratios must satisfy 0 < approach < return < 1")
        if self.persistence < 2:
            raise ValueError("persistence must be at least two frames")
        self._baseline = {
            name: self._raw_view(self.reset_images[name], self.reset_images[name])
            for name in self.feature_cameras
        }
        missing_band = [name for name, item in self._baseline.items() if item["pink_pixels"] < 20]
        if missing_band:
            raise ValueError("pink target band must be visible in every required reset image "
                             f"(at least 20 pixels); insufficient: {', '.join(missing_band)}")
        self._history: deque[dict[str, ViewEvidence]] = deque(maxlen=max(8, persistence + 2))
        self._approach_evidence_count = 0
        self._approach_seen = False
        self._episode = 0
        self._proposal_serial = 0
        self._last_proposal_kind: str | None = None

    def _validate_camera_keys(self, images: Mapping[str, np.ndarray]) -> None:
        if not set(self.feature_cameras) <= set(images) or set(images) - set(CAMERAS):
            raise ValueError(f"RGB images must contain {self.feature_cameras}; "
                             "only known extra cameras are allowed and ignored")

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        result = np.asarray(image)
        if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
            raise ValueError(f"RGB image must be uint8 HxWx3, got {result.dtype} {result.shape}")
        return result

    def _raw_view(self, image: np.ndarray, baseline_image: np.ndarray) -> dict[str, Any]:
        height, width = image.shape[:2]
        pink = _pink_mask(image)
        yellow = _yellow_mask(image)
        pink_pixels, pink_xy = _centroid(pink)
        _, yellow_xy = _centroid(yellow)
        delta = None
        if pink_xy is not None and yellow_xy is not None:
            delta = (pink_xy[0] - yellow_xy[0], pink_xy[1] - yellow_xy[1])
        dynamic_near = None
        rgb = image.astype(np.int16, copy=False)
        base = baseline_image.astype(np.int16, copy=False)
        changed = np.max(np.abs(rgb - base), axis=2) > 22
        if pink_xy is not None:
            dark = rgb.mean(axis=2) < 110
            cx, cy = int(pink_xy[0] * width), int(pink_xy[1] * height)
            half_w, half_h = max(10, width // 7), max(8, height // 10)
            x0, x1 = max(0, cx - half_w), min(width, cx + half_w + 1)
            y0, y1 = max(0, cy - half_h), min(height, cy + half_h + 1)
            region = changed[y0:y1, x0:x1] & dark[y0:y1, x0:x1] & ~pink[y0:y1, x0:x1]
            dynamic_near = float(region.mean()) if region.size else None
        return {
            "pink_pixels": int(pink_pixels),
            "target_xy": pink_xy,
            "target_rack_delta": delta,
            "dark_dynamic_near_target": dynamic_near,
            "global_rgb_change": float(changed.mean()),
        }

    def _features(self, images: Mapping[str, np.ndarray]) -> dict[str, ViewEvidence]:
        self._validate_camera_keys(images)
        result: dict[str, ViewEvidence] = {}
        for name in self.feature_cameras:
            image = self._validate_image(images[name])
            if image.shape != self.reset_images[name].shape:
                raise ValueError(f"{name} shape changed after reset")
            raw = self._raw_view(image, self.reset_images[name])
            baseline = self._baseline[name]
            area_ratio = raw["pink_pixels"] / baseline["pink_pixels"]
            delta_error = None
            if raw["target_rack_delta"] is not None and baseline["target_rack_delta"] is not None:
                delta_error = float(np.linalg.norm(
                    np.asarray(raw["target_rack_delta"]) - np.asarray(baseline["target_rack_delta"])
                ))
            result[name] = ViewEvidence(
                pink_pixels=raw["pink_pixels"],
                pink_area_ratio=float(area_ratio),
                target_visible=raw["pink_pixels"] >= 20,
                target_xy=raw["target_xy"],
                target_rack_delta=raw["target_rack_delta"],
                baseline_delta_error=delta_error,
                dark_dynamic_near_target=raw["dark_dynamic_near_target"],
                global_rgb_change=raw["global_rgb_change"],
            )
        return result

    @staticmethod
    def _json_features(features: Mapping[str, ViewEvidence]) -> dict[str, Any]:
        return {name: dataclasses.asdict(item) for name, item in features.items()}

    def _target_motion(self) -> float | None:
        if len(self._history) < self.persistence:
            return None
        distances: list[float] = []
        recent = list(self._history)[-self.persistence:]
        for name in ("front_rgb", "left_rgb"):
            points = [item[name].target_rack_delta for item in recent]
            if any(point is None for point in points):
                continue
            array = np.asarray(points, dtype=np.float64)
            distances.append(float(np.max(np.linalg.norm(array - array[-1], axis=1))))
        return max(distances) if distances else None

    def _new_proposal(self, kind: str, evidence: dict[str, Any]) -> CriticProposal | None:
        if kind == self._last_proposal_kind:
            return None
        self._last_proposal_kind = kind
        self._proposal_serial += 1
        if kind == "grasp_lost_target_on_fixture":
            candidates = (
                "continue_vla_with_no_change",
                "pause_vla_reopen_retreat_restage_then_regenerate_plan",
                "terminate_if_visual_restage_preconditions_fail",
            )
            success = "RGB target is stable on fixture, gripper has cleared it, and re-entry proposal is emitted"
            escalation = "target disappears, remains in motion, or is displaced outside the validated fixture neighborhood"
        elif kind == "target_displaced_after_failed_grasp":
            candidates = (
                "pause_vla_and_inspect_multiview_rgb",
                "visual_restage_if_target_is_reachable",
                "terminate_or_reset_scene_if_no_safe_rgb-only_restage_exists",
            )
            success = "RGB target returns to a stable reachable fixture pose"
            escalation = "target is missing, unstable, or outside the RGB-validated operating region"
        else:
            candidates = ("resume_vla", "continue_visual_verification", "terminate")
            success = "nominal VLA makes progress without immediate recurrence"
            escalation = "the same lost-grasp proposal recurs immediately after VLA resume"
        return CriticProposal(
            proposal_id=f"rgb-c2-{self._episode:02d}-{self._proposal_serial:04d}",
            kind=kind,
            evidence=evidence,
            candidate_actions=candidates,
            expected_success_signal=success,
            escalation_condition=escalation,
            nominal_vla_reentry_state=(
                "target visible and stable on fixture; gripper clear; task phase known; "
                "fresh VLA approach can start from a reset-like visual state"
            ),
        )

    def observe(self, images: Mapping[str, np.ndarray]) -> CriticProposal | None:
        """Observe one RGB triplet and optionally emit a consolidated proposal."""
        features = self._features(images)
        self._history.append(features)
        front, left, right = (features[name] for name in CAMERAS)
        approach_now = (
            front.pink_area_ratio < self.approach_area_ratio
            or left.pink_area_ratio < self.approach_area_ratio
            or right.pink_area_ratio > 1.45
        )
        self._approach_evidence_count = self._approach_evidence_count + 1 if approach_now else 0
        self._approach_seen |= self._approach_evidence_count >= self.persistence
        if not self._approach_seen:
            return None
        visible_pair = front.target_visible and left.target_visible
        returned = visible_pair and all(
            item.pink_area_ratio >= self.return_area_ratio
            and item.baseline_delta_error is not None
            and item.baseline_delta_error <= self.fixture_delta_tolerance
            for item in (front, left)
        )
        displaced = visible_pair and sum(
            item.baseline_delta_error is not None
            and item.baseline_delta_error >= self.displaced_delta_threshold
            for item in (front, left, right)
        ) >= 2
        motion = self._target_motion()
        stable = motion is not None and motion <= self.motion_tolerance
        common = {
            "rgb_only": True,
            "approach_seen": self._approach_seen,
            "target_motion_normalized": motion,
            "views": self._json_features(features),
            "sim_contact_used": False,
            "sim_object_pose_used": False,
        }
        if returned and stable:
            return self._new_proposal("grasp_lost_target_on_fixture", common)
        if displaced and stable:
            return self._new_proposal("target_displaced_after_failed_grasp", common)
        return None

    def begin_recovery(self) -> None:
        """Start a new RGB verification episode without changing the baseline."""
        self._episode += 1
        self._history.clear()
        self._approach_evidence_count = 0
        self._approach_seen = False
        self._last_proposal_kind = None

    def observe_reentry(self, images: Mapping[str, np.ndarray]) -> CriticProposal | None:
        """Propose VLA resume when RGB post-conditions are stable and reset-like."""
        features = self._features(images)
        self._history.append(features)
        motion = self._target_motion()
        front, left = features["front_rgb"], features["left_rgb"]
        reset_like = all(
            item.target_visible
            and item.pink_area_ratio >= self.return_area_ratio
            and item.baseline_delta_error is not None
            and item.baseline_delta_error <= self.fixture_delta_tolerance
            and item.dark_dynamic_near_target is not None
            and item.dark_dynamic_near_target <= self.reentry_dynamic_threshold
            for item in (front, left)
        )
        reset_like = reset_like and all(
            features[name].global_rgb_change <= threshold
            for name, threshold in {
                "front_rgb": 0.08,
                "left_rgb": 0.08,
                "right_rgb": 0.12,
            }.items()
        )
        stable = motion is not None and motion <= self.motion_tolerance
        if not (reset_like and stable):
            return None
        return self._new_proposal(
            "resume_vla_ready",
            {
                "rgb_only": True,
                "target_motion_normalized": motion,
                "reset_like": reset_like,
                "views": self._json_features(features),
                "sim_contact_used": False,
                "sim_object_pose_used": False,
            },
        )

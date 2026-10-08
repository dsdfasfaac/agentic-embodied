"""Bounded grasp proposal, review and execution contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .contracts import ID, StrictModel


class ProposeGraspArgs(StrictModel):
    engine: Literal["tube_geometry", "contact_graspnet", "graspgen"] = "tube_geometry"
    max_candidates: int = Field(default=8, ge=1, le=32)


class ReviewGraspArgs(StrictModel):
    proposal_id: ID
    phase: Literal["pregrasp", "engage", "lift"] = "pregrasp"
    max_steps: int = Field(default=180, ge=20, le=240)


class ExecuteGraspArgs(StrictModel):
    review_token: ID
    phase: Literal["pregrasp", "engage", "lift"] = "pregrasp"
    max_steps: int = Field(default=180, ge=20, le=240)


class GraspProposalOutput(StrictModel):
    proposal_id: ID
    observation_id: ID
    target_id: str
    engine: str
    proposal_only: Literal[True] = True
    environment_advanced: Literal[False] = False
    candidates: list[dict] = Field(max_length=32)
    evidence: dict


class GraspReviewOutput(StrictModel):
    proposal_id: ID
    observation_id: ID
    phase: str
    eligible: bool
    review_token: ID | None
    checks: list[dict]
    certificate_level: Literal["sensor_tcp_clearance_and_joint_ik"]
    limitations: list[str]


class GraspRecoveryConfig(StrictModel):
    schema_version: Literal["arx.real.grasp.v1"] = "arx.real.grasp.v1"
    # Direct Python model adapters run locally; optional HTTP adapters are loopback only.
    contact_graspnet_endpoint: str | None = None
    graspgen_endpoint: str | None = None
    learned_grasp_to_tcp: list[list[float]] | None = None
    learned_gripper_id: str | None = None
    learned_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    learned_gripper_transfer_verified: bool = False
    # Explicit commissioning scope: permits reviewed open pregrasp only.
    # It never certifies engagement, closing, lifting or an evolved candidate.
    learned_pregrasp_commissioning: bool = False
    learned_target_distance_max_m: float = Field(default=0.025, gt=0, le=0.04)
    learned_parallel_jaw_half_turn: bool = False
    motion_enabled: bool = False
    pregrasp_distance_m: float = Field(default=0.03, ge=0.02, le=0.08)
    lift_distance_m: float = Field(default=0.02, ge=0.01, le=0.03)
    speed_m_s: float = Field(default=0.02, gt=0, le=0.03)
    angular_speed_rad_s: float = Field(default=0.15, gt=0, le=0.30)
    max_travel_m: float = Field(default=0.30, gt=0, le=0.40)
    max_joint_step_rad: float = Field(default=0.035, gt=0, le=0.035)
    # This compares observed label surfaces, including registration error
    # between cameras. Commissioned cross-view observations may need 20 mm.
    target_drift_m: float = Field(default=0.008, gt=0, le=0.020)
    pose_tolerance_m: float = Field(default=0.008, gt=0, le=0.015)
    sensor_max_age_ms: float = Field(default=150.0, gt=0, le=1000)
    sensor_max_skew_ms: float = Field(default=100.0, gt=0, le=1000)
    review_ttl_s: float = Field(default=2.0, gt=0, le=5.0)
    # Conservative TCP sweep, not a full robot collision certificate.
    tcp_clearance_m: float = Field(default=0.012, ge=0.005, le=0.03)
    target_exclusion_radius_m: float = Field(default=0.020, ge=0.005, le=0.025)
    # Explicitly commissioned held upright tube volume above its label.
    # Disabled by default; applies only after target-specific lift admission.
    held_target_upper_extent_m: float | None = Field(default=None, gt=0, le=0.05)
    target_surface_offset_m: float = Field(default=0.0, ge=0, le=0.01)
    geometry_orientation_search_rad: float = Field(default=0.0, ge=0, le=0.1)
    # Optional, task-specific empty-stop evidence. Never infer release from
    # motor current alone; default configurations do not authorize it.
    empty_stop_max_open_fraction: float | None = Field(default=None, ge=0, le=0.03)
    release_target_distance_min_m: float = Field(default=0.035, ge=0.035, le=0.08)
    release_target_lift_max_m: float = Field(default=0.005, gt=0, le=0.005)

    @model_validator(mode="after")
    def endpoints(self):
        from urllib.parse import urlparse
        import numpy as np

        for endpoint in (self.contact_graspnet_endpoint, self.graspgen_endpoint):
            if endpoint is not None:
                url = urlparse(endpoint)
                if (
                    url.scheme != "http"
                    or url.hostname not in {"127.0.0.1", "::1", "localhost"}
                    or url.username
                    or url.password
                    or url.query
                    or url.fragment
                ):
                    raise ValueError("grasp services must run on dodo loopback")
        if self.learned_grasp_to_tcp is not None:
            pose = np.asarray(self.learned_grasp_to_tcp)
            if (
                pose.shape != (4, 4)
                or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(pose[:3, :3]) < 0.999
            ):
                raise ValueError("learned grasp to TCP must be a rigid transform")
        if (
            self.learned_gripper_transfer_verified
            or self.learned_pregrasp_commissioning
        ) and (
            self.learned_grasp_to_tcp is None
            or not self.learned_gripper_id
            or not self.learned_model_sha256
        ):
            raise ValueError(
                "learned gripper transfer requires an identified gripper, model SHA and calibrated transform"
            )
        if (
            self.learned_parallel_jaw_half_turn
            and self.learned_gripper_id != "robotiq_2f_140"
        ):
            raise ValueError(
                "parallel jaw half-turn is configured only for robotiq_2f_140 proposals"
            )
        return self

    def validate_execution_phases(self, phases):
        if (
            self.learned_pregrasp_commissioning
            and not self.learned_gripper_transfer_verified
            and set(phases) - {"pregrasp"}
        ):
            raise ValueError(
                "unverified learned transfer commissioning permits pregrasp only"
            )

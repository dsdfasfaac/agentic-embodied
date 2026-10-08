"""Grasp proposals are read-only; reviewed plans use the sole gateway target loop."""

from __future__ import annotations

import math
import time
import uuid
from copy import deepcopy

import numpy as np

from robots.manipulation.grasp_proposals import (
    LocalGraspService,
    geometry_proposal,
    rigid_pose,
    transfer_grasp_pose,
    parallel_jaw_candidates,
)
from .grasp_contracts import GraspRecoveryConfig
from .motion import rotation
from .tools import ArrayPlan


def _base_points(points, transform):
    return (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]


class MeasuredPosePlan(ArrayPlan):
    def __init__(self, targets, kinematics, goal, tolerance):
        super().__init__(targets, convergence_target=targets[-1])
        self.kinematics, self.goal, self.pose_tolerance = kinematics, goal, tolerance

    def on_commit(self, context):
        super().on_commit(context)
        if self.reached:
            measured = np.asarray(context.observation["hardware"]["measured_state"])
            p, r, _ = self.kinematics.fk(measured[7:13])
            self.reached = bool(
                np.linalg.norm(p - self.goal[:3, 3]) <= self.pose_tolerance
                and np.linalg.norm(r - self.goal[:3, :3]) <= 0.05
            )


class GraspRecovery:
    """Episode-local proposal store, single-use review tokens and bounded IK."""

    def __init__(
        self,
        config: GraspRecoveryConfig,
        observer,
        *,
        closed_policy,
        open_policy,
        control_hz=15.0,
        clock=time.monotonic,
        engines=None,
        joint_bounds=None,
    ):
        self.config, self.observer = config, observer
        self.kinematics = observer.provider.tool_fk
        self.closed, self.open = closed_policy, open_policy
        self.hz, self.clock = control_hz, clock
        self.joint_bounds = (
            None if joint_bounds is None else np.asarray(joint_bounds, dtype=float)
        )
        if self.joint_bounds is not None and (
            self.joint_bounds.shape != (6, 2)
            or not np.isfinite(self.joint_bounds).all()
            or np.any(self.joint_bounds[:, 0] >= self.joint_bounds[:, 1])
        ):
            raise ValueError("grasp requires six finite ordered joint bounds")
        self.engines = engines or {
            "contact_graspnet": LocalGraspService(
                "contact_graspnet",
                config.contact_graspnet_endpoint,
                expected_gripper=config.learned_gripper_id,
                expected_model_sha256=config.learned_model_sha256,
            ),
            "graspgen": LocalGraspService(
                "graspgen",
                config.graspgen_endpoint,
                expected_gripper=config.learned_gripper_id,
                expected_model_sha256=config.learned_model_sha256,
                sampling_options={
                    "num_model_samples": config.graspgen_samples,
                    "sampling_batches": config.graspgen_sampling_batches,
                    "horizontal_closing_max": config.graspgen_horizontal_closing_max,
                    "horizontal_approach_max": config.graspgen_horizontal_approach_max,
                },
            ),
        }
        self.proposals, self.reviews = {}, {}
        self.completed_phases = {}
        self.selected_candidates = {}
        self.geometry = None
        if config.gripper_geometry_file:
            from .gripper_geometry import ArxGripperGeometry

            self.geometry = ArxGripperGeometry(
                config.gripper_geometry_file, config.gripper_geometry_sha256
            )

    def _fresh(self, observation):
        hardware = observation["hardware"]
        age = (time.monotonic_ns() - hardware["observation_completed_ns"]) / 1e6
        health = hardware["device_health"]
        if (
            not 0 <= age <= max(1000.0, 4 * self.config.sensor_max_age_ms)
            or hardware["sensor_age_ms"] > self.config.sensor_max_age_ms
            or hardware["sensor_skew_ms"] > self.config.sensor_max_skew_ms
            or not health["transport_responsive"]
            or health["fault_codes"]
        ):
            raise ValueError(
                "grasp requires fresh synchronized healthy device observations"
            )
        state = np.asarray(hardware["measured_state"], dtype=float)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("grasp requires measured 14D joint state")
        if (
            hardware.get("auxiliary_monotonic_ns", {}).get(
                "right_gripper_current_native"
            )
            != hardware.get("state_monotonic_ns")
            or hardware.get("state_monotonic_ns") is None
        ):
            raise ValueError(
                "grasp requires current and joints from the same fresh sample"
            )
        return state

    def _cloud(self, observation, images, *, camera="auto", region="full"):
        if camera == "auto" and region == "full":
            return self.observer.cloud(observation, images)
        return self.observer.cloud(observation, images, preferred_camera=camera,
            object_region=region, upper_crop_m=(self.config.target_upper_crop_min_m,
                                               self.config.target_upper_crop_max_m))

    def _proposal_cloud(self, stored, observation, images):
        return self._cloud(observation, images, camera=stored.get("camera", "auto"),
                           region=stored.get("object_region", "full"))

    def propose(self, args, context):
        observation, images = context["observation"], context["images"]
        state = self._fresh(observation)
        current = (
            observation["hardware"]
            .get("auxiliary_feedback", {})
            .get("right_gripper_current_native")
        )
        fraction = (state[13] - self.closed) / (self.open - self.closed)
        if current is None or not np.isfinite(current):
            raise ValueError("fresh gripper current is required before recovery")
        if fraction <= 0.30 and abs(current) >= 0.16:
            raise ValueError(
                "possible held object: confirm unloading before opening the gripper"
            )
        if args.object_region != "full" and args.engine == "tube_geometry":
            raise ValueError("upper tube region is a learned-grasp input, not a geometric TCP offset")
        cloud = self._cloud(observation, images, camera=args.camera, region=args.object_region)
        _, orientation, _ = self.kinematics.fk(state[7:13])
        if args.engine == "tube_geometry":
            candidates = geometry_proposal(
                cloud,
                orientation,
                surface_offset_m=self.config.target_surface_offset_m,
                orientation_search_rad=self.config.geometry_orientation_search_rad,
            )[: args.max_candidates]
        else:
            if args.engine == "graspgen" and isinstance(self.engines[args.engine], LocalGraspService):
                if args.horizontal_approach_max is not None and self.config.graspgen_horizontal_closing_max is None:
                    raise ValueError("horizontal approach requires configured closing-axis condition")
                options = self.engines[args.engine].sampling_options
                options["horizontal_approach_max"] = (self.config.graspgen_horizontal_approach_max
                    if args.horizontal_approach_max is None else args.horizontal_approach_max)
                if args.sampling_seed is None:
                    options.pop("seed", None)
                else:
                    options["seed"] = args.sampling_seed
            if (
                args.engine == "graspgen"
                and self.config.graspgen_approach_alignment_min is not None
            ):
                p, _, _ = self.kinematics.fk(state[7:13])
                target_base = _base_points(
                    cloud.target_camera_m[None], cloud.camera_to_base
                )[0]
                direction = target_base - p
                if np.linalg.norm(direction) < 0.01:
                    raise ValueError(
                        "cannot condition approach at near-zero target distance"
                    )
                if isinstance(self.engines[args.engine], LocalGraspService):
                    self.engines[args.engine].sampling_options.update(
                        preferred_approach_camera=(
                            cloud.camera_to_base[:3, :3].T
                            @ (direction / np.linalg.norm(direction))
                        ).tolist(),
                        approach_alignment_min=self.config.graspgen_approach_alignment_min,
                    )
            candidates = self.engines[args.engine].propose(
                cloud, max_candidates=args.max_candidates
            )
            if args.engine == "graspgen" and self.config.learned_parallel_jaw_half_turn:
                candidates = parallel_jaw_candidates(
                    candidates, max_candidates=args.max_candidates
                )
        candidates = [
            dict(
                item,
                transform_base=(
                    cloud.camera_to_base @ rigid_pose(item["transform_camera"])
                ).tolist(),
            )
            for item in candidates
        ]
        identity = "grasp-" + uuid.uuid4().hex
        output = {
            "proposal_id": identity,
            "observation_id": observation["observation_id"],
            "target_id": cloud.target_id,
            "engine": args.engine,
            "proposal_only": True,
            "environment_advanced": False,
            "candidates": candidates,
            "evidence": cloud.evidence,
        }
        if args.engine != "tube_geometry":
            output["evidence"]["model_service"] = deepcopy(
                getattr(self.engines[args.engine], "last_evidence", {})
            )
        self.proposals[identity] = {
            "output": deepcopy(output),
            "cloud": cloud,
            "created": self.clock(),
            "camera": args.camera,
            "object_region": args.object_region,
        }
        return output

    def _goal(self, candidate, engine, phase, state):
        goal = rigid_pose(candidate["transform_base"])
        if engine != "tube_geometry":
            if not self.config.learned_gripper_transfer_verified and not (
                self.config.learned_grasp_commissioning
                or phase == "pregrasp"
                and self.config.learned_pregrasp_commissioning
            ):
                raise ValueError(
                    "learned gripper to ARX transfer has not been verified"
                )
            goal = transfer_grasp_pose(
                candidate["transform_base"], np.eye(4), self.config.learned_grasp_to_tcp
            )
        if phase == "pregrasp":
            # ARX tool centre's +X axis is forward from link-six to the fingers.
            goal[:3, 3] -= self.config.pregrasp_distance_m * goal[:3, 0]
        elif phase == "lift":
            p, r, _ = self.kinematics.fk(state[7:13])
            goal[:3, :3] = r
            goal[:3, 3] = p + np.array([0.0, 0.0, self.config.lift_distance_m])
        return goal

    def _plan(self, command, state, goal, cloud, phase, max_steps, final_joint_hint=None,
              *, scene_state=None):
        observed_state = state if scene_state is None else scene_state
        try:
            return self._direct_plan(command, state, goal, cloud, phase, max_steps,
                                     final_joint_hint=final_joint_hint, scene_state=observed_state)
        except ValueError as original:
            if phase != "pregrasp" or self.config.pregrasp_escape_m <= 0:
                raise
            raised = np.eye(4)
            raised[:3, 3], raised[:3, :3], _ = self.kinematics.fk(state[7:13])
            raised[2, 3] += self.config.pregrasp_escape_m
            first, clearance = self._cartesian_plan(command, state, raised, cloud,
                                                    phase, max_steps, scene_state=observed_state)
            # One settling tail at the final learned pose, not at the waypoint.
            first = first[:-30]
            next_state = state.copy()
            next_state[7:13] = first[-1, 7:13]
            try:
                second, other_clearance = self._direct_plan(first[-1], next_state, goal,
                    cloud, phase, max_steps-len(first), final_joint_hint=final_joint_hint,
                    scene_state=observed_state)
            except ValueError as exc:
                raise ValueError(f"vertical escape could not reach learned pose after {original}: {exc}") from exc
            targets = np.vstack((first, second))
            positions = np.array([self.kinematics.fk(q[7:13])[0]
                                  for q in np.vstack((state, targets))])
            arc = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
            if arc > self.config.max_travel_m:
                raise ValueError("escaped pregrasp arc exceeds Cartesian travel budget")
            return targets, min(clearance, other_clearance)

    def _direct_plan(
        self, command, state, goal, cloud, phase, max_steps, final_joint_hint=None,
        scene_state=None,
    ):
        try:
            return self._cartesian_plan(command, state, goal, cloud, phase, max_steps, scene_state=scene_state)
        except ValueError as original:
            if (
                phase != "pregrasp"
                or self.config.pregrasp_planner != "joint_then_cartesian"
            ):
                raise
            from robots.manipulation.joint_paths import joint_paths
            from scipy.spatial import cKDTree

            obstacles = _base_points(cloud.scene_camera_m, cloud.camera_to_base)
            tree = self._scene_tree(obstacles, cloud, state if scene_state is None else scene_state)
            failures = []
            for path in joint_paths(
                self.kinematics,
                state[7:13],
                goal,
                hz=self.hz,
                speed_m_s=self.config.speed_m_s,
                angular_speed_rad_s=self.config.angular_speed_rad_s,
                max_joint_step_rad=self.config.max_joint_step_rad,
                max_steps=max_steps,
                final_joint_hint=final_joint_hint,
            ):
                clearance = float("inf")
                try:
                    positions = np.asarray(
                        [
                            self.kinematics.fk(q)[:2][0]
                            for q in np.vstack((state[7:13], path))
                        ]
                    )
                    arc_length = float(
                        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
                    )
                    if arc_length > self.config.max_travel_m:
                        raise ValueError(
                            "joint pregrasp arc exceeds Cartesian travel budget"
                        )
                    for q in path:
                        p, r, _ = self.kinematics.fk(q)
                        distance = float(tree.query(p)[0])
                        clearance = min(clearance, distance)
                        if distance < self.config.tcp_clearance_m:
                            raise ValueError(
                                "joint pregrasp TCP sweep intersects observed scene points"
                            )
                        if self.geometry:
                            pose = np.eye(4)
                            pose[:3, :3] = r
                            pose[:3, 3] = p
                            self.geometry.check(tree, pose)
                    targets = np.repeat(command[None], len(path), axis=0)
                    targets[:, 7:13] = path
                    targets = np.r_[targets, np.repeat(targets[-1:], [30], axis=0)]
                    return targets.astype(np.float32), clearance
                except ValueError as exc:
                    failures.append(str(exc))
            raise ValueError(
                "joint pregrasp failed after "
                + str(original)
                + ": "
                + "; ".join(failures or ["no path within rate and step budgets"])
            )

    def _scene_tree(self, obstacles, cloud, state):
        from scipy.spatial import cKDTree

        if self.geometry is not None:
            p, r, _ = self.kinematics.fk(state[7:13])
            pose = np.eye(4)
            pose[:3, :3] = r
            pose[:3, 3] = p
            own = self.geometry.occupied_mask(obstacles, pose)
            target = _base_points(cloud.target_surface_camera_m if cloud.target_surface_camera_m is not None else cloud.object_camera_m, cloud.camera_to_base)
            # Never erase the selected object as robot self geometry.
            own &= cKDTree(target).query(obstacles)[0] > 0.006
            if cloud.robot_self_camera_m is not None and len(cloud.robot_self_camera_m):
                measured_self = _base_points(cloud.robot_self_camera_m, cloud.camera_to_base)
                own |= ((cKDTree(measured_self).query(obstacles)[0] < 1e-6)
                        & (cKDTree(target).query(obstacles)[0] > .006))
            obstacles = obstacles[~own]
        if len(obstacles) < 32:
            raise ValueError(
                "insufficient observed obstacles after self geometry filtering"
            )
        return cKDTree(obstacles)

    def _cartesian_plan(self, command, state, goal, cloud, phase, max_steps, *, scene_state=None):
        q = state[7:13].copy()
        p, r, _ = self.kinematics.fk(q)
        travel = float(np.linalg.norm(goal[:3, 3] - p))
        if travel > self.config.max_travel_m:
            raise ValueError("grasp path exceeds configured Cartesian travel budget")
        # Stable SO(3) interpolation avoids the cross-product IK ambiguity at pi.
        from scipy.spatial.transform import Rotation

        rv = Rotation.from_matrix(goal[:3, :3] @ r.T).as_rotvec()
        count = max(
            1,
            math.ceil(travel / min(0.002, self.config.speed_m_s / self.hz)),
            math.ceil(np.linalg.norm(rv) / (self.config.angular_speed_rad_s / self.hz)),
        )
        if count + 30 > max_steps:
            raise ValueError("grasp path exceeds physical step budget")
        scene = _base_points(cloud.scene_camera_m, cloud.camera_to_base)
        target = _base_points(cloud.target_camera_m[None], cloud.camera_to_base)[0]
        if phase == "pregrasp":
            # Target remains an obstacle during pregrasp. Only engage/lift may enter its ROI.
            obstacles = scene
        else:
            relative = scene - target
            excluded = (
                np.linalg.norm(relative, axis=1)
                <= self.config.target_exclusion_radius_m
            )
            if phase == "lift" and self.config.held_target_upper_extent_m is not None:
                # A label-centred sphere cuts through the held tube's own
                # upper glass wall. The commissioned upright target column
                # moves with the grip; nearby objects outside it remain obstacles.
                excluded |= (
                    (
                        np.linalg.norm(relative[:, :2], axis=1)
                        <= self.config.target_exclusion_radius_m
                    )
                    & (relative[:, 2] >= 0)
                    & (relative[:, 2] <= self.config.held_target_upper_extent_m)
                )
            if self.config.target_surface_exclusion_m:
                from scipy.spatial import cKDTree

                target_points = _base_points(
                    cloud.target_surface_camera_m if cloud.target_surface_camera_m is not None else cloud.object_camera_m, cloud.camera_to_base
                )
                excluded |= (
                    cKDTree(target_points).query(scene)[0]
                    <= self.config.target_surface_exclusion_m
                )
            obstacles = scene[~excluded]
        if len(obstacles) < 32:
            raise ValueError("insufficient scene points for TCP clearance review")
        tree = self._scene_tree(obstacles, cloud, state if scene_state is None else scene_state)
        targets, minimum_clearance = [], float("inf")
        previous = q.copy()
        for index in range(1, count + 1):
            alpha = index / count
            desired = p + alpha * (goal[:3, 3] - p)
            clearance = float(tree.query(desired)[0])
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < self.config.tcp_clearance_m:
                raise ValueError("grasp TCP sweep intersects observed scene points")
            target_r = rotation(alpha * rv) @ r
            if self.geometry:
                pose = np.eye(4)
                pose[:3, :3] = target_r
                pose[:3, 3] = desired
                self.geometry.check(tree, pose)
            q = self.kinematics.solve(q, desired, target_r)
            if np.max(np.abs(q - previous)) > self.config.max_joint_step_rad:
                raise ValueError("grasp IK exceeds joint increment limit")
            if self.joint_bounds is not None and (
                np.any(q < self.joint_bounds[:, 0])
                or np.any(q > self.joint_bounds[:, 1])
            ):
                raise ValueError(
                    "grasp IK exceeds real controller joint command bounds"
                )
            row = command.copy()
            row[7:13] = q
            targets.append(row)
            previous = q.copy()
        targets.extend([targets[-1].copy() for _ in range(30)])
        return np.asarray(targets, dtype=np.float32), minimum_clearance

    def _phase_gate(self, proposal_id, phase, observation, state, images):
        fraction = (state[13] - self.closed) / (self.open - self.closed)
        if phase in {"pregrasp", "engage"} and fraction < 0.6:
            raise ValueError("pregrasp/engage requires an observed open gripper")
        completed = self.completed_phases.get(proposal_id, set())
        if phase == "engage" and "pregrasp" not in completed:
            raise ValueError("engage requires measured pregrasp completion")
        if phase == "lift":
            if "engage" not in completed:
                raise ValueError("lift requires measured engage completion")
            values = self.observer.provider.observe(observation, images)
            if not values["privileged.interaction.gripper_contact"]:
                raise ValueError("lift requires target-specific observed contact")

    def review(self, args, context):
        observation, images = context["observation"], context["images"]
        checks, token, selected = [], None, None
        try:
            state = self._fresh(observation)
            self._phase_gate(args.proposal_id, args.phase, observation, state, images)
            stored = self.proposals[args.proposal_id]
            cloud = self._proposal_cloud(stored, observation, images)
            old = stored["cloud"]
            target = _base_points(cloud.target_camera_m[None], cloud.camera_to_base)[0]
            original = _base_points(old.target_camera_m[None], old.camera_to_base)[0]
            if np.linalg.norm(target - original) > self.config.target_drift_m:
                raise ValueError(
                    "pink target moved since proposal; generate a new proposal"
                )
            if cloud.target_id != old.target_id:
                raise ValueError("grasp target identity changed")
            failures = []
            candidates = (
                [self.selected_candidates[args.proposal_id]]
                if args.phase != "pregrasp"
                and args.proposal_id in self.selected_candidates
                else stored["output"]["candidates"]
            )
            for candidate in sorted(
                candidates,
                key=lambda c: c.get("score", 0.0),
                reverse=True,
            ):
                try:
                    goal = self._goal(
                        candidate, stored["output"]["engine"], args.phase, state
                    )
                    if stored["output"]["engine"] != "tube_geometry":
                        mapped_tcp = np.asarray(
                            candidate["transform_base"]
                        ) @ rigid_pose(self.config.learned_grasp_to_tcp)
                        offset = float(np.linalg.norm(mapped_tcp[:3, 3] - original))
                        if offset > self.config.learned_target_distance_max_m:
                            raise ValueError(
                                "learned mapped TCP is outside selected target distance"
                            )
                    targets, clearance = self._plan(
                        np.asarray(context["command"]),
                        state,
                        goal,
                        cloud,
                        args.phase,
                        args.max_steps,
                    )
                    engage_preview = None
                    if args.phase == "pregrasp" and (self.config.validate_pregrasp_engage or args.require_engage_preview):
                        preview_state = targets[-1].copy()
                        engage_goal = self._goal(candidate, stored["output"]["engine"], "engage", preview_state)
                        engage_targets, engage_clearance = self._plan(
                            targets[-1], preview_state, engage_goal, cloud, "engage", 90,
                            scene_state=state)
                        engage_preview = {"planned_steps": len(engage_targets),
                            "minimum_tcp_clearance_m": engage_clearance,
                            "scope": "same learned pose against currently observed scene; fresh engage review still required"}
                    selected = {
                        "goal": goal,
                        "candidate": deepcopy(candidate),
                        "targets": targets,
                        "state": state,
                        "target": target,
                        "cloud": cloud,
                        "args": args,
                        "created": self.clock(),
                        "used": False,
                        "observation_id": observation["observation_id"],
                    }
                    checks.append(
                        {
                            "check": "bounded-joint-ik-and-tcp-clearance",
                            "gripper_cad_geometry_sha256": (
                                None if self.geometry is None else self.geometry.sha256
                            ),
                            "pregrasp_planner": self.config.pregrasp_planner,
                            "pass": True,
                            "planned_steps": len(targets),
                            "minimum_tcp_clearance_m": clearance,
                            "held_target_upper_extent_m": (
                                self.config.held_target_upper_extent_m
                                if args.phase == "lift"
                                else None
                            ),
                            "goal_tcp_base": goal.tolist(),
                            "target_evidence": cloud.evidence,
                            "engage_preview": engage_preview,
                        }
                    )
                    if stored["output"]["engine"] != "tube_geometry":
                        checks[-1].update(
                            learned_transfer_physically_verified=self.config.learned_gripper_transfer_verified,
                            learned_pregrasp_commissioning=self.config.learned_pregrasp_commissioning,
                            mapped_target_distance_m=offset,
                            selected_candidate=deepcopy(candidate),
                        )
                    break
                except ValueError as exc:
                    failures.append(str(exc))
            if selected is None:
                raise ValueError("no feasible grasp candidate: " + "; ".join(failures))
            token = "grasp-review-" + uuid.uuid4().hex
            self.reviews[token] = selected
        except (KeyError, ValueError) as exc:
            checks.append({"check": "grasp-review", "pass": False, "reason": str(exc)})
        return {
            "proposal_id": args.proposal_id,
            "observation_id": observation["observation_id"],
            "phase": args.phase,
            "eligible": selected is not None,
            "review_token": token,
            "checks": checks,
            "certificate_level": "sensor_tcp_clearance_and_joint_ik",
            "limitations": [
                "No full arm/finger collision certificate.",
                "D405 sees only exposed target surfaces.",
                "Learned scores are model-gripper specific.",
            ],
        }

    def prepare(self, args, context):
        if not self.config.motion_enabled:
            raise ValueError("grasp motion is disabled in the frozen configuration")
        review = self.reviews.get(args.review_token)
        if (
            review is None
            or review["used"]
            or self.clock() - review["created"] > self.config.review_ttl_s
            or args.phase != review["args"].phase
            or args.max_steps != review["args"].max_steps
        ):
            raise ValueError("grasp review token is stale, reused or mismatched")
        state = self._fresh(context.observation)
        if np.max(np.abs(state[7:13] - review["state"][7:13])) > 0.01:
            raise ValueError("arm moved since grasp review")
        self._phase_gate(
            review["args"].proposal_id,
            args.phase,
            context.observation,
            state,
            context.images,
        )
        current_cloud = self._proposal_cloud(self.proposals[review["args"].proposal_id], context.observation, context.images)
        target = _base_points(
            current_cloud.target_camera_m[None], current_cloud.camera_to_base
        )[0]
        if np.linalg.norm(target - review["target"]) > self.config.target_drift_m:
            raise ValueError("target moved since grasp review")
        if current_cloud.target_id != review["cloud"].target_id:
            raise ValueError("grasp target identity changed since review")
        # Recheck the entire sweep against current points, not the old proposal scene.
        targets, _ = self._plan(
            context.command,
            state,
            review["goal"],
            current_cloud,
            args.phase,
            args.max_steps,
            final_joint_hint=review["targets"][-1, 7:13],
        )
        # Preserve the current grip command; an intervening opening/closure invalidates old plans.
        targets = targets.copy()
        targets[:, [6, 13]] = context.command[[6, 13]]
        review["used"] = True
        owner = self
        proposal_id = review["args"].proposal_id

        class Plan(MeasuredPosePlan):
            def on_commit(self, current):
                super().on_commit(current)
                if self.reached:
                    owner.selected_candidates[proposal_id] = deepcopy(
                        review["candidate"]
                    )
                    owner.completed_phases.setdefault(proposal_id, set()).add(
                        args.phase
                    )

        return Plan(
            targets, self.kinematics, review["goal"], self.config.pose_tolerance_m
        )


class ProposeHandler:
    def __init__(self, recovery):
        self.recovery = recovery

    def inspect(self, args, context):
        return self.recovery.propose(args, context)


class ReviewHandler:
    def __init__(self, recovery):
        self.recovery = recovery

    def inspect(self, args, context):
        return self.recovery.review(args, context)


class GraspReentry:
    """VLA resumes only from the measured pregrasp of this recovery's target."""

    def __init__(self, base, grasp):
        self.base, self.grasp = base, grasp

    def inspect(self, args, context):
        result = self.base.inspect(args, context)
        proposal_id = context.get("tool_outputs", {}).get("proposal_id")
        if not proposal_id:
            return result
        observation, images = context["observations"][-1], context["images"][-1]
        passed = False
        try:
            grasp = self.grasp
            if "pregrasp" not in grasp.completed_phases.get(proposal_id, set()):
                raise ValueError("pregrasp_not_completed")
            state = grasp._fresh(observation)
            cloud = grasp.observer.cloud(observation, images)
            original = grasp.proposals[proposal_id]["cloud"]
            target = _base_points(cloud.target_camera_m[None], cloud.camera_to_base)[0]
            before = _base_points(
                original.target_camera_m[None], original.camera_to_base
            )[0]
            p, _, _ = grasp.kinematics.fk(state[7:13])
            passed = bool(
                cloud.target_id == original.target_id
                and np.linalg.norm(target - before) <= grasp.config.target_drift_m
                and np.linalg.norm(target - p)
                <= grasp.config.pregrasp_distance_m + grasp.config.pose_tolerance_m
            )
        except (ValueError, KeyError):
            passed = False
        result["checks"].append(
            {
                "check_id": "target-specific-pregrasp",
                "status": "pass" if passed else "fail",
                "evidence_ids": [observation["observation_id"]],
                "reason_code": (
                    "target_pregrasp_verified"
                    if passed
                    else "target_pregrasp_unverified"
                ),
            }
        )
        if not passed:
            result["status"] = "ineligible"
        return result


class TargetVerifiedGripperPlanner:
    """Guard opening with fresh target evidence and the calibrated empty stop.

    A fully closed PickTube gripper at its observed empty stop can have high
    current from the +0.9 preload. This narrowly scoped admission does not
    certify occupancy for other objects or grippers.
    """

    def __init__(self, planner, grasp):
        self.planner, self.grasp = planner, grasp

    def prepare(self, args, context):
        grasp = self.grasp
        state = grasp._fresh(context.observation)
        fraction = (state[13] - grasp.closed) / (grasp.open - grasp.closed)
        evidence = {
            "observation_id": context.observation["observation_id"],
            "measured_open_fraction": float(fraction),
            "requested_opening": args.opening,
        }
        if args.opening > fraction + 0.01 and fraction < 0.60:
            values = grasp.observer.provider.observe(
                context.observation, context.images
            )
            stop = grasp.config.empty_stop_max_open_fraction
            empty = (
                stop is not None
                and -0.01 <= fraction <= stop
                and values.get("privileged.interaction.gripper_contact") is False
                and values.get("privileged.interaction.grasped") is False
                and values.get("privileged.interaction.success") is False
                and type(values.get("privileged.interaction.lift_m")) in (int, float)
                and abs(values["privileged.interaction.lift_m"])
                <= grasp.config.release_target_lift_max_m
                and type(values.get("privileged.selected.target_gripper_distance_m"))
                in (int, float)
                and values["privileged.selected.target_gripper_distance_m"]
                >= grasp.config.release_target_distance_min_m
            )
            if not empty:
                raise ValueError(
                    "opening requires observed empty stop and separated unlifted target; possible held object"
                )
            evidence.update(
                reason="calibrated_empty_stop_and_target_separation", features=values
            )
        else:
            evidence["reason"] = "closing_or_already_open"
        plan = self.planner.prepare(args, context)
        plan.admission_evidence = evidence
        return plan

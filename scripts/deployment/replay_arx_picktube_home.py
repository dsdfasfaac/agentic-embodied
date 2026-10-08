#!/usr/bin/env python3
"""Return an EMPTY right gripper along the recorded PickTube return corridor.

Default invocation only reads feedback and writes a plan. The recorded six
right joint coordinates are replayed in their original order, slowed down.
Both grippers and the left arm are held; opening/staging is a separate phase.
A fresh posture outside the corridor first needs a checked upward entry.
Failure holds the last measured posture and leaves the controller running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_task_manifest
from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.gateway.arx_ros2_device import ArxRos2Device, Ros2Topics
from robots.arx.gateway.real_config import load_real_hardware_config
from scripts.deployment.stage_arx_picktube_start import _read_fresh

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY = ROOT / 'robots/arx/manifests/real/dodo_picktube_recorded_home.json'
TRAJECTORY_SHA = '302209078d5148338b885b4302fb545fe54cb8f34a37bcc13531f9a54518277d'
JOINT_STEP = .02
JOINT_SPEED = .2


def load_recording(path=TRAJECTORY, digest=TRAJECTORY_SHA):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('recorded return SHA mismatch')
    value = json.loads(raw)
    q = np.asarray(value['right_joints_rad'], dtype=float)
    t = np.asarray(value['timestamps_s'], dtype=float)
    indices = np.asarray(value['frame_indices'])
    xyz = np.asarray(value['right_controller_xyz_m'], dtype=float)
    if (value['schema_version'] != 'arx.recorded.home.v1' or value['task'] != 'PickTube'
            or value['joint_source'] != 'observation.state[:,7:13]'
            or q.shape != (len(t), 6) or xyz.shape != (len(t), 3) or len(t) < 3
            or not np.isfinite(q).all() or not np.isfinite(xyz).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or not np.array_equal(np.diff(indices), np.ones(len(t) - 1))):
        raise ValueError('invalid or reordered recorded return')
    if float(np.max(xyz[:, 2]) - xyz[0, 2]) < .1:
        raise ValueError('recording does not contain a lift before return')
    return value, q, t


def plan_home(current, goal, fk, recording, *, control_hz=15):
    """Create all targets before sending any; preserve taught return ordering."""
    value, recorded, times = recording
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if current.shape != (14,) or not np.isfinite(current).all():
        raise ValueError('home requires finite measured 14D state')
    poses = np.array([fk.fk(q)[0] for q in recorded])
    q = current[7:13].copy()
    initial_p, initial_r, _ = fk.fk(q)
    goal_p = fk.fk(goal[7:13])[0]
    targets, phases, frames = [], [], []
    max_delta = min(JOINT_STEP, JOINT_SPEED / control_hz)

    def append_segment(end, phase, frame=None, duration=0.):
        nonlocal q
        n = max(1, math.ceil(float(np.max(np.abs(end - q))) / max_delta),
                math.ceil(duration * control_hz))
        start = q.copy()
        for k in range(1, n + 1):
            next_q = start + (end - start) * k / n
            fk.fk(next_q)  # All interpolation points obey the reviewed limits.
            target = current.copy()
            target[7:13] = next_q
            targets.append(target)
            phases.append(phase)
            frames.append(frame)
        q = end.copy()

    # Already in the low home area: do not make an unnecessary trip to the rack.
    near_home = (np.max(np.abs(q - goal[7:13])) <= .12
                 and np.linalg.norm(initial_p - goal_p) <= .02)
    join = None
    if not near_home:
        distances = np.max(np.abs(recorded - q), axis=1)
        close = np.flatnonzero((distances <= .12) &
                              (np.linalg.norm(poses - initial_p, axis=1) <= .025))
        if len(close):
            join = int(close[np.argmin(distances[close])])
            # Even a short joint bridge must not lower a gripper near the rack.
            bridge = np.array([fk.fk(q + a * (recorded[join] - q))[0]
                               for a in np.linspace(0, 1, 31)])
            if bridge[:, 2].min() < initial_p[2] - .002:
                close = []
        if not len(close):
            # Join only the raised return corridor; first escape vertically at
            # the CURRENT x/y and orientation. No direct low diagonal shortcut.
            eligible = np.flatnonzero(poses[:, 2] >= max(.10, initial_p[2] + .06))
            if not len(eligible):
                raise ValueError('current pose has no raised recorded entry')
            costs = np.linalg.norm(poses[eligible, :2] - initial_p[:2], axis=1)
            errors = []
            entry_q = q.copy()
            join = None
            for candidate in eligible[np.argsort(costs)]:
                if np.linalg.norm(poses[candidate, :2] - initial_p[:2]) > .12:
                    continue
                height = max(initial_p[2] + .06, poses[candidate, 2])
                trial_q, vertical = entry_q.copy(), []
                try:
                    n = max(1, math.ceil((height - initial_p[2]) / .001))
                    for k in range(1, n + 1):
                        p = initial_p.copy(); p[2] += (height - initial_p[2]) * k / n
                        trial_q = fk.solve(trial_q, p, initial_r)
                        vertical.append(trial_q.copy())
                    bridge = np.array([fk.fk(trial_q + a * (recorded[candidate] - trial_q))[0]
                                       for a in np.linspace(0, 1, 101)])
                    # The raised bridge may dip during wrist rotation. Keep
                    # it above the rack clearance floor, rather than requiring
                    # every point to stay at its highest endpoint altitude.
                    escape_floor = max(.08, initial_p[2] + .04)
                    if bridge[:, 2].min() < escape_floor:
                        raise ValueError('raised entry bridge lowers below escape plane')
                    for step in vertical:
                        append_segment(step, 'vertical_escape')
                    join = int(candidate)
                    break
                except ValueError as exc:
                    errors.append(str(exc))
            if join is None:
                raise ValueError('no upward entry into recorded return: ' + '; '.join(errors[:3]))
        append_segment(recorded[join], 'recorded_entry', value['frame_indices'][join])
        for i in range(join + 1, len(recorded)):
            append_segment(recorded[i], 'recorded_return', value['frame_indices'][i],
                           duration=2 * float(times[i] - times[i - 1]))
    # Only a short final alignment is allowed, after reaching the recorded home.
    if np.max(np.abs(q - goal[7:13])) > .12 or np.linalg.norm(fk.fk(q)[0] - goal_p) > .02:
        raise ValueError('recorded endpoint is not near the frozen task home')
    append_segment(goal[7:13], 'home_alignment')
    return np.asarray(targets), {
        'phases': phases, 'source_frames': frames,
        'join_source_frame': None if join is None else value['frame_indices'][join],
        'already_near_home': bool(near_home),
        'grippers_preserved': current[[6, 13]].tolist(),
        'left_arm_preserved': current[:6].tolist(),
        'initial_tcp_m': initial_p.tolist(), 'goal_tcp_m': goal_p.tolist(),
        'raised_entry_floor_m': max(.08, initial_p[2] + .04),
        'joint_speed_limit_rad_s': JOINT_SPEED,
        'max_joint_step_rad': max_delta,
        'scope': 'Recorded empty-gripper corridor and FK entry; no full-arm collision certification',
    }


def replay_home(hardware_path, hardware_sha, task_path, output, *, execute=False):
    config = load_real_hardware_config(hardware_path, hardware_sha)
    if config.arm_transport != 'arx_ros2' or config.timing.control_hz != 15 or not config.command_right:
        raise ValueError('recorded home requires the dodo 15Hz ROS2 deployment')
    provider = PickTubeRgbdProvider(); provider.validate_hardware(config)
    recording = load_recording()
    # Validate source positions against the controller-EE reference, NOT the
    # tool centre. The recorded gripper coordinate is deliberately discarded.
    positions = [provider.controller_fk.fk(q)[0] for q in recording[1]]
    if np.max(np.linalg.norm(np.asarray(positions) - recording[0]['right_controller_xyz_m'], axis=1)) > .01:
        raise ValueError('recorded joints and controller FK disagree')
    task = load_task_manifest(task_path)
    goal = np.asarray(task.start_state, dtype=float).copy()
    goal[[6, 13]] += np.asarray(task.control.gripper_command_offsets)
    device = ArxRos2Device.from_ros2(topics=Ros2Topics(**config.ros2_topics),
        left_calibration=config.left.calibration(), right_calibration=config.right.calibration(),
        command_left=False, command_right=True, max_status_age_ms=150,
        max_pair_skew_ms=50, keepalive_hz=60, command_lease_s=1.5)
    report = {'schema_version': 'arx.recorded.home.execution.v1', 'status': 'checking',
              'trajectory_sha256': TRAJECTORY_SHA, 'source_sha256': recording[0]['source_sha256'],
              'hardware_sha256': hardware_sha, 'command_log': [], 'controller_disabled': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = None
    motion_started = False
    try:
        sample = _read_fresh(device, time.monotonic() + 5)
        initial = np.asarray(sample.positions, dtype=float)
        targets, plan = plan_home(initial, goal, provider.tool_fk, recording)
        for target in targets:
            device._native_target(target[7:], config.right.calibration())
        report.update(initial_state=initial.tolist(), plan=plan,
                      planned_targets=targets.tolist(), status='planned')
        output.write_text(json.dumps(report, indent=2) + '\n')
        if not execute:
            return report
        tolerance = np.asarray(config.timing.position_tolerance)
        period = 1 / config.timing.control_hz
        for index, target in enumerate(targets):
            started = time.monotonic()
            # Stream the recorded ramp. Pause its clock if feedback lags; do
            # not keep pushing commands into an immobile joint or skip frames.
            deadline = started + 2.
            while np.max(np.abs(sample.positions[7:13] - target[7:13])) > .10:
                if time.monotonic() >= deadline:
                    raise TimeoutError('recorded path tracking stalled before next command')
                sample = _read_fresh(device, deadline)
                time.sleep(.02)
            receipt = device.send(target.astype(np.float32), 'recorded-home-' + uuid.uuid4().hex)
            motion_started = True
            deadline = time.monotonic() + 2.
            while True:
                sample = _read_fresh(device, deadline)
                if sample.monotonic_ns > receipt.sent_monotonic_ns:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('no fresh feedback after home command')
                time.sleep(.01)
            error = np.asarray(sample.positions) - target
            report['command_log'].append({'index': index, 'phase': plan['phases'][index],
                'source_frame': plan['source_frames'][index], 'target': target.tolist(),
                'receipt_status': receipt.status, 'sent_monotonic_ns': receipt.sent_monotonic_ns,
                'measured_state': sample.positions.tolist(), 'measured_monotonic_ns': sample.monotonic_ns,
                'arrival_verified': bool(np.all(np.abs(error[7:14]) <= tolerance[7:14]))})
            if np.max(np.abs(error[7:13])) > .15:
                raise ValueError('recorded path tracking error exceeds 0.15 rad')
            # Checkpoint each segment end. Streaming publication itself is not arrival.
            checkpoint = index == len(targets)-1 or plan['phases'][index+1] != plan['phases'][index]
            if checkpoint:
                deadline = time.monotonic() + 3
                while np.any(np.abs(sample.positions[7:14] - target[7:14]) > tolerance[7:14]):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('home phase did not physically arrive')
                    device.send(target.astype(np.float32), 'home-settle-' + uuid.uuid4().hex)
                    sample = _read_fresh(device, deadline)
                    time.sleep(.02)
                report['command_log'][-1]['arrival_verified'] = True
                report['command_log'][-1]['checkpoint_measured_state'] = sample.positions.tolist()
                report['command_log'][-1]['checkpoint_monotonic_ns'] = sample.monotonic_ns
            output.write_text(json.dumps(report, indent=2) + '\n')
            time.sleep(max(0, period - (time.monotonic() - started)))
        report.update(status='complete', final_state=sample.positions.tolist(),
                      measured_right_home_verified=True)
        return report
    except Exception as exc:
        report.update(status='failed_keep_enabled', error=str(exc))
        if motion_started and sample is not None:
            try:
                sample = _read_fresh(device, time.monotonic() + 1.)
                held = device.send(np.asarray(sample.positions, dtype=np.float32), 'home-fault-hold-' + uuid.uuid4().hex)
                report['fault_hold'] = {'receipt_status': held.status, 'target': sample.positions.tolist(),
                                       'sent_monotonic_ns': held.sent_monotonic_ns}
            except Exception as hold_error:
                report['fault_hold_error'] = str(hold_error)
        return report
    finally:
        device.close()
        output.write_text(json.dumps(report, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware-config', type=Path, required=True)
    p.add_argument('--hardware-sha256', required=True)
    p.add_argument('--task', type=Path, default=ROOT / 'robots/arx/manifests/pickup_test_tube.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    result = replay_home(args.hardware_config, args.hardware_sha256, args.task, args.output, execute=args.execute)
    print(json.dumps({k: result[k] for k in ('status', 'error', 'measured_right_home_verified') if k in result}))
    raise SystemExit(0 if result['status'] in ('planned', 'complete') else 2)


if __name__ == '__main__':
    main()

"""OFFLINE evaluation only; refuse private data until every batch trial is closed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'loop2_stage2_c2_eef_v2'))
from eef_tools import CommandKinematics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def physical_windows(trace, kinematics):
    relative, good, passing = [], [], []
    for row in trace:
        private = row['full_pickup_private']
        pos, rot, _ = kinematics.fk(row['measured_state'][7:13])
        relative.append(rot.T @ (np.asarray(private['target_position']) - pos))
        good.append(private['conservative_vertical_clearance_m'] >= .003
                    and all(row['evaluation']['finger_contacts'])
                    and not private['nonfinger_contact_bodies'])
    for end in range(14, len(trace)):
        pts = np.asarray(relative[end - 14:end + 1])
        drift = np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=2))
        if all(good[end - 14:end + 1]) and drift <= .003:
            passing.append(end)  # trace includes initial frame zero
    return passing


def evaluate(root):
    manifest = json.loads((root / 'frozen_manifest.json').read_text())
    sessions = [root / f'seed_{seed}' for seed in manifest['seeds']]
    # Complete all public checks before opening ANY privileged file.
    for session in sessions:
        decisions = rows(session / 'agent_decisions.jsonl')
        assert decisions and decisions[-1]['tool'] == 'finish', f'not closed: {session}'
        assert (session / 'audit.json').exists() and (session / 'posthoc_trace.json').exists()
        for source, digest in manifest['source_sha256'].items():
            executed = session / 'executed_source' / Path(source).name
            assert hashlib.sha256(executed.read_bytes()).hexdigest() == digest, source
    results = []
    for session in sessions:
        audit = json.loads((session / 'audit.json').read_text())
        trace = json.loads((session / 'posthoc_trace.json').read_text())
        kinematics = CommandKinematics(json.loads((session / 'robot_calibration.json').read_text()))
        passing = physical_windows(trace, kinematics)
        with np.load(session / 'trajectory.npz') as data:
            sources = data['command_sources'].tolist()
        assert len(trace) == len(sources) + 1 == audit['frames'] + 1
        segments = []
        for frame, source in enumerate(sources, 1):
            if not segments or segments[-1]['source'] != source:
                segments.append(dict(source=source, start_frame=frame, end_frame=frame))
            else:
                segments[-1]['end_frame'] = frame
        requests = rows(session / 'cosmos_requests.jsonl')
        decisions = rows(session / 'agent_decisions.jsonl')
        ids = [d['decision_id'] for d in decisions]
        assert len(ids) == len(set(ids))
        assert all(r['approved_by'] in ids for r in requests)
        assert all(b['frame'] > a['frame'] for a, b in zip(requests, requests[1:]))
        assert audit['prefix_command_max_abs_error'] == 0
        assert audit['prefix_measured_state_max_abs_error_posthoc_only'] == 0
        observations = rows(session / 'agent_observations.jsonl')
        proposals = {}
        for observation in observations:
            for proposal in observation['latest_proposals']:
                proposals[(proposal['frame'], proposal['proposal']['kind'])] = proposal
        gates = rows(session / 'pregrasp_proposals.jsonl')
        post_policy = [(frame, row) for frame, row in enumerate(trace)
                       if row['full_pickup_private']['active_tool'] == 'cosmos_after_recovery']
        result = dict(
            seed=audit['seed'], frames=audit['frames'], prefix_frames=audit['prefix_frames'],
            prefix_reproduction_exact=True, frozen_source_hashes_match=True,
            controller_segments=segments, controller_steps=dict(Counter(sources)),
            agent_tool_calls=dict(Counter(d['tool'] for d in decisions)),
            critic_proposals=[dict(frame=f, kind=kind) for f, kind in sorted(proposals)],
            pregrasp_reviews=gates, validated_reentries=rows(session / 'validated_reentries.jsonl'),
            fresh_cosmos_requests=len(requests),
            post_recovery_cosmos_steps=audit['post_recovery_cosmos_steps'],
            handoff_executed=audit['vla_handoff_executed'],
            post_recovery_bilateral_contact_frames=[frame for frame, row in post_policy
                                                    if all(row['evaluation']['finger_contacts'])],
            max_post_recovery_lift_m=max((row['evaluation']['lift'] for _, row in post_policy), default=None),
            terminal_agent_reason=decisions[-1]['reason'],
            legacy_environment_success_ever=audit['legacy_environment_success_ever'],
            full_pickup_passing_window_end_frames=passing,
            final_full_pickup_physical_pass=bool(passing and passing[-1] == len(trace) - 1),
            max_conservative_clearance_m=max(r['full_pickup_private']['conservative_vertical_clearance_m'] for r in trace),
            final_conservative_clearance_m=trace[-1]['full_pickup_private']['conservative_vertical_clearance_m'],
            final_evaluation=trace[-1]['evaluation'],
            formal_stage2_pass=False,
            audit_boundary='Read only after ALL live batch trials closed; never used in online decisions.',
        )
        (session / 'closed_loop_validation.json').write_text(json.dumps(result, indent=2))
        results.append(result)
    summary = dict(seeds=manifest['seeds'], trial_count=len(results),
                   handoffs=sum(r['handoff_executed'] for r in results),
                   final_full_pickup_physical_passes=sum(r['final_full_pickup_physical_pass'] for r in results),
                   scope=manifest['scope'], trials=results)
    (root / 'batch_validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'trials'}, indent=2))
    for r in results:
        print(json.dumps({k: r[k] for k in ('seed', 'frames', 'agent_tool_calls', 'critic_proposals',
              'post_recovery_cosmos_steps', 'fresh_cosmos_requests', 'handoff_executed',
              'full_pickup_passing_window_end_frames', 'final_full_pickup_physical_pass',
              'legacy_environment_success_ever', 'max_conservative_clearance_m', 'final_conservative_clearance_m')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    evaluate(parser.parse_args().batch)

"""Configuration/transport adapter; no recovery selection or new classifier rules."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr import config
sys.path.insert(0, str(ROOT / 'vendor/loop2_stage2_c2_pregrasp_v4'))
from run_pregrasp_session import PregraspSession
import run_agent_session as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--mode', choices=['live', 'vla-first', 'replay-pregrasp', 'smoke'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--approval')
    p.add_argument('--scene', type=Path)
    args = p.parse_args()
    c = config(args.config)
    base.ROOT, base.SCENE = Path(c['repo_root']), Path(c['scene'])
    if args.mode == 'vla-first':
        if args.scene is None:
            raise ValueError('--scene is required for vla-first')
        base.SCENE = args.scene.resolve()
    args.prefix_limit = 512
    args.approved_replay = ROOT / 'fixtures/seed_183173/pregrasp_replay' if args.mode in ('replay-pregrasp', 'smoke') else None
    args.replay_approval = args.approval
    session = PregraspSession(args)
    (args.output / 'delivery_run.json').write_text(json.dumps(dict(mode=args.mode, seed=args.seed,
        recorded_recovery_replay=args.mode != 'live', independent_recovery_trial=args.mode == 'live',
        config=c, argv=sys.argv, one_recovery_cycle_is_agent_protocol=True), indent=2))
    shutil.copy2(ROOT / 'MANIFEST.json', args.output / 'delivery_manifest.json')
    failed = False
    try:
        session.bootstrap()
        if args.mode == 'smoke':
            # Fixed regression assertion, NOT an online Agent decision or an automatic handoff.
            session.execute(dict(decision_id='smoke-gate', evidence_frame=session.frames,
                tool='review_pregrasp', args=dict(measurement_mode='temporal_rgb',
                    observation_frames=[573,594,612,632,652,673,694,715,723]),
                reason='Explicit recorded-action regression: verify saved pregrasp gate; no nominal actions requested.'))
            proposal = json.loads((args.output / 'pregrasp_proposals.jsonl').read_text().splitlines()[-1])
            if session.frames != 723 or not proposal['eligible']:
                raise ValueError('golden pregrasp regression failed')
            (args.output / 'smoke_result.json').write_text(json.dumps(dict(
                frame=session.frames, exact_rgb_checkpoints=True, gate_eligible=True,
                fresh_cosmos_requests=0, independent_recovery_trial=False), indent=2))
        else:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    if not session.execute(json.loads(line)):
                        break
                except Exception as exc:
                    session.log('rejections.jsonl', dict(frame=session.frames, reason=str(exc)))
                    session.snapshot(f'request rejected: {exc}; no automatic fallback')
    except BaseException:
        failed = True
        raise
    finally:
        decisions = args.output / 'agent_decisions.jsonl'
        last = json.loads(decisions.read_text().splitlines()[-1]) if decisions.exists() else {}
        if last.get('tool') != 'finish':
            # Explicitly transport-owned close, not a fabricated Agent decision.
            session.log('transport_events.jsonl', dict(frame=session.frames,
                event='exception_close' if failed else ('smoke_close' if args.mode == 'smoke' else 'stdin_eof_close')))
        session.close()
        (args.output / 'CLOSED.json').write_text(json.dumps(dict(frame=session.frames, failed=failed,
            explicit_agent_finish=last.get('tool') == 'finish', mode=args.mode)))


if __name__ == '__main__':
    main()

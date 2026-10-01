"""H20 integration check: black out only the left RGB stream, reset, hold, close.

No Cosmos request, recovery selection, or pickup success claim. Run using the
configured client Python; this deliberately perturbs observations for testing.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr import client_env, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.h20.json')
    args = parser.parse_args()
    c = config(args.config)
    os.environ.update(client_env(c))
    sys.path.insert(0, c['repo_root'])
    sys.path.insert(0, str(ROOT / 'vendor/loop2_stage2_c2_pregrasp_v4'))
    from run_pregrasp_session import PregraspSession
    import run_agent_session as base
    from phase_critic import PhaseAwareRgbCritic

    original_env = base.ArxMujocoEnv

    class BlackLeftEnv(original_env):
        def reset(self, *a, **kw):
            obs, info = super().reset(*a, **kw)
            obs['left_rgb'] = obs['left_rgb'].copy()
            obs['left_rgb'][:] = 0
            return obs, info

        def step(self, *a, **kw):
            obs, *rest = super().step(*a, **kw)
            obs['left_rgb'] = obs['left_rgb'].copy()
            obs['left_rgb'][:] = 0
            return (obs, *rest)

    base.ArxMujocoEnv = BlackLeftEnv
    base.ROOT, base.SCENE = Path(c['repo_root']), Path(c['scene'])
    session = None
    try:
        session = PregraspSession(SimpleNamespace(
            output=args.output, seed=183173, port=c['port'], max_steps=900,
            prefix=ROOT / 'fixtures/seed_183173/original_prefix.npz',
            prefix_limit=512, approved_replay=None, replay_approval=None))
        assert isinstance(session.critic, PhaseAwareRgbCritic)
        assert set(session.critic.confirmed.reset_images) == {'front_rgb', 'right_rgb'}
        assert session.gate.baseline['left_rgb'] is None
        for _ in range(6):
            session.step(session.command.copy(), 'hold')
        assert set(session.critic.last_features['views']) == {'front_rgb', 'right_rgb'}
        assert not session.rgb()['left_rgb'].any()
        assert session.fresh_requests == 0
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            base.ArxMujocoEnv = original_env
    print(json.dumps(dict(passed=True, frames=session.frames,
        critic_cameras=list(session.critic.CAMERAS), left_rgb='synthetically black',
        fresh_cosmos_requests=0, full_rollout_validated=False)))


if __name__ == '__main__':
    main()

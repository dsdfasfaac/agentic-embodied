"""Maintainer-only deterministic collection; no online behaviour changes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    args = p.parse_args()
    workspace = args.workspace
    batch = workspace / 'outputs/loop2_stage2_c2_pregrasp_v4/eval_other_seeds_20260921'
    original = json.loads((batch / 'frozen_manifest.json').read_text())
    files = list(original['source_sha256']) + [
        'loop2_stage2_c2_phase_v3/test_phase_critic.py',
        'loop2_stage2_c2_phase_v3/annotate_video.py',
        'loop2_stage2_c2_pregrasp_v4/test_pregrasp_gate.py',
        'loop2_stage2_c2_pregrasp_v4/test_evaluate_frozen_batch.py',
        'loop2_stage2_c2_pregrasp_v4/evaluate_frozen_batch.py',
        'loop2_stage2_c2_pregrasp_v4/verify_saved_gate.py']
    for name in files:
        source, target = workspace / name, ROOT / 'vendor' / name
        if name in original['source_sha256']:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == original['source_sha256'][name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Derived replay fixture uses own actions, public RGB and decision provenance only.
    import numpy as np
    source = batch / 'seed_183173'
    target = ROOT / 'fixtures/seed_183173/pregrasp_replay'
    target.mkdir(parents=True, exist_ok=True)
    with np.load(source / 'trajectory.npz') as data:
        np.savez_compressed(target / 'trajectory.npz', raw_targets=data['raw_targets'][:723],
                            command_sources=data['command_sources'][:723])
    observations = [json.loads(line) for line in (source / 'agent_observations.jsonl').read_text().splitlines()
                    if json.loads(line)['frame'] <= 723]
    (target / 'agent_observations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in observations))
    for row in observations:
        name = f"frame_{row['frame']:04d}"
        shutil.copytree(source / name, target / name, dirs_exist_ok=True)
    decisions = [json.loads(line) for line in (source / 'agent_decisions.jsonl').read_text().splitlines()
                 if json.loads(line)['tool'] not in ('resume_vla', 'finish')]
    (target / 'agent_decisions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in decisions))
    provenance = ROOT / 'provenance'
    provenance.mkdir(exist_ok=True)
    shutil.copy2(batch / 'frozen_manifest.json', provenance / 'frozen_runtime_manifest.json')
    result = ROOT / 'reference_results'
    result.mkdir(exist_ok=True)
    shutil.copy2(batch / 'batch_validation.json', result / 'batch_validation.json')
    for seed in (43935,183173,344246):
        dest = result / f'seed_{seed}'
        dest.mkdir(exist_ok=True)
        shutil.copy2(batch / f'seed_{seed}' / 'three_view_controller_labels.mp4', dest / 'three_view_controller_labels.mp4')
    print('Collected frozen runtime, recorded recovery fixture, tests and reference videos.')


if __name__ == '__main__':
    main()

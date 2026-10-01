"""Doctor, unit tests, post-close audit and video labelling. Never selects recovery."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr import config
VENDOR = ROOT / 'vendor'


def doctor(c):
    paths = {key: Path(c[key]).exists() for key in
             ('repo_root', 'scene', 'client_python', 'server_python', 'bundle', 'server_deps',
              'checkpoint', 'model_config', 'libstdcpp')}
    for relative in ('scene.xml', 'task.yaml', 'mapping.json'):
        paths[f'scene/{relative}'] = (Path(c['scene']) / relative).is_file()
    for relative in ('preprocessor_config.json', 'tokenizer.json', 'model.safetensors.index.json'):
        paths[f'checkpoint/{relative}'] = (Path(c['checkpoint']) / relative).is_file()
    for relative in ('models/Wan2.2_VAE.pth',
                     'third_party/cosmos-framework/cosmos_framework/scripts/action_policy_server_arx_task7_edge.py'):
        paths[relative] = (Path(c['bundle']) / relative).exists()
    import imageio_ffmpeg
    ffmpeg = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
    print(json.dumps({'paths': paths, 'python': sys.version, 'python_prefix': sys.prefix, 'ffmpeg': ffmpeg}, indent=2))
    if not all(paths.values()) or not Path(ffmpeg).is_file():
        raise RuntimeError('missing prerequisite; fix config/permissions, do not alter shared installations')
    versions = {}
    for distribution, module in [('numpy', 'numpy'), ('mujoco', 'mujoco'), ('ImageIO', 'imageio'),
                                 ('imageio-ffmpeg', 'imageio_ffmpeg'), ('pillow', 'PIL'),
                                 ('PyYAML', 'yaml'), ('pyzmq', 'zmq'), ('gymnasium', 'gymnasium')]:
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    importlib.import_module('robots.arx.environment')
    importlib.import_module('robots.arx.cosmos_edge_client')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(Path(c['scene']) / 'scene.xml'))
    print(json.dumps({'client_versions': versions, 'scene_compiles': True,
                      'cameras': model.ncam, 'note': 'No GPU model loaded; smoke verifies rendered RGB.'}, indent=2))
    expected_path = ROOT / 'provenance/external_files.json'
    if expected_path.exists():
        mismatches = []
        for row in json.loads(expected_path.read_text()):
            path = Path(c[row['root']]) / row['relative']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                mismatches.append(str(path))
        print(json.dumps({'external_source_hash_mismatches': mismatches}))
        if mismatches:
            raise RuntimeError('external source/config drift; exact reproduction not established')


def audit(session):
    # No private data opened unless this delivery worker has completed close().
    if not (session / 'CLOSED.json').is_file():
        raise ValueError('CLOSED.json missing: finish the session first; for a batch close every session before auditing')
    closed = json.loads((session / 'CLOSED.json').read_text())
    if closed['failed']:
        raise ValueError('run closed with exception; inspect transport log before interpreting results')
    sys.path.insert(0, str(VENDOR / 'loop2_stage2_c2_pregrasp_v4'))
    from evaluate_frozen_batch import physical_windows
    from eef_tools import CommandKinematics
    a = json.loads((session / 'audit.json').read_text())
    trace = json.loads((session / 'posthoc_trace.json').read_text())
    k = CommandKinematics(json.loads((session / 'robot_calibration.json').read_text()))
    passing = physical_windows(trace, k)
    result = dict(mode=closed['mode'], seed=a['seed'], frames=a['frames'],
        recorded_recovery_replay=closed['mode'] != 'live',
        prefix_command_max_abs_error=a['prefix_command_max_abs_error'],
        prefix_measured_state_max_abs_error=a['prefix_measured_state_max_abs_error_posthoc_only'],
        handoff_executed=a['vla_handoff_executed'],
        fresh_cosmos_requests=a['fresh_cosmos_requests'],
        post_recovery_cosmos_steps=a['post_recovery_cosmos_steps'],
        full_pickup_window_end_frames=passing,
        final_physical_pass=bool(passing and passing[-1] == len(trace)-1),
        formal_stage2_pass=False, requires_visual_review=True,
        note='Offline only. Exact replay is not an independent Agent recovery trial; handoff is not success.')
    (session / 'delivery_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['doctor', 'test', 'audit', 'annotate'])
    p.add_argument('--config', required=True)
    p.add_argument('--session', type=Path)
    args = p.parse_args()
    if args.mode == 'doctor':
        doctor(config(args.config))
    elif args.mode == 'test':
        for folder in (VENDOR / 'loop2_stage2_c2_phase_v3', VENDOR / 'loop2_stage2_c2_pregrasp_v4', ROOT / 'tests'):
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(folder), '-p', 'test_*.py', '-v'], check=True)
    elif args.mode == 'audit':
        audit(args.session)
    else:
        if not (args.session / 'CLOSED.json').is_file():
            raise ValueError('close session before video processing')
        subprocess.run([sys.executable, str(VENDOR / 'loop2_stage2_c2_phase_v3/annotate_video.py'), str(args.session)], check=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""H20 delivery CLI; stdlib launcher around unchanged v4.1 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'


def config(path):
    value = json.loads(Path(path).read_text())
    for key in ('repo_root', 'scene', 'client_python', 'bundle', 'server_python',
                'server_deps', 'checkpoint', 'model_config', 'libstdcpp'):
        # A venv Python is often a symlink: resolving it bypasses pyvenv.cfg!
        value[key] = os.path.abspath(os.path.expanduser(value[key]))
    return value


def client_env(c):
    env = os.environ.copy()
    env.update(MUJOCO_GL='osmesa', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=f'/tmp/cosmos-arx-cr-{os.getuid()}',
               PYTHONPATH=os.pathsep.join([c['repo_root'], str(VENDOR / 'loop2_stage2_c2_eef_v2')]),
               LD_PRELOAD=c['libstdcpp'])
    return env


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'package hash mismatch: {name}')
    print(json.dumps({'verified_files': len(manifest['files_sha256']), 'version': manifest['version']}))


def server_command(c, output, gpu, port):
    framework = str(Path(c['bundle']) / 'third_party/cosmos-framework')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), COSMOS_TRAINING='0',
               WAN_VAE_PATH=str(Path(c['bundle']) / 'models/Wan2.2_VAE.pth'),
               EDGE_MODEL_PATH=c['checkpoint'],
               PYTHONPATH=os.pathsep.join([c['server_deps'], framework]))
    cmd = [c['server_python'], '-u', '-m', 'cosmos_framework.scripts.action_policy_server_arx_task7_edge',
           '--checkpoint-path', c['checkpoint'], '--config-file', c['model_config'],
           '--action-normalization', 'auto', '--output-dir', str(output), '--host', '127.0.0.1',
           '--port', str(port), '--seed', '42', '--deterministic-seed', '--action-chunk-size', '32',
           '--served-action-steps', '32', '--num-steps', '4', '--conditioning-fps', '15',
           '--guidance', '1', '--shift', '10', '--sigma-max', '80', '--resolution', '480',
           '--max-state-echo-error', '.05', '--ready-file', str(output / 'ready.json')]
    return cmd, env, framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.h20.json')
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('verify')
    sub.add_parser('doctor')
    sub.add_parser('test')
    server = sub.add_parser('server')
    server.add_argument('--output', type=Path, required=True)
    server.add_argument('--gpu', type=int)
    server.add_argument('--port', type=int)
    server.add_argument('--dry-run', action='store_true')
    for mode in ('live', 'replay-pregrasp', 'smoke'):
        p = sub.add_parser(mode)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--seed', type=int, default=183173)
        p.add_argument('--prefix', type=Path)
        p.add_argument('--port', type=int)
        p.add_argument('--max-steps', type=int, default=1400)
        if mode != 'live':
            p.add_argument('--approval', required=True, help='Explicit approval of recorded-action replay, NOT new Agent decisions')
    for mode in ('audit', 'annotate'):
        p = sub.add_parser(mode)
        p.add_argument('session', type=Path)
    geo = sub.add_parser('geometry')
    geo.add_argument('session', type=Path)
    geo.add_argument('--frame', type=int, required=True)
    geo.add_argument('--temporal-frames', type=int, nargs='+')
    args = parser.parse_args()
    if args.mode == 'verify':
        verify()
        return
    c = config(args.config)
    if args.mode == 'server':
        output = args.output.resolve()
        gpu = c['gpu'] if args.gpu is None else args.gpu
        port = c['port'] if args.port is None else args.port
        cmd, env, cwd = server_command(c, output, gpu, port)
        if args.dry_run:
            print(shlex.join(cmd))
            return
        verify()
        if output.exists():
            raise ValueError('server output already exists; choose a fresh directory')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        memory = {int(a): int(b) for a, b in (line.split(',') for line in inventory.splitlines())}
        if gpu not in memory or memory[gpu] > 1024:
            raise ValueError(f'GPU {gpu} is unavailable or busy; inspect nvidia-smi and select an idle GPU, never kill other jobs')
        # Foreground exec: Ctrl-C stops only this server. Never use broad pkill.
        os.chdir(cwd)
        os.execve(cmd[0], cmd, env)
    if args.mode in ('live', 'replay-pregrasp', 'smoke'):
        verify()
        if not 600 <= args.max_steps <= 1600:
            raise ValueError('max-steps must be 600..1600')
        if args.mode != 'live' and args.seed != 183173:
            raise ValueError('recorded pregrasp fixture is seed183173 only; use live for other seeds')
        prefix = (args.prefix or ROOT / 'fixtures' / f'seed_{args.seed}' / 'original_prefix.npz').resolve()
        if not prefix.is_file():
            raise ValueError('prefix missing; supply the original trajectory.npz with --prefix')
        if args.output.exists():
            raise ValueError('session output already exists; choose a fresh directory')
        cmd = [c['client_python'], '-u', str(ROOT / 'scripts/worker.py'), '--config', str(args.config.resolve()),
               '--mode', args.mode, '--seed', str(args.seed), '--prefix', str(prefix),
               '--output', str(args.output.resolve()), '--port', str(c['port'] if args.port is None else args.port),
               '--max-steps', str(args.max_steps)]
        if args.mode != 'live':
            cmd += ['--approval', args.approval]
        os.execve(cmd[0], cmd, client_env(c))
    if args.mode == 'geometry':
        cmd = [c['client_python'], str(VENDOR / 'loop2_stage2_c2_eef_v2/rgb_geometry.py'),
               '--scene', str(Path(c['scene']) / 'scene.xml'), '--session', str(args.session.resolve()),
               '--frame', str(args.frame)]
        if args.temporal_frames:
            cmd += ['--temporal-frames', *map(str, args.temporal_frames)]
    else:
        cmd = [c['client_python'], str(ROOT / 'scripts/manage.py'), args.mode, '--config', str(args.config.resolve())]
        if hasattr(args, 'session'):
            cmd += ['--session', str(args.session.resolve())]
    subprocess.run(cmd, env=client_env(c), check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

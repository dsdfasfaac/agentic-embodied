#!/usr/bin/env python3
"""Prepare the ten episode scenes and optionally collect seeded VLA rollouts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from queue import Empty, Queue
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/zetta-arx-cache')
import imageio.v2 as imageio
import mujoco
import numpy as np
from robots.arx.environment import ArxMujocoEnv
from robots.arx.mujoco_mapping import MujocoMapping


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def prepare(source, output, eid):
    name=f'episode_{eid:06d}'
    dest=output/name
    xml=source/'episodes'/f'{name}.xml'
    digest=hashlib.sha256(xml.read_bytes()).hexdigest()
    if (dest/'validation.json').exists():
        record=json.loads((dest/'validation.json').read_text())
        if record['source_sha256'] != digest:
            raise ValueError(f'{dest}: source changed; use a fresh output directory')
        return dest
    config=json.loads((source/'runtime_config.json').read_text())
    c=config['controller']
    original=mujoco.MjModel.from_xml_path(str(xml)); initial=mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original,initial,original.key('episode_initial_state').id)
    mujoco.mj_forward(original,initial)
    tree=ET.parse(xml);root=tree.getroot()
    root.find('compiler').set('meshdir','../assets')
    root.find('compiler').set('texturedir','../assets')
    for camera in root.iter('camera'):
        for view in ('front','left','right'):
            if camera.get('name')==view+'_cam':camera.set('name',view+'_rgb')
    for body in root.iter('body'):
        if body.get('name') in ('right_link7','right_link8'):
            body.set('name','right_finger_a' if body.get('name')=='right_link7' else 'right_finger_b')
    # Rename references too (e.g. collision exclusions).
    for node in root.iter():
        for key in ('body1','body2','body'):
            if node.get(key) in ('right_link7','right_link8'):
                node.set(key,'right_finger_a' if node.get(key)=='right_link7' else 'right_finger_b')
    # A failed preparation may have left files but no validation marker.
    dest.mkdir(exist_ok=True)
    tree.write(dest/'scene.xml',encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(dest/'scene.xml'))
    for field in ('body_mass','body_inertia','body_gravcomp','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','geom_contype','geom_conaffinity'):
        np.testing.assert_allclose(getattr(model,field),getattr(original,field),atol=1e-12)
    mapping={'schema_version':2,'joint_names':[],'actuator_names':[],
             'gripper_hardware_ranges':[[-3.4, 0.0]]*2,
             'finger_position_ranges':[[c['finger_closed_m'],c['finger_open_m']]]*2}
    for side in ('left','right'):
        present = [mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_joint{i}') >= 0 for i in range(1, 9)]
        if side == 'left' and not any(present):
            mapping['joint_names'].extend([[] for _ in range(7)])
            mapping['actuator_names'].extend([[] for _ in range(7)])
            continue
        if not all(present):
            raise ValueError(f'{xml}: expected all eight {side} joints, or an entirely absent left arm')
        names=[f'{side}_joint{i}' for i in range(1,7)]+[[f'{side}_joint7',f'{side}_joint8']]
        mapping['joint_names'].extend(names)
        mapping['actuator_names'].extend([n+'_motor' if isinstance(n,str) else [v+'_motor' for v in n] for n in names])
    converter=MujocoMapping.from_mapping(mapping)
    task=json.loads(Path('robots/arx/manifests/pickup_test_tube.yaml').read_text())
    for channel,names in enumerate(converter.joint_names):
        if not names:
            # Keep the manifest's fixed policy state for the absent left arm.
            continue
        value=float(np.mean([initial.qpos[original.jnt_qposadr[original.joint(n).id]] for n in names]))
        task['start_state'][channel]=converter.finger_to_hardware_gripper(channel,value,tolerance=1e-5) if channel in (6,13) else value
    task['starting_scenes']={'selection':'explicit','default_scene_id':name,'allowed_scene_ids':[name]}
    task['control']['gripper_command_offsets']=[0.,0.]
    task['control']['lock_left_arm']=True
    task['success']['parameters'].update(lift_height_m=config['success_proxy']['lift_m'],hold_steps=8)
    # Eight 15-Hz observations approximate the source's 15 holds at 30 Hz.
    write(dest/'task.yaml',task);write(dest/'mapping.json',mapping)
    np.savez(dest/'reset_state.npz',qpos=initial.qpos,qvel=initial.qvel,time=initial.time)
    mujoco.mj_saveModel(model,str(dest/'model.mjb'))
    write(dest/'metadata.json',{'schema_version':'zetta_prepared_scene_v1','scene_id':name,
          'task_name':task['name'],'logical_body_map':{'tube_01':'pink_labelled_target_tube'},
          'preserve_physics_timestep':True,'render_geom_groups':[2],
          'source':str(xml),'source_sha256':digest,'source_mujoco_version':config['mujoco_version'],
          'compiled_mujoco_version':mujoco.__version__})
    env=ArxMujocoEnv(prepared_scene_bundle=str(dest),mapping_path=str(dest/'mapping.json'),task_manifest=str(dest/'task.yaml'),camera_names={n:n for n in ('front_rgb','left_rgb','right_rgb')})
    try:
        obs,info=env.reset(seed=17)
        np.testing.assert_allclose(env.data.qpos,initial.qpos,atol=1e-7)
        np.testing.assert_allclose(env.data.xpos,initial.xpos,atol=1e-7)
        assert np.isfinite(env.data.qacc).all()
        imageio.imwrite(dest/'three_view_frame0.png',np.concatenate([obs[n] for n in ('front_rgb','left_rgb','right_rgb')],axis=1))
        repeat,_=env.reset(seed=17)
        assert all(np.array_equal(obs[k],repeat[k]) for k in obs)
        write(dest/'validation.json',{'source_sha256':digest,'physics_parameters_preserved':True,
              'initial_state_verified':True,'repeatable_reset':True,'physics_timestep':env.model.opt.timestep,
              'physics_steps_per_command':env.physics_steps_per_command,'reset_info':info})
    finally:env.close()
    return dest


def collect(scenes, args, root, output):
    # Each worker owns a server for the entire batch: seeds must not compete
    # with another scene's requests on the same inference server.
    gpus = args.gpus or [None]
    ports = args.ports or [args.port + index for index in range(len(gpus))]
    pending = Queue()
    for record in scenes:
        pending.put(record)

    def worker(gpu, port):
        environment = dict(os.environ)
        if gpu is not None:
            environment['CUDA_VISIBLE_DEVICES'] = gpu
            # EGL sees the selected GPU as device zero inside this process.
            environment['MUJOCO_EGL_DEVICE_ID'] = '0'
        failures = []
        while True:
            try:
                record = pending.get_nowait()
            except Empty:
                return failures
            episode = record['episode']
            print(f'Collecting episode {episode}: GPU={gpu}, server={args.host}:{port}', flush=True)
            cmd = [sys.executable, str(root/'scripts/deployment/run_arx_seed_rollouts.py')]
            for key in ('scene', 'mapping', 'task'):
                cmd += ['--'+key, record[key]]
            for key in ('num_seeds', 'seed_selection', 'chunks', 'host'):
                cmd += ['--'+key.replace('_', '-'), str(getattr(args, key))]
            cmd += ['--port', str(port), '--output', str(output/'trajectories'/f'episode_{episode:06d}')]
            if args.offline_test:
                cmd.append('--offline-test')
            try:
                result = subprocess.run(cmd, cwd=root, env=environment)
                if result.returncode:
                    failures.append(episode)
            except OSError as error:
                print(f'Failed to launch episode {episode}: {error}', file=sys.stderr, flush=True)
                failures.append(episode)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, port) for gpu, port in zip(gpus, ports)]
        failures = sorted(episode for future in futures for episode in future.result())
    if failures:
        raise SystemExit(f'Failed rollout batches: {failures}; inspect per-batch summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('pickup_10_episode_arx_physics'))
    parser.add_argument('--output',type=Path,default=Path('runs/arx_pickup_test_tube_10_new'))
    parser.add_argument('--collect',action='store_true',help='collect trajectories after preparation')
    parser.add_argument('--num-seeds',type=int,default=10)
    parser.add_argument('--seed-selection',type=int,default=17)
    parser.add_argument('--chunks',type=int,default=32)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=5581)
    parser.add_argument('--gpus', nargs='+', help='GPU IDs or UUIDs, one worker per GPU (e.g. --gpus 0 1). Start one inference server per GPU separately.')
    parser.add_argument('--ports', nargs='+', type=int, help='server ports in GPU order; defaults to --port plus GPU index')
    parser.add_argument('--offline-test',action='store_true')
    args=parser.parse_args()
    if args.gpus and (len(set(args.gpus)) != len(args.gpus) or any(not gpu.strip() or ',' in gpu for gpu in args.gpus)):
        parser.error('--gpus must contain distinct, nonempty GPU IDs or UUIDs separated by spaces')
    ports = args.ports or [args.port + index for index in range(len(args.gpus or [None]))]
    if len(ports) != len(args.gpus or [None]):
        parser.error('--ports must contain one port per GPU (or one port without --gpus)')
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        parser.error('server ports must be distinct and in [1, 65535]')
    root=Path(__file__).resolve().parents[2];os.chdir(root)
    source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if not (output/'assets').exists():shutil.copytree(source/'scene/assets',output/'assets')
    scenes=[]
    for eid in range(10):
        print(f'Preparing episode {eid}',flush=True)
        dest=prepare(source,output,eid)
        scenes.append({'episode':eid,'scene':str(dest),'mapping':str(dest/'mapping.json'),'task':str(dest/'task.yaml')})
        write(output/'scenes.json',scenes)
    if args.collect:
        collect(scenes, args, root, output)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Package only the reconstructed PickUpTestTube initial physics state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np

from robots.arx.environment import ArxMujocoEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('scenes/PickUpTestTube/mujoco_physics'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=Path, default=Path('robots/arx/manifests/pickup_test_tube.yaml'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(source / 'model.xml')
    root = tree.getroot()
    with np.load(source / 'dynamics.npz') as saved:
        qpos, qvel = saved['qpos'][0].copy(), saved['qvel'][0].copy()
        camera_pos, camera_quat = saved['mocap_pos'][0], saved['mocap_quat'][0]
    original = mujoco.MjModel.from_xml_path(str(source / 'model.xml'))
    for body in root.iter('body'):
        if body.get('mocap') == 'true':
            index = original.body_mocapid[original.body(body.get('name')).id]
            body.attrib.pop('mocap')
            body.set('pos', ' '.join(map(str, camera_pos[index])))
            body.set('quat', ' '.join(map(str, camera_quat[index])))
        if body.get('name') in ('right_link7', 'right_link8'):
            body.set('name', 'right_finger_a' if body.get('name') == 'right_link7' else 'right_finger_b')
    for camera in root.iter('camera'):
        camera.set('name', camera.get('name') + '_rgb')
    # Retain source joint coordinates, dynamics, collision geometry and limits.
    # These are an X5A reconstruction, not the AC-one XML's local frames.
    shutil.copytree(source / 'assets', output / 'assets')
    tree.write(output / 'scene.xml', encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output / 'scene.xml'))
    assert model.nq == len(qpos) and model.nv == len(qvel) and model.nmocap == 0
    mapping = {
        'schema_version': 2,
        'joint_names': [[] for _ in range(7)] + [f'right_joint{i}' for i in range(1, 7)] + [['right_joint7', 'right_joint8']],
        'actuator_names': [[] for _ in range(7)] + [f'motor_right_joint{i}' for i in range(1, 7)] + [['motor_right_joint7', 'motor_right_joint8']],
        'gripper_hardware_ranges': [[-3.4, 0], [-3.4, 0]],
        'finger_position_ranges': [[0, 0.044], [0, 0.044]],
    }
    task = json.loads(args.task.read_text())
    task['starting_scenes'] = {'selection': 'explicit', 'default_scene_id': 'PickUpTestTube', 'allowed_scene_ids': ['PickUpTestTube']}
    task['control']['lock_left_arm'] = True
    task['control']['zero_left_model_state'] = True
    for i in range(6):
        task['start_state'][7+i] = float(qpos[model.jnt_qposadr[model.joint(f'right_joint{i+1}').id]])
    finger_addresses = [model.jnt_qposadr[model.joint(f'right_joint{i}').id] for i in (7, 8)]
    finger = float(np.mean(qpos[finger_addresses]))
    assert 0 <= finger <= 0.044
    task['start_state'][13] = -3.4 * finger / 0.044
    # The interface commands a coupled pair. Remove nanometre-scale asymmetry.
    qpos[finger_addresses] = finger
    np.savez(output / 'reset_state.npz', qpos=qpos, qvel=qvel, time=0.0)
    mujoco.mj_saveModel(model, str(output / 'model.mjb'))
    metadata = {
        'schema_version': 'zetta_prepared_scene_v1', 'scene_id': 'PickUpTestTube',
        'task_name': task['name'], 'scene_xml': 'scene.xml',
        'logical_body_map': {'tube_01': 'prop_pink', 'tube_green': 'prop_green', 'tube_blue': 'prop_blue',
                             'tube_small_1': 'prop_small_1', 'tube_small_2': 'prop_small_2',
                             'tube_small_5': 'prop_small_5', 'tube_stand_01': 'prop_rack', 'beaker': 'prop_beaker'},
        'reset_source': 'dynamics.npz frame 0; residual velocities retained',
        'source': str(source), 'mujoco_version': mujoco.__version__,
        'source_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ('model.xml', 'dynamics.npz')},
        'limitations': ['Left policy channels are fixed task-state placeholders; left geometry remains static.',
                       'Source X5A joint coordinates retained; hardware policy calibration is not certified.',
                       'Hardware gripper conversion retains the existing 44 mm command span; source 49 mm stops retained.',
                       'Upper-arm and self collisions remain excluded as in source.',
                       'No trajectory controls or later frames are included.'],
    }
    for name, value in [('mapping.json', mapping), ('task.yaml', task), ('metadata.json', metadata)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    env = ArxMujocoEnv(prepared_scene_bundle=str(output), mapping_path=str(output/'mapping.json'),
                       task_manifest=str(output/'task.yaml'),
                       camera_names={name:name for name in ('front_rgb','left_rgb','right_rgb')})
    try:
        obs, info = env.reset(seed=17)
        initial_qpos = env.data.qpos.copy()
        assert np.max(np.abs(initial_qpos-qpos)) < 1e-7
        assert np.isfinite(env.data.qacc).all()
        assert not info['evaluation']['failure']
        for name in ('front_rgb','left_rgb','right_rgb'):
            imageio.imwrite(output/f'{name}_frame0.png', obs[name])
        imageio.imwrite(output/'three_view_frame0.png', np.concatenate([obs[n] for n in ('front_rgb','left_rgb','right_rgb')], axis=1))
        obs2, _ = env.reset(seed=17)
        assert all(np.array_equal(obs[k], obs2[k]) for k in obs)
        audit = {'reset_repeatable': True, 'state': obs['state'].tolist(), 'nq': model.nq,
                 'nv': model.nv, 'nu': model.nu, 'nmocap': model.nmocap,
                 'initial_contacts': env.data.ncon,
                 'max_penetration_m': max([0.0]+[-float(c.dist) for c in env.data.contact]),
                 'reset_info': info, 'executed_policy_steps': 0}
        (output/'validation.json').write_text(json.dumps(audit, indent=2)+'\n')
        print(json.dumps(audit, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()

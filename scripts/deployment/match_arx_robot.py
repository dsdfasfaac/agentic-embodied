"""Transplant the reference robot into the aligned scene, retaining its objects."""
import copy
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np

from robots.arx.environment import ArxMujocoEnv


def main():
    parent = Path('runs/arx_pickup_test_tube')
    source = parent/'dark_silver_aligned_scene'
    reference = parent/'pickup_test_tube_initial_scene'
    output = parent/'dark_silver_aligned_new_robot_scene'
    mapping = '../Zeva_arx/assets/ac_one/ac_one_14d_mapping.json'
    task = 'robots/arx/manifests/pickup_test_tube.yaml'
    old_env = ArxMujocoEnv(prepared_scene_bundle=str(source), mapping_path=mapping, task_manifest=task)
    new_env = ArxMujocoEnv(prepared_scene_bundle=str(reference), mapping_path=str(reference/'mapping.json'), task_manifest=str(reference/'task.yaml'))
    old_env.reset(); new_env.reset()
    om, od, nm, nd = old_env.model, old_env.data, new_env.model, new_env.data
    R = od.body('base_link').xmat.reshape(3, 3).copy()
    offset = od.body('right_link11').xpos - R@nd.body('right_link1').xpos
    tree = ET.parse(source/'scene.xml'); root = tree.getroot()
    ref = ET.parse(reference/'scene.xml').getroot()
    world = root.find('worldbody')
    old_robot = world.find("body[@name='base_link']")
    cameras = []
    for camera in old_robot.iter('camera'):
        cid = om.camera(camera.get('name')).id
        cameras.append((copy.deepcopy(camera), od.cam_xpos[cid].copy(), od.cam_xmat[cid].reshape(3,3).copy()))
    world.remove(old_robot)
    for tag in ('actuator','equality','contact'):
        element = root.find(tag)
        if element is not None: root.remove(element)
    robot = copy.deepcopy(ref.find("./worldbody/body[@name='right_base_link']"))
    for node in robot.iter('body'):
        for camera in list(node.findall('camera')): node.remove(camera)
    def fmt(v): return ' '.join(map(str,v))
    def pose(node, pos, rot):
        q = np.empty(4); mujoco.mju_mat2Quat(q, rot.ravel())
        node.set('pos',fmt(pos));node.set('quat',fmt(q))
    pose(robot, offset, R)
    world.append(robot)
    static = []
    for name in ('static_object_0031','static_object_0032','static_object_0033','static_object_0034'):
        node = copy.deepcopy(ref.find(f"./worldbody/body[@name='{name}']"))
        pose(node, R@nd.body(name).xpos+offset, R@nd.body(name).xmat.reshape(3,3))
        world.append(node);static.append(node)
    # Materialize reference geom defaults only on imported robot geoms.
    defaults = ref.find('./default/geom').attrib
    for node in [robot]+static:
        for geom in node.iter('geom'):
            for key,value in defaults.items(): geom.attrib.setdefault(key,value)
            # Reference inertiafromgeom=false makes visual/collision geometry
            # massless. Preserve that locally without disabling object inertia
            # inference in the destination scene.
            geom.set('mass', '0')
    asset = root.find('asset')
    needed = {e.get(k) for node in [robot]+static for e in node.iter() for k in ('mesh','material') if e.get(k)}
    shutil.copytree(source,output)
    shutil.copytree(reference/'assets',output/'assets/reference_robot')
    for item in ref.find('asset'):
        if item.get('name') in needed:
            item=copy.deepcopy(item)
            if item.get('file'): item.set('file','assets/reference_robot/'+item.get('file').removeprefix('assets/'))
            asset.append(item)
    root.append(copy.deepcopy(ref.find('actuator')))
    option=root.find('option');option.attrib.clear();option.attrib.update(ref.find('option').attrib)
    # Keep camera calibration from the aligned old scene for this experiment.
    wrist_pos=R@nd.body('right_link6').xpos+offset
    wrist_rot=R@nd.body('right_link6').xmat.reshape(3,3)
    for camera,pos,rot in cameras:
        if camera.get('name')=='right_rgb':
            pose(camera,wrist_rot.T@(pos-wrist_pos),wrist_rot.T@rot)
            robot.find(".//body[@name='right_link6']").append(camera)
        else:
            pose(camera,pos,rot);world.append(camera)
    tree.write(output/'scene.xml',encoding='utf-8',xml_declaration=True)
    m=mujoco.MjModel.from_xml_path(str(output/'scene.xml'));d=mujoco.MjData(m)
    for i in range(m.njnt):
        name=m.joint(i).name
        reference_joint=mujoco.mj_name2id(nm,mujoco.mjtObj.mjOBJ_JOINT,name) if name else -1
        # Old unnamed tube free joints are matched by owning body name.
        if reference_joint>=0:
            sm,sd,j=nm,nd,reference_joint
        else:
            sm,sd=om,od
            body=om.body(m.body(m.jnt_bodyid[i]).name).id
            j=om.body_jntadr[body]
        nq,nv=(7,6) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE else (1,1)
        a,b=m.jnt_qposadr[i],sm.jnt_qposadr[j];d.qpos[a:a+nq]=sd.qpos[b:b+nq]
        a,b=m.jnt_dofadr[i],sm.jnt_dofadr[j];d.qvel[a:a+nv]=sd.qvel[b:b+nv]
    np.savez(output/'reset_state.npz',qpos=d.qpos,qvel=d.qvel,time=0.)
    mujoco.mj_saveModel(m,str(output/'model.mjb'))
    for name in ('mapping.json','task.yaml'): shutil.copy2(reference/name,output/name)
    task_config=json.loads((output/'task.yaml').read_text())
    task_config['starting_scenes']={'selection':'explicit','default_scene_id':output.name,'allowed_scene_ids':[output.name]}
    (output/'task.yaml').write_text(json.dumps(task_config,indent=2)+'\n')
    meta=json.loads((source/'metadata.json').read_text())
    for key in ('robot_source_archive','robot_source_archive_sha256','bundle_digest'):
        meta.pop(key,None)
    meta.update(scene_id=output.name,robot_reference=str(reference),reset_source='aligned object state plus reference robot initial state')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    env=ArxMujocoEnv(prepared_scene_bundle=str(output),mapping_path=str(output/'mapping.json'),task_manifest=str(output/'task.yaml'),camera_names={n:n for n in ('front_rgb','left_rgb','right_rgb')})
    obs,info=env.reset()
    for node in [robot]+static:
        for body in node.iter('body'):
            name=body.get('name')
            a,b=env.model.body(name).id,nm.body(name).id
            for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_gravcomp'):
                if nm.body_mass[b] == 0 and field in ('body_ipos','body_iquat'):
                    continue  # Inertial frames of massless bodies have no dynamics.
                np.testing.assert_allclose(getattr(env.model,field)[a],getattr(nm,field)[b],atol=1e-12)
    np.testing.assert_allclose(env.model.body_subtreemass[env.model.body('right_base_link').id],
                               nm.body_subtreemass[nm.body('right_base_link').id],atol=1e-12)
    for i in range(1,9):
        name=f'right_joint{i}';a=env.model.joint(name).id;b=nm.joint(name).id
        for field in ('jnt_type','jnt_axis','jnt_range'):
            np.testing.assert_allclose(getattr(env.model,field)[a],getattr(nm,field)[b])
        da,db=env.model.jnt_dofadr[a],nm.jnt_dofadr[b]
        for field in ('dof_damping','dof_armature','dof_frictionloss'):
            np.testing.assert_allclose(getattr(env.model,field)[da],getattr(nm,field)[db])
        ba,bb=env.model.jnt_bodyid[a],nm.jnt_bodyid[b]
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_gravcomp'):
            np.testing.assert_allclose(getattr(env.model,field)[ba],getattr(nm,field)[bb])
    for field in ('actuator_gainprm','actuator_biasprm','actuator_forcerange'):
        np.testing.assert_allclose(getattr(env.model,field),getattr(nm,field))
    for name in meta['logical_body_map'].values():
        np.testing.assert_allclose(env.data.body(name).xpos,od.body(name).xpos,atol=1e-10)
    for camera,pos,rot in cameras:
        cid=env.model.camera(camera.get('name')).id
        np.testing.assert_allclose(env.data.cam_xpos[cid],pos,atol=1e-7)
        np.testing.assert_allclose(env.data.cam_xmat[cid].reshape(3,3),rot,atol=1e-7)
    imageio.imwrite(output/'three_view_frame0.png',np.concatenate([obs[n] for n in ('front_rgb','left_rgb','right_rgb')],axis=1))
    repeated,_=env.reset();assert all(np.array_equal(obs[k],repeated[k]) for k in obs)
    start=env.data.body(meta['logical_body_map']['tube_01']).xpos.copy()
    mujoco.mj_step(env.model,env.data,300)
    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
    report={'robot_parameters_match':True,'object_poses_preserved':True,'initial_camera_poses_preserved':True,
            'reset_repeatable':True,'stationary_check_seconds':.5,
            'target_displacement_m':(env.data.body(meta['logical_body_map']['tube_01']).xpos-start).tolist(),
            'nq':m.nq,'nv':m.nv,'nu':m.nu,'neq':m.neq,
            'source':str(source),'robot_reference':str(reference),
            'note':'New right robot and static left representation; old object physics and camera calibration retained. Global solver options match reference.'}
    (output/'robot_alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    env.close();old_env.close();new_env.close()


if __name__=='__main__': main()

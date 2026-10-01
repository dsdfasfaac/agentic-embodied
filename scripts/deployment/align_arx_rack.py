"""Uniformly align the old rack assembly to the reconstructed rack footprint."""
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import imageio.v2 as imageio
from robots.arx.environment import ArxMujocoEnv


def points(m, d, prefix):
    result = []
    for i in range(m.ngeom):
        if not m.geom(i).name.startswith(prefix) or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # Use visible meshes only, excluding convex collision decompositions.
        if m.geom_contype[i] or m.geom_conaffinity[i]:
            continue
        mesh = m.geom_dataid[i]
        v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        result.append(v @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i])
    return np.concatenate(result)


def main():
    parent = Path('runs/arx_pickup_test_tube')
    source = parent/'dark_silver_estimated_top_scene_physics_fixed'
    reference = parent/'pickup_test_tube_initial_scene'
    output = parent/'dark_silver_aligned_scene'
    old = mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
    new = mujoco.MjModel.from_binary_path(str(reference/'model.mjb'))
    od, nd = mujoco.MjData(old), mujoco.MjData(new)
    for root, m, d in [(source, old, od), (reference, new, nd)]:
        with np.load(root/'reset_state.npz') as a:
            d.qpos[:] = a['qpos']; d.qvel[:] = a['qvel']
        mujoco.mj_forward(m, d)
    a = points(old, od, 'tube_stand_01__')
    b = points(new, nd, 'mesh_0035_0')
    # Match right shoulder origins and base axes, not unrelated world origins.
    R = od.body('base_link').xmat.reshape(3, 3)
    origin = od.body('right_link11').xpos - R @ nd.body('right_link1').xpos
    b = b @ R.T + origin
    def frame(v):
        center = (v.min(0)+v.max(0))/2
        _, vec = np.linalg.eigh(np.cov(v[:, :2].T))
        axis = vec[:, -1]
        if axis[0] < 0: axis = -axis
        angle = np.arctan2(axis[1], axis[0])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        local = v @ rot
        return rot, (local.max(0)+local.min(0))/2 @ rot.T, np.ptp(local,axis=0)
    A, ac, ae = frame(a); B, bc, be = frame(b)
    scale = float(np.dot(ae,be)/np.dot(ae,ae))
    Q = B @ A.T
    translation = bc - scale*Q@ac
    # Preserve support on the old table rather than placing the group in air.
    zmin = float(a[:,2].min())
    translation[2] = zmin-scale*zmin
    tree = ET.parse(source/'scene.xml')
    root = tree.getroot()
    meshes = set()
    names = []
    def fmt(x): return ' '.join(map(str,x))
    for body in root.findall('./worldbody/body'):
        name = body.get('name','')
        if not name.startswith(('tube_stand_01__','tube_01__','tube_02__','tube_03__')): continue
        names.append(name)
        bid = old.body(name).id
        pos = scale*Q@od.xpos[bid]+translation
        rot = Q@od.xmat[bid].reshape(3,3)
        quat = np.empty(4); mujoco.mju_mat2Quat(quat,rot.ravel())
        body.set('pos',fmt(pos));body.set('quat',fmt(quat))
        for element in body.iter():
            if element is not body and element.get('pos'): element.set('pos',fmt(np.fromstring(element.get('pos'),sep=' ')*scale))
            if element.tag == 'geom':
                if element.get('mesh'): meshes.add(element.get('mesh'))
                elif element.get('size'): element.set('size',fmt(np.fromstring(element.get('size'),sep=' ')*scale))
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('name') in meshes:
            mesh.set('scale',fmt(np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*scale))
    shutil.copytree(source,output)
    tree.write(output/'scene.xml',encoding='utf-8',xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    data = mujoco.MjData(model)
    # Object XML poses now encode the transformed settled state. Robot state
    # and runtime initialization remain the original task's responsibility.
    data.qpos[:] = od.qpos
    for name in names:
        bid = model.body(name).id
        if model.body_jntnum[bid]:
            adr = model.jnt_qposadr[model.body_jntadr[bid]]
            data.qpos[adr:adr+7] = model.qpos0[adr:adr+7]
    data.qvel[:] = od.qvel
    for name in names:
        bid = model.body(name).id
        if model.body_jntnum[bid]:
            adr=model.jnt_dofadr[model.body_jntadr[bid]]
            data.qvel[adr:adr+3]=scale*Q@od.qvel[adr:adr+3]
    np.savez(output/'reset_state.npz',qpos=data.qpos,qvel=data.qvel,time=0.)
    mujoco.mj_saveModel(model,str(output/'model.mjb'))
    report={'scale':scale,'old_rack_dimensions':ae.tolist(),'reference_rack_dimensions':be.tolist(),
            'scaled_rack_dimensions':(ae*scale).tolist(),'rotation':Q.tolist(),'translation':translation.tolist(),
            'vertical_alignment':'old tabletop preserved','source':str(source),'reference':str(reference)}
    (output/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    env=ArxMujocoEnv(prepared_scene_bundle=str(output),mapping_path='../Zeva_arx/assets/ac_one/ac_one_14d_mapping.json',task_manifest='robots/arx/manifests/pickup_test_tube.yaml',camera_names={n:n for n in ['front_rgb','left_rgb','right_rgb']})
    obs,info=env.reset()
    imageio.imwrite(output/'three_view_frame0.png',np.concatenate([obs[n] for n in ['front_rgb','left_rgb','right_rgb']],axis=1))
    assert np.isfinite(env.data.qacc).all()
    print(json.dumps(report,indent=2));print(info)
    env.close()


if __name__=='__main__': main()

"""Read-only RGB triangulation proposal from calibrated cameras + own commands.

No simulator state is loaded. A ray fit is evidence with uncertainty, not an
execution request. The Agent chooses any subsequent move_eef parameters.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import numpy as np
from early_critic import pink_mask
from eef_tools import CommandKinematics, quatmat


def estimate(scene, session, frame):
    k=CommandKinematics.from_scene_xml(scene)
    command=None
    for line in (session/'actions.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['frame']==frame: command=np.asarray(row['command_prediction'])
    if command is None: raise ValueError('no matching own command history')
    tcp,tool_r,_=k.fk(command[7:13]); link=tcp-tool_r@k.tcp
    root=ET.parse(scene).getroot(); parents={c:p for p in root.iter() for c in p}
    rays=[]; evidence=[]
    for cam in root.iter('camera'):
        name=cam.get('name')
        if name not in ('front_rgb','left_rgb','right_rgb'): continue
        im=imageio.imread(session/f'frame_{frame:04d}'/f'{name}.png')
        ys,xs=np.nonzero(pink_mask(im))
        if len(xs)<20: continue
        # An image-border truncation makes the centroid unreliable.
        if xs.min()==0 or xs.max()==im.shape[1]-1 or ys.min()==0 or ys.max()==im.shape[0]-1:
            continue
        cp=np.array(list(map(float,cam.get('pos','0 0 0').split())))
        cr=quatmat(list(map(float,cam.get('quat','1 0 0 0').split())))
        if name=='right_rgb': cp,cr=link+tool_r@cp,tool_r@cr
        else:
            parent=parents[cam]
            if parents[parent].tag!='worldbody' or parent.find('joint') is not None:
                raise ValueError('fixed camera calibration expected')
            pr=quatmat(list(map(float,parent.get('quat','1 0 0 0').split())))
            cp=np.array(list(map(float,parent.get('pos','0 0 0').split())))+pr@cp;cr=pr@cr
        fx,fy=map(float,cam.get('focalpixel').split())
        u,v=float(xs.mean()),float(ys.mean())
        ray=cr@np.array([(u-im.shape[1]/2)/fx,-(v-im.shape[0]/2)/fy,-1.])
        ray/=np.linalg.norm(ray); rays.append((cp,ray))
        evidence.append({'camera':name,'pink_pixels':len(xs),'pixel_uv':[u,v]})
    if len(rays)<2: raise ValueError('fewer than two untruncated RGB detections; depth unknown')
    proj=[np.eye(3)-np.outer(d,d) for _,d in rays]
    system=sum(proj)
    if np.linalg.cond(system)>1000: raise ValueError('triangulation ill-conditioned')
    point=np.linalg.solve(system,sum(p@o for p,(o,d) in zip(proj,rays)))
    residual=max(float(np.linalg.norm(p@(point-o))) for p,(o,d) in zip(proj,rays))
    if residual>.008 or any(np.dot(point-o,d)<=0 for o,d in rays):
        raise ValueError(f'RGB rays inconsistent: residual={residual}')
    delta=point-tcp
    return {'source':'RGB + static calibration + own command history only',
            'frame':frame,'evidence':evidence,'ray_residual_m':residual,
            'rgb_band_to_commanded_tcp_world_m':delta.tolist(),
            'rgb_band_to_commanded_tcp_tool_m':(tool_r.T@delta).tolist(),
            'limitations':'partial occlusion, commanded-pose tracking error, band-center versus pad-contact geometry',
            'proposal_only':True}


def check_historical_projection(scene,session,frame,reference_frame,reference_session=None):
    reference_session=reference_session or session
    ref=estimate(scene,reference_session,reference_frame)
    commands={r['frame']:np.asarray(r['command_prediction']) for r in
              map(json.loads,(session/'actions.jsonl').read_text().splitlines())}
    k=CommandKinematics.from_scene_xml(scene)
    reference_commands={r['frame']:np.asarray(r['command_prediction']) for r in
                        map(json.loads,(reference_session/'actions.jsonl').read_text().splitlines())}
    ref_tcp,_,_=k.fk(reference_commands[reference_frame][7:13])
    point=ref_tcp+np.asarray(ref['rgb_band_to_commanded_tcp_world_m'])
    tcp,r,_=k.fk(commands[frame][7:13]);link=tcp-r@k.tcp
    cam=next(c for c in ET.parse(scene).getroot().iter('camera') if c.get('name')=='right_rgb')
    cp=link+r@np.array(list(map(float,cam.get('pos').split())))
    cr=r@quatmat(list(map(float,cam.get('quat').split())))
    v=cr.T@(point-cp)
    if v[2]>=0: raise ValueError('historical estimate is behind the camera')
    fx,fy=map(float,cam.get('focalpixel').split())
    im=imageio.imread(session/f'frame_{frame:04d}'/'right_rgb.png')
    ys,xs=np.nonzero(pink_mask(im))
    if len(xs)<20: raise ValueError('current target not visible')
    predicted=np.array([im.shape[1]/2+fx*v[0]/(-v[2]),im.shape[0]/2-fy*v[1]/(-v[2])])
    actual=np.array([xs.mean(),ys.mean()]);error=float(np.linalg.norm(actual-predicted))
    clipped=bool(xs.min()==0 or xs.max()==im.shape[1]-1 or ys.min()==0 or ys.max()==im.shape[0]-1)
    return {'status':'historical_projection_consistent' if error<=8 and not clipped else 'unknown',
            'frame':frame,'reference_frame':reference_frame,'current_wrist_pixel_uv':actual.tolist(),
            'reference_session':str(reference_session),
            'current_band_truncated':clipped,
            'historical_wrist_projection_uv':predicted.tolist(),'pixel_residual':error,
            'remaining_tool_delta_from_historical_rgb_m':(r.T@(point-tcp)).tolist(),
            'proposal_only':True,'limitation':'not a new depth measurement; assumes target stationary since reference; wrist reprojection only checks consistency'}


def estimate_temporal(scene,session,frames):
    if len(set(frames))<2: raise ValueError('at least two distinct RGB frames required')
    commands={r['frame']:np.asarray(r['command_prediction']) for r in
              map(json.loads,(session/'actions.jsonl').read_text().splitlines())}
    k=CommandKinematics.from_scene_xml(scene)
    cam=next(c for c in ET.parse(scene).getroot().iter('camera') if c.get('name')=='right_rgb')
    cp_local=np.array(list(map(float,cam.get('pos').split())))
    cr_local=quatmat(list(map(float,cam.get('quat').split())))
    fx,fy=map(float,cam.get('focalpixel').split());rays=[];observations=[]
    for frame in frames:
        tcp,r,_=k.fk(commands[frame][7:13]);cp=tcp-r@k.tcp+r@cp_local;cr=r@cr_local
        im=imageio.imread(session/f'frame_{frame:04d}'/'right_rgb.png')
        ys,xs=np.nonzero(pink_mask(im))
        if len(xs)<20 or xs.min()==0 or xs.max()==im.shape[1]-1 or ys.min()==0 or ys.max()==im.shape[0]-1:
            raise ValueError(f'target absent or clipped at frame {frame}')
        u,v=float(xs.mean()),float(ys.mean())
        d=cr@np.array([(u-im.shape[1]/2)/fx,-(v-im.shape[0]/2)/fy,-1.]);d/=np.linalg.norm(d)
        rays.append((cp,d));observations.append({'frame':frame,'pixel_uv':[u,v],'pink_pixels':len(xs)})
    baseline=max(float(np.linalg.norm(a[0]-b[0])) for a in rays for b in rays)
    if baseline<.018: raise ValueError('active RGB baseline below 18 mm; depth poorly constrained')
    proj=[np.eye(3)-np.outer(d,d) for _,d in rays];system=sum(proj)
    if np.linalg.cond(system)>1000: raise ValueError('active RGB rays ill-conditioned')
    point=np.linalg.solve(system,sum(p@o for p,(o,d) in zip(proj,rays)))
    residual=max(float(np.linalg.norm(p@(point-o))) for p,(o,d) in zip(proj,rays))
    if residual>.005 or any(np.dot(point-o,d)<=0 for o,d in rays): raise ValueError('active RGB rays inconsistent')
    tcp,r,_=k.fk(commands[frames[-1]][7:13])
    return {'source':'temporal wrist RGB + static calibration + own commands; no measured state',
            'frame':frames[-1],'rgb_observations':observations,'baseline_m':baseline,'ray_residual_m':residual,
            'rgb_band_to_commanded_tcp_tool_m':(r.T@(point-tcp)).tolist(),
            'rgb_band_to_commanded_tcp_world_m':(point-tcp).tolist(),'proposal_only':True,
            'limitation':'assumes target stationary between views; ray fit is not a bound on depth error'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--frame',type=int,required=True)
    p.add_argument('--reference-frame',type=int)
    p.add_argument('--reference-session',type=Path)
    p.add_argument('--temporal-frames',type=int,nargs='+')
    a=p.parse_args()
    try:
        result=(estimate_temporal(a.scene,a.session,a.temporal_frames) if a.temporal_frames else
                estimate(a.scene,a.session,a.frame) if a.reference_frame is None else
                check_historical_projection(a.scene,a.session,a.frame,a.reference_frame,a.reference_session))
        print(json.dumps(result,indent=2))
    except ValueError as e: print(json.dumps({'status':'unknown','reason':str(e),'proposal_only':True}))

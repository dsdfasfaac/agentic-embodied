"""Scene-specific pregrasp evidence and one-frame Agent approval gate.

No action selection, simulator access, or success claim. Metric estimates are
RGB ray fits relative to commanded FK, NOT ground truth or collision clearance.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_eef_v2'))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_phase_v3'))
from early_critic import pink_mask, _yellow_mask
from phase_critic import RightFingerTracker


def detection(im,mask_fn):
    ys,xs=np.nonzero(mask_fn(im))
    if len(xs)<20:return None
    return {'pixels':len(xs),'uv':[float(xs.mean()),float(ys.mean())],
            'border':bool(xs.min()==0 or ys.min()==0 or xs.max()==im.shape[1]-1 or ys.max()==im.shape[0]-1)}


def assess(history,geometry):
    """Pure evidence aggregation; returned reasons are never bypassable by prose."""
    reasons=[]
    if len(history)<6:reasons.append('need_six_current_stationary_rgb_frames')
    recent=list(history)[-6:]
    if not recent:return {'eligible':False,'reasons':reasons}
    if any(b['frame']!=a['frame']+1 for a,b in zip(recent,recent[1:])):
        reasons.append('nonconsecutive_visual_history')
    if not all(r['holding'] for r in recent):reasons.append('six_frames_must_be_agent_approved_hold')
    if not all(r['jaw_open'] for r in recent):reasons.append('open_jaw_not_visually_verified')
    if not all(r['command_open'] for r in recent):reasons.append('gripper_open_command_required')
    if not all(r['wrist'] and not r['wrist']['border'] for r in recent):
        reasons.append('wrist_target_missing_or_clipped')
    fixed_ok=[]
    for name in ('front_rgb','left_rgb'):
        ok=all(r['fixed'][name] is not None and not r['fixed'][name]['border'] and
               r['fixed'][name]['fixture_error_px']<=12 for r in recent)
        if ok:
            pts=np.array([r['fixed'][name]['relative_uv'] for r in recent])
            ok=np.max(np.linalg.norm(pts-pts[-1],axis=1))<=2
        if ok:fixed_ok.append(name)
    temporal=bool(geometry and geometry.get('measurement_mode')=='temporal_rgb')
    rack_stable=False
    for name in ('front_rgb','left_rgb'):
        points=[r.get('rack_fixed',{}).get(name) for r in recent]
        if all(p is not None for p in points):
            pts=np.array(points)
            rack_stable|=bool(np.max(np.linalg.norm(pts-pts[-1],axis=1))<=2)
    if not fixed_ok:
        if not temporal:reasons.append('no_stable_unclipped_fixed_view_of_supported_target')
        elif not rack_stable or not all(r.get('wrist_support_cue',False) for r in recent):
            reasons.append('temporal_mode_requires_stable_rack_and_current_wrist_support_cue')
    if all(r['wrist'] for r in recent):
        pts=np.array([r['wrist']['uv'] for r in recent])
        if np.max(np.linalg.norm(pts-pts[-1],axis=1))>2:reasons.append('wrist_target_not_settled')
    if not geometry or geometry.get('status')=='unknown':
        reasons.append('fresh_multiview_depth_unknown')
    else:
        d=np.asarray(geometry['rgb_band_to_commanded_tcp_tool_m'])
        if not np.isfinite(d).all() or d.shape!=(3,):reasons.append('invalid_metric_estimate')
        else:
            if not .015<=d[0]<=.035:reasons.append('approach_standoff_not_15_to_35mm')
            if abs(d[1])>.006:reasons.append('lateral_alignment_error_over_6mm')
            if abs(d[2])>.008:reasons.append('height_alignment_error_over_8mm')
        residual=geometry.get('ray_residual_m',float('inf'))
        if not np.isfinite(residual) or residual>.003:reasons.append('ray_fit_residual_over_3mm')
        if geometry.get('frame')!=recent[-1]['frame']:reasons.append('stale_geometry')
        if temporal:
            baseline=geometry.get('baseline_m',0)
            if not np.isfinite(baseline) or baseline<.025:reasons.append('temporal_baseline_below_25mm')
            if len(geometry.get('rgb_observations',[]))<5:reasons.append('five_temporal_observations_required')
            if residual>.002:reasons.append('temporal_ray_residual_over_2mm')
            disagreement=geometry.get('split_fit_disagreement_m',float('inf'))
            if not np.isfinite(disagreement) or disagreement>.003:
                reasons.append('independent_temporal_fits_disagree')
            if not geometry.get('fresh_window_verified',False):reasons.append('temporal_window_not_verified')
        elif len(geometry.get('evidence',[]))<2:reasons.append('two_current_views_required')
    if not all(r['orientation_ok'] for r in recent):reasons.append('approach_or_closing_axis_not_near_horizontal')
    return {'eligible':not reasons,'reasons':reasons,'stable_fixed_views':fixed_ok,
            'geometry':geometry,'history_frames':[r['frame'] for r in recent],
            'limitation':'development pregrasp envelope, not verified contact/collision safety or guaranteed Cosmos initiation'}


class PregraspGate:
    def __init__(self,reset_rgb):
        self.finger=RightFingerTracker(reset_rgb['right_rgb']);self.history=deque(maxlen=6)
        self.baseline={}
        for n in ('front_rgb','left_rgb'):
            target=detection(reset_rgb[n],pink_mask);rack=detection(reset_rgb[n],_yellow_mask)
            self.baseline[n]=np.array(target['uv'])-rack['uv'] if target and rack else None
        self.pending=None

    def invalidate(self):self.pending=None

    def observe(self,images,frame,command,rotation,source):
        self.invalidate()
        jaw=self.finger.observe(images['right_rgb']);fixed={};rack_fixed={}
        for n in ('front_rgb','left_rgb'):
            target=detection(images[n],pink_mask);rack=detection(images[n],_yellow_mask)
            rack_fixed[n]=rack['uv'] if rack else None
            fixed[n]=None
            if target and rack and self.baseline[n] is not None:
                relative=np.array(target['uv'])-rack['uv']
                fixed[n]=dict(target,relative_uv=relative.tolist(),
                              fixture_error_px=float(np.linalg.norm(relative-self.baseline[n])))
        wrist=detection(images['right_rgb'],pink_mask);rack=detection(images['right_rgb'],_yellow_mask)
        self.history.append({'frame':frame,'holding':source=='hold',
            'jaw_open':jaw['valid'] and abs(jaw['x']-self.finger.baseline_x)<=8,
            'command_open':float(command[13])<=-3.05,
            'orientation_ok':bool(abs(rotation[2,0])<=.35 and abs(rotation[2,1])<=.25),
            'wrist':wrist,'fixed':fixed,'finger_match':jaw,'rack_fixed':rack_fixed,
            'wrist_support_cue':bool(wrist and rack and wrist['pixels']>=100 and
                wrist['uv'][1]+10<=rack['uv'][1])})

    def review(self,frame,geometry):
        result=assess(self.history,geometry)
        if not self.history or self.history[-1]['frame']!=frame:
            result['eligible']=False;result['reasons'].append('stale_visual_history')
        payload=json.dumps({'frame':frame,'evidence':result},sort_keys=True).encode()
        result=dict(result,frame=frame,proposal_only=True,proposal_id=hashlib.sha256(payload).hexdigest()[:20],
                    kind='resume_vla_pregrasp_candidate' if result['eligible'] else 'pregrasp_not_verified')
        self.pending=result
        return result

    def require(self,frame,proposal_id,visual_checks):
        p=self.pending
        if not p or not p['eligible']:raise ValueError('resume blocked: no eligible pregrasp proposal')
        if p['frame']!=frame or p['proposal_id']!=proposal_id:raise ValueError('resume blocked: stale/mismatched pregrasp proposal')
        required=('target_identity','open_finger_corridor','approach_clear','supported_target','appropriate_policy_phase')
        if not isinstance(visual_checks,dict) or not all(isinstance(visual_checks.get(k),str) and
            len(visual_checks[k].strip())>=12 for k in required):
            raise ValueError('resume blocked: Agent must explain each RGB reentry check')
        return p

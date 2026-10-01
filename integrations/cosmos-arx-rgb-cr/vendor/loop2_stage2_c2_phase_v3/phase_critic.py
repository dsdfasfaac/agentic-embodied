"""Scene-specific RGB-only attempted-acquisition critic (not contact sensing).

Unlike v2, approach + target-on-rack can NEVER establish failed acquisition.
Require visible finger closing followed by visual departure without the target.
No action, measured joint, simulator pose/contact, or evaluator input is accepted.
"""
import dataclasses
from collections import deque
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_eef_v2'))
from early_critic import MagentaRgbCritic
from rgb_c2_critic import CriticProposal


class RightFingerTracker:
    """Rigid wrist-camera fingertip template; uncertainty is explicit.

The ROI is specific to this calibrated 320x240 ARX camera. It is not a semantic
segmenter and cannot certify closure/contact when occluded or mismatched.
"""
    def __init__(self,reset):
        if reset.shape!=(240,320,3):raise ValueError('calibrated 320x240 wrist RGB required')
        self.template=reset[141:173,250:277].astype(float).mean(2)
        self.template-=self.template.mean()
        self.norm=np.linalg.norm(self.template)
        self.baseline_x=self.observe(reset)['x']

    def observe(self,image):
        gray=image.astype(float).mean(2);best=(-1.,None,None)
        for dy in range(-3,4):
            region=gray[141+dy:173+dy,180:312]
            windows=np.lib.stride_tricks.sliding_window_view(region,27,axis=1).transpose(1,0,2)
            centered=windows-windows.mean((1,2),keepdims=True)
            scores=(centered*self.template).sum((1,2))/(np.linalg.norm(centered,axis=(1,2))*self.norm+1e-9)
            i=int(np.argmax(scores));score=float(scores[i])
            if score>best[0]:best=(score,i+180,dy)
        return {'score':best[0],'x':best[1],'dy':best[2],'valid':best[0]>=.78}


class PhaseAwareRgbCritic:
    def __init__(self,reset_images):
        self.confirmed=MagentaRgbCritic(reset_images) # features only; never call its failure logic
        self.finger=RightFingerTracker(reset_images['right_rgb'])
        self.frame=0;self.serial=0;self.episode=0
        self.history=deque(maxlen=30);self.events=[];self.last_features=None
        self.reset_phase()

    def reset_phase(self):
        self.phase='approach';self.near_until=-1;self.closing_count=0
        self.attempt_frame=None;self.peak_area=0.;self.peak_xy=None
        self.departure_count=0;self.sent=False;self.unknown_count=0

    def begin_recovery(self):
        # Protocol boundary approved by Agent; supplies no environment observation.
        self.episode+=1;self.history.clear();self.reset_phase()

    def proposal(self,kind,evidence):
        self.serial+=1;self.sent=True
        return CriticProposal(f'rgb-phase-{self.episode}-{self.serial}',kind,evidence,
            ('hold','move_eef','set_gripper','resume_vla','finish'),
            'fresh RGB verifies acquisition or aligned nominal reentry; no contact claim from pixels',
            'occlusion, identity ambiguity, new relative slip or scene disturbance',
            'Agent-approved aligned supported pregrasp or visually stable grasp appropriate for fresh nominal inference')

    def observe(self,images):
        self.frame+=1;views=self.confirmed._features(images)
        jaw=self.finger.observe(images['right_rgb']);w=views['right_rgb']
        self.history.append(views)
        if w.target_visible and w.pink_area_ratio>=1.6:
            self.near_until=self.frame+20
            if w.pink_area_ratio>self.peak_area:
                self.peak_area=w.pink_area_ratio;self.peak_xy=w.target_xy
        # A persistent inward-moving finger in a close-range scene supplies
        # evidence of a grasp attempt, not proof of target contact/acquisition.
        closed=(jaw['valid'] and self.finger.baseline_x-jaw['x']>=25 and self.frame<=self.near_until)
        self.closing_count=self.closing_count+1 if closed else 0
        if self.attempt_frame is None and self.closing_count>=3:
            self.attempt_frame=self.frame;self.phase='attempt_unverified'
        # Both fixed cameras must see target still near fixture baseline.
        fixed=[views[n] for n in ('front_rgb','left_rgb')]
        on_fixture=all(v.target_visible and v.pink_area_ratio>=.62 and
                       v.baseline_delta_error is not None and v.baseline_delta_error<=.065 for v in fixed)
        motions=[]
        if len(self.history)>=12:
            for name in ('front_rgb','left_rgb'):
                pts=[v[name].target_rack_delta for v in list(self.history)[-12:]]
                if all(p is not None for p in pts):
                    array=np.asarray(pts);motions.append(float(np.linalg.norm(array-array[-1],axis=1).max()))
        stable=len(motions)==2 and max(motions)<=.018
        center_shift=(float(np.linalg.norm(np.asarray(w.target_xy)-self.peak_xy))
                      if w.target_visible and self.peak_xy is not None else None)
        departing=(w.target_visible and self.peak_area>=2 and w.pink_area_ratio<.60*self.peak_area
                   and center_shift is not None and center_shift>.08)
        self.departure_count=self.departure_count+1 if departing and on_fixture and stable else 0
        evidence={'rgb_only':True,'phase':self.phase,'attempt_frame':self.attempt_frame,
            'finger_visual_match':jaw,'finger_inward_pixels':self.finger.baseline_x-jaw['x'],
            'peak_wrist_area_ratio':self.peak_area,'wrist_centroid_shift':center_shift,
            'departure_persistence':self.departure_count,'target_on_fixture':on_fixture,
            'target_stable':stable,'views':self.confirmed._json_features(views),
            'limitation':'RGB inference of attempted acquisition; template and color identity can fail; no contact proof'}
        self.last_features=evidence
        if self.sent:return None
        if self.attempt_frame is not None and self.frame-self.attempt_frame>=8 and self.departure_count>=6:
            self.phase='suspected_failed_acquisition'
            evidence['phase']=self.phase
            return self.proposal('attempted_grasp_target_left_behind',evidence)
        # Missing evidence is not a grasp failure. Ask for review only after an
        # attempted acquisition has remained unverifiable for an extended period.
        unknown=self.attempt_frame is not None and not w.target_visible and not on_fixture
        self.unknown_count=self.unknown_count+1 if unknown else 0
        if self.unknown_count>=30:
            return self.proposal('grasp_outcome_unknown',evidence)
        return None

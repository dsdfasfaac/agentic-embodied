"""RGB acquisition checkpoint, separate from confirmed failed acquisition.

Close-range target appearance asks the Agent to verify finger/target alignment
before lifting. It is not itself a failure classification. Occlusion is UNKNOWN.
"""
import sys
import dataclasses
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_rgb_v1'))
from rgb_c2_critic import RgbC2Critic, CriticProposal, _centroid, _yellow_mask


def pink_mask(image):
    rgb=image.astype(np.int16,copy=False)
    r,g,b=(rgb[...,i] for i in range(3))
    # Magenta requires a blue contribution: orange/yellow rack pixels must
    # never count as pink simply because red exceeds green.
    return (r>80)&(r>1.35*g)&(r>1.08*b)&(b>1.15*g)&(b>.30*r)&((r-g)>35)


class MagentaRgbCritic(RgbC2Critic):
    def _raw_view(self,image,baseline_image):
        h,w=image.shape[:2]
        pink=pink_mask(image); yellow=_yellow_mask(image)
        count,xy=_centroid(pink); _,rack=_centroid(yellow)
        delta=None if xy is None or rack is None else (xy[0]-rack[0],xy[1]-rack[1])
        rgb=image.astype(np.int16,copy=False)
        changed=np.max(np.abs(rgb-baseline_image.astype(np.int16,copy=False)),axis=2)>22
        dynamic=None
        if xy is not None:
            x,y=int(xy[0]*w),int(xy[1]*h); hw,hh=max(10,w//7),max(8,h//10)
            region=(changed & (rgb.mean(axis=2)<110) & ~pink)[max(0,y-hh):min(h,y+hh+1),max(0,x-hw):min(w,x+hw+1)]
            dynamic=float(region.mean()) if region.size else None
        return {'pink_pixels':count,'target_xy':xy,'target_rack_delta':delta,
                'dark_dynamic_near_target':dynamic,'global_rgb_change':float(changed.mean())}


class EarlyRgbCritic:
    def __init__(self, reset_images):
        self.confirmed = MagentaRgbCritic(reset_images)
        self.count=0
        self.sent=False

    def observe(self, images):
        features=self.confirmed._features(images)
        wrist=features['right_rgb']
        # Area relative to reset, rather than lateral centroid: compression and
        # partial finger occlusion bias the centroid enough to invalidate a
        # narrow corridor fitted on MP4. A checkpoint may also occur on success.
        risk=(wrist.target_visible and wrist.pink_area_ratio>=1.35
              and .10<wrist.target_xy[0]<.90 and .20<wrist.target_xy[1]<.90)
        self.count=self.count+1 if risk else 0
        confirmed=self.confirmed.observe(images)
        if confirmed is not None:
            return dataclasses.replace(confirmed,
                candidate_actions=('hold_and_inspect_rgb','set_gripper','move_eef','resume_vla','finish'))
        if self.count>=3 and not self.sent:
            self.sent=True
            return CriticProposal(
                proposal_id='rgb-grasp-verification-0001', kind='grasp_verification_required',
                evidence={'rgb_only':True,'status':'acquisition_unverified_not_confirmed_grasp_loss',
                          'consecutive_frames':self.count,
                          'views':self.confirmed._json_features(features),
                          'limitation':'close-range visual checkpoint; Agent must inspect alignment and test co-motion'},
                candidate_actions=('continue_vla','hold_and_inspect_rgb',
                                   'set_gripper','move_eef'),
                expected_success_signal='Agent verifies target between finger pads, then visual co-motion during small test lift',
                escalation_condition='target occluded, displaced, or motion response inconsistent with requested move',
                nominal_vla_reentry_state='visually aligned or stable carried grasp, consistent with successful references')
        return None

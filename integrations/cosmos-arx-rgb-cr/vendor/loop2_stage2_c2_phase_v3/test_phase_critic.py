"""Synthetic temporal logic tests; not substitutes for RGB rollout validation."""
import unittest
from types import SimpleNamespace
import numpy as np
from phase_critic import PhaseAwareRgbCritic


class TemporalTests(unittest.TestCase):
    def setUp(self):
        rgb={n:np.zeros((240,320,3),np.uint8) for n in ('front_rgb','left_rgb','right_rgb')}
        for im in rgb.values():
            im[50:60,50:60]=[220,20,140]
            im[80:90,40:70]=[240,200,0]
        self.rgb=rgb;self.c=PhaseAwareRgbCritic(rgb)
        self.c.finger.baseline_x=250
        self.c.confirmed._json_features=lambda v:{}

    def frames(self,n,area=3.,xy=(.5,.5),closed=False,valid=True,fixture=True,visible=True):
        events=[]
        for _ in range(n):
            fixed=SimpleNamespace(target_visible=fixture,pink_area_ratio=1.,
                baseline_delta_error=0.,target_rack_delta=(0.,0.))
            wrist=SimpleNamespace(target_visible=visible,pink_area_ratio=area,target_xy=xy)
            self.c.confirmed._features=lambda im:{'front_rgb':fixed,'left_rgb':fixed,'right_rgb':wrist}
            self.c.finger.observe=lambda im:{'valid':valid,'x':220 if closed else 250,'score':.9 if valid else .2}
            p=self.c.observe(self.rgb)
            if p:events.append(p.kind)
        return events

    def test_approach_is_not_failure(self):
        self.assertEqual(self.frames(50),[])
        self.assertIsNone(self.c.attempt_frame)

    def test_grasp_without_departure_is_not_failure(self):
        self.assertEqual(self.frames(50,closed=True),[])
        self.assertIsNotNone(self.c.attempt_frame)

    def test_failed_acquisition_requires_attempt_and_departure(self):
        self.frames(20,closed=True)
        self.assertEqual(self.frames(10,area=1.,xy=(.7,.7),closed=True),
                         ['attempted_grasp_target_left_behind'])
        self.assertEqual(self.frames(10,area=1.,xy=(.7,.7),closed=True),[])

    def test_approach_then_depart_without_closure(self):
        self.frames(20)
        self.assertEqual(self.frames(20,area=1.,xy=(.7,.7)),[])

    def test_unreliable_finger_match_cannot_establish_attempt(self):
        self.frames(20,closed=True,valid=False)
        self.assertEqual(self.frames(20,area=1.,xy=(.7,.7),valid=False),[])

    def test_target_missing_is_unknown_not_failure(self):
        self.frames(20,closed=True)
        self.assertEqual(self.frames(35,visible=False,fixture=False),['grasp_outcome_unknown'])

    def test_recovery_clears_stale_attempt(self):
        self.frames(20,closed=True)
        self.c.begin_recovery()
        self.assertEqual(self.frames(20,area=1.,xy=(.7,.7)),[])
        self.assertIsNone(self.c.attempt_frame)


if __name__=='__main__':unittest.main()

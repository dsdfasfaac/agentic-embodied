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
            self.c.confirmed._features=lambda im:{'front_rgb':fixed,'right_rgb':wrist}
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

    def test_main_target_missing_never_means_failed_acquisition(self):
        self.frames(20,closed=True)
        self.assertEqual(self.frames(40,area=1.,xy=(.7,.7),fixture=False),[])


class CameraContractTests(unittest.TestCase):
    def images(self):
        rgb={n:np.zeros((240,320,3),np.uint8) for n in ('front_rgb','right_rgb')}
        for im in rgb.values():
            im[50:60,50:60]=[220,20,140]
            im[80:90,40:70]=[240,200,0]
        return rgb

    def test_two_views_initialize_and_observe(self):
        rgb=self.images();critic=PhaseAwareRgbCritic(rgb)
        self.assertIsNone(critic.observe(rgb))
        self.assertEqual(set(critic.last_features['views']),set(rgb))
        self.assertEqual(critic.last_features['critic_cameras'],list(rgb))

    def test_left_view_black_missing_or_malformed_is_ignored(self):
        rgb=self.images()
        for left in (np.zeros((240,320,3),np.uint8),None,np.ones((1,),np.float32)):
            with self.subTest(left_shape=getattr(left,'shape',None)):
                three=dict(rgb,left_rgb=left)
                critic=PhaseAwareRgbCritic(three)
                self.assertIsNone(critic.observe(three))
                self.assertIsNone(critic.observe(rgb))
                self.assertNotIn('left_rgb',critic.last_features['views'])

    def test_required_band_missing_reports_camera(self):
        for name in ('front_rgb','right_rgb'):
            rgb=self.images();rgb[name][:]=0
            with self.subTest(camera=name),self.assertRaisesRegex(ValueError,'insufficient: '+name):
                PhaseAwareRgbCritic(rgb)

    def test_missing_required_camera_rejected(self):
        for name in ('front_rgb','right_rgb'):
            rgb=self.images();del rgb[name]
            with self.subTest(camera=name),self.assertRaisesRegex(ValueError,'must contain'):
                PhaseAwareRgbCritic(rgb)

    def test_wrist_calibration_still_required(self):
        rgb=self.images();rgb['right_rgb']=rgb['right_rgb'][:200]
        with self.assertRaisesRegex(ValueError,'320x240'):
            PhaseAwareRgbCritic(rgb)

    def test_bad_required_image_dtype_rejected(self):
        rgb=self.images();rgb['front_rgb']=rgb['front_rgb'].astype(np.float32)
        with self.assertRaisesRegex(ValueError,'uint8'):
            PhaseAwareRgbCritic(rgb)

    def test_runtime_shape_change_rejected(self):
        rgb=self.images();critic=PhaseAwareRgbCritic(rgb)
        rgb['front_rgb']=rgb['front_rgb'][:200]
        with self.assertRaisesRegex(ValueError,'shape changed'):
            critic.observe(rgb)

    def test_runtime_band_occlusion_does_not_crash(self):
        rgb=self.images();critic=PhaseAwareRgbCritic(rgb)
        for im in rgb.values():im[:]=0
        self.assertIsNone(critic.observe(rgb))

    def test_left_pixels_cannot_change_evidence(self):
        rgb=self.images()
        a=PhaseAwareRgbCritic(dict(rgb,left_rgb=np.zeros((240,320,3),np.uint8)))
        b=PhaseAwareRgbCritic(dict(rgb,left_rgb=np.full((240,320,3),255,np.uint8)))
        for _ in range(15):
            self.assertEqual(a.observe(rgb),b.observe(dict(rgb,left_rgb=None)))
            self.assertEqual(a.last_features,b.last_features)


if __name__=='__main__':unittest.main()

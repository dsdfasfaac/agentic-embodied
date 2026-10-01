import copy
import unittest
from collections import deque
from pregrasp_gate import assess,PregraspGate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.history=[{'frame':i,'holding':True,'jaw_open':True,'command_open':True,
            'orientation_ok':True,'wrist':{'uv':[160,160],'border':False},
            'fixed':{'front_rgb':{'border':False,'fixture_error_px':1,'relative_uv':[10,20]},
                     'left_rgb':None}} for i in range(1,7)]
        self.geometry={'frame':6,'rgb_band_to_commanded_tcp_tool_m':[.025,0,.004],
                       'ray_residual_m':.001,'evidence':[{},{}]}

    def test_candidate(self):self.assertTrue(assess(self.history,self.geometry)['eligible'])
    def test_no_depth(self):self.assertFalse(assess(self.history,None)['eligible'])
    def test_far_old_recovery_blocked(self):
        self.geometry['rgb_band_to_commanded_tcp_tool_m']=[.078,.085,-.153]
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_lateral_error(self):
        self.geometry['rgb_band_to_commanded_tcp_tool_m'][1]=.02
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_unsettled(self):
        self.history[0]['wrist']['uv']=[160,170]
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_visual_open_required(self):
        self.history[-1]['jaw_open']=False
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_hold_required(self):
        self.history[-1]['holding']=False
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_clipped(self):
        self.history[-1]['wrist']['border']=True
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_displaced_target(self):
        self.history[-1]['fixed']['front_rgb']['fixture_error_px']=30
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_bad_orientation(self):
        self.history[-1]['orientation_ok']=False
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_nan_depth(self):
        self.geometry['ray_residual_m']=float('nan')
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_stale_depth(self):
        self.geometry['frame']=5
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_approval_protocol(self):
        gate=PregraspGate.__new__(PregraspGate);gate.history=deque(self.history);gate.pending=None
        p=gate.review(6,self.geometry)
        checks={k:'Agent inspected current RGB evidence' for k in
            ('target_identity','open_finger_corridor','approach_clear','supported_target','appropriate_policy_phase')}
        self.assertEqual(gate.require(6,p['proposal_id'],checks),p)
        with self.assertRaises(ValueError):gate.require(7,p['proposal_id'],checks)
        with self.assertRaises(ValueError):gate.require(6,'forged-token',checks)
        with self.assertRaises(ValueError):gate.require(6,p['proposal_id'],{})
        gate.invalidate()
        with self.assertRaises(ValueError):gate.require(6,p['proposal_id'],checks)

    def temporal_setup(self):
        for row in self.history:
            row['fixed']={'front_rgb':None,'left_rgb':None}
            row['rack_fixed']={'front_rgb':[160,180],'left_rgb':[270,180]}
            row['wrist_support_cue']=True
        self.geometry.update(measurement_mode='temporal_rgb',baseline_m=.04,
            rgb_observations=[{}]*5,split_fit_disagreement_m=.001,fresh_window_verified=True)

    def test_temporal_candidate(self):
        self.temporal_setup();self.assertTrue(assess(self.history,self.geometry)['eligible'])
    def test_temporal_short_baseline(self):
        self.temporal_setup();self.geometry['baseline_m']=.01
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_temporal_inconsistent_depth(self):
        self.temporal_setup();self.geometry['split_fit_disagreement_m']=.01
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_temporal_missing_support_cue(self):
        self.temporal_setup();self.history[-1]['wrist_support_cue']=False
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_temporal_nan(self):
        self.temporal_setup();self.geometry['split_fit_disagreement_m']=float('nan')
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_missing_frame_in_history(self):
        self.history[1]['frame']=0
        self.assertFalse(assess(self.history,self.geometry)['eligible'])
    def test_review_cannot_relabel_old_images_as_current(self):
        gate=PregraspGate.__new__(PregraspGate);gate.history=deque(self.history);gate.pending=None
        self.assertFalse(gate.review(7,self.geometry)['eligible'])


if __name__=='__main__':unittest.main()

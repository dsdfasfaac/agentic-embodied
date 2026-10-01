import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

spec=importlib.util.spec_from_file_location('online_bridge',Path(__file__).resolve().parents[1]/'scripts/isolated_online_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


class OnlineBoundaryTests(unittest.TestCase):
    def test_proposal_strips_private_keys(self):
        p=bridge.public_proposal({'frame':1,'audit':{'reward':1},'proposal':{
            'kind':'grasp_outcome_unknown','private_pose':[1,2,3],
            'evidence':{'rgb_only':True,'reward':1,'views':{
                'front_rgb':{'pink_pixels':50,'contact':True},'secret':{'pink_pixels':100}}}}})
        self.assertEqual(p['proposal'],{'kind':'grasp_outcome_unknown'})
        self.assertEqual(p['evidence'],{'rgb_only':True,'views':{'front_rgb':{'pink_pixels':50}}})

    def decision(self):return dict(decision_id='one',evidence_frame=10,tool='hold',args={},reason='current RGB')
    def state(self):return dict(pending=False,closed=False,observation={'frame':10})
    def check(self,d=None,state=None,ids=None,frames=None,count=0):
        bridge.validate_decision(d or self.decision(),state or self.state(),ids or set(),frames or {10},count)

    def test_valid_action(self):self.check()
    def test_no_arbitrary_tool(self):
        with self.assertRaises(ValueError):self.check(dict(self.decision(),tool='read_file'))
    def test_no_stale_frame(self):
        with self.assertRaises(ValueError):self.check(dict(self.decision(),evidence_frame=9))
    def test_no_parallel_action(self):
        with self.assertRaises(ValueError):self.check(state=dict(self.state(),pending=True))
    def test_no_duplicate(self):
        with self.assertRaises(ValueError):self.check(ids={'one'})
    def test_no_unseen_history(self):
        with self.assertRaises(ValueError):self.check(dict(self.decision(),args={'observation_frames':[1,10]}))
    def test_own_history_allowed(self):
        self.check(dict(self.decision(),args={'observation_frames':[8,10]}),frames={8,10})
    def test_finish_allowed_at_budget(self):
        with self.assertRaises(ValueError):self.check(count=100)
        self.check(dict(self.decision(),tool='finish'),count=100)


class BoundaryAuditTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('boundary_audit',Path(__file__).resolve().parents[1]/'scripts/audit_online_boundary.py')
        self.audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.audit)

    def test_nested_private_fields_detected(self):
        self.assertEqual(self.audit.forbidden_paths({'tool_result':[{'reward':1}]}),['/tool_result/0/reward'])

    def test_failed_close_rejected_before_log_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);marker=root/'CLOSED.json'
            marker.write_text(json.dumps({'failed':True,'explicit_agent_finish':True}))
            with self.assertRaisesRegex(ValueError,'successful worker close'):
                self.audit.audit(root/'nonexistent-capsule',root/'nonexistent-private',marker)

    def test_transport_close_not_independent_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);marker=root/'CLOSED.json'
            marker.write_text(json.dumps({'failed':False,'explicit_agent_finish':False}))
            with self.assertRaisesRegex(ValueError,'explicit Agent finish'):
                self.audit.audit(root/'nonexistent-capsule',root/'nonexistent-private',marker)


if __name__=='__main__':unittest.main()

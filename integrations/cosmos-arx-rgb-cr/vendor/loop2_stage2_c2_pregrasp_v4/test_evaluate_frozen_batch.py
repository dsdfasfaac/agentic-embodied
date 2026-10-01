import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate_frozen_batch import evaluate, physical_windows


class FixedKinematics:
    def fk(self, _):
        return np.zeros(3), np.eye(3), None


def trace(count=16, clearance=.004, contacts=(True, True), other=()):
    return [dict(measured_state=[0] * 14, evaluation=dict(finger_contacts=contacts),
                 full_pickup_private=dict(target_position=[0, 0, .1],
                                          conservative_vertical_clearance_m=clearance,
                                          nonfinger_contact_bodies=list(other)))
            for _ in range(count)]


class OfflineAuditTests(unittest.TestCase):
    def test_requires_fifteen_frames(self):
        self.assertEqual(physical_windows(trace(14), FixedKinematics()), [])
        self.assertEqual(physical_windows(trace(16), FixedKinematics()), [14, 15])

    def test_clearance_and_contacts(self):
        for data in (trace(clearance=.002), trace(contacts=(True, False)), trace(other=[5])):
            self.assertEqual(physical_windows(data, FixedKinematics()), [])

    def test_relative_drift(self):
        data = trace(15)
        data[-1]['full_pickup_private']['target_position'][0] = .004
        self.assertEqual(physical_windows(data, FixedKinematics()), [])

    def test_all_public_close_checks_precede_private_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'frozen_manifest.json').write_text(json.dumps(dict(seeds=[1, 2], source_sha256={})))
            first, second = root / 'seed_1', root / 'seed_2'
            first.mkdir()
            second.mkdir()
            (first / 'agent_decisions.jsonl').write_text('{"tool":"finish"}\n')
            # Malformed private files would fail JSON loading if opened early.
            (first / 'audit.json').write_text('DO NOT READ WHILE ANOTHER TRIAL IS LIVE')
            (first / 'posthoc_trace.json').write_text('DO NOT READ')
            with self.assertRaisesRegex(AssertionError, 'not closed'):
                evaluate(root)


if __name__ == '__main__':
    unittest.main()

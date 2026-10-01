import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import queue

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('delivery_cr', ROOT / 'cr.py')
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


class DeliveryTests(unittest.TestCase):
    def test_server_loopback_and_fresh_output(self):
        c = cr.config(ROOT / 'config.h20.json')
        cmd, env, _ = cr.server_command(c, Path('/tmp/test-specific-server'), 7, 15611)
        self.assertEqual(cmd[cmd.index('--host') + 1], '127.0.0.1')
        self.assertEqual(cmd[cmd.index('--port') + 1], '15611')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '7')
        self.assertIn('/tmp/test-specific-server/ready.json', cmd)

    def test_client_environment(self):
        env = cr.client_env(cr.config(ROOT / 'config.h20.json'))
        self.assertEqual(env['MUJOCO_GL'], 'osmesa')
        self.assertIn(str(ROOT / 'vendor/loop2_stage2_c2_eef_v2'), env['PYTHONPATH'])

    def test_config_preserves_venv_interpreter_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            interpreter = root / 'venv-python'
            interpreter.symlink_to('/usr/bin/python3')
            data = json.loads((ROOT / 'config.h20.json').read_text())
            data['client_python'] = str(interpreter)
            path = root / 'config.json'
            path.write_text(json.dumps(data))
            self.assertEqual(cr.config(path)['client_python'], str(interpreter))

    def test_audit_refuses_live_session(self):
        spec = importlib.util.spec_from_file_location('delivery_manage', ROOT / 'scripts/manage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'CLOSED.json'):
                module.audit(Path(folder))

    def test_pipe_replay_waits_for_final_checkpoint(self):
        spec = importlib.util.spec_from_file_location('delivery_client', ROOT / 'scripts/agent_client.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = module.AgentSession.__new__(module.AgentSession)
        client.pending = True
        client.latest_frame = None
        client.events = queue.Queue()
        client.events.put(dict(frame=183, reason='verified exact RGB reproduction of previously Agent-reviewed state', rgb={}))
        self.assertEqual(client.wait()['frame'], 183)
        self.assertTrue(client.pending)
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            client.send(dict(evidence_frame=183))
        client.events.put(dict(frame=723, reason='approved RGB-verified state reproduction complete; fresh gate/Agent decision required', rgb={}))
        self.assertEqual(client.wait()['frame'], 723)
        self.assertFalse(client.pending)
        with self.assertRaisesRegex(ValueError, 'current observation'):
            client.send(dict(evidence_frame=183))

    def test_pipe_timeout_keeps_request_pending(self):
        spec = importlib.util.spec_from_file_location('delivery_client_timeout', ROOT / 'scripts/agent_client.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = module.AgentSession.__new__(module.AgentSession)
        client.pending = True
        client.events = queue.Queue()
        with self.assertRaises(queue.Empty):
            client.wait(timeout=.001)
        self.assertTrue(client.pending)


if __name__ == '__main__':
    unittest.main()

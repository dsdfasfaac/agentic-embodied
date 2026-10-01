"""Optional stdlib pipe client. No LLM, no autonomous recovery action selection.

Import AgentSession into an existing Agent orchestrator. It returns image paths,
not embedded images; the caller must load those images and choose each request.
"""
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


class AgentSession:
    def __init__(self, arguments):
        # arguments example: ['live','--seed','183173','--output','/my/runs/new']
        self.process = subprocess.Popen([sys.executable, str(ROOT / 'cr.py'), *arguments],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.events = queue.Queue()
        self.pending = True  # initial bootstrap observation has not arrived yet
        self.latest_frame = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(key in event for key in ('rgb', 'pregrasp_proposal', 'saved')):
                self.events.put(event)
        self.events.put(None)

    def wait(self, timeout=180):
        """Timeout does not cancel/retry a request. Call wait again, not send."""
        event = self.events.get(timeout=timeout)
        if event is None:
            raise RuntimeError(f'worker exited, status={self.process.poll()}')
        # Replay bootstrap emits intermediate checkpoints; only its final
        # "complete" observation makes a new request possible.
        intermediate = event.get('reason', '').startswith('verified exact RGB reproduction')
        if not intermediate:
            self.pending = False
        if 'frame' in event:
            self.latest_frame = event['frame']
        return event

    def send(self, decision):
        """The calling Agent supplies the entire explicit decision."""
        if self.pending:
            raise RuntimeError('previous request/bootstrap still pending; wait for its final response')
        if decision.get('evidence_frame') != self.latest_frame:
            raise ValueError('request must cite current observation frame')
        if self.process.poll() is not None:
            raise RuntimeError('worker exited')
        self.pending = True
        self.process.stdin.write(json.dumps(decision, ensure_ascii=False) + '\n')
        self.process.stdin.flush()

    def close_transport(self):
        """EOF, not an Agent finish decision. Prefer explicitly sending finish."""
        self.process.stdin.close()
        return self.process.wait(timeout=180)

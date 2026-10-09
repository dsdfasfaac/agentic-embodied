"""Bounded background artifact work; SQLite remains owned by the control thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io

import numpy as np

from .contracts import GatewayError
from .public import atomic_write


class ArtifactWriter:
    def __init__(self, *, workers=4, capacity=128, wait_timeout_s=2.0):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='arx-artifact')
        self.capacity, self.timeout = capacity, wait_timeout_s
        self.pending = []
        self.completed = []
        self.closed = False
        self.peak_pending = 0

    def _finish(self, future):
        try:
            record = future.result(timeout=self.timeout)
        except Exception as exc:
            raise GatewayError('ARTIFACT_WRITE_FAILED', str(exc)) from exc
        if record is not None:
            self.completed.append(record)

    def poll(self):
        # Only the owner drains results; workers never touch the journal.
        for future in self.pending.copy():
            if future.done():
                self._finish(future)
                self.pending.remove(future)
        records, self.completed = self.completed, []
        return records

    def submit(self, fn, *args):
        if self.closed:
            raise RuntimeError('artifact writer is closed')
        # Bound memory and apply backpressure rather than dropping observations.
        if len(self.pending) >= self.capacity:
            future = self.pending[0]
            self._finish(future)
            self.pending.remove(future)
        self.pending.append(self.pool.submit(fn, *args))
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def flush(self):
        for future in self.pending.copy():
            self._finish(future)
            self.pending.remove(future)
        return self.poll()

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=False)


def retain_sensor_archive(target, frames, evidence):
    stream = io.BytesIO()
    np.savez_compressed(stream, **frames)
    data = stream.getvalue()
    atomic_write(target, data)
    return ('grasp_sensor_evidence', dict(evidence, sha256=hashlib.sha256(data).hexdigest()))

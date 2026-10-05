from collections import deque
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np

from robots.arx.gateway import real_camera
from robots.arx.gateway.real_backend import CameraIdentity


def test_d405_startup_frames_never_reach_observation_buffer(monkeypatch):
    source = real_camera.RealSenseCameraSource.__new__(real_camera.RealSenseCameraSource)
    source._warmup_s = 1.0
    source._closed = Event()
    source._lock = Lock()
    source._buffers = {"front_rgb": deque(maxlen=8)}
    source._errors = {}
    stamps = iter([0, 500_000_000, 1_500_000_000])
    monkeypatch.setattr(real_camera.time, "monotonic_ns", lambda: next(stamps))

    class Pipeline:
        def __init__(self):
            self.count = 0

        def wait_for_frames(self, timeout):
            self.count += 1
            pixels = np.full((3, 4, 3), self.count, dtype=np.uint8)
            if self.count == 2:
                source._closed.set()
            color = SimpleNamespace(get_data=lambda: pixels)
            return SimpleNamespace(get_color_frame=lambda: color)

    source._pipelines = {"front_rgb": Pipeline()}
    identity = CameraIdentity("front_rgb", "serial", "calibration", "a" * 64, 4, 3)
    spec = real_camera.RealSenseCameraSpec(identity, "serial", Path("unused"), 4, 3, 15)
    source._read_loop(spec)
    assert not source._errors
    frames = list(source._buffers["front_rgb"])
    assert len(frames) == 1
    assert frames[0].monotonic_ns == 1_500_000_000
    assert np.all(frames[0].pixels == 2)

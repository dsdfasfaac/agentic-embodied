# Copyright (c) 2026 Zetta Contributors
"""Fail-closed Linux isolation and bounded observation IPC."""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from robots.arx.gateway.contracts import canonical


class IsolationError(RuntimeError):
    pass


class SandboxedPythonSource:
    def __init__(self, code: bytes, limits, *, timeout_ms):
        self.limits, self.timeout = limits, timeout_ms / 1000
        self.temp = tempfile.TemporaryDirectory(prefix="arx-critic-")
        self.process = None
        self.failed = False
        self.counter = 0
        try:
            self._launch(code)
        except BaseException:
            self.close()
            raise

    def _launch(self, code):
        unshare = shutil.which("unshare")
        if not unshare or not Path("/lib/x86_64-linux-gnu/libseccomp.so.2").exists():
            raise IsolationError(
                "ISOLATION_UNAVAILABLE: Linux namespaces and seccomp required"
            )
        python = str(Path(self.limits.python).absolute())
        # Query the trusted pinned runtime, never candidate code, outside sandbox.
        probe = subprocess.run(
            [
                python,
                "-I",
                "-c",
                'import json,sys,sysconfig,numpy; print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix,stdlib=sysconfig.get_path("stdlib"),numpy=numpy.__file__)))',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        runtime = json.loads(probe.stdout)
        root = Path(self.temp.name)
        package = root / "package"
        package.mkdir()
        (package / "features.py").write_bytes(code)
        # Only stdlib, numpy and its binary libraries are exposed. Do not mount
        # all site-packages (which could contain project/editable installations).
        stdlib = Path(runtime["stdlib"])
        mounts = [(python, python)]
        for item in stdlib.iterdir():
            if item.name not in {"site-packages", "dist-packages", "__pycache__"}:
                mounts.append((str(item), str(item)))
        numpy_root = Path(runtime["numpy"]).parent
        for item in numpy_root.parent.glob("numpy*"):
            if item.name == "numpy" or item.name == "numpy.libs":
                mounts.append((str(item), str(item)))
        # Runtime library directory is trusted binaries only; never simulator assets.
        for directory in [
            "/lib/x86_64-linux-gnu",
            "/lib64",
            str(Path(runtime["base"]) / "lib"),
        ]:
            path = Path(directory)
            if path.exists():
                if directory == str(Path(runtime["base"]) / "lib"):
                    for lib in path.glob("*.so*"):
                        mounts.append((str(lib.resolve()), str(lib)))
                else:
                    mounts.append((str(path.resolve()), directory))
        for prefix in {runtime["prefix"], runtime["base"]}:
            cfg = Path(prefix) / "pyvenv.cfg"
            if cfg.exists():
                mounts.append((str(cfg), str(cfg)))
        worker = Path(__file__).with_name("feature_worker.py").resolve()
        mounts += [(str(package), "/package"), (str(worker), "/worker.py")]
        config = dict(
            self.limits.model_dump(),
            root=str(root / "root"),
            python=python,
            mounts=mounts,
        )
        config_path = root / "launcher.json"
        config_path.write_text(canonical(config))
        launcher = Path(__file__).with_name("sandbox_launcher.py").resolve()
        self.process = subprocess.Popen(
            [
                unshare,
                "--user",
                "--map-root-user",
                "--mount",
                "--net",
                "--pid",
                "--fork",
                "--kill-child",
                python,
                "-I",
                str(launcher),
                str(config_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        ready = self._read(time.monotonic() + self.limits.startup_timeout_s)
        if ready != {"ready": True, "isolation": "namespaces-chroot-seccomp-v1"}:
            raise IsolationError("ISOLATION_PROBE_FAILED")

    def _read_exact(self, size, deadline):
        result = bytearray()
        fd = self.process.stdout.fileno()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise IsolationError("FEATURE_WORKER_TIMEOUT")
            part = os.read(fd, size - len(result))
            if not part:
                raise IsolationError("FEATURE_WORKER_LOST_OR_ISOLATION_UNAVAILABLE")
            result.extend(part)
        return bytes(result)

    def _read(self, deadline):
        size = struct.unpack("!I", self._read_exact(4, deadline))[0]
        if size > self.limits.max_message_bytes:
            raise IsolationError("FEATURE_OUTPUT_TOO_LARGE")
        return json.loads(self._read_exact(size, deadline))

    def _request(self, kind, observation=None, images=None, config=None):
        if self.failed:
            raise IsolationError("feature worker already failed")
        self.counter += 1
        identity = str(self.counter)
        obs_id = observation["observation_id"] if observation else None
        encoded = {}
        for camera, value in (images or {}).items():
            if value.dtype != np.uint8 or value.shape != (
                self.limits.image_height,
                self.limits.image_width,
                3,
            ):
                raise IsolationError("INVALID_CAMERA_SHAPE")
            encoded[camera] = {
                "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode(),
            }
        message = {
            "kind": kind,
            "request_id": identity,
            "observation_id": obs_id,
            "observation": observation,
            "images": encoded,
            "config": config,
        }
        payload = canonical(message).encode()
        if len(payload) > self.limits.max_message_bytes:
            raise IsolationError("FEATURE_INPUT_TOO_LARGE")
        deadline = time.monotonic() + self.timeout
        try:
            data = memoryview(struct.pack("!I", len(payload)) + payload)
            fd = self.process.stdin.fileno()
            os.set_blocking(fd, False)
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                    raise IsolationError("FEATURE_WORKER_TIMEOUT")
                count = os.write(fd, data[:4096])
                data = data[count:]
            response = self._read(deadline)
            if (
                response.get("request_id") != identity
                or response.get("observation_id") != obs_id
                or response.get("ok") is not True
            ):
                raise IsolationError("INVALID_FEATURE_WORKER_RESPONSE")
            if set(response) != {"request_id", "observation_id", "ok", "result"}:
                raise IsolationError("INVALID_FEATURE_WORKER_RESPONSE")
            return response["result"]
        except BaseException:
            self.failed = True
            self.close()
            raise

    def reset(self, observation, images, config):
        if self._request("RESET", observation, images, config) is not None:
            raise IsolationError("invalid reset result")

    def extract(self, observation, images):
        return self._request("EXTRACT", observation, images)

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        self.temp.cleanup()

# Copyright (c) 2026 Zetta Contributors
"""Standalone worker, mounted without project checkout. JSON IPC, never pickle."""

import base64
import ctypes
import errno
import importlib.util
import json
import os
import socket
import struct
import sys

import numpy as np

MAX_MESSAGE = int(os.environ["ARX_MAX_MESSAGE"])


def lock_syscalls():
    lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp unavailable")
    # Filesystem isolation is chroot + read-only mounts. Deny escape, network,
    # subprocess/thread creation, ptrace and kernel mutation at the syscall layer.
    denied = [
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "sendmsg",
        "clone",
        "clone3",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "unshare",
        "setns",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "open_by_handle_at",
        "bpf",
        "perf_event_open",
        "userfaultfd",
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register",
        "init_module",
        "finit_module",
        "delete_module",
        "reboot",
        "kexec_load",
        "keyctl",
        "add_key",
        "request_key",
    ]
    try:
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0)
                != 0
            ):
                raise RuntimeError("cannot install seccomp rule")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("cannot load seccomp")
    finally:
        lib.seccomp_release(context)


def receive():
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError("short header")
    size = struct.unpack("!I", header)[0]
    if size > MAX_MESSAGE:
        raise ValueError("oversized request")
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise ValueError("short request")
    return json.loads(payload)


def send(value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("oversized output")
    protocol.write(struct.pack("!I", len(payload)) + payload)
    protocol.flush()


def images_from(message):
    result = {}
    for name, image in message["images"].items():
        raw = base64.b64decode(image["data"], validate=True)
        value = np.frombuffer(raw, dtype=np.uint8).reshape(image["shape"])
        value.flags.writeable = False
        result[name] = value
    return result


def main():
    lock_syscalls()
    # Access-denial probes execute under exactly the same restrictions as code.
    for path in [
        "/etc/passwd",
        "/proc/self/environ",
        "/data4",
        "/home",
        "/package/features.py",
    ]:
        try:
            if path.startswith("/package"):
                with open(path, "ab"):
                    pass
            else:
                with open(path, "rb"):
                    pass
        except OSError:
            continue
        raise RuntimeError("filesystem isolation probe failed")
    for probe in [lambda: socket.socket(), lambda: os.fork()]:
        try:
            probe()
        except OSError:
            continue
        raise RuntimeError("syscall isolation probe failed")
    send({"ready": True, "isolation": "namespaces-chroot-seccomp-v1"})
    source = None
    initialized = False
    while True:
        message = receive()
        if message is None:
            break
        response = {
            "request_id": message["request_id"],
            "observation_id": message.get("observation_id"),
        }
        try:
            if message["kind"] == "CLOSE":
                if source:
                    source.close()
                send(dict(response, ok=True, result=None))
                break
            if source is None:
                spec = importlib.util.spec_from_file_location(
                    "candidate_features", "/package/features.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                source = module.FeatureExtractor()
            images = images_from(message)
            if message["kind"] == "RESET":
                if initialized:
                    raise ValueError("duplicate reset")
                source.reset(message["observation"], images, message["config"])
                initialized = True
                value = None
            elif message["kind"] == "EXTRACT" and initialized:
                value = source.extract(message["observation"], images)
            else:
                raise ValueError("invalid worker sequence")
            send(dict(response, ok=True, result=value))
        except BaseException:
            send(dict(response, ok=False, error="FEATURE_WORKER_ERROR"))
            break


protocol = sys.stdout.buffer
# Learner prints cannot corrupt protocol or leak arbitrary paths through stderr.
sys.stdout = sys.stderr
if __name__ == "__main__":
    main()

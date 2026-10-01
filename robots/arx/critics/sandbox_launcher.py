# Copyright (c) 2026 Zetta Contributors
"""Trusted Linux user/mount/network namespace launcher. Executed before learner code."""

import json
import os
import resource
import subprocess
import sys
from pathlib import Path


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    root = Path(config["root"])
    root.mkdir()

    def run(*args):
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

    run("/bin/mount", "--make-rprivate", "/")
    # The new root is bounded RAM and contains only runtime, code and scratch.
    run("/bin/mount", "-t", "tmpfs", "-o", "size=32m,nosuid,nodev", "tmpfs", str(root))
    for source, destination in config["mounts"]:
        target = root / destination.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.touch()
        run("/bin/mount", "--bind", source, str(target))
        run("/bin/mount", "-o", "remount,bind,ro,nosuid,nodev", str(target))
    (root / "scratch").mkdir()
    run(
        "/bin/mount",
        "-t",
        "tmpfs",
        "-o",
        f"size={config['scratch_bytes']},nosuid,nodev,noexec",
        "tmpfs",
        str(root / "scratch"),
    )
    run("/bin/mount", "-o", "remount,ro,nosuid,nodev", "tmpfs", str(root))
    os.chroot(root)
    os.chdir("/scratch")
    resource.setrlimit(resource.RLIMIT_AS, (config["memory_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (config["cpu_seconds"],) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (config["scratch_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    os.closerange(3, 1024)
    os.execve(
        config["python"],
        [config["python"], "-I", "/worker.py"],
        {
            "PATH": "/nonexistent",
            "HOME": "/scratch",
            "TMPDIR": "/scratch",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "ARX_MAX_MESSAGE": str(config["max_message_bytes"]),
        },
    )


if __name__ == "__main__":
    main()

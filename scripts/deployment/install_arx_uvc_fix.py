#!/usr/bin/env python3
"""Install or remove a staged UVC overlay for the next boot; never reload it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "rollback"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise PermissionError("run with sudo; this changes module files and initramfs")
    controllers = subprocess.run(["pgrep", "-f", "[X]5Controller|[r]os2 launch arx_x5_controller"], capture_output=True)
    if controllers.returncode != 1:
        raise RuntimeError("robot controller running, or unable to check process status")
    report = json.loads(args.manifest.read_text())
    pin = json.loads((Path(__file__).resolve().parent / "patches/uvc-status-source.json").read_text())
    if report["source"] != pin:
        raise ValueError("staged source/patch identities changed")
    items = report["modules"]
    if not items or len({i["kernel"] for i in items}) != len(items):
        raise ValueError("empty or duplicate module list")
    # Verify the complete plan before making any system change.
    for item in items:
        kernel = item["kernel"]
        if kernel not in pin["kernels"]:
            raise ValueError(f"unsupported kernel: {kernel}")
        source = Path(item["path"])
        if sha(source) != item["sha256"]:
            raise ValueError(f"staged module checksum mismatch: {kernel}")
        magic = subprocess.check_output(["modinfo", "-F", "vermagic", str(source)], text=True).strip()
        if magic != item["vermagic"] or magic.split()[0] != kernel:
            raise ValueError(f"module ABI mismatch: {kernel}")
        destination = Path(f"/lib/modules/{kernel}/updates/arx-uvc/uvcvideo.ko")
        if destination.exists() and sha(destination) != item["sha256"]:
            raise ValueError(f"different overlay already installed: {destination}")
        original = Path(f"/lib/modules/{kernel}/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst")
        original_version = subprocess.check_output(["modinfo", "-F", "srcversion", str(original)], text=True).strip()
        if original_version != pin["expected_original_srcversion"]:
            raise ValueError(f"stock driver changed: {kernel}")
    for item in items:
        kernel = item["kernel"]
        destination = Path(f"/lib/modules/{kernel}/updates/arx-uvc/uvcvideo.ko")
        if args.action == "install":
            subprocess.run(["install", "-D", "-m", "0644", item["path"], str(destination)], check=True)
        elif destination.exists():
            destination.unlink()
        subprocess.run(["depmod", "-a", kernel], check=True)
        subprocess.run(["update-initramfs", "-u", "-k", kernel], check=True)
        selected = subprocess.check_output(["modinfo", "-k", kernel, "-F", "filename", "uvcvideo"], text=True).strip()
        expected = destination if args.action == "install" else Path(f"/lib/modules/{kernel}/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst")
        if Path(selected).resolve() != expected.resolve():
            raise RuntimeError(f"module selection mismatch: {selected}")
        print(f"{args.action}: {kernel} selects {selected}", flush=True)
    print("Disk deployment complete. Running module unchanged; activation requires a separately authorized reboot.")


if __name__ == "__main__":
    main()

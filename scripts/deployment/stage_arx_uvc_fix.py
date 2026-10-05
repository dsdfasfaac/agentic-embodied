#!/usr/bin/env python3
"""Build the pinned UVC fix without loading a module or restarting the host."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", action="append", required=True)
    args = parser.parse_args()
    patches = Path(__file__).resolve().parent / "patches"
    pin = json.loads((patches / "uvc-status-source.json").read_text())
    fix = patches / "uvc-status-self-deadlock-6d27f92.patch"
    if sha(fix) != pin["patch_sha256"]:
        raise ValueError("upstream patch checksum mismatch")
    for name, digest in pin["source_files"].items():
        if Path(name).name != name or sha(args.source / name) != digest:
            raise ValueError(f"pinned source mismatch: {name}")
    kernels = list(dict.fromkeys(args.kernel))
    for kernel in kernels:
        if kernel not in pin["kernels"]:
            raise ValueError(f"unsupported kernel: {kernel}")
        original = f"/lib/modules/{kernel}/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst"
        if output("modinfo", "-F", "srcversion", original) != pin["expected_original_srcversion"]:
            raise ValueError(f"original UVC module changed: {kernel}")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {"source": pin, "activation_pending": True, "modules": []}
    for kernel in kernels:
        for variant in ("baseline", "patched"):
            tree = destination / kernel / variant
            tree.mkdir(parents=True)
            for name in pin["source_files"]:
                shutil.copyfile(args.source / name, tree / name)
            if variant == "patched":
                subprocess.run(["patch", "--batch", "--fuzz=0", "-p5", "-i", str(fix)], cwd=tree, check=True)
            with (tree / "build.log").open("w") as log:
                subprocess.run(["make", "-C", f"/lib/modules/{kernel}/build", f"M={tree}",
                                "CC=x86_64-linux-gnu-gcc-13", "-j4", "modules"], stdout=log, stderr=log, check=True)
            module = tree / "uvcvideo.ko"
            version = output("modinfo", "-F", "srcversion", str(module))
            vermagic = output("modinfo", "-F", "vermagic", str(module))
            if vermagic.split()[0] != kernel:
                raise ValueError(f"module ABI mismatch: {kernel}")
            if variant == "baseline":
                # Out-of-tree modpost hashes local dependencies differently
                # from an in-tree distro build; compare like build modes.
                if version != pin["expected_baseline_srcversion"]:
                    raise ValueError(f"reconstructed baseline differs: {version}")
            else:
                disassembly = output("objdump", "-dr", "--disassemble=uvc_status_stop.part.0", str(module))
                (tree / "status-stop.disassembly.txt").write_text(disassembly + "\n")
                if "current_work" not in disassembly or "cancel_work_sync" not in disassembly:
                    raise ValueError("compiled status-stop guard missing; inspect disassembly")
                report["modules"].append({"kernel": kernel, "path": str(module), "sha256": sha(module),
                                          "srcversion": version, "vermagic": vermagic})
                print(f"Built {kernel}: srcversion={version}", flush=True)
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Staged manifest: {destination / 'manifest.json'}; running driver unchanged")


if __name__ == "__main__":
    main()

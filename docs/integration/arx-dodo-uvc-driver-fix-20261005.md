# Dodo D405 UVC deadlock and staged repair — 2026-10-05

## Current state

The robot controller remains stopped. No new robot commands or paired
evolution rollouts were executed. The user requested no host reboot, and none
was performed. The Model A inference service was not intentionally stopped.

**The repair is installed on disk but not active in the running kernel.** Both
installed kernels, `7.0.0-31-generic` and `7.0.0-34-generic`, select the patched
module for their next boot. Their initramfs images were regenerated. The
currently loaded module still has original srcversion
`D122AF0B4E7F76517D3E4D6`; its deadlocked worker cannot finish or release the
module. Fresh camera capture has not passed after this repair.

## Root cause and evidence

The running kernel is Ubuntu `7.0.0-31.31~24.04.1-generic`, upstream `7.0.14`.
Worker1390108 holds the UVC control/status locks and has this kernel stack:

```text
__flush_work
cancel_work_sync
uvc_status_stop.part.0
uvc_status_put
uvc_pm_put
uvc_ctrl_status_event
uvc_ctrl_status_event_work
process_one_work
```

The status-event worker synchronously cancels the work item it is executing.
Cancellation waits for that same worker to return, so it waits for itself.
Kernel hung-task reports name this worker as the mutex owner blocking the
Python control ioctl and handle cleanup. Worker1378963 (`usb_hub_wq`) also
waits in `usb_disconnect`. This explains why camera unplug/replug, SDK reset,
USB reset and killing Python processes did not restore device enumeration.
Unloading `uvcvideo` was refused because the module remains in use. We did not
force-unload it or modify live kernel memory.

The stack matches the upstream fix
[6d27f92c54ce28cfbd2a8a479a96d6f4a781b7d2](https://github.com/torvalds/linux/commit/6d27f92c54ce28cfbd2a8a479a96d6f4a781b7d2).
It checks `current_work()` before synchronous cancellation, keeps the stopped
status flag set, and drains previous status work on the next start.
[Ubuntu bug2160752](https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2160752)
reports the same RealSense D405 failure and successful backport testing on
other hardware. That validation does not replace acceptance on dodo.
[Ubuntu's advisory](https://ubuntu.com/security/CVE-2026-74437) identifies the
same fault. Inspection of dodo's two original module binaries found no
`current_work` guard; their original srcversions are identical. Booting the
stock `-34` module alone would retain this fault.

Dark initial RGB frames are a separate startup settling issue. The existing
four-second frame warmup remains in place. Exposure-option changes during
diagnosis triggered the kernel fault; effective sensor options must be
rechecked after the driver is recovered.

## Reproducible repair

Tracked files:

- `scripts/deployment/patches/uvc-status-self-deadlock-6d27f92.patch`: upstream patch.
- `scripts/deployment/patches/uvc-status-source.json`: source, archive and patch SHAs.
- `scripts/deployment/stage_arx_uvc_fix.py`: checksum verification and module builds.
- `scripts/deployment/install_arx_uvc_fix.py`: disk install or rollback; never reboots or reloads the module.

Source was extracted from the official Ubuntu source package
`linux-hwe-7.0 7.0.0-31.31~24.04.1`, after validating both original archive and
Ubuntu diff against the published DSC SHA-256 values. Its UVC diff was applied
without fuzzy matching. The 13 resulting source files are individually
pinned. Builds used each installed kernel's headers, symbol versions and
`x86_64-linux-gnu-gcc-13`. Baseline and patched builds passed for both kernels.
Out-of-tree baseline srcversion is `73560DC28F5DF37F5F32893`; modpost's local
dependency hashing differs from the in-tree distro build. The source hashes
and original installed-module identity are checked separately.

Patched srcversion: `481BFC00E4FD5950221E0B9`. Compiled disassembly contains
the `current_work()` guard before `cancel_work_sync()`. Build-time modpost and
vermagic checks passed. Secure Boot was disabled on dodo.

Artifacts on dodo:

```text
/home/dodo/chenfu/.deps/uvc-repair-20261005/ubuntu-source
/home/dodo/chenfu/.deps/uvc-repair-20261005/staged-v2/manifest.json
/lib/modules/7.0.0-31-generic/updates/arx-uvc/uvcvideo.ko
/lib/modules/7.0.0-34-generic/updates/arx-uvc/uvcvideo.ko
```

The stock modules under `kernel/drivers/media/usb/uvc/` remain intact. Overlay
module SHAs and kernel ABI strings are in the staged manifest. Module lookup
selects the overlays after `depmod`; both initramfs updates succeeded.

## Activation and acceptance

Activation requires a separately authorized dodo reboot to clear the existing
kernel deadlock. Restarting Python or reconnecting USB cannot unwind the
self-wait already in progress. Do not restart the robot controller as part of
camera recovery. No robot-related system service was found in the system
unit inventory; this is not proof that arbitrary user startup hooks are absent.

After an authorized reboot, verify the loaded UVC srcversion is the patched
one and no UVC worker is blocked. Keep the controller stopped. Run the
camera-only `scripts/deployment/audit_arx_live_cameras.py` with dodo's Python
environment; verify warm, fresh RGB for all three cameras, aligned front
metric depth, calibrated intrinsics and five observations of the actual pink
tube. Recheck exposure/gain/white balance without uncontrolled option toggles.
Also recheck CAN interface mapping: the USB hub reconnect log included ARX
USB2CAN devices, and `can1`/`can3` were absent in the latest interface inventory.
Restore their mapping before any later authorized controller start.

Only after camera and CAN acceptance, and revocation of the current robot-stop
request, resume the physical parent/candidate comparison. Its task result
remains pending; infrastructure repair is not an evolution success.

Rollback of this disk overlay, on dodo from the repository:

```bash
sudo python3 scripts/deployment/install_arx_uvc_fix.py rollback \
  /home/dodo/chenfu/.deps/uvc-repair-20261005/staged-v2/manifest.json
```

Rollback verifies overlay checksums, removes only these overlays and
regenerates module indexes/initramfs. It does not reload a driver or reboot.

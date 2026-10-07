#!/usr/bin/env python3
"""Discover and dispatch ARX real deployment commands using their existing CLIs.

No hardware is imported or opened until a subcommand is selected. Arguments
and exit codes belong to the original command. ROS and Python environment
selection remain the responsibility of the caller on dodo.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Keep this map explicit: reviewing an entry reveals the actual executable.
COMMANDS = {
    "check": (
        "scripts.deployment.serve_arx_real_gateway",
        ("--check-config",),
        "Offline configuration check; opens no hardware",
    ),
    "observe": (
        "scripts.deployment.audit_arx_live_observation",
        (),
        "Read synchronized sensors; publishes no motion",
    ),
    "camera-audit": (
        "scripts.deployment.audit_arx_live_cameras",
        (),
        "Read cameras; publishes no motion",
    ),
    "freeze": (
        "scripts.deployment.freeze_arx_picktube_inputs",
        (),
        "Generate a frozen catalog and real input contract",
    ),
    "run": (
        "scripts.deployment.run_arx_real_bundle",
        (),
        "Execute a new CandidateBundle rollout; sends robot commands",
    ),
    "continue": (
        "scripts.deployment.continue_arx_grasp_bundle",
        (),
        "Audit or execute a checkpoint suffix; mode flag required",
    ),
    "stage": (
        "scripts.deployment.stage_arx_picktube_start",
        (),
        "Plan or execute measured staging; execution flag required",
    ),
    "finish": (
        "scripts.deployment.finish_arx_real_episode",
        (),
        "Verify home and disable after unloading; execution flag required",
    ),
    "evolution-audit": (
        "scripts.evolution.audit_arx_real_evolution",
        (),
        "Export evidence, shadow replay or compare physical trials",
    ),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use <command> --help for the original command's arguments.",
    )
    parser.add_argument("command", choices=COMMANDS, nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        print("\nCommands:")
        for name, (_, _, description) in COMMANDS.items():
            print(f"  {name:16} {description}")
        return
    args = parser.parse_args(argv)
    module, fixed, _ = COMMANDS[args.command]
    sys.argv = [module, *fixed, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

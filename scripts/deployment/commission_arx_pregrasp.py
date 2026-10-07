#!/usr/bin/env python3
"""Supervised commissioning of a frozen bundle prefix through the real gateway.

No VLA call is made. A one-step hold lets the real bundle critic trigger; only
the bundle's ordered recovery prefix through pregrasp is executed. The optional
physical step cap is a commissioning stop, never reported as critic success.
Closing the gateway leaves the separately managed controllers running.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.deployment.serve_arx_real_gateway import RealCoreFactory
from robots.arx.deployment.bundle_program import resolve_call
from robots.arx.gateway.contracts import ToolRequest
from zetta.evolution.jsonio import file_sha256


def run(args):
    root = Path(__file__).resolve().parents[2]
    provider = root / "robots/arx/deployment/picktube_rgbd_provider.py"
    args.output.mkdir(parents=True, exist_ok=False)
    gateway = args.output / "private/gateway"
    gateway.mkdir(parents=True)
    core = None
    results = []
    report = {"schema_version": "arx.pregrasp.commission.v1", "status": "initializing",
              "termination_reason": "commissioning", "vla_called": False,
              "controller_disabled": False, "max_physical_steps": args.max_physical_steps,
              "bundle_sha256": file_sha256(args.bundle),
              "grasp_config_sha256": file_sha256(args.grasp_config)}
    def cancelled():
        return core is not None and core.step_index >= args.max_physical_steps
    factory = RealCoreFactory(
        hardware_config=str(root / "robots/arx/manifests/real/dodo_picktube_hardware.json"),
        hardware_sha256=args.hardware_sha256,
        task=str(root / "robots/arx/manifests/pickup_test_tube.yaml"),
        model_contract=str(root / "robots/arx/manifests/task7_model_a.yaml"),
        output=str(gateway), episode_id="commission-" + uuid.uuid4().hex,
        limits=dict(max_steps=600, max_decisions=16, max_recoveries=1,
                    operation_timeout_s=120., critic_timeout_s=5., idle_agent_timeout_s=120.,
                    lease_timeout_s=120., shutdown_timeout_s=10.),
        zeva_host="127.0.0.1", zeva_port=5583,
        kinematics_calibration=str(root / "robots/arx/manifests/real/dodo_right_controller_ee_fk.json"),
        bundle=str(args.bundle), tool_catalog=str(args.frozen / "tool-catalog.json"),
        real_input_contract=str(args.frozen / "real-input-contract.json"),
        expected_real_input_sha256=file_sha256(args.frozen / "real-input-contract.json"),
        feature_provider=str(provider), expected_feature_provider_sha256=file_sha256(provider),
        grasp_config=str(args.grasp_config), expected_grasp_config_sha256=file_sha256(args.grasp_config))
    def call(tool, arguments):
        request = ToolRequest(request_id="call-" + uuid.uuid4().hex,
            decision_ref="decision-" + uuid.uuid4().hex,
            observation_id=core.current["observation_id"], control_epoch=core.epoch,
            tool=tool, arguments=arguments, evidence_ids=[core.current["observation_id"]],
            reason="Supervised frozen bundle pregrasp commissioning")
        core.journal.register_decision(request, source="runner", evidence=request.evidence_ids)
        result = core.execute(request)
        results.append(result)
        (args.output / "calls.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"tool": tool, "status": result["status"],
                          "steps": result["executed_steps"], "result": result["result"],
                          "error": result["error"]}), flush=True)
        return result
    try:
        core = factory(cancelled, lambda phase: None)
        # This harness admits only the open-gripper geometric pregrasp prefix.
        # Inspect all programs before reset or the first hold sends a command.
        for program in core.programs.values():
            reached_pregrasp = False
            for entry in program.calls:
                if reached_pregrasp:
                    break
                allowed = (
                    entry.tool == "arx.propose_grasp" and entry.arguments.get("engine", "tube_geometry") == "tube_geometry"
                    or entry.tool == "arx.set_gripper" and entry.arguments.get("opening") == 1.
                    or entry.tool in {"arx.review_grasp", "arx.execute_grasp"}
                    and entry.arguments.get("phase", "pregrasp") == "pregrasp")
                if not allowed:
                    raise ValueError("commissioning admits only open geometric pregrasp calls")
                reached_pregrasp = entry.tool == "arx.execute_grasp"
            if not reached_pregrasp:
                raise ValueError("bundle has no pregrasp commissioning prefix")
        core.reset()
        result = call("arx.hold", {"steps": 1})
        if result["status"] != "interrupted" or core.recovery is None:
            raise ValueError("commissioning critic did not interrupt the hold")
        program = core.programs[core.recovery["binding_id"]]
        for entry in program.calls:
            if entry.tool in {"arx.review_reentry", "arx.zeva"}:
                break
            if entry.tool == "arx.execute_grasp" and entry.arguments.get("phase") != "pregrasp":
                raise ValueError("commissioning supports pregrasp only")
            result = call(entry.tool, resolve_call(entry, core.current["observation_id"],
                          core.program_token, core.program_outputs))
            if result["status"] != "completed":
                report.update(status="stopped", termination_reason=(
                    "commissioning_step_cap" if cancelled() else "gateway_rejected_or_interrupted"))
                break
            if entry.tool == "arx.execute_grasp":
                outcome = result["result"] or {}
                report.update(status="stopped", termination_reason="pregrasp_unverified")
                if outcome.get("command_target_reached") and outcome.get("physical_arrival_verified"):
                    report.update(status="pregrasp_verified", termination_reason="measured_pregrasp")
                break
        report.update(physical_steps=core.step_index, final_observation=core.current,
                      final_features=getattr(core.critic, "last_feature_evidence", None))
        return report
    except Exception as exc:
        report.update(status="failed_keep_enabled", error=str(exc))
        if core is not None and core.current is not None:
            report.update(physical_steps=core.step_index, final_observation=core.current,
                          final_features=getattr(core.critic, "last_feature_evidence", None))
        raise
    finally:
        if core is not None:
            if not core.closed:
                if core.current is None:
                    core.backend.close()
                else:
                    core.close()
            core.journal.close()
        (args.output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "grasp-config", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--max-physical-steps", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute or not 2 <= args.max_physical_steps <= 301:
        parser.error("supervised motion requires --execute and a step cap in 2..301")
    report = run(args)
    print(json.dumps({"status": report["status"], "termination_reason": report["termination_reason"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Supervised commissioning of a frozen bundle prefix through the real gateway.

By default no VLA call is made. A one-step hold lets the real bundle critic
trigger; the bundle's ordered recovery prefix through pregrasp is executed.
--resume-vla-once also admits its reviewed reentry and one fresh VLA chunk. The optional
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

from robots.arx.gateway.real_factory import RealCoreFactory
from robots.arx.deployment.bundle_program import resolve_call
from robots.arx.gateway.contracts import ToolRequest
from zetta.evolution.jsonio import file_sha256


def run(args):
    root = Path(__file__).resolve().parents[2]
    provider = root / "robots/arx/deployment/picktube_rgbd_provider.py"
    settings = json.loads(args.grasp_config.read_text())
    if (
        (
            settings.get("learned_pregrasp_commissioning") is True
            or settings.get("learned_grasp_commissioning") is True
        )
        and not settings.get("learned_gripper_transfer_verified", False)
        and getattr(args, "resume_vla_once", False)
    ):
        raise ValueError("unverified learned commissioning cannot resume VLA")
    args.output.mkdir(parents=True, exist_ok=False)
    gateway = args.output / "private/gateway"
    gateway.mkdir(parents=True)
    core = None
    results = []
    report = {
        "schema_version": "arx.grasp.commission.v2",
        "status": "initializing",
        "termination_reason": "commissioning",
        "vla_called": False,
        "controller_disabled": False,
        "max_physical_steps": args.max_physical_steps,
        "bundle_sha256": file_sha256(args.bundle),
        "grasp_config_sha256": file_sha256(args.grasp_config),
    }

    def cancelled():
        return core is not None and core.step_index >= args.max_physical_steps

    retained = set()

    def phase_changed(phase):
        if (
            phase == "inference"
            and core is not None
            and getattr(args, "resume_vla_once", False)
        ):
            report["vla_called"] = True
        if (
            phase == "critic"
            and getattr(args, "retain_step_sensors", False)
            and core is not None
            and core.current is not None
            and core.current["observation_id"] not in retained
        ):
            core._retain_grasp_sensors()
            retained.add(core.current["observation_id"])

    factory = RealCoreFactory(
        hardware_config=str(
            getattr(args, "hardware_config", None)
            or root / "robots/arx/manifests/real/dodo_picktube_hardware.json"
        ),
        hardware_sha256=args.hardware_sha256,
        task=str(root / "robots/arx/manifests/pickup_test_tube.yaml"),
        model_contract=str(root / "robots/arx/manifests/task7_model_a.yaml"),
        output=str(gateway),
        episode_id="commission-" + uuid.uuid4().hex,
        limits=dict(
            max_steps=600,
            max_decisions=16,
            max_recoveries=1,
            operation_timeout_s=120.0,
            critic_timeout_s=5.0,
            idle_agent_timeout_s=120.0,
            lease_timeout_s=120.0,
            shutdown_timeout_s=10.0,
        ),
        zeva_host="127.0.0.1",
        zeva_port=5583,
        kinematics_calibration=str(
            root / "robots/arx/manifests/real/dodo_right_controller_ee_fk.json"
        ),
        bundle=str(args.bundle),
        tool_catalog=str(args.frozen / "tool-catalog.json"),
        real_input_contract=str(args.frozen / "real-input-contract.json"),
        expected_real_input_sha256=file_sha256(
            args.frozen / "real-input-contract.json"
        ),
        feature_provider=str(provider),
        expected_feature_provider_sha256=file_sha256(provider),
        allow_learned_pregrasp_commissioning=True,
        grasp_config=str(args.grasp_config),
        expected_grasp_config_sha256=file_sha256(args.grasp_config),
    )

    def call(tool, arguments):
        request = ToolRequest(
            request_id="call-" + uuid.uuid4().hex,
            decision_ref="decision-" + uuid.uuid4().hex,
            observation_id=core.current["observation_id"],
            control_epoch=core.epoch,
            tool=tool,
            arguments=arguments,
            evidence_ids=[core.current["observation_id"]],
            reason="Supervised frozen bundle grasp commissioning",
        )
        core.journal.register_decision(
            request, source="runner", evidence=request.evidence_ids
        )
        result = core.execute(request)
        results.append(result)
        (args.output / "calls.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "tool": tool,
                    "status": result["status"],
                    "steps": result["executed_steps"],
                    "result": result["result"],
                    "error": result["error"],
                }
            ),
            flush=True,
        )
        return result

    try:
        core = factory(cancelled, phase_changed)
        # Inspect the complete authorized commissioning prefix before motion.
        # Inspect all programs before reset or the first hold sends a command.
        settings = json.loads(args.grasp_config.read_text())
        full_grasp = getattr(args, "full_grasp", False)
        if full_grasp != (settings.get("learned_grasp_commissioning") is True):
            raise ValueError(
                "full grasp commissioning requires matching explicit configuration"
            )
        learned_commissioning = (
            settings.get("learned_pregrasp_commissioning") is True or full_grasp
        )
        report["learned_grasp_commissioning"] = full_grasp
        report["learned_pregrasp_commissioning"] = learned_commissioning
        for program in core.programs.values():
            pregrasp_indices = [
                i
                for i, c in enumerate(program.calls)
                if c.tool == "arx.execute_grasp"
                and c.arguments.get("phase", "pregrasp") == "pregrasp"
            ]
            if not pregrasp_indices:
                raise ValueError("bundle has no pregrasp commissioning prefix")
            final_pregrasp = pregrasp_indices[-1]
            end = (
                next(
                    (
                        i
                        for i, c in enumerate(program.calls)
                        if c.tool == "arx.review_reentry"
                    ),
                    len(program.calls),
                )
                if full_grasp
                else final_pregrasp + 1
            )
            if full_grasp:
                prefix = program.calls[:end]
                signature = [(c.tool, c.arguments.get("phase")) for c in prefix]
                required = [
                    ("arx.hold", None),
                    ("arx.propose_grasp", None),
                    ("arx.review_grasp", "pregrasp"),
                    ("arx.execute_grasp", "pregrasp"),
                    ("arx.review_grasp", "engage"),
                    ("arx.execute_grasp", "engage"),
                    ("arx.set_gripper", None),
                    ("arx.review_grasp", "lift"),
                    ("arx.execute_grasp", "lift"),
                    ("arx.hold", None),
                ]
                if signature != required or prefix[6].arguments.get("opening") != 0.0:
                    raise ValueError(
                        "full grasp prefix requires ordered pregrasp, engage, close, contact-gated lift and hold"
                    )
            reached_pregrasp = False
            for entry in program.calls[:end]:
                allowed = (
                    entry.tool == "arx.hold"
                    and entry.arguments.get("steps", 1) <= 15
                    or entry.tool == "arx.propose_grasp"
                    and (
                        entry.arguments.get("engine", "tube_geometry")
                        == "tube_geometry"
                        or learned_commissioning
                        and entry.arguments.get("engine") == "graspgen"
                    )
                    or entry.tool == "arx.set_gripper"
                    and (
                        entry.arguments.get("opening") == 1.0
                        or full_grasp
                        and entry.arguments.get("opening") == 0.0
                    )
                    or entry.tool in {"arx.review_grasp", "arx.execute_grasp"}
                    and (
                        entry.arguments.get("phase", "pregrasp") == "pregrasp"
                        or full_grasp
                    )
                )
                if not allowed:
                    raise ValueError(
                        "commissioning admits only open geometric pregrasp calls"
                    )
                reached_pregrasp = reached_pregrasp or entry.tool == "arx.execute_grasp"
            if not reached_pregrasp:
                raise ValueError("bundle has no pregrasp commissioning prefix")
            if getattr(args, "resume_vla_once", False):
                suffix = program.calls[final_pregrasp + 1 :]
                if [c.tool for c in suffix] != [
                    "arx.review_reentry",
                    "arx.zeva",
                ] or suffix[-1].arguments.get("max_chunks") != 1:
                    raise ValueError(
                        "commissioning reentry requires one review and one VLA chunk"
                    )
        core.reset()
        result = call("arx.hold", {"steps": 1})
        if result["status"] != "interrupted" or core.recovery is None:
            raise ValueError("commissioning critic did not interrupt the hold")
        program = core.programs[core.recovery["binding_id"]]
        final_pregrasp = max(
            i
            for i, c in enumerate(program.calls)
            if c.tool == "arx.execute_grasp"
            and c.arguments.get("phase", "pregrasp") == "pregrasp"
        )
        for index, entry in enumerate(program.calls):
            if entry.tool in {"arx.review_reentry", "arx.zeva"} and not getattr(
                args, "resume_vla_once", False
            ):
                break
            if (
                entry.tool == "arx.execute_grasp"
                and entry.arguments.get("phase") != "pregrasp"
                and not full_grasp
            ):
                raise ValueError("commissioning supports pregrasp only")
            result = call(
                entry.tool,
                resolve_call(
                    entry,
                    core.current["observation_id"],
                    core.program_token,
                    core.program_outputs,
                ),
            )
            if result["status"] != "completed":
                report.update(
                    status="stopped",
                    termination_reason=(
                        "commissioning_step_cap"
                        if cancelled()
                        else (result.get("result") or {}).get("completion")
                        or "gateway_rejected_or_interrupted"
                    ),
                )
                break
            if (result.get("result") or {}).get("completion") == "task_success":
                report.update(
                    status="grasp_lift_verified", termination_reason="task_success"
                )
                break
            if (
                entry.tool == "arx.execute_grasp"
                and entry.arguments.get("phase", "pregrasp") == "pregrasp"
            ):
                outcome = result["result"] or {}
                report.update(
                    status="stopped", termination_reason="pregrasp_unverified"
                )
                evidence = core.critic.last_feature_evidence
                features = evidence["features"]
                settings = json.loads(args.grasp_config.read_text())
                target_verified = (
                    evidence.get("feature_observation", {}).get("status") == "observed"
                    and features["privileged.interaction.gripper_closed"] is False
                    and features["privileged.interaction.gripper_contact"] is False
                    and features["privileged.selected.target_gripper_distance_m"]
                    <= settings["pregrasp_distance_m"] + settings["pose_tolerance_m"]
                )
                if (
                    outcome.get("command_target_reached")
                    and outcome.get("physical_arrival_verified")
                    and target_verified
                ):
                    report.update(
                        status="pregrasp_verified",
                        termination_reason="measured_pregrasp",
                    )
                if index == final_pregrasp and (
                    not target_verified
                    or not (full_grasp or getattr(args, "resume_vla_once", False))
                ):
                    break
            elif entry.tool == "arx.review_reentry":
                report["reentry_verified"] = (
                    result["result"].get("assessment", {}).get("status") == "eligible"
                )
            elif entry.tool == "arx.zeva":
                report.update(
                    status="reentry_vla_verified",
                    termination_reason=(
                        "task_success"
                        if result["result"]["completion"] == "task_success"
                        else "bounded_vla_chunk_complete"
                    ),
                )
                break
        report.update(
            physical_steps=core.step_index,
            final_observation=core.current,
            final_features=getattr(core.critic, "last_feature_evidence", None),
        )
        return report
    except Exception as exc:
        report.update(status="failed_keep_enabled", error=str(exc))
        if core is not None and core.current is not None:
            report.update(
                physical_steps=core.step_index,
                final_observation=core.current,
                final_features=getattr(core.critic, "last_feature_evidence", None),
            )
        raise
    finally:
        if core is not None:
            if not core.closed:
                if core.current is None:
                    core.backend.close()
                else:
                    core.close()
            core.journal.close()
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "grasp-config", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--retain-step-sensors", action="store_true")
    parser.add_argument("--resume-vla-once", action="store_true")
    parser.add_argument(
        "--full-grasp",
        action="store_true",
        help="Supervised contact-gated closure and lift; no VLA",
    )
    parser.add_argument("--max-physical-steps", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute or not 2 <= args.max_physical_steps <= 600:
        parser.error("supervised motion requires --execute and a step cap in 2..600")
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "termination_reason": report["termination_reason"],
            }
        )
    )


if __name__ == "__main__":
    main()

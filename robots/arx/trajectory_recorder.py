"""Publish ARX RGB evidence without exporting private gateway provenance."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from zetta.evolution.evidence_policy import ArxRgbPublicEvidencePolicy
from zetta.evolution.jsonio import file_sha256
from zetta.evolution.visual_artifacts import label_event_steps, label_overview_steps


def record_public_trajectory(attempt: Path) -> dict[str, Any]:
    """Read only public observations and image bytes; reject unknown ABI fields."""
    policy = ArxRgbPublicEvidencePolicy()
    source = attempt / "private/gateway/journal.sqlite3"
    if not source.is_file():
        raise ValueError("gateway did not seal its observation journal")
    trajectory = attempt / "trajectory"
    trajectory.mkdir(exist_ok=True)
    frames = attempt / "frames"
    frames.mkdir(exist_ok=True)
    digests: dict[str, str] = {}
    observations = []
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT payload FROM records WHERE kind='ObservationPublished' AND public=1 ORDER BY sequence").fetchall()
        tool_rows = connection.execute("SELECT payload FROM records WHERE kind='tool_result' AND public=1 ORDER BY sequence").fetchall()
    finally:
        connection.close()
    for (payload,) in rows:
        value = json.loads(payload)
        required = {"schema_version", "episode_nonce", "observation_id", "step_index", "simulation_time_s", "lifecycle", "cameras"}
        if not required <= set(value):
            raise ValueError("unexpected gateway observation schema")
        # Privileged observations are authorized private evidence; strip them
        # from the RGB/public trajectory while retaining them in a separate
        # attempt-local artifact below.
        public_value = {key: value[key] for key in required}
        value = public_value
        if not isinstance(value["step_index"], int) or value["step_index"] < 0:
            raise ValueError("invalid public observation step")
        references = {}
        for camera, ref in value["cameras"].items():
            if not camera.isidentifier() or set(ref) != {"content_id", "sha256", "width", "height", "encoding"} or ref["encoding"] != "png":
                raise ValueError("unexpected public image reference")
            content_id = ref["content_id"]
            if not isinstance(content_id, str) or not content_id.startswith("rgb-") or len(content_id) != 68 or not all(x in "0123456789abcdef" for x in content_id[4:]):
                raise ValueError("invalid image content ID")
            origin = attempt / "private/gateway/public/images" / (content_id + ".png")
            if origin.is_symlink() or not origin.is_file() or origin.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("missing or invalid public RGB frame")
            target = frames / (content_id + ".png")
            if not target.exists():
                shutil.copyfile(origin, target)
            digest = file_sha256(target)
            if digest != file_sha256(origin) or digest != ref["sha256"] or content_id != "rgb-" + digest:
                raise ValueError("public RGB frame changed during publication")
            digests[str(target.relative_to(attempt))] = digest
            references[camera] = content_id
        event = {"step_index": value["step_index"], "observation_id": value["observation_id"],
                 "rgb_frame_id": next(iter(references.values())),
                 "events": [{"camera": camera, "rgb_frame_id": content_id}
                            for camera, content_id in sorted(references.items())]}
        policy.validate_public_event(event)
        observations.append(event)
    if not observations:
        raise ValueError("public RGB observation stream is empty")
    observations.sort(key=lambda item: int(item["step_index"]))
    tools = []
    for (payload,) in tool_rows:
        value = json.loads(payload)
        if not isinstance(value, dict) or not isinstance(value.get("status"), str):
            raise ValueError("unexpected public tool result")
        tool = value.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise ValueError("unexpected public tool name")
        row = {"step_index": len(tools), "status": value["status"]}
        if tool is not None:
            row["tool"] = tool
        tools.append(policy.publish_event(row))
    privileged_path = attempt / "privileged_observations.jsonl"
    # Read private rows in a fresh read-only connection.
    privileged_rows = []
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        privileged_rows = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM records WHERE kind='PrivilegedObservation' ORDER BY sequence").fetchall()]
    finally:
        connection.close()
    privileged_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in privileged_rows), encoding="utf-8")
    digests[str(privileged_path.relative_to(attempt))] = file_sha256(privileged_path)
    streams = {
        "chunks": [{"step_index": event["step_index"], "rgb_frame_id": event["rgb_frame_id"]} for event in observations],
        "actions": [{"step_index": event["step_index"], "rgb_frame_id": event["rgb_frame_id"]} for event in observations[1:]],
        "states": observations,
        "tools": tools,
    }
    for name, rows in streams.items():
        path = trajectory / (name + ".jsonl")
        path.write_text("".join(json.dumps(policy.publish_event(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        digests[str(path.relative_to(attempt))] = file_sha256(path)
    # Keep all three ARX views grouped by timestep.  The primary frame ID is
    # retained for compatibility, but diagnosis reads this manifest to avoid
    # treating the three camera streams as one unordered image sequence.
    steps = [int(item["step_index"]) for item in observations]
    overview_steps = label_overview_steps(steps)
    event_rows = list(observations)
    for row in privileged_rows:
        if not isinstance(row, dict):
            continue
        privileged = row.get("privileged", {})
        interaction = privileged.get("interaction", {}) if isinstance(privileged, dict) else {}
        contact = privileged.get("contact_summary", {}) if isinstance(privileged, dict) else {}
        event_rows.append({"step_index": row.get("step_index"), "state": {
            "privileged.task.success": interaction.get("success"),
            "privileged.task.goal.progress": interaction.get("progress"),
            "privileged.task.manipulated_object.grasped": interaction.get("grasped"),
            "privileged.task.manipulated_object.retained": interaction.get("retained"),
            "privileged.task.manipulated_object.in_target": interaction.get("in_target"),
            "privileged.task.stage.name": interaction.get("stage"),
            "privileged.contact.gripper.count": contact.get("gripper_count"),
        }})
    event_steps = label_event_steps(event_rows)
    visual_manifest = {
        "schema_version": "arx.visual_timestep_manifest.v1",
        "camera_order": sorted({event["camera"] for item in observations for event in item["events"]}),
        "timesteps": [
            {"step_index": int(item["step_index"]), "observation_id": item["observation_id"],
             "views": [{"camera": event["camera"], "rgb_frame_id": event["rgb_frame_id"],
             "path": f"frames/step-{int(item['step_index']):06d}-{event['camera']}.png"}
                       for event in item["events"]]}
            for item in observations
        ],
        "labels": {"overview_steps": overview_steps, "event_steps": event_steps},
        "ordering": "ascending step_index; views are grouped within each timestep",
    }
    # Human/file-browser friendly names preserve the chronological order while
    # the content-addressed originals above remain the integrity anchors.
    for item in observations:
        for event in item["events"]:
            source_frame = frames / f"{event['rgb_frame_id']}.png"
            ordered_frame = frames / f"step-{int(item['step_index']):06d}-{event['camera']}.png"
            if not ordered_frame.exists():
                shutil.copyfile(source_frame, ordered_frame)
            if file_sha256(ordered_frame) != file_sha256(source_frame):
                raise ValueError("ordered RGB frame changed during publication")
            digests[str(ordered_frame.relative_to(attempt))] = file_sha256(ordered_frame)
    visual_path = attempt / "visual-timestep-manifest.json"
    visual_path.write_text(json.dumps(visual_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digests[str(visual_path.relative_to(attempt))] = file_sha256(visual_path)
    return {"artifacts": [{"path": name, "sha256": sha} for name, sha in sorted(digests.items())]}

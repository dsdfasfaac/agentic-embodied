"""Author and preflight immutable ARX RGB candidate packages.

This boundary accepts learner text, never executable paths from the learner.
Feature code is parsed by the registry and run only by the isolated critic worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from robots.arx.critics.contracts import WorkerLimits
from robots.arx.critics.packages import PackageManifest, seal_candidate
from robots.arx.critics.registry import validate_candidate


REQUIRED_FILES = frozenset({
    "critic/manifest.json", "critic/config.json", "critic/features.py",
    "critic/feature_schema.json", "skill/SKILL.md", "recovery_bindings.json",
    "reentry/manifest.json", "reentry/config.json", "evidence/claims.json",
    "tests/replay_cases.json",
})


def author_package(
    destination: Path, *, files: Mapping[str, str], metadata: Mapping[str, object],
    parent_sha256: str | None, generation: int, limits: WorkerLimits,
):
    """Seal one fresh draft after enforcing parent, RGB and sandbox contracts."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("candidate draft directory already exists")
    if not isinstance(files, Mapping) or not REQUIRED_FILES <= files.keys():
        raise ValueError("candidate is missing required package files")
    if len(files) > 256:
        raise ValueError("candidate has too many files")
    if metadata.get("parent_package_sha256") != parent_sha256 or metadata.get("generation") != generation:
        raise ValueError("candidate parent or generation does not match campaign")
    if "manifest.json" in files:
        raise ValueError("package manifest is harness-generated")
    validated = []
    for name, contents in files.items():
        path = PurePosixPath(name)
        if (not isinstance(name, str) or path.is_absolute() or ".." in path.parts
                or str(path) != name or not isinstance(contents, str)
                or len(contents.encode("utf-8")) > 4 * 1024 * 1024):
            raise ValueError("invalid candidate file")
        if name.endswith(".json"):
            json.loads(contents)
        validated.append((path.parts, contents))
    # A rejected learner response must not strand an unsealed path which then
    # makes the next deterministic Stage2 attempt unrecoverable.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for parts, contents in validated:
            target = staging.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        package = seal_candidate(staging, dict(metadata))
        manifest = PackageManifest.model_validate_json(package.manifest_bytes)
        if manifest.parent_package_sha256 != parent_sha256 or manifest.generation != generation:
            raise ValueError("sealed candidate parent or generation changed")
        validate_candidate(package, limits)
        replay = package.json("tests/replay_cases.json")
        if not isinstance(replay, dict) or not isinstance(replay.get("cases"), list) or not replay["cases"]:
            raise ValueError("candidate must provide nonempty replay cases")
        os.replace(staging, destination)
        return package
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise


def replay_package(path: Path, limits: WorkerLimits) -> dict[str, object]:
    """Run declared RGB sequences against the actual isolated critic worker.

    The current replay ABI supports constant-red test frames; other fixtures
    must be rejected until they have a defined, independently verifiable image
    encoding. A learner's expected firing steps are assertions, not evidence.
    """
    import numpy as np

    from robots.arx.critics.registry import ArxCriticRegistry

    registry = ArxCriticRegistry(limits=limits)
    package = registry.register_package(path)
    registry.preflight()
    cases = package.json("tests/replay_cases.json")
    if not isinstance(cases, dict) or not isinstance(cases.get("cases"), list) or not cases["cases"]:
        raise ValueError("replay cases are missing")
    if cases.get("shape") != [limits.image_height, limits.image_width, 3]:
        raise ValueError("replay frame shape does not match frozen worker limits")
    checked = []
    for case in cases["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("red"), list)
                or not case["red"] or not all(type(x) is int and 0 <= x <= 255 for x in case["red"])
                or not isinstance(case.get("fires"), list)):
            raise ValueError("unsupported or empty RGB replay case")
        # Each replay needs fresh temporal state. Never reuse a frozen worker.
        fresh = ArxCriticRegistry(limits=limits)
        fresh.register_package(path)
        critic = fresh.freeze()
        observed = []
        try:
            for step, red in enumerate(case["red"]):
                images = {name: np.zeros((limits.image_height, limits.image_width, 3), dtype=np.uint8)
                          for name in package.critic_manifest.cameras}
                for image in images.values():
                    image[:, :, 0] = red
                refs = {name: {"content_id": f"replay-{step}", "sha256": "0" * 64,
                               "width": limits.image_width, "height": limits.image_height,
                               "encoding": "png"} for name in images}
                observation = {"schema_version": "arx.critic.observation.v1", "episode_nonce": "replay",
                               "observation_id": f"obs-{step}", "step_index": step,
                               "simulation_time_s": step / 15, "cameras": refs,
                               "lifecycle": "reset" if step == 0 else "nominal", "event_sequence": step}
                if step == 0:
                    critic.reset(observation, images)
                elif critic.observe(observation, images).events:
                    observed.append(step)
        finally:
            critic.close()
        if observed != case["fires"]:
            raise ValueError(f"RGB replay mismatch: {case.get('name', 'unnamed')}")
        checked.append({"name": case.get("name"), "fires": observed})
    return {"candidate_sha256": package.sha256, "cases": checked, "eligible": True}


def replay_public_episodes(
    path: Path, limits: WorkerLimits, episodes: list[tuple[Path, bool]],
) -> dict[str, object]:
    """Measure detector activation against immutable published RGB episodes.

    This is observational shadow replay: no recovery is executed and no
    simulator state or evaluator fields are passed to the feature worker.
    """
    import io

    import numpy as np
    from PIL import Image

    from robots.arx.critics.registry import ArxCriticRegistry
    from zetta.evolution.evidence_policy import ArxRgbPublicEvidencePolicy
    from zetta.evolution.jsonio import file_sha256

    policy = ArxRgbPublicEvidencePolicy()
    if not episodes:
        raise ValueError("shadow replay needs published RGB episodes")
    outcomes = []
    for attempt, success in episodes:
        states = attempt / "trajectory/states.jsonl"
        if not states.is_file() or states.is_symlink():
            raise ValueError("shadow replay missing public state rows")
        registry = ArxCriticRegistry(limits=limits)
        package = registry.register_package(path)
        critic = registry.freeze()
        fires = []
        try:
            rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                raise ValueError("shadow replay has empty public state rows")
            for position, row in enumerate(rows):
                policy.validate_public_event(row)
                references = {event["camera"]: event["rgb_frame_id"] for event in row.get("events", ())}
                if not set(package.critic_manifest.cameras) <= set(references):
                    raise ValueError("shadow replay camera set differs from package")
                images = {}
                cameras = {}
                for camera in package.critic_manifest.cameras:
                    image_id = references[camera]
                    image_path = attempt / "frames" / (image_id + ".png")
                    if (not image_id.startswith("rgb-") or image_path.is_symlink()
                            or not image_path.is_file() or file_sha256(image_path) != image_id[4:]):
                        raise ValueError("shadow replay RGB frame missing or changed")
                    with Image.open(io.BytesIO(image_path.read_bytes())) as frame:
                        images[camera] = np.asarray(frame.convert("RGB")).copy()
                    height, width = images[camera].shape[:2]
                    if (width, height) != (limits.image_width, limits.image_height):
                        raise ValueError("shadow replay camera geometry differs from worker limits")
                    cameras[camera] = {"content_id": image_id, "sha256": image_id[4:],
                                       "width": width, "height": height, "encoding": "png"}
                observation = {"schema_version": "arx.critic.observation.v1",
                               "episode_nonce": "public-shadow", "observation_id": row["observation_id"],
                               "step_index": row["step_index"], "simulation_time_s": position / 15,
                               "cameras": cameras, "lifecycle": "reset" if position == 0 else "nominal",
                               "event_sequence": position}
                if position == 0:
                    critic.reset(observation, images)
                elif critic.observe(observation, images).events:
                    fires.append(row["step_index"])
        finally:
            critic.close()
        outcomes.append({"states_sha256": file_sha256(states), "success_control": success,
                         "fires": fires})
    failures = [row for row in outcomes if not row["success_control"]]
    controls = [row for row in outcomes if row["success_control"]]
    return {"candidate_sha256": package.sha256, "outcomes": outcomes,
            "target_count": len(failures), "target_detected": sum(bool(row["fires"]) for row in failures),
            "success_control_count": len(controls),
            "success_control_false_positives": sum(bool(row["fires"]) for row in controls),
            "preflight_conclusive": bool(controls and failures)}

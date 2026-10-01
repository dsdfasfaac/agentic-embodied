"""Package authoring is independent of scalar CandidateBundle schemas."""

import json
import hashlib
import io
from pathlib import Path

import pytest

from robots.arx.evolution_candidate_adapter import REQUIRED_FILES, author_package, replay_package
from robots.arx.evolution_candidate_adapter import replay_public_episodes
from robots.arx.critics.packages import seal_candidate
from robots.arx.critics.registry import RgbFeatureReentry, validate_candidate
from robots.arx.gateway.contracts import ReviewArgs
from tests.test_arx_critics import FIXTURE, limits
from PIL import Image


def fixture_input():
    files = {name: (FIXTURE / name).read_text() for name in REQUIRED_FILES}
    metadata = json.loads((FIXTURE / "manifest.json").read_text())
    metadata.pop("files")
    metadata.pop("feature_schema_sha256")
    return files, metadata


def test_author_valid_rgb_package(tmp_path):
    files, metadata = fixture_input()
    package = author_package(tmp_path / "candidate", files=files, metadata=metadata,
                             parent_sha256=None, generation=0, limits=limits())
    assert len(package.sha256) == 64
    assert (tmp_path / "candidate/manifest.json").is_file()
    report = replay_package(tmp_path / "candidate", limits())
    assert report["eligible"] and len(report["cases"]) == 3


def test_reject_parent_and_path_traversal(tmp_path):
    files, metadata = fixture_input()
    with pytest.raises(ValueError, match="parent"):
        author_package(tmp_path / "wrong", files=files, metadata=metadata,
                       parent_sha256="a" * 64, generation=0, limits=limits())
    files["../escape.py"] = "pass"
    with pytest.raises(ValueError, match="invalid candidate file"):
        author_package(tmp_path / "traversal", files=files, metadata=metadata,
                       parent_sha256=None, generation=0, limits=limits())


def test_shadow_replay_uses_published_rgb_frames(tmp_path):
    files, metadata = fixture_input()
    candidate = tmp_path / "candidate"
    author_package(candidate, files=files, metadata=metadata,
                   parent_sha256=None, generation=0, limits=limits())
    attempt = tmp_path / "attempt"
    (attempt / "trajectory").mkdir(parents=True)
    (attempt / "frames").mkdir()
    rows = []
    for step, red in enumerate([0, 0, 255, 255, 0]):
        stream = io.BytesIO()
        Image.new("RGB", (320, 240), (red, 0, 0)).save(stream, format="PNG")
        payload = stream.getvalue()
        image_id = "rgb-" + hashlib.sha256(payload).hexdigest()
        (attempt / "frames" / (image_id + ".png")).write_bytes(payload)
        rows.append({"step_index": step, "observation_id": f"obs-{step}",
                     "rgb_frame_id": image_id,
                     "events": [{"camera": name, "rgb_frame_id": image_id}
                                for name in ("front_rgb", "left_rgb", "right_rgb")]})
    (attempt / "trajectory/states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    report = replay_public_episodes(candidate, limits(), [(attempt, False)])
    assert report["target_detected"] == 1
    assert report["outcomes"][0]["fires"] == [3]


def test_rgb_reentry_can_resume_after_public_feature_clearance(tmp_path):
    files, metadata = fixture_input()
    files["reentry/manifest.json"] = json.dumps({
        "schema_version": "arx.reentry.manifest.v1", "implementation": "rgb_feature_threshold",
        "policy_id": "rgb-clearance"})
    files["reentry/config.json"] = json.dumps({
        "feature": "visual.mean_red", "operator": "le", "threshold": 0.2,
        "minimum_observations": 2})
    bindings = json.loads(files["recovery_bindings.json"])
    bindings[0]["reentry_policy_id"] = "rgb-clearance"
    files["recovery_bindings.json"] = json.dumps(bindings)
    package = author_package(tmp_path / "reentry", files=files, metadata=metadata,
                             parent_sha256=None, generation=0, limits=limits())
    validate_candidate(package, limits())
    import numpy as np
    images = [{"front_rgb": np.full((240, 320, 3), red, np.uint8)} for red in (255, 0, 0)]
    ids = ["obs-0", "obs-1", "obs-2"]
    context = {"observations": [{"observation_id": obs_id} for obs_id in ids],
               "images": images, "recovery_id": "recovery-1", "policy_id": "rgb-clearance"}
    review = RgbFeatureReentry(package, limits())
    assert review.inspect(ReviewArgs(observation_ids=ids), context)["status"] == "eligible"
    context["images"][-1] = {"front_rgb": np.full((240, 320, 3), 255, np.uint8)}
    assert review.inspect(ReviewArgs(observation_ids=ids), context)["status"] == "ineligible"

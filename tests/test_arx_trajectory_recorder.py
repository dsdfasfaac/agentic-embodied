import hashlib
import io
import json
import sqlite3

import pytest
from PIL import Image

from robots.arx.evolution_result import authoritative_success
from robots.arx.trajectory_recorder import record_public_trajectory


def fixture(attempt):
    gateway = attempt / "private/gateway"
    images = gateway / "public/images"
    images.mkdir(parents=True)
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 0, 0)).save(stream, format="PNG")
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    content_id = "rgb-" + sha
    (images / (content_id + ".png")).write_bytes(payload)
    observation = {"schema_version": "arx.public.observation.v1", "episode_nonce": "private-nonce",
                   "observation_id": "obs-0", "step_index": 0, "simulation_time_s": 0,
                   "lifecycle": "reset", "event_sequence": 1,
                   "cameras": {"front": {"content_id": content_id, "sha256": sha, "width": 8, "height": 8, "encoding": "png"}}}
    db = sqlite3.connect(gateway / "journal.sqlite3")
    db.execute("CREATE TABLE records(sequence INTEGER,kind TEXT,public INTEGER,payload TEXT)")
    db.execute("INSERT INTO records VALUES(1,'ObservationPublished',1,?)", (json.dumps(observation),))
    db.execute("INSERT INTO records VALUES(2,'private_evaluation',0,?)", (json.dumps({"evaluation": {"evaluation": {"success": False, "target_pose": [1, 2, 3]}}}),))
    db.commit()
    db.close()
    return gateway


def test_public_recorder_does_not_publish_private_journal(tmp_path):
    gateway = fixture(tmp_path)
    index = record_public_trajectory(tmp_path)
    assert all("private" not in artifact["path"] for artifact in index["artifacts"])
    assert authoritative_success(gateway / "journal.sqlite3") is False
    states = (tmp_path / "trajectory/states.jsonl").read_text()
    assert "private-nonce" not in states
    assert "target_pose" not in states
    assert "rgb_frame_id" in states


def test_recorder_rejects_modified_image(tmp_path):
    gateway = fixture(tmp_path)
    image = next((gateway / "public/images").glob("*.png"))
    image.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="invalid public RGB"):
        record_public_trajectory(tmp_path)


def test_recorder_rejects_private_field_in_public_tool_result(tmp_path):
    gateway = fixture(tmp_path)
    db = sqlite3.connect(gateway / "journal.sqlite3")
    db.execute("INSERT INTO records VALUES(3,'tool_result',1,?)",
               (json.dumps({"status": "ok", "qpos": [1, 2]}),))
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="private ARX evidence field"):
        record_public_trajectory(tmp_path)

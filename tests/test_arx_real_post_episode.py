import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from scripts.deployment import finish_arx_real_episode as cleanup


def setup_trial(tmp_path, monkeypatch, *, held=False, home="complete", arrived=True):
    trial = tmp_path / "trial"
    db_dir = trial / "private/gateway"
    db_dir.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"termination_reason": "critic_interrupted"}))
    with sqlite3.connect(db_dir / "journal.sqlite3") as db:
        db.execute("CREATE TABLE records(sequence INTEGER,kind TEXT,payload TEXT)")
        features = {"privileged.interaction.grasped": held}
        db.execute("INSERT INTO records VALUES(1,'real_feature_evidence',?)", (json.dumps({"features": features}),))
    calls = []
    def audit(*args):
        calls.append("audit")
        return {"task_start_eligible": arrived,
                "features": {"privileged.interaction.gripper_closed": False,
                             "privileged.selected.target_gripper_distance_m": .4}}
    def stage(*args):
        calls.append("home")
        return {"status": home}
    def disable(*args, **kwargs):
        calls.append("disable")
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
    monkeypatch.setattr(cleanup, "audit", audit)
    monkeypatch.setattr(cleanup, "stage", stage)
    monkeypatch.setattr(cleanup.subprocess, "run", disable)
    args = dict(trial=trial, hardware=Path("hardware"), hardware_sha="sha",
                task=Path("task"), model=Path("model"), output=tmp_path / "cleanup",
                controller=Path("controller.sh"))
    return args, calls


def test_home_and_verify_precede_disable(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch)
    result = cleanup.finish(**args)
    assert calls == ["audit", "home", "audit", "disable"]
    assert result["status"] == "homed_and_disabled"
    assert result["homing_is_rollout_step"] is False


def test_homing_failure_keeps_controller_enabled(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, home="partial")
    result = cleanup.finish(**args)
    assert calls == ["audit", "home"]
    assert result["status"] == "failed_keep_enabled"
    assert result["controller_disabled"] is False


def test_bad_measured_home_keeps_controller_enabled(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, arrived=False)
    result = cleanup.finish(**args)
    assert calls == ["audit", "home", "audit"]
    assert result["controller_disabled"] is False


def test_held_tube_requires_unloading_before_home(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, held=True)
    result = cleanup.finish(**args)
    assert calls == []
    assert result["status"] == "requires_unloading"
    result = cleanup.finish(**{**args, "output": tmp_path / "after-unloading"}, operator_unloaded=True)
    assert result["status"] == "homed_and_disabled"
    assert result["operator_confirmed_unloaded"] is True

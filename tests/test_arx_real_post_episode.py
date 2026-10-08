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
    def audit(*args, **kwargs):
        calls.append("audit")
        return {"task_start_eligible": arrived,
                "features": {"privileged.interaction.gripper_closed": False,
                             "privileged.selected.target_gripper_distance_m": .4}}
    def replay(*args, **kwargs):
        calls.append("home")
        return {"status": home}
    def stage(*args):
        calls.append("open")
        return {"status": "complete"}
    def disable(*args, **kwargs):
        calls.append("disable")
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
    monkeypatch.setattr(cleanup, "audit", audit)
    monkeypatch.setattr(cleanup, "replay_home", replay)
    monkeypatch.setattr(cleanup, "stage", stage)
    monkeypatch.setattr(cleanup.subprocess, "run", disable)
    args = dict(trial=trial, hardware=Path("hardware"), hardware_sha="sha",
                task=Path("task"), model=Path("model"), output=tmp_path / "cleanup",
                controller=Path("controller.sh"))
    return args, calls


def test_home_and_verify_precede_disable(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch)
    result = cleanup.finish(**args)
    assert calls == ["audit", "home", "open", "audit", "disable"]
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
    assert calls == ["audit", "home", "open", "audit"]
    assert result["controller_disabled"] is False


def test_held_tube_requires_unloading_before_home(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, held=True)
    result = cleanup.finish(**args)
    assert calls == []
    assert result["status"] == "requires_unloading"
    result = cleanup.finish(**{**args, "output": tmp_path / "after-unloading"}, operator_unloaded=True)
    assert result["status"] == "homed_and_disabled"
    assert result["operator_confirmed_unloaded"] is True


def test_confirmed_unloading_does_not_require_target_visibility(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, held=True)
    scopes = []
    def hardware_audit(*args, require_features=True):
        scopes.append(require_features)
        if require_features:
            raise ValueError("pink tube label is not reliably visible")
        return {"task_start_eligible": True, "features": None,
                "observation_scope": "hardware_home"}
    monkeypatch.setattr(cleanup, "audit", hardware_audit)
    result = cleanup.finish(**args, operator_unloaded=True)
    assert scopes == [False, False]
    assert calls == ["home", "open", "disable"]
    assert result["status"] == "homed_and_disabled"


def test_hardware_failure_after_unloading_keeps_enabled(tmp_path, monkeypatch):
    args, calls = setup_trial(tmp_path, monkeypatch, held=True)
    def unavailable(*args, **kwargs):
        raise RuntimeError("arm device is not healthy")
    monkeypatch.setattr(cleanup, "audit", unavailable)
    result = cleanup.finish(**args, operator_unloaded=True)
    assert calls == []
    assert result["status"] == "failed_keep_enabled"
    assert result["controller_disabled"] is False


def test_observed_empty_open_checkpoint_does_not_need_cold_target_reacquisition(tmp_path, monkeypatch):
    import robots.arx.gateway.real_config as config_module
    args, calls = setup_trial(tmp_path, monkeypatch)
    state = [0.]*14; state[7] = .65; state[13] = -2.45
    (args["trial"] / "result.json").write_text(json.dumps({
        "termination_reason":"gateway_rejected_or_interrupted",
        "final_observation":{"hardware":{"measured_state":state}}}))
    evidence = {"observation_id":"terminal-observed", "feature_observation":{"status":"observed"},
        "features":{"privileged.interaction."+n:False
                    for n in ("gripper_closed","gripper_contact","grasped","success")}}
    with sqlite3.connect(args["trial"] / "private/gateway/journal.sqlite3") as db:
        db.execute("UPDATE records SET payload=?", (json.dumps(evidence),))
    monkeypatch.setattr(config_module,"load_real_hardware_config",lambda *a:SimpleNamespace(
        right_gripper_closed_policy=0.,right_gripper_open_policy=-3.4))
    def hardware_audit(*a, require_features=True):
        assert not require_features
        calls.append("audit")
        return {"measured_state":state, "features":None,"task_start_eligible":True,
                "hardware":{"auxiliary_feedback":{"right_gripper_current_native":.02}}}
    monkeypatch.setattr(cleanup,"audit",hardware_audit)
    result = cleanup.finish(**args)
    assert result["status"] == "homed_and_disabled"
    assert result["empty_home_admission"]["source_observation_id"] == "terminal-observed"
    assert calls == ["audit","home","open","audit","disable"]
    # A controller reset or closing after the checkpoint invalidates admission.
    calls.clear()
    def changed_audit(*a,**kw):
        result = hardware_audit(*a,**kw)
        result["measured_state"] = [0.]*14
        return result
    monkeypatch.setattr(cleanup,"audit",changed_audit)
    result = cleanup.finish(**{**args,"output":tmp_path/"changed"})
    assert result["status"] == "failed_keep_enabled"
    assert calls == ["audit"]

from copy import deepcopy

from scripts.evolution import audit_arx_real_evolution as audit


def pair(monkeypatch, *, success=False, mismatched=False):
    parent = {
        "identities": {name: name for name in (
            "hardware", "feature_provider", "catalog", "task", "model_contract", "runtime_limits")},
        "initial_state": [0.0] * 14,
        "initial_camera_calibration_sha256": {"front_rgb": "calibration"},
        "initial_budget": {"steps": 600, "recoveries": 4, "decisions": 64},
        "result": {"error": None, "counts": {"recoveries": 0, "physical_steps": 600}},
        "event_counts": {}, "verified_task_success": False,
        "distance_m": {"initial": .4, "minimum": .35, "final": .42},
    }
    parent["identities"]["bundle"] = "parent"
    candidate = deepcopy(parent)
    candidate["identities"]["bundle"] = "candidate"
    candidate["result"]["counts"].update(recoveries=1, physical_steps=80 if success else 600)
    candidate["verified_task_success"] = success
    if mismatched:
        candidate["initial_budget"]["steps"] = 1200
    monkeypatch.setattr(audit, "read_trial", lambda path: (parent if path == "parent" else candidate, []))
    return audit.compare("parent", "candidate")


def test_no_task_rescue_rejects_intervening_candidate(monkeypatch):
    result = pair(monkeypatch)
    assert result["decision"] == "reject"
    assert result["promoted"] is False


def test_early_success_can_rescue_but_one_pair_cannot_promote(monkeypatch):
    result = pair(monkeypatch, success=True)
    assert result["attributed_task_rescue"] is True
    assert result["decision"] == "continue_validation"
    assert result["promoted"] is False


def test_changed_execution_budget_invalidates_comparison(monkeypatch):
    result = pair(monkeypatch, success=True, mismatched=True)
    assert result["decision"] == "inconclusive"
    assert result["attributed_task_rescue"] is False

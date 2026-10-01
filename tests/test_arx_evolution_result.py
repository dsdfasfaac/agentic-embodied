from robots.arx.deployment.contracts import Counts, Outcome, RolloutResult
from robots.arx.evolution_result import convert_result


def record(status="completed", success=False):
    result = RolloutResult(
        trial_id="trial", attempt_id="attempt", mode="baseline", status=status,
        termination_reason="terminal", cleanup_status="complete",
        outcome=Outcome(task_success=success), counts=Counts(physical_steps=10),
    )
    return convert_result(result, logical_id="logical", attempt_index=0,
                          generation=0, seed=17, policy_rng=17,
                          started_at="2026-09-28T00:00:00Z", elapsed_s=1,
                          artifact_index={})


def test_valid_failure_has_unknown_divergence_segment():
    value = record()
    assert value.status == "valid"
    assert value.failure_segment.earliest_divergence_step is None
    assert value.failure_segment.end_step == 10


def test_missing_authoritative_outcome_is_invalid():
    value = record(success=None)
    assert value.status == "infra_invalid"
    assert value.success is None


def test_uncertain_execution_is_not_retry_safe():
    value = record(status="execution_uncertain", success=None)
    assert value.status == "infra_invalid"
    assert value.artifact_index["retry_safe"] is False

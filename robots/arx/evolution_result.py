"""Convert sealed ARX outcomes into the shared episode contract."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any

from robots.arx.deployment.contracts import RolloutResult
from zetta.evolution.models import EpisodeRecord, FailureSegment


def authoritative_success(journal_path: Path) -> bool | None:
    """The harness alone reads private evaluator rows after gateway shutdown."""
    if not journal_path.is_file():
        return None
    connection = sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT payload FROM records WHERE kind='private_evaluation' AND public=0 ORDER BY sequence DESC LIMIT 1"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return None
    value = json.loads(rows[0][0])
    evaluation = value.get("evaluation", {}).get("evaluation", {})
    success = evaluation.get("success")
    if not isinstance(success, bool):
        raise ValueError("private evaluator row has no authoritative boolean")
    return success


def convert_result(
    result: RolloutResult, *, logical_id: str, attempt_index: int,
    generation: int, seed: int, policy_rng: int, started_at: str,
    elapsed_s: float, artifact_index: dict[str, Any],
) -> EpisodeRecord:
    """Integrity metadata stays in the harness; artifacts must already be public.

    A completed runner is not sufficient: only an authoritative evaluator
    boolean makes an attempt valid. The caller must seal public artifacts
    before invoking this function.
    """
    valid = result.status == "completed" and isinstance(result.outcome.task_success, bool)
    success = result.outcome.task_success if valid else None
    episode_id = result.episode_id or result.attempt_id
    segments: tuple[FailureSegment, ...] = ()
    if valid and success is False:
        segments = (FailureSegment(
            segment_id=f"{logical_id}-failure", episode_id=episode_id,
            failure_class="task_failure", stage="terminal", tool=None,
            summary="The authoritative evaluator reported task failure.",
            earliest_divergence_step=None, start_step=0,
            end_step=max(0, result.counts.physical_steps),
        ),)
    index = dict(artifact_index)
    index["retry_safe"] = result.status != "execution_uncertain"
    if result.last_operation and result.last_operation.get("write_certainty") == "uncertain":
        index["retry_safe"] = False
    return EpisodeRecord(
        episode_id=episode_id, logical_id=logical_id, generation=generation,
        seed=seed, policy_rng=policy_rng, bundle_sha256=result.package_sha256,
        status="valid" if valid else "infra_invalid", success=success,
        started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_s=elapsed_s, artifact_index=index, attempt_index=attempt_index,
        failure_segment=segments[0] if segments else None, failure_segments=segments,
        invalid_reason=None if valid else (
            "missing_authoritative_outcome" if result.status == "completed"
            else result.status + ":" + result.termination_reason
        ),
    )

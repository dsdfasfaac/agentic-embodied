import pytest

from zetta.evolution.evidence_policy import (
    ArxRgbPublicEvidencePolicy,
    EvidencePolicyError,
)


@pytest.mark.parametrize("field", ["qpos", "qvel", "contact", "object_pose", "target", "reward", "seed", "gateway_journal"])
def test_arx_rejects_private_fields(field):
    with pytest.raises(EvidencePolicyError):
        ArxRgbPublicEvidencePolicy().validate_public_event({field: 1})


def test_arx_allows_public_rgb_and_acknowledgement():
    policy = ArxRgbPublicEvidencePolicy()
    event = {"step": 1, "rgb_frame_id": "frame-1", "action_ack": {"accepted": True}}
    assert policy.publish_event(event) == event


def test_arx_rejects_private_feature_names():
    with pytest.raises(EvidencePolicyError):
        ArxRgbPublicEvidencePolicy().validate_candidate_feature("object_pose_error")


def test_arx_rejects_unknown_public_fields():
    with pytest.raises(EvidencePolicyError):
        ArxRgbPublicEvidencePolicy().publish_event({"unreviewed_payload": {"x": 1}})

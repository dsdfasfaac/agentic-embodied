"""Frozen identifiers for the ARX MuJoCo campaign boundary."""

CANDIDATE_KIND = "arx_rgb_package_v1"
EVIDENCE_POLICY = "arx_rgb_public_v1"
PRIVILEGED_CANDIDATE_KIND = "structured_bundle_v1"
PRIVILEGED_EVIDENCE_POLICY = "arx_privileged_v1"
SAFETY_LAYER = {
    "action_contract": "arx_gateway_named_tools_v1",
    "control_limits": "arx_task7_bounded_operation_limits_v1",
    "simulation_health": "arx_finite_state_and_gateway_lease_v1",
    "joint_limit_shield": "not_implemented",
    "contact_policy": "environment_native_only",
}
HELDOUT_TRIALS = 20

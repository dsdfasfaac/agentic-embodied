"""Proposal-only CandidateBundle critic and real-observation reentry review."""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

from robots.arx.gateway.contracts import Assessment, Proposal
from robots.arx.deployment.real_input import RealFeatureSource
from robots.arx.deployment.feature_observation import FeatureObservationUnavailable
from zetta.evolution.critic import TemporalCritic, resolve_feature
from zetta.evolution.jsonio import file_sha256


class RealFeatureProvider:
    """Loads an operator supplied, SHA-pinned observer; it has no motion handle."""

    def __init__(self, path: Path, expected_sha256: str):
        if file_sha256(path) != expected_sha256:
            raise ValueError("real feature provider SHA-256 mismatch")
        spec = importlib.util.spec_from_file_location("arx_real_feature_provider", path)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load real feature provider")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.impl = module.create_provider()
        self.sources = [RealFeatureSource.model_validate(value) for value in self.impl.feature_sources()]
        if any(source.provider_sha256 != expected_sha256 for source in self.sources):
            raise ValueError("feature provider attestation differs from loaded bytes")
        self.sha256 = expected_sha256

    def validate_hardware(self, config):
        if hasattr(self.impl, "validate_hardware"):
            self.impl.validate_hardware(config)
        self.sources = [RealFeatureSource.model_validate(value) for value in self.impl.feature_sources()]
        if any(source.provider_sha256 != self.sha256 for source in self.sources):
            raise ValueError("configured feature provider attestation differs")

    def augment(self, observation, images):
        hardware = observation.get("hardware")
        if not isinstance(hardware, dict):
            raise ValueError("real feature provider requires hardware observation")
        now = time.monotonic_ns()
        # Image publication and durable journaling occur before the critic.
        # Source freshness is defined when the synchronized observation was
        # acquired, not when its already captured pixels are evaluated.
        reference_ns = hardware.get("observation_completed_ns", now)
        if (not isinstance(reference_ns, int) or reference_ns <= 0
                or reference_ns > now):
            raise ValueError("invalid real observation completion timestamp")
        quality = {"status": "observed", "unavailable_features": []}
        try:
            values = self.impl.observe(observation, images)
            quality = getattr(self.impl, "last_observation_quality", quality)
        except FeatureObservationUnavailable as exc:
            values = dict(exc.available)
            if (set(values) & set(exc.unavailable) or
                    set(values) | set(exc.unavailable) != {source.name for source in self.sources}):
                raise ValueError("invalid unavailable feature declaration") from exc
            values.update({name: None for name in exc.unavailable})
            quality = dict(exc.detail)
        if not isinstance(values, dict) or set(values) != {source.name for source in self.sources}:
            raise ValueError("real feature provider returned wrong feature names")
        result = dict(observation)
        result["feature_observation"] = quality
        for source in self.sources:
            value = values[source.name]
            expected = {"boolean": bool, "integer": int, "string": str}.get(source.scalar_type)
            unavailable = source.name in quality["unavailable_features"]
            if unavailable:
                if value is not None:
                    raise ValueError("unavailable features must not carry cached values")
            elif expected is not None:
                if type(value) is not expected:
                    raise ValueError(f"invalid real feature type: {source.name}")
            elif type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"invalid real feature value: {source.name}")
            stamps = []
            for source_id in source.source_ids:
                if source_id in hardware.get("camera_monotonic_ns", {}):
                    stamp = hardware["camera_monotonic_ns"][source_id]
                elif source_id in hardware.get("depth_monotonic_ns", {}):
                    stamp = hardware["depth_monotonic_ns"][source_id]
                elif source_id in hardware.get("auxiliary_monotonic_ns", {}):
                    stamp = hardware["auxiliary_monotonic_ns"][source_id]
                else:
                    stamp = hardware.get("state_monotonic_ns")
                if not isinstance(stamp, int) or stamp <= 0 or stamp > reference_ns:
                    raise ValueError(f"invalid source timestamp: {source.name}")
                stamps.append(stamp)
            if (reference_ns - min(stamps)) / 1e6 > source.max_age_ms:
                raise ValueError(f"stale real feature source: {source.name}")
            result[source.name] = value
        return result


class BundleMonitor:
    def __init__(self, bundle, provider=None, *, terminal_feature=None):
        self.bundle, self.provider = bundle, provider
        self.temporal = TemporalCritic(bundle.critic_rules)
        self.by_id = {rule.rule_id: rule for rule in bundle.critic_rules}
        self.last_feature_evidence = None
        if terminal_feature is not None and (
            provider is None or not any(source.name == terminal_feature and
                                        source.scalar_type == "boolean"
                                        for source in provider.sources)
        ):
            raise ValueError("terminal feature requires a declared real boolean source")
        self.terminal_feature = terminal_feature

    def _remember_features(self, observation, measured):
        if self.provider:
            self.last_feature_evidence = {
                "observation_id": observation["observation_id"],
                "provider_sha256": self.provider.sources[0].provider_sha256,
                "features": {source.name: measured[source.name]
                             for source in self.provider.sources},
                "feature_observation": measured.get("feature_observation", {"status": "observed"}),
            }

    def completion_evidence(self):
        evidence = self.last_feature_evidence
        if (self.terminal_feature is not None and evidence is not None
                and evidence["features"][self.terminal_feature] is True):
            return {"observation_id": evidence["observation_id"],
                    "feature": self.terminal_feature,
                    "provider_sha256": evidence["provider_sha256"]}
        return None

    def reset(self, observation, images):
        self.temporal.reset()
        if self.provider:
            self._remember_features(observation, self.provider.augment(observation, images))

    def lifecycle(self, event):
        pass

    def observe(self, observation, images):
        measured = self.provider.augment(observation, images) if self.provider else observation
        self._remember_features(observation, measured)
        events = []
        quality = measured.get("feature_observation", {"status": "observed", "unavailable_features": []})
        for item in self.temporal.evaluate(measured, step_index=observation["step_index"],
                                          unavailable_features=set(quality.get("unavailable_features", []))):
            rule = self.by_id[item["rule_id"]]
            events.append(Proposal(
                detector_id="bundle", failure_mode=rule.rule_id, rule_id=rule.rule_id,
                evidence_observation_ids=[observation["observation_id"]],
                reason_code=rule.rule_id, summary=rule.proposal,
            ))
        return Assessment(
            critic_id="bundle", observation_id=observation["observation_id"],
            step_index=observation["step_index"],
            status="failure" if events else "unknown" if quality["status"] == "unknown" else "clear",
            events=events, features={"feature_observation": quality},
        )


class RealBundleReentry:
    def __init__(self, bundle, provider=None, *, max_sensor_age_ms=1000,
                 max_sensor_skew_ms=1000, require_hardware=True, monitor=None):
        self.rules = {rule.rule_id: rule for rule in bundle.critic_rules}
        self.recoveries = {rule.recovery_id: rule for rule in bundle.recovery_rules}
        self.provider = provider
        self.max_age, self.max_skew = max_sensor_age_ms, max_sensor_skew_ms
        self.require_hardware = require_hardware
        self.monitor = monitor

    def inspect(self, args, context):
        obs = context["observations"][-1]
        hardware = obs.get("hardware", {})
        health = hardware.get("device_health", {})
        healthy = (health.get("transport_responsive") is True
                   and not health.get("fault_codes")
                   and all(x.get("responsive") is True for x in hardware.get("camera_health", {}).values())
                   and len(hardware.get("camera_health", {})) == 3)
        completed_ns = hardware.get("observation_completed_ns")
        wall_fresh = (completed_ns is None or
                      (type(completed_ns) is int and 0 <=
                       (time.monotonic_ns() - completed_ns) / 1e6 <= max(1000, 4 * self.max_age)))
        fresh = (type(hardware.get("sensor_age_ms")) in (int, float)
                 and hardware["sensor_age_ms"] <= self.max_age
                 and type(hardware.get("sensor_skew_ms")) in (int, float)
                 and hardware["sensor_skew_ms"] <= self.max_skew
                 and wall_fresh)
        checks = [
            {"check_id": "device-health", "status": "pass" if healthy else "fail",
             "evidence_ids": [obs["observation_id"]], "reason_code": "healthy" if healthy else "device_fault"},
            {"check_id": "sensor-freshness", "status": "pass" if fresh else "fail",
             "evidence_ids": [obs["observation_id"]], "reason_code": "fresh" if fresh else "stale_or_skewed"},
            {"check_id": "measured-arrival", "status": "pass" if hardware.get("arrival_verified") is True else "fail",
             "evidence_ids": [obs["observation_id"]], "reason_code": "arrived" if hardware.get("arrival_verified") is True else "arrival_unverified"},
        ]
        if not self.require_hardware:
            checks = []
        rule = self.recoveries[context["policy_id"]]
        measured = self.provider.augment(obs, context["images"][-1]) if self.provider else obs
        unavailable = set(measured.get("feature_observation", {}).get("unavailable_features", []))
        checks.append({"check_id": "feature-observability", "status": "unknown" if unavailable else "pass",
                       "evidence_ids": [obs["observation_id"]],
                       "reason_code": "target_unobserved" if unavailable else "observed"})
        for rule_id in rule.trigger_rule_ids:
            critic = self.rules[rule_id]
            needed = {critic.feature, *(p.feature for p in critic.activation_conditions)}
            if needed & unavailable:
                checks.append({"check_id": rule_id, "status": "unknown",
                               "evidence_ids": [obs["observation_id"]], "reason_code": "feature_unavailable"})
                continue
            active = all(TemporalCritic._predicate(p, measured) for p in critic.activation_conditions)
            if critic.operator == "stagnant" and active:
                history = (self.monitor.temporal._state[rule_id].history
                           if self.monitor is not None else [])
                status = ("unknown" if len(history) < critic.dwell_steps else
                          "fail" if TemporalCritic._condition(critic, resolve_feature(measured, critic.feature), history)
                          else "pass")
            else:
                status = "fail" if active and TemporalCritic._condition(
                    critic, resolve_feature(measured, critic.feature), [resolve_feature(measured, critic.feature)]
                ) else "pass"
            checks.append({"check_id": rule_id, "status": status,
                           "evidence_ids": [obs["observation_id"]],
                           "reason_code": "rule_clear" if status == "pass" else "rule_not_clear"})
        eligible = all(check["status"] == "pass" for check in checks)
        return {"schema_version": "arx.reentry.assessment.v1",
                "recovery_id": context["recovery_id"], "observation_id": obs["observation_id"],
                "policy_id": context["policy_id"],
                "status": "eligible" if eligible else "ineligible", "checks": checks}

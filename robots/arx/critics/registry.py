# Copyright (c) 2026 Zetta Contributors
"""Frozen registration and RGB feature-rule observation adapter."""

from __future__ import annotations

import ast
import sys
from collections import deque

from robots.arx.gateway.contracts import Assessment, Proposal, digest
from zetta.evolution.critic import TemporalCritic
from zetta.evolution.models import CriticPredicate, CriticRule
from zetta.evolution.evidence_policy import ArxRgbPublicEvidencePolicy, PermissiveEvidencePolicy

from .contracts import CriticObservation, FeatureValue
from .isolation import SandboxedPythonSource
from .packages import load_candidate


def validate_scalar(value, declaration):
    types = {
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "string": (str,),
    }
    if type(value) not in types[declaration.scalar_type]:
        raise ValueError("feature scalar type mismatch: " + declaration.name)
    if (
        declaration.value_range is not None
        and not declaration.value_range[0] <= value <= declaration.value_range[1]
    ):
        raise ValueError("feature value outside declared range")


def validate_candidate(package, limits):
    manifest, config, schema = (
        package.critic_manifest,
        package.config,
        package.feature_schema,
    )
    if manifest.source.kind == "prm_service":
        raise NotImplementedError("PRM_SOURCE_NOT_IMPLEMENTED")
    if manifest.input_profile not in {"arx.rgb_only.v1", "arx.prm_inputs.v1"}:
        raise ValueError("unsupported critic input profile")
    if (
        manifest.history_limit > limits.max_history
        or manifest.evaluation_timeout_ms > limits.max_evaluation_ms
    ):
        raise ValueError("critic exceeds frozen resource limits")
    features = {f.name: f for f in schema.features}
    evidence_policy = PermissiveEvidencePolicy("arx_privileged_v1")
    for f in features.values():
        evidence_policy.validate_candidate_feature(f.name)
        if (
            not set(f.source_cameras) <= set(manifest.cameras)
            or f.history_window > manifest.history_limit
        ):
            raise ValueError("feature cameras/history exceed manifest")
    ids = {r.rule_id for r in config.rules}
    if len(ids) != len(config.rules) or set(config.rule_failure_modes) != ids:
        raise ValueError("duplicate or unmapped rules")
    if set(config.rule_failure_modes.values()) != set(manifest.failure_modes):
        raise ValueError("failure-mode declaration mismatch")
    for rule in config.rules:
        if rule.dwell_steps > manifest.history_limit:
            raise ValueError("rule dwell exceeds history bound")
        for predicate in [rule, *rule.activation_conditions]:
            if predicate.feature not in features:
                raise ValueError("undeclared feature")
            declaration = features[predicate.feature]
            # Threshold is typed, but need not lie in feature range (useful for
            # impossible predicates and stagnant tolerances).
            if (
                type(predicate.threshold)
                not in {
                    "number": (int, float),
                    "integer": (int,),
                    "boolean": (bool,),
                    "string": (str,),
                }[declaration.scalar_type]
            ):
                raise ValueError("threshold type mismatch")
            if predicate.operator not in (
                "eq",
                "ne",
            ) and declaration.scalar_type not in ("number", "integer"):
                raise ValueError("numeric operator needs numeric feature")
            if predicate.operator == "stagnant" and predicate.threshold < 0:
                raise ValueError("negative stagnant tolerance")
    bindings = {b.binding_id: b for b in package.bindings}
    if len(bindings) != len(package.bindings):
        raise ValueError("duplicate binding IDs")
    for mode, binding in manifest.recovery_bindings.items():
        if binding not in bindings or mode not in bindings[binding].failure_modes:
            raise ValueError("missing recovery binding")
    reentry_manifest = package.json("reentry/manifest.json")
    reentry_config = package.json("reentry/config.json")
    if not isinstance(reentry_manifest, dict) or set(reentry_manifest) != {
        "schema_version", "implementation", "policy_id"
    } or reentry_manifest["schema_version"] != "arx.reentry.manifest.v1":
        raise ValueError("unbound reentry policy")
    if reentry_manifest["implementation"] == "always_ineligible":
        if reentry_config != {}:
            raise ValueError("unsupported reentry policy")
    elif reentry_manifest["implementation"] == "rgb_feature_threshold":
        if (not isinstance(reentry_config, dict)
                or set(reentry_config) != {"feature", "operator", "threshold", "minimum_observations"}
                or reentry_config["feature"] not in features
                or reentry_config["operator"] not in {"lt", "le", "gt", "ge", "eq", "ne"}
                or not isinstance(reentry_config["minimum_observations"], int)
                or not 1 <= reentry_config["minimum_observations"] <= manifest.history_limit):
            raise ValueError("unsupported RGB reentry predicate")
        predicate = reentry_config
        declaration = features[predicate["feature"]]
        threshold = predicate["threshold"]
        if type(threshold) not in {"number": (int, float), "integer": (int,),
                                   "boolean": (bool,), "string": (str,)}[declaration.scalar_type]:
            raise ValueError("RGB reentry threshold type mismatch")
        if predicate["operator"] not in {"eq", "ne"} and declaration.scalar_type not in {"number", "integer"}:
            raise ValueError("RGB reentry numeric comparison requires numeric feature")
    else:
        raise ValueError("unsupported reentry policy")
    if any(b.reentry_policy_id != reentry_manifest["policy_id"] for b in bindings.values()):
        raise ValueError("unbound reentry policy")
    # Static import checking improves diagnostics. It is NOT the isolation boundary.
    tree = ast.parse(package.file("critic/features.py"))
    allowed = set(sys.stdlib_module_names) | {"numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports not admitted")
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        if not set(names) <= allowed:
            raise ValueError("dependency not in pinned stdlib/numpy runtime")


class ArxCriticRegistry:
    """Harness-only registration. Freeze before reset; learner code never imports here."""

    def __init__(self, *, limits):
        self.limits = limits
        self._packages = {}
        self._frozen = False

    def register(self, manifest, config, feature_schema, package_root):
        if self._frozen:
            raise ValueError("critic registry frozen")
        package = load_candidate(package_root)
        for supplied, actual in [
            (manifest, package.critic_manifest),
            (config, package.config),
            (feature_schema, package.feature_schema),
        ]:
            supplied = (
                supplied.model_dump(mode="json")
                if hasattr(supplied, "model_dump")
                else supplied
            )
            if supplied != actual.model_dump(mode="json"):
                raise ValueError("registration payload differs from sealed package")
        validate_candidate(package, self.limits)
        critic_id = package.critic_manifest.critic_id
        existing_rules = {
            r.rule_id for p in self._packages.values() for r in p.config.rules
        }
        if critic_id in self._packages or existing_rules & {
            r.rule_id for r in package.config.rules
        }:
            raise ValueError("duplicate critic/rule identity")
        self._packages[critic_id] = package

    def register_package(self, package_root, **identities):
        package = load_candidate(package_root, **identities)
        self.register(
            package.critic_manifest,
            package.config,
            package.feature_schema,
            package_root,
        )
        return package

    def describe(self):
        entries = [
            {
                "critic_id": k,
                "package_sha256": p.sha256,
                "manifest": p.critic_manifest.model_dump(mode="json"),
            }
            for k, p in sorted(self._packages.items())
        ]
        return {
            "schema_version": "arx.critic.catalog.v1",
            "critics": entries,
            "critic_sha256": digest(entries),
            "frozen": self._frozen,
        }

    def preflight(self):
        """Import/reset/extract submitted code in disposable isolated workers."""
        import numpy as np

        from .contracts import FeatureValue

        for package in self._packages.values():
            source = SandboxedPythonSource(
                package.file("critic/features.py"),
                self.limits,
                timeout_ms=package.critic_manifest.evaluation_timeout_ms,
            )
            images = {
                name: np.zeros(
                    (self.limits.image_height, self.limits.image_width, 3),
                    dtype=np.uint8,
                )
                for name in package.critic_manifest.cameras
            }
            refs = {
                name: {
                    "content_id": "preflight",
                    "sha256": "0" * 64,
                    "width": self.limits.image_width,
                    "height": self.limits.image_height,
                    "encoding": "png",
                }
                for name in images
            }
            obs = {
                "schema_version": "arx.critic.observation.v1",
                "episode_nonce": "preflight",
                "observation_id": "obs-0",
                "step_index": 0,
                "simulation_time_s": 0.0,
                "cameras": refs,
                "lifecycle": "reset",
                "event_sequence": 0,
            }
            try:
                source.reset(obs, images, package.config.extractor_config)
                obs.update(
                    observation_id="obs-1",
                    step_index=1,
                    simulation_time_s=1 / 15,
                    lifecycle="nominal",
                    event_sequence=1,
                )
                output = source.extract(obs, images)
                if not isinstance(output, dict) or set(output) != {
                    f.name for f in package.feature_schema.features
                }:
                    raise ValueError("preflight feature names mismatch")
                for feature in package.feature_schema.features:
                    value = FeatureValue.model_validate(output[feature.name])
                    if value.valid:
                        validate_scalar(value.value, feature)
            finally:
                source.close()

    def freeze(self):
        if self._frozen or not self._packages:
            raise ValueError("registry already frozen or empty")
        self._frozen = True
        critics = []
        try:
            for _, package in sorted(self._packages.items()):
                source = SandboxedPythonSource(
                    package.file("critic/features.py"),
                    self.limits,
                    timeout_ms=package.critic_manifest.evaluation_timeout_ms,
                )
                critics.append(FeatureRuleCritic(package, source))
            return FrozenCritic(critics, self.describe()["critic_sha256"])
        except BaseException:
            for critic in critics:
                critic.close()
            raise


class FeatureRuleCritic:
    def __init__(self, package, source):
        self.package, self.source = package, source
        self.manifest, self.config, self.schema = (
            package.critic_manifest,
            package.config,
            package.feature_schema,
        )
        self.rules = {}
        for rule in self.config.rules:
            adapted = CriticRule(
                rule_id=rule.rule_id,
                title=rule.title,
                feature=rule.feature,
                operator=rule.operator,
                threshold=rule.threshold,
                dwell_steps=rule.dwell_steps,
                cooldown_steps=rule.cooldown_steps,
                proposal=rule.proposal,
                evidence_ids=tuple(rule.evidence_ids),
                activation_conditions=tuple(
                    CriticPredicate(**p.model_dump())
                    for p in rule.activation_conditions
                ),
            )
            self.rules[rule.rule_id] = TemporalCritic((adapted,))
        self.initialized = False
        self.last = None
        self.cached = None
        self.history = deque(maxlen=self.manifest.history_limit)

    def _input(self, observation, images):
        obs = CriticObservation.from_public(observation)
        if not set(self.manifest.cameras) <= set(images) or not set(
            self.manifest.cameras
        ) <= set(obs.cameras):
            raise ValueError("missing required critic camera")
        selected = {name: images[name] for name in self.manifest.cameras}
        for name, value in selected.items():
            ref = obs.cameras[name]
            if value.shape != (ref.height, ref.width, 3):
                raise ValueError("camera metadata mismatch")
        return obs.model_dump(mode="json"), selected

    def reset(self, observation, images):
        if self.initialized or observation["step_index"] != 0:
            raise ValueError("critic reset exactly once at step zero")
        obs, selected = self._input(observation, images)
        self.source.reset(obs, selected, self.config.extractor_config)
        self.initialized = True
        self.last = (obs["observation_id"], 0)
        self.history.append(obs["observation_id"])

    def observe(self, observation, images):
        obs, selected = self._input(observation, images)
        identity = (obs["observation_id"], obs["step_index"])
        if self.initialized and identity == self.last and self.cached is not None:
            return self.cached.model_copy(deep=True)
        if (
            not self.initialized
            or obs["step_index"] != self.last[1] + 1
            or obs["observation_id"] in self.history
        ):
            raise ValueError("out-of-order critic observation")
        try:
            output = self.source.extract(obs, selected)
            if not isinstance(output, dict) or set(output) != {
                f.name for f in self.schema.features
            }:
                raise ValueError("undeclared or missing feature output")
            values = {}
            for declaration in self.schema.features:
                value = FeatureValue.model_validate(output[declaration.name])
                if value.valid:
                    validate_scalar(value.value, declaration)
                values[declaration.name] = value
            self.history.append(obs["observation_id"])
            events, unknown = [], False
            for rule in self.config.rules:
                evaluator = self.rules[rule.rule_id]
                required = {rule.feature} | {
                    p.feature for p in rule.activation_conditions
                }
                if not all(values[name].valid for name in required):
                    evaluator.reset()
                    unknown = True
                    continue
                fired = evaluator.evaluate(
                    {
                        name: value.value
                        for name, value in values.items()
                        if value.valid
                    },
                    step_index=obs["step_index"],
                )
                if fired:
                    events.append(
                        Proposal(
                            detector_id=self.manifest.critic_id,
                            rule_id=rule.rule_id,
                            failure_mode=self.config.rule_failure_modes[rule.rule_id],
                            evidence_observation_ids=list(self.history)[
                                -rule.dwell_steps :
                            ],
                            reason_code="feature_rule_fired",
                            summary=rule.title,
                            limitations=[
                                "RGB features do not establish contact or task success."
                            ],
                        )
                    )
            assessment = Assessment(
                critic_id=self.manifest.critic_id,
                observation_id=obs["observation_id"],
                step_index=obs["step_index"],
                status="failure" if events else "unknown" if unknown else "clear",
                events=events,
                features={name: value.model_dump() for name, value in values.items()},
            )
            self.last, self.cached = identity, assessment
            return assessment.model_copy(deep=True)
        except BaseException:
            self.source.close()
            raise

    def close(self):
        self.source.close()


class FrozenCritic:
    def __init__(self, critics, sha256):
        self.critics, self.sha256 = critics, sha256

    def reset(self, observation, images):
        try:
            for critic in self.critics:
                critic.reset(observation, images)
        except BaseException:
            self.close()
            raise

    def observe(self, observation, images):
        try:
            assessments = [c.observe(observation, images) for c in self.critics]
            events = [e for a in assessments for e in a.events]
            return Assessment(
                critic_id="registered-critics",
                observation_id=observation["observation_id"],
                step_index=observation["step_index"],
                status="failure"
                if events
                else "unknown"
                if any(a.status == "unknown" for a in assessments)
                else "clear",
                events=events,
                features={a.critic_id: a.features for a in assessments},
            )
        except BaseException:
            self.close()
            raise

    def lifecycle(self, event):
        # Audit-only: never reset dwell/history or re-evaluate cached pixels.
        if event["kind"] == "episode_closed":
            self.close()

    def close(self):
        for critic in self.critics:
            critic.close()


class AlwaysIneligibleReentry:
    def inspect(self, args, context):
        return {
            "schema_version": "arx.reentry.assessment.v1",
            "recovery_id": context["recovery_id"],
            "observation_id": args.observation_ids[-1],
            "policy_id": context["policy_id"],
            "status": "ineligible",
            "checks": [
                {
                    "check_id": "no-clearance",
                    "status": "fail",
                    "evidence_ids": args.observation_ids,
                    "reason_code": "policy_never_resumes",
                }
            ],
        }


class RgbFeatureReentry:
    """Review public RGB with an isolated extractor; no simulator state access."""

    def __init__(self, package, limits):
        self.package, self.limits = package, limits

    def inspect(self, args, context):
        import operator

        config = self.package.json("reentry/config.json")
        compare = {"lt": operator.lt, "le": operator.le, "gt": operator.gt,
                   "ge": operator.ge, "eq": operator.eq, "ne": operator.ne}[config["operator"]]
        source = SandboxedPythonSource(
            self.package.file("critic/features.py"), self.limits,
            timeout_ms=self.package.critic_manifest.evaluation_timeout_ms,
        )
        passed = 0
        try:
            for index, (observation, images) in enumerate(zip(context["observations"], context["images"])):
                if index == 0:
                    source.reset(observation, images, self.package.config.extractor_config)
                value = FeatureValue.model_validate(source.extract(observation, images)[config["feature"]])
                if value.valid and compare(value.value, config["threshold"]):
                    passed += 1
                else:
                    passed = 0
        finally:
            source.close()
        eligible = passed >= config["minimum_observations"]
        return {
            "schema_version": "arx.reentry.assessment.v1",
            "recovery_id": context["recovery_id"],
            "observation_id": args.observation_ids[-1],
            "policy_id": context["policy_id"],
            "status": "eligible" if eligible else "ineligible",
            "checks": [{"check_id": "public-rgb-feature-clearance",
                        "status": "pass" if eligible else "fail",
                        "evidence_ids": args.observation_ids,
                        "reason_code": "rgb_feature_clear" if eligible else "rgb_feature_not_clear"}],
        }

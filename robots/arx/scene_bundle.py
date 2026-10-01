# Copyright (c) 2026 Zetta Contributors
"""Immutable Real2Sim starting-scene bundle validation and selection."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from robots.arx.contracts import ArxStartingScenes

__all__ = [
    "ArtifactRef",
    "Real2SimSceneBundle",
    "load_scene_bundle",
    "select_scene_id",
]

_SCHEMA = "zetta_real2sim_scene_bundle_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"attempt", "randomized_variant"})


def _strict(cls: type[Any], value: Any, context: str) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    payload = {str(key): item for key, item in value.items()}
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {context}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactRef:
    """One content-addressed file under a Real2Sim bundle root."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if (
            not self.path
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in self.path
        ):
            raise ValueError(f"artifact path must be safe and relative: {self.path!r}")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must contain 64 lowercase hex characters")

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> ArtifactRef:
        return _strict(cls, value, context)


@dataclasses.dataclass(frozen=True, slots=True)
class SceneSource:
    real2sim_commit: str
    run_id: str
    kind: str
    index: int
    retry: int
    bundle_root: str

    def __post_init__(self) -> None:
        if not self.real2sim_commit or not self.run_id:
            raise ValueError("Real2Sim commit and run_id must not be empty")
        if self.kind not in _KINDS:
            raise ValueError(f"scene source kind must be one of {sorted(_KINDS)}")
        if self.index < 0 or self.retry < 0:
            raise ValueError("scene source index/retry must be non-negative")
        root = Path(self.bundle_root).expanduser()
        if not root.is_absolute():
            raise ValueError("bundle_root must be absolute")


@dataclasses.dataclass(frozen=True, slots=True)
class SceneArtifacts:
    active_layout: ArtifactRef
    assembly_report: ArtifactRef
    final_scene: ArtifactRef
    scene_xml: ArtifactRef
    model_mjb: ArtifactRef
    settled_state: ArtifactRef

    @classmethod
    def from_mapping(cls, value: Any) -> SceneArtifacts:
        if not isinstance(value, Mapping):
            raise ValueError("artifacts must be an object")
        payload = {
            str(key): ArtifactRef.from_mapping(item, f"artifacts.{key}")
            for key, item in value.items()
        }
        return _strict(cls, payload, "artifacts")


@dataclasses.dataclass(frozen=True, slots=True)
class SceneObjects:
    required_ids: tuple[str, ...]
    target_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.required_ids or any(not item for item in self.required_ids):
            raise ValueError("required_ids must not be empty")
        if len(set(self.required_ids)) != len(self.required_ids):
            raise ValueError("required_ids must be unique")
        if not set(self.target_ids).issubset(self.required_ids):
            raise ValueError("target_ids must be a subset of required_ids")


@dataclasses.dataclass(frozen=True, slots=True)
class SceneComposition:
    frame_transform: str
    robot_source_archive: str
    robot_source_archive_sha256: str
    robot_xml: str
    robot_xml_sha256: str
    mapping: str
    mapping_sha256: str
    cameras: str

    def __post_init__(self) -> None:
        if not self.frame_transform or not self.cameras:
            raise ValueError("composition transform/cameras must not be empty")
        for name in ("robot_source_archive", "robot_xml", "mapping"):
            if not Path(getattr(self, name)).expanduser().is_absolute():
                raise ValueError(f"composition {name} must be absolute")
        for name in (
            "robot_source_archive_sha256",
            "robot_xml_sha256",
            "mapping_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"composition {name} is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class SceneReset:
    source: str = "settled_state"
    settle_after_composition_steps: int = 0

    def __post_init__(self) -> None:
        if self.source != "settled_state":
            raise ValueError("scene reset source must be settled_state")
        if self.settle_after_composition_steps < 0:
            raise ValueError("settle_after_composition_steps must be non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class Real2SimSceneBundle:
    """Validated Real2Sim source bundle and deterministic composition inputs."""

    schema_version: str
    scene_id: str
    task_name: str
    source: SceneSource
    artifacts: SceneArtifacts
    objects: SceneObjects
    composition: SceneComposition
    reset: SceneReset

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError(f"scene bundle schema_version must be {_SCHEMA!r}")
        if not self.scene_id or not self.task_name:
            raise ValueError("scene_id and task_name must not be empty")

    @classmethod
    def from_mapping(cls, value: Any) -> Real2SimSceneBundle:
        if not isinstance(value, Mapping):
            raise ValueError("scene bundle must be an object")
        payload = {str(key): item for key, item in value.items()}
        payload["source"] = _strict(SceneSource, payload.get("source"), "source")
        payload["artifacts"] = SceneArtifacts.from_mapping(payload.get("artifacts"))
        for name, target in (
            ("objects", SceneObjects),
            ("composition", SceneComposition),
            ("reset", SceneReset),
        ):
            item = dict(payload.get(name) or {})
            if name == "objects":
                item["required_ids"] = tuple(item.get("required_ids", ()))
                item["target_ids"] = tuple(item.get("target_ids", ()))
            payload[name] = _strict(target, item, name)
        return _strict(cls, payload, "scene bundle")

    @property
    def bundle_root(self) -> Path:
        return Path(self.source.bundle_root).expanduser().resolve(strict=True)

    def artifact_path(self, reference: ArtifactRef) -> Path:
        root = self.bundle_root
        path = (root / reference.path).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"artifact escapes bundle_root: {reference.path}")
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {reference.path}")
        return path

    def validate_files(self) -> str:
        """Validate source files/provenance and return a canonical bundle digest."""
        for name, expected in (
            ("robot_source_archive", self.composition.robot_source_archive_sha256),
            ("robot_xml", self.composition.robot_xml_sha256),
            ("mapping", self.composition.mapping_sha256),
        ):
            path = Path(getattr(self.composition, name)).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"composition {name} is not a regular file: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"composition hash mismatch for {name}: {actual} != {expected}"
                )

        paths: dict[str, Path] = {}
        for field in dataclasses.fields(SceneArtifacts):
            reference = getattr(self.artifacts, field.name)
            path = self.artifact_path(reference)
            actual = _sha256(path)
            if actual != reference.sha256:
                raise ValueError(
                    f"artifact hash mismatch for {field.name}: {actual} != {reference.sha256}"
                )
            paths[field.name] = path

        layout = _read_json(paths["active_layout"])
        report = _read_json(paths["assembly_report"])
        final = _read_json(paths["final_scene"])
        if final.get("schema_version") != "1.0":
            raise ValueError("Real2Sim final_scene schema must be 1.0")
        if final.get("backend") != "mujoco" or final.get("status") != "succeeded":
            raise ValueError("Real2Sim final_scene must be a successful MuJoCo result")
        for name, value in (("layout", layout), ("assembly report", report), ("final scene", final)):
            if value.get("run_id") != self.source.run_id:
                raise ValueError(f"{name} run_id does not match scene source")
        layout_ids = {str(item.get("id")) for item in layout.get("objects", [])}
        report_ids = {str(item.get("logical_id")) for item in report.get("objects", [])}
        missing = set(self.objects.required_ids) - layout_ids
        if missing or not set(self.objects.required_ids).issubset(report_ids):
            raise ValueError(f"required logical objects are missing: {sorted(missing)}")
        expected_paths = {
            "scene_xml": final.get("scene_xml") or final.get("scene"),
            "model_mjb": final.get("compiled_model"),
            "settled_state": final.get("state"),
        }
        for name, declared in expected_paths.items():
            if not isinstance(declared, str) or Path(declared).name != paths[name].name:
                raise ValueError(f"final_scene {name} path does not match bundle artifact")

        canonical = {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "task_name": self.task_name,
            "source": dataclasses.asdict(self.source),
            "artifacts": dataclasses.asdict(self.artifacts),
            "objects": dataclasses.asdict(self.objects),
            "composition": dataclasses.asdict(self.composition),
            "reset": dataclasses.asdict(self.reset),
        }
        canonical["source"].pop("bundle_root")
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_scene_bundle(path: str | Path) -> Real2SimSceneBundle:
    return Real2SimSceneBundle.from_mapping(_read_json(Path(path)))


def select_scene_id(
    scenes: ArxStartingScenes, *, seed: int, explicit_scene_id: str | None = None
) -> str:
    """Select an allowlisted scene without depending on registry insertion order."""
    allowed = tuple(sorted(scenes.allowed_scene_ids))
    if explicit_scene_id is not None:
        if explicit_scene_id not in allowed:
            raise ValueError(f"scene_id is not allowlisted: {explicit_scene_id!r}")
        if scenes.selection not in {"explicit", "fixed", "seeded_uniform"}:
            raise ValueError("scene selection mode does not accept an explicit ID")
        return explicit_scene_id
    if scenes.selection == "explicit":
        raise ValueError("explicit scene selection requires scene_id")
    if scenes.selection == "fixed":
        return scenes.default_scene_id
    digest = hashlib.sha256(str(int(seed)).encode("ascii")).digest()
    return allowed[int.from_bytes(digest[:8], "big") % len(allowed)]

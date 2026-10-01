# Copyright (c) 2026 Zetta Contributors
"""Read and seal immutable candidate bytes without importing candidate code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field

from robots.arx.gateway.contracts import (
    ID,
    RecoveryBinding,
    StrictModel,
    canonical,
    digest,
)

from .contracts import SHA, CriticConfig, CriticManifest, FeatureSchema


class PackageFile(StrictModel):
    relative_path: str
    sha256: SHA
    media_type: str


class PackageManifest(StrictModel):
    schema_version: Literal["arx.deployment.package.v1"]
    package_id: ID
    parent_package_sha256: SHA | None
    generation: Annotated[int, Field(ge=0)]
    contract_sha256: SHA
    tool_catalog_sha256: SHA
    feature_schema_sha256: SHA
    deployment_bootstrap_sha256: SHA
    files: Annotated[list[PackageFile], Field(min_length=1, max_length=256)]
    mechanism_hypothesis: str
    changed_components: list[str]
    predicted_effect: str
    evidence_ids: list[ID]
    validation_plan: str


@dataclass(frozen=True)
class CandidatePackage:
    """Byte strings are the authority; callers receive fresh parsed models."""

    manifest_bytes: bytes
    payloads: tuple[tuple[str, bytes], ...]

    @property
    def sha256(self):
        return digest(json.loads(self.manifest_bytes))

    def file(self, name):
        return dict(self.payloads)[name]

    def json(self, name):
        return json.loads(self.file(name))

    @property
    def critic_manifest(self):
        return CriticManifest.model_validate_json(self.file("critic/manifest.json"))

    @property
    def config(self):
        return CriticConfig.model_validate_json(self.file("critic/config.json"))

    @property
    def feature_schema(self):
        return FeatureSchema.model_validate_json(
            self.file("critic/feature_schema.json")
        )

    @property
    def bindings(self):
        return tuple(
            RecoveryBinding.model_validate(x)
            for x in self.json("recovery_bindings.json")
        )


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_candidate(
    root: Path,
    *,
    expected_contract=None,
    expected_catalog=None,
    expected_bootstrap=None,
):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink package root")
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = PackageManifest.model_validate_json(manifest_bytes)
    expected = {
        "contract_sha256": expected_contract,
        "tool_catalog_sha256": expected_catalog,
        "deployment_bootstrap_sha256": expected_bootstrap,
    }
    for name, value in expected.items():
        if value is not None and getattr(manifest, name) != value:
            raise ValueError("package identity mismatch: " + name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("nonregular package file")
    listed = set()
    payloads = []
    total = 0
    for item in manifest.files:
        path = PurePosixPath(item.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != item.relative_path
            or item.relative_path in listed
            or item.relative_path == "manifest.json"
        ):
            raise ValueError("invalid or duplicate package path")
        listed.add(item.relative_path)
        file = root.joinpath(*path.parts)
        if file.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("package file too large")
        content = file.read_bytes()
        total += len(content)
        if total > 16 * 1024 * 1024 or sha_bytes(content) != item.sha256:
            raise ValueError("package size or file hash mismatch")
        payloads.append((item.relative_path, content))
    if actual != listed | {"manifest.json"}:
        raise ValueError("unlisted or missing files")
    required = {
        "critic/manifest.json",
        "critic/config.json",
        "critic/features.py",
        "critic/feature_schema.json",
        "skill/SKILL.md",
        "recovery_bindings.json",
        "reentry/manifest.json",
        "reentry/config.json",
        "evidence/claims.json",
        "tests/replay_cases.json",
    }
    if not required <= listed:
        raise ValueError("incomplete deployment package")
    if any(x.startswith("critic/") and x not in required for x in listed):
        raise ValueError("only single-file extractor packages are admitted")
    package = CandidatePackage(manifest_bytes, tuple(payloads))
    cm = package.critic_manifest
    if sha_bytes(package.file("critic/features.py")) != cm.code_sha256:
        raise ValueError("critic code hash mismatch")
    for filename, expected_sha in [
        ("config", cm.config_sha256),
        ("feature_schema", cm.feature_schema_sha256),
    ]:
        if digest(package.json("critic/" + filename + ".json")) != expected_sha:
            raise ValueError("critic " + filename + " hash mismatch")
    if manifest.feature_schema_sha256 != cm.feature_schema_sha256:
        raise ValueError("feature schema identity mismatch")
    return package


def seal_candidate(root: Path, metadata: dict):
    """Learner/harness authoring helper; registration still validates everything."""
    root = Path(root)
    cm = json.loads((root / "critic/manifest.json").read_text())
    cm.update(
        code_sha256=sha_bytes((root / "critic/features.py").read_bytes()),
        config_sha256=digest(json.loads((root / "critic/config.json").read_text())),
        feature_schema_sha256=digest(
            json.loads((root / "critic/feature_schema.json").read_text())
        ),
    )
    (root / "critic/manifest.json").write_text(canonical(cm) + "\n")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks forbidden")
        if path.is_file() and path != root / "manifest.json":
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha_bytes(path.read_bytes()),
                    "media_type": "application/json"
                    if path.suffix == ".json"
                    else "text/plain",
                }
            )
    manifest = PackageManifest.model_validate(
        dict(metadata, feature_schema_sha256=cm["feature_schema_sha256"], files=files)
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return load_candidate(root)

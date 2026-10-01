# Copyright (c) 2026 Zetta Contributors
"""Content-addressed candidate artifacts shared across campaign runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from zetta.evolution.jsonio import canonical_sha256, read_json
from zetta.evolution.models import CandidateBundle

STRUCTURED = "structured_bundle_v1"
ARX = "arx_rgb_package_v1"


@dataclass(frozen=True, slots=True)
class CandidateRef:
    sha256: str
    kind: str
    path: Path
    parent_sha256: str | None
    diagnosis_sha256: str | None
    generation: int
    candidate_id: str


class CandidateArtifactAdapter(Protocol):
    kind: str

    def load(self, path: Path, sha256: str) -> CandidateRef: ...


class StructuredBundleAdapter:
    kind = STRUCTURED

    def load(self, path: Path, sha256: str) -> CandidateRef:
        if not path.is_file():
            raise ValueError("structured candidate bundle is missing")
        payload = read_json(path)
        if canonical_sha256(payload) != sha256:
            raise ValueError("candidate artifact digest mismatch")
        bundle = CandidateBundle.from_dict(payload)
        return CandidateRef(
            sha256, self.kind, path.resolve(), bundle.parent_sha256,
            bundle.diagnosis_sha256, bundle.generation, bundle.candidate_id,
        )


class ArxPackageAdapter:
    kind = ARX

    def load(self, path: Path, sha256: str) -> CandidateRef:
        from robots.arx.critics.packages import PackageManifest, load_candidate

        package = load_candidate(path)
        if package.sha256 != sha256:
            raise ValueError("ARX candidate package digest mismatch")
        manifest = PackageManifest.model_validate_json(package.manifest_bytes)
        return CandidateRef(
            sha256, self.kind, path.resolve(), manifest.parent_package_sha256,
            None, manifest.generation,
            manifest.package_id,
        )


_ADAPTERS: dict[str, CandidateArtifactAdapter] = {
    STRUCTURED: StructuredBundleAdapter(),
    ARX: ArxPackageAdapter(),
}


def candidate_kind(runtime: dict[str, Any]) -> str:
    kind = runtime.get("candidate_kind", STRUCTURED)
    if kind not in _ADAPTERS:
        raise ValueError(f"unsupported candidate kind: {kind}")
    return str(kind)


def load_artifact(path: Path, sha256: str, kind: str) -> CandidateRef:
    if kind not in _ADAPTERS:
        raise ValueError(f"unsupported candidate kind: {kind}")
    return _ADAPTERS[kind].load(path, sha256)


def resolve_candidate_artifact(root: Path, runtime: dict[str, Any], sha256: str | None) -> str:
    if sha256 is None:
        return "none"
    kind = candidate_kind(runtime)
    configured = runtime.get("bundle_files_by_sha", {})
    candidates: list[Path] = []
    if isinstance(configured, dict) and sha256 in configured:
        selected = Path(str(configured[sha256]))
        candidates.append(selected if selected.is_absolute() else root / selected)
    if kind == STRUCTURED:
        candidates.append(root / "candidates" / sha256 / "bundle.json")
        candidates.extend(sorted((root / "bundles").glob("*.json")))
    else:
        candidates.append(root / "candidates" / sha256 / "package")
    for path in candidates:
        if path.exists():
            try:
                if kind == STRUCTURED:
                    if not path.is_file() or canonical_sha256(read_json(path)) != sha256:
                        continue
                else:
                    load_artifact(path, sha256, kind)
            except (ValueError, KeyError, OSError):
                continue
            return str(path.resolve())
    label = "bundle" if kind == STRUCTURED else "candidate"
    raise ValueError(f"frozen {label} artifact is missing: {sha256}")

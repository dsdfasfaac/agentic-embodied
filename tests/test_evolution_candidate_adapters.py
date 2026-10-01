# Copyright (c) 2026 Zetta Contributors
"""Candidate paths are resolved by bytes, not by mutable directory names."""

from pathlib import Path

import pytest

from robots.arx.critics.packages import load_candidate
from zetta.evolution.candidate_artifacts import ARX, candidate_kind, load_artifact, resolve_candidate_artifact


FIXTURE = Path(__file__).parent / "fixtures" / "arx_candidate_rgb_v1"


def test_arx_package_resolves_by_digest(tmp_path: Path) -> None:
    sha256 = load_candidate(FIXTURE).sha256
    runtime = {"candidate_kind": ARX, "bundle_files_by_sha": {sha256: str(FIXTURE)}}
    assert candidate_kind(runtime) == ARX
    resolved = resolve_candidate_artifact(tmp_path, runtime, sha256)
    ref = load_artifact(Path(resolved), sha256, ARX)
    assert ref.kind == ARX
    assert ref.sha256 == sha256
    assert ref.path == FIXTURE.resolve()
    with pytest.raises(ValueError, match="missing"):
        resolve_candidate_artifact(tmp_path, runtime, "0" * 64)


def test_arx_package_rejects_modified_payload(tmp_path: Path) -> None:
    import shutil

    candidate = tmp_path / "package"
    shutil.copytree(FIXTURE, candidate)
    sha256 = load_candidate(candidate).sha256
    (candidate / "skill" / "SKILL.md").write_text("modified skill")
    with pytest.raises(ValueError):
        load_artifact(candidate, sha256, ARX)

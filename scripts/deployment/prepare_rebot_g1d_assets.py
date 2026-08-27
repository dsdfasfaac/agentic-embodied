#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Safely prepare the minimal reBot G1-D MuJoCo bundle outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from rollout_runtime.backends.rebot_g1d_session import (
    compute_rebot_asset_manifest,
)

_AUDITED_CODE = (
    "simulation/g1d_mujoco_env.py",
    "simulation/g1d_mujoco_env_bin.py",
    "simulation/geometric_grasp.py",
    "scripts/sim_geometric_grasp.py",
    "scripts/sim_fallen_bottle_to_bin.py",
)
_CONFIGS = (
    "config/g1d_mujoco.yaml",
    "config/g1d_fallen_bottle_to_bin.yaml",
)
_URDF = "models/g1_d_description/g1_d_with_dex1_1.urdf"
_MAX_ENTRIES = 10_000
_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe zip member path: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the pinned reBot MuJoCo adapters, audited Agentic "
            "skills, G1-D scene configs, URDF, and meshes; model weights and "
            "generated outputs are excluded."
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-archive-sha256",
        required=True,
        help="required lowercase SHA-256 pin for the source zip",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    archive = args.archive.expanduser().resolve(strict=True)
    output = args.output.expanduser()
    if not output.is_absolute():
        raise ValueError("output must be an absolute path outside the repository")
    output = output.resolve(strict=False)
    repository = Path(__file__).resolve().parents[2]
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError(
            "refusing to place reBot simulator assets inside the repository"
        )
    expected = args.expected_archive_sha256
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "--expected-archive-sha256 must be 64 lowercase hex characters"
        )
    actual = _sha256(archive)
    if actual != expected:
        raise ValueError(
            f"archive SHA-256 mismatch: expected {expected}, calculated {actual}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing asset directory: {output}")

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > _MAX_ENTRIES:
            raise ValueError(f"archive has too many entries: {len(infos)}")
        total_size = sum(info.file_size for info in infos)
        if total_size > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"archive expands beyond 1 GiB: {total_size} bytes")
        for info in infos:
            _safe_name(info.filename)
            if _is_symlink(info):
                raise ValueError(f"archive contains a symbolic link: {info.filename!r}")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member paths")

        primary_adapter = _AUDITED_CODE[0]
        adapter_suffix = f"/{primary_adapter}"
        adapters = [
            info.filename
            for info in infos
            if not info.is_dir()
            and (
                info.filename == primary_adapter
                or info.filename.endswith(adapter_suffix)
            )
        ]
        if len(adapters) != 1:
            raise ValueError(
                "archive must contain exactly one simulation/g1d_mujoco_env.py"
            )
        prefix = adapters[0][: -len(primary_adapter)]

        members = {
            info.filename[len(prefix) :]: info
            for info in infos
            if not info.is_dir() and info.filename.startswith(prefix)
        }
        urdf_info = members.get(_URDF)
        if urdf_info is None:
            raise ValueError(f"archive is missing required reBot asset: {_URDF}")
        try:
            urdf_root = ET.fromstring(bundle.read(urdf_info))
        except ET.ParseError as exc:
            raise ValueError(f"archive contains an invalid reBot URDF: {exc}") from exc
        required_meshes: set[str] = set()
        for mesh in urdf_root.iter("mesh"):
            filename = mesh.get("filename")
            if not filename:
                raise ValueError("reBot URDF contains a mesh without filename")
            mesh_name = _safe_name(filename)
            relative = PurePosixPath(_URDF).parent.joinpath(mesh_name).as_posix()
            required_meshes.add(_safe_name(relative).as_posix())

        selected: list[tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            relative = info.filename[len(prefix) :]
            if (
                relative in _AUDITED_CODE
                or relative in _CONFIGS
                or relative == _URDF
                or relative in required_meshes
            ):
                selected.append((info, relative))
        selected_names = {relative for _info, relative in selected}
        required = {*_AUDITED_CODE, _URDF, *_CONFIGS, *required_meshes}
        missing = sorted(required - selected_names)
        if missing:
            raise ValueError(f"archive is missing required reBot assets: {missing}")
        if not required_meshes:
            raise ValueError("archive contains no G1-D mesh assets")

        output.mkdir(parents=True)
        for info, relative in selected:
            destination = output.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)

    scene_manifests = {
        scene_config: compute_rebot_asset_manifest(output, scene_config)
        for scene_config in _CONFIGS
    }
    manifest = {
        "schema_version": 2,
        "source_archive": archive.name,
        "source_archive_sha256": actual,
        "selected_file_count": len(selected),
        "excluded_file_count": len(infos) - len(selected),
        "scenes": scene_manifests,
    }
    manifest_path = output / "asset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

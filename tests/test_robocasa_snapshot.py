# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import importlib
import json
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def snapshot_module(monkeypatch):
    fake_mujoco = SimpleNamespace(
        mjtState=SimpleNamespace(
            mjSTATE_INTEGRATION=0,
            mjSTATE_TIME=1,
            mjSTATE_QPOS=2,
            mjSTATE_QVEL=3,
            mjSTATE_ACT=4,
            mjSTATE_WARMSTART=5,
            mjSTATE_CTRL=6,
            mjSTATE_QFRC_APPLIED=7,
            mjSTATE_XFRC_APPLIED=8,
            mjSTATE_EQ_ACTIVE=9,
            mjSTATE_MOCAP_POS=10,
            mjSTATE_MOCAP_QUAT=11,
            mjSTATE_USERDATA=12,
            mjSTATE_PLUGIN=13,
        ),
        mjtObj=SimpleNamespace(
            mjOBJ_BODY=1,
            mjOBJ_JOINT=2,
            mjOBJ_GEOM=3,
            mjOBJ_SITE=4,
            mjOBJ_ACTUATOR=5,
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    sys.modules.pop("robots.robocasa.snapshot", None)
    module = importlib.import_module("robots.robocasa.snapshot")
    yield module, fake_mujoco
    sys.modules.pop("robots.robocasa.snapshot", None)


def test_python_state_round_trip_preserves_numeric_nested_values(
    snapshot_module, monkeypatch
) -> None:
    module, _ = snapshot_module
    owner = SimpleNamespace(
        scalar=7,
        nested={"goal": np.array([1.0, 2.0]), "flags": [True, 3.5]},
        ignored="text",
    )
    monkeypatch.setattr(module, "python_state_owners", lambda env: {"owner": owner})

    arrays, manifest = module.capture_python_state(object())
    owner.scalar = -1
    owner.nested["goal"][:] = -2
    owner.nested["flags"] = [False, -3.5]
    module.restore_python_state(object(), arrays, manifest)

    assert owner.scalar == 7
    np.testing.assert_array_equal(owner.nested["goal"], [1.0, 2.0])
    assert owner.nested["flags"] == [True, 3.5]
    assert owner.ignored == "text"


def test_model_identity_fingerprints_large_static_assets(
    snapshot_module, monkeypatch
) -> None:
    module, fake_mujoco = snapshot_module
    raw_model = SimpleNamespace(
        nq=1,
        nv=1,
        na=0,
        nu=1,
        nbody=1,
        njnt=1,
        ngeom=1,
        nsite=1,
        ncam=0,
        nlight=0,
        neq=0,
        ntendon=0,
        nmat=1,
        ntex=1,
        nmesh=1,
        nmeshvert=1,
        nmeshface=1,
        tex_data=np.array([1, 2, 3], dtype=np.uint8),
        mesh_vert=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
    )
    fake_mujoco.mj_id2name = lambda model, object_type, index: f"{object_type}:{index}"
    monkeypatch.setattr(
        module,
        "raw_model_and_data",
        lambda env: (SimpleNamespace(), raw_model, SimpleNamespace()),
    )

    before = module.model_identity(object())
    raw_model.tex_data[1] = 9
    after = module.model_identity(object())

    assert before["static_asset_sha256"] != after["static_asset_sha256"]
    assert before["sha256"] != after["sha256"]


def test_refresh_render_resources_reuploads_gpu_assets_and_closes_cache(
    snapshot_module, monkeypatch
) -> None:
    module, fake_mujoco = snapshot_module
    events: list[tuple[str, int] | str] = []
    render_context = SimpleNamespace(
        upload_texture=lambda index: events.append(("texture", index)),
        con=object(),
        gl_ctx=SimpleNamespace(make_current=lambda: events.append("current")),
    )
    sim = SimpleNamespace(
        _render_context_offscreen=render_context,
        _close_isolated_rgb_renderers=lambda: events.append("close"),
    )
    raw_model = SimpleNamespace(ntex=2, nmesh=2, nhfield=1)
    fake_mujoco.mjr_uploadMesh = lambda model, context, index: events.append(
        ("mesh", index)
    )
    fake_mujoco.mjr_uploadHField = lambda model, context, index: events.append(
        ("hfield", index)
    )
    monkeypatch.setattr(
        module,
        "raw_model_and_data",
        lambda env: (sim, raw_model, SimpleNamespace()),
    )

    assert module.refresh_render_resources(object()) is True
    assert events == [
        ("texture", 0),
        ("texture", 1),
        "current",
        ("mesh", 0),
        ("mesh", 1),
        ("hfield", 0),
        "close",
    ]


def test_save_snapshot_uses_pickle_free_compressed_npz(
    snapshot_module, monkeypatch, tmp_path
) -> None:
    module, _ = snapshot_module
    arrays = {
        "physics": np.zeros(4096, dtype=np.float64),
        "model::geom_rgba": np.ones((8, 4), dtype=np.float32),
    }
    manifest = {"schema_version": module.SCHEMA_VERSION}
    monkeypatch.setattr(
        module,
        "capture_snapshot",
        lambda env: (arrays, manifest),
    )

    npz_path, json_path = module.save_snapshot(object(), tmp_path / "keyframe")

    assert json.loads(json_path.read_text()) == manifest
    with np.load(npz_path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["physics"], arrays["physics"])
    with zipfile.ZipFile(npz_path) as archive:
        assert archive.infolist()
        assert all(
            entry.compress_type == zipfile.ZIP_DEFLATED for entry in archive.infolist()
        )

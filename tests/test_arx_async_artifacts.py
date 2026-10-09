"""Slow or failed storage cannot bypass the step critic or lose evidence silently."""
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from robots.arx.gateway.artifacts import ArtifactWriter
from robots.arx.gateway.public import ImageStore
from scripts.deployment.prepare_arx_trial_storage import prepare
from tests.test_arx_gateway import make_core, call, limits, ScriptCritic


def test_blocked_image_disk_does_not_block_motion_or_critic(tmp_path, monkeypatch):
    from robots.arx.gateway import public
    entered, release = threading.Event(), threading.Event()
    original = public.atomic_write
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(public, 'atomic_write', blocked)
    critic = ScriptCritic({2: 'a'})
    core, backend, _ = make_core(tmp_path, critic, config=limits(async_sensor_storage=True))
    try:
        assert entered.wait(1)
        result, _ = call(core)
        assert backend.steps == 2
        assert result['status'] == 'interrupted'
        assert critic.seen == [1, 2]
        assert not release.is_set()
    finally:
        release.set()
        core.close()
        core.journal.close()


def test_background_png_bytes_match_reference_and_original_pixels(tmp_path):
    writer = ArtifactWriter()
    store = ImageStore(tmp_path/'images', writer)
    images = {k: np.full((20, 25, 3), i+10, np.uint8) for i,k in enumerate(
        ('front_rgb','left_rgb','right_rgb'))}
    original = {k:v.copy() for k,v in images.items()}
    try:
        refs, owned = store.publish(images)
        for pixels in images.values():
            pixels[:] = 255
        writer.flush()
        for k,ref in refs.items():
            raw = (store.root/(ref['content_id']+'.png')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref['sha256']
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(raw))), original[k])
            assert not owned[k].flags.writeable
    finally:
        writer.close()


def test_sensor_archive_record_only_after_persistence_and_close_flushes(tmp_path):
    core, _, _ = make_core(tmp_path, config=limits(async_sensor_storage=True))
    core._retain_grasp_sensors()
    core.close()
    rows = [(k,json.loads(v)) for k,v in core.journal.db.execute(
        'SELECT kind,payload FROM records ORDER BY sequence')]
    sensor = next(v for k,v in rows if k == 'grasp_sensor_evidence')
    path = tmp_path/sensor['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sensor['sha256']
    with np.load(path) as frames:
        assert set(frames.files) == {'front_rgb','left_rgb','right_rgb'}
    kinds = [k for k,_ in rows]
    assert kinds.index('grasp_sensor_evidence') < kinds.index('artifact_finalization')
    assert next(v for k,v in rows if k=='artifact_finalization')['finalization']=='complete'
    core.journal.close()


def test_write_failure_prevents_next_command_and_complete_marker(tmp_path, monkeypatch):
    from robots.arx.gateway import public
    def fail(*args):
        raise OSError('disk failed')
    monkeypatch.setattr(public, 'atomic_write', fail)
    core, backend, _ = make_core(tmp_path, config=limits(async_sensor_storage=True))
    for future in core.artifacts.pending:
        with pytest.raises(OSError):
            future.result(timeout=1)
    result,_ = call(core)
    assert backend.steps == 0
    assert result['error']['code'] == 'ARTIFACT_WRITE_FAILED'
    # The execution error path already closes and marks incomplete artifacts.
    assert core.closed
    markers = [json.loads(r[0]) for r in core.journal.db.execute(
        'SELECT payload FROM records WHERE kind=?', ('artifact_finalization',))]
    assert markers[-1]['finalization']=='incomplete'
    core.journal.close()


def test_bounded_queue_applies_backpressure_without_dropping(tmp_path):
    entered,release = threading.Event(),threading.Event()
    writer = ArtifactWriter(workers=1,capacity=1)
    def first():
        entered.set()
        assert release.wait(2)
        return ('sample', {'i':1})
    writer.submit(first)
    assert entered.wait(1)
    try:
        with ThreadPoolExecutor(max_workers=1) as owner:
            blocked = owner.submit(writer.submit, lambda: ('sample', {'i':2}))
            assert not blocked.done()
            release.set()
            blocked.result(timeout=1)
        assert writer.flush()==[('sample',{'i':1}),('sample',{'i':2})]
        assert writer.peak_pending==1
    finally:
        release.set()
        writer.close()


def test_storage_layout_resolves_small_journal_and_large_data_separately(tmp_path):
    output = tmp_path/'large/trial05'
    result = prepare(output,tmp_path/'ssd')
    gateway = output/'private/gateway'
    assert gateway.resolve() == Path(result['gateway'])
    assert (gateway/'public/images').resolve() == output/'sensor-artifacts/images'
    assert (gateway/'grasp-sensors').resolve() == output/'sensor-artifacts/grasp-sensors'
    with pytest.raises(ValueError,match='new'):
        prepare(output,tmp_path/'ssd')


def test_registered_pending_http_image_waits_for_background_bytes(tmp_path, monkeypatch):
    from robots.arx.gateway import public
    from robots.arx.gateway.journal import Journal
    from robots.arx.gateway.service import create_app
    from fastapi.testclient import TestClient
    release = threading.Event()
    original = public.atomic_write
    def blocked(*args):
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(public, 'atomic_write', blocked)
    writer = ArtifactWriter()
    store = ImageStore(tmp_path/'public/images',writer)
    images = {k:np.zeros((5,5,3),np.uint8) for k in ('front_rgb','left_rgb','right_rgb')}
    refs,_ = store.publish(images)
    journal = Journal(tmp_path/'journal.sqlite3')
    journal.record('ObservationPublished',{'cameras':refs},public=True)
    worker = SimpleNamespace(episode_id='test',lock=threading.RLock(),journal=journal,output=tmp_path)
    app = create_app(worker,agent_capability='a'*32,harness_capability='b'*32)
    ref = refs['front_rgb']
    try:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as requests:
            pending = requests.submit(client.get,'/v1/episodes/test/images/'+ref['content_id'],
                                      headers={'authorization':'Bearer '+'a'*32})
            assert not pending.done()
            release.set()
            response = pending.result(timeout=2)
            assert response.status_code==200
            assert hashlib.sha256(response.content).hexdigest()==ref['sha256']
    finally:
        release.set()
        writer.flush()
        writer.close()
        journal.close()

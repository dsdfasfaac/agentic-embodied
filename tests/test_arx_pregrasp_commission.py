"""Unsafe commissioning prefixes must reject before any motor command."""
import json
from types import SimpleNamespace

import pytest

from scripts.deployment import commission_arx_pregrasp as module


@pytest.mark.parametrize('tool,arguments', [
    ('arx.propose_grasp', {'engine': 'graspgen'}),
    ('arx.set_gripper', {'opening': 0.}),
    ('arx.execute_grasp', {'phase': 'engage'}),
    ('arx.zeva', {'max_chunks': 1}),
])
def test_non_pregrasp_prefix_rejects_before_reset_or_hold(tmp_path, monkeypatch, tool, arguments):
    events = []
    core = SimpleNamespace(
        programs={'test': SimpleNamespace(calls=[
            SimpleNamespace(tool='arx.execute_grasp', arguments={'phase': 'pregrasp'}),
            SimpleNamespace(tool=tool, arguments=arguments),
            SimpleNamespace(tool='arx.execute_grasp', arguments={'phase': 'pregrasp'}),
        ])}, current=None, closed=False,
        backend=SimpleNamespace(close=lambda: events.append('backend_closed')),
        journal=SimpleNamespace(close=lambda: events.append('journal_closed')),
        reset=lambda: events.append('reset'), execute=lambda request: events.append('motor_command'),
    )
    monkeypatch.setattr(module, 'RealCoreFactory', lambda **kwargs: lambda *args: core)
    for name in ('bundle', 'grasp-config'):
        (tmp_path / (name + '.json')).write_text('{}')
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    (frozen / 'real-input-contract.json').write_text('{}')
    args = SimpleNamespace(bundle=tmp_path / 'bundle.json',
        grasp_config=tmp_path / 'grasp-config.json', frozen=frozen,
        output=tmp_path / 'output', hardware_sha256='0' * 64, max_physical_steps=6)
    with pytest.raises(ValueError, match='only open geometric pregrasp'):
        module.run(args)
    assert events == ['backend_closed', 'journal_closed']
    assert json.loads((args.output / 'result.json').read_text())['status'] == 'failed_keep_enabled'


def test_full_grasp_rejects_closure_before_measured_engage(tmp_path, monkeypatch):
    events = []
    calls = [
        SimpleNamespace(tool='arx.hold', arguments={'steps': 15}),
        SimpleNamespace(tool='arx.propose_grasp', arguments={'engine': 'graspgen'}),
        SimpleNamespace(tool='arx.review_grasp', arguments={'phase': 'pregrasp'}),
        SimpleNamespace(tool='arx.execute_grasp', arguments={'phase': 'pregrasp'}),
        SimpleNamespace(tool='arx.set_gripper', arguments={'opening': 0.0}),
        SimpleNamespace(tool='arx.review_reentry', arguments={}),
    ]
    core = SimpleNamespace(programs={'test': SimpleNamespace(calls=calls)},
        current=None, closed=False,
        backend=SimpleNamespace(close=lambda: events.append('backend_closed')),
        journal=SimpleNamespace(close=lambda: events.append('journal_closed')),
        reset=lambda: events.append('reset'))
    monkeypatch.setattr(module, 'RealCoreFactory', lambda **kwargs: lambda *args: core)
    (tmp_path / 'bundle.json').write_text('{}')
    (tmp_path / 'grasp-config.json').write_text(json.dumps({'learned_grasp_commissioning': True}))
    frozen = tmp_path / 'frozen'; frozen.mkdir()
    (frozen / 'real-input-contract.json').write_text('{}')
    args = SimpleNamespace(bundle=tmp_path / 'bundle.json', grasp_config=tmp_path / 'grasp-config.json',
        frozen=frozen, output=tmp_path / 'output', hardware_sha256='0'*64,
        max_physical_steps=600, full_grasp=True)
    with pytest.raises(ValueError, match='ordered pregrasp, engage, close'):
        module.run(args)
    assert events == ['backend_closed', 'journal_closed']


def test_frozen_narrow_at_home_program_validates_before_hardware_reset(tmp_path, monkeypatch):
    from pathlib import Path
    from robots.arx.deployment.real_input import _load_bundle
    from robots.arx.deployment.bundle_program import compile_programs
    root=Path(__file__).resolve().parents[1]
    d=root/'docs/experiments/arx-graspgen-live-20261008'
    candidate,_=_load_bundle(d/'bundle-preshape-commission.json')
    programs=compile_programs(candidate,max_tool_calls=16,max_physical_steps=600,nominal_chunk_steps=16)
    events=[]
    def reset():
        events.append('reset');raise RuntimeError('validated_before_motion')
    core=SimpleNamespace(programs=programs,current=None,closed=False,
        backend=SimpleNamespace(close=lambda:events.append('backend_closed')),
        journal=SimpleNamespace(close=lambda:events.append('journal_closed')),reset=reset)
    monkeypatch.setattr(module,'RealCoreFactory',lambda **kw:lambda *a:core)
    args=SimpleNamespace(bundle=d/'bundle-preshape-commission.json',
        grasp_config=d/'grasp-config-preshape-commission.json',frozen=d/'frozen-preshape',
        output=tmp_path/'out',hardware_sha256='0'*64,max_physical_steps=600,full_grasp=True)
    with pytest.raises(RuntimeError,match='validated_before_motion'):module.run(args)
    assert events==['reset','backend_closed','journal_closed']

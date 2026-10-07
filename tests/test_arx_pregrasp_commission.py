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

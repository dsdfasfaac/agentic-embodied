"""Missing target evidence cannot masquerade as success or authorize a new write."""
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from robots.arx.deployment.feature_observation import FeatureObservationUnavailable
from robots.arx.deployment.real_input import RealFeatureSource
from robots.arx.gateway.backend import HardwareEvidence
from robots.arx.gateway.bundle_runtime import BundleMonitor, RealFeatureProvider, RealBundleReentry
from robots.arx.gateway.contracts import Assessment
from tests.test_arx_bundle_execution import bundle
from tests.test_arx_gateway import FakeBackend, make_core, limits, call, ScriptCritic
from zetta.evolution.critic import TemporalCritic


def test_unknown_clears_temporal_dwell_instead_of_triggering_numeric_rule():
    candidate = bundle()
    rule = replace(candidate.critic_rules[0], dwell_steps=2)
    critic = TemporalCritic((rule,))
    assert not critic.evaluate({'real_error': .2}, step_index=1)
    assert not critic.evaluate({'real_error': None}, step_index=2, unavailable_features={'real_error'})
    assert not critic.evaluate({'real_error': .2}, step_index=3)
    assert critic.evaluate({'real_error': .2}, step_index=4)


def test_expected_loss_is_nullable_and_stale_devices_still_error():
    source = RealFeatureSource(name='real_error', source_kind='camera_rgbd', source_ids=['front_rgb'],
        scalar_type='number', units='m', provider_id='test', provider_sha256='a'*64, max_age_ms=150)
    def missing(*args):
        raise FeatureObservationUnavailable('target hidden', available={}, unavailable=['real_error'],
                                            reason_code='target_not_visible')
    adapter = RealFeatureProvider.__new__(RealFeatureProvider)
    adapter.sources = [source]
    adapter.impl = SimpleNamespace(observe=missing)
    stamp = time.monotonic_ns()
    obs = {'observation_id':'obs-1', 'step_index':1, 'hardware':{
        'observation_completed_ns':stamp, 'camera_monotonic_ns':{'front_rgb':stamp}}}
    measured = adapter.augment(obs, {})
    assert measured['real_error'] is None
    monitor = BundleMonitor(bundle(), adapter)
    assert monitor.observe(obs, {}).status == 'unknown'
    assert monitor.last_feature_evidence['features']['real_error'] is None
    obs['hardware']['camera_monotonic_ns']['front_rgb'] -= 200_000_000
    with pytest.raises(ValueError, match='stale real feature'):
        adapter.augment(obs, {})


class SensorBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.reads = 0
    def commit(self):
        return replace(super().commit(), hardware=HardwareEvidence({}, None, True, {}))
    def observe(self):
        self.reads += 1
        return self.commit()


class MissingCritic(ScriptCritic):
    def __init__(self, backend, persistent=False):
        super().__init__()
        self.backend = backend
        self.persistent = persistent
        self.write_counts_at_assessment = []
    def observe(self, obs, images):
        self.write_counts_at_assessment.append(self.backend.steps)
        missing = obs['step_index'] == 1 and (self.persistent or self.backend.reads < 2)
        return Assessment(critic_id='test', observation_id=obs['observation_id'], step_index=obs['step_index'],
            status='unknown' if missing else 'clear', features={'feature_observation':{
                'status':'unknown' if missing else 'observed', 'unavailable_features':['target'] if missing else []}})


def test_short_occlusion_reads_only_then_resumes_same_plan(tmp_path):
    backend = SensorBackend()
    critic = MissingCritic(backend)
    core, _, _ = make_core(tmp_path, critic, backend, config=limits(observation_reacquire_timeout_s=.3))
    result, _ = call(core)
    assert result['status'] == 'completed'
    assert backend.steps == 4 and backend.reads == 2
    assert critic.write_counts_at_assessment[:3] == [1,1,1]
    events = core.journal.events(0)
    assert any(e['kind']=='observation_reacquired' for e in events)
    assert not any(e['kind']=='interrupt' for e in events)


def test_persistent_occlusion_interrupts_without_error_or_disabling(tmp_path):
    backend = SensorBackend()
    critic = MissingCritic(backend, persistent=True)
    core, _, _ = make_core(tmp_path, critic, backend, config=limits(observation_reacquire_timeout_s=.12))
    result, _ = call(core)
    assert result['status'] == 'interrupted' and result['error'] is None
    assert result['result']['completion'] == 'observation_unavailable'
    assert result['executed_steps'] == backend.steps == 1
    assert core.state == 'INTERRUPTED' and not backend.closed
    assert core.incidents == 0
    interrupts = [e['payload'] for e in core.journal.events(0) if e['kind']=='interrupt']
    assert interrupts[-1]['code'] == 'OBSERVATION_UNAVAILABLE'
    assert interrupts[-1]['task_failure'] is False


def test_unknown_blocks_reentry_and_success():
    class Provider:
        sources = [SimpleNamespace(name='real_error', scalar_type='number', provider_sha256='a'*64),
                   SimpleNamespace(name='success', scalar_type='boolean', provider_sha256='a'*64)]
        def augment(self, obs, images):
            return dict(obs, real_error=None, success=None, feature_observation={
                'status':'unknown','unavailable_features':['real_error','success']})
    provider=Provider()
    monitor=BundleMonitor(bundle(), provider, terminal_feature='success')
    obs={'observation_id':'obs-2','step_index':2}
    assert monitor.observe(obs,{}).status == 'unknown'
    assert monitor.completion_evidence() is None
    reviewer=RealBundleReentry(bundle(),provider,require_hardware=False)
    outcome=reviewer.inspect(None,{'observations':[obs], 'images':[{}], 'policy_id':'recover','recovery_id':'r'})
    assert outcome['status'] != 'eligible'
    assert any(x['status']=='unknown' for x in outcome['checks'])


def test_runner_reports_observation_loss_without_a_recovery_binding(tmp_path):
    from tests.test_arx_deployment import runner
    backend=SensorBackend()
    core,_,_=make_core(tmp_path/'core', MissingCritic(backend,persistent=True), backend,
                        config=limits(observation_reacquire_timeout_s=.06))
    call(core)
    run=runner(tmp_path/'runner',core)
    assert run.loop() == 'observation_unavailable'
    assert run.outcome.task_success is None


def wrist_provider(monkeypatch):
    from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
    from robots.arx.gateway.motion import CommandKinematics
    p=PickTubeRgbdProvider()
    p.closed_policy=0.; p.open_policy=-3.4
    p.wrist_mount=np.eye(4);p.wrist_mount_sha='a'*64;p.wrist_intrinsics={};p.wrist_tracker=SimpleNamespace(last_centre=(0.,0.))
    p.link6_fk=CommandKinematics(p.controller_fk.calibration.model_copy(update={'tcp_offset':[0.,0.,0.]}))
    def sample(rgb,depth,tracker,transform,deproject,camera,hardware):
        point=rgb
        if point is None:
            raise ValueError('pink tube label is not reliably visible in front RGB')
        return {'camera':camera,'stamp':hardware['camera_monotonic_ns'][camera],
                'point_left':np.array(point),'transform_left':transform}
    monkeypatch.setattr(p,'_label_sample',sample)
    return p


def hardware_for_provider(p,stamp):
    from robots.arx.deployment.picktube_rgbd_provider import RIGHT_SERIAL,RIGHT_INTRINSICS_SHA256
    state=np.zeros(14);state[13]=-2.5
    ee,_,_=p.controller_fk.fk(state[7:13])
    return {'measured_state':state.tolist(),'right_tcp_xyz_m':ee.tolist(),
        'right_tcp_monotonic_ns':stamp,'state_monotonic_ns':stamp,'right_tcp_frame':'right_arm_local_base',
        'camera_health':{'right_rgb':{'device_id':RIGHT_SERIAL}},
        'camera_calibration_sha256':{'right_rgb':RIGHT_INTRINSICS_SHA256},
        'camera_monotonic_ns':{'front_rgb':stamp,'right_rgb':stamp},
        'depth_monotonic_ns':{'front_depth_mm':stamp,'right_depth_mm':stamp},
        'auxiliary_feedback':{'right_gripper_current_native':.07},
        'auxiliary_monotonic_ns':{'right_gripper_current_native':stamp}}


def test_wrist_handoff_needs_three_agreements_and_cancels_source_height_bias(monkeypatch):
    p=wrist_provider(monkeypatch)
    front=np.array([.2,-.3,.1]);right=front+[0.,0.,.006]
    images={'front_rgb':front,'right_rgb':right}
    for stamp in range(1,4):
        values=p.observe({'observation_id':f'obs-{stamp}','hardware':hardware_for_provider(p,stamp)},images)
        assert p.last_observation_quality['camera']=='front_rgb'
    values=p.observe({'observation_id':'obs-4','hardware':hardware_for_provider(p,4)},
                     {'front_rgb':None,'right_rgb':right})
    assert p.last_observation_quality['camera']=='right_rgb'
    assert values['privileged.interaction.lift_m']==0.
    assert not values['privileged.interaction.success']
    # A repeated camera frame cannot contribute another hold frame.
    p.success_hold_frames=4
    with pytest.raises(FeatureObservationUnavailable,match='frame is not new'):
        p.observe({'observation_id':'obs-5','hardware':hardware_for_provider(p,4)},
                  {'front_rgb':None,'right_rgb':right})
    assert p.success_hold_frames==0


def test_unverified_or_disagreeing_wrist_does_not_replace_hidden_target(monkeypatch):
    p=wrist_provider(monkeypatch)
    front=np.array([.2,-.3,.1]);right=front+[0.,0.,.006]
    p.observe({'observation_id':'obs-1','hardware':hardware_for_provider(p,1)},
              {'front_rgb':front,'right_rgb':right})
    with pytest.raises(FeatureObservationUnavailable):
        p.observe({'observation_id':'obs-2','hardware':hardware_for_provider(p,2)},
                  {'front_rgb':None,'right_rgb':right})
    for stamp in range(3,6):
        p.observe({'observation_id':f'obs-{stamp}','hardware':hardware_for_provider(p,stamp)},
                  {'front_rgb':front,'right_rgb':front+[.03,0.,0.]})
    assert p.wrist_validations==0
    with pytest.raises(FeatureObservationUnavailable):
        p.observe({'observation_id':'obs-6','hardware':hardware_for_provider(p,6)},
                  {'front_rgb':None,'right_rgb':right})


def test_occluded_front_distractor_does_not_override_validated_wrist(monkeypatch):
    p=wrist_provider(monkeypatch)
    front=np.array([.2,-.3,.1]);right=front+[0.,0.,.006]
    for stamp in range(1,4):
        p.observe({'observation_id':f'obs-{stamp}','hardware':hardware_for_provider(p,stamp)},
                  {'front_rgb':front,'right_rgb':right})
    values=p.observe({'observation_id':'obs-4','hardware':hardware_for_provider(p,4)},
                      {'front_rgb':front+[.1,0.,.1],'right_rgb':right})
    assert p.last_observation_quality['camera']=='right_rgb'
    assert p.wrist_validations >= 3
    assert values['privileged.interaction.lift_m']==0.
    assert not values['privileged.interaction.success']


def test_confirmed_wrist_identity_survives_temporary_loss_but_requires_fresh_target(monkeypatch):
    p=wrist_provider(monkeypatch)
    front=np.array([.2,-.3,.1]);right=front+[0.,0.,.006]
    for stamp in range(1,4):
        p.observe({'observation_id':f'obs-{stamp}','hardware':hardware_for_provider(p,stamp)},
                  {'front_rgb':front,'right_rgb':right})
    old_centre=p.wrist_tracker.last_centre
    with pytest.raises(FeatureObservationUnavailable):
        p.observe({'observation_id':'obs-4','hardware':hardware_for_provider(p,4)},
                  {'front_rgb':None,'right_rgb':None})
    assert p.wrist_validations == 3 and p.wrist_tracker.last_centre == old_centre
    with pytest.raises(FeatureObservationUnavailable):
        p.observe({'observation_id':'obs-5','hardware':hardware_for_provider(p,5)},
                  {'front_rgb':None,'right_rgb':right+[.1,0.,0.]})
    values=p.observe({'observation_id':'obs-6','hardware':hardware_for_provider(p,6)},
                      {'front_rgb':None,'right_rgb':right})
    assert p.last_observation_quality['camera']=='right_rgb'
    assert not values['privileged.interaction.success']


def test_sparse_depth_on_visible_label_cannot_disprove_wrist_identity():
    from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
    p=PickTubeRgbdProvider()
    mask=np.zeros((240,320),dtype=bool);mask[40:50,100:110]=True
    tracker=SimpleNamespace(_pink_component=lambda rgb:mask)
    depth=np.zeros((240,320),dtype=np.uint16);depth[40:46,100:110]=500
    with pytest.raises(ValueError,match='insufficient valid metric depth'):
        p._label_sample(np.zeros((240,320,3),dtype=np.uint8),depth,tracker,np.eye(4),
                        lambda x,y,z:np.array([x,y,z]),'front_rgb',{})

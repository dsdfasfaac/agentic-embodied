"""A missing frame is admissible for empty return only with a fresh paused same-pose witness."""
from copy import deepcopy

import pytest

from scripts.deployment.replay_arx_picktube_home import resolve_empty_feature_witnesses


def rows():
    f={'observation_id':'obs-1','feature_observation':{'status':'unknown'},'features':{}}
    valid={'observation_id':'obs-1-reacquire-1','feature_observation':{'status':'observed'},
           'features':{'privileged.interaction.'+n:False for n in ('gripper_contact','grasped','success')}}
    valid['features']['privileged.selected.target_gripper_distance_m']=.35
    obs={'observation_id':'obs-1','step_index':1,'hardware':{'measured_state':[0.]*14,
         'arrival_verified':True,'observation_completed_ns':100}}
    after=deepcopy(obs);after['observation_id']=valid['observation_id']
    after['hardware']['observation_completed_ns']=200;after['hardware']['measured_state'][7]=.006
    return [(1,'ObservationPublished',obs),(2,'real_feature_evidence',f),
            (3,'observation_wait_started',{'step':1,'robot_commands_sent':False}),
            (4,'ObservationPublished',after),(5,'real_feature_evidence',valid),
            (6,'observation_reacquired',{'step':1,'observation_id':valid['observation_id']}),
            (7,'step_intent',{'step':2})]


def test_readonly_reacquisition_retains_unknown_and_records_actual_witness():
    source=rows();features,resolved=resolve_empty_feature_witnesses(source)
    assert source[1][2]['feature_observation']['status']=='unknown'
    assert len(features)==1
    assert resolved['obs-1']['observation_id']=='obs-1-reacquire-1'
    assert resolved['obs-1']['maximum_joint_drift_rad']==pytest.approx(.006)


@pytest.mark.parametrize('failure',['motion','drift','held','no_wait','no_arrival','stale'])
def test_unresolved_or_changed_pose_cannot_authorize_return(failure):
    source=rows()
    if failure=='motion':source.insert(3,(3.5,'command_sent',{'command_id':'unexpected'}))
    if failure=='drift':source[3][2]['hardware']['measured_state'][7]=.04
    if failure=='held':source[4][2]['features']['privileged.interaction.grasped']=True
    if failure=='no_wait':source.pop(2)
    if failure=='no_arrival':source[3][2]['hardware']['arrival_verified']=False
    if failure=='stale':source[3][2]['hardware']['observation_completed_ns']=100
    with pytest.raises(ValueError):resolve_empty_feature_witnesses(source)

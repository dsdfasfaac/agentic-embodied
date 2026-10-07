"""Continuation cannot invent engage evidence or refund the closing budget."""
import json
import sqlite3
from types import SimpleNamespace
import pytest
from robots.arx.deployment.bundle_program import ProgramCall, RecoveryProgram, resolve_call, retain_tool_outputs
from robots.arx.deployment.grasp_continuation import read_checkpoint
from zetta.evolution.jsonio import file_sha256


def checkpoint_fixture(tmp_path, *, verified=True):
    binding=SimpleNamespace(binding_id='grasp')
    entries=[
        ProgramCall(0,0,'arx.propose_grasp',{'engine':'tube_geometry','max_candidates':1},''),
        ProgramCall(1,0,'arx.review_grasp',{'proposal_id':'proposal-from-last','phase':'pregrasp','max_steps':60},''),
        ProgramCall(2,0,'arx.execute_grasp',{'review_token':'grasp-token-from-review','phase':'pregrasp','max_steps':60},''),
        ProgramCall(3,0,'arx.review_grasp',{'proposal_id':'proposal-from-last','phase':'engage','max_steps':60},''),
        ProgramCall(4,0,'arx.execute_grasp',{'review_token':'grasp-token-from-review','phase':'engage','max_steps':60},''),
        ProgramCall(5,0,'arx.set_gripper',{'opening':0.,'max_steps':60},'')]
    program=RecoveryProgram(binding,tuple(entries));g=tmp_path/'private/gateway';g.mkdir(parents=True)
    j=g/'journal.sqlite3';db=sqlite3.connect(j)
    db.executescript('create table records(sequence integer primary key,kind text,payload text); create table operations(request_id text,request text);create table snapshot(payload text);')
    def record(kind,value):db.execute('insert into records(kind,payload) values(?,?)',(kind,json.dumps(value)))
    record('interrupt',{'recovery_context':dict(binding_id='grasp',remaining_steps=180,remaining_decisions=10)})
    record('tool_result',dict(tool='arx.zeva',result={'completion':'critic_interrupted'}))
    record('recovery_acknowledged',{'binding_id':'grasp'})
    outputs={}
    for i,e in enumerate(entries):
        rid=str(i);args=resolve_call(e,'old-obs',None,outputs)
        db.execute('insert into operations values(?,?)',(rid,json.dumps(dict(tool=e.tool,arguments=args,observation_id='old-obs'))))
        value=dict(command_target_reached=True,physical_arrival_verified=verified if i==4 else True)
        if e.tool=='arx.propose_grasp':value.update(proposal_id='p',candidates=[{}],observation_id='old-obs')
        if e.tool=='arx.review_grasp':value.update(eligible=True,review_token='r'+rid)
        if i==5:value.update(completion='observation_unavailable',command_target_reached=False)
        result=dict(request_id=rid,tool=e.tool,status='interrupted' if i==5 else 'completed',executed_steps=18 if i==5 else 0,result=value)
        record('tool_result',result)
        if i!=5:retain_tool_outputs(e,result,outputs)
    for i in range(18):record('command_sent',{'command_id':str(i)});record('arrival_observed',{'verified':True})
    db.execute('insert into snapshot values(?)',(json.dumps(dict(observation={'hardware':{'measured_state':[0.]*14}},budget_remaining={'decisions':52})),))
    db.commit();db.close()
    (tmp_path/'result.json').write_text(json.dumps(dict(termination_reason='observation_unavailable',package_sha256='a'*64,counts={'recoveries':1,'physical_steps':18})))
    return j,program


def test_checkpoint_preserves_consumed_budget_and_verified_phase(tmp_path):
    journal,program=checkpoint_fixture(tmp_path)
    value=read_checkpoint(tmp_path,file_sha256(journal),program,'a'*64)
    assert value['steps']==18 and value['decisions']==12
    assert value['closing_steps_remaining']==42 and value['program'].calls[-1].arguments['max_steps']==42
    assert value['recovery']['remaining_steps']==162
    assert value['recovery']['remaining_decisions']==4
    assert value['phases']['p']=={'pregrasp','engage'}


def test_checkpoint_rejects_unmeasured_engage(tmp_path):
    journal,program=checkpoint_fixture(tmp_path,verified=False)
    with pytest.raises(ValueError,match='arrival unverified'):
        read_checkpoint(tmp_path,file_sha256(journal),program,'a'*64)


def test_checkpoint_rejects_changed_journal_and_candidate(tmp_path):
    journal,program=checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA differs'):read_checkpoint(tmp_path,'f'*64,program,'a'*64)
    with pytest.raises(ValueError,match='same bundle'):read_checkpoint(tmp_path,file_sha256(journal),program,'b'*64)

"""After-close input-boundary audit; does not certify an OS-level sandbox."""
import argparse
import hashlib
import json
from pathlib import Path

FORBIDDEN_KEYS={'reward','measured_state','measured_joints','object_pose','target_position',
                'finger_contacts','full_pickup_private','evaluation','audit','posthoc_trace'}


def forbidden_paths(value,path=''):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            child=f'{path}/{key}'
            if key in FORBIDDEN_KEYS:found.append(child)
            found.extend(forbidden_paths(item,child))
    elif isinstance(value,list):
        for i,item in enumerate(value):found.extend(forbidden_paths(item,f'{path}/{i}'))
    return found


def audit(capsule,private,closed_path):
    # Public completion marker must be verified before reading episode logs.
    closed=json.loads(closed_path.read_text())
    if closed.get('failed') or not closed.get('explicit_agent_finish'):
        raise ValueError('require successful worker close and explicit Agent finish')
    frozen=json.loads((private/'frozen_input_manifest.json').read_text())
    mismatches=[name for name,digest in frozen['skill_sha256'].items()
        if not (capsule/name).is_file() or hashlib.sha256((capsule/name).read_bytes()).hexdigest()!=digest]
    exposures=[json.loads(line) for line in (private/'agent_exposure.jsonl').read_text().splitlines()]
    violations=[{'exposure_index':i,'path':path} for i,event in enumerate(exposures)
                for path in forbidden_paths(event)]
    decisions=[json.loads(line) for line in (private/'agent_decisions.jsonl').read_text().splitlines()]
    ids=[d['decision_id'] for d in decisions]
    seen=set();unseen_geometry=[]
    for event in exposures:
        if event.get('observation'):seen.add(event['observation']['frame'])
    for d in decisions:
        frames=d.get('args',{}).get('observation_frames',[])
        if not set(frames)<=seen:unseen_geometry.append(d['decision_id'])
    return dict(skill_unchanged=not mismatches,skill_hash_mismatches=mismatches,
        forbidden_fields_in_exposures=violations,unique_decision_ids=len(ids)==len(set(ids)),
        geometry_uses_published_snapshots=not unseen_geometry,online_decisions=len(decisions),
        explicit_finish=bool(decisions and decisions[-1]['tool']=='finish'),
        input_boundary_checks_passed=not mismatches and not violations and not unseen_geometry and len(ids)==len(set(ids)),
        limitation='Audits published inputs and declared context separation, not all filesystem access or OS isolation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capsule','private','closed'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=json.dumps(audit(a.capsule,a.private,a.closed),indent=2)
    if a.output:a.output.write_text(result+'\n')
    print(result)


if __name__=='__main__':main()

"""Check current gate against saved online-only evidence, without audit inputs."""
import argparse
import json
from pathlib import Path
from pregrasp_gate import assess


def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);a=p.parse_args()
    evidence=[json.loads(line) for line in (a.session/'pregrasp_proposals.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (a.session/'pregrasp_features.jsonl').read_text().splitlines()]
    for proposal in evidence:
        selected=[r for r in rows if r['frame'] in proposal['history_frames']]
        result=assess(selected,proposal['geometry'])
        assert result['eligible']==proposal['eligible'],(proposal,result)
        print(json.dumps({'frame':proposal['frame'],'reproduced_eligibility':result['eligible'],
                          'source':'online RGB/own-command evidence only'}))


if __name__=='__main__':main()

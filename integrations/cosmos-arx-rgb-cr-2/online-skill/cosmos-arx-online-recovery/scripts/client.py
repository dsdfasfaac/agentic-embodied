"""Restricted online transport. Reads only its capsule's connection metadata."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('state','send'))
    p.add_argument('decision',nargs='?')
    a=p.parse_args()
    c=json.loads((Path(__file__).resolve().parents[1]/'connection.json').read_text())
    data=json.dumps(json.loads(a.decision)).encode() if a.command=='send' else None
    request=urllib.request.Request(c['url']+('/action' if data else '/state'),data=data,
        headers={'Authorization':'Bearer '+c['token'],'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        print(response.read().decode())


if __name__=='__main__':main()

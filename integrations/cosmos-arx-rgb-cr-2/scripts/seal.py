"""Maintainer: seal the delivery after payload/provenance/docs are finalized."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = {}
for p in sorted(root.rglob('*')):
    relative = p.relative_to(root)
    if not p.is_file() or p.name == 'MANIFEST.json' or '__pycache__' in relative.parts or relative.parts[0] == 'runs':
        continue
    files[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
(root / 'MANIFEST.json').write_text(json.dumps(dict(version='1.1.0', runtime='v4.1_front_wrist_independent_agent',
    files_sha256=files, note='Package integrity only, not a cryptographic signature or proof of task success.'), indent=2))
print(f'Sealed {len(files)} files')

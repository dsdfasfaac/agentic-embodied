"""Maintainer: generate a portable archive without macOS resource forks or run data."""
import gzip
import hashlib
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
output = root.parent / 'cosmos-arx-rgb-cr-h20-v1.1.0.tar.gz'
with output.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz:
    with tarfile.open(fileobj=gz, mode='w') as archive:
        for p in [root, *sorted(root.rglob('*'))]:
            relative = p.relative_to(root)
            if '__pycache__' in relative.parts or (relative.parts and relative.parts[0] == 'runs') or p.name.startswith('._'):
                continue
            info = archive.gettarinfo(str(p), arcname=str(Path(root.name) / relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.pax_headers = {}
            info.mode = 0o755 if p.is_dir() else 0o644
            if p.is_file():
                with p.open('rb') as source:
                    archive.addfile(info, source)
            else:
                archive.addfile(info)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
(output.parent / (output.name + '.sha256')).write_text(f'{digest}  {output.name}\n')
print(output, output.stat().st_size, digest)

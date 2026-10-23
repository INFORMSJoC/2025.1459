#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];manifest=root/'scripts/config/SHA256SUMS.json'
if not manifest.exists():raise SystemExit('Release checksum manifest has not been generated')
issues=[];entries=json.loads(manifest.read_text())
for x in entries:
 p=root/x['path']
 if not p.is_file():issues.append('missing '+x['path']);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if p.stat().st_size!=x['bytes'] or h.hexdigest()!=x['sha256']:issues.append('changed '+x['path'])
print(json.dumps(dict(checked=len(entries),issues=issues),indent=2));sys.exit(bool(issues))

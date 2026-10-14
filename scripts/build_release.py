#!/usr/bin/env python3
"""Freeze a review ZIP and file hashes; never includes local depots/new runs/caches."""
from pathlib import Path
import hashlib,json,zipfile,argparse
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=root.parent/'HPRQP_reproducibility_review.zip');a=p.parse_args();dest=a.output.resolve()
if root==dest or root in dest.parents:raise SystemExit('ZIP output must be outside the package root')
def included(p):
 rel=p.relative_to(root)
 return p.is_file() and not p.is_symlink() and p.name!='SHA256SUMS.json' and not any(x in ['.venv','.julia-depot','__pycache__','.git','.CondaPkg','.pixi','.reviewer_locks'] for x in rel.parts) and rel.parts[:2] not in [('results','new'),('data','downloads')] and not (rel.parts[0]=='experiments' and 'runs' in rel.parts) and p.suffix not in ['.pyc','.tmp','.lock']
bundled={x['path'] for n in ['manifest.json','competitor_qap_manifest.json'] for x in json.loads((root/'data'/n).read_text()) if x.get('bundled')}
files=sorted(p for p in root.rglob('*') if included(p) and (p.relative_to(root).parts[:2]!=('data','inputs') or str(p.relative_to(root/'data/inputs')) in bundled));entries=[]
for p in files:
 if p.relative_to(root).parts[0]=='experiments' and 'raw' not in p.relative_to(root).parts:continue # Generated views; raw verification evidence stays checksummed.
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 entries.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
manifest=root/'SHA256SUMS.json';manifest.write_text(json.dumps(entries,indent=2)+'\n');files.append(manifest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.zip')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for p in files:z.write(p,root.name+'/'+str(p.relative_to(root)))
with zipfile.ZipFile(tmp) as z:
 bad=z.testzip()
 if bad:raise SystemExit('ZIP integrity failed: '+bad)
tmp.replace(dest);sha=hashlib.sha256(dest.read_bytes()).hexdigest();dest.with_suffix(dest.suffix+'.sha256').write_text(sha+'  '+dest.name+'\n');print(json.dumps(dict(zip=str(dest),files=len(files),bytes=dest.stat().st_size,sha256=sha),indent=2))

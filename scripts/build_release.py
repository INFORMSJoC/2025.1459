#!/usr/bin/env python3
"""Freeze a review ZIP with selected evidence, excluding unused attempts and caches."""
from pathlib import Path
import hashlib,json,zipfile,argparse,csv
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=root.parent/'HPRQP_reproducibility_review.zip');a=p.parse_args();dest=a.output.resolve()
if root==dest or root in dest.parents:raise SystemExit('ZIP output must be outside the package root')
selected_runs=set()
for table in (root/'results/reproduced_results').glob('*/solver_*/summary.csv'):
 with table.open(newline='') as f:
  for row in csv.DictReader(f):
   raw=Path(row.get('raw_file','').split('#')[0])
   if raw.is_absolute():
    try:raw=raw.relative_to(root)
    except ValueError:continue
   if 'runs' in raw.parts:selected_runs.add(Path(*raw.parts[:raw.parts.index('runs')+2]))
def included(p):
 rel=p.relative_to(root);parts=rel.parts
 if not p.is_file() or p.is_symlink() or p.name=='SHA256SUMS.json':return False
 if any(x in ['.venv','.julia-depot','__pycache__','.git','.CondaPkg','.pixi','.reviewer_locks','.reviewer_runtime'] for x in parts):return False
 if parts[0] not in ['src','data','scripts','results','docs'] and len(parts)>1:return False
 if parts[:2]==('results','setup_logs'):return False
 if parts[:2]==('results','reproduced_results') and 'runs' in parts and Path(*parts[:parts.index('runs')+2]) not in selected_runs:return False
 if parts[0]=='data' and 'inputs' in parts:return False
 return p.suffix not in ['.pyc','.tmp','.lock']
files=sorted(p for p in root.rglob('*') if included(p));entries=[]
for p in files:
 rel=p.relative_to(root)
 # Rebuilt CSV/HTML/PDF views can change after local runs; evidence stays checked.
 if rel.parts[:2] in [('results','paper_results'),('results','reproduced_results')] and 'raw' not in rel.parts and 'sources' not in rel.parts and p.suffix in ['.csv','.pdf'] and p.name!='summary.csv':continue
 if rel.parts[:2]==('results','reproduced_results') and p.name=='summary.csv':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 entries.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=h.hexdigest()))
manifest=root/'scripts/config/SHA256SUMS.json';manifest.write_text(json.dumps(entries,indent=2)+'\n');files.append(manifest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.zip')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for p in files:z.write(p,'HPRQP_reproducibility/'+str(p.relative_to(root)))
with zipfile.ZipFile(tmp) as z:
 bad=z.testzip()
 if bad:raise SystemExit('ZIP integrity failed: '+bad)
tmp.replace(dest);sha=hashlib.sha256(dest.read_bytes()).hexdigest();dest.with_suffix(dest.suffix+'.sha256').write_text(sha+'  '+dest.name+'\n');print(json.dumps(dict(zip=str(dest),files=len(files),bytes=dest.stat().st_size,sha256=sha),indent=2))

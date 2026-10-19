#!/usr/bin/env python3
"""Check the release inventory against the manuscript's explicit case list."""
from pathlib import Path
import json,sys,csv,tomllib
from catalog import load_catalog
r=Path(__file__).resolve().parents[1];gs=json.loads((r/'scripts/config/experiments.json').read_text());scope=json.loads((r/'scripts/config/paper_scope.json').read_text());m=json.loads((r/'scripts/config/data/manifest.json').read_text());issues=[]
actual={g['id']:[c['name'] for c in g['cases']] for g in gs}
if actual!=scope:issues.append('Experiment configuration differs from the paper scope')
variants={g['variant'] for g in gs}
if {x.name for x in (r/'src/variants').iterdir() if x.is_dir()}!=variants:issues.append('Missing or extra source variants')
paths={c['path'] for g in gs for c in g['cases']}
if {x['path'] for x in m}!=paths:issues.append('Data inventory differs from required paper inputs')
for x in m:
 expected={g['id'] for g in gs if any(c['path']==x['path'] for c in g['cases'])}
 if set(x['groups'])!=expected:issues.append('Incorrect input membership '+x['path'])
for f in (r/'data').rglob('*'):
 if f.is_file() and f.suffix in ['.mps','.mat'] and str(f.relative_to(r/'data')) not in ['Support/warmup.mps','Appendix-Example/instance.mps'] and str(f.relative_to(r/'data')) not in paths | {x['path'] for x in json.loads((r/'scripts/config/data/competitor_qap_manifest.json').read_text())}:issues.append('Extra bundled input '+str(f))
for g in gs:
 if not (r/'src/variants'/g['variant']/g['entry']).is_file():issues.append('Missing source entry '+g['id'])
 if g['probe'] not in actual[g['id']]:issues.append('Probe outside paper scope '+g['id'])
index=json.loads((r/'scripts/config/data/archives/archive_index.json').read_text())
if index.get('complete') and {x['path'] for d in index['datasets'] for x in d['inputs']}!=paths:issues.append('Completed archive index does not cover every paper input')
for d in index['datasets']:
 if any(x['path'] not in paths for x in d['inputs']):issues.append('Archive contains non-paper input '+d['dataset'])
for path in ['data/Appendix-Example/instance.mps','scripts/run_concrete_example.jl']:
 if not (r/path).is_file():issues.append('Missing appendix example file '+path)
for table in (r/'results/reproduced_results').glob('*/solver_*/summary.csv'):
 for row in csv.DictReader(table.open()):
  path=row.get('raw_file','').split('#')[0]
  if path and not (r/path).is_file():issues.append('Missing native result '+path)
cat=load_catalog()
inputs=m+json.loads((r/'scripts/config/data/competitor_qap_manifest.json').read_text())
inventory={x['path']:x for x in inputs}
extra_index=r/'scripts/config/data/archives/competitor_qap_index.json'
if extra_index.exists():
 extra=json.loads(extra_index.read_text())
 expected={x['path'] for x in inputs if x['path'].startswith('QAPLIB/inputs/explicit/')}
 actual_extra={x['path'] for d in extra['datasets'] for x in d['inputs']}
 if extra.get('complete') and expected!=actual_extra:issues.append('Competitor archive index does not cover exact QAP inventory')
 for d in extra['datasets']:
  for x in d['inputs']:
   if x['path'] not in inventory or any(x[k]!=inventory[x['path']][k] for k in ['bytes','sha256']):issues.append('Competitor archive input mismatch '+x['path'])
if len(inventory)!=len(inputs):issues.append('Duplicate canonical input paths')
for x in inputs:
 p=Path(x['path'])
 if p.is_absolute() or '..' in p.parts:issues.append('Nonportable canonical input '+str(p))
 if str(p).startswith('LASSO/') and p.parts[1] not in ['uci','synthetic']:issues.append('Obsolete Lasso input path '+str(p))
for manifest in (r/'scripts/environments').rglob('Manifest.toml'):
 for name,packages in tomllib.loads(manifest.read_text()).get('deps',{}).items():
  for package in packages:
   if 'path' in package:
    p=Path(package['path']);target=(manifest.parent/p).resolve()
    if p.is_absolute() or not target.is_dir() or r.resolve() not in target.parents:issues.append('Nonportable Julia dependency '+name+' in '+str(manifest.relative_to(r)))
for e in cat:
 for solver,sp in e['solvers'].items():
  for c in sp['cases']:
   if c['group']!='appendix_concrete':
    x=inventory.get(c['path'])
    if not x:issues.append('Uninventoried input '+c['path'])
    elif c.get('sha256')!=x['sha256'] or c.get('bytes')!=x['bytes']:issues.append('Catalog/input hash or size mismatch '+e['name']+'/'+solver+'/'+c['key'])
   for row in c['archived']:
    f=row.get('raw_file','').split('#')[0]
    if f and not (r/f).is_file():issues.append('Missing archived reference '+f)
print(json.dumps(dict(appendix_example='2 methods, separate tiny input',groups=len(gs),cases=sum(map(len,actual.values())),source_variants=len(variants),distinct_inputs=len(paths),inputs_including_competitors=len(inventory),portable_dependency_paths_checked=True,issues=issues),indent=2))
sys.exit(bool(issues))

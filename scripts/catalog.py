"""Join execution settings with the paper's per-experiment CSV results."""
from pathlib import Path
import csv,json
from result_paths import solver_path
ROOT=Path(__file__).resolve().parents[1]
NUMERIC={'tolerance','iterations','seconds','objective','residual','wall_seconds'}
def load_groups():
 groups=json.loads((ROOT/'scripts/config/experiments.json').read_text())
 for g in groups:
  with (ROOT/g['reference_csv']).open(newline='') as f:refs={r['name']:{k:v for k,v in r.items() if v!=''} for r in csv.DictReader(f)}
  for c in g['cases']:c['reference']=dict(refs[c['name']])
  if g.get('reference_curves_csv'):
   with (ROOT/g['reference_curves_csv']).open(newline='') as f:
    for r in csv.DictReader(f):
     ref=next(c['reference'] for c in g['cases'] if c['name']==r['case'])
     ref.setdefault('curve_iteration',[]).append(int(r['iteration']));ref.setdefault('curve_optimality',[]).append(float(r['optimality']))
 return groups

def load_catalog():
 catalog=json.loads((ROOT/'scripts/config/reviewer_experiments.json').read_text())
 for e in catalog:
  for solver,sp in e['solvers'].items():
   with (solver_path(ROOT/'results',e['name'],solver,True)/'summary.csv').open(newline='') as f:rows=list(csv.DictReader(f))
   for r in rows:
    for field in NUMERIC:r[field]=float(r[field]) if r.get(field,'')!='' else None
    if r['solved'].lower() not in ['true','false']:raise ValueError('Invalid solved flag')
    r['solved']=r['solved'].lower()=='true'
   for c in sp['cases']:c['archived']=[r for r in rows if r['case_id']==c['key']]
 return catalog

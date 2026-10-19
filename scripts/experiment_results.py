#!/usr/bin/env python3
"""Shared result schema and transparent archived/fresh experiment summaries."""
from pathlib import Path
import csv,json,math,html,os,tempfile,fcntl
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['solver','case_id','case','group','subset','tolerance','status','solved','iterations','seconds','objective','residual','source','measurement_origin','raw_file','status_basis','objective_basis','wall_seconds','note','model_revision']
def number(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None
def atomic_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
 with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
 os.replace(tmp,path)
def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r)) or FIELDS
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
 with os.fdopen(fd,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def read_csv(path):
 if not Path(path).exists():return []
 with Path(path).open() as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k in ['tolerance','iterations','seconds','objective','residual','wall_seconds']:
   if k in r:r[k]=number(r[k])
  r['solved']=str(r.get('solved','')).lower()=='true'
 return rows
def row(case,solver,tol,**kw):
 d={k:None for k in FIELDS};d.update(solver=solver,case_id=case['key'],case=case['name'],group=case.get('group',''),subset=case.get('subset','all'),tolerance=tol,source='fresh',measurement_origin='local_run',status_basis='native solver',objective_basis='full objective');d.update(kw);return d
def hpr_rows(case,solver,d,raw_file='',origin='local_run'):
 r=d.get('result',{});res=number(r.get('residuals'));final=number(r.get('iter'));rows=[]
 for t in case['tolerances']:
  suffix={1e-4:'4',1e-6:'6',1e-8:None}.get(t);it=number(r.get('iter_'+suffix)) if suffix else final
  tm=number(r.get('time_'+suffix)) if suffix else number(r.get('time'))
  solved=(res is not None and res<=t) or (suffix is not None and it is not None and final is not None and it<final)
  status='TOLERANCE_REACHED' if solved and suffix else r.get('output_type','EXECUTION_ERROR')
  rows.append(row(case,solver,t,status=status,solved=solved,iterations=it,seconds=tm,objective=number(r.get('primal_obj')) if not suffix else None,residual=res if not suffix else None,raw_file=raw_file,measurement_origin=origin,note=d.get('error','')))
 return rows
def jump_rows(case,solver,d,raw_file='',origin='local_run'):
 r=d.get('result',{});t=d['tolerance'];status=r.get('status','EXECUTION_ERROR')
 obj=r.get('native_primal_objective',r.get('pobj')) if solver in ['SCS','CuClarabel'] else r.get('objective')
 # The old SCS MM 1e-6 reference CSV uses the full JuMP objective.
 if case.get('objective_basis',{}).get(str(t))=='full objective':obj=r.get('objective')
 return [row(case,solver,t,status=status,solved=status=='OPTIMAL',iterations=number(r.get('iterations')),seconds=number(r.get('solve_time')),objective=number(obj),residual=number(r.get('res_primal',r.get('res_pri',r.get('ConstrVio')))),raw_file=raw_file,measurement_origin=origin,objective_basis=case.get('objective_basis',{}).get(str(t),'native log objective' if solver in ['SCS','CuClarabel'] else 'full objective'),note=d.get('error',''))]
def pdqp_rows(case,d,raw_file='',origin='local_run'):
 r=d['solution_stats'];conv=r.get('convergence_information',[{}])[0];rows=[];final=number(d.get('iteration_count'))
 for t in case['tolerances']:
  suffix={1e-4:'4',1e-6:'6',1e-8:None}[t];it=number(r.get('iter_'+suffix)) if suffix else final;tm=number(r.get('time_'+suffix)) if suffix else number(d.get('solve_time_sec'))
  optimal=d.get('termination_string')=='OPTIMAL';solved=optimal or (suffix is not None and it is not None and final is not None and it<final)
  rows.append(row(case,'PDQP',t,status='TOLERANCE_REACHED' if solved and suffix else d.get('termination_string','UNKNOWN'),solved=solved,iterations=it,seconds=tm,objective=number(conv.get('primal_objective')) if not suffix else None,residual=number(conv.get('relative_optimality_gap')) if not suffix else None,raw_file=raw_file,measurement_origin=origin))
 return rows
def sgm(values):
 return math.exp(sum(math.log(x+10) for x in values)/len(values))-10 if values else None
def refresh(experiment,output):
 """Publish one experiment atomically; every fresh row overrides its archived row, even failures."""
 folder=Path(output)/'paper_results/experiments'/experiment['name'];folder.mkdir(parents=True,exist_ok=True)
 with (folder/'.summary.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  effective=[];comparisons=[];summaries=[]
  for solver,spec in experiment['solvers'].items():
   sd=folder/solver;sd.mkdir(exist_ok=True)
   archive=[dict(r,source='archived',measurement_origin='paper_archive') for c in spec['cases'] for r in c['archived']]
   prior=[r for r in read_csv(ROOT/'results/raw_results/verification_snapshot.csv') if r.pop('experiment',None)==experiment['name'] and r['solver']==solver];fresh={(r['case_id'],r['tolerance']):r for r in prior};errors=0
   for f in sorted((Path(output)/'raw_results'/experiment['name']/solver/'runs').glob('*/record.json')):
    try:d=json.loads(f.read_text())
    except (OSError,ValueError):continue
    if d.get('state') not in ['completed']:errors+=1
    for r in d.get('rows',[]):fresh[r['case_id'],r['tolerance']]=r
   expected_models={c['key']:c.get('model_revision') for c in spec['cases']}
   unusable_archives={c['key'] for c in spec['cases'] if not c.get('archive_usable',True)}
   fresh={k:r for k,r in fresh.items() if not expected_models.get(k[0]) or r.get('model_revision')==expected_models[k[0]]}
   write_csv(sd/'archived.csv',archive,FIELDS);write_csv(sd/'fresh.csv',list(fresh.values()),FIELDS)
   used=[]
   for old in archive:
    new=fresh.get((old['case_id'],old['tolerance']));chosen=dict(new or old)
    if old['case_id'] in unusable_archives:
     chosen['note']='Corrected original-MPS model; historical HPR objective/iterations are not a valid reproduction target.'
     if not new:chosen.update(status='MODEL_CORRECTION_PENDING',solved=False,seconds=None,iterations=None,objective=None,residual=None,note='Corrected run pending; historical old-loader result excluded.')
    used.append(chosen);effective.append(chosen)
    delta=None
    if new and number(new.get('objective')) is not None and number(old.get('objective')) is not None and new.get('objective_basis')==old.get('objective_basis'):
     delta=abs(new['objective']-old['objective'])/max(1,abs(old['objective']))
    comparisons.append(dict(solver=solver,case=old['case'],group=old['group'],tolerance=old['tolerance'],source=chosen['source'],archived_status=old['status'],fresh_status=new.get('status') if new else None,archived_seconds=old['seconds'],fresh_seconds=new.get('seconds') if new else None,archived_iterations=old['iterations'],fresh_iterations=new.get('iterations') if new else None,archived_objective=old['objective'],fresh_objective=new.get('objective') if new else None,scaled_objective_difference=delta,raw_file=chosen.get('raw_file'),note=chosen.get('note')))
   subsets=['all']+sorted({r['subset'] for r in archive if r['subset'] not in ['all','',None]})
   for subset in subsets:
    for tol in sorted({r['tolerance'] for r in archive}):
     rs=[r for r in used if r['tolerance']==tol and (subset=='all' or r['subset']==subset)]
     old=[r for r in archive if r['tolerance']==tol and (subset=='all' or r['subset']==subset)]
     if not rs:continue
     fresh_count=sum(r['source']=='fresh' for r in rs);bad=[r for r in rs if r['status'] in ['HARD_TIME_LIMIT','MEMORY_LIMIT','INTERRUPTED','EXECUTION_ERROR','MISSING_RESULT','MODEL_CORRECTION_PENDING']]
     def times(xs):
      if any(r['solved'] and number(r['seconds']) is None for r in xs):return []
      return [min(float(r['seconds']),spec['time_limit']) if r['solved'] and number(r['seconds']) is not None else spec['time_limit'] for r in xs]
     # Curves/appendix have no defensible historical timing aggregate.
     aggregate=spec['kind'] not in ['curve','appendix']
     vals=times(rs) if aggregate and not bad else [];oldvals=times(old) if aggregate else []
     its=[number(r['iterations']) for r in rs]
     summaries.append(dict(solver=solver,subset=subset,tolerance=tol,cases=len(rs),fresh_cases=fresh_count,archived_cases=len(rs)-fresh_count,source='fresh' if fresh_count==len(rs) else 'mixed' if fresh_count else 'archived',solved=sum(r['solved'] for r in rs),archived_solved=sum(r['solved'] for r in old),time_SGM10=sgm(vals),archived_time_SGM10=sgm(oldvals),iteration_SGM10=sgm(its) if aggregate and all(x is not None for x in its) and not bad else None,complete_fresh=fresh_count==len(rs) and not bad,guard_or_execution_failures=len(bad),historical_attempt_errors=errors,curves_reproduced=sum(r['source']=='fresh' and r.get('note')=='Full-curve comparison: curve_reproduced' for r in rs) if spec['kind']=='curve' else None,note='Mixed sources: not a full fresh benchmark' if 0<fresh_count<len(rs) else 'External guard/interruption: timing aggregate withheld' if bad else 'Archived fallback' if not fresh_count else 'Fresh measurements; shared-hardware times may vary'))
  for summary in summaries:
   if any(r['solver']==summary['solver'] and r['status']=='MODEL_CORRECTION_PENDING' and r['tolerance']==summary['tolerance'] and (summary['subset']=='all' or r['subset']==summary['subset']) for r in effective):summary['note']='Corrected model rerun pending; historical scaled results excluded; aggregate withheld'
  write_csv(folder/'results.csv',effective,FIELDS);write_csv(folder/'comparison.csv',comparisons);write_csv(folder/'summary.csv',summaries)
  columns=['solver','subset','tolerance','cases','fresh_cases','archived_cases','source','solved','time_SGM10','archived_time_SGM10','historical_attempt_errors','curves_reproduced','note']
  def show(v):return '' if v is None else format(v,'.6g') if isinstance(v,float) else str(v)
  h=['<!doctype html><meta charset="utf-8"><title>'+html.escape(experiment['title'])+'</title><style>body{font:15px system-ui;margin:32px;color:#222}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eee}</style>', '<h1>'+html.escape(experiment['name']+': '+experiment['title'])+'</h1>','<p>'+html.escape(experiment['paper'])+'</p>','<p>Archived paper results are the baseline. Each available fresh row replaces its archived row, including unsuccessful solves. Unrun cases retain archived results. <strong>Mixed summaries are not full fresh benchmarks.</strong> Native failures use the paper time cap; guard stops and execution errors withhold the effective timing aggregate.</p>','<p><a href="summary.csv">Summary CSV</a> · <a href="comparison.csv">Archived versus fresh</a> · <a href="results.csv">Selected per-case results and sources</a></p>','<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>']
  if experiment.get('validation_note'):h.insert(3,'<p><strong>'+html.escape(experiment['validation_note'])+'</strong></p>')
  for r in summaries:h.append('<tr>'+''.join('<td>'+html.escape(show(r[c]))+'</td>' for c in columns)+'</tr>')
  h.append('</table><h2>Solver files</h2><ul>')
  for solver in experiment['solvers']:h.append('<li>'+html.escape(solver)+': <a href="'+solver+'/archived.csv">archived.csv</a> · <a href="'+solver+'/fresh.csv">fresh.csv</a></li>')
  h.append('</ul><p>Input provenance, settings, native residuals and logs are linked by raw_file. For curve experiments, curves_reproduced counts matching sample sequences; solved still counts attainment of the stopping tolerance. Different solver residual definitions are not a common KKT certificate. Iteration differences and historical manuscript discrepancies remain visible.</p>')
  (folder/'summary.html').write_text('\n'.join(h))
 return summaries

#!/usr/bin/env python3
"""Shared result schema and paper/reproduced experiment summaries."""
from pathlib import Path
import csv,json,math,os,tempfile,fcntl,hashlib
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
def save_run(folder,record):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 write_csv(folder/'run.csv',[{k:v for k,v in record.items() if k!='rows'}])
 write_csv(folder/'measurements.csv',record.get('rows',[]),FIELDS)

def load_run(folder):
 with (Path(folder)/'run.csv').open(newline='') as f:d=next(csv.DictReader(f))
 d['rows']=read_csv(Path(folder)/'measurements.csv')
 return d

def hpr_csv_record(path,name):
 with Path(path).open(newline='') as f:r=next(x for x in csv.DictReader(f) if x['name']==name)
 fields={'alg_time':'time','res':'residuals','status':'output_type'}
 return dict(execution='completed',result={fields.get(k,k):v if k=='status' else number(v) for k,v in r.items() if k!='name'})

def aggregate(rows,spec,expected,solver):
 summaries=[]
 for subset in ['all']+sorted({r.get('subset','all') for r in expected}-{'all','',None}):
  for tol in sorted({r['tolerance'] for r in expected}):
   old=[r for r in expected if r['tolerance']==tol and (subset=='all' or r.get('subset')==subset)]
   rs=[r for r in rows if r['tolerance']==tol and (subset=='all' or r.get('subset')==subset)]
   if not old:continue
   bad=[r for r in rs if r['status'] in EXECUTION_FAILURES]
   complete=len(rs)==len(old) and not bad
   vals=[min(r['seconds'],spec['time_limit']) if r['solved'] and number(r['seconds']) is not None else spec['time_limit'] for r in rs]
   valid=complete and spec['kind'] not in ['curve','appendix'] and all(not r['solved'] or number(r['seconds']) is not None for r in rs)
   its=[number(r['iterations']) for r in rs]
   summaries.append(dict(solver=solver,subset=subset,tolerance=tol,expected_cases=len(old),measured_cases=len(rs),solved=sum(r['solved'] for r in rs),time_SGM10=sgm(vals) if valid else None,iteration_SGM10=sgm(its) if valid and all(v is not None for v in its) else None,complete_fresh=complete,guard_or_execution_failures=len(bad),note='Complete measured dataset' if complete else 'Incomplete dataset; no paper-value fallback; aggregate withheld'))
 return summaries

COMPARISON_FIELDS=['solver','case','group','subset','tolerance','differences','expected','note','paper_status','reproduced_status','paper_iterations','reproduced_iterations','paper_objective','reproduced_objective','scaled_objective_difference','paper_seconds','reproduced_seconds','paper_raw_file','reproduced_raw_file']
SUMMARY_FIELDS=['result_set','solver','subset','tolerance','expected_cases','measured_cases','pending_cases','solved','time_SGM10','iteration_SGM10','status','execution_failures','note']
EXECUTION_FAILURES={'HARD_TIME_LIMIT','MEMORY_LIMIT','INTERRUPTED','EXECUTION_ERROR','MISSING_RESULT','MODEL_CORRECTION_PENDING','INPUT_ERROR'}

def result_file(record):
 path=Path(record['raw_file'].split('#',1)[0])
 return path if path.is_absolute() else ROOT/path

def measured_curves(rows):
 """Read the curve associated with each selected measurement, never an old run."""
 curves={}
 for r in rows:
  if r['status'] in EXECUTION_FAILURES:continue
  source=result_file(r).parent/'curves.csv'
  if not source.is_file():continue
  with source.open(newline='') as f:
   samples=[s for s in csv.DictReader(f) if s['case']==r['case']]
  if samples:curves[r['case']]=samples
 return curves

def remove_figure(path):
 for p in [Path(path),Path(path).with_suffix('.csv')]:p.unlink(missing_ok=True)

def file_hash(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def invalidate_figures(folder,statistics):
 """A refresh cannot leave a plot advertising an ineligible solver series."""
 for source in (folder/'figures').glob('*.csv'):
  try:tol=float(source.stem.rsplit('_',1)[1])
  except ValueError:continue
  eligible={r['solver'] for r in statistics if r['subset']=='all' and r['tolerance']==tol and r['complete_fresh'] and r['time_SGM10'] is not None}
  with source.open(newline='') as f:records=list(csv.DictReader(f))
  plotted={r['solver'] for r in records}
  unchanged=all(r.get('summary_sha256') and (folder.parent.parent/r['source']).is_file() and file_hash(folder.parent.parent/r['source'])==r['summary_sha256'] for r in records)
  if not plotted or not plotted<=eligible or not unchanged:remove_figure(source.with_suffix('.pdf'))

def difference_row(experiment,solver,old,new):
 """Only observed differences; timing and synonymous solved labels are omitted.

 Objective screening uses the existing scaled 1e-6 validation threshold.
 An explanation is not treated as established merely because both runs solved.
 """
 if new is None:return None
 differences=[];notes=[];expectations=[]
 def add(field,expected,note):
  differences.append(field);expectations.append(expected);notes.append(note)
 corrected=new.get('model_revision')=='original_mps_objective_v1'
 both_solved=old['solved'] and new['solved']
 both_timeout=not old['solved'] and not new['solved'] and old['status']=='TIME_LIMIT' and new['status']=='TIME_LIMIT'
 if old['solved']!=new['solved']:
  add('outcome','not established','Solved outcome changed at this tolerance; review the stopping residual and native records.')
 elif not both_solved and old['status']!='REFERENCE_CURVE' and old['status'].upper()!=new['status'].upper():
  add('termination','not established','Termination reason changed; review the native records before treating this as reproduced.')
 a=number(old.get('objective'));b=number(new.get('objective'))
 delta=abs(a-b)/max(1,abs(a)) if a is not None and b is not None and old.get('objective_basis')==new.get('objective_basis') else None
 if delta is not None and delta>1e-6:
  if corrected and new['solved']:
   add('objective','yes','Corrected original-MPS objective replaces the historical scaled-model formulation; the two objectives are not the same target.')
  elif both_timeout:
   add('objective','yes','Both runs reached the time limit; different stopping iterates can have different unconverged objectives.')
  elif solver=='CuClarabel' and format(a,'.5g')==format(b,'.5g'):
   add('objective','consistent with rounding','Objectives agree to five significant digits, consistent with the limited precision of the preserved CuClarabel reference.')
  else:
   add('objective','not established','Scaled objective difference exceeds 1e-6 on the same objective basis; the cause has not been established.')
 ai=number(old.get('iterations'));bi=number(new.get('iterations'))
 if ai is not None and bi is not None and ai!=bi:
  if corrected and new['solved']:
   add('iterations','yes','The corrected model can change the iteration count.')
  elif both_timeout:
   add('iterations','yes','A wall-clock time limit permits different iteration counts on different runs.')
  else:
   detail='Objectives agree within the scaled 1e-6 threshold.' if both_solved and delta is not None and delta<=1e-6 else 'Objective agreement cannot be inferred from this iteration comparison.'
   add('iterations','not established','Iteration count differs. '+detail+' Numerical or environment variation is possible; the specific cause is unverified.')
 if 'Full-curve comparison: mismatch_review' in (new.get('note') or ''):
  add('curve','not established','Saved iteration grids or pointwise residual curves fail the curve agreement check.')
 if not differences:return None
 expected='not established' if 'not established' in expectations else 'consistent with rounding' if 'consistent with rounding' in expectations else 'yes'
 return dict(solver=solver,case=old['case'],group=old['group'],subset=old.get('subset','all'),tolerance=old['tolerance'],differences='; '.join(differences),expected=expected,note=' '.join(notes),paper_status=old['status'],reproduced_status=new['status'],paper_iterations=ai,reproduced_iterations=bi,paper_objective=a,reproduced_objective=b,scaled_objective_difference=delta,paper_seconds=old['seconds'],reproduced_seconds=new.get('seconds'),paper_raw_file=old.get('raw_file'),reproduced_raw_file=new.get('raw_file'))

def reported_statistics(experiment,statistics):
 """Use printed R2 aggregates in paper overviews; retain computed statistics."""
 from result_paths import experiment_path
 source=experiment_path(ROOT/'results',experiment,True)/'sources/reported_statistics.csv'
 reported={}
 if source.exists():
  with source.open(newline='') as f:
   reported={(r['solver'],r['subset'],float(r['tolerance']),int(r['expected_cases'])):r for r in csv.DictReader(f)}
 result=[]
 for r in statistics:
  r=dict(r);printed=reported.get((r['solver'],r['subset'],r['tolerance'],r['expected_cases']),{})
  for field in ['time_SGM10','iteration_SGM10','solved']:
   if printed.get(field,'')!='':r[field]=number(printed[field])
  result.append(r)
 return result

def summary_sections(paper,reproduced):
 """Two labeled result sets in one rectangular CSV, with explicit pending cells."""
 result=[]
 for label,rows in [('paper',paper),('reproduced',reproduced)]:
  for r in rows:
   pending=r['expected_cases']-r['measured_cases'];failures=r['guard_or_execution_failures']
   status='to be reproduced' if pending else 'rerun required' if failures else 'available'
   missing='to be reproduced' if pending else 'unavailable' if failures else 'not available'
   note=f"{pending} cases to be reproduced. Solved count covers only the {r['measured_cases']} measured cases." if pending else ''
   if failures:note=(note+' Execution failure; full aggregates withheld.').strip()
   result.append(dict(result_set=label,solver=r['solver'],subset=r['subset'],tolerance=r['tolerance'],expected_cases=r['expected_cases'],measured_cases=r['measured_cases'],pending_cases=pending,solved=r['solved'] if r['measured_cases'] else 'to be reproduced',time_SGM10=r['time_SGM10'] if r['time_SGM10'] is not None else missing,iteration_SGM10=r['iteration_SGM10'] if r['iteration_SGM10'] is not None else missing,status=status,execution_failures=failures,note=note))
 return result

def refresh(experiment,output):
 """Select latest runs without overwriting the distributed native evidence."""
 from result_paths import solver_path,experiment_path
 folder=experiment_path(output,experiment['name']);folder.mkdir(parents=True,exist_ok=True)
 with (folder/'.summary.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  comparisons=[];summaries=[];paper_summaries=[]
  for solver,spec in experiment['solvers'].items():
   sd=solver_path(output,experiment['name'],solver);sd.mkdir(parents=True,exist_ok=True)
   archive=[dict(r) for c in spec['cases'] for r in c['archived']]
   prior=read_csv(solver_path(ROOT/'results',experiment['name'],solver)/'raw/measurements.csv')
   fresh={(r['case_id'],r['tolerance']):r for r in prior}
   runs=sorted((sd/'raw/runs').glob('*/run.csv'))
   for f in runs:
    try:d=load_run(f.parent)
    except (OSError,ValueError,StopIteration):continue
    for r in d['rows']:fresh[r['case_id'],r['tolerance']]=r
   expected_models={c['key']:c.get('model_revision') for c in spec['cases']}
   allowed={(r['case_id'],r['tolerance']) for r in archive}
   fresh={k:r for k,r in fresh.items() if k in allowed and (not expected_models.get(k[0]) or r.get('model_revision')==expected_models[k[0]])}
   measured=[fresh[k] for k in sorted(fresh)]
   write_csv(sd/'summary.csv',measured,FIELDS)
   stats=aggregate(measured,spec,archive,solver);summaries+=stats
   write_csv(sd/'statistics.csv',stats)
   ps=solver_path(output,experiment['name'],solver,True)
   write_csv(ps/'summary.csv',archive,FIELDS)
   pstats=aggregate(archive,spec,archive,solver);paper_summaries+=pstats;write_csv(ps/'statistics.csv',pstats)
   for old in archive:
    new=fresh.get((old['case_id'],old['tolerance']))
    difference=difference_row(experiment['name'],solver,old,new)
    if difference:comparisons.append(difference)
  paper_summaries=reported_statistics(experiment['name'],paper_summaries)
  invalidate_figures(folder,summaries)
  if experiment['name']=='Restart-Sigma':
   source=folder/'QSCSD1.csv'
   records=[]
   if source.exists():
    with source.open(newline='') as f:records=list(csv.DictReader(f))
   current=folder/'solver_HPR-QP/summary.csv'
   if any(not r['complete_fresh'] for r in summaries) or not records or records[0].get('summary_sha256')!=file_hash(current):remove_figure(folder/'QSCSD1.pdf')
  write_csv(folder/'comparison.csv',comparisons,COMPARISON_FIELDS)
  write_csv(folder/'summary.csv',summary_sections(paper_summaries,summaries),SUMMARY_FIELDS)
  write_csv(experiment_path(output,experiment['name'],True)/'summary.csv',paper_summaries)
 return summaries

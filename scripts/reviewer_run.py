#!/usr/bin/env python3
"""Paper experiment + solver entry point, bounded workers, and source-labelled results."""
from pathlib import Path
import argparse,copy,csv,datetime,fcntl,hashlib,json,os,queue,shutil,signal,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiment_results import ROOT,atomic_json,row,refresh,hpr_rows,jump_rows,pdqp_rows,number
from process_watchdog import run_guarded
from data_paths import locate_input
CAT=json.loads((ROOT/'config/reviewer_experiments.json').read_text())
def norm(s):return s.lower().replace('_','-')
def resolve_exp(s):
 return next((e for e in CAT if norm(s) in [norm(e['name'])]+[norm(x) for x in e['aliases']]),None)
def resolve_solver(s):
 aliases={'hprqp':'HPR-QP','hpr-qp-cqp':'HPR-QP-explicit','hpr(p1)':'HPR-p1','hpr(p2)':'HPR-p2','clarabel':'CuClarabel'}
 return aliases.get(norm(s),next((x for e in CAT for x in e['solvers'] if norm(x)==norm(s)),None))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('experiment',nargs='?',default='list',help='Descriptive experiment name, ALL, list, refresh, or setup')
 p.add_argument('solver',nargs='?',default='HPR-QP',help='HPR-QP, HPR-p1, HPR-p2, HPR-QP-explicit, PDQP, SCS, CuClarabel, Gurobi, or ALL')
 p.add_argument('--cases',nargs='+',help='Exact names or group__name keys from the mapping; duplicate synthetic names select both groups')
 p.add_argument('--groups',nargs='+',help='Optional source group ids, e.g. synthetic_paper or revision_denser')
 p.add_argument('--tolerances',nargs='+',type=float,choices=[1e-4,1e-6,1e-8])
 p.add_argument('--data-root',type=Path,default=Path(os.environ.get('HPR_DATA_ROOT',str(ROOT/'data/inputs'))))
 p.add_argument('--output',type=Path,default=ROOT/'experiments')
 p.add_argument('--julia',default=os.environ.get('JULIA_BIN','julia'))
 p.add_argument('--gpu',default='0');p.add_argument('--gpus',help='Comma-separated physical GPUs; one solver worker at a time on each GPU')
 p.add_argument('--jobs',type=int,default=1,help='Concurrent workers; default 1 for timing isolation')
 p.add_argument('--cpu-threads',type=int,default=32,help='Gurobi threads; 32 follows the archived default; CPU Gurobi jobs never overlap within this package')
 p.add_argument('--time-limit',type=float,help='Override solver time limit; changes the paper protocol and is recorded')
 p.add_argument('--hard-time-limit',type=float,help='Wall seconds per Julia worker, including loading/warmup; TERM then KILL')
 p.add_argument('--run-time-limit',type=float,help='Optional whole-invocation deadline in seconds')
 p.add_argument('--memory-gb',type=float,default=0,help='Optional host RSS guard per worker (0 disables); does not cap GPU memory')
 p.add_argument('--plan',action='store_true');p.add_argument('--rerun',action='store_true',help='Create new attempts even when matching local runs completed')
 a=p.parse_args();a.output=a.output.resolve()
 if a.jobs<1 or a.cpu_threads<1 or a.memory_gb<0:p.error('Jobs/CPU threads must be positive and memory must be nonnegative')
 for v in [a.time_limit,a.hard_time_limit,a.run_time_limit]:
  if v is not None and v<=0:p.error('Time limits must be positive')
 if a.experiment.lower() in ['list','help']:
  print('Command name         Paper experiment / solver choices')
  for e in CAT:print(f"{e['name']:20} {e['title']} | {', '.join(e['solvers'])}")
  print('\nRecommended dataset: ./run.sh QAPLIB HPR-QP')
  print('Preview:     ./run.sh QAPLIB HPR-QP --plan')
  print('Optional case check: ./run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps')
  print('Dataset costs and paper mapping: docs/EXPERIMENTS.md')
  print('Results:     experiments/<name>/summary.html (source-labelled)')
  return 0
 if a.experiment.lower()=='refresh':
  for e in CAT:refresh(e,a.output)
  print('Updated experiment summaries in',a.output);return 0
 solver='ALL' if a.solver.upper()=='ALL' else resolve_solver(a.solver)
 if not solver:p.error('Unknown solver '+a.solver)
 if a.experiment.lower()=='setup':
  names=['HPR-QP','PDQP','SCS','CuClarabel','Gurobi'] if solver=='ALL' else [solver]
  for s in names:
   project=ROOT/'environment' if s.startswith('HPR') else ROOT/'environment'/s
   env=dict(os.environ);env['JULIA_LOAD_PATH']='@:@stdlib';env.pop('LD_LIBRARY_PATH',None)
   env['JULIA_DEPOT_PATH']=os.environ.get('HPR_DEPOT_PATH' if s.startswith('HPR') else 'COMPETITOR_DEPOT_PATH',os.environ.get('JULIA_DEPOT_PATH',str(ROOT/'.julia-depot')))
   logfile=a.output/'setup_logs'/(s+'.log');print('Installing pinned environment:',s,flush=True)
   result=run_guarded([a.julia,'--startup-file=no','--project='+str(project),'-e','using Pkg; @assert VERSION==v"1.10.4" "Use Julia 1.10.4"; Pkg.instantiate()'],env,logfile,a.hard_time_limit or 3600)
   if result['state']!='completed':print('Setup failed:',result['state'],logfile);return 2
  return 0
 chosen=CAT if a.experiment.upper()=='ALL' else [resolve_exp(a.experiment)]
 if any(x is None for x in chosen):p.error('Unknown experiment; use ./run.sh list')
 gpus=(a.gpus or a.gpu).split(',')
 if len(set(gpus))!=len(gpus) or any(not x.strip() for x in gpus):p.error('GPU identifiers must be nonempty and unique')
 tasks={};matched_filters=set();active=[]
 for e in chosen:
  names=list(e['solvers']) if solver=='ALL' else [solver] if solver in e['solvers'] else []
  if not names and len(chosen)==1:p.error('Solver is not reported for '+e['name']+'; supported: '+', '.join(e['solvers']))
  for s in names:
   spec=e['solvers'][s]
   for c0 in spec['cases']:
    c=copy.deepcopy(c0)
    if a.groups and c['group'] not in a.groups:continue
    if a.cases:
     hits=set(a.cases)&{c['name'],c['key'],Path(c['path']).name}
     if not hits:continue
     matched_filters.update(hits)
    ts=c['tolerances']
    if a.tolerances:ts=[t for t in ts if t in a.tolerances]
    if not ts:continue
    c['tolerances']=ts
    soft=a.time_limit or spec['time_limit'];hard=a.hard_time_limit or soft+(14400 if s.startswith('HPR') else 900)
    for tol in ts if spec['kind']=='jump' else [None]:
     key=json.dumps([s,spec['kind'],c['group'],c['name'],tol,soft,hard,a.cpu_threads,a.memory_gb])
     task=tasks.setdefault(key,dict(solver=s,kind=spec['kind'],soft=soft,hard=hard,tol=tol,targets=[],token=hashlib.sha256(key.encode()).hexdigest()[:20]))
     task['targets'].append((e,spec,c))
    if e not in active:active.append(e)
 if a.cases and set(a.cases)!=matched_filters:p.error('Unmatched cases: '+str(set(a.cases)-matched_filters))
 if not tasks:p.error('No cases matched this experiment/solver')
 tasks=list(tasks.values());estimate=0;missing=set();known_failures=0
 def locate(c):
  if c['group']=='appendix_concrete':return (ROOT/c['path']).resolve() if (ROOT/c['path']).is_file() else None
  return locate_input(c['path'],a.data_root)
 for task in tasks:
  e,sp,c=task['targets'][0];path=locate(c);task['input']=str(path) if path else None
  if not path:missing.add(c['path'])
  refs=[r for r in c['archived'] if r['tolerance']==(task['tol'] or 1e-8)]
  if refs:estimate+=number(refs[0]['seconds']) or 0;known_failures+=sp['kind'] not in ['curve','appendix'] and not refs[0]['solved']
 print(f"Selected {len(tasks)} Julia workers across {len(active)} experiments; {len(missing)} missing inputs.")
 print(f"Archived solver-time total: {estimate/60:.1f} minutes (excludes loading, warmup, compilation; shared-device timings vary).")
 if a.experiment.upper()=='ALL' or known_failures:
  print(f"WARNING: this selection includes {known_failures} archived unsuccessful worker outcomes. Full ALL may take days to weeks; huge QAP alone historically took 35 hours. Competing solvers can exceed available memory or spend a long time in setup. External wall deadlines apply. Ctrl-C stops worker groups and retains partial results.")
 if a.jobs>1:print('PARALLEL RUN: each GPU is exclusive within this package; only one Gurobi job at a time. CPU preprocessing and memory bandwidth still interact. Use --jobs 1 on otherwise idle hardware for timing comparisons.')
 for task in tasks:
  e,sp,c=task['targets'][0]
  if a.plan:print(f"{e['name']}/{task['solver']}: {c['key']} tol={task['tol'] or 'checkpoints to 1e-8'} hard={task['hard']:g}s {'READY' if task['input'] else 'MISSING'}")
 for e in active:
  if e.get('validation_note'):print('MODEL CORRECTION: '+e['validation_note'])
 if a.plan:return 2 if missing else 0
 exe=shutil.which(a.julia) or (a.julia if Path(a.julia).is_file() else None)
 if not exe:p.error('Julia 1.10.4 not found; set JULIA_BIN or use --julia')
 for e in active:refresh(e,a.output)
 stop=threading.Event();resource_queue=queue.Queue()
 for g in gpus:resource_queue.put(g)
 old_handlers={sig:signal.getsignal(sig) for sig in [signal.SIGINT,signal.SIGTERM]}
 def interrupted(sig,frame):print('\nStopping worker groups and saving partial results…',flush=True);stop.set()
 for sig in old_handlers:signal.signal(sig,interrupted)
 run_budget=a.run_time_limit;timer=threading.Timer(run_budget,stop.set) if run_budget else None
 if timer:timer.start()
 locks=ROOT/'.reviewer_locks';locks.mkdir(exist_ok=True);hash_cache={};hash_lock=threading.Lock();print_lock=threading.Lock()
 def message(s):
  with print_lock:print(s,flush=True)
 def completed_for(path,case):
  try:d=json.loads(path.read_text())
  except (OSError,ValueError):return False
  return d.get('state')=='completed' and set(case['tolerances']) <= {r['tolerance'] for r in d.get('rows',[]) if not case.get('model_revision') or r.get('model_revision')==case['model_revision']}
 def execute(task):
  e,sp,c0=task['targets'][0];c=copy.deepcopy(c0);c['tolerances']=sorted({t for _,_,cc in task['targets'] for t in cc['tolerances']});s=task['solver'];folders=[a.output/ee['name']/s for ee,_,_ in task['targets']]
  if not a.rerun and all(any(completed_for(f,next(cc for ee,_,cc in task['targets'] if a.output/ee['name']/s==d)) for f in (d/'runs').glob('*_'+task['token']+'/record.json')) for d in folders):
   message('Resume: already completed '+e['name']+'/'+s+'/'+c['name']);return True
  if stop.is_set():return False
  gpu=None;lf=None
  try:
   if s!='Gurobi':
    while not stop.is_set():
     try:gpu=resource_queue.get(timeout=.2);break
     except queue.Empty:pass
    if gpu is None:return False
   lf=(locks/('cpu-Gurobi.lock' if s=='Gurobi' else 'gpu-'+re_safe(gpu)+'.lock')).open('w')
   while not stop.is_set():
    try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:time.sleep(.2)
   if stop.is_set():return False
   stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+task['token'];run=folders[0]/'runs'/stamp;run.mkdir(parents=True,exist_ok=True)
   record=dict(state='running',solver=s,experiment=e['name'],case_id=c['key'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=[],input_path=task['input'],gpu=gpu,cpu_threads=a.cpu_threads,solver_time_limit=task['soft'],hard_time_limit=task['hard'],host_memory_limit_gb=a.memory_gb)
   atomic_json(run/'record.json',record)
   def finish():
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(run/'record.json',record)
    for ee,ss,cc in task['targets']:
     dst=a.output/ee['name']/s/'runs'/stamp
     atomic_json(dst/'record.json',dict(record,experiment=ee['name'],rows=[r for r in record['rows'] if r['tolerance'] in cc['tolerances']]))
     refresh(ee,a.output)
   if not task['input']:
    record.update(state='INPUT_ERROR',error='Missing '+c['path']+'; see docs/EXPERIMENTS.md and data inventories');finish();message(record['error']);return False
   path=Path(task['input']);cachekey=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
   with hash_lock:sha=hash_cache.get(cachekey)
   if sha is None:
    h=hashlib.sha256()
    with path.open('rb') as f:
     for block in iter(lambda:f.read(8*1024*1024),b''):
      if stop.is_set():record['state']='INTERRUPTED';finish();return False
      h.update(block)
    sha=h.hexdigest()
    with hash_lock:hash_cache[cachekey]=sha
   record['input_sha256']=sha
   if c.get('sha256') and sha!=c['sha256']:
    record.update(state='INPUT_ERROR',error='Input SHA256 mismatch: '+str(path));finish();message(record['error']);return False
   env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.update(JULIA_LOAD_PATH='@:@stdlib',JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=gpu or '',GKSwstype='100')
   env['JULIA_DEPOT_PATH']=os.environ.get('HPR_DEPOT_PATH' if s.startswith('HPR') else 'COMPETITOR_DEPOT_PATH',os.environ.get('JULIA_DEPOT_PATH',str(ROOT/'.julia-depot')))
   project=ROOT/'environment' if s.startswith('HPR') else ROOT/'environment'/s
   command=[exe,'--startup-file=no','--project='+str(project)]
   if task['kind'] in ['hpr','curve']:
    g=copy.deepcopy(next(g for g in json.loads((ROOT/'config/experiments.json').read_text()) if g['id']==c['group']));case=next(x for x in g['cases'] if x['name']==c['name']);case['path']=str(path);g['params']['time_limit']=task['soft'];g['cases']=[case]
    job=dict(id=task['token'],group=g,cases=[case],phase='reviewer_entrypoint',output_root=str(run));atomic_json(run/'job.json',job)
    command += [str(ROOT/'scripts/run_worker.jl'),str(run/'job.json')]
   elif task['kind']=='jump':
    settings=dict(c.get('settings',{}))
    if s=='Gurobi':settings['Threads']=a.cpu_threads
    jc=dict(family=c['family'],name=Path(c['path']).name,path=str(path),tolerances=[task['tol']],settings=settings)
    warm=dict(family='MM',name='CVXQP3_S.mps',path=str(ROOT/'data/inputs/MM_mps/CVXQP3_S.mps'))
    atomic_json(run/'job.json',dict(solver=s,cases=[jc],tolerances=[task['tol']],output=str(run/'native'),warmup_case=warm,solver_time_limit=task['soft']))
    command += [str(ROOT/'scripts/jump_worker.jl'),str(run/'job.json')]
   elif task['kind']=='pdqp':
    (run/'native').mkdir();env['HPR_COMPETITOR_META']=str(run/'native/environment.json')
    command += ['--load',str(ROOT/'scripts/pdqp_environment.jl'),str(ROOT/'src/competitors/PDQP/scripts/solve_0.jl'),'--instance_path',str(path),'--output_directory',str(run/'native'),'--tolerance','1e-8','--time_sec_limit',str(task['soft']),'--use_gpu','1']
   else:command += [str(ROOT/'scripts/run_concrete_example.jl'),str(run/'native')]
   message('Running '+e['name']+'/'+s+'/'+c['name']+' on '+('CPU' if gpu is None else 'GPU '+gpu))
   result=run_guarded(command,env,run/'launch.log',task['hard'],stop,a.memory_gb);record.update(result)
   rr=[]
   if result['state']=='completed':
    try:
     if task['kind'] in ['hpr','curve']:
      nf=run/'results'/c['group']/(c['name']+'.json');native=json.loads(nf.read_text());rr=hpr_rows(c,s,native,str(nf));
      for result_row in rr:result_row['model_revision']=native.get('model_revision','')
      if native.get('execution')!='completed':record['state']='EXECUTION_ERROR'
      if task['kind']=='curve':
       from compare_results import compare
       verdict=compare(native)['verdict']
       for r in rr:r['note']='Full-curve comparison: '+verdict
     elif task['kind']=='jump':
      nf=next((run/'native').glob('*.json'));native=json.loads(nf.read_text());rr=jump_rows(c,s,native,str(nf))
      if native.get('execution')!='completed':record['state']='EXECUTION_ERROR'
     elif task['kind']=='pdqp':
      nf=next((run/'native').glob('*_summary.json'));rr=pdqp_rows(c,json.loads(nf.read_text()),str(nf))
     else:
      nf=run/'native/raw_results.csv';native=next(x for x in csv.DictReader(nf.open()) if x['method']==c['method']);rr=[row(c,s,1e-8,status=native['status'],solved=native['status']=='OPTIMAL',iterations=number(native['iter']),seconds=number(native['time']),objective=number(native['primal_obj']),residual=number(native['residual']),raw_file=str(nf),note='Reconstructed appendix driver; historical dual count remains unverified')]
    except (OSError,ValueError,KeyError,StopIteration) as ex:record.update(state='MISSING_RESULT',error=str(ex))
   if record['state']!='completed':
    ts=[task['tol']] if task['tol'] else c['tolerances'];rr=[row(c,s,t,status=record['state'],solved=False,wall_seconds=result['wall_seconds'],raw_file=str(run/'launch.log'),note=record.get('error','External guard or interruption; no archived substitution for this attempted row')) for t in ts]
   for r in rr:
    if c.get('model_revision'):r['model_revision']=c['model_revision']
    r['wall_seconds']=result['wall_seconds']
    try:r['raw_file']=str(Path(r['raw_file']).relative_to(ROOT))
    except ValueError:pass
   record['rows']=rr;finish();message(record['state']+' '+e['name']+'/'+s+'/'+c['name']);return record['state']=='completed'
  except Exception as ex:
   message('Launcher error '+e['name']+'/'+s+'/'+c['name']+': '+str(ex))
   if 'record' in locals():record.update(state='EXECUTION_ERROR',error=str(ex));finish()
   return False
  finally:
   if lf:lf.close()
   if gpu is not None:resource_queue.put(gpu)
 try:
  with ThreadPoolExecutor(max_workers=a.jobs) as pool:
   futures=[pool.submit(execute,t) for t in tasks];ok=[f.result() for f in as_completed(futures)]
 finally:
  if timer:timer.cancel()
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
  for e in active:refresh(e,a.output)
 print('Reports:',a.output,'— open <experiment>/summary.html; archived/fresh sources are explicit.')
 return 130 if stop.is_set() else 0 if all(ok) else 2

def re_safe(s):return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)
if __name__=='__main__':raise SystemExit(main())

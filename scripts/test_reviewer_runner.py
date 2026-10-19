#!/usr/bin/env python3
"""Local orchestration tests; these do not substitute for real Julia/GPU validation."""
import os,json,sys,tempfile,subprocess,unittest,threading,time
from pathlib import Path
from process_watchdog import run_guarded
from experiment_results import ROOT as ORIGINAL_ROOT,read_csv,load_run,save_run
import shutil,hashlib
from catalog import load_catalog

# The orchestration tests use clearly artificial inputs and simulated workers.
# No third-party benchmark bytes or GPU are required by this test suite.
_FIXTURE=tempfile.TemporaryDirectory(prefix='hpr test package ')
ROOT=Path(_FIXTURE.name)/'package'
shutil.copytree(ORIGINAL_ROOT,ROOT,ignore=shutil.ignore_patterns('.git','__pycache__','.julia-depot','.reviewer_locks','runs','inputs'))
_catalog=ROOT/'scripts/config/reviewer_experiments.json'
_cases=json.loads(_catalog.read_text())
for name in ['CVXQP3_S.mps','QSCSD6.mps']:
 blob=('synthetic orchestration fixture, not a benchmark: '+name).encode()
 file=ROOT/'data/MM/inputs'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(blob)
 for e in _cases:
  for sp in e['solvers'].values():
   for c in sp['cases']:
    if c['path']=='MM/inputs/'+name:c.update(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
_catalog.write_text(json.dumps(_cases))
def tearDownModule():_FIXTURE.cleanup()

class RunnerChecks(unittest.TestCase):
 def test_watchdog_kills_process_group(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); code='import subprocess,time,pathlib,sys; c=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); pathlib.Path(sys.argv[1]).write_text(str(c.pid)); time.sleep(60)'
   r=run_guarded([sys.executable,'-c',code,str(p/'pid')],dict(os.environ),p/'log',.6)
   self.assertEqual(r['state'],'HARD_TIME_LIMIT');self.assertLess(r['wall_seconds'],8)
   stat=Path('/proc')/(p/'pid').read_text()/'stat'
   self.assertTrue(not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z')
 def test_interruption(self):
  with tempfile.TemporaryDirectory() as d:
   stop=threading.Event();timer=threading.Timer(.3,stop.set);timer.start()
   r=run_guarded([sys.executable,'-c','import time;time.sleep(60)'],dict(os.environ),Path(d)/'log',60,stop);timer.join()
   self.assertEqual(r['state'],'INTERRUPTED')
 def test_shared_hpr_case_retains_all_checkpoints_and_resume(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fake=p/'julia';fake.write_text('''#!/usr/bin/env python3
import json,pathlib,sys
import csv
j=json.load(open(sys.argv[-1]));c=j['cases'][0];f=pathlib.Path(j['output_root'])/'native/HPRQP_zxwyw_result.csv';f.parent.mkdir(parents=True,exist_ok=True)
r=dict(name=c['name'],iter=300,iter_4=100,iter_6=200,alg_time=3,time_4=1,time_6=2,primal_obj=0,res=1e-9,status='OPTIMAL')
with f.open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(r));w.writeheader();w.writerow(r)
''');fake.chmod(0o755)
   cmd=[sys.executable,str(ROOT/'scripts/reviewer_run.py'),'ALL','HPR-QP','--cases','QSCSD6.mps','--julia',str(fake),'--output',str(p/'out')]
   one=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(one.returncode,0,one.stdout+one.stderr)
   mm=load_run(next((p/'out/reproduced_results/exp_MM/solver_HPR-QP/raw/runs').glob('*/run.csv')).parent);pd=load_run(next((p/'out/reproduced_results/exp_Primal-Dual/solver_HPR-QP/raw/runs').glob('*/run.csv')).parent)
   self.assertEqual(len(mm['rows']),3);self.assertEqual(len(pd['rows']),1)
   raw=p/'out/reproduced_results/exp_MM/solver_HPR-QP/raw'
   self.assertFalse(list(raw.rglob('*.json')))
   import csv
   with Path(mm['rows'][0]['raw_file'].split('#')[0]).open() as f:dataset=list(csv.DictReader(f))
   self.assertEqual(float(next(r for r in dataset if r['name']=='QSCSD6.mps')['iter']),300)
   two=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(two.returncode,0,two.stdout+two.stderr);self.assertIn('Resume:',two.stdout)
   # Running into the distributed result tree must not invalidate its evidence hashes.
   build=subprocess.run([sys.executable,str(ROOT/'scripts/build_release.py'),'--output',str(p/'fixture.zip')],capture_output=True,text=True)
   self.assertEqual(build.returncode,0,build.stdout+build.stderr)
   manifest=json.loads((ROOT/'scripts/config/SHA256SUMS.json').read_text())
   local=cmd[:-1]+[str(ROOT/'results')]
   run=subprocess.run(local,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   for entry in manifest:
    file=ROOT/entry['path']
    self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),entry['sha256'],entry['path'])
 def test_guard_failure_overrides_archive_and_withholds_aggregate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fake=p/'julia';fake.write_text('#!/usr/bin/env python3\nimport time; time.sleep(60)\n');fake.chmod(0o755)
   figures=p/'out/reproduced_results/exp_MM/figures';figures.mkdir(parents=True)
   for tol in ['1e-04','1e-06','1e-08']:
    (figures/f'profile_{tol}.pdf').write_bytes(b'old plot')
    (figures/f'profile_{tol}.csv').write_text('solver,source,cases\nHPR-QP,old.csv,137\n')
   r=subprocess.run([sys.executable,str(ROOT/'scripts/reviewer_run.py'),'MM','HPR-QP','--cases','CVXQP3_S.mps','--julia',str(fake),'--hard-time-limit','.5','--output',str(p/'out')],capture_output=True,text=True)
   self.assertEqual(r.returncode,2,r.stdout+r.stderr)
   rows=read_csv(p/'out/reproduced_results/exp_MM/solver_HPR-QP/summary.csv');bad=[x for x in rows if x['solver']=='HPR-QP' and x['case']=='CVXQP3_S.mps'];self.assertEqual(len(bad),3);self.assertTrue(all(x['status']=='HARD_TIME_LIMIT' and x['source']=='fresh' for x in bad))
   summary=read_csv(p/'out/reproduced_results/exp_MM/summary.csv')
   self.assertTrue(all(x['time_SGM10']=='unavailable' and x['status']=='rerun required' for x in summary if x['solver']=='HPR-QP' and x['result_set']=='reproduced'))
   self.assertTrue(all(float(x['time_SGM10'])>0 for x in summary if x['solver']=='HPR-QP' and x['result_set']=='paper'))
   self.assertEqual(list(figures.iterdir()),[])

class FigureChecks(unittest.TestCase):
 def test_profile_is_removed_after_measurement_changes(self):
  from experiment_results import invalidate_figures,file_hash,write_csv
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'reproduced_results/exp_MM';figures=folder/'figures';figures.mkdir(parents=True)
   summary=folder/'solver_HPR-QP/summary.csv';summary.parent.mkdir();summary.write_text('seconds\n1\n')
   pdf=figures/'profile_1e-08.pdf';pdf.write_bytes(b'plot of the first measurement')
   write_csv(pdf.with_suffix('.csv'),[dict(solver='HPR-QP',source=str(summary.relative_to(root)),summary_sha256=file_hash(summary))])
   statistics=[dict(solver='HPR-QP',subset='all',tolerance=1e-8,complete_fresh=True,time_SGM10=1)]
   invalidate_figures(folder,statistics);self.assertTrue(pdf.exists())
   summary.write_text('seconds\n2\n')
   invalidate_figures(folder,statistics);self.assertFalse(pdf.exists());self.assertFalse(pdf.with_suffix('.csv').exists())
 def test_curve_selection_follows_latest_measurement(self):
  from experiment_results import measured_curves
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name,value in [('baseline',1),('latest',2)]:
    native=p/name;native.mkdir()
    (native/'curves.csv').write_text(f'case,iteration,optimality\nno_restart,100,{value}\n')
   row=dict(case='no_restart',raw_file=str(p/'latest/result.csv'),status='MAX_ITER')
   self.assertEqual(measured_curves([row])['no_restart'][0]['optimality'],'2')
   self.assertEqual(measured_curves([dict(row,status='EXECUTION_ERROR')]),{})
   (p/'latest/curves.csv').unlink()
   self.assertEqual(measured_curves([row]),{})
class SolverRuntimeChecks(unittest.TestCase):
 def test_jump_resume_checks_each_workers_tolerance(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fake=p/'julia';fake.write_text('''#!/usr/bin/env python3
import json,pathlib,sys
j=json.load(open(sys.argv[-1]));out=pathlib.Path(j['output']);out.mkdir(parents=True,exist_ok=True)
(out/'sample.json').write_text(json.dumps(dict(execution='completed',tolerance=j['tolerances'][0],result=dict(status='OPTIMAL',iterations=7,solve_time=1.0,native_primal_objective=0.0))))
''');fake.chmod(0o755)
   cmd=[sys.executable,str(ROOT/'scripts/reviewer_run.py'),'MM','CuClarabel','--cases','CVXQP3_S.mps','--cuclarabel-julia',str(fake),'--output',str(p/'out')]
   one=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(one.returncode,0,one.stdout+one.stderr)
   runs=p/'out/reproduced_results/exp_MM/solver_CuClarabel/raw/runs'
   before=set(runs.glob('*/run.csv'));self.assertEqual(len(before),3)
   two=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(two.returncode,0,two.stdout+two.stderr)
   self.assertEqual(set(runs.glob('*/run.csv')),before,two.stdout+two.stderr)
   self.assertEqual(two.stdout.count('Resume:'),3)
 def test_separate_cuclarabel_runtime_for_setup_and_worker(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);calls=p/'calls.txt'
   template = r'''#!/usr/bin/env python3
import json,pathlib,sys,os
expected=EXPECTED
project=next(a.split("=",1)[1] for a in sys.argv if a.startswith("--project="))
solver="CuClarabel" if project.endswith("/CuClarabel") else "other"
assert (solver=="CuClarabel")== (expected=="1.11.5")
with open(os.environ["TEST_RUNTIME_CALLS"],"a") as f:f.write(expected+" "+solver+"\n")
if "-e" in sys.argv:
 assert 'VERSION==v"'+expected+'"' in sys.argv[-1]
else:
 j=json.load(open(sys.argv[-1]));c=j["cases"][0];out=pathlib.Path(j["output"]);out.mkdir(parents=True,exist_ok=True)
 (out/"sample.json").write_text(json.dumps(dict(execution="completed",tolerance=j["tolerances"][0],result=dict(status="OPTIMAL",iterations=7,solve_time=1.0,objective=0.0,native_primal_objective=0.0,res_primal=1e-9))))
'''
   for version in ['1.10.4','1.11.5']:
    f=p/('julia-'+version);f.write_text(template.replace('EXPECTED',repr(version)));f.chmod(0o755)
   env=dict(os.environ,TEST_RUNTIME_CALLS=str(calls),CUCLARABEL_JULIA_BIN=str(p/'julia-1.11.5'))
   common=['--julia',str(p/'julia-1.10.4'),'--output',str(p/'out')]
   r=subprocess.run([sys.executable,str(ROOT/'scripts/reviewer_run.py'),'setup','ALL',*common],env=env,capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   r=subprocess.run([sys.executable,str(ROOT/'scripts/reviewer_run.py'),'MM','CuClarabel','--cases','CVXQP3_S.mps','--tolerances','1e-8',*common],env=env,capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   self.assertEqual(calls.read_text().splitlines().count('1.11.5 CuClarabel'),2)
   self.assertEqual(calls.read_text().splitlines().count('1.10.4 other'),4)

class ModelCorrectionChecks(unittest.TestCase):
 def test_missing_rerun_never_uses_paper_values(self):
  import copy,csv
  from experiment_results import refresh
  e=copy.deepcopy(next(e for e in load_catalog() if e['name']=='MM'))
  c=copy.deepcopy(e['solvers']['HPR-QP']['cases'][0]);c['key']='unrun_test_case'
  for r in c['archived']:r['case_id']=c['key']
  e['solvers']={'HPR-QP':dict(e['solvers']['HPR-QP'],cases=[c])}
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);stats=refresh(e,out)
   self.assertEqual(read_csv(out/'reproduced_results/exp_MM/solver_HPR-QP/summary.csv'),[])
   self.assertEqual(len(read_csv(out/'paper_results/exp_MM/solver_HPR-QP/summary.csv')),3)
   with (out/'reproduced_results/exp_MM/comparison.csv').open() as f:comparison=list(csv.DictReader(f))
   self.assertEqual(comparison,[])
   with (out/'reproduced_results/exp_MM/summary.csv').open() as f:overview=list(csv.DictReader(f))
   self.assertEqual([r['result_set'] for r in overview],['paper']*3+['reproduced']*3)
   self.assertTrue(all(r['status']=='to be reproduced' and r['time_SGM10']=='to be reproduced' and r['solved']=='to be reproduced' and r['pending_cases']=='1' for r in overview if r['result_set']=='reproduced'))
   self.assertTrue(all(r['measured_cases']==0 and r['time_SGM10'] is None for r in stats))
 def test_old_model_cannot_fill_corrected_result(self):
  import copy
  from experiment_results import refresh
  cat=load_catalog()
  e=copy.deepcopy(next(e for e in cat if e['name']=='Synthetic-CQP'))
  c=next(c for c in e['solvers']['HPR-QP']['cases'] if c['key']=='synthetic_denser__huber_5000.mps')
  c['key']='model_guard_test_case'
  for archived in c['archived']:archived['case_id']=c['key']
  e['solvers']={'HPR-QP':dict(e['solvers']['HPR-QP'],cases=[c])}
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);run=out/'reproduced_results'/('exp_'+e['name'])/'solver_HPR-QP/raw/runs/old';run.mkdir(parents=True)
   rows=[dict(r,source='fresh') for r in c['archived']]
   save_run(run,dict(state='completed',rows=rows))
   summary=refresh(e,out)
   self.assertTrue(all(r['time_SGM10'] is None and not r['complete_fresh'] for r in summary))
   selected=read_csv(out/'reproduced_results'/('exp_'+e['name'])/'solver_HPR-QP/summary.csv')
   self.assertEqual(selected,[])
   for r in rows:r['model_revision']='original_mps_objective_v1'
   save_run(run,dict(state='completed',rows=rows))
   summary=refresh(e,out)
   self.assertTrue(all(r['complete_fresh'] and r['time_SGM10'] is not None for r in summary))

class ComparisonChecks(unittest.TestCase):
 def setUp(self):
  self.paper=dict(case='sample',group='sample',subset='all',tolerance=1e-8,status='PAPER_SOLVED',solved=True,iterations=100,seconds=2,objective=10,objective_basis='full objective',raw_file='paper.csv')
 def test_only_observed_nontiming_differences(self):
  from experiment_results import difference_row
  new=dict(self.paper,status='OPTIMAL',seconds=20,objective=10.00000001)
  self.assertIsNone(difference_row('MM','HPR-QP',self.paper,new))
  self.assertIsNone(difference_row('MM','HPR-QP',self.paper,None))
  r=difference_row('MM','HPR-QP',self.paper,dict(new,objective=11))
  self.assertEqual(r['differences'],'objective');self.assertEqual(r['expected'],'not established')
  r=difference_row('MM','HPR-QP',self.paper,dict(new,iterations=200))
  self.assertEqual(r['differences'],'iterations');self.assertEqual(r['expected'],'not established')
  self.assertIn('Objectives agree',r['note'])
  curve=dict(self.paper,status='REFERENCE_CURVE',solved=False,objective=None)
  self.assertIsNone(difference_row('Restart-Sigma','HPR-QP',curve,dict(curve,status='MAX_ITER',note='Full-curve comparison: curve_reproduced')))
  mismatch=difference_row('Restart-Sigma','HPR-QP',curve,dict(curve,status='MAX_ITER',note='Full-curve comparison: mismatch_review'))
  self.assertEqual(mismatch['differences'],'curve')
 def test_explained_differences_and_changed_outcome(self):
  from experiment_results import difference_row
  old=dict(self.paper,status='TIME_LIMIT',solved=False)
  new=dict(old,iterations=200,objective=20)
  self.assertEqual(difference_row('MM','HPR-QP',old,new)['expected'],'yes')
  changed=difference_row('MM','HPR-QP',old,dict(new,solved=True,status='OPTIMAL'))
  self.assertIn('outcome',changed['differences']);self.assertEqual(changed['expected'],'not established')
  corrected=dict(self.paper,iterations=200,objective=20,model_revision='original_mps_objective_v1')
  self.assertEqual(difference_row('Synthetic-CQP','HPR-QP',self.paper,corrected)['expected'],'yes')
  old=dict(self.paper,objective=1669)
  rounded=difference_row('Lasso-UCI','CuClarabel',old,dict(old,objective=1668.988))
  self.assertEqual(rounded['expected'],'consistent with rounding')
 def test_partial_summary_keeps_paper_and_pending_separate(self):
  from experiment_results import aggregate,summary_sections
  old=[self.paper,dict(self.paper,case='second')]
  spec=dict(time_limit=3600,kind='hpr')
  paper=aggregate(old,spec,old,'HPR-QP');partial=aggregate(old[:1],spec,old,'HPR-QP')
  rows=summary_sections(paper,partial)
  self.assertEqual(rows[0]['result_set'],'paper');self.assertEqual(rows[0]['time_SGM10'],paper[0]['time_SGM10'])
  self.assertEqual(rows[1]['pending_cases'],1);self.assertEqual(rows[1]['measured_cases'],1)
  self.assertEqual(rows[1]['time_SGM10'],'to be reproduced');self.assertEqual(rows[1]['solved'],1)

class DataLayoutChecks(unittest.TestCase):
 def test_launcher_preserves_caller_relative_paths(self):
  import shutil
  with tempfile.TemporaryDirectory(prefix='hpr caller with spaces ') as d:
   root=Path(d);data=root/'external data/MM_mps';data.mkdir(parents=True)
   shutil.copy2(ROOT/'data/MM/inputs/CVXQP3_S.mps',data/'CVXQP3_S.mps')
   fake=root/'fake-julia';fake.write_text('''#!/usr/bin/env python3
import json,pathlib,sys
import csv
j=json.load(open(sys.argv[-1]));c=j['cases'][0];f=pathlib.Path(j['output_root'])/'native/HPRQP_zxwyw_result.csv';f.parent.mkdir(parents=True,exist_ok=True)
r=dict(name=c['name'],iter=300,iter_4=100,iter_6=200,alg_time=3,time_4=1,time_6=2,primal_obj=0,res=1e-9,status='OPTIMAL')
with f.open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(r));w.writeheader();w.writerow(r)
''');fake.chmod(0o755)
   env=dict(os.environ,JULIA_BIN='./fake-julia')
   run=subprocess.run([str(ROOT/'scripts/run.sh'),'MM','HPR-QP','--cases','CVXQP3_S.mps','--data-root','external data','--output','fresh results'],cwd=root,env=env,capture_output=True,text=True)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   record=load_run(next((root/'fresh results/reproduced_results/exp_MM/solver_HPR-QP/raw/runs').glob('*/run.csv')).parent)
   self.assertEqual(record['input_path'],str(data/'CVXQP3_S.mps'))
   self.assertEqual(record['state'],'completed')
 def test_canonical_and_legacy_lookup(self):
  from data_paths import locate_input,LEGACY
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for relative in ['Lasso-UCI/inputs/mpg_scale_expanded7.mps','Lasso-Synthetic/inputs/lasso_10000_500000.mps']:
    old=root/LEGACY[relative][0];old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(b'old exact input')
    self.assertEqual(locate_input(relative,root),old)
    new=root/relative;new.parent.mkdir(parents=True,exist_ok=True);new.write_bytes(b'new exact input')
    self.assertEqual(locate_input(relative,root),new)
 def test_legacy_archive_extracts_new_directories(self):
  import hashlib,tarfile,io,shutil
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'scripts').mkdir();(root/'scripts/config/data/archives').mkdir(parents=True);(root/'cache').mkdir()
   for name in ['manage_data.py','data_paths.py']:shutil.copy2(ROOT/'scripts'/name,root/'scripts'/name)
   data=[('LASSO/uci/mpg.mps','LASSO/uci/1e-3/mpg.mps',b'uci bytes'),('LASSO/synthetic/lasso.mps','LASSO/02/lasso.mps',b'synthetic bytes')]
   entries=[dict(path=new,legacy_paths=[old],bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),groups=['test']) for new,old,blob in data]
   (root/'scripts/config/data/manifest.json').write_text(json.dumps(entries))
   archive=io.BytesIO()
   with tarfile.open(fileobj=archive,mode='w:gz') as tar:
    for new,old,blob in data:
     member=tarfile.TarInfo(old);member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
   payload=archive.getvalue();parts=[]
   for i,part in enumerate([payload[:len(payload)//2],payload[len(payload)//2:]]):
    name=f'test.tar.gz.part{i:03}';(root/'cache'/name).write_bytes(part);parts.append(dict(file=name,bytes=len(part),sha256=hashlib.sha256(part).hexdigest()))
   spec=dict(complete=True,datasets=[dict(dataset='LASSO',parts=parts,inputs=[dict(path=e['path'],archive_path=e['legacy_paths'][0],bytes=e['bytes'],sha256=e['sha256']) for e in entries])])
   (root/'scripts/config/data/archives/archive_index.json').write_text(json.dumps(spec))
   run=subprocess.run([sys.executable,str(root/'scripts/manage_data.py'),'extract','--cache',str(root/'cache')],capture_output=True,text=True)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   for new,old,blob in data:
    self.assertEqual((root/'data'/new).read_bytes(),blob)
    self.assertFalse((root/'data'/old).exists())
   # The default include-competitors route must load the separate archive index.
   payload=io.BytesIO();blob=b'competitor model';relative='QAP/mps/example.mps'
   with tarfile.open(fileobj=payload,mode='w:gz') as tar:
    member=tarfile.TarInfo(relative);member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
   raw=payload.getvalue();name='competitor.tar.gz.part000';(root/'cache'/name).write_bytes(raw)
   entry=dict(path=relative,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),groups=['test_comp'])
   (root/'scripts/config/data/competitor_qap_manifest.json').write_text(json.dumps([entry]))
   spec=dict(complete=True,datasets=[dict(dataset='QAP/mps',parts=[dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())],inputs=[entry])])
   (root/'scripts/config/data/archives/competitor_qap_index.json').write_text(json.dumps(spec))
   run=subprocess.run([sys.executable,str(root/'scripts/manage_data.py'),'extract','--include-competitors','--cache',str(root/'cache')],capture_output=True,text=True)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   self.assertEqual((root/'data'/relative).read_bytes(),blob)
if __name__=='__main__':unittest.main()

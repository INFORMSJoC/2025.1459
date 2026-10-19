#!/usr/bin/env python3
"""Local orchestration tests; these do not substitute for real Julia/GPU validation."""
import os,json,sys,tempfile,subprocess,unittest,threading,time
from pathlib import Path
from process_watchdog import run_guarded
from experiment_results import ROOT as ORIGINAL_ROOT,read_csv
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
j=json.load(open(sys.argv[-1]));c=j['cases'][0];f=pathlib.Path(j['output_root'])/'results'/j['group']['id']/(c['name']+'.json');f.parent.mkdir(parents=True,exist_ok=True)
f.write_text(json.dumps(dict(execution='completed',result=dict(iter=300,iter_4=100,iter_6=200,time=3,time_4=1,time_6=2,primal_obj=0,residuals=1e-9,output_type='OPTIMAL'))))
''');fake.chmod(0o755)
   cmd=[sys.executable,str(ROOT/'scripts/reviewer_run.py'),'ALL','HPR-QP','--cases','QSCSD6.mps','--julia',str(fake),'--output',str(p/'out')]
   one=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(one.returncode,0,one.stdout+one.stderr)
   mm=json.loads(next((p/'out/raw_results/MM/HPR-QP/runs').glob('*/record.json')).read_text());pd=json.loads(next((p/'out/raw_results/Primal-Dual/HPR-QP/runs').glob('*/record.json')).read_text())
   self.assertEqual(len(mm['rows']),3);self.assertEqual(len(pd['rows']),1)
   two=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(two.returncode,0,two.stdout+two.stderr);self.assertIn('Resume:',two.stdout)
 def test_guard_failure_overrides_archive_and_withholds_aggregate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fake=p/'julia';fake.write_text('#!/usr/bin/env python3\nimport time; time.sleep(60)\n');fake.chmod(0o755)
   r=subprocess.run([sys.executable,str(ROOT/'scripts/reviewer_run.py'),'MM','HPR-QP','--cases','CVXQP3_S.mps','--julia',str(fake),'--hard-time-limit','.5','--output',str(p/'out')],capture_output=True,text=True)
   self.assertEqual(r.returncode,2,r.stdout+r.stderr)
   rows=read_csv(p/'out/paper_results/experiments/MM/results.csv');bad=[x for x in rows if x['solver']=='HPR-QP' and x['case']=='CVXQP3_S.mps'];self.assertEqual(len(bad),3);self.assertTrue(all(x['status']=='HARD_TIME_LIMIT' and x['source']=='fresh' for x in bad))
   summary=read_csv(p/'out/paper_results/experiments/MM/summary.csv');self.assertTrue(all(x['time_SGM10']=='' for x in summary if x['solver']=='HPR-QP'))
class ModelCorrectionChecks(unittest.TestCase):
 def test_old_model_cannot_fill_corrected_result(self):
  import copy
  from experiment_results import refresh
  cat=load_catalog()
  e=copy.deepcopy(next(e for e in cat if e['name']=='Synthetic-CQP'))
  c=next(c for c in e['solvers']['HPR-QP']['cases'] if c['key']=='revision_denser__huber_5000.mps')
  c['key']='model_guard_test_case'
  for archived in c['archived']:archived['case_id']=c['key']
  e['solvers']={'HPR-QP':dict(e['solvers']['HPR-QP'],cases=[c])}
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);run=out/'raw_results'/e['name']/'HPR-QP/runs/old';run.mkdir(parents=True)
   rows=[dict(r,source='fresh') for r in c['archived']]
   (run/'record.json').write_text(json.dumps(dict(state='completed',rows=rows)))
   summary=refresh(e,out)
   self.assertTrue(all(r['time_SGM10'] is None and not r['complete_fresh'] for r in summary))
   selected=read_csv(out/'paper_results/experiments'/e['name']/'results.csv')
   self.assertTrue(all(r['status']=='MODEL_CORRECTION_PENDING' for r in selected))
   for r in rows:r['model_revision']='original_mps_objective_v1'
   (run/'record.json').write_text(json.dumps(dict(state='completed',rows=rows)))
   summary=refresh(e,out)
   self.assertTrue(all(r['complete_fresh'] and r['time_SGM10'] is not None for r in summary))

class DataLayoutChecks(unittest.TestCase):
 def test_launcher_preserves_caller_relative_paths(self):
  import shutil
  with tempfile.TemporaryDirectory(prefix='hpr caller with spaces ') as d:
   root=Path(d);data=root/'external data/MM_mps';data.mkdir(parents=True)
   shutil.copy2(ROOT/'data/MM/inputs/CVXQP3_S.mps',data/'CVXQP3_S.mps')
   fake=root/'fake-julia';fake.write_text('''#!/usr/bin/env python3
import json,pathlib,sys
j=json.load(open(sys.argv[-1]));c=j['cases'][0];f=pathlib.Path(j['output_root'])/'results'/j['group']['id']/(c['name']+'.json');f.parent.mkdir(parents=True,exist_ok=True)
f.write_text(json.dumps(dict(execution='completed',result=dict(iter=300,iter_4=100,iter_6=200,time=3,time_4=1,time_6=2,primal_obj=0,residuals=1e-9,output_type='OPTIMAL'))))
''');fake.chmod(0o755)
   env=dict(os.environ,JULIA_BIN='./fake-julia')
   run=subprocess.run([str(ROOT/'scripts/run.sh'),'MM','HPR-QP','--cases','CVXQP3_S.mps','--data-root','external data','--output','fresh results'],cwd=root,env=env,capture_output=True,text=True)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   record=json.loads(next((root/'fresh results/raw_results/MM/HPR-QP/runs').glob('*/record.json')).read_text())
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

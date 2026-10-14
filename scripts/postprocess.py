#!/usr/bin/env python3
"""Regenerate auditable tables and grayscale figures from preserved references."""
from pathlib import Path
import argparse,csv,json,math,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/paper_exports');p.add_argument('--no-plots',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
groups=json.loads((ROOT/'config/experiments.json').read_text());by={g['id']:g for g in groups};papers=ROOT/'config/reference/paper'
# Use the experiment's selected results for corrected synthetic tables/profiles.
# Original raw paper files stay immutable and available as historical evidence.
from experiment_results import refresh, read_csv
catalog=json.loads((ROOT/'config/reviewer_experiments.json').read_text())
synthetic=next(e for e in catalog if e['name']=='Synthetic-CQP')
refresh(synthetic,ROOT/'experiments')
selected=read_csv(ROOT/'experiments/Synthetic-CQP/results.csv')
corrected={(r['case_id'],r['tolerance']):r for r in selected if r['solver']=='HPR-QP'}
if any(r['status']=='MODEL_CORRECTION_PENDING' for r in corrected.values()):
 raise SystemExit('Corrected synthetic results missing; refusing to export historical scaled-model values as current results.')
synthetic_case_rows={int(c['archived'][0]['raw_file'].rsplit('row',1)[1]):c['key'] for c in synthetic['solvers']['PDQP']['cases']}
def num(x):
 try:return float(x)
 except (ValueError,TypeError):return math.nan
def sgm(xs):
 xs=list(xs);return math.expm1(sum(math.log1p(x/10) for x in xs)/len(xs))*10 if xs and all(math.isfinite(x) and x>=0 for x in xs) else ''
def write(name,rows):
 if not rows:return
 with (a.output/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cap(x):return min(x,3600) if math.isfinite(x) and x>=0 else 3600.
summary=[];details=[]
for g in groups:
 if g['id']=='figure_QSCSD1':continue
 for c in g['cases']:
  r=c['reference'];details.append(dict(group=g['id'],instance=c['name'],data_path=c['path'],iterations=r.get('iter'),time_1e4=r.get('time_4',''),time_1e6=r.get('time_6',''),time_1e8=r.get('alg_time'),residual=r.get('res'),objective=r.get('primal_obj'),status=r.get('status'),power_time=r.get('power_time','')))
 for tol,key in [('1e-4','time_4'),('1e-6','time_6'),('1e-8','alg_time')]:
  vals=[num(c['reference'].get(key)) for c in g['cases']]
  # Do not silently turn a missing threshold field (primal variants) into a failure.
  complete=all(math.isfinite(x) for x in vals)
  limit=g['params']['time_limit']
  summary.append(dict(group=g['id'],tolerance=tol,n=len(vals),time_SGM10=sgm(min(x,limit) for x in vals) if complete else '',time_limit=limit,below_time_limit=sum(0<=x<limit for x in vals) if complete else '',iteration_SGM10=sgm(num(c['reference'].get('iter')) for c in g['cases']) if tol=='1e-8' else '',note='Configured time cap; below limit is a timing count, inspect status/residual separately' if complete else 'Threshold field absent in archived schema; no value inferred'))
write('HPR_reference_details.csv',details);write('HPR_reference_summary.csv',summary)
write('Large_QAP_raw_times.csv',[x for x in details if x['group']=='QAP_large'])
TABLE1=['CVXQP1_L','CVXQP1_M','CVXQP2_L','CVXQP2_M','CVXQP3_L','CVXQP3_M','EXDATA','LASER','MOSARQP1','Q25FV47','QGFRDXPN','QPILOTNO','QSCFXM3','QSCRS8','QSCSD6','QSCSD8','QSCTAP2','QSCTAP3','QSEBA','QSHELL','QSHIP04L','QSHIP04S','QSHIP08L','QSHIP08S','QSHIP12L','QSHIP12S','QSIERRA','QSTANDAT','STCQP1','STCQP2']
table=[];totals=[]
for name in TABLE1:
 row=dict(instance=name)
 for group,label in [('MM_test_version','HPR_QP'),('primal1','HPR_p1'),('primal2','HPR_p2')]:
  c=next(c for c in by[group]['cases'] if Path(c['name']).stem==name);row[label+'_iter']=int(float(c['reference']['iter']));row[label+'_time']=float(c['reference']['alg_time'])
 table.append(row)
for label in ['HPR_QP','HPR_p1','HPR_p2']:totals.append(dict(solver=label,iteration_SGM10=sgm(r[label+'_iter'] for r in table),time_SGM10=sgm(r[label+'_time'] for r in table),n=30))
write('Table1_S1_details.csv',table);write('Table1_summary.csv',totals)
profiles=[];combined_summary=[];source_map=[]
for family,folder,prefix in [('MM','MM','MM_plot_'),('QAP','QAP','QAP_')]:
 for tol in ['1e-4','1e-6','1e-8']:
  source=papers/'Results'/folder/(prefix+tol+('_plot' if family=='QAP' else '')+'.csv');rows=[r for r in csv.DictReader(source.open()) if r.get('name',r.get('instance','')).strip()];solvers=list(rows[0])[1:];series={s:[cap(num(r[s])) for r in rows] for s in solvers}
  assert len(rows)==(137 if family=='MM' else 36)
  if family=='MM' and tol=='1e-8':
   # The old plotting CSV lost two Sub-optimal statuses and misjoined eight runtimes.
   # Preserve it unchanged; use the original named raw records for Gurobi instead.
   raw_source=papers/'Results/MM/Gurobi/Gurobi_MMmps_1e-8_2025Jun17.csv'
   raw_by={Path(x['name']).name:x for x in csv.DictReader(raw_source.open())}
   assert set(raw_by)=={x['name'] for x in rows}
   solver=next(k for k in series if k.lower()=='gurobi')
   series[solver]=[cap(num(raw_by[x['name']]['time'])) if raw_by[x['name']]['status']=='Optimal' else 3600. for x in rows]
   write('MM_Gurobi_1e-8_reconciled.csv',[dict(instance=x['name'],original_plot_seconds=x[solver],raw_seconds=raw_by[x['name']]['time'],raw_status=raw_by[x['name']]['status'],used_seconds=v) for x,v in zip(rows,series[solver])])
   source_map.append(dict(output='MM1e-8.pdf',source=str(raw_source.relative_to(ROOT)),sha256=hashlib.sha256(raw_source.read_bytes()).hexdigest()))
  profiles.append((family+tol,series,len(rows)));source_map.append(dict(output=family+tol+'.pdf',source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
  for s,v in series.items():combined_summary.append(dict(experiment=family,subset='all',solver=s,tolerance=tol,n=len(v),time_SGM10=sgm(v),below_3600=sum(x<3600 for x in v)))
source=papers/'Response/RANDOM_combined_results.csv';raw_with_rows=[(i,r) for i,r in enumerate(csv.reader(source.open()),1) if r and r[0].endswith('.mps')];raw=[r for i,r in raw_with_rows];assert len(raw)==60
sizes=[]
for i,r in raw_with_rows:sizes.append(dict(row_id=i,instance=r[0],m=r[16],n=r[17],nnzA=r[18],nnzQ=r[19],nnz_total=r[20],size_class='Medium' if num(r[20])<1e7 else 'Large' if num(r[20])<1e8 else 'Huge'))
write('Synthetic_dimensions.csv',sizes)
# Frozen transcription from the separately submitted manuscript.
uci=list(csv.DictReader((ROOT/'config/reference/UCI_dimensions.csv').open()))
assert len(uci)==11
write('UCI_dimensions.csv',uci)

for j,tol in enumerate(['1e-4','1e-6','1e-8']):
 series={s:[cap(num(r[1+k*3+j])) for r in raw] for k,s in enumerate(['HPR-QP','PDQP','SCS','CuClarabel','GUROBI'])}
 historical_hpr=list(series['HPR-QP'])
 hpr_rows=[corrected[synthetic_case_rows[i],float(tol)] for i,r in raw_with_rows]
 if any(r['status'] in ['HARD_TIME_LIMIT','MEMORY_LIMIT','INTERRUPTED','EXECUTION_ERROR','MISSING_RESULT'] for r in hpr_rows):
  raise SystemExit('Synthetic HPR execution incomplete; consult experiment summary before generating profiles.')
 series['HPR-QP']=[cap(r['seconds']) if r['solved'] else 3600. for r in hpr_rows]
 profiles.append(('Random'+tol,series,60));source_map.append(dict(output='Random'+tol+'.pdf',source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
 selected_path=ROOT/'experiments/Synthetic-CQP/results.csv'
 source_map.append(dict(output='Random'+tol+'.pdf',source=str(selected_path.relative_to(ROOT)),sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest()))
 write('Synthetic_HPR_selected_'+tol+'.csv',[dict(case_id=r['case_id'],case=r['case'],tolerance=tol,seconds=r['seconds'],solved=r['solved'],source=r['source'],raw_file=r['raw_file'],historical_paper_seconds=old) for r,old in zip(hpr_rows,historical_hpr)])
 for subset in ['all','Medium','Large','Huge']:
  inds=[i for i in range(60) if subset=='all' or sizes[i]['size_class']==subset]
  for s,vals in series.items():
   v=[vals[i] for i in inds];combined_summary.append(dict(experiment='Synthetic_60',subset=subset,solver=s,tolerance=tol,n=len(v),time_SGM10=sgm(v),below_3600=sum(x<3600 for x in v)))
# R2 Table 3 audit: keep manuscript values visible beside regenerated values.
r2_hpr={'Medium':[2.5,4.3,7.1],'Large':[10.4,16.4,22.0],'Huge':[29.3,44.4,61.1],'all':[10.2,15.6,21.3]}
comparison=[]
for subset,values in r2_hpr.items():
 for j,tol in enumerate(['1e-4','1e-6','1e-8']):
  h=next(x for x in combined_summary if x['experiment']=='Synthetic_60' and x['subset']==subset and x['solver']=='HPR-QP' and x['tolerance']==tol)
  comparison.append(dict(subset=subset,tolerance=tol,cases=h['n'],R2_HPR_seconds=values[j],selected_HPR_seconds=h['time_SGM10'],rounded_selected_seconds=round(h['time_SGM10'],1),difference_seconds=h['time_SGM10']-values[j],solved=sum(x['solved'] for x in selected if x['solver']=='HPR-QP' and x['tolerance']==float(tol) and (subset=='all' or x['subset']==subset)),note='Corrected-model verification; shared hardware; manuscript not edited'))
write('Synthetic_HPR_R2_comparison.csv',comparison)
source=papers/'Response/result/TRmat/respond_TR_results.csv';tr=list(csv.reader(source.open()))[2:];tr=[r for r in tr if r and r[0]];assert len(tr)==17
trplots=[];trdetails=[]
for j,tol in enumerate(['1e-4','1e-6','1e-8']):
 series={s:[cap(num(r[1+k*3+j])) for r in tr] for k,s in enumerate(['HPR-QP','SCS','CuClarabel','GUROBI'])}
 for s,v in series.items():combined_summary.append(dict(experiment='TR_17_MAT',subset='all',solver=s,tolerance=tol,n=17,time_SGM10=sgm(v),below_3600=sum(x<3600 for x in v)))
 trplots.append(('TR_solve_time_comparison_'+tol,series));source_map.append(dict(output=trplots[-1][0]+'.pdf',source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
 for i,r in enumerate(tr):trdetails.append(dict(alias_in_old_CSV=r[0],actual_MAT_name='d'+re.search(r'_d(\d+)_',r[0])[1]+'.mat',tolerance=tol,**{s:v[i] for s,v in series.items()}))
qscsd=ROOT/'config/reference/QSCSD1_line_styles.pdf'
source_map.append(dict(output='QSCSD1.pdf',source=str(qscsd.relative_to(ROOT)),sha256=hashlib.sha256(qscsd.read_bytes()).hexdigest()))
write('TR17_details.csv',trdetails);write('Combined_table_summaries.csv',combined_summary);write('Figure_source_map.csv',source_map)
expected={'MM_test_version':[7.4,10.5,12.6],'LASSO_UCI':[4.3,8.5,13.2],'LASSO_synthetic':[11.2,18.5,25.5],'LASSO_UCI_R1_explicit':[27.9,44.9,66.8],'LASSO_synthetic_R1_explicit':[26.1,35.3,46.8],'QAPLIB':[.4,1.8,4.7]}
checks=[]
for group,paper in expected.items():
 for j,tol in enumerate(['1e-4','1e-6','1e-8']):
  value=next(x['time_SGM10'] for x in summary if x['group']==group and x['tolerance']==tol);checks.append(dict(item=group,tolerance=tol,R2_value=paper[j],recomputed=value,rounded_match=round(value,1)==paper[j],source='selected raw HPR CSV'))
for j,tol in enumerate(['1e-4','1e-6','1e-8']):
 value=next(x['time_SGM10'] for x in combined_summary if x['experiment']=='Synthetic_60' and x['subset']=='all' and x['solver']=='HPR-QP' and x['tolerance']==tol);checks.append(dict(item='Synthetic_60_corrected_HPR',tolerance=tol,R2_value=[10.2,15.6,21.3][j],recomputed=value,rounded_match=round(value,1)==[10.2,15.6,21.3][j],source='Selected HPR verification results; corrected original objective. Competitor times remain archived; not a simultaneous performance benchmark.'))
write('Manuscript_summary_checks.csv',checks)
if not a.no_plots:
 import os,tempfile
 plot_cache=tempfile.TemporaryDirectory(prefix='hprqp-mpl-');os.environ.setdefault('MPLCONFIGDIR',plot_cache.name)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.2})
 styles=['-',(0,(7,3)),(0,(7,3,1,3)),(0,(1,2)),(0,(10,3,3,3))];markers=['o','s','^','D','v'];colors=['0','0.25','0.45','0.6','0.15']
 figdir=a.output/'figures';figdir.mkdir(exist_ok=True)
 for name,series,n in profiles:
  fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
  xs=np.logspace(-3,math.log10(3600),400)
  for k,(s,vals) in enumerate(series.items()):ax.plot(xs,[sum(v<=x and v<3600 for v in vals)/n for x in xs],label=s,color=colors[k],linestyle=styles[k],linewidth=2,marker=markers[k],markevery=45,markersize=4)
  ax.set(xscale='log',xlim=(.001,3600),ylim=(0,1.02),xlabel='Solving time (seconds)',ylabel='Fraction of instances solved');ax.legend(loc='lower right',fontsize=9);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
 for name,series in trplots:
  order=sorted(range(len(tr)),key=lambda i:int(re.search(r'_d(\d+)_',tr[i][0])[1]));dims=[int(re.search(r'_d(\d+)_',tr[i][0])[1]) for i in order];fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
  for k,(s,v) in enumerate(series.items()):ax.plot(dims,[v[i] for i in order],label=s,color=colors[k],linestyle=styles[k],linewidth=2,marker=markers[k],markersize=4)
  ax.set(xscale='log',yscale='log',xlabel='Dimension',ylabel='Solving time (seconds; capped at 3600)');ax.legend(fontsize=9);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
 # Preserve the approved QSCSD1 drawing geometry, width=4, medium dash lengths and scientific x ticks.
 import shutil
 source=ROOT/'config/reference/QSCSD1_line_styles.pdf'
 from pypdf import PdfReader,PdfWriter
 from pypdf.generic import ContentStream,FloatObject
 reader=PdfReader(source);writer=PdfWriter();writer.clone_document_from_reader(reader)
 for page in writer.pages:
  content=ContentStream(page.get_contents(),writer);converted=[]
  for operands,op in content.operations:
   if op in [b'RG',b'rg']:
    gray=sum(float(c)*w for c,w in zip(operands,[.299,.587,.114]));converted.append(([FloatObject(gray)],b'G' if op==b'RG' else b'g'))
   else:converted.append((operands,op))
  content.operations=converted;page.replace_contents(content)
 with (figdir/'QSCSD1.pdf').open('wb') as f:writer.write(f)
print(json.dumps(dict(output=str(a.output),HPR_groups=len({x['group'] for x in summary}),manuscript_checks=len(checks),summary_mismatches=[x for x in checks if not x['rounded_match']],table1=totals),indent=2))

print('Synthetic tables/profiles use selected HPR results, including the corrected denser model; competitor profiles retain archived paper times. Other exports and HPR_reference_* remain historical references. Differences from manuscript values are reported, not overwritten.')

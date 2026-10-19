#!/usr/bin/env python3
"""Generate tables and figures separately from paper and reproduced CSVs."""
from pathlib import Path
import argparse,os,tempfile,csv,math,shutil
from catalog import ROOT,load_catalog
from result_paths import experiment_path,solver_path
from experiment_results import refresh,read_csv,write_csv,remove_figure,measured_curves,EXECUTION_FAILURES,file_hash
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=ROOT/'results',help='Results root containing paper_results and reproduced_results')
p.add_argument('--no-plots',action='store_true');a=p.parse_args()
cat=load_catalog()
for e in cat:refresh(e,a.output)
if a.no_plots:
 print('Updated per-solver CSV summaries/statistics and per-experiment comparisons.');raise SystemExit(0)
cache=tempfile.TemporaryDirectory(prefix='hprqp-mpl-');os.environ.setdefault('MPLCONFIGDIR',cache.name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.2})
# Preserve the paper profile palette by solver identity, even for partial plots.
profile_styles={'HPR-QP':('#0072BD',None),'PDQP':('#D95319','o'),'SCS':('#EDB120','+'),'CuClarabel':('#7E2F8E','o'),'Gurobi':('#77AC30','X')}
tr_styles={'HPR-QP':('blue','o'),'SCS':('red','s'),'CuClarabel':('green','D'),'Gurobi':('purple','^')}
outputs=[]
for paper in [True,False]:
 for e in cat:
  folder=experiment_path(a.output,e['name'],paper)
  data={s:read_csv(solver_path(a.output,e['name'],s,paper)/'summary.csv') for s in e['solvers']}
  if e['name'] in ['MM','QAPLIB','Synthetic-CQP','Trust-Region']:
   for tol in [1e-4,1e-6,1e-8]:
    dest=folder/'figures'/('solve_times_' if e['name']=='Trust-Region' else 'profile_')
    dest=dest.with_name(dest.name+format(tol,'.0e')+'.pdf')
    series={};sources=[]
    for s,rows in data.items():
     expected=[r for c in e['solvers'][s]['cases'] for r in c['archived'] if r['tolerance']==tol]
     rs=[r for r in rows if r['tolerance']==tol]
     valid=len(rs)==len(expected) and all(r['status'] not in EXECUTION_FAILURES and (not r['solved'] or r['seconds'] is not None) for r in rs)
     if not valid:continue
     source=solver_path(a.output,e['name'],s,paper)/'summary.csv'
     series[s]=rs;sources.append(dict(solver=s,source=str(source.relative_to(a.output)),cases=len(rs),summary_sha256=file_hash(source)))
    if not series:
     remove_figure(dest);continue
    fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
    for i,(s,rs) in enumerate(series.items()):
     limit=e['solvers'][s]['time_limit']
     if e['name']=='Trust-Region':
      rs=sorted((r for r in rs if r['solved'] and r['seconds']<limit),key=lambda r:int(Path(r['case']).stem[1:]));xs=[int(Path(r['case']).stem[1:]) for r in rs];ys=[r['seconds'] for r in rs]
      color,marker=tr_styles[s]
      if rs:ax.plot(xs,ys,label=s,color=color,marker=marker,linewidth=2)
     else:
      vals=[min(r['seconds'],limit) if r['solved'] else limit for r in rs];xs=np.logspace(-3,math.log10(limit),400)
      color,marker=profile_styles[s]
      ax.plot(xs,[sum(v<=x and v<limit for v in vals)/len(vals) for x in xs],label=s,color=color,linewidth=2,marker=marker,markevery=45,markersize=5,markerfacecolor='none' if s=='PDQP' else color)
    if e['name']=='Trust-Region':ax.set(xscale='log',yscale='log',xlabel='Dimension',ylabel='Solving time (seconds)')
    else:ax.set(xscale='log',xlim=(.001,3600),ylim=(0,1.02),xlabel='Solving time (seconds)',ylabel='Fraction of instances solved')
    ax.set_title(e['name']+r' ($\mathrm{Tol}=10^{'+str(int(math.log10(tol)))+'}$)',fontsize=13)
    ax.legend(fontsize=9,loc='best');dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest);plt.close(fig)
    write_csv(dest.with_suffix('.csv'),sources);outputs.append(dest)
  elif e['name']=='Restart-Sigma':
   # The paper drawing is preserved as supplied; rerun curves use measured samples.
   if paper:
    source=experiment_path(ROOT/'results',e['name'],True)/'QSCSD1.pdf';dest=folder/'QSCSD1.pdf'
    if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
    continue
   curves=measured_curves(data['HPR-QP']);dest=folder/'QSCSD1.pdf'
   if len(curves)!=len(e['solvers']['HPR-QP']['cases']):
    remove_figure(dest);continue
   fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
   styles=['-',(0,(7,3)),(0,(7,3,1,3))];labels=[r'no restart, $\sigma=1$',r'adaptive restart, $\sigma=1$',r'adaptive restart and $\sigma$']
   colors=['#009AF9','#E36F47','#3DA44D']
   for i,name in enumerate(['no_restart','adaptive_restart','adaptive_restart_sigma']):
    rs=curves[name]
    ax.semilogy([int(r['iteration']) for r in rs],[float(r['optimality']) for r in rs],label=labels[i],color=colors[i],linestyle=styles[i],linewidth=3)
   ax.set(xlabel='Iteration',ylabel='KKT residual');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.legend()
   fig.savefig(dest);plt.close(fig);outputs.append(dest)
   source=solver_path(a.output,e['name'],'HPR-QP')/'summary.csv'
   write_csv(dest.with_suffix('.csv'),[dict(solver='HPR-QP',source=str(source.relative_to(a.output)),summary_sha256=file_hash(source))])
print('Generated',len(outputs),'figures from separate paper/reproduced measurements. Incomplete solver datasets are excluded from profiles.')

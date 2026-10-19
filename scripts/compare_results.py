import math

def number(v):
    try:return float(v)
    except:return math.nan

def compare(d):
    b=d['reference'];a=d.get('result',{});s=a.get('output_type','ERROR');bs=b.get('status','')
    ar=number(a.get('residuals'));ao=number(a.get('primal_obj'));bo=number(b.get('primal_obj'));diff=abs(ao-bo)/max(1,abs(bo))
    matched=number(a.get('iter'))==number(b.get('iter'))
    if d['group']=='figure_QSCSD1' and d['execution']=='completed':
        x=a.get('plot_iter',[]);y=a.get('plot_KKT',[]);by=b['curve_optimality']
        curve_ok=x==b['curve_iteration'] and len(y)==len(by) and all(v>0 and w>0 and abs(math.log10(v/w))<=0.01 for v,w in zip(y,by))
        verdict='curve_reproduced' if curve_ok and matched else 'mismatch_review'
    elif d['execution']!='completed':verdict='execution_error' 
    elif d.get('model_revision')=='original_mps_objective_v1':verdict='corrected_model_converged' if s=='OPTIMAL' and ar<=1.01e-8 else 'corrected_model_review'
    elif bs=='OPTIMAL' and s=='OPTIMAL' and ar<=1.01e-8 and diff<=1e-6:verdict='reproduced' if matched else 'converged_iteration_difference'
    elif bs!='OPTIMAL' and s=='OPTIMAL' and ar<=1.01e-8:verdict='newly_converged_review'
    elif bs!='OPTIMAL' and s!='OPTIMAL':verdict='both_unsolved_review'
    else:verdict='mismatch_review'
    return dict(group=d['group'],name=d['name'],verdict=verdict,status=s,saved_status=bs,iter=a.get('iter'),saved_iter=b.get('iter'),iteration_match=matched,residual=a.get('residuals'),saved_residual=b.get('res'),objective=a.get('primal_obj'),saved_objective=b.get('primal_obj'),scaled_objective_difference=diff if math.isfinite(diff) else None,seconds=a.get('time'),saved_seconds=b.get('alg_time'),gpu=d.get('gpu'),error=d.get('error',''),iter_3=a.get('iter_3'),saved_iter_3=b.get('iter_3'),iter_4=a.get('iter_4'),saved_iter_4=b.get('iter_4'),iter_6=a.get('iter_6'),saved_iter_6=b.get('iter_6'))

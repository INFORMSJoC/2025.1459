using JuMP, LinearAlgebra, SparseArrays, JSON, SHA, Pkg, Dates
BLAS.set_num_threads(1)
job=JSON.parsefile(ARGS[1]);name=job["solver"];out=job["output"];mkpath(out)
if name=="SCS"
 @eval using SCS, SCS_GPU_jll
 optimizer=SCS.Optimizer
elseif name=="CuClarabel"
 @eval using Clarabel, CUDA
 CUDA.device!(0);optimizer=Clarabel.Optimizer
elseif name=="Gurobi"
 @eval using Gurobi
 optimizer=Gurobi.Optimizer
else
 error("Unknown solver")
end
versions=Dict(p.name=>string(p.version) for p in values(Pkg.dependencies()) if p.version!==nothing)
environment=Dict("julia"=>string(VERSION),"packages"=>versions,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"cuda_visible_devices"=>get(ENV,"CUDA_VISIBLE_DEVICES",""),"manifest_sha256"=>bytes2hex(sha256(read(joinpath(dirname(Base.active_project()),"Manifest.toml")))))
if name=="CuClarabel"
 environment["cuda_runtime"]=string(CUDA.runtime_version());environment["gpu"]=CUDA.name(CUDA.device())
end
function settings(tol,family)
 limit=get(job,"solver_time_limit",3600.0)
 if name=="SCS"
  return Dict{String,Any}("eps_rel"=>tol,"eps_abs"=>tol,"time_limit_secs"=>limit,"max_iters"=>typemax(Int32),"linear_solver"=>SCS.GpuIndirectSolver)
 elseif name=="CuClarabel"
  d=Dict{String,Any}("tol_gap_rel"=>tol,"tol_gap_abs"=>tol,"tol_feas"=>tol,"reduced_tol_gap_rel"=>tol,"reduced_tol_gap_abs"=>tol,"reduced_tol_feas"=>tol,"time_limit"=>limit,"max_iter"=>typemax(Int32),"direct_solve_method"=>:cudss)
  if family!="MM";d["tol_infeas_abs"]=0.0;d["tol_infeas_rel"]=0.0;end
  return d
 else
  # Explicit barrier selection follows the manuscript; original QP logs used
  # automatic method selection, which also selected barrier on these inputs.
  return Dict{String,Any}("BarConvTol"=>tol,"OptimalityTol"=>tol,"FeasibilityTol"=>tol,"TimeLimit"=>limit,"Presolve"=>0,"OutputFlag"=>1,"Method"=>2,"Threads"=>1)
 end
end
function load_model(c)
 if c["family"]=="TR"
  @eval using MAT
  vars=Base.invokelatest(MAT.matread,c["path"]);H=vars["H"];b=vars["b"] isa Number ? [vars["b"]] : vec(vars["b"]);v=vec(vars["c"]);A=vars["A"]
  m=Model();@variable(m,y[1:length(v)]);@objective(m,Min,0.5*y'*H*y+dot(v,y));@constraint(m,A*y.==b);@constraint(m,y in SecondOrderCone());return m
 end
 read_from_file(c["path"])
end
function solve(c,tol)
 model=load_model(c);set_optimizer(model,optimizer);attrs=settings(tol,c["family"])
 merge!(attrs,get(c,"settings",Dict{String,Any}()))
 for (k,v) in attrs;set_optimizer_attribute(model,k,v);end
 wall=@elapsed optimize!(model)
 r=Dict{String,Any}("status"=>string(termination_status(model)),"raw_status"=>raw_status(model),"solver_name"=>solver_name(model),"solve_time_moi"=>solve_time(model),"wall_optimize_seconds"=>wall,"solve_time"=>solve_time(model))
 if has_values(model);r["objective"]=objective_value(model);end
 if name!="Gurobi";try;r["dual_objective"]=dual_objective_value(model);catch;end;end
 if name=="SCS"
  r["iterations"]=MOI.get(backend(model),SCS.ADMMIterations())
  info=SCS.HPR_SAMPLE_LAST_INFO[]
  r["solve_time"]=info.solve_time/1000;r["setup_time"]=info.setup_time/1000
  for f in [:res_pri,:res_dual,:gap,:pobj,:dobj];r[string(f)]=getproperty(info,f);end
  r["native_primal_objective"]=info.pobj
  if haskey(r,"objective");r["objective_offset"]=r["objective"]-info.pobj;end
 elseif name=="CuClarabel"
  r["iterations"]=MOI.get(backend(model),MOI.BarrierIterations())
  r["objective_offset"]=unsafe_backend(model).objconstant
  info=unsafe_backend(model).solver_info
  r["native_primal_objective"]=info.cost_primal
  r["native_dual_objective"]=info.cost_dual
  for f in [:setup_time,:solve_time,:total_time,:res_primal,:res_dual,:gap_abs,:gap_rel];r[string(f)]=getproperty(info,f);end
 else
  r["iterations"]=MOI.get(backend(model),MOI.BarrierIterations())
  for f in ["ConstrVio","BoundVio","DualVio","ComplVio"]
   try;r[f]=MOI.get(backend(model),Gurobi.ModelAttribute(f));catch;end
  end
 end
 return r,Dict(k=>(v isa Symbol || v isa DataType ? string(v) : v) for (k,v) in attrs)
end
# Separate same-case warmup; measured solves reconstruct a fresh model.
open(joinpath(out,"warmup.log"),"w") do io
 redirect_stdout(io) do
  redirect_stderr(io) do
   try;solve(get(job,"warmup_case",job["cases"][1]),1e-4);catch e;println("WARMUP_ERROR ",sprint(showerror,e));end
  end
 end
end
for c in job["cases"],tol in get(c,"tolerances",job["tolerances"])
 id=c["family"]*"__"*c["name"]*"__"*string(tol);path=joinpath(out,id*".json");isfile(path)&&continue
 record=Dict{String,Any}("solver"=>name,"family"=>c["family"],"name"=>c["name"],"tolerance"=>tol,"input_path"=>c["path"],"input_sha256"=>bytes2hex(sha256(read(c["path"]))),"environment"=>environment,"started"=>string(now()),"settings"=>Dict(k=>string(v) for (k,v) in settings(tol,c["family"])))
 open(joinpath(out,id*".log"),"w") do io
  redirect_stdout(io) do
   redirect_stderr(io) do
    try
     r,attrs=solve(c,tol);record["result"]=r;record["settings"]=attrs;record["execution"]="completed"
    catch e
     record["execution"]="error";record["error"]=sprint(showerror,e,catch_backtrace());println(record["error"])
    end
   end
  end
 end
 record["finished"]=string(now());open(path*".tmp","w") do io;JSON.print(io,record,2);end;mv(path*".tmp",path,force=true)
 println(name," ",id," ",record["execution"]);flush(stdout)
end

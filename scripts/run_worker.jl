using JSON, Dates, LinearAlgebra
BLAS.set_num_threads(1)
root=dirname(@__DIR__)
job=JSON.parsefile(ARGS[1])
g=job["group"]
function atomic_json(path, value)
    mkpath(dirname(path));tmp=path*".tmp";open(tmp,"w") do io;JSON.print(io,value,2);end;mv(tmp,path,force=true)
end
include(joinpath(root,"scripts","verify_environment.jl"))
include(joinpath(root,"src","variants",g["variant"],g["entry"]))
M=g["module"]=="Main" ? Main : getfield(Main,Symbol(g["module"]))
if !g["cpu"]
    @eval using CUDA
    CUDA.device!(0)
end
function parameters()
    ctor=g["kind"] in ["tr_mat","tr_jld"] ? :HPRSOCP_parameters : :HPRQP_parameters
    p=getfield(M,ctor)()
    for (k,v) in g["params"]
        if hasproperty(p,Symbol(k));setproperty!(p,Symbol(k),convert(fieldtype(typeof(p),Symbol(k)),v));end
    end
    if hasproperty(p,:warm_up);p.warm_up=false;end
    return p
end
function solve_case(case, warm=false)
    file=case["path"]
    if g["kind"]=="primal"
        return M.run_file(file,true,warm ? 200 : 2147483647,g["params"]["time_limit"],g["params"]["stoptol"])
    end
    p=parameters()
    for (k,v) in get(case,"params",Dict());setproperty!(p,Symbol(k),convert(fieldtype(typeof(p),Symbol(k)),v));end
    if warm;p.max_iter=200;end
    if g["kind"]=="tr_mat"
        vars=M.matread(file)
        H=vars["H"];c=vec(vars["c"]);A=vars["A"];b=vars["b"] isa Number ? [vars["b"]] : vec(vars["b"])
        return M.run_TR(H,A,b,c,p)
    elseif g["kind"]=="tr_jld"
        H,c,A,b=M.load_trust_region_problem(file)
        return M.run_TR(H,A,b,c,p)
    end
    if g["variant"] == "main"
        return M.run_file(file,p; force_general_q=get(g,"force_general_q",false))
    end
    return M.run_file(file,p)
end
function results_dict(r)
    d=Dict{String,Any}()
    for k in [:iter,:iter_3,:iter_4,:iter_6,:time,:time_3,:time_4,:time_6,:power_time,:primal_obj,:residuals,:output_type,:plot_iter,:plot_KKT]
        if hasproperty(r,k) && isdefined(r,k);v=getproperty(r,k);d[string(k)]=v isa AbstractFloat && !isfinite(v) ? string(v) : v;end
    end
    return d
end
output_root=job["output_root"]
out=joinpath(output_root,"results",g["id"])
mkpath(out)
# Warm-up every fresh Julia process; main solves always restore original settings.
try
    println("WARMUP ",job["cases"][1]["name"]);flush(stdout)
    solve_case(job["cases"][1],true)
catch e
    println("WARMUP_ERROR ",sprint(showerror,e));flush(stdout)
end
GC.gc(true)
if !g["cpu"];CUDA.reclaim();end
for case in job["cases"]
    name=case["name"];dest=joinpath(out,name*".json")
    isfile(dest) && continue
    atomic_json(joinpath(output_root,"progress",job["id"]*".json"),Dict("group"=>g["id"],"case"=>name,"started_epoch"=>time(),"time_limit"=>g["params"]["time_limit"]))
    record=Dict{String,Any}("name"=>name,"group"=>g["id"],"input_path"=>case["path"],"reference"=>case["reference"],"started"=>string(now()),"gpu"=>get(ENV,"CUDA_VISIBLE_DEVICES",""),"phase"=>job["phase"],"environment"=>environment_record,"parameters"=>merge(g["params"],get(case,"params",Dict())),"source_variant"=>g["variant"],"force_general_q"=>get(g,"force_general_q",false),"model_revision"=>get(g,"model_revision",""))
    try
        open(joinpath(out,name*".log"),"w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    println("CASE ",name," settings=",g["params"])
                    r=solve_case(case)
                    record["result"]=results_dict(r)
                    record["execution"]="completed"
                end
            end
        end
    catch e
        record["execution"]="error";record["error"]=sprint(showerror,e,catch_backtrace())
    end
    record["finished"]=string(now());atomic_json(dest,record)
    println("FINISHED ",g["id"]," ",name," ",record["execution"]);flush(stdout)
    GC.gc(true)
    if !g["cpu"];CUDA.reclaim();end
end

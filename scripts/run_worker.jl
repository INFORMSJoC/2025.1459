using JSON, Dates, LinearAlgebra, CSV, DataFrames
BLAS.set_num_threads(1)
root=dirname(@__DIR__)
job=JSON.parsefile(ARGS[1])
g=job["group"]
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
out=joinpath(job["output_root"],"native");mkpath(out)
# Preserve the same separate warmup used by the bounded launcher.
try
    println("WARMUP ",job["cases"][1]["name"]);flush(stdout)
    solve_case(job["cases"][1],true)
catch e
    println("WARMUP_ERROR ",sprint(showerror,e));flush(stdout)
end
GC.gc(true)
if !g["cpu"];CUDA.reclaim();end
if g["kind"]=="figure"
    for case in job["cases"]
        r=open(joinpath(out,"HPRQP_log.txt"),"w") do io
            redirect_stdout(io) do
                solve_case(case)
            end
        end
        value(k)=hasproperty(r,k) && isdefined(r,k) ? getproperty(r,k) : missing
        CSV.write(joinpath(out,"HPRQP_zxwyw_result.csv"),DataFrame(name=[case["name"]],iter=[r.iter],alg_time=[r.time],res=[r.residuals],primal_obj=[r.primal_obj],status=[r.output_type],iter_4=[value(:iter_4)],time_4=[value(:time_4)],iter_6=[value(:iter_6)],time_6=[value(:time_6)],power_time=[value(:power_time)]))
        CSV.write(joinpath(out,"curves.csv"),DataFrame(case=fill(case["name"],length(r.plot_iter)),sample=collect(0:length(r.plot_iter)-1),iteration=r.plot_iter,optimality=r.plot_KKT))
    end
else
    # The original dataset writers receive a temporary directory containing only
    # the requested inputs. Their CSV/log output remains native dataset output.
    mktempdir() do inputs
        for case in job["cases"]
            symlink(case["path"],joinpath(inputs,case["name"]))
        end
        if g["kind"]=="primal"
            M.run_dataset(inputs*"/",out*"/",g["params"]["time_limit"],g["params"]["stoptol"],false,0,true)
        else
            p=parameters();p.do_warm_up=false
            if g["variant"]=="main"
                M.run_dataset(inputs*"/",out*"/",p,0;force_general_q=get(g,"force_general_q",false))
            else
                M.run_dataset(inputs*"/",out*"/",p,0)
            end
        end
    end
end
# The source is already versioned under src; omit dataset-writer source copies.
for filename in ["HPRQP.jl","HPRSOCP.jl"]
    path=joinpath(out,filename);isfile(path) && rm(path)
end
println("FINISHED ",g["id"]);flush(stdout)

# Reconstructed appendix driver; not recovered historical experiment code.
# Uses the existing HPR-QP/ HPR(p1) algorithms, with unscaled input and exact
# spectral bounds specified in the manuscript. Typed overloads below are confined
# to this process; they supply paper constants instead of power estimates.
using JSON, Dates, SparseArrays, LinearAlgebra, CUDA, CUDA.CUSPARSE, CSV, DataFrames
BLAS.set_num_threads(1)
root=dirname(@__DIR__);g=Dict("cpu"=>false)
include(joinpath(root,"scripts","verify_environment.jl"))
include(joinpath(root,"src","variants","main","HPRQP_zxwyw.jl"))
module PrimalExample
include(joinpath(@__DIR__,"..","src","variants","primal1","solver.jl"))
end
input_path=joinpath(root,"data","Appendix-Example","instance.json")
d=JSON.parsefile(input_path)
if length(ARGS)>=2
 d["check_iter"]=parse(Int,ARGS[2]);@assert d["check_iter"]>0
 d["check_iter_provenance"]="Explicit diagnostic override; not evidence of the historical setting"
end
out=isempty(ARGS) ? joinpath(root,"results","raw_results","Appendix-Example","standalone") : abspath(ARGS[1]);mkpath(out)
Q=permutedims(hcat(d["Q"]...));A=permutedims(hcat(d["A"]...));c=Float64.(d["c"]);b=Float64.(d["b"])
Sw=Q*(d["lambda_Q"]*I-Q);Sy=d["lambda_A"]*I-A*A';Sx=d["lambda_Q"]*I-Q+d["sigma"]*(d["lambda_A"]*I-A'*A)
Ty=d["sigma"]*Sy+d["sigma"]^2*A*Q*inv(d["sigma"]*Q*Q+Q+d["sigma"]*Sw)*Q*A'
T1=[Ty zeros(2,2);zeros(2,2) d["sigma"]*Sw]
rows=NamedTuple[]
for (name,mat,paper) in [("Sw",Sw,19.99),("Sy",Sy,11.86),("T1",T1,19.99),("Sx",Sx,31.59)]
 push!(rows,(operator=name,spectral_norm=opnorm(mat,2),paper_rounded_norm=paper,rounded_match=round(opnorm(mat,2),digits=2)==paper,matrix_json=JSON.json(mat)))
end
CSV.write(joinpath(out,"proximal_norms.csv"),DataFrame(rows))
results=NamedTuple[]
for (label,M) in [("dual",HPRQP_zxwyw),("primal1",PrimalExample)]
 @eval M begin
  power_iteration_A_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR)=$(d["lambda_A"])
  power_iteration_Q_gpu(Q::CuSparseMatrixCSR)=$(d["lambda_Q"])
 end
 p=M.initialize_parameters();p.useGPU=true;p.sigma=1.0;p.sigma_fixed=true;p.mode="linearize";p.lambda_factor=1.0;p.check_iter=d["check_iter"];p.restart=d["restart"];p.stoptol=d["stoptol"];p.max_iter=100000;p.time_limit=60.0
 function solve()
  q=M.qp_formulation(sparse(Float64.(Q)),c,sparse(Float64.(A)),fill(-Inf,2),b,zeros(2),fill(Inf,2),0.0)
  s=M.scaling!(q,false,false,false,false)
  common=(CuSparseMatrixCSR(q.Q),CuVector(q.c),CuSparseMatrixCSR(q.A),CuSparseMatrixCSR(q.A'),CuVector(q.AL),CuVector(q.AU),CuVector(q.l),CuVector(q.u),CuVector(q.idxI),q.m1,q.obj_constant,CuVector(q.diag_Q),q.Q_is_diag)
  qg=label=="dual" ? M.QP_info_gpu(common...,CuVector(q.x0),CuVector(q.y0),q.noC) : M.QP_info_gpu(common...)
  sg=M.Scaling_info_gpu(CuVector(s.l_org),CuVector(s.u_org),CuVector(s.row_norm),CuVector(s.col_norm),s.b_scale,s.c_scale,s.norm_b,s.norm_c,s.norm_b_org,s.norm_c_org)
  M.HPRQP(qg,sg,p)
 end
 started=string(now())
 r=open(joinpath(out,label*".log"),"w") do io
  redirect_stdout(io) do
   println("Reconstructed appendix example; check_iter=",p.check_iter,"; fixed sigma=1; no scaling; lambda_A=12.2; lambda_Q=21.0")
   solve()
  end
 end
 raw=Dict{String,Any}()
 for k in [:iter,:iter_3,:iter_4,:iter_6,:time,:time_3,:time_4,:time_6,:power_time,:primal_obj,:residuals,:output_type,:plot_iter,:plot_KKT]
  if hasproperty(r,k) && isdefined(r,k);raw[string(k)]=getproperty(r,k);end
 end
 record=Dict("group"=>"appendix_concrete","name"=>label,"execution"=>"completed","input_path"=>input_path,"parameters"=>d,"environment"=>environment_record,"started"=>started,"finished"=>string(now()),"result"=>raw,"provenance"=>"Reconstructed driver; historical iteration counts not independently verified")
 open(joinpath(out,label*".json"),"w") do io;JSON.print(io,record,2);end
 push!(results,(method=label,iter=r.iter,paper_approx_iter=d["paper_approximate_iterations"][label],time=r.time,primal_obj=r.primal_obj,residual=r.residuals,status=r.output_type,check_iter=p.check_iter,sigma=p.sigma,lambda_A=d["lambda_A"],lambda_Q=d["lambda_Q"],scaling=false))
 println(label," iter=",r.iter," residual=",r.residuals)
end
CSV.write(joinpath(out,"raw_results.csv"),DataFrame(results))

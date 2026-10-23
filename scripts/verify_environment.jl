using SHA, TOML, CUDA
@assert VERSION == v"1.10.4" "This reproduction requires Julia 1.10.4"
expected = TOML.parsefile(joinpath(root,"scripts","environments","LOCK.toml"))
for f in keys(expected["sha256"])
    @assert bytes2hex(sha256(read(joinpath(root,"scripts","environments",f)))) == expected["sha256"][f] "Environment file changed: $f"
end
@assert pkgversion(CUDA) == v"5.11.0" "CUDA.jl version changed"
environment_record=Dict{String,Any}("julia"=>string(VERSION),"CUDA_jl"=>string(pkgversion(CUDA)),"manifest_sha256"=>expected["sha256"]["Manifest.toml"],"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"load_path"=>copy(LOAD_PATH))
@assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
if !g["cpu"]
    @assert CUDA.functional() "CUDA is not functional"
    @assert CUDA.runtime_version() == v"13.2.0" "CUDA runtime changed; inspect environment before continuing"
    environment_record["cuda_runtime"]=string(CUDA.runtime_version())
    environment_record["cuda_driver_api"]=string(CUDA.driver_version())
    environment_record["gpu_name"]=CUDA.name(CUDA.device())
end
println("ENVIRONMENT ",environment_record);flush(stdout)

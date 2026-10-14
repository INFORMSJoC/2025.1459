using JSON, Pkg, SHA, LinearAlgebra, CUDA
BLAS.set_num_threads(1)
r=Dict("julia"=>string(VERSION),"packages"=>Dict(p.name=>string(p.version) for p in values(Pkg.dependencies()) if p.version!==nothing),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"cuda_runtime"=>string(CUDA.runtime_version()),"gpu"=>CUDA.name(CUDA.device()),"manifest_sha256"=>bytes2hex(sha256(read(joinpath(dirname(Base.active_project()),"Manifest.toml")))))
open(ENV["HPR_COMPETITOR_META"],"w") do io;JSON.print(io,r,2);end

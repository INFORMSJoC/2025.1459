using QPSReader
using SparseArrays
using LinearAlgebra
using Statistics
using CUDA
using CUDA.CUSPARSE
using Printf
using CSV
using DataFrames
using Random
using Logging
# using CUDSS

function read_mps_SCS(file::String)
    # Read data from file
    if file[end-3:end] == ".mps" || file[end-4:end] == ".MPS"
        println("Reading MPS file: ", file)
        qp = QPSReader.readqps(file, mpsformat=:free)
    elseif file[end-3:end] == ".QPS" || file[end-4:end] == ".qps"
        println("Reading QPS file: ", file)
        qp = QPSReader.readqps(file, mpsformat=:fixed)
    else
        error("Unsupported file format. Please provide a .mps or .qps file.")
    end

    # important to have 4th and 5th arguments that mip library has empty cols in A (guess by A cols < c dim).
    A = sparse(qp.arows, qp.acols, qp.avals, qp.ncon, qp.nvar)
    lcon = qp.lcon
    ucon = qp.ucon

    # quadratic part
    Q = sparse(qp.qrows, qp.qcols, qp.qvals, qp.nvar, qp.nvar)
    # the Q matrix is not symmetric, so we need to symmetrize it
    diag_Q = diag(Q)
    Q = Q + Q' - Diagonal(diag_Q)

    # linear part
    c = qp.c
    c0 = qp.c0

    # bounds
    lvar = qp.lvar
    uvar = qp.uvar

    # Return the modified qp
    return Q, c, A, lcon, ucon, lvar, uvar, c0
end

mutable struct QP_info_cpu
    Q::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    idxI::Vector{Int}
    m1::Int
    obj_constant::Float64
    diag_Q::Vector{Float64}
    Q_is_diag::Bool
end

mutable struct Scaling_info
    l_org::Vector{Float64}
    u_org::Vector{Float64}
    row_norm::Vector{Float64}
    col_norm::Vector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
end

mutable struct HPRQP_parameters
    stoptol::Float64
    sigma::Float64
    max_iter::Int
    sigma_fixed::Bool
    time_limit::Float64
    restart::Int
    lambda_factor::Float64
    # 0 for best, 1 for current
    check_iter::Int
    useGPU::Bool
    sigma_method::Int
    # direct, linearize
    mode::String
    HPRQP_parameters() = new(1e-6, 1.0, 1000, false, 3600.0, -1, 1.005, 10, false, 1, "direct")
end

mutable struct HPRQP_residuals
    err_Rp_org_bar::Float64
    err_Rd_org_bar::Float64
    err_Rp_bar::Float64
    err_Rd_bar::Float64
    err_K_org_bar::Float64
    err_z_org_bar::Float64
    err_Comp_org_bar::Float64
    primal_obj::Float64
    dual_obj::Float64
    is_updated::Bool

    primal_feas_org_bar::Float64
    dual_feas_org_bar::Float64
    KKTx_and_gap_org_bar::Float64
    KKTx_org_bar::Float64

    primal_obj_bar::Float64
    rel_gap_bar::Float64

    KKTx_bar::Float64
    dual_obj_bar::Float64

    # Define a default constructor
    HPRQP_residuals() = new()
end

mutable struct HPRQP_results
    iter::Int
    iter_3::Int
    iter_6::Int
    time::Float64
    time_3::Float64
    time_6::Float64
    power_time::Float64
    primal_obj::Float64
    residuals::Float64
    gap::Float64
    output_type::String
    sigma::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    HPRQP_results() = new()
end

mutable struct HPRQP_workspace_gpu
    x::CuVector{Float64}
    x_hat::CuVector{Float64}
    x_bar::CuVector{Float64}
    dx::CuVector{Float64}
    y::CuVector{Float64}
    y_hat::CuVector{Float64}
    y_bar::CuVector{Float64}
    dy::CuVector{Float64}
    z_bar::CuVector{Float64}
    s_bar::CuVector{Float64}
    Q::CuSparseMatrixCSR{Float64,Int32}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    c::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    idxI::CuVector{Int}
    Rp::CuVector{Float64}
    Rd::CuVector{Float64}
    m1::Int
    m::Int
    n::Int
    sigma::Float64
    lambda_max_A::Float64
    lambda_max_Q::Float64
    Ax::CuVector{Float64}
    ATy::CuVector{Float64}
    ATy_bar::CuVector{Float64}
    Qx::CuVector{Float64}
    last_x::CuVector{Float64}
    last_y::CuVector{Float64}
    diag_Q::CuVector{Float64}
    fact1::CuVector{Float64}
    fact2::CuVector{Float64}
    idx_boxed::CuVector{Bool}
    idx_unbounded::CuVector{Bool}
    idx_linf::CuVector{Bool}
    idx_uinf::CuVector{Bool}
    update_z::Bool
    tempv::CuVector{Float64}
    HPRQP_workspace_gpu() = new()
end

mutable struct HPRQP_workspace_cpu
    x::Vector{Float64}
    x_hat::Vector{Float64}
    x_bar::Vector{Float64}
    dx::Vector{Float64}
    y::Vector{Float64}
    y_hat::Vector{Float64}
    y_bar::Vector{Float64}
    dy::Vector{Float64}
    z_bar::Vector{Float64}
    s_bar::Vector{Float64}
    Q::SparseMatrixCSC{Float64,Int32}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    AL::Vector{Float64}
    AU::Vector{Float64}
    c::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    idxI::Vector{Int}
    Rp::Vector{Float64}
    Rd::Vector{Float64}
    m1::Int
    m::Int
    n::Int
    sigma::Float64
    lambda_max_A::Float64
    lambda_max_Q::Float64
    Ax::Vector{Float64}
    ATy::Vector{Float64}
    ATy_bar::Vector{Float64}
    Qx::Vector{Float64}
    last_x::Vector{Float64}
    last_y::Vector{Float64}
    idx_boxed::Vector{Bool}
    idx_unbounded::Vector{Bool}
    idx_linf::Vector{Bool}
    idx_uinf::Vector{Bool}
    update_z::Bool
    tempv::Vector{Float64}
    HPRQP_workspace_cpu() = new()
end

mutable struct HPRQP_restart
    # 0: no restart, 1: restart with avg, 2: restart with current
    restart_flag::Int

    # 0: no restart, 1: fixed step restart, 2: adaptive restart
    restart_type::Int
    first_restart::Bool
    last_gap::Float64
    current_gap::Float64
    save_gap::Float64
    inner::Int
    step::Int

    sufficient::Int
    necessary::Int
    long::Int
    ratio::Int
    times::Int

    weighted_norm::Float64

    HPRQP_restart() = new()
end

# mutable struct QP_info_gpu
#     Q::CuSparseMatrixCSR{Float64,Int32}
#     c::CuVector{Float64}
#     A::CuSparseMatrixCSR{Float64,Int32}
#     AT::CuSparseMatrixCSR{Float64,Int32}
#     AL::CuVector{Float64}
#     AU::CuVector{Float64}
#     l::CuVector{Float64}
#     u::CuVector{Float64}
#     idxI::CuVector{Int}
#     m1::Int
#     obj_constant::Float64
# end

mutable struct QP_info_gpu
    Q::CuSparseMatrixCSR{Float64,Int32}
    c::CuVector{Float64}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    idxI::CuVector{Int}
    m1::Int
    obj_constant::Float64
    diag_Q::CuVector{Float64}
    Q_is_diag::Bool
end


mutable struct Scaling_info_gpu
    l_org::CuVector{Float64}
    u_org::CuVector{Float64}
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
end


function qp_formulation(Q, c, A, AL, AU, l, u, c0=0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    keep_rows = setdiff(1:size(A, 1), del_row)
    println("Deleted ", length(del_row), " rows of A that are all zeros.")
    A = A[keep_rows, :]
    lp_AL = AL[keep_rows]
    lp_AU = AU[keep_rows]

    idxE = findall(lp_AL .== lp_AU)
    idxG = findall((lp_AL .> -Inf) .& (lp_AU .== Inf))
    idxL = findall((lp_AL .== -Inf) .& (lp_AU .< Inf))
    idxB = findall((lp_AL .> -Inf) .& (lp_AU .< Inf))
    idxB = setdiff(idxB, idxE)
    print("has ", length(idxG), " >= constraints\n")
    print("has ", length(idxL), " <= constraints\n")
    print("has ", length(idxB), " boxed constraints\n")
    # print("has ", length(idxL) + length(idxG), " inequality constraints\n")
    # print("has ", length(idxE), " equality constraints\n")
    # print("has ", length(lp_AL), " constraints in total\n")
    # @assert length(lp_AL) == length(idxE) + length(idxG) + length(idxL) + length(idxB)
    # print("has ", length(l), " variables\n")
    # print("has ", nnz(A), " non-zero elements in the constraint matrix\n")
    # print("has ", nnz(Q), " non-zero elements in the objective matrix\n")

    # check dimension of Q, c, A, l, u, AL, AU
    println("dimension of Q is ", size(Q))
    println("dimension of A is ", size(A))
    println("dimension of b is ", size(lp_AL))
    println("dimension of c is ", size(c))
    println("dimension of l is ", size(l))
    println("dimension of u is ", size(u))
    println("number of equalities = ", length(idxE))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(l) == size(Q, 1)
    @assert length(lp_AL) == length(lp_AU)
    @assert length(lp_AL) == size(A, 1)


    idxI = setdiff(1:length(lp_AL), idxE)

    AE = A[idxE, :]
    AI = A[idxI, :]

    ALE = lp_AL[idxE]
    AUE = lp_AU[idxE]
    ALI = lp_AL[idxI]
    AUI = lp_AU[idxI]

    A = vcat(AE, AI)
    AL = vcat(ALE, ALI)
    AU = vcat(AUE, AUI)

    m1 = length(idxE)

    # idxL is from m1+1 to m
    m = length(AL)
    idxI = (m1+1):m

    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, idxI, m1, c0, [], false)

    # Return the modified qp
    return standard_qp
end

function qp_formulation_noC(Q, c, A, AL, AU, l, u, c0=0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    keep_rows = setdiff(1:size(A, 1), del_row)
    println("Deleted ", length(del_row), " rows of A that are all zeros.")
    A = A[keep_rows, :]
    lp_AL = AL[keep_rows]
    lp_AU = AU[keep_rows]

    idxE = findall(lp_AL .== lp_AU)
    idxG = findall((lp_AL .> -Inf) .& (lp_AU .== Inf))
    idxL = findall((lp_AL .== -Inf) .& (lp_AU .< Inf))
    idxB = findall((lp_AL .> -Inf) .& (lp_AU .< Inf))
    idxB = setdiff(idxB, idxE)
    print("has ", length(idxG), " >= constraints\n")
    print("has ", length(idxL), " <= constraints\n")
    print("has ", length(idxB), " boxed constraints\n")

    # print("has ", length(idxL) + length(idxG), " inequality constraints\n")
    # print("has ", length(idxE), " equality constraints\n")
    # print("has ", length(lp_AL), " constraints in total\n")
    # @assert length(lp_AL) == length(idxE) + length(idxG) + length(idxL) + length(idxB)
    # print("has ", length(l), " variables\n")
    # print("has ", nnz(A), " non-zero elements in the constraint matrix\n")
    # print("has ", nnz(Q), " non-zero elements in the objective matrix\n")

    # check dimension of Q, c, A, l, u, AL, AU
    println("dimension of Q is ", size(Q))
    println("dimension of A is ", size(A))
    println("dimension of b is ", size(AL))
    println("dimension of c is ", size(c))
    println("dimension of l is ", size(l))
    println("dimension of u is ", size(u))
    println("number of equalities = ", length(idxE))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(l) == size(Q, 1)
    @assert length(lp_AL) == length(lp_AU)
    @assert length(lp_AL) == size(A, 1)


    idxI = setdiff(1:length(lp_AL), idxE)

    AE = A[idxE, :]
    AI = A[idxI, :]


    ALE = lp_AL[idxE]
    AUE = lp_AU[idxE]
    ALI = lp_AL[idxI]
    AUI = lp_AU[idxI]

    A = vcat(AE, AI, spdiagm(ones(length(l))))
    AL = vcat(ALE, ALI, l)
    AU = vcat(AUE, AUI, u)
    l .= -Inf * ones(length(l))
    u .= Inf * ones(length(u))

    m1 = length(idxE)

    # idxL is from m1+1 to m
    m = length(AL)
    idxI = (m1+1):m

    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, idxI, m1, c0)

    # Return the modified qp
    return standard_qp
end

function qp_formulation_Axb(Q, c, A, L, U, l, u, c0=0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((L .== -Inf) .& (U .== Inf)))

    keep_rows = setdiff(1:size(A, 1), del_row)
    println("Deleted ", length(del_row), " rows of A that are all zeros.")
    A = A[keep_rows, :]
    L = L[keep_rows]
    U = U[keep_rows]

    # Find indices for different types of constraints, i.e., equality, >=, <=, and boxed constraints
    idxE = findall(L .== U)
    idxG = findall((L .> -Inf) .& (U .== Inf))
    idxL = findall((L .== -Inf) .& (U .< Inf))
    idxB = findall((L .> -Inf) .& (U .< Inf))
    idxB = setdiff(idxB, idxE)

    AE = A[idxE, :]
    AG = A[idxG, :]
    AL = A[idxL, :]
    AB = A[idxB, :]

    num_E = length(idxE)
    num_G = length(idxG)
    num_L = length(idxL)
    num_B = length(idxB)
    println("has ", num_G, " >= constraints")
    println("has ", num_L, " <= constraints")
    println("has ", num_B, " boxed constraints")

    AE_new = hcat(AE, spzeros(num_E, num_G + num_L + num_B))
    AG_new = hcat(AG, -spdiagm(ones(num_G)), spzeros(num_G, num_L + num_B))
    AL_new = hcat(AL, spzeros(num_L, num_G), spdiagm(ones(num_L)), spzeros(num_L, num_B))
    AB_new = hcat(AB, spzeros(num_B, num_G + num_L), -spdiagm(ones(num_B)))

    A_new = vcat(AE_new, AG_new, AL_new, AB_new)

    b_new = vcat(L[idxE], L[idxG], U[idxL], zeros(num_B))

    c_new = vcat(c, zeros(num_G + num_L + num_B))

    l_new = vcat(l, zeros(num_G + num_L), L[idxB])

    u_new = vcat(u, Inf * ones(num_G + num_L), U[idxB])

    # check CPU resource
    Q_new = vcat(hcat(Q, spzeros(length(c), num_G + num_L + num_B)),
        spzeros(num_G + num_L + num_B, length(c) + num_G + num_L + num_B))

    # Check dimensions
    println("Original dimensions:")
    println("dimension of Q is ", size(Q))
    println("dimension of A is ", size(A))
    println("dimension of b is ", size(L))
    println("dimension of c is ", size(c))
    println("dimension of l is ", size(l))
    println("dimension of u is ", size(u))
    println("number of equalities = ", length(idxE))

    println("\nNew dimensions after adding slack variables:")
    println("dimension of Q_new is ", size(Q_new))
    println("dimension of A_new is ", size(A_new))
    println("dimension of b_new is ", size(b_new))
    println("dimension of c_new is ", size(c_new))
    println("dimension of l_new is ", size(l_new))
    println("dimension of u_new is ", size(u_new))
    println("number of slack variables = ", num_G + num_L + num_B)

    # Assertions for new dimensions
    @assert size(Q_new, 1) == size(Q_new, 2)
    @assert size(Q_new, 1) == length(c_new)
    @assert size(A_new, 2) == length(c_new)
    @assert length(l_new) == length(u_new)
    @assert length(l_new) == size(Q_new, 1)
    @assert length(b_new) == size(A_new, 1)

    # Create standard QP with new matrices
    standard_qp = QP_info_cpu(Q_new, c_new, A_new, A_new', b_new, copy(b_new), l_new, u_new,
        Int[], size(A_new, 1), c0)  # No inequality constraints after slack variable conversion

    return standard_qp
end

# function scaling!(qp, use_Ruiz_scaling, use_Pock_Chambolle_scaling, use_bc_scaling, use_l2_scaling)
#     m, n = size(qp.A)
#     row_norm = ones(m)
#     col_norm = ones(n)

#     # Preallocate temporary arrays
#     temp_norm_A_row = zeros(m)
#     temp_norm_A_col = zeros(n)
#     temp_norm_Q = zeros(n)
#     DR = spdiagm(temp_norm_A_row)
#     DC = spdiagm(temp_norm_A_col)

#     AL_nInf = copy(qp.AL)
#     AU_nInf = copy(qp.AU)
#     AL_nInf[qp.AL.==-Inf] .= 0.0
#     AU_nInf[qp.AU.==Inf] .= 0.0

#     scaling_info = Scaling_info(copy(qp.l), copy(qp.u), row_norm, col_norm, 1, 1, 1, 1, 1 + norm(max(abs.(AL_nInf), abs.(AU_nInf)), Inf), 1 + norm(qp.c, Inf))
#     println("norm_b_org: ", scaling_info.norm_b_org)
#     println("norm_c_org: ", scaling_info.norm_c_org)

#     if use_Ruiz_scaling
#         for _ in 1:10
#             # Ruiz_scaling!(qp, scaling_info)
#             temp_norm_Q .= maximum(abs, qp.Q, dims=1)[1, :]
#             # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

#             temp_norm_A_col .= maximum(abs, qp.A, dims=1)[1, :]
#             # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
#             temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))
#             temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

#             temp_norm_A_row .= sqrt.(maximum(abs, qp.A, dims=2)[:, 1])
#             temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

#             row_norm .*= temp_norm_A_row
#             col_norm .*= temp_norm_A_col


#             DR .= spdiagm(1.0 ./ temp_norm_A_row)
#             DC .= spdiagm(1.0 ./ temp_norm_A_col)
#             # temp_norm2 .= max.(temp_norm2,1.0)

#             qp.Q .= DC * qp.Q * DC
#             qp.c ./= temp_norm_A_col
#             qp.A .= DR * qp.A * DC

#             qp.AL ./= temp_norm_A_row
#             qp.AU ./= temp_norm_A_row
#             qp.l .*= temp_norm_A_col
#             qp.u .*= temp_norm_A_col
#         end
#     end

#     if use_bc_scaling
#         # bc_scaling!(qp, scaling_info)
#         AL_nInf = copy(qp.AL)
#         AU_nInf = copy(qp.AU)
#         AL_nInf[qp.AL.==-Inf] .= 0.0
#         AU_nInf[qp.AU.==Inf] .= 0.0
#         b_scale = 1 + norm(abs.(AL_nInf) + abs.(AU_nInf))
#         c_scale = 1 + norm(qp.c)
#         println("b_scale: ", b_scale)
#         println("c_scale: ", c_scale)
#         qp.Q .*= b_scale / c_scale
#         qp.AL ./= b_scale
#         qp.AU ./= b_scale
#         qp.c ./= c_scale
#         qp.l ./= b_scale
#         qp.u ./= b_scale
#         scaling_info.b_scale = b_scale
#         scaling_info.c_scale = c_scale
#     else
#         scaling_info.b_scale = 1.0
#         scaling_info.c_scale = 1.0
#     end

#     if use_l2_scaling
#         # compute the l2 norm of the rows and columns of A
#         temp_norm_Q .= sum(t -> t^2, qp.Q, dims=1)[1, :]
#         # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

#         temp_norm_A_col .= sum(t -> t^2, qp.A, dims=1)[1, :]
#         # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

#         ## choose one
#         temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
#         temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
#         # temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))

#         temp_norm_A_row .= sqrt.(sum(t -> t^2, qp.A, dims=2)[:, 1])
#         temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

#         row_norm .*= temp_norm_A_row
#         col_norm .*= temp_norm_A_col

#         DR .= spdiagm(1.0 ./ temp_norm_A_row)
#         DC .= spdiagm(1.0 ./ temp_norm_A_col)
#         # temp_norm2 .= max.(temp_norm2,1.0)

#         qp.Q .= DC * qp.Q * DC
#         qp.c ./= temp_norm_A_col
#         qp.A .= DR * qp.A * DC

#         qp.AL ./= temp_norm_A_row
#         qp.AU ./= temp_norm_A_row
#         qp.l .*= temp_norm_A_col
#         qp.u .*= temp_norm_A_col
#     end

#     if use_Pock_Chambolle_scaling
#         # Pock_Chambolle_scaling!(qp, scaling_info)
#         temp_norm_Q .= sum(abs, qp.Q, dims=1)[1, :]
#         # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

#         temp_norm_A_col .= sum(abs, qp.A, dims=1)[1, :]
#         # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

#         ## choose one
#         temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
#         temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
#         # temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))

#         temp_norm_A_row .= sqrt.(sum(abs, qp.A, dims=2)[:, 1])
#         temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

#         row_norm .*= temp_norm_A_row
#         col_norm .*= temp_norm_A_col


#         DR .= spdiagm(1.0 ./ temp_norm_A_row)
#         DC .= spdiagm(1.0 ./ temp_norm_A_col)
#         # temp_norm2 .= max.(temp_norm2,1.0)

#         qp.Q .= DC * qp.Q * DC
#         qp.c ./= temp_norm_A_col
#         qp.A .= DR * qp.A * DC

#         qp.AL ./= temp_norm_A_row
#         qp.AU ./= temp_norm_A_row
#         qp.l .*= temp_norm_A_col
#         qp.u .*= temp_norm_A_col
#     end

#     AL_nInf = copy(qp.AL)
#     AU_nInf = copy(qp.AU)
#     AL_nInf[qp.AL.==-Inf] .= 0.0
#     AU_nInf[qp.AU.==Inf] .= 0.0
#     scaling_info.norm_b = norm(max(abs.(AL_nInf), abs.(AU_nInf)))
#     scaling_info.norm_c = norm(qp.c)
#     qp.AT = qp.A'
#     # eliminate the numerical error cause the asymmetry of Q
#     qp.Q = (qp.Q + transpose(qp.Q)) / 2
#     scaling_info.row_norm = row_norm
#     scaling_info.col_norm = col_norm
#     return scaling_info
# end



function scaling!(qp, use_Ruiz_scaling, use_bc_scaling, use_l2_scaling, use_Pock_Chambolle_scaling, x_path="", y_path="")
    m, n = size(qp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    # Preallocate temporary arrays
    temp_norm_A_row = zeros(m)
    temp_norm_A_col = zeros(n)
    temp_norm_Q = zeros(n)
    DR = spdiagm(temp_norm_A_row)
    DC = spdiagm(temp_norm_A_col)

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info = Scaling_info(copy(qp.l), copy(qp.u), row_norm, col_norm, 1, 1, 1, 1, norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), norm(qp.c, Inf))
    println("norm_b_org: ", scaling_info.norm_b_org)
    println("norm_c_org: ", scaling_info.norm_c_org)


    if use_Ruiz_scaling
        for _ in 1:10
            # Ruiz_scaling!(qp, scaling_info)
            temp_norm_Q .= maximum(abs, qp.Q, dims=1)[1, :]
            # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

            temp_norm_A_col .= maximum(abs, qp.A, dims=1)[1, :]
            # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
            temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))
            temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

            temp_norm_A_row .= sqrt.(maximum(abs, qp.A, dims=2)[:, 1])
            temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

            row_norm .*= temp_norm_A_row
            col_norm .*= temp_norm_A_col


            DR .= spdiagm(1.0 ./ temp_norm_A_row)
            DC .= spdiagm(1.0 ./ temp_norm_A_col)
            # temp_norm2 .= max.(temp_norm2,1.0)

            qp.Q .= DC * qp.Q * DC
            qp.c ./= temp_norm_A_col
            qp.A .= DR * qp.A * DC
            # qp.x0 .*= temp_norm_A_col
            # qp.y0 .*= temp_norm_A_row

            qp.AL ./= temp_norm_A_row
            qp.AU ./= temp_norm_A_row
            qp.l .*= temp_norm_A_col
            qp.u .*= temp_norm_A_col
        end
    end



    if use_l2_scaling
        # compute the l2 norm of the rows and columns of A
        temp_norm_Q .= sum(t -> t^2, qp.Q, dims=1)[1, :]
        # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

        temp_norm_A_col .= sum(t -> t^2, qp.A, dims=1)[1, :]
        # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

        ## choose one
        temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
        temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
        # temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))

        temp_norm_A_row .= sqrt.(sum(t -> t^2, qp.A, dims=2)[:, 1])
        temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col

        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)
        # temp_norm2 .= max.(temp_norm2,1.0)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC
        # qp.x0 .*= temp_norm_A_col
        # qp.y0 .*= temp_norm_A_row

        qp.AL ./= temp_norm_A_row
        qp.AU ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col
        qp.u .*= temp_norm_A_col
    end

    if use_Pock_Chambolle_scaling
        # Pock_Chambolle_scaling!(qp, scaling_info)
        temp_norm_Q .= sum(abs, qp.Q, dims=1)[1, :]
        # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

        temp_norm_A_col .= sum(abs, qp.A, dims=1)[1, :]
        # temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

        ## choose one
        temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
        temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0
        # temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))

        temp_norm_A_row .= sqrt.(sum(abs, qp.A, dims=2)[:, 1])
        temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col


        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)
        # temp_norm2 .= max.(temp_norm2,1.0)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC
        # qp.x0 .*= temp_norm_A_col
        # qp.y0 .*= temp_norm_A_row

        qp.AL ./= temp_norm_A_row
        qp.AU ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col
        qp.u .*= temp_norm_A_col
    end

    if use_bc_scaling
        # bc_scaling!(qp, scaling_info)
        b_scale = 1 + norm(min.(qp.AL, qp.AU))
        c_scale = 1 + norm(qp.c)
        # b_scale = norm(qp.x0)+1
        # c_scale = norm(qp.y0)+1
        # qp.x0 ./= b_scale
        # qp.y0 ./= c_scale
        println("b_scale: ", b_scale)
        println("c_scale: ", c_scale)
        qp.Q .*= b_scale / c_scale
        qp.AL ./= b_scale
        qp.AU ./= b_scale
        qp.c ./= c_scale
        qp.l ./= b_scale
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    temp_norm_Q .= sum(abs, qp.Q, dims=1)[1, :]
    # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

    # the diagonal matrix of the Q matrix
    diag_Q = diag(qp.Q)
    Q_is_diag = (sum(temp_norm_Q .== diag_Q) == length(temp_norm_Q))
    qp.Q_is_diag = Q_is_diag
    qp.diag_Q = diag_Q


    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(qp.c)
    qp.AT = qp.A'
    # eliminate the numerical error cause the asymmetry of Q
    qp.Q = (qp.Q + transpose(qp.Q)) / 2
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    return scaling_info
end


function power_iteration_gpu2(A, AT, max_iterations=50000, tolerance=1e-4)
    seed = 1
    # Random.seed!(seed)

    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    # z = CuVector(rand(m)) .+ 1e-8
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1
    for i in 1:max_iterations
        q .= z
        q ./= LinearAlgebra.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = LinearAlgebra.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        if LinearAlgebra.norm(q) < tolerance
            # lambda_max = dot(b_new, A * (AT * b_new))
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", LinearAlgebra.norm(q))
    return lambda_max
end



function power_iteration_A_gpu(A, AT, max_iterations=5000, tolerance=1e-4)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= LinearAlgebra.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = LinearAlgebra.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = LinearAlgebra.norm(q) / (LinearAlgebra.norm(z) + lambda_max)
        if error < tolerance
            # lambda_max = dot(b_new, A * (AT * b_new))
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

function power_iteration_Q_gpu(Q, max_iterations=5000, tolerance=1e-4)
    seed = 1
    n, n = size(Q)
    z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    # z = CuVector(rand(m)) .+ 1e-8
    q = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= LinearAlgebra.norm(q)
        # mul!(ATq, AT, q)
        CUDA.CUSPARSE.mv!('N', 1, Q, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = LinearAlgebra.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = LinearAlgebra.norm(q) / (LinearAlgebra.norm(z) + lambda_max)

        if error < tolerance
            # lambda_max = dot(b_new, A * (AT * b_new))
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

function compute_weighted_norm_gpu(ws::HPRQP_workspace_gpu, Q_is_diag::Bool)
    if Q_is_diag
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.x_hat, ws.dx, ws.n)
        axpby_gpu!(1.0, ws.y_bar, -1.0, ws.y_hat, ws.dy, ws.m)
        CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.dx, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        dot_prod = 2 * LinearAlgebra.dot(ws.Ax, ws.dy)
        dy_squarenorm = LinearAlgebra.dot(ws.dy, ws.dy)
        dx_squarenorm = LinearAlgebra.dot(ws.dx, ws.dx)
        # println("dx_squarenorm: ", dx_squarenorm)
        # println("dy_squarenorm: ", dy_squarenorm)
        weighted_norm = (ws.sigma * ws.lambda_max_A) * dx_squarenorm + (dy_squarenorm) / ws.sigma - dot_prod
    else
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.x_hat, ws.dx, ws.n)
        axpby_gpu!(1.0, ws.y_bar, -1.0, ws.y_hat, ws.dy, ws.m)
        CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.dx, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dx, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        xQx = LinearAlgebra.dot(ws.dx, ws.Qx)
        dot_prod = 2 * LinearAlgebra.dot(ws.Ax, ws.dy)
        dy_squarenorm = LinearAlgebra.dot(ws.dy, ws.dy)
        dx_squarenorm = LinearAlgebra.dot(ws.dx, ws.dx)
        weighted_norm = (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A) * dx_squarenorm - xQx + (dy_squarenorm) / ws.sigma + dot_prod
    end
    # println("weighted_norm: ", weighted_norm)
    if weighted_norm < 0
        error("The estimated value of lambda_max is too small! Please increase params.lambda_factor!")
        # ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
        # weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    return weighted_norm
end

function compute_weighted_norm_cpu(ws::HPRQP_workspace_cpu)
    ws.dx .= ws.x .- ws.x_hat
    ws.dy .= ws.y .- ws.y_hat
    mul!(ws.Ax, ws.A, ws.dx)
    mul!(ws.Qx, ws.Q, ws.dx)
    xQx = dot(ws.dx, ws.Qx)
    dot_prod = 2 * dot(ws.Ax, ws.dy)
    dy_squarenorm = norm(ws.dy)^2
    dx_squarenorm = norm(ws.dx)^2

    weighted_norm = (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A) * dx_squarenorm - xQx + (dy_squarenorm) / ws.sigma + dot_prod
    # println("weighted_norm: ", weighted_norm)
    if weighted_norm < 0
        error("The estimated value of lambda_max is too small! Please increase params.lambda_factor!")
        # ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
        # weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    return weighted_norm
end

function axpby_kernel!(a, x, b, y, z, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a, x, b, y, z, n)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

function combined_kernel_x!(x, x_bar, Qx, x_hat, l, u, fact, ATy, c, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        x_bar[i] = x[i] + fact * (-Qx[i] + ATy[i] - c[i])
        x_bar[i] = (x_bar[i] < l[i]) ? l[i] : ((x_bar[i] > u[i]) ? u[i] : x_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
    end
    return
end

function update_x_gpu!(ws::HPRQP_workspace_gpu)
    # mul!(ws.ATy, ws.AT, ws.y)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_hat, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x!(ws.x, ws.x_bar, ws.Qx, ws.x_hat, ws.l, ws.u, 1 / (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A), ws.ATy, ws.c, ws.n)
end

function combined_kernel_xz!(x, x_bar, z_bar, Qx, x_hat, last_x, l, u, fact, fact2, ATy, c, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        z_bar[i] = x[i] + fact * (-Qx[i] + ATy[i] - c[i])
        x_bar[i] = (z_bar[i] < l[i]) ? l[i] : ((z_bar[i] > u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = fact2 * (x_bar[i] - z_bar[i])
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

function update_xz_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_hat, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_xz!(ws.x, ws.x_bar, ws.z_bar, ws.Qx, ws.x_hat, ws.last_x, ws.l, ws.u, 1 / (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A), (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A), ws.ATy, ws.c, Halpern_fact1, Halpern_fact2, ws.n)
end





function combined_kernel_xz_Q_diag!(x, x_bar, z_bar, Qx, x_hat, last_x, l, u, fact1, fact2, ATy, c, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        z_bar[i] = fact2[i] * x[i] + fact1[i] * (ATy[i] - c[i])
        x_bar[i] = (z_bar[i] < l[i]) ? l[i] : ((z_bar[i] > u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / fact1[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

function update_xz_gpu_Q_diag!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_hat, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_xz_Q_diag!(ws.x, ws.x_bar, ws.z_bar, ws.Qx, ws.x_hat, ws.last_x, ws.l, ws.u, ws.fact1, ws.fact2, ws.ATy, ws.c, Halpern_fact1, Halpern_fact2, ws.n)
end
@inline function cust_combined_kernel_xz!(rowPtrAT, colValAT, nzValAT, rowPtrQ, colValQ, nzValQ, y_hat, x, x_bar, z_bar, Qx, x_hat, last_x, l, u, fact, fact2, ATy, c, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_hat[colValAT[k]]
        end
        ATy[i] = sAT

        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * x[colValQ[k]]
        end
        Qx[i] = qr1

        z_bar[i] = x[i] + fact * (-Qx[i] + ATy[i] - c[i])
        x_bar[i] = (z_bar[i] < l[i]) ? l[i] : ((z_bar[i] > u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = fact2 * (x_bar[i] - z_bar[i])
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

function cust_update_xz_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_hat, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_combined_kernel_xz!(ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
        ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.y_hat, ws.x, ws.x_bar, ws.z_bar, ws.Qx, ws.x_hat, ws.last_x, ws.l, ws.u, 1 / (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A), (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A), ws.ATy, ws.c, Halpern_fact1, Halpern_fact2, ws.n)
end



@inline function cust_combined_kernel_xz_Q_diag!(rowPtrAT, colValAT, nzValAT, rowPtrQ, colValQ, nzValQ, y_hat, x, x_bar, z_bar, Qx, x_hat, last_x, l, u, fact1, fact2, ATy, c, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_hat[colValAT[k]]
        end
        ATy[i] = sAT

        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * x[colValQ[k]]
        end
        Qx[i] = qr1

        z_bar[i] = fact2[i] * x[i] + fact1[i] * (ATy[i] - c[i])
        x_bar[i] = (z_bar[i] < l[i]) ? l[i] : ((z_bar[i] > u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / fact1[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]



    end
    return
end

function cust_update_xz_gpu_Q_diag!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_hat, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_combined_kernel_xz_Q_diag!(ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
        ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.y_hat, ws.x, ws.x_bar, ws.z_bar, ws.Qx, ws.x_hat, ws.last_x, ws.l, ws.u, ws.fact1, ws.fact2, ws.ATy, ws.c, Halpern_fact1, Halpern_fact2, ws.n)
end


function update_x_cpu!(ws::HPRQP_workspace_cpu)
    mul!(ws.ATy, ws.AT, ws.y_hat)
    mul!(ws.Qx, ws.Q, ws.x)
    x = ws.x
    ATy = ws.ATy
    Qx = ws.Qx
    l = ws.l
    u = ws.u
    fact = 1 / (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A)
    x_bar = ws.x_bar
    x_hat = ws.x_hat
    c = ws.c
    @inbounds @simd for i in eachindex(x)
        x_bar[i] = x[i] + fact * (-Qx[i] + ATy[i] - c[i])
        x_bar[i] = (x_bar[i] < l[i]) ? l[i] : ((x_bar[i] > u[i]) ? u[i] : x_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
    end
end

function update_xz_cpu!(ws::HPRQP_workspace_cpu)
    mul!(ws.ATy, ws.AT, ws.y_hat)
    mul!(ws.Qx, ws.Q, ws.x)
    x = ws.x
    ATy = ws.ATy
    Qx = ws.Qx
    l = ws.l
    u = ws.u
    fact = 1 / (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A)
    fact2 = (ws.lambda_max_Q + ws.sigma * ws.lambda_max_A)
    x_bar = ws.x_bar
    x_hat = ws.x_hat
    c = ws.c
    z_bar = ws.z_bar
    @inbounds @simd for i in eachindex(x)
        z_bar[i] = x[i] + fact * (-Qx[i] + ATy[i] - c[i])
        x_bar[i] = (z_bar[i] < l[i]) ? l[i] : ((z_bar[i] > u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = fact2 * (x_bar[i] - z_bar[i])
    end
end

function update_y_kernel!(y_bar, y, y_hat, AL, AU, Ax, sigma, inv_sigma, m)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        y_bar[i] = Ax[i] - inv_sigma * y[i]
        y_bar[i] = (y_bar[i] < AL[i]) ? (AL[i] - y_bar[i]) : ((y_bar[i] > AU[i]) ? (AU[i] - y_bar[i]) : 0.0)
        y_bar[i] = sigma * y_bar[i]
        y_hat[i] = 2 * y_bar[i] - y[i]
    end
    return
end

function update_y_gpu!(ws::HPRQP_workspace_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.y_bar, ws.y, ws.y_hat, ws.AL, ws.AU, ws.Ax, ws.sigma, 1 / ws.sigma, ws.m)
end

function update_ys_kernel!(y_bar, y, y_hat, last_y, s_bar, AL, AU, Ax, sigma, inv_sigma, Halpern_fact1, Halpern_fact2, m)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        y_bar[i] = Ax[i] - inv_sigma * y[i]
        s_bar[i] = (y_bar[i] < AL[i]) ? AL[i] : ((y_bar[i] > AU[i]) ? AU[i] : y_bar[i])
        y_bar[i] = sigma * (s_bar[i] - y_bar[i])
        y_hat[i] = 2 * y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * y_hat[i]
    end
    return
end

function update_ys_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_ys_kernel!(ws.y_bar, ws.y, ws.y_hat, ws.last_y, ws.s_bar, ws.AL, ws.AU, ws.Ax, ws.sigma, 1 / ws.sigma, Halpern_fact1, Halpern_fact2, ws.m)
end


# @inline function cust_update_ys_kernel!(rowPtrA, colValA, nzValA, x, y_bar, y, y_hat, last_y, s_bar, AL, AU, Ax, sigma, inv_sigma, Halpern_fact1, Halpern_fact2, m)
#     @inbounds begin
#         # compute global thread index (1-based)
#         i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#         if i <= m
#             # load row pointers
#             start = rowPtrA[i]
#             stop = rowPtrA[i+1] - 1
#             # accumulate A * x
#             Ai = 0
#             for k in start:(stop)
#                 Ai += nzValA[k] * x[colValA[k]]
#             end
#             # compute update
#             yi = y[i]
#             yb = Ai - inv_sigma * yi
#             # clamp between AL and AU
#             sbi = min(max(yb, AL[i]), AU[i])
#             yb = sigma * (sbi - yb)
#             # write back
#             yh = 2 * yb - yi
#             s_bar[i] = sbi

#             y_bar[i] = yb

#             y_hat[i] = yh
#             # final Halpern step
#             y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * yh
#         end
#     end
#     return
# end


@inline function cust_update_ys_kernel!(rowPtrA, colValA, nzValA, x, y_bar, y, y_hat, last_y, s_bar, AL, AU, Ax, sigma, inv_sigma, Halpern_fact1, Halpern_fact2, m)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        startA = rowPtrA[i]
        stopA = rowPtrA[i+1] - 1
        Ai = 0.0
        @inbounds for k in startA:stopA
            Ai += nzValA[k] * x[colValA[k]]
        end
        y_bar[i] = Ai - inv_sigma * y[i]
        s_bar[i] = (y_bar[i] < AL[i]) ? AL[i] : ((y_bar[i] > AU[i]) ? AU[i] : y_bar[i])
        y_bar[i] = sigma * (s_bar[i] - y_bar[i])
        y_hat[i] = 2 * y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * y_hat[i]
    end
    return
end

function cust_update_ys_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_ys_kernel!(ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
        ws.x, ws.y_bar, ws.y, ws.y_hat, ws.last_y, ws.s_bar, ws.AL, ws.AU, ws.Ax, ws.sigma, 1 / ws.sigma, Halpern_fact1, Halpern_fact2, ws.m)
end


function update_y_cpu!(ws::HPRQP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.x)
    y = ws.y
    Ax = ws.Ax
    AL = ws.AL
    AU = ws.AU
    inv_sigma = 1 / ws.sigma
    sigma = ws.sigma
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    @inbounds @simd for i in eachindex(y)
        y_bar[i] = Ax[i] - inv_sigma * y[i]
        y_bar[i] = (y_bar[i] < AL[i]) ? (AL[i] - y_bar[i]) : ((y_bar[i] > AU[i]) ? (AU[i] - y_bar[i]) : 0.0)
        y_bar[i] = sigma * y_bar[i]
        y_hat[i] = 2 * y_bar[i] - y[i]
    end
end


function update_ys_cpu!(ws::HPRQP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.x)
    y = ws.y
    Ax = ws.Ax
    AL = ws.AL
    AU = ws.AU
    inv_sigma = 1 / ws.sigma
    sigma = ws.sigma
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    s_bar = ws.s_bar
    @inbounds @simd for i in eachindex(y)
        y_bar[i] = Ax[i] - inv_sigma * y[i]
        s_bar[i] = (y_bar[i] < AL[i]) ? AL[i] : ((y_bar[i] > AU[i]) ? AU[i] : y_bar[i])
        y_bar[i] = sigma * (s_bar[i] - y_bar[i])
        # y_bar[i] = y[i] + sigma * (AU[i] - Ax[i])
        y_hat[i] = 2 * y_bar[i] - y[i]
    end
end

function compute_Rd_kernel!(ATy, z, c, Qx, Rd, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qx[i] + c[i] - ATy[i] - z[i]
    end
    return
end

function compute_Rd_kernel2!(ATy, x, c, Qx, Rd, l, u, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = x[i] - min(u[i], max(l[i], x[i] - Qx[i] + ATy[i] - c[i]))
    end
    return
end


function compute_Rd_gpu!(ws)
    # mul!(ws.ATy, ws.AT, ws.y_bar) 
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATy, ws.z_bar, ws.c, ws.Qx, ws.Rd, ws.n)
end

function compute_Rd_gpu2!(ws)
    # mul!(ws.ATy, ws.AT, ws.y_bar) 
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel2!(ws.ATy, ws.x_bar, ws.c, ws.Qx, ws.Rd, ws.l, ws.u, ws.n)
end


function compute_err_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    m1::Int, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= m1
        Rp[i] = AU[i] - Ax[i]
    elseif i <= m
        Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : ((Ax[i] > AU[i]) ? (AU[i] - Ax[i]) : 0.0)
    end
    return
end

function compute_err_Rp_gpu!(ws)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_err_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, ws.m1, ws.m)
end

function compute_err_Rp_cpu!(ws)
    mul!(ws.Ax, ws.A, ws.x_bar)
    m1 = ws.m1
    m = ws.m
    Ax = ws.Ax
    Rp = ws.Rp
    AL = ws.AL
    AU = ws.AU
    @inbounds @simd for i in eachindex(Rp)
        # Threads.@threads for i in eachindex(Rp)
        if i <= m1
            Rp[i] = AU[i] - Ax[i]
        elseif i <= m
            Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : ((Ax[i] > AU[i]) ? (AU[i] - Ax[i]) : 0.0)
        end
    end
end

function compute_err_Rp_cpu_original!(ws, qp_org)
    mul!(ws.Ax, qp_org.A, ws.dx)
    m1 = ws.m1
    m = ws.m
    Ax = ws.Ax
    Rp = ws.Rp
    AL = qp_org.AL
    AU = qp_org.AU
    @inbounds @simd for i in eachindex(Rp)
        # Threads.@threads for i in eachindex(Rp)
        if i <= m1
            Rp[i] = AU[i] - Ax[i]
        elseif i <= m
            Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : ((Ax[i] > AU[i]) ? (AU[i] - Ax[i]) : 0.0)
        end
    end
end

function Halpern_update_gpu!(ws::HPRQP_workspace_gpu, restart_info::HPRQP_restart)
    fact1 = 1.0 / (restart_info.inner + 2.0)
    fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
    axpby_gpu!(fact1, ws.last_x, fact2, ws.x_hat, ws.x, ws.n)
    axpby_gpu!(fact1, ws.last_y, fact2, ws.y_hat, ws.y, ws.m)
end

function Halpern_update_cpu!(ws::HPRQP_workspace_cpu, restart_info::HPRQP_restart)
    fact1 = 1.0 / (restart_info.inner + 2.0)
    fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
    ws.x .= fact1 .* ws.last_x .+ fact2 .* ws.x_hat
    ws.y .= fact1 .* ws.last_y .+ fact2 .* ws.y_hat
end


function compute_residuals_cpu(ws::Union{HPRQP_workspace_cpu,HPRQP_workspace_gpu},
    qp::Union{QP_info_cpu,QP_info_gpu},
    sc::Union{Scaling_info,Scaling_info_gpu},
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    print_yes::Bool,
    check_iter::Int,
    iter::Int,
)
    res.is_updated = true


    ### obj
    # res.primal_obj = LinearAlgebra.dot(ws.c, ws.x_bar)
    mul!(ws.Qx, ws.Q, ws.x_bar)
    mul!(ws.ATy, ws.AT, ws.y_bar)
    # ws.z_bar .= ws.Qx .- ws.ATy .+ ws.c
    # proj_z_cpu!(ws)

    # @assert norm(ws.z_bar) < 1e-10

    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (dot(ws.c, ws.x_bar) + 0.5 * dot(ws.x_bar, ws.Qx))
    +qp.obj_constant

    res.dual_obj_bar =
        sc.b_scale * sc.c_scale *
        (-0.5 * dot(ws.x_bar, ws.Qx)
         + dot(ws.y_bar, ws.s_bar)
         #  + min.(ws.y_bar, 0)' * ws.AU + max.(ws.y_bar, 0)' * ws.AL
         + dot(ws.z_bar, ws.x_bar))
    #  + min.(ws.z_bar, 0)' * ws.u + max.(ws.z_bar, 0)' * ws.l)
    +qp.obj_constant
    # res.dual_obj_bar = sc.b_scale * sc.c_scale * res.dual_obj + qp.obj_constant

    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))



    ### Rd
    ws.Rd .= -ws.Qx .+ ws.ATy .- ws.c .+ ws.z_bar
    res.err_Rd_bar = norm(ws.Rd)

    ws.Rd .*= sc.col_norm
    ws.ATy .*= sc.col_norm
    ws.Qx .*= sc.col_norm
    res.err_Rd_org_bar = sc.c_scale * norm(ws.Rd, Inf) / maximum([sc.norm_c_org, 1 + norm(ws.ATy, Inf), 1 + norm(ws.Qx, Inf)])
    # res.err_Rd_org_bar = sc.c_scale * norm(ws.Rd, Inf)

    ### Rp
    if ws.m > 0
        compute_err_Rp_cpu!(ws)
        res.err_Rp_bar = norm(ws.Rp)
        ws.Rp .*= sc.row_norm
        ws.Ax .*= sc.row_norm
        res.err_Rp_org_bar = sc.b_scale * norm(ws.Rp, Inf) / maximum([sc.norm_b_org, 1 + norm(ws.Ax, Inf)])
        # res.err_Rp_org_bar = sc.b_scale * norm(ws.Rp, Inf)
    end

    if iter == 0
        res.err_Rp_org_bar += norm(ws.x_bar - max.(min.(ws.x_bar, ws.u), ws.l), Inf)
    end

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
    # res.KKTx_org_bar = max(res.err_Rp_org_bar, res.err_K_org_bar, res.err_Rd_org_bar, res.err_z_org_bar, res.err_Comp_org_bar) 

end



function update_Q_factors_kernel!(
    fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64}, diag_Q::CuDeviceVector{Float64},
    sigma::Float64,
    lambda_max_A::Float64,
    N::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        t = 1.0 / (sigma * lambda_max_A + diag_Q[i])
        fact1[i] = t
        fact2[i] = t * sigma * lambda_max_A
    end
    return
end


function update_Q_factors_gpu!(
    fact1::CuVector{Float64},
    fact2::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64,
    lambda_max_A::Float64,
)
    N = length(diag_Q)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks update_Q_factors_kernel!(fact1, fact2, diag_Q, sigma, lambda_max_A, N)
    return
end


function proj_z_kernel!(z::CuDeviceVector{Float64},
    idx_boxed::CuDeviceVector{Bool},
    idx_unbounded::CuDeviceVector{Bool},
    idx_uinf::CuDeviceVector{Bool},
    idx_linf::CuDeviceVector{Bool},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        if idx_boxed[i]
            z[i] = z[i]
        elseif idx_unbounded[i]
            z[i] = 0.0
        elseif idx_uinf[i]
            z[i] = max(0.0, z[i])
        elseif idx_linf[i]
            z[i] = min(0.0, z[i])
        end
    end
    return
end

function proj_z_gpu!(ws::HPRQP_workspace_gpu)
    # projK_mPolar = (x) -> (y = CuVector(zeros(length(x))); y[idx_boxed] .= x[idx_boxed]; y[idx_unbounded] .= 0.0; y[idx_uinf] .= max.(0, x[idx_uinf]); y[idx_linf] .= min.(0, x[idx_linf]); y)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) proj_z_kernel!(ws.z_bar, ws.idx_boxed, ws.idx_unbounded, ws.idx_uinf, ws.idx_linf, ws.n)
end

function proj_z_cpu!(ws::HPRQP_workspace_cpu)
    # projK_mPolar = (x) -> (y = CuVector(zeros(length(x))); y[idx_boxed] .= x[idx_boxed]; y[idx_unbounded] .= 0.0; y[idx_uinf] .= max.(0, x[idx_uinf]); y[idx_linf] .= min.(0, x[idx_linf]); y)
    z = ws.z_bar
    idx_boxed = ws.idx_boxed
    idx_unbounded = ws.idx_unbounded
    idx_uinf = ws.idx_uinf
    idx_linf = ws.idx_linf
    @inbounds @simd for i in eachindex(z)
        if idx_boxed[i]
            z[i] = z[i]
        elseif idx_unbounded[i]
            z[i] = 0.0
        elseif idx_uinf[i]
            z[i] = max(0.0, z[i])
        elseif idx_linf[i]
            z[i] = min(0.0, z[i])
        end
    end
end

function compute_residuals_gpu(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    print_yes::Bool,
    check_iter::Int,
    iter::Int,
)
    res.is_updated = true

    ### obj
    # res.primal_obj = LinearAlgebra.dot(ws.c, ws.x_bar)
    CUSPARSE.mv!('N', 1, ws.Q, ws.x_bar, 0, ws.Qx, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.z_bar .= ws.Qx .- ws.ATy .+ ws.c
    # proj_z_gpu!(ws)

    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (LinearAlgebra.dot(ws.c, ws.x_bar) + 0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx))
    +qp.obj_constant

    res.dual_obj_bar =
        sc.b_scale * sc.c_scale *
        (-0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx)
         #  + LinearAlgebra.dot(min.(ws.y_bar, 0), ws.AU) + LinearAlgebra.dot(max.(ws.y_bar, 0), ws.AL)
         + LinearAlgebra.dot(ws.y_bar, ws.s_bar)
         + LinearAlgebra.dot(ws.z_bar, ws.x_bar))
    #  + LinearAlgebra.dot(min.(ws.z_bar, 0), ws.u) + LinearAlgebra.dot(max.(ws.z_bar, 0), ws.l))
    +qp.obj_constant
    # res.dual_obj_bar = sc.b_scale * sc.c_scale * res.dual_obj + qp.obj_constant

    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))


    ### Rd
    compute_Rd_gpu!(ws)


    res.err_Rd_bar = LinearAlgebra.norm(ws.Rd)
    # println("Rd norm = ", res.err_Rd_bar)
    ws.Rd .*= sc.col_norm
    ws.ATy .*= sc.col_norm
    ws.Qx .*= sc.col_norm
    res.err_Rd_org_bar = sc.c_scale * LinearAlgebra.norm(ws.Rd, Inf) / maximum([sc.norm_c_org, 1 + LinearAlgebra.norm(ws.ATy, Inf), 1 + LinearAlgebra.norm(ws.Qx, Inf)])

    ### Rp
    if ws.m > 0
        compute_err_Rp_gpu!(ws)
        res.err_Rp_bar = LinearAlgebra.norm(ws.Rp)
        ws.Rp .*= sc.row_norm
        ws.Ax .*= sc.row_norm
        res.err_Rp_org_bar = sc.b_scale * LinearAlgebra.norm(ws.Rp, Inf) / maximum([sc.norm_b_org, 1 + LinearAlgebra.norm(ws.Ax, Inf)])
    end

    if iter == 0
        res.err_Rp_org_bar += LinearAlgebra.norm(ws.x_bar - max.(min.(ws.x_bar, ws.u), ws.l), Inf)
    end

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
    # res.KKTx_org_bar = max(res.err_Rp_org_bar, res.err_K_org_bar, res.err_Rd_org_bar, res.err_z_org_bar, res.err_Comp_org_bar) 

end


function sigma_update_step(iter)
    if iter < 100
        res = 3
    elseif iter < 1000
        res = 100
    elseif iter < 10000
        res = 500
    else
        res = 2000
    end
    return res
end

function update_sigma(params::HPRQP_parameters,
    restart_info::Union{HPRQP_restart,HPRQP_restart},
    iter::Int,
    sc::Union{Scaling_info,Scaling_info_gpu},
    ws::Union{HPRQP_workspace_cpu,HPRQP_workspace_gpu},
    residuals::HPRQP_residuals,
    Q_is_diag::Bool
)
    if ~params.sigma_fixed && ((restart_info.restart_flag >= 1) || (restart_info.restart_type == 0 && (iter < 5e5 && rem(iter, sigma_update_step(iter)) == 1)))
        if params.sigma_method == 0
            primal_move = norm(ws.x_bar .- ws.last_x)
            dual_move = norm(ws.y_bar .- ws.last_y)
            if primal_move > 1e-16 && dual_move > 1e-16
                ratio = primal_move / dual_move
                #sigma = ratio
                ws.sigma = exp(0.5 * log(ratio) + 0.5 * log(ws.sigma))
            end
        elseif params.sigma_method == 1
            if params.useGPU
                axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
                if ws.m > 0
                    axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
                end

                dual_move = LinearAlgebra.norm(ws.dy)
                primal_move = LinearAlgebra.norm(ws.dx) * sqrt(ws.lambda_max_A)

            else
                ws.dx .= ws.x_bar .- ws.last_x
                if ws.m > 0
                    ws.dy .= ws.y_bar .- ws.last_y
                end

                dual_move = norm(ws.dy)
                primal_move = norm(ws.dx) * sqrt(ws.lambda_max_A)
            end
            # ratio_infeas = residuals.err_Rd_bar / residuals.err_Rp_bar
            # ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar

            # print(" err_Rp_bar = ", residuals.err_Rp_bar, " err_Rd_bar = ", residuals.err_Rd_bar, "\n")
            # print(" dual_infeas/primal_infeas = ", ratio_infeas, "\n")

            if restart_info.restart_flag >= 1 && restart_info.restart_flag <= 3
                if primal_move > 1e-14 && dual_move > 1e-14
                    sigma_new = dual_move / primal_move
                else
                    sigma_new = ws.sigma
                end
                # sigma_new = 10 * max(dual_move, 1e-12) / max(primal_move, 1e-12)

                # if primal_move > 1e-14 && dual_move > 1e-14
                #     sigma_new = max(dual_move, 1e-12) / max(primal_move, 1e-12)
                # else
                #     sigma_new = ws.sigma
                # end

                fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
                print("sigma old: ", ws.sigma)

                ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))

            end
        end
        println(" sigma_new = ", ws.sigma, " primal_move = ", primal_move, " dual_move = ", dual_move)
    end

end

function check_restart(restart_info::Union{HPRQP_restart,HPRQP_restart},
    residuals::HPRQP_residuals,
    iter::Int,
    check_iter::Int,
)

    restart_info.restart_flag = 0
    if (restart_info.restart_type > 0)
        if restart_info.restart_type == 1
            # fixed step restart
            if restart_info.inner == restart_info.step
                restart_info.current_gap = residuals.KKTx_bar
                restart_info.restart_flag = 1
            end
        else
            # adaptive restart
            if restart_info.first_restart
                if iter == check_iter
                    # restart_info.current_gap = restart_info.weighted_norm
                    restart_info.first_restart = false
                    restart_info.restart_flag = 1
                    restart_info.weighted_norm = restart_info.current_gap
                    println("first restart, current gap = ", restart_info.current_gap, "\n")

                end
            else
                if rem(iter, check_iter) == 0
                    # restart_info.current_gap = restart_info.weighted_norm
                    if restart_info.current_gap < 0
                        restart_info.current_gap = 1e-6
                        println("current_gap < 0")
                    end

                    if restart_info.current_gap <= 0.2 * restart_info.last_gap
                        restart_info.sufficient += 1
                        restart_info.restart_flag = 1
                        print("will restart by sufficient decrease, last restart gap = ", restart_info.last_gap, " current gap = ", restart_info.current_gap, "\n")
                    end

                    if (restart_info.current_gap <= 0.8 * restart_info.last_gap) && (restart_info.current_gap > 1.00 * restart_info.save_gap)
                        restart_info.necessary += 1
                        restart_info.restart_flag = 2
                        print("will restart by necessary decrease, last restart gap = ", restart_info.last_gap, " current gap = ", restart_info.current_gap, " last check point gap = ", restart_info.save_gap, "\n")
                    end
                    if restart_info.current_gap / restart_info.weighted_norm > 1e-1
                        fact = 0.5
                    else
                        fact = 0.2
                    end

                    # sigma_ratio = ws.sigma_can / ws.sigma
                    if restart_info.inner >= fact * iter
                        # if restart_info.inner >= 0.1 * iter  && (restart_info.current_gap > 0.99 * restart_info.save_gap)
                        restart_info.long += 1
                        restart_info.restart_flag = 3
                        print("will restart by long time\n")
                    end

                    # ratio = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                    # if ratio > 1e8 || ratio < 1e-8
                    #     restart_info.ratio += 1
                    #     restart_info.restart_flag = 3
                    #     print("will restart by large ratio\n")
                    # end
                    restart_info.save_gap = restart_info.current_gap
                end
            end
        end
    end
end

function try_restart(restart_info, ws::Union{HPRQP_workspace_cpu,HPRQP_workspace_gpu}, iter::Int)
    if restart_info.restart_flag > 0
        println()
        ws.x .= ws.x_bar
        ws.y .= ws.y_bar
        ws.last_x .= ws.x_bar
        ws.last_y .= ws.y_bar
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf
        println("restart at current iter ", iter)
        restart_info.times += 1
        restart_info.inner = 0
        # elseif restart_info.inner > 1e4
        #     ws.x .= ws.x_bar
        #     ws.y .= ws.y_bar
        #     ws.last_x .= ws.x_bar
        #     ws.last_y .= ws.y_bar
    end

end

function check_break(residuals::HPRQP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRQP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

function collect_results_gpu!(
    ws::HPRQP_workspace_gpu,
    residuals::HPRQP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
)
    results = HPRQP_results()
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    results.sigma = ws.sigma
    ### copy the results to the CPU ### 
    results.x = Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y = Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z = Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))

    return results
end


function collect_results_cpu!(
    ws::HPRQP_workspace_cpu,
    residuals::HPRQP_residuals,
    sc::Scaling_info,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
)
    results = HPRQP_results()
    results.x = Vector{Float64}(undef, ws.n)
    results.y = Vector{Float64}(undef, ws.m)
    results.z = Vector{Float64}(undef, ws.n)
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    results.sigma = ws.sigma
    results.x .= sc.b_scale * (ws.x_bar ./ sc.col_norm)
    results.y .= sc.c_scale * (ws.y_bar ./ sc.row_norm)
    results.z .= sc.c_scale * (ws.z_bar .* sc.col_norm)

    return results
end

function adaptive_check_iter(iter)
    if iter < 1000
        check_iter = 10
    elseif iter < 10000
        check_iter = 50
    elseif iter < 100000
        check_iter = 200
    else
        check_iter = 500
    end
    return check_iter
end

function allocate_workspace_gpu(qp::QP_info_gpu,
    params::HPRQP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64
)
    ws = HPRQP_workspace_gpu()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    ws.m1 = qp.m1
    ws.sigma = params.sigma
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.diag_Q = qp.diag_Q
    ws.fact1 = CUDA.zeros(Float64, n)
    ws.fact2 = CUDA.zeros(Float64, n)
    ws.x = CUDA.zeros(Float64, n)
    ws.x_hat = CUDA.zeros(Float64, n)
    ws.x_bar = CUDA.zeros(Float64, n)
    ws.dx = CUDA.zeros(Float64, n)
    ws.y = CUDA.zeros(Float64, m)
    ws.y_hat = CUDA.zeros(Float64, m)
    ws.y_bar = CUDA.zeros(Float64, m)
    ws.dy = CUDA.zeros(Float64, m)
    ws.z_bar = CUDA.zeros(Float64, n)
    ws.s_bar = CUDA.zeros(Float64, m)
    ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u
    ws.idxI = qp.idxI
    ws.Rp = CUDA.zeros(Float64, m)
    ws.Rd = CUDA.zeros(Float64, n)
    ws.ATy = CUDA.zeros(Float64, n)
    ws.ATy_bar = CUDA.zeros(Float64, n)
    ws.Ax = CUDA.zeros(Float64, m)
    ws.Qx = CUDA.zeros(Float64, n)
    ws.last_x = CUDA.zeros(Float64, n)
    ws.last_y = CUDA.zeros(Float64, m)
    ws.update_z = false
    ws.tempv = CUDA.zeros(Float64, n)

    if ws.m > 0
        ws.AL[ws.AL.==-Inf] .= -1e20
        ws.AU[ws.AU.==Inf] .= 1e20
    end
    l_inf = ws.l .== -Inf
    u_inf = ws.u .== Inf
    ws.l[l_inf] .= -1e20
    ws.u[u_inf] .= 1e20
    ws.idx_boxed = (.~l_inf) .& (.~u_inf)
    ws.idx_unbounded = l_inf .& u_inf
    ws.idx_uinf = (.~l_inf) .& u_inf
    ws.idx_linf = l_inf .& (.~u_inf)
    return ws
end

function allocate_workspace_cpu(qp::QP_info_cpu,
    params::HPRQP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
)
    ws = HPRQP_workspace_cpu()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    ws.m1 = qp.m1
    ws.sigma = params.sigma
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.x = Vector(zeros(n))
    ws.x_hat = Vector(zeros(n))
    ws.x_bar = Vector(zeros(n))
    ws.dx = Vector(zeros(n))
    ws.y = Vector(zeros(m))
    ws.y_hat = Vector(zeros(m))
    ws.y_bar = Vector(zeros(m))
    ws.dy = Vector(zeros(m))
    ws.z_bar = Vector(zeros(n))
    ws.s_bar = Vector(zeros(m))
    ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u
    ws.idxI = qp.idxI
    ws.Rp = Vector(zeros(m))
    ws.Rd = Vector(zeros(n))
    ws.ATy = Vector(zeros(n))
    ws.ATy_bar = Vector(zeros(n))
    ws.Ax = Vector(zeros(m))
    ws.Qx = Vector(zeros(n))
    ws.last_x = Vector(zeros(n))
    ws.last_y = Vector(zeros(m))
    ws.update_z = false
    ws.tempv = Vector(zeros(n))

    if ws.m > 0
        ws.AL[ws.AL.==-Inf] .= -1e20
        ws.AU[ws.AU.==Inf] .= 1e20
    end
    l_inf = ws.l .== -Inf
    u_inf = ws.u .== Inf
    ws.l[l_inf] .= -1e20
    ws.u[u_inf] .= 1e20
    ws.idx_boxed = (.~l_inf) .& (.~u_inf)
    ws.idx_unbounded = l_inf .& u_inf
    ws.idx_uinf = (.~l_inf) .& u_inf
    ws.idx_linf = l_inf .& (.~u_inf)
    return ws
end

function initialize_restart(params::HPRQP_parameters)
    restart_info = HPRQP_restart()
    if params.restart == -1
        ## adaptive restart
        restart_info.restart_type = 2
    elseif params.restart == 0
        ## no restart
        restart_info.restart_type = 0
    else
        ## fixed restart
        restart_info.restart_type = 1
        restart_info.step = params.restart
    end
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.ratio = 0
    restart_info.restart_flag = 0
    restart_info.weighted_norm = Inf
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end


function power_iteration_A_cpu(A, AT, max_iterations=5000, tolerance=1e-3)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    lambda_max = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q        # error 
        if norm(q) < tolerance
            # lambda_max = dot(b_new, A * (AT * b_new))
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", LinearAlgebra.norm(q))
    return lambda_max
end

# function power_iteration_A_gpu(A, AT, max_iterations=5000, tolerance=1e-3)
#     seed = 1
#     m, n = size(A)
#     z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
#     q = CUDA.zeros(Float64, m)
#     ATq = CUDA.zeros(Float64, n)
#     lambda_max = 1.0
#     for i in 1:max_iterations
#         q .= z
#         q ./= LinearAlgebra.norm(q)
#         CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         lambda_max = LinearAlgebra.dot(q, z)
#         q .= z .- lambda_max .* q        # error 
#         if LinearAlgebra.norm(q) < tolerance
#             # lambda_max = dot(b_new, A * (AT * b_new))
#             println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
#             return lambda_max
#         end
#     end
#     println("Power iteration did not converge within the specified tolerance.")
#     println("The maximum iteration is ", max_iterations, " and the error is ", LinearAlgebra.norm(q))
#     return lambda_max
# end

function power_iteration_Q_cpu(Q, max_iterations=5000, tolerance=1e-3)
    seed = 1
    n, n = size(Q)
    z = Vector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, n)
    # ATq = zeros(Float64, n)
    lambda_max = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        # mul!(ATq, AT, q)
        mul!(z, Q, q)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q        # error 
        if norm(q) < tolerance
            # lambda_max = dot(b_new, A * (AT * b_new))
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", LinearAlgebra.norm(q))
    return lambda_max
end

# function power_iteration_Q_gpu(Q, max_iterations=5000, tolerance=1e-3)
#     seed = 1
#     n, n = size(Q)
#     z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
#     # z = CuVector(rand(m)) .+ 1e-8
#     q = CUDA.zeros(Float64, n)
#     lambda_max = 1.0
#     for i in 1:max_iterations
#         q .= z
#         q ./= LinearAlgebra.norm(q)
#         # mul!(ATq, AT, q)
#         CUDA.CUSPARSE.mv!('N', 1, Q, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         lambda_max = LinearAlgebra.dot(q, z)
#         q .= z .- lambda_max .* q        # error 
#         if LinearAlgebra.norm(q) < tolerance
#             # lambda_max = dot(b_new, A * (AT * b_new))
#             println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
#             return lambda_max
#         end
#     end
#     println("Power iteration did not converge within the specified tolerance.")
#     println("The maximum iteration is ", max_iterations, " and the error is ", LinearAlgebra.norm(q))
#     return lambda_max
# end

# function power_iteration_Q_gpu_new(Q, max_iterations=5000, tolerance=1e-4)
#     seed = 1
#     n, n = size(Q)
#     q = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
#     q ./= LinearAlgebra.norm(q)
#     # z = CuVector(rand(m)) .+ 1e-8
#     z = CUDA.zeros(Float64, n)
#     q_new = CUDA.zeros(Float64, n)
#     lambda_max = 1.0
#     error = 1.0
#     for i in 1:max_iterations
#         # mul!(ATq, AT, q)
#         CUDA.CUSPARSE.mv!('N', 1, Q, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         q_new .= z / LinearAlgebra.norm(z)
#         error = LinearAlgebra.norm(q_new - q)
#         if error < tolerance
#             CUDA.CUSPARSE.mv!('N', 1, Q, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#             lambda_max = LinearAlgebra.dot(q_new, z)
#             println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
#             return lambda_max
#         end
#         q .= q_new
#     end
#     println("Power iteration did not converge within the specified tolerance.")
#     println("The maximum iteration is ", max_iterations, " and the error is ", error)
#     CUDA.CUSPARSE.mv!('N', 1, Q, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#     lambda_max = LinearAlgebra.dot(q_new, z)
#     return lambda_max
# end

function power_iteration_Q_cpu_new(Q, max_iterations=5000, tolerance=1e-3)
    seed = 1
    n, n = size(Q)
    q = Vector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q ./= norm(q)
    z = zeros(Float64, n)
    q_new = zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        # mul!(ATq, AT, q)
        mul!(z, Q, q)
        q_new .= z / norm(z)
        error = norm(q_new - q)
        if error < tolerance
            mul!(z, Q, q_new)
            lambda_max = dot(q_new, z)
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
        q .= q_new
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    mul!(z, Q, q_new)
    lambda_max = dot(q_new, z)
    return lambda_max
end

function power_iteration_A_gpu_new(A, AT, max_iterations=5000, tolerance=1e-4)
    seed = 1
    m, n = size(A)
    q = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q ./= LinearAlgebra.norm(q)
    # z = CuVector(rand(m)) .+ 1e-8
    z = CUDA.zeros(Float64, n)
    q_new = CUDA.zeros(Float64, m)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        # mul!(ATq, AT, q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, z, 0, q_new, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        q_new ./= LinearAlgebra.norm(q_new)
        error = LinearAlgebra.norm(q_new - q)
        if error < tolerance
            CUDA.CUSPARSE.mv!('N', 1, AT, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            CUDA.CUSPARSE.mv!('N', 1, A, z, 0, q, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            lambda_max = LinearAlgebra.dot(q_new, q)
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
        q .= q_new
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    CUDA.CUSPARSE.mv!('N', 1, AT, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, A, z, 0, q, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    lambda_max = LinearAlgebra.dot(q_new, q)
    return lambda_max
end

function power_iteration_A_cpu_new(A, AT, max_iterations=5000, tolerance=1e-3)
    seed = 1
    m, n = size(A)
    q = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q ./= norm(q)
    # z = CuVector(rand(m)) .+ 1e-8
    z = zeros(Float64, n)
    q_new = zeros(Float64, m)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        # mul!(ATq, AT, q)
        mul!(z, AT, q)
        mul!(q_new, A, z)
        q_new ./= norm(q_new)
        error = norm(q_new - q)
        if error < tolerance
            mul!(z, AT, q_new)
            mul!(q, A, z)
            lambda_max = dot(q_new, q)
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
        q .= q_new
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    mul!(z, AT, q_new)
    mul!(q, A, z)
    lambda_max = dot(q_new, q)
    return lambda_max
end



function HPRQP(qp, scaling_info, params)

    useGPU = params.useGPU

    m, n = size(qp.A)

    ### power iteration to estimate lambda_max ###

    # CUDA.synchronize()
    t_start_alg = time()

    if params.useGPU
        # lambda_max = power_iteration_gpu2(qp.A, qp.AT) * params.lambda_factor
        # lambda_max = 1.0
        if m > 0
            lambda_max_A = power_iteration_A_gpu(qp.A, qp.AT) * params.lambda_factor
        else
            lambda_max_A = 0.0
        end
        if length(qp.Q.nzVal) > 0
            if !qp.Q_is_diag
                lambda_max_Q = power_iteration_Q_gpu(qp.Q) * params.lambda_factor
            else
                lambda_max_Q = maximum(qp.Q.nzVal)
            end
        else
            lambda_max_Q = 0.0
        end
    else
        if m > 0
            lambda_max_A = power_iteration_A_cpu_new(qp.A, qp.AT) * params.lambda_factor
        else
            lambda_max_A = 0.0
        end
        if length(qp.Q.nzval) > 0
            lambda_max_Q = power_iteration_Q_cpu_new(qp.Q) * params.lambda_factor
        else
            lambda_max_Q = 0.0
        end
    end
    # return
    CUDA.synchronize()
    power_time = time() - t_start_alg
    println("lambda_max A = ", lambda_max_A)
    println("lambda_max Q = ", lambda_max_Q)

    ### Initialization ###
    residuals = HPRQP_residuals()

    restart_info = initialize_restart(params)

    if useGPU
        ws = allocate_workspace_gpu(qp, params, lambda_max_A, lambda_max_Q)
    else
        ws = allocate_workspace_cpu(qp, params, lambda_max_A, lambda_max_Q)
    end

    println("HPRQP solver starts...")
    println("iter    errRp     errRd     last_gap    current_gap     p_obj    d_obj    gap    sigma    time")

    if useGPU
        # # obsolete memory diagnostic omitted
    end


    iter_3 = 0
    time_3 = 0.0
    iter_6 = 0
    time_6 = 0.0
    first_3 = true
    first_6 = true
    # params.sigma_fixed = true
    
    if ws.m > 0
            max_nnz_A_row = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=2)))
            max_nnz_A_col = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=1)))
        else
            max_nnz_A_row = 0
            max_nnz_A_col = 0
        end
        max_nnz_Q_col = maximum(vec(sum(t -> (abs(t) > 0), qp.Q, dims=1)))
        if (max_nnz_A_row > max(0.01 * n, 500)) || (max_nnz_A_col > max(0.01 * m, 500)) || (max_nnz_Q_col > max(0.01 * n, 500))
            spmv_mode = "CUSPARSE"
        else
            spmv_mode = "customized"
        end

        if m > 1e5 && n > 1e5
            spmv_mode = "CUSPARSE"
        end
    println("SPMV mode = ", spmv_mode)


    if qp.Q_is_diag
        update_Q_factors_gpu!(
            ws.fact1, ws.fact2, ws.diag_Q, ws.sigma, ws.lambda_max_A
        )
    end


    for iter = 0:params.max_iter
        # if residuals.KKTx_and_gap_org_bar > 10 * params.stoptol
        check_iter = params.check_iter
        # if iter < 100
        #     check_iter = 1
        # end
        # else
        #     check_iter = 50
        # end

        ### compute residuals ###
        if rem(iter, check_iter) == 0
            if params.useGPU
                compute_residuals_gpu(ws, qp, scaling_info, residuals, params, false, params.check_iter, iter)
            else
                compute_residuals_cpu(ws, qp, scaling_info, residuals, params, false, params.check_iter, iter)
                # compute_residuals_cpu_original(ws, qp, qp_org, scaling_info, residuals, params, false, params.check_iter, iter)
            end
            residuals.is_updated = true
        else
            residuals.is_updated = false
        end

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        # restart_info.weighted_norm = sqrt(residuals.err_Rp_bar^2 * ws.sigma + residuals.err_Rd_bar^2 / ws.sigma)
        check_restart(restart_info, residuals, iter, check_iter)

        print_yes = ((rem(iter, max(print_step(iter), check_iter)) == 0) || (iter == params.max_iter) ||
                     (time() - t_start_alg > params.time_limit) || (restart_info.inner == check_iter) ||
                     (restart_info.restart_flag > 0))


        ### print the log ##
        if print_yes || (status != "CONTINUE")
            # if (rem(iter, print_step(iter)) == 0) || (restart_info.restart_flag > 0) || (status != "CONTINUE")

            print(@sprintf("%4.0f", iter),
                @sprintf("    %3.2e", residuals.err_Rp_org_bar),
                @sprintf("    %3.2e", residuals.err_Rd_org_bar),
                @sprintf("    %3.2e", restart_info.last_gap),
                @sprintf("    %3.2e", restart_info.current_gap),
                @sprintf("    %7.6e", residuals.primal_obj_bar),
                @sprintf("    %7.6e", residuals.dual_obj_bar),
                @sprintf("    %3.2e", residuals.rel_gap_bar))
            # end
            print(@sprintf("    %3.2e", ws.sigma),
                @sprintf("    %6.2f", time() - t_start_alg),
                " ", restart_info.times, " ", restart_info.sufficient, " ",
                restart_info.necessary, " ", restart_info.long, " ", restart_info.ratio)
            print("\n")
        end



        ### update sigma ###
        # if residuals.KKTx_and_gap_org_bar < 1e-1
        # params.sigma_fixed = false
        # end
        sigma_old = ws.sigma
        update_sigma(params, restart_info, iter, scaling_info, ws, residuals, qp.Q_is_diag)

        ### restart if needed ###
        try_restart(restart_info, ws, iter)
        ### collect results and return ###
        if residuals.KKTx_and_gap_org_bar < 1e-3 && first_3
            time_3 = time() - t_start_alg
            iter_3 = iter
            first_3 = false
            println("KKT < 1e-3 at iter = ", iter)
        end
        if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6
            time_6 = time() - t_start_alg
            iter_6 = iter
            first_6 = false
            println("KKT < 1e-6 at iter = ", iter)
        end
        if status != "CONTINUE"
            if params.useGPU
                results = collect_results_gpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time)
                results.output_type = status
                # return results
            else
                results = collect_results_cpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time)
                results.output_type = status
                # return results
            end
            results.time_3 = time_3 == 0.0 ? results.time : time_3
            results.iter_3 = iter_3 == 0 ? iter : iter_3
            results.time_6 = time_6 == 0.0 ? results.time : time_6
            results.iter_6 = iter_6 == 0 ? iter : iter_6
            # println(results)
            return results
        end



        ## main iteatrion 
        Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
        Halpern_fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
        # update y


        if qp.Q_is_diag
            if abs(sigma_old - ws.sigma) > 1e-15
                update_Q_factors_gpu!(
                    ws.fact1, ws.fact2, ws.diag_Q, ws.sigma, ws.lambda_max_A
                )
            end
        end



        if params.useGPU
        else
            update_ys_cpu!(ws)
        end
        # update x
        if params.useGPU
            if spmv_mode == "customized"
                if qp.Q_is_diag
                    cust_update_ys_gpu!(ws, Halpern_fact1, Halpern_fact2)
                    cust_update_xz_gpu_Q_diag!(ws, Halpern_fact1, Halpern_fact2)
                else
                    cust_update_ys_gpu!(ws, Halpern_fact1, Halpern_fact2)
                    cust_update_xz_gpu!(ws, Halpern_fact1, Halpern_fact2)
                end
            else
                if qp.Q_is_diag
                    update_ys_gpu!(ws, Halpern_fact1, Halpern_fact2)
                    update_xz_gpu_Q_diag!(ws, Halpern_fact1, Halpern_fact2)
                else
                    update_ys_gpu!(ws, Halpern_fact1, Halpern_fact2)
                    update_xz_gpu!(ws, Halpern_fact1, Halpern_fact2)
                end
            end
        else
            update_xz_cpu!(ws)
        end
        # ws.x_hat .= ws.x_bar


        if restart_info.restart_flag > 0
            if params.useGPU
                restart_info.last_gap = compute_weighted_norm_gpu(ws, qp.Q_is_diag)
            else
                restart_info.last_gap = compute_weighted_norm_cpu(ws)
            end
            # ws.lambda_max_A = lambda_max_A
            # ws.lambda_max_Q = lambda_max_Q
        end
        if rem(iter + 1, check_iter) == 0
            if params.useGPU
                restart_info.current_gap = compute_weighted_norm_gpu(ws, qp.Q_is_diag)
            else
                restart_info.current_gap = compute_weighted_norm_cpu(ws)
            end
        end

        ### Halpern update ###
        if params.restart != 0
            if params.useGPU
                # Halpern_update_gpu!(ws, restart_info)
            else
                Halpern_update_cpu!(ws, restart_info)
            end
            restart_info.inner += 1
        else
            ws.x .= ws.x_bar
            ws.y .= ws.y_bar
        end

        # ws.lambda_max_A /= (restart_info.inner+1) / (restart_info.inner)
        # ws.lambda_max_Q /= (restart_info.inner+1) / (restart_info.inner)


    end

end


function initialize_parameters()
    params = HPRQP_parameters()
    params.useGPU = true
    params.check_iter = 100
    params.max_iter = typemax(Int32)
    params.sigma_fixed = false
    params.sigma = 1.0

    params.lambda_factor = 1.01
    params.restart = -1
    params.stoptol = 1e-8
    params.time_limit = 15000

    return params
end


function run_file(FILE_NAME, useGPU=false, iter=typemax(Int32), time_limit=3600, stoptol=1e-8)
    # io = open(FILE_NAME)
    # qp = Logging.with_logger(Logging.NullLogger()) do
    #     readqps(io, mpsformat=:free)
    # end
    # close(io)
    # CUDA.device!(3)
    Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps_SCS(FILE_NAME)
    standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
    # standard_qp.Q .= spdiagm(1*ones(size(standard_qp.Q, 1)))
    # standard_qp.Q.nzval[1] = 0.0
    # return standard_qp 
    # standard_qp.Q .+= spdiagm(1e-6*ones(699))
    # qp_org = deepcopy(standard_qp)
    # qp_org.AL[qp_org.AL.==-Inf] .= -1e20
    # qp_org.AU[qp_org.AU.==Inf] .= 1e20
    # qp_org.l[qp_org.l.==-Inf] .= -1e20
    # qp_org.u[qp_org.u.==Inf] .= 1e20
    # return qp_org
    if useGPU
        scaling_info = scaling!(standard_qp, true, false, false, true)
        # return standard_qp,  scaling_info
        if standard_qp.Q_is_diag
            println("Q is diagonal")
        end
        standard_qp.Q_is_diag = false

        standard_qp_gpu = QP_info_gpu(
            CuSparseMatrixCSR(standard_qp.Q),
            CuVector(standard_qp.c),
            CuSparseMatrixCSR(standard_qp.A),
            CuSparseMatrixCSR(standard_qp.A'),
            CuVector(standard_qp.AL),
            CuVector(standard_qp.AU),
            CuVector(standard_qp.l),
            CuVector(standard_qp.u),
            CuVector(standard_qp.idxI),
            standard_qp.m1,
            standard_qp.obj_constant,
            CuVector(standard_qp.diag_Q),
            standard_qp.Q_is_diag,)
        # return standard_qp_gpu
        scaling_info_gpu = Scaling_info_gpu(
            CuVector(scaling_info.l_org),
            CuVector(scaling_info.u_org),
            CuVector(scaling_info.row_norm),
            CuVector(scaling_info.col_norm),
            scaling_info.b_scale,
            scaling_info.c_scale,
            scaling_info.norm_b,
            scaling_info.norm_c,
            scaling_info.norm_b_org,
            scaling_info.norm_c_org)
    else
        scaling_info = scaling!(standard_qp, true, true, false, true)
    end

    # return standard_qp

    params = initialize_parameters()
    params.max_iter = iter
    params.time_limit = time_limit
    params.stoptol = stoptol
    params.useGPU = useGPU
    params.sigma_fixed = false
    params.restart = -1
    params.mode = "linearize"

    norm_b = scaling_info.norm_b
    norm_c = scaling_info.norm_c
    if norm_c > 1e-16 && norm_b > 1e-16 && norm_b < 1e16 && norm_c < 1e16
        params.sigma = norm_c / norm_b
    else
        params.sigma = 1.0
    end
    # params.sigma = 1e-3
    println("initial sigma = ", params.sigma)


    if useGPU
        results = HPRQP(standard_qp_gpu, scaling_info_gpu, params)
    else
        results = HPRQP(standard_qp, scaling_info, params)
    end

    return results

end

function mean(x::Vector{Float64})
    return sum(x) / length(x)
end

function run_dataset(data_path, result_path, time_limit=3600, tol=1e-8, do_warm_up=false, device_number=3, use_gpu=true)
    CUDA.device!(device_number)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRQP_result.csv"

    # redirect the output to a file
    log_path = result_path * "HPRQP_log.txt"

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter3list = Vector{Any}(result_table.iter_3[1:end-2])
        time3list = Vector{Any}(result_table.time_3[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter3list = []
        time3list = []
        iter6list = []
        time6list = []
    end


    skip_files = ["LISWET1.mps", "LISWET10.mps", "LISWET11.mps", "LISWET12.mps", "LISWET2.mps", "LISWET7.mps", "LISWET8.mps", "LISWET9.mps"]
    for i = 1:length(files)
        file = files[i]
        if file in skip_files
#            continue
        end
        if (occursin(".QPS", file) || occursin(".mps", file)) && !(file in namelist)
            FILE_NAME = data_path * file
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
            # println(file)

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                if do_warm_up
                    println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                    t_start_all = time()
                    results = run_file(FILE_NAME, use_gpu, 200, time_limit, tol)
                    all_time = time() - t_start_all
                    println("warm up time: ", all_time)
                    println("warm up ends ----------------------------------------------------------------------------------------------------------")
                end


                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, use_gpu, typemax(Int32), time_limit, tol)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter3list, results.iter_3)
                push!(time3list, min(results.time_3, time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, time_limit))

                # when the problem is not solved, save the x, y and w to a binary file
                # if results.time >= time_limit
                #     # save the x, y and w to a binary file
                #     x_file = result_path * file[1:end-4] * "_x" * ".bin"
                #     y_file = result_path * file[1:end-4] * "_y" * ".bin"
                #     w_file = result_path * file[1:end-4] * "_w" * ".bin"
                #     open(x_file, "w") do io
                #         write(io, results.x)
                #     end
                #     open(y_file, "w") do io
                #         write(io, results.y)
                #     end
                #     open(w_file, "w") do io
                #         write(io, results.w)
                #     end
                # end
            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_3=iter3list,
                time_3=time3list,
                iter_6=iter6list,
                time_6=time6list,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_3 = exp(mean(log.(time3list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_3 = exp(mean(log.(iter3list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_3, geomean_time_3, geomean_iter_6, geomean_time_6])

            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < time_limit, timelist)
            solved_3 = count(x -> x < time_limit, time3list)
            solved_6 = count(x -> x < time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_3, "", solved_6])

            CSV.write(csv_file, result_table)
        end
    end

    close(io)
end


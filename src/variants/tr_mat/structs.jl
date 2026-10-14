# problem: min 0.5 * x' * Q * x + c' * x
#           s.t. A_ineq * x >= b_ineq
#                A_eq * x = b_eq
#                A_SOC * x - b_SOC ∈ SOC
#                x ∈ K (>=0, free, =0, SOC)



mutable struct SOCP_info_cpu
    Q::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    b::Vector{Float64}
    SOC_con_idx::Vector{Int}
    number_ineq::Int
    number_eq::Int
    l::Vector{Float64}
    u::Vector{Float64}
    SOC_var_idx::Vector{Int}
    number_lu_x::Int
    obj_constant::Float64
end

mutable struct Scaling_info_cpu
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
    SOC_con_scaling_factor::Vector{Float64}
    SOC_con_scaling_factor_inv::Vector{Float64}
    SOC_var_scaling_factor::Vector{Float64}
    SOC_var_scaling_factor_inv::Vector{Float64}
end

mutable struct SOCP_info_gpu
    Q::CuSparseMatrixCSR{Float64,Int32}
    c::CuVector{Float64}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    b::CuVector{Float64}
    SOC_con_idx::CuVector{Int}
    number_ineq::Int
    number_eq::Int
    l::CuVector{Float64}
    u::CuVector{Float64}
    SOC_var_idx::CuVector{Int}
    number_lu_x::Int
    obj_constant::Float64
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
    SOC_con_scaling_factor::CuVector{Float64}
    SOC_con_scaling_factor_inv::CuVector{Float64}
    SOC_var_scaling_factor::CuVector{Float64}
    SOC_var_scaling_factor_inv::CuVector{Float64}
end

mutable struct HPRSOCP_parameters
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
    do_warm_up::Bool
    spmv_mode::String
    alpha::Float64
    HPRSOCP_parameters() = new(1e-6, -1, 1000, false, 3600.0, -1, 1.01, 100, false, 1, "direct", false, "auto", 2.0)
end

mutable struct HPRSOCP_residuals
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
    HPRSOCP_residuals() = new()
end

mutable struct HPRSOCP_results
    iter::Int
    iter_4::Int
    iter_6::Int
    time::Float64
    time_4::Float64
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
    w::Vector{Float64}
    HPRSOCP_results() = new()
end

mutable struct HPRSOCP_workspace_gpu
    w::CuVector{Float64}
    w_hat::CuVector{Float64}
    w_bar::CuVector{Float64}
    dw::CuVector{Float64}
    x::CuVector{Float64}
    x_hat::CuVector{Float64}
    x_bar::CuVector{Float64}
    dx::CuVector{Float64}
    y::CuVector{Float64}
    y_hat::CuVector{Float64}
    y_bar::CuVector{Float64}
    dy::CuVector{Float64}
    z_bar::CuVector{Float64}
    dz::CuVector{Float64}
    Q::CuSparseMatrixCSR{Float64,Int32}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    b::CuVector{Float64}
    c::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    SOC_con_idx::CuVector{Int}
    SOC_var_idx::CuVector{Int}
    number_ineq::Int
    number_eq::Int
    number_lu_x::Int
    number_SOC_con::Int
    number_SOC_var::Int
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
    ATdy::CuVector{Float64}
    QATdy::CuVector{Float64}
    s::CuVector{Float64}
    s_org::CuVector{Float64}
    Qw::CuVector{Float64}
    Qw_hat::CuVector{Float64}
    Qw_bar::CuVector{Float64}
    Qx::CuVector{Float64}
    dQw::CuVector{Float64}
    last_x::CuVector{Float64}
    last_y::CuVector{Float64}
    last_Qw::CuVector{Float64}
    last_w::CuVector{Float64}
    last_ATy::CuVector{Float64}
    update_z::Bool
    tempv::CuVector{Float64}
    diag_Q::CuVector{Float64}
    fact1::CuVector{Float64}
    fact2::CuVector{Float64}
    fact::CuVector{Float64}
    fact_M::CuVector{Float64}
    SOC_con_scaling_factor::CuVector{Float64}
    SOC_con_scaling_factor_inv::CuVector{Float64}
    SOC_var_scaling_factor::CuVector{Float64}
    SOC_var_scaling_factor_inv::CuVector{Float64}
    SOC_norms_temp::CuVector{Float64}  # Temporary array for precomputed SOC norms (one per SOC)
    HPRSOCP_workspace_gpu() = new()
end

mutable struct HPRSOCP_restart
    # 0: no restart, 1: restart with avg, 2: restart with current
    restart_flag::Int

    # 0: no restart, 1: fixed step restart, 2: adaptive restart
    restart_type::Int
    first_restart::Bool
    # first_gap::Float64
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

    HPRSOCP_restart() = new()
end
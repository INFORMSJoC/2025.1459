mutable struct QAP_Q_operator
    A::Matrix{Float64}
    B::Matrix{Float64}
    S::Matrix{Float64}
    T::Matrix{Float64}
    n::Int
end

mutable struct QAP_Q_operator_gpu
    A::CuMatrix{Float64}
    B::CuMatrix{Float64}
    S::CuMatrix{Float64}
    T::CuMatrix{Float64}
    n::Int
end

mutable struct LASSO_Q_operator_gpu
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
end


mutable struct LASSO_C_operator
    lambda::Float64
end

mutable struct QP_info_cpu
    Q::Union{QAP_Q_operator,
             SparseMatrixCSC{Float64,Int64}}
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int64}
    AT::SparseMatrixCSC{Float64,Int64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    m1::Int
    obj_constant::Float64
    diag_Q::Vector{Float64}
    Q_is_diag::Bool
    lambda::Float64
    # C::LASSO_C_operator
end

mutable struct QP_info_gpu
    Q::Union{QAP_Q_operator_gpu,
             CuSparseMatrixCSR{Float64,Int32}, LASSO_Q_operator_gpu}
    c::CuVector{Float64}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    m1::Int
    obj_constant::Float64
    diag_Q::CuVector{Float64}
    Q_is_diag::Bool
    lambda::CuVector{Float64}
    # C::LASSO_C_operator
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
    do_warm_up::Bool
    spmv_mode::String
    alpha::Float64
    problem_type::String
    HPRQP_parameters() = new(1e-6, -1, 1000, false, 3600.0, -1, 1.05, 100, false, 1, "direct", false, "auto", 2.0, "QP")
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
    HPRQP_results() = new()
end

mutable struct HPRQP_workspace_gpu
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
    ATdy::CuVector{Float64}
    QATdy::CuVector{Float64}
    s::CuVector{Float64}
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
    temp1::CuVector{Float64}
    HPRQP_workspace_gpu() = new()
end

mutable struct HPRQP_restart
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

    HPRQP_restart() = new()
end
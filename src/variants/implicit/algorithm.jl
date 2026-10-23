function compute_residuals_gpu(ws::Union{HPRQP_workspace_gpu},
    qp::Union{QP_info_cpu,QP_info_gpu},
    sc::Union{Scaling_info_gpu},
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    print_yes::Bool,
    check_iter::Int,
    iter::Int,
)
    res.is_updated = false

    if rem(iter, check_iter) == 0
        res.is_updated = true


        ### obj
        # res.primal_obj = LinearAlgebra.dot(ws.c, ws.x_bar)
        # CUSPARSE.mv!('N', 1, ws.Q, ws.x_bar, 0, ws.Qx, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        if typeof(qp.Q) <: Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
            QAP_Qmap!(ws.x_bar, ws.Qx, ws.temp1, qp.Q)
        elseif typeof(qp.Q) <: CuSparseMatrixCSR
            ws.Qx .= qp.diag_Q .* ws.x_bar
        end
        # CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        res.primal_obj_bar = sc.b_scale * sc.c_scale *
                             (LinearAlgebra.dot(ws.c, ws.x_bar) + 0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx)) + qp.obj_constant



        res.dual_obj_bar =
            sc.b_scale * sc.c_scale *
            (-0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx)
             # (-0.5 * dot(ws.w_bar, ws.Qw_bar)
             #  + LinearAlgebra.dot(min.(ws.y_bar, 0), ws.AU) + LinearAlgebra.dot(max.(ws.y_bar, 0), ws.AL)
             + LinearAlgebra.dot(ws.y_bar, ws.s)
             #  + LinearAlgebra.dot(min.(ws.z_bar, 0), ws.u) + LinearAlgebra.dot(max.(ws.z_bar, 0), ws.l))
             + LinearAlgebra.dot(ws.z_bar, ws.x_bar)) + qp.obj_constant
        # res.dual_obj_bar = sc.b_scale * sc.c_scale * res.dual_obj + qp.obj_constant

        # println("primal_obj_bar = ", res.primal_obj_bar)
        if length(qp.lambda) > 0
            ws.Rd .= qp.lambda .* ws.x_bar
            # println("l1 norm of Rd = ", LinearAlgebra.norm(ws.Rd, 1))
            # println(norm(ws.x_bar, 1))
            l1_norm = LinearAlgebra.norm(ws.Rd, 1)
            # println("l1_norm = ", l1_norm)
            res.primal_obj_bar += sc.b_scale * sc.c_scale * l1_norm
            res.dual_obj_bar += sc.b_scale * sc.c_scale * l1_norm
        end


        res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))


        ### Rd
        compute_Rd_gpu!(ws)
        res.err_Rd_bar = LinearAlgebra.norm(ws.Rd)

        ws.Rd .*= sc.col_norm * sc.c_scale
        ws.ATdy .*= sc.col_norm * sc.c_scale
        ws.Qx .*= sc.col_norm * sc.c_scale
        res.err_Rd_org_bar = LinearAlgebra.norm(ws.Rd, Inf) / (1.0 + maximum([sc.norm_c_org, LinearAlgebra.norm(ws.ATdy, Inf), LinearAlgebra.norm(ws.Qx, Inf)]))

        ### Rp
        if ws.m > 0
            compute_Rp_gpu!(ws)
            res.err_Rp_bar = LinearAlgebra.norm(ws.Rp)
            ws.Rp .*= sc.row_norm * sc.b_scale
            ws.Ax .*= sc.row_norm * sc.b_scale
            res.err_Rp_org_bar = LinearAlgebra.norm(ws.Rp, Inf) / (1.0 + max(sc.norm_b_org, LinearAlgebra.norm(ws.Ax, Inf)))
        else
            res.err_Rp_bar = 0.0
            res.err_Rp_org_bar = 0.0
        end

        if params.problem_type != "LASSO"
            if iter == 0
                res.err_Rp_org_bar += LinearAlgebra.norm(ws.x_bar - max.(min.(ws.x_bar, ws.u), ws.l), Inf)
            end
        end
        res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
        # res.KKTx_org_bar = max(res.err_Rp_org_bar, res.err_K_org_bar, res.err_Rd_org_bar, res.err_z_org_bar, res.err_Comp_org_bar) 

    end
end


function update_sigma(params::HPRQP_parameters,
    restart_info::Union{HPRQP_restart,HPRQP_restart},
    iter::Int,
    sc::Union{Scaling_info_gpu},
    ws::Union{HPRQP_workspace_gpu},
    qp::Union{QP_info_cpu,QP_info_gpu},
    residuals::HPRQP_residuals,
    solver_AAT,
    Q_is_diag::Bool=false,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
        if ws.m > 0
            axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
            axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.last_ATy, ws.ATdy, ws.n)
            if typeof(qp.Q) <: Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
                QAP_Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
            elseif typeof(qp.Q) <: CuSparseMatrixCSR
                ws.QATdy .= qp.diag_Q .* ws.ATdy
            end
            a += ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy) - 2 * LinearAlgebra.dot(ws.dQw, ws.ATdy)
            # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
        axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
        # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        if typeof(qp.Q) <: Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
            QAP_Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
        elseif typeof(qp.Q) <: CuSparseMatrixCSR
            ws.dQw .= qp.diag_Q .* ws.dw
        end

        b += LinearAlgebra.dot(ws.dx, ws.dx)

        if Q_is_diag
            # if Q_is_diag
            a += LinearAlgebra.norm(ws.dQw)^2
        else
            a += ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
            if ws.m > 0
                c += LinearAlgebra.dot(ws.ATdy, ws.QATdy)
                d += ws.lambda_max_Q
            end
        end

        println("a = ", a, " b = ", b, " c = ", c, " d = ", d)
        a = max(a, 1e-12)
        b = max(b, 1e-12)
        # c = max(c, 1e-12)
        if ws.m == 0
            c = 0.0
        end

        println("a = ", a, " b = ", b, " c = ", c, " d = ", d)
        if ws.m > 0
            if b > 1e-14 && a > 1e-14
                if Q_is_diag
                    sigma_new = golden_Q_diag(a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv; lo=1e-12, hi=1e12, tol=1e-13)
                else
                    # min a * x + b / x + c * x^2 / (1 + d * x)

                    sigma_new = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
                end
            else
                sigma_new = ws.sigma
            end
        else
            sigma_new = sqrt(b / a)
        end
        fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
        ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))
        # println(" primal_move = ", primal_move, " dual_move = ", dual_move)
    end

end

function check_restart(restart_info::Union{HPRQP_restart,HPRQP_restart},
    residuals::HPRQP_residuals,
    iter::Int,
    check_iter::Int,
    ws::Union{HPRQP_workspace_gpu},
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

                    # sigma_ratio = ws.sigma_can / ws.sigma
                    # if sigma_ratio > 5000000 || sigma_ratio < 0.0002
                    #     # if restart_info.inner >= 0.1 * iter  && (restart_info.current_gap > 0.99 * restart_info.save_gap)

                    #     restart_info.long += 1
                    #     restart_info.restart_flag = 3
                    #     print("will restart by sigma: $(sigma_ratio),\n")
                    # end

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

function try_restart(restart_info, ws::Union{HPRQP_workspace_gpu}, qp::QP_info_gpu, iter::Int)
    if restart_info.restart_flag > 0
        # println()
        ws.x .= ws.x_bar
        ws.w .= ws.w_bar
        ws.last_x .= ws.x_bar
        ws.last_w .= ws.w_bar
        if ws.m > 0
            ws.y .= ws.y_bar
            ws.last_y .= ws.y_bar
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            ws.last_ATy .= ws.ATy_bar
            ws.ATy .= ws.ATy_bar
        end
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf
        println("restart at current iter ", iter)
        restart_info.times += 1
        restart_info.inner = 0
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
    results.w = Vector(sc.b_scale * (ws.w_bar ./ sc.col_norm))
    results.x = Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y = Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z = Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))

    return results
end

function allocate_workspace_gpu(qp::QP_info_gpu,
    params::HPRQP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
    scaling_info::Scaling_info_gpu,
)
    ws = HPRQP_workspace_gpu()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    ws.m1 = qp.m1
    if params.sigma == -1
        norm_b = scaling_info.norm_b
        norm_c = scaling_info.norm_c
        println("norm_b = ", norm_b)
        println("norm_c = ", norm_c)
        if norm_c > 1e-16 && norm_b > 1e-16 && norm_b < 1e16 && norm_c < 1e16
            ws.sigma = norm_b / norm_c
        else
            ws.sigma = 1.0
        end
    end
    println("initial sigma = ", ws.sigma)
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.diag_Q = qp.diag_Q
    ws.w = CUDA.zeros(Float64, n)
    # ws.w = copy(qp.x0)
    ws.w_hat = CUDA.zeros(Float64, n)
    ws.w_bar = CUDA.zeros(Float64, n)
    # ws.w_bar = copy(qp.x0)
    ws.dw = CUDA.zeros(Float64, n)
    ws.x = CUDA.zeros(Float64, n)
    # ws.x = copy(qp.x0)
    ws.x_hat = CUDA.zeros(Float64, n)
    ws.x_bar = CUDA.zeros(Float64, n)
    # ws.x_bar = copy(qp.x0)
    ws.dx = CUDA.zeros(Float64, n)
    ws.y = CUDA.zeros(Float64, m)
    # ws.y = copy(qp.y0)
    ws.y_hat = CUDA.zeros(Float64, m)
    ws.y_bar = CUDA.zeros(Float64, m)
    # ws.y_bar = copy(qp.y0)
    ws.dy = CUDA.zeros(Float64, m)
    ws.s = CUDA.zeros(Float64, m)
    ws.z_bar = CUDA.zeros(Float64, n)
    # ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u
    if ws.m > 0
        ws.AL[ws.AL.==-Inf] .= -1e20
        ws.AU[ws.AU.==Inf] .= 1e20
        ws.l[ws.l.==-Inf] .= -1e20
        ws.u[ws.u.==Inf] .= 1e20
    end
    # ws.idxI = qp.idxI
    ws.Rp = CUDA.zeros(Float64, m)
    ws.Rd = CUDA.zeros(Float64, n)
    ws.ATy = CUDA.zeros(Float64, n)
    ws.ATy_bar = CUDA.zeros(Float64, n)
    ws.ATdy = CUDA.zeros(Float64, n)
    ws.QATdy = CUDA.zeros(Float64, n)
    ws.Ax = CUDA.zeros(Float64, m)
    ws.Qw = CUDA.zeros(Float64, n)
    ws.Qw_bar = CUDA.zeros(Float64, n)
    ws.Qw_hat = CUDA.zeros(Float64, n)
    ws.Qx = CUDA.zeros(Float64, n)
    ws.dQw = CUDA.zeros(Float64, n)
    ws.last_x = CUDA.zeros(Float64, n)
    ws.last_y = CUDA.zeros(Float64, m)
    ws.last_Qw = CUDA.zeros(Float64, n)
    ws.last_w = CUDA.zeros(Float64, n)
    # ws.last_x = copy(qp.x0)
    # ws.last_y = copy(qp.y0)
    # ws.last_w = copy(qp.x0)
    ws.last_ATy = CUDA.zeros(Float64, n)
    ws.update_z = false
    ws.tempv = CUDA.zeros(Float64, n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.last_Qw = copy(ws.Qw)
    # ws.Qw_bar .= copy(ws.Qw)
    # ws.last_ATy = copy(ws.ATy)
    # ws.z_bar .= ws.Qw - ws.ATy + ws.c
    ws.fact1 = CUDA.zeros(Float64, n)
    ws.fact2 = CUDA.zeros(Float64, n)
    ws.fact = CUDA.zeros(Float64, n)
    ws.fact_M = CUDA.zeros(Float64, n)
    if params.problem_type == "LASSO"
        ws.temp1 = CUDA.zeros(Float64, qp.Q.A.dims[1])
    else
        ws.temp1 = CUDA.zeros(Float64, n)
    end
    # println(Vector(ws.z_bar))
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


function HPRQP(qp, scaling_info, params)
    useGPU = params.useGPU

    m, n = size(qp.A)

    ### power iteration to estimate lambda_max ###

    # CUDA.synchronize()
    t_start_alg = time()

    if params.mode == "linearize"
        # lambda_max = power_iteration_gpu2(qp.A, qp.AT) * params.lambda_factor
        # lambda_max = 1.0
        if m > 0
            lambda_max_A = power_iteration_A_gpu(qp.A, qp.AT) * params.lambda_factor
        else
            lambda_max_A = 0.0
        end
        if !qp.Q_is_diag
            if params.problem_type == "LASSO"
                lambda_max_Q = power_iteration_Q_LASSO_gpu(qp.Q) * params.lambda_factor
            else
                lambda_max_Q = power_iteration_Q_gpu(qp.Q) * params.lambda_factor
            end
        else
            lambda_max_Q = maximum(qp.diag_Q)
        end
        CUDA.synchronize()
        power_time = time() - t_start_alg
        println("lambda_max A = ", lambda_max_A)
        println("lambda_max Q = ", lambda_max_Q)
    elseif params.mode == "direct"
        lambda_max_A = 0.0
        lambda_max_Q = 0.0
        power_time = 0.0
    else
        error("Unknown mode: ", params.mode)
    end

    ### Initialization ###
    residuals = HPRQP_residuals()

    restart_info = initialize_restart(params)

    # x0 = CUDA.zeros(Float64, n)
    # y0 = CUDA.zeros(Float64, m)
    # x0 .= copy(qp.x0)
    # y0 .= copy(qp.y0)

    # qp.x0 .= CUDA.zeros(Float64, n)
    # qp.y0 .= CUDA.zeros(Float64, m)

    ws = allocate_workspace_gpu(qp, params, lambda_max_A, lambda_max_Q, scaling_info)

    println("HPRQP solver starts...")
    println("iter    errRp     errRd     last_gap    current_gap     p_obj    d_obj    gap    sigma    time")

    if qp.Q_is_diag
        update_Q_factors_gpu!(
            ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma
        )
    end

    if useGPU
        # # obsolete memory diagnostic omitted
    end

    if params.mode == "direct"
        if params.useGPU
            Q_I = ws.Q + Diagonal(ones(ws.n) / ws.sigma |> CuVector)
            solver = CudssSolver(Q_I, "SPD", 'F')
            cudss("analysis", solver, ws.w_bar, ws.Qx)
            cudss("factorization", solver, ws.w_bar, ws.Qx)

            if m > 0
                solver_AAT = CudssSolver(ws.A * ws.AT, "G", 'F')
                cudss("analysis", solver_AAT, ws.y_bar, ws.Rp)
                cudss("factorization", solver_AAT, ws.y_bar, ws.Rp)
            else
                solver_AAT = 0
            end
        else
            Q_I = ws.Q + Diagonal(ones(ws.n) / ws.sigma)
            solver = cholesky(Q_I)

            # AAT = ws.A ws.A'
            # # upper triangular
            # AAT = UpperTriangular(AAT)
            if m > 0
                solver_AAT = cholesky(ws.A * ws.AT)
            else
                solver_AAT = 0
            end
        end
    else
        solver = 0
        solver_AAT = 0
    end

    iter_4 = 0
    time_4 = 0.0
    iter_6 = 0
    time_6 = 0.0
    first_4 = true
    first_6 = true

    spmv_mode = params.spmv_mode
    if params.spmv_mode == "auto"
        max_nnz_A_row = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=2)))
        max_nnz_A_col = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=1)))
        max_nnz_Q_col = maximum(vec(sum(t -> (abs(t) > 0), qp.Q, dims=1)))
        println("max_nnz_A_row = ", max_nnz_A_row)
        println("max_nnz_A_col = ", max_nnz_A_col)
        println("max_nnz_Q_col = ", max_nnz_Q_col)
        if (max_nnz_A_row > max(0.01 * n, 500)) || (max_nnz_A_col > max(0.01 * m, 500)) || (max_nnz_Q_col > max(0.01 * n, 500))
            spmv_mode = "CUSPARSE"
            println("use CUSPARSE")
        else
            spmv_mode = "customized"
            println("use customized")
        end

        # if (max_nnz_A_row > max(0.01 * n, 500))
        #     spmv_mode_A_row = "CUSPARSE"
        #     println("use CUSPARSE")
        # else
        #     spmv_mode_A_row = "customized"
        #     println("use customized")
        # end
        # if (max_nnz_A_col > max(0.01 * m, 500))
        #     spmv_mode_A_col = "CUSPARSE"
        #     println("use CUSPARSE")
        # else
        #     spmv_mode_A_col = "customized"
        #     println("use customized")
        # end
        # if (max_nnz_Q_col > max(0.01 * n, 500))
        #     spmv_mode_Q = "CUSPARSE"
        #     println("use CUSPARSE")
        # else
        #     spmv_mode_Q = "customized"
        #     println("use customized")
        # end


    end

    # qp.Q_is_diag = false
    # if qp.Q_is_diag
    #     update_Q_factors_gpu!(
    #         ws.fact2, ws.fact, ws.fact1, ws.fact_M,
    #         ws.diag_Q, ws.sigma
    #     )
    # end

    # plot_ATy_error = []
    check_iter = params.check_iter

    for iter = 0:params.max_iter
        compute_residuals_gpu(ws, qp, scaling_info, residuals, params, true, check_iter, iter)

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        check_restart(restart_info, residuals, iter, params.check_iter, ws)

        print_yes = ((rem(iter, print_step(iter)) == 0) || (iter == params.max_iter) ||
                     (time() - t_start_alg > params.time_limit) ||
                     (restart_info.restart_flag > 0))

        ### update sigma ###
        sigma_old = ws.sigma
        # if residuals.KKTx_and_gap_org_bar < 1e-4
        update_sigma(params, restart_info, iter, scaling_info, ws, qp, residuals, solver_AAT, qp.Q_is_diag)
        # if params.mode == "direct"
        #     if abs(sigma_old - ws.sigma) > 1e-18
        #         if params.useGPU
        #             Q_I = ws.Q + Diagonal(ones(ws.n) / ws.sigma |> CuVector)
        #             cudss_set(solver, Q_I)
        #             cudss("refactorization", solver, ws.w_bar, ws.Qx)
        #         else
        #             Q_I = ws.Q + Diagonal(ones(ws.n) / ws.sigma)
        #             cholesky!(solver, Q_I)
        #         end
        #     end
        # end
        # end



        ### restart if needed ###
        try_restart(restart_info, ws, qp, iter)

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
            println()
        end


        ### collect results and return ###
        if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4
            time_4 = time() - t_start_alg
            iter_4 = iter
            first_4 = false
            println("KKT < 1e-3 at iter = ", iter)
        end
        if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6
            time_6 = time() - t_start_alg
            iter_6 = iter
            first_6 = false
            println("KKT < 1e-6 at iter = ", iter)
        end
        if status != "CONTINUE"
            # plt = plot(1:length(plot_ATy_error), log10.(plot_ATy_error), title="ATy error", xlabel="iter", ylabel="error")
            # display(plt)
            results = collect_results_gpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time)
            results.output_type = status
            results.time_4 = time_4 == 0.0 ? results.time : time_4
            results.iter_4 = iter_4 == 0 ? iter : iter_4
            results.time_6 = time_6 == 0.0 ? results.time : time_6
            results.iter_6 = iter_6 == 0 ? iter : iter_6
            # println(results)
            return results
        end



        ## main iteatrion 


        # update z
        if rem(iter + 1, check_iter) == 0 || true
            ws.update_z = true
        end

        Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
        Halpern_fact2 = 1.0 - Halpern_fact1

        if qp.Q_is_diag
            if abs(sigma_old - ws.sigma) > 1e-15
                update_Q_factors_gpu!(
                    ws.fact2, ws.fact, ws.fact1, ws.fact_M,
                    ws.diag_Q, ws.sigma
                )
            end
        end

        # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        # update_zx_gpu!(ws, Halpern_fact1, Halpern_fact2)
        if qp.Q_is_diag
            update_zxw1_diagQ_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_y_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_w2_diagQ_gpu!(ws, Halpern_fact1, Halpern_fact2)
        else
            if spmv_mode == "customized"
                cust_update_zxw1_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)
            else
                if params.problem_type == "LASSO"
                    if typeof(qp.Q) <: Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
                        QAP_Qmap!(ws.w, ws.Qw, ws.temp1, qp.Q)
                    elseif typeof(qp.Q) <: CuSparseMatrixCSR
                        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                    end
                    # ws.Qw .= ws.Q * ws.w
                    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
                    fact1 = 1.0 - fact2
                    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw_kernel!(qp.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
                else
                    update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
                    update_y_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
                    update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)
                end
            end
        end


        if restart_info.restart_flag > 0
            restart_info.last_gap = compute_M_norm_gpu3!(ws, qp)
            # M_c = compute_M_norm_gpu!(ws, qp.Q_is_diag, qp.noC)
        end
        if rem(iter + 1, check_iter) == 0
            restart_info.current_gap = compute_M_norm_gpu3!(ws, qp)
        end

        if params.restart != 0
            restart_info.inner += 1
        else
            ws.x .= ws.x_bar
            ws.y .= ws.y_bar
            ws.Qw .= ws.Qw_bar
            ws.w .= ws.w_bar
            ws.ATy .= ws.ATy_bar
        end


    end

end

function read_QAP_mat(FILE_NAME)
    vars = matread(FILE_NAME)
    #QAPdata = vars["QAPdata"]
    QAPdata = vars
    A = QAPdata["A"]
    B = QAPdata["B"]
    S = QAPdata["S"]
    T = QAPdata["T"]
    n = size(A, 1)

    A_gpu = CuMatrix(A)
    B_gpu = CuMatrix(B)
    S_gpu = CuMatrix(S)
    T_gpu = CuMatrix(T)

    Q = QAP_Q_operator_gpu(A_gpu, B_gpu, S_gpu, T_gpu, n)
    c = CUDA.zeros(Float64, n^2)
    ee = ones(Float64, n)
    Id = spdiagm(ones(Float64, n))
    A = sparse(vcat(kron(ee', Id), kron(Id, ee')))

    A_gpu = CuSparseMatrixCSR(A)
    AT_gpu = CuSparseMatrixCSR(A')
    b = CuVector(ones(Float64, 2 * n))
    l = CUDA.zeros(Float64, n^2)
    u = Inf * CUDA.ones(Float64, n^2)
    # Convert to standard QP_info format
    standard_qp = QP_info_gpu(Q, c, A_gpu, AT_gpu, b, copy(b), l, u, 2 * n, 0.0, [], false, [])

    return standard_qp
end

function run_file(FILE_NAME, params, x_path="", y_path="")
    # io = open(FILE_NAME)
    # qp = Logging.with_logger(Logging.NullLogger()) do
    #     readqps(io, mpsformat=:free)
    # end
    # close(io)
    # QAP reader
    if endswith(FILE_NAME, ".mat")
        t_start = time()
        standard_qp_gpu = read_QAP_mat(FILE_NAME)
        println("QAP read time: ", time() - t_start)

        t_start = time()
        scaling_info_gpu = scaling_inexplicit!(standard_qp_gpu, false, false, false, false)
        println("QAP scaling time: ", time() - t_start)

        # scaling_info_gpu = Scaling_info_gpu(copy(standard_qp_gpu.l), copy(standard_qp_gpu.u), CuVector(ones(m)), CuVector(ones(n)), 1, 1, 1, 1, norm(abs.(standard_qp_gpu.AL), Inf), norm(standard_qp_gpu.c, Inf))
    elseif endswith(FILE_NAME, ".mps") && params.problem_type == "LASSO"
        t_start = time()
        Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps_SCS(FILE_NAME)
        println("LASSO read time: ", time() - t_start)
        # Convert to standard QP_info format
        t_start = time()
        standard_qp_gpu = qp_formulation_LASSO_ATA2(Q, c, A, lcon, ucon, lvar, uvar, c0)
        println("LASSO formulation time: ", time() - t_start)

        t_start = time()
        scaling_info_gpu = scaling_inexplicit!(standard_qp_gpu, false, false, false, false)
        println("LASSO scaling time: ", time() - t_start)
    else
        Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps_SCS(FILE_NAME)
        standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
        # if problem
        if params.useGPU
            scaling_info = scaling!(standard_qp, true, false, true, true)
            if standard_qp.Q_is_diag
                println("Q is diagonal")
            end
            standard_qp_gpu = QP_info_gpu(
                CuSparseMatrixCSR(standard_qp.Q),
                CuVector(standard_qp.c),
                CuSparseMatrixCSR(standard_qp.A),
                CuSparseMatrixCSR(standard_qp.A'),
                CuVector(standard_qp.AL),
                CuVector(standard_qp.AU),
                CuVector(standard_qp.l),
                CuVector(standard_qp.u),
                standard_qp.m1,
                standard_qp.obj_constant,
                CuVector(standard_qp.diag_Q),
                standard_qp.Q_is_diag
            )
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
            scaling_info = scaling!(standard_qp, true, false, true, true)
        end
    end

    if params.useGPU
        results = HPRQP(standard_qp_gpu, scaling_info_gpu, params)
    else
        results = HPRQP(standard_qp, scaling_info, params)
    end

    return results

end

function run_dataset(data_path, result_path, params, device_number=0)
    current_file = @__FILE__
    cp(current_file, result_path * "HPRQP.jl", force=true)
    CUDA.device!(device_number)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRQP_zxwyw_result.csv"

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
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time3list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        powerlist = Vector{Any}(result_table.power_time[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time3list = []
        iter6list = []
        time6list = []
        powerlist = []
    end


    if params.do_warm_up
        println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
        t_start_all = time()
        # Warm up the selected dataset without requiring a large bundled Lasso file.
        warmup_file = findfirst(f -> isfile(joinpath(data_path, f)) &&
            lowercase(splitext(f)[2]) in (".mps", ".qps", ".mat"), files)
        warmup_file === nothing && error("No supported input file found in " * data_path)
        results = run_file(joinpath(data_path, files[warmup_file]), params)
        all_time = time() - t_start_all
        println("warm up time: ", all_time)
        println("warm up ends ----------------------------------------------------------------------------------------------------------")
    end
    skip_files = ["LISWET1.QPS", "LISWET10.QPS", "LISWET11.QPS", "LISWET12.QPS", "LISWET2.QPS", "LISWET7.QPS", "LISWET8.QPS", "LISWET9.QPS", "YAO.QPS"]
    for i = 1:length(files)
        file = files[i]
        if file in skip_files
            continue
        end
        if (occursin(".QPS", file) || occursin(".mps", file) || occursin(".mat", file)) && !(file in namelist)
            FILE_NAME = data_path * file
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
            # println(file)

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                # if params.do_warm_up
                #     println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                #     t_start_all = time()
                #     results = run_file(FILE_NAME, params)
                #     all_time = time() - t_start_all
                #     println("warm up time: ", all_time)
                #     println("warm up ends ----------------------------------------------------------------------------------------------------------")
                # end


                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter4list, results.iter_4)
                push!(time3list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
                push!(powerlist, results.power_time)

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
                iter_4=iter4list,
                time_4=time3list,
                iter_6=iter6list,
                time_6=time6list,
                power_time=powerlist,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time3list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, ""])

            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_3 = count(x -> x < params.time_limit, time3list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_3, "", solved_6, ""])

            CSV.write(csv_file, result_table)
        end
    end

    close(io)
end

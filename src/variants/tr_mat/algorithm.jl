function compute_residuals_gpu(ws::Union{HPRSOCP_workspace_gpu},
    socp::Union{SOCP_info_cpu,SOCP_info_gpu},
    sc::Union{Scaling_info_gpu},
    res::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    print_yes::Bool,
    check_iter::Int,
    iter::Int,
)
    res.is_updated = false

    if rem(iter, check_iter) == 0
        res.is_updated = true


        ### obj
        # res.primal_obj = LinearAlgebra.dot(ws.c, ws.x_bar)
        CUSPARSE.mv!('N', 1, ws.Q, ws.x_bar, 0, ws.Qx, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        # CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        res.primal_obj_bar = sc.b_scale * sc.c_scale *
                             (LinearAlgebra.dot(ws.c, ws.x_bar) + 0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx)) + socp.obj_constant


        res.dual_obj_bar =
            sc.b_scale * sc.c_scale *
            (-0.5 * LinearAlgebra.dot(ws.x_bar, ws.Qx)
             # (-0.5 * dot(ws.w_bar, ws.Qw_bar)
             #  + LinearAlgebra.dot(min.(ws.y_bar, 0), ws.AU) + LinearAlgebra.dot(max.(ws.y_bar, 0), ws.AL)
             + LinearAlgebra.dot(ws.y_bar, ws.b)
             #  + LinearAlgebra.dot(min.(ws.z_bar, 0), ws.u) + LinearAlgebra.dot(max.(ws.z_bar, 0), ws.l))
             + LinearAlgebra.dot(ws.z_bar, ws.x_bar)) + socp.obj_constant
        # res.dual_obj_bar = sc.b_scale * sc.c_scale * res.dual_obj + socp.obj_constant

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
        end

        if iter == 0
            if ws.number_SOC_var > 0
                @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) compute_x_SOC_kernel!(ws.dx, ws.x_bar, ws.SOC_var_idx, ws.number_SOC_var)
            end
            if ws.number_lu_x > 0
                @cuda threads = 256 blocks = ceil(Int, ws.number_lu_x / 256) compute_xlu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, ws.number_lu_x)
            end
            res.err_Rp_org_bar += LinearAlgebra.norm(ws.dx, Inf)
        end
        res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
        # res.KKTx_org_bar = max(res.err_Rp_org_bar, res.err_K_org_bar, res.err_Rd_org_bar, res.err_z_org_bar, res.err_Comp_org_bar) 

    end
end

function compute_residuals_gpu_unscale(ws::Union{HPRSOCP_workspace_gpu},
    socp::Union{SOCP_info_cpu,SOCP_info_gpu},
    socp_org::Union{SOCP_info_cpu,SOCP_info_gpu},
    sc::Union{Scaling_info_gpu},
    res::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    print_yes::Bool,
    check_iter::Int,
    iter::Int,
)
    res.is_updated = false

    if rem(iter, check_iter) == 0
        res.is_updated = true

        ws.dw .= sc.b_scale * (ws.w_bar ./ sc.col_norm)
        ws.dx .= sc.b_scale * (ws.x_bar ./ sc.col_norm)
        ws.dy .= sc.c_scale * (ws.y_bar ./ sc.row_norm)
        ws.dz .= sc.c_scale * (ws.z_bar .* sc.col_norm)


        ### obj
        # res.primal_obj = LinearAlgebra.dot(ws.c, ws.x_bar)
        CUSPARSE.mv!('N', 1, socp_org.Q, ws.dx, 0, ws.Qx, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        # CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        res.primal_obj_bar = 0.5 * LinearAlgebra.dot(ws.dx, ws.Qx) + LinearAlgebra.dot(socp_org.c, ws.dx) + socp.obj_constant

        res.dual_obj_bar = (LinearAlgebra.dot(ws.dy, socp_org.b)
                            -
                            0.5 * LinearAlgebra.dot(ws.dx, ws.Qx)
                            +
                            LinearAlgebra.dot(ws.dz, ws.dx)) + socp.obj_constant
        # res.dual_obj_bar = sc.b_scale * sc.c_scale * res.dual_obj + socp.obj_constant

        res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))


        ### Rd
        CUDA.CUSPARSE.mv!('N', 1, socp_org.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.dz, socp_org.c, ws.Qx, ws.Rd, ws.n)
        # res.err_Rd_bar = LinearAlgebra.norm(ws.Rd)

        res.err_Rd_org_bar = LinearAlgebra.norm(ws.Rd, Inf) / (1.0 + maximum([sc.norm_c_org, LinearAlgebra.norm(ws.ATdy, Inf), LinearAlgebra.norm(ws.Qx, Inf)]))

        ### Rp
        if ws.m > 0
            CUDA.CUSPARSE.mv!('N', 1, socp_org.A, ws.dx, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            @cuda threads = 256 blocks = ceil(Int, (ws.number_ineq + ws.number_eq) / 256) compute_Rp_kernel!(ws.Rp, socp_org.b, ws.Ax, ws.number_ineq, ws.number_eq, ws.SOC_con_idx, ws.number_SOC_con, ws.m)
            if ws.number_SOC_con > 0
                @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_con / 256) compute_Rp_SOC_kernel!(ws.Rp, socp_org.b, ws.Ax, ws.number_ineq, ws.number_eq, ws.SOC_con_idx, ws.number_SOC_con, ws.m)
            end

            # res.err_Rp_bar = LinearAlgebra.norm(ws.Rp)
            res.err_Rp_org_bar = LinearAlgebra.norm(ws.Rp, Inf) / (1.0 + max(sc.norm_b_org, LinearAlgebra.norm(ws.Ax, Inf)))
        end

        if iter == 0
            if ws.number_SOC_var > 0
                @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) compute_x_SOC_kernel!(ws.x_hat, ws.x_bar, ws.SOC_var_idx, ws.number_SOC_var)
            end
            if ws.number_lu_x > 0
                @cuda threads = 256 blocks = ceil(Int, ws.number_lu_x / 256) compute_xlu_kernel!(ws.x_hat, ws.x_bar, socp_org.l, socp_org.u, ws.number_lu_x)
            end
            res.err_Rp_org_bar += LinearAlgebra.norm(ws.x_hat, Inf)
        end
        res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
        # res.KKTx_org_bar = max(res.err_Rp_org_bar, res.err_K_org_bar, res.err_Rd_org_bar, res.err_z_org_bar, res.err_Comp_org_bar) 

    end
end


function update_sigma(params::HPRSOCP_parameters,
    restart_info::Union{HPRSOCP_restart,HPRSOCP_restart},
    iter::Int,
    sc::Union{Scaling_info_gpu},
    ws::Union{HPRSOCP_workspace_gpu},
    residuals::HPRSOCP_residuals,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
        if ws.m > 0
            axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
            axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.last_ATy, ws.ATdy, ws.n)
            CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
        axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        a = ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy) - 2 * LinearAlgebra.dot(ws.dQw, ws.ATdy)
        b = LinearAlgebra.dot(ws.dx, ws.dx)
        a += ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
        c = LinearAlgebra.dot(ws.ATdy, ws.QATdy)
        d = ws.lambda_max_Q
        a = max(a, 1e-12)
        b = max(b, 1e-12)
        if b > 1e-14 && a > 1e-14
            # if Q_is_diag
            #     sigma_new = golden_Q_diag(a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv; lo=1e-12, hi=1e12, tol=1e-13)
            # else
            # min a * x + b / x + c * x^2 / (1 + d * x)
            sigma_new = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
            # end
        else
            sigma_new = ws.sigma
        end
        fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
        ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))

        # fact = 1 - min(restart_info.current_gap / restart_info.weighted_norm, 1)
        # ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))

        # ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(max(residuals.err_Rd_bar, 1e-10) / max(residuals.err_Rp_bar, 1e-10)))

        # ws.sigma = min(max(ws.sigma, 1e-6), 1e6)
        # else
        # ws.sigma = 1
        # end
        # ws.sigma = exp(0.2 * log(sigma_new) + 0.8 * log(ws.sigma))

        # else
        # ws.sigma = 1.0
        # end


        # println(" primal_move = ", primal_move, " dual_move = ", dual_move)
    end

end

function check_restart(restart_info::Union{HPRSOCP_restart,HPRSOCP_restart},
    residuals::HPRSOCP_residuals,
    iter::Int,
    check_iter::Int,
    ws::Union{HPRSOCP_workspace_gpu},
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

function try_restart(restart_info, ws::Union{HPRSOCP_workspace_gpu}, iter::Int)
    if restart_info.restart_flag > 0
        # println()
        ws.x .= ws.x_bar
        ws.y .= ws.y_bar
        ws.w .= ws.w_bar
        ws.last_x .= ws.x_bar
        ws.last_y .= ws.y_bar
        ws.last_w .= ws.w_bar
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        ws.Qw .= ws.Qw_bar
        ws.last_Qw .= ws.Qw_bar
        ws.last_ATy .= ws.ATy
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf
        println("restart at current iter ", iter)
        restart_info.times += 1
        restart_info.inner = 0
    end
end

function check_break(residuals::HPRSOCP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRSOCP_parameters,
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
    ws::HPRSOCP_workspace_gpu,
    residuals::HPRSOCP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
)
    results = HPRSOCP_results()
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    results.sigma = ws.sigma
    ### copy the results to the CPU ### 
    results.w = Vector(sc.b_scale * (ws.w_bar ./ sc.col_norm))
    # println(" scaling col norm: ", sc.col_norm)
    # println(" sc.b_scale: ", sc.b_scale )
    results.x = Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y = Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z = Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))

    return results
end

function allocate_workspace_gpu(socp::SOCP_info_gpu,
    params::HPRSOCP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
    scaling_info::Scaling_info_gpu,
)
    ws = HPRSOCP_workspace_gpu()
    m, n = size(socp.A)
    ws.m = m
    ws.n = n
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
    else
        ws.sigma = params.sigma
    end
    println("initial sigma = ", ws.sigma)
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.w = CUDA.zeros(Float64, n)
    # ws.w = copy(socp.x0)
    ws.w_hat = CUDA.zeros(Float64, n)
    ws.w_bar = CUDA.zeros(Float64, n)
    # ws.w_bar = copy(socp.x0)
    ws.dw = CUDA.zeros(Float64, n)
    ws.x = CUDA.zeros(Float64, n)
    # ws.x = copy(socp.x0)
    ws.x_hat = CUDA.zeros(Float64, n)
    ws.x_bar = CUDA.zeros(Float64, n)
    # ws.x_bar = copy(socp.x0)
    ws.dx = CUDA.zeros(Float64, n)
    ws.y = CUDA.zeros(Float64, m)
    # ws.y = copy(socp.y0)
    ws.y_hat = CUDA.zeros(Float64, m)
    ws.y_bar = CUDA.zeros(Float64, m)
    # ws.y_bar = copy(socp.y0)
    ws.dy = CUDA.zeros(Float64, m)
    ws.s = CUDA.zeros(Float64, m)
    ws.s_org = CUDA.zeros(Float64, m)
    ws.z_bar = CUDA.zeros(Float64, n)
    ws.dz = CUDA.zeros(Float64, n)
    ws.Q = socp.Q
    ws.A = socp.A
    ws.AT = socp.AT
    ws.b = socp.b
    ws.c = socp.c
    ws.l = socp.l
    ws.u = socp.u
    ws.SOC_con_idx = socp.SOC_con_idx
    ws.SOC_var_idx = socp.SOC_var_idx
    ws.number_ineq = socp.number_ineq
    ws.number_eq = socp.number_eq
    ws.number_lu_x = socp.number_lu_x
    ws.number_SOC_con = length(ws.SOC_con_idx) - 1
    ws.number_SOC_var = length(ws.SOC_var_idx) - 1
    if ws.m > 0
        ws.l[ws.l.==-Inf] .= -1e20
        ws.u[ws.u.==Inf] .= 1e20
    end
    # ws.idxI = socp.idxI
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
    # ws.last_x = copy(socp.x0)
    # ws.last_y = copy(socp.y0)
    ws.last_w = CUDA.zeros(Float64, n)
    ws.last_ATy = CUDA.zeros(Float64, n)
    ws.update_z = false
    ws.tempv = CUDA.zeros(Float64, n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    ws.last_Qw = copy(ws.Qw)
    ws.Qw_bar .= copy(ws.Qw)
    ws.last_ATy = copy(ws.ATy)
    # ws.z_bar .= ws.Qw - ws.ATy + ws.c
    ws.fact1 = CUDA.zeros(Float64, n)
    ws.fact2 = CUDA.zeros(Float64, n)
    ws.fact = CUDA.zeros(Float64, n)
    ws.fact_M = CUDA.zeros(Float64, n)
    ws.SOC_con_scaling_factor = CUDA.zeros(Float64, ws.m)
    ws.SOC_con_scaling_factor_inv = CUDA.zeros(Float64, ws.m)
    ws.SOC_var_scaling_factor = CUDA.zeros(Float64, ws.n)
    ws.SOC_var_scaling_factor_inv = CUDA.zeros(Float64, ws.n)
    ws.SOC_con_scaling_factor .= scaling_info.SOC_con_scaling_factor
    ws.SOC_con_scaling_factor_inv .= scaling_info.SOC_con_scaling_factor_inv
    ws.SOC_var_scaling_factor .= scaling_info.SOC_var_scaling_factor
    ws.SOC_var_scaling_factor_inv .= scaling_info.SOC_var_scaling_factor_inv
    # Allocate temporary array for precomputed SOC norms (one per SOC variable)
    ws.SOC_norms_temp = CUDA.zeros(Float64, ws.number_SOC_var)
    println(sum(ws.SOC_con_scaling_factor .== 1.0))
    println(sum(ws.SOC_con_scaling_factor_inv .== 1.0))
    # println(Vector(ws.z_bar))
    return ws
end

function initialize_restart(params::HPRSOCP_parameters)
    restart_info = HPRSOCP_restart()
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
    params = HPRSOCP_parameters()
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


function solve(socp, scaling_info, params, socp_org, A_SOC, b_SOC)
    useGPU = params.useGPU

    m, n = size(socp.A)

    ### power iteration to estimate lambda_max ###

    # CUDA.synchronize()
    t_start_alg = time()

    # lambda_max = power_iteration_gpu2(socp.A, socp.AT) * params.lambda_factor
    # lambda_max = 1.0
    if m > 0
        lambda_max_A = power_iteration_A_gpu(socp.A, socp.AT) * params.lambda_factor
    else
        lambda_max_A = 0.0
    end
    if length(socp.Q.nzVal) > 0
        lambda_max_Q = power_iteration_Q_gpu(socp.Q) * params.lambda_factor
    else
        lambda_max_Q = 0.0
    end
    CUDA.synchronize()
    power_time = time() - t_start_alg
    println("lambda_max A = ", lambda_max_A)
    println("lambda_max Q = ", lambda_max_Q)

    ### Initialization ###
    residuals = HPRSOCP_residuals()

    restart_info = initialize_restart(params)

    # x0 = CUDA.zeros(Float64, n)
    # y0 = CUDA.zeros(Float64, m)
    # x0 .= copy(socp.x0)
    # y0 .= copy(socp.y0)

    # socp.x0 .= CUDA.zeros(Float64, n)
    # socp.y0 .= CUDA.zeros(Float64, m)

    
    ws = allocate_workspace_gpu(socp, params, lambda_max_A, lambda_max_Q, scaling_info)

    println("HPRSOCP solver starts...")
    println("iter    errRp     errRd     last_gap    current_gap     p_obj    d_obj    gap    sigma    time")

    if useGPU
        # # obsolete memory diagnostic omitted
    end

    AAT = A_SOC * A_SOC'
    SOC_len = length(b_SOC)
    dy = CUDA.zeros(Float64, size(AAT, 1))
    Rp = CUDA.zeros(Float64, size(AAT, 1))
    if params.mode == "direct"
        # if params.useGPU
        # Q_I = ws.Q + Diagonal(ones(ws.n) / ws.sigma |> CuVector)
        # solver = CudssSolver(Q_I, "SPD", 'F')
        # cudss("analysis", solver, ws.w_bar, ws.Qx)
        # cudss("factorization", solver, ws.w_bar, ws.Qx)

        if m > 0
            solver_AAT = CudssSolver(AAT, "G", 'F')
            cudss("analysis", solver_AAT, dy, Rp)
            cudss("factorization", solver_AAT, dy, Rp)
        else
            solver_AAT = 0
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

    # plot_ATy_error = []
    check_iter = params.check_iter

    # return
    for iter = 0:params.max_iter
        compute_residuals_gpu_unscale(ws, socp, socp_org, scaling_info, residuals, params, true, check_iter, iter)
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
        update_sigma(params, restart_info, iter, scaling_info, ws, residuals)



        ### restart if needed ###
        try_restart(restart_info, ws, iter)

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

        update_zxw1_gpu!(ws, Halpern_fact1, Halpern_fact2)

        # update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
        if params.mode == "direct"
            if ws.m > 0
                CUDA.CUSPARSE.mv!('N', 1, A_SOC, ws.x_hat, 0, Rp, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                Rp .= (b_SOC .- Rp) ./ ws.sigma
                cudss("solve", solver_AAT, dy, Rp)
                @cuda threads = 256 blocks = ceil(Int, SOC_len / 256) add_partial_vector!(ws.y_bar, ws.y, ws.last_y, ws.dy, dy, m - SOC_len + 1, m, Halpern_fact1, Halpern_fact2)
                # ws.y_bar .= ws.dy .+ ws.y
                # ws.y_hat .= 2 * ws.y_bar .- ws.y
                # ws.y .= Halpern_fact1 * ws.last_y .+ Halpern_fact2 * ws.y_hat
            else
                ws.y_hat .= 2 * ws.y_bar .- ws.y
            end
        else
            update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
        end

        update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)


        if restart_info.restart_flag > 0
            restart_info.last_gap = compute_M_norm_gpu2!(ws)
            # M_c = compute_M_norm_gpu!(ws, socp.Q_is_diag, socp.noC)
        end
        if rem(iter + 1, check_iter) == 0
            restart_info.current_gap = compute_M_norm_gpu2!(ws)
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

function generate_sqrt_lasso_HPRSOCP(m, n, seed=1)
    Random.seed!(seed)

    println("Generating a random SOCP problem for sqrt lasso with m = $m, n = $n...")
    t_start = time()

    A = sprand(m, n, 1e-3)  # Sparse matrix with 10% density
    x0 = randn(n)            # Random vector of size n
    # generate random permuation of indices
    perm = randperm(n)
    x0[perm[1:Int(ceil(n / 2))]] .= 0
    b = A * x0 .+ 1e-6  # Generate b with some noise

    lambda = norm(A' * b, Inf)

    Inn = spdiagm(ones(n))  # Identity matrix of size n
    On1 = sparse(zeros(n, 1))
    O1n = sparse(zeros(1, n))
    A_0 = vcat(hcat(On1, Inn, Inn), hcat(On1, -Inn, Inn))
    A_1 = vcat(hcat(1, O1n, O1n), hcat(sparse(zeros(m, 1)), A, sparse(zeros(m, n))))

    A_new = vcat(A_0, A_1)
    c = vcat(1, zeros(n), lambda * ones(n))
    b_new = vcat(zeros(size(A_0, 1)), 0, b)

    @assert size(A_new, 1) == size(b_new, 1) "The number of rows in A must match the number of elements in b"
    @assert size(A_new, 2) == size(c, 1) "The number of columns in A must match the number of elements in c"

    number_eq = 0
    number_ineq = size(A_0, 1)
    SOC_con_idx = vcat(number_ineq + 1, number_ineq + 1 + size(A_1, 1))

    number_lu_x = length(c)
    l = -Inf * ones(length(c))
    u = Inf * ones(length(c))
    SOC_var_idx = [number_lu_x + 1]

    obj_constant = 0.0
    Q = sparse([], [], [], length(c), length(c))

    println("Problem generation completed in $(time() - t_start) seconds.")
    return SOCP_info_cpu(Q, c, A_new, A_new', b_new, SOC_con_idx, number_ineq, number_eq, l, u, SOC_var_idx, number_lu_x, obj_constant)
end

function run_file(FILE_NAME, params, x_path="", y_path="")
    # io = open(FILE_NAME)
    # socp = Logging.with_logger(Logging.NullLogger()) do
    #     readqps(io, mpsformat=:free)
    # end
    # close(io)
    # standard_socp = read_cbf(FILE_NAME)
    # return standard_socp

    Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps_SCS(FILE_NAME)
    standard_socp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)

    # standard_socp = generate_sqrt_lasso_HPRSOCP(5000, 10000)
    # println(standard_socp)
    SOC_idx = standard_socp.SOC_con_idx[1]:standard_socp.SOC_con_idx[end]-1
    # A_SOC = standard_socp.A[socp.SOC_con_idx[1]:socp.SOC_con_idx[end] - 1, :]
    # AAT = A_SOC * A_SOC'
    # return standard_socp
    # socp_formulation_SOC_Axb!(standard_socp)
    # return standard_socp

    socp_org = deepcopy(standard_socp)
    # return socp_org
    # return standard_socp
    # noC = true
    # standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
    # return standard_qp
    # standard_qp.Q .*= 0
    # standard_qp.l .= zeros(Float64, length(standard_qp.l))
    # return standard_qp 

    # return standard_qp

    # return standard_qp

    # scaling_info = scaling!(standard_qp, true, true, true)

    # return 
    # print norm b, norm c
    if params.useGPU
        scaling_info = scaling!(standard_socp, false, false, false, false, x_path, y_path)
        # return socp_org
        A_SOC = standard_socp.A[SOC_idx, :]
        b_SOC = standard_socp.b[SOC_idx]
        AAT = A_SOC * A_SOC'
        # return scaling_info
        # print("maximum nonzero value in A: ", maximum(abs.(standard_qp.A.nzval)), "\n")
        # print("minimum nonzero value in A: ", minimum(abs.(standard_qp.A.nzval)), "\n")
        # print("maximum nonzero value in Q: ", maximum(abs.(standard_qp.Q.nzval)), "\n")
        # print("minimum nonzero value in Q: ", minimum(abs.(standard_qp.Q.nzval)), "\n")
        # return 
        # standard_qp.Q_is_diag = false
        # return standard_qp, scaling_info
        standard_scop_gpu = SOCP_info_gpu(
            CuSparseMatrixCSR(standard_socp.Q),
            CuVector(standard_socp.c),
            CuSparseMatrixCSR(standard_socp.A),
            CuSparseMatrixCSR(standard_socp.AT),
            CuVector(standard_socp.b),
            CuVector(standard_socp.SOC_con_idx),
            standard_socp.number_ineq,
            standard_socp.number_eq,
            CuVector(standard_socp.l),
            CuVector(standard_socp.u),
            CuVector(standard_socp.SOC_var_idx),
            standard_socp.number_lu_x,
            standard_socp.obj_constant
        )
        socp_org_gpu = SOCP_info_gpu(
            CuSparseMatrixCSR(socp_org.Q),
            CuVector(socp_org.c),
            CuSparseMatrixCSR(socp_org.A),
            CuSparseMatrixCSR(socp_org.AT),
            CuVector(socp_org.b),
            CuVector(socp_org.SOC_con_idx),
            socp_org.number_ineq,
            socp_org.number_eq,
            CuVector(socp_org.l),
            CuVector(socp_org.u),
            CuVector(socp_org.SOC_var_idx),
            socp_org.number_lu_x,
            socp_org.obj_constant
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
            scaling_info.norm_c_org,
            CuVector(scaling_info.SOC_con_scaling_factor),
            CuVector(scaling_info.SOC_con_scaling_factor_inv),
            CuVector(scaling_info.SOC_var_scaling_factor),
            CuVector(scaling_info.SOC_var_scaling_factor_inv))
    else
        scaling_info = scaling!(standard_socp, true, false, false, true)
    end

    # return standard_qp

    # params = initialize_parameters()
    # params.max_iter = iter
    # params.time_limit = time_limit
    # params.stoptol = stoptol
    # params.useGPU = useGPU
    # params.sigma_fixed = false
    # params.restart = -1
    # params.mode = "linearize"
    # params.sigma_method = 0


    # norm_b = scaling_info.norm_b
    # norm_c = scaling_info.norm_c
    # println("norm_b = ", norm_b)
    # println("norm_c = ", norm_c)
    # if norm_c > 1e-16 && norm_b > 1e-16 && norm_b < 1e16 && norm_c < 1e16
    #     params.sigma = norm_b / norm_c
    # else
    #     params.sigma = 1.0
    # end
    # # params.sigma = 8e5
    # println("initial sigma = ", params.sigma)



    if params.useGPU
        results = solve(standard_scop_gpu, scaling_info_gpu, params, socp_org_gpu, CuSparseMatrixCSR(A_SOC), CuVector(b_SOC))
    else
        results = solve(standard_qp, scaling_info, params)
    end

    return results

end

function run_TR(Q, A, b, c, params, x_path="", y_path="")

    # Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps_SCS(FILE_NAME)
    
    standard_socp = SOCP_info_cpu(SparseMatrixCSC(Q), copy(c), SparseMatrixCSC(A), SparseMatrixCSC(A)', copy(b), [size(A, 1) + 1], 0, size(A,1), -Inf * ones(0), Inf * ones(0), [1;length(c) + 1], 0, 0.0)

    SOC_idx = standard_socp.SOC_con_idx[1]:standard_socp.SOC_con_idx[end]-1

    
    socp_org = deepcopy(standard_socp)
    
    # return 
    # print norm b, norm c
    println(" useGPU = ", params.useGPU)
    if params.useGPU
        scaling_info = scaling!(standard_socp, true, false, false, true, x_path, y_path)
        # return socp_org
        A_SOC = standard_socp.A[SOC_idx, :]
        b_SOC = standard_socp.b[SOC_idx]
        standard_socp_gpu = SOCP_info_gpu(
            CuSparseMatrixCSR(standard_socp.Q),
            CuVector(standard_socp.c),
            CuSparseMatrixCSR(standard_socp.A),
            CuSparseMatrixCSR(standard_socp.AT),
            CuVector(standard_socp.b),
            CuVector(standard_socp.SOC_con_idx),
            standard_socp.number_ineq,
            standard_socp.number_eq,
            CuVector(standard_socp.l),
            CuVector(standard_socp.u),
            CuVector(standard_socp.SOC_var_idx),
            standard_socp.number_lu_x,
            standard_socp.obj_constant
        )
        socp_org_gpu = SOCP_info_gpu(
            CuSparseMatrixCSR(socp_org.Q),
            CuVector(socp_org.c),
            CuSparseMatrixCSR(socp_org.A),
            CuSparseMatrixCSR(socp_org.AT),
            CuVector(socp_org.b),
            CuVector(socp_org.SOC_con_idx),
            socp_org.number_ineq,
            socp_org.number_eq,
            CuVector(socp_org.l),
            CuVector(socp_org.u),
            CuVector(socp_org.SOC_var_idx),
            socp_org.number_lu_x,
            socp_org.obj_constant
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
            scaling_info.norm_c_org,
            CuVector(scaling_info.SOC_con_scaling_factor),
            CuVector(scaling_info.SOC_con_scaling_factor_inv),
            CuVector(scaling_info.SOC_var_scaling_factor),
            CuVector(scaling_info.SOC_var_scaling_factor_inv))
        
        println("norm b after scaling: ", scaling_info.norm_b)
        println("norm c after scaling: ", scaling_info.norm_c)
    else
        scaling_info = scaling!(standard_socp, true, false, false, true)
    end


    if params.useGPU
        results = solve(standard_socp_gpu, scaling_info_gpu, params, socp_org_gpu, CuSparseMatrixCSR(A_SOC), CuVector(b_SOC))
    else
        results = solve(standard_qp, scaling_info, params)
    end

    return results

end

function load_trust_region_problem(filename::String)
    # Add .jld2 extension if not present
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end
    
    data = load(filename)
    
    # Fix b if it's a scalar (old format) - convert to vector
    b = data["b"]
    if isa(b, Number)
        b = [b]
        println("Note: Converted scalar b to vector [b]")
    end
    
    println("Loaded trust region problem from ", filename)
    println("  Problem size: ", size(data["A"], 1), " equality constraints, ", size(data["A"], 2), " variables")
    println("  H is sparse: ", issparse(data["H"]), ", size = ", size(data["H"]))
    if issparse(data["H"])
        println("    nnz(H) = ", nnz(data["H"]))
    end
    println("  A is sparse: ", issparse(data["A"]), ", size = ", size(data["A"]))
    if issparse(data["A"])
        println("    nnz(A) = ", nnz(data["A"]))
    end
    
    return (data["H"], data["c"], data["A"], b)
end

function run_dataset(data_path, result_path, params, device_number=0)
    current_file = @__FILE__
    cp(current_file, result_path * "HPRSOCP.jl", force=true)
    CUDA.device!(device_number)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRSOCP_zxwyw_result.csv"

    # redirect the output to a file
    log_path = result_path * "HPRSOCP_log.txt"

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


    skip_files = ["LISWET1.SOCPS", "LISWET10.SOCPS", "LISWET11.SOCPS", "LISWET12.SOCPS", "LISWET2.SOCPS", "LISWET7.SOCPS", "LISWET8.SOCPS", "LISWET9.SOCPS", "YAO.SOCPS"]
    warmupdone = false
    for i = 1:length(files)
        file = files[i]
        if file in skip_files
            # continue
        end
        if (occursin(".mat", file) && !(file in namelist))
            FILE_NAME = data_path * file
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
            # println(file)
            # Load the problem data
#            H, c, A, b = load_trust_region_problem(FILE_NAME)
	    matfile = matopen(FILE_NAME)
H = read(matfile, "H")  # SparseMatrixCSC{Float64,Int}
c = read(matfile, "c")  # Vector{Float64}
c = c[:,1]  # Convert to 1D vector
A = read(matfile, "A")  # SparseMatrixCSC{Float64,Int}
b = read(matfile, "b")  # Float64 (scalar)
b = [b]  # Convert to 1D vector
close(matfile)
            # Run the solver

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                if params.do_warm_up && !warmupdone
                    warmupdone = true
                    max_iter = params.max_iter
                    params.max_iter = 200
                    println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                    t_start_all = time()
                    # results = run_file(FILE_NAME, params)
                    results = run_TR(H, A, b, c, params)
                    params.max_iter = max_iter
                    all_time = time() - t_start_all
                    println("warm up time: ", all_time)
                    println("warm up ends ----------------------------------------------------------------------------------------------------------")
                end


                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                # results = run_file(FILE_NAME, params)
                results = run_TR(H, A, b, c, params)
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

function check_dataset(data_path)
    # check the data path, save the valid file names into a csv file
    if !isdir(data_path)
        error("data path does not exist: ", data_path)
    end
    files = readdir(data_path)
    println("total files: ", length(files))
    valid_files = []
    nrow_list = []
    ncol_list = []
    nnzA_list = []
    number_soc_con_list = []
    min_soc_con_len_list = []
    max_soc_con_len_list = []
    total_soc_con_len_list = []
    number_soc_var_list = []
    min_soc_var_len_list = []
    max_soc_var_len_list = []
    total_soc_var_len_list = []


    problem = []
    for i = 1:length(files)
        file = files[i]
        if !occursin(".cbf", file)
            continue
        end
        println("check the ", i, "th file: ", file)
        try
            problem = my_read_cbf(data_path * file)
        catch e
            println("Error reading file ", file, ": ", e)
            continue
        end
        if !isempty(problem.dcoord) || !isempty(problem.hcoord) || !isempty(problem.psdcon)
            println("The CBF file contains PSD constraints, which is not supported.")
            continue
        end
        if !isempty(problem.fcoord) || !isempty(problem.psdvar)
            println("The CBF file contains matrix variables, which is not supported.")
            continue
        end
        if !isempty(problem.intlist)
            println("The CBF file contains integer constraints, which is not supported.")
            continue
        end
        support_cones = ["L+", "L-", "L=", "Q", "F"]
        support = true
        for con in problem.con
            if !(con[1] in support_cones)
                println("The CBF file contains unsupported constraint type: ", con[1])
                support = false
                break
            end
        end
        if !support
            continue
        end

        support = true
        for var in problem.var
            if !(var[1] in support_cones)
                println("The CBF file contains unsupported variable type: ", var[1])
                support = false
                break
            end
        end
        if !support
            continue
        end

        push!(valid_files, file)
        m = sum(getindex.(problem.con, 2))
        @assert problem.nconstr == m
        n = sum(getindex.(problem.var, 2))
        @assert problem.nvar == n

        push!(nrow_list, m)
        push!(ncol_list, n)

        nnzA = length(problem.acoord)
        push!(nnzA_list, nnzA)

        number_soc_con = 0
        min_soc_con_len = Inf
        max_soc_con_len = 0
        total_soc_con_len = 0
        # Ax - b ∈ SOC 
        for i in 1:length(problem.con)
            if problem.con[i][1] == "Q"
                number_soc_con += 1
                len = problem.con[i][2]
                total_soc_con_len += len
                if len < min_soc_con_len
                    min_soc_con_len = len
                end
                if len > max_soc_con_len
                    max_soc_con_len = len
                end
            end
        end
        push!(number_soc_con_list, number_soc_con)
        push!(min_soc_con_len_list, min_soc_con_len)
        push!(max_soc_con_len_list, max_soc_con_len)
        push!(total_soc_con_len_list, total_soc_con_len)

        number_soc_var = 0
        min_soc_var_len = Inf
        max_soc_var_len = 0
        total_soc_var_len = 0
        # SOC variables
        for i in 1:length(problem.var)
            if problem.var[i][1] == "Q"
                number_soc_var += 1
                len = problem.var[i][2]
                total_soc_var_len += len
                if len < min_soc_var_len
                    min_soc_var_len = len
                end
                if len > max_soc_var_len
                    max_soc_var_len = len
                end
            end
        end
        push!(number_soc_var_list, number_soc_var)
        push!(min_soc_var_len_list, min_soc_var_len)
        push!(max_soc_var_len_list, max_soc_var_len)
        push!(total_soc_var_len_list, total_soc_var_len)



        csv_file = "mycbf_valid_files.csv"

        result_table = DataFrame(
            file=valid_files,
            nrow=nrow_list,
            ncol=ncol_list,
            nnzA=nnzA_list,
            number_soc_con=number_soc_con_list,
            min_soc_con_len=min_soc_con_len_list,
            max_soc_con_len=max_soc_con_len_list,
            total_soc_con_len=total_soc_con_len_list,
            number_soc_var=number_soc_var_list,
            min_soc_var_len=min_soc_var_len_list,
            max_soc_var_len=max_soc_var_len_list,
            total_soc_var_len=total_soc_var_len_list
        )
        CSV.write(csv_file, result_table)
    end
end

CUDA.@fastmath @inline function update_zxw1_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-12
        #     tmp = 0.0
        # end
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
    end
    return
end

function update_zxw1_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.Qw .= ws.Q * ws.w
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
    # println(Vector(ws.last_x))
    # println(Vector(ws.x_bar))
    # println(Vector(ws.x_bar - ws.last_x))
end

CUDA.@fastmath @inline function update_zxw1_diagQ_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-12
        #     tmp = 0.0
        # end
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
    end
    return
end

function update_zxw1_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    # fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_diagQ_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
    # println(Vector(ws.last_x))
    # println(Vector(ws.x_bar))
    # println(Vector(ws.x_bar - ws.last_x))
end


CUDA.@fastmath @inline function cust_update_zxw1_kernel!(dx, rowPtrQ, colValQ, nzValQ, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w[colValQ[k]]
        end
        Qw[i] = qr1
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-12
        #     tmp = 0.0
        # end
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
    end
    return
end

function cust_update_zxw1_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw_hat, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_zxw1_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)

    # if (LinearAlgebra.norm(ws.Qw_hat - ws.Qw) > 1e-11)
    #     error("Qw_hat: ", LinearAlgebra.norm(ws.Qw_hat - ws.Qw))
    # end
end



CUDA.@fastmath @inline function cust_update_zxw1_diagQ_kernel!(dx, rowPtrQ, colValQ, nzValQ, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w[colValQ[k]]
        end
        Qw[i] = qr1
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-12
        #     tmp = 0.0
        # end
        z_bar[i] = x[i] + sigma * tmp
        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i]
        w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
    end
    return
end

function cust_update_zxw1_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    # fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_zxw1_diagQ_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end



function cust_update_w1_kernel!(x_hat, w, w_bar, sigma, tempv, fact1, fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function cust_update_w1_gpu!(ws::HPRQP_workspace_gpu)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x_hat, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w1_kernel!(ws.x_hat, ws.w, ws.w_bar, ws.sigma, ws.tempv, fact1, fact2, ws.n)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_compute_r2_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.Qw, ws.sigma, ws.x_hat, ws.tempv, ws.n)
end

function cust_compute_r2_kernel!(rowPtrQ, colValQ, nzValQ, w_bar, Qw, sigma, x_hat, tempv, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        tempv[i] = x_hat[i] + sigma * (Qw[i] - qr1)
    end
    return
end

function cust_update_w1_diagQ_kernel!(x_hat, w, w_bar, sigma, tempv, fact1, fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function cust_update_w1_diagQ_gpu!(ws::HPRQP_workspace_gpu)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x_hat, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w1_diagQ_kernel!(ws.x_hat, ws.w, ws.w_bar, ws.sigma, ws.tempv, ws.fact1, ws.fact2, ws.n)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_compute_r2_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.Qw, ws.sigma, ws.x_hat, ws.tempv, ws.n)
end

# function update_w1_kernel!(w, x_hat, w_bar, sigma, tempv, fact1, fact2, n)
#     i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
#     if i <= n
#         w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
#         # tempv[i] = tempv[i] + sigma * (w[i] - w_bar[i])
#     end
#     return
# end

# function update_w1_gpu!(ws::HPRQP_workspace_gpu)
#     # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x_hat, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#     fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
#     fact1 = 1.0 - fact2
#     # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w1_kernel!(ws.w, ws.x_hat, ws.w_bar, ws.sigma, ws.tempv, fact1, fact2, ws.n)
#     CUDA.CUSPARSE.mv!('N', -ws.sigma, ws.Q, ws.w_bar, 1.0, ws.tempv, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#     axpby_gpu!(1.0, ws.tempv, ws.sigma, ws.Qw, ws.tempv, ws.n)
# end

# function update_w1_diagQ_kernel!(w, x_hat, w_bar, sigma, tempv, fact1, fact2, n)
#     i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
#     if i <= n
#         w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
#         # tempv[i] = tempv[i] + sigma * (w[i] - w_bar[i])
#     end
#     return
# end

# function update_w1_diagQ_gpu!(ws::HPRQP_workspace_gpu)
#     # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.x_hat, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#     # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w1_diagQ_kernel!(ws.w, ws.x_hat, ws.w_bar, ws.sigma, ws.tempv, ws.fact1, ws.fact2, ws.n)
#     CUDA.CUSPARSE.mv!('N', -ws.sigma, ws.Q, ws.w_bar, 1.0, ws.tempv, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#     axpby_gpu!(1.0, ws.tempv, ws.sigma, ws.Qw, ws.tempv, ws.n)
# end


function update_w2_kenel!(dw, ATdy, w, w_bar, last_w, last_ATy, ATy, ATy_bar, fact, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = w_bar[i] + fact * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function Halpern_Qw_kernel!(Qw, Qw_bar, last_Qw, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

function update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.ATy, ws.ATy, ws.n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATy, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w2_kenel!(ws.dw, ws.ATdy, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q), Halpern_fact1, Halpern_fact2, ws.n)
    # CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # axpby_gpu!(Halpern_fact1, ws.last_Qw, 2*Halpern_fact2, ws.Qw_bar, ws.Qw, ws.n)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) Halpern_Qw_kernel!(ws.Qw, ws.Qw_bar, ws.last_Qw, Halpern_fact1, Halpern_fact2, ws.n)
end


function update_w2_diagQ_kenel!(dw, ATdy, w, w_bar, last_w, last_ATy, ATy, ATy_bar, fact, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = w_bar[i] + fact[i] * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function update_w2_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.ATy, ws.ATy, ws.n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATy, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w2_diagQ_kenel!(ws.dw, ws.ATdy, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.fact, Halpern_fact1, Halpern_fact2, ws.n)
    # CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # axpby_gpu!(Halpern_fact1, ws.last_Qw, 2*Halpern_fact2, ws.Qw_bar, ws.Qw, ws.n)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) Halpern_Qw_kernel!(ws.Qw, ws.Qw_bar, ws.last_Qw, Halpern_fact1, Halpern_fact2, ws.n)
end




CUDA.@fastmath @inline function compute_ATy_kernel!(rowPtrAT, colValAT, nzValAT, y, ATy, ATdy, ATy_bar, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy_bar[i] = sAT
        ATdy[i] = sAT - ATy[i]
    end
    return
end

CUDA.@fastmath @inline function cust_update_w2_kernel!(dw, ATdy, rowPtrAT, colValAT, nzValAT, y_bar, w, w_bar, last_w, last_ATy, ATy, ATy_bar, fact, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_bar[colValAT[k]]
        end
        ATy_bar[i] = sAT
        # ATdy[i] = sAT - ATy[i]

        w_bar[i] = w_bar[i] + fact * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function cust_update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_ATy_kernel!(ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar, ws.ATy, ws.ATdy, ws.ATy_bar, ws.n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATy, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w2_kernel!(ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar,
        ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q), Halpern_fact1, Halpern_fact2, ws.n)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_Halpern_Qw_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.w_bar, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function cust_Halpern_Qw_kernel!(rowPtrQ, colValQ, nzValQ, Qw, Qw_bar, last_Qw, w_bar, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qx = 0.0
        @inbounds for k in startQ:stopQ
            qx += nzValQ[k] * w_bar[colValQ[k]]
        end
        Qw_bar[i] = qx
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

CUDA.@fastmath @inline function cust_update_w2_diagQ_kernel!(dw, ATdy, rowPtrAT, colValAT, nzValAT, y_bar, w, w_bar, last_w, last_ATy, ATy, ATy_bar, fact, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_bar[colValAT[k]]
        end
        ATy_bar[i] = sAT
        # ATdy[i] = sAT - ATy[i]

        w_bar[i] = w_bar[i] + fact[i] * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function cust_update_w2_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_ATy_kernel!(ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar, ws.ATy, ws.ATdy, ws.ATy_bar, ws.n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATy, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w2_diagQ_kernel!(ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar,
        ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.fact, Halpern_fact1, Halpern_fact2, ws.n)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_Halpern_Qw_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.w_bar, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_kernel!(dy, y_bar, y, last_y, s, AL, AU, Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, m)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        s[i] = Ax[i] - fact1 * y[i]
        y_bar[i] = s[i] < AL[i] ? (AL[i] - s[i]) : (s[i] > AU[i] ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    CUDA.CUSPARSE.mv!('N', 1.0, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.tempv .+= ws.Qw_bar
    axpby_gpu!(1.0, ws.tempv, -ws.sigma, ws.Qw_bar, ws.tempv, ws.n)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
end

CUDA.@fastmath @inline function cust_update_y_kernel!(dy, rowPtrA, colValA, nzValA, tempv, y_bar, y, last_y, s, AL, AU, fact1, fact2, Halpern_fact1, Halpern_fact2, m)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        startA = rowPtrA[i]
        stopA = rowPtrA[i+1] - 1
        Ai = 0.0
        @inbounds for k in startA:stopA
            Ai += nzValA[k] * tempv[colValA[k]]
        end
        s[i] = Ai - fact1 * y[i]
        y_bar[i] = s[i] < (AL[i]) ? (AL[i] - s[i]) : (s[i] > (AU[i]) ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        # y_bar[i] = y[i] + fact2 * (AL[i] - Ai)
        # y_bar[i] = y_bar[i] < 1e-8 ? 0.0 : y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function cust_update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    # CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_compute_r2_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.Qw, ws.sigma, ws.x_hat, ws.tempv, ws.n)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.tempv, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.lambda_max_A * ws.sigma, 1 / (ws.lambda_max_A * ws.sigma), Halpern_fact1, Halpern_fact2, ws.m)
end


function compute_Rd_kernel!(ATy, z, c, Qw, Rd, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qw[i] + c[i] - ATy[i] - z[i]
    end
    return
end

function compute_Rd_gpu!(ws)
    # mul!(ws.ATy, ws.AT, ws.y_bar) 
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, ws.n)
end

function compute_Rd_noC_kernel!(ATy, c, Qw, Rd, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qw[i] + c[i] - ATy[i]
    end
    return
end

function compute_Rd_noC_gpu!(ws)
    # mul!(ws.ATy, ws.AT, ws.y_bar) 
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_noC_kernel!(ws.ATdy, ws.c, ws.Qx, ws.Rd, ws.n)
end

function compute_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    m1::Int, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= m1
        Rp[i] = AU[i] - Ax[i]
    elseif i <= m
        Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : (Ax[i] > AU[i] ? (AU[i] - Ax[i]) : 0.0)
    end
    return
end

function compute_Rp_gpu!(ws)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, ws.m1, ws.m)
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



CUDA.@fastmath @inline function update_w_noC_kernel!(dw, x, w, w_bar, last_w, ATy, Qw, c, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end



CUDA.@fastmath @inline function cust_update_w_noC_kernel!(dw, rowPtrAT, colValAT, nzValAT, y, x, w, w_bar, last_w, ATy, Qw, c, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function cust_update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end


CUDA.@fastmath @inline function cust_update_w_noC_diagQ_kernel!(dw, rowPtrAT, colValAT, nzValAT, y, x, w, w_bar, last_w, ATy, Qw, c, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function cust_update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    # fact1 = 1.0 - fact2
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_diagQ_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end




CUDA.@fastmath @inline function update_w_noC_diagQ_kernel!(dw, x, w, w_bar, last_w, ATy, Qw, c, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    # fact1 = 1.0 - fact2
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_diagQ_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function update_x_noC_kernel!(dx, x_bar, x_hat, tempv, last_x, x, Qw, Qw_bar, last_Qw, sigma, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])

        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_x_noC_kernel!(ws.dx, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end



CUDA.@fastmath @inline function cust_update_x_noC_kernel!(dx, rowPtrQ, colValQ, nzValQ, w_bar, x_bar, x_hat, tempv, last_x, x, Qw, Qw_bar, last_Qw, sigma, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        Qw_bar[i] = qr1

        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        dx[i] = x_bar[i] - x[i]
        x_hat[i] = 2 * x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])

        # tempv[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
    end
    return
end

function cust_update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_x_noC_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end






function update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)

end

function cust_update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    # CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.x_hat, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
end


function compute_M_norm_gpu!(ws::HPRQP_workspace_gpu, Q_is_diag::Bool, noC::Bool)
    if noC
        M_norm = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
        M_norm += 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        M_norm += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
        if !Q_is_diag
            M_norm += ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
            M_norm -= ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
        end
    else
        M_norm = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        M_norm += 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
        M_norm -= 2 * ws.sigma * LinearAlgebra.dot(ws.dQw, ws.ATdy)
        M_norm += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
        M_norm -= 2 * LinearAlgebra.dot(ws.dQw, ws.dx)

        if Q_is_diag
            ws.ATdy .*= ws.fact_M
            M_norm += LinearAlgebra.dot(ws.ATdy, ws.QATdy)
        else
            M_norm += (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * LinearAlgebra.dot(ws.ATdy, ws.QATdy)
        end

        if Q_is_diag
            M_norm += ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
        else
            M_norm += ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
        end
    end
    # M_norm = max(1e-24, M_norm)
    return sqrt(M_norm)
end


# function compute_M_norm_gpu2!(ws::HPRQP_workspace_gpu, Q_is_diag::Bool, noC::Bool)
#     if noC
#         M_1 = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
#         M_1 += 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
#         CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         M_1 += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
#         if !Q_is_diag
#             M_2 = ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
#             M_2 -= 2 * ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
#             M_norm = max(M_2, 0) + max(M_1, 0)
#             if min(M_1, M_2) < -1e-8
#                 println("M_1 = $M_1,M_2 = $M_2")
#             end
#         end
#     else
#         M_norm = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
#         CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
#         M_norm += 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
#         M_norm -= 2 * ws.sigma * LinearAlgebra.dot(ws.dQw, ws.ATdy)
#         M_norm += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
#         M_norm -= 2 * LinearAlgebra.dot(ws.dQw, ws.dx)

#         if Q_is_diag
#             ws.ATdy .*= ws.fact_M
#             M_norm += LinearAlgebra.dot(ws.ATdy, ws.QATdy) # sGS term
#             M_norm += ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
#         else
#             M_norm += (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * LinearAlgebra.dot(ws.ATdy, ws.QATdy)  # sGS term
#             M_norm += ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
#         end
#         # M_2 = M_1

#         # M_norm = M_1
#         # M_norm = max(M_2, 0) + max(M_3, 0)
#         # if min(M_1, M_2, M_3) < -1e-8
#         #     println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3")
#         # end
#     end
#     return sqrt(M_norm)
# end




function compute_M_norm_gpu2!(ws::HPRQP_workspace_gpu, Q_is_diag::Bool, noC::Bool)
    if noC
        M_1 = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
        M_1 += 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        M_1 += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
        M_2 = 0
        if !Q_is_diag
            M_2 = ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
            M_2 -= ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
        end
        M_norm = max(M_2, 0) + max(M_1, 0)
        if min(M_1, M_2) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2")
        end
    else
        M_1 = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        M_1 -= 2 * ws.sigma * LinearAlgebra.dot(ws.dQw, ws.ATdy)
        M_2 = 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
        M_2 += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
        M_2 -= 2 * LinearAlgebra.dot(ws.dQw, ws.dx)
        if Q_is_diag
            ws.ATdy .*= ws.fact_M
            M_3 = LinearAlgebra.dot(ws.ATdy, ws.QATdy) # sGS term
            M_1 += ws.sigma * LinearAlgebra.dot(ws.dQw, ws.dQw)
        else
            M_3 = (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * LinearAlgebra.dot(ws.ATdy, ws.QATdy)  # sGS term
            M_1 += ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
        end
        M_2 += max(M_1, 0)
        # M_norm = M_2 + M_3
        M_norm = max(M_2, 0) + max(M_3, 0)
        if min(M_1, M_2, M_3) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3")
        end
    end
    return sqrt(M_norm)
end



@inline function f_dev(x, a, b, c, d)
    return a * x + b / x + c * x^2 / (1 + d * x)
end

function golden_kernel!(
    a_p::Float64, b_p::Float64, c_p::Float64, d_p::Float64,
    φ::Float64,
    lo::Float64, hi::Float64,
    tol::Float64, maxiter::Int32
)
    a = lo
    b = hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    f_c = f_dev(c, a_p, b_p, c_p, d_p)
    f_d = f_dev(d, a_p, b_p, c_p, d_p)

    for i in 1:maxiter
        if f_d < f_c
            a, c, f_c = c, d, f_d
            d = a + φ * (b - a)
            f_d = f_dev(d, a_p, b_p, c_p, d_p)
        else
            b, d, f_d = d, c, f_c
            c = b - φ * (b - a)
            f_c = f_dev(c, a_p, b_p, c_p, d_p)
        end
        if (b - a) < tol
            break
        end
    end

    x_sol = 0.5 * (a + b)
    return x_sol
end

function golden(
    a::Float64, b::Float64, c::Float64, d::Float64;
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200
)
    # golden ratio constant
    φ = (sqrt(5.0) - 1.0) / 2.0

    maxiter32 = Int32(maxiter)

    x = golden_kernel!(
        a, b, c, d,
        φ,
        lo, hi,
        tol, maxiter32
    )
    return x
end



# Golden-section search for minimizing 
# f(x) = a*x + b/x + x^2 * dot(c, (I + x*Q) \ d)
# GPU‑enabled golden‐section search for 
# f(x) = a*x + b/x + x^2 * dot(c, d ./ (1 + x*Q))
function golden_Q_diag(a::Float64, b::Float64, Q::CuArray{Float64}, c::CuArray{Float64}, d::CuArray{Float64}, tempv::CuArray{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0



    # Objective using GPU operations, reusing tempv
    function f_gpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * LinearAlgebra.dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_gpu(x1)
    f2 = f_gpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_gpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_gpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end



using CUDA

#############################
# CUDA kernel to update all four factors in one pass
#############################
function update_Q_factors_kernel!(
    fact2::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64},
    fact1::CuDeviceVector{Float64},
    fact_M::CuDeviceVector{Float64},
    diag_Q::CuDeviceVector{Float64},
    sigma::Float64,
    s2::Float64,
    N::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

#############################
# High-level wrapper to launch the above kernel
#############################
function update_Q_factors_gpu!(
    fact2::CuVector{Float64},
    fact::CuVector{Float64},
    fact1::CuVector{Float64},
    fact_M::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    threads = 256
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks update_Q_factors_kernel!(
        fact2, fact, fact1, fact_M,
        diag_Q, sigma, s2, N
    )
    return
end
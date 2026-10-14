CUDA.@fastmath @inline function update_zxw1_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-15
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

CUDA.@fastmath @inline function update_zxw1_SOC_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_var, SOC_var_idx)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1
        norm_s = 0.0
        z_bar[start] = x[start] + sigma * (-Qw[start] + ATy[start] - c[start])
        for j in start+1:stop
            z_bar[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c[j])
            norm_s += z_bar[j]^2
        end
        norm_s = sqrt(norm_s)
        t = z_bar[start]
        if norm_s <= -t
            for j in start:stop
                x_bar[j] = 0.0
            end
        elseif norm_s <= t
            for j in start:stop
                x_bar[j] = z_bar[j]
            end
        else
            fact = (1 + t / norm_s) / 2
            x_bar[start] = (norm_s + t) / 2
            for j in start+1:stop
                x_bar[j] = fact * z_bar[j]
            end
        end

        for j in start:stop
            x_hat[j] = 2 * x_bar[j] - x[j]
            z_bar[j] = (x_bar[j] - z_bar[j]) / sigma
            dx[j] = x_bar[j] - x[j]
            x[j] = Halpern_fact1 * last_x[j] + Halpern_fact2 * x_hat[j]
            tempv[j] = x_hat[j] + sigma * Qw[j]
            w_bar[j] = fact1 * w[j] + fact2 * x_hat[j]
        end
    end
    return
end

# Single SOC version - Step 1: Compute z_bar in parallel
CUDA.@fastmath @inline function compute_zbar_single_SOC_kernel!(z_bar, x, Qw, ATy, c, sigma, start_idx, stop_idx)
    j = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    soc_size = stop_idx - start_idx + 1

    @inbounds if j <= soc_size
        idx = start_idx + j - 1
        z_bar[idx] = x[idx] + sigma * (-Qw[idx] + ATy[idx] - c[idx])
    end
    return
end

# Single SOC version - Step 2: Apply projection and update in parallel
# norm_s is computed outside the kernel, t is read from z_bar[start_idx]
CUDA.@fastmath @inline function project_and_update_single_SOC_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, start_idx, stop_idx, norm_s, t_val)
    j = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    soc_size = stop_idx - start_idx + 1

    @inbounds if j <= soc_size
        idx = start_idx + j - 1

        # Use precomputed t value (passed as parameter to avoid race condition)
        t = t_val

        # Apply projection based on norm_s and t
        if norm_s <= -t
            # Zero projection
            x_bar[idx] = 0.0
        elseif norm_s <= t
            # Identity projection
            x_bar[idx] = z_bar[idx]
        else
            # Standard projection
            if idx == start_idx
                # Head element: t_proj = (norm_s + t) / 2
                x_bar[idx] = (norm_s + t) / 2
            else
                # Tail elements: s_proj = ((1 + t/norm_s) / 2) * s
                fact = (1 + t / norm_s) / 2
                x_bar[idx] = fact * z_bar[idx]
            end
        end

        # Update remaining variables in parallel
        x_hat[idx] = 2 * x_bar[idx] - x[idx]
        z_bar[idx] = (x_bar[idx] - z_bar[idx]) / sigma
        dx[idx] = x_bar[idx] - x[idx]
        x[idx] = Halpern_fact1 * last_x[idx] + Halpern_fact2 * x_hat[idx]
        tempv[idx] = x_hat[idx] + sigma * Qw[idx]
        w_bar[idx] = fact1 * w[idx] + fact2 * x_hat[idx]
    end
    return
end


function update_zxw1_SOC_kernel_scaling!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_var, SOC_var_idx, SOC_var_scaling_factor, SOC_var_scaling_factor_inv)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1
        norm_s = 0.0
        norm_s_invscaling = 0.0
        z_bar[start] = x[start] + sigma * (-Qw[start] + ATy[start] - c[start])
        for j in start+1:stop
            z_bar[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c[j])
            norm_s += (SOC_var_scaling_factor[j] * z_bar[j])^2
            norm_s_invscaling += (SOC_var_scaling_factor_inv[j] * z_bar[j])^2
        end
        norm_s = sqrt(norm_s)
        norm_s_invscaling = sqrt(norm_s_invscaling)
        t = z_bar[start]
        if norm_s <= -t
            for j in start:stop
                x_bar[j] = 0.0
            end
        elseif norm_s_invscaling <= t
            for j in start:stop
                x_bar[j] = z_bar[j]
            end
        elseif abs(t) < 1e-15
            x_bar[start] = 0.0
            for j in start+1:stop
                x_bar[j] = z_bar[j] / (1 + SOC_var_scaling_factor_inv[j]^2)
                x_bar[start] += (z_bar[j] / (SOC_var_scaling_factor[j] + SOC_var_scaling_factor_inv[j]))^2
            end
            x_bar[start] = sqrt(x_bar[start])
        else
            # bisection
            if t > 0.0
                lo = 0.0
                up = 0.5
                mid = 0.25
                for iter in 1:100
                    mid = (lo + up) / 2
                    t_c = -t^2 / (1 - 2 * mid)^2
                    for j in start+1:stop
                        t_c += (SOC_var_scaling_factor_inv[j] / (1 + 2 * mid * SOC_var_scaling_factor_inv[j]^2))^2 * z_bar[j]^2
                    end
                    if t_c < -1e-15
                        up = mid
                    elseif t_c > 1e-15
                        lo = mid
                    else
                        break
                    end
                    if up - lo < 1e-15
                        break
                    end
                end
            else
                lo = 0.5
                up = 1.0
                mid = 0.75
                t_c = -1.0
                for iter in 1:100
                    t_c = -t^2 / (1 - 2 * up)^2
                    for j in start+1:stop
                        t_c += (SOC_var_scaling_factor_inv[j] / (1 + 2 * up * SOC_var_scaling_factor_inv[j]^2))^2 * z_bar[j]^2
                    end
                    up *= 2.0
                    if t_c > 0.0
                        break
                    end
                end
                for iter in 1:100
                    mid = (lo + up) / 2
                    t_c = -t^2 / (1 - 2 * mid)^2
                    for j in start+1:stop
                        t_c += (SOC_var_scaling_factor_inv[j] / (1 + 2 * mid * SOC_var_scaling_factor_inv[j]^2))^2 * z_bar[j]^2
                    end
                    if t_c < -1e-15
                        lo = mid
                    elseif t_c > 1e-15
                        up = mid
                    else
                        break
                    end
                    if up - lo < 1e-15
                        break
                    end
                end
            end
            lambda = mid
            # Qw[1] = lambda
            # Qw[2] = norm_s
            # Qw[3] = t
            x_bar[start] = t / (1 - 2 * lambda)
            for j in start+1:stop
                x_bar[j] = z_bar[j] / (1 + 2 * lambda * SOC_var_scaling_factor_inv[j]^2)
            end

            # fact = (1 + t / norm_s) / 2
            # x_bar[start] = (norm_s + t) / 2
            # for j in start+1:stop
            #     x_bar[j] = fact * z_bar[j]
            # end
        end

        for j in start:stop
            x_hat[j] = 2 * x_bar[j] - x[j]
            z_bar[j] = (x_bar[j] - z_bar[j]) / sigma
            dx[j] = x_bar[j] - x[j]
            x[j] = Halpern_fact1 * last_x[j] + Halpern_fact2 * x_hat[j]
            tempv[j] = x_hat[j]
            w_bar[j] = fact1 * w[j] + fact2 * x_hat[j]
        end
    end
    return
end

# Scalar scaling SOC kernel - Step 1: Compute z_bar for each SOC
CUDA.@fastmath @inline function compute_zbar_SOC_kernel_scalar!(z_bar, x, Qw, ATy, c, sigma, number_SOC_var, SOC_var_idx)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1
        for j in start:stop
            z_bar[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c[j])
        end
    end
    return
end

# Scalar scaling SOC kernel - Step 2: Apply projection with norm_s computed outside
CUDA.@fastmath @inline function update_zxw1_SOC_kernel_scalar!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_var, SOC_var_idx, SOC_con_scaling_factor, SOC_var_scaling_factor, norm_s_array)
    # This kernel assumes uniform scalar scaling per SOC (same scale for all elements in a cone).
    # norm_s_array contains the precomputed norms for each SOC (computed outside on GPU).
    # It maps the current z_bar into the scaled (canonical) cone space, applies the
    # standard closed-form SOC projection, then maps the projected result back.
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1

        # Read scaling scalars for this SOC (assumed uniform across the block)
        con_scale = SOC_con_scaling_factor[start]
        var_scale = SOC_var_scaling_factor[start]

        # Read precomputed norm (computed outside kernel)
        norm_s_un = norm_s_array[i]
        t = z_bar[start]

        # Project onto standard SOC: ||s|| ≤ t (no scaling transformations)
        if norm_s_un <= -t
            # Projection is zero
            for j in start:stop
                x_bar[j] = 0.0
            end
        elseif norm_s_un <= t
            # Already inside cone
            for j in start:stop
                x_bar[j] = z_bar[j]
            end
        else
            # Standard SOC projection
            tproj = (norm_s_un + t) / 2
            fact = (1 + t / norm_s_un) / 2

            x_bar[start] = tproj
            for j in start+1:stop
                x_bar[j] = fact * z_bar[j]
            end
        end

        # Common updates
        for j in start:stop
            x_hat[j] = 2 * x_bar[j] - x[j]
            z_bar[j] = (x_bar[j] - z_bar[j]) / sigma
            dx[j] = x_bar[j] - x[j]
            x[j] = Halpern_fact1 * last_x[j] + Halpern_fact2 * x_hat[j]
            tempv[j] = x_hat[j] + sigma * Qw[j]
            w_bar[j] = fact1 * w[j] + fact2 * x_hat[j]
        end
    end
    return
end

@inline function project_and_update_single_SOC_kernel_test!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, tempv, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, start_idx, stop_idx, norm_s)
    j = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    soc_size = stop_idx - start_idx + 1

    # Early exit for out-of-bounds threads
    if j > soc_size
        return
    end

    idx = start_idx + j - 1
    
    # Cache the head element value in shared memory for better performance
    # All threads read this, so we use shared memory to reduce global memory traffic
    t = @inbounds z_bar[start_idx]
    
    # Cache current z_bar value for this thread
    z_bar_val = @inbounds z_bar[idx]
    
    # Small epsilon for numerical stability
    eps = 1e-16
    
    # Apply projection based on norm_s and t
    x_bar_val = if norm_s <= -t
        # Zero projection: cone constraint violated, project to origin
        0.0
    elseif norm_s <= t
        # Identity projection: already in cone
        z_bar_val
    else
        # Standard projection: project onto cone boundary
        if idx == start_idx
            # Head element: t_proj = (norm_s + t) / 2
            (norm_s + t) * 0.5
        else
            # Tail elements: s_proj = ((1 + t/norm_s) / 2) * s
            # Safe division: only compute when norm_s > t > 0 (from outer condition)
            # Add epsilon to prevent division by very small numbers
            fact = (1.0 + t / max(norm_s, eps)) * 0.5
            fact * z_bar_val
        end
    end
    
    # Update remaining variables in parallel
    # Use local variables to reduce memory transactions
    @inbounds begin
        x_val = x[idx]
        x_bar[idx] = x_bar_val
        
        x_hat_val = 2.0 * x_bar_val - x_val
        x_hat[idx] = x_hat_val
        
        # Safe division by sigma (should always be positive, but add safety)
        z_bar[idx] = (x_bar_val - z_bar_val) / max(sigma, eps)
        
        dx[idx] = x_bar_val - x_val
        
        x[idx] = Halpern_fact1 * last_x[idx] + Halpern_fact2 * x_hat_val
        
        tempv[idx] = x_hat_val + sigma * Qw[idx]
        
        w_bar[idx] = fact1 * w[idx] + fact2 * x_hat_val
    end
    
    return
end

function update_zxw1_SOC_kernel_scaling_test!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_var, SOC_var_idx, SOC_var_scaling_factor, SOC_var_scaling_factor_inv)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1
        norm_s = 0.0
        norm_s_invscaling = 0.0
        z_bar[start] = x[start] + sigma * (-Qw[start] + ATy[start] - c[start])
        for j in start+1:stop
            z_bar[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c[j])
            norm_s += (SOC_var_scaling_factor[j] * z_bar[j])^2
            norm_s_invscaling += (SOC_var_scaling_factor_inv[j] * z_bar[j])^2
        end
        norm_s = sqrt(norm_s)
        norm_s_invscaling = sqrt(norm_s_invscaling)
        t = z_bar[start]
        if norm_s <= -t
            for j in start:stop
                x_bar[j] = 0.0
            end
        elseif norm_s_invscaling <= t
            for j in start:stop
                x_bar[j] = z_bar[j]
            end
        elseif abs(t) < 1e-15
            x_bar[start] = 0.0
            for j in start+1:stop
                x_bar[j] = z_bar[j] / (1 + SOC_var_scaling_factor_inv[j]^2)
                x_bar[start] += (z_bar[j] / (SOC_var_scaling_factor[j] + SOC_var_scaling_factor_inv[j]))^2
            end
            x_bar[start] = sqrt(x_bar[start])
        else
            fact = (1 + t / norm_s) / 2
            x_bar[start] = (norm_s + t) / 2
            for j in start+1:stop
                x_bar[j] = fact * z_bar[j]
            end
        end

        for j in start:stop
            x_hat[j] = 2 * x_bar[j] - x[j]
            z_bar[j] = (x_bar[j] - z_bar[j]) / sigma
            dx[j] = x_bar[j] - x[j]
            x[j] = Halpern_fact1 * last_x[j] + Halpern_fact2 * x_hat[j]
            tempv[j] = x_hat[j] + sigma * Qw[j]
            w_bar[j] = fact1 * w[j] + fact2 * x_hat[j]
        end
    end
    return
end

function update_zxw1_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.ATy, ws.AT, ws.y)
    # CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.Qw .= ws.Q * ws.w
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    if ws.number_lu_x > 0
        @cuda threads = 256 blocks = ceil(Int, ws.number_lu_x / 256) update_zxw1_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_lu_x)
    end
    if ws.number_SOC_var > 0
        # NEW: Use scalar scaling kernel (uniform scaling per SOC - preserves cone structure)
        # This enables simple closed-form projection instead of bisection
        
        # Step 1: Compute z_bar for all SOCs in parallel
        #@cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) compute_zbar_SOC_kernel_scalar!(
        #    ws.z_bar, ws.x, ws.Qw, ws.ATy, ws.c, ws.sigma, ws.number_SOC_var, ws.SOC_var_idx
        #)
        
        # Step 2: Compute norm (single SOC case - compute directly on GPU)
        #v = @view ws.z_bar[2:ws.n]
        #norm_s = sqrt(LinearAlgebra.dot(v, v))
        # Store in temp array (avoid scalar indexing by using fill!)
        #CUDA.fill!(ws.SOC_norms_temp, norm_s)
        
        # Step 3: Apply projection with precomputed norms
        #@cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) update_zxw1_SOC_kernel_scalar!(
        #    ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.tempv,
        #    ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_SOC_var, ws.SOC_var_idx, 
        #    ws.SOC_con_scaling_factor, ws.SOC_var_scaling_factor, ws.SOC_norms_temp
        #)
        
        # OLD APPROACHES (commented out):
        # 1. Elementwise scaling with bisection (100x slower):
        # @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) update_zxw1_SOC_kernel_scaling!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_SOC_var, ws.SOC_var_idx, ws.SOC_var_scaling_factor, ws.SOC_var_scaling_factor_inv)
        
        # 2. No scaling (assumes already normalized):
        # @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_var / 256) update_zxw1_SOC_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_SOC_var, ws.SOC_var_idx)
        
        # 3. Single SOC special case (3-step approach):
        # Step 1: Compute z_bar in parallel
        @cuda threads=256 blocks=ceil(Int, ws.n / 256) compute_zbar_single_SOC_kernel!(ws.z_bar, ws.x, ws.Qw, ws.ATy, ws.c, ws.sigma, 1, ws.n)
        # Step 2: Compute norm on GPU (only non-head elements)
        v = @view ws.z_bar[2:ws.n]
        norm_s = sqrt(LinearAlgebra.dot(v, v))
	t_val = CUDA.@allowscalar ws.z_bar[1]
        # Step 3: Apply projection and update in parallel (pass t as parameter to avoid race condition)
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) project_and_update_single_SOC_kernel!(
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.tempv,
            ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2,
            1, ws.n, norm_s, t_val
        )
    end
    # verify_soc_constraints_gpu(ws.x_bar, ws.SOC_var_idx, ws.number_SOC_var)
    # println(ws.SOC_var_scaling_factor)
    # temp_m = Vector(ws.SOC_var_scaling_factor)
    # println("SOC_var_scaling_factor: ", temp_m)
    # temp_m = Vector(ws.SOC_var_scaling_factor_inv)
    # println("SOC_var_scaling_factor_inv: ", temp_m)
    # Qw_cpu = Vector(ws.Qw)
    # println("lambda: ", Qw_cpu[1])
    # println("norm_s: ", Qw_cpu[2])
    # println("t: ", Qw_cpu[3])
    # println("1-2*lambda: ", 1 - 2 * Qw_cpu[1])
    # println("2t/(norm_s + t):", 2 * Qw_cpu[3] / (Qw_cpu[2] + Qw_cpu[3]))
    # println(Vector(ws.last_x))
    # println(Vector(ws.x_bar))
    # println(Vector(ws.x_bar - ws.last_x))
end

CUDA.@fastmath @inline function update_zxw1_diagQ_kernel!(dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x, Qw, ATy, c, tempv, l, u, sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, n)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        # if abs(tmp) < 1e-15
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

function update_zxw1_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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
        # if abs(tmp) < 1e-15
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

function cust_update_zxw1_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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
        # if abs(tmp) < 1e-15
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

function cust_update_zxw1_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_w1_gpu!(ws::HPRSOCP_workspace_gpu)
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

function cust_update_w1_diagQ_gpu!(ws::HPRSOCP_workspace_gpu)
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

# function update_w1_gpu!(ws::HPRSOCP_workspace_gpu)
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

# function update_w1_diagQ_gpu!(ws::HPRSOCP_workspace_gpu)
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

function update_w2_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function update_w2_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_w2_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_w2_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_ATy_kernel!(ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar, ws.ATy, ws.ATdy, ws.ATy_bar, ws.n)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATy, 0, ws.Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w2_diagQ_kernel!(ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar,
        ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.fact, Halpern_fact1, Halpern_fact2, ws.n)
    # @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_Halpern_Qw_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.w_bar, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_kernel!(dy, y_bar, y, last_y, b, Ax, fact, Halpern_fact1, Halpern_fact2, number_eq, number_ineq)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_eq
        y_bar[i] = y[i] + fact * (b[i] - Ax[i])
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    elseif i <= number_eq + number_ineq
        y_bar[i] = max(y[i] + fact * (b[i] - Ax[i]), 0.0)
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function update_y_SOC_kernel!(dy, y_bar, y, last_y, s, b, Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_con, SOC_con_idx)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_con
        start = SOC_con_idx[i]
        stop = SOC_con_idx[i+1] - 1
        norm_s = 0.0
        s[start] = Ax[start] - b[start] - fact1 * y[start]
        for j in start+1:stop
            s[j] = Ax[j] - b[j] - fact1 * y[j]
            norm_s += s[j]^2
        end
        norm_s = sqrt(norm_s)
        t = s[start]
        if norm_s <= -t
            for j in start:stop
                y_bar[j] = -s[j]
            end
        elseif norm_s <= t
            for j in start:stop
                y_bar[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            y_bar[start] = (norm_s + t) / 2 - s[start]
            for j in start+1:stop
                y_bar[j] = fact * s[j] - s[j]
            end
        end
        for j in start:stop
            s[j] = s[j] + y_bar[j]
            y_bar[j] = fact2 * y_bar[j]
            dy[j] = y_bar[j] - y[j]
            y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2 * y_bar[j] - y[j])
        end
    end
    return
end


function update_y_SOC_kernel_scaling2!(dy, y_bar, y, last_y, s, s_org, b, Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_con, SOC_con_idx, SOC_con_scaling_factor, SOC_con_scaling_factor_inv)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_con
        start = SOC_con_idx[i]
        stop = SOC_con_idx[i+1] - 1
        norm_s = 0.0
        s[start] = Ax[start] - b[start] - fact1 * y[start]
        s_org[start] = s[start] * SOC_con_scaling_factor[start]
        for j in start+1:stop
            s[j] = Ax[j] - b[j] - fact1 * y[j]
            s_org[j] = s[j] * SOC_con_scaling_factor[j]
            norm_s += (s_org[j])^2
        end
        norm_s = sqrt(norm_s)
        t = s[start]
        if norm_s <= -t
            for j in start:stop
                y_bar[j] = 0.0
            end
        elseif norm_s <= t
            for j in start:stop
                y_bar[j] = s_org[j]
            end
        else
            fact = (1 + t / norm_s) / 2
            y_bar[start] = (norm_s + t) / 2 - s[start]
            for j in start+1:stop
                y_bar[j] = fact * s[j] - s[j]
            end
        end
        for j in start:stop
            s[j] = s[j] + y_bar[j]
            y_bar[j] = fact2 * y_bar[j]
            dy[j] = y_bar[j] - y[j]
            y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2 * y_bar[j] - y[j])
        end
    end
    return
end

function update_y_SOC_kernel_scaling!(dy, y_bar, y, last_y, s, b, Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, number_SOC_con, SOC_con_idx, SOC_con_scaling_factor, SOC_con_scaling_factor_inv)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= number_SOC_con
        start = SOC_con_idx[i]
        stop = SOC_con_idx[i+1] - 1
        norm_s = 0.0
        norm_s_invscaling = 0.0
        s[start] = Ax[start] - b[start] - fact1 * y[start]
        for j in start+1:stop
            s[j] = Ax[j] - b[j] - fact1 * y[j]
            norm_s += (SOC_con_scaling_factor[j] * s[j])^2
            norm_s_invscaling += (SOC_con_scaling_factor_inv[j] * s[j])^2
        end
        norm_s = sqrt(norm_s)
        norm_s_invscaling = sqrt(norm_s_invscaling)
        t = s[start]
        if norm_s <= -t
            for j in start:stop
                y_bar[j] = 0.0
            end
        elseif norm_s_invscaling <= t
            for j in start:stop
                y_bar[j] = s[j]
            end
        elseif abs(t) < 1e-15
            y_bar[start] = 0.0
            for j in start+1:stop
                y_bar[j] = s[j] / (1 + SOC_con_scaling_factor_inv[j]^2)
                y_bar[start] += (s[j] / (SOC_con_scaling_factor[j] + SOC_con_scaling_factor_inv[j]))^2
            end
            y_bar[start] = sqrt(y_bar[start])
        else
            # bisection
            if t > 0.0
                lo = 0.0
                up = 0.5
                mid = 0.25
                max_iter = 100
                for iter in 1:max_iter
                    mid = (lo + up) / 2
                    t_c = -t^2 / (1 - 2 * mid)^2
                    for j in start+1:stop
                        t_c += (SOC_con_scaling_factor_inv[j] / (1 + 2 * mid * SOC_con_scaling_factor_inv[j]^2))^2 * s[j]^2
                    end
                    if t_c < -1e-15
                        up = mid
                    elseif t_c > 1e-15
                        lo = mid
                    else
                        break
                    end
                    if up - lo < 1e-15
                        break
                    end
                end
            else
                lo = 0.5
                up = 100.0
                mid = 0.75
                t_c = -1.0
                for iter in 1:100
                    t_c = -t^2 / (1 - 2 * up)^2
                    for j in start+1:stop
                        t_c += (SOC_con_scaling_factor_inv[j] / (1 + 2 * up * SOC_con_scaling_factor_inv[j]^2))^2 * s[j]^2
                    end
                    up *= 2.0
                    if t_c > 0.0
                        break
                    end
                end
                up /= 2.0
                for iter in 1:100
                    mid = (lo + up) / 2
                    t_c = -t^2 / (1 - 2 * mid)^2
                    for j in start+1:stop
                        t_c += (SOC_con_scaling_factor_inv[j] / (1 + 2 * mid * SOC_con_scaling_factor_inv[j]^2))^2 * s[j]^2
                    end
                    if t_c < -1e-15
                        lo = mid
                    elseif t_c > 1e-15
                        up = mid
                    else
                        break
                    end
                    if up - lo < 1e-15
                        break
                    end
                end
            end
            lambda = mid

            # fact = (1 + t / norm_s) / 2
            # y_bar[start] = (norm_s + t) / 2
            # for j in start+1:stop
            #     y_bar[j] = fact * s[j]
            # end
            y_bar[start] = t / (1-2*lambda)
            for j in start+1:stop
                y_bar[j] = s[j]/(1+2*lambda*SOC_con_scaling_factor_inv[j]^2)
            end
        end
        for j in start:stop
            y_bar[j] -= s[j]
            s[j] = s[j] + y_bar[j]
            y_bar[j] = fact2 * y_bar[j]
            dy[j] = y_bar[j] - y[j]
            y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2 * y_bar[j] - y[j])
        end
    end
    return
end


function update_y_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    CUDA.CUSPARSE.mv!('N', 1.0, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # ws.tempv .+= ws.Qw_bar
    axpby_gpu!(1.0, ws.tempv, -ws.sigma, ws.Qw_bar, ws.tempv, ws.n)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, (ws.number_eq + ws.number_ineq) / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.b, ws.Ax, fact2, Halpern_fact1, Halpern_fact2, ws.number_eq, ws.number_ineq)
    if ws.number_SOC_con > 0
        # @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_con / 256) update_y_SOC_kernel_scaling!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.b, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_SOC_con, ws.SOC_con_idx, ws.SOC_con_scaling_factor, ws.SOC_con_scaling_factor_inv)
        @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_con / 256) update_y_SOC_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.b, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.number_SOC_con, ws.SOC_con_idx)
    end
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

function cust_update_y_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function compute_Rd_gpu_unscale!(ws, socp_org)
    # mul!(ws.ATy, ws.AT, ws.y_bar) 
    CUDA.CUSPARSE.mv!('N', 1, socp_org.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.dz, socp_org.c, ws.Qx, ws.Rd, ws.n)
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
    b::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    number_ineq::Int,
    number_eq::Int,
    SOC_con_idx::CuDeviceVector{Int},
    number_SOC_con::Int,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= number_eq
        Rp[i] = b[i] - Ax[i]
    elseif i <= number_ineq + number_eq
        Rp[i] = max(b[i] - Ax[i], 0.0)
    end
    return
end

function compute_Rp_SOC_kernel!(Rp::CuDeviceVector{Float64},
    b::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    number_ineq::Int,
    number_eq::Int,
    SOC_con_idx::CuDeviceVector{Int},
    number_SOC_con::Int,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= number_SOC_con
        start = SOC_con_idx[i]
        stop = SOC_con_idx[i+1] - 1
        norm_s = 0.0
        Rp[start] = Ax[start] - b[start]
        for j in start+1:stop
            Rp[j] = Ax[j] - b[j]
            norm_s += Rp[j]^2
        end
        norm_s = sqrt(norm_s)
        t = Rp[start]
        if norm_s <= -t
        elseif norm_s <= t
            for j in start:stop
                Rp[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            Rp[start] = (norm_s + t) / 2 - Rp[start]
            for j in start+1:stop
                Rp[j] = fact * Rp[j] - Rp[j]
            end
        end
    end
    return
end

function compute_xlu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    number_linear_vars::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= number_linear_vars
        dx[i] = 0.0
        if x[i] < l[i]
            dx[i] = l[i] - x[i]
        elseif x[i] > u[i]
            dx[i] = u[i] - x[i]
        end
    end
    return
end

function compute_x_SOC_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    SOC_var_idx::CuDeviceVector{Int},
    number_SOC_var::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= number_SOC_var
        start = SOC_var_idx[i]
        stop = SOC_var_idx[i+1] - 1
        norm_s = 0.0
        dx[start] = x[start]
        for j in start+1:stop
            dx[j] = x[j]
            norm_s += dx[j]^2
        end
        norm_s = sqrt(norm_s)
        t = dx[start]
        if norm_s <= -t
        elseif norm_s <= t
            for j in start:stop
                dx[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            dx[start] = (norm_s + t) / 2 - dx[start]
            for j in start+1:stop
                dx[j] = fact * dx[j] - dx[j]
            end
        end
    end
    return
end

function compute_Rp_gpu!(ws)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, (ws.number_ineq + ws.number_eq) / 256) compute_Rp_kernel!(ws.Rp, ws.b, ws.Ax, ws.number_ineq, ws.number_eq, ws.SOC_con_idx, ws.number_SOC_con, ws.m)
    if ws.number_SOC_con > 0
        @cuda threads = 256 blocks = ceil(Int, ws.number_SOC_con / 256) compute_Rp_SOC_kernel!(ws.Rp, ws.b, ws.Ax, ws.number_ineq, ws.number_eq, ws.SOC_con_idx, ws.number_SOC_con, ws.m)
    end
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

function update_w_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_w_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_w_noC_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function update_w_noC_diagQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function update_x_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
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

function cust_update_x_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_x_noC_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end






function update_y_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)

end

function cust_update_y_noC_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1, Halpern_fact2)
    # mul!(ws.Ax, ws.A, ws.x_hat)
    # CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.x_hat, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
end


function compute_M_norm_gpu!(ws::HPRSOCP_workspace_gpu, Q_is_diag::Bool, noC::Bool)
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


# function compute_M_norm_gpu2!(ws::HPRSOCP_workspace_gpu, Q_is_diag::Bool, noC::Bool)
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




function compute_M_norm_gpu2!(ws::HPRSOCP_workspace_gpu)
    M_1 = ws.sigma * ws.lambda_max_A * LinearAlgebra.dot(ws.dy, ws.dy)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    M_1 -= 2 * ws.sigma * LinearAlgebra.dot(ws.dQw, ws.ATdy)
    M_2 = 1 / ws.sigma * LinearAlgebra.dot(ws.dx, ws.dx)
    M_2 += 2 * LinearAlgebra.dot(ws.ATdy, ws.dx)
    M_2 -= 2 * LinearAlgebra.dot(ws.dQw, ws.dx)
    M_3 = (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * LinearAlgebra.dot(ws.ATdy, ws.QATdy)  # sGS term
    M_1 += ws.sigma * ws.lambda_max_Q * LinearAlgebra.dot(ws.dw, ws.dQw)
    M_2 += max(M_1, 0)
    # M_norm = M_2 + M_3
    M_norm = max(M_2, 0) + max(M_3, 0)
    if min(M_1, M_2, M_3) < -1e-8
        println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3")
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
    tol::Float64=1e-15,
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
    tol::Float64=1e-15,
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

function add_partial_vector!(y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    dy::CuDeviceVector{Float64},
    src::CuDeviceVector{Float64}, start::Int, stop::Int, Halpern_fact1::Float64, Halpern_fact2::Float64)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= stop - start + 1
        dy[start+i-1] = src[i]
        y_bar[start+i-1] = src[i] + y[start+i-1]
        y[start+i-1] = Halpern_fact1 * last_y[start+i-1] + Halpern_fact2 * (2 * y_bar[start+i-1] - y[start+i-1])
    end
    return
end

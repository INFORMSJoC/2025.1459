
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

    println("nnz A = ", nnz(A))
    # quadratic part
    Q = sparse(qp.qrows, qp.qcols, qp.qvals, qp.nvar, qp.nvar)
    # the Q matrix is not symmetric, so we need to symmetrize it
    diag_Q = diag(Q)
    Q = Q + Q' - Diagonal(diag_Q)
    println("nnz Q = ", nnz(Q))

    # linear part
    c = qp.c
    c0 = qp.c0

    # bounds
    lvar = qp.lvar
    uvar = qp.uvar

    # Return the modified qp
    return Q, c, A, lcon, ucon, lvar, uvar, c0
end

function qp_formulation_noC(Q, c, A, AL, AU, l, u, c0=0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    A = vcat(A, spdiagm(ones(length(l))))
    AL = vcat(AL, l)
    AU = vcat(AU, u)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    keep_rows = setdiff(1:size(A, 1), del_row)
    println("Deleted ", length(del_row), " rows of A that are all zeros.")
    A = A[keep_rows, :]
    lp_AL = AL[keep_rows]
    lp_AU = AU[keep_rows]

    # del_lu = findall((l .== -Inf) .& (u .== Inf))
    # keep_lu = setdiff(1:length(l), del_lu)
    # println("Deleted ", length(del_lu), " rows of l and u that are all -Inf and Inf.")
    # l = l[keep_lu]
    # u = u[keep_lu]

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
    @assert length(c) == size(Q, 1)
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
    # l .= -Inf * ones(length(l))
    # u .= Inf * ones(length(u))


    m1 = length(idxE)

    # idxL is from m1+1 to m
    m = length(AL)
    idxI = (m1+1):m

    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, idxI, m1, c0, [], false, [], [], true)

    # Return the modified qp
    return standard_qp
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

    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, idxI, m1, c0, [], false, [], [], false)

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
        Int[], size(A_new, 1), c0, [], false, [], [], false)  # No inequality constraints after slack variable conversion

    return standard_qp
end

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
    scaling_info = Scaling_info_cpu(copy(qp.l), copy(qp.u), row_norm, col_norm, 1, 1, 1, 1, norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), norm(qp.c, Inf))
    println("norm_b_org: ", scaling_info.norm_b_org)
    println("norm_c_org: ", scaling_info.norm_c_org)

    if x_path != ""
        # read x from file
        x = Array{Float64}(undef, n)
        read!(x_path, x)
        qp.x0 = x
    else
        # initialize x to zeros
        qp.x0 = zeros(n)
    end
    if y_path != ""
        # read y from file
        y = Array{Float64}(undef, m)
        read!(y_path, y)
        qp.y0 = y
    else
        # initialize y to zeros
        qp.y0 = zeros(m)
    end
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
            qp.x0 .*= temp_norm_A_col
            qp.y0 .*= temp_norm_A_row

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
        qp.x0 .*= temp_norm_A_col
        qp.y0 .*= temp_norm_A_row

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
        qp.x0 .*= temp_norm_A_col
        qp.y0 .*= temp_norm_A_row

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
        qp.x0 ./= b_scale
        qp.y0 ./= c_scale
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


function power_iteration_Q_gpu_new(Q, max_iterations=500, tolerance=1e-4)
    seed = 1
    n, n = size(Q)
    q = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q ./= LinearAlgebra.norm(q)
    # z = CuVector(rand(m)) .+ 1e-8
    z = CUDA.zeros(Float64, n)
    q_new = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        # mul!(ATq, AT, q)
        CUDA.CUSPARSE.mv!('N', 1, Q, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        q_new .= z / LinearAlgebra.norm(z)
        error = LinearAlgebra.norm(q_new - q)
        if error < tolerance
            CUDA.CUSPARSE.mv!('N', 1, Q, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            lambda_max = LinearAlgebra.dot(q_new, z)
            println("Power iteration converged in ", i, " iterations. The largest eigenvalue is ", lambda_max)
            return lambda_max
        end
        q .= q_new
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    CUDA.CUSPARSE.mv!('N', 1, Q, q_new, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    lambda_max = LinearAlgebra.dot(q_new, z)
    return lambda_max
end

function power_iteration_A_gpu_new(A, AT, max_iterations=500, tolerance=1e-4)
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

function mean(x::Vector{Float64})
    return sum(x) / length(x)
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
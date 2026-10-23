using JLD2

mutable struct my_cbf_data
    acoord
    bcoord
    con
    dcoord
    fcoord
    hcoord
    intlist
    nconstr
    nvar
    var
    objacoord
    objfcoord
    objoffset
    psdcon
    psdvar
    sense
end

"""
    save_trust_region_problem(filename, H, c, A, b)

Save a trust region problem (H, c, A, b) to a JLD2 file. Handles sparse matrices natively.

# Arguments
- `filename`: Output filename (without extension, .jld2 will be added)
- `H`: Objective Hessian matrix (can be sparse or dense)
- `c`: Linear objective coefficient vector
- `A`: Constraint matrix (can be sparse or dense)
- `b`: Right-hand side vector

The problem represents:
    min  (1/2)⟨y, Hy⟩ + ⟨c, y⟩
    s.t. Ay = b
         ||y||₂ ≤ radius
"""
function save_trust_region_problem(filename::String, H, c, A, b)
    # Add .jld2 extension if not present
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end
    
    jldsave(filename;
        H = H,
        c = c,
        A = A,
        b = b
    )
    
    println("Saved trust region problem to ", filename)
    println("  Problem size: ", size(A, 1), " equality constraints, ", size(A, 2), " variables")
    println("  H is sparse: ", issparse(H), ", size = ", size(H))
    if issparse(H)
        println("    nnz(H) = ", nnz(H))
    end
    println("  A is sparse: ", issparse(A), ", size = ", size(A))
    if issparse(A)
        println("    nnz(A) = ", nnz(A))
    end
end

"""
    load_trust_region_problem(filename)

Load a trust region problem from a JLD2 file.

# Returns
A tuple containing: (H, c, A, b)
"""
function load_trust_region_problem(filename::String)
    # Add .jld2 extension if not present
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end
    
    data = load(filename)
    
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
    
    return (data["H"], data["c"], data["A"], data["b"])
end

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



function my_read_cbf(filename::String)
    # problem may contain
    # acoord     bcoord     con        dcoord
    # fcoord     hcoord     intlist    name
    # nconstr    nvar       objacoord  objfcoord
    # objoffset  psdcon     psdvar     sense
    # var
    file = open(filename, "r")

    lines = readlines(file)
    close(file)
    acoord = []
    bcoord = []
    nconstr = 0
    con = []
    dcoord = []
    fcoord = []
    hcoord = []
    intlist = []
    nvar = 0
    var = []
    objacoord = []
    objfcoord = []
    objoffset = 0.0
    psdcon = []
    psdvar = []
    sense = "min"
    i = 1
    while i <= length(lines)
        if startswith(lines[i], "#")
            i += 1
            continue
        end
        line = lines[i]
        if line == "ACOORD"
            nnz = parse(Int, lines[i+1])
            acoord = Vector{Tuple{Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 3
                    acoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Float64, parts[3]))
                    # acoord = vcat(acoord, (parse(Int, parts[1])+1, parse(Int, parts[2])+1, parse(Float64, parts[3])))
                else
                    println("Error in ACOORD line: ", lines[i])
                end
            end
        end
        if line == "BCOORD"
            nnz = parse(Int, lines[i+1])
            bcoord = Vector{Tuple{Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 2
                    bcoord[j] = (parse(Int, parts[1]) + 1, parse(Float64, parts[2]))
                    # bcoord = vcat(bcoord, (parse(Int, parts[1])+1, parse(Float64, parts[2])))
                else
                    println("Error in BCOORD line: ", lines[i])
                end
            end
        end
        if line == "CON"
            parts = split(lines[i+1])
            nconstr = parse(Int, parts[1])
            number_con_groups = parse(Int, parts[2])
            con = Vector{Tuple{String,Int}}(undef, number_con_groups)
            i += 1
            for j in 1:number_con_groups
                i += 1
                parts = split(lines[i])
                if length(parts) == 2
                    con[j] = (parts[1], parse(Int, parts[2]))
                    # con = vcat(con, (parts[1], parse(Int, parts[2])))

                else
                    println("Error in con line: ", lines[i])
                end
            end
        end
        if line == "DCOORD"
            nnz = parse(Int, lines[i+1])
            dcoord = Vector{Tuple{Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 4
                    dcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Float64, parts[4]))
                    # dcoord = vcat(dcoord, (parse(Int, parts[1])+1, parse(Int, parts[2])+1, parse(Float64, parts[3])))
                else
                    println("Error in DCOORD line: ", lines[i])
                end
            end
        end
        if line == "FCOORD"
            nnz = parse(Int, lines[i+1])
            fcoord = Vector{Tuple{Int,Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 5
                    dcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Int, parts[4]) + 1, parse(Float64, parts[5]))
                    # dcoord = vcat(dcoord, (parse(Int, parts[1])+1, parse(Int, parts[2])+1, parse(Float64, parts[3])))
                else
                    println("Error in FCOORD line: ", lines[i])
                end
            end
        end
        if line == "hCOORD"
            nnz = parse(Int, lines[i+1])
            hcoord = Vector{Tuple{Int,Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 5
                    dcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Int, parts[4]) + 1, parse(Float64, parts[5]))
                    # dcoord = vcat(dcoord, (parse(Int, parts[1])+1, parse(Int, parts[2])+1, parse(Float64, parts[3])))
                else
                    println("Error in HCOORD line: ", lines[i])
                end
            end
        end
        if line == "INT"
            nnz = parse(Int, lines[i+1])
            intlist = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 1
                    intlist[j] = (parse(Int, parts[1]) + 1)
                    # intlist = vcat(intlist, (parse(Int, parts[1])+1, parse(Int, parts[2])))
                else
                    println("Error in INT line: ", lines[i])
                end
            end
        end
        if line == "VAR"
            parts = split(lines[i+1])
            nvar = parse(Int, parts[1])
            number_var_groups = parse(Int, parts[2])
            var = Vector{Tuple{String,Int}}(undef, number_var_groups)
            i += 1
            for j in 1:number_var_groups
                i += 1
                parts = split(lines[i])
                if length(parts) == 2
                    var[j] = (parts[1], parse(Int, parts[2]))
                    # con = vcat(con, (parts[1], parse(Int, parts[2])))

                else
                    println("Error in VAR line: ", lines[i])
                end
            end
        end
        if line == "OBJACOORD"
            nnz = parse(Int, lines[i+1])
            objacoord = Vector{Tuple{Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 2
                    objacoord[j] = (parse(Int, parts[1]) + 1, parse(Float64, parts[2]))
                    # bcoord = vcat(bcoord, (parse(Int, parts[1])+1, parse(Float64, parts[2])))
                else
                    println("Error in OBJACOORD line: ", lines[i])
                end
            end
        end
        if line == "OBJFCOORD"
            nnz = parse(Int, lines[i+1])
            objfcoord = Vector{Tuple{Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 4
                    objfcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Float64, parts[4]))
                    # dcoord = vcat(dcoord, (parse(Int, parts[1])+1, parse(Int, parts[2])+1, parse(Float64, parts[3])))
                else
                    println("Error in OBJFCOORD line: ", lines[i])
                end
            end
        end
        if line == "OBJBCOORD"
            objoffset = parse(Float64, lines[i+1])
            i += 1
        end
        if line == "PSDCON"
            nnz = parse(Int, lines[i+1])
            psdcon = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 1
                    psdcon[j] = parse(Int, parts[1]) + 1
                    # psdcon = vcat(psdcon, (parse(Int, parts[1])+1, parse(Int, parts[2])))
                else
                    println("Error in PSDCON line: ", lines[i])
                end
            end
        end
        if line == "PSDVAR"
            nnz = parse(Int, lines[i+1])
            psdvar = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                if length(parts) == 1
                    psdvar[j] = parse(Int, parts[1]) + 1
                    # psdvar = vcat(psdvar, (parse(Int, parts[1])+1, parse(Int, parts[2])))
                else
                    println("Error in PSDVAR line: ", lines[i])
                end
            end
        end
        if line == "SENSE"
            sense = lines[i+1]
            i += 1
        end

        i += 1
    end

    # Create a struct to hold the problem data
    problem = my_cbf_data(
        acoord,
        bcoord,
        con,
        dcoord,
        fcoord,
        hcoord,
        intlist,
        nconstr,
        nvar,
        var,
        objacoord,
        objfcoord,
        objoffset,
        psdcon,
        psdvar,
        sense
    )
    return problem
end


function read_cbf(file::String)
    if file[end-3:end] != ".cbf" && file[end-4:end] != ".CBF"
        println("The file is not a CBF file. Please provide a .cbf file.")
        return "error"
    end
    problem = my_read_cbf(file)
    # problem = readcbfdata(file)
    # problem may contain
    # acoord     bcoord     con        dcoord
    # fcoord     hcoord     intlist    name
    # nconstr    nvar       objacoord  objfcoord
    # objoffset  psdcon     psdvar     sense
    # var
    if !isempty(problem.dcoord) || !isempty(problem.hcoord) || !isempty(problem.psdcon)
        println("The CBF file contains PSD constraints, which is not supported.")
        return "error"
    end
    if !isempty(problem.fcoord) || !isempty(problem.psdvar)
        println("The CBF file contains matrix variables, which is not supported.")
        return "error"
    end
    if !isempty(problem.intlist)
        println("The CBF file contains integer constraints, which is not supported.")
        return "error"
    end
    support_cones = ["L+", "L-", "L=", "Q", "F"]
    for con in problem.con
        if !(con[1] in support_cones)
            println("The CBF file contains unsupported constraint type: ", con[1])
            return "error"
        end
    end
    for var in problem.var
        if !(var[1] in support_cones)
            println("The CBF file contains unsupported variable type: ", var[1])
            return "error"
        end
    end

    m = sum(getindex.(problem.con, 2))
    @assert problem.nconstr == m
    n = sum(getindex.(problem.var, 2))
    @assert problem.nvar == n
    println("The problem has ", m, " constraints and ", n, " variables.")
    arows = getindex.(problem.acoord, 1)
    acols = getindex.(problem.acoord, 2)
    avals = getindex.(problem.acoord, 3)
    A = sparse(arows, acols, avals, m, n)
    b = zeros(m)
    for i in 1:length(problem.bcoord)
        b[problem.bcoord[i][1]] = problem.bcoord[i][2]
    end

    A_new = []
    b_new = []
    A_ineq = sparse([], [], [], 0, n)
    A_eq = sparse([], [], [], 0, n)
    A_SOC = sparse([], [], [], 0, n)
    b_ineq = []
    b_eq = []
    b_SOC = []
    current_row = 1
    SOC_con_idx = []
    # Ax - b = 0 from number_ineq+1 to number of equality constraints
    for i in 1:length(problem.con)
        if problem.con[i][1] == "L="
            len = problem.con[i][2]
            if isempty(A_eq)
                A_eq = A[current_row:current_row+len-1, :]
            else
                A_eq = vcat(A_eq, A[current_row:current_row+len-1, :])
            end
            b_eq = vcat(b_eq, -b[current_row:current_row+len-1])
            current_row += len
        elseif problem.con[i][1] == "L+"
            len = problem.con[i][2]
            if isempty(A_ineq)
                A_ineq = A[current_row:current_row+len-1, :]
            else
                A_ineq = vcat(A_ineq, A[current_row:current_row+len-1, :])
            end
            b_ineq = vcat(b_ineq, -b[current_row:current_row+len-1])
            current_row += len
        elseif problem.con[i][1] == "L-"
            len = problem.con[i][2]
            if isempty(A_ineq)
                A_ineq = -A[current_row:current_row+len-1, :]
            else
                A_ineq = vcat(A_ineq, -A[current_row:current_row+len-1, :])
            end
            b_ineq = vcat(b_ineq, b[current_row:current_row+len-1])
            current_row += len
        elseif problem.con[i][1] == "Q"
            len = problem.con[i][2]
            if isempty(A_SOC)
                A_SOC = A[current_row:current_row+len-1, :]
                SOC_con_idx = vcat(SOC_con_idx, len)
            else
                A_SOC = vcat(A_SOC, A[current_row:current_row+len-1, :])
                SOC_con_idx = vcat(SOC_con_idx, len)
            end
            b_SOC = vcat(b_SOC, -b[current_row:current_row+len-1])
            current_row += len
        end
    end
    number_eq = size(A_eq, 1)
    number_ineq = size(A_ineq, 1)
    A_new = vcat(A_eq, A_ineq, A_SOC)
    b_new = vcat(b_eq, b_ineq, b_SOC)
    println("The problem has ", number_eq, " equality constraints.")
    println("The problem has ", number_ineq, " inequality constraints.")
    @assert size(A_new, 1) == m
    @assert size(A_new, 2) == n
    @assert length(b_new) == m
    # println("The problem has ", size(A_new, 1), " constraints and ", size(A_new, 2), " variables.")

    SOC_con_idx_new = [number_eq + number_ineq + 1]
    for i in 1:length(SOC_con_idx)
        SOC_con_idx_new = vcat(SOC_con_idx_new, SOC_con_idx_new[end] + SOC_con_idx[i])
    end
    @assert SOC_con_idx_new[end] - 1 == m
    println("The problem has ", length(SOC_con_idx_new) - 1, " SOCs in constraints, SOC row number is ", SOC_con_idx_new[end] - SOC_con_idx_new[1], ".")

    # don't count free constraints

    # objective function
    c = zeros(n)
    for i in 1:length(problem.objacoord)
        c[problem.objacoord[i][1]] = problem.objacoord[i][2]
    end
    c0 = problem.objoffset
    if problem.sense == "max"
        c = -c
        c0 = -c0
    end


    # variable bounds
    lb = []
    ub = []
    idx_linear = []
    for i in 1:length(problem.var)
        if problem.var[i][1] == "L+"
            lb = vcat(lb, zeros(problem.var[i][2]))
            ub = vcat(ub, Inf * ones(problem.var[i][2]))
            idx_linear = vcat(idx_linear, i)
        end
        if problem.var[i][1] == "L-"
            lb = vcat(lb, -Inf * ones(problem.var[i][2]))
            ub = vcat(ub, zeros(problem.var[i][2]))
            idx_linear = vcat(idx_linear, i)
        end
        if problem.var[i][1] == "L="
            lb = vcat(lb, zeros(problem.var[i][2]))
            ub = vcat(ub, zeros(problem.var[i][2]))
            idx_linear = vcat(idx_linear, i)
        end
        if problem.var[i][1] == "F"
            lb = vcat(lb, -Inf * ones(problem.var[i][2]))
            ub = vcat(ub, Inf * ones(problem.var[i][2]))
            idx_linear = vcat(idx_linear, i)
        end
    end
    number_linear_vars = length(lb)
    println("The problem has ", number_linear_vars, " linear variables.")

    SOC_var_idx = [number_linear_vars + 1]
    # SOC variables
    for i in 1:length(problem.var)
        if problem.var[i][1] == "Q"
            len = problem.var[i][2]
            SOC_var_idx = vcat(SOC_var_idx, SOC_var_idx[end] + len)
        end
    end
    println("The problem has ", length(SOC_var_idx) - 1, " SOCs in variables, SOC variable number is ", SOC_var_idx[end] - SOC_var_idx[1], ".")
    number_lu_x = length(lb)
    println(SOC_var_idx[end])
    @assert SOC_var_idx[end] - 1 == n

    idx_SOC = setdiff(1:n, idx_linear)
    A_new = hcat(A_new[:, idx_linear], A_new[:, idx_SOC])
    c = vcat(c[idx_linear], c[idx_SOC])

    return SOCP_info_cpu(sparse([], [], [], n, n), c, sparse(A_new), sparse(A_new)', b_new, SOC_con_idx_new, number_ineq, number_eq, lb, ub, SOC_var_idx, number_lu_x, c0)

end

function socp_formulation_Axb!(socp::SOCP_info_cpu)
    m, n = size(socp.A)
    number_SOC_con = m - socp.number_eq - socp.number_ineq
    A_eq = socp.A[1:socp.number_eq, :]
    A_ineq = socp.A[socp.number_eq+1:socp.number_eq+socp.number_ineq, :]
    A_SOC = socp.A[socp.number_eq+socp.number_ineq+1:end, :]
    c_lu = socp.c[1:socp.number_lu_x]
    c_SOC = socp.c[socp.number_lu_x+1:end]

    # A_new = hcat(socp.A, vcat(sparse(zeros(socp.number_eq, socp.number_eq)), -spdiagm(ones(socp.number_ineq)), -spdiagm(ones(number_SOC_con))))
    A_eq = hcat(A_eq, sparse(zeros(socp.number_eq, socp.number_ineq + number_SOC_con)))
    A_ineq = hcat(A_ineq, -spdiagm(ones(socp.number_ineq)), sparse(zeros(socp.number_ineq, number_SOC_con)))
    A_SOC = hcat(A_SOC, sparse(zeros(number_SOC_con, socp.number_ineq)), -spdiagm(ones(number_SOC_con)))
    A_new = vcat(A_eq, A_ineq, A_SOC)


    A_new_lu = hcat(A_new[:, 1:socp.number_lu_x], A_new[:, n+1:n+socp.number_ineq])
    A_new_SOC = hcat(A_new[:, socp.number_lu_x+1:n], A_new[:, n+socp.number_ineq+1:end])
    A_new2 = hcat(A_new_lu, A_new_SOC)
    c_new = vcat(c_lu, zeros(socp.number_ineq), c_SOC, zeros(number_SOC_con))

    l_new = vcat(socp.l, zeros(socp.number_ineq))
    u_new = vcat(socp.u, Inf * ones(socp.number_ineq))

    if length(socp.SOC_var_idx) > 1

        for i in 1:length(socp.SOC_var_idx)
            socp.SOC_var_idx[i] += socp.number_ineq
        end
        for i in 1:length(socp.SOC_con_idx)-1
            len = socp.SOC_con_idx[i+1] - socp.SOC_con_idx[i]
            push!(socp.SOC_var_idx, socp.SOC_var_idx[end] + len)
        end
    else
        # println(socp.SOC_con_idx)
        # println(socp.SOC_var_idx)
        socp.SOC_var_idx = [socp.number_lu_x + socp.number_ineq + 1]
        for i in 1:length(socp.SOC_con_idx)-1
            len = socp.SOC_con_idx[i+1] - socp.SOC_con_idx[i]
            push!(socp.SOC_var_idx, socp.SOC_var_idx[end] + len)
        end
    end

    socp.number_lu_x = socp.number_lu_x + socp.number_ineq
    socp.number_eq = m
    socp.number_ineq = 0
    socp.SOC_con_idx = [m + 1]

    n = size(A_new2, 2)
    socp.Q = sparse([], [], [], n, n)  # No quadratic term in the Axb formulation
    socp.A = A_new2
    socp.c = c_new
    socp.AT = A_new2'
    socp.l = l_new
    socp.u = u_new
end


function socp_formulation_SOC_Axb!(socp::SOCP_info_cpu)
    m, n = size(socp.A)
    number_SOC_con = m - socp.number_eq - socp.number_ineq
    A_eq = socp.A[1:socp.number_eq, :]
    A_ineq = socp.A[socp.number_eq+1:socp.number_eq+socp.number_ineq, :]
    A_SOC = socp.A[socp.number_eq+socp.number_ineq+1:end, :]
    c_lu = socp.c[1:socp.number_lu_x]
    c_SOC = socp.c[socp.number_lu_x+1:end]

    # A_new = hcat(socp.A, vcat(sparse(zeros(socp.number_eq, socp.number_eq)), -spdiagm(ones(socp.number_ineq)), -spdiagm(ones(number_SOC_con))))
    A_eq = hcat(A_eq, sparse(zeros(socp.number_eq, number_SOC_con)))
    A_ineq = hcat(A_ineq, sparse(zeros(socp.number_ineq, number_SOC_con)))
    A_SOC = hcat(A_SOC, -spdiagm(ones(number_SOC_con)))
    A_new = vcat(A_eq, A_ineq, A_SOC)


    # A_new_lu = hcat(A_new[:, 1:socp.number_lu_x])
    # A_new_SOC = hcat(A_new[:, socp.number_lu_x+1:n], A_new[:, n+1:end])
    # A_new2 = hcat(A_new_lu, A_new_SOC)
    A_new2 = A_new
    c_new = vcat(c_lu, c_SOC, zeros(number_SOC_con))

    l_new = socp.l
    u_new = socp.u

    for i in 1:length(socp.SOC_con_idx)-1
        len = socp.SOC_con_idx[i+1] - socp.SOC_con_idx[i]
        push!(socp.SOC_var_idx, socp.SOC_var_idx[end] + len)
    end

    # socp.number_lu_x = socp.number_lu_x + socp.number_ineq
    # socp.number_eq = m
    # socp.number_ineq = 0
    socp.SOC_con_idx = [m + 1]

    n = size(A_new2, 2)
    socp.Q = sparse([], [], [], n, n)  # No quadratic term in the Axb formulation
    socp.A = A_new2
    socp.c = c_new
    socp.AT = A_new2'
    socp.l = l_new
    socp.u = u_new
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
    println("dimension of b is ", size(AL))
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
    # AI = A[idxI, :]
    AG = A[idxG, :]
    AL = A[idxL, :]
    A = vcat(AE, AG, -AL)
    b = vcat(lp_AL[idxE], lp_AL[idxG], -lp_AU[idxL])

    n = size(A, 2)
    # ALE = lp_AL[idxE]
    # AUE = lp_AU[idxE]
    # ALI = lp_AL[idxI]
    # AUI = lp_AU[idxI]

    # A = vcat(AE, AI)
    # AL = vcat(ALE, ALI)
    # AU = vcat(AUE, AUI)


    m1 = length(idxE)

    # idxL is from m1+1 to m
    m = length(b)
    idxI = (m1+1):m

    println("number of equality constraints = ", m1)
    println("number of inequality constraints = ", m - m1)

    standard_qp = SOCP_info_cpu(Q, c, A, A', b, [m+1], m-m1, m1, l, u, [n+1], n, c0)

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
    SOC_con_scaling_factor = ones(m)
    SOC_con_scaling_factor_inv = ones(m)
    SOC_var_scaling_factor = ones(n)
    SOC_var_scaling_factor_inv = ones(n)

    # Preallocate temporary arrays
    temp_norm_A_row = zeros(m)
    temp_norm_A_col = zeros(n)
    temp_norm_Q = zeros(n)
    DR = spdiagm(temp_norm_A_row)
    DC = spdiagm(temp_norm_A_col)

    # AL_nInf = copy(qp.AL)
    # AU_nInf = copy(qp.AU)
    # AL_nInf[qp.AL.==-Inf] .= 0.0
    # AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info = Scaling_info_cpu(copy(qp.l), copy(qp.u), row_norm, col_norm, 1, 1, 1, 1, norm(qp.b, Inf), norm(qp.c, Inf), SOC_con_scaling_factor, SOC_con_scaling_factor_inv, SOC_var_scaling_factor, SOC_var_scaling_factor_inv)
    println("norm_b_org: ", scaling_info.norm_b_org)
    println("norm_c_org: ", scaling_info.norm_c_org)

    if use_Ruiz_scaling
        println("Using Ruiz scaling")
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

            # Apply uniform scalar scaling to SOC constraints
            for i in 1:length(qp.SOC_con_idx)-1
                start_idx = qp.SOC_con_idx[i]
                end_idx = qp.SOC_con_idx[i+1] - 1
                maxv = maximum(temp_norm_A_row[start_idx:end_idx])
                for j in start_idx:end_idx
                    temp_norm_A_row[j] = maxv
                end
            end

            # Apply uniform scalar scaling to SOC variables
            for i in 1:length(qp.SOC_var_idx)-1
                start_idx = qp.SOC_var_idx[i]
                end_idx = qp.SOC_var_idx[i+1] - 1
                maxv = maximum(temp_norm_A_col[start_idx:end_idx])
                for j in start_idx:end_idx
                    temp_norm_A_col[j] = maxv
                end
            end

            row_norm .*= temp_norm_A_row
            col_norm .*= temp_norm_A_col


            DR .= spdiagm(1.0 ./ temp_norm_A_row)
            DC .= spdiagm(1.0 ./ temp_norm_A_col)
            # temp_norm2 .= max.(temp_norm2,1.0)

            qp.Q .= DC * qp.Q * DC
            qp.c ./= temp_norm_A_col
            qp.A .= DR * qp.A * DC

            qp.b ./= temp_norm_A_row
            qp.l .*= temp_norm_A_col[1:qp.number_lu_x]
            qp.u .*= temp_norm_A_col[1:qp.number_lu_x]
        end
    end



    if use_l2_scaling
        println("Using l2 scaling")
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

        # for i in 1:length(qp.SOC_con_idx)-1
        #     start_idx = qp.SOC_con_idx[i]
        #     end_idx = qp.SOC_con_idx[i+1] - 1
        #     v = 0.0
        #     for j in start_idx:end_idx
        #         v += temp_norm_A_row[j]^2
        #     end
        #     v = sqrt(v)
        #     for j in start_idx:end_idx
        #         temp_norm_A_row[j] = v
        #     end
        # end

        # for i in 1:length(qp.SOC_var_idx)-1
        #     start_idx = qp.SOC_var_idx[i]
        #     end_idx = qp.SOC_var_idx[i+1] - 1
        #     v = 0.0
        #     for j in start_idx:end_idx
        #         v += temp_norm_A_col[j]^2
        #     end
        #     v = sqrt(v)
        #     for j in start_idx:end_idx
        #         temp_norm_A_col[j] = v
        #     end
        # end

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col

        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)
        # temp_norm2 .= max.(temp_norm2,1.0)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC

        qp.b ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col[1:qp.number_lu_x]
        qp.u .*= temp_norm_A_col[1:qp.number_lu_x]
    end

    if use_Pock_Chambolle_scaling
        println("Using Pock-Chambolle scaling")
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

        # Apply uniform scalar scaling to SOC constraints
        for i in 1:length(qp.SOC_con_idx)-1
            start_idx = qp.SOC_con_idx[i]
            end_idx = qp.SOC_con_idx[i+1] - 1
            maxv = maximum(temp_norm_A_row[start_idx:end_idx])
            for j in start_idx:end_idx
                temp_norm_A_row[j] = maxv
            end
        end

        # Apply uniform scalar scaling to SOC variables
        for i in 1:length(qp.SOC_var_idx)-1
            start_idx = qp.SOC_var_idx[i]
            end_idx = qp.SOC_var_idx[i+1] - 1
            maxv = maximum(temp_norm_A_col[start_idx:end_idx])
            for j in start_idx:end_idx
                temp_norm_A_col[j] = maxv
            end
        end

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col


        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)
        # temp_norm2 .= max.(temp_norm2,1.0)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC

        qp.b ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col[1:qp.number_lu_x]
        qp.u .*= temp_norm_A_col[1:qp.number_lu_x]
    end

    if use_bc_scaling
        println("Using b and c scaling")
        # bc_scaling!(qp, scaling_info)
        b_scale = 1 + norm(qp.b)
        c_scale = 1 + norm(qp.c)
        # b_scale = norm(qp.x0)+1
        # c_scale = norm(qp.y0)+1
        println("b_scale: ", b_scale)
        println("c_scale: ", c_scale)
        qp.Q .*= b_scale / c_scale
        qp.b ./= b_scale
        qp.c ./= c_scale
        qp.l ./= b_scale
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end


    # ========== ORIGINAL CODE (COMMENTED OUT - Elementwise Scaling) ==========
    # NOTE: This elementwise scaling transforms SOC into ellipsoidal cone,
    #       requiring expensive bisection for projection (100 iterations).
    #       Kept for reference only.
    # for i in 1:length(qp.SOC_con_idx)-1
    #     start_idx = qp.SOC_con_idx[i]
    #     end_idx = qp.SOC_con_idx[i+1] - 1
    #     for j in start_idx:end_idx
    #         SOC_con_scaling_factor[j] = 1 / row_norm[j] / scaling_info.b_scale
    #     end
    #     for j in start_idx+1:end_idx
    #         SOC_con_scaling_factor[j] /= SOC_con_scaling_factor[start_idx]
    #         SOC_con_scaling_factor_inv[j] = 1 / SOC_con_scaling_factor[j]
    #     end
    #     SOC_con_scaling_factor[start_idx] = 1.0
    #     SOC_con_scaling_factor_inv[start_idx] = 1.0
    # end
    # ========================================================================

    # NEW: Scalar (uniform) scaling per SOC constraint using max norm
    # This preserves SOC structure and enables simple closed-form projection
    for i in 1:length(qp.SOC_con_idx)-1
        start_idx = qp.SOC_con_idx[i]
        end_idx = qp.SOC_con_idx[i+1] - 1
        
        # Use MAXIMUM norm for uniform scalar scaling (best stability)
        max_norm = maximum(row_norm[start_idx:end_idx])
        scale = 1 / max_norm / scaling_info.b_scale
        
        # Apply SAME scaling to ALL elements of this SOC
        for j in start_idx:end_idx
            SOC_con_scaling_factor[j] = scale
            SOC_con_scaling_factor_inv[j] = 1 / scale
        end
    end

    # ========== ORIGINAL CODE (COMMENTED OUT - Elementwise Scaling) ==========
    # NOTE: This elementwise scaling transforms SOC into ellipsoidal cone,
    #       requiring expensive bisection for projection (100 iterations).
    #       Kept for reference only.
    # for i in 1:length(qp.SOC_var_idx)-1
    #     start_idx = qp.SOC_var_idx[i]
    #     end_idx = qp.SOC_var_idx[i+1] - 1
    #     for j in start_idx:end_idx
    #         SOC_var_scaling_factor[j] = col_norm[j] / scaling_info.c_scale
    #     end
    #     for j in start_idx+1:end_idx
    #         SOC_var_scaling_factor[j] /= SOC_var_scaling_factor[start_idx]
    #         SOC_var_scaling_factor_inv[j] = 1 / SOC_var_scaling_factor[j]
    #     end
    #     SOC_var_scaling_factor[start_idx] = 1.0
    #     SOC_var_scaling_factor_inv[start_idx] = 1.0
    # end
    # ========================================================================

    # NEW: Scalar (uniform) scaling per SOC variable using max norm
    # This preserves SOC structure and enables simple closed-form projection
    for i in 1:length(qp.SOC_var_idx)-1
        start_idx = qp.SOC_var_idx[i]
        end_idx = qp.SOC_var_idx[i+1] - 1
        
        # Use MAXIMUM norm for uniform scalar scaling (best stability)
        max_norm = maximum(col_norm[start_idx:end_idx])
        scale = 1 / max_norm / scaling_info.c_scale
        
        # Apply SAME scaling to ALL elements of this SOC
        for j in start_idx:end_idx
            SOC_var_scaling_factor[j] = scale
            SOC_var_scaling_factor_inv[j] = 1 / scale
        end
    end

    # temp_norm_Q[iszero.(temp_norm_Q)] .= 1.0

    # the diagonal matrix of the Q matrix
    # diag_Q = diag(qp.Q)
    # Q_is_diag = (sum(temp_norm_Q .== diag_Q) == length(temp_norm_Q))
    # qp.Q_is_diag = Q_is_diag
    # qp.diag_Q = diag_Q

    scaling_info.norm_b = norm(qp.b)
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
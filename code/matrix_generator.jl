using LinearAlgebra, Random, SparseArrays

has_multi_edges(H, nedges) = ( nnz(H) != nedges )

# build a parity check matrix given degree profile, size and number of edges
#  assumes all of the parameters are consistent
#  follows Luby, "Efficient erasure correcting codes", doi: 10.1109/18.910575.
# polynomials are from the NODE POINT OF VIEW
function ldpc_matrix(n::Integer, m::Integer, nedges::Integer, Lambda, Rho,
    edgesleft=fill(zero(n), nedges), edgesright=copy(edgesleft);
    rng = Random.GLOBAL_RNG,
    vperm = randperm(rng, n), fperm = randperm(rng, m),
    maxtrials=1000,
    isacceptable = !has_multi_edges)

    check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for t in 1:maxtrials
        H = one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright,
            rng=rng, vperm=vperm, fperm=fperm)
        isacceptable(H, nedges) && return H
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright;
    rng = Random.GLOBAL_RNG,
    vperm = randperm(rng, n), fperm = randperm(rng, m))
    v = r = 1
    for i = 1:lastindex(Lambda)
        ni = Int(round(n*Lambda[i], digits=8))
        for _ in 1:ni
            edgesleft[r:r+i-1] .= vperm[v]
            v += 1; r += i
        end
    end
    shuffle!(rng, edgesleft)
    f = r = 1
    for j = 1:lastindex(Rho)
        nj = Int(round(m*Rho[j], digits=10))
        for _ in 1:nj
            edgesright[r:r+j-1] .= fperm[f]
            f += 1; r += j
        end
    end
    sparse(edgesleft, edgesright, trues(nedges), n, m)
end

function check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for (i,l) in pairs(Lambda)
        if !isinteger(round(n*l, digits=8))
           println("Non integer number of variables with degree i: got ", n*l) 
        end
    end
    for (j,r) in pairs(Rho)
        if !isinteger(round(m*r, digits=8))
            println("Non integer number of factors with degree j: got ", m*r) 
         end
    end
    nedges_variables = n*sum(i*l for (i,l) in pairs(Lambda))
    nedges_factors = m*sum(j*r for (j,r) in pairs(Rho))
    if nedges_variables == nedges_factors
        nedges = nedges_variables
    else
        
    end

    @assert isapprox(n*sum(i*l for (i,l) in pairs(Lambda)), nedges, atol=1e-1) 
    @assert isapprox(m*sum(j*r for (j,r) in pairs(Rho)), nedges, atol=1e-1)
    @assert isapprox(sum(Lambda), 1, atol=1e-8)
    @assert isapprox(sum(Rho), 1, atol=1e-8)
end

# switch between node and edge degree convention
function edges2nodes(lambda::Vector{<:Real}, 
        rho::Vector{<:Real})
    lambda_new = [lambda[i]/i for i in eachindex(lambda)]
    rho_new = [rho[j]/j for j in eachindex(rho)]
    return lambda_new./sum(lambda_new), rho_new./sum(rho_new)
end
function nodes2edges(lambda::Vector{<:Real}, 
    rho::Vector{<:Real})
    lambda_new = [lambda[i]*i for i in eachindex(lambda)]
    rho_new = [rho[j]*j for j in eachindex(rho)]
    return lambda_new./sum(lambda_new), rho_new./sum(rho_new)
end

# build a valid degree profile as close as possible to the given one
function valid_degrees(Prows, Pcols, A::Int=2*3*5*7; B::Int=1,
        verbose=false, tol=1e-2)
    # convert to rationals
    Pr = Rational.(round.(Int, Prows*A), A)
    Pc = Rational.(round.(Int, Pcols*A), A)
    # make sure that both P's still sum to 1
    nz_rows = findall(!iszero, Pr); nz_cols = findall(!iszero, Pc)
    Pr[nz_rows[end]] = 1 - sum(Pr[nz_rows[1:end-1]])
    Pc[nz_cols[end]] = 1 - sum(Pc[nz_cols[1:end-1]])

    R = dot(Pr,eachindex(Pr)) / dot(Pc,eachindex(Pc))
    # find least common denominator
    m = lcm(((R*Pc..., Pr..., R) .|> denominator)...)
    nrows = B*m
    ncols = Int(nrows*R)
    nedges = Int(nrows*dot(Pr,eachindex(Pr)))
    Pr_new = float(Pr)
    Pc_new = float(Pc)
    if verbose
        err = max(maximum(abs,Prows.-Pr_new), maximum(abs,Pcols.-Pc_new))
        err > tol && error("Discrepancy with given degree profile larger than tol $tol")
        println("Max discrepancy with given degree profile: err=",err)
    end
    return nrows, ncols, nedges, Pr_new, Pc_new
end

function rate(Λ::AbstractVector, K::AbstractVector)
    α = (Λ'eachindex(Λ)) / (K'eachindex(K))
    1 - α
end

function full_adjmat(H::SparseMatrixCSC, T::Type=eltype(H))
    m,n = size(H)
    A = [zeros(T,m,m) H;
         permutedims(H) zeros(T,n,n)]
    return A
end

function isfullcore(H::SparseMatrixCSC, Ht=permutedims(H))
    rowperm, dep, indep = leaf_removal(H, Ht)
    length(rowperm) == size(H, 1)
end

function isfullrank(H::SparseMatrixCSC, Ht=permutedims(H))
    B, indep = findbasis_slow(BitMatrix(H))
    length(indep) == size(H,2) - size(H,1)
end

#### GF(q)

function ldpc_matrix_gfq(Q::Integer, n::Integer, m::Integer, nedges::Integer, 
    Lambda, Rho, edgesleft=fill(zero(n), nedges), edgesright=copy(edgesleft);
    rng = Random.GLOBAL_RNG, T::Type=Int,
    vperm = randperm(rng, n), fperm = randperm(rng, m),
    accept_multi_edges=true, maxtrials=1000, verbose=false,
    isacceptable = !has_multi_edges)

    check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for t in 1:maxtrials
        H = one_ldpc_matrix_gfq(Q, n, m, nedges, Lambda, Rho, edgesleft, edgesright,
            rng=rng, T=T, vperm=vperm, fperm=fperm)
        found = isacceptable(H, nedges)
        if found
           verbose && println("Factor graph generated after $t trials") 
           return H
        end
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function one_ldpc_matrix_gfq(Q, n, m, nedges, Lambda, Rho, edgesleft, edgesright;
    rng = Random.GLOBAL_RNG, T::Type=Int,
    vperm = randperm(rng, n), fperm = randperm(rng, m))
    v = r = 1
    for i = 1:lastindex(Lambda)
        ni = Int(round(n*Lambda[i], digits=10))
        for _ in 1:ni
            edgesleft[r:r+i-1] .= vperm[v]
            v += 1; r += i
        end
    end
    shuffle!(rng, edgesleft)
    f = r = 1
    for j = 1:lastindex(Rho)
        nj = Int(round(m*Rho[j], digits=10))
        for _ in 1:nj
            edgesright[r:r+j-1] .= fperm[f]
            f += 1; r += j
        end
    end
    nz = rand(rng, convert.(T,1:Q-1), nedges)
    combine(x,y) = rand(rng, (x,y))     # manages duplicates 
    sparse(edgesleft, edgesright, nz, n, m, combine)
end

function cycle_code(q::Int, n::Int, R::Real; kw...)
    Lambda = [0,1]
    nedges = 2n
    α = 1-R
    m = Int(round(α*n,digits=10))
    k = floor(Int, 2/α)
    s = k+1-2/α
    K = [fill(0,k-1); s; 1-s]
    K .*= K .> 1e-10
    K ./ sum(K)
    H = permutedims(ldpc_matrix_gfq(q, n, m, nedges, Lambda, K, 
        accept_multi_edges=false; kw...))
end

function cycle_code(n::Int, R::Real; kw...)
    Lambda = [0,1]
    nedges = 2n
    α = 1-R
    m = Int(round(α*n,digits=10))
    k = floor(Int, 2/α)
    s = k+1-2/α
    K = [fill(0,k-1); s; 1-s]
    K .*= K .> 1e-10
    K ./ sum(K)
    H = permutedims(ldpc_matrix(n, m, nedges, Lambda, K; kw...))
end
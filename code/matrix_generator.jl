using LinearAlgebra, Random, SparseArrays

# build a parity check matrix given degree profile, size and number of edges
#  assumes all of the parameters are consistent
#  follows Luby, "Efficient erasure correcting codes", doi: 10.1109/18.910575.
# polynomials are from the NODE POINT OF VIEW
function ldpc_matrix(n::Integer, m::Integer, nedges::Integer, Lambda, Rho,
    edgesleft=fill(zero(n), nedges), edgesright=copy(edgesleft);
    rng = MersenneTwister(0),
    vperm = randperm(rng, n), fperm = randperm(rng, m),
    accept_multi_edges=true, maxtrials=1000)

    check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for t in 1:maxtrials
        H = one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright,
            rng=rng, vperm=vperm, fperm=fperm)
        (nnz(H) == nedges || accept_multi_edges) && return H
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright;
    rng = MersenneTwister(0),
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
    sparse(edgesleft, edgesright, trues(nedges), n, m)
end

function check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for l in Lambda
        @assert isinteger(round(n*l, digits=10))
    end
    for r in Rho
        @assert isinteger(round(m*r, digits=10))
    end
    @assert isapprox(n*sum(i*l for (i,l) in pairs(Lambda)), nedges, atol=1e-8) 
    @assert isapprox(m*sum(j*r for (j,r) in pairs(Rho)), nedges, atol=1e-8)
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

function full_adjmat(H::SparseMatrixCSC, T::Type)
    m,n = size(H)
    A = [zeros(T,m,m) H;
         permutedims(H) zeros(T,n,n)]
    return A
end


#### GF(q)

function ldpc_matrix_gfq(Q::Integer, n::Integer, m::Integer, nedges::Integer, 
    Lambda, Rho, edgesleft=fill(zero(n), nedges), edgesright=copy(edgesleft);
    rng = MersenneTwister(0), T::Type=Int,
    vperm = randperm(rng, n), fperm = randperm(rng, m),
    accept_multi_edges=true, maxtrials=1000)

    check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for t in 1:maxtrials
        H = one_ldpc_matrix_gfq(Q, n, m, nedges, Lambda, Rho, edgesleft, edgesright,
            rng=rng, T=T, vperm=vperm, fperm=fperm)
        (nnz(H) == nedges || accept_multi_edges) && return H
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function one_ldpc_matrix_gfq(Q, n, m, nedges, Lambda, Rho, edgesleft, edgesright;
    rng = MersenneTwister(0), T::Type=Int,
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
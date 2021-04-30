include("slim_graphs.jl")
include("bp.jl")

using OffsetArrays, Statistics

struct LDGM{F,M}
    G :: SparseMatrixCSC{F,Int}     # size (nfactors,nvars)
    X :: SparseMatrixCSC{Int,Int}   # to get neighbors of factor nodes
    h :: Vector{M}                  # messages var -> factor_perm
    u :: Vector{M}                  # messages factor -> var
    efield :: Vector{M}             # external field
    belief :: Vector{M}
end
nfactors(bp::LDGM) = size(bp.G,1)
nvars(bp::LDGM) = size(bp.G,2)
function LDGM(G::SparseMatrixCSC, efield = fill((0.5,0.5), size(G,1)))
    n,k = size(G)
    X = sparse(SparseMatrixCSC(size(G)...,G.colptr,G.rowval,collect(1:length(G.nzval)))')
    h = fill((0.5,.5), nnz(G))
    u = fill((0.5,.5), nnz(G))
    belief = fill((0.5,.5), k)
    LDGM(G, X, h, u, copy(efield), belief)
end

# function LDGM(n, m, nedges, Lambda, Rho, efield=fill((0.5,0.5),size(G,1)), 
#         h=fill((0.5,.5),nedges), u=fill((0.5,.5),nedges),  
#         belief=fill((0.5,.5),size(G,2)), args...; kw...)
#     G = sparse(ldgm_matrix(n, m, nedges, Lambda, Rho, args...; kw...)')
#     X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
#     LDGM(G, X, h, u, copy(efield), belief)
# end

msg_conv(h1::Tuple, h2::Tuple) = (h1[1]*h2[1]+h1[2]*h2[2], h1[1]*h2[2]+h1[2]*h2[1]) 
msg_mult(u1::Tuple, u2::Tuple) = u1 .* u2
msg_maxconv(h1::Tuple, h2::Tuple) = (max(h1[1]+h2[1],h1[2]+h2[2]), max(h1[1]+h2[2],h1[2]+h2[1])) 
msg_sum(u1::Tuple, u2::Tuple) = u1 .+ u2

function update_var_bp!(bp::LDGM, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.G, i)
    b = (.5,.5)
    for a in ∂i
        hnew = (.5,.5)
        for c in ∂i
            c==a && continue
            hnew = msg_mult(hnew, bp.u[c])
        end
        hnew = hnew ./ sum(hnew)
        ε = max(ε, abs(hnew[1]-bp.h[a][1]), abs(hnew[2]-bp.h[a][2]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_mult(b, bp.u[a]) 
    end
    iszero(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = b ./ sum(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_bp!(bp::LDGM, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    for i in ∂a
        unew = bp.efield[a]
        for j in ∂a
            j==i && continue
            unew = msg_conv(unew, bp.h[j])
        end
        unew = unew ./ sum(unew)  
        ε = max(ε, abs(unew[1]-bp.u[i][1]), abs(unew[2]-bp.u[i][2]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end

function update_var_ms!(bp::LDGM, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.G, i)
    b = (0.0,0.0)
    for a in ∂i
        hnew = (0.0,0.0)
        for c in ∂i
            c==a && continue
            hnew = msg_sum(hnew, bp.u[c])
        end
        hnew = hnew .- maximum(hnew)
        ε = max(ε, abs(hnew[1]-bp.h[a][1]), abs(hnew[2]-bp.h[a][2]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_sum(b, bp.u[a]) 
    end
    isnan(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = b .- maximum(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_ms!(bp::LDGM, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    for i in ∂a
        unew = bp.efield[a]
        for j in ∂a
            j==i && continue
            unew = msg_maxconv(unew, bp.h[j])
        end
        unew = unew .- maximum(unew)  
        ε = max(ε, abs(unew[1]-bp.u[i][1]), abs(unew[2]-bp.u[i][2]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end


function iteration!(bp::LDGM; maxiter=10^3, tol=1e-12, damp=0.0, rein=0.0, 
        update_f! = update_factor_bp!, update_v! = update_var_bp!,
        callback=(x...)->false)
    # pre-allocate memory for the indices of neighbors
    factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.G,1)]
    ε = 0.0
    for it = 1:maxiter
        ε = 0.0
        for a = 1:size(bp.G,1)
            errf = update_f!(bp, a, factor_neigs[a], damp=damp)
            errf == -1 && return -1,it
            ε = max(ε, errf)
        end
        for i = 1:size(bp.G,2)
            errv = update_v!(bp, i, damp=damp, rein=rein)
            errv == -1 && return -1,it
            ε = max(ε, errv)
        end
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

function performance(bp::LDGM, s::AbstractVector)
    x = argmax.(bp.belief) .== 2
    z = bp.G*x .% 2
    y = s .== -1
    dist = mean(z .!= y)
    ovl = 1-2*dist
    ovl, dist
end

function avg_dist(bp::LDGM, s::AbstractVector)
    ovl = 0.0
    n = nfactors(bp)
    Gt = sparse(bp.G')
    for a in 1:n
        σ = s[a]
        ∂a = rowvals(Gt)[nzrange(Gt, a)]
        for i in ∂a
            σ *= (bp.belief[i][1]-bp.belief[i][2])
        end
        ovl += σ
    end
    0.5*(1-ovl/n)
end


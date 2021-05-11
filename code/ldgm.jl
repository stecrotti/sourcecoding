include("slim_graphs.jl")
include("matrix_generator.jl")

using OffsetArrays, Statistics

struct LDGM{F,M}
    G :: SparseMatrixCSC{F,Int}     # size (nfactors,nvars)
    X :: SparseMatrixCSC{Int,Int}   # to get neighbors of factor nodes
    h :: Vector{M}                  # messages var -> factor
    u :: Vector{M}                  # messages factor -> var
    efield :: Vector{M}             # external field for source
    belief :: Vector{M}
    s :: Vector{Int}                # source vector
    prior :: Vector{M}              # external field for variables
end
nfactors(bp::LDGM) = size(bp.G,1)
nvars(bp::LDGM) = size(bp.G,2)
function LDGM(G::SparseMatrixCSC, s, efield = fill((0.5,0.5), size(G,1)))
    n,k = size(G)
    X = sparse(SparseMatrixCSC(size(G)...,G.colptr,G.rowval,collect(1:length(G.nzval)))')
    h = fill((0.5,.5), nnz(G))
    u = fill((0.5,.5), nnz(G))
    belief = fill((0.5,.5), k)
    prior = fill((0.5,.5), k)
    LDGM(G, X, h, u, copy(efield), belief, s, prior)
end

function ldgm(n, k, nedges, Lambda, Rho, s=rand((-1,1),n); 
        efield=fill((0.5,0.5),n), 
        h=fill((0.5,.5),nedges), u=fill((0.5,.5),nedges),  
        belief=fill((0.5,.5),k), prior = fill((0.5,.5), k), kw...)
    G = ldgm_matrix(n, k, nedges, Lambda, Rho; kw...)
    X = sparse(SparseMatrixCSC(size(G)...,G.colptr,G.rowval,collect(1:length(G.nzval)))')
    LDGM(G, X, h, u, copy(efield), belief, s, prior)
end

function ldgm_matrix(n::Int, k::Int, nedges::Int, Lambda, Rho;
    edgesleft=zeros(Int, nedges), edgesright=copy(edgesleft),
    accept_multi_edges=true, maxtrials=1000)

    check_consistency_polynomials_ldgm(n,k,nedges,Lambda,Rho)
    for t in 1:maxtrials
        # call LDPC but with rho and lambda inverted
        H = one_ldpc_matrix(n, k, nedges, Rho, Lambda, edgesleft, edgesright)
        (nnz(H) == nedges || accept_multi_edges) && return H
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function check_consistency_polynomials_ldgm(n,k,nedges,Lambda,Rho)
    for l in Lambda
        @assert isinteger(round(k*l, digits=10))
    end
    for r in Rho
        @assert isinteger(round(n*r, digits=10))
    end
    @assert isapprox(k*sum(i*l for (i,l) in pairs(Lambda)), nedges, atol=1e-8) 
    @assert isapprox(n*sum(j*r for (j,r) in pairs(Rho)), nedges, atol=1e-8)
    @assert isapprox(sum(Lambda), 1, atol=1e-8)
    @assert isapprox(sum(Rho), 1, atol=1e-8)
end

msg_conv(h1::Tuple, h2::Tuple) = (h1[1]*h2[1]+h1[2]*h2[2], h1[1]*h2[2]+h1[2]*h2[1]) 
msg_mult(u1::Tuple, u2::Tuple) = u1 .* u2
msg_maxconv(h1::Tuple, h2::Tuple) = (max(h1[1]+h2[1],h1[2]+h2[2]), max(h1[1]+h2[2],h1[2]+h2[1])) 
msg_sum(u1::Tuple, u2::Tuple) = u1 .+ u2

function update_var_bp!(bp::LDGM, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.G, i)
    b = bp.prior[i]
    for a in ∂i
        hnew = bp.prior[i]
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
    try
    bp.prior[i] = bp.prior[i] .* bp.belief[i].^rein
    catch
        @show bp.belief[i]
        return -1
    end
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
    b = bp.prior[i]
    for a in ∂i
        hnew = bp.prior[i]
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
    bp.prior[i] = bp.prior[i] .+ rein.*bp.belief[i]
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
        dist = fill(NaN,maxiter),
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.G,1)],
        callback=(x...)->false)
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
        dist[it] = performance(bp)[2]
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

function performance(bp::LDGM)
    x = argmax.(bp.belief) .== 2
    z = bp.G*x .% 2
    y = bp.s .== -1
    dist = mean(z .!= y)
    ovl = 1-2*dist
    ovl, dist
end

function avg_dist(bp::LDGM)
    ovl = 0.0
    n = nfactors(bp)
    Gt = sparse(bp.G')
    for a in 1:n
        σ = bp.s[a]
        ∂a = rowvals(Gt)[nzrange(Gt, a)]
        for i in ∂a
            σ *= (bp.belief[i][1]-bp.belief[i][2])
        end
        ovl += σ
    end
    0.5*(1-ovl/n)
end

magnetiz(t) = (t[1]-t[2]) / 2
function free_energy(bp::LDGM)
    n,k = size(bp.G)
    # fa
    Fa = 0.0
    for a in 1:n
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]
        p = bp.s[a]*mapreduce(magnetiz, *, bp.h[∂a])
        Fa += -1+2*(p<0)
    end
    # fi
    Fi = 0.0
    for i in 1:k
        ∂i = nzrange(bp.G, i)
        Fi += -abs(sum(magnetiz,u for u in bp.u[∂i]))
        Fi += sum(abs(u) for u in magnetiz.(bp.u[∂i]))
    end
    #fai
    Fai = 0.0
    for ai in eachindex(bp.u)
        Fai += -abs(magnetiz(bp.h[ai].+bp.u[ai])) + abs(magnetiz(bp.h[ai])) 
            + abs(magnetiz(bp.u[ai]))
    end
    F = (Fi+Fa-Fai) / n
end

#### DECIMATION

# try Tmax times to reach zero unsat with decimation
# returns nunsat, ovl, dist
function decimate!(bp::LDGM, efield; Tmax=1, 
        fair_decimation=false, 
        
        kw...)
    freevars = falses(nvars(bp))
    for t in 1:Tmax
        ε, nunsat, ovl, dist, iters = decimate1!(bp, efield, freevars; 
            fair_decimation = fair_decimation, factor_neigs = factor_neigs, kw...)
        print("Trial $t of $Tmax: ")
        ε == -1 && print("contradiction found. ")
        nunsat == 0 && return nunsat, ovl, dist
        freevars .= false
    end
    return -1, NaN, NaN
end

# 1 trial of decimation
function decimate!(bp::LDGM, efield; 
        u_init=fill((0.5,0.5), length(bp.u)), h_init=fill((0.5,0.5), length(bp.h)),
        prior_init=fill((0.5,0.5),length(bp.prior)),
        callback=(ε,nunsat,args...) -> (ε==-1||nunsat==0), 
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.G,1)],
        fair_decimation = false, kw...)
    freevars = trues(nvars(bp))
    # reset messages
    bp.h .= h_init; bp.u .= u_init; bp.prior .= prior_init
    bp.efield .= efield; fill!(bp.belief, (0.5,0.5))
    # warmup bp run
    ε, iters = iteration!(bp; tol=1e-15, kw...) 
    println("Avg distortion after 1st BP round: ", avg_dist(bp))
    ovl, dist = performance(bp)
    nfree = sum(freevars)
    callback(ε, bp, nfree, ovl, dist, iters, 0, -Inf) && return ε, ovl, dist, iters

    for t in 1:nfree
        maxfield, tofix, newfield = find_most_biased(bp, freevars, 
            fair_decimation = fair_decimation)
        freevars[tofix] = false
        # fix most decided variable by applying a strong field 
        bp.prior[tofix] = newfield
        ε, iters = iteration!(bp; kw...)
        ovl, dist = performance(bp)
        callback(ε, bp, nfree-t, ovl, dist, iters, t, maxfield) && return ε, ovl, dist, iters
    end
    ε, ovl, dist, iters
end

# returns max prob, idx of variable, sign of the belief
function find_most_biased(bp::LDGM, freevars::BitArray{1}; 
        fair_decimation=false)
    m = -Inf; mi = 1; s = 0
    newfields = [(1.0,0.0), (0.0,1.0)]
    newfield = (0.5,0.5)
    for (i,h) in pairs(bp.belief)
        if freevars[i] && maximum(h)>m
                m, q = findmax(h); mi = i
                if fair_decimation
                    # sample fixing field from the distribution given by the belief
                    newfield = rand() < h[1] ? newfields[1] : newfields[2]
                else
                    # use as fixing field the one corresponding to max belief
                    newfield = newfields[q]
                end
        end
    end
    m, mi, newfield
end

function cb_decimation(ε, bp::LDGM, nfree, ovl, dist, iters, step, maxfield, args...)
    @printf(" Step  %3d. Free = %3d. Maxfield = %1.2E. ε = %6.2E. Ovl = %.3f. Iters %d\n", 
            step, nfree, maxfield, ε, ovl, iters)
    return ε==-1
end


include("slim_graphs.jl")
include("cavity.jl")
include("matrix_generator.jl")

using OffsetArrays, Statistics, Printf

struct BPFull{F,M}
    H :: SparseMatrixCSC{F,Int}     # size (nfactors,nvars)
    X :: SparseMatrixCSC{Int,Int}   # to get neighbors of factor nodes
    h :: Vector{M}                  # messages var -> factor_perm
    u :: Vector{M}                  # messages factor -> var
    efield :: Vector{M}             # external field
    belief :: Vector{M}
end
nfactors(bp::BPFull) = size(bp.H,1)
nvars(bp::BPFull) = size(bp.H,2)
function BPFull(H::SparseMatrixCSC, efield = fill((0.5,0.5), size(H,2)))
    n = size(H,2)
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    neutralel = eltype(efield[1][1])==Int ? (0,0) : (0.5,0.5)
    h = fill(neutralel,nnz(H))
    u = fill(neutralel,nnz(H))
    belief = fill(neutralel,n)
    BPFull(H, X, h, u, copy(efield), belief)
end

function bp_full(n, m, nedges, Lambda, Rho, efield=fill((0.5,0.5),n), 
        args...; kw...)
    H = sparse(ldpc_matrix(n, m, nedges, Lambda, Rho, args...; kw...)')
    BPFull(H, copy(efield))
end

msg_conv(h1::Tuple, h2::Tuple) = (h1[1]*h2[1]+h1[2]*h2[2], h1[1]*h2[2]+h1[2]*h2[1]) 
msg_mult(u1::Tuple, u2::Tuple) = u1 .* u2
normalize_prob(u::Tuple) = u ./ sum(u)


function update_var_bp!(bp::BPFull, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    b = bp.efield[i] 
    for a in ∂i
        hnew = bp.efield[i] 
        for c in ∂i
            c==a && continue
            hnew = msg_mult(hnew, bp.u[c])
        end
        hnew = normalize_prob(hnew)
        ε = max(ε, abs(hnew[1]-bp.h[a][1]), abs(hnew[2]-bp.h[a][2]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_mult(b, bp.u[a]) 
    end
    iszero(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = normalize_prob(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_bp!(bp::BPFull, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    for i in ∂a
        unew = (1.0,0.0)
        for j in ∂a
            j==i && continue
            unew = msg_conv(unew, bp.h[j])
        end
        unew = normalize_prob(unew)    # should not be needed
        ε = max(ε, abs(unew[1]-bp.u[i][1]), abs(unew[2]-bp.u[i][2]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end

# alias for calling `iteration!` with maxsum updates
function iteration_ms!(ms::BPFull; kw...)
    iteration!(ms; update_f! = update_factor_ms!, 
        update_v! = update_var_ms!, kw...)
end

function iteration!(bp::BPFull; maxiter=10^3, tol=1e-12, damp=0.0, rein=0.0, 
        update_f! = update_factor_bp!, update_v! = update_var_bp!,
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.H,1)],
        callback=(it, ε, bp)->false)
    
    ε = 0.0
    for it = 1:maxiter
        ε = 0.0
        for a = 1:size(bp.H,1)
            errf = update_f!(bp, a, factor_neigs[a], damp=damp)
            errf == -1 && return -1,it
            ε = max(ε, errf)
        end
        for i = 1:size(bp.H,2)
            errv = update_v!(bp, i, damp=damp, rein=rein)
            errv == -1 && return -1,it
            ε = max(ε, errv)
        end
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

function parity(bp::BPFull, x::AbstractVector)
    z = sum(bp.H*x .% 2)
    return z 
end
function parity(H::SparseMatrixCSC, x::AbstractVector)
    z = sum(H*x .% 2)
    return z 
end
function distortion(x::AbstractVector, y::AbstractVector)
    d = 0
    for (xx,yy) in zip(x,y)
        d += sign(xx)!=sign(yy)
    end
    d/length(x)
end
function performance(bp::BPFull, s::AbstractVector,
        x=falses(length(s)))
    x .= argmax.(bp.belief) .== 2
    nunsat = parity(bp, x)
    y = s .== -1
    dist = mean(x .!= y)
    ovl = 1-2*dist
    nunsat, ovl, dist
end
function avg_dist(bp::BPFull, s::AbstractVector)
    ovl = 0.0
    for i in 1:nvars(bp)
        ovl += s[i]*(bp.belief[i][1]-bp.belief[i][2])
    end
    0.5(1-ovl/nvars(bp))
end


#### DECIMATION

# try Tmax times to reach zero unsat with decimation
# returns nunsat, ovl, dist
function decimate!(bp::BPFull, efield, indep, s; Tmax=1, 
        fair_decimation=false, 
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.H,1)],
        kw...)
    freevars = falses(nvars(bp)); freevars[indep] .= true
    for t in 1:Tmax
        ε, nunsat, ovl, dist, iters = decimate1!(bp, efield, freevars, s; 
            fair_decimation = fair_decimation, factor_neigs = factor_neigs, kw...)
        print("Trial $t of $Tmax: ")
        ε == -1 && print("contradiction found. ")
        println(nunsat, " unsat. Dist = ", round(dist,digits=3))
        nunsat == 0 && return nunsat, ovl, dist
        freevars .= false; freevars[indep] .= true
    end
    return -1, NaN, NaN
end

# 1 trial of decimation
function decimate1!(bp::BPFull, efield, freevars::BitArray{1}, s; 
        u_init=fill((0.5,0.5), length(bp.u)), h_init=fill((0.5,0.5), length(bp.h)),
        callback=(ε,nunsat,args...) -> (ε==-1||nunsat==0), 
        fair_decimation = false, kw...)
    # reset messages
    bp.h .= h_init; bp.u .= u_init
    bp.efield .= efield; fill!(bp.belief, (0.5,0.5))
    # warmup bp run
    ε, iters = iteration!(bp; tol=1e-15, kw...) 
    println("Avg distortion after 1st BP round: ", avg_dist(bp,s))
    nunsat, ovl, dist = performance(bp, s)
    nfree = sum(freevars)
    callback(ε, nunsat, bp, nfree, ovl, dist, iters, 0, -Inf) && return ε, nunsat, ovl, dist, iters

    for t in 1:nfree
        maxfield, tofix, newfield = find_most_biased(bp, freevars, 
            fair_decimation = fair_decimation)
        freevars[tofix] = false
        # fix most decided variable by applying a strong field 
        bp.efield[tofix] = newfield
        ε, iters = iteration!(bp; kw...)
        nunsat, ovl, dist = performance(bp, s)
        callback(ε, nunsat, bp, nfree-t, ovl, dist, iters, t, maxfield) && return ε, nunsat, ovl, dist, iters
    end
    ε, nunsat, ovl, dist, iters
end

# returns max prob, idx of variable, sign of the belief
function find_most_biased(bp::BPFull, freevars::BitArray{1}; 
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

function cb_decimation(ε, nunsat, bp::BPFull, nfree, ovl, dist, iters, step, maxfield, args...)
    @printf(" Step  %3d. Free = %3d. Maxfield = %1.2E. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
            step, nfree, maxfield, ε, nunsat,  ovl, iters)
    return ε==-1 || nunsat==0
end


#### MAX-SUM 
msg_maxconv(h1::Tuple, h2::Tuple) = (max(h1[1]+h2[1],h1[2]+h2[2]), max(h1[1]+h2[2],h1[2]+h2[1])) 
msg_sum(u1::Tuple, u2::Tuple) = u1 .+ u2
normalize_max(h::Tuple) = h .- maximum(h)

function update_var_ms!(bp::BPFull, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    b = bp.efield[i] 
    for a in ∂i
        hnew = bp.efield[i] 
        for c in ∂i
            c==a && continue
            hnew = msg_sum(hnew, bp.u[c])
        end
        hnew = normalize_max(hnew)
        ε = max(ε, abs(hnew[1]-bp.h[a][1]), abs(hnew[2]-bp.h[a][2]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_sum(b, bp.u[a]) 
    end
    isnan(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = normalize_max(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_ms!(bp::BPFull, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    for i in ∂a
        unew = (0.0,-Inf)
        for j in ∂a
            j==i && continue
            unew = msg_maxconv(unew, bp.h[j])
        end
        unew = normalize_max(unew)    
        ε = max(ε, abs(unew[1]-bp.u[i][1]), abs(unew[2]-bp.u[i][2]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end


### VANISHING FIELDS
Msg{T,M} = Tuple{Tuple{T,T}, Tuple{M,M}}

function msg_maxconv(h1::Msg{T,M}, h2::Msg{T,M}) where {T,M}
    u = msg_maxconv(h1[1], h2[1])
    utilde_plus = (u[1]==h1[1][1]+h2[1][1])*h1[2][1]*h2[2][1] +
                  (u[1]==h1[1][1]+h2[1][2])*h1[2][2]*h2[2][2]
    utilde_minus = (u[2]==h1[1][2]+h2[1][2])*h1[2][2]*h2[2][1] +
                   (u[2]==h1[1][2]+h2[1][1])*h1[2][2]*h2[2][1]
    (u, (utilde_plus,utilde_minus))::Msg{T,M}
end

function msg_sum(u1::Msg{T,M}, u2::Msg{T,M}) where {T,M}
    h = u1[1] .+ u2[1]
    htilde = u1[2] .* u2[2]
    (h, htilde)::Msg{T,M}
end

function normalize_max(h::Msg{T,M}) where {T,M}
    hnew = normalize_max(h[1])
    hnewtilde = normalize_prob(h[2])
    (hnew, hnewtilde)::Msg{T,M}
end

function ms_full_vanishing(H::SparseMatrixCSC, 
    efield::Vector{Tuple{Tuple{T,T},Tuple{M,M}}}) where {T,M}
    n = size(H,2)
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    neutralel = ((zero(T),zero(T)), (zero(M), zero(M)))
    h = fill(neutralel,nnz(H))
    u = fill(neutralel,nnz(H))
    belief = fill(neutralel,n)
    BPFull(H, X, h, u, copy(efield), belief)
end


### OBSERVABLES
function normalize_bp!(bp::BPFull)
    nor(t) = t ./ sum(t)
    bp.belief .= nor.(bp.belief)
    bp.h .= nor.(bp.h)
    bp.u .= nor.(bp.u)
    nothing
end
function normalize_ms!(bp::BPFull)
    nor(t) = t .- maximum(t)
    bp.belief .= nor.(bp.belief)
    bp.h .= nor.(bp.h)
    bp.u .= nor.(bp.u)
    nothing
end
function free_energy_entropic(bp::BPFull)
    # normalize_bp!(bp)
    zi = 0.0
    for i in 1:nvars(bp)
        ∂i = nzrange(bp.H, i)
        p = reduce(msg_mult, bp.u[∂i]) .* bp.efield[i]
        zi += sum(p)
    end
    za = 0.0
    for a in 1:nfactors(bp)
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]
        p = reduce(msg_conv, bp.h[∂a])
        za += p[1]
    end
    zai = 0.0
    for (u,h) in zip(bp.u,bp.h)
        p = u .* h
        zai += sum(p)
    end
    @show zi, za, zai
    F = -log(zi) -log(za) +log(zai)
    f = F / nvars(bp)
end

function entropy(bp::BPFull)
    # normalize_bp!(bp)
    # variables
    si = 0.0
    for i in 1:nvars(bp)
        p = bp.belief[i] .* log.(bp.belief[i])
        ∂i = nzrange(bp.H, i)
        si += sum(p)*(length(∂i)-1)
    end
    # factors
    sa = 0.0
    for a in 1:nfactors(bp)
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]
        ba = []
        for sigmas in Iterators.product(fill((-1,1),length(∂a))...)
            prod(sigmas)==-1 && continue
            idx = replace(collect(sigmas), -1=>2)
            b = prod(h[i] for (h,i) in zip(bp.h[∂a],idx))
            push!(ba,b)
        end
        ba = ba ./ sum(ba)
        sa += - sum(ba.*log.(ba))
    end

    S = si + sa
    s = S / nvars(bp)
end

function free_energy_energetic(ms::BPFull)
    # h(σ)=hσ+const => h=(h(+)-h(-))/2
    h = [(hh[1]-hh[2])/2 for hh in ms.h]
    u = [(uu[1]-uu[2])/2 for uu in ms.u]
    f = [(ss[1]-ss[2])/2 for ss in ms.efield]
    fi = 0.0
    for i in 1:nvars(ms)
        ∂i = nzrange(ms.H, i)
        fi += -abs(f[i]+sum(u[∂i]))
        fi += sum(abs,u[∂i])
    end
    fa = 0.0
    for a in 1:nfactors(ms)
        ∂a = nonzeros(ms.X)[nzrange(ms.X, a)]
        hh = h[∂a]
        fa += (prod(hh)<0) * 2*minimum(abs,hh)
    end
    fai = 0.0
    for ai in eachindex(ms.u)
        fai += -abs(h[ai]+u[ai]) + abs(h[ai]) + abs(u[ai])
    end
    F = fi + fa - fai
    f = F / nvars(ms)
end

#### SLOW VERSIONS
function update_var_bp_slow!(bp::BPFull, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    u = bp.u[∂i]
    h = copy(bp.h[∂i])
    full = cavity!(h, u, msg_mult, bp.efield[i])
    iszero(sum(full)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = full ./ sum(full)
    for (hnew,a) in zip(h, ∂i)
        hnew = hnew ./ sum(hnew)
        ε = max(ε, maximum(abs, hnew.-bp.h[a]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
    end
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_bp_slow!(bp::BPFull, a::Int, 
    ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    u = copy(bp.u[∂a])
    h = bp.h[∂a]
    full = cavity!(u, h, msg_conv, (1.0,0.0))
    for (unew,i) in zip(u, ∂a)
        unew = unew ./ sum(unew)    # should not be needed
        ε = max(ε, maximum(abs, unew.-bp.u[i]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end

function iteration_slow!(bp::BPFull; maxiter=10^3, tol=1e-12, damp=0.0, rein=0.0, 
        update_f! = update_factor_bp_slow!, update_v! = update_var_bp_slow!, 
        callback=(x...)->false)
    ε = 0.0
    for it = 1:maxiter
        ε = 0.0
        for a=1:size(bp.H,1)
            errf = update_f!(bp, a, damp=damp)
            errf == -1 && return -1,it
            ε = max(ε, errf)
        end
        for i=1:size(bp.H,2)
            errv = update_v!(bp, i, damp=damp, rein=rein)
            errv == -1 && return -1,it
            ε = max(ε, errv)
        end
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

function update_var_ms_slow!(bp::BPFull, i::Int, 
        h=copy(bp.h[nzrange(bp.H, i)]); damp=0.0, rein=0.0)
        ε = 0.0
    ∂i = nzrange(bp.H, i)
    u = bp.u[∂i]
    full = cavity!(h, u, msg_sum, bp.efield[i])
    ## 
    # some check against i.e. contradictions??
    ##
    bp.belief[i] = full .- maximum(full)
    for (hnew,a) in zip(h, ∂i)
        hnew = hnew .- maximum(hnew)
        ε = max(ε, maximum(abs, hnew.-bp.h[a]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
    end
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor_ms_slow!(bp::BPFull, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0)
    ε = 0.0
    u = bp.u[nonzeros(bp.X)[nzrange(bp.X, a)]]
    h = bp.h[∂a]
    full = cavity!(u, h, msg_maxconv, (0.0,-Inf))
    for (unew,i) in zip(u, ∂a)
        unew = unew .- maximum(unew)  
        ε = max(ε, maximum(abs, unew.-bp.u[i]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end


# Extremely fast, but... not working
function update_var_bp_new!(bp::BPFull, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    nzp = 0
    nzn = 0
    h = bp.efield[i]    # initialize
    for a in ∂i
        u = bp.u[a]
        all(iszero,u) && println("All zero u")
        if u[1] < 1e-300
            nzp += 1
        elseif u[2] < 1e-300
            nzn += 1
        else
            h = msg_mult(h, u)
        end
    end
    iszero(sum(h)) && return -1.0  # normaliz of belief is zero
    any(isnan,h) && @show h, bp.efield[i]
    bp.belief[i] = h ./ sum(h)
    for a in ∂i
        u = bp.u[a]
        if u[1] < 1e-300
            hnew = nzp==1 ? h : (0.0,1.0)
        elseif u[2] < 1e-300
            hnew = nzn==1 ? h : (1.0,0.0)
        else
            hnew = msg_mult(h, reverse(u)) # instead of dividing, mult times the swapped message
        end
        any(isnan, hnew ./ sum(hnew)) && @show hnew, h, u
        hnew = hnew ./ sum(hnew)
        ε = max(ε, abs(hnew[1]-bp.h[a][1]), abs(hnew[2]-bp.h[a][2]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
    end
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    any(isnan, bp.efield[i]) && @show bp.belief[i]
    ε
end

function update_factor_bp_new!(bp::BPFull, a::Int; damp=0.0)
    ε = 0.0
    ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]
    nunif = 0
    u = (1.0,0.0)
    for i in ∂a
        h = bp.h[i]
        if h[1]==h[2]
            nunif += 1
        else
            u = msg_conv(u,h)
        end
    end
    for i in ∂a
        h = bp.h[i]
        if h[1]==h[2]
            unew = nunif==1 ? u : (0.5,0.5)
        else
            # "invert" the convolution
            hinv = (h[1],-h[2])
            unew = msg_conv(u,hinv)
        end
        unew = unew ./ sum(unew) 
        any(isnan,unew) && println("Nan in update factor")
        ε = max(ε, abs(unew[1]-bp.u[i][1]), abs(unew[2]-bp.u[i][2]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
        any(isnan,bp.u[i]) && @show bp.u[i]
    end
    ε
end

include("slim_graphs.jl")
include("cavity.jl")
include("matrix_generator.jl")
include("bp.jl")

using OffsetArrays, Statistics, Printf, Plots

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
function BPFull(H::SparseMatrixCSC, efield = fill((0.5,0.5), size(H,2));
        neutralel = eltype(efield[1][1])==Int ? (0,0) : (0.5,0.5))
    n = size(H,2)
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
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


function update_var_bp!(bp::BPFull{F,M}, i::Int; damp=0.0, rein=0.0) where 
        {F, M<:Tuple{Real,Real}}
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
    bp.efield[i] = bp.efield[i] .* bp.belief[i].^rein
    ε
end

function update_factor_bp!(bp::BPFull{F,M}, a::Int, 
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0) where {F, M<:Tuple{Real,Real}}
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
        vars=rand(1:size(bp.H,2), size(bp.H,2)÷2), 
        factors=rand(1:size(bp.H,1), size(bp.H,1)÷2),
        callback=(it, ε, bp)->false)
    
    errv = zeros(nvars(bp)); errf = zeros(nfactors(bp)); ε = 0.0
    for it = 1:maxiter
        Threads.@threads for i ∈ vars
            errv[i] = update_v!(bp, i, damp=damp, rein=rein)
            errv[i] == -1 && return -1,it
            # ε = max(ε, errv)
            # ε += errv
        end
        ε = maximum(errv)
        Threads.@threads for a ∈ factors
            errf[a] = update_f!(bp, a, factor_neigs[a], damp=damp)
            errf[a] == -1 && return -1,it
            # ε = max(ε, errf)
            # ε += errf
        end
        ε = max(ε, maximum(errf))
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
        vars .= rand(1:size(bp.H,2), length(vars))
        factors .= rand(1:size(bp.H,1), length(factors))
    end
    ε, maxiter
end

function parity(bp::BPFull{Bool,T}) where T 
    b = map(b->b[1]-b[2], bp.belief)
    undec = sum(iszero,b) 
    undec >0 && error("Found $undec undecided variables")   
    x = b .< 0 
    parity(bp.H, x)
end
function parity(H::SparseMatrixCSC{Bool,Int}, x::AbstractVector)
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
    nunsat = parity(bp.H, x)
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

#### MAX-SUM 
msg_maxconv(h1::Tuple, h2::Tuple) = (max(h1[1]+h2[1],h1[2]+h2[2]), max(h1[1]+h2[2],h1[2]+h2[1])) 
msg_sum(u1::Tuple, u2::Tuple) = u1 .+ u2
function normalize_max(h::Tuple)
    m = maximum(h)
    return isinf(m) ? h : h .- maximum(h)
end
# we want Inf-Inf=0
myabsdiff(x,y) = x==y ? zero(x) : abs(x-y)
maxabsdiff(t1::Tuple, t2::Tuple, ε=0.0) = max(myabsdiff.(t1,t2)..., ε)
function damping(tnew::Tuple, told::Tuple, damp::Real)
    t = tnew.*(1-damp)
    if damp != 0
        t = t .+ told.*damp
    end
    t
end
function reinforcement_ms(oldfield::Tuple, belief::Tuple, rein::Real)
    newfield = oldfield
    if rein != 0
        newfield = newfield .+ rein .* belief 
    end
    newfield
end
 
function update_var_ms!(bp::BPFull{F,M}, i::Int; damp=0.0, rein=0.0) where 
    {F, M<:Tuple{Real,Real}}
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    b = bp.efield[i] 
    for a in ∂i
        hnew = bp.efield[i] 
        for c in ∂i
            c==a && continue
            hnew = msg_sum(hnew, bp.u[c])
        end
        any(isnan, normalize_max(hnew)) && @warn "NaN in h-message $a: $hnew"
        hnew = normalize_max(hnew)
        ε = maxabsdiff(hnew, bp.h[a], ε)
        # isnan(ε) && @show hnew, bp.h[a]
        # bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        # bp.h[a] = hnew
        bp.h[a] = damping(hnew, bp.h[a], damp)
        any(isnan, bp.h[a]) && @warn "NaN in message $a"
        b = msg_sum(b, bp.u[a]) 
    end
    any(isnan, normalize_max(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = normalize_max(b)
    # bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    bp.efield[i] = reinforcement_ms(bp.efield[i], bp.belief[i], rein)
    ε == -1 && println("Error in var $i")
    ε
end

function update_factor_ms!(bp::BPFull{F,M}, a::Int, 
    ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; damp=0.0) where {F, M<:Tuple{Real,Real}}
    ε = 0.0
    for i in ∂a
        unew = (0.0,-Inf)
        for j in ∂a
            j==i && continue
            unew = msg_maxconv(unew, bp.h[j])
        end
        any(isnan, normalize_max(unew)) && @warn "NaN in u-message $unew"
        unew = normalize_max(unew)    
        ε = maxabsdiff(unew, bp.u[i], ε)
        # bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
        # bp.u[i] = unew
        bp.u[i] = damping(unew, bp.u[i], damp)
        any(isnan, bp.u[i]) && @warn "NaN in message $i. unew=$unew. Nneigs=$(length(∂a))"
    end
    ε == -1 && println("Error in factor $a")
    ε
end



#### DECIMATION

# try Tmax times to reach zero unsat with decimation
# returns nunsat, ovl, dist
function decimate!(bp::BPFull, efield, indep, s, B; Tmax=1, 
        fair_decimation=false, verbose=true,
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.H,1)],
        kw...)
    freevars = falses(nvars(bp)); freevars[indep] .= true
    dist = zeros(Tmax)
    for t in 1:Tmax
        ε, nunsat, ovl, d, iters = decimate1!(bp, efield, freevars, s; 
            fair_decimation = fair_decimation, factor_neigs = factor_neigs, kw...)
        x = argmax.(bp.belief) .== 2
        σ = fix_indep!(x, B, indep)   
        dist[t] = distortion(σ,s)
        verbose && print("Trial $t of $Tmax: ")
        ε == -1 && print("contradiction found. ")
        verbose && println(nunsat, " unsat. Dist = ", round(dist[t],digits=3))
        # nunsat == 0 && return nunsat, ovl, dist
        freevars .= false; freevars[indep] .= true
    end
    return minimum(dist)
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
    # println("Avg distortion after 1st BP round: ", avg_dist(bp,s))
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


function beliefs_as_fields(belief)
    [Int((b[2]-b[1])/2) for b in belief]
end

function beliefs_hist(belief)
    n = length(belief)
    h = fill(0, -n:n)
    for i in 1:n
        b = belief[i]
        bb = Int((b[2]-b[1])/2)
        h[bb] += 1
    end
    h ./ sum(h)
end





### VANISHING FIELDS
# Msg{T,M} = Tuple{Tuple{T,T}, Tuple{M,M}}

# function msg_maxconv(h1::Msg{T,M}, h2::Msg{T,M}) where {T,M}
#     u = msg_maxconv(h1[1], h2[1])
#     utilde_plus = (u[1]==h1[1][1]+h2[1][1])*h1[2][1]*h2[2][1] +
#                   (u[1]==h1[1][1]+h2[1][2])*h1[2][2]*h2[2][2]
#     utilde_minus = (u[2]==h1[1][2]+h2[1][2])*h1[2][2]*h2[2][1] +
#                    (u[2]==h1[1][2]+h2[1][1])*h1[2][2]*h2[2][1]
#     (u, (utilde_plus,utilde_minus))::Msg{T,M}
# end

# function msg_sum(u1::Msg{T,M}, u2::Msg{T,M}) where {T,M}
#     h = u1[1] .+ u2[1]
#     htilde = u1[2] .* u2[2]
#     (h, htilde)::Msg{T,M}
# end

# function normalize_max(h::Msg{T,M}) where {T,M}
#     hnew = normalize_max(h[1])
#     hnewtilde = normalize_prob(h[2])
#     (hnew, hnewtilde)::Msg{T,M}
# end

# function ms_full_vanishing(H::SparseMatrixCSC, 
#     efield::Vector{Tuple{Tuple{T,T},Tuple{M,M}}}) where {T,M}
#     n = size(H,2)
#     X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
#     neutralel = ((zero(T),zero(T)), (zero(M), zero(M)))
#     h = fill(neutralel,nnz(H))
#     u = fill(neutralel,nnz(H))
#     belief = fill(neutralel,n)
#     BPFull(H, X, h, u, copy(efield), belief)
# end



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

# PLOTTING
function plot_rdb(; f30=true, f3=false) 
    DD = 0.0001:0.000001:0.5
    RR = 1 .+ DD.*log2.(DD) + (1 .- DD).*log2.(1 .- DD)
    pl = Plots.plot(RR, DD, label="Information bound")
    Plots.plot!(pl, RR, 0.5*(1 .- RR), label="Naive compression")
    if f30
        # RS curve for f3=0 : variables have degree 1 or 2
        R_f30 = 0.01:0.01:0.99
        D_f30 = [0.4546474073128681,0.43565049932361133,0.42092765046912317,0.40839439222208573,0.3972457420215144,0.3870702443196218,0.3776242651788484,0.36874955806591975,0.3603365352601175,0.3523056385673838,0.34459697668198624,0.33716414626734653,0.3299703418625086,0.3229857971761951,0.31618604029437997,0.30955066735206993,0.30306245811343413,0.2967067238049428,0.2904708168067388,0.2843437556958753,0.2783159341345032,0.27237889177802405,0.26652513178257226,0.2607479738209666,0.2550414345013817,0.24940012917780852,0.2438191906350483,0.2382942012118009,0.2328211357178961,0.22739631308980668,0.22201635516998064,0.21667815133009227,0.211378827914124,0.20688352899753953,0.20278317659003975,0.19868817793711935,0.1945975994415784,0.19051056552023182,0.18642625421128822,0.18234389327612,0.17826275673045133,0.17418216174918932,0.17010146589683228,0.16602006464188257,0.16193738911916555,0.15785290410864755,0.1537661062033548,0.14967652214243293,0.1455837072883871,0.14148724423013948,0.137998366107847,0.13448740638286893,0.13095421069018287,0.12739865696194796,0.12382065470933767,0.12022014422138577,0.11659709568295457,0.1129515082152911,0.10928340884389087,0.10559285139949881,0.10231362606715677,0.0989986151678956,0.09564794614910743,0.0922617930523803,0.0888403753926773,0.08538395674018595,0.0820151560863665,0.0788488704133426,0.07563678673548502,0.07237911796887381,0.06907615625262697,0.06591319349557895,0.06283584277258297,0.05970319879491964,0.056515457092013055,0.053562785457060835,0.05054512262289507,0.04752198628196386,0.04463641809759822,0.04167585667427448,0.0388827836051418,0.03604608454052277,0.033304960430999175,0.03060809220840449,0.027952952887639493,0.025359146185103376,0.022832861932172044,0.020370784947590925,0.017979055383501308,0.015666129177374533,0.013449978144655372,0.0113201258591068,0.009302254807509736,0.007397829094439923,0.0056264302423451595,0.004010648891537405,0.002577264143732827,0.0013672330158682189,0.0004509986577261871]
        plot!(pl, R_f30, D_f30, lc=:gray, ls=:dash, 
            label="RS K=fₖδ(k) + (1-fₖ)δ(k+1) Λ=δ₂")
    end
    if f3
        # RS curve for degree-3 factors and varying portion of degree-3 variables
        R_f3 = [0.33333333333333337,0.33000000000000007,0.32666666666666666,0.32333333333333336,0.31999999999999995,0.31666666666666676,0.31333333333333335,0.31000000000000005,0.30666666666666664,0.30333333333333334,0.29999999999999993,0.29666666666666675,0.29333333333333333,0.29000000000000004,0.2866666666666666,0.2833333333333333,0.2799999999999999,0.2766666666666667,0.2733333333333333,0.2699999999999999,0.2666666666666666,0.2633333333333333,0.2599999999999999,0.2566666666666667,0.2533333333333333,0.25,0.2466666666666667,0.2433333333333333,0.23999999999999988,0.2366666666666667,0.2333333333333334,0.2300000000000001,0.22666666666666668,0.22333333333333327,0.22000000000000008,0.21666666666666679,0.21333333333333326,0.20999999999999996,0.20666666666666667,0.20333333333333348,0.19999999999999984,0.19666666666666666,0.19333333333333336,0.18999999999999995,0.18666666666666654,0.18333333333333324,0.18000000000000005,0.17666666666666675,0.17333333333333334,0.16999999999999993,0.16666666666666663,0.16333333333333344,0.16000000000000003,0.15666666666666662,0.15333333333333332,0.15000000000000002,0.1466666666666666,0.1433333333333332,0.14,0.13666666666666671,0.13333333333333341,0.1299999999999999,0.1266666666666666,0.1233333333333334,0.1200000000000001,0.11666666666666659,0.11333333333333329,0.10999999999999999,0.1066666666666668,0.10333333333333339,0.10000000000000009,0.09666666666666668,0.09333333333333327,0.08999999999999997,0.08666666666666678,0.08333333333333337,0.07999999999999996,0.07666666666666666,0.07333333333333336,0.06999999999999995,0.06666666666666654,0.06333333333333335,0.05999999999999994,0.05666666666666664,0.053333333333333344,0.050000000000000155,0.04666666666666652,0.043333333333333335,0.040000000000000036,0.036666666666666736,0.033333333333333215,0.029999999999999916,0.026666666666666727,0.023333333333333428,0.020000000000000018,0.01666666666666672,0.013333333333333308,0.009999999999999898,0.00666666666666671,0.0033333333333334103,0.0]
        D_f3 = [0.20962056095628745,0.21045655312219785,0.21129398045076497,0.21213285191160997,0.21297317651789593,0.213814963331392,0.21465822146743357,0.21550296009981335,0.21634918846557782,0.2171969158697719,0.21804615169011754,0.2188969053816388,0.21974918648125297,0.22060300461230448,0.22145836948909192,0.22231529092135494,0.22317377881874867,0.22403384319531316,0.22489549417392346,0.22575874199075563,0.2266235969997512,0.22749006967709307,0.22835817062570685,0.22922791057977476,0.23009930040929016,0.23097235112463632,0.23184707388120812,0.23272347998408488,0.2336015808927387,0.23448138822581466,0.2353629137659588,0.23624616946472055,0.23713116744752138,0.23801792001870203,0.23890643966665281,0.2397967390690277,0.2406888310980555,0.24158272882594994,0.24247844553042286,0.2433759947003037,0.24427539004128584,0.24517664548178786,0.2460797751789402,0.24698479352472213,0.2478917151522217,0.248800554942059,0.2497113280289589,0.2506240498084894,0.2515387359439647,0.2524554023735355,0.25337406531746137,0.25429474128557106,0.25521744708493777,0.2561421998277597,0.2570690169394544,0.25799791616700474,0.2589289155875188,0.2598620336170625,0.26079728901973587,0.2617347009170402,0.26267428879751364,0.26361607252667024,0.2645600723572469,0.26550630893977045,0.2664548033334655,0.26740557701750556,0.26835865190263664,0.2693140503431725,0.2702717951493969,0.271231909600374,0.2721944174571924,0.27315934297666117,0.27412671092547497,0.2750965465948728,0.2760688758158149,0.2770437249746786,0.2780211210295341,0.27900109152699526,0.2799836646196798,0.28096886908431673,0.28195673434051494,0.28294729047022854,0.283940568237963,0.2849365991117401,0.2859354152848619,0.28693704969851397,0.28794153606524925,0.2889489088934002,0.28995920351242044,0.290972456099293,0.2919887037059378,0.2930079842877797,0.29403033673345685,0.2950558008957639,0.2960844176238681,0.29711622879689686,0.298151277358935,0.2991896073555008,0.30023126397162553,0.30127629357155217,0.30232474374018364]
        plot!(pl, R_f3, D_f3, label="RS K=δ₃ Λ=(1-f₃)δ₂+f₃*δ₃", lc=:magenta)
    end
    xlabel!(pl, "R"); ylabel!(pl, "D")
    pl
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

# set the dependent variables in x according to the indep ones and 
#  a basis B
function fix_indep!(x, B, indep)
    x .= B * x[indep] .% 2
end
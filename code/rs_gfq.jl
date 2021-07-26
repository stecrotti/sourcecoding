include("bp_full_gfq.jl")
using StaticArrays, ProgressMeter, StatsBase

hamming(x,y) = count_ones((x-1) ⊻ (y-1))

function iter_var_ms(us, s::Int, Q)
    h = - hamming.(SVector{Q}(1:Q), s)
    for u in us
        h = msg_sum(h, u)
    end
    h
    # all(isinf, h) && @show h, us, s
    h
end

function iter_factor_ms(hs, Hs, H0, gfmult, gfdiv, Q; uaux = -Inf*ones(MVector{Q,Float64}))
    u_tilde = neutral_prob_ms(Q)
    for (h,H) in zip(hs,Hs)
        h_tilde = h[SVector{Q}(gfdiv[:,H])]
        uaux .= -Inf
        u_tilde = msg_maxconv_gfq!(uaux, u_tilde, h_tilde)
    end
    u = u_tilde[SVector{Q}(gfmult[:,H0])]
    # all(isinf, u) && @show u_tilde, u, hs, Hs, H0, uaux
    u
end

function update_var(popP, Q::Int, wΛ)
    d = sample(wΛ)
    s = rand(1:Q)
    idx = rand(1:N, SVector{d-1})
    us = @view popP[idx]
    h = iter_var_ms(us, s, Q)
    any(isnan, h) && error("nan in update var. non-norm h=$h")
    h .- maximum(h)
end
function update_factor(popQ, Q::Int, wP, gfmult, gfdiv; 
        uaux=-Inf*ones(MVector{Q,Float64}))
    k = sample(wP)
    Hs = rand(2:Q, SVector{k-1})
    H0 = rand(2:Q)
    idx = rand(1:N, SVector{k-1})
    hs = @view popQ[idx]
    u = iter_factor_ms(hs, Hs, H0, gfmult, gfdiv, Q)
    any(isnan, u) && error("nan in update factor. non-norm u=$u")
    u .- maximum(u)
end


function RS_gfq!(popP, popQ, Λ, Pk, Q, gfmult, gfdiv; maxiter=10^2, tol=1e-5, 
        toliter=(length(popP)/Q)^(-1), showprogress=true, cb=(x...)->nothing) 
    Λ_red = Λ.*eachindex(Λ)
    Pk_red = Pk.*eachindex(Pk)
    wΛ = StatsBase.weights(Λ_red)
    wP = StatsBase.weights(Pk_red)
    N = length(popP)
    avgP = mean(popP)
    uauxs = fill(-Inf*ones(MVector{Q,Float64}),Threads.nthreads())
    err = -Inf
    prog = ProgressMeter.Progress(maxiter, showprogress ? 1 : Inf)
    for it in 1:maxiter
        # update Q(h)
        Threads.@threads for i in rand(1:N, N÷2)
            popQ[i] = update_var(popP, Q, wΛ)
        end
        # update P(u)
        Threads.@threads for i in rand(1:N, N÷2)
            popP[i] = update_factor(popQ, Q, wP, gfmult, gfdiv, 
                uaux=uauxs[Threads.threadid()])
        end
        avgP_old = avgP
        avgP = mean(popP)
        err = maximum(abs.(avgP_old.-avgP))
        ProgressMeter.next!(prog, showvalues=[(:it, it), (:err, "$err/$toliter")])
        cb(it, err)
        if err < toliter
            break
        end
    end
    err
end

function freenrj_factor(wP, popQ, gfmult, gfdiv, Q;
        uaux = -Inf*ones(MVector{Q,Float64}))
    N = length(popQ)
    k = sample(wP)
    Hs = rand(2:Q, SVector{k})
    idx = rand(1:N, SVector{k})
    hs = @view popQ[idx]
    u = iter_factor_ms(hs, Hs, 1, gfmult, gfdiv, Q) / log2(Q)
    any(isnan, u) && error("nan in update factor. non-norm u=$u")
    Fa = -u[1] 
end
function freenrj_edge(popP, popQ, Q)
    h = rand(popQ)
    u = rand(popP)
    Fia = -maximum(msg_sum(h, u)) / log2(Q)
end
function freenrj_var(wΛ, popP, Q)
    N = length(popP)
    d = sample(wΛ)
    s = rand(1:Q)
    idx = rand(1:N, SVector{d})
    us = @view popP[idx]
    h = iter_var_ms(us, s, Q) ./ log2(Q)
    any(isnan, h) && error("nan in update var. non-norm h=$h")
    Fi = -maximum(h)
end

function freenrj(Λ, Pk, Q, popP, popQ, gfmult, gfdiv; nsamples=10^3, 
        showprogress=true, cb=(x...)->nothing, F = zeros(nsamples))
    Λ_red = Λ.*eachindex(Λ)
    Pk_red = Pk.*eachindex(Pk)
    mΛ = sum(Λ_red); mP = sum(Pk_red)
    α = mΛ/mP
    wΛ = StatsBase.weights(Λ)
    wP = StatsBase.weights(Pk)
    uaux = fill(-Inf*ones(MVector{Q,Float64}),Threads.nthreads())
    prog = ProgressMeter.Progress(nsamples, showprogress ? 1 : Inf)
    Threads.@threads for t = 1:nsamples
        Fa = freenrj_factor(wP, popQ, gfmult, gfdiv, Q;
            uaux=uaux[Threads.threadid()])
        Fi = freenrj_var(wΛ, popP, Q)
        Fia = freenrj_edge(popP, popQ, Q)
        F[t] = Fi + α*Fa - mΛ*Fia
        cb(t, F, Fa, Fi, Fia)
        ProgressMeter.next!(prog)
    end
    mean(F)
end
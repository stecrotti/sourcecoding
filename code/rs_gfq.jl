include("bp_full_gfq.jl")
using StaticArrays, ProgressMeter, StatsBase

hamming(x,y) = count_ones((x-1) ⊻ (y-1))

function iter_var_ms(us, s::Int, Q)
    h = - hamming.(SVector{Q}(1:Q), s)
    for u in us
        h = msg_sum(h, u)
    end
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
end

function RS_gfq!(popP, popQ, Λ, Pk, Q, gfmult, gfdiv; maxiter=10^2, tol=1e-5, 
        toliter=1/sqrt(length(popP))*Q, showprogress=true) 
    ks = [k for k in eachindex(Pk) if Pk[k] > tol]
    ds = [d for d in eachindex(Λ) if Λ[d] > tol]
    @assert sum(Pk[ks]) ≈ 1 && sum(Λ[ds]) ≈ 1    
    Λ_red = [d*Λ[d] for d in eachindex(Λ)]; Λ_red ./= sum(Λ_red)
    P_red = [d*Pk[d] for d in eachindex(Pk)]; P_red ./= sum(P_red)

    N = length(popP)
    avgP = mean(popP)
    uaux = -Inf*ones(MVector{Q,Float64})
    err = -Inf
    prog = ProgressMeter.Progress(maxiter, showprogress ? 1 : Inf)
    for it in 1:maxiter
        # update Q
        # for i in 1:N
        for i in rand(1:N, N÷2)
            d = sample(eachindex(Λ_red), StatsBase.weights(Λ_red))
            s = rand(1:Q)
            idx = sample(1:N, d)
            us = @view popP[idx]
            h = iter_var_ms(us, s, Q)
            popQ[i] = h .- maximum(h)
        end
        # update P(u)
        for i in rand(1:N, N÷2)
            k = sample(eachindex(P_red), StatsBase.weights(P_red))
            Hs = rand(2:Q,k)
            H0 = rand(2:Q)
            idx = sample(1:N, k)
            hs = @view popQ[idx]
            u = iter_factor_ms(hs, Hs, H0, gfmult, gfdiv, Q; uaux=uaux)
            popP[i] = u .- maximum(u)
        end
        avgP_old = avgP
        avgP = mean(popP)
        err = maximum(abs.(avgP_old.-avgP))
        ProgressMeter.next!(prog, showvalues=[(:it, it), (:err, "$err/$toliter")])
        if err < toliter
            break
        end
    end
    err
end

function freenrj_factor(Pk, popQ, gfmult, gfdiv, Q)
    N=length(popQ)
    k = sample(eachindex(Pk), weights(Pk))
    Hs = rand(2:Q,k)
    idx = sample(1:N, k)
    hs = @view popQ[idx]
    u = iter_factor_ms(hs, Hs, 1, gfmult, gfdiv, Q)
    Fa = -u[1]
end
function freenrj_edge(popP, popQ)
    N=length(popP)
    h=popQ[rand(1:N)]
    u=popP[rand(1:N)]
    Fia = -maximum(msg_sum(h, u))
end
function freenrj_var(Λ, popP, Q)
    N = length(popP)
    d = sample(eachindex(Λ), weights(Λ))
    s = rand(1:Q)
    idx = sample(1:N, d)
    us = @view popP[idx]
    Fi = -maximum(iter_var_ms(us, s, Q))
end

function freenrj(Λ, Pk, Q, popP, popQ, gfmult, gfdiv; samples=10^3, showprogress=true)
    mK = sum(k*Pk[k] for k=eachindex(Pk))
    mΛ = sum(d*Λ[d] for d=eachindex(Λ))
    α = mΛ/mK
    Fa=Fi=Fia=0 
    prog = ProgressMeter.Progress(samples, showprogress ? 1 : Inf)
    for t=1:samples
        Fa += freenrj_factor(Pk, popQ, gfmult, gfdiv, Q)
        Fi += freenrj_var(Λ, popP, Q)
        Fia += freenrj_edge(popP, popQ)
        F = (Fi + α*Fa - mΛ*Fia)/t/log2(Q)
        ProgressMeter.next!(prog, showvalues=[(:it, t), (:F, F)])
    end
    # Fi *= 1/samples/log2(Q)
    # Fa *= α/samples/log2(Q)
    # Fia *= -mΛ/samples/log2(Q)
    # F = Fi + Fa + Fia
    # F, Fi, Fa, Fia
    F = (Fi + α*Fa - mΛ*Fia)/samples/log2(Q)
end
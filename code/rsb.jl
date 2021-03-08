function Cached_Factor_Iterator(J)
    a = fill(0.0, -J-1:J+1)
    function iter_factor!(p, Q)
        a[0:J] .= 1
        a[-J:-1] .= 0
        mass = 1.0
        @assert extrema(eachindex(p)) == (-J,J)
        @inbounds for q ∈ Q
            @assert extrema(eachindex(q)) == (-J,J)
            Σp, Σm = 0.0, 0.0
            for h=J:-1:1
                ap, am = a[h], a[-h]
                Σp += q[h]; Σm += q[-h]
                a[+h] = ap*Σp + am*Σm
                a[-h] = am*Σp + ap*Σm
            end
            a[0] *= Σp + Σm
            mass *= Σp + Σm + q[0]
        end
        @inbounds for u = 1:J
            p[+u] = a[+u] - a[+u+1]
            p[-u] = a[-u] - a[-u-1]
        end
        p[0] = mass-a[0]
        p ./= sum(p)
    end
end

function Cached_Var_Iterator(J, maxd, y)
    buf = (zeros(maxd*2J+1), zeros(maxd*2J+1))
    expy = OffsetArray([exp(y*abs(h)) for h=-maxd*J-1:maxd*J+1], -maxd*J-1:maxd*J+1)
    expmy = OffsetArray([exp(-y*abs(h)) for h=-J:J], -J:J)
    function iter_var!(q, P, s)
        @assert length(P) ≤ maxd
        first = last = s
        q1, q2 = buf[1], buf[2]
        q1[1] = 1.0
        @inbounds for (i,p) in enumerate(P)
            #q2[1:last-first+1+2J] .= 0
            q2 .= 0
            for u2 = eachindex(p)
                paux = p[u2] * expmy[u2]
                for u1 = 1:last-first+1
                    q2[u1+u2+J] += q1[u1] * paux
                end
            end
            first -= J
            last += J
            q1, q2 = q2, q1
        end
        q .= 0
        @inbounds for (h,h1) in enumerate(first:last)
            q[clamp(h1,firstindex(q),lastindex(q))] += q1[h] * expy[h1]
        end
        #q ./= sum(q)
        q
    end
end

residual(x) = (p=OffsetVector((x .* eachindex(x))[1:end], 0:lastindex(x)-1); p./=sum(p))

function moments!(avp, popP)
    avp .= sum(popP, dims=2)[:,1] ./ size(popP,2)
end

function checkRS(popP)
    V=0.0
    for t=1:size(popP,2)
        m1 = sum(i*popP[i,t] for i in eachindex(popP[:,t]))
        m2 = sum(i^2*popP[i,t] for i in eachindex(popP[:,t]))
        V+= m2-m1^2
    end
    V /= size(popP,2)
end

function RSB(Λ, K; 
        J=10, 
        maxiter=100, 
        popsize=1000, 
        tol=1/(J*√popsize),
        popP = fill(1/(2J+1), -J:J, 1:popsize),
        popQ = fill(1/(2J+1), -J:J, 1:popsize), 
        y=0.0)
    J = lastindex(popP[:,1])
    mt = MersenneTwister()
    function Sampler(P, R)
        idx = fill(0,10)
        wP = StatsBase.weights(P)
        function s()
            d = sample(mt, eachindex(P), wP)
            resize!(idx, d)
            rand!(mt, idx, R)
        end
    end
    N = size(popP,2)
    sampled = Sampler(residual(Λ), 1:N)
    samplek = Sampler(residual(K), 1:N)
    iter_var! = Cached_Var_Iterator(J, lastindex(Λ), y)
    iter_factor! = Cached_Factor_Iterator(J)
    avp = fill(0.0, -J:J)
    avp1 = fill(0.0, -J:J)
    moments!(avp, popP); err = Inf
    p = Progress(maxiter)
    for t = 1:maxiter
        for i = 1:N
            P = eachcol(@view popP[:, sampled()])
            s = rand(mt, (-1,1))
            iter_var!((@view popQ[:,i]), P, s)
        end
        for i = 1:N
            Q = eachcol(@view popQ[:, samplek()])
            iter_factor!((@view popP[:,i]), Q)
        end
        moments!(avp1, popP);
        err = maximum(abs.(avp .- avp1))
        if err < tol
            @show t
            break
        end
        ProgressMeter.next!(p; showvalues = [(:err,"$err/$tol")])
        avp, avp1 = avp1, avp
    end
    @show err
    popP, popQ
end

function Cached_Overlap_Var(J, maxd, y)
    F! = Cached_Var_Iterator(J, maxd, y)
    q = fill(0.0, -J*maxd-1:J*maxd+1)
    pau = fill(0.0, -J:J)
    function overlap(P, s)
        F!(q, P, s)
        o = 0.0
        z = 0.0
        for h ∈ eachindex(q)
            o += -abs(h)*q[h]
            z += q[h]
        end
        for pa ∈ P
            pau .= pa .* abs.(eachindex(pa))
            Pa = (pb === pa ? pau : pb for pb in P)
            F!(q, Pa, s)
            o += sum(q)
        end
        o/z, log(z)
    end
end

function Cached_Overlap_Factor(J)
    p = fill(0.0, -J:J)
    F! = Cached_Factor_Iterator(J)
    function overlap(Q, J, y)
        F!(p, Q)
        o = 0.0
        z = 1.0
        for f=1:J
            e = exp(-2y*f)
            o += p[-f]*2f*e
            z += p[-f]*(e-1)
        end
        o/z, log(max(z,0))
    end
end

function overlap_slow_edge(p, q, J, y=0.0)
    z = 0.0
    o = 0.0
    for h in -J:J, u in -J:J
        Fia = -abs(u+h)+abs(h)+abs(u)
        w = p[u]*q[h]*exp(-y*Fia)
        o += Fia*w
        z += w
    end
    o/z, log(z)
end

function overlap1RSB(Λ, K; 
        popP, 
        popQ, 
        samples=size(popP,2), 
        y=0.0)
    mt = MersenneTwister()
    function Sampler(P, R)
        idx = fill(0,10)
        wP = StatsBase.weights(P)
        function s()
            d = sample(mt, eachindex(P), wP)
            resize!(idx, d)
            rand!(mt, idx, R)
        end
    end
    N = size(popP,2)
    sampled = Sampler(Λ, 1:N)
    samplek = Sampler(K, 1:N)
    J = lastindex(popQ[:,1])
    mK = sum(k*K[k] for k=eachindex(K))
    mΛ = sum(d*Λ[d] for d=eachindex(Λ))
    α = mΛ/mK

    N = size(popP,2)

    O_factor = F_factor = 0.0
    O_var = F_var = 0.0
    O_edge = F_edge = 0.0

    cached_overlap_factor = Cached_Overlap_Factor(J)
    cached_overlap_var = Cached_Overlap_Var(J, lastindex(Λ), y)
    progress = Progress(samples)
    for t = 1:samples
        Q = eachcol(@view popQ[:, samplek()])
        o, f = cached_overlap_factor(Q, J, y)
        O_factor += o
        F_factor += f

        P = eachcol(@view popP[:, sampled()])
        s = rand(mt, (-1,1))
        o, f = cached_overlap_var(P, s)
        O_var += o
        F_var += f

        p = popP[:, rand(1:N)]
        q = popQ[:, rand(1:N)]
        o, f = overlap_slow_edge(p, q, J, y)
        O_edge += o
        F_edge += f
        O = -(α*O_factor + O_var - mΛ*O_edge)/t
        F = -1/y*(α*F_factor + F_var - mΛ*F_edge)/t
        ProgressMeter.next!(progress; showvalues = [(:F,F),(:O,O),(:D,(1-O)/2)])
    end
    O = -(α*O_factor + O_var - mΛ*O_edge)/samples
    F = -1/y*(α*F_factor + F_var - mΛ*F_edge)/samples
    C = -O - F
    O,F,C
end
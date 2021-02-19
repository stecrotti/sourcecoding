using SparseArrays, Random


mutable struct SurveyPropagation{F,M}
    H :: SparseMatrixCSC{F,Int}
    X :: SparseMatrixCSC{Int,Int}
    P :: Vector{M}
    Q :: Vector{M}
    survey :: Vector{M}
    efield :: Vector{Int}
    y :: Float64
    J :: Int
end

# computes in linear time and inplace dst[i] = 
# src[1] op … src[i-1] op src[i+1] op … src[end] 
# 1-base indexing required

function cavity!(dest, source, op, init)
    @assert length(dest) == length(source)
    isempty(source) && return init
    if length(source) == 1
        dest[begin] = init 
        return source[begin]
    end
    if length(source) == 2
        dest[begin] = op(source[end], init)
        dest[end] = op(source[begin], init)
        return op(dest[begin], source[begin])
    end
    accumulate!(op, dest, source)
    full = op(dest[end], init)
    right = init
    for i=length(source):-1:2
        dest[i] = op(dest[i-1], right);
        right = op(source[i], right);
    end
    dest[1] = right
    full
end

function survey_propagation(H; field, init, y)
    H = sparse(H)
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    P = [copy(init) for i=1:length(H.nzval)]
    Q = [copy(init) for i=1:length(H.nzval)]
    survey = [copy(init) for i=1:size(H,2)]
    SurveyPropagation(H, X, P, Q, survey, field, y, lastindex(init))
end


function update_factor!(sp::SurveyPropagation, b; damp = 0.0)
    ε = 0.0
    J = sp.J
    ∂b = nonzeros(sp.X)[nzrange(sp.X, b)]
    for i ∈ ∂b
        a = fill(0.0, -J-1:J+1)
        a[0:J] .= 1
        for j ∈ ∂b
            i == j && continue
            q = sp.Q[j]
            Σp, Σm = 0.0, 0.0
            for h=J:-1:1
                ap, am = a[h], a[-h]
                Σp += q[h]; Σm += q[-h]
                a[+h] = ap*Σp + am*Σm
                a[-h] = am*Σp + ap*Σm
            end
            a[0] *= 1 - q[0]
        end
        p = fill(0.0, -J:J)
        for u = 1:J
            p[+u] = a[+u]-a[u+1]
            p[-u] = (a[-u]-a[-u-1])
            #*exp(2sp.y*u)
        end
        p[0] = 1 - a[0]
        p ./= sum(p)
        ε = max(ε, maximum(abs, sp.P[i] - p))
        for u in eachindex(p)
            sp.P[i][u] = damp * sp.P[i][u] + (1-damp) * p[u]
        end
    end
    ε
end


function ⊛(p1, p2)
    q = fill(0.0,firstindex(p1)+firstindex(p2):lastindex(p1)+lastindex(p2))
    for f1 in eachindex(p1), f2 in eachindex(p2)
        q[f1+f2] += p1[f1]*p2[f2]
    end
    q
end


function update_var_slow!(sp::SurveyPropagation, i; damp = 0.0)
    ε = 0.0
    J = sp.J
    s = sp.efield[i]
    y = sp.y
    ∂i = nzrange(sp.H, i)
    q = fill(1.0, s:s)
    for a in ∂i
        p = sp.P[a]
        q = q ⊛ (p .* exp.(y .* abs.(eachindex(p))))
    end
    si = sp.survey[i]
    si .= 0.0
    for h in eachindex(q)
        si[clamp(h,-J,J)] += q[h] * exp(-y*abs(h))
    end
    si ./= sum(si)
    for a in ∂i
        q = fill(1.0, s:s)
        for b ∈ ∂i
            b == a && continue
            p = sp.P[b]
            q = q ⊛ (p .* exp.(y .* abs.(eachindex(p))))
        end
        qnew = fill(0.0, -J:J)
        for h in eachindex(q)
            qnew[clamp(h,-J,J)] += q[h] * exp(-y * abs(h))
        end

        qnew ./= sum(qnew)
        ε = max(ε, maximum(abs, qnew - sp.Q[a]))
        sp.Q[a] .= qnew
        @show qnew
    end
    ε
end


function update_var!(sp::SurveyPropagation, i; damp = 0.0, rein=0.0)
    ε = 0.0
    J = sp.J
    s = sp.efield[i]
    ∂i = nzrange(sp.H, i)
    P = [p .* exp.(sp.y * abs.(eachindex(p))) for p in sp.P[∂i]]
    init = fill(1.0, s:s)
    Q = [fill(1.0, 0:0) for a ∈ 1:length(∂i)]
    qfull = cavity!(Q, P, ⊛, init)
    for h in eachindex(qfull)
         sp.survey[i][clamp(h,-J,J)] += qfull[h] * exp(-sp.y*abs(h))
    end
    sp.survey[i] ./= sum(sp.survey[i])

    qnew = fill(0.0, -J:J)
    for (qcav,q) ∈ zip(Q, sp.Q[∂i])
        qnew .= 0.0
        for h in eachindex(qcav)
            qnew[clamp(h,-J,J)] += qcav[h] * exp(-sp.y*abs(h))
        end
        qnew .*= sp.survey[i].^rein
        qnew ./= sum(qnew)
        ε = max(ε, maximum(abs, qnew - q))
        q .= damp .* q .+ (1-damp).* qnew
    end
    sp.survey[i][0] *= 1 - rein
    sp.survey[i][-J:-1] .^= 1 + rein
    sp.survey[i][1:J] .^= 1 + rein
    ε
end

function update_var_zeroT!(sp::SurveyPropagation, i; damp = 0.0, rein = 0.0)
    ε = 0.0
    J = sp.J
    s = sp.efield[i]
    ∂i = nzrange(sp.H, i)
    # Functions for max-sum convolution
    P = [abs.(OffsetArray(-J:J, -J:J)) - p for p in sp.P[∂i]]
    # Init: "log(delta)" centered at s
    init = fill(0.0, s:s)
    Q = [fill(-Inf, 0:0) for a ∈ 1:length(∂i)]
    qfull = cavity!(Q, P, msc, init)
    si = sp.survey[i]
    si .= -Inf
    for h in eachindex(qfull)
         si[clamp(h,-J,J)] = max(si[clamp(h,-J,J)], abs(h) - qfull[h])
    end
    si .-= maximum(si)

    qnew = fill(-Inf, -J:J)
    for (qcav,q) ∈ zip(Q, sp.Q[∂i])
        qnew .= -Inf
        for h in eachindex(qcav)
            qnew[clamp(h,-J,J)] = max(qnew[clamp(h,-J,J)], abs(h) - qcav[h])
        end
        qnew .+= si.*rein
        qnew .-= maximum(qnew)
        ε = max(ε, maximum(abs, qnew - q))
        q .= damp .* q .+ (1-damp).* qnew
        # @show q
    end
    si[0] *= 1 - rein
    si[-J:-1] .*= 1 + rein
    si[1:J] .*= 1 + rein
    ε
end

function update_var_zeroT_slow!(sp::SurveyPropagation, i; damp = 0.0, rein = 0.0)
    ε = 0.0
    J = sp.J
    s = sp.efield[i]
    ∂i = nzrange(sp.H, i)
    # compute survey
    P = sp.P[∂i]
    si = sp.survey[i]
    si .= -Inf
    for us in Iterators.product(fill(-J:J, length(P))...)
        h = sum(us) + s
        si[clamp(h,-J,J)] = max(si[clamp(h,-J,J)], 
            abs(h) - sum(abs.(u)-p[u] for (p,u) in zip(P,us)))
    end
    si .-=  maximum(si)
    # compute cavity fields
    for a in ∂i
        P = [sp.P[∂i][b] for b in eachindex(sp.P[∂i]) if b!=a]
        q = fill(-Inf, -J:J)
        for us in Iterators.product(fill(-J:J, length(P))...)
            h = sum(us) + s
            q[clamp(h,-J,J)] = max(q[clamp(h,-J,J)], 
                abs(h) - sum(abs.(u)-p[u] for (p,u) in zip(P,us)))
        end
        q .-= maximum(q)
        q .+= si.*rein
        ε = max(ε, maximum(abs, q - sp.Q[a]))
        sp.Q[a] .= damp*sp.Q[a] + (1-damp)*q
        # @show sp.Q[a]
    end
    si[0] *= 1 - rein
    si[-J:-1] .*= 1 + rein
    si[1:J] .*= 1 + rein
    ε
end  

function update_factor_zeroT!(sp::SurveyPropagation, b; damp = 0.0)
    ε = 0.0
    J = sp.J
    ∂b = nonzeros(sp.X)[nzrange(sp.X, b)]
    for i ∈ ∂b
        p = fill(0.0, -J:J)
        qstar = 0.0
        qmax = -Inf
        b = fill(-Inf,J:J); b[1:J] .= 0.0
        a = fill(-Inf,J:J)
        for j ∈ ∂b
            j == i && continue
            q = sp.Q[j]
            # recursion for u=0
            qstar += maximum(q)
            qmax = max(qmax, q[0]-qstar)
            # recursion for |u|>0
            for u in 1:J
                anew = fill(-Inf,J:J)
                bnew = fill(-Inf,J:J)
                # recursion for a uses b from the previous round
                for h in J:-1:u+1
                    anew[u] = max(anew[u], q[h]+a[u], q[-h]+a[-u])
                    anew[-u] = max(anew[-u], q[h]+a[-u], q[-h]+a[u])
                end 
                anew[u] = max(anew[u], q[u]+b[u], q[-u]+b[-u])
                anew[-u] = max(anew[-u], q[u]+b[-u], q[-u]+b[u])
                # recursion for b
                for h in J:-1:u
                    bnew[u] = max(bnew[u], q[h]+b[u], q[-h]+b[-u])
                    bnew[-u] = max(bnew[-u], q[h]+b[-u], q[-h]+b[u])
                end
                a .= anew
                b .= bnew
            end
        end
        p[0] = -sum(qstar) - qmax
        p[1:J] = a[1:J]
        p[-J:-1] = a[-J:-1]
        p .-= maximum(p)
        ε = max(ε, maximum(abs, sp.P[i] - p))
        for u in eachindex(p)
            sp.P[i][u] = damp * sp.P[i][u] + (1-damp) * p[u]
        end
    end
    ε
end


function iteration!(sp::SurveyPropagation; maxiter = 1000, tol=1e-3, γ=0.0, damp=0.0, rein=0.0, callback=(x...)->false)
    errf = fill(0.0, size(H,1))
    errv = fill(0.0, size(H,2))
    @inbounds for t = 1:maxiter
        Threads.@threads for a=1:size(H,1)
            errf[a] = update_factor!(sp, a, damp=damp)
        end
        Threads.@threads for i=1:size(H,2)
            errv[i] = update_var!(sp, i, damp=damp, rein=rein)
        end
        ε = max(maximum(errf), maximum(errv))
        callback(t, ε, sp) && break
        ε < tol && break
    end
    
end

# Max-Sum convolution
function msc(f1,f2)
    g = fill(0.0,firstindex(f1)+firstindex(f2):lastindex(f1)+lastindex(f2))
    for x1 in eachindex(f1), x2 in eachindex(f2)
        v = f1[x1]+f2[x2]
        if v > g[clamp(x1+x2,-J,J)]
            g[clamp(x1+x2,-J,J)] = v
        end
    end
    g
end
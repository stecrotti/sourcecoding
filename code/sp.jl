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


function update_factor!(fg::SurveyPropagation, b; damp = 0.0)
    ε = 0.0
    J = fg.J
    ∂b = nonzeros(fg.X)[nzrange(fg.X, b)]
    for i ∈ ∂b
        a = fill(0.0, -J-1:J+1)
        a[0:J] .= 1
        for j ∈ ∂b
            i == j && continue
            q = fg.Q[j]
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
            #*exp(2fg.y*u)
        end
        p[0] = 1 - a[0]
        p ./= sum(p)
        ε = max(ε, maximum(abs, fg.P[i] - p))
        for u in eachindex(p)
            fg.P[i][u] = damp * fg.P[i][u] + (1-damp) * p[u]
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


function update_var_slow!(fg::SurveyPropagation, i; damp = 0.0)
    ε = 0.0
    J = fg.J
    s = fg.efield[i]
    y = fg.y
    ∂i = nzrange(fg.H, i)
    q = fill(1.0, s:s)
    for a in ∂i
        p = sp.P[a]
        q = q ⊛ (p .* exp.(y .* abs.(eachindex(p))))
    end
    si = fg.survey[i]
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
        ε = max(ε, maximum(abs, qnew - fg.Q[a]))
        fg.Q[a] .= qnew
        @show qnew
    end
    ε
end




function update_var!(fg::SurveyPropagation, i; damp = 0.0, rein=0.0)
    ε = 0.0
    J = fg.J
    s = fg.efield[i]
    ∂i = nzrange(fg.H, i)
    P = [p .* exp.(fg.y * abs.(eachindex(p))) for p in fg.P[∂i]]
    init = fill(1.0, s:s)
    Q = [fill(1.0, 0:0) for a ∈ 1:length(∂i)]
    qfull = cavity!(Q, P, ⊛, init)
    for h in eachindex(qfull)
         fg.survey[i][clamp(h,-J,J)] += qfull[h] * exp(-fg.y*abs(h))
    end
    fg.survey[i] ./= sum(fg.survey[i])

    qnew = fill(0.0, -J:J)
    for (qcav,q) ∈ zip(Q, fg.Q[∂i])
        qnew .= 0.0
        for h in eachindex(qcav)
            qnew[clamp(h,-J,J)] += qcav[h] * exp(-fg.y*abs(h))
        end
        qnew .*= fg.survey[i].^rein
        qnew ./= sum(qnew)
        ε = max(ε, maximum(abs, qnew - q))
        q .= damp .* q .+ (1-damp).* qnew
    end
    fg.survey[i][0] *= 1 - rein
    fg.survey[i][-J:-1] .^= 1 + rein
    fg.survey[i][1:J] .^= 1 + rein
    ε
end

function iteration!(fg::SurveyPropagation; maxiter = 1000, tol=1e-3, γ=0.0, damp=0.0, rein=0.0, callback=(x...)->false)
    errf = fill(0.0, size(H,1))
    errv = fill(0.0, size(H,2))
    @inbounds for t = 1:maxiter
        Threads.@threads for a=1:size(H,1)
            errf[a] = update_factor!(fg, a, damp=damp)
        end
        Threads.@threads for i=1:size(H,2)
            errv[i] = update_var!(fg, i, damp=damp, rein=rein)
        end
        ε = max(maximum(errf), maximum(errv))
        callback(t, ε, fg) && break
        ε < tol && break
    end
end


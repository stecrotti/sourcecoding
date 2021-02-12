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

# computes in linear time and inplace dest[i] = 
# src[1] op … src[i-1] op src[i+1] op … src[end] 
# 1-base indexing required
function cavity!(dest, source, op, init)
    @assert length(dest) == length(source)
    isempty(source) && return init
    accumulate!(op, dest, source, init=init)
    full = dest[end]
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
            for f=J:-1:1
                ap, am = a[f], a[-f]
                Σp += q[f]; Σm += q[-f]
                a[+f] = ap*Σp + am*Σm
                a[-f] = am*Σp + ap*Σm
            end
            a[0] *= 1-q[0]
        end
        p = fill(0.0, -J:J)
        for f = 1:J
            p[+f] = (a[+f]-a[+f+1])
            p[-f] = (a[-f]-a[-f-1])*exp(-2fg.y*f)
        end
        p[0] = 1 - a[0]
        p ./= sum(p)
        ε = max(ε, maximum(abs, fg.P[i] - p))
        fg.P[i] .= damp .* fg.P[i] .+ (1-damp) .* p
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

function update_var!(fg::SurveyPropagation, i; damp = 0.0)
    ε = 0.0
    J = fg.J
    ∂i = nzrange(fg.H, i)
    P = push!([p .* exp.(-fg.y .* abs.(eachindex(p))) for p ∈ @view fg.P[∂i]],
              fill(1.0, fg.efield[i]:fg.efield[i]))
    init = fill(1.0, 0:0)
    Q = [fill(1.0, 0:0) for a ∈ 1:length(∂i)+1]
    qfull = cavity!(Q, P, ⊛, init)
    for f in eachindex(qfull)
        fg.survey[i][clamp(f,-J,J)] += qfull[f]
    end
    fg.survey[i] ./= sum(fg.survey[i])

    q = fill(0.0, -J:J)
    for (q1,qout) ∈ zip(Q, @view fg.Q[∂i])
        q1 .*= exp.(fg.y .* abs.(eachindex(q1)))
        q .= 0.0
        for f in eachindex(q1)
            q[clamp(f,-J,J)] += q1[f]
        end
        q ./= sum(q)
        ε = max(ε, maximum(abs, qout - q))
        qout .= damp .* q .+ (1-damp).* q
    end
    ε
end

function iteration!(fg::SurveyPropagation; maxiter = 1000, tol=1e-3, γ=0.0, damp=0.0, callback=(x...)->false)
    errf = fill(0.0, size(H,1))
    errv = fill(0.0, size(H,2))
    @inbounds for t = 1:maxiter
        Threads.@threads for a=size(H,1)
            errf[a] = update_factor!(fg, a, damp=damp)
        end
        Threads.@threads for i=size(H,2)
            errv[i] = update_var!(fg, i, damp=damp)
        end
        ε = max(maximum(errf), maximum(errv))
        ε < tol && break
        callback(t, ε, fg) && break
    end
end


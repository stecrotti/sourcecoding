include("bp_full.jl")
using StaticArrays, OffsetArrays, DelimitedFiles

msg_mult(u1::SVector, u2::SVector) = u1 .* u2
normalize_prob(u::SVector) = u ./ sum(u)
function neutral_prob_bp(Q::Integer)
    v = zero(SVector{Q})
    v = setindex(v, 1.0, 1)
end
getQ(bp::BPFull{F,SVector{Q,T}}) where {F,Q,T} = length(bp.u[1])

function bp_full_gfq(H::SparseMatrixCSC, Q::Integer, src::Vector{<:Integer}, HH::Real; 
        neutralel=1/Q*ones(SVector{Q}))
    
    n = size(H,2)
    efield = [SVector{Q}(ntuple(x->exp(-HH*(count_ones((x-1)⊻(s[i]-1)))),Q)) for i in 1:n]
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    h = fill(neutralel,nnz(H))
    u = fill(neutralel,nnz(H))
    belief = fill(neutralel,n)
    BPFull(H, X, h, u, copy(efield), belief)
end

function ms_full_gfq(H::SparseMatrixCSC, Q::Int, src::Vector{<:Integer}; 
    neutralel=zeros(SVector{Q}))

    n = size(H,2)
    efield = [SVector{Q}(ntuple(
        x->-float(count_ones((x-1)⊻(s[i]-1))),Q)) for i in 1:n]
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    h = fill(neutralel,nnz(H))
    u = fill(neutralel,nnz(H))
    belief = fill(neutralel,n)
    BPFull(H, X, h, u, copy(efield), belief)
end

function msg_conv_gfq!(u::MVector{Q,T}, h1::SVector{Q,T}, h2::SVector{Q,T}) where {Q,T}
    for x1 in eachindex(h1), x2 in eachindex(h2)
        # adjust for indices starting at 1 instead of 0
        u[((x1-1) ⊻ (x2-1)) + 1] += h1[x1]*h2[x2]
    end
    SVector(u)
end
function msg_conv_gfq(h1::SVector{Q,T}, h2::SVector{Q,T}) where {Q,T}
    u = zero(MVector{Q,T})
    msg_conv_gfq!(u,h1,h2)
end

function update_factor_bp!(bp::BPFull{F,SVector{Q,T}}, a::Int, 
        gfmult::SMatrix{Q,Q,<:Integer,L}, gfdiv::SMatrix{Q,Q,<:Integer,L},
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; uaux=zero(MVector{Q,T}),
        damp=0.0) where {F,Q,T,L}
    ε = 0.0
    for i in ∂a
        unew = neutral_prob_bp(Q)
        for j in ∂a
            j==i && continue
            # adjust for convolution weights
            hj_tilde = bp.h[j][SVector{Q}(gfdiv[:,nonzeros(bp.H)[j]+1])] 
            uaux .= zero(T)
            unew = msg_conv_gfq!(uaux, unew, hj_tilde)
        end
        # adjust result for convolution weights
        unew = unew[SVector{Q}(gfmult[:,nonzeros(bp.H)[i]+1])]
        unew = normalize_prob(unew)    # should not be needed
        ε = max(ε, maximum(abs,unew-bp.u[i]))
        bp.u[i] = bp.u[i].^damp .* unew.^(1-damp)
    end
    ε
end

function update_var_bp!(bp::BPFull{F,SVector{Q,T}}, i::Int; damp=0.0, rein=0.0) where {F,Q,T}
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
        ε = max(ε, maximum(abs,hnew-bp.h[a]))
        bp.h[a] = bp.h[a].^damp .* hnew.^(1-damp)
        b = msg_mult(b, bp.u[a]) 
    end
    iszero(sum(b)) && return Inf  # normaliz of belief is zero
    bp.belief[i] = normalize_prob(b) 
    bp.efield[i] = bp.efield[i] .* bp.belief[i].^rein
    ε
end


function iteration!(bp::BPFull{F,SVector{Q,T}}; maxiter=10^3, tol=1e-12, 
        damp=0.0, rein=0.0, 
        update_f! = update_factor_bp!, update_v! = update_var_bp!,
        factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.H,1)],
        gftab = gftables(Val(getQ(bp))), uaux=fill(zero(MVector{Q,T}),nfactors(bp)),
        callback=(it, ε, bp)->false) where {F,Q,T}

    ε = 0.0
    err = zeros(Threads.nthreads())
    m,n = size(bp.H)
    for it = 1:maxiter
        ε = 0.0
        # for i = 1:size(bp.H,2)
        Threads.@threads for i = rand(1:n, n÷3*2) 
            errv = update_v!(bp, i, damp=damp, rein=rein)
            errv == Inf && @warn "Contradiction found updating var $i"
            err[Threads.threadid()] = max(err[Threads.threadid()],errv)
        end
        # for a = 1:size(bp.H,1)
        for a = rand(1:m, m÷3*2)
            errf = update_f!(bp, a, gftab..., factor_neigs[a], #uaux=uaux[a], 
                damp=damp)
            err[Threads.threadid()] = max(err[Threads.threadid()],errf)
        end
        ε = maximum(err)
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

### MAX-SUM
msg_sum(u1::SVector, u2::SVector) = u1 .+ u2
normalize_max(u::SVector) = u .- maximum(u)
function neutral_prob_ms(Q::Integer)
    v = -Inf*ones(SVector{Q})
    v = setindex(v, 0.0, 1)
end
function msg_maxconv_gfq!(u::MVector{Q,T}, h1::SVector{Q,T}, h2::SVector{Q,T}) where {Q,T}
    for x1 in eachindex(h1), x2 in eachindex(h2)
        # adjust for indices starting at 1 instead of 0
        x = ((x1-1) ⊻ (x2-1)) + 1
        v = h1[x1]+h2[x2]
        v > u[x] && (u[x]=v)
    end
    SVector(u)
end
function msg_maxconv_gfq(h1::SVector{Q,T}, h2::SVector{Q,T}) where {Q,T}
    u = -Inf*ones(MVector{Q,T})
    msg_maxconv_gfq!(u,h1,h2)
end

function update_factor_ms!(bp::BPFull{F,SVector{Q,T}}, a::Int, 
        gfmult::SMatrix{Q,Q,<:Integer,L}, gfdiv::SMatrix{Q,Q,<:Integer,L},
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; uaux=-Inf*ones(MVector{Q,T}),
        damp=0.0) where {F,Q,T,L}
    ε = 0.0
    for i in ∂a
        unew = neutral_prob_ms(Q)
        for j in ∂a
            j==i && continue
            # adjust for convolution weights
            hj_tilde = bp.h[j][SVector{Q}(gfdiv[:,nonzeros(bp.H)[j]+1])] 
            uaux .= -Inf
            unew = msg_maxconv_gfq!(uaux, unew, hj_tilde)
        end
        # adjust result for convolution weights
        unew = unew[SVector{Q}(gfmult[:,nonzeros(bp.H)[i]+1])]
        unew = normalize_max(unew)    # should not be needed
        ε = max(ε, maximum(abs,unew-bp.u[i]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end

function update_var_ms!(bp::BPFull{F,SVector{Q,T}}, i::Int; damp=0.0, rein=0.0) where {F,Q,T}
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
        ε = max(ε, maximum(abs,hnew-bp.h[a]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_sum(b, bp.u[a]) 
    end
    isnan(sum(b)) && return Inf  # normaliz of belief is zero
    bp.belief[i] = normalize_max(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

# alias for calling `iteration!` with maxsum updates
function iteration_ms!(ms::BPFull{F,SVector{Q,T}}; kw...) where {F,Q,T}
    iteration!(ms; update_f! = update_factor_ms!, 
        update_v! = update_var_ms!, 
        uaux = fill(-Inf*ones(MVector{Q}), nfactors(ms)), kw...)
end

function parity(bp::BPFull{F,SVector{Q,T}}, x::Vector{<:Integer}=argmax.(bp.belief), 
        Ht = permutedims(bp.H),
        gfmult=gftables(Val(Q))[1]) where {F,Q,T}

    z = p = 0
    rows = rowvals(Ht)
    for a in 1:size(Ht,2)
        for i in nzrange(Ht,a)
            Hai = nonzeros(Ht)[i] + 1
            z = xor(z, gfmult[Hai, x[rowvals(Ht)[i]]] - 1)
        end
        p += count_ones(z) 
        z = 0
    end
    p
end
function distortion(x::Vector{<:Integer}, s::Vector{<:Integer}, Q::Int)
    z = xor.(x.-1,s.-1)
    d = mean(count_ones, z)
    d / log2(Q)
end
function performance(bp::BPFull{F,SVector{Q,T}}, s::Vector{<:Integer},
        Ht = permutedims(bp.H),
        gfmult=gftables(Val(Q))[1]) where {F,Q,T}
    x = argmax.(bp.belief)
    nunsat = parity(bp, x, Ht, gfmult)
    dist = distortion(x, s, Q)
    nunsat, dist
end
function mult_gfq(H::SparseMatrixCSC{T,<:Integer}, x::AbstractVector, Q::Integer,
        Ht = permutedims(H), gfmult=gftables(Val(Q))[1]) where T
    m,n = size(H)
    y = zeros(T,m)
    rows = rowvals(Ht)
    for a in 1:size(Ht,2)
        z = 0
        for i in nzrange(Ht,a)
            Hai = nonzeros(Ht)[i] + 1
            z = xor(z, gfmult[Hai, x[rowvals(Ht)[i]]] - 1)
        end
        y[a] = count_ones(z) 
    end
    y
end
function gfqto2(H::SparseMatrixCSC{<:Integer,<:Integer}, Q::Int)
    m,n = size(H)
    k = Int(log2(Q))
    Hnew = falses(m, n*k)
    rows = rowvals(H); vals = nonzeros(H)
    for j = 1:n
        for i in nzrange(H, j)
            row = rows[i]
            val = vals[i]
            Hnew[row, k*(j-1)+1:k*j] .= reverse(digits(val, base=2, pad=k))
        end
    end
    Hnew
end
function gfqto2(s::AbstractVector{<:Integer}, Q::Int)
    n = length(s)
    k = Int(log2(Q))
    snew = falses(n*k)
    for j = 1:n
        snew[k*(j-1)+1:k*j] .= reverse(digits(s[j], base=2, pad=k))
    end
    snew
end
function gf2toq(x::BitVector, Q::Int; T::Type=Int)
    n = length(x)
    k = Int(log2(Q))
    @assert mod(n,k)==0
    nnew = Int(n/k)
    xnew = zeros(T, nnew)
    for j in 1:nnew
        xtmp = @view x[k*(j-1)+1:k*j]
        xnew[j] = sum(xtmp[i]*2^(k-i) for i in eachindex(xtmp))
    end
    xnew
end

### In-place, linear-time cavity updates. Damping not possible!
# alias for calling `iteration!` with quick updates
function iteration_bp_quick!(bp::BPFull{F,SVector{Q,T}}; kw...) where {F,Q,T}
    iteration!(bp; update_f! = update_factor_bp!, 
        update_v! = update_var_bp_quick!, 
        uaux = fill(zero(MVector{Q}), nfactors(bp)), kw...)
end

function update_var_bp_quick!(bp::BPFull{F,SVector{Q,T}}, i::Int; 
        damp=0.0, rein=0.0) where {F,Q,T}
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    # sweep forward
    bp.h[∂i[1]] = bp.efield[i]
    for a in 2:length(∂i)
        bp.h[∂i[a]] = msg_mult(bp.h[∂i[a-1]], bp.u[∂i[a-1]])
    end
    bp.h[∂i[end]] = normalize_prob(bp.h[∂i[end]])
    b = msg_mult(bp.h[∂i[end]], bp.u[∂i[end]])
    # sweep backward
    h = bp.u[∂i[end]]
    for a in length(∂i)-1:-1:1
        bp.h[∂i[a]] = msg_mult(bp.h[∂i[a]], h)
        bp.h[∂i[a]] = normalize_prob(bp.h[∂i[a]])
        h = msg_mult(bp.u[∂i[a]], h)
    end
    iszero(sum(b)) && return Inf  # normaliz of belief is zero
    b = normalize_prob(b)
    ε = maximum(abs, bp.belief[i] .- b)
    bp.efield[i] = bp.efield[i] .* b.^rein
    bp.belief[i] = b
    ε
end

function update_factor_bp_quick!(bp::BPFull{F,SVector{Q,T}}, a::Int, 
        gfmult::SMatrix{Q,Q,<:Integer,L}, gfdiv::SMatrix{Q,Q,<:Integer,L},
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; uaux=zero(MVector{Q,T}),
        damp=0.0) where {F,Q,T,L}
    
    bp.u[∂a[1]] = neutral_prob_bp(Q)
    for i in 2:length(∂a)
        j = ∂a[i-1]
        h_tilde = bp.h[j][SVector{Q}(gfdiv[:,nonzeros(bp.H)[j]+1])] 
        uaux .= zero(T)
        bp.u[∂a[i]] = msg_conv_gfq!(uaux, bp.u[j], h_tilde)
    end
    u = bp.h[∂a[end]][SVector{Q}(gfdiv[:,nonzeros(bp.H)[∂a[end]]+1])] 
    for i in length(∂a)-1:-1:1
        uaux .= zero(T)
        bp.u[∂a[i]] = msg_conv_gfq!(uaux, bp.u[∂a[i]], u)
        h_tilde = bp.h[∂a[i]][SVector{Q}(gfdiv[:,nonzeros(bp.H)[∂a[i]]+1])] 
        uaux .= zero(T)
        u = msg_conv_gfq!(uaux, h_tilde, u)
    end
    # adjust for weights
    for i in ∂a
        bp.u[i] = bp.u[i][SVector{Q}(gfmult[:,nonzeros(bp.H)[i]+1])]
        bp.u[i] = normalize_prob(bp.u[i])
    end
    -Inf
end

# alias for calling `iteration!` with quick updates
function iteration_ms_quick!(ms::BPFull{F,SVector{Q,T}}; kw...) where {F,Q,T}
    iteration!(ms; update_f! = update_factor_ms_quick!, 
        update_v! = update_var_ms_quick!, 
        uaux = fill(-Inf*ones(MVector{Q}),nfactors(ms)), kw...)
end


function update_factor_ms_quick!(bp::BPFull{F,SVector{Q,T}}, a::Int, 
        gfmult::SMatrix{Q,Q,<:Integer,L}, gfdiv::SMatrix{Q,Q,<:Integer,L},
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; uaux=-Inf*ones(MVector{Q,T}),
        damp=0.0) where {F,Q,T,L}

    bp.u[∂a[1]] = neutral_prob_ms(Q)
    for i in 2:length(∂a)
        j = ∂a[i-1]
        h_tilde = bp.h[j][SVector{Q}(gfdiv[:,nonzeros(bp.H)[j]+1])] 
        uaux .= -Inf
        bp.u[∂a[i]] = msg_maxconv_gfq!(uaux, bp.u[j], h_tilde)
    end
    u = bp.h[∂a[end]][SVector{Q}(gfdiv[:,nonzeros(bp.H)[∂a[end]]+1])] 
    for i in length(∂a)-1:-1:1
        uaux .= -Inf
        bp.u[∂a[i]] = msg_maxconv_gfq!(uaux, bp.u[∂a[i]], u)
        h_tilde = bp.h[∂a[i]][SVector{Q}(gfdiv[:,nonzeros(bp.H)[∂a[i]]+1])] 
        uaux .= -Inf
        u = msg_maxconv_gfq!(uaux, h_tilde, u)
    end
    # adjust for weights
    for i in ∂a
        bp.u[i] = bp.u[i][SVector{Q}(gfmult[:,nonzeros(bp.H)[i]+1])]
        bp.u[i] = normalize_max(bp.u[i])
    end
    -Inf
end

function update_var_ms_quick!(bp::BPFull{F,SVector{Q,T}}, i::Int; 
        damp=0.0, rein=0.0) where {F,Q,T}
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    # sweep forward
    bp.h[∂i[1]] = bp.efield[i]
    for a in 2:length(∂i)
        bp.h[∂i[a]] = msg_sum(bp.h[∂i[a-1]], bp.u[∂i[a-1]])
    end
    bp.h[∂i[end]] = normalize_max(bp.h[∂i[end]])
    b = msg_sum(bp.h[∂i[end]], bp.u[∂i[end]])
    # sweep backward
    h = bp.u[∂i[end]]
    for a in length(∂i)-1:-1:1
        bp.h[∂i[a]] = msg_sum(bp.h[∂i[a]], h)
        bp.h[∂i[a]] = normalize_max(bp.h[∂i[a]])
        h = msg_sum(bp.u[∂i[a]], h)
    end
    isnan(sum(b)) && return Inf  # normaliz of belief is zero
    b = normalize_max(b)
    ε = maximum(abs, bp.belief[i] .- b)
    bp.efield[i] = bp.efield[i] .+ b.*rein
    bp.belief[i] = b
    ε
end


function gftables(Q::Val{2})
    gfmult = SA[1 1; 
                1 2]
    gfdiv = SA[0 1;
               0 2]
    gfmult, gfdiv
end
function gftables(Q::Val{4})
    gfmult = SA[1 1 1 1;
                1 2 3 4;
                1 3 4 2;
                1 4 2 3]
    gfdiv = SA[0 1 1 1;
               0 2 4 3;
               0 3 2 4;
               0 4 3 2] 
    gfmult, gfdiv
end
function gftables(Q::Val{8})
    gfmult = SA[1  1  1  1  1  1  1  1
                1  2  3  4  5  6  7  8
                1  3  5  7  4  2  8  6
                1  4  7  6  8  5  2  3
                1  5  4  8  7  3  6  2
                1  6  2  5  3  8  4  7
                1  7  8  2  6  4  3  5
                1  8  6  3  2  7  5  4]
    gfdiv = SA[ 0  1  1  1  1  1  1  1
                0  2  6  7  8  3  4  5
                0  3  2  8  6  5  7  4
                0  4  5  2  3  7  6  8
                0  5  3  6  2  4  8  7
                0  6  8  4  7  2  5  3
                0  7  4  3  5  8  2  6
                0  8  7  5  4  6  3  2]  
    gfmult, gfdiv
end
function gftables(Q::Val{16})
    gfmult = SA[1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
                1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
                1   3   5   7   9  11  13  15   4   2   8   6  12  10  16  14
                1   4   7   6  13  16  11  10  12   9  14  15   8   5   2   3
                1   5   9  13   4   8  12  16   7   3  15  11   6   2  14  10
                1   6  11  16   8   3  14   9  15  12   5   2  10  13   4   7
                1   7  13  11  12  14   8   2   6   4  10  16  15   9   3   5
                1   8  15  10  16   9   2   7  14  11   4   5   3   6  13  12
                1   9   4  12   7  15   6  14  13   5  16   8  11   3  10   2
                1  10   2   9   3  12   4  11   5  14   6  13   7  16   8  15
                1  11   8  14  15   5  10   4  16   6   9   3   2  12   7  13
                1  12   6  15  11   2  16   5   8  13   3  10  14   7   9   4
                1  13  12   8   6  10  15   3  11   7   2  14  16   4   5   9
                1  14  10   5   2  13   9   6   3  16  12   7   4  15  11   8
                1  15  16   2  14   4   3  13  10   8   7   9   5  11  12   6
                1  16  14   3  10   7   5  12   2  15  13   4   9   8   6  11]
    gfdiv = SA[ 0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
                0   2  10  15  14  12   8   7  16   3  13   6  11   5   4   9
                0   3   2  16  10   6  15  13  14   5  12  11   8   9   7   4
                0   4   9   2   5  15  10  11   3   7   8  16  14  13   6  12
                0   5   3  14   2  11  16  12  10   9   6   8  15   4  13   7
                0   6  12   4  13   2   9  14   7  11  10   3   5   8  16  15
                0   7   4   3   9  16   2   8   5  13  15  14  10  12  11   6
                0   8  11  13   6   5   7   2  12  15   3   9   4  16  10  14
                0   9   5  10   3   8  14   6   2   4  11  15  16   7  12  13
                0  10  14   8  16  13  11   4  15   2   7  12   6   3   9   5
                0  11   6   7  12   3   4  10  13   8   2   5   9  15  14  16
                0  12  13   9   7  10   5  16   4   6  14   2   3  11  15   8
                0  13   7   5   4  14   3  15   9  12  16  10   2   6   8  11
                0  14  16  11  15   7   6   9   8  10   4  13  12   2   5   3
                0  15   8  12  11   9  13   3   6  16   5   4   7  14   2  10
                0  16  15   6   8   4  12   5  11  14   9   7  13  10   3   2]
    gfmult, gfdiv
end
function gftables(Q::Val{32})
    gfmult = SA[1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
                1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32
                1   3   5   7   9  11  13  15  17  19  21  23  25  27  29  31   6   8   2   4  14  16  10  12  22  24  18  20  30  32  26  28
                1   4   7   6  13  16  11  10  25  28  31  30  21  24  19  18  22  23  20  17  26  27  32  29  14  15  12   9   2   3   8   5
                1   5   9  13  17  21  25  29   6   2  14  10  22  18  30  26  11  15   3   7  27  31  19  23  16  12   8   4  32  28  24  20
                1   6  11  16  21  18  31  28  14   9   8   3  26  29  20  23  27  32  17  22  15  12   5   2  24  19  30  25   4   7  10  13
                1   7  13  11  25  31  21  19  22  20  26  32  14  12   2   8  16  10   4   6  24  18  28  30  27  29  23  17   3   5  15   9
                1   8  15  10  29  28  19  22  30  27  20  21   2   7  16   9  32  25  18  23   4   5  14  11   3   6  13  12  31  26  17  24
                1   9  17  25   6  14  22  30  11   3  27  19  16   8  32  24  21  29   5  13  18  26   2  10  31  23  15   7  28  20  12   4
                1  10  19  28   2   9  20  27   3  12  17  26   4  11  18  25   5  14  23  32   6  13  24  31   7  16  21  30   8  15  22  29
                1  11  21  31  14   8  26  20  27  17  15   5  24  30   4  10  18  28   6  16  29  23   9   3  12   2  32  22   7  13  19  25
                1  12  23  30  10   3  32  21  19  26   5  16  28  17  14   7   2  11  24  29   9   4  31  22  20  25   6  15  27  18  13   8
                1  13  25  21  22  26  14   2  16   4  24  28  27  23   3  15  31  19   7  11  12   8  20  32  18  30  10   6   5   9  29  17
                1  14  27  24  18  29  12   7   8  11  30  17  23  28  13   2  15   4  21  26  32  19   6   9  10   5  20  31  25  22   3  16
                1  15  29  19  30  20   2  16  32  18   4  14   3  13  31  17  28  22   8  10   7   9  27  21   5  11  25  23  26  24   6  12
                1  16  31  18  26  23   8   9  24  25  10   7  15   2  17  32  12   5  22  27  19  30  13   4  29  20   3  14   6  11  28  21
                1  17   6  22  11  27  16  32  21   5  18   2  31  15  28  12  14  30   9  25   8  24   3  19  26  10  29  13  20   4  23   7
                1  18   8  23  15  32  10  25  29  14  28  11  19   4  22   5  30  13  27  12  20   3  21   6   2  17   7  24  16  31   9  26
                1  19   2  20   3  17   4  18   5  23   6  24   7  21   8  22   9  27  10  28  11  25  12  26  13  31  14  32  15  29  16  30
                1  20   4  17   7  22   6  23  13  32  16  29  11  26  10  27  25  12  28   9  31  14  30  15  21   8  24   5  19   2  18   3
                1  21  14  26  27  15  24   4  18   6  29   9  12  32   7  19   8  20  11  31  30  10  17   5  23   3  28  16  13  25   2  22
                1  22  16  27  31  12  18   5  26  13  23   4   8  19   9  30  24   3  25  14  10  29   7  20  15  28   2  21  17   6  32  11
                1  23  10  32  19   5  28  14   2  24   9  31  20   6  27  13   3  21  12  30  17   7  26  16   4  22  11  29  18   8  25  15
                1  24  12  29  23   2  30  11  10  31   3  22  32   9  21   4  19   6  26  15   5  20  16  25  28  13  17   8  14  27   7  18
                1  25  22  14  16  24  27   3  31   7  12  20  18  10   5  29  26   2  13  21  23  15   4  28   8  32  19  11   9  17  30   6
                1  26  24  15  12  19  29   6  23  16   2  25  30   5  11  20  10  17  31   8   3  28  22  13  32   7   9  18  21  14   4  27
                1  27  18  12   8  30  23  13  15  21  32   6  10  20  25   3  29   7  14  24  28   2  11  17  19   9   4  26  22  16   5  31
                1  28  20   9   4  25  17  12   7  30  22  15   6  31  23  14  13  24  32   5  16  21  29   8  11  18  26   3  10  19  27   2
                1  29  30   2  32   4   3  31  28   8   7  27   5  25  26   6  20  16  15  19  13  17  18  14   9  21  22  10  24  12  11  23
                1  30  32   3  28   7   5  26  20  15  13  18   9  22  24  11   4  31  29   2  25   6   8  27  17  14  16  19  12  23  21  10
                1  31  26   8  24  10  15  17  12  22  19  13  29   3   6  28  23   9  16  18   2  32  25   7  30   4   5  27  11  21  20  14
                1  32  28   5  20  13   9  24   4  29  25   8  17  16  12  21   7  26  30   3  22  11  15  18   6  27  31   2  23  10  14  19]
    gfdiv = SA[ 0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
                0   2  19  29  10  24  15  13  23   5  26  17   8  16   7  14  12  25   3  30  31  27   9   6  18  11  22  32   4  20  21  28
                0   3   2  30  19  12  29  25  10   9  24   6  15  31  13  27  23  22   5  32  26  18  17  11   8  21  16  28   7   4  14  20
                0   4  20   2  28  29  19  21  32  13  15  22  10  18  11  24  30  14   7   3   8  12  25  16  23  31  27   5   6  17  26   9
                0   5   3  32   2  23  30  22  19  17  12  11  29  26  25  18  10  16   9  28  24   8   6  21  15  14  31  20  13   7  27   4
                0   6  17   4   9   2  20  26   5  21  19  27  28  23  31  29   3  24  11   7  10  30  14  18  32   8  12  13  16  22  15  25
                0   7   4   3  20  30   2  14  28  25  29  16  19   8  21  12  32  27  13   5  15  23  22  31  10  26  18   9  11   6  24  17
                0   8  18  31  27  11  16   2  14  29   6  32  22   9  19   7  21   3  15  26  17  13  30  28  25  20   5  24  10  23   4  12
                0   9   5  28   3  10  32  16   2   6  23  21  30  24  22   8  19  31  17  20  12  15  11  14  29  27  26   4  25  13  18   7
                0  10  23   8  12  31  18   4  24   2  16   5  27  25  20  11  26   7  19  15  22  21   3   9  14  17  13  29  28  32   6  30
                0  11   6   7  17   3   4  24   9  14   2  18  20  10  26  30   5  12  21  13  19  32  27   8  28  15  23  25  31  16  29  22
                0  12  24  27  26  22  14  28  31  10  25   2  21   7  32  17  16  20  23  18  13   6  19   3  11   5   4   8  30  29   9  15
                0  13   7   5   4  32   3  27  20  22  30  31   2  15  14  23  28  18  25   9  29  10  16  26  19  24   8  17  21  11  12   6
                0  14  21  25  11   9  13  23   6  18   5  15   7   2  12  28  17  10  27  22   3  20   8  29   4  30  19  16  24  26  32  31
                0  15   8  26  18  21  31   3  27  30  11  28  16  17   2  13  14   5  29  24   6  25  32  20  22   4   9  12  19  10   7  23
                0  16  22   6  25   4  17  15  13  26  20  12   9  32   8   2   7  29  31  11  28   3  24  23   5  10  30  21  18  27  19  14
                0  17   9  20   5  19  28  31   3  11  10  14  32  12  16  15   2  26   6   4  23  29  21  27  30  18  24   7  22  25   8  13
                0  18  27  16  14   6  22  19  21  15  17  30  25   5  10   4  11   2   8  31   9   7  29  32  13  28   3  26  23  12  20  24
                0  19  10  15  23  26   8   7  12   3  31   9  18  22   4  21  24  13   2  29  16  14   5  17  27   6  25  30  20  28  11  32
                0  20  28  19  32  15  10  11  30   7   8  25  23  27   6  26  29  21   4   2  18  24  13  22  12  16  14   3  17   9  31   5
                0  21  11  13   6   5   7  12  17  27   3   8   4  19  24  32   9  23  14  25   2  28  18  15  20  29  10  22  26  31  30  16
                0  22  25  17  13  20   9   8   7  31  28  24   5  30  18  19   4  15  16   6  32   2  26  12   3  23  29  11  27  14  10  21
                0  23  12  18  24  16  27  20  26  19  22   3  14  13  28   6  31   4  10   8  25  11   2   5  21   9   7  15  32  30  17  29
                0  24  26  14  31  25  21  32  16  23  13  19  11   4  30   9  22  28  12  27   7  17  10   2   6   3  20  18  29  15   5   8
                0  25  13   9   7  28   5  18   4  16  32  26   3  29  27  10  20   8  22  17  30  19  31  24   2  12  15   6  14  21  23  11
                0  26  31  21  16  13  11  30  22  12   7  10   6  20  29   5  25  32  24  14   4   9  23  19  17   2  28  27  15   8   3  18
                0  27  14  22  21  17  25  10  11   8   9  29  13   3  23  20   6  19  18  16   5   4  15  30   7  32   2  31  12  24  28  26
                0  28  32  10  30   8  23   6  29   4  18  13  12  14  17  31  15  11  20  19  27  26   7  25  24  22  21   2   9   5  16   3
                0  29  15  24   8  14  26   5  18  32  21  20  31   6   3  25  27   9  30  12  11  22  28   4  16   7  17  23   2  19  13  10
                0  30  29  12  15  27  24   9   8  28  14   4  26  11   5  22  18  17  32  23  21  16  20   7  31  13   6  10   3   2  25  19
                0  31  16  11  22   7   6  29  25  24   4  23  17  28  15   3  13  30  26  21  20   5  12  10   9  19  32  14   8  18   2  27
                0  32  30  23  29  18  12  17  15  20  27   7  24  21   9  16   8   6  28  10  14  31   4  13  26  25  11  19   5   3  22   2]
    gfmult, gfdiv
end
function gftables(Q::Val{64})
    wd = pwd()
    cd(@__DIR__)
    m = readdlm("tables_gfq/gf64_mul.txt", Int)
    gfmult = SMatrix{64,64,Int}(m)
    d = readdlm("tables_gfq/gf64_div.txt", Int)
    gfdiv = SMatrix{64,64,Int}(d)
    cd(wd)
    gfmult, gfdiv
end
function gftables(Q::Val{128})
    wd = pwd()
    cd(@__DIR__)
    m = readdlm("tables_gfq/gf128_mul.txt", Int)
    gfmult = SMatrix{128,128,Int}(m)
    d = readdlm("tables_gfq/gf128_div.txt", Int)
    gfdiv = SMatrix{128,128,Int}(d)
    cd(wd)
    gfmult, gfdiv
end
function gftables(Q::Val{256})
    wd = pwd()
    cd(@__DIR__)
    m = readdlm("tables_gfq/gf256_mul.txt", Int)
    gfmult = SMatrix{256,256,Int}(m)
    d = readdlm("tables_gfq/gf256_div.txt", Int)
    gfdiv = SMatrix{256,256,Int}(d)
    cd(wd)
    gfmult, gfdiv
end


# GF(q) values here go from 0 to q-1
function gfdot(x::AbstractVector{Int}, y::AbstractVector{Int}, q::Int;
        mult::AbstractArray{Int,2}=gftables(Val(q))[1])
    z = 0
    for (xx,yy) in zip(x,y)
        if xx!=0 && yy!=0
            z = xor(z, mult[xx+1,yy+1]-1)
        end
    end
    z
end

function gfmult(A::AbstractArray{Int,2}, x::AbstractVector{Int}, q::Int;
        mult::AbstractArray{Int,2}=gftables(Val(q))[1])
    m, n = size(A)
    @assert length(x)==n
    b = zeros(Int, m)
    for i in eachindex(b)
        b[i] = gfdot(A[i,:], x, q; mult=mult)
    end
    b
end

function gfmult(A::AbstractArray{Int,2}, B::AbstractArray{Int,2}, q::Int;
        mult::AbstractArray{Int,2}=gftables(Val(q))[1])
    m, n = size(A)
    a, b = size(B)
    @assert a==n
    C = zeros(Int, m, b)
    for j in 1:b 
        for i in 1:m
            C[i,j] = gfdot(A[i,:], B[:,j], q; mult=mult)
        end
    end
    C
end


# Here the matrix elements range in 0:q-1
function gfrref!(H::AbstractArray{Int,2}, q::Int;
        gftab = gftables(Val(q)))
    mult, gfdiv = gftab 
    m, n = size(H)
    # Initialize pivot to zero
    dep = Int[]
    p = 0
    for c = 1:n
        nz = findfirst(!iszero, @view H[p+1:end,c])
        if nz === nothing
            continue
        else
            p += 1
            push!(dep, c)
            if nz != 1
                H[p,:], H[p+nz-1,:] = H[nz+p-1,:], H[p,:]
            end
            pvt = H[p,c] + 1
            # normalize row `p` so that pivot equal 1
            for j in c:n
                H[p,j] = gfdiv[H[p,j]+1, pvt] - 1
            end
            for r = 1:m
                r==p && continue
                # adjust row `r` to have all zeros below pivot
                if H[r,c] != 0
                    k = H[r,c]
                    for j in c:n
                        H[r,j] = xor(H[r,j], mult[k+1,H[p,j]+1]-1)
                    end
                end
            end
            if p == m 
                break
            end
        end
    end
    issparse(H) && dropzeros!(H)
    return H, dep
end
gfrref(H::AbstractArray{Int,2}, q::Int; gftab = gftables(Val(q))) = 
    gfrref!(copy(H), q, gftab=gftab)

function findbasis_slow(H::AbstractArray{Int,2}, q::Int; 
        gftab = gftables(Val(q)))
    H = Matrix(H)
    A, dep = gfrref(H, q, gftab=gftab)
    indep = setdiff(1:size(H,2), dep)
    colperm = [dep; indep]
    B = [A[1:length(dep),indep];I]
    B .= B[invperm(colperm),:]
    B, indep
end

function fix_indep!(x, B, indep, q::Int)
    x .= gfmult(B, x[indep], q)
end
include("bp_full.jl")
using StaticArrays, OffsetArrays

msg_mult(u1::SVector, u2::SVector) = u1 .* u2
normalize_prob(u::SVector) = u ./ sum(u)
function neutral_prob_bp(Q::Int)
    v = zero(SVector{Q})
    v = setindex(v, 1.0, 1)
end
getQ(bp::BPFull{F,SVector{Q,T}}) where {F,Q,T} = length(bp.u[1])

function bp_full_gfq(H::SparseMatrixCSC, Q::Int, src::Vector{Int}, HH::Real; 
        neutralel=1/Q*ones(SVector{Q}))
    
    n = size(H,2)
    efield = [SVector{Q}(ntuple(x->exp(-HH*(count_ones((x-1)⊻(s[i]-1)))),Q)) for i in 1:n]
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    h = fill(neutralel,nnz(H))
    u = fill(neutralel,nnz(H))
    belief = fill(neutralel,n)
    BPFull(H, X, h, u, copy(efield), belief)
end

function ms_full_gfq(H::SparseMatrixCSC, Q::Int, src::Vector{Int}; 
    neutralel=zeros(SVector{Q}))

    n = size(H,2)
    efield = [SVector{Q}(ntuple(x->-float(count_ones((x-1)⊻(s[i]-1))),Q)) for i in 1:n]
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
        gfmult::SMatrix{Q,Q,Int,L}, gfdiv::SMatrix{Q,Q,Int,L},
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
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
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
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
        b = msg_mult(b, bp.u[a]) 
    end
    iszero(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = normalize_prob(b) 
    bp.efield[i] = bp.efield[i] .* bp.belief[i].^rein
    ε
end


function iteration!(bp::BPFull{F,SVector{Q,T}}; maxiter=10^3, tol=1e-12, 
        damp=0.0, rein=0.0, 
    update_f! = update_factor_bp!, update_v! = update_var_bp!,
    factor_neigs = [nonzeros(bp.X)[nzrange(bp.X, a)] for a = 1:size(bp.H,1)],
    gftab = gftables(Val(getQ(bp))), uaux=zero(MVector{Q,T}),
    parities=zeros(Int, maxiter), dist=zeros(maxiter), s=argmax.(bp.efield),
    callback=(it, ε, bp)->false) where {F,Q,T}

    ε = 0.0
    for it = 1:maxiter
        ε = 0.0
        for i = 1:size(bp.H,2)
            errv = update_v!(bp, i, damp=damp, rein=rein)
            errv == -1 && return -1,it
            ε = max(ε, errv)
        end
        for a = 1:size(bp.H,1)
            errf = update_f!(bp, a, gftab..., factor_neigs[a], uaux=uaux, damp=damp)
            errf == -1 && return -1,it
            ε = max(ε, errf)
        end
        x = argmax.(bp.belief)
        parities[it] = parity(bp, x)
        dist[it] = distortion(x,s,getQ(bp))
        callback(it, ε, bp) && return ε,it
        ε < tol && return ε, it
    end
    ε, maxiter
end

### MAX-SUM
msg_sum(u1::SVector, u2::SVector) = u1 .+ u2
normalize_max(u::SVector) = u .- maximum(u)
function neutral_prob_ms(Q::Int)
    v = -Inf*ones(SVector{Q})
    v = setindex(v, 0.0, 1)
end
function msg_maxconv_gfq!(u::MVector{Q,T}, h1::SVector{Q,T}, h2::SVector{Q,T}) where {Q,T}
    for x1 in eachindex(h1), x2 in eachindex(h2)
        # adjust for indices starting at 1 instead of 0
        u[((x1-1) ⊻ (x2-1)) + 1] = max(u[((x1-1) ⊻ (x2-1)) + 1], h1[x1]+h2[x2])
    end
    SVector(u)
end

function update_factor_ms!(bp::BPFull{F,SVector{Q,T}}, a::Int, 
        gfmult::SMatrix{Q,Q,Int,L}, gfdiv::SMatrix{Q,Q,Int,L},
        ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]; uaux=zero(MVector{Q,T}),
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
    iszero(sum(b)) && return -1.0  # normaliz of belief is zero
    bp.belief[i] = normalize_max(b) 
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

# alias for calling `iteration!` with maxsum updates
function iteration_ms!(ms::BPFull{F,SVector{Q,T}}; kw...) where {F,Q,T}
    iteration!(ms; update_f! = update_factor_ms!, 
        update_v! = update_var_ms!, kw...)
end



function parity(bp::BPFull{F,SVector{Q,T}}, x::Vector{Int}, 
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

function mult_gfq(H::SparseMatrixCSC{T,Int}, x::AbstractVector, Q::Int,
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

function distortion(x::Vector{Int}, s::Vector{Int}, Q::Integer)
    z = xor.(x.-1,s.-1)
    d = mean(count_ones, z)
    d / log2(Q)
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
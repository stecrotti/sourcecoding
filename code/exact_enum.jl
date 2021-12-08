include("slim_graphs.jl")

using Plots

function distortion(x::BitVector, y::BitVector)
    @assert length(x) == length(y)
    d = 0
    for (xx, yy) in zip(x.chunks, y.chunks)
        d += count_ones( xor(xx, yy) )
    end
    d / length(x)
end

distortion(x::Integer, y::Integer) = count_ones( xor(x, y) )

function hamming_weight(x::BitVector)
    w = 0
    for xc in x.chunks
        w += count_ones(xc)
    end
    w
end

function hamming(A::BitArray, B::BitArray)
    #size(A) == size(B) || throw(DimensionMismatch("sizes of A and B must match"))
    Ac,Bc = A.chunks, B.chunks
    W = 0
    for i = 1:(length(Ac)-1)
        W += count_ones(Ac[i] ⊻ Bc[i])
    end
    W += count_ones(Ac[end] ⊻ Bc[end] & Base._msk_end(A))
    return W
end

function augment_basis(B)
    n, k = size(B)
    @assert k < 64
    r = 64 * ( floor(Int, n/64) + 1)
    BB = [B; falses(r-n, k)]
end

function bitmult_fast!(y::BitVector, B::BitMatrix, x::BitVector)
    n, k = size(B)
    @assert k < 64 "Adjust code for k larger than 64" 
    @assert mod(n, 64) == 0 "number of rows must be a multiple of 64. Can use `augment_basis`"
    fill!(y, false)
    nchunks = Int(length(B.chunks) / k)
    for j in eachindex(x)
        if x[j] != 0
            for i in eachindex(y.chunks)
                y.chunks[i] ⊻= B.chunks[i + (j-1)*nchunks]
            end
        end
    end
    y
end

function bitmult_fast!(y::BitVector, B::BitMatrix, x::Integer)
    @assert 0 <= x <= UInt64(2) ^ 64 - 1
    z = falses(64)
    z.chunks[1] = x
    bitmult_fast!(y, B, z)
end

# WARNING: size of y is determined by size of B which must be a multiple of 64
function bitmult_fast(B::BitMatrix, x)
    y = falses(size(B,1))
    bitmult_fast!(y, B, x)
end

# Given a basis `B` and a list of `sources`, computes the exact weight 
#  enumeration function w.r.t the zero codeword (`h0`) and with the
#  sources (`h`). Also returns the (normalized) minimum distortions for
#  each source
function exact_wef(B, sources=BitVector[]; showprogress=true,
    y = BitVector(undef, size(B,1)),
    x = BitVector(undef, size(B,2)),
    h0 = zeros(Int, size(B,1)+1),
    h = [zeros(Int, size(B,1)+1) for _ in sources],
    mins = fill(Inf, length(sources)),
    argmins = fill(0, length(sources)))

    n, k = size(B)
    @assert all(x->length(x)==n, sources)
    @assert k < 64 "Adjust code for k larger than 64" 

    r = 64 * ( floor(Int, n/64) + 1)
    # extend B so that #rows is a multiple of 64 => easier for multiplication
    BB = [B; falses(r-n, k)]
    
    dt = showprogress ? 1.0 : Inf
    prog = ProgressMeter.Progress(2^k, dt=dt)
    for i in 0:2^k-1
        x.chunks[1] = i 
        bitmult_fast!(y, BB, x)
        d = hamming_weight(y)
        h0[d+1] += 1
        for (j,s) in enumerate(sources)
            d = distortion(y, s)
            h[j][d+1] += 1
            if d < mins[j]
                mins[j] = d
                argmins[j] = i
            end
        end
        ProgressMeter.next!(prog)
    end
    h0, h, mins ./ n
end


# computes the next integer with the same number of zeros in the 
#  binary representation
function _next(v::Integer)
    t = (v | (v - 1)) + 1;  
    t | ((((t & -t) ÷ (v & -v)) >> 1) - 1)
end

function exact_wef_fast(B, indep, sources; 
    y = BitVector(undef, size(B,1)),
    x = BitVector(undef, size(B,2)),
    dist = zeros(sources))

    n, k = size(B)
    @assert k < 64 "Adjust code for k larger than 64" 
    @assert length(s) == n

    r = 64 * ( floor(Int, n/64) + 1 )
    BB = [B; falses(r-n, k)]
    s_indep = s[indep]
    m = n
    for d in 1:k
        d ≥ m && break
        z = (1 << d) - 1
        for _ in 1:binomial(k, d)
            x.chunks[1] = s_indep.chunks[1] ⊻ z
            bitmult_fast!(y, BB, x)
            dd = distortion(y, s)
            if dd < m
                m = dd
            end
            z = _next(z)
        end
    end
    m
end

function plot_wef!(pl::Plots.Plot, h::Vector{<:Real}; 
        label="WEF", plotmin=true, seriestype=:bar, kw...)
    n = length(h) - 1
    ff = findfirst(!iszero, h)
    ylab = "normalized counts"
    xlab = "distortion"
    if ff === nothing 
        @warn "WEF vector is empty"
        plotmin = false
        return Plots.plot!(pl, [0.5], [NaN], label=label, xlabel=xlab, ylabel=ylab; kw...)
    end
    m = ff - 1
    x = 0:n
    h = h ./ sum(h) .* n
    x = x ./ n
    m = m / n
    Plots.plot!(pl, x, h, label=label, xlabel=xlab, ylabel=ylab, st=seriestype; kw...)
    plotmin && Plots.vline!(pl, [m], label="min=$m")
    pl
end

function plot_wef(h::Vector{Int}; kw...)
    pl = Plots.plot()
    plot_wef!(pl, h; kw...)
end 

function plot_wef_prob(h::Vector{Int}, β::Real, kw...)
    n = length(h) - 1
    m = ( findfirst(!iszero, h) - 1 ) / n
    x = (0:n) ./ n
    xlab = "distortion"; ylab = "prob"
    p = [exp(-β*x[i+1])*h[i+1] for i in 0:n]
    p ./= sum(p)
    pl = Plots.bar(x, p, label="p(d)", xlabel=xlab, ylabel=ylab)
    Plots.vline!(pl, [m], label="min=$m")
    Plots.plot!(pl; kw...)
    pl
end

# multiplies B*x given the transpose of B 
# function bitmult!(y::BitVector, Bt::SparseMatrixCSC{Bool,Int}, x::BitVector)
#     rows = rowvals(Bt)
#     for j in 1:size(Bt, 2)
#         z = false
#         for k in nzrange(Bt, j)
#             i = rows[k]
#             z = xor(z, x[i]*Bt[i,j])
#         end
#         y[j] = z
#     end
#     y
# end
# function bitmult!(y::BitVector, Bt::AbstractMatrix{Bool}, x::BitVector)
#     for j in 1:size(Bt, 2)
#         z = false
#         for i in 1:size(Bt, 1)
#             z = xor(z, x[i]*Bt[i,j])
#         end
#         y[j] = z
#     end
#     y
# end


#### BENCHMARK
# include("matrix_generator.jl")
# using BenchmarkTools, Profile

# n = 50
# R = 0.28
# m = round(Int, n*(1-R))
# f3 = 1-3R
# Λ = [0,1-f3,f3]
# K = [0, 0, 1]
# nedges = 3m
# H = permutedims(ldpc_matrix(n,m,nedges,Λ,K))
# B, indep = findbasis_slow(BitMatrix(H))

# Profile.clear()
# @profile exact_wef(B)
# Profile.print()


# # version with multiplication
# foo!(y, B, x) = (mul!(y, B, x) .% 2; nothing)

# println("\n n=$n \n")
# println("Loop version, uses transposed matrix")
# @btime bitmult!($y, $Bt, $x);

# @btime bitmult_fast!($y, )
# println("Just multiplication and then .% 2")
# @btime foo!($y, $B, $x);


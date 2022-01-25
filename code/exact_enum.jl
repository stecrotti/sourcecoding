include("slim_graphs.jl")

using Plots

function distortion(x::BitVector, y::BitVector)
    @assert length(x) == length(y)
    hamming(x,y) / length(x)
end

distortion(x::Integer, y::Integer) = count_ones( xor(x, y) )

function hamming(x::BitVector)
    w = 0
    xc = x.chunks
    @inbounds for i in 1:(length(xc)-1)
        w += count_ones(xc[i])
    end
    w += count_ones(xc[end] & Base._msk_end(x))
    w
end

function hamming(A::BitArray, B::BitArray)
    #size(A) == size(B) || throw(DimensionMismatch("sizes of A and B must match"))
    Ac,Bc = A.chunks, B.chunks
    W = 0
    @inbounds for i = 1:(length(Ac)-1)
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
    @inbounds for j in eachindex(x)
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
        d = hamming(y)
        h0[d+1] += 1
        for (j,s) in enumerate(sources)
            d = hamming(y, s)
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

function exact_wef_fast(B, indep, s; 
    y = BitVector(undef, size(B,1)),
    x = BitVector(undef, size(B,2)))

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

function merge_wefs(W::Vector{Vector{Int}})
    L = maximum(length, W)
    w = zeros(Int, L)
    for i in eachindex(W)
        li = length(W[i])
        w[1:li] .+= W[i]
    end
    return w ./ length(W)
end

function plot_wef!(pl::Plots.Plot, h::Vector{<:Real}; 
        label="WEF", plotmin=true, seriestype=:bar, 
        normalize=true, kw...)
    n = length(h) - 1
    ff = findfirst(!iszero, h)
    ylab = "normalized counts"
    xlab = "distortion"
    if ff === nothing 
        @warn "WEF vector is empty"
        plotmin = false
        return Plots.plot!(pl, [0.5], [NaN], label=label, xlabel=xlab, ylabel=ylab; kw...)
    end
    if normalize
        h = h ./ sum(h) .* n
    end
    m = ff - 1
    x = 0:n
    x = x ./ n
    m = m / n
    Plots.plot!(pl, x, h, label=label, xlabel=xlab, ylabel=ylab, st=seriestype; kw...)
    plotmin && Plots.vline!(pl, [m], label="min=$m")
    pl
end

function plot_wef(h::Vector{<:Real}; kw...)
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

### LIGHTEST BASIS    

# list the numbers of all codewords with weight == w
function cws_of_weight_w(BB, w::Int, n::Int; 
        y = falses(n), x = falses(size(BB,2)))
    nn, k = size(BB)
    @assert mod(nn, 64)==0 "nn=$nn"
    cws = typeof(x.chunks[1])[]
    for i in 0:2^k-1
        @inbounds x.chunks[1] = i 
        bitmult_fast!(y, BB, x)
        d = hamming(y)
        if d == w
            push!(cws, i)
        end
    end
    cws
end

function islinearindep(B, v; Baux=[B v])
    gfrcefGF2!(Baux)
    return !all(iszero, Baux[:,end])
end

function lightest_basis(BB, indep, n::Int; y = falses(n), x = falses(size(BB,2)),
        showprogress=true)
    nn, k = size(BB)
    @assert length(indep) == k 
    @assert mod(nn, 64)==0 "nn=$nn"
    Blight = falses(k, 0)
    dt = showprogress ? 1.0 : Inf
    prog = ProgressUnknown(desc="Finding lightest basis", dt=dt)
    for w in 1:n
        cws = cws_of_weight_w(BB, w, n; y=y, x=x)
        for c in cws 
            @inbounds z = bitmult_fast(BB, c)[indep]
            # try to add c to the basis
            islinearindep(Blight, z) && (Blight = [Blight z])
            # if basis complete, return
            if size(Blight, 2) == k
                @inbounds Blight_full = reduce(hcat, bitmult_fast(BB, Blight[:,j]) for j in 1:size(Blight,2))
                return Blight_full, w # the identity part adds weight 1 to each vector
            end
        end
        next!(prog, showvalues=[("weight", w)])
    end
    error("Something went wrong")
end

# compute WEF only for basis vectors B
function basis_wef(B, w=zeros(Int, size(B,1)))    
    for j in 1:size(B,2)
        x = sum(B[:,j])
        w[x] += 1
    end
    w
end

### STUFF FOR CLUSTER EXPLORATION 2 NOTEBOOK

# save idx and distortion of the 2 best codewords
function exact_wef2(B, sources=BitVector[]; showprogress=true,
    y = BitVector(undef, size(B,1)),
    x = BitVector(undef, size(B,2)),
    h0 = zeros(Int, size(B,1)+1),
    h = [zeros(Int, size(B,1)+1) for _ in sources])

    n, k = size(B)
    @assert k < 64 "Adjust code for k larger than 64" 

    r = 64 * ( floor(Int, n/64) + 1)
    # extend B so that #rows is a multiple of 64 => easier for multiplication
    BB = [B; falses(r-n, k)]
    
    m1 = fill(Inf, length(sources)); m2 = copy(m1)
    idx1 = zeros(Int, length(sources)); idx2 = copy(idx1)
    
    dt = showprogress ? 1.0 : Inf
    prog = ProgressMeter.Progress(2^k, dt=dt, desc="Exact WEF for 2 closest CWs ")
    for i in 0:2^k-1
        x.chunks[1] = i 
        bitmult_fast!(y, BB, x)
        d = hamming(y)
        h0[d+1] += 1
        for (j,s) in enumerate(sources)
            d = hamming(y, s)
            h[j][d+1] += 1
            if d < m1[j]
                m2[j] = m1[j]; m1[j] = d
                idx2[j] = idx1[j]; idx1[j] = i
            elseif d < m2[j]
                idx2[j] = i; m2[j] = d
            end
        end
        ProgressMeter.next!(prog)
    end
    (m1, m2) ./ n, (idx1, idx2)
end

# compute the threshold value of distortion so that a proportion α of the 
#  Boltzmann probability mass is made up by codewords with less than that dist
function distortion_thresh(h, β, α=0.99)
    @assert α ≤ 1
    p = [h[i]*exp(-β*i) for i in eachindex(h)]
    p ./= sum(p)
    c = 0.0
    for i in eachindex(p)
        c += p[i]
        c ≥ α && return (i-1) / length(h), sum(@view h[1:i])
    end
    return (lastindex(p)-1) / length(h), sum(@view h[1:i])  
end

# count codewords up to distortion maxdistortion
function ncw(h, maxdistortion)   
    nc = 0
    n = length(h) - 1
    for i in eachindex(h)
        nc += h[i]
        ((i-1) / n) ≥ maxdistortion && return nc
    end
    error("something went wrong")
end

# compute A = {indices of cws up to weight maxdistance}
#  xor of A with any codeword gives its "neighbors"
function neighbors(BB, maxdistance::Int, ref::BitVector,
        y = falses(length(ref)), x = falses(size(BB,2)))
    nn, k = size(BB)
    @assert mod(nn, 64)==0 "nn=$nn"
    neigs = typeof(x.chunks[1])[]
    dist = Int[]
    for i in 0:2^k-1
        x.chunks[1] = i 
        bitmult_fast!(y, BB, x)
        y == ref && continue
        d = hamming(y, ref)
        if d <= maxdistance
            push!(neigs, i)
            push!(dist, d)
        end
    end
    neigs, dist
end

# connected component of codeword `ref`
function connected_component_fast(B, maxdistortion::Real, maxdistance::Int, ref::Integer, 
        s::BitVector; ncodewords = 0, showprogress=true,
        y = BitVector(undef, size(B,1)), x = BitVector(undef, size(B,2)),
        z = BitVector(undef, size(B,1)), t = BitVector(undef, size(B,2)),
        target::Integer=-1)
    BB = augment_basis(B)
    n, k = size(B)
    visited = OffsetArray(falses(2^k), -1)
    Q = [ref]
    neigs, _ = neighbors(BB, maxdistance, falses(size(BB,1)))
    neigs_aux = deepcopy(neigs)
    dt = showprogress ? 1.0 : Inf
    prog = ProgressMeter.ProgressUnknown(dt=dt, desc="Codewords visited")
    while !isempty(Q)
        w = popfirst!(Q)
        w == target && (visited[w] = true; return visited)
        visited[w] == true && continue
        visited[w] = true
        neigs_aux .= xor.(w, neigs)
        for r in neigs_aux
            x.chunks[1] = r
            bitmult_fast!(y, BB, x)
            distortion(y, s) <= maxdistortion && push!(Q, r)
        end   
        st = ncodewords == 0 ? "" : "/"*string(ncodewords)
        ProgressMeter.next!(prog, showvalues=[("added to cc", "$(sum(visited))"*st), (:ncodewords,2^k)])
    end
    visited    
end

# returns the first distortion for which c1, c2 are in the same connected component
function two_cw_in_same_cc(B, maxdistortions, maxdistance::Int, 
        c1::Integer, c2::Integer,
        s::BitVector; ncodewords=zeros(Int, length(maxdistortions)),
        showprogress=true)
    for (i, maxdistortion) in enumerate(maxdistortions)
        visited1 = connected_component_fast(B, maxdistortion, maxdistance, c1, s, ncodewords=ncodewords[i],
        showprogress=showprogress, target=c2)
        # visited2 = connected_component_fast(B, maxdistortion, maxdistance, c2, s, ncodewords=ncodewords[i],
        # showprogress=showprogress, target=c1)
        visited1[c2] && return maxdistortion
        # (visited1[c2] || visited2[c1]) && return maxdistortion
    end
    return Inf
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


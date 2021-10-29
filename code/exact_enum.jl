include("slim_graphs.jl")

using Plots

function distortion(x::BitVector, y::BitVector)
    d = 0
    for (xx, yy) in zip(x, y)
        d += xor(xx, yy)
    end
    d
end

# multiplies B*x given the transpose of B 
function bitmult!(y::BitVector, Bt::SparseMatrixCSC{Bool,Int}, x::BitVector)
    rows = rowvals(Bt)
    for j in 1:size(Bt, 2)
        z = false
        for k in nzrange(Bt, j)
            i = rows[k]
            z = xor(z, x[i]*Bt[i,j])
        end
        y[j] = z
    end
    y
end
function bitmult!(y::BitVector, Bt::AbstractMatrix{Bool}, x::BitVector)
    for j in 1:size(Bt, 2)
        z = false
        for i in 1:size(Bt, 1)
            z = xor(z, x[i]*Bt[i,j])
        end
        y[j] = z
    end
    y
end

# Given a basis `B` and a list of `sources`, computes the exact weight 
#  enumeration function w.r.t the zero codeword (`h0`) and with the
#  sources (`h`). Also returns the (normalized) minimum distortions for
#  each source
function exact_wef(B, sources=[]; showprogress=true,
    y = BitVector(undef, size(B,1)),
    x = BitVector(undef, size(B,2)),
    h0 = zeros(Int, size(B,1)+1),
    h = [zeros(Int, size(B,1)+1) for _ in sources],
    mins = fill(size(B,1), length(sources)),
    Bt = permutedims(B))

    n, k = size(B)
    @assert k < 64 "Adjust code for k larger than 64" 
    
    prog = ProgressMeter.Progress(2^k, enabled=showprogress, dt=1.0)
    for i in 0:2^k-1
        x.chunks[1] = i 
        bitmult!(y, Bt, x)  # y = B*x .% 2 , but faster for sparse B
        d = sum(y)
        h0[d+1] += 1
        for (j,s) in enumerate(sources)
            d = distortion(y, s)
            h[j][d+1] += 1
            mins[j] = min(mins[j], d)
        end
        ProgressMeter.next!(prog)
    end
    h0, h, mins ./ n
end

function plot_wef(h::Vector{Int}; normalize=true, rescale=true, kw...)
    n = length(h) - 1
    ff = findfirst(!iszero, h)
    ff === nothing && error("WEF vector is empty")
    m = ff - 1
    x = 0:n
    xlab = "distortion"; ylab = "counts"
    if normalize 
        h = h ./ sum(h)
        ylab = "normalized counts"
    end
    if rescale 
        x = x ./ n
        m = m / n
        xlab = "distortion / n"
    end
    pl = Plots.bar(x, h, label="WEF", xlabel=xlab, ylabel=ylab)
    Plots.vline!(pl, [m], label="min=$m")
    Plots.plot!(pl; kw...)
    pl
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

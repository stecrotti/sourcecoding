#### Convolutions on 𝐑 and 𝔾𝔽(2ᵏ) ####
using OffsetArrays

Fun = OffsetArray
OffsetArray(n::Integer, v = 0.0) = OffsetArray(fill(v, n), 0:n-1)
OffsetArray(X::AbstractVector) = Fun(Vector(X), 0:length(X)-1)
OffsetArray(X::Base.Generator) = Fun(collect(X), 0:length(X)-1)
OffsetArray(X::Array{<:Number,2}) = Fun(X, 0:size(X,1)-1, 0:size(X,2)-1)
# # Base.getindex(f::Fun, r::Union{UnitRange,StepRange}) = Fun(f.parent[r.+1])
domain!(f::Fun, n::Int, v=zero(eltype(f))) = n <= length(f.parent) ?  resize!(f.parent, n) :
    append!(f.parent, fill(v, n - length(f)))
import Base.*
*(f1::Fun, f2::Fun) = Fun(f1.parent .* f2.parent)
import Base.exp
exp(f::Fun) = exp.(f)

# Normal convolution
function convolution(f1::Fun, f2::Fun)
    q1, q2 = length(f1), length(f2)
    f3 = Fun(q1 + q2 -1, 0.0)
    for x1 in 0:q1-1
        for x2 in 0:q2-1
            f3[x1+x2] += f1[x1]*f2[x2]
        end
    end
    return f3
end

# Convolution on GF(2^k)
function gfconv(f1::Fun, f2::Fun)
    q = length(f1)
    f3 = Fun(q, 0.0)
    for x1 in 0:q-1
        @inbounds for x2 in 0:q-1
            f3[xor(x1,x2)] += f1[x1]*f2[x2]
        end
    end
    return f3
end

# Max-Sum convolution on GF(2^k)
function gfmsc(f1::Fun, f2::Fun)
    q = length(f1)
    f3 = OffsetArray(q, -Inf)
    @inbounds for x3 in 0:q-1
        for x1 in 0:q-1
            v = f1[x1] + f2[xor(x1, x3)]
            f3[x3] < v && (f3[x3] = v)
        end
    end
    return f3
end

# No allocation versions. Don't work well with 'reduce'
function gfconv!(f3::Fun, f1::Fun, f2::Fun)
    q = length(f1)
    for x3 in 0:q-1
        f3[x3] = f1[0]*f2[x3]
        @inbounds for x1 in 1:q-1
            f3[x3] += f1[x1]*f2[xor(x1, x3)]
        end
    end
    return f3
end

function gfmsc!(f3::Fun, f1::Fun, f2::Fun)
    q = length(f1)
    for x3 in 0:q-1
        f3[x3] = f1[0] + f2[x3]
        @inbounds for x1 in 1:q-1
            v = f1[x1] + f2[xor(x1, x3)]
            f3[x3] < v && (f3[x3] = v)
        end
    end
    return f3
end

# Convolutions with weights
function gfconvw(F, gfdiv, w::Vector{Int}=ones(Int, length(F)),
        neutral=Fun(x == 0 ? 1.0 : 0.0 for x=0:length(F[1])-1))

    any(<=(0), w) && error("Weights must be positive")

    Fweighted = [ F[i][ gfdiv[:,w[i]] ] for i in eachindex(w)]::Array{OffsetArray{Float64,1,Array{Float64,1}},1}
    f = reduce(gfconv, Fweighted, init=neutral)::OffsetArray{Float64,1,Vector{Float64}}
    return f
end

function gfmscw(F, gfdiv, w::Vector{Int}=ones(Int, length(F)),
        neutral=Fun(x == 0 ? 0.0 : -Inf for x=0:length(F[1])-1))

    any(<=(0), w) && error("Weights must be positive")

    # Fweighted = [ F[i][ gfdiv[:,w[i]] ] for i in eachindex(w)]
    # f = reduce(gfmsc, Fweighted, init=neutral)
    for i in eachindex(w)
        F[i] .= F[i][gfdiv[:,w[i]]]
    end
    f = reduce(gfmsc, F, init=neutral)
    return f
end


### Not used
# function gfconvweighted(f1::Fun, f2::Fun, w1::Int=1, w2::Int=1;
#     mult::OffsetArray{Int,2}=gftables(length(f1))[1],
#     gfinv::Vector{Int}=gftables(length(f1))[2])
#
#     g1 = f1[mult[gfinv[w1],:]]
#     g2 = f2[mult[gfinv[w2],:]]
#     return gfconv(g1,g2)
# end

function gfmscweighted(f1::Fun, f2::Fun, w1::Int=1, w2::Int=1;
    mult::OffsetArray{Int,2}=gftables(length(f1))[1],
    gfinv::Vector{Int}=gftables(length(f1))[2])

    g1 = f1[mult[gfinv[w1],:]]
    g2 = f2[mult[gfinv[w2],:]]
    return gfmsc(g1,g2)
end

function gfconvlist(lf, conv=gfconv)
    n = length(lf)
    n == 1 && return lf[1]
    n == 0 && return Fun([1])    # return neutral element of convolution
    f1 = gfconvlist(lf[1:n÷2], conv)
    f2 = gfconvlist(lf[n÷2+1:end], conv)
    conv(f1, f2)
end

function gfmsclist(lf, conv=gfmsc)
    n = length(lf)
    n == 1 && return lf[1]
    n == 0 && return Fun([0])   # return neutral element of MSC
    f1 = gfmsclist(lf[1:n÷2], conv)
    f2 = gfmsclist(lf[n÷2+1:end], conv)
    conv(f1, f2)
end

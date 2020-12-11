#### A wrapper for a FactorGraph object with temperature(s), current guess for the solution, source vector ####
using LinearAlgebra

mutable struct LossyModel
    fg::FactorGraph     # Factor graph instance
    x::Vector{Int}      # Current state
    beta1::Real         # Inverse temperature for checks
    beta2::Real         # Inverse temperature for overlap with input vector y
    y::Vector{Int}      # Vector to be compressed
end

# Constructor for lossy model with LDPC matrix
function LossyModel(q::Int, n::Int, m::Int; beta1::Real=Inf, beta2::Real=1.0,
        nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
        rho::Vector{T}=generate_polyn(n,m)[3],
        fields = [Fun(1e-3*randn(q)) for v in 1:n], verbose=false,
        arbitrary_mult = false,
        randseed::Int=0) where {T<:AbstractFloat}

    !ispow2(q) && warning("The value of q you inserted (q=$q) is not a power of 2")
    fg = ldpc_graph(q,n,m, nedges, lambda, rho, fields, verbose=false,
        arbitrary_mult=arbitrary_mult, randseed=randseed)
    x = zeros(Int, n)
    y = rand(MersenneTwister(randseed), 0:q-1, n)
    return LossyModel(fg, x, beta1, beta2, y)
end

function LossyModel(fg::FactorGraph)
    x = zeros(Int, fg.n)
    beta1 = Inf
    beta2 = 1.0
    y = rand(0:fg.q-1, fg.n)
    return LossyModel(fg, x, beta1, beta2, y)
end

function Base.show(io::IO, lm::LossyModel)
    println(io, "Lossy compression model:")
    println(io, " - ", dispstring(lm.fg))
    println(io, " - Inverse temperatures β₁=$(lm.beta1) for checks and",
        " β₂=$(lm.beta2) for overlap")
end

function rate(lm::LossyModel) 
    n_indep_rows = rank(lm)
    r = 1 - n_indep_rows/lm.fg.n
    return r
end
function distortion(lm::LossyModel, x::Vector{Int}=lm.x)
    # return hd(x,lm.y)/(lm.fg.n*log2(lm.fg.q))
    return distortion(lm.fg, lm.y, x)
end
adjmat(lm::LossyModel) = adjmat(lm.fg)
import LinearAlgebra.nullspace, LinearAlgebra.rank
nullspace(fg::FactorGraph) = gfnullspace(adjmat(fg), fg.q)
nullspace(lm::LossyModel) = gfnullspace(lm.fg)
log_nsolutions(lm::LossyModel)::Int = size(nullspace(lm), 2)
nsolutions(lm::LossyModel)::Int = lm.fg.q^log_nsolutions(lm)
rank(fg::FactorGraph)::Int = gfrank(adjmat(fg), fg.q)
rank(lm::LossyModel)::Int = gfrank(lm.fg)
isfullrank(lm::LossyModel)::Bool = rank(lm::LossyModel)==lm.fg.m

function breduction!(lm::LossyModel, args...; kwargs...)
    b = breduction!(lm.fg, args...; kwargs...)
    return b
end

# Support for general input x (can also be a matrix)
function paritycheck(lm::LossyModel, x::Array{Int,2}, varargin...)
    return paritycheck(lm.fg, x, varargin...)
end

# Input as a vector instead of 2d array
function paritycheck(lm::LossyModel, x::Vector{Int}=lm.x, varargin...)
    return paritycheck(lm.fg, x[:,:], varargin...)
end

function parity(lm::LossyModel, args...)
    return sum(paritycheck(lm, args...))
end

function energy(lm::LossyModel, x::Vector{Int}=lm.x)
    ener_checks = energy_checks(lm ,x)
    ener_overlap = energy_overlap(lm, x)
    return ener_checks + ener_overlap
end

function energy_checks(lm::LossyModel, x::Union{Vector{Int},Array{Int,2}}=lm.x;
                        f::Union{Int,Vector{Int},Array{Int,2}}=collect(1:lm.fg.m))
    # Unsatisfied checks
    z = paritycheck(lm, x, f)
    hw_checks = sum(hw(z))
    # In principle this should jsut be lm.beta1*hw_checks, but gotta take into
    #  account the case beta1=Inf, for which we choose the convention Inf*0=0
    ener_checks = hw_checks == 0 ? 0 : lm.beta1*hw_checks
end

function energy_overlap(lm::LossyModel, x::Union{Vector{Int},Array{Int,2}}=lm.x)
    return lm.beta2*hd(x, lm.y)
end

function refresh!(lm::LossyModel, args...)
    return refresh!(lm.fg, lm.y, args...)
end

# Gaussian elimination on the graph
function gfref!(lm::LossyModel)
    H = adjmat(lm)
    gfref!(H, lm.fg.q, lm.fg.mult, lm.fg.gfdiv)
    lm.fg = FactorGraph(H)
    return nothing
end

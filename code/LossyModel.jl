#### A wrapper for a FactorGraph object with temperature(s), current guess for the solution, source vector ####

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
    fg = ldpc_graph(q,n,m, nedges, lambda, rho, fields, verbose=verbose,
        arbitrary_mult=arbitrary_mult, randseed=randseed)
    x = zeros(Int, n)
    y = rand(MersenneTwister(randseed), 0:q-1, n)
    return LossyModel(fg, x, beta1, beta2, y)
end

function Base.show(io::IO, lm::LossyModel)
    println(io, "Lossy compression model:")
    println(io, " - ", dispstring(lm.fg))
    println(io, " - Inverse temperatures β₁=$(lm.beta1) for checks and β₂=$(lm.beta2) for overlap")
end

rate(lm::LossyModel) = 1 - lm.fg.m/lm.fg.n
function distortion(lm::LossyModel, x::Vector{Int}=lm.x)
    # return hd(x,lm.y)/(lm.fg.n*log2(lm.fg.q))
    return distortion(lm.fg, lm.y, x)
end
adjmat(lm::LossyModel) = adjmat(lm.fg)
basis(lm::LossyModel) = gfnullspace(adjmat(lm), lm.fg.q)
gfrank(lm::LossyModel) = gfrank(adjmat(lm), lm.fg.q)

# Support for general input x (can also be a matrix)
function paritycheck(lm::LossyModel, x::Array{Int,2}, varargin...)
    return paritycheck(lm.fg, x, varargin...)
end

# Input as a vector instead of 2d array
function paritycheck(lm::LossyModel, x::Vector{Int}=lm.x, varargin...)
    return paritycheck(lm.fg, x[:,:], varargin...)
end

function energy(lm::LossyModel, x::Vector{Int}=lm.x)
    ener_checks = energy_checks(lm ,x)
    ener_overlap = energy_overlap(lm)
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



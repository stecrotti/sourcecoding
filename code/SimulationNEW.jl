#### Perform simulations and store results ####
using Parameters    # constructors with default values

@with_kw struct Simulation{T<:LossyAlgo}
    algo::T = T()
    q::Int = 2
    n::Int = 100
    R::Float64 = 0.5
    b::Int = 0
    navg::Int = 50
    it::Int = 1
    arbitrary_mult::Bool = false
    # opts::AlgoOptions = AlgoOptions(algo)
    results::Dict{Symbol,Any} = Dict{Symbol,Any}()
    runtimes::Vector{Float64} = zeros(navg)
end

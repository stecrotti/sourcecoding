#### Perform simulations and store results ####
using Parameters    # constructors with default values

@with_kw struct Simulation{T<:LossyAlgo}
    algo::T = T()
    q::Int = 2
    n::Int = 100
    m::Int = 50
    niter::Int = 50
    b::Int = 0
    # it::Int = 1
    arbitrary_mult::Bool = false
    results::Vector{Dict{Symbol,Any}} = Vector{Dict{Symbol,Any}}(undef,niter)
    runtimes::Vector{Float64} = zeros(niter)
end

function Simulation(lm::LossyModel, algo::LossyAlgo, niter::Int=50; kwargs...)
    q = lm.fg.q
    n = lm.fg.n
    m = lm.fg.m 
    arbitrary_mult = (lm.fg.mult == gftables(lm.fg.q))
    for it in 1:niter
        (results[it], runtimes[it]) = @timed solve!(lm, algo)
    end
    return Simulation(algo, q, n, m, arbitrary_mult=arbitrary_mult;
        kwargs...)
end
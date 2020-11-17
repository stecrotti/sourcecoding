#### Perform simulations and store results ####
using Parameters    # constructors with default values

@with_kw struct Simulation{T<:LossyAlgo}
    # algo::T = T()
    q::Int
    n::Int
    m::Int
    niter::Int
    b::Int = 0
    arbitrary_mult::Bool = false
    results::Vector{LossyResults} = Vector{LossyResults}(undef, niter)
    runtimes::Vector{Float64} = zeros(niter)
end

function Simulation(q::Int, n::Int, m::Int, algo::LossyAlgo; 
    b::Int=0, 
    niter::Int=50,
    arbitrary_mult::Bool = false,   # Shuffle multiplication table
    samegraph = false,              # If true, only 1 graph is extracted and all |navg| simulations are run on it
    samevector = false,             # If true, only 1 vector is extracted and all |navg| simulations are run on it
    randseed = 100,                 # For reproducibility
    verbose = true)

    lm = LossyModel(q, n, m+b, beta2=init_beta2(algo), verbose=verbose, 
        arbitrary_mult=arbitrary_mult,
        randseed=randseed)

    breduction!(lm.fg, b, randseed=randseed)

    

    arbitrary_mult = (lm.fg.mult == gftables(lm.fg.q))
    results = Vector{LossyResults}(undef, niter)
    runtimes = zeros(niter)
    for it in 1:niter
        (results[it], runtimes[it]) = @timed solve!(lm, algo)
    end
    return Simulation{typeof(algo)}(q=q, n=n, m=m, arbitrary_mult=arbitrary_mult,
        results=results, niter=niter)
end
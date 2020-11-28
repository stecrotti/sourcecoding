#### Perform simulations and store results ####
using Parameters, Lazy  

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
    verbose::Bool = true,
    showprogress::Bool = verbose)

    results = Vector{LossyResults}(undef, niter)
    runtimes = zeros(niter)

    lm = LossyModel(q, n, m+b, beta2=beta2_init(algo), verbose=verbose, 
        arbitrary_mult=arbitrary_mult, randseed=randseed)

    breduction!(lm.fg, b, randseed=randseed)

    verbose && println()
    for it in 1:niter
        if !samevector
            lm.y .= rand(MersenneTwister(randseed+it), 0:q-1, n)
        end
        if !samegraph
            lm.fg = ldpc_graph(q, n, m+b, verbose=false,
                randseed=randseed+it, arbitrary_mult=arbitrary_mult)
            breduction!(lm.fg, b, randseed=randseed+it)
        end
        (results[it], runtimes[it]) = @timed solve!(lm, algo, randseed=randseed,
            verbose=verbose, showprogress=showprogress)
        verbose && println("# Finished iter $it of $niter: ", output_str(results[it]), "\n")
        # Reinitialize messages if you're gonna reuse the same graph
        samegraph && refresh!(lm.fg)
    end
    arbitrary_mult = (lm.fg.mult == gftables(lm.fg.q))
    return Simulation{typeof(algo)}(q,n,m,niter,b,arbitrary_mult,results,runtimes)
end


# Binary entropy function
function H2(x::Real)
    if 0<x<1
        return -x*log2(x)-(1-x)*log2(1-x)
    elseif x==0 || x==1
        return 0
    else
        error("$x is outside the domain [0,1]")
    end
end

rdb(D::Real) = 1-H2(D)

function plot!(pl::Plots.Plot, sims::Vector{Simulation{T}}; 
    label::String="Experimental data") where {T<:LossyAlgo}

    dist = distortion.(sims)
    rate = [1-sim.m/sim.n for sim in sims]
    Plots.scatter!(pl, rate, dist, label=label)
    xlabel!(pl, "Rate")
    ylabel!(pl, "Distortion")
    return pl
end

function plot(sims::Vector{Simulation{T}}; kwargs...) where {T<:LossyAlgo}
    d = LinRange(0,0.5,100)
    r = LinRange(0, 1, 100)
    pl = Plots.plot(rdb.(d), d, label="RDB")
    return plot!(pl, sims; kwargs...)
end

function distortion(results::Vector{LossyResults})
    D = [r.distortion for r in results]
end
@forward Simulation.results distortion
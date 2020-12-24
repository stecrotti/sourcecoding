#### Perform simulations and store results ####
using Parameters, Lazy, StatsBase, Dates, Printf

@with_kw struct Simulation{T<:LossyAlgo}
    q::Int
    n::Int
    m::Int
    algo::LossyAlgo
    niter::Int
    b::Int = 0
    arbitrary_mult::Bool = false
    results::Vector{LossyResults} = Vector{LossyResults}(undef, niter)
    runtimes::Vector{Float64} = zeros(niter)
end


function Simulation(q::Int, n::Int, m::Int, algo::LossyAlgo; 
    b::Int=0, 
    gauss_elim::Bool=false,
    niter::Int=50,
    arbitrary_mult::Bool = false,   # Shuffle multiplication table
    samegraph::Bool = false,              # If true, only 1 graph is extracted and all |navg| simulations are run on it
    samevector::Bool = false,             # If true, only 1 vector is extracted and all |navg| simulations are run on it
    randseed::Int = 100,                 # For reproducibility
    verbose::Bool = true,
    showprogress::Bool = false)

    results = Vector{LossyResults}(undef, niter)
    runtimes = zeros(niter)

    lm = LossyModel(q, n, m+b, beta2=beta2_init(algo), verbose=verbose, 
        arbitrary_mult=arbitrary_mult, randseed=randseed)
    breduction!(lm.fg, b, randseed=randseed)

    gauss_elim && gfref!(lm)

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
        if verbose 
            it_str = @sprintf("%3d", it)
            println("# Iter "*it_str*" of $niter: ", output_str(results[it]))
        end
        # Reinitialize messages if you're gonna reuse the same graph
        samegraph && refresh!(lm.fg)
    end
    arbitrary_mult = (lm.fg.mult == gftables(lm.fg.q))
    return Simulation{typeof(algo)}(q,n,m,algo,niter,b,arbitrary_mult,results,
        runtimes)
end

#### GETTERS
function rate(sim::Simulation{<:LossyAlgo})
    return 1 - sim.m/sim.n
end
function iterations(sim::Simulation{<:LossyAlgo}; convergedonly::Bool=false)
    return [r.iterations for r in sim.results if (r.converged || !convergedonly)]
end
function trials(sim::Simulation{<:LossyAlgo}; convergedonly::Bool=false)
    return [r.trials for r in sim.results if (r.converged || !convergedonly)]
end
function runtime(sim::Simulation{<:LossyAlgo}; convergedonly::Bool=false)
    conv = [r.converged for r in sim.results]
    run = sim.runtimes
    return sum(run[conv .| !convergedonly])
end
function runtime(sims::Vector{Simulation{T}}; kwargs...) where {T<:LossyAlgo}
    return sum(runtime(sim; kwargs...) for sim in sims)
end
function nconverged(sim::Simulation{<:LossyAlgo})
    return sum(r.converged for r in sim.results)
end
convergenceratio(sim::Simulation{<:LossyAlgo}) = nconverged(sim)/sim.niter
function nunconverged(sim::Simulation{<:LossyAlgo})
    return sim.niter - nconverged(sim)
end
function convergence_ratio(sim::Simulation{<:LossyAlgo})
    return nconverged(sim)/sim.niter
end
function distortion(results::Vector{LossyResults}, convergedonly::Bool=false)
    D = [r.distortion for r in results if (r.converged || !convergedonly)]
end
@forward Simulation.results distortion



#### PRINTERS
function Base.show(io::IO, sim::Simulation{<:LossyAlgo})
    println(io, "\nSimulation{$(typeof(sim.algo))} with q=", sim.q,
        ", n=", sim.n, ", R=", 
        round(1-sim.m/sim.n,digits=2),
         ", b=", sim.b,", niter=", sim.niter)
    return nothing
end
function runtime_str(sim::Union{Simulation{<:T},Vector{Simulation{T}}}; 
        kwargs...) where {T<:LossyAlgo}
    seconds = Second(round(runtime(sim; kwargs...)))
    c = canonicalize(seconds)
    return string(c)
end


#### RATE-DISTORTION
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
rdbinv(R::Real) = H2inv(1-R)


#### PLOTTING
import Plots: plot!, plot, histogram, bar

function plot!(pl::Plots.Plot, sims::Vector{Simulation{T}}; 
        allpoints::Bool=false,
        label::String="Experimental data", 
        convergedonly::Bool=false, msw::Number=0.5, plotkw...) where {T<:LossyAlgo}
    dist = distortion.(sims, convergedonly)
    npoints = length.(dist)
    r = [rate(sim) for sim in sims]
    if allpoints
        rate_c_augmented = vcat([rate(sim)*ones(nconverged(sim)) for sim in sims]...)
        rate_u_augmented = vcat([rate(sim)*ones(nunconverged(sim)) for sim in sims]...)
        dist_c = [[r.distortion for r in sim.results if r.converged] for sim in sims] 
        dist_u = [[r.distortion for r in sim.results if !r.converged] for sim in sims] 
        dist_c_augmented = vcat(dist_c...)
        dist_u_augmented = vcat(dist_u...)
        Plots.scatter!(pl, rate_c_augmented, dist_c_augmented, markersize=3,
            label="Converged"; msw=msw, plotkw...)
        Plots.scatter!(pl, rate_u_augmented, dist_u_augmented, markersize=3,
            label="Unconverged"; msw=msw, plotkw...)
    else
        dist_avg = mean.(dist)
        if Plots.backend() == Plots.UnicodePlotsBackend()
            Plots.scatter!(pl, r, dist_avg, label=label, size=(300,200); plotkw...)
        else
            dist_sd = std.(distortion.(sims)) ./ [sqrt(npoints[i]) for i in eachindex(sims)]
            Plots.scatter!(pl, r, dist_avg, label=label, yerror=dist_sd; 
                msw=msw, plotkw...)
        end
    end
    xlabel!(pl, "R")
    ylabel!(pl, "D")
    return pl
end
function plot!(pl::Plots.Plot, sim::Simulation{<:LossyAlgo}; kwargs...)
    plot!(pl, [sim]; kwargs...)
end

function plot(sims::Union{Simulation{T},Vector{Simulation{T}}}; 
        kwargs...) where {T<:LossyAlgo}
    d = LinRange(0,0.5,100)
    r = LinRange(0, 1, 100)
    pl = Plots.plot(rdb.(d), d, label="RDB")
    Plots.plot!(pl, r, 0.5*(-r.+1), label="Naive compression")
    
    return plot!(pl, sims; kwargs...)
end


function iters_hist(sim::Simulation{<:LossyAlgo}; kwargs...)
    iters = iterations(sim; kwargs...)
    h = Plots.histogram(iters, nbins=50)
    return h
end
function trials_hist(sim::Simulation{<:LossyAlgo}; kwargs...)
    t = trials(sim; kwargs...)
    h = counts(t, 1:maximum(t))
    b = Plots.bar(h, bar_width=1)
    return b
end


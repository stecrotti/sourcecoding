#### Perform simulations and store results ####
using Parameters, Lazy, StatsBase, Dates, Printf

@with_kw struct Simulation{T<:LossyAlgo}
    q::Int
    n::Int
    m::Int
    algo::T
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

    lm = LossyModel(Val(q), n, m+b, beta2=beta2_init(algo), verbose=false, 
        arbitrary_mult=arbitrary_mult, randseed=randseed)
    breduction!(lm.fg, b, randseed=randseed)

    gauss_elim && gfref!(lm)

    for it in 1:niter
        if !samevector
            lm.y .= rand(MersenneTwister(randseed+it), 0:q-1, n)
        end
        if !samegraph
            lm.fg = ldpc_graph(Val(q), n, m+b, verbose=false,
                randseed=randseed+it, arbitrary_mult=arbitrary_mult)
            breduction!(lm.fg, b, randseed=randseed+it)
        end
        (results[it], runtimes[it]) = @timed solve!(lm, algo, randseed=randseed,
            verbose=verbose, showprogress=showprogress)
        if verbose 
            it_str = @sprintf("%3d", it)
            println(it_str*" of $niter: ", output_str(results[it]))
        end
    end
    arbitrary_mult = (lm.fg.mult != gftables(lm.fg.q)[1])
    return Simulation{typeof(algo)}(q,n,m,algo,niter,b,arbitrary_mult,results,
        runtimes)
end

#### GETTERS
function rate(sim::Simulation)
    return 1 - sim.m/sim.n
end
function iterations(sim::Simulation; convergedonly::Bool=false)
    return [r.iterations for r in sim.results if (r.converged || !convergedonly)]
end
function trials(sim::Simulation; convergedonly::Bool=false)
    return [r.trials for r in sim.results if (r.converged || !convergedonly)]
end
function runtime(sim::Simulation; convergedonly::Bool=false)
    conv = [r.converged for r in sim.results]
    run = sim.runtimes
    return sum(run[conv .| !convergedonly])
end
function runtime(sims::Vector{Simulation{T}}; kwargs...) where {T<:LossyAlgo}
    return sum(runtime(sim; kwargs...) for sim in sims)
end
function nconverged(sim::Simulation)
    return sum(r.converged for r in sim.results)
end
convergenceratio(sim::Simulation) = nconverged(sim)/sim.niter
function nunconverged(sim::Simulation)
    return sim.niter - nconverged(sim)
end
function convergence_ratio(sim::Simulation)
    return nconverged(sim)/sim.niter
end
function distortion(results::Vector{<:LossyResults}, convergedonly::Bool=false)
    D = [r.distortion for r in results if (r.converged || !convergedonly)]
end
@forward Simulation.results distortion



#### PRINTERS
function Base.show(io::IO, sim::Simulation)
    println(io, "\nSimulation{$(typeof(sim.algo))} with q=", sim.q,
        ", n=", sim.n, ", R=", 
        round(1-sim.m/sim.n,digits=2),
         ", b=", sim.b,", niter=", sim.niter)
    return nothing
end
function runtime_str(sim::Union{Simulation,Vector{Simulation{T}}}; 
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
naive_compression_inv(R::Real) = 0.5*(1-R)
naive_compression(D::Real) = 1 - 2*D


#### PLOTTING
import Plots: plot!, plot, histogram, bar

function Plots.plot!(pl::Plots.Plot, sims::Vector{Simulation{T}}; 
        allpoints::Bool=false,
        label::String="Experimental data", 
        convergedonly::Bool=false, msw::Number=0.5, plotkw...) where {T<:LossyAlgo}
    dist = distortion.(sims, convergedonly)
    npoints = length.(dist)
    r = [rate(sim) for sim in sims]
    length(sims)==0 && return pl
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
            Plots.scatter!(pl, r, dist_avg, label=label, size=(300,200); 
                plotkw...)
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
function Plots.plot!(pl::Plots.Plot, sim::Simulation; kwargs...)
    plot!(pl, [sim]; kwargs...)
end

function Plots.plot!(pl::Plots.Plot, 
    sims_vec::Vector{Vector{Simulation{T}}}; 
    labels::Vector{String}=["GF($(s[1].q))" for s in sims_vec],
    kw...) where {T<:LossyAlgo}

    markers = [:circle, :diamond, :rect, :utriangle,:star6]

    for i in eachindex(sims_vec)
        plot!(pl, sims_vec[i]; label=labels[i], 
            markershape=markers[mod1(i,length(markers))], kw...)
    end
    return pl
end


function Plots.plot(sims::Union{Simulation{T},Vector{Simulation{T}},Vector{Vector{Simulation{T}}}}; 
        size=(500,500), kwargs...) where {T<:LossyAlgo}
    d = LinRange(0,0.5,100)
    r = LinRange(0, 1, 100)
    pl = Plots.plot(rdb.(d), d, label="Information bound")
    Plots.plot!(pl, r, naive_compression_inv.(r), label="Naive compression")
    return plot!(pl, sims; size=size, kwargs...)
end

function plot_rdb(; rs=true, kw...) 
    pl = Plots.plot(Simulation{BP}[]; kw...)
    if rs
        Rm = 0.01:0.01:0.99
        Dm = [0.4546474073128681,0.43565049932361133,0.42092765046912317,0.40839439222208573,0.3972457420215144,0.3870702443196218,0.3776242651788484,0.36874955806591975,0.3603365352601175,0.3523056385673838,0.34459697668198624,0.33716414626734653,0.3299703418625086,0.3229857971761951,0.31618604029437997,0.30955066735206993,0.30306245811343413,0.2967067238049428,0.2904708168067388,0.2843437556958753,0.2783159341345032,0.27237889177802405,0.26652513178257226,0.2607479738209666,0.2550414345013817,0.24940012917780852,0.2438191906350483,0.2382942012118009,0.2328211357178961,0.22739631308980668,0.22201635516998064,0.21667815133009227,0.211378827914124,0.20688352899753953,0.20278317659003975,0.19868817793711935,0.1945975994415784,0.19051056552023182,0.18642625421128822,0.18234389327612,0.17826275673045133,0.17418216174918932,0.17010146589683228,0.16602006464188257,0.16193738911916555,0.15785290410864755,0.1537661062033548,0.14967652214243293,0.1455837072883871,0.14148724423013948,0.137998366107847,0.13448740638286893,0.13095421069018287,0.12739865696194796,0.12382065470933767,0.12022014422138577,0.11659709568295457,0.1129515082152911,0.10928340884389087,0.10559285139949881,0.10231362606715677,0.0989986151678956,0.09564794614910743,0.0922617930523803,0.0888403753926773,0.08538395674018595,0.0820151560863665,0.0788488704133426,0.07563678673548502,0.07237911796887381,0.06907615625262697,0.06591319349557895,0.06283584277258297,0.05970319879491964,0.056515457092013055,0.053562785457060835,0.05054512262289507,0.04752198628196386,0.04463641809759822,0.04167585667427448,0.0388827836051418,0.03604608454052277,0.033304960430999175,0.03060809220840449,0.027952952887639493,0.025359146185103376,0.022832861932172044,0.020370784947590925,0.017979055383501308,0.015666129177374533,0.013449978144655372,0.0113201258591068,0.009302254807509736,0.007397829094439923,0.0056264302423451595,0.004010648891537405,0.002577264143732827,0.0013672330158682189,0.0004509986577261871]
        plot!(pl, Rm, Dm, lc=:gray, ls=:dash, label="RS")
    end
    pl
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


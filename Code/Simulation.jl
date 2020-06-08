struct Simulation
    q::Int
    n::Int
    R::Float64
    b::Int
    navg::Int
    maxiter::Int
    converged::BitArray{1}
    parity::Vector{Int}
    distortions::Vector{Float64}
    iterations::Vector{Int}
    runtimes::Vector{Float64}
    maxdiff::Vector{Vector{Float64}}
    codeword::Vector{BitArray{1}}
    maxchange::Vector{Vector{Float64}}
    trials::Vector{Int}
end

# Run simulation and store
function Simulation(
    algo::Union{BP,MS},
    q::Int,
    n::Int,                             # Number of nodes, fixed
    m::Int;                             # m=n-k, the number of rows in the system
    L::Real = 1,                        # Factor in the fields expression
    navg = 10,                          # Number of runs for each value of m
    maxiter = Int(1e3),                 # Max number of iteration for each run of BP
    convergence = :decvars,             # Convergence criterion: can be either :decvars or :messages
    nmin = 100,                         # If :decvars is chosen
    tol = 1e-7,                         # If :messages is chosen
    b = 0,                              # Number of factors to be removed
    gamma = 0,                          # Reinforcement parameter
    alpha = 0,                          # Damping parameter
    Tmax = 1,                           # Number of trials
    samegraph = false,                  # If true, only 1 graph is extracted and all |navg| simulations are run on it
    samevector = false,                 # If true, only 1 vector is extracted and all |navg| simulations are run on it
    randseed = 100,                     # For reproducibility
    verbose = true)

    converged = falses(navg)
    parity = zeros(Int,navg)
    runtimes = zeros(navg)
    distortions = 0.5*ones(navg)
    iterations = zeros(Int,navg)
    Y = [zeros(Int,n) for _ in 1:navg]
    H = [Array{Int,2}(undef,m,n) for _ in 1:navg]
    maxdiff = [zeros(maxiter) for i in 1:navg]
    codeword = [falses(maxiter) for i in 1:navg]
    maxchange = [fill(-Inf, maxiter) for i in 1:navg]
    trials = zeros(Int, navg)

    FG = ldpc_graph(q, n, m+b, verbose=verbose, randseed=randseed)
    breduction!(FG, b, randseed=randseed)
    y = rand(MersenneTwister(randseed), 0:q-1, n)

    t = @timed begin
        verbose && println()
        for it in 1:navg
            if !samevector
                y .= rand(MersenneTwister(randseed+it), 0:q-1, n)
            end
            if !samegraph
                FG = ldpc_graph(q, n, m+b, verbose=verbose,
                    randseed=randseed+it)
                breduction!(FG, b, randseed=randseed+it)
            end
            FG.fields .= extfields(q,y,algo,L, randseed=randseed+it*Tmax)
            (res,iterations[it],trials[it]), runtimes[it] = @timed bp!(FG, algo, y,
                maxiter, convergence, nmin, tol, gamma, alpha, Tmax, randseed+it*Tmax,
                maxdiff[it], codeword[it], maxchange[it], verbose=false)

            if res == :converged
                converged[it] = true
            end
            parity[it] = sum(paritycheck(FG))
            if parity[it] == 0
                distortions[it] = hd(guesses(FG),y)/(n*log2(q))
            end
            res_string = res==:converged ? "C" : "U"

            if convergence==:messages
                verbose && println("Run ", it, " of ",navg,": ",res_string,
                    " after ", iterations[it], " iters. ",
                    "Parity ", parity[it],
                    ". Max change ", round.(maxchange[it][iterations[it]], sigdigits=3),
                    ". Trials ", trials[it]#=,
                    ". Seed ", randseed=#)
            else
                verbose && println("Run ", @sprintf("%3d", it),
                    " of ",navg,": ",
                    res_string, " after ", @sprintf("%3d", iterations[it]),
                    " iters. ", "Parity ", @sprintf("%2d", parity[it]),
                    ". Trials ", trials[it]#=,
                    ". Seed ", randseed=#)
            end
            samegraph && refresh!(FG)   # Reset messages
        end
    end
    totaltime = t[2]
    R = 1-m/n
    return Simulation(q, n, R, b, navg, maxiter, converged, parity, distortions, iterations,
        runtimes, maxdiff, codeword, maxchange, trials)
end

import Base.show
function show(io::IO, sim::Simulation)
    println(io, "Simulation with n=", sim.n, ", R=", round(sim.R,digits=2),
     ", b=", sim.b,", navg=", sim.navg)
end

function plotdist(R=Float64[], D=Float64[]; backend=:unicode, errorbars=false,
     sigma=0, linename="Simulation results")

    d = LinRange(0,0.5,100)
    r = LinRange(0, 1, 100)
    if backend==:pyplot
        fig1 = PyPlot.figure("Rate-distortion bound")
        PyPlot.plot(rdb.(d),d, label="Lower bound")
        PyPlot.plot(r, (1 .- r)/2, label="Naive compression")
        if errorbars
            if length(R) != 0
                PyPlot.errorbar(R, D, sigma, fmt="o",
                                    label=linename, ms=4, capsize=4)
            end
        elseif length(R) != 0
            PyPlot.plot(R, D, "o", ms=4, label=linename)
        end
        plt.:xlabel("Rate")
        plt.:ylabel("Distortion")
        plt.:legend()
        return fig1
    elseif backend==:unicode
        myplt = UnicodePlots.lineplot(rdb.(d),d, name="RDB", xlabel = "R", ylabel="D",
            canvas = DotCanvas, width=60, height = 20)
        UnicodePlots.lineplot!(myplt, r, (1 .- r)/2, name="Naive compression")
        length(R) != 0 && UnicodePlots.scatterplot!(myplt, R, D, name=linename)
        return myplt
    else
        error("Backend $backend not supported")
    end
end

function plot(sims::Vector{Simulation}; backend=:unicode, plotparity=true,
        errorbars=false, title="Mean distortion")
    R = [sim.R for sim in sims]
    dist = [mean(sim.distortions[sim.parity .== 0]) for sim in sims]
    sigma = [std(sim.distortions[sim.parity .== 0])/sqrt(length(sim.distortions[sim.parity .== 0])) for sim in sims]
    dist_tot = [mean(sim.distortions) for sim in sims]
    sigma_tot = [std(sim.distortions)/sqrt(length(sim.distortions)) for sim in sims]

    if backend == :pyplot
        fig1 = plotdist(R, dist_tot, backend=:pyplot, linename="Total distortion",
                        errorbars=errorbars, sigma=sigma)
        ax = fig1.axes[1]
        ax.set_title(title)
        if plotparity
            if errorbars
                ax.errorbar(R, dist, sigma, fmt="o", ms=4, capsize=4, label="Parity fulfilled")
            else
                ax.plot(R, dist, "o", ms=4, label="Parity fulfilled")
            end
        end
        plt.:legend()
        return fig1
    elseif backend == :unicode
        println()
        myplt = plotdist(R, dist_tot, backend=:unicode, linename="Total distortion")
        plotparity && scatterplot!(myplt, R, dist, name="Parity fulfilled")
        title!(myplt, title)
        return myplt
    end
end

function plott(sim::Simulation; backend=:unicode, plotparity=true,
        errorbars=false, title="Mean distortion")
    return plot([sim], backend=backend, plotparity=plotparity,
                errorbars=errorbars, title=title)
end

function plot(simsvec::Vector{Vector{Simulation}}; backend=:pyplot,
        errorbars = false, title="Mean distortion")
    if backend == :pyplot
        PyPlot.close("all")
        fig1 = plotdist(backend=:pyplot)
        ax = fig1.axes[1]
        ax.set_title(title)
        for sims in simsvec
            R = [sim.R for sim in sims]
            dist_tot = [mean(sim.distortions) for sim in sims]
            if errorbars
                sigma = [std(sim.distortions)/sqrt(length(sim.distortions)) for sim in sims]
                ax.errorbar(R, dist_tot, sigma, fmt="o",
                    ms=4, capsize=4, label="GF($(sims[1].q))")
            else
                ax.plot(R, dist_tot, "o", ms=4, label="GF($(sims[1].q))")
            end
        end
        plt.:legend()
        return fig1
    elseif backend == :unicode
        myplt = plotdist(backend=:unicode)
        for sims in simsvec
            R = [sim.R for sim in sims]
            dist_tot = [mean(sim.distortions) for sim in sims]
            scatterplot!(myplt, R, dist_tot, name="GF($(sims[1].q))")
        end
        title!(myplt, title)
        return myplt
    end
end

function trials_hist(sims::Vector{Simulation}; backend=:unicode,
    title="Trials for convergence")

    T = [sim.iterations[sim.converged] for sim in sims]
    tmax = maximum(maximum(T))

    if backend == :unicode
        for j in eachindex(T)
            println()
            display(UnicodePlots.barplot(string.(1:maximum(T[j])), StatsBase.counts(T[j]),
                symb="\u2588", title=title * ". R = $(round(sims[j].R, digits=2))"))
        end
   elseif backend == :pyplot
       PyPlot.close("all")
       for j in eachindex(T)
           PyPlot.subplot(2, length(sims)÷2, j)
           PyPlot.hist(T[j])
           ax = PyPlot.gca()
           ax.set_title("R = $(round(sims[j].R, digits=2))")
       end
       fig = gcf()
       fig.suptitle(title)
       PyPlot.tight_layout()
   end
end

function iters_hist(sims::Vector{Simulation}; backend=:unicode,
    title="Iterations for convergence")

    T = [sim.iterations[sim.converged] for sim in sims]
    tmax = maximum.(T)
    tmin = minimum.(T)

    if backend == :unicode
        for j in eachindex(T)
            println()
            # display(UnicodePlots.barplot(string.(tmin[j]:tmax[j]), StatsBase.counts(T[j]),
            #     symb="\u2588", title=title * ". R = $(round(sims[j].R, digits=2))"))
            display(UnicodePlots.histogram(T[j], nbins=10,
                symb="\u2588", title=title * ". R = $(round(sims[j].R, digits=2))"))
        end
   elseif backend == :pyplot
       PyPlot.close("all")
       for j in eachindex(T)
           PyPlot.subplot(2, length(sims)÷2, j)
           PyPlot.hist(T[j], bins=tmax)
           ax = PyPlot.gca()
           ax.set_title("R = $(round(sims[j].R, digits=2))")
           fig = gcf()
           fig.suptitle(title)
       end
       PyPlot.tight_layout()
   end
end

# Print results
import Base.print
function print(io::IO, sim::Simulation; options=:short)
    println(io, "Simulation with n = ", sim.n, ", R = ", round(sim.R,digits=2),
    ", b = ", sim.b, ", maxiter = ", sim.maxiter)
    println("Average distortion for instances that fulfilled parity: ",
        round(mean(sim.distortions[sim.parity .== 0]),digits=2))
    println("Average distortion for all instances: ",
            round(mean(sim.distortions),digits=2))
    println("Iterations for converged instances: ", mean_sd_string(sim.iterations[sim.converged .== true]))

    M = fill("",3,3)
    M[2,2] = string(sum(sim.converged.*(sim.parity.==0)))
    M[2,3] = string(sum(sim.converged.*(sim.parity.!=0)))
    M[3,2] = string(sum(.!sim.converged.*(sim.parity.==0)))
    M[3,3] = string(sum(.!sim.converged.*(sim.parity.!=0)))
    M[2,1] = string(sum(sim.converged))
    M[3,1] = string(sum(.!sim.converged))
    M[1,2] = string(sum(sim.parity.==0))
    M[1,3] = string(sum(sim.parity.!=0))
    M[1,1] = string(length(sim.converged))

    h = Highlighter(
        f = (data,i,j) -> (i,j) in [(2,4)] && data[i,j] != "0";
        crayon = crayon"bg:red"
    )

    data = hcat(["Total"; "Convergence Y"; "Convergence N"], M)
    time_tot = Int(round(sum(sim.runtimes)))
    time_hour = div(time_tot,60*60)
    mm = mod(time_tot, 60*60)
    time_min = div(mm, 60)
    time_sec = mod(mm, 60)
    avg_min = Int(fld(mean(sim.runtimes),60))
    avg_sec = Int(round(mod(mean(sim.runtimes),60)))
    println(io, "Runtime: ", time_hour, "h ", time_min, "m ", time_sec,
        "s. Average runtime per instance: ", avg_min, "m ", avg_sec, "s")
    pretty_table(io, data, ["" "Total" "Parity Y" "Parity N"], alignment=:c,
        hlines = [1,2], highlighters = h)
    println(io)

    if options==:full
        println("L = ", sim.L)
        # println("Number of edges = ", sim.nedges)
        # println("Lambda = ", sim.lambda)
        # println("Rho = ", sim.rho)
        if sim.convergence==:decvars
            println("Convergence criterion: decisional variables")
            println("Minimum number of consecutive iterations with no changes: ", sim.nmin)
        elseif sim.convergence==:messages
            println("Convergence criterion: messages")
            println("Tolerance for messages to be considered equal: ", sim.tol)
        end
        println("b (number of factors removed to improve convergence) = ", sim.b)
        println("gamma (soft decimation factor) = ", sim.gamma)
        if sim.samegraph
            println("All $(sim.navg) simulations were run on the same graph")
        else
            println("A new graph was created for each input vector")
        end
        if sim.samevector
            println("All $(sim.navg) simulations were run on the same vector")
        else
            println("A new input vector was created each time")
        end
    elseif options != :short
        println("Option $option not available")
    end
end

function print(io::IO, sims::Vector{Simulation}; options=:short)
    for sim in sims
        print(io, sim, options=options)
    end
end

# Returns mean distortion
# If parityonly is set to false, a distortion 0.5 is added for each instance
# that didn't converge
function meandist(sim::Simulation; convergedonly::Bool=true)
    if convergedonly
        idx = (sim.converged .== true)
    else
        idx = eachindex(sim.distortions)
    end
    return mean(sim.distortions[idx])
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

function mean_sd_string(v::AbstractVector, digits::Int=2)
    m = mean(v)
    s = std(v)
    output = string(round(m,digits=digits), " ± ", round(s,digits=digits))
    return output
end

# function plot(sim::Simulation; options=:short, backend=:unicode, errorbars=false)
#     dist = mean(sim.distortions[sim.parity .== 0])
#     dist_tot = mean(sim.distortions)
#     if backend==:pyplot
#         sigma = std(sim.distortions[sim.parity .== 0])
#         fig1 = plotdist([sim.R], [dist], backend=:pyplot, errorbars=errorbars,
#             error = sigma./sqrt.(length(sigma)))
#         ax = fig1.axes[1]
#         ax.set_title("Mean disortionsim = Simulation(MS(), 2, 10, 5)\n n = $(sim.n)")
#         if options==:full
#             fig2 = PyPlot.figure("Detailed plots")
#             PyPlot.subplot(311)
#             ax1 = PyPlot.gca()
#             x = 1:sim.navg
#             ax1.plot(x, sim.iterations, "bo")
#             ax1.axhline(sim.maxiter, c="b", ls="--", lw=0.8)
#             ax1.set_ylim((0,sim.maxiter+100))
#             ax1.set_xticks(x)
#             ax1.set_xlabel("Index")
#             ax1.set_ylabel("Iterations", color="b")
#             ax2 = ax1.twinx()
#             ax2.plot(x, sim.parity, "ro")
#             ax2.set_yticks(0:maximum(sim.parity)[1])
#             ax2.axhline(0,c="r", ls="--", lw=0.8)
#             ax2.set_ylabel("Unfulfilled checks", color="r")
#             ax1.set_title("Number of iterations and unfulfilled parity checks")
#             PyPlot.tight_layout()
#
#             PyPlot.subplot(312)
#             ax1 = PyPlot.gca()
#             x = 1:sim.navg
#             ax1.plot(x, sim.distortions, "go")
#             ax1.axhline(1/2,c="g", ls="--", lw=0.8)
#             ax1.set_ylim((0,1))
#             ax1.set_xticks(x)
#             ax1.set_xlabel("Index")
#             ax1.set_ylabel("Distortion", color="g")
#             ax2 = ax1.twinx()
#             ax2.plot(x, sim.parity, "ro")
#             ax2.set_yticks(0:maximum(sim.parity)[1])
#             ax2.axhline(0,c="r", ls="--", lw=0.8)
#             ax2.set_ylabel("Unfulfilled checks", color="r")
#             ax1.set_title("Distortions and unfulfilled parity checks")
#             PyPlot.tight_layout()
#
#             PyPlot.subplot(313)
#             ax1 = PyPlot.gca()
#             x = 1:sim.navg
#             ax1.plot(x, sim.distortions, "go")
#             ax1.axhline(1/2,c="g", ls="--", lw=0.8)
#             ax1.set_ylim((0,1))
#             ax1.set_xticks(x)
#             ax1.set_xlabel("Index")
#             ax1.set_ylabel("Distortion", color="g")
#             ax2 = ax1.twinx()
#             ax2.plot(x, sim.iterations, "bo")
#             ax2.axhline(sim.maxiter,c="b", ls="--", lw=0.8)
#             ax2.set_ylim((0,sim.maxiter+100))
#             ax2.set_ylabel("Iterations", color="b")
#             ax1.set_title("Distortions and Number of iterations")
#             PyPlot.tight_layout()
#             return fig1, fig2
#         end
#         return fig1
#     elseif backend==:unicode
#         myplt = plotdist([sim.R], [dist], backend=:unicode, linename="Parity fulfilled")
#         lineplot!(myplt, [sim.R, sim.R], [dist_tot, dist_tot], name="Total")
#         title!(myplt, "Mean distortion \n n = $(sim.n)")
#         return myplt
#     end
# end

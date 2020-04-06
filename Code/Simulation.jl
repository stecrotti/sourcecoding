struct Simulation
    n::Int
    m::Int
    navg::Int
    converged::BitArray{1}
    parity::Vector{Int}
    distortions::Vector{Float64}
    iterations::Vector{Int}
    runtimes::Vector{Float64}
    totaltime::Float64
    maxiter::Int
    L::Real
    nedges::Int
    lambda::Vector{Float64}
    rho::Vector{Float64}
    convergence::Symbol
    nmin::Int
    tol::Float64
    b::Int
    gamma::Float64
    samegraph::Bool
    samevector::Bool
    y::Vector{Vector{Int}}
    H::Vector{Array{Int,2}}
end

# Run simulation and store
function Simulation(
    algo::Union{BP,MS}, q::Int,
    n::Int,                             # Number of nodes, fixed
    m::Int,                             # m=n-k, the number of rows in the system
    L::Real,                            # Factor in the fields expression
    nedges::Int,                        #
    lambda::Vector{Float64},            # Parameters for graph construction
    rho::Vector{Float64};               #
    navg = 10,                          # Number of runs for each value of m
    maxiter = Int(1e3),                 # Max number of iteration for each run of BP
    convergence = :decvars,             # Convergence criterion: can be either :decvars or :messages
    nmin = 100,                         # If :decvars is chosen
    tol = 1e-7,                         # If :messages is chosen
    b = 0,                              # Number of factors to be removed
    gamma = 0,                          # Reinforcement parameter
    samegraph = false,                  # If true, only 1 graph is extracted and all |navg| simulations are run on it
    samevector = false,                 # If true, only 1 vector is extracted and all |navg| simulations are run on it
    verbose = false)

    converged = falses(navg)
    parity = zeros(Int,navg)
    runtimes = zeros(navg)
    rawdistortion = zeros(navg)
    iterations = zeros(Int,navg)
    Y = [zeros(Int,n) for _ in 1:navg]
    H = [Array{Int,2}(undef,m,n) for _ in 1:navg]

    verbose && println("----------- Simulation starting -----------")
    if samegraph
        FG = ldpc_graph(q, n, m, nedges, lambda, rho, verbose=verbose)
        # b-reduction
        for _ in 1:b
            deletefactor!(FG)
            m -= 1
        end
        y = rand(0:q-1, n)
    end
    t = @timed begin
        for it in 1:navg
            if !samegraph
                FG = ldpc_graph(q, n, m, nedges, lambda, rho, verbose=verbose)
                # b-reduction
                for _ in 1:b
                    deletefactor!(FG)
                    m -= 1
                end
                y = rand(0:q-1, n)

            end
            Y[it] .= y
            H[it] .= adjmat(FG)
            FG.fields .= extfields(q,y,algo,L)
            if convergence == :decvars
                ((res,iters), runtimes[it]) = @timed bp!(FG, algo, maxiter=maxiter, gamma=gamma, nmin=nmin)
            elseif convergence == :messages
                ((res,iters), runtimes[it]) = @timed bp_msg!(FG, algo, maxiter=maxiter, gamma=gamma, tol=tol)
            else
                error("Field 'convergence' must be either :decvars or :messages")
            end
            res != :unconverged && (converged[it] = true)
            parity[it] = sum(paritycheck(FG))
            rawdistortion[it] = hd(guesses(FG),y)/n
            iterations[it] = iters
                        if verbose && isinteger(10*it/navg)
                println("Finished ",Int(it/navg*100), "%")
            end
            samegraph && refresh!(FG)   # Reset messages
        end
    end
    totaltime = t[2]
    return Simulation(n, m, navg, converged, parity, rawdistortion, iterations,
        runtimes, totaltime, maxiter, L, nedges, lambda, rho, convergence, nmin,
        tol, b, gamma, samegraph, samevector, Y, H)
end

import Base.show
function show(io::IO, sim::Simulation)
    print(io, sim, options=:short)
end


import PyPlot.plot
function plot(sim::Simulation; options=:short)
    d = LinRange(0.001,0.5-0.001,100)
    r = LinRange(0, 1, 100)
    fig1 = PyPlot.figure("Rate-distortion bound")
    PyPlot.plot(rdb.(d),d);
    PyPlot.plot(r, (1 .- r)/2)

    R = 1 - sim.m/sim.n
    dist = meandist(sim)
    PyPlot.plot(R, dist, "o", ms=5)
    plt.:xlabel("Rate")
    plt.:ylabel("Distortion")
    plt.:legend(["Lower bound", "Random compression"])
    plt.:title("Mean disortion for instances that fulfill parity \n n = $(sim.n)")

    if options==:full
        fig2 = PyPlot.figure("Detailed plots")
        PyPlot.subplot(311)
        ax1 = PyPlot.gca()
        x = 1:sim.navg
        ax1.plot(x, sim.iterations, "bo")
        ax1.axhline(sim.maxiter,c="b", ls="--", lw=0.8)
        ax1.set_ylim((0,sim.maxiter+100))
        ax1.set_xticks(x)
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Iterations", color="b")
        ax2 = ax1.twinx()
        ax2.plot(x, sim.parity, "ro")
        ax2.set_yticks(0:maximum(sim.parity)[1])
        ax2.axhline(0,c="r", ls="--", lw=0.8)
        ax2.set_ylabel("Unfulfilled checks", color="r")
        ax1.set_title("Number of iterations and unfulfilled parity checks")
        PyPlot.tight_layout()

        PyPlot.subplot(312)
        ax1 = PyPlot.gca()
        x = 1:sim.navg
        ax1.plot(x, sim.distortions, "go")
        ax1.axhline(1/2,c="g", ls="--", lw=0.8)
        ax1.set_ylim((0,1))
        ax1.set_xticks(x)
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Distortion", color="g")
        ax2 = ax1.twinx()
        ax2.plot(x, sim.parity, "ro")
        ax2.set_yticks(0:maximum(sim.parity)[1])
        ax2.axhline(0,c="r", ls="--", lw=0.8)
        ax2.set_ylabel("Unfulfilled checks", color="r")
        ax1.set_title("Distortions and unfulfilled parity checks")
        PyPlot.tight_layout()

        PyPlot.subplot(313)
        ax1 = PyPlot.gca()
        x = 1:sim.navg
        ax1.plot(x, sim.distortions, "go")
        ax1.axhline(1/2,c="g", ls="--", lw=0.8)
        ax1.set_ylim((0,1))
        ax1.set_xticks(x)
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Distortion", color="g")
        ax2 = ax1.twinx()
        ax2.plot(x, sim.iterations, "bo")
        ax2.axhline(sim.maxiter,c="b", ls="--", lw=0.8)
        ax2.set_ylim((0,sim.maxiter+100))
        ax2.set_ylabel("Iterations", color="b")
        ax1.set_title("Distortions and Number of iterations")
        PyPlot.tight_layout()
        return fig1, fig2
    end
    return fig1
end

function plot(sims::Vector{Simulation})
    for sim in sims
        plot(sim)
    end
end

# Print results
import Base.print
function print(io::IO, sim::Simulation; options=:short)
    println(io)
    R = 1 - sim.m/sim.n
    println(io, "Simulation with n = ", sim.n, ", average over ",
        length(sim.converged), " trials")
    println("Average distortion for instances that fulfill parity: ", round(meandist(sim),digits=2))
    totaltime_min = Int(fld(sim.totaltime,60))
    totaltime_sec = Int(round(mod(sim.totaltime,60)))
    println(io, "Total elapsed time: ", totaltime_min, "m ",
        totaltime_sec, "s\n")

    println(io, "\t    k = $(sim.n-sim.m)  /  R = ", round(R, digits=2), "\n")

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
    time_min = Int(fld(sum(sim.runtimes),60))
    time_sec = Int(round(mod(sum(sim.runtimes),60)))
    avg_min = Int(fld(mean(sim.runtimes),60))
    avg_sec = Int(round(mod(mean(sim.runtimes),60)))
    println(io, "Runtime: ", time_min, "m ", time_sec, "s. Average runtime per instance: ",
        avg_min, "m ", avg_sec, "s")
    pretty_table(io, data, ["" "Total" "Parity Y" "Parity N"], alignment=:c,
        hlines = [1,2], highlighters = h)

    if options==:full
        println("L = ", sim.L)
        println("Number of edges = ", sim.nedges)
        println("Lambda = ", sim.lambda)
        println("Rho = ", sim.rho)
        if sim.convergence==:decvars
            println("Convergence criterion: decisional variables")
            println("Minimum number of consecutive iterations with no changes: ", sim.nmin)
        else
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
    else
        options != :short && println("Option $option not available")
    end
end


### Used internally
function meandist(sim::Simulation)
    return mean(sim.distortions[(sim.parity.==0) .& (sim.converged.==true)])
end

rdb(D) = 1-(-D*log2(D)-(1-D)*log2(1-D))

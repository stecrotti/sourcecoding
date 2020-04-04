struct Simulation
    n::Int
    m::Int
    converged::BitArray{1}
    parity::Vector{Int}
    rawdistortion::Vector{Float64}
    iterations::Vector{Int}
    runtimes::Vector{Float64}
    totaltime::Float64
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
    verbose = false)

    converged = falses(navg)
    parity = zeros(Int,navg)
    runtimes = zeros(navg)
    rawdistortion = zeros(navg)
    iterations = zeros(Int,navg)

    println("----------- Simulation starting -----------")

    t = @timed begin
        for it in 1:navg
            FG = ldpc_graph(q, n, m, nedges, lambda, rho, verbose=verbose)
            # b-reduction
            for _ in 1:b
                deletefactor!(FG)
                m -= 1
            end
            y = rand(0:q-1, n)
            FG.fields .= extfields(q,y,algo,L)
            if convergence == :decvars
                ((res,iters), runtimes[it]) = @timed bp!(FG, algo, maxiter=maxiter, gamma=gamma, nmin=nmin, verbose=verbose)
            elseif convergence == :messages
                ((res,iters), runtimes[it]) = @timed bp_msg!(FG, algo, maxiter=maxiter, gamma=gamma, tol=tol, verbose=verbose)
            else
                error("Field 'convergence' must be either :decvars or :messages")
            end
            res != :unconverged && (converged[it] = true)
            parity[it] = sum(paritycheck(FG))
            rawdistortion[it] = hd(guesses(FG),y)/n
            # refresh!(FG)
            yield()
            isinteger(10*it/maxiter) && println("Finished ", Int(it/maxiter*100), "%")
        end

    end
    totaltime = t[2]
    return Simulation(n,m,converged, parity, rawdistortion, runtimes, totaltime)
end

import Base.show
function show(io::IO, sim::Simulation)
    print(io, sim)
end

# Plot distortions
import PyPlot.plot
function plot(sim::Simulation)
    d = LinRange(0.001,0.5-0.001,100)
    r = LinRange(0, 1, 100)
    PyPlot.plot(rdb.(d),d);
    PyPlot.plot(r, (1 .- r)/2)

    R = 1 .- sim.m/sim.n
    dist = distortions(sim)
    for j in 1:length(sim.m)
        PyPlot.plot(R[j], dist[j], "o", ms=5)
    end
    plt.:xlabel("Rate")
    plt.:ylabel("Distortion")
    plt.:legend(["Lower bound", "Random compression"])
    plt.:title("Mean disortion for instances that fulfill parity \n n = $(sim.n)")
end

# Print results
import Base.print
function print(io::IO, sim::Simulation)
    R = 1 .- sim.m/sim.n
    println(io, "Simulation with n = ", sim.n, ", average over ",
        length(sim.converged[1]), " trials")
    println(io, "k = ", sim.n .- sim.m)
    println(io, "R = ", round.(R, digits=2))
    totaltime_min = Int(fld(sim.totaltime,60))
    totaltime_sec = Int(round(mod(sim.totaltime,60)))
    println(io, "Total elapsed time: ", totaltime_min, "m ",
        totaltime_sec, "s\n")

    for j in 1:length(sim.m)
        println(io, "---------------------------------------------")
        println(io, "\t    k = $(sim.n-sim.m[j])  /  R = ", round(R[j], digits=2))
        println(io, "---------------------------------------------\n")

        M = fill("",3,3)
        M[2,2] = string(sum(sim.converged[j].*(sim.parity[j].==0)))
        M[2,3] = string(sum(sim.converged[j].*(sim.parity[j].!=0)))
        M[3,2] = string(sum(.!sim.converged[j].*(sim.parity[j].==0)))
        M[3,3] = string(sum(.!sim.converged[j].*(sim.parity[j].!=0)))
        M[2,1] = string(sum(sim.converged[j]))
        M[3,1] = string(sum(.!sim.converged[j]))
        M[1,2] = string(sum(sim.parity[j].==0))
        M[1,3] = string(sum(sim.parity[j].!=0))
        M[1,1] = string(length(sim.converged[1]))

        h = Highlighter(
            f = (data,i,j) -> (i,j) in [(2,4)] && data[i,j] != "0";
            crayon = crayon"red bold"
        )

        data = hcat(["Total"; "Convergence Y"; "Convergence N"], M)
        time_min = Int(fld(sum(sim.runtimes[j]),60))
        time_sec = Int(round(mod(sum(sim.runtimes[j]),60)))
        avg_min = Int(fld(mean(sim.runtimes[j]),60))
        avg_sec = Int(round(mod(mean(sim.runtimes[j]),60)))
        println(io, "Runtime: ", time_min, "m ", time_sec, "s. Average runtime per instance: ",
            avg_min, "m ", avg_sec, "s")
        println("Average distortion for instances that fulfill parity: ", round(distortions(sim)[j],digits=2))
        pretty_table(io, data, ["" "Total" "Parity Y" "Parity N"], alignment=:c,
            hlines = [1,2], highlighters = h)
    end
end


### Used internally
function distortions(sim::Simulation)
    return [mean(sim.rawdistortion[j][(sim.parity[j].==0) .& (sim.converged[j].==true)]) for j in 1:length(sim.m)]
end

rdb(D) = 1-(-D*log2(D)-(1-D)*log2(1-D))

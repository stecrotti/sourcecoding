include("./../headers.jl")
using JLD2

q = 2
gamma = 5e-3
n = Int(round(420*5/log2(q)))
R = collect(0.21:0.1:0.81) 
m = Int.(round.(n*(1 .- R)))
b = Int(round(n/100))*ones(Int, length(m))
maxiter = Int(3e2)
niter = 20
randseed = 1234
Tmax = 5

# Three algorithms
maxsum = MS(maxiter=maxiter, gamma=gamma, Tmax=Tmax)
simanneal = SA(mc_move=MetropSmallJumps(), nsamples=200, 
    betas=[Inf 0.1; Inf 1.0; Inf 10.0]);
optimalcycle = OptimalCycle();

sims_cycles = Vector{Simulation{OptimalCycle}}(undef, length(m))
sims_sa = Vector{Simulation{SA}}(undef, length(m))
sims_ms = Vector{Simulation{MS}}(undef, length(m))


for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sims_sa[j] = Simulation(q, n, m[j], simanneal, b=b[j], niter=niter,
        randseed=randseed, showprogress=true)
    sims_cycles[j] = Simulation(q, n, m[j], optimalcycle, b=0, niter=niter,
        randseed=randseed, showprogress=true)
    sims_ms[j] = Simulation(q, n, m[j], maxsum, niter=niter, b=b[j], 
        randseed=randseed, showprogress=true)
end

ms = 3 # marker size

pl = plot(sims_ms, label="Max-Sum", size=(600,400), legend=:outertopright, dpi=200, ms=ms, marker=:square)
plot!(pl, sims_cycles, label="Optimal cycle", ms=ms)
plot!(pl, sims_sa, label="Simulated annealing", ms=ms, marker=:diamond)

JLD2.@save "./method_comparison3.jld" sims_sa sims_cycles sims_ms

send_notification()

include("./../headers.jl")
using JLD2

q = 2
gamma = 5e-3
n = Int(round(420*4/log2(q)))
R = collect(0.21:0.1:0.81) 
m = Int.(round.(n*(1 .- R)))
maxiter = Int(5e2)
niter = 20
randseed = 12345
Tmax = 1

nbetas = Int(2e5)
beta2 = collect(LinRange(5e-1, 5e0, nbetas))
nsamples = 1

# Three algorithms
maxsum = MS(maxiter=maxiter, gamma=gamma, Tmax=Tmax)
simanneal = SA(MetropBasisCoeffs(), beta2, nsamples=nsamples, 
    init_state=fix_indep_from_src);
optimalcycle = OptimalCycle();

sims_cycles = Vector{Simulation{OptimalCycle}}(undef, length(m))
sims_sa = Vector{Simulation{SA}}(undef, length(m))
sims_ms = Vector{Simulation{MS}}(undef, length(m))


for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------") 
    sims_ms[j] = Simulation(q, n, m[j], maxsum, niter=niter, b=0, 
        randseed=randseed, showprogress=false)
    sims_cycles[j] = Simulation(q, n, m[j], optimalcycle, niter=niter,
        randseed=randseed, showprogress=true)
    sims_sa[j] = Simulation(q, n, m[j], simanneal, niter=niter, 
        b=Int(round(n/10)), randseed=randseed, showprogress=false)
end

ms = 3 # marker size

pl = plot(sims_ms, label="Max-Sum", size=(600,400), legend=:outertopright, dpi=200, ms=ms, marker=:square)
plot!(pl, sims_cycles, label="Optimal cycle", ms=ms)
plot!(pl, sims_sa, label="Simulated annealing", ms=ms, marker=:diamond)

JLD2.@save "./method_comparison4.jld" pl

send_notification()

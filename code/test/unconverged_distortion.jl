include("./../headers.jl")
using JLD2

const q = 2
const gamma = 5e-3
const n = Int(round(420*4/log2(q)))
const R = collect(0.21:0.1:0.81) 
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/100))*ones(Int, length(m))
const maxiter = Int(2e2)
const niter = 20
const randseed = 1234
const Tmax = 5

fcns = [naive_compression_distortion, fix_indep_from_src, fix_indep_from_ms]
sims = [Vector{Simulation{MS}}(undef, length(m)) for _ in eachindex(fcns)]

for i in eachindex(sims)
    ms = MS(maxiter=maxiter, gamma=gamma, Tmax=Tmax, default_distortion=fcns[i])
    for j in 1:length(m)
        println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
        sim = Simulation(q, n, m[j], ms, niter=niter, b=b[j], randseed=randseed, 
            showprogress=true)
        print(sim)
        sims[i][j] = sim
    end
end

JLD2.@save "./unconverged_distortion1.jld" sims

lt = @layout [a;b;c]
p1 = plot(sims[1], allpoints=true, title="Naive compression")
p2 = plot(sims[2], allpoints=true, title="Fix dependent vars to source")
p3 = plot(sims[3], allpoints=true, title="Fix dependent vars to MS output")
plot(p1,p2,p3, layout=lt)
include("./../headers.jl")
# using JLD2

q = 2
gamma = 5e-3
n = Int(round(420*1/log2(q)))
R = collect(0.21:0.1:0.81) 
m = Int.(round.(n*(1 .- R)))
b = Int(round(n/100))*ones(Int, length(m))
maxiter = Int(2e2)
niter = 1
randseed = 1234
Tmax = 1

fcns = [naive_compression_distortion, fix_indep_from_src, fix_indep_from_ms]
sims = [Vector{Simulation{MS}}(undef, length(m)) for _ in eachindex(fcns)]

# for i in eachindex(sims)
for i in [3]
    ms = MS(maxiter=50, gamma=gamma, Tmax=1, default_distortion=fcns[i])
    for j in 1:length(m)
        println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
        sim = Simulation(q, n, m[j], ms, niter=niter, b=b[j], randseed=randseed, 
            showprogress=true)
        print(sim)
        sims[i][j] = sim
    end
end

s = sims[3]
# JLD2.@save "./unconverged_distortion2.jld" sims = s
# plot(s)
println("Finished!")

# lt = @layout [a;b;c]
# p1 = plot(sims[1], allpoints=true, title="Naive compression")
# p2 = plot(sims[2], allpoints=true, title="Fix dependent vars to source")
# p3 = plot(sims[3], allpoints=true, title="Fix dependent vars to MS output")
# plot(p1,p2,p3, layout=lt, size=(400,800))
include("./../headers.jl")
using JLD2

q = 8
gamma = 1e-3
n = round(Int, 420*4 ./log2(q))
R = collect(0.21:0.1:0.81) 
m = Int.(round.(n*(1 .- R)))
maxiter = Int(5e2)
navg = 20
randseed = 100
Tmax = 1

algo = MS(maxiter=maxiter, Tmax=Tmax, gamma=gamma)
sims_gfq = Vector{Simulation}(undef, length(m))
sims_arb = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sims_arb[j] = Simulation(q, n, m[j], algo, arbitrary_mult=true, b=1)
    print(sims_arb[j])
end

for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sims_gfq[j] = Simulation(q, n, m[j], algo, arbitrary_mult=false, b=1)
    print(sims_gfq[j])
end


date = Dates.format(now(), "yyyymmdd_HHMM")
@save "arbitr-"*date*".jld" sims_arb sims_gfq

fn = @__FILE__
send_notification(fn*" finished execution")
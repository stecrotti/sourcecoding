include("./../headers.jl")
using JLD2

qq = 2 .^ [1 3 5]
gamma = 1e-3
nn = Int.(round.(420*6 ./log2.(qq)))
R = collect(0.21:0.1:0.81) 
mm = [Int.(round.(n*(1 .- R))) for n in nn]
maxiter = [1000, 2000, 5000]
navg = 10
randseed = 100
Tmax = 2

sims_vec = [Vector{Simulation{MS}}(undef, length(m)) for m in mm]


for (i,q) in enumerate(qq)
    println("#### q=$q. Order $i of $(length(qq)) ####")
    for j in eachindex(mm[i])
        println("---------- Simulation $j of ", length(mm[i])," | R = ",R[j]," -----------")
        algo = MS(maxiter=maxiter[i], Tmax=Tmax, gamma=gamma)
        sims_vec[i][j] = Simulation(q, nn[i], mm[i][j], algo, niter=navg, b=1)
    end
end

fn = @__FILE__
send_notification(fn*" finished execution")

date = Dates.format(now(), "yyyymmdd_HHMM")
@save "gfq-"*date*".jld" sims_vec


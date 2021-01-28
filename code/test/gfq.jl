include("./../headers.jl")
using JLD2

qq = 2 .^ [1 4 6 8]
gamma = 1e-3
nn = Int.(round.(420*6 ./log2.(qq)))
R = collect(0.21:0.1:0.81) 
mm = [Int.(round.(n*(1 .- R))) for n in nn]
maxiter = Int(2e3)
navg = 20
randseed = 100
Tmax = 1

sims_vec = [Vector{Simulation{MS}}(undef, length(m)) for m in mm]
algo = MS(maxiter=maxiter, Tmax=Tmax, gamma=gamma)

for (i,q) in enumerate(qq)
    println("#### q=$q. Order $i of $(length(qq)) ####")
    for j in eachindex(mm[i])
        println("---------- Simulation $j of ", length(mm[i])," | R = ",R[j]," -----------")
        sims_vec[i][j] = Simulation(q, nn[i], mm[i][j], algo, niter=navg)
    end
end

fn = @__FILE__
send_notification(fn*" finished execution")

date = Dates.format(now(), "yyyymmdd_HHMM")
@save "gfq-"*date*".jld" sims_vec


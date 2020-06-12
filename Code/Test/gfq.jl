include("../headers.jl")

const R = collect(0.1:0.1:0.9)
const Rvals = length(R)
const gamma = 1e-2
const maxiter = Int(3e2)
const navg = 200
randseed = 109
const Tmax = 6

qrange = 2 .^(1:5)
qvals = length(qrange)

simsvec = [Vector{Simulation}(undef, Rvals) for q in qrange]

for (i,q) in enumerate(qrange)
    println("################################################")
    println("                 q = $q")
    println("################################################")
    n = 420*Int(7-log2(q))
    m = Int.(round.(n*(1 .- R)))
    b = Int.(round.(n/2*(-R.^2/14 .+ R/7 .+ 1/10)))

    for j in 1:length(m)
        println("---------- q=$q - Simulation $j of ", length(m)," | R = ",R[j]," -----------")
        simsvec[i][j] = Simulation(MS(), q, n, m[j],
            navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
            tol=1e-20, b=b[j], samegraph=false, samevector=false,
            randseed=randseed+navg*Tmax*j, verbose=true)
        print(simsvec[i][j])
    end
    print(simsvec[i])
    plot(simsvec[i])
end

plot(simsvec, title="Mean distortion\n n=$(simsvec[1][1].n), gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot, errorbars=true)
ax = gca()
ax.annotate("b=$(Int.(round.(420*Int(7-log2(2))/2*(-R.^2/14 .+ R/7 .+ 1/10))))", (0,0))
ax.annotate("maxiter=$(maxiter)", (0,0.05))
date = string(Dates.today())
savefig("../images/gfq-"*date, bbox_inches="tight")

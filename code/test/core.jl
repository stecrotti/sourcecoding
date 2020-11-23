using Plots
unicodeplots()
include("./../headers.jl")

q = 2
n = 10000
mvals = Int.(round.(n*(0.5:0.05:0.9)))
b = 1
navg = 100
seed = 12345

Nc = [zeros(navg) for _ in mvals]

for (j,m) in enumerate(mvals)
    for i = 1:navg  
        fg = ldpc_graph(q,n,m,randseed=seed+i)
        breduction!(fg, b)
        lr!(fg)
        Nc[j][i] = nvars(fg)/n
    end
    println("Finished m=", m)
end

Plots.plot(mvals./n, mean.(Nc))




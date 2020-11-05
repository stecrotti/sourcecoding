include("../headers.jl")

n = 100
const q = 2

for m in 10:10:100
    println("m = ", m)
    FG = ldpc_graph(q,n,m)
    println(polyn(FG))
end

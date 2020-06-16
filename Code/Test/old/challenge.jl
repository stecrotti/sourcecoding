include("../headers.jl")
using DelimitedFiles

# LOAD DATA
ncols = 1280
m = ncols÷2
M1 = readdlm("challenge_H.txt", String)
M2 = vec(M1)
length(M2)!=m && error("Wrong input size")
M3 = fill("0", m, m)
for i in eachindex(M2)
    M3[i,:] = split(M2[i],"")
end
M = parse.(Int, M3)
Mt = transpose(M)
Id = Matrix(1I, m, m)
Hinit = [Id Mt]

# MOVE TO GF(2^k)
k = 1
const q = 2^k
n = ncols÷k
H = gf2toq(Hinit, k)
println("Data loaded")

# BUILD FACTOR GRAPH
algo = MS()
L = 0.01
FG = FactorGraph(q, H)
y = zeros(Int, n)
y[end] = 1
fields = extfields(q, y, algo)
fields[end][1] = 1e3
FG.fields .= fields
print("Built ")
show(FG)

# RUN MAX-SUM
println("Max-Sum starting...")
maxiter = Int(1e3)
nmin = 300
gamma = 1e-4
randseed = 1996
verbose = true

(result, iters) = bp!(FG, algo, maxiter=maxiter, gamma=gamma, nmin=nmin,
    randseed = randseed, verbose=verbose)
println("Result: ", result, " after ", iters, " iterations")
yhat = guesses(FG)
weight = sum(yhat)

println("Weight = ", weight)

print("\a") # beep

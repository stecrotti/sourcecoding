#### A factor graph type thought for GF(q) belief propagation ####

struct FactorGraph
    q::Int                              # field order
    mult::OffsetArray{Int,2}            # multiplication matrix in GF(q)
    gfinv::Vector{Int}                  # inverses in GF(q). It has q-1 indices, since 0 has no inverse
    gfdiv::OffsetArray{Int,2}
    n::Int                              # number of variable nodes
    m::Int                              # number of factor nodes
    Vneigs::Vector{Vector{Int}}         # neighbors of variable nodes.
    Fneigs::Vector{Vector{Int}}         # neighbors of factor nodes (containing only factor nodes)
    fields::Vector{OffsetArray{Float64,1,Array{Float64,1}}}             # Prior probabilities in the form of external fields
    hfv::Vector{Vector{Int}}          # Non-zero elements of the parity check matrix
    mfv::Vector{Vector{OffsetArray{Float64,1,Array{Float64,1}}}}          # Messages from factor to variable with index starting at 0
end

# Basic constructor for empty object
function FactorGraph(q::Int, n::Int, m::Int)
    mult = OffsetArray(zeros(Int,q,q), 0:q-1, 0:q-1)
    gfinv = zeros(Int, q-1)
    gfdiv = OffsetArray(zeros(Int, q,q-1), 0:q-1,1:q-1)
    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    fields = [OffsetArray(fill(0.0, q), 0:q-1) for v in 1:n]
    hfv = [Int[] for f in 1:m]
    mfv = Vector{Vector{OffsetArray{Float64,1,Array{Float64,1}}}}()
    return FactorGraph(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
end

# Construct graph from adjacency matrix (for checks with simple examples)
function FactorGraph(q::Int, A::Array{Int,2},
    fields = [Fun(1e-3*randn(q)) for v in 1:size(A,2)])

    m,n = size(A)
    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    hfv = [Int[] for f in 1:m]
    mfv = [OffsetArray{Float64,1,Array{Float64,1}}[] for f in 1:m]

    for f in 1:m
        for v in 1:n
            if A[f,v]<0 || A[f,v]>q-1
                error("Entry of the adjacency matrix must be 0≤h_ij<q")
            elseif A[f,v] > 0
                push!(Fneigs[f], v)
                push!(Vneigs[v], f)
                push!(hfv[f], A[f,v])
                push!(mfv[f], OffsetArray(1/q*ones(q), 0:q-1))
            end
        end
    end
    mult, gfinv, gfdiv = gftables(q)
    return FactorGraph(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
end

function adjmat(fg::FactorGraph)
    A = zeros(Int,fg.m, fg.n)
    for f in 1:fg.m
        for (v_idx,v) in enumerate(fg.Fneigs[f])
            A[f,v] = fg.hfv[f][v_idx]
        end
    end
    A
end

dispstring(fg::FactorGraph) = "Factor Graph with n=$(fg.n) variables and m=$(fg.m) factors defined on GF($(fg.q))"

function Base.show(io::IO, fg::FactorGraph)
    println(io, dispstring(fg))
end


# Degree of variable node
function vardegree(fg::FactorGraph, v::Int)::Int
    v > fg.n && error("Variable $v is not in the graph")
    return length(fg.Vneigs[v])
end

# Degree of factor node
function factdegree(fg::FactorGraph, f::Int)::Int
    f > fg.m && error("Factor $f is not in the graph")
    return length(fg.Fneigs[f])
end

isvarleaf(fg::FactorGraph, v::Int)::Bool = vardegree(fg,v)==1
isfactleaf(fg::FactorGraph, f::Int)::Bool = factdegree(fg,f)==1

vardegrees(fg::FactorGraph) = [vardegree(fg,v) for v in eachindex(fg.Vneigs)]
factdegrees(fg::FactorGraph) = [factdegree(fg,f) for f in eachindex(fg.Fneigs)]

# deletes elements in vec that are equal to val
function deleteval!(vec::Vector{T}, val::T) where T
    deleteat!(vec, findall(x->x==val, vec))
end

function deletefactor!(fg::FactorGraph, f::Int=rand(filter(ff -> factdegree(fg,ff)!=0, 1:fg.m)))
    for v in fg.Fneigs[f]
        # delete factor from its neighbors' lists
        deleteval!(fg.Vneigs[v],f)
    end
    # delete messages from f
    fg.mfv[f] = OffsetArray{Float64,1,Array{Float64,1}}[]
    # delete factor f
    fg.Fneigs[f] = []
    return f
end

function deletevar!(fg::FactorGraph, v::Int=rand(filter(vv -> vardegree(fg,vv)!=0, 1:fg.n)))
    for f in eachindex(fg.Fneigs)
        # delete i from its neighbors' neighbor lists
        v_idx = findall(isequal(v), fg.Fneigs[f])
        deleteat!(fg.Fneigs[f],v_idx)
        # delete messages to v
        deleteat!(fg.mfv[f], v_idx)
        # delete weight on the adjacency matrix
        deleteat!(fg.hfv[f], v_idx)
    end
    # delete node v
    fg.Vneigs[v] = []
    return v
end

# Leaf removal
function lr!(fg::FactorGraph)
    flag = false    # raised if there are still leaves to remove
    for v in eachindex(fg.Vneigs)
        if vardegree(fg,v)==1
            deletefactor!(fg, fg.Vneigs[v][1])
            flag = true
        end
    end
    flag && lr!(fg)
    nothing
end

# Remove only 1 leaf
function onelr!(fg::FactorGraph, idx::Vector{Int}=randperm(fg.n))
    for v in idx
        if vardegree(fg,v)==1
            deletefactor!(fg, fg.Vneigs[v][1])
            # deletevar!(fg, v)
            return v
        end
    end
    return 0
end

# The following 2 are used to get the number of variables or factors left in
# the graph, which might be different from n,m i.e. the original ones

function nvars(fg::FactorGraph)   # number of variables in the core
    Nvars = 0
    for v in fg.Vneigs
        v != [] && (Nvars += 1)
    end
    return Nvars
end

function nfacts(fg::FactorGraph)    # number of hyperedges in the core
     Nfact = 0
     Fneigs = fg.Fneigs
     for f in Fneigs
         f != [] && (Nfact += 1)
     end
     return Nfact
end

function breduction!(fg::FactorGraph, b::Int; randseed::Int=0)
    randseed != 0 && Random.seed!(randseed)     # for reproducibility
    for _ in 1:b
        deletefactor!(fg)
    end
end

function polyn(fg::FactorGraph)
    fd = countmap(factdegrees(fg))    # degree => number of factors with that degree
    rho = zeros(maximum(keys(fd)))
    for j in keys(fd)
        rho[j] = j*fd[j]
    end
    rho ./= sum(rho)

    vd = countmap(vardegrees(fg))
    lambda = zeros(maximum(keys(vd)))
    for i in keys(vd)
        lambda[i] = i*vd[i]
    end
    lambda ./= sum(lambda)

    return lambda, rho
end

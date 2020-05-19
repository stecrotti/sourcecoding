struct FactorGraph
    q::Int                              # field order
    mult::OffsetArray{Int,2}            # multiplication matrix in GF(q)
    gfinv::Vector{Int}                  # inverses in GF(q). It has q-1 indices, since 0 has no inverse
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
    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    fields = [OffsetArray(fill(0.0, q), 0:q-1) for v in 1:n]
    hfv = [Int[] for f in 1:m]
    mfv = Vector{Vector{OffsetArray{Float64,1,Array{Float64,1}}}}()
    return FactorGraph(q, mult, gfinv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
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
    mult, gfinv = gftables(q)
    return FactorGraph(q, mult, gfinv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
end

function adjmat(FG::FactorGraph)
    A = zeros(Int,FG.m, FG.n)
    for f in 1:FG.m
        for (v_idx,v) in enumerate(FG.Fneigs[f])
            A[f,v] = FG.hfv[f][v_idx]
        end
    end
    A
end

function Base.show(io::IO, FG::FactorGraph)
    println(io, "Factor Graph with $(FG.n) variables and $(FG.m) factors defined on GF($(FG.q))")
end


# Degree of variable node
function vardegree(FG::FactorGraph, v::Int)::Int
    v > FG.n && error("Variable $v is not in the graph")
    return length(FG.Vneigs[v])
end

# Degree of factor node
function factdegree(FG::FactorGraph, f::Int)::Int
    f > FG.m && error("Factor $f is not in the graph")
    return length(FG.Fneigs[f])
end

vardegrees(FG::FactorGraph) = [vardegree(FG,v) for v in eachindex(FG.Vneigs)]
factdegrees(FG::FactorGraph) = [factdegree(FG,f) for f in eachindex(FG.Fneigs)]

# deletes elements in vec that are equal to val
function deleteval!(vec::Vector{T}, val::T) where T
    deleteat!(vec, findall(x->x==val, vec))
end

function deletefactor!(FG::FactorGraph, f::Int=rand(filter(ff -> factdegree(FG,ff)!=0, 1:FG.m)))
    for v in FG.Fneigs[f]
        # delete factor from its neighbors' lists
        deleteval!(FG.Vneigs[v],f)
    end
    # delete messages from f
    FG.mfv[f] = OffsetArray{Float64,1,Array{Float64,1}}[]
    # delete factor f
    FG.Fneigs[f] = []
    return f
end

function deletevar!(FG::FactorGraph, v::Int=rand(filter(vv -> vardegree(FG,vv)!=0, 1:FG.n)))
    for f in eachindex(FG.Fneigs)
        # delete i from its neighbors' neighbor lists
        v_idx = findall(isequal(v), FG.Fneigs[f])
        deleteat!(FG.Fneigs[f],v_idx)
        # delete messages to v
        deleteat!(FG.mfv[f], v_idx)
        # delete weight on the adjacency matrix
        deleteat!(FG.hfv[f], v_idx)
    end
    # delete node v
    FG.Vneigs[v] = []
    return v
end

# Leaf removal
function lr!(FG::FactorGraph)
    flag = false    # raised if there are still leaves to remove
    for v in eachindex(FG.Vneigs)
        if vardegree(FG,v)==1
            deletefactor!(FG, FG.Vneigs[v][1])
            flag = true
        end
    end
    flag && lr!(FG)
    nothing
end

# Remove only 1 leaf
function onelr!(FG::FactorGraph, idx::Vector{Int}=randperm(FG.n))
    for v in idx
        if vardegree(FG,v)==1
            deletefactor!(FG, FG.Vneigs[v][1])
            # deletevar!(FG, v)
            return v
        end
    end
    return 0
end

# The following 2 are used to get the number of variables or factors left in
# the graph, which might be different from n,m i.e. the original ones

function nvars(FG::FactorGraph)   # number of variables in the core
    Nvars = 0
    for v in FG.Vneigs
        v != [] && (Nvars += 1)
    end
    return Nvars
end

function nfacts(FG::FactorGraph)    # number of hyperedges in the core
     Nfact = 0
     Fneigs = FG.Fneigs
     for f in Fneigs
         f != [] && (Nfact += 1)
     end
     return Nfact
end

function breduction!(FG::FactorGraph, b::Int; randseed::Int=0)
    randseed != 0 && Random.seed!(randseed)     # for reproducibility
    for _ in 1:b
        deletefactor!(FG)
    end
end

function polyn(FG::FactorGraph)
    fd = countmap(factdegrees(FG))    # degree => number of factors with that degree
    rho = zeros(maximum(keys(fd)))
    for j in keys(fd)
        rho[j] = j*fd[j]
    end
    rho ./= sum(rho)

    vd = countmap(vardegrees(FG))
    lambda = zeros(maximum(keys(vd)))
    for i in keys(vd)
        lambda[i] = i*vd[i]
    end
    lambda ./= sum(lambda)

    return lambda, rho
end

#### A factor graph type thought for GF(q) belief propagation ####
using OffsetArrays, StatsBase, LightGraphs, GraphRecipes, Plots

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
function FactorGraph(A::Array{Int,2}, q::Int= nextpow(2,maximum(A)+0.5),
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

# Returns the proper square adjacency matrix
function full_adjmat(fg::FactorGraph)
    H = adjmat(fg)
    (m,n) = (fg.m, fg.n)
    A = [zeros(Int,m,m) H;
         permutedims(H) zeros(Int,n,n)]
    return A
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

varleaves(fg::FactorGraph) = [v for v in 1:fg.n if isvarleaf(fg,v)]
factleaves(fg::FactorGraph) = [v for v in 1:fg.n if isfactleaf(fg,v)]

nvarleaves(fg::FactorGraph)::Int = sum(vardegrees(fg).==1)
nfactleaves(fg::FactorGraph)::Int = sum(factdegrees(fg).==1)

vardegrees(fg::FactorGraph) = [vardegree(fg,v) for v in eachindex(fg.Vneigs)]
vardegrees_distr(fg::FactorGraph) = proportionmap(vardegrees(fg))
factdegrees(fg::FactorGraph) = [factdegree(fg,f) for f in eachindex(fg.Fneigs)]
factdegrees_distr(fg::FactorGraph) = proportionmap(factdegrees(fg))

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
function deletefactors!(fg::FactorGraph, ff::Vector{Int}) 
    for f in ff
        deletefactor!(fg,f)
    end
    return ff
end


function deletevar!(fg::FactorGraph, 
    v::Int=rand(filter(vv -> vardegree(fg,vv)!=0, 1:fg.n)))
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
function deletevars!(fg::FactorGraph, vv::Vector{Int}) 
    for v in vv
        deletevar!(fg,v)
    end
    return vv
end

# Recursive leaf removal
function lr!(fg::FactorGraph; plotting::Bool=false)
    flag = false    # raised if there are still leaves to remove
    nremoved = 0
    plotting && plot(fg)
    for v in eachindex(fg.Vneigs)
        if vardegree(fg,v)==1
            # delete factor and all its edges
            deletefactor!(fg, fg.Vneigs[v][1])
            nremoved += 1
            flag = true
            if plotting
                display(plot(fg))
                sleep(1)
            end
        end
    end
    flag && lr!(fg)
    return nremoved
end

# Leaf removal but starting from leaf factors!
function lr_factors!(fg::FactorGraph)
    flag = false    # raised if there are still leaves to remove
    for f in eachindex(fg.Fneigs)
        if factdegree(fg,f)==1
            deletevar!(fg, fg.Fneigs[f][1])
            flag = true
        end
    end
    flag && lr_factors!(fg)
    nothing
end

# Experiment! build a graph with only factors (for degree-2 variables only)
# Weights in the final graph are number of mini-loops
function only_factors_graph(fg::FactorGraph)
    fg2 = deepcopy(fg)
    # Ensure all variables have degree 2 and we're working on GF(2)
    @assert all(vardegrees(fg2) .<= 2) && fg2.q==2
    # Delete variable leaves
    deletevars!(fg2, varleaves(fg2))
    lr_factors!(fg2)
    # Start
    g = SimpleWeightedGraph(fg2.m)
    
    # Remove all leaves, being factors or variables
    lr_factors!(fg2)

    H = adjmat(fg2)
    (m,n) = size(H)
    for j in 1:n
        involved = findall(Bool.(H[:,j]))
        involved == [] && continue
        if has_edge(g, (involved...))
            g.weights[involved...] += 1
        else
            add_edge!(g, (involved...))
        end
    end
    return g
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

function breduction!(fg::FactorGraph, b::Int=1; randseed::Int=0)
    randseed != 0 && Random.seed!(randseed)     # for reproducibility
    to_be_removed = rand(filter(ff -> factdegree(fg,ff)!=0, 1:fg.m),b)
    for _ in 1:b
        deletefactors!(fg, to_be_removed)
    end
    return b
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

function plot(fg::FactorGraph;
    highlighted_nodes=Int[], highlighted_factors=Int[])
    m = fg.m
    if typeof(highlighted_nodes)==Int
        highlighted_nodes = [highlighted_nodes]
    end
    g = SimpleGraph(full_adjmat(fg))
    if ne(g) == 0
        println("Graph contains no edges")
        return nothing
    end
    node_idx = [ones(Int,fg.m); 2*ones(Int,fg.n)]
    node_idx[highlighted_factors] .= 3
    node_idx[m .+ highlighted_nodes] .= 4
    shapes = [:rect, :circle, :rect, :circle]
    nodeshape = shapes[node_idx]
    colors = [:yellow, :white, :orange, :red]
    nodecolor = colors[node_idx]
    
    return graphplot(g, curves=false, names = [1:fg.m; 1:fg.n],
        nodeshape = nodeshape, nodecolor=colors[node_idx],
        method=:spring, nodesize=0.15, fontsize=8, nodestrokewidth=0.5)
end
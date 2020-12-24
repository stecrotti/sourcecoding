using BlossomV, LinearAlgebra, SparseArrays

# Implements algorithm 1 in Improved Algorithms for Detecting Negative Cost Cycles in Undirected Graphs.
# Xiaofeng Gu1, Kamesh Madduri, K. Subramani and Hong-Jian Lai

function optimal_cycle(G)
    G = sparse(G)
    @assert issymmetric(G)
    rows, weights  = rowvals(G), nonzeros(G)
    ∂(i) = zip((@view rows[nzrange(G,i)]), (@view weights[nzrange(G,i)]))
    edge(i,j,w) = add_edge(E, i-1, j-1, w)
    match(i) = get_match(E, i - 1) + 1
    n = size(G,1);
    m = Int(length(rows)/2)
    E = Matching(Float64, 2n+2m)
    D = Tuple{Int,Int}[]
    k = 2n+1
    for i=1:n
        edge(i, i+n, 0)
        for (j,w) in ∂(i)
            if i < j
                edge(i,     k,   w)
                edge(i + n, k,   w)
                edge(j,     k+1, w)
                edge(j + n, k+1, w)
                edge(k,     k+1, 0)
                push!(D, (i,j))
                k+=2
            end
        end
    end
    @assert k == 2n+2m+1

    solve(E)

    cycle = [D[k] for k=1:m if match(2k + 2n - 1) != 2k + 2n]
    weight = isempty(cycle) ? 0.0 : sum(G[edge...] for edge in cycle)
    return cycle, weight
end 

# Wizardry to go from edge numbering in the graph used to find cycles (1:m+n) 
#  to actual variable (1:n) and factor (1:m) indices
function variables_from_cycle(cy::Array{Tuple{Int64,Int64},1}, m::Int)
    cy_ = Array{Tuple{Int64,Int64},1}(undef, length(cy))
    for i in eachindex(cy)
        edge_as_vec = sort(collect(cy[i]))
        cy_[i] = (edge_as_vec[1], edge_as_vec[2]-m)
    end
    return cy_    
end

function one_loop_flip(lm::LossyModel)
    H = weighted_full_adjmat(lm)
    op = variables_from_cycle(optimal_cycle(H)[1], lm.fg.m)
    to_flip = unique!([tup[2] for tup in op])
    lm.x[to_flip] .⊻= 1
    return op, to_flip, w
end

# If there are leaves, add a (redundant) factor so that all solutions are loops
function neutralize_leaves!(lm::LossyModel)
    lm.fg = add_factor(lm.fg)
    return lm
end


##### SOLVE LOSSY COMPRESSION
struct OptimalCycle <: LossyAlgo; end

@with_kw struct OptimalCycleResults <: LossyResults
    converged::Bool
    parity::Int
    distortion::Float64
end

function solve!(lm::LossyModel, algo::OptimalCycle, 
    maxiter::Int=min(lm.fg.m,5);
    randseed::Int=abs(rand(Int)), verbose::Bool=true, 
    showprogress::Bool=verbose)

    # Close leaves in a loops
    neutralize_leaves!(lm)

    dist = Float64[]
    finished = false
    # Loop maxiter only as a precaution in case something goes wrong and the 
    #  procedure doesn't stop
    for it in 1:maxiter
        op,to_flip, w = one_loop_flip(lm)
        push!(dist, distortion(lm))
        showprogress && println("Iter ", length(dist), ". Distortion ", 
            round(dist[end], digits=4), ". Cycle weight", round(w,digits=4))
        if isempty(op)
            return OptimalCycleResults(true, parity(lm), distortion(lm))
        end
    end
    return OptimalCycleResults(false, parity(lm), distortion(lm))
end
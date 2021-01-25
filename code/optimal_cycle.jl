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
    E = BlossomV.Matching(Float64, 2n+2m, 5m+n)
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
# The output is of the form (fact,var) for each edge
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
    op_, w = optimal_cycle(float.(H))
    op = variables_from_cycle(op_, lm.fg.m)
    to_flip = unique!([tup[2] for tup in op])
    # @show parity(lm)
    lm.x[to_flip] .⊻= 1
    if parity(lm) != 0
        only_the_flipped = zeros(Int, length(lm.x))
        only_the_flipped[to_flip] .= 1
        # @show only_the_flipped
        # @show parity(lm, only_the_flipped)
        # g = SimpleGraph(full_adjmat(lm.fg))
        # @show length(connected_components(g))
        error("Error: parity non zero")
    end
    return op, to_flip, w
end

# If there are leaves, add a (redundant) factor so that all solutions are loops
function neutralize_leaves!(lm::LossyModel)
    if nvarleaves(lm.fg) > 0
        lm.fg = add_factor(lm.fg)
    end
    return lm
end


@with_kw struct OptimalCycle <: LossyAlgo
    # Function to initialize internal state x
    init_state::Function = (zero_codeword(lm::LossyModel)=zeros(Int, lm.fg.n))       
end

@with_kw struct OptimalCycleResults <: LossyResults
    parity::Int
    distortion::Float64
    converged::Bool=true     # Doesn't mean anything, here just for consistency with the other Results types
end

function solve!(lm::LossyModel, algo::OptimalCycle;
    maxiter::Int=50,
    randseed::Int=abs(rand(Int)), verbose::Bool=true, 
    showprogress::Bool=verbose)

    lm.x = algo.init_state(lm)

    # Close leaves in a loop
    neutralize_leaves!(lm)

    dist = Float64[]
    finished = false
    # Loop maxiter only as a precaution in case something goes wrong and the 
    #  procedure doesn't stop
    E = energy(lm)
    for it in 1:maxiter
        op,to_flip, w = one_loop_flip(lm)
        push!(dist, distortion(lm))

        Echecks = energy_checks(lm)
        Eoverlap = energy_overlap(lm)
        Enew = energy(lm)
        deltaE = Enew - E
        if isinf(Enew)
            @show Echecks, Eoverlap
            error("Inf found in energy")
        end
        if isnan(deltaE)
            @show Enew, E
           error("NaN found in energy shift")
        end
        showprogress && println("Iter ", length(dist), ". Distortion ", 
            round(dist[end], digits=4), ". Cycle weight ", round(w,digits=4),
            ". Energy shift ", deltaE)
        
        if deltaE == 0
            return OptimalCycleResults(parity=parity(lm), 
                distortion=distortion(lm))
        else
            E = Enew
        end
    end
    return OptimalCycleResults(parity=parity(lm), distortion=distortion(lm))
end
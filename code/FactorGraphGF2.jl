using SparseArrays

struct FactorGraphGF2 <: FactorGraph
    q::Int                              # field order
    mult::OffsetArray{Int,2}            # multiplication matrix in GF(q)
    gfinv::Vector{Int}                  # inverses in GF(q). It has q-1 indices, since 0 has no inverse
    gfdiv::OffsetArray{Int,2}
    n::Int                              # number of variable nodes
    m::Int                              # number of factor nodes
    Vneigs::Vector{Vector{Int}}         # neighbors of variable nodes.
    Fneigs::Vector{Vector{Int}}         # neighbors of factor nodes (containing only factor nodes)
    fields::Vector{Float64}             # Prior probabilities in the form of external fields
    H::SparseMatrixCSC{Int64,Int64}     # Adjacency matrix
    mfv::Vector{Vector{Float64}}        # Messages from factor to variable with index starting at 0
end

# Construct graph from adjacency matrix (for checks with simple examples)
function FactorGraphGF2(A::AbstractArray{Int,2}, 
        fields::Vector{Float64} = zeros(size(A,2)))  
    m,n = size(A)
    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    mfv = [Float64[] for f in 1:m]

    for f in 1:m
        for v in 1:n
            if A[f,v]<0 || A[f,v]>1
                error("Entry of the adjacency matrix must be 0≤h_ij<q")
            elseif A[f,v] > 0
                push!(Fneigs[f], v)
                push!(Vneigs[v], f)
                push!(mfv[f], 0.0)
            end
        end
    end
    mult, gfinv, gfdiv = gftables(2)
    H = issparse(A) ? A : sparse(A)
    return FactorGraphGF2(2, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, fields, H, mfv)
end

function ldpc_graphGF2(n::Int, m::Int,
    nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
    rho::Vector{T}=generate_polyn(n,m)[3],
    fields::Vector{Float64} = zeros(n); verbose=false,
    arbitrary_mult = false,
    randseed::Int=0) where {T<:AbstractFloat}

    m < 2 && error("Cannot build a graph with m≤1 factors")

    randseed != 0 && Random.seed!(randseed)      # for reproducibility

    ### Argument validation ###
    _check_consistency_polynomials(lambda, rho, nedges, n, m)
    
    if verbose
        println("Building factor graph...")
        println("lambda = ", lambda, "\nrho = ", rho)
    end

    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    H = SparseArrays.spzeros(m,n)
    edgesleft = edgesright = zeros(Int, nedges)

    too_many_trials = 1000
    multi_edge_found = false
    for t in 1:too_many_trials
        multi_edge_found = false
        Vneigs .= [Int[] for v in 1:n]
        Fneigs .= [Int[] for f in 1:m]
        H .= SparseArrays.spzeros(m,n)
        edgesleft .= edgesright .= zeros(Int, nedges)

        ### Irregular Tanner Graph construction and FactorGraph object initialization ###
        # Assign each edge "on the left" to a variable node
        v = 1; r = 1
        for i in 1:length(lambda)
            deg = Int(round(lambda[i]/i*nedges,digits=10))   # number of edges incident on variable v
            for _ in 1:deg
                edgesleft[r:r+i-1] .= v
                r += i; v += 1
            end
        end

        shuffle!(edgesleft)   # Permute nodes on the left

        # Assign each edge "on the right" to a factor node
        f = s = 1
        for j in 1:length(rho)
            deg = Int(round(rho[j]/j*nedges,digits=10))
            for _ in 1:deg
                for v in edgesleft[s:s+j-1]
                    if findall(isequal(v), Fneigs[f])!=[]
                        verbose && println("Multi-edge discarded")
                        multi_edge_found = true
                        break
                    end
                    # Initialize neighbors
                    push!(Fneigs[f], v)
                    push!(Vneigs[v], f)
                    # Initalize parity check matrix elements
                    H[f,v] = 1
                end
                s += j
                f += 1
            end
        end
        if !multi_edge_found
            # Initialize messages
            mfv = [zeros(length(neigs)) for neigs in Fneigs]
            # Get multiplication and iverse table for GF(q)
            mult, gfinv, gfdiv = gftables(2, arbitrary_mult)
            # Build FactorGraph object
            fg = FactorGraphGF2(2, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, 
                fields, H, mfv)
            # Check that the number of connected components is 1
            fg_ = deepcopy(fg)
            breduction!(fg_,1)
            depths,_,_ = lr!(fg_)
            all(depths .!= 0) && return fg 
        end
    end
    # If you got here, multi-edges made it impossible to build a graph
    if multi_edge_found
        error("Could not build factor graph. Too many multi-edges were popping up") 
    else
        error("Could not build factor graph. Too many instances were coming up ",
            "with more than one connected component")
    end   
end

guesses(fg::FactorGraphGF2) = floor.(Int, 0.5*(1 .- sign.(fg.fields)))

function onebpiter!(fg::FactorGraphGF2, algo::MS,
    neutral=neutralel(algo,fg.q))
    maxdiff = diff = 0.0
    aux = Float64[]
    # Loop over factors
    for f in randperm(fg.m)
        # Loop over neighbors of `f`
        for (v_idx, v) in enumerate(fg.Fneigs[f])
            # Subtract message from belief
            fg.fields[v] -= fg.mfv[f][v_idx]          
            # Collect (var->fact) messages from the other neighbors of `f`
            aux = [fg.fields[vprime] - fg.mfv[f][vprime_idx] 
                for (vprime_idx,vprime) in enumerate(fg.Fneigs[f]) 
                if vprime_idx != v_idx]
            # Apply formula to update message
            fg.mfv[f][v_idx] = prod(sign, aux)*reduce(min, abs.(aux), init=Inf)
            # Update belief after updating the message
            fg.fields[v] += fg.mfv[f][v_idx]
            # Look for maximum message
            diff = abs(fg.mfv[f][v_idx])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return maxdiff
end


# Re-initialize messages
function refresh!(fg::FactorGraphGF2)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= 0.0
    end
    return nothing
end


function reinforce!(fg::FactorGraphGF2, algo::Union{BP,MS})
    for (v,gv) in enumerate(fg.fields)
        if algo.gamma > 0
            if typeof(algo)==BP
                fg.fields[v] *= gv.^algo.gamma
            elseif typeof(algo)==MS
                fg.fields[v] += gv*algo.gamma
            end
        end
    end
    return nothing
end

# Parity-check for the adjacency matrix of a factor graph.
# f specifies which factors to consider (default: all 1:m)
function paritycheck(fg::FactorGraphGF2, x::Vector{Int}=guesses(fg))
    z = (fg.H * x) .% 2
    return z
end

function free_energy(fg::FactorGraphGF2)
    O = 0.0
     for (a_idx,a) in enumerate(fg.Fneigs)
        F = [fg.fields[i] - fg.mfv[a_idx][i_idx] for (i_idx,i) in enumerate(a)]
        O += sum(abs.(F)) - (prod(F)<0)*2*minimum(abs.(F)) 
     end
     O -= sum(abs.(fg.fields))
 end

#  function _fix_indep(fg::FactorGraphGF2, z::Vector{Int}, x::Vector{Int}; 
#     triang_form=permute_to_triangular(fg))
    
#     # Retrieve permuted parity-check matrix in the form [T|U]
#     M, col_perm = triang_form
#     m,n = size(M)
#     dependent = col_perm[1:m]
#     independent = col_perm[m+1:end]
#     x[independent] = z[independent]
#     b = M[:,m+1:end] * z[independent] .% 2
#     x[dependent] .= gf_invert_ut(M[:,1:m], b, fg.q, fg.mult, fg.gfdiv, x[dependent])
#     return x
# end

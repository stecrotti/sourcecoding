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
    hfv::Vector{Vector{Int}}            # Non-zero elements of the parity check matrix
    mfv::Vector{Vector{Float64}}        # Messages from factor to variable with index starting at 0
end

# Construct graph from adjacency matrix (for checks with simple examples)
function FactorGraphGF2(A::Array{Int,2}, q::Int=nextpow(2,maximum(A)+0.5),
    fields = [Fun(1e-3*randn(q)) for v in 1:size(A,2)])
    m,n = size(A)
    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    hfv = [Int[] for f in 1:m]
    mfv = [Vector{Float64}[] for f in 1:m]

    for f in 1:m
        for v in 1:n
            if A[f,v]<0 || A[f,v]>q-1
                error("Entry of the adjacency matrix must be 0≤h_ij<q")
            elseif A[f,v] > 0
                push!(Fneigs[f], v)
                push!(Vneigs[v], f)
                push!(hfv[f], A[f,v])
                push!(mfv[f], 0.0)
            end
        end
    end
    mult, gfinv, gfdiv = gftables(q)
    return FactorGraphGF2(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
end


guesses(fg::FactorGraphGF2) = floor.(Int, 0.5*(1 .- sign.(fg.fields)))

function onebpiter!(fg::FactorGraphGF2, algo::MS)
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
    return guesses(fg), maxdiff
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
function paritycheck(fg::FactorGraphGF2, x::Vector{Int}=guesses(fg),
    f::Vector{Int}=collect(1:fg.m))
    B = adjmat(fg)[f,:]
    z = B * x
    return z
end
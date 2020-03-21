function ldpc_graph(q::Int, n::Int, k::Int,
    nedges::Int, lambda::Vector{T}=[0.0, 1.0], rho::Vector{T}=[0.0, 0.5, 0.5],
    fields = [OffsetArray(fill(1/q, q), 0:q-1) for v in 1:n]) where {T<:AbstractFloat}

    ### Argument validation ###
    if sum(lambda) != 1 || sum(rho) != 1
        error("Vector lambda and rho must sum to 1")
    elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
        error("n, lambda and nedges incompatible")
    elseif k != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
        error("k, rho and nedges incompatible")
    end

    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:k]
    hfv = [Dict() for f in 1:k]
    mfv = [Dict() for f in 1:k]

    edgesleft = zeros(Int, nedges)
    edgesright = zeros(Int, nedges)

    ### Irregular Tanner Graph construction and FactorGraph object initialization ###
    # Assign each edge "on the left" to a variable node
    v = 1
    r = 1
    for i in 1:length(lambda)
        deg = Int(lambda[i]/i*nedges)   # number of edges incident on variable v
        for _ in 1:deg
            edgesleft[r:r+i-1] = v*ones(Int,i)
            r += i
            v += 1
        end
    end

    perm = edgesleft[randperm(length(edgesleft))]   # Permute nodes on the left

    # Assign each edge "on the right" to a factor node
    f = 1
    s = 1
    for j in 1:length(rho)
        deg = Int(rho[j]/j*nedges)
        for _ in 1:deg
            for v in  perm[s:s+j-1]
                ########
                # If we want to avoid multi-edges, this is probably the right place to do something about it
                ########
                # Initialize neighbors
                push!(Fneigs[f], v)
                push!(Vneigs[v], f)
                # Initalize parity check matrix elements
                hfv[f][v] = rand(1:q-1)
                # While we're here, initialize messages factor->variable
                mfv[f][v] = OffsetArray(1/q*ones(q), 0:q-1)
            end
            s += j
            f += 1
        end
    end

    # Get multiplication and iverse table for GF(q)
    mult, gfinv = gftables(q)

    return FactorGraph(q, mult, gfinv, n, k, Vneigs, Fneigs, fields, hfv, mfv)
end

function gftables(q)
    if q==2
        elems = [0,1]
    else
        G,x = GaloisField(q,:x)
        elems = collect(G)
    end
    ##########
    # What if q = p^1 ?
    #########
    M = [findfirst(isequal(x*y),elems)-1 for x in elems, y in elems]
    mult = OffsetArray(M, 0:q-1, 0:q-1)
    gfinv = [findfirst(isequal(inv(x)), elems)-1 for x in elems[2:end]]

    return mult, gfinv
end

# Creates fields for the priors: the closest to y, the stronger the field
# The prior distr is given by exp(field)
function extfields(q::Int, y::Vector{Int}, L::Real=1.0)
    fields = [OffsetArray(fill(1.0, q), 0:q-1) for v in eachindex(y)]
    for v in eachindex(fields)
        for a in 0:q-1
            fields[v][a] = L*hd(a,y[v])
        end
    end
    return fields
end

# Hamming distance, works when q is a power of 2
function hd(x::Int,y::Int)
    count(isequal('1'), bitstring(xor.(x,y)))
end

function hd(x::Vector{Int}, y::Vector{Int})
    sum(x.!=y)
end

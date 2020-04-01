function ldpc_graph(q::Int, n::Int, m::Int,
    nedges::Int, lambda::Vector{T}=[0.0, 1.0], rho::Vector{T}=[0.0, 0.5, 0.5],
    fields = [Fun(1e-3*randn(q)) for v in 1:n]; verbose=false) where {T<:AbstractFloat}

    ### Argument validation ###
    if sum(lambda) != 1 || sum(rho) != 1
        error("Vector lambda and rho must sum to 1")
    elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
        error("n, lambda and nedges incompatible")
    elseif m != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
        error("m, rho and nedges incompatible")
    end

    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    hfv = [Int[] for f in 1:m]
    mfv = [OffsetArray{Float64,1,Array{Float64,1}}[] for f in 1:m]

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
                if findall(isequal(v), Fneigs[f])!=[]
                    verbose && println("Multi-edge discarded")
                    continue
                end
                # Initialize neighbors
                push!(Fneigs[f], v)
                push!(Vneigs[v], f)
                # Initalize parity check matrix elements
                push!(hfv[f], rand(1:q-1))
                # While we're here, initialize messages factor->variable
                push!(mfv[f], OffsetArray(1/q*ones(q), 0:q-1))
            end
            s += j
            f += 1
        end
    end

    # Get multiplication and iverse table for GF(q)
    mult, gfinv = gftables(q)

    return FactorGraph(q, mult, gfinv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
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
# A small noise with amplitude sigma is added to break the symmetry
function extfields(q::Int, y::Vector{Int}, algo::Union{BP,MS}, L::Real=1.0, sigma::Real=1e-4)
    fields = [OffsetArray(fill(0.0, q), 0:q-1) for v in eachindex(y)]
    for v in eachindex(fields)
        for a in 0:q-1
            fields[v][a] = -L*hd(a,y[v]) + sigma*randn()
            typeof(algo)==BP && (fields[v][a] .= exp.(fields[v][a]))
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

# Works only for GF(2^k)
function paritycheck(H::Array{Int,2}, y::Vector{Int}, mult::OffsetArray{Int,2,Array{Int,2}})
    m,n = size(H)
    @assert length(y) == n
    z = zeros(Int, m)
    for i in eachindex(z)
        s = 0
        for j in eachindex(y)
            s = xor(s, mult[H[i,j], y[j]])
        end
        z[i] = s
    end
    return z
end

#### Not used
function ldpc_adjmat(q::Int, n::Int, m::Int,
    nedges::Int, lambda::Vector{T}=[0.0, 1.0], rho::Vector{T}=[0.0, 0.5, 0.5];
     verbose=false) where {T<:AbstractFloat}

    ### Argument validation ###
    if sum(lambda) != 1 || sum(rho) != 1
        error("Vector lambda and rho must sum to 1")
    elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
        error("n, lambda and nedges incompatible")
    elseif m != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
        error("m, rho and nedges incompatible")
    end

    H = zeros(Int, m, n)
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
                if findall(isequal(v), H[f,:])!=[]
                    verbose && println("Warning: I'm building a multi-edge")
                    continue
                end
                H[f,v] = rand(1:q-1)
            end
            s += j
            f += 1
        end
    end
    return H
 end

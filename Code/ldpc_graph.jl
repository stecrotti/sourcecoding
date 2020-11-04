function ldpc_graph(q::Int, n::Int, m::Int,
    nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
    rho::Vector{T}=generate_polyn(n,m)[3],
    fields = [Fun(1e-3*randn(q)) for v in 1:n]; verbose=false,
    arbitrary_mult = false,
    randseed::Int=0) where {T<:AbstractFloat}

    randseed != 0 && Random.seed!(randseed)      # for reproducibility

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
        deg = Int(round(lambda[i]/i*nedges,digits=10))   # number of edges incident on variable v
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
        deg = Int(round(rho[j]/j*nedges,digits=10))
        for _ in 1:deg
            for v in perm[s:s+j-1]
                if findall(isequal(v), Fneigs[f])!=[]
                    # verbose && println("Multi-edge discarded")
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
    mult, gfinv, gfdiv = gftables(q, arbitrary_mult)

    return FactorGraph(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, fields, hfv, mfv)
end

function generate_polyn(n::Int, m::Int)
    # This part is fixed
    lambda = [0.0, 1.0]
    nedges = 2*n
    # Find the right r
    r = Int(ceil(nedges/m - 1))
    # Initialize and fill rho
    rho = zeros(r+1)
    rho[r] = r*((r+1)*m - nedges)
    rho[r+1] = (r+1)*(nedges - r*m)
    # Normalize
    rho ./= nedges

    return nedges, lambda, rho
end

function gftables(q::Int, arbitrary_mult::Bool=false)
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
    if arbitrary_mult
        gfinv = zeros(Int, q-1)
        # gfinv[1] = 1
        # for r in 2:q-1
        #     if gfinv[r] == 0
        #         gfinv[r] = rand(findall(gfinv .== 0))
        #         gfinv[gfinv[r]] = r
        #     end
        # end
        #
        # for r in 2:q-1
        #     mult[r, gfinv[r]] = 1
        #     others = [i for i in 2:q-1 if i!=r]
        #     mult[r, [2:gfinv[r]-1; gfinv[r]+1:q-1] ] = shuffle(others)
        # end
        for r in 1:q-1
            mult[r, 1:q-1] .= shuffle(mult[r, 1:q-1])
        end

    else
        gfinv = [findfirst(isequal(1), mult[r,1:end]) for r in 1:q-1]
    end

    div = OffsetArray(zeros(Int, q,q-1), 0:q-1,1:q-1)
    for r in 1:q-1
        for c in 1:q-1
            div[r,c] = findfirst(isequal(r), [mult[c,k] for k in 1:q-1])
        end
    end

    return mult, gfinv, div
end


# Creates fields for the priors: the closest to y, the stronger the field
# The prior distr is given by exp(field)
# A small noise with amplitude sigma is added to break the symmetry
function extfields(q::Int, y::Vector{Int}, algo::Union{BP,MS}, L::Real=1.0,
    sigma::Real=1e-4; randseed::Int=0)
    randseed != 0 && Random.seed!(randseed)      # for reproducibility
    fields = [OffsetArray(fill(0.0, q), 0:q-1) for v in eachindex(y)]
    for v in eachindex(fields)
        for a in 0:q-1
            fields[v][a] = -L*hd(a,y[v]) + sigma*randn()
            typeof(algo)==BP && (fields[v][a] = exp.(fields[v][a]))
        end
    end
    return fields
end

# Hamming distance, works when q is a power of 2
function hd(x::Int,y::Int)::Int
    count(isequal('1'), bitstring(xor.(x,y)))
end

function hd(x::Vector{Int}, y::Vector{Int})::Int
    sum(hd.(x,y))
end

# Hamming weight
function hw(x::Int)::Int
    hd(x, 0)
end

function hw(v::Vector{Int})::Int
    sum(hw.(v))
end

# Works only for GF(2^k)
function paritycheck(H::Array{Int,2}, x::Vector{Int}, q::Int,
                    mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    m,n = size(H)
    r,p = size(mult)
    @assert length(x) == n
    @assert r == p
    z = zeros(Int, m)
    for i in eachindex(z)
        # s = 0
        # for j in eachindex(y)
        #     s = hd(s, mult[H[i,j], y[j]])
        # end
        # z[i] = s
        z[i] = reduce(xor, [mult[H[i,j], x[j]] for j in eachindex(x)], init=0)
    end
    return z
end

# Parity-check for the adjacency matrix of a factor graph.
# f specifies which factors to consider (default: all 1:m)
function paritycheck(fg::FactorGraph, x::Vector{Int}=guesses(fg),
    f::Vector{Int}=collect(1:fg.m))
    return paritycheck(adjmat(fg)[f,:], x, fg.q, fg.mult)
end
# Support f as a single index, not forcefully a vector
function paritycheck(fg::FactorGraph, x::Vector{Int}, f::Int)
    return paritycheck(fg, x, [f])
end


# Groups bits together to transform GF(2)->GF(2^k)
function gf2toq(H::Array{Int,2}, k::Int=1)
    m,n = size(H)
    nnew = div(n,k)
    Hnew = zeros(Int, m, nnew)
    for f in 1:m
        Hnew[f,:] = gf2toq(H[f,:], k)
    end
    return Hnew
end

function gf2toq(x::Vector{Int}, k::Int=1)
    mod(length(x),k) != 0 && error("Length of vector x must be a multiple of k")
    newlen = div(length(x),k)
    return [bits2int(x[k*(v-1)+1:k*(v-1)+k]) for v in 1:newlen]
end

function bits2int(x::Vector{Int})
    bitcheck = prod(in.(x, Ref([0,1])))
    !bitcheck && error("Input vector x must be made of bits")
    return sum(y*2^(i-1) for (i,y) in enumerate(reverse(x)))
end

function int2bits(x::Int, len::Int)
    x > 2^len-1 && error("x is too large to fit in a bit vector of length len")
    b = bitstring(x)
    y = split(b,"")
    return parse.(Int,y[end-len+1:end])
end

function gfqto2(y::Vector{Int}, k::Int)
    z = zeros(Int, length(y)*k)
    for (i,s) in enumerate(y)
        z[k*(i-1)+1:k*(i-1)+k] = int2bits(s,k)
    end
    return z
end

function gfqto2(H::Array{Int,2}, k::Int=1)
    m,n = size(H)
    nnew = n*k
    Hnew = zeros(Int, m, nnew)
    for f in 1:m
        Hnew[f,:] = gfqto2(H[f,:], k)
    end
    return Hnew
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

#### Build and handle LDPC graphs, Hamming distances and more, on 𝔾𝔽(2ᵏ) ####
using GaloisFields, Random

function ldpc_graph(q::Int, n::Int, m::Int,
    nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
    rho::Vector{T}=generate_polyn(n,m)[3],
    fields = [Fun(1e-3*randn(q)) for v in 1:n]; verbose=false,
    arbitrary_mult = false,
    randseed::Int=0) where {T<:AbstractFloat}

    m < 2 && error("Cannot build a graph with m≤1 factors")

    randseed != 0 && Random.seed!(randseed)      # for reproducibility

    ### Argument validation ###
    if sum(lambda) != 1 || sum(rho) != 1
        error("Vector lambda and rho must sum to 1")
    elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
        error("n, lambda and nedges incompatible")
    elseif m != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
        error("m, rho and nedges incompatible")
    end

    if verbose
        println("Building factor graph...")
        println("lambda = ", lambda, "\nrho = ", rho)
    end

    Vneigs = [Int[] for v in 1:n]
    Fneigs = [Int[] for f in 1:m]
    hfv = [Int[] for f in 1:m]
    mfv = [OffsetArray{Float64,1,Array{Float64,1}}[] for f in 1:m]

    too_many_trials = 1000
    multi_edge_found = false
    for t in 1:too_many_trials

        multi_edge_found = false
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
                        verbose && println("Multi-edge discarded")
                        multi_edge_found = true
                        break
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
        if !multi_edge_found
            # Get multiplication and iverse table for GF(q)
            mult, gfinv, gfdiv = gftables(q, arbitrary_mult)
            # Build FactorGraph object
            fg = FactorGraph(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, 
                fields, hfv, mfv)
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

function generate_polyn(n::Int, m::Int; degree_type::Symbol=:edges)
    @assert degree_type in [:edges, :nodes]
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
    if degree_type == :edges
        return nedges, lambda, rho
    else
        lambda_n, rho_n = edges2nodes(lambda, rho)
        return nedges, lambda_n, rho_n
    end
end
# Switch from the 2 alternative representations for graph degree profiles
# "Edges" is the one expressing everything in terms of edge degree, as used in
#   https://ieeexplore.ieee.org/document/910576
# "Nodes" is the one expressing everything in terms of node degrees, as in
#  https://web.stanford.edu/~montanar/RESEARCH/book.html 

function edges2nodes(lambda::Vector{<:AbstractFloat}, 
        rho::Vector{<:AbstractFloat})
    lambda_new = [lambda[i]/i for i in eachindex(lambda)]
    rho_new = [rho[j]/j for j in eachindex(rho)]
    return lambda_new./sum(lambda_new), rho_new./sum(rho_new)
end
function nodes2edges(lambda::Vector{<:AbstractFloat}, 
    rho::Vector{<:AbstractFloat})
lambda_new = [lambda[i]*i for i in eachindex(lambda)]
rho_new = [rho[j]*j for j in eachindex(rho)]
return lambda_new./sum(lambda_new), rho_new./sum(rho_new)
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

# Hamming distance, works when q is a power of 2
function hd(x::Int,y::Int)::Int
    z = xor(x,y)
    return sum(int2bits(z))
end

function hd(x::Vector{Int}, y::Vector{Int})::Int
    sum(hd.(x,y))
end

# Hamming weight
function hw(x::Int)::Int
    return sum(int2bits(x))
end

hw(v::Vector{Int})::Int = sum(hw.(v))
hw(v::Array{Int,2})::Int = hw(vec(v))

function paritycheck(fg::FactorGraph, x::Array{Int,2}, f::Int)
    return gfmatrixmult(adjmat(fg)[[f],:], x, fg.q, fg.mult)
end

function paritycheck(fg::FactorGraph, x::Array{Int,2},
    f::Vector{Int}=collect(1:fg.m))
    return gfmatrixmult(adjmat(fg)[f,:], x, fg.q, fg.mult)
end

parity(fg::FactorGraph, args...) = hw(paritycheck(fg, args...))

# Parity-check for the adjacency matrix of a factor graph.
# f specifies which factors to consider (default: all 1:m)
function paritycheck(fg::FactorGraph, x::Vector{Int}=guesses(fg),
    f::Vector{Int}=collect(1:fg.m))
    return gfmatrixmult(adjmat(fg)[f,:], x[:,:], fg.q, fg.mult)
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

function int2bits(x::Int, pad::Int=ceil(Int,log2(x+1)))
    x > 2^pad && error("Input $x is too large to fit in a bit vector of length $pad")
    return reverse(digits(x, base=2, pad=pad))
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

function int2gfq(x::Int, k::Int=1, pad::Int=ndigits(x,base=2^k))
    # x > 2^pad && error("Input x is too large to fit in a bit vector of length $pad")
    return reverse(digits(x, base=2^k, pad=pad))
end

function int2gfq(y::Vector{Int}, k::Int=1, pad::Int=ndigits(maximum(y),base=2^k))
    return [int2gfq(x, k, pad) for x in y]
end



#### Not used
# function ldpc_adjmat(q::Int, n::Int, m::Int,
#     nedges::Int, lambda::Vector{T}=[0.0, 1.0], rho::Vector{T}=[0.0, 0.5, 0.5];
#      verbose=false) where {T<:AbstractFloat}
#
#     ### Argument validation ###
#     if sum(lambda) != 1 || sum(rho) != 1
#         error("Vector lambda and rho must sum to 1")
#     elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
#         error("n, lambda and nedges incompatible")
#     elseif m != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
#         error("m, rho and nedges incompatible")
#     end
#
#     H = zeros(Int, m, n)
#     edgesleft = zeros(Int, nedges)
#     edgesright = zeros(Int, nedges)
#
#     ### Irregular Tanner Graph construction and FactorGraph object initialization ###
#     # Assign each edge "on the left" to a variable node
#     v = 1
#     r = 1
#     for i in 1:length(lambda)
#         deg = Int(lambda[i]/i*nedges)   # number of edges incident on variable v
#         for _ in 1:deg
#             edgesleft[r:r+i-1] = v*ones(Int,i)
#             r += i
#             v += 1
#         end
#     end
#
#     perm = edgesleft[randperm(length(edgesleft))]   # Permute nodes on the left
#
#     # Assign each edge "on the right" to a factor node
#     f = 1
#     s = 1
#     for j in 1:length(rho)
#         deg = Int(rho[j]/j*nedges)
#         for _ in 1:deg
#             for v in  perm[s:s+j-1]
#                 ########
#                 # If we want to avoid multi-edges, this is probably the right place to do something about it
#                 ########
#                 if findall(isequal(v), H[f,:])!=[]
#                     verbose && println("Warning: I'm building a multi-edge")
#                     continue
#                 end
#                 H[f,v] = rand(1:q-1)
#             end
#             s += j
#             f += 1
#         end
#     end
#     return H
#  end
#
#  # Works only for GF(2^k)
#  function paritycheck(H::Array{Int,2}, x::Vector{Int}, q::Int,
#                      mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
#      m,n = size(H)
#      r,p = size(mult)
#      @assert length(x) == n
#      @assert r == p
#      z = zeros(Int, m)
#      for i in eachindex(z)
#          # s = 0
#          # for j in eachindex(y)
#          #     s = hd(s, mult[H[i,j], y[j]])
#          # end
#          # z[i] = s
#          z[i] = reduce(xor, [mult[H[i,j], x[j]] for j in eachindex(x)], init=0)
#      end
#      return z
#  end

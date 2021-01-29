#### Build and handle LDPC graphs, Hamming distances and more, on 𝔾𝔽(2ᵏ) ####
using GaloisFields, Random

function ldpc_graph(::Val{2}, args...; kw...)
    return ldpc_graphGF2(args...; kw...)
end
function ldpc_graph(::Val{q}, args...; kw...) where {q} 
    return ldpc_graphGFQ(q, args...; kw...)
end

function ldpc_graphGFQ(q::Int, n::Int, m::Int,
    nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
    rho::Vector{T}=generate_polyn(n,m)[3],
    fields = [Fun(1e-3*randn(q)) for v in 1:n]; verbose=false,
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
                    H[f,v] = rand(1:q-1)
                    # While we're here, initialize messages factor->variable
                end
                s += j
                f += 1
            end
        end
        if !multi_edge_found
            # Initialize messages
            mfv = [fill(OffsetArray(1/q*ones(q), 0:q-1), length(neigs)) for neigs in Fneigs]
            # Get multiplication and iverse table for GF(q)
            mult, gfinv, gfdiv = gftables(q, arbitrary_mult)
            # Build FactorGraph object
            # if q > 2
            fg = FactorGraphGFQ(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, 
                fields, H, mfv)
            # else
            #     fields = zeros(n)
            #     fg = FactorGraphGF2(q, mult, gfinv, gfdiv, n, m, Vneigs, Fneigs, 
            #         fields, hfv, mfv)
            # end
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
    gfmult = OffsetArray(M, 0:q-1, 0:q-1)
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
        # for c in 2:q-1
            gfmult[r, [1:r-1; r+1:q-1]] .= shuffle(gfmult[r, [1:r-1; r+1:q-1]])
            # gfmult[1:q-1,c] .= shuffle(gfmult[1:q-1,c])
        end

    else
        gfinv = [findfirst(isequal(1), gfmult[r,1:end]) for r in 1:q-1]
    end

    gfdiv = OffsetArray(zeros(Int, q,q-1), 0:q-1,1:q-1)
    for r in 1:q-1
        for c in 1:q-1
            gfdiv[r,c] = findfirst(isequal(r), [gfmult[c,k] for k in 1:q-1])
        end
    end

    return gfmult, gfinv, gfdiv
end

# Hamming distance, works when q is a power of 2
function hd(x::Int,y::Int)::Int
    z = xor(x,y)
    return hw(z)
end

function hd(x::Vector{Int}, y::Vector{Int})::Int
    sum(hd.(x,y))
end


hw(v::Vector{Int})::Int = sum(count_ones, v)
hw(v::Array{Int,2})::Int = hw(vec(v))

# Parity-check for the adjacency matrix of a factor graph
function paritycheck(fg::FactorGraph, x::AbstractVector{Int}=guesses(fg))
    return gfmatrixmult(fg.H, x, fg.q, fg.mult)
end
# function paritycheck(fg::FactorGraph, x::AbstractArray{Int,2})
#     return gfmatrixmult(fg.H, vec(x), fg.q, fg.mult)
# end
function paritycheck(fg::FactorGraphGF2, x::AbstractVector{Int}=guesses(fg))
    z = 0
    for f in eachindex(fg.Fneigs)
        z += reduce(+, x[v] for v in fg.Fneigs[f], init=0) % 2
    end
    return z
end

parity(fg::FactorGraph, args...) = hw(paritycheck(fg, args...))

function free_energy(fg::FactorGraphGF2)
    O = 0.0
    # Loop only on factors with neighbors
     for (a_idx,a) in enumerate(fg.Fneigs)
        a == [] && continue
        F = [fg.fields[i] - fg.mfv[a_idx][i_idx] for (i_idx,i) in enumerate(a)]
        O += sum(abs,F) - (prod(F)<0)*2*minimum(abs,F) 
     end
     O -= sum(abs,fg.fields)
     E = 0.5*(fg.n-O)
     return E
 end

# Groups bits together to transform GF(2)->GF(2^k)
function gf2toq(H::AbstractArray{Int,2}, k::Int=1)
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

function gfqto2(H::AbstractArray{Int,2}, k::Int=1)
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

####### SUBROUTINES
function _check_consistency_polynomials(lambda, rho, nedges, n, m)
    if sum(lambda) != 1 || sum(rho) != 1
        error("Vector lambda and rho must sum to 1")
    elseif n != round(nedges*sum(lambda[i]/i for i in eachindex(lambda)))
        error("n, lambda and nedges incompatible")
    elseif m != round(nedges*(sum(rho[j]/j for j in eachindex(rho))))
        error("m, rho and nedges incompatible")
    end
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

#### Perform simulated annealing on a LossyModel object. Flexible for new MC moves to be added ####
using ProgressMeter, Statistics, Printf

abstract type MCMove end
struct Metrop1 <: MCMove; end
# Propose states by flipping coefficients of expansion on a basis
@with_kw mutable struct MetropBasisCoeffs <: MCMove
    basis::Array{Int,2}=Array{Int,2}(undef,0,0)
    getbasis::Function=lightbasis
end
# Propose states by flipping seaweeds
@with_kw mutable struct MetropSmallJumps <: MCMove
    depths::Vector{Int}=Vector{Int}(undef,0)
    isincore::BitArray{1}=falses(0)
end

function adapt_to_model!(mc_move::MCMove, lm::LossyModel)
    return mc_move
end
function adapt_to_model!(mc_move::MetropSmallJumps, lm::LossyModel)
    # If graph has no leaves, remove one factor
    if nvarleaves(lm.fg) == 0
        breduction!(lm, 1)
    end
    depths,_,_ = lr(lm.fg)
    isincore = (depths .== 0)
    mc_move.depths = depths
    mc_move.isincore = isincore
    return mc_move
end
function adapt_to_model!(mc_move::MetropBasisCoeffs, lm::LossyModel)
    mc_move.basis = mc_move.getbasis(lm)
    return mc_move
end


@with_kw struct SAResults <: LossyResults
    parities::Vector{Vector{Int}}
    distortion::Float64
    acceptance_ratio::Vector{Float64}
    dE::Vector{Vector{Float64}}
    beta_argmin::Vector{Float64}                # Temperatures for which the min was achieved
    converged::Bool=true     # Doesn't mean anything, here just for consistency with the other Results types
end


#### SIMULATED ANNEALING
@with_kw struct SA <: LossyAlgo
    mc_move::MCMove = MetropBasisCoeffs()                                   # Single MC move
    betas:: Array{Float64,2} = [1.0 1.0]                                    # Cooling schedule
    nsamples::Int = Int(1e2)                                                # Number of samples
    init_state!::Function = zero_codeword!       # Function to initialize internal state
end
function SA(mc_move::MCMove, beta2::Vector{Float64}; kw...)
    betas = hcat(fill(Inf, length(beta2)), beta2)
    return SA(mc_move=mc_move, betas=betas; kw...)
end
function zero_codeword!(lm::LossyModel, args...)
    zeros(eltype(lm.x), lm.fg.n)
end
function random_state!(lm::LossyModel, rng::AbstractRNG)
    lm.x .= rand(rng, [0,1], length(lm.x))
end
function random_state!(lm::LossyModelGF2, rng::AbstractRNG)
    lm.x .= bitrand(length(lm.x))
end

function solve!(lm::LossyModel, algo::SA,
    distortions::Vector{Vector{Float64}}=[fill(0.5, algo.nsamples) 
        for _ in 1:size(algo.betas,1)],
    accepted::Vector{BitArray{1}}=[falses(algo.nsamples) 
        for _ in 1:size(algo.betas,1)],
    parities::Vector{Vector{Int}}=[fill(typemax(Int), algo.nsamples) 
        for _ in 1:size(algo.betas,1)],
    dE::Vector{Vector{Float64}}=[fill(0.5, algo.nsamples) 
        for _ in 1:size(algo.betas,1)];          
    to_display::String="Running Monte Carlo...",
    verbose::Bool=false, randseed::Int=0, showprogress::Bool=verbose)  

    
    rng = Random.MersenneTwister(randseed)
    # Initialize to the requested initial state
    algo.init_state!(lm)
    # Adapt mc_move parameters to the current model
    adapt_to_model!(algo.mc_move, lm)

    nbetas = size(algo.betas,1)
    mindist = zeros(nbetas)

    wait_time = showprogress ? 1 : Inf   # trick not to display progess
    prog = ProgressMeter.Progress(nbetas, wait_time, to_display)
    en = energy(lm)
    @inbounds for b in 1:nbetas
        # Update temperature
        lm.beta1, lm.beta2 = algo.betas[b,1], algo.betas[b,2]
        # Run MC
        mc!(lm, algo, rng, accepted[b], distortions[b], parities[b], dE[b],
            current_energy = en)
        # Compute minimum distortion
        mindist[b] = minimum(distortions[b])
        next!(prog)
    end
    (minimum_dist, beta_argmin) = findmin(mindist)

    return SAResults(parities=parities, distortion=minimum_dist,
        beta_argmin=algo.betas[beta_argmin,:], 
        acceptance_ratio=mean.(accepted), dE=dE)
end


#### Monte Carlo subroutines
function mc!(lm::LossyModel, algo::SA, rng::AbstractRNG,
    accepted::BitArray{1} = falses(algo.nsamples),
    distortions::Vector{Float64} = fill(NaN, algo.nsamples),
    parities::Vector{Int}=fill(typemax(Int),algo.nsamples),
    dE::Vector{Float64} = fill(NaN, algo.nsamples);
    current_energy::Real = energy(lm), 
    showprogress::Bool=true)

    for n in 1:algo.nsamples
        acc, deltaE = onemcstep!(lm, algo.mc_move, current_energy, rng)
        acc && (current_energy += deltaE)
        accepted[n] = acc
        dE[n] = deltaE
        # Check parity only for those MC moves that don't always stay on cw's
        par = typeof(algo.mc_move) in (MetropBasisCoeffs,MetropSmallJumps) ? 0 :
            parity(lm)
        # Save distortion only if parity is satisfied
        par == 0 && (distortions[n] = distortion(lm))
        parities[n] = par
    end
    return nothing
end

function onemcstep!(lm::LossyModel, mc_move::MCMove, current_energy::Real,
        rng::AbstractRNG=Random.MersenneTwister(0))
    to_flip, newvals = propose(mc_move, lm, rng)
    acc, dE = accept(mc_move, lm, to_flip, newvals, current_energy, rng)
    return acc, dE
end

# In general, accept by computing the full energy
# For particular choices of proposed new state, this method can be overridden for
#  something faster that exploits the fact that only a few terms in the energy change
function accept(mc_move::MCMove, lm::LossyModel, to_flip::Vector{Int},
        newvals::Vector, energy_old::Real, rng::AbstractRNG)
    xnew = copy(lm.x)
    xnew[to_flip] .= newvals
    dE = energy(lm, xnew) - energy_old
    acc = metrop_accept(dE, rng)
    acc && (lm.x = xnew)
    return acc, dE
end

function accept(mc_move::Metrop1, lm::LossyModelGF2, to_flip::Int,
    newval::Bool, energy_old::Real, rng::AbstractRNG)

    # COMPUTE ENERGY DIFFERENCE
    dE = 0.0
    # Look for neighbors of `to_flip` and compute delta energy for checks
    for a in lm.fg.Vneigs[to_flip]
        z = 0
        for w in lm.fg.Fneigs[a]    
            z = xor(z, lm.x[w])    
        end
        dE += lm.beta1*(1-2*z)
        @assert !isnan(dE)
    end
    # Delta energy for overlap
    dE += lm.beta2*(1-2*xor(lm.x[to_flip],lm.y[to_flip]))
    # ACCEPT OR NOT
    acc = metrop_accept(dE, rng)
    acc && (lm.x[to_flip] = newval)
    return acc, dE
end

function accept(mc_move::Union{MetropBasisCoeffs,MetropSmallJumps}, 
        lm::LossyModel, to_flip::Vector{Int},
        newvals::Vector, rng::AbstractRNG)
    # Compare energy shift only wrt those variables `to_flip`
    dE = energy_overlap(lm, newvals, sites=to_flip) - 
        energy_overlap(lm, lm.x[to_flip], sites=to_flip)
    acc = metrop_accept(dE, rng)
    acc && (lm.x[to_flip] .= newvals)
    return acc, dE
end

function accept(mc_move::Union{MetropBasisCoeffs,MetropSmallJumps}, 
        lm::LossyModelGF2, to_flip::AbstractVector,
        newvals::Vector, rng::AbstractRNG)
    # Compare energy shift only wrt those variables `to_flip`
    dE = 0
    for i in to_flip
        dE += 1-2*(xor(lm.x[i],lm.y[i]))
    end
    dE *= lm.beta2
    acc = metrop_accept(dE, rng)
    acc && (lm.x[to_flip] .= .!lm.x[to_flip])
    return acc, dE
end

# Changes value to 1 spin taken uniform random to a uniform random value in {0,...,q-1}
function propose(mc_move::Metrop1, lm::LossyModel, rng::AbstractRNG)
    # Pick a site at random
    to_flip = rand(rng, 1:lm.fg.n)
    # Pick a new value 
    newval = rand(rng, [k for k=0:lm.fg.q-1 if k!=lm.x[to_flip]])
    return to_flip, newval
end

function propose(mc_move::Metrop1, lm::LossyModelGF2, rng::AbstractRNG)
    # Pick a site at random
    to_flip = rand(rng, 1:lm.fg.n)
    # Pick a new value 
    newval = !lm.x[to_flip]
    return to_flip, newval
end

# Flips one basis coefficient at every move
function propose(mc_move::MetropBasisCoeffs, lm::LossyModel, rng::AbstractRNG)
    k = size(mc_move.basis,2)
    # Pick one coefficient to be flipped
    coeff_to_flip = rand(rng, 1:k)
    # Pick its new value in {1,2,...,q-1}
    coeff = rand(rng, 1:lm.fg.q-1)
    # Indices of variables appearing in the chosen basis vector
    basis_vector = mc_move.basis[:, coeff_to_flip]
    to_flip = findall(!iszero, basis_vector)
    # New values
    newvals = xor.(lm.x[to_flip], basis_vector[to_flip])
    return to_flip, newvals
end

# Flips a seaweed at every move. Only works for q = 2
function propose(mc_move::MetropSmallJumps, lm::LossyModel, rng::AbstractRNG)
    xnew = copy(lm.x)
    @assert lm.fg.q == 2 "Seaweed only doable for GF(q=2)"
    # Pick a site at random that is not in the core and start a seaweed from there
    not_in_core = (1:lm.fg.n)[.!mc_move.isincore]
    site = rand(rng, not_in_core)

    to_flip_bool=falses(lm.fg.n)
    seaweed(lm.fg, site, to_flip=to_flip_bool)
    to_flip = (1:lm.fg.n)[to_flip_bool]
    newvals = xor.(1, lm.x[to_flip])
    return to_flip, newvals
end

function metrop_accept(dE::Real, rng::AbstractRNG)::Bool
    if dE < 0
        return true
    else
        return rand(rng) < exp(-dE)
    end
end

function output_str(res::SAResults)
    out_str = 
    # "Parity " * string(res.parity) * ". " *
              "Distortion " * @sprintf("%.3f ", res.distortion) *
              "at β₁=" * string(res.beta_argmin[1]) * ", β₂=" * 
                string(round(res.beta_argmin[2],digits=3)) * 
            #   ". Acceptance: " * 
            #     string(res.acceptance_ratio) *
                "."
    return out_str
end


### Build a seaweed as described in https://arxiv.org/pdf/cond-mat/0207140.pdf

function seaweed(fg::FactorGraph, seed::Int, depths::Vector{Int}=lr(fg)[1], 
        isincore::BitArray{1} = (depths .== 0);
        to_flip::BitArray{1}=falses(fg.n))
    # Check that there is at least 1 leaf
    @assert nvarleaves(fg) > 0 "Graph must contain at least 1 leaf"
    # Check that seed is one of the variables in the graph
    @assert seed <= fg.n "Cannot use var $seed as seed since FG has $(fg.n) variables"
    # Check that seed is a variable outside the core    
    @assert !isincore[seed] "Cannot grow seaweed starting from a variable in the core"
    # Grow the seaweed
    grow!(fg, seed, 0, depths, to_flip, isincore)
    # check that the resulting seaweed satisfies parity
    to_flip_int = Int.(to_flip)
    @assert parity(fg, to_flip_int)==0
    return to_flip_int
end

function grow!(fg::FactorGraph, v::Int, f::Int, depths::Vector{Int}, 
        to_flip::BitArray{1}, isincore::BitArray{1})
    # flip v
    to_flip[v] = !(to_flip[v])
    # branches must grow in all directions except the one we're coming from,
    #  i.e. factor f
    neigs_of_v = fg.Vneigs[v]
    i = findfirst(ee->isbelow(ee, v, fg, depths) && ee!=f, neigs_of_v)
    # grow upwards branches everywhere except where you came from (factor f) and
    #  factor below (factor with index i)
    for (j,ee) in enumerate(neigs_of_v); if j!=i && ee!=f
        grow_upwards!(fg, v, ee, depths, to_flip, isincore)
    end; end
    # grow downwards to maximum 1 factor
    if !isnothing(i)
        grow_downwards!(fg, v, neigs_of_v[i], depths, to_flip, isincore)
    end
    return nothing
end

function grow_downwards!(fg::FactorGraph, v::Int, f::Int, depths::Vector{Int}, 
    to_flip::BitArray{1}, isincore::BitArray{1})
    # consider all neighbors except the one we're coming from (v)
    neigs_of_f = [w for w in fg.Fneigs[f] if (w != v && !isincore[w])]
    if !isempty(neigs_of_f)
        # if v is not the only neighbor of minimum depth, pick one of the others
        mindepth = minimum(depths[neigs_of_f])
        if depths[v]==mindepth
            argmindepth = findall(depths[neigs_of_f] .== mindepth)
            # pick at random one of the neighbors with equal min depth
            new_v = neigs_of_f[rand(argmindepth)]
            grow!(fg, new_v, f, depths, to_flip, isincore)
        else
            # find maximum depth
            maxdepth = maximum(depths[neigs_of_f])
            argmaxdepth = findall(depths[neigs_of_f] .== maxdepth)
            # pick at random one of the neighbors with equal max depth
            new_v = neigs_of_f[rand(argmaxdepth)]
            grow!(fg, new_v, f, depths, to_flip, isincore)
        end
    else
        println("Warning: no neighbors found going down from v to f")
    end
    return nothing
end

# function grow_upwards!(fg::FactorGraph, v::Int, f::Int, depths::Vector{Int}, 
#     to_flip::BitArray{1}, isincore::BitArray{1})
#     # consider all neighbors except the one we're coming from (v)
#     neigs_of_f = [w for w in fg.Fneigs[f] if (w != v && !isincore[w])]
#     if !isempty(neigs_of_f)
#         # find minimum depth
#         mindepth = minimum(depths[neigs_of_f])
#         argmindepth = findall(depths[neigs_of_f] .== mindepth)
#         # pick at random one of the nieghbors with equal min depth
#         new_v = neigs_of_f[rand(argmindepth)]
#         grow!(fg, new_v, f, depths, to_flip, isincore)
#     end
#     return nothing
# end

# function grow_downwards!(fg::FactorGraph, v::Int, f::Int, depths::Vector{Int}, 
#     to_flip::BitArray{1}, isincore::BitArray{1})
#     # consider all neighbors except the one we're coming from (v)
#     neigs_of_f = fg.Fneigs[f]
#     mindepth = typemax(Int); maxdepth = 0
#     argmindepth = -1
#     for w in neigs_of_f
#         if w != v && !isincore[w] 
#             if depths[w]<mindepth
#                 argmindepth = w
#                 mindepth = depths[w]
#             end
#             if depths[w]>maxdepth
#                 maxdepth = depths[w]
#             end
#         end
#         if depths[v]==mindepth
#             # if v is not the only neighbor of minimum depth, pick one of the others
#             grow!(fg, argmindepth, f, depths, to_flip, isincore)
#         else
#             argmaxdepth = [w for w in neigs_of_f if w != v && !isincore[w] && depths[w]==maxdepth]
#             if !isempty(argmaxdepth) 
#                 new_v = rand(argmaxdepth)
#                 grow!(fg, new_v, f, depths, to_flip, isincore)
#             end    
#         end
#     end
#     return nothing
# end


function grow_upwards!(fg::FactorGraph, v::Int, f::Int, depths::Vector{Int}, 
    to_flip::BitArray{1}, isincore::BitArray{1})
    mindepth = typemax(Int)
    neigs_of_f = fg.Fneigs[f]
    for w in neigs_of_f
        if w != v && !isincore[w] && depths[w]<mindepth
            mindepth = depths[w]
        end
    end
    argmindepth = [w for w in neigs_of_f if w != v && !isincore[w] && depths[w]==mindepth]
    if !isempty(argmindepth) 
        new_v = rand(argmindepth)
        grow!(fg, new_v, f, depths, to_flip, isincore)
    end
    return nothing
end

function isbelow(e::Int, v::Int, fg::FactorGraph, depths::Vector{Int})
    neigs_of_e = fg.Fneigs[e]
    mindepth_idx = argmin(depths[neigs_of_e])
    isbel = (neigs_of_e[mindepth_idx] == v)
    return isbel
end

function plot_seaweed(fg::FactorGraph, seed::Int)
    depths = lr(fg)[2]
    sw = seaweed(fg, seed, depths)
    sw_idx = (1:fg.n)[Bool.(sw)]
    highlighted_edges = Tuple{Int,Int}[]
    for v in sw_idx
        for f in fg.Vneigs[v]
            push!(highlighted_edges, (f,v))
        end
    end
    plot(fg, highlighted_edges=highlighted_edges, varnames = depths)
end
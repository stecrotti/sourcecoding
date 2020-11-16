#### Perform simulated annealing on a LossyModel object. Flexible for new MC moves to be added ####
using ProgressMeter

abstract type MCmove end
struct Metrop1 <: MCmove; end
@with_kw struct MetropBasisCoeffs <: MCmove
    basis::Array{Int,2}=Array{Int,2}(undef,0,0)
    basis_coeffs::Vector{Int}=Vector{Int}(undef,0)
end


#### SIMULATED ANNEALING
@with_kw struct SA <: LossyAlgo
    mc_move::MCmove = MetropBasisCoeffs()                                             # Single MC move
    # betas:: Array{Float64,2} = [0.1 .^ (3:-1:-1) ones(Float64, 5)]          # Cooling schedule
    betas:: Array{Float64,2} = [1.0 1.0]
    nsamples::Int = Int(1e2)                                                # Number of samples
    sample_every::Int = Int(1e0)                                            # Frequency of sampling
    stop_crit::Function = (crit(varargs...) = false)                        # Stopping criterion callback
    init_state::Function = (init(lm::LossyModel)=zeros(Int, lm.fg.n))       # Function to initialize internal state
end

# Quick constructor that adapts the number of iterations to the size of the problem
function SA(lm::LossyModel; kwargs...)
    nsamples = 10*(lm.fg.n-lm.fg.m)
    basis = nullspace(lm)
    mc_move = MetropBasisCoeffs(basis, zeros(Int,size(basis,2)))
    return SA(nsamples=nsamples, mc_move=mc_move; kwargs...)
end

function solve!(lm::LossyModel, algo::SA,
        distortions::Vector{Vector{Float64}}=[zeros(algo.nsamples) for _ in 1:size(algo.betas,1)];
        verbose::Bool=true)    
    # Initialize to the requested initial state
    lm.x = algo.init_state(lm)
    nbetas = size(algo.betas,1)
    min_dist = Inf
    argmin_beta = nbetas+1
    for b in 1:nbetas
        # Update temperature
        lm.beta1 = algo.betas[b,1]
        lm.beta2 = algo.betas[b,2]
        # Run MC
        distortions[b], acceptance_ratio = mc!(lm, algo, verbose=verbose,
            to_display="MC running β₁=$(lm.beta1), "*
            "β₂=$(lm.beta2)")
        (m,i) = findmin(distortions[b])
        if m < min_dist
            min_dist = m
            argmin_beta = b
        end
        # Check stopping criterion
        if algo.stop_crit(lm, distortions[b], acceptance_ratio)
            return (:stopped, min_dist, algo.betas[b,:], argmin_beta)
        end
        if verbose
            println("Temperature $b of $nbetas:",
            "(β₁=$(algo.betas[b,1]),β₂=$(algo.betas[b,2])).",
            "Energy = $(energy(lm)). Distortion = $(min_dist)")
        end
    end
    return (:finished, min_dist, algo.betas[nbetas], argmin_beta)
end


#### Monte Carlo subroutines

function mc!(lm::LossyModel, algo::SA; 
    verbose::Bool=false, to_display::String="Running Monte Carlo...")
    distortions = fill(+Inf, algo.nsamples)
    acceptance_ratio = zeros(algo.nsamples)
    # Initial energy
    en = energy(lm)
    verbose && (prog = Progress(algo.nsamples, to_display))
    for n in 1:algo.nsamples
        for it in 1:algo.sample_every
            acc, dE = onemcstep!(lm , algo.mc_move)
            en = en + dE*acc
            acceptance_ratio[n] += acc
        end
        acceptance_ratio[n] = acceptance_ratio[n]/algo.sample_every
        # Store distortion only if parity is fulfilled
        if sum(paritycheck(lm)) == 0
            distortions[n] = distortion(lm)
        end
        verbose && next!(prog)
    end
    return distortions, acceptance_ratio
end

function onemcstep!(lm::LossyModel, mc_move::MCmove)
    xnew = propose(mc_move, lm)
    acc, dE = accept(mc_move, lm, xnew)
    if acc
        lm.x = xnew
    end
    return acc, dE
end

# In general, accept by computing the full energy
# For particular choices of proposed new state, this method is overridden for
#  something faster that exploits the fact that only a few terms in the energy change
function accept(mc_move::MCmove, lm::LossyModel, xnew::Vector{Int})
    dE = energy(lm, xnew) - energy(lm, lm.x)
    return metrop_accept(dE), dE
end

# Changes value to 1 spin taken uniform random to a uniform random value in {0,...,q-1}
function propose(mc_move::Metrop1, lm::LossyModel)
    q = lm.fg.q
    n = lm.fg.n
    # current state
    x = lm.x
    # Pick a site at random
    site = rand(1:n)
    # Pick a new value for x[site]
    xnew = copy(x)
    xnew[site] = rand([k for k=0:q-1 if k!=x[site]])
    return xnew, site
end

# Accept according to the Metropolis rule for 1 spin flip
# function accept(mc_move::Metrop1, lm::LossyModel, xnew::Vector{Int}, site::Int)
#     # Evaluate energy delta E(xnew)-E(x)
#     dE_checks = sum(hw(paritycheck(lm.fg, xnew, f)) -
#         hw(paritycheck(lm.fg, lm.x, f)) for f in lm.fg.Vneigs[site])

#     dE_checks = energy_checks(lm, xnew, f=lm.fg.Vneigs[site]) -
#         energy_checks(lm, lm.x, f=lm.fg.Vneigs[site])

#     dE_overlap = (hd(xnew, lm.y) - hd(lm.x, lm.y))
#     dE_overlap = energy_overlap(lm, xnew) - energy_overlap(lm)
#     dE = dE_checks + dE_overlap
#     # Accept or not, return also delta energy
#     acc = metrop_accept(dE)
#     return acc, dE
# end

function propose(mc_move::MetropBasisCoeffs, lm::LossyModel)
    q = lm.fg.q
    k = log_nsolutions(lm)
    coeffs_old = mc_move.basis_coeffs
    # Pick one coefficient to be flipped
    to_be_flipped = rand(1:k)
    coeffs_new = copy(coeffs_old)
    # Flip
    coeffs_new[to_be_flipped] = rand([k for k=0:q-1 if k!=coeffs_old[to_be_flipped]])
    # Project on basis to get the new state
    x_new = mc_move.basis*coeffs_new
    return x_new
end

function metrop_accept(dE::Real)::Bool
    if dE < 0
        return true
    else
        r = rand()
        return r < exp(-dE)
    end
end

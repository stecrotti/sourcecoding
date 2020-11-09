#### Perform simulated annealing on a LossyModel object. Flexible for new MC moves to be added ####

abstract type MCmove end
struct Metrop1 <: MCmove; end

#### SIMULATED ANNEALING
@with_kw struct SA <: LossyAlgo
    mc_move::MCmove = Metrop1()                                             # Single MC move
    betas:: Array{Float64,2} = [0.1 .^ (3:-1:-1) ones(Float64, 5)]          # Cooling schedule
    nsamples::Int = Int(1e2)                                                # Number of samples
    sample_every::Int = Int(1e3)                                            # Frequency of sampling
    stop_crit::Function = (crit(varargs...) = false)                        # Stopping criterion callback
    init_state::Function = (init(lm::LossyModel)=rand(0:lm.fg.q-1,lm.fg.n)) # Function to initialize internal state
end

SA(lm::LossyModel) = SA(nsamples=10*lm.fg.n)

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
        distortions[b], acceptance_ratio = mc!(lm, algo)
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

function mc!(lm::LossyModel, algo::SA)
    distortions = fill(+Inf, algo.nsamples)
    acceptance_ratio = zeros(algo.nsamples)
    # Initial energy
    en = energy(lm)
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
    end
    return distortions, acceptance_ratio
end

function onemcstep!(lm::LossyModel, mc_move::MCmove)
    xnew, site = propose(mc_move, lm)
    acc, dE = accept(mc_move, lm, xnew, site)
    if acc
        lm.x = xnew
    end
    return acc, dE
end

# In general, accept by computing the full energy
# For particular choices of proposed new state, this method is overridden for
#  something faster that exploits the fact that only a few terms in the energy change
function accept(mc_move::MCmove, lm::LossyModel, xnew::Vector{Int})::Bool
    dE = energy(lm, xnew) - energy(lm, lm.x)
    return metrop_accept(dE), dE
end

# Changes value to 1 spin taken uniform random to a uniform random value in {0,...,q-1}
function propose(algo::Metrop1, lm::LossyModel)
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
function accept(algo::Metrop1, lm::LossyModel, xnew::Vector{Int}, site::Int)
    # Evaluate energy delta E(xnew)-E(x)
    dE_checks = sum(hw(paritycheck(lm.fg, xnew, f)) -
        hw(paritycheck(lm.fg, lm.x, f)) for f in lm.fg.Vneigs[site])

    dE_checks = energy_checks(lm, xnew, f=lm.fg.Vneigs[site]) -
        energy_checks(lm, lm.x, f=lm.fg.Vneigs[site])

    dE_overlap = (hd(xnew, lm.y) - hd(lm.x, lm.y))
    dE_overlap = energy_overlap(lm, xnew) - energy_overlap(lm)
    dE = dE_checks + dE_overlap
    # Accept or not, return also delta energy
    acc = metrop_accept(dE)
    return acc, dE
end

# struct Metrop2 <: MCmove; end
# # Changes value to 2 NEIGHBOR spins taken uniform random to a uniform random value in {0,...,q-1}
# function propose(algo::Metrop2, lm::LossyModel)
#     q = lm.fg.q
#     n = lm.fg.n
#     # current state
#     x = lm.x
#     # Pick 2 sites at random
#     site1 = rand(1:n)
#     site2 = rand([v for v in []])
#     return xnew, sites
# end
#
# function accept(algo::Metrop2, lm::LossyModel, xnew::Vector{Int}, site::Int)
#     # Evaluate energy delta E(xnew)-E(x)
#
#
#     # Safe for beta1=inf and dE_checks=0
#     if dE_checks == 0
#         dE = lm.beta2*dE_overlap
#     else
#         dE = lm.beta1*dE_checks + lm.beta2*dE_overlap
#     end
#     # Accept or not
#     return metrop_accept(dE)
# end

function metrop_accept(dE::Real)::Bool
    if dE < 0
        return true
    else
        r = rand()
        return r < exp(-dE)
    end
end

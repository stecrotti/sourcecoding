abstract type Metrop end

# In general, accept by computing the full energy
# For particular choices of proposed new state, this method is overridden for
#  something faster that exploits the fact that only a few terms in the energy change
function accept(algo::Metrop, lm::LossyModel, xnew::Vector{Int})::Bool
    dE = energy(lm, xnew) - energy(lm, lm.x)
    return metrop_accept(dE), dE
end

struct Metrop1 <: Metrop; end

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

function accept(algo::Metrop1, lm::LossyModel, xnew::Vector{Int}, site::Int)
    # Evaluate energy delta E(xnew)-E(x)
    dE_checks = sum(hw(paritycheck(lm.fg, xnew, f)) -
        hw(paritycheck(lm.fg, lm.x, f)) for f in lm.fg.Vneigs[site])
    dE_overlap = (hd(xnew, lm.y) - hd(lm.x, lm.y))

    # Safe for beta1=inf and dE_checks=0
    if dE_checks == 0
        dE = lm.beta2*dE_overlap
    else
        dE = lm.beta1*dE_checks + lm.beta2*dE_overlap
    end
    # Accept or not, return also delta energy
    acc = metrop_accept(dE)
    return acc, dE
end

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
# struct Metrop2 <: Metrop; end

function onemcstep!(lm::LossyModel, mc_algo::Metrop)
    xnew, site = propose(mc_algo, lm)
    acc, dE = accept(mc_algo, lm, xnew, site)
    if acc
        lm.x = xnew
    end
    return acc, dE
end

function mc!(lm::LossyModel, mc_algo::Metrop; nsamples::Int=Int(1e4),
                sample_every::Int=Int(1e2),
                x0::Vector{Int}=lm.x)
    energies = zeros(nsamples)
    acceptance_ratio = zeros(nsamples)
    lm.x = x0
    # Initial energy
    en = energy(lm)
    for n in 1:nsamples
        for it in 1:sample_every
            acc, dE = onemcstep!(lm , mc_algo)
            en = en + dE*acc
            acceptance_ratio[n] += acc
        end
        acceptance_ratio[n] = acceptance_ratio[n]/sample_every
        energies[n] = en
    end
    return energies, acceptance_ratio
end

struct SA; end      # Simulated Annealing

function anneal!(lm::LossyModel, mc_algo::Metrop,
                betas:: Array{<:Real,2};
                nsamples::Int=Int(1e4),
                sample_every::Int=Int(1e2), stop_crit = (args...)->false,
                x0::Vector{Int}=rand(0:lm.fg.q-1,lm.fg.n), verbose::Bool=true)
    # Initialize to the requested initial state
    lm.x = x0
    nbetas = size(betas,1)
    for b in 1:nbetas
        # Update temperature
        lm.beta1 = betas[b,1]
        lm.beta2 = betas[b,2]
        # Run MC
        energies, acceptance_ratio =
            mc!(lm, mc_algo, nsamples=nsamples, sample_every=sample_every, x0=lm.x)
        if stop_crit(lm, energies, acceptance_ratio)
            return (:stopped, betas[b,:], energy(lm))
        end
    if verbose
        println("Finished temperature $b of $nbetas: (β₁=$(betas[b,1]),β₂=$(betas[b,2])). Energy = $(energy(lm))")
    end
    end
    return (:finished, betas[nbetas], energy(lm))
end


function metrop_accept(dE::Real)::Bool
    if dE < 0
        return true
    else
        r = rand()
        return r < exp(-dE)
    end
end

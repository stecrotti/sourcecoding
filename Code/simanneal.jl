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

function onemcstep!(lm::LossyModel, algo::Metrop)
    xnew, site = propose(algo, lm)
    acc, dE = accept(algo, lm, xnew, site)
    if acc
        lm.x = xnew
    end
    return acc, dE
end

function mc!(lm::LossyModel, algo::Metrop; nsamples::Int=Int(1e4),
                sample_every::Int=Int(1e2),
                x0::Vector{Int}=lm.x)
    energies = zeros(nsamples)
    acceptance_ratio = zeros(nsamples)
    lm.x = x0
    # Initial energy
    en = energy(lm)
    for n in 1:nsamples
        for it in 1:sample_every
            acc, dE = onemcstep!(lm , algo)
            en = en + dE*acc
            acceptance_ratio[n] += acc
        end
        acceptance_ratio[n] = acceptance_ratio[n]/sample_every
        energies[n] = en
        # if stop_crit(lm, acceptance_ratio, energies)
        #      verbose && println("Stopping criterion reached after $n iters. E=$en")
        #     return :stopped, n, energies, acceptance_ratio
        # end
    end
    return energies, acceptance_ratio
end

function anneal!(lm::LossyModel, algo::Metrop,
                betas::Tuple{Vector{<:Real},Vector{<:Real}};
                nsamples::Int=Int(1e4),
                sample_every::Int=Int(1e2), stop_crit = (lm, ar, e)->false,
                x0::Vector{Int}=rand(0:lm.fg.q-1,lm.fg.n), verbose::Bool=true)
    # Initialize to the requested initial state
    lm.x = x0
    nbetas = length.(betas)[1]
    for b in 1:nbetas
        # Update temperature
        lm.beta1 = betas[1][b]
        lm.beta2 = betas[2][b]
        # Run MC
        energies, acceptance_ratio =
            mc!(lm, algo, nsamples=nsamples, sample_every=sample_every, x0=lm.x)
        if stop_crit(lm, energies, acceptance_ratio)
            return (:stopped, b, energy(lm))
        end
    end
    return (:finished, nbetas, energy(lm))
end


function metrop_accept(dE::Real)::Bool
    if dE < 0
        return true
    else
        r = rand()
        return r < exp(-dE)
    end
end

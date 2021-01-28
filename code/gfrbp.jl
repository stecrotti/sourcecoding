#### Reinforced belief propagation and max-sum on 𝔾𝔽(2ᵏ) ####
using Parameters, ProgressMeter, Printf

abstract type LossyAlgo end

# convenience method
beta2_init(algo::LossyAlgo) = 1.0

# Belief Propagation
@with_kw struct BP <: LossyAlgo
    maxiter::Int = Int(1e3)             # Max mun of iterations
    convergence::Symbol = :messages     # Convergence criterion
    @assert convergence in [:messages, :decvars, :parity]
    nmin::Int = 300                     # Min number of consecutive unchanged decision vars
    tol::Float64 = 1e-12                # Tol for messages convergence
    gamma::Float64 = 0.0                # Reinforcement
    Tmax::Int = 1                       # Max number of restarts with new random init
    beta2::Float64 = 1.0                # Inverse temperature for overlap energy
    sigma::Float64 = 1e-4               # Random noise on external fields
    default_distortion::Function=fix_indep_from_ms
end

# Max-sum
@with_kw struct MS <: LossyAlgo
    maxiter::Int = Int(1e3)             # Max mun of iterations
    convergence::Symbol = :parity       # Convergence criterion
    @assert convergence in [:messages, :decvars, :parity]
    nmin::Int = 300                     # Min number of consecutive unchanged decision vars
    tol::Float64 = 1e-12                # Tol for messages convergence
    gamma::Float64 = 1e-2               # Reinforcement
    Tmax::Int = 5                       # Max number of restarts with new random init
    beta2::Float64 = 1.0                # Inverse temperature for overlap energy
    sigma::Float64 = 1e-4               # Random noise on external fields
    default_distortion::Function=fix_indep_from_ms
end

beta2_init(algo::T) where {T<:Union{BP,MS}} = algo.beta2

abstract type LossyResults end

function output_str(res::LossyResults)
    out_str = "Parity " * string(res.parity) * ". " *
              "Dist " * @sprintf("%.4f ", res.distortion) * "."
    return out_str
end

@with_kw struct BPResults{T<:Union{BP,MS}} <: LossyResults
    converged::Bool = false
    parity::Int = 0
    distortion::Float64 = 1.0
    trials::Int = 0
    iterations::Int = 0
    maxdiff::Vector{Float64} = Vector{Float64}(undef,0)
    codeword::BitArray{1} = BitArray{1}(undef,0)
    maxchange::Vector{Float64} = Vector{Float64}(undef,0)
end

function output_str(res::BPResults{<:Union{BP,MS}})
    outcome_str = res.converged ? "C" : "U"
    out_str = outcome_str * " after " * 
            @sprintf("%4d", res.iterations) * " iters, " *
            @sprintf("%1d", res.trials) * " trials. " *
            "Parity " * @sprintf("%3d", res.parity) * ". " *
            "Dist " * @sprintf("%.3f", res.distortion) *
            "."
    return out_str
end

function onebpiter!(fg::FactorGraph, algo::BP, neutral=neutralel(algo,fg.q))

    maxdiff = diff = 0.0
    for f in randperm(length(fg.Fneigs))
        for (v_idx, v) in enumerate(fg.Fneigs[f])
            # Divide message from belief
            fg.fields[v] ./= fg.mfv[f][v_idx]
            # Restore possible n/0 or 0/0
            fg.fields[v][isnan.(fg.fields[v])] .= 1.0
            # Define functions for weighted convolution
            funclist = Fun[]
            weightlist = Int[]
            for (vprime_idx,vprime) in enumerate(fg.Fneigs[f])
                if vprime != v
                    func = fg.fields[vprime] ./ fg.mfv[f][vprime_idx]
                    func[isnan.(func)] .= 1.0
                    # adjust for weights
                    # func .= func[fg.mult[fg.gfinv[fg.hfv[f][vprime_idx]], fg.mult[fg.hfv[f][v_idx],:]]]
                    push!(funclist, func)
                    push!(weightlist, fg.hfv[f][vprime_idx])
                end
            end
            fg.mfv[f][v_idx] = gfconvw(funclist, fg.gfdiv, weightlist,
                neutral)
            # Adjust final weight
            fg.mfv[f][v_idx] .= fg.mfv[f][v_idx][ fg.mult[fg.hfv[f][v_idx],:] ]

            fg.mfv[f][v_idx][isnan.(fg.mfv[f][v_idx])] .= 0.0
            if sum(isnan.(fg.mfv[f][v_idx])) > 0
                println()
                @show funclist
                @show fg.mfv[f][v_idx]
                error("NaN in message ($f,$v)")
            end
            # Normalize message
            if !isinf(sum(fg.mfv[f][v_idx]))
                fg.mfv[f][v_idx] ./= sum(fg.mfv[f][v_idx])
            end
            # Update belief after updating the message
            fg.fields[v] .*=  fg.mfv[f][v_idx]
            # Normalize belief
            if sum(fg.fields[v])!= 0
                fg.fields[v] ./= sum(fg.fields[v])
            end
            # Check for NaNs
            if sum(isnan.(fg.fields[v])) > 0
                @show funclist
                @show fg.mfv[f][v_idx]
                error("Belief $v has a NaN")
            end
            # Look for maximum message (difference)
            diff = abs(fg.mfv[f][v_idx][0]-fg.mfv[f][v_idx][1])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return guesses(fg), maxdiff
end

function onebpiter!(fg::FactorGraph, algo::MS,
    neutral=neutralel(algo,fg.q))

    maxdiff = diff = 0.0
    for f in randperm(length(fg.Fneigs))
        for (v_idx, v) in enumerate(fg.Fneigs[f])
            # Subtract message from belief
            fg.fields[v] .-= fg.mfv[f][v_idx]
            # if "Inf-Inf=NaN" happened, restore 0.0
            replace!(fg.fields[v], NaN=>0.0)
            # Define functions for weighted convolution
            funclist = [fg.fields[vprime] - fg.mfv[f][vprime_idx] 
                for (vprime_idx, vprime) in enumerate(fg.Fneigs[f])
                if vprime != v]
            weights = [fg.H[f,vprime] 
                for (vprime_idx, vprime) in enumerate(fg.Fneigs[f])
                if vprime != v]

            fg.mfv[f][v_idx] .= gfmscw(funclist, fg.gfdiv, weights,
                neutral)
            # Adjust final weight
            fg.mfv[f][v_idx] .= fg.mfv[f][v_idx][ fg.mult[fg.H[f,v],:] ]
            # Normalize message
            fg.mfv[f][v_idx] .-= maximum(fg.mfv[f][v_idx])
            replace!(fg.mfv[f][v_idx], NaN=>0.0)
            # Update belief after updating the message
            fg.fields[v] .+= fg.mfv[f][v_idx]
            # Normalize belief
            fg.fields[v] .-= maximum(fg.fields[v])
            replace!(fg.fields[v], NaN=>0.0)
            # Look for maximum message (difference)
            diff = abs(fg.mfv[f][v_idx][0]-fg.mfv[f][v_idx][1])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return maxdiff
end

function onebpiter!(fg::FactorGraphGF2, algo::MS,
    neutral=neutralel(algo,fg.q))
    maxdiff = diff = 0.0
    aux = Float64[]
    # Loop over factors
    for f in randperm(fg.m)
        # Loop over neighbors of `f`
        for (v_idx, v) in enumerate(fg.Fneigs[f])
            # Subtract message from belief
            fg.fields[v] -= fg.mfv[f][v_idx]          
            # Collect (var->fact) messages from the other neighbors of `f`
            aux = [fg.fields[vprime] - fg.mfv[f][vprime_idx] 
                for (vprime_idx,vprime) in enumerate(fg.Fneigs[f]) 
                if vprime_idx != v_idx]
            # Apply formula to update message
            fg.mfv[f][v_idx] = prod(sign, aux)*reduce(min, abs.(aux), init=Inf)
            # Update belief after updating the message
            fg.fields[v] += fg.mfv[f][v_idx]
            # Look for maximum message
            diff = abs(fg.mfv[f][v_idx])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return maxdiff
end

function onebpiter_fast!(fg::FactorGraphGF2, algo::MS, neutral=neutralel(algo,fg.q))
    maxdiff = 0.0
    # Loop over factors
    for f in randperm(fg.m)
        fmin = fmin2 = Inf
        imin = 1
        s = 1.0
        # Loop over neighbors of `f`, computing:
        # - prod of signs `s`
        # - first min (and index) and second min of abs values of messages
        for (i, v) in enumerate(fg.Fneigs[f])
            # Subtract message from belief
            fg.fields[v] -= fg.mfv[f][i]
            s *= sign(fg.fields[v])
            m = abs(fg.fields[v])
            if fmin > m
                fmin = m
                imin = i
            elseif fmin2 > m
                fmin2 = m
            end
        end
        for (i, v) in enumerate(fg.Fneigs[f])
            # Apply formula to update message
            m = (i == imin ? fmin2 : fmin) * s * sign(fg.fields[v])
            fg.mfv[f][i] = m
            # Update belief after updating the message
            fg.fields[v] += m
            # Look for maximum message
            maxdiff = max(maxdiff, abs(m))
        end
    end
    maxdiff
end

function guesses(beliefs::AbstractVector)
    return [findmax(b)[2] for b in beliefs]
end
guesses(fg::FactorGraph) = guesses(fg.fields)
guesses(fg::FactorGraphGF2) = floor.(Int, 0.5*(1 .- sign.(fg.fields)))

function bp!(fg::FactorGraph, algo::Union{BP,MS}, y::Vector{Int},
    maxdiff=zeros(algo.maxiter), codeword=falses(algo.maxiter),
    maxchange=zeros(algo.maxiter); randseed::Int=0, neutral=neutralel(algo,fg.q),
    verbose::Bool=false, showprogress::Bool=verbose)

    randseed != 0 && Random.seed!(randseed)      # for reproducibility
    newguesses = zeros(Int,fg.n)
    oldguesses = guesses(fg)
    oldmessages = deepcopy(fg.mfv)
    newmessages = deepcopy(fg.mfv)
    par = parity(fg)
    wait_time = showprogress ? 1 : Inf
    n = 0
    
    refresh!(fg)
    extfields!(fg,y,algo,randseed=randseed)

    for trial in 1:algo.Tmax
        prog = ProgressMeter.Progress(algo.maxiter, wait_time, 
            "Trial $trial/$(algo.Tmax) ")
        for t in 1:algo.maxiter
            maxdiff[t] = onebpiter!(fg, algo, neutral)
            newguesses .= guesses(fg)
            newmessages .= fg.mfv
            par = parity(fg, newguesses)
            codeword[t] = (par==0)
            if algo.convergence == :messages
                for f in eachindex(newmessages)
                    for (v_idx,msg) in enumerate(newmessages[f])
                        change = maximum(abs,msg - oldmessages[f][v_idx])
                        if change > maxchange[t]
                            maxchange[t] = change
                        end
                    end
                end
                if maxchange[t] <= algo.tol
                    return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y), trials=trial, iterations=t,
                        maxdiff=maxdiff, codeword=codeword, maxchange=maxchange)
                end
                oldmessages .= deepcopy(newmessages)
            elseif algo.convergence == :decvars
                if newguesses == oldguesses
                    n += 1
                    if n >= algo.nmin
                        return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y), trials=trial, iterations=t,
                        maxdiff=maxdiff, codeword=codeword, maxchange=maxchange)
                    end
                else
                    n=0
                end
                oldguesses .= newguesses
            elseif algo.convergence == :parity
                if par == 0
                    showprogress && println()
                    return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y), trials=trial, iterations=t,
                        maxdiff=maxdiff, codeword=codeword, maxchange=maxchange)
                end
            else
                error("Field convergence must be one of :messages, :decvars, :parity")
            end
            algo.gamma != 0 && reinforce!(fg, algo)
            ProgressMeter.next!(prog)
        end
        if trial != algo.Tmax
            # If convergence not reached, re-initialize random fields and start again
            refresh!(fg)
            extfields!(fg,y,algo,randseed=randseed+trial)
            oldguesses .= guesses(fg)
            oldmessages .= deepcopy(fg.mfv)
            fill!(maxchange, -Inf)
            n = 0
        end
    end
    return BPResults{typeof(algo)}(converged=false, parity=par,
                distortion=algo.default_distortion(fg,y), trials=algo.Tmax, 
                iterations=algo.maxiter, maxdiff=maxdiff, codeword=codeword, 
                maxchange=maxchange)
end

function solve!(lm::LossyModel, algo::Union{BP,MS}, args...; randseed::Int=0,
    verbose::Bool=false, kwargs...)
    output = bp!(lm.fg, algo, lm.y, args...; randseed=randseed, kwargs...)
    lm.x = guesses(lm.fg)
    return output
end

function reinforce!(fg::FactorGraph, algo::Union{BP,MS})
    for (v,gv) in enumerate(fg.fields)
        if algo.gamma > 0
            if typeof(algo)==BP
                fg.fields[v] .*= gv.^algo.gamma
                # Normalize belief
                if sum(fg.fields[v])!= 0
                    fg.fields[v] ./= sum(fg.fields[v])
                end
            elseif typeof(algo)==MS
                fg.fields[v] .+= gv*algo.gamma
            end
        end
    end
    return nothing
end
function reinforce!(fg::FactorGraphGF2, algo::Union{BP,MS})
    for (v,gv) in enumerate(fg.fields)
        if algo.gamma > 0
            if typeof(algo)==BP
                fg.fields[v] *= gv.^algo.gamma
            elseif typeof(algo)==MS
                fg.fields[v] += gv*algo.gamma
            end
        end
    end
    return nothing
end

neutralel(algo::BP, q::Int) = Fun(x == 0 ? 1.0 : 0.0 for x=0:q-1)
neutralel(algo::MS, q::Int) = Fun(x == 0 ? 0.0 : -Inf for x=0:q-1)

# Creates fields for the priors: the closest to y, the stronger the field
# The prior distr is given by exp(field)
# A small noise with amplitude sigma is added to break the symmetry
function extfields!(fg::FactorGraph, y::Vector{Int}, algo::Union{BP,MS}; randseed::Int=0)
    randseed != 0 && Random.seed!(randseed) 
    q = fg.q
    if q > 2
        fields = [OffsetArray(fill(0.0, q), 0:q-1) for v in eachindex(y)]
        for v in eachindex(fields)
            for a in 0:q-1
                fields[v][a] = -algo.beta2*hd(a,y[v]) + algo.sigma*randn()
                typeof(algo)==BP && (fields[v][a] = exp.(fields[v][a]))
                fg.fields .= fields
            end
        end
    else
        fields = algo.beta2*(1 .- 2*y) + algo.sigma*randn(length(y))
        fg.fields .= fields
    end
    return nothing
end

# Re-initialize messages
function refresh!(fg::FactorGraph)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= [OffsetArray(1/fg.q*ones(fg.q), 0:fg.q-1) 
            for v in eachindex(fg.mfv[f])]
    end
    return nothing
end
function refresh!(fg::FactorGraphGF2)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= 0.0
    end
    return nothing
end

function refresh!(fg::FactorGraph, y::Vector{Int},
    algo::Union{BP,MS}=MS(); randseed::Int=0)
    refresh!(fg)
    extfields!(fg, y, algo, randseed=randseed)
    return nothing
end


function extfields!(lm::LossyModel, algo::Union{BP,MS}; randseed::Int=0)
    extfields!(lm.fg, lm.y, algo, randseed=randseed)
end

function distortion(fg::FactorGraph, y::Vector{Int}, x::Vector{Int}=guesses(fg))
    return hd(x,y)/(fg.n*log2(fg.q))
end

#### Distortion for non-converged instances
naive_compression_distortion(fg::FactorGraph,args...;kw...) = 0.5*(nfacts(fg)/nvars(fg))

# Fix the independent variables to their value in the source vector
# Pass x as an argument to then be able to retrieve it
function fix_indep_from_src(fg::FactorGraph, y::Vector{Int}, 
        x::Vector{Int}=zeros(Int, fg.n); 
        independent::BitArray{1}=falses(fg.n), basis=lightbasis(fg, independent))
    x .= _fix_indep(fg,y,x, basis=basis, independent=independent)
    return distortion(fg, y, x)
end

# Fix the independent variables to the decision variables outputted by max-sum
# Pass x as an argument to then be able to retrieve it
function fix_indep_from_ms(fg::FactorGraph, y::Vector{Int}, 
        x::Vector{Int}=zeros(Int, fg.n); 
        independent::BitArray{1}=falses(fg.n), basis=lightbasis(fg, independent))
    x .= _fix_indep(fg, guesses(fg), x, basis=basis, independent=independent)
    return distortion(fg, y, x)
end

function _fix_indep(fg::FactorGraph, z::Vector{Int}, x::Vector{Int};
    basis::AbstractArray{Int,2}, independent::BitArray{1})

    x[independent] .= z[independent]
    x[.!independent] .= gfmatrixmult(basis[.!independent,:],  x[independent], 
        fg.q, fg.mult)
    return x
end

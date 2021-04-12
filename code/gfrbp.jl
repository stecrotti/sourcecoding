#### Reinforced belief propagation and max-sum on 𝔾𝔽(2ᵏ) ####
using Parameters, ProgressMeter, Printf

abstract type LossyAlgo end

# convenience method
beta2_init(algo::LossyAlgo) = 1.0

# Belief Propagation
@with_kw struct BP <: LossyAlgo
    maxiter::Int = Int(1e3)             # Max num of iterations
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
    maxiter::Int = Int(1e3)             # Max num of iterations
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

function onebpiter_slow!(fg::FactorGraph, algo::BP, neutral=neutralel(algo,fg.q))

    maxdiff = diff = 0.0
    for f in randperm(length(fg.Fneigs))
        for (v_idx, v) in enumerate(fg.Fneigs[f])
            # Divide message from belief
            fg.fields[v] /= fg.mfv[f][v_idx]
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
    neutral=neutralel(algo,fg.q); fact_perm = randperm(fg.m))

    maxdiff = diff = 0.0
    for f in fact_perm
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

function onebpiter_slow!(fg::FactorGraphGF2, algo::MS,
    neutral=neutralel(algo,fg.q);
    fact_perm = randperm(fg.m))
    maxdiff = diff = 0.0
    aux = Float64[]
    # Loop over factors
    for f in fact_perm
        # if degree 1, just forces to +1 its only neighbor
        if length(fg.Fneigs[f])==1
            fg.fields[only(fg.Fneigs[f])] = Inf
            continue
        end
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

struct Prod{T}
    p::T
    n::Int  # number of zeros
end
Prod{T}() where T = Prod(one(T), 0)
Base.:*(P::Prod{T}, x) where T = iszero(x) ? Prod(P.p, P.n+1) : Prod(P.p * x, P.n)
function Base.:/(P::Prod{T}, x) where T
    if iszero(x)
        return P.n==1 ? P.p : zero(T)
    else
        return P.n==0 ? P.p/x : zero(T)
    end
end

function onebpiter!(fg::FactorGraphGF2, algo::MS, neutral=neutralel(algo,fg.q);
        fact_perm = randperm(fg.m))
    maxchange = 0.0
    # Loop over factors
    for f in fact_perm
        # if degree 1, just forces to +1 its only neighbor
        if length(fg.Fneigs[f])==1
            fg.fields[only(fg.Fneigs[f])] = Inf
            continue
        end
        fmin = fmin2 = Inf
        imin = 1
        s = Prod{Int}()
        # Loop over neighbors of `f`, computing:
        # - prod of signs `s`
        # - first min (and index) and second min of abs values of messages
        for (i, v) in enumerate(fg.Fneigs[f])
            # Subtract message from belief
            fg.fields[v] -= fg.mfv[f][i]
            s *= sign(fg.fields[v])
            m = abs(fg.fields[v])
            if fmin > m
                fmin2 = fmin
                fmin = m
                imin = i
            elseif fmin2 > m
                fmin2 = m
            end
        end
        for (i, v) in enumerate(fg.Fneigs[f])
            # Apply formula to update message
            m = (i == imin ? fmin2 : fmin) * (s / sign(fg.fields[v]))
            isinf(m) && error("m infinite. Degree(a)=", length(fg.Fneigs[f]),
                ". i=$i, v=$v, neigs=$(fg.Fneigs[f]). sign=", s / sign(fg.fields[v]))
            # Look for maximum change in message
            maxchange = max(maxchange, abs(m-fg.mfv[f][i]))
            fg.mfv[f][i] = m
            # Update belief after updating the message
            # fg.fields[v] + m == 0 && @show fg.fields[v], m, imin, fmin
            fg.fields[v] += m
            isnan(fg.fields[v]) && error("NaN in field")
        end
    end
    maxchange
end

function onebpiter_old!(fg::FactorGraphGF2, algo::BP, neutral=neutralel(algo,fg.q);
    fact_perm = randperm(fg.m))
    maxchange = 0.0
    # Loop over factors
    for f in fact_perm
        # if degree 1, just forces to +1 its only neighbor
        if length(fg.Fneigs[f])==1
            fg.fields[only(fg.Fneigs[f])] = Inf
            continue
        end
        t = Prod{Float64}()
        for (i, v) in enumerate(fg.Fneigs[f])
            # Avoid Inf-Inf=NaN
            if fg.fields[v] == fg.mfv[f][i] 
                fg.fields[v] = 0.0
            else
                fg.fields[v] -= fg.mfv[f][i]
            end
            # isnan((t*tanh(fg.fields[v])).p) && @show t, tanh(fg.fields[v]) 
            t *= tanh(fg.fields[v])
        end
        for (i, v) in enumerate(fg.Fneigs[f])
            m = atanh(t/tanh(fg.fields[v]))
            # isnan(m) && @show t,tanh(fg.fields[v])
            # Look for maximum change in message, avoid Inf-Inf
            m != fg.mfv[f][i] && (maxchange = max(maxchange, abs(m-fg.mfv[f][i])))
            # Update belief after updating the message
            if isnan(fg.fields[v]+m) 
                @show f,v,fg.fields[v],m 
                # error("NaN in field")
                return -1.0
            end
            fg.mfv[f][i] = m
            fg.fields[v] += m
        end
    end
    maxchange
end

function onebpiter!(fg::FactorGraphGF2, algo::BP, neutral=neutralel(algo,fg.q);
        fact_perm = randperm(fg.m))
    maxchange = 0.0
    for f in fact_perm
        t = Prod{Float64}()
        for (i, v) in enumerate(fg.Fneigs[f])
            if fg.fields[v] == fg.mfv[f][i]
                fg.fields[v] = 0.0
            else
                fg.fields[v] = (fg.fields[v]-fg.mfv[f][i])/(1-fg.fields[v]*fg.mfv[f][i])
            end
            t *= fg.fields[v]
        end
        for (i, v) in enumerate(fg.Fneigs[f])
            m = t/fg.fields[v]
            maxchange = max(maxchange, abs(m-fg.mfv[f][i]))
            fg.mfv[f][i] = m
            newfield = (m+fg.fields[v])/(1+m*fg.fields[v])
            if isnan(newfield)
                return -1.0
            end
            fg.fields[v] = newfield
            # abs(fg.fields[v]) > 1 && error("Something that should be tanh has abs >1")
        end
    end
    maxchange
end

function guesses(beliefs::AbstractVector)
    return argmax.(beliefs)
end
function guesses(fg::FactorGraph, g::Vector{Int}=zeros(Int,fg.n))     
    g .= guesses(fg.fields)
end
function guesses(fg::FactorGraphGF2, g::Vector{Int}=zeros(Int,fg.n)) 
    for i in eachindex(g)
        g[i] = fg.fields[i] < 0 ? 1 : 0  
    end
    return g
end

function bp!(fg::FactorGraph, algo::Union{BP,MS}, y::AbstractVector,
    codeword=falses(algo.maxiter),
    maxchange=fill(NaN, algo.maxiter); randseed::Int=0, neutral=neutralel(algo,fg.q),
    verbose::Bool=false, showprogress::Bool=verbose, oneiter!::Function=onebpiter!, 
    independent::BitArray{1}=falses(fg.n), basis=lightbasis(fg, independent)[1])

    randseed != 0 && Random.seed!(randseed)      # for reproducibility
    newguesses = zeros(Int,fg.n)
    oldguesses = guesses(fg)
    par = parity(fg)
    wait_time = showprogress ? 1 : Inf
    n = 0
    
    fact_perm = randperm(fg.m)

    for trial in 1:algo.Tmax
        prog = ProgressMeter.Progress(algo.maxiter, wait_time, 
            "Trial $trial/$(algo.Tmax) ")
        for t in 1:algo.maxiter
            maxchange[t] = oneiter!(fg, algo, neutral, fact_perm=fact_perm)
            shuffle!(fact_perm)
            newguesses .= guesses(fg, newguesses)
            par = parity(fg, newguesses)
            codeword[t] = (par==0)
            if algo.convergence == :messages
                if maxchange[t] <= algo.tol
                    return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y, newguesses), trials=trial, iterations=t,
                        codeword=codeword, maxchange=maxchange)
                end
            elseif algo.convergence == :decvars
                if newguesses == oldguesses
                    n += 1
                    if n >= algo.nmin
                        return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y, newguesses), trials=trial, iterations=t,
                        codeword=codeword, maxchange=maxchange)
                    end
                else
                    n=0
                end
            elseif algo.convergence == :parity
                if par == 0
                    showprogress && println()
                    return BPResults{typeof(algo)}(converged=true, parity=par,
                        distortion=distortion(fg, y, newguesses), trials=trial, iterations=t,
                        codeword=codeword, maxchange=maxchange)
                end
            else
                error("Field convergence must be one of :messages, :decvars, :parity")
            end
            newguesses,oldguesses = oldguesses,newguesses
            algo.gamma != 0 && reinforce!(fg, algo)
            ProgressMeter.next!(prog)
        end
        if trial != algo.Tmax
            # If convergence not reached, re-initialize random fields and start again
            refresh!(fg, algo)
            extfields!(fg,y,algo,randseed=randseed+trial)
            fill!(maxchange, NaN)
            n = 0
        end
    end
    return BPResults{typeof(algo)}(converged=false, parity=par,
                distortion=algo.default_distortion(fg,y,independent=independent,
                basis=basis), trials=algo.Tmax, 
                iterations=algo.maxiter, codeword=codeword, 
                maxchange=maxchange)
end

function solve!(lm::LossyModel, algo::Union{BP,MS}, args...; randseed::Int=0,
        verbose::Bool=false, beta=lm.beta2, kwargs...)
    refresh!(lm.fg, algo)
    extfields!(lm.fg,lm.y,algo,randseed=randseed)
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
                fg.fields[v] += gv.*algo.gamma
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
function extfields!(fg::FactorGraph, y::AbstractVector, algo::Union{BP,MS}; 
        randseed::Int=0)
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
        fg.fields .= algo.beta2*(1 .- 2*y) + algo.sigma*randn(length(y))
    end
    return nothing
end

# Re-initialize messages
function refresh!(fg::FactorGraph, algo)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= [OffsetArray(1/fg.q*ones(fg.q), 0:fg.q-1) 
            for v in eachindex(fg.mfv[f])]
    end
    return nothing
end
function refresh!(fg::FactorGraphGF2, algo::MS)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= 0.0
    end
    return nothing
end
function refresh!(fg::FactorGraphGF2, algo::BP)
    for f in eachindex(fg.mfv)
        fg.mfv[f] .= algo.sigma*randn(length(fg.mfv[f]))
    end
    return nothing
end

function refresh!(fg::FactorGraph, y::AbstractVector,
    algo::Union{BP,MS}=MS(); randseed::Int=0)
    refresh!(fg)
    extfields!(fg, y, algo, randseed=randseed)
    return nothing
end


function extfields!(lm::LossyModel, algo::Union{BP,MS}; randseed::Int=0)
    extfields!(lm.fg, lm.y, algo, randseed=randseed)
end

function distortion(fg::FactorGraph, y::AbstractVector, x::AbstractVector=guesses(fg))
    return hd(x,y)/(fg.n*log2(fg.q))
end

function distortion(fg::FactorGraphGF2, y::AbstractVector, x::AbstractVector=guesses(fg))
    d = 0.0
    for (xx,yy) in zip(x,y)
        d += sign(xx)!=sign(yy)
    end
    d/fg.n
end

#### Distortion for non-converged instances
naive_compression_distortion(fg::FactorGraph,args...;kw...) = 0.5*(nfacts(fg)/nvars(fg))

# Fix the independent variables to their value in the source vector
# Pass x as an argument to then be able to retrieve it
function fix_indep_from_src(fg::FactorGraph, y::AbstractVector, 
        x::AbstractVector=zeros(Int, fg.n); 
        independent::BitArray{1}=falses(fg.n), basis=lightbasis(fg, independent)[1])
    x .= _fix_indep(fg,y,x, basis, independent)
    return distortion(fg, y, x)
end

# Fix the independent variables to the decision variables outputted by max-sum
# Pass x as an argument to then be able to retrieve it
function fix_indep_from_ms(fg::FactorGraph, y::AbstractVector, 
        x=typeof(fg)==FactorGraphGF2 ? falses(fg.n) : zeros(Int, fg.n); 
        independent::BitArray{1}=falses(fg.n), basis=lightbasis(fg, independent)[1])
    x .= _fix_indep(fg, guesses(fg), x, basis, independent)
    return distortion(fg, y, x)
end

function _fix_indep(fg::FactorGraph, z::Vector{Int}, x::Vector{Int},
    basis, independent::BitArray{1})

    x[independent] .= z[independent]
    x[.!independent] .= gfmatrixmult(basis[.!independent,:],  x[independent], 
        fg.q, fg.mult)
    return x
end

function _fix_indep(fg::FactorGraphGF2, z, x, basis, independent::BitArray{1})

    x[independent] .= z[independent]
    x[.!independent] .= basis[.!independent,:] * x[independent] .% 2
    return x
end

#### DECIMATION
function performance(fg::FactorGraph, fields, σ=ones(Int,fg.n), x=falses(fg.n))
    σ .= sign.(fg.fields)
    x .= σ .== -1
    nunsat = parity(fg, x)
    dist = distortion(fg, fields, σ)
    ovl = 1-2*dist
    nunsat, ovl, dist
end

function decimate1!(fg, fields, freevars; ndec=1, maxiter=200, F=1.0, verbose=true, tol=1e-12)
    ε = 0.0
    iters = 0
    for f in 1:m; fg.mfv[f] .= zeros(length(fg.mfv[f])) end
    fg.fields .= copy(fields)  
    # pre-allocate for speed
    fact_perm = randperm(fg.m); σ=ones(Int, fg.n); x=falses(fg.n)
    for it in 1:maxiter
        ε = onebpiter!(fg, BP(); fact_perm=fact_perm) 
        if ε==-1
            return -2, NaN, NaN, it
        end
        if ε < tol; (iters = it; break) else; iters=maxiter end 
        shuffle!(fact_perm)
    end
    nunsat, ovl, dist = performance(fg, fields, σ, x)
    if verbose 
        @printf(" Step   0. Free = %3d. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
            length(freevars), ε, nunsat,  ovl, iters)
    end
    cnt = 1
    iters = 0
    while !isempty(freevars)
        sort!(freevars, by=j->abs(fg.fields[j]))
        freevars, tofix = freevars[1:end-ndec], freevars[max(1,end-ndec+1):end]
        for tf in tofix; fg.fields[tf] == 0 && (fg.fields[tf] = fields[tf]) end
        fg.fields[tofix] .= F*sign.(fg.fields[tofix])
        for it in 1:maxiter
            ε = onebpiter!(fg, BP(); fact_perm=fact_perm)
            ε == -1.0 && return -1, NaN, NaN, it   # when a contradiction is found
            if ε < tol; (iters = it; break) else; iters=maxiter end 
            shuffle!(fact_perm)
        end
        nunsat, ovl, dist = performance(fg, fields, σ, x)
        if verbose 
            @printf(" Step %3d. Fixing %6s. Free = %3d. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
                cnt, string(tofix), length(freevars), ε, nunsat,  ovl, iters)
        end
        nunsat == 0 && return nunsat, ovl, dist, iters
        cnt += 1
    end
    nunsat, ovl, dist, iters
end

function decimate!(fg, fields, free=collect(1:fg.n); Tmax=1, kw...)
    freevars = copy(free)
    for t in 1:Tmax
        freevars .= copy(free)
        nunsat, ovl, dist, iters = decimate1!(fg, fields, freevars; kw...)
        str = "$nunsat unsat"
        nunsat==-1 && (str="contradiction found after $iters iters")
        nunsat==-2 && (str="contradiction found already in the first BP run before decimation, after $iters iters")
        print("# Trial $t of $Tmax: ", str)
        if nunsat == 0 
            if all(sign.(fg.fields) .!= 0)
                println()
                return nunsat, ovl, dist
            else
               print(" but ", sum(sign.(fg.fields).==0), " undecided") 
            end
        end
        println()
    end
    println("No zero-unsat found after $Tmax trials")
    return -1, NaN, NaN
end
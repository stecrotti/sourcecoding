struct MS; end
struct BP; end

function onebpiter!(FG::FactorGraph, algo::BP, neutral=neutralel(algo,FG.q);
    alpha = 0)

    maxdiff = diff = 0.0
    for f in randperm(length(FG.Fneigs))
        for (v_idx, v) in enumerate(FG.Fneigs[f])
            # If message is a delta, don't to anything, that variable is already decided
            # FG.mfv[f][v_idx]==neutral  && continue

            # Divide message from belief
            FG.fields[v] ./= FG.mfv[f][v_idx]

            # Restore possible n/0 or 0/0
            FG.fields[v][isnan.(FG.fields[v])] .= 1.0

            # Define functions for weighted convolution
            funclist = Fun[]
            for (vprime_idx,vprime) in enumerate(FG.Fneigs[f])
                if vprime != v
                    func = FG.fields[vprime] ./ FG.mfv[f][vprime_idx]
                    func[isnan.(func)] .= 1.0
                    # adjust for weights
                    func .= func[FG.mult[FG.mult[FG.hfv[f][v_idx],FG.gfinv[FG.hfv[f][vprime_idx]]],:]]
                    push!(funclist, func)
                end
            end
            FG.mfv[f][v_idx] = reduce(gfconv, funclist, init=neutral)
            FG.mfv[f][v_idx][isnan.(FG.mfv[f][v_idx])] .= 0.0
            if sum(isnan.(FG.mfv[f][v_idx])) > 0 
                println()
                @show funclist
                @show FG.mfv[f][v_idx]
                error("NaN in message ($f,$v)")
            end
            # Normalize message
            if !isinf(sum(FG.mfv[f][v_idx]))
                FG.mfv[f][v_idx] ./= sum(FG.mfv[f][v_idx])
            end
            # Update belief after updating the message
            FG.fields[v] .*=  FG.mfv[f][v_idx]
            # Normalize belief
            if sum(FG.fields[v])!= 0
                FG.fields[v] ./= sum(FG.fields[v])
            end
            # Check for NaNs
            if sum(isnan.(FG.fields[v])) > 0
                @show funclist
                @show FG.mfv[f][v_idx]
                error("Belief $v has a NaN")
            end
            # Look for maximum message (difference)
            diff = abs(FG.mfv[f][v_idx][0]-FG.mfv[f][v_idx][1])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return guesses(FG), maxdiff
end

function onebpiter!(FG::FactorGraph, algo::MS,
    neutral=neutralel(algo,FG.q);
    wrong = Fun(FG.q, -Inf), alpha = 0)

    maxdiff = diff = 0.0
    for f in randperm(length(FG.Fneigs))
        for (v_idx, v) in enumerate(FG.Fneigs[f])
            # If field has -Inf, don't to anything, that variable is already decided
            # sum(isinf.(FG.fields[v]))!=0  && continue
            # FG.mfv[f][v_idx]==neutral  && continue

            # If a leaf, uniform message!
            # if vardegree(FG,v) == 1
                # FG.fields[v] = OffsetArray(1/FG.q*ones(FG.q), 0:FG.q-1)
            # else
                # Subtract message from belief
                FG.fields[v] .-= FG.mfv[f][v_idx]
                ####### EXPERIMENT ######
                FG.fields[v][isnan.(FG.fields[v])] .= 0.0
            # end
            # Define functions for weighted convolution
            funclist = Fun[]
            for (vprime_idx, vprime) in enumerate(FG.Fneigs[f])
                if vprime != v
                    func = FG.fields[vprime] - FG.mfv[f][vprime_idx]
                    # adjust for weights
                    func .= func[FG.mult[FG.mult[FG.hfv[f][v_idx],FG.gfinv[FG.hfv[f][vprime_idx]]],:]]
                    push!(funclist, func)
                end
            end
            # Update with damping
            oldmessage = alpha > 0 ? alpha*FG.mfv[f][v_idx] : Fun(FG.q)
            FG.mfv[f][v_idx] .= oldmessage + (1-alpha)*reduce(gfmsc, funclist, init=neutral)
            FG.mfv[f][v_idx] .-= maximum(FG.mfv[f][v_idx])
            # Send warning if messages are all NaN
            sum(isnan.(FG.mfv[f][v_idx])) > 0 && error("Message ($f,$v) has a NaN")
            # Update belief after updating the message
            FG.fields[v] .+= FG.mfv[f][v_idx]
            # Normalize belief
            FG.fields[v] .-= maximum(FG.fields[v])
            sum(isnan.(FG.fields[v])) > 0 && error("Belief $v has a NaN")
            # Look for maximum message (difference)
            diff = abs(FG.mfv[f][v_idx][0]-FG.mfv[f][v_idx][1])
            diff > maxdiff && (maxdiff = diff)
        end
    end
    return guesses(FG), maxdiff
end

function guesses(beliefs::AbstractVector)
    return [findmax(b)[2] for b in beliefs]
end
function guesses(FG::FactorGraph)
    return guesses(FG.fields)
end

function bp!(FG::FactorGraph, algo::Union{BP,MS}, y::Vector{Int}, maxiter=Int(1e3),
    convergence=:messages, nmin=300, tol=1e-7, gamma=0, alpha=0 , Tmax=1, L=1,
    randseed=0, maxdiff=zeros(maxiter), codeword=falses(maxiter),
    maxchange=zeros(maxiter); verbose=false)

    randseed != 0 && Random.seed!(randseed)      # for reproducibility
    newguesses = zeros(Int,FG.n)
    oldguesses = guesses(FG)
    oldmessages = deepcopy(FG.mfv)
    newmessages = deepcopy(FG.mfv)
    n = 0
    for trial in 1:Tmax
        for t in 1:maxiter
            newguesses,maxdiff[t] = onebpiter!(FG, algo, alpha=alpha)
            newmessages .= FG.mfv
            codeword[t] = (sum(paritycheck(FG))==0)
            if convergence == :messages
                for f in eachindex(newmessages)
                    for (v_idx,msg) in enumerate(newmessages[f])
                        change = maximum(abs.(msg - oldmessages[f][v_idx]))
                        if change > maxchange[t]
                            maxchange[t] = change
                        end
                    end
                end
                maxchange[t] < tol && return :converged, t, trial
                oldmessages .= deepcopy(newmessages)
            elseif convergence == :decvars
                if newguesses == oldguesses
                    n += 1
                    n >= nmin && return :converged, t, trial
                else
                    n=0
                end
                oldguesses .= newguesses
            elseif convergence == :parity
                if sum(paritycheck(FG)) == 0
                    return :converged, t, trial
                end
            else
                error("Field convergence must be one of :messages, :decvars, :parity")
            end
            softdecimation!(FG, gamma*t, algo)
            (verbose && isinteger(10*t/maxiter)) && println("BP/MS Finished ",Int(t/maxiter*100), "%")
        end
        if trial != Tmax
            refresh!(FG)
            FG.fields .= extfields(FG.q,y,algo,L,randseed=randseed+trial)
            oldguesses .= guesses(FG)
            oldmessages .= deepcopy(FG.mfv)
            maxchange .= fill(-Inf, maxiter)
            n = 0
        end
    end
    return :unconverged, maxiter, Tmax
end

function softdecimation!(FG::FactorGraph, gamma::Real, algo::Union{BP,MS})
    for (v,gv) in enumerate(FG.fields)
        if gamma > 0
            if typeof(algo)==BP
                FG.fields[v] .*= gv.^gamma
                # Normalize belief
                if sum(FG.fields[v])!= 0
                    FG.fields[v] ./= sum(FG.fields[v])
                end
            else
                FG.fields[v] .+= gv*gamma
            end
        end
    end
    return nothing
end

neutralel(algo::BP, q::Int) = Fun(x == 0 ? 1.0 : 0.0 for x=0:q-1)
neutralel(algo::MS, q::Int) = Fun(x == 0 ? 0.0 : -Inf for x=0:q-1)


# Re-initialize messages
function refresh!(FG::FactorGraph)
    for f in eachindex(FG.mfv)
        FG.mfv[f] .= [OffsetArray(1/FG.q*ones(FG.q), 0:FG.q-1) for v in eachindex(FG.mfv[f])]
    end
    return nothing
end

function refresh!(FG::FactorGraph, y::Vector{Int}, q::Int=2, algo::Union{BP,MS}=MS(),
    L::Real=1.0, sigma::Real=1e-4; randseed::Int=0)

    refresh!(FG)
    FG.fields .= extfields(q, y, algo, L, randseed=randseed)
    return nothing
end

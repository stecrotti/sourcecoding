struct MS; end
struct BP; end

function onebpiter!(FG::FactorGraph, algo::BP, neutral=neutralel(algo,FG.q);
    alpha = 0)

    maxdiff = diff = 0.0
    for f in randperm(length(FG.Fneigs))
        for (v_idx, v) in enumerate(FG.Fneigs[f])
            # Divide message from belief
            FG.fields[v] ./= FG.mfv[f][v_idx]
            # Restore possible n/0 or 0/0
            FG.fields[v][isnan.(FG.fields[v])] .= 1.0
            # Define functions for weighted convolution
            funclist = Fun[]
            weightlist = Int[]
            for (vprime_idx,vprime) in enumerate(FG.Fneigs[f])
                if vprime != v
                    func = FG.fields[vprime] ./ FG.mfv[f][vprime_idx]
                    func[isnan.(func)] .= 1.0
                    # adjust for weights
                    # func .= func[FG.mult[FG.gfinv[FG.hfv[f][vprime_idx]], FG.mult[FG.hfv[f][v_idx],:]]]
                    push!(funclist, func)
                    push!(weightlist, FG.hfv[f][vprime_idx])
                end
            end
            FG.mfv[f][v_idx] = gfconvw(funclist, FG.gfdiv, weightlist,
                neutral)
            # Adjust final weight
            FG.mfv[f][v_idx] .= FG.mfv[f][v_idx][ FG.mult[FG.hfv[f][v_idx],:] ]

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
                # Subtract message from belief
                FG.fields[v] .-= FG.mfv[f][v_idx]
                # if "Inf-Inf=NaN" happened, restore 0.0
                FG.fields[v][isnan.(FG.fields[v])] .= 0.0
            # Define functions for weighted convolution
            funclist = Fun[]
            weightlist = Int[]
            for (vprime_idx, vprime) in enumerate(FG.Fneigs[f])
                if vprime != v
                    func = FG.fields[vprime] - FG.mfv[f][vprime_idx]
                    # adjust for weights
                    # func .= func[FG.mult[FG.gfinv[FG.hfv[f][vprime_idx]], FG.mult[FG.hfv[f][v_idx],:]]]
                    push!(funclist, func)
                    push!(weightlist, FG.hfv[f][vprime_idx])
                end
            end
            # Try new way
            FG.mfv[f][v_idx] = gfmscw(funclist, FG.gfdiv, weightlist,
                neutral)
            # Adjust final weight
            FG.mfv[f][v_idx] .= FG.mfv[f][v_idx][ FG.mult[FG.hfv[f][v_idx],:] ]
            # Normalize message
            FG.mfv[f][v_idx] .-= maximum(FG.mfv[f][v_idx])
            FG.mfv[f][v_idx][isnan.(FG.mfv[f][v_idx])] .= 0.0
            # end
            # Send warning if messages are all NaN
            if sum(isnan.(FG.mfv[f][v_idx])) > 0
                @show reduce(gfmsc, funclist, init=neutral)
                error("Message ($f,$v) has a NaN")
            end
            # Update belief after updating the message
            FG.fields[v] .+= FG.mfv[f][v_idx]
            # Normalize belief
            FG.fields[v] .-= maximum(FG.fields[v])
            FG.fields[v][isnan.(FG.fields[v])] .= 0.0
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
                if maxchange[t] < tol
                    return :converged, t, trial
                end
                oldmessages .= deepcopy(newmessages)
                performance_name = "max change"
                performance_value = maxchange[t]
            elseif convergence == :decvars
                if newguesses == oldguesses
                    n += 1
                    if n >= nmin
                        return :converged, t, trial
                    end
                else
                    n=0
                end
                oldguesses .= newguesses
                performance_name = "iters with same decision vars"
                performance_value = n
            elseif convergence == :parity
                parity = sum(paritycheck(FG))
                if parity == 0
                    return :converged, t, trial
                end
                performance_name = "parity"
                performance_value = parity
            else
                error("Field convergence must be one of :messages, :decvars, :parity")
            end
            reinforce!(FG, gamma*t, algo)
            if verbose
                steps_tot = 20
                progress = div(t*steps_tot, maxiter)
                print("\u1b[2K")    # clear line
                println("  [", "-"^progress," "^(steps_tot-progress), "] ",
                    "$t/$maxiter, trial $trial/$Tmax, ",
                    performance_name, ": ", performance_value)
                print("\u1b[1F")    # move cursor to beginning of line
            end
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
    # if verbose
    #     print("\u1b[2K")    # clear line
    #     print("\u1b[1F")    # move cursor to beginning of line
    # end
    return :unconverged, maxiter, Tmax
end

function reinforce!(FG::FactorGraph, gamma::Real, algo::Union{BP,MS})
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

struct MS
end

struct BP
end

function onebpiter!(FG::FactorGraph, algo::BP,
     neutral=Fun(x == 0 ? 1.0 : 0.0 for x=0:FG.q-1))
    mult = FG.mult
    gfinv = FG.gfinv
    # factor -> variable
    for f in randperm(length(FG.Fneigs))
        for (v_idx, v) in enumerate(FG.Fneigs[f])
            # Divide message from belief
            FG.fields[v] .= FG.fields[v] ./ FG.mfv[f][v_idx]
            # Define functions for weighted convolution
            funclist = Fun[]
            for (vprime_idx,vprime) in enumerate(FG.Fneigs[f])
                if vprime != v
                    func = FG.fields[vprime] ./ FG.mfv[f][vprime_idx]
                    # adjust for weights
                    func .= func[mult[mult[FG.hfv[f][v_idx],gfinv[FG.hfv[f][vprime_idx]]],:]]
                    push!(funclist, func)
                end
            end
            FG.mfv[f][v_idx] = reduce(gfconv, funclist, init=neutral)
            FG.mfv[f][v_idx] ./= sum(FG.mfv[f][v_idx])
            # Update belief after updating the message
            FG.fields[v] .= FG.fields[v] .* FG.mfv[f][v_idx]
        end
    end
    return (FG)
end

function onebpiter!(FG::FactorGraph, algo::MS,
    neutral=Fun(x == 0 ? 0.0 : -Inf for x=0:FG.q-1),
    wrong = Fun(q, -Inf))

    for f in randperm(length(FG.Fneigs))
        for (v_idx, v) in enumerate(FG.Fneigs[f])
            # Subtract message from belief
            FG.fields[v] .-= FG.mfv[f][v_idx]
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
            FG.mfv[f][v_idx] = reduce(gfmsc, funclist, init=neutral)
            FG.mfv[f][v_idx] .-= maximum(FG.mfv[f][v_idx])
            # Send warning if messages are all -Inf
            FG.mfv[f][v_idx] == wrong && println("Warning: message $f->$v is all -Inf")
            # Update belief after updating the message
            FG.fields[v] .+= FG.mfv[f][v_idx]
        end
    end
    return guesses(FG)
end

function guesses(beliefs::AbstractVector)
    return [findmax(b)[2] for b in beliefs]
end
function guesses(FG::FactorGraph)
    return guesses(FG.fields)
end

# BP with convergence criterion: guesses
function bp!(FG::FactorGraph, algo::Union{BP,MS}; maxiter=Int(3e2),
    gamma=0, nmin=100, verbose=false)
    if  typeof(algo) == BP
        neutral = Fun(x == 0 ? 1.0 : 0.0 for x=0:FG.q-1)
    else
        neutral = Fun(x == 0 ? 0.0 : -Inf for x=0:FG.q-1)
    end
    newguesses = zeros(Int,FG.n)
    oldguesses = guesses(FG)
    n = 0   # number of consecutive times for which the guesses are left unchanged by one BP iteration
    for it in 1:maxiter
        newguesses = onebpiter!(FG, algo, neutral)
        if newguesses == oldguesses
            n += 1
            if n >= nmin
                verbose && println("BP/MS converged after $it steps")
                return :converged, it
            end
        else
            n=0
        end
        oldguesses = newguesses
        # Soft decimation
        for (v,gv) in enumerate(FG.fields)
            if typeof(algo)==BP
                FG.fields[v] .*= gv^(gamma*it)
            else
                FG.fields[v] .+= (gamma*it)*gv
            end
        end
    end
    verbose && println("BP/MS unconverged after $maxiter steps")
    return :unconverged, maxiter
end

# BP with convergence criterion: messages
function bp_msg!(FG::FactorGraph, algo::Union{BP,MS}; maxiter=Int(3e2),
    gamma=0, tol=1e-4, verbose=false)
    if  typeof(algo) == BP
        neutral = Fun(x == 0 ? 1.0 : 0.0 for x=0:FG.q-1)
    else
        neutral = Fun(x == 0 ? 0.0 : -Inf for x=0:FG.q-1)
    end
    oldmessages = deepcopy(FG.mfv)
    maxchange = 0.0     # Maximum change in messages from last step
    for it in 1:maxiter
        maxchange = 0.0
        onebpiter!(FG, algo, neutral)
        newmessages = FG.mfv
        for f in eachindex(newmessages)
            for (v_idx,msg) in enumerate(newmessages[f])
                change = maximum(abs.(msg - oldmessages[f][v_idx]))
                if change > maxchange
                    maxchange = change
                end
            end
        end

        if maxchange < tol
            verbose && println("BP/MS converged after $it steps")
            return :converged, it
        end
        # Soft decimation
        for (v,gv) in enumerate(FG.fields)
            if typeof(algo)==BP
                FG.fields[v] .*= gv^(gamma*it)
            else
                FG.fields[v] .+= (gamma*it)*gv
            end
        end
        oldmessages = deepcopy(newmessages)
    end
    verbose && println("BP/MS unconverged after $maxiter steps. Max change in messages: $maxchange")
    return :unconverged, maxiter
end

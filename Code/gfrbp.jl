struct MS
end

struct BP
end

function onebpiter!(FG::FactorGraph, algo::BP)
    mult = FG.mult
    gfinv = FG.gfinv
    # factor -> variable
    for f in randperm(length(FG.Fneigs))
        for v in FG.Fneigs[f]
            # Define functions for weighted convolution
            funclist = Fun[]
            for vprime in FG.Fneigs[f]
                if vprime != v
                    # find all messages f'->v'
                    neigs = [FG.mfv[fprime][vprime] for fprime in FG.Vneigs[vprime] if fprime!=f]
                    # product
                    func = exp.(-FG.fields[vprime]) * (neigs==[] ? Fun(q,1) : prod(neigs))
                    # adjust for weights
                    func .= func[mult[mult[FG.hfv[f][v],gfinv[FG.hfv[f][vprime]]],:]]
                    push!(funclist, func)
                end
            end

            FG.mfv[f][v] = gfconvlist(funclist)
            FG.mfv[f][v] ./= sum(FG.mfv[f][v])
            domain!(FG.mfv[f][v], FG.q, 0.0)   # Ensure the correct length by padding with neutral element
        end
    end
    return guesses(FG, algo)
end

function onebpiter!(FG::FactorGraph, algo::MS)
    mult = FG.mult
    gfinv = FG.gfinv
    # factor -> variable
    for f in randperm(length(FG.Fneigs))
        for v in FG.Fneigs[f]
            # Define functions for weighted convolution
            funclist = Fun[]
            for vprime in FG.Fneigs[f]
                if vprime != v
                    # find all messages f'->v'
                    neigs = [FG.mfv[fprime][vprime] for fprime in FG.Vneigs[vprime] if fprime!=f]
                    # product
                    func = -FG.fields[vprime] .+ (neigs==[] ? Fun(q,) : sum(neigs))
                    # adjust for weights
                    func .= func[mult[mult[FG.hfv[f][v],gfinv[FG.hfv[f][vprime]]],:]]
                    push!(funclist, func)
                end
            end
            FG.mfv[f][v] = gfconvlist(funclist)
            FG.mfv[f][v] .-= maximum(FG.mfv[f][v])
            domain!(FG.mfv[f][v], FG.q, -Inf)   # Ensure the correct length by padding with neutral element
        end
    end
    return guesses(FG, algo)
end

function beliefs(FG::FactorGraph, algo::BP)
    q = FG.q
    g = [OffsetArray(fill(1/q, q), 0:q-1) for v in 1:FG.n]
    for (v, neigs_of_v) in enumerate(FG.Vneigs)
        if neigs_of_v != []
            neigs = [FG.mfv[f][v] for f in neigs_of_v]
            g[v] = exp.(-FG.fields[v]) .* prod(neigs)   # I defined '*' for OffsetArray, hence also prod is well defined
        else
            g[v] = exp.(-FG.fields[v])
        end
        g[v] ./= sum(g[v])
    end
    return g
end

function beliefs(FG::FactorGraph, algo::MS)
    q = FG.q
    g = [OffsetArray(fill(0.0, q), 0:q-1) for v in 1:FG.n]
    for (v, neigs_of_v) in enumerate(FG.Vneigs)
        if neigs_of_v != []
            neigs = [FG.mfv[f][v] for f in neigs_of_v]
            g[v] = -FG.fields[v] + sum(neigs)
        else
            g[v] = -FG.fields[v]
        end
        g[v] .-= maximum(g[v])
    end
    return g
end

function guesses(FG::FactorGraph, algo::Union{BP,MS})
    return [findmax(b)[2] for b in beliefs(FG,algo)]
end

function bp!(FG::FactorGraph, algo::Union{BP,MS}; max_iter=Int(3e2),
    gamma=0, nmin=10, verbose=false)
    newguesses = zeros(Int,FG.n)
    oldguesses = guesses(FG,algo)
    n = 0   # number of consecutive times for which the guesses are left unchanged by one BP iteration
    for it in 1:max_iter
        newguesses = onebpiter!(FG, algo)
        if newguesses == oldguesses
            n += 1
            if n >= nmin
                verbose && println("BP/MS converged after $it steps")
                return newguesses
            end
        else
            n=0
        end
        # Soft decimation
        for (v,gv) in enumerate(beliefs(FG,algo))
            if typeof(algo)==BP
                FG.fields[v] .+= (gamma*log.(gv))
            else
                FG.fields[v] .+= gamma*gv
            end
        end
        oldguesses = newguesses
    end
    # error("BP/MS unconverged after $max_iter steps")
    return :unconverged
end

using OffsetArrays
using StatsBase, ProgressMeter, Random
residual(x) = (p=OffsetVector((x .* eachindex(x))[1:end], 0:lastindex(x)-1); p./=sum(p))

function BP_th(H, tus)
    p1 = prod(tus[i][1] for i=1:length(tus))*exp(H)
    p2 = prod(tus[i][2] for i=1:length(tus))*exp(-H)
    if p1+p2==0
        println("conflicting messages in BP_th")
    end
    th1 = p1/(p1+p2)
    th2 = p2/(p1+p2)
    (th1, th2)
end

function BP_tu(s, ths)
    tu1=0.0; tu2=0.0
    for sig in Iterators.product(fill(-1:2:1,2)...)
        ind = [(sig[i]==1 ? 1 : 2) for i=1:length(sig)]
        if prod(sig) == s
            tu1 += prod(ths[i][ind[i]] for i=1:length(ths))
        else
            tu2 += prod(ths[i][ind[i]] for i=1:length(ths))
        end
    end
    (tu1, tu2)
end

conv(h1::Tuple, h2::Tuple) = tuple(h1[1]*h2[1]+h1[2]*h2[2], h1[1]*h2[2]+h1[2]*h2[1]) 
function BP_th_conv(H, tus)
    th = (exp(H), exp(-H))
    for tu in tus
        th = th.*tu
    end
    th ./ sum(th)
end
function BP_tu_conv(s, ths)
   tu = s==1 ? tuple(1.0,0.0) : tuple(0.0,1.0)
   for i in eachindex(ths)
       tu = conv(tu, ths[i]) 
    end
    tu
end

function dist_sigmas(s, ths_0, σs, ν)
    k = length(ths_0)
    i = 0
    for sig in Iterators.product(fill(-1:2:1,k)...)
        i += 1
        ind = [(sig[j]==1 ? 1 : 2) for j=1:length(sig)]
        σs[i] = sig
        if prod(sig) == s
            ν[i] += prod(ths_0[j][ind[j]] for j=1:length(ths_0))
        end
    end
    ν ./= sum(ν)
    ν, σs
end

function RS(Λ, K, H; 
        maxiter=100, 
        popsize=1000, 
        popP_RS = fill((0.5,0.5), 1:popsize),
        popQ_RS = fill((0.5,0.5), 1:popsize),
        q0 = fill(NaN, 1:maxiter),
        dist_RS=fill(NaN, 1:maxiter),
        Fe_RS=fill(NaN, 1:maxiter)
    )

    Λ1 = residual(Λ)
    K1 = residual(K)
    wΛ1 = weights(Λ1.parent)#wΛ1 = weights(Λ1)
    wK1 = weights(K1)

    
    @showprogress for t = 1:maxiter
        
        for i = 1:popsize
            k = sample(eachindex(K1), wK1)
            ind_ths = rand(1:popsize, k)
            ths = popP_RS[ind_ths]            
            s = rand((-1,1))
            popQ_RS[i] = BP_tu_conv(s, ths)
        end
        
        q0[t]=0.0
        
        for i = 1:popsize
            d = sample(collect(eachindex(Λ1)), wΛ1)#d = sample(eachindex(Λ1), wΛ1)
            ind_tus = rand(1:popsize, d)
            tus = popQ_RS[ind_tus]
            popP_RS[i] = BP_th_conv(H, tus)
            
            q0[t]+=(popP_RS[i][1]-popP_RS[i][2])^2
        end
        
        q0[t] = q0[t]/length(popP_RS)
        dist_RS[t]=distorsion_RS(Λ, K, H, popP_RS, popQ_RS)
        Fe_RS[t]=F_RS(Λ, K, H, popP_RS, popQ_RS, maxiter=10^5)
    end
    popP_RS, popQ_RS, q0, dist_RS, Fe_RS
end

function distorsion_RS(Λ, K, H, popP_RS, popQ_RS; 
        maxiter=length(popP_RS) 
    )
    wΛ = weights(Λ)
    popsize = length(popP_RS)

    O = 0.0
    for t = 1:maxiter
        d = sample(eachindex(Λ), wΛ)
        ind_tus = rand(1:popsize, d)
        tus = popQ_RS[ind_tus]
        th = BP_th_conv(H, tus)
        O += th[1]-th[2]
    end
    O =O/maxiter
    (1-O)/2
end

function log_Zi_RS(Λ, H, popQ_RS, maxiter)
    wΛ = weights(Λ)
    popsize = length(popQ_RS)

    logZ_i = 0.0
    for t = 1:maxiter
        d = sample(eachindex(Λ), wΛ)
        ind_tus = rand(1:popsize, d)
        
        tus = popQ_RS[ind_tus]
        th = (exp(H), exp(-H))
        for tu in tus
            th = th.*tu
        end
        Z_i = sum(th)
        logZ_i += log(Z_i)
    end
    logZ_i /= maxiter
end

function log_Za_RS(K, popP_RS,maxiter)
    wK = weights(K)
    popsize = length(popP_RS)
    
    logZ_a = 0.0
    for t=1:maxiter
        k = sample(eachindex(K), wK)
        ind_ths = rand(1:popsize, k)
        ths = popP_RS[ind_ths]
        tu = reduce(conv, ths, init=(1,0))
        Z_a = rand(tu)
        logZ_a += log(Z_a)
    end
    logZ_a /= maxiter
end

function log_Zia_RS(popP_RS, popQ_RS, maxiter)
    logZ_ia = 0.0
    for t=1:maxiter
        th = rand(popP_RS)
        tu = rand(popQ_RS)
        Z_ia = sum(th .* tu)
        logZ_ia += log(Z_ia)
    end
    logZ_ia /= maxiter
end

function F_RS(Λ, K, H, popP_RS, popQ_RS; maxiter=length(popP_RS))
    mK = sum(k*K[k] for k=eachindex(K))
    mΛ = sum(d*Λ[d] for d=eachindex(Λ))
    α = mΛ/mK
    F = log_Zi_RS(Λ, H, popQ_RS, maxiter) + α* log_Za_RS(K, popP_RS, maxiter) - mΛ*log_Zia_RS(popP_RS, popQ_RS, maxiter)
    F=-F/H
end

function init_pop(pop_RS, ϵ)
    N=length(pop_RS)
    pop = fill((NaN, NaN), -1:1, 1:N)
    pop[0,:]=popP_RS
    pop[1,:] =fill( ( 1-ϵ, ϵ), 1:N)
    pop[-1,:] = fill( ( ϵ, 1-ϵ), 1:N)
    pop
end
function init_pop_alternative(pop_RS, ϵ)
    N=length(pop_RS)
    pop = fill((NaN, NaN), -1:1, 1:N)
    pop[0,:]=popP_RS
    for i=1:N
        pop[1,i]=(rand()<ϵ ? pop[0,i] : (1.0, 0.0))
        pop[-1,i]=(rand()<ϵ ? pop[0,i] : (0.0, 1.0))
    end
    pop
end

function checkRS(popP)
    q0=0.0; q1=0.0
    for ind=1:size(popP, 2)
        th = popP[:,ind]
        m0 = th[0][1]-th[0][2]; m1 = th[1][1]-th[1][2]; mm1=th[-1][1]-th[-1][2]
        q0 += m0^2
        q1 += m1*th[0][1] - mm1*th[0][2]
    end
    q1= q1/size(popP, 2); q0=q0/size(popP, 2)
    q0, q1, q1-q0
end

function checkRS_comparePops(popP)
    C1 = [0.0, 0.0]; Cm1=[0.0, 0.0]
    for ind=1:size(popP, 2)
        th = popP[:,ind]
        C1[1] += th[1][1]-th[0][1]
        C1[2] += th[1][2]-th[0][2]
        Cm1[1] += th[-1][1]-th[0][1]
        Cm1[2] += th[-1][2]-th[0][2]
    end
    C1= C1./size(popP, 2); Cm1=Cm1./size(popP, 2)
    C1, Cm1
end

function checkHardFields(pop)
    p=(0.0, 0.0)
    for i=1:length(pop)
        p = p .+ (pop[i] .== 0.0)
    end
    p=p./length(pop)
end

function prop_small_m(pop, th)
    p1 = mean([pop[i][1]<th for i=1:popsize])
    p2 = mean([pop[i][2]<th for i=1:popsize])
    p1, p2
end

function RSB_entropic_m1(Λ, K, H, popP_RS, popQ_RS; 
        maxiter=100, 
        popsize=1000, 
        ϵ=0.01,
        popP = init_pop(popP_RS, ϵ),
        popQ = init_pop(popQ_RS, ϵ),
        q0=fill(0.0, 1:maxiter),
        V=fill(0.0, 1:maxiter),
        p1=fill((0.0, 0.0), 1:maxiter),
        pm1=fill((0.0, 0.0), 1:maxiter),
        int_freenrj=fill(NaN, 1:maxiter)
    )
    
    Λ1 = residual(Λ)
    K1 = residual(K)
    wΛ1 = weights(Λ1.parent)#wΛ1 = weights(Λ1)
    wK1 = weights(K1)
    
    @showprogress for t = 1:maxiter
        for i = 1:popsize
            k = sample(eachindex(K1), wK1)
            ind_ths = rand(1:popsize, k)
            ths = popP[:,ind_ths]            
            s = rand((-1,1))
            popQ[0,i] = BP_tu_conv(s, ths[0,:])
            #@show ind_ths, s
            
            ν = fill(0.0, 1:2^k)
            σs = fill(tuple(fill(NaN, k)...), 1:2^k)
            ν, σs = dist_sigmas(s, ths[0,:], σs, ν)
            wν = weights(ν)
            ind = sample(eachindex(ν), wν)
            σ = convert.(Int64, σs[ind])
            #@show σ
            elts = [ths[s,i] for (i,s) ∈ zip(eachindex(σ), σ)]
            popQ[1,i] = BP_tu_conv(s, elts)

            ν = fill(0.0, 1:2^k)
            σs = fill(tuple(fill(NaN, k)...), 1:2^k)
            ν, σs = dist_sigmas(-s, ths[0,:], σs, ν)
            wν = weights(ν)
            ind = sample(eachindex(ν), wν)
            σ = convert.(Int64, σs[ind])
            #@show σ
            elts = [ths[s,i] for (i,s) ∈ zip(eachindex(σ), σ)]
            popQ[-1,i] = BP_tu_conv(s, elts)
        end

        for i = 1:popsize
            d = sample(collect(eachindex(Λ1)), wΛ1)#d = sample(eachindex(Λ1), wΛ1)
            ind_tus = rand(1:popsize, d)
            tus = popQ[:,ind_tus]
            #@show ind_tus
            popP[0,i] = BP_th_conv(H, tus[0,:])
            popP[-1,i] = BP_th_conv(H, tus[-1,:])
            popP[1,i] = BP_th_conv(H, tus[1,:])
            
            m0 = popP[0,i][1]-popP[0,i][2]; m1 = popP[1,i][1]-popP[1,i][2]; mm1 = popP[-1,i][1]-popP[-1,i][2]
            q0[t]+= m0^2
            V[t]+= m1*popP[0,i][1] - mm1*popP[0,i][2]
            p1[t] = p1[t] .+ (popP[1,i] .== 0.0)
            pm1[t] = pm1[t] .+ (popP[-1,i] .== 0.0)
        end
        q0[t] = q0[t]/size(popP, 2)
        V[t] = V[t]/size(popP, 2)
        p1[t] = p1[t]./size(popP, 2)
        pm1[t] = pm1[t]./size(popP, 2)
        int_freenrj[t] = internal_freeenergy(Λ, K, H, popP, popQ, maxiter=10^5)
    end    
    V = V .- q0
    #popP, popQ
    popP, popQ, V, p1, pm1, int_freenrj
end

function variable_internal_freeenergy(Λ, H, popQ, maxiter)
    wΛ = weights(Λ)
    popsize = length(popQ[0,:])
    logZ_i = 0.0
    for t = 1:maxiter
        d = sample(eachindex(Λ), wΛ)
        ind_tus = rand(1:popsize, d)
        tus = popQ[:,ind_tus]
        
        pr = (exp(H), exp(-H))
        for tu in tus[0,:]
            pr = pr.*tu
        end
        pr = pr./sum(pr)
        wpr = weights([pr[1], pr[2]])
        sig = sample(1:-2:-1, wpr)
        
        th = (exp(H), exp(-H))
        for tu in tus[sig,:]
            th = th.*tu
        end
        Z_i = sum(th)
        #@show Z_i
        logZ_i += log(Z_i)
    end
    logZ_i /= maxiter    
end

function factor_internal_freeenergy(K, popP, maxiter)
    wK = weights(K)
    popsize = length(popP[0,:])
    logZ_a = 0.0
    for t=1:maxiter
        k = sample(eachindex(K), wK)
        ind_ths = rand(1:popsize, k)
        ths = popP[:,ind_ths]
        
        s = rand((-1,1))
        ν = fill(0.0, 1:2^k)
        σs = fill(tuple(fill(NaN, k)...), 1:2^k)
        ν, σs = dist_sigmas(s, ths[0,:], σs, ν)
        wν = weights(ν)
        ind = sample(eachindex(ν), wν)
        σ = convert.(Int64, σs[ind])
        #@show σ
        
        elts = [ths[s,i] for (i,s) ∈ zip(eachindex(σ), σ)]
        tu = reduce(conv, elts, init=(1,0))
        Z_a = tu[s == 1 ? 1 : 2]
        logZ_a += log(Z_a)
    end
    logZ_a /= maxiter
end

function edge_internal_freeenergy(popP, popQ, maxiter)
    popsize = length(popQ[0,:])
    logZ_ia = 0.0
    for t = 1:maxiter
        th = popP[:,rand(1:popsize)]
        tu = popQ[:,rand(1:popsize)]
        
        pr = th[0].* tu[0]
        pr = pr./sum(pr)
        wpr = weights([pr[1], pr[2]])
        sig = sample(1:-2:-1, wpr)
        
        Z_ia = sum(th[sig].* tu[sig])
        #@show Z_ia
        logZ_ia += log(Z_ia)
    end
    logZ_ia /= maxiter
end

function internal_freeenergy(Λ, K, H, popP, popQ; maxiter=length(popP[0,:]))
    mK = sum(k*K[k] for k=eachindex(K))
    mΛ = sum(d*Λ[d] for d=eachindex(Λ))
    α = mΛ/mK
    F = variable_internal_freeenergy(Λ, H, popQ, maxiter) + α* factor_internal_freeenergy(K, popP, maxiter) - mΛ*edge_internal_freeenergy(popP, popQ, maxiter)
    F=-F/H
end

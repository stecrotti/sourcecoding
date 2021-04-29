using SparseArrays, Random, Printf, Plots

include("slim_graphs.jl")   # methods to compute basis

struct BeliefPropagation{F,M}
    H :: SparseMatrixCSC{F,Int}     # size (nvars,nfactors)
    m :: Vector{M}                  # messages (parametrized with magnetization)
    efield :: Vector{M}             # external field (parametrized with magnetization)
end
nfactors(bp::BeliefPropagation) = size(bp.H,2)
nvars(bp::BeliefPropagation) = size(bp.H,1)

# build a parity check matrix given degree profile, size and number of edges
# assumes all of the parameters are consistent
# follows Luby, "Efficient erasure correcting codes", doi: 10.1109/18.910575.
function ldpc_matrix(n::Int, m::Int, nedges::Int, Lambda, Rho,
    edgesleft=zeros(Int, nedges), edgesright=copy(edgesleft);
    accept_multi_edges=true, maxtrials=1000)

    check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for t in 1:maxtrials
        H = one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright)
        (nnz(H) == nedges || accept_multi_edges) && return H
    end
    error("Could not build graph after $maxtrials trials: multi-edges were popping up")
end

function one_ldpc_matrix(n, m, nedges, Lambda, Rho, edgesleft, edgesright)
    v = r = 1
    for i = 1:lastindex(Lambda)
        ni = round(Int, n*Lambda[i])
        for _ in 1:ni
            edgesleft[r:r+i-1] .= v
            v += 1; r += i
        end
    end
    shuffle!(edgesleft)
    f = r = 1
    for j = 1:lastindex(Rho)
        nj = round(Int, m*Rho[j])
        for _ in 1:nj
            edgesright[r:r+j-1] .= f
            f += 1; r += j
        end
    end
    sparse(edgesleft, edgesright, trues(nedges))
end

function check_consistency_polynomials(n,m,nedges,Lambda,Rho)
    for l in Lambda
        @assert isinteger(round(n*l, digits=10))
    end
    for r in Rho
        @assert isinteger(round(m*r, digits=10))
    end
    @assert isapprox(n*sum(i*l for (i,l) in pairs(Lambda)), nedges, atol=1e-8) 
    @assert isapprox(m*sum(j*r for (j,r) in pairs(Rho)), nedges, atol=1e-8)
    @assert isapprox(sum(Lambda) == sum(Rho), 1, atol=1e-8)
end

function belief_propagation(n, m, nedges, Lambda, Rho, efield=zeros(n), 
        msg=zeros(nedges),  args...; kw...)
    H = ldpc_matrix(n, m, nedges, Lambda, Rho, args...; kw...)
    BeliefPropagation(H, msg, copy(efield))
end

# BP ROUTINES
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

# returns -1 if a contradiction is found, the max absolute change in message otherwise
function update_factor!(bp::BeliefPropagation, a::Int; damp=0.0, rein=0.0)
    maxchange = 0.0
    t = Prod{Float64}()
    vars = rowvals(bp.H)
    for i in nzrange(bp.H,a)
        v = vars[i]
        # if bp.efield[v] == bp.m[i]
        #     # avoid 0/0 when the two are equal and have absolute value 1
        #     bp.efield[v] = 0.0
        if abs(bp.efield[v])!=1
        # else
            bp.efield[v] = (bp.efield[v]-bp.m[i])/(1-bp.efield[v]*bp.m[i])
        end
        t *= bp.efield[v]
    end
    for i in nzrange(bp.H,a)
        v = vars[i]
        m = t/bp.efield[v]
        maxchange = max(maxchange, abs(m-bp.m[i]))
        m = m*(1-damp) + bp.m[i]*damp
        newfield = (m+bp.efield[v])/(1+m*bp.efield[v])
        # contradiction: m and field[v] were completely polarized but opposite
        isnan(newfield) && return -1.0
        bp.m[i] = m
        bp.efield[v] = sign(newfield)*abs(newfield)^(max(1.0-rein,0.0))
    end
    maxchange
end

# max-sum
function update_factor_ms!(bp::BeliefPropagation, a::Int; damp=0.0, rein=0.0)
    maxchange = 0.0
    fmin = fmin2 = Inf
    imin = 1
    s = Prod{Int}()
    vars = rowvals(bp.H)
    for i in nzrange(bp.H,a)
        v = vars[i]
        if bp.efield[v] == bp.m[i]
            # avoid 0/0 when the two are equal and have absolute value Inf
            bp.efield[v] = 0
        else
            bp.efield[v] = bp.efield[v] - bp.m[i]
        end
        s *= sign(bp.efield[v])
        m = abs(bp.efield[v])
        if fmin > m
            fmin2 = fmin
            fmin = m
            imin = i
        elseif fmin2 > m
            fmin2 = m
        end
    end
    for i in nzrange(bp.H,a)
        v = vars[i]
        m = (i == imin ? fmin2 : fmin) * (s / sign(bp.efield[v]))
        maxchange = max(maxchange, abs(m-bp.m[i]))
        m = m*(1-damp) + bp.m[i]*damp
        newfield = m + bp.efield[v]
        # contradiction: m and field[v] were completely polarized but opposite
        isnan(newfield) && return -1.0
        bp.m[i] = m
        bp.efield[v] = newfield*(1.0+rein)
    end
    maxchange
end

# returns ε=-1 if a contradiction is found and the number of the last iteration
function iteration!(bp::BeliefPropagation; factor_perm=randperm(nfactors(bp)), 
        maxiter=1000, tol=1e-12, damp=0.0, rein=0.0, callback=(x...)->false,
        update! = update_factor!)
    ε  = 0.0
    for it in 1:maxiter 
        ε  = 0.0
        for a in factor_perm
            maxchange = update!(bp, a, damp=damp, rein=rein*it)
            maxchange == -1 && return -1.0, it
            ε = max(ε, maxchange)
        end
        shuffle!(factor_perm)
        callback(it, ε, bp) && return ε, it
        ε < tol && return ε, it
    end
    ε, maxiter
end

# given a basis and the values of x[indep], compute what x[dep] must be
function fix_indep!(x, B, indep)
    n,k = size(B)
    dep = setdiff(1:n, indep)
    x[dep] .= B[dep,:]*x[indep] .% 2
    σ = 1 .- 2x
end

function parity(bp::BeliefPropagation, x::AbstractVector)
    z = p = 0
    vars = rowvals(bp.H)
    for a in 1:nfactors(bp)
        for i in nzrange(bp.H,a)
            p += x[vars[i]]
        end
        z += p % 2  
        p = 0
    end
    return z 
end
function distortion(x::AbstractVector, y::AbstractVector)
    d = 0
    for (xx,yy) in zip(x,y)
        d += sign(xx)!=sign(yy)
    end
    d/length(x)
end
function performance(bp::BeliefPropagation, fields, 
        x = sign.(bp.efield) .== -1)
    x .= sign.(bp.efield) .== -1
    nunsat = parity(bp, x)
    dist = distortion(fields, bp.efield)
    ovl = 1-2*dist
    nunsat, ovl, dist
end


#### DECIMATION

# try Tmax times to reach zero unsat with decimation
function decimate!(bp::BeliefPropagation, fields, indep; Tmax=1, kw...)
    freevars = falses(nvars(bp)); freevars[indep] .= true
    for t in 1:Tmax
        print("Trial $t of $Tmax: ")
        ε, nunsat, ovl, dist, iters = decimate1!(bp, fields, freevars; kw...)
        ε == -1 && print("contradiction found. ")
        println(nunsat, " unsat")
        nunsat == 0 && return nunsat, ovl, dist
        freevars .= false; freevars[indep] .= true
    end
    return -1, NaN, NaN
end

# 1 trial of decimation
function decimate1!(bp::BeliefPropagation, fields, freevars::BitArray{1}; 
        callback=(ε,nunsat,args...) -> (ε==-1||nunsat==0), kw...)
    # reset messages
    bp.m .= 0; bp.efield .= copy(fields)
    # pre-allocate for speed
    factor_perm = randperm(nfactors(bp)); x=falses(nvars(bp))
    # warmup bp run
    ε, iters = iteration!(bp, factor_perm=factor_perm; kw...)
    nunsat, ovl, dist = performance(bp, fields, x)
    nfree = sum(freevars)
    callback(ε, nunsat, bp, nfree, ovl, dist, iters, 0) && return nunsat, ovl, dist, iters

    for t in 1:nfree
        maxfield, tofix = find_most_biased(bp, freevars)
        freevars[tofix] = false
        # if tofix is undecided, give it its value in the source    
        bp.efield[tofix] = maxfield==0 ? fields[tofix] : sign(bp.efield[tofix])
        ε, iters = iteration!(bp, factor_perm=factor_perm; kw...)
        nunsat, ovl, dist = performance(bp, fields  , x)
        callback(ε, nunsat, bp, nfree-t, ovl, dist, iters, t) && return ε, nunsat, ovl, dist, iters
    end
    ε, nunsat, ovl, dist, iters
end

function find_most_biased(bp::BeliefPropagation, freevars::BitArray{1})
    m = -Inf; mi = 1
    for (i,h) in pairs(bp.efield)
       if freevars[i] && abs(h)>m
            m = abs(h); mi = i
       end
    end
    m, mi
end

function cb_decimation(ε, nunsat, bp::BeliefPropagation, nfree, ovl, dist, iters, step)
    @printf(" Step  %3d. Free = %3d. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
            step, nfree, ε, nunsat,  ovl, iters)
    (ε==-1 || nunsat==0) && return true
    false
end


# PLOTTING
function plot_rdb(; f30=true, f3=true) 
    DD = 0.001:0.01:0.5
    RR = 1 .+ DD.*log2.(DD) + (1 .- DD).*log2.(1 .- DD)
    pl = Plots.plot(RR, DD, label="Information bound")
    Plots.plot!(pl, RR, 0.5*(1 .- RR), label="Naive compression")
    if f30
        # RS curve for f3=0 : variables have degree 1 or 2
        R_f30 = 0.01:0.01:0.99
        D_f30 = [0.4546474073128681,0.43565049932361133,0.42092765046912317,0.40839439222208573,0.3972457420215144,0.3870702443196218,0.3776242651788484,0.36874955806591975,0.3603365352601175,0.3523056385673838,0.34459697668198624,0.33716414626734653,0.3299703418625086,0.3229857971761951,0.31618604029437997,0.30955066735206993,0.30306245811343413,0.2967067238049428,0.2904708168067388,0.2843437556958753,0.2783159341345032,0.27237889177802405,0.26652513178257226,0.2607479738209666,0.2550414345013817,0.24940012917780852,0.2438191906350483,0.2382942012118009,0.2328211357178961,0.22739631308980668,0.22201635516998064,0.21667815133009227,0.211378827914124,0.20688352899753953,0.20278317659003975,0.19868817793711935,0.1945975994415784,0.19051056552023182,0.18642625421128822,0.18234389327612,0.17826275673045133,0.17418216174918932,0.17010146589683228,0.16602006464188257,0.16193738911916555,0.15785290410864755,0.1537661062033548,0.14967652214243293,0.1455837072883871,0.14148724423013948,0.137998366107847,0.13448740638286893,0.13095421069018287,0.12739865696194796,0.12382065470933767,0.12022014422138577,0.11659709568295457,0.1129515082152911,0.10928340884389087,0.10559285139949881,0.10231362606715677,0.0989986151678956,0.09564794614910743,0.0922617930523803,0.0888403753926773,0.08538395674018595,0.0820151560863665,0.0788488704133426,0.07563678673548502,0.07237911796887381,0.06907615625262697,0.06591319349557895,0.06283584277258297,0.05970319879491964,0.056515457092013055,0.053562785457060835,0.05054512262289507,0.04752198628196386,0.04463641809759822,0.04167585667427448,0.0388827836051418,0.03604608454052277,0.033304960430999175,0.03060809220840449,0.027952952887639493,0.025359146185103376,0.022832861932172044,0.020370784947590925,0.017979055383501308,0.015666129177374533,0.013449978144655372,0.0113201258591068,0.009302254807509736,0.007397829094439923,0.0056264302423451595,0.004010648891537405,0.002577264143732827,0.0013672330158682189,0.0004509986577261871]
        plot!(pl, R_f30, D_f30, lc=:gray, ls=:dash, label="RS K=δ₃ Λ=f₁*δ₁+(1-f₁)δ₂")
    end
    if f3
        # RS curve for degree-3 factors and varying portion of degree-3 variables
        R_f3 = [0.33333333333333337,0.33000000000000007,0.32666666666666666,0.32333333333333336,0.31999999999999995,0.31666666666666676,0.31333333333333335,0.31000000000000005,0.30666666666666664,0.30333333333333334,0.29999999999999993,0.29666666666666675,0.29333333333333333,0.29000000000000004,0.2866666666666666,0.2833333333333333,0.2799999999999999,0.2766666666666667,0.2733333333333333,0.2699999999999999,0.2666666666666666,0.2633333333333333,0.2599999999999999,0.2566666666666667,0.2533333333333333,0.25,0.2466666666666667,0.2433333333333333,0.23999999999999988,0.2366666666666667,0.2333333333333334,0.2300000000000001,0.22666666666666668,0.22333333333333327,0.22000000000000008,0.21666666666666679,0.21333333333333326,0.20999999999999996,0.20666666666666667,0.20333333333333348,0.19999999999999984,0.19666666666666666,0.19333333333333336,0.18999999999999995,0.18666666666666654,0.18333333333333324,0.18000000000000005,0.17666666666666675,0.17333333333333334,0.16999999999999993,0.16666666666666663,0.16333333333333344,0.16000000000000003,0.15666666666666662,0.15333333333333332,0.15000000000000002,0.1466666666666666,0.1433333333333332,0.14,0.13666666666666671,0.13333333333333341,0.1299999999999999,0.1266666666666666,0.1233333333333334,0.1200000000000001,0.11666666666666659,0.11333333333333329,0.10999999999999999,0.1066666666666668,0.10333333333333339,0.10000000000000009,0.09666666666666668,0.09333333333333327,0.08999999999999997,0.08666666666666678,0.08333333333333337,0.07999999999999996,0.07666666666666666,0.07333333333333336,0.06999999999999995,0.06666666666666654,0.06333333333333335,0.05999999999999994,0.05666666666666664,0.053333333333333344,0.050000000000000155,0.04666666666666652,0.043333333333333335,0.040000000000000036,0.036666666666666736,0.033333333333333215,0.029999999999999916,0.026666666666666727,0.023333333333333428,0.020000000000000018,0.01666666666666672,0.013333333333333308,0.009999999999999898,0.00666666666666671,0.0033333333333334103,0.0]
        D_f3 = [0.20962056095628745,0.21045655312219785,0.21129398045076497,0.21213285191160997,0.21297317651789593,0.213814963331392,0.21465822146743357,0.21550296009981335,0.21634918846557782,0.2171969158697719,0.21804615169011754,0.2188969053816388,0.21974918648125297,0.22060300461230448,0.22145836948909192,0.22231529092135494,0.22317377881874867,0.22403384319531316,0.22489549417392346,0.22575874199075563,0.2266235969997512,0.22749006967709307,0.22835817062570685,0.22922791057977476,0.23009930040929016,0.23097235112463632,0.23184707388120812,0.23272347998408488,0.2336015808927387,0.23448138822581466,0.2353629137659588,0.23624616946472055,0.23713116744752138,0.23801792001870203,0.23890643966665281,0.2397967390690277,0.2406888310980555,0.24158272882594994,0.24247844553042286,0.2433759947003037,0.24427539004128584,0.24517664548178786,0.2460797751789402,0.24698479352472213,0.2478917151522217,0.248800554942059,0.2497113280289589,0.2506240498084894,0.2515387359439647,0.2524554023735355,0.25337406531746137,0.25429474128557106,0.25521744708493777,0.2561421998277597,0.2570690169394544,0.25799791616700474,0.2589289155875188,0.2598620336170625,0.26079728901973587,0.2617347009170402,0.26267428879751364,0.26361607252667024,0.2645600723572469,0.26550630893977045,0.2664548033334655,0.26740557701750556,0.26835865190263664,0.2693140503431725,0.2702717951493969,0.271231909600374,0.2721944174571924,0.27315934297666117,0.27412671092547497,0.2750965465948728,0.2760688758158149,0.2770437249746786,0.2780211210295341,0.27900109152699526,0.2799836646196798,0.28096886908431673,0.28195673434051494,0.28294729047022854,0.283940568237963,0.2849365991117401,0.2859354152848619,0.28693704969851397,0.28794153606524925,0.2889489088934002,0.28995920351242044,0.290972456099293,0.2919887037059378,0.2930079842877797,0.29403033673345685,0.2950558008957639,0.2960844176238681,0.29711622879689686,0.298151277358935,0.2991896073555008,0.30023126397162553,0.30127629357155217,0.30232474374018364]
        plot!(pl, R_f3, D_f3, label="RS K=δ₃ Λ=(1-f₃)δ₂+f₃*δ₃", lc=:magenta)
    end
    xlabel!(pl, "R"); ylabel!(pl, "D")
    pl
end



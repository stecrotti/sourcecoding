using GaloisFields, DelimitedFiles, OffsetArrays

function build_tables(q::Int)
    if q==2
        elems = [0,1]
    else
        G,x = GaloisField(q,:x)
        elems = collect(G)
    end
    M = [findfirst(isequal(x*y),elems)-1 for x in elems, y in elems]
    gfmult = OffsetArray(M, 0:q-1, 0:q-1)

    gfdiv = OffsetArray(zeros(Int, q,q-1), 0:q-1,1:q-1)
    for r in 1:q-1
        for c in 1:q-1
            gfdiv[r,c] = findfirst(isequal(r), [gfmult[c,k] for k in 1:q-1])
        end
    end

    D = cat(-ones(Int,q), gfdiv.parent, dims=2) .+ 1
    M.+1,D
end

function write_tables(q::Int)
    M,D = build_tables(q)
    writedlm("gf$(q)_mul.txt", M)
    writedlm("gf$(q)_div.txt",D)
end

function write_all_tables(qs::Vector{Int}=2 .^ (1:8))
    for q in qs
        write_tables(q)
    end
end
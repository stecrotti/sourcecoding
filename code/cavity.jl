# computes in linear time and inplace dst[i] = 
# src[1] op … src[i-1] op src[i+1] op … src[end] op init
# returns src[1] op … op src[end] op init

function cavity!(dest, source, op, init)
    @assert length(dest) == length(source)
    isempty(source) && return init
    if length(source) == 1
        dest[begin] = init 
        return op(first(source), init)
    end
    Iterators.accumulate!(op, dest, source)
    full = op(dest[end], init)
    right = init
    for (i,s)=zip(lastindex(dest):-1:firstindex(dest)+1,Iterators.reverse(source))
        dest[i] = op(dest[i-1], right);
        right = op(s, right);
    end
    dest[begin] = right
    full
end


# Non-allocating version for cases where the elements of source and dest are
#  themselves vectors
# op! must be of the form op!(out, in1, in2)
function cavity!(dest::Vector{Vector{T}}, source::Vector{Vector{T}}, op!, init;
        full = similar(source[1]), right = similar(dest[1])) where T
    @assert length(dest) == length(source)
    isempty(source) && return init
    if length(source) == 1
        dest[begin] .= init 
        op!(full, first(source), init)
        return full
    end
    dest[1] = source[1]
    for i in 2:lastindex(dest)
        op!(dest[i], dest[i-1], source[i])
    end
    right .= init
    for i in lastindex(dest):-1:firstindex(dest)+1
        op!(dest[i], dest[i-1], right)
        op!(right, source[i], right)
    end
    copyto!(dest[begin], right)
    op!(full, dest[end], init)
    full
end

# mult!(a, b, c) = begin
#     a .= b.*c
# end

# function time_test()
#     dest = [randn(10) for _ in 1:4]
#     source = [randn(10) for _ in 1:4]
#     op! = mult! 
#     init = ones(10)
#     full = similar(source[1])
#     right = similar(dest[1])

#     @assert length(dest) == length(source)
#     @timeit to "nest 1" begin               
#         isempty(source) && return init
#         if length(source) == 1
#             @timeit to "level 1.1" dest[begin] .= init 
#             @timeit to "level 1.2" op!(full, first(source), init)
#             return full
#         end
#     end
#     @timeit to "accum" begin
#         # Iterators.accumulate!(.*, dest, source)
#         dest[1] = source[1]
#         for i in 2:lastindex(dest)
#             op!(dest[i], dest[i-1], source[i])
#         end
#     end
#     @timeit to "nest 2" copyto!(right, init)
#     @timeit to "nest 3" begin
#         for i in lastindex(dest):-1:firstindex(dest)+1
#             @timeit to "level 3.1" op!(dest[i], dest[i-1], right)
#             @timeit to "level 3.2" op!(right, source[i], right)
#         end
#         @timeit to "level 3.3" dest[begin] .= right
#         @timeit to "level 3.4" op!(full, dest[end], init)
#     end
# end

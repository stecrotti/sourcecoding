# computes in linear time and inplace dst[i] = 
# src[1] op … src[i-1] op src[i+1] op … src[end] op init
# returns src[1] op … op src[end] op init
# 1-base indexing required

function cavity!(dest, source, op, init)
    @assert length(dest) == length(source)
    isempty(source) && return init
    if length(source) == 1
        dest[begin] = init 
        return op(source[begin], init)
    end
    if length(source) == 2
        dest[begin] = op(source[end], init)
        dest[end] = op(source[begin], init)
        return op(dest[begin], source[begin])
    end
    accumulate!(op, dest, source)
    full = op(dest[end], init)
    right = init
    for i=length(source):-1:2
        dest[i] = op(dest[i-1], right);
        right = op(source[i], right);
    end
    dest[1] = right
    full
end


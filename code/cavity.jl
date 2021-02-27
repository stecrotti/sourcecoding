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
    if length(source) == 2
        dest[begin] = op(first(Iterators.reverse(source)), init)
        dest[end] = op(first(source), init)
        return op(dest[begin], first(source))
    end
    dest[begin], rest = Iterators.peel(source)
    for (s,i) = zip(rest, eachindex(dest))
        dest[i+1] = op(dest[i], s)
    end
    full = op(dest[end], init)
    right = init
    for (i,s)=zip(lastindex(dest):-1:firstindex(dest)+1,Iterators.reverse(source))
        dest[i] = op(dest[i-1], right);
        right = op(s, right);
    end
    dest[begin] = right
    full
end


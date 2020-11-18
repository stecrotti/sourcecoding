## Loops as basis elements
The problem amounts to a choice to include or not every basis vector.
If loops are not adjacent -> decoupling and reduction of complexity.
Is there any hope of finding decoupled loops in large graphs?
Or maybe b-reduction breaks one of these loops?

A pool of loops is a connected component in the dual graph. 
The size of pools (number of loops in a pool) can be checked by computing the sizes of such connected components.
If the size grows subextensively....


## Rank of problem if all variables have degree equal to 2
> rk = #factors - #CC

(From Euler's formula). Does this work only for planar graphs??

## Sometimes removing a factor doesn't change anything
Suppose the factor has degree 2 and is part of a loop. 
When you remove it, the number of solutions is left unchanged: a loop was removed but a path was created.

Monte Carlo over basis coefficients in this case won't feel any change, because the energy for vectors satisfying all checks will be the same as it was before the b-reduction

## Region graphs
https://www.ics.uci.edu/~fowlkes/papers/planar.pdf: planar graphs with variable node degrees <= 2 are 'good' for region-based free energy approximations.

https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
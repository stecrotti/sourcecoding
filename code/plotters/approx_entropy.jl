function newton(f::Function, fprime::Function, x::Real;
                tol::Float64=1e-12, maxiter::Int=100, verbose::Bool=false)

        it = 0
        y = x
        while abs(f(y)) > tol
          it ≥ maxiter && return (:uncoverged, y, f(y),it)
          it += 1
          y = x - f(x)/fprime(x)
          verbose && println("it: $it sol: $y val: $f(y)")
          x = y
        end
        return (:converged, y, f(y), it)
end

# Approximate the inverse of binary entropy function
# Start from a very crude approx 1/2*x^2 and then refine with Newton
function H2inv(y::Real;
    tol::Float64=1e-12, maxiter::Integer=100, verbose::Bool=false)

    (res, h2inv, err, it) = newton(x->H2(x)-y, H2prime, 0.5*y^2,
                            tol=tol, maxiter=maxiter, verbose=verbose)
    if res == :converged
      return h2inv
    else
      error("Something went wrong with Newton's method")
      return nothing
    end
end

H2prime(D::Real) = log2((1-D)/D)

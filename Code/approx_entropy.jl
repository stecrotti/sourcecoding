include("headers.jl")

function Newton(f::Function, f_prime::Function, a::Number;
                tol::Float64=1e-12, maxiter::Integer=100, verbose::Bool=false)

        it = 0
        b = a
        while abs(f(a)) > tol
          it ≥ maxiter && return (:uncoverged, b, f(b),it)
          it += 1
          b = a - f(a)/f_prime(a)
          verbose && println("it: $it sol: $b val: $f(b)")
          a = b
        end
        return (:converged, b, f(b), it)
end

function H2inv(y::Number;
    tol::Float64=1e-12, maxiter::Integer=100, verbose::Bool=false)

    (res, h2inv, err, it) = Newton(x->H2(x)-y, H2prime, 0.5*y^2,
                            tol=tol, maxiter=maxiter, verbose=verbose)
    if res == :converged
      return h2inv
    else
      error("Something went wrong with Newton's method")
      return nothing
    end
end

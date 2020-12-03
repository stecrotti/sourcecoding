# This code computes the averaged Weigth Enumerator Function (WEF), using the expression (11.17) in the book by Mezard and Montanari. The function Phi is the annealed average, Phi(w)=lim (1/N)\log E[WEF(w)], with w\in[0,1], that gives the log of averaged fraction of solutions at distance Nw from 0. The code solves the three equations in (11.18), (11.19) for the special degree profile P(X)=X^k, \Lambda(X)=lambda1*X+lambda2*X^2 and plug the solution (x,y,z) in the expression (11.17) of Phi
# The function Phi takes in arguments the Hamming distance w\in[0,1], the fraction lambda1 of variables with degree 1, the fraction lambda2=1-lambda1 of variables with degree 2, the function degree k, and a starting point y0 for the resolution of the fixed point equation y=F(w, y, lambda1, lambda2, k)

function h(w, x, y, lambda1, lambda2) # equation (11.18)
    t1=1+x*y
    t2=1+x*(y^2)
    return w*t1*t2 - t2*lambda1*x*y - t1*lambda2*x*(y^2)
end

function f(z, k) # 1st equation (11.19) y = f(z, k)
    num = (1+z)^(k-1)-(1-z)^(k-1)
    den = (1+z)^(k-1)+(1-z)^(k-1) 
    return num/den
end

function u(x, y, lambda1, lambda2) # 2nd equation (11.19) z = u(x, y) 
    t1=1+x*y
    t2=1+x*(y^2)
    num = lambda1*x*t2 + 2*lambda2*x*y*t1
    den = lambda1*t2 + 2*lambda2*t1
    return num/den
end

function xsol(w, y, lambda1, lambda2)
# resolution h=0 on x at fixed w, y
    A = (y^3)*(1-w)
    B = lambda1*y + lambda2*(y^2) - w*(y + y^2)
    C = -w
    Delta = B^2 - 4*A*C
    if Delta < 0
        println("negative discriminant")
        exit()
    end
    sol = ( -B + sqrt(Delta) )/(2*A)
    return sol
end

function g(w, y, lambda1, lambda2)
# substitute the solution x=xsol(y) in the function u
    tmp = xsol(w, y, lambda1, lambda2)
    return u(tmp, y, lambda1, lambda2)
end

function F(w, y, lambda1, lambda2, k)
    tmp = g(w, y, lambda1, lambda2)
    return f(tmp, k)
end

function ysol(w, lambda1, lambda2, k, y0)
    return find_zero(y -> F(w, y, lambda1, lambda2, k)-y, y0)
end

function fixedpoint(f, x0; tol = 10^(-10), maxiter=10^15)    
    residual = tol + 1
    iter = 1
    xold = x0
    while residual > tol && iter < maxiter
        xnew = f(xold)        
        residual = abs(xold - xnew);
        xold = xnew
        iter += 1
    end
    return xold
end

function ysolfp(w, lambda1, lambda2, k, y0)
    y = fixedpoint(y -> F(w, y, lambda1, lambda2, k), y0)
end

function Phi(w, lambda1, lambda2, k, y0)
    if w == 0
        P = 0
    else
        m = lambda1 + 2*lambda2
        ys = ysolfp(w, lambda1, lambda2, k, y0)
        xs = xsol(w, ys, lambda1, lambda2)
        zs = u(xs, ys, lambda1, lambda2)
        q = 0.5*((1+zs)^k + (1-zs)^k)
        P = - w*log(xs) - m*log(1 + ys*zs) + lambda1*log(1 + xs*ys) + lambda2*log(1 + xs*(ys^2)) + m*log(q)/k
    end
    return P
end

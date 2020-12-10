# This code computes the averaged Weigth Enumerator Function (WEF), using the expression (11.17) in the book by Mezard and Montanari. The function Phi is the annealed average, Phi(w)=lim (1/N)\log E[WEF(w)], with w\in[0,1], that gives the log of averaged fraction of solutions at distance Nw from 0. The code solves the three equations in (11.18), (11.19) for degree profile P(X)=\sum_k P_k*X^k, \Lambda(X)=lambda1*X+lambda2*X^2 and plug the solution (x,y,z) in the expression (11.17) of Phi
# The function Phi takes in arguments the Hamming distance w\in[0,1], the fraction lambda1 of variables with degree 1, the degree profile of the check nodes P (an Array of tuples [(k_min, P_{k_min}), ..., (k_max, P_{k_max})] ), and a starting point y0 for the resolution of the fixed point equation y=F(w, y, lambda1, P)

function xsol(w, y, lambda1)
# resolution equation (11.18) on x at fixed w, y
	lambda2 = 1-lambda1
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

function f(z, P) # 1st equation (11.19) y = f(z; P) with a generic degree profile P(x)=\sum_k P_k x^k
	num = 0
	den = 0
	for i =1:length(P)
		k = P[i][1]
		Pk = P[i][2]
		num += k*Pk*( (1+z)^(k-1) - (1-z)^(k-1) )/( (1+z)^k + (1-z)^k )
		den += k*Pk*( (1+z)^(k-1) + (1-z)^(k-1) )/( (1+z)^k + (1-z)^k )
	end
    return num/den
end

function u(x, y, lambda1) # 2nd equation (11.19) z = u(x, y) 
	lambda2 = 1-lambda1
    t1=1+x*y
    t2=1+x*(y^2)
    num = lambda1*x/t1 + 2*lambda2*x*y/t2
    den = lambda1/t1 + 2*lambda2/t2
    return num/den
end

function g(w, y, lambda1)
# substitute the solution x=xsol(y) in the function u
    tmp = xsol(w, y, lambda1)
    return u(tmp, y, lambda1)
end

function F(w, y, lambda1, P)
# inject z=u(xsol(y), y) in the function f
    tmp = g(w, y, lambda1)
    return f(tmp, P)
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

function ysolfp(w, lambda1, P, y0)
    y = fixedpoint(y -> F(w, y, lambda1, P), y0)
end

function Phi(w, lambda1, P, y0)
	lambda2 = 1-lambda1
    if w == 0
        res = 0
    else
        mL = lambda1 + 2*lambda2
        ys = ysolfp(w, lambda1, P, y0)
        xs = xsol(w, ys, lambda1)
        zs = u(xs, ys, lambda1)
        mP = 0
        logq = 0
		for i =1:length(P)
			k = P[i][1]
			Pk = P[i][2]
			mP += k*Pk
	        q = 0.5*( (1+zs)^k + (1-zs)^k )
			logq += Pk*log(q)
		end       
        res = - w*log(xs) - mL*log(1 + ys*zs) + lambda1*log(1 + xs*ys) + lambda2*log(1 + xs*(ys^2)) + mL*logq/mP
    end
    return res
end

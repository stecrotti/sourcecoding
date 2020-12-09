# This code computes E[\cal{N}(d)] the averaged number of solutions at a distance d=N*delta from a random source vector x_*. The components of x_* are i.i.d from a Bernouilli law of parameter p\in[0,1]. It uses the WEF function Phi writen in "SolverWEF.jl" The function computed is Psi, the annealed average : Psi(delta)=lim (1/N) log E[\cal{N}(d)], with delta\in[0,1]. It performs a maximization over w\in[0,1] : Psi(delta, p)=sup_w{Phi(w)+Xsi(delta, w, p)}, where Phi is the exponential growth rate of the WEF (the average number of solutions at distance w from 0, that depends on the degree profiles), and where Xsi is a term dependent on the randomness of the source vector.

function solalpha(delta, w, p)
	if delta == 1
		sol = 0
	else
	    A = delta
    	B = delta*( (1-p)/p + p/(1-p) ) - (1-w)*p/(1-p) - w*(1-p)/p
    	C = delta - 1
    	Delta = B^2-4*A*C
    	if Delta < 0
    	    println("negative discriminant")
    	    exit()
    	end
    	sol = ( -B + sqrt(Delta) )/(2*A)
    end
    return sol
end

function Xsi(delta, w, p)
	if delta == 0 #then u0=0 and u1=0
		res = w*log(p) + (1 - w)*log(1-p)
	elseif delta == 1 #then u0=1-w and u1=w
		res = (1-w)*log(p) + w*log(1-p)
	else
	    alpha = solalpha(delta, w, p)
	    u0 = (1 - w)/(1 + alpha*(1 - p)/p)
	    u1 = w/(1 + alpha*p/(1 - p))
	    res = w*log(w) + (1 - w)*log(1 - w) - u0*log(u0) - (1 - w - u0)*log(1 - w - u0) - u1*log(u1) - (w - u1)*log(w - u1) + (w + u0 - u1)*log(p) + (1 - w - u0 + u1)*log(1-p)
	end
    return res
end

function Psi(delta, p, lambda1, P, y0, len::Int64=100)
# increase 'len' the length of the vector Xs+Ph to increase precision in finding maximum
	if p == 0
    	res = Phi(delta, lambda1, P, y0)
    elseif p == 1
    	res = Phi(1 - delta, lambda1, P, y0)
    else
		W=range(0.001,0.999,length=len)
		tP(w)=Phi(w, lambda1, P, y0)
		Ph=tP.(W)
		Xs=Xsi.(delta, W, p)
		res = findmax(Xs+Ph)
		#println("position of max: ", W[res[2]]) # look at the position of the maximum
		res = res[1] 
	end
    return res
end

function zeroPsi(p, lambda1, P, y0, len::Int64=100, tol::Float64=10^(-10))
# set tolerance tol=xatol for bisection algorithms
	dinit = 0.001
	dend = 0.5
	Pinit = Psi(dinit, p, lambda1, P, y0, len)
	Pend = Psi(dend, p, lambda1, P, y0, len)
	if Pinit*Pend > 0
		#println("same sign")
		delta0 = 0.
	else
		delta0 = find_zero(delta -> Psi(delta, p, lambda1, P, y0, len), (dinit,dend); xatol = tol)
	end
	return delta0
end


using Optim, Statistics, ForwardDiff, LinearAlgebra, Printf

# ----------------------------------------- #
# this function calculates a GMM estimate by:
# (1) running LBFGS for a fixed number of iterations
# (2) updating the weighting matrix
# (3) running Newton's method with the new weighting matrix
# (4) doing a one-step update with optimal weighting
# (5) reporting all estimates
function estimate_gmm(x0,gfunc!,W,N,nresids,args...)
    nmom = size(W,1)
    # step (1)
    r1 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x0,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=200))
    # step (2)
    Ω = moment_variance(r1.minimizer,gfunc!,N,nmom,nresids,args...)
    W = inv(Ω)
    # step (3)
    r2 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),r1.minimizer,Newton(),autodiff=:forward,Optim.Options(show_trace=true))
    # calculating this variance for now just to compare
    V = parameter_variance_gmm(r2.minimizer,gfunc!,W,N,nresids,args...)
    # step (4)
    Ω = moment_variance(r2.minimizer,gfunc!,N,nmom,nresids,args...)
    W = inv(Ω)
    r2 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),r2.minimizer,Newton(),autodiff=:forward,Optim.Options(show_trace=true))

    dG = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),r2.minimizer)
    avar = inv(dG'*W*dG)
    #gn = moment_func(r2.minimizer,gfunc!,N,nmom,nresids,args...)
    x_est = r2.minimizer# .-  avar*dG'*W*gn

    # return some output
    var = avar / N
    se = sqrt.(diag(var))
    se_old = sqrt.(diag(V))
    return (est1 = x_est,est2 = r1.minimizer,Ω = Ω,avar = avar,se = se,se_old=se_old)
end

# ----- function that calculates the sample mean of gfunc
# idea: pass other arguments here
# g = N^{-1}∑gfunc(x,i...)
# gfunc = [ϵ_{1,i} × Z_{1,i}, ϵ_{2,i} × Z_{2,i}, .... ]
function gmm_criterion(x,gfunc!,W,N,nresids,args...)
    nmom = size(W)[1]
    g = moment_func(x,gfunc!,N,nmom,nresids,args...)
    return g'*W*g / 2
end
# - used in gmm_criterion, calculates sample moments
function moment_func(x,gfunc!,N,nmom,nresids,args...)
    g = zeros(typeof(x[1]),nmom) #<- pre-allocate an array for the moment function to write to
    resids = zeros(typeof(x[1]),nresids) #<- pre-allocate an array to write the residuals.
    for n in 1:N
        #println(n)
        gfunc!(x,n,g,resids,args...)
    end
    g /= N #
end

# ----------------------------------------- #
# --- function that calculates the variance of the moment
function moment_variance(x,gfunc!,N,nmom,nresids,args...)
    G = zeros(N,nmom)
    resids = zeros(nresids)
    for n in 1:N
        @views gfunc!(x,n,G[n,:],resids,args...)
    end
    return cov(G)
end

# -------
# - Given estimates, a function that calculates the parameter variance
function parameter_variance_gmm(x_est,gfunc!,W,N,nresids,args...)
    nmom = size(W)[1]
    dg = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),x_est)
    Σ = moment_variance(x_est,gfunc!,N,nmom,nresids,args...)
    bread = inv(dg'*W*dg)
    peanut_butter = dg'*W*Σ*W*dg
    return (1/N)  * bread * peanut_butter * bread'
end

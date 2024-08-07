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
    return (est = x_est,Ω = Ω,avar = avar,se = se)
end

function bootstrap_gmm(x0,gfunc!,W,N,nresids,args...;ntrials=50,seed=71024,trace=true)
    np = length(x0)
    Xb = zeros(np,ntrials)
    Random.seed!(seed)
    for b in 1:ntrials
        index = rand(1:N,N)
        if trace
            println("Doing bootstrap trial $b of $ntrials")
        end
        r1 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...;index),x0,Newton(),autodiff=:forward,Optim.Options(iterations=1))
        Xb[:,b] .= r1.minimizer
    end
    return Xb
end

# ----- function that calculates the sample mean of gfunc
# idea: pass other arguments here
# g = N^{-1}∑gfunc(x,i...)
# gfunc = [ϵ_{1,i} × Z_{1,i}, ϵ_{2,i} × Z_{2,i}, .... ]
function gmm_criterion(x,gfunc!,W,N,nresids,args...;index = 1:N)
    nmom = size(W)[1]
    g = moment_func(x,gfunc!,N,nmom,nresids,args...;index)
    return g'*W*g / 2
end
# - used in gmm_criterion, calculates sample moments
function moment_func(x,gfunc!,N,nmom,nresids,args...;index = 1:N)
    g = zeros(typeof(x[1]),nmom) #<- pre-allocate an array for the moment function to write to
    resids = zeros(typeof(x[1]),nresids) #<- pre-allocate an array to write the residuals.
    for n in index
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

# -----
# - Conduct a lagrange multiplier of dg/dx_est where x_est are restricted estimates and g is a relaxed version of the function
# this test only works if W is the inverse of a consistent estimate of the variance of g
function LM_test(x_est,r,gfunc!,W,N,nresids,args...)
    nmom = size(W,1)

    dG = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),x_est)
    gn = moment_func(x_est,gfunc!,N,nmom,nresids,args...)
    Binv = inv(dG'*W*dG)

    # test statistic and p-value
    pre = dG'*W*gn
    test_stat = N*pre'*Binv*pre
    pval = 1-cdf(Chisq(r),test_stat)
    return test_stat,pval
end

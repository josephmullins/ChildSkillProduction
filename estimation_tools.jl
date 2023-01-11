using Optim, Statistics, ForwardDiff, LinearAlgebra

# ---------- Utility functions for GMM estimation ------------ #

# function that calculates the sample mean of gfunc
function gmm_criterion(x,gfunc!,data,W,N)
    nmom = size(W)[1]
    g = moment_func(x,gfunc!,data,N,nmom)
    return g'*W*g
end

# when we benchmarked, didn't seem like the type instability cost anything
function moment_func(x,gfunc!,data,N,nmom)
    g = zeros(typeof(x[1]),nmom)
    for n in 1:N
        gfunc!(x,g,data,n)
    end
    g /= N #
end

# function moment_func(x,gfunc!,data,N,nmom)
#     g = zeros(nmom)
#     for n in 1:N
#         gfunc!(x,g,data,n)
#     end
#     g /= N #
# end

# function that calculate the variance of the moment
function moment_variance(x,gfunc!,data,N,nmom)
    G = zeros(N,nmom)
    for n in 1:N
        @views gfunc!(x,G[n,:],data,n)
    end
    return cov(G)
end

function parameter_variance_gmm(x_est,gfunc!,data,W,N)
    nmom = size(W)[1]
    dg = ForwardDiff.jacobian(x->moment_func(x,gfunc!,data,N,nmom),x_est)
    Σ = moment_variance(x_est,gfunc!,data,N,nmom)
    bread = inv(dg'*W*dg)
    peanut_butter = dg'*W*Σ*W*dg
    return (1/N)  * bread * peanut_butter * bread'
end

# when a M-valued function *resid* must be interacted with M different sets of instruments
# NOTE: this is really two functions! maybe make it one?
function stack_gmm!(x,g,rvec,resid!,data,z_vars,n)
    resid!(x,rvec,data,n) #fill in rvec with residuals
    pos = 1
    for k in eachindex(z_vars) 
        for m in eachindex(z_vars[k])
            zv = z_vars[k][m]
            g[pos] += rvec[k]*data[n,zv]
            pos += 1
        end
    end
end
# this version assumes we have the residuals already. I think this is better
function stack_moments!(g,rvec,data,z_vars,n)
    pos = 1
    for k in eachindex(z_vars) 
        for m in eachindex(z_vars[k])
            zv = z_vars[k][m]
            g[pos] += rvec[k]*data[n,zv]
            pos += 1
        end
    end
end

function estimate_gmm_iterative(x0,gfunc!,data,W,N,iter)
    # x0: the initial parameter guess
    # gfunc: the function used to evalute the moment: gn(x) = (1/N)*∑ gfunc(x,i)
    # W: the initial weighting matrix
    # lb,ub: lower and upper bounds on parameter space
    # iter: number of iterations

    x1 = x0
    nmom = size(W)[1]
    for i=1:iter
        println("----- Iteration $i ----------")
        res = optimize(x->gmm_criterion(x,gfunc!,data,W,N),x1,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=30))
        x1 = res.minimizer
        Ω = moment_variance(x1,gfunc!,data,N,nmom)
        W = inv(Ω)
    end
    println("----- Final Iteration ----------")
    res = optimize(x->gmm_criterion(x,gfunc!,data,W,N),x1,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true))
    x1 = res.minimizer
    V = parameter_variance_gmm(x1,gfunc!,data,W,N)
    return x1,sqrt.(diag(V))
end


# ---- Other Utility Functions

# function to make a set of dummy variables in the dataframe
function make_dummy(data,var::Symbol)
    vals = unique(skipmissing(data[!,var]))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end

# function to take a linear combination given a vector and a list of variables
# assumption: we have type dummies in data. then it's straightforward.
function linear_combination(β,vars,data,n)
    r = 0 #<- not assuming a constant term
    for j in eachindex(vars)
        if vars[j]==:const
            r += β[j]
        else
            @views r += β[j]*data[n,vars[j]]
        end
    end
    return r
end



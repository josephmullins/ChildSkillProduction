using ForwardDiff
using NLopt
using LinearAlgebra

# this function conducts an iterative gmm procedure
# at each stage the weighting matrix is updated to the inverse of the moment variance given current estimates
function EstimateGMMIterative(x0,gfunc,data,W,N,lb,ub,iter)
    # x0: the initial parameter guess
    # gfunc: the function used to evalute the moment: gn(x) = (1/N)*∑ gfunc(x,i)
    # W: the initial weighting matrix
    # lb,ub: lower and upper bounds on parameter space
    # iter: number of iterations

    np = length(x0)
    opt = Opt(:LD_LBFGS,np)
    lower_bounds!(opt,lb)
    upper_bounds!(opt,ub)
    maxeval!(opt,30)
    x1 = x0
    for i=1:iter
        println("----- Iteration $i ----------")
        min_objective!(opt,(x,g)->GMMCriterion(x,g,gfunc,data,W,N))
        res = optimize(opt,x1)
        x1 = res[2]
        Ω = MomentVariance(x1,gfunc,data,W,N)
        W = inv(Ω)
    end
    println("----- Final Iteration ----------")
    maxeval!(opt,1000000)
    min_objective!(opt,(x,g)->GMMCriterion(x,g,gfunc,data,W,N))
    res = optimize(opt,x1)
    V = ParameterVariance(res[2],gfunc,data,W,N)
    return res,sqrt.(diag(V))
end

# evaluates gn'*W*gn where gn is the sample mean of gfunc
function GMMCriterion(x,gfunc,data,W,N)
    np = length(x)
    K = size(W)[1]
    gn = zeros(Real,K)
    for i=1:N
        gi = gfunc(x,data,i)
        gn += gi
    end
    gn /= N
    Qn = gn'*W*gn
    return Qn
end

# same as above but fills in a derivative
function GMMCriterion(x,g,gfunc,data,W,N)
    np = length(x)
    K = size(W)[1]
    dG = zeros(np,K)
    gn = zeros(K)
    for i=1:N
        gi = gfunc(x,data,i)
        dgi = ForwardDiff.jacobian(y->gfunc(y,data,i),x)
        gn += gi
        dG .+= dgi'
    end
    dG /= N
    gn /= N
    Qn = gn'*W*gn
    println(Qn)
    g[:] = 2*dG*W*gn
    return Qn
end


function ParameterVariance(x,gfunc,data,W,N)
    V = MomentVariance(x,gfunc,data,W,N)
    np = length(x)
    K = size(W)[1]
    gn,dG = MomentFunc(x,gfunc,data,W,N)
    bread = inv(dG*W*dG')*dG*W
    meat = V
    return (1/N)*bread*meat*bread'
end

# returns a vector of moments as well as the Jacobian using ForwardDiff
function MomentFunc(x,gfunc,data,W,N)
    # x is parameter 
    # gfunc evaluates a function at each i such that gn = N^{-1}∑gfunc(x,data,i)
    # data is the data object used in gfunc
    # W: is the weighting matrix
    # N: the number of observations
    np = length(x)
    K = size(W)[1]
    dG = zeros(np,K)
    gn = zeros(K)
    for i=1:N
        gi = gfunc(x,data,i)
        dgi = ForwardDiff.jacobian(y->gfunc(y,data,i),x)
        gn += gi
        dG .+= dgi'
    end
    dG /= N
    gn /= N
    return gn,dG
end

# evalutes the variance of the moment function: E[gfunc*gfunc']
# we need this to evaluate the parameter variance
function MomentVariance(x,gfunc,data,W,N)
    np = length(x)
    K = size(W)[1]
    G = zeros(N,K)
    for i=1:N
        G[i,:] = gfunc(x,data,i)
    end
    return cov(G)
end

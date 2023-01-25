using Optim, Statistics, ForwardDiff, LinearAlgebra

# ---------- Utility functions for GMM estimation ------------ #

# function that calculates the sample mean of gfunc
# idea: pass other arguments here
# g = N^{-1}∑gfunc(x,i...)
# gfunc = [ϵ_{1,i} × Z_{1,i}, ϵ_{2,i} × Z_{2,i}, .... ]
function gmm_criterion(x,gfunc!,W,N,nresids,args...)
    nmom = size(W)[1]
    g = moment_func(x,gfunc!,N,nmom,nresids,args...)
    return g'*W*g
end

# issue: want the type of the storage vector to change
# when we benchmarked, didn't seem like the type instability cost anything
function moment_func(x,gfunc!,N,nmom,nresids,args...)
    g = zeros(typeof(x[1]),nmom)
    resids = zeros(typeof(x[1]),nresids)
    for n in 1:N
        gfunc!(x,n,g,resids,args...)
    end
    g /= N #
end


# function that calculate the variance of the moment
function moment_variance(x,gfunc!,N,nmom,nresids,args...)
    G = zeros(N,nmom)
    resids = zeros(nresids)
    for n in 1:N
        @views gfunc!(x,n,G[n,:],resids,args...)
    end
    return cov(G)
end

function parameter_variance_gmm(x_est,gfunc!,W,N,nresids,args...)
    nmom = size(W)[1]
    dg = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),x_est)
    Σ = moment_variance(x_est,gfunc!,N,nmom,nresids,args...)
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
# but it's wrong!
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

function estimate_gmm_iterative(x0,gfunc!,iter,W,N,nresids,args...)
    # x0: the initial parameter guess
    # gfunc: the function used to evalute the moment: gn(x) = (1/N)*∑ gfunc(x,i)
    # W: the initial weighting matrix
    # lb,ub: lower and upper bounds on parameter space
    # iter: number of iterations

    x1 = x0
    nmom = size(W)[1]
    for i=1:iter
        println("----- Iteration $i ----------")
        res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x1,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=30))
        x1 = res.minimizer
        Ω = moment_variance(x1,gfunc!,N,nmom,nresids,args...)
        W = inv(Ω)
    end
    println("----- Final Iteration ----------")
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x1,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true))
    x1 = res.minimizer
    V = parameter_variance_gmm(x1,gfunc!,W,N,nresids,args...)
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

# ---- Functions from main_wages


function get_wage_data(data,vlist::Array{Symbol,1},fe)
    N = size(data)[1]
    index_select = .!ismissing.(data.logwage_m)
    for v in vlist
        index_select = index_select .& (.!ismissing.(data[!,v]))
    end
    d=data[index_select,:]
    lW=Vector{Float64}(d[!,:logwage_m]) #return non-demeans
    if fe

        d=data[index_select,:]
        d=groupby(d,:MID)
        d=transform(d, :logwage_m => mean) 
        d[!,:logwage_m_demean] = d.logwage_m-d.logwage_m_mean #overwrites log_wage with the de-meaned
        lW=Vector{Float64}(d[!,:logwage_m_demean])

        for i in 1:length(vlist)
            d=groupby(d,:MID)
            d=transform(d, vlist[i] => mean) #creates a new mean column
            d[!,vlist[i]] = d[!,vlist[i]]-d[!,end] #subtracts off the newest column, which should be the newly constructed mean
        end
    end
    return lW,Matrix{Float64}(data[index_select,vlist]),d
end

function wage_regression(data,vlist,fe)
    
    if fe

        lW,X,d = get_wage_data(data,vlist,fe) #using the demeaned values
        coef=inv(X'X)*X'lW

        d.resid = missings(Float64, nrow(d))
        tcoef=transpose(coef)
    
        for i in 1:nrow(d)
            d.resid[i]=tcoef*X[i,:]
        end    

        d.resid=d.logwage_m-d.resid #residuals from non-demeaned

        d=groupby(d,:MID)
        d=transform(d, :resid => mean) #a row of the means repeating for each group; if not desired form use select

        return Vector{Float64}(coef),Vector{Float64}(d.resid),d

    else

        lW,X,d = get_wage_data(data,vlist,fe)
        coef=inv(X'X)*X'lW

        d.resid = missings(Float64, nrow(d))
        tcoef=transpose(coef)
    
        for i in 1:nrow(d)
            d.resid[i]=tcoef*X[i,:]
        end    

        d.resid=lW-d.resid #herelW is just the logwage normally

        return Vector{Float64}(coef),Vector{Float64}(d.resid) 
    end
end


function wage_clustering(wage_reg,fe)
    if fe
        df=wage_reg[3]
        dat=df[:,:resid_mean]
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=d.MID,cluster=a)
    else
        df=wage_reg[2]
        dat=df[:,:resid_mean]
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=d.MID,cluster=a)
    end
    return clusters,centers
end



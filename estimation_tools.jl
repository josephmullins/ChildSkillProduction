using Optim, Statistics, ForwardDiff, LinearAlgebra, Printf

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


# ----------- Tools for writing results to file
function write_line!(io,format,M,v::Symbol,i::Int=0,vname::String="")
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            write(io,format(getfield(M[s],v)),"&")
        else
            write(io,format(getfield(M[s],v)[i]),"&")
        end
    end
    write(io,"\\\\","\n")
end

function write_observables!(io,format,formatse,M,SE,specs,labels,var::Symbol,specvar::Symbol,constant=true)

    nspec = length(M)
    if constant
        # write the constant:
        write_line!(io,format,M,var,1,"Const.")
        write_line!(io,formatse,SE,var,1)
    end
    vlist = union([s[specvar] for s in specs]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v] #<-?
        else
            vname = string(v)
        end
        write(io,vname,"&")
        # write estimates
        for j in 1:nspec
            i = findfirst(specs[j][specvar].==v)
            if isnothing(i)
                write(io,"-","&")
            else
                if constant
                    write(io,format(getfield(M[j],var)[1+i]),"&")
                else
                    write(io,format(getfield(M[j],var)[i]),"&")
                end
            end
        end
        write(io,"\\\\")
        # write standard errors
        write(io,"&")
        for j in 1:nspec
            i = findfirst(specs[j][specvar].==v)
            if isnothing(i)
                write(io,"","&")
            else
                if constant
                    write(io,formatse(getfield(SE[j],var)[1+i]),"&")
                else
                    write(io,formatse(getfield(SE[j],var)[i]),"&")
                end
            end
        end
        write(io,"\\\\")

    end
end

# specs are an array
function writetable(M,SE,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    
    nspec = length(M)
    
    # Write the header
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec+1),"}\\\\\\toprule","\n")
    write(io,"&",["($s)&" for s in 1:nspec]...,"\\\\\\cmidrule(r){2-$(nspec+2)}")

    
    # Write the elasticity parameters 
    v = [:ρ,:γ]
    vname = ["\$\\rho\$","\$\\gamma\$"]
    for j in 1:2
        write_line!(io,form,M,v[j],0,vname[j])
        write_line!(io,formse,SE,v[j],0)
    end
    # δ_1 and δ_2
    for j in 1:2
        write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
        write_line!(io,formse,SE,:δ,j)
    end

    # Factor share Parameters:
    # a_{m}
    # write the header:
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{m}\$: Mother's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βm,:vm)
    # a_{f}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{f}\$: Father's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βf,:vf)
    # a_{g}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{g}\$: Goods}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βg,:vg)

    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end
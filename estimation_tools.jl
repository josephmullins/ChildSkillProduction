using Optim, Statistics, ForwardDiff, LinearAlgebra, Printf
using Clustering

# NOTE:
# - The setup below allows for the moment function to take the form:
# g_{ij}(x,θ) = ϵ_{i}(θ)× Z_{j} with a unique set of instruments for each residual
# - the function stack_moments will perform the stacking into a vector of known size


# ---------- Utility functions for GMM estimation ------------ #

# function that calculates the sample mean of gfunc
# idea: pass other arguments here
# g = N^{-1}∑gfunc(x,i...)
# gfunc = [ϵ_{1,i} × Z_{1,i}, ϵ_{2,i} × Z_{2,i}, .... ]
function gmm_criterion(x,gfunc!,W,N,nresids,args...)
    nmom = size(W)[1]
    g = moment_func(x,gfunc!,N,nmom,nresids,args...)
    return g'*W*g / 2
end

# issue: want the type of the storage vector to change
# when we benchmarked, didn't seem like the type instability cost anything
function moment_func(x,gfunc!,N,nmom,nresids,args...)
    g = zeros(typeof(x[1]),nmom) #<- pre-allocate an array for the moment function to write to
    resids = zeros(typeof(x[1]),nresids) #<- pre-allocate an array to write the residuals.
    for n in 1:N
        #println(n)
        gfunc!(x,n,g,resids,args...)
    end
    g /= N #
end


# function that calculates the variance of the moment
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

# when a M-vector of resids, *resid*, must be interacted with M different sets of instruments and stacked.

# this version assumes we have the residuals already. I think this is better
function stack_moments!(g,rvec,data::DataFrame,z_vars,n)
    # g: the vector to add moments to
    # rvec: the vector of residuals
    # z_vars: an array of arrays of instrument names to take from data. z_vars[k] is the array of instrument names for the kth residual
    # n: the row number for the data
    pos = 1
    for k in eachindex(z_vars) 
        for m in eachindex(z_vars[k])
            zv = z_vars[k][m]
            g[pos] += rvec[k]*data[n,zv]
            pos += 1
        end
    end
end

function stack_moments!(g,rvec,data,z_vars,n)
    # g: the vector to add moments to
    # rvec: the vector of residuals
    # z_vars: an array of arrays of instrument names to take from data. z_vars[k] is the array of instrument names for the kth residual
    # n: the row number for the data
    pos = 1
    for k in eachindex(z_vars) 
        for m in eachindex(z_vars[k])
            zv = z_vars[k][m]
            g[pos] += rvec[k]*data[zv][n]
            pos += 1
        end
    end
end

# this function is an iterative estimator that updates the weighting matrix after 30 function calls in the minimization routine, and does this iter times before finishing the minimization routine off.
function estimate_gmm_iterative(x0,gfunc!,iter,W,N,nresids,args...)
    # x0: the initial parameter guess
    # gfunc: the function used to evalute the moment: gn(x) = (1/N)*∑ gfunc(x,i)
    # W: the initial weighting matrix
    # iter: number of iterations

    x1 = x0
    nmom = size(W)[1]
    for i=1:iter
        println("----- Iteration $i ----------")
        res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x1,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=50))
        x1 = res.minimizer
        Ω = moment_variance(x1,gfunc!,N,nmom,nresids,args...)
        W = inv(Ω)
    end
    println("----- Final Iteration ----------")
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x1,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true)) #,x_tol = 1e-3,f_calls_limit=10))
    x1 = res.minimizer
    V = parameter_variance_gmm(x1,gfunc!,W,N,nresids,args...)
    return x1,sqrt.(diag(V))
end

# this function estimates by:
# (1) running LBFGS for a fixed number of iterations
# (2) updating the weighting matrix
# (3) running Newton's method with the new weighting matrix
# (4) doing a one-step update with optimal weighting
# (5) reporting all estimates
# edit this function? to just run Newton again with updated weighting?
function estimate_gmm(x0,gfunc!,W,N,nresids,args...)
    nmom = size(W,1)
    # step (1)
    r1 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x0,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=100))
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

function newton_raphson_approx_step(x1,gfunc!,W,N,nresids,args...)
    dG = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),x1)
    Hinv = inv(dG'*W*dG)
    gn = moment_func(x1,gfunc!,N,nmom,nresids,args...)
    x2 = x1 - Hinv*dG'*W*gn
    return x2,Hinv
end


function newton_raphson_exact_step(x1,gfunc!,W,N,nresids,args...)
    dG = ForwardDiff.jacobian(x->moment_func(x,gfunc!,N,nmom,nresids,args...),x1)
    H = ForwardDiff.hessian(x->gmm_criterion(x,gfunc!,W,N,nresids,args...),x1)
    Hinv = inv(H)
    gn = moment_func(x1,gfunc!,N,nmom,nresids,args...)
    x2 = x1 - Hinv*dG'*W*gn
    return x2,Hinv
end

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

# ---- Other Utility Functions

# function to make a set of dummy variables in the dataframe
function make_dummy(data,var::Symbol)
    vals = sort(unique(skipmissing(data[!,var])))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end

# function to create interaction terms in the data and spit out a list of their names
function make_interactions(data,V1::Vector{Symbol},V2::Vector{Symbol})
    names = []
    for v1 in V1, v2 in V2
        name = Symbol(v1,"_x_",v2)
        data[!,name] = data[!,v1].*data[!,v2]
        push!(names,name)
    end
    return names
end

# function to take a linear combination given a vector and a list of variables
# assumption: we have type dummies in data. then it's straightforward.
function linear_combination(β,vars::Vector{Symbol},data::DataFrame,n::Int64)
    r = 0. #<- not assuming a constant term
    for j in eachindex(vars)
        if vars[j]==:constant #<- this condition is unnecessary: we simply add a variable =1 named :constant to the dataset
            r += β[j]
        else
            r += β[j]*data[n,vars[j]]
        end
    end
    return r
end

function linear_combination(β,vars,data,n)
    r = 0. #<- not assuming a constant term
    for j in eachindex(vars)
        if vars[j]==:constant #<- this condition is unnecessary: we simply add a variable =1 named :constant to the dataset
            r += β[j]
        else
            r += β[j] * data[vars[j]][n]
        end
    end
    return r
end

# ---- Functions to cluster on wages

function get_wage_data(data,vlist::Array{Symbol,1},fe)
    N = size(data)[1]
    d = dropmissing(select(data, [:MID;:logwage_m;vlist]))
    d2 = copy(d)
    # if including individual fixed effect, first de-mean here
    if fe
        demean(x) = x .- mean(x)
        gd = groupby(d2,:MID)
        for v in [:logwage_m;vlist]
            transform!(gd,v => demean => v)
        end
    end
    lW = Vector{Float64}(d2.logwage_m)
    return lW,Matrix{Float64}(d2[!,vlist]),d
end

function predict_wage(data,vlist,β)
    return [linear_combination(β,vlist,data,n) for n=1:size(data,1)]
end

function wage_regression(data,vlist,fe)
    lW,X,d = get_wage_data(data,vlist,fe)
    coef=inv(X'X)*X'lW
    d[!,:resid] = d.logwage_m .- predict_wage(d,vlist,coef)
    return coef,d
end

##reduce to unique cases prior to clustering

function generate_cluster_assignment(dat,vlist,fe,nclusters)
    coef,d = wage_regression(dat,vlist,fe) #<- this function returns d, which drops missing observations from dat and writes a residual called :resid
    # command below takes the mean residual for each individual after dropping missing data
    mean_resids = combine(groupby(d,:MID),:resid => mean)
    result = kmeans(mean_resids[!,:resid_mean]', nclusters; maxiter=100, display=:iter)
    relabel = sortperm(result.centers[:])
    position_map = [findfirst(relabel.==k) for k=1:nclusters] #<- this uses the ordering to map each cluster to its new position
    mean_resids[!,:cluster] = position_map[result.assignments]
    mean_resids[!,:center] = result.centers[result.assignments]
    return mean_resids
end

function cluster_routine_robust(dat,vlist,nclusters,maxiter = 100)
    lW,X,d = get_wage_data(dat,[:constant;vlist],false)
    # get an initial assignment by clustering on residuals

    coef=inv(X'X)*X'lW
    d[!,:resid] = d.logwage_m .- predict_wage(d,[:constant;vlist],coef)
    dn = combine(groupby(d,:MID),:resid => mean)
    result = kmeans(dn[!,:resid_mean]', nclusters; maxiter=100, display=:iter)
    dn[!,:cluster] = result.assignments
    select!(dn,Not(:resid_mean))
    d = innerjoin(d,dn,on=:MID)

    eps = Inf
    iter = 0
    assignment = result.assignments
    gd = groupby(d,:MID)
    μk0 = zeros(nclusters)
    X = X[:,2:end] #<- drop the intercept term
    while (eps>1e-10) & (iter<maxiter)
        iter +=1 
        # calculate the group fixed effect
        μk,β = wagereg_group_fe(d,vlist)
        # calculate the residual error *not including* the fe:
        d[!,:xb] = d[!,:logwage_m] .- X*β
        # get a new assignment of individuals to groups
        μk = 0.5*μk + 0.5*μk0
        assignment_new = [assign_cluster(d,μk) for d in gd]
        eps = sum((μk .- μk0).^2)
        #eps = sum((assignment .- assignment_new).^2)
        dn[!,:cluster] = assignment_new
        assignment[:] = assignment_new
        #println(μk)
        μk0[:] = μk
        select!(d,Not(:cluster))
        d = innerjoin(d,dn,on=:MID)
        println(eps," ",iter)
    end
    relabel = sortperm(μk0) #<- this tells the ordering of the cluster
    position_map = [findfirst(relabel.==k) for k=1:nclusters] #<- this uses the ordering to map each cluster to its new position
    dn[!,:cluster] = position_map[assignment]
    dn[!,:center] = μk0[assignment]
    return dn
end

# function of the data frame
function assign_cluster(d,μk)
    nclusters = length(μk)
    ssq = [sum((d.xb .- μk[k]).^2) for k in 1:nclusters]
    return argmin(ssq)
end

# function: 

function wagereg_group_fe(dat,vlist)
    cluster_dummies=make_dummy(dat,:cluster)
    ncluster = length(cluster_dummies)
    lW,X,d = get_wage_data(dat,[cluster_dummies;vlist],false)
    coef=inv(X'X)*X'lW
    μk = coef[1:ncluster]
    β = coef[ncluster+1:end]
    return μk,β
end


# ----------- Tools for writing results to file

# this function writes to file the value of parameter v in for each element of the vector of results M. This may or may not be a vector.
function write_line!(io,format,M,v::Symbol,i::Int=0,vname::String="",endline=true)
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            write(io,format(getfield(M[s],v)),"&")
        else
            write(io,format(getfield(M[s],v)[i]),"&")
        end
    end
    if endline
        write(io,"\\\\","\n")
    end
end

# this function does the same as above but includes p-values
function write_pars!(io,format,M,Pv,v::Symbol,i::Int=0,vname::String="",endline = true)
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            pval = getfield(Pv[s],v)
            pstr = pval_ind(pval)
            write(io,format(getfield(M[s],v)),pstr,"&")
        else
            pval = getfield(M[s],v)[i]
            pstr = pval_ind(pval)
            write(io,format(getfield(M[s],v)[i]),pstr,"&")
        end
    end
    if endline
        write(io,"\\\\","\n")
    end
end

function pval_ind(p,char = "*")
    if p<0.001
        return string("^{",repeat(char,3),"}")
    elseif p<0.01
        return string("^{",repeat(char,2),"}")
    elseif p<0.05
        return string("^{",char,"}")
    else
        return ""
    end
end

function format_pval(form,x,p)
    return string("\$",form(x),pval_ind(p),"\$")
end


function write_observables!(io,format,formatse,M,SE,specs,labels,var::Symbol,specvar::Symbol)
    nspec = length(M)
    vlist = union([s[specvar] for s in specs]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
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
                write(io,format(getfield(M[j],var)[i]),"&")
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
                write(io,formatse(getfield(SE[j],var)[i]),"&")
            end
        end
        write(io,"\\\\")

    end
end

function write_production_table(M,SE,Pp,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\rho\$} & \\multicolumn{$nspec}{c}{\$\\gamma\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")
    
    # -- now write the estimates:
    
    write(io,[string("&",format_pval(form,M[s].ρ,Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",format_pval(form,M[s].γ,Pp[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(SE[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",4*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\phi_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{Y}\$: Childcare} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βg,:βθ]
        svarlist = [:vm,:vf,:vg,:vθ]#<- I'm an idiot for calling these different things
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    xval = getfield(M[j],var)[i]
                    if var==:βθ
                        write(io,"&",form(xval))
                    else                        
                        pval = getfield(Pp[j],var)[i]
                        write(io,"&",format_pval(form,xval,pval))
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",formse(getfield(SE[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end




function writetable(M,SE,specs,labels,pvals,outfile::String,production = false)
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
    if production
        for j in 1:2
            write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
            write_line!(io,formse,SE,:δ,j)
        end
        write_line!(io,form,M,:λ,0,"\$\\lambda_{AP}\$")
        write_line!(io,formse,SE,:λ,0) 
    end
    # δ_1 and δ_2 #are there delta parameters here?
    #for j in 1:2
    #    write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
    #    write_line!(io,formse,SE,:δ,j)
    #end

    # Factor share Parameters:
    # a_{m}
    # write the header:
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{m}\$: Mother's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βm,:vm)
    # a_{f}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{f}\$: Father's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βf,:vf)
    # a_{g}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{Y}\$: Childcare}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βg,:vg)

    # test results
    write(io,"& \\multicolumn{$(nspec+1)}{c}{Residual Correlation Test}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write(io,"p-value","&")
    for s in 1:nspec
        write(io,form(pvals[s]),"&")
    end
    write(io,"\\\\","\n")


    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end

function write_production_table_unrestricted(P1,P2,Pu,SE1,SE2,spec,labels,test_stat,p_val,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*2)-$(1+s*2)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{lccccccc}","\\toprule","\n")
    #write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{2}{c}{\$\\rho\$} & \\multicolumn{2}{c}{\$\\gamma\$} & {\$\\delta_{1}\$} & {\$\\delta_{2}\$} & \$2N(Q_{N} - \\tilde{Q}_{N})\$ ","\\\\\n")
    write(io," & (R) & (U) & (R) & (U) & - & - & - \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-6}","\\cmidrule(r){7-7}","\\cmidrule(r){8-8}","\n")
    
    # -- now write the estimates:
    
    write(io,"&",form(P1.ρ))
    if Pu.ρ
        write(io,"&",form(P2.ρ))
    else
        write(io,"& - ")
    end
    write(io,"&",form(P1.γ))
    if Pu.γ
        write(io,"&",form(P2.γ))
    else
        write(io,"& - ")
    end
    write(io,"&",form(P2.δ[1]),"&",form(P2.δ[2]),"&",form(test_stat),"\\\\\n")

    # ----- standard errors:
    write(io,"&",formse(SE1.ρ))
    if Pu.ρ
        write(io,"&",formse(SE2.ρ))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE1.γ))
    if Pu.γ
        write(io,"&",formse(SE2.γ))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE2.δ[1]),"&",formse(SE2.δ[2]), "&", formse(p_val),"\\\\\n")
    
    write(io,"\\\\\n")
    write(io,repeat("&",7),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{2}{c}{\$\\phi_{m}\$: Mother's Time} & \\multicolumn{2}{c}{\$\\phi_{f}\$: Father's Time} & \\multicolumn{2}{c}{\$\\phi_{Y}\$: Childcare} &{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io," & (R) & (U) & (R) & (U) & (R) & (U) & -  \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-7}","\\cmidrule(r){8-8}","\n")

    vlist = union([spec[specvar] for specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βg,:βθ]
        svarlist = [:vm,:vf,:vg,:vθ]#<- I'm an idiot for calling these different things
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            i = findfirst(spec[specvar].==v)
            if isnothing(i)
                if var==:βθ
                    write(io,"&","-")
                else
                    write(io,"& - & -")
                end
            else
                if var==:βθ
                    xval = getfield(P2,var)[i]
                    write(io,"&",form(xval))
                else
                    xval = getfield(P1,var)[i]
                    write(io,"&",form(xval))
                    if getfield(Pu,var)[i]
                        xval = getfield(P2,var)[i]
                        write(io,"&",form(xval))
                    else
                        write(io,"& -") 
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            i = findfirst(spec[specvar].==v)
            if isnothing(i)
                if var==:βθ
                    write(io,"&") #,"-")
                else
                    write(io," & &")
                end
            else
                if var==:βθ
                    xval = getfield(SE2,var)[i]
                    write(io,"&",formse(xval))
                else
                    xval = getfield(SE1,var)[i]
                    write(io,"&",formse(xval))
                    if getfield(Pu,var)[i]
                        xval = getfield(SE2,var)[i]
                        write(io,"&",formse(xval))
                    else
                        write(io,"&") # -") 
                    end
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end

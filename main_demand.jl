using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim

#D = DataFrame(CSV.File("../../../PSID_CDS/data-derived/gmm_data.csv",missingstring = "NA"))
D = DataFrame(CSV.File("../../../PSID_CDS/data-derived/ChildPanelCDS.csv",missingstring = "NA"))
D = subset(groupby(D,:KID),[:year,:age] => (x,y)->fill(sum((x.==1997) .& (y.<=8))>0,length(x)))


# ------ Data prep: all of this we should be doing in other code, I think
D[!,:logwage_m] = log.(D.m_wage)
D[!,:age_sq] = D.age_mother.^2
function fill_missing(wage)
    if wage<=0
        return missing
    else
        return wage
    end
end
transform!(D,:f_wage => ByRow(passmissing(fill_missing)) => :f_wage)
D[!,:logwage_f] = log.(D.f_wage)
D[!,:logprice_g] = log.(D.price_g)
D[!,:logprice_c] = log.(D.p_4c)
D[!,:log_chcare] = replace(log.(D.chcare),-Inf => missing)
D[!,:log_mtime] = replace(log.(D.tau_m),-Inf => missing)
D[!,:log_ftime] = replace(log.(D.tau_f),-Inf => missing)
D = D[.!ismissing.(D.mar_stat),:]
D = D[.!ismissing.(D.m_wage),:]
D = D[.!(D.mar_stat .& ismissing.(D.f_wage)),:]
D = D[.!ismissing.(D.m_ed),:]
D = D[.!(D.mar_stat .& ismissing.(D.f_ed)),:]
D[!,:goods] = D.Toys .+ D.tuition .+ D.comm_grps .+ D.lessons .+ D.tutoring .+ D.sports .+ D.SchSupplies
D[!,:log_good] = replace(log.(D.goods),-Inf => missing)


# -------------
function make_dummy(data,var::Symbol)
    vals = unique(skipmissing(data[!,var]))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end


m_ed = make_dummy(D,:m_ed)
f_ed = make_dummy(D,:f_ed)
D[.!D.mar_stat,f_ed] .= 0. #<- make zero by default. this won't work all the time

# ----------


@with_kw struct CESmod
    # elasticity parameters
    ρ = -1.5 #
    γ = -3. 
    δ = [0.05,0.95]
    # coefficient vectors for factor shares
    βm = zeros(2)
    βf = zeros(2)
    βg = zeros(2)
    βθ = zeros(2)
    spec = (vm = [:const,:mar_stat],vf = [:const],vθ = [:const,:mar_stat],vg = [:const,:mar_stat])
end

# an exmaple of specification
spec = (vm = [:mar_stat;:age;:num_0_5;m_ed],vf = [:age;:num_0_5;f_ed],vθ = [:const,:mar_stat,:age,:num_0_5],vg = [:mar_stat;:age;:num_0_5;m_ed;f_ed[2:end]]) 

function CESmod(spec)
    return CESmod(βm = zeros(length(spec.vm)),βf = zeros(length(spec.vf)),βg = zeros(length(spec.vg)),βθ = zeros(length(spec.vθ)),spec=spec)
end
P = CESmod(spec)



# an example of an update function. This is should be written in a customized way for each specification
# TODO: write more sophisticated version that works for any list of Symbols
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    nm = length(spec.vm)
    βm = x[3:2+nm]
    pos = 3+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    return CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
end
function update_inv(pars)
    @unpack ρ,γ,βm,βf,βg = pars
    return [ρ;γ;βm;βf;βg]
end
x0 = update_inv(P)


# a simple utility function that returns a linear combination of β and each variable in the list vars
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


function log_input_ratios(ρ,γ,ay,am,af,ag,logwage_m,logwage_f,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕf = 1/(ρ-1)*log(ag/af) + 1/(ρ-1)*(logwage_f - logprice_g)
    lϕc = 1/(γ-1)*log(ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag)^(γ/ρ) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm) + exp(logwage_f)*exp(lϕf))
    return lϕm,lϕf,lϕc,log_price_index,Φg
end

# same function as above but for singles (no wage of father to pass)
function log_input_ratios(ρ,γ,ay,am,ag,logwage_m,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕc = 1/(γ-1)*log(ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + ag)^(γ/ρ) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm))
    return lϕm,lϕc,log_price_index,Φg
end

# QUESTION: how do we want to do this?
# splitting the dataframe may slow us down a lot
function log_input_ratios(pars,data,it)
    @unpack ρ,γ = pars
    if data.mar_stat[it]
        ag,am,af = factor_shares(pars,data,it,true) #<- returns the factor shares.
        lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,1.,am,af,ag,data.logwage_m[it],data.logwage_f[it],data.logprice_g[it],data.logprice_c[it])
        return lϕm,lϕf,lϕc,log_price_index,Φg
    else
        ag,am = factor_shares(pars,data,it,false)
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,1.,am,ag,data.logwage_m[it],data.logprice_g[it],data.logprice_c[it])
        lϕf = 0.
        return lϕm,lϕf,lϕc,log_price_index,Φg
    end
end

function factor_shares(pars,data,it,mar_stat)
    @unpack βm,βf,βg,spec = pars
    if mar_stat
        am = linear_combination(βm,spec.vm,data,it)
        af = linear_combination(βf,spec.vf,data,it)
        ag = linear_combination(βg,spec.vg,data,it)
        return exp(ag),exp(am),exp(af)
    else
        am = linear_combination(βm,spec.vm,data,it)
        ag = linear_combination(βg,spec.vg,data,it)
        return exp(am),exp(ag)
    end 
end


#function:
# suppose 97 and 02 are uncorrelated
#
# use different ratios (right now must code a different function)
# recall: 97: use c/m,f/m 
# recall: 02: use c/m,f/m,c/g,m/g,f/g
function calc_demand_resids!(it,R97,R02,data,pars)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<- does this factor in missing data?
    if data.year[it]==1997
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R97[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R97[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
    # recall: 02: use c/m,f/m,c/g,m/g,f/g
    elseif data.year[it]==2002
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R02[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R02[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
        if !ismissing(data.log_good[it])
            if !ismissing(data.log_chcare[it])
                R02[3] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
            end
            if !ismissing(data.log_mtime[it]) #& !ismissing(data.logwage_m[it]) #<- include for missing wage?
                R02[4] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
            end
            if !ismissing(data.log_ftime[it]) #& !ismissing(data.logwage_f[it]) #<- include for missing wage?
                R02[5] = data.log_ftime[it] - data.log_good[it] - lϕf + data.logprice_g[it]
            end
        end
    end
end

function calc_demand_resids!(R97,R02,data,pars)
    gd = groupby(D,:KID)
    for i in 1:gd.ngroups
        #println(i)
        R97[i,:] .= 0.
        R02[i,:] .= 0.
        for it in gd.starts[i]:gd.ends[i]
            @views calc_demand_resids!(it,R97[i,:],R02[i,:],data,pars)
        end
    end
end



# R97 = zeros(gd.ngroups,2)
# R02 = zeros(gd.ngroups,5)


# TODO: 
function weighted_nlls(P,W97,W02,data)
    ssq = 0
    r97 = zeros(typeof(P.βm[1]),2)
    r02 = zeros(typeof(P.βm[2]),5)
    gd = groupby(data,:KID)
    for i in 1:gd.ngroups
        #println(i)
        r97[:] .= 0.
        r02[:] .= 0.
        for it in gd.starts[i]:gd.ends[i]
            calc_demand_resids!(it,r97,r02,data,P)
        end
        ssq += r97'*W97*r97 + r02'*W02*r02
    end
    return ssq/gd.ngroups
end

function weighted_nlls(gd,i,P,W97,W02,data)
    r97 = zeros(typeof(P.ρ),2)
    r02 = zeros(typeof(P.ρ),5)
    for it in gd.starts[i]:gd.ends[i]
        calc_demand_resids!(it,r97,r02,data,P)
    end
    return r97'*W97*r97 + r02'*W02*r02
end

# This is really slow. 1.29 M allocations is way too many. where from?
@time weighted_nlls(P,I(2),I(5),D)

using Optim

res = optimize(x->weighted_nlls(update(x,spec),I(2),I(5),D),x0,LBFGS(),autodiff=:forward)
Pest = update(res.minimizer,spec)


# try estimating with f/m removed from regressions
W97 = [1 0;0 0]
W02 = I(5); W02[2,2] = 0;
res2 = optimize(x->weighted_nlls(update(x,spec),W97,W02,D),x0,LBFGS(),autodiff=:forward)
P2est = update(res2.minimizer,spec)

# TODO: load old data and estimate to compare results
# -- are differneces coming from parameter estimates or not?
# write or import NLLS module for estimating and computing standard errors


# gd = groupby(D,:KID)
# R97 = zeros(gd.ngroups,2)
# R02 = zeros(gd.ngroups,5)
# calc_demand_resids!(R97,R02,D,P2est)
# v97 = cov(R97)
# v02 = cov(R02)
# w97 = [1/v97[1,1] 0; 0 0]
# w02 = inv(v02); w02[2,:] .= 0; w02[:,2] .= 0;
# res3 = optimize(x->weighted_nlls(update(x,spec),w97,w02,D),x0,LBFGS(),autodiff=:forward)
# P3est = update(res3.minimizer,spec)


# ---- Experiment: run the estimation routine now on the old dataset
# Step 1: create the data object
D2 = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
#d = DataFrame(CSV.File("data/gmm_full_horizontal.csv",missingstring = "NA"))
D2[!,:mar_stat] = D2.mar_stable
D2[!,:logwage_m] = log.(D2.m_wage)
#D2[!,:age_sq] = D2.age_mother.^2
D2[!,:logwage_f] = log.(D2.f_wage)
D2[!,:logprice_g] = log.(D2.price_g)
D2[!,:logprice_c] = log.(D2.p_4f) # alternative is p_4c
#D2[!,:log_chcare] = replace(log.(D2.chcare),-Inf => missing)
D2[!,:log_mtime] = D2.log_mtime .- D2.logwage_m
D2[!,:log_ftime] = D2.log_ftime .- D2.logwage_f
#D2 = D2[.!ismissing.(D2.mar_stat),:]
D2 = D2[.!ismissing.(D2.price_g),:]
D2 = D2[.!ismissing.(D2.m_wage),:]
D2 = D2[.!(D2.mar_stat .& ismissing.(D2.f_wage)),:]
# D2[!,:goods] = D2.Toys .+ D2.tuition .+ D2.comm_grps .+ D2.lessons .+ D2.tutoring .+ D2.sports .+ D2.SchSupplies
# D2[!,:log_good] = replace(log.(D2.goods),-Inf => missing)
m_ed = make_dummy(D2,:m_ed)
f_ed = make_dummy(D2,:f_ed)
D2[.!D2.mar_stat,f_ed] .= 0. #<- make zero by default. this won't work all the time


@time weighted_nlls(update(x0,spec),W97,W02,D2)
res4 = optimize(x->weighted_nlls(update(x,spec),W97,W02,D2),x0,LBFGS(),autodiff=:forward)
P4est = update(res4.minimizer,spec)

using ForwardDiff
# -------
# this code works for calculating standard errors. It's slow though
H = ForwardDiff.hessian(x->weighted_nlls(update(x,spec),W97,W02,D2),res4.minimizer)
gd = groupby(D2,:KID)
S = zeros(gd.ngroups,length(x0))
for i=1:gd.ngroups
    S[i,:] = ForwardDiff.gradient(x->weighted_nlls(gd,i,update(x,spec),W97,W02,D2),res4.minimizer)
end
Vx = inv(H)*cov(S)*inv(H) / gd.ngroups
# ---------
[res4.minimizer sqrt.(diag(Vx))]

res5 = optimize(x->weighted_nlls(update(x,spec),I(2),I(5),D2),x0,LBFGS(),autodiff=:forward)
P5est = update(res5.minimizer,spec)


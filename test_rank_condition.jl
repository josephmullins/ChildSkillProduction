using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")
include("production.jl")
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
# temporary:
#panel_data.mid[ismissing.(panel_data.mid)] .= 6024032
#panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


panel_data = panel_data[panel_data.year.<=2002,:] #<- for now, limit to years <=2002. We need to update code eventually.
include("temp_prep_data.jl")

#---- write the update function:
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    δ = x[3:4] #<- factor shares
    nm = length(spec.vm)
    βm = x[5:4+nm]
    pos = 5+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    pos += ng
    nθ = length(spec.vθ)
    βθ = x[pos:pos+nθ-1]
    pos+= nθ
    λ = x[pos]
    P = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
    return P
end
function update_inv(pars)
    @unpack ρ,γ,δ,βm,βf,βg,βθ,λ = pars
    return [ρ;γ;δ;βm;βf;βg;βθ;λ]
end

include("specifications.jl")

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    x0[3:4] = [0.1,0.9] #<- initial guess for δ
    return x0
end

N = length(unique(panel_data.kid))
nmom = spec_1p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_1p)

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec)


P = update(x0,spec_1p)
np_demand = 2+length(spec_1.vm)+length(spec_1.vf)+length(spec_1.vg)

unrestricted = fill(true,np_demand)
unrestricted[1:12] .= true

xcount = collect(1:51)
P1_idx,P2_idx = update(xcount,spec_1p,unrestricted)


Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

# test:
P1,P2 = update(x1,spec_1p,unrestricted)

gfunc2!(x,n,g,resids,data,spec) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec)

dG = ForwardDiff.jacobian(x->moment_func(x,gfunc2!,N,nmom,5,panel_data,spec_1p),x1)

nmom = spec_1p_x.g_idx_prod[end][end]
W = I(nmom)
dG = ForwardDiff.jacobian(x->moment_func(x,gfunc2!,N,nmom,5,panel_data,spec_1p_x),x1)

Ω = moment_variance(x1,gfunc2!,N,nmom,5,panel_data,spec_1p_x)


# first let's estimate the restricted version:

x0 = initial_guess(spec_1p_x)

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec)

res1 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_1p_x)

np_demand = 2+length(spec_1.vm)+length(spec_1.vf)+length(spec_1.vg)

P = update(res1.est1,spec_1p_x)

W = inv(res1.Ω)

# Ω = moment_variance(res1.est1,gfunc!,N,nmom,5,panel_data,spec_1p_x)
# W = inv(Ω)

# try a version where everything is calculated up-front


gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec)

# test restrictions jointly:
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)
LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

# test restrictions invidually
# THE ISSUE here is that the restricted version does not have zero derivatives (due to updating of the weighting matrix. we would have to return the original weighting matrix to get a stable comparison)

tvec = zeros(np_demand)
pvec = zeros(np_demand)
for p in 1:np_demand
    unrestricted = fill(false,np_demand)
    unrestricted[p] = true
    Pu = update_demand(unrestricted,spec_1)
    x1 = update_inv(P,P,Pu)
    tvec[p],pvec[p] = LM_test(x1,1,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)
end

LM_test(res1.est1,1,gfunc!,inv(res1.Ω),N,5,panel_data,spec_1p_x)

# attempt to estimate the unrestricted version:
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),x1,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),res1u.minimizer,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),res1u.minimizer,Newton(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))
unrestricted),res1u.minimizer,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))


g0 = gmm_criterion(x1,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)
g1 = res1u.minimum

# ---- experiment 1: update the elasticities only
unrestricted = fill(false,np_demand)
unrestricted[1:2] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

# this doesn't work.
res1u = estimate_gmm(x1,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)


# ---- experiment 2: update the intercept terms in the factor shares:
P_idx = update_demand(collect(1:np_demand),spec_1p_x)
unrestricted = fill(false,np_demand)
unrestricted[[P_idx.βm[1:2];P_idx.βf[1]]] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

P1,P2 = update(x1,spec_1p_x,unrestricted)

LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

nresids = 5
args = (panel_data,spec_1p_x,unrestricted)
# we hit a failure here
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,nresids,args...),x1,LBFGS(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))

# but it looks like there are some changes in the parameters
P1_idx,P2_idx = update(collect(1:length(x1)),spec_1p_x,unrestricted)
i_r = [P1_idx.βm[1:2];P2_idx.βm[1:2];P1_idx.βf[1];P2_idx.βf[1]]
#res1u = estimate_gmm(x1,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

# experiment 3: relax the same parameters, but fix the others
# - does minimization perform better in this case? 
# - the answer is no
function gfunc3!(x,n,g,resids,data,spec,unrestricted,idx_fill,x0)
    x0[idx_fill] .= x
    production_demand_moments_stacked!(update(x0,spec,unrestricted)...,n,g,resids,data,spec)
end

x0 = zeros(Real,length(x1))
x0[:] .= x1
x1 = x1[i_r]

res1u = optimize(x->gmm_criterion(x,gfunc3!,W,N,nresids,args...,i_r,x0),x1,Newton(),autodiff=:forward,Optim.Options(show_trace=true,f_calls_limit=100))

g1 = gmm_criterion(x1,gfunc3!,W,N,nresids,args...,i_r,x0)
g2 = gmm_criterion(res1u.minimizer,gfunc3!,W,N,nresids,args...,i_r,x0)

dq = ForwardDiff.gradient(x->gmm_criterion(x,gfunc3!,W,N,nresids,args...,i_r,x0),x1)


dQ = ForwardDiff.gradient(x->gmm_criterion(x,gfunc2!,W,N,nresids,panel_data,spec_1p_x,unrestricted),x1)


dG = ForwardDiff.jacobian(x->moment_func(x,gfunc2!,N,nmom,5,panel_data,spec_1p_x),x1)
gn = moment_func(x1,gfunc2!,N,nmom,5,panel_data,spec_1p_x)
Binv = inv(dG'*W*dG)

# test all restrictions:
pre = dG'*W*gn
pre'*Binv*pre * N
using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim, JLD2
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

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec)

# - load the estimates from spec 3
r = JLD2.load("results/production_restricted.jld2")
res3 = r["res3"]
W = inv(res3.Ω)
np_demand = 2+length(spec_3p_x.vm)+length(spec_3p_x.vf)+length(spec_3p_x.vg)
P_idx = update_demand(collect(1:np_demand),spec_3p_x)

# --- try relaxing just elasticities
unrestricted = fill(false,np_demand)
unrestricted[1:2] .= true
P = update(res3.est1,spec_3p_x)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv(P,P,Pu)

t3,p3 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)

res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
p1,p2 = update(res3u.minimizer,spec_3p_x,unrestricted)
v = parameter_variance_gmm(res3u.minimizer,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)
SE1,SE2 = update(sqrt.(diag(v)),spec_3p_x,unrestricted)

# -- now relax the intercept term in factor shares:

P_idx = update_demand(collect(1:np_demand),spec_3p_x)
unrestricted = fill(false,np_demand)
unrestricted[[P_idx.βm[1];P_idx.βf[1];P_idx.βg[1]]] .= true
P = update(res3.est1,spec_3p_x)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv(P,P,Pu)

t3,p3 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)

res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
p1,p2 = update(res3u.minimizer,spec_3p_x,unrestricted)
v = parameter_variance_gmm(res3u.minimizer,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)
SE1,SE2 = update(sqrt.(diag(v)),spec_3p_x,unrestricted)

# TODO: write an update function that only updates the relaxed parameters and not the others
using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("model.jl")
include("moment_functions.jl")
include("testing_tools.jl")

# --------  read in the data:
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
include("prep_data.jl")
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
#panel_data = DataFrame(filter(x->sum(ismissing.(x.age))==0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


# - load the specifications that we want to use. See that script for more details.
include("specifications.jl")

N = length(unique(panel_data.kid))
# define the moment functions
gfunc!(x,n,g,resids,data,spec,case) = production_demand_moments_strict!(update(x,spec,case),n,g,resids,data)
gfunc2!(x,n,g,resids,data,spec,unrestricted,case) = production_demand_moments_relaxed!(update_relaxed(x,spec,unrestricted,case)...,n,g,resids,data)

# ------  Unconstrained case

case = "uc"


# ---- specification (3)
data = child_data(panel_data,spec_3p_x)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec_3p_x,case)
res3 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_3p_x,case)

# ------ Re-estimate with mother's factor share relaxed

np_demand = 2 + length(spec_3.vm) + length(spec_3.vf) + length(spec_3.vy)
P_index = update_demand(1:np_demand,spec_3)
unrestricted = fill(false,np_demand)
unrestricted[P_index.βm[1]] = true
P = update(res3.est1,spec_3p_x,case)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv_relaxed(P,P,Pu,case)

W = inv(res3.Ω)
data = child_data(panel_data,spec_3p_x)
res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
g0 = gmm_criterion(x1,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case)

g1 = res3u.minimum
DM = 2N*max(g0-g1,0.)
p_val = 1 - cdf(Chisq(sum(unrestricted)),DM)
p1,p2 = update_relaxed(res3u.minimizer,spec_3p_x,unrestricted,case)
v = parameter_variance_gmm(res3u.minimizer,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case)
SE1,SE2 = update_relaxed(sqrt.(diag(v)),spec_3p_x,unrestricted,case)

include("table_tools.jl")
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :constant => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

write_production_table_unrestricted(p1,p2,Pu,SE1,SE2,spec_3p_x,labels,DM,p_val,"tables/demand_production_mothershare_relaxed.tex")


# ---- No borrowing or saving case

case = "nbs"

res3 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_3p_x,case)

# ------ Re-estimate with mother's factor share relaxed

P = update(res3.est1,spec_3p_x,case)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv_relaxed(P,P,Pu,case)

W = inv(res3.Ω)
data = child_data(panel_data,spec_3p_x)
res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
g0 = gmm_criterion(x1,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case)

g1 = res3u.minimum
DM = 2N*max(g0-g1,0.)
p_val = 1 - cdf(Chisq(sum(unrestricted)),DM)
p1,p2 = update_relaxed(res3u.minimizer,spec_3p_x,unrestricted,case)
v = parameter_variance_gmm(res3u.minimizer,gfunc2!,W,N,length(data.Z),data,spec_3p_x,unrestricted,case)
SE1,SE2 = update_relaxed(sqrt.(diag(v)),spec_3p_x,unrestricted,case)


write_production_table_unrestricted(p1,p2,Pu,SE1,SE2,spec_3p_x,labels,DM,p_val,"tables/demand_production_mothershare_relaxed_nbs.tex")

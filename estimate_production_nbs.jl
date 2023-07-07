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
gfunc!(x,n,g,resids,data,spec) = production_demand_moments_strict!(update(x,spec),n,g,resids,data,false)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_relaxed!(update(x,spec,unrestricted)...,n,g,resids,data,false)

# ---- Part 1: estimate the restricted estimator and conduct tests of the equality constraints using the LM statistic
# ---- specification (1)
data = child_data(panel_data,spec_1p_x)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec_1p_x)
res1 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_1p_x)

# - test restrictions (inidividual and joint)
W = inv(res1.Ω)
t1,p1 = test_joint_restrictions(res1.est1,W,N,spec_1p_x,data)
tvec1,pvec1 = test_individual_restrictions(res1.est1,W,N,spec_1p_x,data)

# ---- specification (2)
data = child_data(panel_data,spec_2p_x)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec_2p_x)
res2 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_2p_x)

# - test restrictions (individual and joint)
W = inv(res2.Ω)
t2,p2 = test_joint_restrictions(res2.est1,W,N,spec_2p_x,data)
tvec2,pvec2 = test_individual_restrictions(res2.est1,W,N,spec_2p_x,data)


# ---- specification (3)
data = child_data(panel_data,spec_3p_x)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec_3p_x)
res3 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_3p_x)

# - test restrictions (inidividual and joint)
W = inv(res3.Ω)
t3,p3 = test_joint_restrictions(res3.est1,W,N,spec_3p_x,data)
tvec3,pvec3 = test_individual_restrictions(res3.est1,W,N,spec_3p_x,data)

# ---- specification 5
data = child_data(panel_data,spec_5p_x)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec_5p_x)
res5 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec_5p_x)

# - test restrictions (inidividual and joint)
W = inv(res5.Ω)
t5,p5 = test_joint_restrictions(res5.est1,W,N,spec_5p_x,data)
tvec5,pvec5 = test_individual_restrictions(res5.est1,W,N,spec_5p_x,data)


# Write results to a table
include("table_tools.jl")
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :constant => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1.est1,spec_1p_x),update(res2.est1,spec_2p_x),update(res3.est1,spec_3p_x),update(res5.est1,spec_5p_x)]
se_vec = [update(res1.se,spec_1p_x),update(res2.se,spec_2p_x),update(res3.se,spec_3p_x),update(res5.se,spec_5p_x)]
pval_vec = [update_demand(pvec1,spec_1),update_demand(pvec2,spec_2),update_demand(pvec3,spec_3p_x),update_demand(pvec5,spec_5p_x)]
write_production_table(par_vec,se_vec,pval_vec,[spec_1p_x,spec_2p_x,spec_3p_x,spec_5p_x],labels,"tables/demand_production_restricted_nbs.tex"
)


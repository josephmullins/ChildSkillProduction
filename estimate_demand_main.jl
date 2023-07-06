using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("model.jl")
include("moment_functions.jl")

# --------  read in the data:
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
include("prep_data.jl")
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("wage_types.csv"))

# - load the specifications that we want to use. See that script for more details.
include("specifications.jl")


gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update_demand(x,spec),n,g,resids,data)

N = length(unique(panel_data.kid))
x0 = demand_guess(spec_1)
data = child_data(panel_data,spec_1)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)

res1 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_1)

# Specification (2): 
x0 = demand_guess(spec_2)
data = child_data(panel_data,spec_2)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)

res2 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_2)

# Specification (3): 
x0 = demand_guess(spec_3)
data = child_data(panel_data,spec_3)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)

res3 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_3)

# Specification (4): 
x0 = demand_guess(spec_4)
data = child_data(panel_data,spec_4)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)

res4 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_4)

# Specification (5): 
x0 = demand_guess(spec_5)
data = child_data(panel_data,spec_5)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)

res5 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_5)


# ----- Write results to a LaTeX table

cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1.est1,spec_1),update(res2.est1,spec_2),update(res3.est1,spec_3),update(res4.est1,spec_4),update(res5.est1,spec_5)]
#par_vec = [update(res2,spec_1),update(res3,spec_2),update(res4,spec_3),update(res5,spec_4),update(res6,spec_5)]
se_vec = [update(res1.se,spec_1),update(res2.se,spec_2),update(res3.se,spec_3),update(res4.se,spec_4),update(res5.se,spec_5)]
results = [residual_test(panel_data,N,p) for p in par_vec]
pvals = [r[2] for r in results]

writetable(par_vec,se_vec,[spec_1,spec_2,spec_3,spec_4,spec_5],labels,pvals,"tables/relative_demand.tex")

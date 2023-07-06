using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics, Distributions
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("model.jl")
include("moment_functions.jl")

# --------  read in the data:
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
include("prep_data.jl")
#panel_data = DataFrame(filter(x->sum(ismissing.(x.age))==0,groupby(panel_data,:kid)))
#panel_data = DataFrame(filter(x->sum(x.ind_not_sample.==0)>0,groupby(panel_data,:kid)))
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))

wage_types = DataFrame(CSV.File("wage_types.csv"))
#wage_types2 = DataFrame(CSV.File("wage_types_old.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types


cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

include("specifications.jl")

gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update_demand(x,spec),n,g,resids,data)

gmm_criterion(x0,gfunc!,W,N,9,data,spec_1)

N = length(unique(panel_data.kid))
x0 = demand_guess(spec_1)
data = child_data(panel_data,spec_1)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)


res1 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_1)

using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics, Distributions
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("model.jl")
include("moment_functions.jl")

# --------  read in the data:
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))

# temporary: we need to fix this!!
#panel_data.mid[ismissing.(panel_data.mid)] .= 6024032

panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID, matchmissing = :notequal) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

include("prep_data.jl")
include("specifications.jl")


gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update_demand(x,spec),n,g,resids,data)

N = length(unique(panel_data.kid))
x0 = demand_guess(spec_1)
data = child_data(panel_data,spec_1)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)


res1 = estimate_gmm(x0,gfunc!,W,N,9,data,spec_1)

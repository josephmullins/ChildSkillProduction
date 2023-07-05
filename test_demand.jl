using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("relative_demand.jl")
include("other_functions.jl")

#TODO:
# update specifications for demand to "include" instruments

# --------  read in the data:
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))

# temporary: we need to fix this!!
#panel_data.mid[ismissing.(panel_data.mid)] .= 6024032

panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID, matchmissing = :notequal) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

include("temp_prep_data.jl")
include("specifications_alt.jl")


gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update(x,spec),n,g,resids,data)

N = length(unique(panel_data.kid))

W = I(nmom)
x0 = initial_guess(spec_1)
cd = child_data(panel_data,spec_1)
nmom = sum([size(z,1)*!isempty(z) for z in cd.Z])

g = moment_func(x0,gfunc!,N,nmom,9,cd,spec_1)
Ω = moment_variance(x0,gfunc!,N,nmom,9,cd,spec_1)
R = zeros(9)
gfunc!(x0,1,g,R,cd,spec_1)

res1 = estimate_gmm(x0,gfunc!,W,N,9,cd,spec_1)
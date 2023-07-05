using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("gmm_data.jl")
include("other_functions.jl")
include("production_alt.jl")

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


gfunc!(x,n,g,resids,data,spec) = production_demand_moments_strict!(update(x,spec),n,g,resids,data,true)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_relaxed!(update(x,spec,unrestricted)...,n,g,resids,data,true)
N = length(unique(panel_data.kid))

x0 = initial_guess(spec_1p_x)
cd = child_data(panel_data,spec_1p_x)
nmom = sum([size(z,1)*!isempty(z) for z in cd.Z])
W = I(nmom)

#not sure what's happening here.
R = zeros(length(cd.Z))
g = zeros(nmom)
p = CESmod(spec_1p_x)
#gfunc!(x0,1,g,R,cd,spec_1p_x)

@time production_demand_moments_strict!(p,1,g,R,cd,true)
@time gfunc!(x0,1,g,R,cd,spec_1p_x)

g1 = moment_func(x0,gfunc!,N,nmom,length(cd.Z),cd,spec_1p_x)

res1 = estimate_gmm(x0,gfunc!,W,N,length(cd.Z),cd,spec_1p_x)

np_demand = 2+length(spec_1.vm)+length(spec_1.vf)+length(spec_1.vy)
unrestricted = fill(false,np_demand)

@time production_demand_moments_relaxed!(p,p,1,g,R,cd,true)
g2 = moment_func(x0,gfunc2!,N,nmom,length(cd.Z),cd,spec_1p_x,unrestricted)

@time gfunc2!(x0,1,g2,R,cd,spec_1p_x,unrestricted)

res1a = estimate_gmm(x0,gfunc2!,W,N,length(cd.Z),cd,spec_1p_x,unrestricted)

    
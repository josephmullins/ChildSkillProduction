using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("../src/model.jl")
include("../src/estimation.jl")

# --------  read in the data:
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("data/wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

spec1,spec2,spec3,spec4 = get_specifications(m_ed,f_ed,cluster_dummies)

N = length(unique(panel_data.kid))

case = "uc"

# define the moment function
gfunc!(x,n,g,resids,data,spec,case) = production_demand_moments!(update(x,spec,case),n,g,resids,data)
#gfunc!(x,n,g,resids,data,spec,unrestricted,case) = production_demand_moments!(update_relaxed(x,spec,unrestricted,case)...,n,g,resids,data)

data = child_data(panel_data,spec1)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
W = I(nmom)
x0 = initial_guess(spec1,"uc")
unrestricted = fill(false,length(x0))
#res1 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec1,unrestricted,"uc")
res1 = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec1,"uc")

#p1,p2 = update_relaxed(res1.est2,spec1,unrestricted,"uc")

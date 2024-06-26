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

# get the four specifications we settle on in the paper
spec1,spec2,spec3,spec4 = get_specifications(m_ed,f_ed,cluster_dummies)

# define the moment function we use in estimation
gfunc!(x,n,g,resids,data,spec,unrestricted,case) = production_demand_moments!(update_relaxed(x,spec,unrestricted,case)...,n,g,resids,data)

# define the set of labels that convert symbols to strings for our output tables
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))
other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :constant => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")
labels = merge(other_labels,cluster_labels,ed_labels)


# ------- Case 1: unconstrained (κ=0)

res1 = run_restricted_estimation(panel_data,spec1,"uc",gfunc!)
res2 = run_restricted_estimation(panel_data,spec2,"uc",gfunc!)
res3 = run_restricted_estimation(panel_data,spec3,"uc",gfunc!)
res4 = run_restricted_estimation(panel_data,spec4,"uc",gfunc!)

# write these results to a .tex table
# Write results to a table
write_production_table([res1,res2,res3,res4],[spec1,spec2,spec3,spec4],labels,"tables/demand_production_restricted.tex")

# run the unrestricted version for our preferred specification
res3u = run_unrestricted_estimation(panel_data,spec3,"uc",gfunc!,res3)

write_production_table_unrestricted(res3u,spec3,labels,"tables/demand_production_unrestricted.tex")

# --------- Case 2: No borrowing or savings (κ=1)

res1_nbs = run_restricted_estimation(panel_data,spec1,"nbs",gfunc!)
res2_nbs = run_restricted_estimation(panel_data,spec2,"nbs",gfunc!)
res3_nbs = run_restricted_estimation(panel_data,spec3,"nbs",gfunc!)
res4_nbs = run_restricted_estimation(panel_data,spec4,"nbs",gfunc!)

# write these results to a .tex table
# Write results to a table
write_production_table([res1_nbs,res2_nbs,res3_nbs,res4_nbs],[spec1,spec2,spec3,spec4],labels,"tables/demand_production_restricted_nbs.tex")

# run the unrestricted version for our preferred specification
res3u_nbs = run_unrestricted_estimation(panel_data,spec3,"nbs",gfunc!,res3_nbs)
write_production_table_unrestricted(res3u_nbs,spec3,labels,"tables/demand_production_unrestricted_nbs.tex")

# Finally: write a summary table for specification three (our preferred)

# THEN: write the relative demand case in here as well.


# THEN: run the estimation for older children.

# THEN: run the estimation with the relaxed mother share.
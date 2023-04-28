using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")

# --------  read in the data:
# -- Read in wage data from the mother's panel;
wage_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
wage_data[!,:logwage_m] = wage_data.ln_wage_m
wage_data = subset(wage_data,:m_wage => x->x.>0,skipmissing=true)
wage_data[!,:constant] .= 1.
m_ed = make_dummy(wage_data,:m_ed)
vl=[m_ed[2:end];:m_exper;:m_exper2]
num_clusters = 3


wage_types = cluster_routine_robust(wage_data,vl,num_clusters)

wage_types_k10 = cluster_routine_robust(wage_data,vl,10,500)
wage_types_k10 = rename(select(wage_types_k10,[:MID,:center]),:center => :mu_k)

# --- alternatively using the naive clustering algorithm which won't work if we include education in vl
# wage_types = generate_cluster_assignment(wage_data,vl,true,num_clusters)
# wage_types_k10 = generate_cluster_assignment(wage_data,vl,true,10)
# wage_types_k10 = rename(select(wage_types_k10,[:MID,:center]),:center => :mu_k)

wage_types = innerjoin(wage_types,wage_types_k10,on=:MID)

CSV.write("wage_types.csv",wage_types)
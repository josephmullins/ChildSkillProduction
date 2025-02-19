include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")

# =======================   read in the data ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))

# ======= Introduce alternative normalization of test scores ====== #
scores = DataFrame(CSV.File("../../../PSID_CDS/data-cds/assessments/AssessmentPanel.csv",missingstring=["","NA"]))
scores = select(scores,[:KID,:year,:LW_raw,:AP_raw,:AP_std,:LW_std])
scores = rename(scores,:KID => :kid)
panel_data = sort(leftjoin(panel_data,scores,on=[:kid,:year]),[:kid,:year])
mLW = mean(skipmissing(panel_data.LW_raw[panel_data.age.==12]))
sLW = std(skipmissing(panel_data.LW_raw[panel_data.age.==12]))
mAP = mean(skipmissing(panel_data.AP_raw[panel_data.age.==12]))
sAP = std(skipmissing(panel_data.AP_raw[panel_data.age.==12]))

using DataFramesMeta
panel_data = @chain panel_data begin
    # groupby(:age)
    @transform :LW = (:LW_raw .- mLW)/sLW :AP = (:AP_raw .- mAP)/sAP
end

# =======================   run the clustering routine on wages ===================================== #

println(" ======= Running the Clustering Algorithm on Wage Data ========= ")
wage_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
wage_data[!,:logwage_m] = wage_data.ln_wage_m
wage_data = subset(wage_data,:m_wage => x->x.>0,skipmissing=true)
wage_data[!,:constant] .= 1.
m_ed = make_dummy(wage_data,:m_ed)
yr = make_dummy(wage_data,:year)
vl=[yr[2:end];m_ed[2:end];:m_exper;:m_exper2;:mar_stat ; :num_0_5]
wage_data.mar_stat = coalesce.(wage_data.mar_stat,0)
num_clusters = 3

Random.seed!(2724)
wage_types = cluster_routine_robust(wage_data,vl,num_clusters)
wage_types_k10 = cluster_routine_robust(wage_data,vl,10,500)
wage_types_k10 = rename(select(wage_types_k10,[:MID,:center]),:center => :mu_k)

wage_types = innerjoin(wage_types,wage_types_k10,on=:MID)

CSV.write("data/wage_types_alt_clustering.csv",wage_types)

# =======================  Do other basic setup work ===================================== #

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

# get the four specifications we settle on in the paper
spec1,spec2,spec3,spec4 = get_specifications(m_ed,f_ed,cluster_dummies)

# define the moment function we use in estimation
gfunc!(x,n,g,resids,data,spec,unrestricted,case) = production_demand_moments!(update_relaxed(x,spec,unrestricted,case)...,n,g,resids,data)

# define the set of labels that convert symbols to strings for our output tables
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father some coll.","Father coll+","Mother some coll.","Mother coll+"]))
other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. of children 0-5", :constant => "Constant", :mu_k => "\$\\mu_{k}\$", :age => "Child's age", :ind02 => "Year = 2002")
labels = merge(other_labels,cluster_labels,ed_labels)

# =======================  Run the specifications ===================================== #

# ------- Case 1: unconstrained (κ=0)
println(" ====== Estimating Main Specifications for κ = 0 ========= ")
res1 = run_restricted_estimation(panel_data,spec1,"uc",gfunc!)
res2 = run_restricted_estimation(panel_data,spec2,"uc",gfunc!)
res3 = run_restricted_estimation(panel_data,spec3,"uc",gfunc!)
res4 = run_restricted_estimation(panel_data,spec4,"uc",gfunc!)

# Write results to a table
write_production_table([res1,res2,res3,res4],[spec1,spec2,spec3,spec4],labels,"tables/demand_production_restricted_alt_clustering.tex")

# --------- Case 2: No borrowing or savings (κ=1)
println(" ====== Estimating Main Specifications for κ = 0 ========= ")
res1_nbs = run_restricted_estimation(panel_data,spec1,"nbs",gfunc!)
res2_nbs = run_restricted_estimation(panel_data,spec2,"nbs",gfunc!)
res3_nbs = run_restricted_estimation(panel_data,spec3,"nbs",gfunc!)
res4_nbs = run_restricted_estimation(panel_data,spec4,"nbs",gfunc!)

# Write results to a table
write_production_table([res1_nbs,res2_nbs,res3_nbs,res4_nbs],[spec1,spec2,spec3,spec4],labels,"tables/demand_production_restricted_nbs_alt_clustering.tex")


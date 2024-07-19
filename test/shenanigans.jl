include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")

# =======================   read in the data ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("data/wage_types.csv"))

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

break
# =======================  Run the specifications ===================================== #
res1 = run_restricted_estimation(panel_data,spec1,"uc",gfunc!)

# -------- Read in assessment data
scores = DataFrame(CSV.File("../../../PSID_CDS/data-cds/assessments/AssessmentPanel.csv",missingstring=["","NA"]))
scores = select(scores,[:KID,:year,:LW_raw,:AP_raw,:AP_std,:LW_std])
scores = rename(scores,:KID => :kid)
panel_data2 = sort(leftjoin(panel_data,scores,on=[:kid,:year]),[:kid,:year])
mLW = mean(skipmissing(panel_data2.LW_raw[panel_data2.age.==12]))
sLW = std(skipmissing(panel_data2.LW_raw[panel_data2.age.==12]))
mAP = mean(skipmissing(panel_data2.AP_raw[panel_data2.age.==12]))
sAP = std(skipmissing(panel_data2.AP_raw[panel_data2.age.==12]))

break

using DataFramesMeta
panel_data2 = @chain panel_data2 begin
    # groupby(:age)
    @transform :LW = (:LW_raw .- mean(skipmissing(:LW_raw)))/sLW :AP = (:AP_raw .- mean(skipmissing(:AP_raw)))/sAP
    #@transform :LW = (:LW_std .- 100)/15 :AP2 = (:AP_std .- 100)/15
end

res2 = run_restricted_estimation(panel_data2,spec1,"uc",gfunc!)

res3 = run_restricted_estimation(panel_data,spec1,"uc",gfunc!)


# @chain panel_data begin
#     @subset .!ismissing.(:LW)
#     @select :kid :year :LW :LW2
#     #@combine :a = maximum(abs.(:LW .- :LW2))
# end

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
vl=[yr[2:end];m_ed[2:end];:m_exper;:m_exper2]
num_clusters = 3

Random.seed!(2724)
wage_types = cluster_routine_robust(wage_data,vl,num_clusters)

Random.seed!(2724)
wage_data.mar_stat = coalesce.(wage_data.mar_stat,0)
vl=[yr[2:end];m_ed[2:end];:m_exper;:m_exper2;:mar_stat ;:num_child]
wage_types_2 = cluster_routine_robust(wage_data,vl,num_clusters)

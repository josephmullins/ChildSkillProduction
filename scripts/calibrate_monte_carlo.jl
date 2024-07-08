# a script to pick parameters for the monte-carlo simulation

include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")

# =======================   read in the data ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))

wage_types = DataFrame(CSV.File("data/wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

# get the four specifications we settle on in the paper
spec1,spec2,spec3,spec4 = get_specifications(m_ed,f_ed,cluster_dummies)

# read in estimates from spec 3
est = readdlm("output/est_nbs_spec3")[:]

p = update(est,spec3,"nbs")

# things to calculate

# (1) residual variation of child care prices relative to mother's wage
# (2) covariance of demand residuals across time periods
# (3) variance of demand residuals
# (4) variance of residual in outcome equation
# (5) variance of the mother's time input (logged, conditional on Zθ)
N = length(unique(panel_data.kid))
data = child_data(panel_data,spec3)
R = zeros(30,N)
for n in 1:N
    @views demand_residuals_all!(R[1:9,n],p,n,data)
    @views production_residuals_all!(R[10:end,n],p,p,n,data)
end


r_keep = sum(R[1:2,:].!=0,dims=1)[:].>1 #<- keep only non-missing residuals in both years
varξ1 = cov(R[1,r_keep],R[2,r_keep])
varξ2 = var(R[1,r_keep]) - varξ1
varζ = var(R[10,:]) 

I_keep = (panel_data.year.==1997) .& .!ismissing.(panel_data.logprice_c_m) .& .!ismissing.(panel_data.div) .& .!ismissing.(panel_data.log_mtime)
X = Matrix{Float64}(panel_data[I_keep,spec3.vθ[1:end-1]])
Y1 = Vector{Float64}(panel_data.logprice_c_m[I_keep])
Y2 = Vector{Float64}(panel_data.log_mtime[I_keep])

varx1 = var(Y2 .- X*inv(X' * X) * X' * Y2)
varπ = var(Y1 .- X*inv(X' * X) * X' * Y1)

# --- This script loads estimates and runs a regression of skill outcomes on investment proxies.

# - the goal is to highlight and issue with the model and also understand more transparently our results

using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim, JLD2
include("estimation_tools.jl")
include("gmm_data.jl")
include("model.jl")
include("moment_functions.jl")

panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
include("prep_data.jl")
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
#panel_data = DataFrame(filter(x->sum(ismissing.(x.age))==0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

# - load the specifications that we want to use. See that script for more details.
include("specifications.jl")



# get some results

# spec_3p_x = build_spec_prod((vm = spec_3.vm,vf = spec_3.vf, vg = spec_3.vg,vθ = spec_3.vm,
# zlist_prod_t = [0,5],
# zlist_prod = [[[spec_3.vg;interactions_3;:LW],[spec_3.vg;interactions_3;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]]))

# results = JLD2.load("results/production_restricted.jld2")
# res = results["res3"]

# now write a function to make a dataset
function model_test(pars1,pars2,panel_data,kid_data,spec)
    panel_data[!,:Phi_m] .= 0.
    for it in axes(panel_data,1)
        if kid_data.all_prices[it]
            #lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,pars2,kid_data,it)
            lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,kid_data,it)
            panel_data[it,:Phi_m] = - lΦm
        end
    end
    d97 = panel_data[(panel_data.year.==2002) .& (panel_data.all_prices) .& (panel_data.mtime_valid) .& (panel_data.age.<=12),:]
    d02 = panel_data[(panel_data.year.==2007) .& (panel_data.mtime_valid),:]
    rename!(d97,:AP => :AP97,:LW => :LW97,:log_mtime => :log_mtime_97)
    keep97 = [:kid;:AP97;:LW97;:log_mtime_97;:log_chcare_input;:log_ftime;:Phi_m;spec.vθ;:logprice_g;:logprice_c;:logwage_m;:logwage_f;:logprice_m_g;:logprice_c_g;:logprice_f_g]
    keep02 = [:kid;:log_mtime;:AP;:LW]
    return innerjoin(d97[:,keep97],d02[:,keep02],on=:kid)
end

P = update(res3.est1,spec_3p_x)
d = model_test(P,P,panel_data,data,spec_3p_x)

using FixedEffectModels
d[!,:agesq] = d.age.^2

# this regression shows how differently Phi_m and mother's time enter the outcome equation:
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:AP97))
reg(d,term(:LW) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:LW97))

# can switch signs on here by switching LW and AP in the lag
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))
reg(d,term(:LW) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:LW97) ~ term(:log_mtime) + term(:AP97)))


reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))
reg(d,term(:LW) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:LW97) ~ term(:log_mtime) + term(:AP97)))



d[!,:lX97] = d.log_mtime_97 .+ d.Phi_m

# this regression doesn't quite show what we want, but maybe not surprising
reg(d,term(:LW) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + (term(:lX97) + term(:LW97) ~ term(:log_mtime) + term(:AP97)))
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + (term(:lX97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))

# first stages:
reg(d,term(:log_mtime_97) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:LW97))
reg(d,term(:AP97) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:LW97))

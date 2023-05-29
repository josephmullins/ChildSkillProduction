# --- This script loads estimates and runs a regression of skill outcomes on investment proxies.

# - the goal is to highlight and issue with the model and also understand more transparently our results

using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim, JLD2
include("estimation_tools.jl")
include("relative_demand.jl")
include("production.jl")
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
# temporary:
#panel_data.mid[ismissing.(panel_data.mid)] .= 6024032
#panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


panel_data = panel_data[panel_data.year.<=2002,:] #<- for now, limit to years <=2002. We need to update code eventually.
include("temp_prep_data.jl")

#---- write the update function:
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    δ = x[3:4] #<- factor shares
    nm = length(spec.vm)
    βm = x[5:4+nm]
    pos = 5+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    pos += ng
    nθ = length(spec.vθ)
    βθ = x[pos:pos+nθ-1]
    pos+= nθ
    λ = x[pos]
    P = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
    return P
end

include("specifications_alt_demand.jl")

spec_3p_x = build_spec_prod((vm = spec_3.vm,vf = spec_3.vf, vg = spec_3.vg,vθ = spec_3.vm,
zlist_prod_t = [0,5],
zlist_prod = [[[spec_3.vg;interactions_3;:LW],[spec_3.vg;interactions_3;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]]))

results = JLD2.load("results/production_restricted.jld2")
res = results["res3"]

# now write a function to make a dataset
function model_test(pars1,pars2,data,spec)
    d97 = data[(data.year.==1997) .& (data.all_prices) .& (data.mtime_valid) .& (data.age.<=12),:]
    d02 = data[(data.year.==2002) .& (data.mtime_valid),:]
    d97[!,:Phi_m] .= 0.
    for n in 1:size(d97,1)
        # this function call is assumed to return a coefficient lΦm where $X_{it} = τ_{m,it} / exp(lΦm)
        lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,pars2,d97,n)
        #Ψ0 = pars2.δ[1]*pars2.δ[2]^4*lX97
        Ψ0 = 0
        d97[n,:Phi_m] = - lΦm
    end
    rename!(d97,:AP => :AP97,:LW => :LW97,:log_mtime => :log_mtime_97)
    keep97 = [:kid;:AP97;:LW97;:log_mtime_97;:log_chcare_input;:log_ftime;:Phi_m;spec.vθ;:logprice_g;:logprice_c;:logwage_m;:logwage_f;:logprice_m_g;:logprice_c_g;:logprice_f_g]
    keep02 = [:kid;:log_mtime;:AP;:LW]
    return innerjoin(d97[:,keep97],d02[:,keep02],on=:kid)
end

P = update(res.est1,spec_3p_x)
d = model_test(P,P,panel_data,spec_3p_x)

using FixedEffectModels
# this regression shows how differently Phi_m and mother's time enter the outcome equation:
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))

d[!,:lX97] = d.log_mtime_97 .+ d.Phi_m

# this regression doesn't quite show what we want, but maybe not surprising
reg(d,term(:LW) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + (term(:lX97) + term(:LW97) ~ term(:log_mtime) + term(:AP97)))

# first stages:
reg(d,term(:log_mtime_97) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:LW97))
reg(d,term(:AP97) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + term(:log_mtime) + term(:LW97))

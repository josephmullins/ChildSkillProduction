using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
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
function update_inv(pars)
    @unpack ρ,γ,δ,βm,βf,βg,βθ,λ = pars
    return [ρ;γ;δ;βm;βf;βg;βθ;λ]
end

include("specifications_alt_demand.jl")

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    x0[3:4] = [0.1,0.9] #<- initial guess for δ
    return x0
end

function test_individual_restrictions(est,W,N,spec,data)
    np_demand = 2+length(spec.vm)+length(spec.vf)+length(spec.vg)
    tvec = zeros(np_demand)
    pvec = zeros(np_demand)
    for i in 1:np_demand
        unrestricted = fill(false,np_demand)
        unrestricted[i] = true
        P = update(est,spec)
        Pu = update_demand(unrestricted,spec)
        x1 = update_inv(P,P,Pu)
        tvec[i],pvec[i] = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,data,spec,unrestricted)
    end
    return tvec,pvec
end

function test_joint_restrictions(est,W,N,spec,data)
    P = update(est,spec)
    np_demand = 2+length(spec.vm)+length(spec.vf)+length(spec.vg)
    unrestricted = fill(true,np_demand)
    Pu = update_demand(unrestricted,spec)
    x1 = update_inv(P,P,Pu)
    return LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,data,spec,unrestricted)
end

N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec)

# ---- Part 1: estimate the restricted estimator and conduct tests of the equality constraints using the LM statistic

# ---- specification (1)
spec_1p_x = build_spec_prod(
        (vm = spec_1.vm,vf = spec_1.vf, vg = spec_1.vg,vθ = spec_1.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_1;:AP],[interactions_1;:LW],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)
nmom = spec_1p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_1p_x)
res1 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_1p_x)

# -- test restrictions (inidividual and joint)
W = inv(res1.Ω)
t1,p1 = test_joint_restrictions(res1.est1,W,N,spec_1p_x,panel_data)
tvec1,pvec1 = test_individual_restrictions(res1.est1,W,N,spec_1p_x,panel_data)


# ---- specification (2)
spec_2p_x = build_spec_prod(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_2;:AP],[interactions_2;:LW],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)
nmom = spec_2p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_2p_x)
res2 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_2p_x)

# - test restrictions
W = inv(res2.Ω)
t2,p2 = test_joint_restrictions(res2.est1,W,N,spec_2p_x,panel_data)
tvec2,pvec2 = test_individual_restrictions(res2.est1,W,N,spec_2p_x,panel_data)

# ---- specification (3)
# previously we didn't include levels of covariates. This may have been an issue?
spec_3p_x = build_spec_prod((vm = spec_3.vm,vf = spec_3.vf, vg = spec_3.vg,vθ = spec_3.vm,
zlist_prod_t = [0,5],
zlist_prod = [[[spec_3.vg;:LW],[spec_3.vg;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]]))
nmom = spec_3p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_3p_x)
res3 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_3p_x)


# - test restrictions
W = inv(res3.Ω)
t3,p3 = test_joint_restrictions(res3.est1,W,N,spec_3p_x,panel_data)
tvec3,pvec3 = test_individual_restrictions(res3.est1,W,N,spec_3p_x,panel_data)

# ---- specification 5
interactions_5 = make_interactions(panel_data,price_ratios,spec_5.vm)

spec_5p_x = build_spec_prod(
        (vm = spec_5.vm,vf = spec_5.vf, vg = spec_5.vg,vθ = spec_5.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_5;:AP],[interactions_5;:LW],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)
nmom = spec_5p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_5p_x)
res5 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_5p_x)

# - test restrictions
W = inv(res5.Ω)
t5,p5 = test_joint_restrictions(res5.est1,W,N,spec_5p_x,panel_data)
tvec5,pvec5 = test_individual_restrictions(res5.est1,W,N,spec_5p_x,panel_data)


# using JLD2
# jldsave("results/production_restricted.jld2"; res1, res2, res3,res5)

# results = jldopen("results/production_restricted.jld2")
# res1 = results["res1"]
# res2 = results["res2"]
# res3 = results["res3"]
# spec_1p_x = results["spec_1p_x"]
# spec_2p_x = results["spec_2p_x"]
# spec_3p_x = results["spec_3p_x"]


# Write results to a table
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :constant => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1.est1,spec_1p_x),update(res2.est1,spec_2p_x),update(res3.est1,spec_3p_x),update(res5.est1,spec_5p_x)]
se_vec = [update(res1.se,spec_1p_x),update(res2.se,spec_2p_x),update(res3.se,spec_3p_x),update(res5.se,spec_5p_x)]
pval_vec = [update_demand(pvec1,spec_1),update_demand(pvec2,spec_2),update_demand(pvec3,spec_3p_x),update_demand(pvec5,spec_5p_x)]
write_production_table(par_vec,se_vec,pval_vec,[spec_1p_x,spec_2p_x,spec_3p_x,spec_5p_x],labels,"tables/demand_production_restricted.tex"
)

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

P = update(res3.est1,spec_3p_x)
d = model_test(P,P,panel_data,spec_3p_x)
using FixedEffectModels
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))

reg(d,@formula(Phi_m ~ logprice_m_g + logprice_c_g))

# this regression has some clues: a higher relative price of mother's time is negatively correlated, conditional on time investment.
# -- this may go away if we instrument for wages.
# -- but what is the intuition here???
# -- also: perhaps we should substract the average by age for each test score
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:logprice_m_g) + term(:logprice_c_g) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))

reg(d,term(:LW) ~  term(:logprice_m_g) + term(:logprice_c_g) + (term(:log_mtime_97) + term(:LW97) ~ term(:log_mtime) + term(:AP97)))


break
## NOW: for specification (3), my preferred, let's relax coefficients with pvalues<0.05.
 # (1) ρ (2) γ; (2); num_children and type dummies in βm and βg 
np_demand = 2+length(spec_3.vm)+length(spec_3.vf)+length(spec_3.vg)
P_idx = update_demand(collect(1:np_demand),spec_3p_x)

# ---- using just the 0.01 cutoff
unrestricted = pvec3.<0.01
#unrestricted = fill(false,np_demand)
#unrestricted[[P_idx.ρ;P_idx.γ;P_idx.βg]] .= true
P = update(res3.est1,spec_3p_x)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv(P,P,Pu)

W = inv(res3.Ω)
res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))

# --- temporary test to see what's happening :-)
p1,p2 = update(res3u.minimizer,spec_3p_x,unrestricted)

d = model_test(p1,p2,panel_data,spec_3p_x)
reg(d,term(:AP) ~ term(:div) + term(:num_0_5) + term(:age) + term(:cluster_2) + term(:cluster_3) + term(m_ed[2]) + term(:m_ed_16) + term(:Phi_m) + (term(:log_mtime_97) + term(:AP97) ~ term(:log_mtime) + term(:LW97)))

# -----------

Ω = moment_variance(res3u.minimizer,gfunc2!,N,size(W,1),8,panel_data,spec_3p_x,unrestricted)
W = inv(Ω)
res3ub = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),res3u.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))


res4u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=3,show_trace=true))
Ω = moment_variance(res4u.minimizer,gfunc2!,N,size(W,1),8,panel_data,spec_3p_x,unrestricted)
W = inv(Ω)
res4ub = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),res4u.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))


g0 = gmm_criterion(x1,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)

g1 = res3u.minimum
DM = 2N*(g0-g1)

# ---- now using the 0.05 cutoff
unrestricted = pvec3.<0.05
#unrestricted = fill(false,np_demand)
#unrestricted[[P_idx.ρ;P_idx.γ;P_idx.βg]] .= true
P = update(res3.est1,spec_3p_x)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv(P,P,Pu)

W = inv(res3.Ω)
res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))

g0 = gmm_criterion(x1,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)

g1 = res3u.minimum
DM = 2N*(g0-g1)

p1,p2 = update(res3u.minimizer,spec_3p_x,unrestricted)

# this code shows we're still rejecting the joint on the other (but we need to update the weighting matrix, and potentially re-estimate too with new weighting?)
np_demand = 2+length(spec_3p_x.vm)+length(spec_3p_x.vf)+length(spec_3p_x.vg)
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_3p_x)
x1 = update_inv(p1,p2,Pu)
LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)


# OPTIONS:
# (1) run quasi-bayes with fully unrestricted, using priors from the first stage
# (2) just show estimates and move on. They are crazy. So what?
# (3) try using a subset of factor shares that appear significant, go from there.
# (4)

break

res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),res3u.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
V = parameter_variance_gmm(res3u.minimizer,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted)
se = sqrt.(diag(V))
p1,p2 = update(res3u.minimizer,spec_3p_x,unrestricted)
se1,se2 = update(se,spec_3p_x,unrestricted)

unrestricted = fill(false,np_demand)
unrestricted[P_idx.βg[[3,4,5,6,10]]] .= true
P = update(res3.est1,spec_3p_x)
Pu = update_demand(unrestricted,spec_3)
x1 = update_inv(P,P,Pu)


W = inv(res3.Ω)
res3u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_3p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))


break

# GET a rank issue. Perhaps we have to update interactions?
# ALSO: maybe we need to try doing that first. interactions only include spec.vm?
# ---- specification (6)
interactions_6 = make_interactions(panel_data,price_ratios,spec_3.vg)

spec_6p_x = build_spec_prod(
        (vm = spec_3.vg,vf = deleteat!(copy(spec_3.vg),2), vg = spec_3.vg,vθ = spec_3.vg,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_6;:AP],[interactions_6;:LW],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)
nmom = spec_6p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_6p_x)
res6 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_6p_x)

# - test restrictions
W = inv(res6.Ω)
t6,p6 = test_joint_restrictions(res6.est1,W,N,spec_6p_x,panel_data)
tvec6,pvec6 = test_individual_restrictions(res6.est1,W,N,spec_6p_x,panel_data)

# tried fully unrestricted case and was too crazy

break
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=4,show_trace=true))
#V = parameter_variance_gmm(res1u.minimizer,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)
# look at some factor shares
for it in 1:1000
    if panel_data.all_prices[it] && (panel_data.year[it]==1997)
        println(factor_shares(p2,panel_data,it,panel_data.mar_stat[it]))
    end
end

res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted),res1u.minimizer,NewtonTrustRegion(),autodiff=:forward,Optim.Options(iterations=20,show_trace=true))


# ----- Here we try a one-step and don't reject because the variance is insane
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=1,show_trace=true))

dG = ForwardDiff.jacobian(x->moment_func(x,gfunc2!,N,nmom,8,panel_data,spec_1p_x,unrestricted),res1u.minimizer)
var = inv(dG'*W*dG) / N

R = zeros(np_demand,length(x1))
p1_idx,p2_idx = update(collect(1:length(x1)),spec_1p_x,unrestricted)
idx1 = [p1_idx.ρ;p1_idx.γ;p1_idx.βm;p1_idx.βf;p1_idx.βg]
idx2 = [p2_idx.ρ;p2_idx.γ;p2_idx.βm;p2_idx.βf;p2_idx.βg]
for i in 1:sum(unrestricted); R[i,idx1[i]] = 1; R[i,idx2[i]] = -1; end
wald_joint = (R*res1u.minimizer)'*inv(R*var*R')*(R*res1u.minimizer)
# --------

# try estimating just γ and ρ:
unrestricted = fill(false,np_demand)
unrestricted[1:2] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)
t1,p1 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted)

# we hit a NaN problem
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))



# now try the unrestricted estimator:
#res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),x1,LBFGS(),autodiff=:forward,Optim.Options(f_calls_limit=200,show_trace=true))

#res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(iterations=14,show_trace=true))

# ---- experiment 2: update the intercept terms in the factor shares:
P_idx = update_demand(collect(1:np_demand),spec_1p_x)
unrestricted = fill(false,np_demand)
unrestricted[[P_idx.βm[1];P_idx.βf[1];P_idx.βg[1]]] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

P1,P2 = update(x1,spec_1p_x,unrestricted)

t1u,p1u = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted)

nresids = 5
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))

p1,p2 = update(res1u.minimizer,spec_1p_x,unrestricted)


# now test restrictions again
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(p1,p2,Pu)
t1,p1 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

# experiment 3: change all factor shares

unrestricted = fill(false,np_demand)
unrestricted[[P_idx.βm;P_idx.βf]] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)


t1u,p1u = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

# starts again from the initial estimates
res1u2 = optimize(x->gmm_criterion(x,gfunc2!,W,N,nresids,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))

p1,p2 = update(res1u2.minimizer,spec_1p_x,unrestricted)

# experiment 4: change only coefficient terms in factor shares:

unrestricted = fill(false,np_demand)
unrestricted[[P_idx.βm[2:end];P_idx.βf[2:end];P_idx.βg[2:end]]] .= true
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)

P1,P2 = update(x1,spec_1p_x,unrestricted)

t1u,p1u = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted)

nresids = 5
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,8,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))

p1,p2 = update(res1u.minimizer,spec_1p_x,unrestricted)

break
# but it looks like there are some changes in the parameters
P1_idx,P2_idx = update(collect(1:length(x1)),spec_1p_x,unrestricted)
i_r = [P1_idx.βm[1:2];P2_idx.βm[1:2];P1_idx.βf[1];P2_idx.βf[1]]

# ----  experiment 5: test individual parameters using LM

break

# ---- specification (2)
nmom = spec_2p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_2p)

res2 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_2p_x)

# test restrictions jointly:
P = update(res2.est1,spec_2p_x)
W = inv(res2.Ω)
np_demand = 2+length(spec_2.vm)+length(spec_2.vf)+length(spec_2.vg)
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_2)
x1 = update_inv(P,P,Pu)
t2,p2 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_2p_x,unrestricted)


# ---- specification (3)
nmom = spec_3p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_3p)

res3 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_3p_x)

# test restrictions jointly:
P = update(res3.est1,spec_3p_x)
W = inv(res3.Ω)
np_demand = 2+length(spec_3.vm)+length(spec_3.vf)+length(spec_3.vg)
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_3)
x1 = update_inv(P,P,Pu)
t3,p3 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_3p_x,unrestricted)

# ----- Write results to a LaTeX table

# the code below doesn't work yet.
break

cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res2,spec_1p),update(res3,spec_2p),update(res4,spec_3p)]
se_vec = [update(se2,spec_1p),update(se3,spec_2p),update(se4,spec_3p)]
results = [residual_test(panel_data,N,p) for p in par_vec]
pvals = [r[2] for r in results]

writetable(par_vec,se_vec,[spec_1,spec_2,spec_3],labels,pvals,"tables/demand_production_restricted.tex",true)

# I don't recall convergence being an issue before
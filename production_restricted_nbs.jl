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

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec,false)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec,false)

# ---- Part 1: estimate the restricted estimator and conduct tests of the equality constraints using the LM statistic

# ---- specification (1)
spec_1p_x = build_spec_prod(
        (vm = spec_1.vm,vf = spec_1.vf, vg = spec_1.vg,vθ = spec_1.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[spec_1.vg;interactions_1;:LW],[spec_1.vg;interactions_1;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
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
        zlist_prod = [[[spec_2.vg;interactions_2;:LW],[spec_2.vg;interactions_2;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
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
zlist_prod = [[[spec_3.vg;interactions_3;:LW],[spec_3.vg;interactions_3;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]]))
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
        zlist_prod = [[[spec_5.vg;interactions_5;:LW],[spec_5.vg;interactions_5;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)
nmom = spec_5p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_5p_x)
res5 = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_5p_x)

# - test restrictions
W = inv(res5.Ω)
t5,p5 = test_joint_restrictions(res5.est1,W,N,spec_5p_x,panel_data)
tvec5,pvec5 = test_individual_restrictions(res5.est1,W,N,spec_5p_x,panel_data)


# save results to file for future use
using JLD2
JLD2.jldsave("results/production_restricted_nbs.jld2"; res1, res2, res3,res5)

#results = JLD2.load("results/production_restricted.jld2")


# Write results to a table
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :constant => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1.est1,spec_1p_x),update(res2.est1,spec_2p_x),update(res3.est1,spec_3p_x),update(res5.est1,spec_5p_x)]
se_vec = [update(res1.se,spec_1p_x),update(res2.se,spec_2p_x),update(res3.se,spec_3p_x),update(res5.se,spec_5p_x)]
pval_vec = [update_demand(pvec1,spec_1),update_demand(pvec2,spec_2),update_demand(pvec3,spec_3p_x),update_demand(pvec5,spec_5p_x)]
write_production_table(par_vec,se_vec,pval_vec,[spec_1p_x,spec_2p_x,spec_3p_x,spec_5p_x],labels,"tables/demand_production_restricted_nbs.tex"
)

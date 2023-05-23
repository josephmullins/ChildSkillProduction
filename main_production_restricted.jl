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

include("specifications.jl")

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    x0[3:4] = [0.1,0.9] #<- initial guess for δ
    return x0
end

N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,resids,data,spec)
gfunc2!(x,n,g,resids,data,spec,unrestricted) = production_demand_moments_stacked!(update(x,spec,unrestricted)...,n,g,resids,data,spec)


# ---- specification (1)
nmom = spec_1p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_1p_x)

res1 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_1p_x)

# test restrictions jointly:
P = update(res1.est1,spec_1p_x)
W = inv(res1.Ω)
np_demand = 2+length(spec_1.vm)+length(spec_1.vf)+length(spec_1.vg)
unrestricted = fill(true,np_demand)
Pu = update_demand(unrestricted,spec_1)
x1 = update_inv(P,P,Pu)
t1,p1 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)


break
# try the one-step estimator:
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted),x1,Newton(),autodiff=:forward,Optim.Options(iterations=4,show_trace=true))
#V = parameter_variance_gmm(res1u.minimizer,gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

for it in 1:1000; if panel_data.year[it]==1997 && panel_data.all_prices[it]; println(factor_shares(p2,panel_data,it,panel_data.mar_stat[it])); end ;end

# ----- Here we try a one-step and don't reject because the variance is insane
dG = ForwardDiff.jacobian(x->moment_func(x,gfunc2!,N,nmom,nresids,panel_data,spec_1p_x,unrestricted),res1u.minimizer)
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
t1,p1 = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

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

t1u,p1u = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

nresids = 5
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,nresids,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))

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

t1u,p1u = LM_test(x1,sum(unrestricted),gfunc2!,W,N,5,panel_data,spec_1p_x,unrestricted)

nresids = 5
res1u = optimize(x->gmm_criterion(x,gfunc2!,W,N,nresids,panel_data,spec_1p_x,unrestricted),x1,NewtonTrustRegion(),autodiff=:forward,Optim.Options(show_trace=true))

p1,p2 = update(res1u.minimizer,spec_1p_x,unrestricted)

break
# but it looks like there are some changes in the parameters
P1_idx,P2_idx = update(collect(1:length(x1)),spec_1p_x,unrestricted)
i_r = [P1_idx.βm[1:2];P2_idx.βm[1:2];P1_idx.βf[1];P2_idx.βf[1]]




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
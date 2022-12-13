#include("GMMRoutines.jl")
include("direct_estimation.jl")
using Optim
using ForwardDiff
using Statistics
using CSV
using DataFrames

d = DataFrame(CSV.File("../../../PSID_CDS/data-derived/gmm_data.csv",missingstring = "NA"))

d = d[.~ismissing.(d.mar_stat),:]

d.f_ed[.~d.mar_stat .& ismissing.(d.f_ed)] .= "--"

d = dropmissing(d)

idrop = ((d.mar_stat.==1) .& (d.tau_f.==0)) .| (d.tau_m.==0) .| ((d.formal .+ d.informal).==0) .| (d.goods.==0)
idrop2 = ((d.mar_stat.==1) .& (d.tau_f.==0)) .| (d.tau_m.==0) .| (d.chcare.==0)
d2 = d[.~idrop2,:] # still too few!!!
d = d[.~idrop,:]
#d = d[d.age.<=12,:]

# vars = [:KID,:mar_ind,:LW02,:LW97,:AP97,:log_chcare,:log_mtime,:log_good,:p_4c,:price_g]
# d3 = innerjoin(dropmissing(d2[:,vars]),d2[:,[:KID,:log_ftime]],on=:KID)
# idrop = ismissing.(d3.log_ftime) .& d3.mar_ind
# d3 = d3[.~idrop,:]

# vars = [:LW02,:LW97,:AP97,:log_chcare,:log_mtime,:log_good,]
# dropmissing(d2[:,vars]) #<- only 100 observations
# # count the zeros?

# data = (mar = d3.mar_ind,LW02 = (d3.LW02 - 100)/15,LW97 = (d3.LW97 - 100)/15,AP97 = (d3.AP97 - 100)/15,chcare = exp.(d3.log_chcare)./d3.p_4c,mtime = exp.(d3.log_mtime),ftime = exp.(d3.log_ftime),goods = exp.(d3.log_good)./d3.price_g)
# add an education dummy
ed_levels = unique(d.m_ed)
level_names = Dict([("<12","HSdrop"),("12","HS"),("13-15","SomeColl"),("16","Coll"),(">16","CollMore")])

for v in ed_levels
    vname = Symbol("m_ed_",level_names[v])
    d[:,vname] = (d.m_ed.==v)
end


data = (mar = d.mar_stat, LW02 = d.LW02,LW97 = d.LW97,AP97 = d.AP97, mtime = d.tau_m,ftime = d.tau_f,
    goods = d.goods ./ d.price_g,formal = d.formal,informal = d.informal,
    m_ed_HS = d.m_ed_HS,m_ed_SomeColl = d.m_ed_SomeColl,m_ed_Coll = d.m_ed_Coll,m_ed_CollMore = d.m_ed_CollMore, age = d.age)


spec = (vm = [:mar,], vf = [], vg = [:mar],vθ = [:mar,:m_ed_HS,:m_ed_SomeColl,:m_ed_Coll,:m_ed_CollMore],vY = [:formal, :informal])

spec2 = (vm = [:mar,:m_ed_HS,:m_ed_SomeColl,:m_ed_Coll,:m_ed_CollMore,:age], vf = [], vg = [:mar,:age],vθ = [:mar,:m_ed_HS,:m_ed_SomeColl,:m_ed_Coll,:m_ed_CollMore,:age],vY = [:formal, :informal])


up(x,ρ,γ) = [x[1:2];ρ;γ;x[3:end]]


# -- First Specification --- #

P = CESmod(spec)
x0 = CESvec(P)


ρfix = -1.2
γfix = -2.0

x0[3:4] = [ρfix,γfix]
Z = get_instruments([x0[1:2];x0[5:end]],x->up(x,ρfix,γfix),spec,data) #<- pass update function also
res = optimize(x->gmm_criterion(up(x,ρfix,γfix),Z,spec,data),[x0[1:2];x0[5:end]],LBFGS(),autodiff = :forward)
x_est = res.minimizer
# parameter variance: pass update function also
#V = parameter_variance(x_est,Z,spec,data,up)
P1 = CESmod(up(x_est,ρfix,γfix),spec)
# se0 = CESmod(std(X,dims=2)[:],spec)
#se = CESmod(sqrt.(diag(V)),spec)
X = bootstrap(x_est,Z,spec,data,x->up(x,ρfix,γfix),1010,50,true)
se = std(X,dims=2)[:]
se1 = CESmod([se[1:2];0;0;se[3:end]],spec)

# --- Second Specification --- #

ρfix = -0.5
γfix = -0.5

x0[3:4] = [ρfix,γfix]
#up(x) = [x[1:2];ρfix;γfix;x[3:end]]
Z = get_instruments([x0[1:2];x0[5:end]],x->up(x,ρfix,γfix),spec,data) #<- pass update function also
res = optimize(x->gmm_criterion(up(x,ρfix,γfix),Z,spec,data),[x0[1:2];x0[5:end]],LBFGS(),autodiff = :forward)
x_est = res.minimizer
P2 = CESmod(up(x_est,ρfix,γfix),spec)
# se0 = CESmod(std(X,dims=2)[:],spec)
#se = CESmod(sqrt.(diag(V)),spec)
X = bootstrap(x_est,Z,spec,data,x->up(x,ρfix,γfix),1010,50,true)
se = std(X,dims=2)[:]
se2 = CESmod([se[1:2];0;0;se[3:end]],spec)



# ---- Third Specification
P = CESmod(spec2)
x0 = CESvec(P)
ρfix = -1.2
γfix = -2.0

x0[3:4] = [ρfix,γfix]

Z = get_instruments([x0[1:2];x0[5:end]],x->up(x,ρfix,γfix),spec2,data) #<- pass update function also
res = optimize(x->gmm_criterion(up(x,ρfix,γfix),Z,spec2,data),[x0[1:2];x0[5:end]],LBFGS(),autodiff = :forward)
x_est = res.minimizer
P3 = CESmod(up(x_est,ρfix,γfix),spec2)
# se0 = CESmod(std(X,dims=2)[:],spec2)
#se = CESmod(sqrt.(diag(V)),spec2)
X = bootstrap(x_est,Z,spec2,data,x->up(x,ρfix,γfix),1010,50,true)
se = std(X,dims=2)[:]
se3 = CESmod([se[1:2];0;0;se[3:end]],spec2)



# ----- Write these two specifications to table:

labels = (mar = "Married",m_ed_HS = "Mother: HS",m_ed_SomeColl = "Mother: Some Coll.",m_ed_Coll = "Mother: Coll.", m_ed_CollMore = "Mother: College+",
    formal = "Formal Care", informal = "Informal Care",age = "Child Age")
writetable([P1,P2,P3],[se1,se2,se3],[spec,spec,spec2],labels,"tables/results_fixed_rho_gamma.tex")

break
# get initial guess using NLLS:
# x0[3:4] = [-1.,-1.]
# res = optimize(x->sumsq(x,spec,data),x0,NelderMead())
# res = optimize(x->sumsq(x,spec,data),res.minimizer,LBFGS(),autodiff=:forward)
# x1 = res.minimizer
# f1 = res.minimum

# x0[3:4] = [0.5,0.5]
# res = optimize(x->sumsq(x,spec,data),x0,NelderMead())
# res = optimize(x->sumsq(x,spec,data),res.minimizer,LBFGS(),autodiff=:forward)
# x2 = res.minimizer
# f2 = res.minimum

x0[3:4] = [-0.5,-0.5]
res = optimize(x->sumsq(x,spec,data),x0,NelderMead(),Optim.Options(iterations=2000))
# res = optimize(x->sumsq(x,spec,data),res.minimizer,LBFGS(),autodiff=:forward)
# x3 = res.minimizer
# f3 = res.minimum

P = CESmod(spec)
x0 = CESvec(P)
P2 = CESmod(x0,spec)
break

Z = get_instruments(x0,P,spec,data)
gmm_criterion(x0,Z,spec,data)

break

# exercise: use three starting values and compare the final function values
# 0.5,-1.,-2.

# x0[3:4] .= [0.5,0.5]
# x1 = copy(x0)
# x1[3:4] = [-1.,-1.]
# x2 = copy(x0)
# x2[3:4] = [-2.,-2.]

Z = get_instruments(res.minimizer,P,spec,data)


res = optimize(x->gmm_criterion(x,Z,spec,data),x0,LBFGS(),autodiff = :forward)
Z = get_instruments(res.minimizer,P,spec,data)
res = optimize(x->gmm_criterion(x,Z,spec,data),x0,LBFGS(),autodiff = :forward)
res = optimize(x->gmm_criterion(x,Z,spec,data),res.minimizer,LBFGS(),autodiff = :forward)
x0_est = res.minimizer
V0 = parameter_variance(x0_est,Z,spec,data)
# X = bootstrap(x0_est,Z,spec,data,x->x,1010,50,true)
P0 = CESmod(x0_est,spec)
# se0 = CESmod(std(X,dims=2)[:],spec)
se0 = CESmod(sqrt.(diag(V0)),spec)

res = optimize(x->gmm_criterion(x,Z,spec,data),x1,LBFGS(),autodiff = :forward)
x1_est = res.minimizer
V1 = parameter_variance(x1_est,Z,spec,data)
#X = bootstrap(x1_est,Z,spec,data,x->x,1010,50,true)
P1 = CESmod(x1_est,spec)
#se1 = CESmod(std(X,dims=2)[:],spec)
se1 = CESmod(sqrt.(diag(V1)),spec)

res = optimize(x->gmm_criterion(x,Z,spec,data),x2,LBFGS(),autodiff = :forward)
x2_est = res.minimizer
V2 = parameter_variance(x2_est,Z,spec,data)
#X = bootstrap(x2_est,Z,spec,data,x->x,1010,50,true)
P2 = CESmod(x2_est,spec)
#se1 = CESmod(std(X,dims=2)[:],spec)
se2 = CESmod(sqrt.(diag(V2)),spec)


labels = (mar = "Married",m_ed_HS = "Mother: HS",m_ed_SomeColl = "Mother: Some Coll.",m_ed_Coll = "Mother: Coll.", m_ed_CollMore = "Mother: College+")
writetable([P0,P1,P2],[se0,se1,se2],[spec,spec,spec],labels,"tables/table_weak_id.tex")

break

res = optimize(x->gmm_criterion(x,Z,spec,data),x0,LBFGS(),autodiff = :forward)
x1 = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x1,NelderMead())
x2 = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x2,NelderMead())
x3 = res.minimizer

x0b = copy(x0)
x0b[3:4] = [-0.5,-0.5]
res = optimize(x->gmm_criterion(x,Z,spec,data),x0b,LBFGS(),autodiff = :forward,Optim.Options(iterations=3000))
x1b = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x1b,NelderMead())
x2b = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x2b,NelderMead())
x3b = res.minimizer
# this does better.
# very weakly identified

x0c = copy(x0)
x0c[3:4] = [0.5,0.5]
res = optimize(x->gmm_criterion(x,Z,spec,data),x0c,LBFGS(),autodiff = :forward)
x1c = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x1c,NelderMead())
x2c = res.minimizer
res = optimize(x->gmm_criterion(x,Z,spec,data),x2c,NelderMead())
x3c = res.minimizer


X = bootstrap(x1,Z,spec,data,x->x,1010,50,true)

P = CESmod(x1,spec)
se = CESmod(std(X,dims=2)[:],spec)

labels = (mar = "Married",m_ed_HS = "Mother: HS",m_ed_SomeColl = "Mother: Some Coll.",m_ed_Coll = "Mother: Coll.", m_ed_CollMore = "Mother: College+")

P2 = CESmod(spec)
x0 = CESvec(P2)
res = optimize(x->gmm_criterion([0.05;0.95;x],Z,spec,data),x0[3:end],LBFGS(),autodiff = :forward)
x2 = [0.05;0.95;res.minimizer]
P2 = CESmod(x2,spec)
# ERROR: can't do the bootstrap because something goes wrong with estimation
X2 = bootstrap(x2[3:end],Z,spec,data,x->[0.05;0.95;x],2020)
se2 = CESmod([0.;0.;std(X2,dims=2)[:]],spec)

writetable([P,P2],[se,se2],[spec,spec],labels,"tables/test_table.tex")


break

Z2 = get_instruments(x1,P,spec,data)
res = optimize(x->gmm_criterion(x,Z2,spec,data),x0,LBFGS(),autodiff = :forward)
x2 = res.minimizer

Z3 = get_instruments(x2,P,spec,data)
res = optimize(x->gmm_criterion(x,Z3,spec,data),x0,LBFGS(),autodiff = :forward)
x3 = res.minimizer

Z4 = get_instruments(x3,P,spec,data)
res = optimize(x->gmm_criterion(x,Z4,spec,data),x0,LBFGS(),autodiff = :forward)
x4 = res.minimizer


V = moment_variance(x4,Z,spec,data)
dg = ForwardDiff.jacobian(x->gfunc(x,Z4,spec,data),x4)

bread = inv(dg'*dg)


parameter_variance(x1,Z,spec,data)
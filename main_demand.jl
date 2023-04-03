using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")



# --------  read in the data:
# -- For now, we're using the old data. A task is to replicate how these data were created
# Step 1: create the data object
panel_data = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
ind_data = DataFrame(CSV.File("CLMP_v1/data/gmm_full_horizontal.csv",missingstring = "NA"))
nclusters = 3

# -- Read in wage data from the mother's panel;
wage_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
wage_data = subset(wage_data,:m_wage => x->x.>0,skipmissing=true)
wage_data[!,:logwage_m] = log.(wage_data.m_wage)
wage_data[!,:age_sq] = wage_data.age_mother.^2
wage_data[!,:const] .= 1.
m_ed = make_dummy(wage_data,:m_ed)
vl=[m_ed[2:end];:age_mother;:age_sq]



wage_types = cluster_routine_robust(wage_data,vl,nclusters)

wage_types_k10 = cluster_routine_robust(wage_data,vl,10,500)
wage_types_k10 = rename(select(wage_types_k10,[:MID,:center]),:center => :mu_k)

# --- alternatively using the naive clustering algorithm which won't work if we include education in vl
# wage_types = generate_cluster_assignment(wage_data,vl,true,nclusters)
# wage_types_k10 = generate_cluster_assignment(wage_data,vl,true,10)
# wage_types_k10 = rename(select(wage_types_k10,[:MID,:center]),:center => :mu_k)

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
panel_data = innerjoin(panel_data,wage_types_k10,on = :MID) # mergining in centers for K=10 clustering
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

panel_data[!,:mar_stat] = panel_data.mar_stable
panel_data[!,:logwage_m] = log.(panel_data.m_wage)
panel_data[!,:logwage_f] = log.(panel_data.f_wage)
panel_data[!,:logprice_g] = log.(panel_data.price_g)
panel_data[!,:logprice_c] = log.(panel_data.p_4f) # alternative is p_4c
#panel_data[!,:log_chcare] = replace(log.(panel_data.chcare),-Inf => missing)
panel_data[!,:log_mtime] = panel_data.log_mtime .- panel_data.logwage_m
panel_data[!,:log_ftime] = panel_data.log_ftime .- panel_data.logwage_f


# in this section we drop for missing variables and this makes the panel unbalanced.
# --- we want to think aboyt how else to do this potentially
# -- example: build in a missing data check in the moment function

#panel_data = panel_data[.!ismissing.(panel_data.mar_stat),:]
panel_data = panel_data[.!ismissing.(panel_data.price_g),:] #<- drop observations with missing prices (goods)
panel_data = panel_data[.!ismissing.(panel_data.m_wage),:] #<- drop observations with missing prices (mother's wage)
panel_data = panel_data[.!(panel_data.mar_stat .& ismissing.(panel_data.f_wage)),:] #<- drop with missing prices (father's wage)
# panel_data[!,:goods] = panel_data.Toys .+ panel_data.tuition .+ panel_data.comm_grps .+ panel_data.lessons .+ panel_data.tutoring .+ panel_data.sports .+ panel_data.SchSupplies
# panel_data[!,:log_good] = replace(log.(panel_data.goods),-Inf => missing)
panel_data.m_ed = replace(panel_data.m_ed,">16" => "16","<12" => "12") #<-  simplify the education categories
panel_data.f_ed = replace(panel_data.f_ed,">16" => "16","<12" => "12")
m_ed = make_dummy(panel_data,:m_ed)
f_ed = make_dummy(panel_data,:f_ed)
panel_data[.!panel_data.mar_stat,f_ed] .= 0. #<- make zero by default. this won't work all the time
panel_data = panel_data[.!ismissing.(panel_data.age),:]
panel_data.logwage_f = coalesce.(panel_data.logwage_f,0) #<- make these into zeros to avoid a problem with instruments
panel_data[!,:div] = .!panel_data.mar_stat
panel_data[!,:const] .= 1.
panel_data = panel_data[panel_data.price_missing.==0,:]
#panel_data = subset(panel_data,:year => x->(x.==1997) .| (x.==2002)) #<- for now, limit to years 1997 and 2002

# ordering of residuals: c/m,f/m,c/g,m/g,f/g
# -- make some price ratios to pass as instruments:
# -- put this in prep data or even in earlier R code
panel_data[!,:logprice_c_m] = panel_data.logprice_c .- panel_data.logwage_m
panel_data[!,:logprice_m_f] = panel_data.logwage_m .- panel_data.logwage_f
panel_data[!,:logprice_c_g] = panel_data.logprice_c .- panel_data.logprice_g
panel_data[!,:logprice_m_g] = panel_data.logwage_m .- panel_data.logprice_g
panel_data[!,:logprice_f_g] = panel_data.logwage_f .- panel_data.logprice_g

# overwrite age and marital status to see if this gets old estimates back
#select!(panel_data,Not(:age))
#panel_data = innerjoin(panel_data,ind_data[:,[:KID,:age]],on=:KID)


break
# ----------------------------- #

#---- write the update function:
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    nm = length(spec.vm)
    βm = x[3:2+nm]
    pos = 3+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    return CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
end
function update_inv(pars)
    @unpack ρ,γ,βm,βf,βg = pars
    return [ρ;γ;βm;βf;βg]
end

# - load the specifications that we want to use. See that script for more details.
include("specifications.jl")

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    return x0
end


gfunc!(x,n,g,resids,data,gd,gmap,spec) = demand_moments_stacked!(update(x,spec),n,g,resids,data,spec)

# all of the specifications here calculate moments using the child as the observational unit
gd = groupby(panel_data,:KID)


# Specification (1): spec_1 with version_1 of moments
x0 = initial_guess(spec_1)

n97s = length(spec_1.vm)
n02s = length(spec_1.vm)*3
n97m = length(spec_1.vg)
n02m = length(spec_1.vg)*2 + length(spec_1.vm) + length(spec_1.vf) + 1
nmom = n97s+n02s+n97m+n02m
N = length(unique(panel_data.KID))
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v1,spec_1)
res1,se1 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v1,spec_1)

# Specification (2): spec_1 with version_2 of moments
n97 = length(spec_1.vg) + 1 
n02 = (length(spec_1.vg)+1)*2 + length(spec_1.vf) + length(spec_1.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_1)
res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_1)

# Specification (3): spec_2 with version_2 of moments
x0 = initial_guess(spec_2)

n97 = length(spec_2.vg) + 1 
n02 = (length(spec_2.vg)+1)*2 + length(spec_2.vf) + length(spec_2.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_2)
res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_2)

# Specification (4): spec_3 with version_2 of moments
x0 = initial_guess(spec_3)

n97 = length(spec_3.vg) + 1 
n02 = (length(spec_3.vg)+1)*2 + length(spec_3.vf) + length(spec_3.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_3)
res4,se4 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_3)

# Specification (5): spec_4 with version_2 of moments
x0 = initial_guess(spec_4)

n97 = length(spec_4.vg) + 1 
n02 = (length(spec_4.vg)+1)*2 + length(spec_4.vf) + length(spec_4.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_4)
res5,se5 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_4)

# Specification (6): spec_5 with version_2 of moments
x0 = initial_guess(spec_5)

n97 = length(spec_5.vg) + 1 
n02 = (length(spec_5.vg)+1)*2 + length(spec_5.vf) + length(spec_5.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_5)
res6,se6 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_5)

# ----- Write results to a LaTeX table

cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1,spec_1),update(res2,spec_1),update(res3,spec_2),update(res4,spec_3),update(res5,spec_4),update(res6,spec_5)]

results = [residual_test(panel_data,gd,p) for p in par_vec]
pvals = [r[2] for r in results]

writetable(par_vec,[update(se1,spec_1),update(se2,spec_1),update(se3,spec_2),update(se4,spec_3),update(se5,spec_4),update(se6,spec_5)],[spec_1,spec_1,spec_2,spec_3,spec_4,spec_5],labels,pvals,"tables/relative_demand.tex")

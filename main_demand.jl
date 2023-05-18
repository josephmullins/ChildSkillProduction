using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")

# --------  read in the data:
# -- For now, we're using the old data. A task is to replicate how these data were created
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))

# temporary:
panel_data.mid[ismissing.(panel_data.mid)] .= 6024032


panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


panel_data = panel_data[panel_data.year.<=2002,:] #<- for now, limit to years <=2002. We need to update code eventually.
include("temp_prep_data.jl")

# TEMPORARY EXERCISE: let's merge with the new data to see if it's the new observations that cause the issue:
# panel_data[!,:KID] = panel_data.kid
# old_data = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
# panel_data = innerjoin(panel_data,old_data[:,[:KID,:year]],on=[:KID,:year])

sort!(panel_data,[:kid,:year])


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


gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update(x,spec),n,g,resids,data,spec)

# call this function instead and get it to work using a production specification
#production_moments_stacked!(update(x,spec),n,g,resids,data,spec)

N = length(unique(panel_data.kid))

# Specification (1)

nmom = spec_1.g_idx_02[end]
W = I(nmom)
x0 = initial_guess(spec_1)

res1 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_1)
#res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_1)

# Specification (2): 
x0 = initial_guess(spec_2)
nmom = spec_2.g_idx_02[end]
W = I(nmom)

res2 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_2)
#res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2)

# Specification (3): 
x0 = initial_guess(spec_3)
nmom = spec_3.g_idx_02[end]
W = I(nmom)
res3 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_3)
#res4,se4 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_3)

# Specification (4): 
x0 = initial_guess(spec_4)
nmom = spec_4.g_idx_02[end]
W = I(nmom)
res4 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_4)
#res5,se5 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_4)


# Specification (5): 
x0 = initial_guess(spec_5)
nmom = spec_5.g_idx_02[end]
W = I(nmom)
res5 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_5)

#res6,se6 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_5)

# ----- Write results to a LaTeX table

cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res1.est1,spec_1),update(res2.est1,spec_2),update(res3.est1,spec_3),update(res4.est1,spec_4),update(res5.est1,spec_5)]
#par_vec = [update(res2,spec_1),update(res3,spec_2),update(res4,spec_3),update(res5,spec_4),update(res6,spec_5)]
se_vec = [update(res1.se,spec_1),update(res2.se,spec_2),update(res3.se,spec_3),update(res4.se,spec_4),update(res5.se,spec_5)]
results = [residual_test(panel_data,N,p) for p in par_vec]
pvals = [r[2] for r in results]

writetable(par_vec,se_vec,[spec_1,spec_2,spec_3,spec_4,spec_5],labels,pvals,"tables/relative_demand.tex")

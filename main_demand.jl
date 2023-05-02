using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")

# --------  read in the data:
# -- For now, we're using the old data. A task is to replicate how these data were created
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


panel_data = panel_data[panel_data.year.<=2002,:] #<- for now, limit to years <=2002. We need to update code eventually.
include("temp_prep_data.jl")


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

# Specification (1)

n97 = length(spec_1.vg) + 1 
n02 = (length(spec_1.vg)+1)*2 + length(spec_1.vf) + length(spec_1.vm) + 2
nmom = n97+n02
W = I(nmom)

g = zeros(nmom)
resids = zeros(5)
x0 = initial_guess(spec_1)

N = length(unique(panel_data.kid))
@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_1)

res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_1)

# Specification (2): 
x0 = initial_guess(spec_2)
n97 = length(spec_2.vg) + 1 
n02 = (length(spec_2.vg)+1)*2 + length(spec_2.vf) + length(spec_2.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_2)
res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2)

# Specification (3): 
x0 = initial_guess(spec_3)

n97 = length(spec_3.vg) + 1 
n02 = (length(spec_3.vg)+1)*2 + length(spec_3.vf) + length(spec_3.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_3)
res4,se4 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_3)

# Specification (4): 
x0 = initial_guess(spec_4)

n97 = length(spec_4.vg) + 1 
n02 = (length(spec_4.vg)+1)*2 + length(spec_4.vf) + length(spec_4.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(res4,gfunc!,W,N,5,panel_data,spec_4)
res5,se5 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_4)

# Specification (5): 
x0 = initial_guess(spec_5)

n97 = length(spec_5.vg) + 1 
n02 = (length(spec_5.vg)+1)*2 + length(spec_5.vf) + length(spec_5.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_5)
res6,se6 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_5)

# ----- Write results to a LaTeX table

cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.", :mu_k => "\$\\mu_{k}\$", :age => "Child Age")

labels = merge(other_labels,cluster_labels,ed_labels)

par_vec = [update(res2,spec_1),update(res3,spec_2),update(res4,spec_3),update(res5,spec_4),update(res6,spec_5)]

results = [residual_test(panel_data,N,p) for p in par_vec]
pvals = [r[2] for r in results]

writetable(par_vec,[update(se2,spec_1),update(se3,spec_2),update(se4,spec_3),update(se5,spec_4),update(se6,spec_5)],[spec_1,spec_2,spec_3,spec_4,spec_5],labels,pvals,"tables/relative_demand.tex")

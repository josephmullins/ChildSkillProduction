using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")

# --------  read in the data:
# -- For now, we're using the old data. A task is to replicate how these data were created
# Step 1: create the data object
panel_data = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
#panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))

panel_data.m_ed = replace(panel_data.m_ed,">16" => "16","<12" => "12") #<-  simplify the education categories
panel_data.f_ed = replace(panel_data.f_ed,">16" => "16","<12" => "12")
m_ed = make_dummy(panel_data,:m_ed)
f_ed = make_dummy(panel_data,:f_ed)
panel_data[!,:constant] .= 1.
panel_data[!,:mar_stat] = panel_data.mar_ind
panel_data[!,:div] = .!panel_data.mar_stat
panel_data[!,f_ed] = coalesce.(panel_data[:,f_ed],0.)
panel_data[!,:ind_not_sample] .= false
# demographics:
N = length(unique(panel_data.KID))
for n in 1:N, t in 1:5
    it97 = (n-1)*6+1
    panel_data.age[it97+t] = panel_data.age[it97]+t
end

# prices:
#v_prices = [:prices_observed,:logwage_m,:logwage_f,:logprice_g,:logprice_c,:logprice_c_m,:logprice_m_f,:logprice_c_g,:logprice_m_g,:logprice_f_g]

panel_data[!,:logwage_m] = log.(panel_data.m_wage)
panel_data[!,:logwage_f] = coalesce.(log.(panel_data.f_wage),0) #<- make these into zeros to avoid a problem with instruments
panel_data[!,:prices_observed] = .!panel_data.price_missing
panel_data[!,:logprice_g] = log.(panel_data.price_g)
panel_data[!,:logprice_c] = log.(panel_data.p_4c) #log.(panel_data.p_yocent_e_cps_cpkt) .- log(33*52)
### relative prices
panel_data[!,:logprice_c_m] = panel_data.logprice_c .- panel_data.logwage_m
panel_data[!,:logprice_m_f] = panel_data.logwage_m .- panel_data.logwage_f
panel_data[!,:logprice_c_g] = panel_data.logprice_c .- panel_data.logprice_g
panel_data[!,:logprice_m_g] = panel_data.logwage_m .- panel_data.logprice_g
panel_data[!,:logprice_f_g] = panel_data.logwage_f .- panel_data.logprice_g


# inputs and outputs
#v_inputs = [:log_mtime,:log_ftime,:log_chcare,:log_good,:log_total_income]

panel_data[!,:log_mtime] = panel_data.log_mtime .- panel_data.logwage_m
panel_data[!,:log_ftime] = panel_data.log_ftime .- panel_data.logwage_f
#panel_data[!,:log_chcare] = panel_data.ln_chcare_exp
#panel_data[!,:log_good] = panel_data.ln_hhinvest
#panel_data[!,:log_total_income] = log.(panel_data.m_wage .+ coalesce.(panel_data.f_wage,0))
panel_data[!,:mtime_valid] = .!ismissing.(panel_data.log_mtime) .& .!ismissing.(panel_data.m_ed)
panel_data[!,:ftime_valid] = .!ismissing.(panel_data.log_ftime) .| .!panel_data.mar_stat

# TODO: run the code with this switched off vs switched on
#panel_data[!,:log_mtime] = coalesce.(panel_data.log_mtime,0.) # this could be the issue!!! because this check might fuck up everything else!!!
#panel_data[!,:log_ftime] = coalesce.(panel_data.log_ftime,0.) #


wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


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

N = length(unique(panel_data.KID))
@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_1)

res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_1)

break
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

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_4)
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

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

# Read in the old data
panel_data[!,:KID] = panel_data.kid
old_data = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
panel_data2 = innerjoin(panel_data,old_data[:,[:KID,:year]],on=[:KID,:year])

sort!(panel_data,[:kid,:year])

# do prep on the old data:
old_data.m_ed = replace(old_data.m_ed,">16" => "16","<12" => "12") #<-  simplify the education categories
old_data.f_ed = replace(old_data.f_ed,">16" => "16","<12" => "12")
m_ed = make_dummy(old_data,:m_ed)
f_ed = make_dummy(old_data,:f_ed)
old_data[!,:constant] .= 1.
old_data[!,:mar_stat] = old_data.mar_ind
old_data[!,:div] = .!old_data.mar_stat
old_data[!,f_ed] = coalesce.(old_data[:,f_ed],0.)
old_data[!,:ind_not_sample] .= false
# demographics:
N = length(unique(old_data.KID))
for n in 1:N, t in 1:5
    it97 = (n-1)*6+1
    old_data.age[it97+t] = old_data.age[it97]+t
end

# prices:
#v_prices = [:prices_observed,:logwage_m,:logwage_f,:logprice_g,:logprice_c,:logprice_c_m,:logprice_m_f,:logprice_c_g,:logprice_m_g,:logprice_f_g]

old_data[!,:logwage_m] = log.(old_data.m_wage)
old_data[!,:logwage_f] = coalesce.(log.(old_data.f_wage),0) #<- make these into zeros to avoid a problem with instruments
old_data[!,:prices_observed] = .!old_data.price_missing
old_data[!,:logprice_g] = log.(old_data.price_g)
old_data[!,:logprice_c] = log.(old_data.p_4c) #log.(old_data.p_yocent_e_cps_cpkt) .- log(33*52)
### relative prices
old_data[!,:logprice_c_m] = old_data.logprice_c .- old_data.logwage_m
old_data[!,:logprice_m_f] = old_data.logwage_m .- old_data.logwage_f
old_data[!,:logprice_c_g] = old_data.logprice_c .- old_data.logprice_g
old_data[!,:logprice_m_g] = old_data.logwage_m .- old_data.logprice_g
old_data[!,:logprice_f_g] = old_data.logwage_f .- old_data.logprice_g


# inputs and outputs
#v_inputs = [:log_mtime,:log_ftime,:log_chcare,:log_good,:log_total_income]

old_data[!,:log_mtime] = old_data.log_mtime .- old_data.logwage_m
old_data[!,:log_ftime] = old_data.log_ftime .- old_data.logwage_f
old_data[!,:mtime_valid] = .!ismissing.(old_data.log_mtime) .& .!ismissing.(old_data.m_ed)
old_data[!,:ftime_valid] = .!ismissing.(old_data.log_ftime) .| .!old_data.mar_stat


old_data=innerjoin(old_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(old_data,:cluster) #cluster dummies made


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


N = length(unique(panel_data.kid))


# Specification (3): 
x0 = initial_guess(spec_3)

n97 = length(spec_3.vg) + 1 
n02 = (length(spec_3.vg)+1)*2 + length(spec_3.vf) + length(spec_3.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_3)
N = length(unique(panel_data.KID))
res1,se1 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_3)

N = length(unique(panel_data2.KID))
res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data2,spec_3)

N = length(unique(old_data.KID))
res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,old_data,spec_3)

# lesson (1): the change in the sample does lead to a bit of a difference.

# lesson (2): the change in all the data leads to big differences! 
    # the main difference is in factor shares, you can see that this is mostly coming from differences in the measurement of time investment

[mean(skipmissing(panel_data2.log_mtime)) mean(skipmissing(old_data.log_mtime))]

[mean(skipmissing(panel_data2.log_ftime)) mean(skipmissing(old_data.log_ftime))]

[mean(skipmissing(panel_data2.log_good)) mean(skipmissing(old_data.log_good))]

[mean(skipmissing(panel_data2.log_chcare)) mean(skipmissing(old_data.log_chcare))]

# prices:
[mean(skipmissing(panel_data2.logprice_c)) mean(skipmissing(old_data.logprice_c));
mean(skipmissing(panel_data2.logprice_g)) mean(skipmissing(old_data.logprice_g));
mean(skipmissing(panel_data2.logwage_m)) mean(skipmissing(old_data.logwage_m));
mean(skipmissing(panel_data2.logwage_f)) mean(skipmissing(old_data.logwage_f))
]

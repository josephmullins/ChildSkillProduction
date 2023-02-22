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

# -- Read in wage data from the mother's pane;
wage_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
wage_data[!,:logwage_m] = log.(wage_data.m_wage)
wage_data[!,:age_sq] = wage_data.age_mother.^2

output=generate_cluster_assignment(wage_data,true,nclusters) #getting our clustering assignments
c=output[1] #dataframe for clusters

panel_data=innerjoin(panel_data, c, on = :MID) #merging in 
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

panel_data[!,:mar_stat] = panel_data.mar_stable
panel_data[!,:logwage_m] = log.(panel_data.m_wage)
panel_data[!,:logwage_f] = log.(panel_data.f_wage)
panel_data[!,:logprice_g] = log.(panel_data.price_g)
panel_data[!,:logprice_c] = log.(panel_data.p_4f) # alternative is p_4c
#panel_data[!,:log_chcare] = replace(log.(panel_data.chcare),-Inf => missing)
panel_data[!,:log_mtime] = panel_data.log_mtime .- panel_data.logwage_m
panel_data[!,:log_ftime] = panel_data.log_ftime .- panel_data.logwage_f

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
panel_data = subset(panel_data,:year => x->(x.==1997) .| (x.==2002)) #<- for now, limit to years 1997 and 2002

# ordering of residuals: c/m,f/m,c/g,m/g,f/g
# -- make some price ratios to pass as instruments:
# -- put this in prep data or even in earlier R code
panel_data[!,:logprice_c_m] = panel_data.logprice_c .- panel_data.logwage_m
panel_data[!,:logprice_m_f] = panel_data.logwage_m .- panel_data.logwage_f
panel_data[!,:logprice_c_g] = panel_data.logprice_c .- panel_data.logprice_g
panel_data[!,:logprice_m_g] = panel_data.logwage_m .- panel_data.logprice_g
panel_data[!,:logprice_f_g] = panel_data.logwage_f .- panel_data.logprice_g



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

#rename(D2, :const => :constant) #using const mucked things up for the write tools

#spec = (vm = [:mar_stat;:div;m_ed[2:3];:age;:num_0_5;],vf = [:const;f_ed[2:3];:age;:num_0_5],vθ = [:const,:mar_stat,:age,:num_0_5],vg = [:mar_stat;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])
#cluster dummies have been added to vm


P = CESmod(spec_1)
x0 = update_inv(P)
x0[1:2] .= -2. #<- initial guess consistent with last time

gfunc!(x,n,g,resids,data,gd,gmap,spec) = demand_moments_stacked!(update(x,spec),n,g,resids,data,gd,gmap,spec)
# all of the specifications here calculate moments using the child as the observational unit
gd = groupby(panel_data,:KID)


# Specification (1): spec_1 with version_1 of moments
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
n97 = length(spec_2.vg) + 1 
n02 = (length(spec_2.vg)+1)*2 + length(spec_2.vf) + length(spec_2.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_2)
res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_2)

# Specification (4): spec_3 with version_2 of moments
n97 = length(spec_3.vg) + 1 
n02 = (length(spec_3.vg)+1)*2 + length(spec_3.vf) + length(spec_3.vm) + 2
nmom = n97+n02
W = I(nmom)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,gd,gmap_v2,spec_3)
res4,se4 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,gd,gmap_v2,spec_3)


cluster_labels = Dict(zip(cluster_dummies[2:3],["Type $s" for s in 2:nclusters]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father: College+","Father: Some College","Mother: Some College","Mother: College+"]))

other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. Children 0-5", :const => "Const.")

labels = merge(other_labels,cluster_labels,ed_labels)


writetable([update(res1,spec_1),update(res2,spec_1),update(res3,spec_2)],[update(se1,spec_1),update(se2,spec_1),update(se3,spec_2)],[spec_1,spec_1,spec_2],labels,"tables/relative_demand.tex")

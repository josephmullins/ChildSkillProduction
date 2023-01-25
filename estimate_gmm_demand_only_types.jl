include("estimation_tools.jl")
include("relative_demand.jl")

# --------  read in the data:
# -- For now, we're using the old data. A task is to replicate how these data were created
# Step 1: create the data object
D2 = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
d = DataFrame(CSV.File("CLMP_v1/data/gmm_full_horizontal.csv",missingstring = "NA"))
# NOTE: if this code is used, the age of the child in 1997 is used for both 1997 and 2002 observations which is a bit weird.
# rename!(D2,:age => :age2)
# D2 = innerjoin(D2,d[:,[:KID,:age]],on=:KID)

# run clustering on wages here and merge with D2 using :MID -> get a variable :type

D2[!,:mar_stat] = D2.mar_stable
D2[!,:logwage_m] = log.(D2.m_wage)
#D2[!,:age_sq] = D2.age_mother.^2
D2[!,:logwage_f] = log.(D2.f_wage)
D2[!,:logprice_g] = log.(D2.price_g)
D2[!,:logprice_c] = log.(D2.p_4f) # alternative is p_4c
#D2[!,:log_chcare] = replace(log.(D2.chcare),-Inf => missing)
D2[!,:log_mtime] = D2.log_mtime .- D2.logwage_m
D2[!,:log_ftime] = D2.log_ftime .- D2.logwage_f
#D2 = D2[.!ismissing.(D2.mar_stat),:]
D2 = D2[.!ismissing.(D2.price_g),:] #<- drop observations with missing prices (goods)
D2 = D2[.!ismissing.(D2.m_wage),:] #<- drop observations with missing prices (mother's wage)
D2 = D2[.!(D2.mar_stat .& ismissing.(D2.f_wage)),:] #<- drop with missing prices (father's wage)
# D2[!,:goods] = D2.Toys .+ D2.tuition .+ D2.comm_grps .+ D2.lessons .+ D2.tutoring .+ D2.sports .+ D2.SchSupplies
# D2[!,:log_good] = replace(log.(D2.goods),-Inf => missing)
D2.m_ed = replace(D2.m_ed,">16" => "16","<12" => "12") #<-  simplify the education categories
D2.f_ed = replace(D2.f_ed,">16" => "16","<12" => "12")
m_ed = make_dummy(D2,:m_ed)
f_ed = make_dummy(D2,:f_ed)
D2[.!D2.mar_stat,f_ed] .= 0. #<- make zero by default. this won't work all the time
D2 = D2[.!ismissing.(D2.age),:]
D2.logwage_f = coalesce.(D2.logwage_f,0) #<- make these into zeros to avoid a problem with instruments
D2[!,:div] = .!D2.mar_stat
D2[!,:const] .= 1.
D2 = D2[D2.price_missing.==0,:]
D2 = subset(D2,:year => x->(x.==1997) .| (x.==2002)) #<- for now, limit to years 1997 and 2002
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

# by clustering, we add a variable :type to data frame D2
# call:
# type = make_dummy(D2,:type)


# ---- write the specification we want to use. This is the same as the original draft. add type[2:end] to vm and vg
spec = (vm = [:mar_stat;:div;m_ed[2:3];:age;:num_0_5],vf = [:const;f_ed[2:3];:age;:num_0_5],vθ = [:const,:mar_stat,:age,:num_0_5],vg = [:mar_stat;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

P = CESmod(spec)
x0 = update_inv(P)
x0[1:2] .= -2. #<- initial guess consistent with last time

# ordering of residuals: c/m,f/m,c/g,m/g,f/g
# -- make some price ratios to pass as instruments:
# -- put this in prep data or even in earlier R code
D2[!,:logprice_c_m] = D2.logprice_c .- D2.logwage_m
D2[!,:logprice_m_f] = D2.logwage_m .- D2.logwage_f
D2[!,:logprice_c_g] = D2.logprice_c .- D2.logprice_g
D2[!,:logprice_m_g] = D2.logwage_m .- D2.logprice_g
D2[!,:logprice_f_g] = D2.logwage_f .- D2.logprice_g


# -------- Specification (2)
# this version is the same as above but we are now overlaying single and married moments
function gmap_spec2(data,it,spec)
    n97 = length(spec.vg)+1
    n02 = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
    if data.year[it]==1997
        g_idx = 1:n97
        zlist =  [[spec.vg;:logprice_c_m]]
        r_idx = [1] 
    else
        g_idx = (n97+1):(n97+n02)
        zlist = [[spec.vm;:logprice_m_g],
        [spec.vf;:logprice_f_g],
        [spec.vg;:logprice_c_g],
        [spec.vg;:logprice_c_m]]
        r_idx = [4,5,3,1]
    end
    return g_idx,zlist,r_idx
end

n97 = length(spec.vg) + 1 
n02 = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2

nmom = n97+n02
N = nrow(D2)
W = I(nmom)

gfunc_spec2!(x,n,g,resids,data,gd,gmap_spec2,spec) = demand_moments_stacked2!(update(x,spec),n,g,resids,data,gd,gmap_spec2,spec)
@time gmm_criterion(x0,gfunc_spec2!,W,N,5,D2,gd,gmap_spec2,spec)
res,se = estimate_gmm_iterative(x0,gfunc_spec2!,5,W,N,5,D2,gd,gmap_spec2,spec)

# ------------ Specification (3): each child-year is one observation now
# there is a simpler vesion than this FYI (what is it?) have to review code to figure it out
function gmap_spec3(data,it,spec)
    nmom = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
    g_idx = 1:nmom
    zlist = [[spec.vm;:logprice_m_g],
    [spec.vf;:logprice_f_g],
    [spec.vg;:logprice_c_g],
    [spec.vg;:logprice_c_m]]
    r_idx = [4,5,3,1]
    return g_idx,zlist,r_idx
end
nmom = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
N = nrow(D2) #length(unique(D2.KID))
gd2 = groupby(D2,[:KID,:year])
W = I(nmom)

gfunc_spec3!(x,n,g,resids,data,gd,gmap_spec3,spec) = demand_moments_stacked2!(update(x,spec),n,g,resids,data,gd,gmap_spec3,spec)
@time gmm_criterion(x0,gfunc_spec3!,W,N,5,D2,gd2,gmap_spec3,spec)
res3,se3 = estimate_gmm_iterative(x0,gfunc_spec3!,5,W,N,5,D2,gd2,gmap_spec3,spec)


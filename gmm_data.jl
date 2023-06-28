using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
#include("relative_demand.jl")

# --------  read in the data:
# Step 1: create the data object
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))

# temporary: we need to fix this!!
#panel_data.mid[ismissing.(panel_data.mid)] .= 6024032

panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID, matchmissing = :notequal) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

include("temp_prep_data.jl")
include("specifications.jl")

struct child_data
    Xm::Matrix{Float64}
    Xf::Matrix{Float64}
    Xy::Matrix{Float64}
    Xθ::Matrix{Float64}
    year::Vector{Int64}
    age::Vector{Int64}
    # logicals:
    mar_stat::Vector{Bool}
    prices_observed::Vector{Bool}
    ind_not_sample::Vector{Bool}
    # prices 
    logprice_g::Vector{Float64}
    logprice_c::Vector{Float64}
    logwage_m::Vector{Float64}
    logwage_f::Vector{Float64}
    # inputs
    log_mtime::Vector{Float64}
    log_ftime::Vector{Float64}
    log_chcare::Vector{Float64}
    log_good::Vector{Float64}

    mtime_missing::Vector{Bool}
    ftime_missing::Vector{Bool}
    chcare_missing::Vector{Bool}
    goods_missing::Vector{Bool}

    # instruments:
    Z::Vector{Matrix{Float64}}

end

function child_data(data,spec)
    Xm = hcat([data[!,v] for v in spec_1.vm]...)'
    Xf = hcat([data[!,v] for v in spec_1.vf]...)'
    Xy = hcat([data[!,v] for v in spec_1.vy]...)'
    Xθ = hcat([data[!,v] for v in spec_1.vθ]...)'
    Z = []
    for zv in spec.zlist_97
        push!(Z,coalesce.(hcat([data[data.year.==1997,v] for v in zv]...)',0.))
    end
    for zv in spec.zlist_02
        push!(Z,coalesce.(hcat([data[data.year.==2002,v] for v in zv]...)',0.))
    end
    for zv in spec.zlist_07
        push!(Z,coalesce.(hcat([data[data.year.==2007,v] for v in zv]...)',0.))
    end
    return child_data(coalesce.(Xm,0.),
    coalesce.(Xf,0.),
    coalesce.(Xy,0.),
    coalesce.(Xθ,0.),
    data.year,
    data.age,
    coalesce.(data.mar_stat,false),
    data.prices_observed,
    data.ind_not_sample,
    coalesce.(data.logprice_g,0.),
    coalesce.(data.logprice_c,0.),
    coalesce.(data.logwage_m,0.),
    coalesce.(data.logwage_f,0.),
    coalesce.(data.log_mtime,0.),
    coalesce.(data.log_ftime,0.),
    coalesce.(data.log_chcare,0.),
    coalesce.(data.log_good,0.),
    ismissing.(data.log_mtime),
    ismissing.(data.log_ftime),
    ismissing.(data.log_chcare),
    ismissing.(data.log_good),
    Z)
end
include("other_functions.jl")

gfunc!(x,n,g,resids,data,spec) = demand_moments_stacked!(update(x,spec),n,g,resids,data)

N = length(unique(panel_data.kid))

W = I(nmom)
x0 = initial_guess(spec_1)
cd = child_data(panel_data,spec_1)
nmom = sum([size(z,1) for z in cd.Z])

gmm_criterion(x0,gfunc!,W,N,9,cd,spec_1)

res1 = estimate_gmm(x0,gfunc!,W,N,9,cd,spec_1)

include("relative_demand.jl")
gfunc2!(x,n,g,resids,data,spec) = demand_moments_stacked!(update2(x,spec),n,g,resids,data,spec)

gmm_criterion(x0,gfunc2!,W,N,9,panel_data,spec_1)


# p0 = CESmod(spec_1)
# @code_warntype demand_moments_stacked!(p0,1,g,R,cd)
# @code_warntype calc_demand_resids!(1,R,cd,p0)

# @code_warntype log_input_ratios(p0,cd,1)

# @code_warntype factor_shares(p0,cd,1,cd.mar_stat[1])

# @time demand_moments_stacked!(p0,1,g,R,cd)
# @time calc_demand_resids!(1,R,cd,p0)

# @time log_input_ratios(p0,cd,1)

# @time factor_shares(p0,cd,1,cd.mar_stat[1])

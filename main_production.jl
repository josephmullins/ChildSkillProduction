using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")
include("production.jl")
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data[!,:MID] = panel_data.mid

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
    return CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
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

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked!(update(x,spec),n,g,resids,data,spec)

# specification (1)
nmom = spec_1p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_1p)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_1p)
res2,se2 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_1p)

p = update(res2,spec_1p)
p_se = update(se2,spec_1p)

break

# specification (2)
nmom = spec_2p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_2p)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_2p)

g = zeros(nmom)
r = zeros(5)
@time gfunc!(x0,10,g,r,panel_data,spec_2p)

res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2p)

p2 = update(res3,spec_2p)
p2_se = update(se3,spec_2p)


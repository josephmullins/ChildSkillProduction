using CSV, DataFrames, Parameters, Random, Printf, LinearAlgebra, Statistics
using Optim
include("estimation_tools.jl")
include("relative_demand.jl")
include("production.jl")
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
# temporary:
panel_data.mid[ismissing.(panel_data.mid)] .= 6024032
panel_data[!,:MID] = panel_data.mid

wage_types = DataFrame(CSV.File("wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made


panel_data = panel_data[panel_data.year.<=2002,:] #<- for now, limit to years <=2002. We need to update code eventually.
include("temp_prep_data.jl")

# what's free and what isn't initially?

# first let's try making the factor shares completely free but elasticities the same
#---- write the update function:
# function update(x,spec)
#     ρ = x[1]
#     γ = x[2]
#     δ = x[3:4] #<- factor shares
#     nm = length(spec.vm)
#     βm = x[5:4+nm]
#     pos = 5+nm
#     nf = length(spec.vf)
#     βf = x[pos:pos+nf-1]
#     ng = length(spec.vg)
#     pos += nf
#     βg = x[pos:pos+ng-1]
#     pos += ng
#     nm = length(spec.vm)
#     P1 = CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
#     βm = x[pos:pos+nm-1]
#     pos += nm
#     nf = length(spec.vf)
#     βf = x[pos:pos+nf-1]
#     ng = length(spec.vg)
#     pos += nf
#     βg = x[pos:pos+ng-1]
#     pos += ng
#     nθ = length(spec.vθ)
#     βθ = x[pos:pos+nθ-1]
#     pos+= nθ
#     λ = x[pos]
#     P2 = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
#     return P1,P2
# end

# CAREFUL: we are pulling δ from the wrong spot here, not the correct inverse
# function update_inv(P1,P2)
#     @unpack ρ,γ,δ,βm,βf,βg = P1
#     x1 = [ρ;γ;δ;βm;βf;βg]
#     @unpack ρ,γ,δ,βm,βf,βg,βθ,λ = P2
#     return [x1;βm;βf;βg;βθ;λ]
# end
# try relaxing elasticities only
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    ρ2 = x[3]
    γ2 = x[4]
    δ = x[5:6] #<- factor shares
    nm = length(spec.vm)
    βm = x[7:6+nm]
    pos = 7+nm
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
    P1 = CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
    P2 = CESmod(ρ=ρ2,γ=γ2,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
    return P1,P2
end
function update_inv(P1,P2)
    @unpack ρ,γ,δ,βm,βf,βg = P1
    x1 = [ρ,γ] #;δ;βm;βf;βg]
    @unpack ρ,γ,δ,βm,βf,βg,βθ,λ = P2
    return [x1;ρ;γ;δ;βm;βf;βg;βθ;λ]
end


include("specifications.jl")

# function to get the initial guess
# function initial_guess(spec)
#     P = CESmod(spec)
#     x0 = update_inv(P,P)
#     x0[1:2] .= -2. #<- initial guess consistent with last time
#     x0[3:4] = [0.1,0.9] #<- initial guess for δ
#     return x0
# end
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P,P)
    x0[1:4] .= -2. #<- initial guess consistent with last time
    x0[5:6] = [0.1,0.9] #<- initial guess for δ
    return x0
end


N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked!(update(x,spec)...,n,g,resids,data,spec)


# specification (1)
nmom = spec_1p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_1p)


# specification (2)
nmom = spec_2p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_2p)

@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_2p)

res3,se3 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2p)

p2 = update(res3,spec_2p)
p2_se = update(se3,spec_2p)

# specification (3)
nmom = spec_3p.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_3p)
@time gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_3p)

res4,se4 = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_3p)

p3 = update(res4,spec_3p)
p3_se = update(se4,spec_3p)


[p.δ p_se.δ p2.δ p2_se.δ p3.δ p3_se.δ]

display([p.βm p_se.βm spec_1p.vm; "-" "-" "-"; p2.βm p2_se.βm spec_2p.vm; "-" "-" "-" ; p3.βm p3_se.βm spec_3p.vm])

display([p.βθ p_se.βθ spec_1p.vθ; "-" "-" "-"; p2.βθ p2_se.βθ spec_2p.vθ;  "-" "-" "-"; p3.βθ p3_se.βθ spec_3p.vθ])



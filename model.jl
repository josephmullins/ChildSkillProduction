using Parameters
# -- this script contains:
# (1) definition of the model parameters and update functions given a vector of parameters
# (2) functions to calculate relative demand
# (3) functions to calculate price indices in both the case where production parameters are accurately perceived and the case when this is relaxed (i.e. parameters that determine demand are different from those determining outcomes)

# ------- Definition of the model parameters object, with update functions
@with_kw struct CESmod{R}
    # elasticity parameters
    ρ::R = R(-2.)
    γ::R = R(-2.)
    δ::Vector{R} = zeros(R,2) #R[0.05,0.95]
    # coefficient vectors for factor shares
    βm::Vector{R} = zeros(R,2)
    βf::Vector{R} = zeros(R,2)
    βy::Vector{R} = zeros(R,2)
    βθ::Vector{R} = zeros(R,2)
    λ::R = R(1.)
    κ::R = R(0.)
end
# - a simple constructor:
function CESmod(spec)
    return CESmod{Float64}(βm = zeros(length(spec.vm)),βf = zeros(length(spec.vf)),βy = zeros(length(spec.vy)),βθ = zeros(length(spec.vθ)))
end

# - function to update just demand parameters
function update_demand(x,spec)
    R = eltype(x)
    ρ = x[1]
    γ = x[2]
    nm = length(spec.vm)
    βm = x[3:2+nm]
    pos = 3+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ny = length(spec.vy)
    pos += nf
    βy = x[pos:pos+ny-1]
    return CESmod{R}(ρ=ρ,γ=γ,βm = βm,βf = βf,βy=βy)
end
function demand_inv(pars)
    @unpack ρ,γ,βm,βf,βy = pars
    return [ρ;γ;βm;βf;βy]
end
function demand_guess(spec)
    P = CESmod(spec)
    x0 = demand_inv(P)
    return x0
end

# - function to update all parameters (restricted case)
function update(x,spec,case="hybrid")
    R = eltype(x)
    ρ = x[1]
    γ = x[2]
    δ = x[3:4] #<- factor shares
    nm = length(spec.vm)
    βm = x[5:4+nm]
    pos = 5+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vy)
    pos += nf
    βy = x[pos:pos+ng-1]
    pos += ng
    nθ = length(spec.vθ)
    βθ = x[pos:pos+nθ-1]
    pos+= nθ
    λ = x[pos]
    if case=="hybrid"
        κ = x[pos+1]
    elseif case=="uc"
        κ = R(0.)
    else
        κ = R(1.)
    end
    P = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βy=βy,βθ=βθ,λ=λ,κ=κ)
    return P
end
function update_inv(pars,case="hybrid")
    @unpack ρ,γ,δ,βm,βf,βy,βθ,λ,κ = pars
    if case=="hybrid"
        return [ρ;γ;δ;βm;βf;βy;βθ;λ;κ]
    else
        return [ρ;γ;δ;βm;βf;βy;βθ;λ]
    end
end
# function to get the initial guess
function initial_guess(spec,case="hybrid")
    P = CESmod(spec)
    x0 = update_inv(P,case)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    x0[3:4] = [0.1,0.9] #<- initial guess for δ
    return x0
end

# update function in the unrestricted case
function update_relaxed(x,spec,unrestricted,case="hybrid")
    R = eltype(x)
    pos = 1 #<- tracks position in the vector of restriction indicators
    pos2 = 1 #<- tracks position in the vector of parameters
    ρ = x[1]
    pos2 += 1
    if unrestricted[1]
        ρ2 = x[2]
        pos2 += 1
    else
        ρ2 = ρ 
    end
    pos += 1
    
    γ = x[pos2]
    pos2 += 1
    if unrestricted[pos]
        γ2 = x[pos2]
        pos2 += 1
    else
        γ2 = γ
    end
    pos += 1
    
    # cobb-douglas factor shares
    δ = x[pos2:pos2+1] 
    pos2 += 2 
    

    np = length(spec.vm)
    βm = x[pos2:pos2+np-1]
    pos2 += np
    @views r = unrestricted[pos:pos+np-1]
    βm2 = copy(βm)
    nu = sum(r)
    βm2[r] = x[pos2:pos2+nu-1]
    pos2 += nu
    pos += np

    np = length(spec.vf)
    βf = x[pos2:pos2+np-1]
    pos2 += np
    @views r = unrestricted[pos:pos+np-1]
    βf2 = copy(βf)
    nu = sum(r)
    βf2[r] = x[pos2:pos2+nu-1]
    pos2 += nu
    pos += np

    np = length(spec.vy)
    βy = x[pos2:pos2+np-1]
    pos2 += np
    @views r = unrestricted[pos:pos+np-1]
    βy2 = copy(βy)
    nu = sum(r)
    βy2[r] = x[pos2:pos2+nu-1]
    pos2 += nu
    pos += np

    nθ = length(spec.vθ)
    βθ = x[pos2:pos2+nθ-1]
    pos2 += nθ
    λ = x[pos2]
    if case=="hybrid"
        κ = x[pos2+1]
    elseif case=="uc"
        κ = R(0.)
    else
        κ = R(1.)
    end

    P1 = CESmod{R}(ρ=ρ,γ=γ,βm = βm,βf = βf,βy=βy)
    P2 = CESmod{R}(ρ=ρ2,γ=γ2,δ = δ,βm = βm2,βf = βf2,βy=βy2,βθ=βθ,λ=λ,κ = κ)
    return P1,P2
end

# an inverse of the update function assumes a CESmod object that holds indicators of restrictions
function update_inv_relaxed(P1,P2,Pu,case="hybrid")
    x = [P1.ρ]
    if Pu.ρ
        push!(x,P2.ρ)
    end
    push!(x,P1.γ)
    if Pu.γ
        push!(x,P2.γ)
    end

    push!(x,P2.δ...)

    push!(x,P1.βm...)
    push!(x,P2.βm[Pu.βm]...)

    push!(x,P1.βf...)
    push!(x,P2.βf[Pu.βf]...)

    push!(x,P1.βy...)
    push!(x,P2.βy[Pu.βy]...)

    push!(x,P2.βθ...)
    push!(x,P2.λ)
    if case=="hybrid"
        push!(x,P2.κ)
    end
    return x
end

# ------------------------------------------------------ #

# --------- Functions to calculate relative demand

# return factor shares
function factor_shares(pars,data,it,mar_stat)
    @unpack βm,βf,βy = pars
    @views if mar_stat
        am = exp(dot(βm,data.Xm[:,it]))
        af = exp(dot(βf,data.Xf[:,it]))
        ay = exp(dot(βy,data.Xy[:,it]))
        denom_inner = am+af+1
        denom_outer = 1+ay
    else
        am = exp(dot(βm,data.Xm[:,it]))
        ay = exp(dot(βy,data.Xy[:,it]))
        denom_inner = 1+am
        denom_outer = 1+ay
        af = 0.
    end 
    return 1/denom_inner,am/denom_inner,af/denom_inner,ay /denom_outer
end

# -------- calculate log input ratios
function log_input_ratios(ρ,γ,ay,am,af,ag,logwage_m,logwage_f,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕf = 1/(ρ-1)*log(ag/af) + 1/(ρ-1)*(logwage_f - logprice_g)
    lϕc = 1/(γ-1)*log((1-ay)*ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag)^(γ/ρ)*(1-ay) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm) + exp(logwage_f)*exp(lϕf))
    return lϕm,lϕf,lϕc,log_price_index,Φg
end
# same function as above but for singles (no wage of father to pass)
function log_input_ratios(ρ,γ,ay,am,ag,logwage_m,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕc = 1/(γ-1)*log((1-ay)*ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + ag)^(γ/ρ)*(1-ay) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm))
    return lϕm,lϕc,log_price_index,Φg
end
function log_input_ratios(pars,data,it)
    @unpack ρ,γ = pars
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(pars,data,it,true) #<- returns the factor shares.
        lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,ay,am,af,ag,data.logwage_m[it],data.logwage_f[it],data.logprice_g[it],data.logprice_c[it])
    else
        ag,am,af,ay = factor_shares(pars,data,it,false)
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,ay,am,ag,data.logwage_m[it],data.logprice_g[it],data.logprice_c[it])
        lϕf = 0.
    end
    return lϕm,lϕf,lϕc,log_price_index,Φg
end
# ---------------------------------------------------------------- #

# --------- calculate price indices ------------------------ #

# strict case:
function calc_Φ_m(pars,data,it)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}, X_{t} = τ_{m,t} / (Φg * ϕ_{m})
    lΦg = log(Φg)
    lΦm = lΦg + lϕm #<- now we have that X_{t} = τ_{m,t} / Φm
    lΦf = lΦg + lϕf
    lΦc = lΦg + lϕc
    return lΦm,lΦf,lΦg,lΦc,log_price_index
end

# unrestricted case:
function calc_Φ_m(pars1,pars2,data,it)
    
    lϕm,lϕf,lϕc,lp,Φg = log_input_ratios(pars1,data,it) #relative input ratios from perceived parameters
    # lϕm is log(mother's time / goods)
    # so exp(-lϕm) is goods / mother's time
    # lϕc is log(childcare / goods) so 
    # lϕc - lϕm is log(childcare / mother's time)
    
    @unpack ρ,γ = pars2 #implied 

    #g_t = Φg*τ_{m,t}, τ_{m,t} = g_t/Φg
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(pars2,data,it,true)
        Φm = ((am + af*exp(lϕf - lϕm)^ρ + ag*exp(-lϕm)^ρ)^(γ/ρ)*(1-ay)+ay*exp(lϕc - lϕm)^γ)^(1/γ)
        lΦm=-log(Φm) 
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) + exp(data.logprice_c[it])*exp(lϕc) + exp(data.logwage_m[it])*exp(lϕm) + exp(data.logwage_f[it])*exp(lϕf))
    else
        ag,am,af,ay = factor_shares(pars2,data,it,false)
        Φm = ((am+ag*exp(-lϕm)^ρ)^(γ/ρ)*(1-ay)+ay*exp(lϕc - lϕm)^γ)^(1/γ) #here I have X_{t}/τ_{m,t} as  composite investment relative to mothers time = Φm
        lΦm=-log(Φm) 
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) + exp(data.logprice_c[it])*exp(lϕc) + exp(data.logwage_m[it])*exp(lϕm))
    end
    # if X = τm / Φm, and τf / τm = ϕf / ϕm, then: X = (ϕm/ϕf * τm) / Φ_{m}
    # so: Φ_{f} = Φ_{m} * ϕ_{f} / ϕ_{m}
    lΦf = lΦm + lϕf - lϕm # 
    lΦc = lΦm + lϕc - lϕm
    return lΦm,lΦf,lΦg,lΦc,log_price_index
end
# -------------------------------------------------------------- #

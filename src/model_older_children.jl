# this file contains equivalent functions to those found in model.jl but without childcare. 
# We use these functions to estimate the parameters on older children.

using Parameters
# -- this script contains:
# (1) definition of the model parameters and update functions given a vector of parameters
# (2) functions to calculate relative demand
# (3) functions to calculate price indices in both the case where production parameters are accurately perceived and the case when this is relaxed (i.e. parameters that determine demand are different from those determining outcomes)

# ------- Definition of the model parameters object, with update functions

# - function to update just demand parameters
function update_demand_older(x,spec)
    R = eltype(x)
    ρ = x[1]
    nm = length(spec.vm)
    βm = x[2:1+nm]
    pos = 2+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    pos += nf
    return CESmod{R}(ρ=ρ,γ=0,βm = βm,βf = βf)
end
function demand_inv_older(pars)
    @unpack ρ,βm,βf = pars
    return [ρ;βm;βf]
end
function demand_guess_older(spec)
    P = CESmod(spec)
    x0 = demand_inv_older(P)
    return x0
end

# - function to update all parameters (restricted case)
function update_older(x,spec,case="hybrid")
    R = eltype(x)
    ρ = x[1]
    δ = x[2:3] #<- factor shares
    nm = length(spec.vm)
    βm = x[4:3+nm]
    pos = 4+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    pos += nf
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
    P = CESmod{R}(ρ=ρ,γ=0,δ = δ,βm = βm,βf = βf,βθ=βθ,λ=λ,κ=κ)
    return P
end
function update_inv_older(pars,case="hybrid")
    @unpack ρ,δ,βm,βf,βθ,λ,κ = pars
    if case=="hybrid"
        return [ρ;δ;βm;βf;βθ;λ;κ]
    else
        return [ρ;δ;βm;βf;βθ;λ]
    end
end
# function to get the initial guess
function initial_guess_older(spec,case="hybrid")
    P = CESmod(spec)
    x0 = update_inv_older(P,case)
    x0[1] = -2. #<- initial guess consistent with last time
    x0[2:3] = [0.1,0.9] #<- initial guess for δ
    return x0
end

# update function in the unrestricted case
function update_relaxed_older(x,spec,unrestricted,case="hybrid")
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

    P1 = CESmod{R}(ρ=ρ,βm = βm,βf = βf)
    P2 = CESmod{R}(ρ=ρ2,δ = δ,βm = βm2,βf = βf2,βθ=βθ,λ=λ,κ = κ)
    return P1,P2
end

# an inverse of the update function assumes a CESmod object that holds indicators of restrictions
function update_inv_relaxed_older(P1,P2,Pu,case="hybrid")
    x = [P1.ρ]
    if Pu.ρ
        push!(x,P2.ρ)
    end

    push!(x,P2.δ...)

    push!(x,P1.βm...)
    push!(x,P2.βm[Pu.βm]...)

    push!(x,P1.βf...)
    push!(x,P2.βf[Pu.βf]...)

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
function factor_shares_older(pars,data,it,mar_stat)
    @unpack βm,βf = pars
    @views if mar_stat
        am = exp(dot(βm,data.Xm[:,it]))
        af = exp(dot(βf,data.Xf[:,it]))
        denom = am+af+1
    else
        am = exp(dot(βm,data.Xm[:,it]))
        denom = 1+am
        af = 0.
    end 
    return 1/denom,am/denom,af/denom
end

# -------- calculate log input ratios
function log_input_ratios_older(ρ,am,af,ag,logwage_m,logwage_f,logprice_g)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕf = 1/(ρ-1)*log(ag/af) + 1/(ρ-1)*(logwage_f - logprice_g)
    Φg = (am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag)^(-1/ρ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logwage_m)*exp(lϕm) + exp(logwage_f)*exp(lϕf))
    return lϕm,lϕf,log_price_index,Φg
end
# same function as above but for singles (no wage of father to pass)
function log_input_ratios_older(ρ,am,ag,logwage_m,logprice_g)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    Φg = (am*exp(ρ*lϕm) + ag)^(-1/ρ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logwage_m)*exp(lϕm))
    return lϕm,log_price_index,Φg
end
function log_input_ratios_older(pars,data,it)
    @unpack ρ,γ = pars
    if data.mar_stat[it]
        ag,am,af = factor_shares_older(pars,data,it,true) #<- returns the factor shares.
        lϕm,lϕf,log_price_index,Φg = log_input_ratios_older(ρ,am,af,ag,data.logwage_m[it],data.logwage_f[it],data.logprice_g[it])
    else
        ag,am,af = factor_shares_older(pars,data,it,false)
        lϕm,log_price_index,Φg = log_input_ratios_older(ρ,am,ag,data.logwage_m[it],data.logprice_g[it])
        lϕf = 0.
    end
    return lϕm,lϕf,log_price_index,Φg
end
# ---------------------------------------------------------------- #

# --------- calculate price indices ------------------------ #

# strict case:
function calc_Φ_m_older(pars,data,it)
    lϕm,lϕf,log_price_index,Φg = log_input_ratios_older(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}, X_{t} = τ_{m,t} / (Φg * ϕ_{m})
    lΦg = log(Φg)
    lΦm = lΦg + lϕm #<- now we have that X_{t} = τ_{m,t} / Φm
    lΦf = lΦg + lϕf
    return lΦm,lΦf,lΦg,log_price_index
end

# unrestricted case:
function calc_Φ_m_older(pars1,pars2,data,it)
    
    lϕm,lϕf,lp,Φg = log_input_ratios_older(pars1,data,it) #relative input ratios from perceived parameters
    # lϕm is log(mother's time / goods)
    # so exp(-lϕm) is goods / mother's time
    
    @unpack ρ = pars2 #implied 

    #g_t = Φg*τ_{m,t}, τ_{m,t} = g_t/Φg
    if data.mar_stat[it]
        ag,am,af = factor_shares_older(pars2,data,it,true)
        Φm = (am + af*exp(lϕf - lϕm)^ρ + ag*exp(-lϕm)^ρ)^(1/ρ)
        lΦm=-log(Φm)
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) +  exp(data.logwage_m[it])*exp(lϕm) + exp(data.logwage_f[it])*exp(lϕf))
    else
        ag,am,af = factor_shares_older(pars2,data,it,false)
        Φm = (am+ag*exp(-lϕm)^ρ)^(1/ρ) #here I have X_{t}/τ_{m,t} as  composite investment relative to mothers time = Φm
        lΦm=-log(Φm) 
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) +  exp(data.logwage_m[it])*exp(lϕm))
    end
    # if X = τm / Φm, and τf / τm = ϕf / ϕm, then: X = (ϕm/ϕf * τm) / Φ_{m}
    # so: Φ_{f} = Φ_{m} * ϕ_{f} / ϕ_{m}
    lΦf = lΦm + lϕf - lϕm # 
    return lΦm,lΦf,lΦg,log_price_index
end
# -------------------------------------------------------------- #
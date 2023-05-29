# this script will use functions in relative_demand.jl

# below: two versions of the functions that take either one or two parameter objects

function calc_production_resids!(n,R,data,pars1,pars2,savings)
    it97 = (n-1)*6 + 1
    it02 = n*6
    # this function call is assumed to return a coefficient lΦm where $X_{it} = τ_{m,it} / exp(lΦm)
    lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,pars2,data,it97)
    #Ψ0 = pars2.δ[1]*pars2.δ[2]^4*lX97
    Ψ0 = 0
    coeff_X = pars2.δ[1]*pars2.δ[2]^4
    for t=1:4
        lΦm,lΦf,lΦg,lΦc,log_price_index = calc_Φ_m(pars1,pars2,data,it97+t)
        #lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars1,data,it97+t)
        if savings
            coeff_X += pars2.δ[1]*pars2.δ[2]^(4-t)
            Ψ0 += pars2.δ[1]*pars2.δ[2]^(4-t)*(log_price_97 - log_price_index)
        else
            Ψ0 += pars2.δ[1]*pars2.δ[2]^(4-t)*(data.log_total_income[it97+t] - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    Ψ0 += linear_combination(pars2.βθ,pars2.spec.vθ,data,it97)
    r1 = data.AP[it02] / pars2.λ - Ψ0 - pars2.δ[2]^5 * data.AP[it97] / pars2.λ
    r2 = data.LW[it02] - Ψ0 - pars2.δ[2]^5 * data.LW[it97]
    if data.mtime_valid[it97]
        lX97 = data.log_mtime[it97] - lΦm #<- investment proxy using time investment
        R[1] = r1 - coeff_X*lX97
        R[2] = r2 - coeff_X*lX97
    end
    if data.ftime_valid[it97]
        # need to write childcare input as a function of other stuff now too
        #lX97 = data.log_chcare[it97] - lΦc - log_price_97
        lX97 = data.log_ftime[it97] - lΦf
        R[3] = r1 - coeff_X*lX97
        R[4] = r2 - coeff_X*lX97
    end
    if data.chcare_valid[it97]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_chcare[it97] - lΦc - log_price_97
        #lX97 = data.log_ftime[it97] - lΦf
        R[5] = r1 - coeff_X*lX97
        R[6] = r2 - coeff_X*lX97
    end

    R[7] = (data.AP[it02] - pars2.λ*data.LW[it02])*data.LW[it97]
    R[8] = (data.AP[it02]*data.AP[it97] - pars2.λ^2*data.LW[it02]*data.LW[it97])
end

# version with one parameter instead of two
function calc_production_resids!(n,R,data,pars1,savings)
    it97 = (n-1)*6 + 1
    it02 = n*6
    # this function call is assumed to return a coefficient lΦm where $X_{it} = τ_{m,it} / exp(lΦm)
    lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,data,it97)
    Ψ0 = 0
    coeff_X = pars1.δ[1]*pars1.δ[2]^4
    for t=1:4
        #lΦ,log_price_index = calc_Φ_m(pars1,pars1,data,it97+t)
        lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars1,data,it97+t)
        if savings
            coeff_X += pars1.δ[1]*pars1.δ[2]^(4-t)
            Ψ0 += pars1.δ[1]*pars1.δ[2]^(4-t)*(log_price_97 - log_price_index)
        else
            Ψ0 += pars1.δ[1]*pars1.δ[2]^(4-t)*(data.log_total_income[it97+t] - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    Ψ0 += linear_combination(pars1.βθ,pars1.spec.vθ,data,it97)
    r1 = data.AP[it02] / pars1.λ - Ψ0 - pars1.δ[2]^5 * data.AP[it97] / pars1.λ
    r2 = data.LW[it02] - Ψ0 - pars1.δ[2]^5 * data.LW[it97]
    if data.mtime_valid[it97]
        lX97 = data.log_mtime[it97] - lΦm #<- investment proxy using time investment
        R[1] = r1 - coeff_X*lX97
        R[2] = r2 - coeff_X*lX97
    end
    if data.ftime_valid[it97]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_ftime[it97] - lΦf
        R[3] = r1 - coeff_X*lX97
        R[4] = r2 - coeff_X*lX97
    end
    if data.chcare_valid[it97]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_chcare[it97] - lΦc - log_price_97
        #lX97 = data.log_ftime[it97] - lΦf
        R[5] = r1 - coeff_X*lX97
        R[6] = r2 - coeff_X*lX97
    end
    R[7] = (data.AP[it02] - pars1.λ*data.LW[it02])*data.LW[it97]
    R[8] = (data.AP[it02]*data.AP[it97] - pars1.λ^2*data.LW[it02]*data.LW[it97])
end


# this function creates a stacked vector of moment conditions from a vector of residuals
function production_demand_moments_stacked!(pars1,pars2,n,g,R,data,spec,savings=true)
    # first do relative demand moments
    demand_moments_stacked!(pars1,n,g,R,data,spec)

    production_moments_stacked!(pars1,pars2,n,g,R,data,spec,savings)

end
# version with one parameter instead of two
function production_demand_moments_stacked2!(pars1,n,g,R,data,spec,savings=true)
    # first do relative demand moments
    demand_moments_stacked!(pars1,n,g,R,data,spec)

    production_moments_stacked!(pars1,n,g,R,data,spec,savings)

end

# this function requires the following elements in the specification, spec:
# - z_list_prod_t: a collection of time indices that indicate which time periods each set of instruments come from
# - z_list_prod: a collection of collections (of collections) of instrument names for each time period in z_list_prod_t for each residual
# - g_idx_prod: a collection of unit ranges that indicate where stacked_moments should write to for each t
# example: suppose we want to use AP in 97 and tau_m in 02 for R[1] and LW in 97 and tau_m in 02 for R[2], and just a constant for R[3] and R[4], then we would have:
 # zlist_prod_t = [0,5]
 # zlist_prod = ([[:AP],[:LW]],[[:tau_m],[:tau_m]]) # let's check this
function production_moments_stacked!(pars1,pars2,n,g,R,data,spec,savings=true)
    R[:] .= 0.
    it97 = (n-1)*6+1
    it02 = it97+5
    if data.all_prices[it97] && data.mtime_valid[it97] && data.mtime_valid[it02] && (data.age[it97]<=12)
        calc_production_resids!(n,R,data,pars1,pars2,savings)
        #resids = view(R,1:6) #<- is this necessary? I don't think so.
        for j in eachindex(spec.zlist_prod)
            g_n = view(g,spec.g_idx_prod[j])
            it = it97 + spec.zlist_prod_t[j]
            stack_moments!(g_n,R,data,spec.zlist_prod[j],it)
        end
    end
end
# version with one parameter instead of two
function production_moments_stacked!(pars1,n,g,R,data,spec,savings=true)
    R[:] .= 0.
    it97 = (n-1)*6+1
    it02 = it97+5
    if data.all_prices[it97] && data.mtime_valid[it97] && data.mtime_valid[it02] && (data.age[it97]<=12)
        calc_production_resids!(n,R,data,pars1,savings)
        #resids = view(R,1:6)
        for j in eachindex(spec.zlist_prod)
            g_n = view(g,spec.g_idx_prod[j])
            it = it97 + spec.zlist_prod_t[j]
            stack_moments!(g_n,R,data,spec.zlist_prod[j],it)
        end
    end
end

# need multiple instruments? not sure?
function calc_Φ_m(pars,data,it)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}, X_{t} = τ_{m,t} / (Φg * ϕ_{m})
    lΦg = log(Φg)
    lΦm = lΦg + lϕm #<- now we have that X_{t} = τ_{m,t} / Φm
    lΦf = lΦg + lϕf
    lΦc = lΦg + lϕc
    return lΦm,lΦf,lΦg,lΦc,log_price_index
end

# that takes relative input ratios (log_input_ratios) using pars1 #<- perceived pars
# and calculates *actual* composite investment (X) relative to mothers time (using pars2, implied input ratios)
# notes: factor_shares will return factor shares (i.e. evalutes the function a_m(ϕ_{m},Z_{n,t} ... see relative_demand.jl->log_input_ratios for usage)
# notes: log_input_ratios will return input ratios relative to goods
# notes: results_new.pdf describes how to calculate X

# function to return lΦm where X_{i,t} = τ_{m} / exp(lΦm)
function calc_Φ_m(pars1,pars2,data,it)
    
    lϕm,lϕf,lϕc,lp,Φg = log_input_ratios(pars1,data,it) #relative input ratios from perceived parameters
    # lϕm is log(mother's time / goods)
    # so exp(-lϕm) is goods / mother's time
    # lϕc is log(childcare / goods) so 
    # lϕc - lϕm is log(childcare / mother's time)
    
    @unpack ρ,γ = pars2 #implied 

    #g_t = Φg*τ_{m,t}, τ_{m,t} = g_t/Φg
    #not sure I'm interpreting these equations correctly/might be combining equations I'm not supposed to be
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(pars2,data,it,true)
        Φm = ((am + af*exp(lϕf - lϕm)^ρ + ag*exp(-lϕm)^ρ)^(γ/ρ)*(1-ay)+ay*exp(lϕc - lϕm)^γ)^(1/γ)
        lΦm=-log(Φm) 
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) + exp(data.logprice_c[it])*exp(lϕc) + exp(data.logwage_m[it])*exp(lϕm))
    else
        ag,am,ay = factor_shares(pars2,data,it,false)
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


# update function for just demand parameters (shouldn't go here!!)
function update_demand(x,spec)
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
function demand_inv(pars)
    @unpack ρ,γ,βm,βf,βg = pars
    return [ρ;γ;βm;βf;βg]
end

# attempting to write an update function with some restrictions and some not.
function update(x,spec,unrestricted)
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

    np = length(spec.vg)
    βg = x[pos2:pos2+np-1]
    pos2 += np
    @views r = unrestricted[pos:pos+np-1]
    βg2 = copy(βg)
    nu = sum(r)
    βg2[r] = x[pos2:pos2+nu-1]
    pos2 += nu
    pos += np

    nθ = length(spec.vθ)
    βθ = x[pos2:pos2+nθ-1]
    pos2 += nθ
    λ = x[pos2]
    P1 = CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
    P2 = CESmod(ρ=ρ2,γ=γ2,δ = δ,βm = βm2,βf = βf2,βg=βg2,βθ=βθ,λ=λ,spec=spec)
    return P1,P2
end

# this function assumes a CESmod object that holds indicators of restrictions
function update_inv(P1,P2,Pu)
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

    push!(x,P1.βg...)
    push!(x,P2.βg[Pu.βg]...)

    push!(x,P2.βθ...)
    push!(x,P2.λ)
    return x
end


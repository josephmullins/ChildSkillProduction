# content:
# (1) functions to calculate residuals in relative demand
# (2) functions to construct the vector of demand moments
# (3) functions to calculate production residuals in the restricted and unrestricted case
# (4) functions to construct the vector of demand and production moments

# ------ demand residuals

function calc_demand_resids!(it,R,data,pars)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<- does this factor in missing data?
    # in 97: c/m (no goods available in 97)
    # no c/f? or f / m?
    if data.year[it]==1997
        if !data.mtime_missing[it]
            if !data.chcare_missing[it]
                R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
        end
    # recall: 02: use c/m,f/m,c/g,m/g,f/g
    elseif data.year[it]==2002 || data.year[it]==2007
        if !data.mtime_missing[it]
            if !data.chcare_missing[it]
                R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
        end
        if !data.goods_missing[it]
            if !data.chcare_missing[it]
                R[2] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
            end
            if !data.mtime_missing[it] 
                R[3] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
            end
            if !data.ftime_missing[it]
                R[4] = data.log_ftime[it] - data.log_good[it] - lϕf + data.logprice_g[it]
            end
        end
    end
end

function demand_residuals_all!(R,pars,n,data)
    # --- 1997 residuals
    it = (n-1)*11 + 1
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        calc_demand_resids!(it,R,data,pars)
    end

    # --- 2002 residuals
    it = (n-1)*11 + 6 
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        resids = view(R,2:5)
        calc_demand_resids!(it,resids,data,pars)
    end

    # --- 2007 residuals
    it = n*11
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        resids = view(R,6:9)
        calc_demand_resids!(it,resids,data,pars)
    end 
end
# ------------------------------------------------------- #
# demand moments:
function demand_moments_stacked!(pars,n,g,R,data)
    # assume a balanced panel of observations
    fill!(R,0.)
    demand_residuals_all!(R,pars,n,data)

    stack_moments!(g,R,data,n)
end
# --------------------------------------------------------- #
# production residuals: relaxed
function calc_production_resids!(it0,R,data,pars1,pars2,savings)
    # it0 indicates the period in which skills are initially measured
    # it5 is the period in whichs skills are next measured:
    it5 = it0 + 5
    # this function call is assumed to return a coefficient lΦm where $X_{it} = τ_{m,it} / exp(lΦm)
    lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,pars2,data,it0)
    #Ψ0 = pars2.δ[1]*pars2.δ[2]^4*lX97
    Ψ0 = 0
    coeff_X = pars2.δ[1]*pars2.δ[2]^4
    for t=1:4
        lΦm,lΦf,lΦg,lΦc,log_price_index = calc_Φ_m(pars1,pars2,data,it0+t)
        #lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars1,data,it0+t)
        if savings
            coeff_X += pars2.δ[1]*pars2.δ[2]^(4-t)
            Ψ0 += pars2.δ[1]*pars2.δ[2]^(4-t)*(log_price_97 - log_price_index)
        else
            Ψ0 += pars2.δ[1]*pars2.δ[2]^(4-t)*(data.log_total_income[it0+t] - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    @views Ψ0 += dot(pars2.βθ,data.Xθ[:,it0])
    r1 = data.AP[it5] / pars2.λ - Ψ0 - pars2.δ[2]^5 * data.AP[it0] / pars2.λ
    r2 = data.LW[it5] - Ψ0 - pars2.δ[2]^5 * data.LW[it0]
    if !data.mtime_missing[it0]
        lX97 = data.log_mtime[it0] - lΦm #<- investment proxy using time investment
        if !data.AP_missing[it5] && !data.AP_missing[it0]
            R[1] = r1 - coeff_X*lX97
        end
        if !data.LW_missing[it5] && !data.LW_missing[it0]
            R[2] = r2 - coeff_X*lX97
        end
    end
    if !data.ftime_missing[it0]
        # need to write childcare input as a function of other stuff now too
        #lX97 = data.log_chcare[it0] - lΦc - log_price_97
        lX97 = data.log_ftime[it0] - lΦf
        R[3] = r1 - coeff_X*lX97
        R[4] = r2 - coeff_X*lX97
    end
    if !data.chcare_missing[it0]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_chcare[it0] - lΦc - log_price_97
        #lX97 = data.log_ftime[it0] - lΦf
        R[5] = r1 - coeff_X*lX97
        R[6] = r2 - coeff_X*lX97
    end

    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0]
        R[7] = (data.AP[it5] - pars2.λ*data.LW[it5])*data.LW[it0]
    end
    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0] && !data.AP_missing[it0]
        R[8] = (data.AP[it5]*data.AP[it0] - pars2.λ^2*data.LW[it5]*data.LW[it0])
    end
end
function production_residuals_all!(R,pars1,pars2,n,data,savings)
    it97 = (n-1)*11+1
    it02 = it97+5
    it07 = it02+5
    if data.all_prices[it97] && !data.mtime_missing[it97] && !data.mtime_missing[it02] && (data.age[it97]<=12)
        @views calc_production_resids!(it97,R[1:8],data,pars1,pars2,savings)
        #resids = view(R,1:6) #<- is this necessary? I don't think so.
    end
    if data.all_prices[it02] && !data.mtime_missing[it02] && !data.mtime_missing[it07] && (data.age[it02]<=12)
        @views calc_production_resids!(it02,R[9:16],data,pars1,pars2,savings)
    end
end


# production residuals: restricted
function calc_production_resids!(it0,R,data,pars1,savings)
    # it0 indicates the period in which skills are initially measured
    # it5 is the period in whichs skills are next measured:
    it5 = it0 + 5

    # this function call is assumed to return a coefficient lΦm where $X_{it} = τ_{m,it} / exp(lΦm)
    lΦm,lΦf,lΦg,lΦc,log_price_97 = calc_Φ_m(pars1,data,it0)
    Ψ0 = 0
    coeff_X = pars1.δ[1]*pars1.δ[2]^4
    for t=1:4
        #lΦ,log_price_index = calc_Φ_m(pars1,pars1,data,it0+t)
        lΦm,lΦf,lΦg,lΦc,log_price_index = calc_Φ_m(pars1,data,it0+t)
        #lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars1,data,it0+t)
        if savings
            coeff_X += pars1.δ[1]*pars1.δ[2]^(4-t)
            Ψ0 += pars1.δ[1]*pars1.δ[2]^(4-t)*(log_price_97 - log_price_index)
        else
            Ψ0 += pars1.δ[1]*pars1.δ[2]^(4-t)*(data.log_total_income[it0+t] - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    @views Ψ0 += dot(pars1.βθ,data.Xθ[:,it0])
    r1 = data.AP[it5] / pars1.λ - Ψ0 - pars1.δ[2]^5 * data.AP[it0] / pars1.λ
    r2 = data.LW[it5] - Ψ0 - pars1.δ[2]^5 * data.LW[it0]
    if !data.mtime_missing[it0]
        lX97 = data.log_mtime[it0] - lΦm #<- investment proxy using time investment
        if !data.AP_missing[it5] && !data.AP_missing[it0]
            R[1] = r1 - coeff_X*lX97
        end
        if !data.LW_missing[it5] && !data.LW_missing[it0]
            R[2] = r2 - coeff_X*lX97
        end
    end
    if !data.ftime_missing[it0]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_ftime[it0] - lΦf
        R[3] = r1 - coeff_X*lX97
        R[4] = r2 - coeff_X*lX97
    end
    if !data.chcare_missing[it0]
        # need to write childcare input as a function of other stuff now too
        lX97 = data.log_chcare[it0] - lΦc - log_price_97
        #lX97 = data.log_ftime[it0] - lΦf
        R[5] = r1 - coeff_X*lX97
        R[6] = r2 - coeff_X*lX97
    end
    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0]
        R[7] = (data.AP[it5] - pars1.λ*data.LW[it5])*data.LW[it0]
    end
    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0] && !data.AP_missing[it0]
        R[8] = (data.AP[it5]*data.AP[it0] - pars1.λ^2*data.LW[it5]*data.LW[it0])
    end
end
function production_residuals_all!(R,pars1,n,data,savings)
    it97 = (n-1)*11+1
    it02 = it97+5
    it07 = it02+5
    if data.all_prices[it97] && !data.mtime_missing[it97] && !data.mtime_missing[it02] && (data.age[it97]<=12)
        @views calc_production_resids!(it97,R[1:8],data,pars1,savings)
    end
    if data.all_prices[it02] && !data.mtime_missing[it02] && !data.mtime_missing[it07] && (data.age[it02]<=12)
        @views calc_production_resids!(it02,R[9:16],data,pars1,savings)
    end
end

# -------------------------------------------------------------- #

# production and demand moments: relaxed
function production_demand_moments_relaxed!(pars1,pars2,n,g,R,data,savings=true)
    fill!(R,0.)
    # first do relative demand moments
    @views demand_residuals_all!(R[1:9],pars1,n,data)

    @views production_residuals_all!(R[10:end],pars1,pars2,n,data,savings)
    stack_moments!(g,R,data,n)
end
# production and demand moments: strict case
function production_demand_moments_strict!(pars1,n,g,R,data,savings=true)
    fill!(R,0.)
    @views demand_residuals_all!(R[1:9],pars1,n,data)

    @views production_residuals_all!(R[10:end],pars1,n,data,savings)

    stack_moments!(g,R,data,n)
end

# ----------------------------------------------------------- #
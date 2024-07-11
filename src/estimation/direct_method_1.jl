
function update_method_1(x,spec)
    dpars = update_demand_method_1(x,spec)
    pos = sum(length(x) for x in [spec.zc,spec.zf,spec.zg,spec.zτ])+20
    ppars = update(x[pos+1:end],spec,"uc")
    return dpars,ppars
end
function update_demand_method_1(x,spec)
    nc = length(spec.zc)
    nf = length(spec.zf)
    ng = length(spec.zg)
    nτ = length(spec.zτ)
    βc = x[1:nc]
    pos = nc
    βf = x[pos+1:pos+nf]
    pos += nf
    βg = x[pos+1:pos+ng]
    pos += ng
    βτ = x[pos+1:pos+nτ]
    pos += nτ
    γc = x[pos+1:pos+5]
    γf = x[pos+6:pos+10]
    γg = x[pos+11:pos+15]
    γτ = x[pos+16:pos+20]
    return (;βc,βf,βg,βτ,γc,γf,γg,γτ)
end


function log_input_ratios_method_1(pars,data,it)
    (;βc,βg,βf,γc,γg,γf) = pars
    @views lϕc = dot(βc,data.Zc[:,it]) + dot(γc,data.Ω[:,it])
    @views lϕg = dot(βg,data.Zg[:,it]) + dot(γg,data.Ω[:,it])
    @views lϕf = dot(βf,data.Zf[:,it]) + dot(γf,data.Ω[:,it])
    return lϕc,lϕg,lϕf
end

function calc_demand_resids_method_1!(it,R,data,pars)
    lϕc,lϕg,lϕf = log_input_ratios_method_1(pars,data,it)
    # in 97: c/m (no goods available in 97)
    # no c/f? or f / m?
    if data.year[it]==1997
        if !data.mtime_missing[it]
            if !data.chcare_missing[it]
                R[1] = data.log_chcare[it] - data.log_mtime[it] - lϕc - data.logprice_c[it]
            end
            if !data.ftime_missing[it]
                R[2] = data.log_ftime[it] - data.log_mtime[it] - lϕf
            end
            if !data.mtime_missing[it+5]
                lτ = log_τ_ratio(pars,data,it,5)
                R[3] = data.log_mtime[it+5] - data.log_mtime[it] - lτ
            end
        end
    # recall: 02: use c/m,f/m,g/m
    elseif data.year[it]==2002
        if !data.mtime_missing[it]
            if !data.chcare_missing[it]
                R[1] = data.log_chcare[it] - data.log_mtime[it] - lϕc - data.logprice_c[it]
            end
            if !data.goods_missing[it]
                R[2] = data.log_good[it] - data.log_mtime[it] - lϕg - data.logprice_g[it]
            end
            if !data.ftime_missing[it]
                R[3] = data.log_ftime[it] - data.log_mtime[it] - lϕf
            end
            if !data.mtime_missing[it+5]
                lτ = log_τ_ratio(pars,data,it,5)
                R[4] = data.log_mtime[it+5] - data.log_mtime[it] - lτ
            end
        end
    elseif data.year[it]==2007
        if !data.mtime_missing[it]
            if !data.chcare_missing[it]
                R[1] = data.log_chcare[it] - data.log_mtime[it] - lϕc - data.logprice_c[it]
            end
            if !data.ftime_missing[it]
                R[2] = data.log_ftime[it] - data.log_mtime[it] - lϕf
            end
        end
    end
end

function demand_residuals_all_method_1!(R,pars,n,data)
    # --- 1997 residuals
    it = (n-1)*11 + 1
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        calc_demand_resids_method_1!(it,R,data,pars)
    end

    # --- 2002 residuals
    it = (n-1)*11 + 6 
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        resids = view(R,4:7)
        calc_demand_resids_method_1!(it,resids,data,pars)
    end

    # --- 2007 residuals
    it = n*11
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        resids = view(R,8:9)
        calc_demand_resids_method_1!(it,resids,data,pars)
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

# returns production function assuming τₘ=1 and optimal input ratios
# dpars: parameters of demand system
# ppars: parameters of production
function log_f_method_1(dpars,ppars,data,it)
    lϕc,lϕg,lϕf = log_input_ratios_method_1(dpars,data,it)
    (;ρ,γ) = ppars
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(ppars,data,it,true)
        return ((am + af*exp(lϕf)^ρ + ag*exp(lϕg)^ρ)^(γ/ρ)*(1-ay) + ay*exp(lϕc)^γ)^(1/γ)
    else
        ag,am,af,ay = factor_shares(ppars,data,it,true)
        return ((am + ag*exp(lϕg)^ρ)^(γ/ρ)*(1-ay) + ay*exp(lϕc)^γ)^(1/γ)
    end
end

# calculates predicted log(τₜ₊ₛ/τₜ)
# have to fix this!
function log_τ_ratio(pars,data,it,s)
    (;βτ,γτ) = pars
    @views lτ = dot(βτ,data.Zτ[:,it+s] .- data.Zτ[:,it]) + dot(γτ,data.Ω[:,it+s] .- data.Ω[:,it])
    return lτ
end

function calc_production_resids_method_1!(it0,R,data,dpars,ppars)
    # it0 indicates the period in which skills are initially measured
    # it5 is the period in whichs skills are next measured:
    it5 = it0 + 5
    Ψ0 = 0
    coeff_X = ppars.δ[1]*ppars.δ[2]^4
    for t=1:4
        lτ = log_τ_ratio(dpars,data,it0,t) 
        lf = log_f_method_1(dpars,ppars,data,it0+t)  
        Ψ0 += ppars.δ[1]*ppars.δ[2]^(4-t)*(lτ + lf)
        coeff_X += ppars.δ[1]*ppars.δ[2]^(4-t)
    end
    @views Ψ0 += dot(ppars.βθ,data.Xθ[:,it0])
    r1 = data.AP[it5] / ppars.λ - Ψ0 - ppars.δ[2]^5 * data.LW[it0] #/ ppars.λ
    r2 = data.LW[it5] - Ψ0 - ppars.δ[2]^5 * data.LW[it0]
    lf = log_f_method_1(dpars,ppars,data,it0)
    if !data.mtime_missing[it0]
        lX97 = data.log_mtime[it0] + lf #<- investment proxy using time investment
        if !data.AP_missing[it5] && !data.AP_missing[it0]
            R[1] = r1 - coeff_X*lX97
        end
        if !data.LW_missing[it5] && !data.LW_missing[it0]
            R[2] = r2 - coeff_X*lX97
        end
    end

    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0]
        R[3] = (data.AP[it5] - ppars.λ*data.LW[it5])*data.LW[it0]
    end
    if !data.AP_missing[it5] && !data.LW_missing[it5] && !data.LW_missing[it0] && !data.AP_missing[it0]
        R[4] = (data.AP[it5]*data.AP[it0] - ppars.λ^2*data.LW[it5]*data.LW[it0])
    end
end
function production_residuals_all_method_1!(R,dpars,ppars,n,data)
    it97 = (n-1)*11+1
    it02 = it97+5
    it07 = it02+5
    if data.all_prices[it97] && !data.mtime_missing[it97] && !data.mtime_missing[it02] && (data.age[it97]<=12)
        @views calc_production_resids_method_1!(it97,R[1:4],data,dpars,ppars)
    end
    if data.all_prices[it02] && !data.mtime_missing[it02] && !data.mtime_missing[it07] && (data.age[it02]<=12)
        @views calc_production_resids_method_1!(it02,R[5:8],data,dpars,ppars)
    end
end

function demand_moments_method_1!(dpars,n,g,R,data)
    fill!(R,0.)
    demand_residuals_all_method_1!(R,dpars,n,data)
    stack_moments!(g,R,data,n)
end

function production_demand_moments_method_1!(dpars,ppars,n,g,R,data)
    fill!(R,0.)

    @views demand_residuals_all_method_1!(R[1:9],dpars,n,data)

    @views production_residuals_all_method_1!(R[10:end],dpars,ppars,n,data)
    stack_moments!(g,R,data,n)
end

function production_moments_method_1!(dpars,ppars,n,g,R,data)
    fill!(R,0.)
    @views production_residuals_all_method_1!(R,dpars,ppars,n,data)
    stack_moments!(g,R,data,n)
end
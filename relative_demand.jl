using Parameters, Distributions

# note:
# - the major steps here are calculating the residuals for relative demand given marital status and year, then interacting them with the appropriate instruments.
# - We want two things:
# (1) to be able to select/change the residuals and the instruments flexibly; and
# (2) to change how the moments are arranged flexibly
# - the function demand_moments_stacked calculates the moments by calling
# - the key is that this function uses an input function *gmap*: a function that tells demand_moments_stacked where to write the moments, which residuals to use, and which instruments to use for each residual given the year and marital status of the individual. The gmap function must be written specially for each specification


@with_kw struct CESmod2
    # elasticity parameters
    ρ = -1.5 #
    γ = -3. 
    δ = [0.05,0.95]
    # coefficient vectors for factor shares
    βm = zeros(2)
    βf = zeros(2)
    βy = zeros(2)
    βθ = zeros(2)
    λ = 1.
    spec = (vm = [:constant,:mar_stat],vf = [:constant],vθ = [:constant,:mar_stat],vg = [:constant,:mar_stat])
end

function CESmod2(spec)
    return CESmod2(βm = zeros(length(spec.vm)),βf = zeros(length(spec.vf)),βy = zeros(length(spec.vy)),βθ = zeros(length(spec.vθ)),spec=spec)
end


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

# a function that uses the functions above depending on marital status (and whether data are available?)
function log_input_ratios(pars,data,it)
    @unpack ρ,γ = pars
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(pars,data,it,true) #<- returns the factor shares.
        lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,ay,am,af,ag,data.logwage_m[it],data.logwage_f[it],data.logprice_g[it],data.logprice_c[it])
        return lϕm,lϕf,lϕc,log_price_index,Φg
    else
        ag,am,ay = factor_shares(pars,data,it,false)
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,ay,am,ag,data.logwage_m[it],data.logprice_g[it],data.logprice_c[it])
        lϕf = 0.
        return lϕm,lϕf,lϕc,log_price_index,Φg
    end
end

function factor_shares(pars,data,it,mar_stat)
    @unpack βm,βf,βy,spec = pars
    if mar_stat
        am = exp(linear_combination(βm,spec.vm,data,it))
        af = exp(linear_combination(βf,spec.vf,data,it))
        ay = exp(linear_combination(βy,spec.vy,data,it))
        denom_inner = am+af+1
        denom_outer = 1+ay
        return 1/denom_inner,am/denom_inner,af/denom_inner,ay /denom_outer
    else
        am = exp(linear_combination(βm,spec.vm,data,it))
        ay = exp(linear_combination(βy,spec.vy,data,it))
        denom_inner = 1+am
        denom_outer = 1+ay
        return 1/denom_inner,am/denom_inner,ay /denom_outer
    end 
end


# this function calculates residuals in relative demand 
# TODO: compare to version with relative expenditures
# NOTE: this does not check if data are available, that must be checked before calling this function
function calc_demand_resids!(it,R,data,pars)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<- does this factor in missing data?
    # in 97: c/m, f/m (no goods available in 97)
    if data.year[it]==1997
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
    # recall: 02: use c/m,f/m,c/g,m/g,f/g
    elseif data.year[it]==2002 || data.year[it]==2007
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
        if !ismissing(data.log_good[it])
            if !ismissing(data.log_chcare[it])
                R[3] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
            end
            if !ismissing(data.log_mtime[it]) #& !ismissing(data.logwage_m[it]) #<- include for missing wage?
                R[4] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
            end
            if !ismissing(data.log_ftime[it]) #& !ismissing(data.logwage_f[it]) #<- include for missing wage?
                R[5] = data.log_ftime[it] - data.log_good[it] - lϕf + data.logprice_g[it]
            end
        end
    end
end

# residual test
function residual_test(data,N,pars)
    R = zeros(N,5)
    r = zeros(5)
    for n=1:N
        it97 = (n-1)*11 + 1
        r[:] .= 0.
        if data.prices_observed[it97] && (data.age[it97]<=12) && (data.ind_not_sample[it97]==0)
            calc_demand_resids!(it97,r,data,pars)
            R[n,1] = r[1]
        end
        it02 = (n-1)*11 + 6
        r[:] .= 0.
        if data.prices_observed[it02] && (data.age[it02]<=12) && (data.ind_not_sample[it97]==0)
            calc_demand_resids!(it02,r,data,pars)
            R[n,2] = r[3] - r[4]
        end
    end
    test_stat = sqrt(N)*mean(R[:,1].*R[:,2]) / std(R[:,1])*std(R[:,2])
    pval = 2*cdf(Normal(),-abs(test_stat))
    return test_stat,pval
end


# this function creates a stacked vector of moment conditions from a vector of residuals
function demand_moments_stacked!(pars,n,g,R,data,spec)
    # assume a balanced panel of observations

    # --- 1997 relative demand moments
    it = (n-1)*11 + 1
    R[:] .= 0.
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        calc_demand_resids!(it,R,data,pars)
        resids = view(R,[1])
        g_it = view(g,spec.g_idx_97)
        stack_moments!(g_it,resids,data,spec.zlist_97,it)
    end

    # --- 2002 relative demand moments
    it = (n-1)*11 + 6 
    r_idx = [4,5,3,1]
    R[:] .= 0.
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        calc_demand_resids!(it,R,data,pars)
        resids = view(R,r_idx)
        g_it = view(g,spec.g_idx_02)
        stack_moments!(g_it,resids,data,spec.zlist_02,it)
    end

    # --- 2007 relative demand moments
    it = n*11
    r_idx = [4,5,3,1]
    R[:] .= 0.
    if data.prices_observed[it] && (data.age[it]<=12) && (data.ind_not_sample[it]==0)
        calc_demand_resids!(it,R,data,pars)
        resids = view(R,r_idx)
        g_it = view(g,spec.g_idx_07)
        stack_moments!(g_it,resids,data,spec.zlist_02,it)
    end
end

function update2(x,spec)
    ρ = x[1]
    γ = x[2]
    nm = length(spec.vm)
    βm = x[3:2+nm]
    pos = 3+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vy)
    pos += nf
    βy = x[pos:pos+ng-1]
    return CESmod2(ρ=ρ,γ=γ,βm = βm,βf = βf,βy=βy,spec=spec)
end
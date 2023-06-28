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
end

function CESmod(spec)
    return CESmod(βm = zeros(length(spec.vm)),βf = zeros(length(spec.vf)),βy = zeros(length(spec.vy)),βθ = zeros(length(spec.vθ)))
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

function calc_demand_resids!(it,R,data,pars)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<- does this factor in missing data?
    # in 97: c/m, f/m (no goods available in 97)
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

function demand_moments_stacked!(pars,n,g,R,data)
    # assume a balanced panel of observations

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

    stack_moments!(g,R,data,n)
end

#---- write the update function:
function update(x,spec)
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
function update_inv(pars)
    @unpack ρ,γ,βm,βf,βy = pars
    return [ρ;γ;βm;βf;βy]
end

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    return x0
end


# this function returns a vector of moment conditions
function gmm_inputs(par,data,Z_prod,Z_prodF,i,savings=true)
    # par: the guess of parameters
    # data: the data object
    # Z_prod and Z_prodF: instruments for the production moments
    # i: the observationl unit in the data (i.e. child)
    # savings: a boolean that indicates which assumption on credit markets is used (no binding constraints or no borrowing/saving)

    K0 = data.Ks[1]+data.Ks[2]+2*data.Ks[3]
    K1 = sum(data.Km[1:3])+2*data.Km[5]
    K2 = 2*size(Z_prod[1])[2]
    if data.mar[i]==1
        g0 = zeros(Real,K0)
        g1 = gmm_shares_married(par,data,i)
        #g2 = zeros(Real,K2)
        g2 = production_moments_married_time(par,data,Z_prod,Z_prodF,i,savings)
    else
        g0 = gmm_shares_single(par,data,i)
        g1 = zeros(Real,K1)
        g2 = production_moments_single_time(par,data,Z_prod,Z_prodF,i,savings)
        #g3 = zeros(Real,2*K2)
    end
    return [g0;g1;g2] #<- try this
end

function gmm_shares(par,data,i)
    K0 = data.Ks[1]+data.Ks[2]+2*data.Ks[3]
    K1 = sum(data.Km[1:3])+2*data.Km[5]
    if data.mar[i]==1
        g0 = zeros(Real,K0)
        g1 = gmm_shares_married(par,data,i)
    else
        g0 = gmm_shares_single(par,data,i)
        g1 = zeros(Real,K1)
    end
    return [g0;g1]
end


# moment functions for married couples:

function gmm_shares_married(par,data,i)
    # get ratio moments in 1997 (don't need expenditure for now, maybe later)
    g1 = ratio_moments_married_1997(par,data,i)
    # get ratio moments in 2002 (don't need expenditure for now, maybe later)
    g2 = ratio_moments_married(par,data,i,6)
    return [g1;g2]
end

function ratio_moments_married_1997(par,data,i)
    t = 1
    tt = (i-1)*6 + t
    if data.price_missing[tt]==0
        lϕm,lϕf,lϕc,log_cprice,Φg = get_ratios_married(par,data,i,t)
        if data.mtime_missing[tt]==0
            if data.chcare_missing[tt]==0
                m5 = data.log_chcare[tt] - data.log_mtime[tt] - (lϕc-lϕm) - (data.logprice_c[tt] - data.logwage_m[tt])
                m5 = m5*data.Zm[5][tt,:]
            else
                m5 = zeros(Real,data.Km[5])
            end
        else
            m5 = zeros(Real,data.Km[5])
        end
        return m5
    else
        return zeros(Real,data.Km[5])
    end
end

function ratio_moments_married(par,data,i,t)
    tt = (i-1)*6 + t
    if data.price_missing[tt]==0
        lϕm,lϕf,lϕc,log_cprice,Φg = get_ratios_married(par,data,i,t)
        if data.good_missing[tt]==0
            if data.mtime_missing[tt]==0
                m1 = data.log_mtime[tt]-data.log_good[tt] - lϕm - (data.logwage_m[tt] - data.logprice_g[tt])
                m1 = m1*data.Zm[1][tt,:]
            else
                m1 = zeros(Real,data.Km[1])
            end
            if data.ftime_missing[tt]==0
                m2 = data.log_ftime[tt] - data.log_good[tt] - lϕf - (data.logwage_f[tt] - data.logprice_g[tt])
                m2 = m2*data.Zm[2][tt,:]
            else
                m2 = zeros(Real,data.Km[2])
            end
            if data.chcare_missing[tt]==0
                m3 = data.log_chcare[tt] - data.log_good[tt] - lϕc - (data.logprice_c[tt] - data.logprice_g[tt])
                m3 = m3*data.Zm[3][tt,:]
            else
                m3 = zeros(Real,data.Km[3])
            end
        else
            m1 = zeros(Real,data.Km[1])
            m2 = zeros(Real,data.Km[2])
            m3 = zeros(Real,data.Km[3])
        end
        # do with respect to mother's time
        if data.mtime_missing[tt]==0
            if data.chcare_missing[tt]==0
                m5 = data.log_chcare[tt] - data.log_mtime[tt] - (lϕc-lϕm) - (data.logprice_c[tt] - data.logwage_m[tt])
                m5 = m5*data.Zm[5][tt,:]
            else
                m5 = zeros(Real,data.Km[5])
            end
        else
            m5 = zeros(Real,data.Km[5])
        end
        return [m1;m2;m3;m5]
    else
        return zeros(Real,data.Km[1]+data.Km[2]+data.Km[3]+data.Km[5])
    end
    # for now assume one set of instruments for each ratio
end

function get_ratios_married(par,data,i,t) #<- do separate time periods?
    tt = (i-1)*6 + t
    am = exp(dot(par.βm,data.Xm[tt,:]))
    af = exp(dot(par.βf,data.Xf[tt,:]))
    agm = exp(dot(par.βg,data.Xg[tt,:]))
    ay = par.ay
    lϕm = 1/(par.ρ-1)*log(agm*par.αm/am) + 1/(par.ρ-1)*(data.logwage_m[tt] - data.logprice_g[tt])
    lϕf = 1/(par.ρ-1)*log(agm*par.αf/af) + 1/(par.ρ-1)*(data.logwage_f[tt] - data.logprice_g[tt])
    lϕc = 1/(par.γ-1)*log(agm/ay) + (par.γ-par.ρ)/(par.ρ*(par.γ-1))*log(am*exp(par.ρ*lϕm) + af*exp(par.ρ*lϕf) + agm) + 1/(par.γ-1)*(data.logprice_c[tt]-data.logprice_g[tt])
    Φg = ((am*exp(par.ρ*lϕm) + af*exp(par.ρ*lϕf) + agm)^(par.γ/par.ρ) + ay*exp(lϕc)^par.γ)^(-1/par.γ)
    #lΦg = -(1/par.γ)*log((par.am*exp(par.ρ*lϕm) + par.af*exp(par.ρ*lϕf) + par.agm)^(par.γ/par.ρ) + par.ay*exp(lϕc)^par.γ)
    log_cprice = log(Φg) + log(data.price_g[tt] + data.price_c[tt]*exp(lϕc) + data.wage_m[tt]*par.αm*exp(lϕm) + data.wage_f[tt]*par.αf*exp(lϕf))
    return lϕm,lϕf,lϕc,log_cprice,Φg
end

function production_moments_married_time(par,data,Z_prod,Z_prodF,i,savings=true)
    if data.all_prices[i]==1
        log_cprice = zeros(Real,5)
        lϕm,lϕf,l3,log_cprice[1],Φg = get_ratios_married(par,data,i,1)
        #for t=2:5
        #    l1,l2,l3,log_cprice[t],Φg = get_ratios_married(par,data,i,t)
        #end
        lphim = lϕm + log(Φg)
        lphif = lϕf + log(Φg)
        g3 = production_moments_mtime(par.λ,par.βθ,par.δ,data,i,log_cprice,Z_prod,lphim,savings)
        if data.ltau_f97[i]!=0
            g4 = production_moments_ftime(par.λ,par.βθf,par.δ,data,i,log_cprice,Z_prodF,lphif,savings)
        else
            g4 = zeros(Real,2*size(Z_prodF[1])[2])
        end
        g5 = [(data.A02[i]-par.λ*data.L02[i])*data.L97[i],data.A02[i]*data.A97[i] - par.λ^2*data.L97[i]*data.L02[i]]
    else
        g3 = zeros(Real,2*size(Z_prod[1])[2])
        g4 = zeros(Real,2*size(Z_prodF[1])[2])
        g5 = zeros(Real,2)
    end
    return [g3;g4;g5]
end

function production_moments_mtime(λ,θ,δ,data,i,log_cprice,Z_prod,lphi,savings=true)
    a = data.age[i]
    tt0 = (i-1)*6 + 1
    K = dot(θ,data.Xθ[i,:])
    K2 = dot(θ,data.Xθ[i,:])
    K += δ[2]^4*δ[1]*(data.ltau_m97[i] - lphi)
    K2 += δ[2]^4*δ[1]*(data.ltau_m97[i] - lphi)
    if savings
        for t=2:5
            tt = (i-1)*6 + t
            K += δ[1]*δ[2]^(5-t)*(data.ltau_m97[i] - lphi + log_cprice[t]-log_cprice[1])
        end
    else
        for t=2:5
            tt = (i-1)*6 + t
            K += δ[1]*δ[2]^(5-t)*(data.lY[tt] - log_cprice[t])
        end
    end
    g1 = (data.A02[i]/λ - K - δ[2]^5*data.A97[i]/λ)*Z_prod[1][i,:]
    g2 = (data.L02[i] - K - δ[2]^5*data.L97[i])*Z_prod[2][i,:]
    return [g1;g2]
end

function production_moments_ftime(λ,θ,δ,data,i,log_cprice,Z_prod,lphi,savings=true)
    a = data.age[i]
    tt0 = (i-1)*6 + 1
    K = dot(θ,data.Xθf[i,:])
    K2 = dot(θ,data.Xθf[i,:])
    K += δ[2]^4*δ[1]*(data.ltau_f97[i] - lphi)
    K2 += δ[2]^4*δ[1]*(data.ltau_f97[i] - lphi)
    if savings
        for t=2:5
            tt = (i-1)*6 + t
            K += δ[1]*δ[2]^(5-t)*(data.ltau_f97[i] - lphi + log_cprice[t]-log_cprice[1])
        end
    else
        for t=2:5
            tt = (i-1)*6 + t
            K += δ[1]*δ[2]^(5-t)*(data.lY[tt] - log_cprice[t])
        end
    end

    g1 = (data.A02[i]/λ - K - δ[2]^5*data.A97[i]/λ)*Z_prod[1][i,:]
    g2 = (data.L02[i] - K - δ[2]^5*data.L97[i])*Z_prod[2][i,:]
    return [g1;g2]
end

# moment functions for singles:

function gmm_shares_single(par,data,i)
    # get ratio moments in 1997 (don't need expenditure for now, maybe later)
    g1 = ratio_moments_single_1997(par,data,i)
    # get ratio moments in 2002 (don't need expenditure for now, maybe later)
    g2 = ratio_moments_single(par,data,i,6)
    return [g1;g2]
end


function ratio_moments_single_1997(par,data,i)
    t = 1
    tt = (i-1)*6 + t
    if data.price_missing[tt]==0
        lϕm,lϕc,log_cprice,Φg = get_ratios_single(par,data,i,t)
        # do with respect to mother's time
        if data.mtime_missing[tt]==0
            if data.chcare_missing[tt]==0
                m3 = data.log_chcare[tt] - data.log_mtime[tt] - (lϕc-lϕm) - (data.logprice_c[tt] - data.logwage_m[tt])
                m3 = m3*data.Zs[3][tt,:]
            else
                m3 = zeros(Real,data.Ks[3])
            end
        else
            m3 = zeros(Real,data.Ks[3])
        end
        return m3
    else
        return zeros(Real,data.Ks[3])
    end
end


function ratio_moments_single(par,data,i,t)
    tt = (i-1)*6 + t
    if data.price_missing[tt]==0
        lϕm,lϕc,log_cprice,Φg = get_ratios_single(par,data,i,t)
        if data.good_missing[tt]==0
            if data.mtime_missing[tt]==0
                m1 = data.log_mtime[tt]-data.log_good[tt] - lϕm - (data.logwage_m[tt] - data.logprice_g[tt])
                m1 = m1*data.Zs[1][tt,:]
            else
                m1 = zeros(Real,data.Ks[1])
            end
            if data.chcare_missing[tt]==0
                m2 = data.log_chcare[tt] - data.log_good[tt] - lϕc - (data.logprice_c[tt] - data.logprice_g[tt])
                m2 = m2*data.Zs[2][tt,:]
            else
                m2 = zeros(Real,data.Ks[2])
            end
        else
            m1 = zeros(Real,data.Ks[1])
            m2 = zeros(Real,data.Ks[2])
        end
        # do with respect to mother's time
        if data.mtime_missing[tt]==0
            if data.chcare_missing[tt]==0
                m3 = data.log_chcare[tt] - data.log_mtime[tt] - (lϕc-lϕm) - (data.logprice_c[tt] - data.logwage_m[tt])
                m3 = m3*data.Zs[3][tt,:]
            else
                m3 = zeros(Real,data.Ks[3])
            end
        else
            m3 = zeros(Real,data.Ks[3])
        end
        return [m1;m2;m3]
    else
        return zeros(Real,data.Ks[1]+data.Ks[2]+data.Ks[3])
    end
end


function get_ratios_single(par,data,i,t) #<- do separate time periods?
    tt = (i-1)*6 + t
    aτ = exp(dot(par.βm,data.Xm[tt,:]))
    ag = exp(dot(par.βg,data.Xg[tt,:]))
    ay = par.ay
    lϕm = 1/(par.ρ-1)*log(ag*par.αm/aτ) + 1/(par.ρ-1)*(data.logwage_m[tt] - data.logprice_g[tt])
    lϕc = 1/(par.γ-1)*log(ag/ay) + (par.γ-par.ρ)/(par.ρ*(par.γ-1))*log(aτ*exp(par.ρ*lϕm) + ag) + 1/(par.γ-1)*(data.logprice_c[tt]-data.logprice_g[tt])
    Φg = ((aτ*exp(par.ρ*lϕm) + ag)^(par.γ/par.ρ) + ay*exp(par.γ*lϕc))^(-1/par.γ)
    log_cprice = log(Φg) + log(data.price_g[tt] + data.price_c[tt]*exp(lϕc) + data.wage_m[tt]*par.αm*exp(lϕm))
    return lϕm,lϕc,log_cprice,Φg
end


function production_moments_single_time(par,data,Z_prod,Z_prodF,i,savings=true)
    if data.all_prices[i]==1
        log_cprice = zeros(Real,5)
        lϕm,l2,log_cprice[1],Φg = get_ratios_single(par,data,i,1)
        # for t=2:5
        #     l1,l2,log_cprice[t],lg = get_ratios_single(par,data,i,t)
        # end
        lphi = lϕm + log(Φg)
        g3 = production_moments_mtime(par.λ,par.βθ,par.δ,data,i,log_cprice,Z_prod,lphi,savings)
        g4 = zeros(Real,2*size(Z_prodF[1])[2])
        g5 = [(data.A02[i]-par.λ*data.L02[i])*data.L97[i],data.A02[i]*data.A97[i] - par.λ^2*data.L97[i]*data.L02[i]]
    else
        g3 = zeros(Real,2*size(Z_prod[1])[2]) #( 0 x instruments)
        g4 = zeros(Real,2*size(Z_prodF[1])[2])
        g5 = zeros(Real,2)
    end
    return [g3;g4;g5]
end

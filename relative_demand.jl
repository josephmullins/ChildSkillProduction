using Parameters

# note:
# - the major steps here are calculating the residuals for relative demand given marital status and year, then interacting them with the appropriate instruments.
# - We want two things:
# (1) to be able to select/change the residuals and the instruments flexibly; and
# (2) to change how the moments are arranged flexibly
# - the function demand_moments_stacked calculates the moments by calling
# - the key is that this function uses an input function *gmap*: a function that tells demand_moments_stacked where to write the moments, which residuals to use, and which instruments to use for each residual given the year and marital status of the individual. The gmap function must be written specially for each specification
# - is this the best way to do it?


@with_kw struct CESmod
    # elasticity parameters
    ρ = -1.5 #
    γ = -3. 
    δ = [0.05,0.95]
    # coefficient vectors for factor shares
    βm = zeros(2)
    βf = zeros(2)
    βg = zeros(2)
    βθ = zeros(2)
    spec = (vm = [:const,:mar_stat],vf = [:const],vθ = [:const,:mar_stat],vg = [:const,:mar_stat])
end

function CESmod(spec)
    return CESmod(βm = zeros(length(spec.vm)),βf = zeros(length(spec.vf)),βg = zeros(length(spec.vg)),βθ = zeros(length(spec.vθ)),spec=spec)
end


function log_input_ratios(ρ,γ,ay,am,af,ag,logwage_m,logwage_f,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕf = 1/(ρ-1)*log(ag/af) + 1/(ρ-1)*(logwage_f - logprice_g)
    lϕc = 1/(γ-1)*log(ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + af*exp(ρ*lϕf) + ag)^(γ/ρ) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm) + exp(logwage_f)*exp(lϕf))
    return lϕm,lϕf,lϕc,log_price_index,Φg
end

# same function as above but for singles (no wage of father to pass)
function log_input_ratios(ρ,γ,ay,am,ag,logwage_m,logprice_g,logprice_c)
    lϕm = 1/(ρ-1)*log(ag/am) + 1/(ρ-1)*(logwage_m - logprice_g)
    lϕc = 1/(γ-1)*log(ag/ay) + (γ-ρ)/(ρ*(γ-1))*log(am*exp(ρ*lϕm) + ag) + 1/(γ-1)*(logprice_c-logprice_g)
    Φg = ((am*exp(ρ*lϕm) + ag)^(γ/ρ) + ay*exp(lϕc)^γ)^(-1/γ)
    log_price_index = log(Φg) + log(exp(logprice_g) + exp(logprice_c)*exp(lϕc) + exp(logwage_m)*exp(lϕm))
    return lϕm,lϕc,log_price_index,Φg
end

# QUESTION: how do we want to do this?
# splitting the dataframe may slow us down a lot
function log_input_ratios(pars,data,it)
    @unpack ρ,γ = pars
    if data.mar_stat[it]
        ag,am,af = factor_shares(pars,data,it,true) #<- returns the factor shares.
        lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,1.,am,af,ag,data.logwage_m[it],data.logwage_f[it],data.logprice_g[it],data.logprice_c[it])
        return lϕm,lϕf,lϕc,log_price_index,Φg
    else
        ag,am = factor_shares(pars,data,it,false)
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(ρ,γ,1.,am,ag,data.logwage_m[it],data.logprice_g[it],data.logprice_c[it])
        lϕf = 0.
        return lϕm,lϕf,lϕc,log_price_index,Φg
    end
end

function factor_shares(pars,data,it,mar_stat)
    @unpack βm,βf,βg,spec = pars
    if mar_stat
        am = linear_combination(βm,spec.vm,data,it)
        af = linear_combination(βf,spec.vf,data,it)
        ag = linear_combination(βg,spec.vg,data,it)
        return exp(ag),exp(am),exp(af)
    else
        am = linear_combination(βm,spec.vm,data,it)
        ag = linear_combination(βg,spec.vg,data,it)
        return exp(am),exp(ag)
    end 
end

#function:
# suppose 97 and 02 are uncorrelated
#
# use different ratios (right now must code a different function)
# recall: 97: use c/m,f/m 
# recall: 02: use c/m,f/m,c/g,m/g,f/g
function calc_demand_resids!(it,R97,R02,data,pars)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<- does this factor in missing data?
    if data.year[it]==1997
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R97[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R97[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
    # recall: 02: use c/m,f/m,c/g,m/g,f/g
    elseif data.year[it]==2002
        if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
            if !ismissing(data.log_chcare[it])
                R02[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
            end
            if !ismissing(data.log_ftime[it]) #<- what about missing prices?
                R02[2] = data.log_ftime[it] - data.log_mtime[it] - (lϕf - lϕm)
            end
        end
        if !ismissing(data.log_good[it])
            if !ismissing(data.log_chcare[it])
                R02[3] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
            end
            if !ismissing(data.log_mtime[it]) #& !ismissing(data.logwage_m[it]) #<- include for missing wage?
                R02[4] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
            end
            if !ismissing(data.log_ftime[it]) #& !ismissing(data.logwage_f[it]) #<- include for missing wage?
                R02[5] = data.log_ftime[it] - data.log_good[it] - lϕf + data.logprice_g[it]
            end
        end
    end
end

# this function calculates residuals in relative demand after checking that the data are available
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
    elseif data.year[it]==2002
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

# this function creates a stacked vector of moment conditions from a vector of residuals
# -- it relies on the function gmap to tell it, given variables data at [it], which residuals to use, which instruments to use, and where to place them in the vector g
#
# this version does the same but assumes that the function spits out the list of instruments instead of an index for existing instruments
# args... contains all the arguments that might potentially be needed by the function gmap
function demand_moments_stacked!(pars,n,g,R,data,gd,gmap,args...)
    for it=gd.starts[n]:gd.ends[n]
        R[:] .= 0.
        g_idx,zlist,r_idx = gmap(data,it,args...) #<- returns which part of g to write to, which instruments to use, and which residuals to use, based on [it] observables. Not all residuals are calculated or used for each observation.
        calc_demand_resids!(it,R,data,pars)
        resids = view(R,r_idx)
        g_it = view(g,g_idx)
        stack_moments!(g_it,resids,data,zlist,it)
    end
end
# thought: this can be a general function in estimation_tools by requiring residuals be calculated elsewhere


# functions below for the nonlinear least squares estimator. Possibly deprecated.: 
function weighted_nlls(P,W97,W02,data)
    ssq = 0
    r97 = zeros(typeof(P.βm[1]),2)
    r02 = zeros(typeof(P.βm[2]),5)
    gd = groupby(data,:KID)
    for i in 1:gd.ngroups
        #println(i)
        r97[:] .= 0.
        r02[:] .= 0.
        for it in gd.starts[i]:gd.ends[i]
            calc_demand_resids!(it,r97,r02,data,P)
        end
        ssq += r97'*W97*r97 + r02'*W02*r02
    end
    return ssq/gd.ngroups
end

function weighted_nlls(gd,i,P,W97,W02,data)
    r97 = zeros(typeof(P.ρ),2)
    r02 = zeros(typeof(P.ρ),5)
    for it in gd.starts[i]:gd.ends[i]
        calc_demand_resids!(it,r97,r02,data,P)
    end
    return r97'*W97*r97 + r02'*W02*r02
end
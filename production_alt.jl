# this script will use functions in relative_demand.jl

# below: two versions of the functions that take either one or two parameter objects

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

# version with one parameter instead of two
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


# this function creates a stacked vector of moment conditions from a vector of residuals
function production_demand_moments_relaxed!(pars1,pars2,n,g,R,data,savings=true)
    # first do relative demand moments
    @views demand_residuals_all!(R[1:9],pars1,n,data)

    @views production_residuals_all!(R[10:end],pars1,pars2,n,data,savings)
    stack_moments!(g,R,data,n)
end
# version with one parameter instead of two
function production_demand_moments_strict!(pars1,n,g,R,data,savings=true)
    fill!(R,0.)
    @views demand_residuals_all!(R[1:9],pars1,n,data)

    @views production_residuals_all!(R[10:end],pars1,n,data,savings)

    stack_moments!(g,R,data,n)
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
# version with one parameter instead of two
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
    if data.mar_stat[it]
        ag,am,af,ay = factor_shares(pars2,data,it,true)
        Φm = ((am + af*exp(lϕf - lϕm)^ρ + ag*exp(-lϕm)^ρ)^(γ/ρ)*(1-ay)+ay*exp(lϕc - lϕm)^γ)^(1/γ)
        lΦm=-log(Φm) 
        lΦg = lΦm - lϕm
        log_price_index = lΦg + log(exp(data.logprice_g[it]) + exp(data.logprice_c[it])*exp(lϕc) + exp(data.logwage_m[it])*exp(lϕm) + exp(data.logwage_f[it])*exp(lϕf))
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
    ng = length(spec.vy)
    pos += nf
    βy = x[pos:pos+ng-1]
    return CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βy=βy,spec=spec)
end
function demand_inv(pars)
    @unpack ρ,γ,βm,βf,βy = pars
    return [ρ;γ;βm;βf;βy]
end

# attempting to write an update function with some restrictions and some not.
function update(x,spec,unrestricted)
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
    P1 = CESmod{R}(ρ=ρ,γ=γ,βm = βm,βf = βf,βy=βy)
    P2 = CESmod{R}(ρ=ρ2,γ=γ2,δ = δ,βm = βm2,βf = βf2,βy=βy2,βθ=βθ,λ=λ)
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

    push!(x,P1.βy...)
    push!(x,P2.βy[Pu.βy]...)

    push!(x,P2.βθ...)
    push!(x,P2.λ)
    return x
end

# TODO: fix everything below here. a bit to do still.

function test_individual_restrictions(est,W,N,spec,data)
    np_demand = 2+length(spec.vm)+length(spec.vf)+length(spec.vy)
    tvec = zeros(np_demand)
    pvec = zeros(np_demand)
    for i in 1:np_demand
        unrestricted = fill(false,np_demand)
        unrestricted[i] = true
        P = update(est,spec)
        Pu = update_demand(unrestricted,spec)
        x1 = update_inv(P,P,Pu)
        tvec[i],pvec[i] = LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,data,spec,unrestricted)
    end
    return tvec,pvec
end

function test_joint_restrictions(est,W,N,spec,data)
    P = update(est,spec)
    np_demand = 2+length(spec.vm)+length(spec.vf)+length(spec.vy)
    unrestricted = fill(true,np_demand)
    Pu = update_demand(unrestricted,spec)
    x1 = update_inv(P,P,Pu)
    return LM_test(x1,sum(unrestricted),gfunc2!,W,N,8,data,spec,unrestricted)
end

function update(x,spec)
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
    P = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βy=βy,βθ=βθ,λ=λ)
    return P
end
function update_inv(pars)
    @unpack ρ,γ,δ,βm,βf,βy,βθ,λ = pars
    return [ρ;γ;δ;βm;βf;βy;βθ;λ]
end

# function to get the initial guess
function initial_guess(spec)
    P = CESmod(spec)
    x0 = update_inv(P)
    x0[1:2] .= -2. #<- initial guess consistent with last time
    x0[3:4] = [0.1,0.9] #<- initial guess for δ
    return x0
end

function get_moment_names(spec)
    rname = [["c/m","c/m","c/g","m/g","f/g","c/m","c/g","m/g","f/g"];
    repeat(["AP-m","LW-m","AP-c","LW-c","AP-f","LW-f","L1","L2"],2)]
    mname = String[]
    znames = [spec.zlist_97;spec.zlist_02;spec.zlist_07]
    zp = [vcat([z for z in Z]...) for Z in spec.zlist_prod]
    zlist = [znames;zp;zp]

    for r in eachindex(rname)
        for z in zlist[r]
            push!(mname,string(rname[r],z))
        end
    end
    return mname
end

mn = get_moment_names(spec_1p_x)
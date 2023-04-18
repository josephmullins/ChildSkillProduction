# this script will use functions in relative_demand.jl


# TODO:
# push then switch to new branch for these edits

# then write to add production moments to the bottom
#   - need to add some version of prices!
#   - potential idea: just write two versions of moment function (w/ vs without production)
#   - and make them flexible functions of specification. This seems the easiest to read!

# how do we want to do this???
# from previous version:
#   g1 = (data.A02[i]/λ - K - δ[2]^5*data.A97[i]/λ)*Z_prod[1][i,:]
#   g2 = (data.L02[i] - K - δ[2]^5*data.L97[i])*Z_prod[2][i,:]
#   g5 = [(data.A02[i]-par.λ*data.L02[i])*data.L97[i],data.A02[i]*data.A97[i] - par.λ^2*data.L97[i]*data.L02[i]]

function calc_production_resids!(n,R,data,pars,savings=true)
    it97 = (n-1)*6 + 1
    it02 = n*6

    lΦm,log_price_97 = calc_Φ_m(pars,data,it97)
    lX97 = data.log_mtime[it97] - lΦm
    Ψ0 = pars.δ[1]*pars.δ[2]^4*lX97
    for t=1:4
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it97+t)
        if savings
            Ψ0 += pars.δ[1]*pars.δ[2]^(4-t)*(lX97 + log_price_97 - log_price_index)
        else
            Ψ0 += pars.δ[1]*pars.δ[2]^(4-t)*(data.log_total_income - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    Ψ0 += linear_combination(pars.βθ,spec.vm,data,it)
    R[1] = data.AP[it02] / pars.λ - Ψ0 - pars.δ[2]^5 * data.AP[it97] / pars.λ
    R[2] = data.LW[it02] - Ψ0 - pars.δ[2]^5 * data.LW[it97]
    R[3] = (data.AP[it02] - pars.λ*data.LW[it02])*data.LW[it97]
    R[4] = (data.AP[it02]*data.AP[it97] - pars.λ^2*data.LW[it02]*data.LW[it97])
end


# this function creates a stacked vector of moment conditions from a vector of residuals
function production_demand_moments_stacked!(pars,n,g,R,data,spec,savings=true)
    # first do relative demand moments
    demand_moments_stacked!(pars,n,g,R,data,spec,savings)

    production_moments_stacked!(pars,n,g,R,data,spec,savings)

end

function production_moments_stacked!(pars,n,g,R,data,spec,savings=true)
    R[:] .= 0.
    it97 = (n-1)*6+1
    if data.all_prices[it97]
        calc_production_resids!(n,R,data,pars,savings)
        resids = view(R,1:4)
        g_n = view(g,spec.g_idx_prod)
        stack_moments!(g_n,resids,data,spec.zlist_prod,it97)
    end
end


function calc_Φ_m(pars,data,it)
    lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}, X_{t} = τ_{m,t} / (Φg * ϕ_{m})
    lΦm = log(Φg) + lϕm #<- now we have that X_{t} = τ_{m,t} / Φm
    return lΦm,log_price_index
end

# that takes relative input ratios (log_input_ratios) using pars1 #<- perceived pars
# and calculates *actual* composite investment (X) relative to mothers time (using pars2, implied input ratios)
# notes: factor_shares will return factor shares (i.e. evalutes the function a_m(ϕ_{m},Z_{n,t} ... see relative_demand.jl->log_input_ratios for usage)
# notes: log_input_ratios will return input ratios relative to goods
# notes: results_new.pdf describes how to calculate X

function calc_Φ_m(pars1,pars2,data,it)
end

# this script will use functions in relative_demand.jl


# TODO:
# push then switch to new branch for these edits

# then write to add production moments to the bottom
#   - need to add some version of prices!
#   - potential idea: just write two versions of moment function (w/ vs without production)
#   - and make them flexible functions of specification. This seems the easiest to read!

# how do we want to do this???
function calc_production_resids!(n,R,data,pars,savings=true)
    it97 = (n-1)*6 + 1

    #θpredict = [data.LW[it97]*pars.δ[2]^5/pars.λ[1], data.AP[it97]*pars.δ[2]^5 / pars.λ[2]]
    lΦm,log_price_97 = calc_Φ_m(pars,data,it97)
    lX97 = data.log_mtime[it97] - lΦm
    Ψ0 = pars.δ[1]*pars.δ[2]^4*lX97
    for t=1:4
        lϕm,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it97+t)
        if savings
            Ψ0 += pars.δ[1]*pars.δ[2]^(4-t)*(lX97 + log_price_97 - log_price_index)
        else
            Ψ0 += pars.δ[1]*pars.δ[2]^(4-t)*(data.logwage_m + data.logwage_f - log_price_index) #<- assume that father's log wage is coded as zero for single parents
        end
    end
    Ψ0 += linear_combination(pars.βθ,spec.vm,data,it)

end

function calc_Φ_m(pars,data,it)
    lϕm,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}, X_{t} = τ_{m,t} / (Φg * ϕ_{m})
    lΦm = log(Φg) + lϕm #<- now we have that X_{t} = τ_{m,t} / Φm
    return lΦm,log_price_index
end
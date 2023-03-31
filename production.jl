# this script will use functions in relative_demand.jl


# TODO:
# push then switch to new branch for these edits
# re-write demand moments to fit the one version we like
#   - idea: just use both years, hard code info on how many moments it will be
# then write to add production moments to the bottom
#   - need to add some version of prices!
#   - potential idea: just write two versions of moment function (w/ vs without production)
#   - and make them flexible functions of specification. This seems the easiest to read!

# how do we want to do this???
function calc_production_resids!(it,R,data,pars)

end

function calc_Φ_m(pars,data,it)
    lϕm,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it)
    # eq(7) in original draft says: X_{t} = g_{t}/Φg, and τ_{m,t} = ϕ_m*g_{t}
    Φm = Φg / exp(lϕm) #<- now we have that X_{t} = Φm * τ_{m,t}
    return Φm,log_price_index
end
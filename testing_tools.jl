using Distributions
# ----- functions for running tests on parameter estimates

# residual test for correlation between demand residuals in 97 and 02

function residual_test(data,N,pars)
    R = zeros(N,9)
    r = zeros(9)
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
            R[n,2] = r[1] #r[3] - r[4]
        end
    end
    test_stat = sqrt(N)*mean(R[:,1].*R[:,2]) / std(R[:,1])*std(R[:,2])
    pval = 2*cdf(Normal(),-abs(test_stat))
    return test_stat,pval
end

# - test parameter restrictions one by one using LM test
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
        tvec[i],pvec[i] = LM_test(x1,sum(unrestricted),gfunc2!,W,N,25,data,spec,unrestricted)
    end
    return tvec,pvec
end

# conduct joint test of parameter restrictions using LM stat:
function test_joint_restrictions(est,W,N,spec,data)
    P = update(est,spec)
    np_demand = 2+length(spec.vm)+length(spec.vf)+length(spec.vy)
    unrestricted = fill(true,np_demand)
    Pu = update_demand(unrestricted,spec)
    x1 = update_inv(P,P,Pu)
    return LM_test(x1,sum(unrestricted),gfunc2!,W,N,25,data,spec,unrestricted)
end
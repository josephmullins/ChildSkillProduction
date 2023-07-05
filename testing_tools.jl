# functions for running tests on parameter estimates

# residual test for correlation between demand residuals in 97 and 02

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
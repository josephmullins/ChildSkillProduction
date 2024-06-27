
include("estimation/data_tools.jl")
include("estimation/specifications.jl")
include("estimation/clustering.jl")
include("estimation/moment_functions.jl")
include("estimation/gmm.jl")
include("estimation/testing.jl")
include("estimation/input_output.jl")

# And finally a function that does everything we want to for our baseline restricted results, it takes as an argument:
# - the panel data
# - the specification (spec)
# and runs the estimation routine, and tests the restrictions collectively and individually

function run_restricted_estimation(panel_data,spec,case,gfunc!)
    N = length(unique(panel_data.kid))
    data = child_data(panel_data,spec)
    nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
    W = I(nmom)
    x0 = initial_guess(spec,case)
    unrestricted = fill(false,length(x0))
    res = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec,unrestricted,case)

    # test the restrictions
    W = inv(res.Ω)
    t_joint,p_joint = test_joint_restrictions(res.est,W,N,spec,data,case,gfunc!)
    t_indiv,p_indiv = test_individual_restrictions(res.est,W,N,spec,data,case,gfunc!)
    return (;res...,t_joint,p_joint,t_indiv,p_indiv)
end

function run_unrestricted_estimation(panel_data,spec,case,gfunc!,res)
    N = length(unique(panel_data.kid))
    (;Ω,p_indiv,est) = res
    unrestricted = p_indiv.<0.05
    P = update(est,spec,case)
    Pu = update_demand(unrestricted,spec)
    x1 = update_inv_relaxed(P,P,Pu,case)

    W = inv(Ω)
    data = child_data(panel_data,spec)
    res3u = optimize(x->gmm_criterion(x,gfunc!,W,N,length(data.Z),data,spec,unrestricted,case),x1,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))
    g0 = gmm_criterion(x1,gfunc!,W,N,length(data.Z),data,spec,unrestricted,case)

    g1 = res3u.minimum
    DM = 2N*max(g0-g1,0.)
    if sum(unrestricted)>0
        p_val = 1 - cdf(Chisq(sum(unrestricted)),DM)
    else
        p_val = 1.
    end
    #p1,p2 = update_relaxed(res3u.minimizer,spec,unrestricted,case)
    var = parameter_variance_gmm(res3u.minimizer,gfunc!,W,N,length(data.Z),data,spec,unrestricted,case)
    se = sqrt.(diag(var))
    return (;est = res3u.minimizer,se,p_val,DM,unrestricted,case)
end

function run_demand_estimation(panel_data,spec,gfunc!)
    N = length(unique(panel_data.kid))

    x0 = demand_guess(spec)
    data = child_data(panel_data,spec)
    nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
    W = I(nmom)
    res = estimate_gmm(x0,gfunc!,W,N,9,data,spec)
    p = update_demand(res.est,spec)
    pse = update_demand(res.se,spec)
    r = residual_test(data,N,p)
    pval = r[2]
    return (;res...,p,pse,pval)
end
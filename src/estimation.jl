using CSV, DataFrames, Random, Printf, LinearAlgebra, Statistics, Distributions
using Optim

include("estimation/data_tools.jl")
include("estimation/specifications.jl")
include("estimation/clustering.jl")
include("estimation/moment_functions.jl")
include("estimation/moment_functions_older.jl")
include("estimation/gmm.jl")
include("estimation/testing.jl")
include("estimation/input_output.jl")

# And finally a set of functions that run the various estimation routines to produce final results

# - runs joint gmm estimation in the completely restricted case and tests individual restrictions on the parameters
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
    return (;res...,t_joint,p_joint,t_indiv,p_indiv,case)
end

# - runs joint gmm estimation with parameters that fail the LM test above allowed to be estimated separately
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

# runs joint gmm estimation with the intercept term on the mother's factor share allowed to differ in demand vs production
function run_unrestricted_estimation_mothershare(panel_data,spec,case,gfunc!,res)
    N = length(unique(panel_data.kid))
    (;Ω,est) = res
    np_demand = 2 + length(spec.vm) + length(spec.vf) + length(spec.vy)
    P_index = update_demand(1:np_demand,spec)
    unrestricted = fill(false,np_demand)
    unrestricted[P_index.βm[1]] = true
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

# runs joint gmm estimation using only children aged 8-12 and assuming no childcare
function run_restricted_estimation_older(panel_data,spec,case,gfunc!)
    N = length(unique(panel_data.kid))
    data = child_data(panel_data,spec)
    nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
    W = I(nmom)
    x0 = initial_guess_older(spec,case)
    unrestricted = fill(false,length(x0))
    res = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec,unrestricted,case)

    # test the restrictions
    W = inv(res.Ω)
    t_joint,p_joint = test_joint_restrictions_older(res.est,W,N,spec,data,case,gfunc!)
    t_indiv,p_indiv = test_individual_restrictions_older(res.est,W,N,spec,data,case,gfunc!)
    return (;res...,t_joint,p_joint,t_indiv,p_indiv,case)
end

# runs gmm estimation with only relative demand moments (no production moments and no parameters from that appear only intertemporally i.e. δ₁,δ₂,ϕ_θ)
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
using Random, Distributions

function gen_data(p1,p2,N)
    (;δ,σξ,σπ) = p2
    rel_price = rand(LogNormal(0,σπ),N)
    x1 = rand(LogNormal(3.,1.),N)
    x2 = (p1.a/(1-p1.a))^(1/(p1.ρ-1)) .* rel_price.^(1/(p1.ρ-1)) .* x1
    y = δ * log.( (p2.a .* x1 .^ p2.ρ .+ (1 - p2.a) .* x2 .^ p2.ρ ) .^ (1/p2.ρ) ) .+ rand(Normal(0,σξ),N)
    x2_obs = log.(x2) .+ rand(Normal(0,0.2),N)
    x1_obs = log.(x1)
    return (;y,x1,x2,x1_obs,x2_obs,rel_price)
end

function gen_data(p,N)
    (;a,ρ,σξ,δ,σπ) = p
    x1 = rand(LogNormal(3.,1),N)
    x2 = rand(LogNormal(3.,σπ),N)
    y = δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ ) .^ (1/ρ) ) .+ rand(Normal(0,σξ),N)
    return (;y,x1,x2)
end

p = (;ρ = -2.,a = 0.5, δ = 0.2,σξ = 0.5,σπ = 0.2)
dat = gen_data(p,p,500)

# no measurement error
function Q_nlls(p1,p2,data)
    (;y,x1,x2,x1_obs,x2_obs,rel_price) = data
    (;δ) = p2
    r1 = x2_obs .- x1_obs .- (1/(p1.ρ-1))*log(p1.a/(1-p1.a)) .- (1/(p1.ρ-1))*log.(rel_price)
    #x2 = (p1.a/(1-p1.a))^(1/(p1.ρ-1)) .* rel_price.^(1/(p1.ρ-1)) .* x1
    r2 = y .- δ * log.( (p2.a .* x1 .^ p2.ρ .+ (1 - p2.a) .* x2 .^ p2.ρ) .^ (1/p2.ρ) )
    return sum(r1.^2) + sum(r2.^2)
end

logit(x) = exp(x)/(1+exp(x))
logit_inv(x) = log(x/(1-x))

function pars_restricted(x)
    ρ = x[1]
    a = logit(x[2])
    δ = x[3]
    p = (;ρ,a,δ)
    return p,p
end
function pars_relaxed(x)
    p1 = (;ρ = x[1], a = logit(x[2]))
    p2 = (;ρ = x[3], a = logit(x[4]),δ = x[5])
    return p1,p2
end

using Optim

function monte_carlo(N,B,p)
    Xb = zeros(3,B)
    x0 = [p.ρ,logit_inv(p.a),p.δ]
    for b in 1:B
        Random.seed!(11211+b)
        println("Doing round $b of $B trials")
        dat = gen_data(p,p,N)
        f_obj(x) = Q_nlls(pars_restricted(x)...,dat)
        res = optimize(f_obj,x0,LBFGS(),autodiff=:forward,Optim.Options(show_trace=false)) 
        Xb[:,b] = res.minimizer
    end
    return Xb
end

Xb = monte_carlo(1000,100,p)

using Plots
histogram(Xb[1,:])

break

# Random.seed!(11211+2)
# dat = gen_data(p,p,N)
# f_obj(x) = Q_nlls(pars_restricted(x)...,dat)
# res = optimize(f_obj,x0,LBFGS(),autodiff=:forward,Optim.Options(show_trace=false)) 
# p1,_ = pars_restricted(res.minimizer)
# Q_nlls(p1,p1,dat)
# Q_nlls(p,p,dat)

function monte_carlo(N,B,p)
    Xb = zeros(5,B)
    x0 = [p.ρ,logit_inv(p.a),p.ρ,logit_inv(p.a),p.δ]
    for b in 1:B
        Random.seed!(11211+b)
        println("Doing round $b of $B trials")
        dat = gen_data(p,p,N)
        f_obj(x) = Q_nlls(pars_relaxed(x)...,dat)
        res = optimize(f_obj,x0,LBFGS(),autodiff=:forward,Optim.Options(show_trace=false)) 
        Xb[:,b] = res.minimizer
    end
    return Xb
end

Xb = monte_carlo(1000,100,p)

using Plots
histogram(Xb[3,:])

function Q_nlls(x,data)
    (;y,x1,x2) = data
    ρ = x[1]
    a = logit(x[2])
    δ = x[3]
    r2 = y .- δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ) .^ (1/ρ) )
    return sum(r2.^2)
end

function monte_carlo(N,B,p)
    Xb = zeros(3,B)
    x0 = [p.ρ,logit_inv(p.a),p.δ]
    for b in 1:B
        Random.seed!(11211+b)
        println("Doing round $b of $B trials")
        dat = gen_data(p,p,N)
        res = optimize(x->Q_nlls(x,dat),x0,LBFGS(),autodiff=:forward,Optim.Options(show_trace=false)) 
        Xb[:,b] = res.minimizer
    end
    return Xb
end

# how much variation in relative prices do we need? maybe a shite load?
# note: in our case having prices and demand parameters is an equivalent formulation
p = (;p...,σπ = 2.)
Xb = monte_carlo(1000,100,p)
histogram(Xb[1,:])


# ------------- simplest possible case -------------- #
function Q_nlls(x,data)
    (;y,x1,x2) = data
    ρ = x
    a = 0.5 #<- true value
    δ = 0.2 #<- true value
    r2 = y .- δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ) .^ (1/ρ) )
    return sum(r2.^2)
end

function monte_carlo(N,B,p)
    Xb = zeros(B)
    Xb2 = zeros(B)
    for b in 1:B
        Random.seed!(11211+b)
        println("Doing round $b of $B trials")
        dat = gen_data(p,p,N)
        res = optimize(x->Q_nlls(x,dat),-100.,1.) #[-2.]) #-300,1.) 
        Xb[b] = res.minimizer
        dat = gen_data(p,N)
        res = optimize(x->Q_nlls(x,dat),-100.,1.) #[-2.]) #-300,1.) 
        Xb2[b] = res.minimizer
    end
    return Xb,Xb2
end

# how much variation in relative prices do we need? maybe a shite load?

# note: in our case having prices and demand parameters is an equivalent formulation
# HERE IS THE MOST IMPORTANT THING
# the variance of the estimator depends on sample size but also on the variance of relative prices. 
# As these get smaller, the left tail of the distribution of the estimator grows (variance increases).
# We could probably run a simulation to show that this is true. Could we even characterize the variance?
# NOTE: this could also depend on the value of ρ itself (potentially).
# Another issue: if I set the lower bound too low, the estimate converges to the same number for every trial.
# NOTE: this seems to be related to whatever I set the mean of x1 to be also.
# this is caused by the lower bound for ρ hitting an infinite number. we need to dynamically find the initial guess.
p = (;p...,σπ = 1.5)
Xb = monte_carlo(1000,500,p)
histogram(Xb)
dat = gen_data(p,p,5_000)
Q_nlls(-500.,dat) # so this is the issue!!!

σπ_vec = [0.5,1.,1.5]
N_vec = [500,1_000,5_000]
H = []
H2 = []
for i in 1:3
    for j in 1:3
       p = (;p...,σπ = σπ_vec[i])
       Xb,Xb2 = monte_carlo(N_vec[j],200,p)
       push!(H,histogram(Xb,bins=20))
       push!(H2,histogram(Xb2,bins=20))
    end
end
plot(H...)
plot(H2...)


dat = gen_data(p,p,1000)
Q_nlls(0.99,dat)

xg = LinRange(-3.,-1.,100)
plot(xg,[Q_nlls(x,dat) for x in xg])

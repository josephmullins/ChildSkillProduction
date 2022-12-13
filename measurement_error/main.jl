using Distributions,Optim,Random,Parameters

# Four sets of parameters
# π - mixture probabilities
# β_3 - mean and covariance of each normal mixture
# β_2 - production parameters and variance of the TFP shock
# β_1 - measurement parameters
# π - assume two mixtures (K=2)

# β_2:
@with_kw mutable struct CESparams{R}
    ρ::R = -1.5
    γ::R = -3.
    σ_η::Float64 = 0.5 #<- we don't need to take a derivative with respect to this parameter so we can fix the type.
    logθ::R = 1.
    a::Array{R,1} = [0.2,0.4,0.3,1.] #<= order: a_m,a_f,a_g,a_Y, with normalization that a_{Y}=1.
    δ::Array{R,1} = [0.5,0.5]
end
P = CESparams() #<- this calls the constructor with the default values as given above

# β_3:
Random.seed!(1409)
C = rand(5,5)
Σ1 = C'*C
C = rand(5,5)
Σ2 = C'*C

μ = [ones(5),-ones(5)] #<- μ[k] is the mean for the kth mixture
Σ = [Σ1,Σ2]
π0 = [0.5,0.5]

#<- The Distributions struct "Mixture Model"  simplifies how we draw from the initial distribution. We no longer need the function "initial_draw"
# We also don't need the function "draw_data" since we can just call rand(F0,N)
F0 = MixtureModel([MultivariateNormal(μ[i],Σ[i]) for i=1:2],π0) 
x = rand(F0)

#β_1: assume J=2
@with_kw mutable struct MeasurementModel
    J::Int64 = 2 #<- number of measurements
    λ::Array{Float64,1} = [1.,0.9]
    σ_ζ::Array{Float64,1} = [0.4,0.4]
end

MM = MeasurementModel()

# function to evaluate nested CES production function 
# -- this is total human capital investment
# assume x = [τ_m,τ_f,g,Y,Ψ_0]
function logCES(τ_m,τ_f,g,Y,logΨ_0,P) #<- notice how we only need to pass the struct now
    @unpack δ,ρ,γ,a,logθ = P  #<- this macro comes from the Parameters package
    home_input = (a[1]*τ_m^ρ + a[2]*τ_f^ρ + a[3]*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + a[4]*Y^γ) + δ[2]*logΨ_0
end

function logCES(x,P)
    logCES(exp(x[1]),exp(x[2]),exp(x[3]),exp(x[4]),x[5],P)
end

function draw_logΨ1(τ_m,τ_f,g,Y,logΨ_0,P)
    return logCES(τ_m,τ_f,g,Y,logΨ_0,P) + rand(Normal(0,P.σ_η)) #<- notice we don't *have* to use @unpack, it's just for convenience. We can always use P.
end 

function draw_logΨ1(x,P)
    return logCES(x,P) + rand(Normal(0,P.σ_η)) #<- edited this line to call logCES directly
end

##total human capital investment a child receives as a function of all inputs (investment inputs)

logΨ1 = draw_logΨ1(x,P)

# function to return a 2 x 1 vector of measurements, M given a single skill logΨ:
function draw_M(logΨ,MM)
    @unpack λ, σ_ζ = MM
    return  λ*logΨ .+ rand(MultivariateNormal([0,0],σ_ζ))
end

M = draw_M(logΨ1,MM)

#function to draw N observations given parameters


N = 10000
dat0 = rand(F0,N)

#N x 2 Array of Skills 
function draw_skills(dat0,P)
    @unpack δ,a,γ,ρ,logθ,σ_η = P
    N = size(dat0)[2]
    dat = Array{Float64, 1}(undef, N)
    for i in 1:N
        @views dat[i] = draw_logΨ1(dat0[:,i],P)
    end
    return dat
end

dat1 = draw_skills(dat0,P);
@time dat1 = draw_skills(dat0,P);

# return N x 4 array of measurements (our M)
function draw_measurements(dat1,dat0,MM) 
    datM = Array{Float64, 3}(undef, 2, 2, N)
    for i in 1:N
        datM[:,1,i]=draw_M(dat0[5,i],MM)
        datM[:,2,i]=draw_M(dat1[i],MM)
    end
    return datM
end

@time datM = draw_measurements(dat1,dat0,MM); 

#next: code up a NLLS criterion, write estimation routine and check that it works
function pred_error(i,P,dat1,dat0)
    return dat1[i]-logCES(dat0[:,i],P)
end


function ssq(P::CESparams,dat1,dat0)
    sumsq = 0
    for i=1:length(dat1)
        sumsq += pred_error(i,P,dat1,dat0)^2
    end
    return sumsq
end

ssq(P,dat1,dat0)

# add some functions to (1) return the vector of transformed parameters given CESpars
getPars(x::Array) = CESparams(δ=exp.(x[1:2]),a=[exp.(x[3:5]);1.],γ=x[6],ρ=x[7],logθ=x[8])
# and (2): return CESpars given the vector of transformed parameters
getPars(P::CESparams) = [log.(P.δ);log.(P.a[1:3]);P.γ;P.ρ;P.logθ]
# compare these two versions of ssq and notice how we are using type-dependent dispatch

# notice again here that we have two ssq functions that rely on type-dependent dispatch
function ssq(x::Array{R,1},dat1,dat0) where R<:Real 
    return ssq(getPars(x),dat1,dat0)
end

# issue: σ_η isn't updated in this estimation step. that's the problem.

x0 = getPars(P)

ssq(x0,dat1,dat0)

@time res = optimize(x->ssq(x,dat1,dat0),x0)

@time res2 = optimize(x->ssq(x,dat1,dat0),x0,BFGS(),autodiff=:forward)

@time res3 = optimize(x->ssq(x,dat1,dat0),x0,Newton(),autodiff=:forward)

# notice now that we have fixed the normalization issue, the results are consistent across routines
[res.minimizer res2.minimizer res3.minimizer x0]


N = 1000
B = 100
Xb = zeros(8,B)
#Xb2 = zeros(9,B)
R = zeros(Bool,B)

P = CESparams(σ_η = 0.3) #<- reduce the standard deviation of η

for b=1:B
    println(b)
    dat0 = rand(F0,N)
    dat1 = draw_skills(dat0,P);
    res = optimize(x->ssq(x,dat1,dat0),x0,BFGS(),autodiff=:forward)
    Xb[:,b] = res.minimizer
    R[b] = res.ls_success
end

[x0 mean(Xb,dims=2)[:]]

using Plots
histogram(Xb[4,:])
plot!([x0[4],x0[4]],[0,ylims()[2]],linewidth=3.)
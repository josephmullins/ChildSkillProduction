using Distributions,Optim,Random,Parameters,LinearAlgebra

# objects to track:
# (1) parameters of the CES (including σ_η)
# (2) weights for the Model
# (3) iteration number?
# (4) Simulated Draws: logΨ1,logΨ0

# functions:
# (1) Evaluate log-likelihood / likelihood
#    (1.1) of measures given factor draws
#    (1.2) of Ψ1 given Ψ0,I, and k
#    (1.3) of Ψ0 given I, k, and parameters
#    (1.4) of I given k
# (2) Update measurement parameter estimates
# (3) Update CES parameter estimates
# (4) Calculate weights for each combination of n,k,r


R = 10
K = 2 #<- number of mixtures
J = 2 #<- number of measurements
N = 1000 #<- number of observations

wghts = zeros(N,K,R)


@with_kw mutable struct MeasurementModel
    J::Int64 = 2 #<- number of measurements
    λ::Array{Float64,1} = [1.,0.9]
    σ_ζ::Array{Float64,1} = [0.4,0.4]
end

@with_kw mutable struct InitDist
    K::Int64 = 2 #<- number of mixtures. TODO: RENAME THIS TO K
    μ::Array{Array{Float64,1},1} = [fill(-1.,4),fill(1.,4)]
    μΨ::Array{Float64,1} = [-1.,1.]
    Σ::Array{Array{Float64,2},1} = [I(4),I(4)]
    B::Array{Array{Float64,1},1} = [fill(0.1,4),fill(-0.1,4)]
    σΨ::Array{Float64,1} = [0.5,0.5]
    π0::Array{Float64,1} = [0.5,0.5]
end

MM = MeasurementModel()
F0 = InitDist()

@with_kw mutable struct CESparams{R}
    ρ::R = -1.5
    γ::R = -3.
    σ_η::Float64 = 0.5 #<- we don't need to take a derivative with respect to this parameter so we can fix the type.
    logθ::R = 1.
    a::Array{R,1} = [0.2,0.4,0.3,1.] #<= order: a_m,a_f,a_g,a_Y, with normalization that a_{Y}=1.
    δ::Array{R,1} = [0.5,0.5]
end
P = CESparams() #<- this calls the constructor with the default values as given above

# ---------- functions to draw from the initial distribution  ------------------- #
function draw_data(F0::InitDist) 
    @unpack μ,μΨ,Σ,B,σΨ,π0 = F0
    k = rand(Categorical(π0)) #<- draw the mixture
    logI = rand(MultivariateNormal(μ[k],Σ[k]))
    logΨ0 = μΨ[k] + dot(B[k],logI) + rand(Normal(0,σΨ[k]))
    return logΨ0,logI
end
# function to draw a sample from InitDist
function draw_data!(logI,logΨ0,F0::InitDist)
    N = length(logΨ0)
    for n=1:N
        logΨ0[n],logI[:,n] = draw_data(F0)
    end
end
function draw_data(F0::InitDist,N) 
    logI = zeros(4,N)
    logΨ0 = zeros(N)
    draw_data!(logI,logΨ0,F0::InitDist)
    return logΨ0,logI
end

logΨ0,logI = draw_data(F0,1000)
Inv = exp.(logI)

# ---------------- functions to evaluate the CES function (in log-terms) ------------------ #
function logCES(τ_m,τ_f,g,Y,logΨ_0,P) #<- notice how we only need to pass the struct now
    @unpack δ,ρ,γ,a,logθ = P  #<- this macro comes from the Parameters package
    home_input = (a[1]*τ_m^ρ + a[2]*τ_f^ρ + a[3]*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + a[4]*Y^γ) + δ[2]*logΨ_0
end

function logCES(I,logΨ0,P)
    logCES(I[1],I[2],I[3],I[4],logΨ0,P)
end

@time logCES(Inv[:,1],logΨ0[1],P)

# ----------------- functions to draw skills in second period given initial skills and investment ------ #

function draw_logΨ1(Inv,logΨ0,P)
    return logCES(Inv,logΨ0,P) + rand(Normal(0,P.σ_η)) #<
end

function draw_logΨ1!(logΨ1::Array{Float64,1},Inv::Array{Float64,2},logΨ0,P)
    N = size(Inv)[2]
    for n=1:N
        @views logΨ1[n] = draw_logΨ1(Inv[:,n],logΨ0[n],P)
    end
end
function draw_logΨ1(Inv::Array{Float64,2},logΨ0,P)
    N = size(Inv)[2]
    logΨ1 = zeros(N)
    draw_logΨ1!(logΨ1::Array{Float64,1},Inv::Array{Float64,2},logΨ0,P)
    return logΨ1
end

@time logΨ1 = draw_logΨ1(Inv,logΨ0,P);

# -------------- functions to simulate measurement error  ---------- #

# function to return a 2 x 1 vector of measurements, M, given a single skill logΨ:
function draw_M!(M,logΨ::Float64,MM)
    @unpack J, λ, σ_ζ = MM
    for j=1:J
        M[j] = λ[j]*logΨ + rand(Normal(0,σ_ζ[j]))
    end
end
function draw_M(logΨ::Float64,MM)
    @unpack J = MM
    M = zeros(J)
    draw_M!(M,logΨ,MM)
    return  M
end
function draw_M!(M::Array{Float64,2},logΨ::Array{Float64,1},MM)
    J,N = size(M)
    for n=1:N
        @views draw_M!(M[:,n],logΨ[n],MM)
    end
end
function draw_M(logΨ::Array{Float64,1},MM)
    @unpack J = MM
    N = length(logΨ)
    M = zeros(J,N)
    draw_M!(M,logΨ,MM)
    return  M
end


#function to draw N observations given parameters
function draw_data(F0::InitDist,P::CESparams,MM::MeasurementModel,N)
    logΨ0,logI = draw_data(F0,N)
    Inv = exp.(logI)
    logΨ1 = draw_logΨ1(Inv,logΨ0,P);
    M0 = draw_M(logΨ0,MM)
    M1 = draw_M(logΨ1,MM)
    return M0,M1,logI,Inv
end

mutable struct EM_model #<
    wghts::Array{Float64,3}
    M0::Array{Float64,2} #<- first period measurements
    M1::Array{Float64,2} #<- second period measurements
    Inv::Array{Float64,2} #<- observed investments
    logI::Array{Float64,2} #<- just keep the log version on hand also.
    logΨ0::Array{Float64,3} #<- simulated draws of Ψ0
    logΨ1::Array{Float64,3} #<- simulated draws of Ψ1
    MM::MeasurementModel #<- measurement parameters (?)
    P::CESparams{Float64} #<- CES parameters
    F0::InitDist #<- Initial distribution
end

# write a constructor function
function EM_model(N,K,R)
    wghts = zeros(N,K,R)
    F0 = InitDist()
    P = CESparams()
    MM = MeasurementModel()
    M0,M1,logI,Inv = draw_data(F0,P,MM,N)
    return EM_model(wghts,M0,M1,Inv,logI,zeros(N,K,R),zeros(N,K,R),MM,P,F0)
end

EM = EM_model(1000,2,10);

## ----- Function to calculate weights for the EM algorithm  -------- #

# (1.1) likelihood of measures given factor draws
# this calculation assumes independence of measurement error over j
function f_M_given_Ψ(M,logΨ,MM)
    @unpack J,λ,σ_ζ = MM
    f = 1.
    for j=1:MM.J
        f *= pdf(Normal(λ[j]*logΨ,σ_ζ[j]),M[j])
    end
    return f
end

# NOTE: from the readme
# f(M,Ψ,I,k)#<- want to integrate over this.  
# it is equal to f(M|Ψ)*f(Ψ1|I,Ψ0)*f(Ψ0|I,k)*f(I|k)*π[k]
# we draw from f(Ψ1|I,Ψ0)*f(Ψ0|I,k) and calculate the other components in the weights: f(M|Ψ)*f(I|k)*π[k]

function get_wght(n,k,r,EM)
    @unpack M0,M1,logΨ0,logΨ1,MM,F0,logI = EM
    f = 1.
    # likelihood of first measurement
    f *= f_M_given_Ψ(M0[:,n],logΨ0[n,k,r],MM)
    # likelihood of second measurement
    f *= f_M_given_Ψ(M1[:,n],logΨ1[n,k,r],MM)
    # likelihood of investments given mixture
    @unpack μ,Σ,π0 = F0
    @views f *= pdf(MultivariateNormal(μ[k]),logI[:,n])
    # likelihood of mixture
    f *= π0[k]
    return f
end

get_wght(1,1,1,EM)

function get_wght!(EM::EM_model)
    N,K,R = size(EM.wghts)
    for n in 1:N
        for k in 1:K, r in 1:R
            EM.wghts[n,k,r] = get_wght(n,k,r,EM)
        end
        # now normalize:
        @views EM.wghts[n,:,:] ./= sum(EM.wghts[n,:,:])
    end
end

get_wght!(EM)

function draw_Ψ0_Ψ1!(EM::EM_model)
    N,K,R = size(EM.wghts)
    @unpack P,F0,logI,Inv = EM
    @unpack μΨ,B,μ,σΨ = F0
    for n in 1:N, k in 1:K, r in 1:R
        @views EM.logΨ0[n,k,r] = μΨ[k] + dot(B[k],logI[:,n]) + rand(Normal(0,σΨ[k]))
        @views EM.logΨ1[n,k,r] = draw_logΨ1(Inv[:,n],EM.logΨ0[n,k,r],P)
    end
end

draw_Ψ0_Ψ1!(EM)
get_wght!(EM)

# -------- Code to Calculate the M-Step for production parameters ---------- #

###########################################################################



function λ_update!(EM::EM_model) ##this only updates the second value
    N,K,R = size(EM.wghts)
    numerator = 0
    denominator = 0
    for n in 1:N, k in 1:K, r in 1:R
        numerator += (EM.wghts[n,k,r]*EM.logΨ0[n,k,r]*EM.M0[k,n]+EM.wghts[n,k,r]*EM.logΨ1[n,k,r]*EM.M1[k,n])
        denominator += (EM.wghts[n,k,r]*(EM.logΨ0[n,k,r])^2+EM.wghts[n,k,r]*(EM.logΨ1[n,k,r])^2)
    end
    EM.MM.λ[2] = numerator/denominator
end

λ_update!(EM)

function π_update!(EM::EM_model) ##edited - struct
    N,K,R = size(EM.wghts)
    for k in 1:K
        EM.F0.π0[k] = 0.
        for n in 1:N, r in 1:R
            EM.F0.π0[k] += EM.wghts[n,k,r]
        end
        EM.F0.π0[k]=EM.F0.π0[k]/sum(EM.wghts)
    end
end

π_update!(EM)

# NOTE: this function only works properly as part of the M-step if λ_update! is called first
function σ_ζ_update!(EM::EM_model) ##edited - struct
    N,K,R = size(EM.wghts)
    J=EM.MM.J
    for j in 1:J
        numerator = 0
        for n in 1:N, k in 1:K, r in 1:R
            numerator +=  EM.wghts[n,k,r]*((EM.M0[k,n]-EM.MM.λ[j]*EM.logΨ0[n,k,r])^2+
                            (EM.M1[k,n]-EM.MM.λ[j]*EM.logΨ1[n,k,r])^2)
        end
        EM.MM.σ_ζ[j] = numerator/(2N)
    end
end

σ_ζ_update!(EM) 

function μ_update!(EM::EM_model) ##edited - struct
    N,K,R = size(EM.wghts)
    NI = size(EM.logI)[1]
    denominator = sum(EM.wghts,dims=[1,3])
    for k in 1:K, i in 1:NI
        EM.F0.μ[k][i] = 0.
        for n in 1:N, r in 1:R
            EM.F0.μ[k][i] += EM.wghts[n,k,r]*EM.logI[i,n]  
        end
        @views EM.F0.μ[k][i] = EM.F0.μ[k][i]/denominator[k]
    end
end

μ_update!(EM::EM_model)


function Σ_update!(EM::EM_model) ##edited - struc
    N,K,R = size(EM.wghts) #zero out the current value
    M,L = size(EM.F0.Σ[1])
    EM.F0.Σ = EM.F0.Σ*0 
    denominator = sum(EM.wghts,dims=[1,3])
    for k in 1:K
        for l in 1:L, m in l:M
            for n in 1:N, r in 1:R
                @views EM.F0.Σ[k][m,l] += EM.wghts[n,k,r]*(EM.logI[l,n]-EM.F0.μ[k][l])*(EM.logI[m,n]-EM.F0.μ[k][m])
            end
            EM.F0.Σ[k][m,l] /= denominator[k]
            EM.F0.Σ[k][l,m] = EM.F0.Σ[k][m,l]
        end
    end
end

Σ_update!(EM::EM_model)

function generate_C(EM::EM_model) #something still off here
    N,K,R = size(EM.wghts)
    x=[ones(1,N);EM.logI]
    C1=zeros(5,5,2)
    C2=zeros(5,2)
    C=zeros(5,2)
    for k in 1:K
        for n in 1:N, r in 1:R
            C1[:,:,k] += EM.wghts[n,k,r].*(x[:,n]*x[:,n]')
            C2[:,k] += EM.wghts[n,k,r].*(x[:,n]).*EM.logΨ0[n,k,r]
        end
        C[:,k]=inv(C1[k])*C2[:,k]
    end
    return C
end

@time C = generate_C(EM)

# TODO: maddi to fix this function
function update_σΨ!(EM::EM_model)
    N,K,R = size(EM.wghts)
    x=[ones(1,N);logI]
    generate_C(EM::EM_model)
    for k in 1:K
        for n in 1:N, r in 1:R
            EM.F0.σΨ[k] += EM.wghts[n,k,r]*(EM.logΨ0[n,k,r]-x[:,n]'*C[:,k])^2
        end
        EM.F0.σΨ[k]=EM.F0.σΨ[k]/((sum(sum(EM.wghts, dims=1),dims=3))[k]) 
    end  
end

# add exclamation points
function M_step(EM::EM_model)
    π_update(EM::EM_model)
    λ_update(EM::EM_model)
    σ_ζ_update(EM::EM_model)
    μ_update(EM::EM_model)
    Σ_update(EM::EM_model)
    update_σΨ(EM::EM_model)
end




###########################################################################

#next: code up a NLLS criterion, write estimation routine and check that it works
function pred_error(n,k,r,P,EM::EM_model)
    @unpack logΨ1,logΨ0,Inv = EM
    return logΨ1[n,k,r]-logCES(Inv[:,n],logΨ0[n,k,r],P)
end


function ssq(P::CESparams,EM::EM_model)
    sumsq = 0
    @unpack wghts = EM
    for ind in CartesianIndices(wghts)
        n,k,r = Tuple(ind)
        sumsq += wghts[ind]*pred_error(n,k,r,P,EM)^2
    end
    return sumsq
end

ssq(P,EM)

# add some functions to (1) return the vector of transformed parameters given CESpars
getPars(x::Array) = CESparams(δ=exp.(x[1:2]),a=[exp.(x[3:5]);1.],γ=x[6],ρ=x[7],logθ=x[8])
# and (2): return CESpars given the vector of transformed parameters
getPars(P::CESparams) = [log.(P.δ);log.(P.a[1:3]);P.γ;P.ρ;P.logθ]
# compare these two versions of ssq and notice how we are using type-dependent dispatch

# notice again here that we have two ssq functions that rely on type-dependent dispatch
function ssq(x::Array{R,1},dat1,dat0) where R<:Real 
    return ssq(getPars(x),dat1,dat0)
end

x0 = getPars(P)

# ----- Some Monte-Carlo Simulations that should eventually go elsewhere. No need to run this! -------- #

EM = EM_model(1000,2,10);
# here is the solution with 10 simulations per observation
B = 50
Xb = zeros(8,B)
for b=1:B
    #println(b)
    draw_Ψ0_Ψ1!(EM)
    get_wght!(EM)
    res = optimize(x->ssq(getPars(x),EM),x0,LBFGS(),autodiff=:forward)
    Xb[:,b] = res.minimizer
end

[x0 mean(Xb,dims=2)]

# here is the solution with 1000 simulations per observation

EM = EM_model(2000,2,10);

# here is the solution with 10 simulations per observation
Xb2 = zeros(8,B)

for b=1:B
    #println(b)
    draw_Ψ0_Ψ1!(EM)
    get_wght!(EM)
    res = optimize(x->ssq(getPars(x),EM),x0,LBFGS(),autodiff=:forward)
    Xb2[:,b] = res.minimizer
end

[x0 mean(Xb2,dims=2)]

# ------- NEXT: Code to calculate the M-Step for F0 and MM parameters.
# - See the README file for formulae
# - Wghts can be calculated using get_wght!(EM) and stored in EM.wghts
# - All other data stored in EM

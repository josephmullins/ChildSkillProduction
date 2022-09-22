using Distributions,Optim,Random

# Four sets of parameters
# π - mixture probabilities
# β_3 - mean and covariance of each normal mixture
# β_2 - production parameters and variance of the TFP shock
# β_1 - measurement parameters

# π - assume two mixtures (K=2)

π0 = [0.5,0.5]
# β_3:
Random.seed!(1409)
C = rand(5,5)
Σ1 = C'*C
C = rand(5,5)
Σ2 = C'*C

μ = (ones(5),-ones(5)) #<- μ[k] is the mean for the kth mixture
Σ = (Σ1,Σ2)

# β_2:
ρ = -1.5
γ = -3
σ_η = 0.5
θ = 1.
a = [0.2,0.2,0.2,0.4] #<= order: a_m,a_f,a_g,a_Y
δ = [0.5,0.5]
logθ = 0.

#β_1: assume J=2
λ = [1.,0.9]
σ_ζ = [0.4,0.4]


# function to draw (ψ_0,I_1)
function initial_draw(π0,μ,Σ)
    if (rand()<π0[1])
        k=1
    else
        k=2
    end
    x = rand(MultivariateNormal(μ[k],Σ[k]))
    return x
end


x = rand(MultivariateNormal(μ[1],Σ[1]))

# function to evaluate nested CES production function 
# -- this is total human capital investment
# assume x = [τ_m,τ_f,g,Y,Ψ_0]
function logCES(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ)
    home_input = (a[1]*τ_m^ρ + a[2]*τ_f^ρ + a[3]*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + a[4]*Y^γ) + δ[2]*logΨ_0
end

function logCES(x,δ,a,γ,ρ,logθ)
    logCES(exp(x[1]),exp(x[2]),exp(x[3]),exp(x[4]),x[5],δ,a,γ,ρ,logθ)
end


function draw_logΨ1(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ,σ_η)
    return logCES(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ) + rand(Normal(0,σ_η))
end 

function draw_logΨ1(x,δ,a,γ,ρ,logθ,σ_η)
    return draw_logΨ1(exp(x[1]),exp(x[2]),exp(x[3]),exp(x[4]),x[5],δ,a,γ,ρ,logθ,σ_η)
end

##total human capital investment a child receives as a function of all inputs (investment inputs)

logΨ1 = draw_logΨ1(x,δ,a,γ,ρ,logθ,σ_η)

# function to return a 2 x 1 vector of measurements, M:
function draw_M(logΨ,λ,σ_ζ)
    return  λ*logΨ .+ rand(MultivariateNormal([0,0],σ_ζ))
end

M = draw_M(logΨ1,λ,σ_ζ)

#function to draw N observations given parameters
function draw_data(N,π0,μ,Σ) 
    dat = Array{Float64, 2}(undef, 5, N)
    for i in 1:N
        dat[:,i]=initial_draw(π0,μ,Σ)
    end
    return dat
end

N = 10000
dat0 = draw_data(N,π0,μ,Σ)

#N x 2 Array of Skills 
function draw_skills(dat0,δ,a,γ,ρ,logθ,σ_η)
    N = size(dat0)[2]
    dat = Array{Float64, 1}(undef, N)
    for i in 1:N
        @views dat[i] =draw_logΨ1(dat0[:,i],δ,a,γ,ρ,logθ,σ_η)
    end
    return dat
end

dat1 = draw_skills(dat0,δ,a,γ,ρ,logθ,σ_η);
@time dat1 = draw_skills(dat0,δ,a,γ,ρ,logθ,σ_η);

# return N x 4 array of measurements (our M)
function draw_measurements(dat1,dat0,λ,σ_ζ) 
    datM = Array{Float64, 3}(undef, 2, 2, N)
    for i in 1:N
        datM[:,1,i]=draw_M(dat0[5,i],λ,σ_ζ)
        datM[:,2,i]=draw_M(dat1[i],λ,σ_ζ)
    end
    return datM
end

datM = draw_measurements(dat1,dat0,λ,σ_ζ) 

#next: code up a NLLS criterion, write estimation routine and check that it works
function pred_error(i, δ, a,γ,ρ,logθ,dat1,dat0)
    return dat1[i]-logCES(dat0[:,i],δ,a,γ,ρ,logθ)
end


function ssq(δ,a,γ,ρ,logθ,dat1,dat0)
    sumsq = 0
    for i=1:length(dat1)
        sumsq += pred_error(i,δ, a,γ,ρ,logθ,dat1,dat0)^2
    end
    return sumsq
end

ssq(δ,a,γ,ρ,logθ,dat1,dat0)


function ssq(x,dat1,dat0)
    ssq(exp.(x[1:2]),exp.(x[3:6]),x[7],x[8],x[9],dat1,dat0)
end

x0 = [log.(δ);log.(a);γ;ρ;logθ]

ssq(x0,dat1,dat0)

@time res = optimize(x->ssq(x,dat1,dat0),x0)

@time res2 = optimize(x->ssq(x,dat1,dat0),x0,BFGS(),autodiff=:forward)

@time res3 = optimize(x->ssq(x,dat1,dat0),x0,Newton(),autodiff=:forward)


[res.minimizer res2.minimizer res3.minimizer x0]

N = 1000
B = 100
Xb = zeros(9,B)
#Xb2 = zeros(9,B)
R = zeros(Bool,B)

for b=1:B
    println(b)
    dat0 = draw_data(N,π0,μ,Σ)
    dat1 = draw_skills(dat0,δ,a,γ,ρ,logθ,0.3);
    res = optimize(x->ssq(x,dat1,dat0),x0,BFGS(),autodiff=:forward)
    Xb[:,b] = res.minimizer
    R[b] = res.ls_success
end

[x0 mean(Xb,dims=2)[:]]

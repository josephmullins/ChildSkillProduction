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

#β_1: assume J=2
λ = 0.9
σ_ζ = [0.4,0.4]


# function to draw (ψ_0,I_1)
function initial_draw(π0,μ,Σ)
    k = 1 + rand()<π0[1] #<- draw the mixture
    x = rand(MultivariateNormal(μ[k],Σ[k]))
    return x
end

# function to evaluate nested CES production function
# assume x = [τ_m,τ_f,g,Y,Ψ_0]
function logCES(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ)
    home_input = (a[1]*τ_m^ρ + a[2]*τ_f^ρ + a[3]*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + a[4]*Y^γ) + δ[2]*logΨ_0
end

function draw_logΨ1(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ,σ_η)
    return logCES(τ_m,τ_f,g,Y,logΨ_0,δ,a,γ,ρ,logθ) + rand(Normal(0,σ_η))
end

# return the vector:
function draw_M(logΨ0,logΨ1,λ,σ_ζ)

end

# function to draw N observations given parameters
# return N x 4 array of measurements, N x 2 Array of Skills, N x 4 array of inputs




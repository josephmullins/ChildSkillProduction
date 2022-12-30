include("GMMRoutines.jl")
include("moment_functions.jl")
using Distributions
using LinearAlgebra
using CSV
using DataFrames

# create the data object and the instruments
include("prep_data.jl")

# come up with an initial guess of the parameters
x0 = [-2.,-2.,0.1,0.9]
βg0 = zeros(8)
βf0 = zeros(5)
βm0 = zeros(6)
βθ0 = zeros(7)
βθf = zeros(6)
x0 = [x0;βg0;βm0;βf0;βθ0;βθf;1]
αm = [0.1,0.5,1.]
αf = [0.9,0.5,1.]

# write an update function that works with the specification we are using here
# returns a named tuple
function update(x,αm0,αf0)
    return (ay = 1, ρ=x[1],γ=x[2],αm=αm0,αf=αf0,δ = x[3:4],βg=x[5:12],βm=x[13:18],βf=x[19:23],βθ=x[24:30],βθf=x[31:36],λ=x[37])
end

lb = [[-Inf,-Inf,0.,0.];-Inf*ones(32);0.]
ub = Inf*ones(37)


# Run the gmm routine to estimate the model
gfunc(x,data,i) = gmm_inputs(update(x,1.,1.),data,Z_prod,Z_prodF,i)
gtest = gfunc(x0,data,1)
K = length(gtest) #<- get the number of moments in total
W = Matrix{Float64}(I,K,K) #<- start with the identity weighting matrix
res,se = EstimateGMMIterative(x0,gfunc,data,W,N,lb,ub,4)

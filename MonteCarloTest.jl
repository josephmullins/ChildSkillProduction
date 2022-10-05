include("EM.jl")


x0 = getPars(P)
x1 = getPars(EM.P)

B = 50 #<- number of trials in the monte-carlo
K = 2 #<- two mixtures types for initial distribution
N = 1000 #<- number observations
R = 20 #<- simulations per observation

Xb = zeros(8,B)

for b=1:B
    println(" ---- Running iteration $b ------")
    Random.seed!(1000+b)
    EM = EM_model(N,K,R)
    # (1) get estimates from E-M routine
    EMRoutine(EM,1e-3)
    # (2) save estimates for production parameters only
    Xb[:,b] = getPars(EM.P)
end

# note: set seed at 1006 and get something weird
# TODO: check what's happening here
Random.seed!(1006)
EM = EM_model(N,K,R)

# (1) check estimates from E-M routine
#EMRoutine(EM,1e-3,10)
[x0 mean(Xb,dims=2)]

break

# Does bias disappear when we increase N? Run one more trial to compare:

Xb2 = zeros(8,B)
N = 2000

for b=1:B
    println(" ---- Running iteration $b ------")
    Random.seed!(1000+b)
    EM = EM_model(N,K,R)
    # (1) get estimates from E-M routine
    EMRoutine(EM,1e-3)
    # (2) save estimates for production parameters only
    Xb2[:,b] = getPars(EM.P)
end

[x0 mean(Xb2,dims=2)]

# a little snip of code here to see that the estimates are mostly ok but occasionally go crazy
using Plots

j = 5
histogram(Xb2[j,:])
plot!([x0[j],x0[j]],[0,ylims()[2]],linewidth=2.0)

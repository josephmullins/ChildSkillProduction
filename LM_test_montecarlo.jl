include("estimation_tools.jl")
# gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked2!(update(x,spec),n,g,
# t1,p1 = LM_test(x1,sum(unrestricted),gfunc!,W,N,5,args...)

# model: y = β1*x1 + β1^2*x2 + β2*x3 + β1*β2*x4

φ(β) = [β[1],β[1]^2,β[1],β[1] * β[2]]

using Distributions, Random
β0 = [0.9,0.7]

function sim_data(β,N)
    eps = 0.5*rand(Normal(),N)
    Z = rand(Normal(),N,5)

    X = 0.5*Z[:,1:4] .+ 0.5*Z[:,5] .+ eps .+ 0.3*rand(Normal(),N,4)
    γ = φ(β) #[β[1],β[1]^2,β[2],β[1]*β[2]]
    Y = X*γ .+ eps
    return (X=X,Y=Y,Z=Z)
end

function gfunc!(γ,n,g,resids,data)
    e = data.Y[n] .- dot(data.X[n,:],γ)
    g[:] .+= e*data.Z[n,:]
end
gfunc_r!(β,n,g,resids,data) = gfunc!(φ(β),n,g,resids,data)

function boot_LM(β0,B,N)
    p_boot = zeros(2,B)
    test_boot = zeros(B)
    W = I(9)
    for b in 1:B
        dat = sim_data(β0,N)
        Z = [dat.Z rand(Normal(),N,4)]
        dat = (Y = dat.Y,X = dat.X, Z = Z)
        r1 = estimate_gmm(β0,gfunc_r!,W,N,5,dat)
        b1 = r1.est2
        p_boot[:,b] = r1.est2
        xest = φ(b1)
        W = inv(r1.Ω)
        t1,p1 = LM_test(xest,2,gfunc!,W,N,5,dat)
        test_boot[b] = t1
    end
    return test_boot,p_boot
end

tb,pb = boot_LM(β0,250,5000)

# the simulations show that the test works as intended

# dat = sim_data(β0,200)
# W = I(5)
# r1 = estimate_gmm(β0,gfunc_r!,W,200,5,dat)
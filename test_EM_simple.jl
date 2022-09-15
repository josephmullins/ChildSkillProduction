using Distributions
# this simple script tests the EM routine for estimating a gaussian mixture model
# a three dimensional mixture
C1 = [1 0 0;0.2 1 0;0.1 0.1 1]
Σ1 = C1*C1'
C2 = [1 0 0;-0.2 1 0;-0.1 -0.1 1]
Σ2 = C2*C2'
C3 = [1 0 0;0.2 1 0;-0.1 0.1 1]
Σ3 = C3*C3'

Ftrue = MixtureModel(MultivariateNormal,[(fill(-1,3),Σ1),(fill(0,3),Σ2),(fill(1,3),Σ3)],[0.2,0.4,0.4])

pars0 = [(fill(-1,3),Σ1),(fill(0,3),Σ2),(fill(1,3),Σ3)]
prior0 = [0.2,0.4,0.4]
# simulate from this true distribution:
N = 1000
X = rand(Ftrue,N)

function estep(mixpars,prior,X)
    Fest = MixtureModel(MultivariateNormal,mixpars,prior)
    N = size(X)[2]
    wghts = zeros(3,N)
    for k=1:3
        for i=1:N
            wghts[k,i] = pdf(MultivariateNormal(mixpars[k]...),X[:,i]) *prior[k]
        end
    end
    wghts ./= sum(wghts,dims=1)
    return wghts
end

estep(pars0,prior0,X)

function mstep(X,wghts)
    
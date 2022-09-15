using Parameters,ForwardDiff

#ok this seems to work

@with_kw struct params{R}
    α::Array{R,1} = fill(1.,2)
    β::R = 1.
end

@with_kw struct params2
    α::Array{Float64,1} = fill(1.,2)
    β::Float64 = 1.
end


p = params()
p2 = params2()

function test_function(params)
    @unpack α,β = params
    return α[1]^2 + α[2]^2 + β
end

function test_function2(x)
    p = params(α=x[1:2],β=x[3])
    return test_function(p)
end

function test_function3(x)
    p = params2(α=x[1:2],β=x[3])
    return test_function(p)
end


# evalute the function
test_function2([1.,1.,1.])
test_function3([1.,1.,1.])


# test that test_function2 can be used with automatic differentiation
ForwardDiff.gradient(test_function2,ones(3))

# profile test_function2 against test_function3 which uses the non parametrically-typed version
@time test_function2([1.,1.,1.]) #<- 
@time test_function3([1.,1.,1.]) #<- same number of allocations

# show that test_function3 *cannot* be used with automatic differentiation
ForwardDiff.gradient(test_function3,ones(3))


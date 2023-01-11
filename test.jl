
function f(x,g)
    a = 0
    for i=1:1000
        a += g(x)
    end
    return a
end

function f(x,g,b)
    a = 0
    for i=1:1000
        a += g(x,b)
    end
    return a
end

g(x,b) = b*x

@time f(1.,x->g(x,2.))
@time f(1.,g,2.)

# LESSON: passing anonymous function without arguments is a big slow down
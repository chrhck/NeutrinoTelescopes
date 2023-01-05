using Enzyme

function test(x)
    n = 1
    for k in 1:2
        y = falses(n)        
    end
    return x
end
Enzyme.API.printall!(true)
a = Enzyme.autodiff(Enzyme.Reverse, test, Active, Active(1E4))
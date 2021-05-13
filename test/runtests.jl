module GridapPETScTests

using Test

@time @testset "Bindings" begin include("BindingsTests.jl") end

end # module

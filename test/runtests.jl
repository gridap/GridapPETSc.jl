module GridapPETScTests

using Test

@time @testset "PETSC" begin include("PETSCTests.jl") end

end # module

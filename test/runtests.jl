module GridapPETScTests

using Test

@time @testset "PETSC" begin include("PETSCTests.jl") end

@time @testset "PETScSolvers" begin include("PETScSolversTests.jl") end

end # module

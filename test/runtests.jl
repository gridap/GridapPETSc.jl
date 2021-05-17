module GridapPETScTests

using Test

@time @testset "PETSC" begin include("PETSCTests.jl") end

@time @testset "PETScVectors" begin include("PETScVectorsTests.jl") end

@time @testset "PETScSolvers" begin include("PETScSolversTests.jl") end

@time @testset "PoissonDriver" begin include("PoissonDriver.jl") end

end # module

module GridapPETScTests

using Test

@time @testset "PETSC" begin include("PETSCTests.jl") end

@time @testset "PETScArrays" begin include("PETScArraysTests.jl") end

@time @testset "PETScSolvers" begin include("PETScSolversTests.jl") end

@time @testset "PETScAssembly" begin include("PETScAssemblyTests.jl") end

@time @testset "PoissonDriver" begin include("PoissonDriver.jl") end

@time @testset "ElasticityDriver" begin include("ElasticityDriver.jl") end

end # module

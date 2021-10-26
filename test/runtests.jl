module GridapPETScTests

using Test

@time @testset "PETSC" begin include("PETSCTests.jl") end

@time @testset "PETScArrays" begin include("PETScArraysTests.jl") end

@time @testset "PartitionedArrays (sequential)" begin include("PartitionedArraysTests_sequential.jl") end

@time @testset "PartitionedArrays (mpi)" begin include("PartitionedArraysTests_mpi.jl") end

@time @testset "PETScLinearSolvers" begin include("PETScLinearSolversTests.jl") end

@time @testset "PETScNonLinearSolvers" begin include("PETScNonLinearSolversTests.jl") end

@time @testset "PETScAssembly" begin include("PETScAssemblyTests.jl") end

@time @testset "PoissonDriver" begin include("PoissonDriver.jl") end

@time @testset "ElasticityDriver" begin include("ElasticityDriver.jl") end

end # module

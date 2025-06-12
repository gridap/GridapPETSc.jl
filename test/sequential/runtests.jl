module GridapPETScSequentialTests

using Test
using MPI

@time @testset "PartitionedArrays (sequential)" begin include("PartitionedArraysTests.jl") end

@time @testset "PoissonTests" begin include("PoissonTests.jl") end

@time @testset "PLaplacianTests" begin include("PLaplacianTests.jl") end

if MPI.Initialized()
    MPI.Finalize()
end

end # module
module GridapPETScMPITests
using Test
using MPI

if MPI.Initialized()
    MPI.Finalize()
end

@time @testset "PartitionedArrays" begin include("PartitionedArraysTestsRun.jl") end
@time @testset "PLaplacianTests" begin include("PLaplacianTestsRun.jl") end
@time @testset "GCTests" begin include("GCTestsRun.jl") end
@time @testset "PoissonTests" begin include("PoissonTestsRun.jl") end
@time @testset "DarcyTests" begin include("DarcyTestsRun.jl") end

end # module

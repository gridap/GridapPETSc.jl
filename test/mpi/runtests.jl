module GridapPETScMPITests
using Test
@time @testset "PartitionedArrays" begin include("PartitionedArraysTestsRun.jl") end
@time @testset "PLaplacianTests" begin include("PLaplacianTestsRun.jl") end
@time @testset "GCTests" begin include("GCTestsRun.jl") end
end # module

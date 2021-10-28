module GridapPETScMPITests
using Test
@time @testset "PartitionedArrays" begin include("PartitionedArraysTestsRun.jl") end
@time @testset "PLaplacianTests" begin include("PLaplacianTestsRun.jl") end
end # module

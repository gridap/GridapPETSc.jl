module PartitionedArraysTests
include("../PartitionedArraysTests.jl")
nparts = 3
prun(partitioned_tests,sequential,nparts)
end # module

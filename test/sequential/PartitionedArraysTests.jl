module PartitionedArraysTests
include("../PartitionedArraysTests.jl")
nparts = 3
with_debug() do distribute
  partitioned_tests(distribute,nparts)
end
end # module

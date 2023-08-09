include("../PartitionedArraysTests.jl")
nparts = 3
with_mpi() do distribute
  partitioned_tests(distribute,nparts)
end

include("../PartitionedArraysTests.jl")
nparts = 3
with_mpi() do distribute
  parts = distribute(LinearIndices((prod(nparts),)))
  partitioned_tests(parts)
end

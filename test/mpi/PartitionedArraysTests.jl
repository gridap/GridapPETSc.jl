include("../PartitionedArraysTests.jl")
nparts = 3
prun(partitioned_tests,mpi,nparts)

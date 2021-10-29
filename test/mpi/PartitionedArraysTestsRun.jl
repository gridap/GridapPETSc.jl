module PartitionedArraysTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=3,file="PartitionedArraysTests.jl")
end # module

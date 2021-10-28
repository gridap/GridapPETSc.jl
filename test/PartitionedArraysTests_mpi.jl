module PartitionedArraysTests_mpi

include("mpiexec.jl")
run_mpi_driver(procs=3,file="PartitionedArraysTests_mpi_driver.jl")

end # module

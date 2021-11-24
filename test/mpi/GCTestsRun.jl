module GCTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=4,file="GCTests.jl")
end # module

module DarcyTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=4,file="DarcyTests.jl")
end # module

module PoissonTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=4,file="PoissonTests.jl")
end # module

module PLaplacianTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=2,file="PLaplacianTests.jl")
end # module

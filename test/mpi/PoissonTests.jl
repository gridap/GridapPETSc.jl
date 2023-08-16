include("../PoissonTests.jl")
nparts = (2,2)
with_mpi() do distribute
  main(distribute,nparts)
end

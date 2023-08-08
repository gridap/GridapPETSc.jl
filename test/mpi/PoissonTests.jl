include("../PoissonTests.jl")
nparts = (2,2)
with_mpi() do distribute
  parts = distribute(LinearIndices((prod(nparts),)))
  main(parts)
end

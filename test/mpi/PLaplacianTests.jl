include("../PLaplacianTests.jl")
nparts = (2,1)
with_mpi() do distribute
  parts = distribute(LinearIndices((prod(nparts),)))
  main(parts)
end
with_mpi() do distribute
  parts = distribute(LinearIndices((prod(nparts),)))
  main(parts)
end

include("../PLaplacianTests.jl")
nparts = (2,1)
with_mpi() do distribute
  main(distribute,nparts)
end
with_mpi() do distribute
  main(distribute,nparts)
end

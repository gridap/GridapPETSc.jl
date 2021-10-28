
include("PLaplacianDriver.jl")

nparts = (2,1)
prun(main,mpi,nparts)

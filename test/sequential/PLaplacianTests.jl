module PLaplacianTests
include("../PLaplacianTests.jl")
nparts = (2,2)
prun(main,sequential,nparts)
end # module

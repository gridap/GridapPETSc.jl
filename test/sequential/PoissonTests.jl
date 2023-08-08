module PoissonTests
include("../PoissonTests.jl")
nparts = (2,2)
with_debug() do distribute
  main(distribute,nparts)
end
end # module

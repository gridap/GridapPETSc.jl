module GridapPETScTests

using Test

@time @testset "SEQUENTIAL" begin include("sequential/runtests.jl") end
@time @testset "MPI" begin include("mpi/runtests.jl") end

end # module

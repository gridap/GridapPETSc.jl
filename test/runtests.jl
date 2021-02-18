using GridapPETSc
using Test

# Skip tests if library was not properly loaded
if GridapPETSc.PETSC_LOADED[]
    using MPI
    if !MPI.Initialized()
        MPI.Init()
    end

    # @testset "PETSc tests" begin include("PETSc.jl") end
    # @testset "Linear Solver tests" begin include("LinearSolver.jl") end
    @testset "FEM driver" begin include("femdriver.jl") end

    if MPI.Initialized() & !MPI.Finalized()
        MPI.Finalize()
    end
else
    @warn   """
            PETSc library is not properly loaded.
            GridapPETSc tests are not going to be performed.
            """
end

using GridapPETSc
using MPI
using Test

if !MPI.Initialized()
    MPI.Init()
end

@testset "PETSc tests" begin include("PETSc.jl") end
@testset "Linear Solver tests" begin include("LinearSolver.jl") end

if MPI.Initialized() & !MPI.Finalized()
    MPI.Finalize()
end

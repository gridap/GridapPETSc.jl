
struct PETScLinearSolver{F} <: LinearSolver
  setup::F
  comm::MPI.Comm
end

ksp_from_options(ksp) = @check_error_code PETSC.KSPSetFromOptions(ksp[])

function PETScLinearSolver(comm::MPI.Comm)
  PETScLinearSolver(ksp_from_options,comm)
end

PETScLinearSolver() = PETScLinearSolver(MPI.COMM_WORLD)

PETScLinearSolver(setup::Function) = PETScLinearSolver(setup,MPI.COMM_WORLD)

struct PETScLinearSolverSS{F} <: SymbolicSetup
  solver::PETScLinearSolver{F}
end

function Algebra.symbolic_setup(solver::PETScLinearSolver,mat::AbstractMatrix)
  PETScLinearSolverSS(solver)
end

mutable struct PETScLinearSolverNS{T} <: NumericalSetup
  A::T
  B::PETScMatrix
  comm::MPI.Comm
  ksp::Ref{KSP}
  initialized::Bool
  function PETScLinearSolverNS(A,B::PETScMatrix,comm::MPI.Comm)
    T=typeof(A)
    new{T}(A,B,comm,Ref{KSP}(),false)
  end
end

function Init(a::PETScLinearSolverNS)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize,a)
end

function Finalize(ns::PETScLinearSolverNS)
  if ns.initialized && GridapPETSc.Initialized()
    @check_error_code PETSC.PetscObjectRegisterDestroy(ns.ksp[].ptr)
    ns.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function Algebra.numerical_setup(ss::PETScLinearSolverSS,A::AbstractMatrix)
  B = convert(PETScMatrix,A)
  ns = PETScLinearSolverNS(A,B,ss.solver.comm)
  @check_error_code PETSC.KSPCreate(ns.comm,ns.ksp)
  ss.solver.setup(ns.ksp)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  Init(ns)
end

function Algebra.solve!(x::PETScVector,ns::PETScLinearSolverNS,b::AbstractVector)
  B = convert(PETScVector,b)
  @check_error_code PETSC.KSPSolve(ns.ksp[],B.vec[],x.vec[])
  x
end

function Algebra.solve!(x::Vector{PetscScalar},ns::PETScLinearSolverNS,b::AbstractVector)
  X = convert(PETScVector,x)
  solve!(X,ns,b)
  x
end

function Algebra.solve!(x::AbstractVector,ns::PETScLinearSolverNS,b::AbstractVector)
  X = convert(Vector{PetscScalar},x)
  solve!(X,ns,b)
  x .= X
  x
end

function Algebra.solve!(x::PVector,ns::PETScLinearSolverNS,b::PVector)
  X = similar(b,(axes(ns.A)[2],))
  B = similar(b,(axes(ns.A)[2],))
  copy!(X,x)
  copy!(B,b)
  Y = convert(PETScVector,X)
  solve!(Y,ns,B)
  copy!(x,Y)
end

function Algebra.numerical_setup!(ns::PETScLinearSolverNS,A::AbstractMatrix)
  ns.A = A
  ns.B = convert(PETScMatrix,A)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  ns
end

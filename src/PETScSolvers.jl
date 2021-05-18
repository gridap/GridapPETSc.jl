
struct PETScSolver{F} <: LinearSolver
  setup::F
  comm::MPI.Comm
end

from_options(ksp) = @check_error_code PETSC.KSPSetFromOptions(ksp[])

function PETScSolver(comm::MPI.Comm)
  PETScSolver(from_options,comm)
end

PETScSolver() = PETScSolver(MPI.COMM_WORLD)

PETScSolver(setup::Function) = PETScSolver(setup,MPI.COMM_WORLD)

struct PETScSolverSS{F} <: SymbolicSetup
  solver::PETScSolver{F}
end

function Algebra.symbolic_setup(solver::PETScSolver,mat::AbstractMatrix)
  PETScSolverSS(solver)
end

mutable struct PETScSolverNS <: NumericalSetup
  A::PETScMatrix
  comm::MPI.Comm
  ksp::Ref{KSP}
  initialized::Bool
  function PETScSolverNS(A::PETScMatrix,comm::MPI.Comm)
    new(A,comm,Ref{KSP}(),false)
  end
end

function Init(a::PETScSolverNS)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize,a)
end

function Finalize(ns::PETScSolverNS)
  if ns.initialized && GridapPETSc.Initialized()
    @check_error_code PETSC.KSPDestroy(ns.ksp)
    ns.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function Algebra.numerical_setup(ss::PETScSolverSS,A::AbstractMatrix)
  B = convert(PETScMatrix,A)
  ns = PETScSolverNS(B,ss.solver.comm)
  @check_error_code PETSC.KSPCreate(ns.comm,ns.ksp)
  ss.solver.setup(ns.ksp)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  Init(ns)
end

function Algebra.solve!(x::PETScVector,ns::PETScSolverNS,b::AbstractVector)
  B = convert(PETScVector,b)
  @check_error_code PETSC.KSPSolve(ns.ksp[],B.vec[],x.vec[])
  x
end

function Algebra.solve!(x::Vector{PetscScalar},ns::PETScSolverNS,b::AbstractVector)
  X = convert(PETScVector,x)
  solve!(X,ns,b)
  x
end

function Algebra.solve!(x::AbstractVector,ns::PETScSolverNS,b::AbstractVector)
  X = convert(Vector{PetscScalar},x)
  solve!(X,ns,b)
  x .= X
  x
end

function Algebra.numerical_setup!(ns::PETScSolverNS,A::AbstractMatrix)
  ns.A = convert(PETScMatrix,A)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  ns
end


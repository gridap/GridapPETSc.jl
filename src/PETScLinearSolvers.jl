
struct PETScLinearSolver{F} <: LinearSolver
  setup::F
end

ksp_from_options(ksp) = @check_error_code PETSC.KSPSetFromOptions(ksp[])

function PETScLinearSolver()
  PETScLinearSolver(ksp_from_options)
end

struct PETScLinearSolverSS{F} <: SymbolicSetup
  solver::PETScLinearSolver{F}
end

function Algebra.symbolic_setup(solver::PETScLinearSolver,mat::AbstractMatrix)
  PETScLinearSolverSS(solver)
end

mutable struct PETScLinearSolverNS <: NumericalSetup
  A::PETScMatrix
  X::PETScVector
  B::PETScVector
  ksp::Ref{KSP}
  initialized::Bool
  function PETScLinearSolverNS(A::PETScMatrix,X::PETScVector,B::PETScVector)
    new(A,X,B,Ref{KSP}(),false)
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
    if ns.A.comm == MPI.COMM_SELF
      @check_error_code PETSC.KSPDestroy(ns.ksp)
    else
      @check_error_code PETSC.PetscObjectRegisterDestroy(ns.ksp[].ptr)
    end
    ns.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function Algebra.numerical_setup(ss::PETScLinearSolverSS,_A::AbstractMatrix)
  A = convert(PETScMatrix,_A)
  X = convert(PETScVector,allocate_in_domain(_A))
  B = convert(PETScVector,allocate_in_domain(_A))
  ns = PETScLinearSolverNS(A,X,B)
  @check_error_code PETSC.KSPCreate(A.comm,ns.ksp)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  ss.solver.setup(ns.ksp)
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  Init(ns)
end

function Algebra.solve!(x::AbstractVector{PetscScalar},ns::PETScLinearSolverNS,b::AbstractVector{PetscScalar})
  X, B = ns.X, ns.B
  copy!(B,b)
  @check_error_code PETSC.KSPSolve(ns.ksp[],B.vec[],X.vec[])
  copy!(x,X)
  return x
end

function Algebra.numerical_setup!(ns::PETScLinearSolverNS,A::AbstractMatrix)
  ns.A = convert(PETScMatrix,A)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  ns
end

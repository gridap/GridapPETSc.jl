
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

mutable struct PETScLinearSolverNS{T} <: NumericalSetup
  A::T
  B::PETScMatrix
  ksp::Ref{KSP}
  initialized::Bool
  function PETScLinearSolverNS(A,B::PETScMatrix)
    T=typeof(A)
    new{T}(A,B,Ref{KSP}(),false)
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
    if ns.B.comm == MPI.COMM_SELF
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

function Algebra.numerical_setup(ss::PETScLinearSolverSS,A::AbstractMatrix)
  B = convert(PETScMatrix,A)
  ns = PETScLinearSolverNS(A,B)
  @check_error_code PETSC.KSPCreate(B.comm,ns.ksp)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  ss.solver.setup(ns.ksp)
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  Init(ns)
end

function Algebra.solve!(x::PETScVector,ns::PETScLinearSolverNS,b::PETScVector)
  @check_error_code PETSC.KSPSolve(ns.ksp[],b.vec[],x.vec[])
  return x
end

function Algebra.solve!(x::PETScVector,ns::PETScLinearSolverNS,b::AbstractVector)
  # Jordi: Somehow, I think this destroys PETSc objects that are
  # still in use. This then leads to a PETSc error 62 when calling KSPSolve.
  # Instead, I have added GridapPETSc.Finalize(...) calls for the specific PETSc
  # objects that we are creating internally.
  #
  # if (x.comm != MPI.COMM_SELF)
  #   gridap_petsc_gc() # Do garbage collection of PETSc objects
  # end

  B = convert(PETScVector,b)
  solve!(x,ns,B)
  GridapPETSc.Finalize(B)
  return x
end

function Algebra.solve!(x::AbstractVector,ns::PETScLinearSolverNS,b::AbstractVector)
  X = convert(PETScVector,x)
  solve!(X,ns,b)
  copy!(x,X)
  GridapPETSc.Finalize(X)
  return x
end

# When x is a Vector{PetscScalar}, the memory is aliased with the PETSc Vec object, i.e
# we do not need to copy the data back into x.
function Algebra.solve!(x::Vector{PetscScalar},ns::PETScLinearSolverNS,b::AbstractVector)
  X = convert(PETScVector,x)
  solve!(X,ns,b)
  return x
end

# In the case of PVectors, we need to ensure that ghost layouts match. In the case they
# do not, we have to create a new vector and copy (which is less efficient, but necessary).
function Algebra.solve!(x::PVector,ns::PETScLinearSolverNS,b::PVector)
  rows, cols = axes(ns.A)
  if partition(axes(x,1)) !== partition(cols)
    y = pzeros(PetscScalar,partition(cols))
    copy!(y,x)
  else
    y = x
  end
  if partition(axes(b,1)) !== partition(rows)
    c = pzeros(PetscScalar,partition(rows))
    copy!(c,b)
  else
    c = b
  end

  X = convert(PETScVector,y)
  solve!(X,ns,c)
  copy!(x,X)
  GridapPETSc.Finalize(X)
  return x
end

function Algebra.numerical_setup!(ns::PETScLinearSolverNS,A::AbstractMatrix)
  ns.A = A
  ns.B = convert(PETScMatrix,A)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  return ns
end

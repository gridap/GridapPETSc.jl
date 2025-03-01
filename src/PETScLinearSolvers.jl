
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
  x
end

function Algebra.solve!(x::PETScVector,ns::PETScLinearSolverNS,b::AbstractVector)
  if (x.comm != MPI.COMM_SELF)
    gridap_petsc_gc() # Do garbage collection of PETSc objects
  end

  B = convert(PETScVector,b)
  solve!(x,ns,B)
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
  x
end

# NOTE: 
# We previously threw away PETSc's matrix `ns.B`, and re-set the KSP object (commented code). 
# This is not only unnecessary, but can also cause some issues with MUMPS. 
# I am not completely sure why, but here are some notes on the issue: 
#   - It is matrix-dependent, and only happens in parallel (nprocs > 1). 
#   - It has to do with the re-use of the symmetric permutation created by MUMPS to 
#     find the pivots.
# I think it probably re-orders the matrix internally, and does not re-order it again when we swap it
# using `KSPSetOperators`. So when accessing the new (non-permuted) matrix using the old permutation, 
# it throws a stack overflow error.
# In fact, when updating the SNES setups in the nonlinear solvers, we do not re-set the matrix,
# but use `copy!` instead.
#function Algebra.numerical_setup!(ns::PETScLinearSolverNS,A::AbstractMatrix)
#  # ns.A = A
#  # ns.B = convert(PETScMatrix,A)
#  # @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
#  @assert nnz(ns.A) == nnz(A) # This is weak, but it might catch some errors
#  copy!(ns.B,A)
#  @check_error_code PETSC.KSPSetUp(ns.ksp[])
#  ns
#end

function Algebra.numerical_setup!(ns::PETScLinearSolverNS,A::AbstractMatrix)
  if ns.A === A
    copy!(ns.B,A)
    @check_error_code PETSC.KSPSetUp(ns.ksp[])
  else
    ns.A = A
    ns.B = convert(PETScMatrix,A)
    @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
    @check_error_code PETSC.KSPSetUp(ns.ksp[])
  end
  ns
end
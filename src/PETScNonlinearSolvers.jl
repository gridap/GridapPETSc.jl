
mutable struct PETScNonlinearSolver{F} <: NonlinearSolver
  setup::F
  comm::MPI.Comm
end

mutable struct PETScNonlinearSolverCache{A,B,C,D}
  initialized::Bool
  snes::Ref{SNES}
  op::NonlinearOperator

  # Julia LA data structures
  x_julia_vec::A
  res_julia_vec::A
  jac_julia_mat_A::B
  jac_julia_mat_P::B

  # Counterpart PETSc data structures
  x_petsc_vec::C
  res_petsc_vec::C
  jac_petsc_mat_A::D
  jac_petsc_mat_P::D

  function PETScNonlinearSolverCache(snes::Ref{SNES}, op::NonlinearOperator,
                                     x_julia_vec::A,res_julia_vec::A,
                                     jac_julia_mat_A::B,jac_julia_mat_P::B,
                                     x_petsc_vec::C,res_petsc_vec::C,
                                     jac_petsc_mat_A::D,jac_petsc_mat_P::D) where {A,B,C,D}

      cache=new{A,B,C,D}(true, snes, op,
                         x_julia_vec, res_julia_vec, jac_julia_mat_A, jac_julia_mat_P,
                         x_petsc_vec, res_petsc_vec, jac_petsc_mat_A, jac_petsc_mat_P)
      finalizer(Finalize,cache)
  end
end

function snes_residual(csnes::Ptr{Cvoid},
                       cx::Ptr{Cvoid},
                       cfx::Ptr{Cvoid},
                       ctx::Ptr{Cvoid})::PetscInt
  cache  = unsafe_pointer_to_objref(ctx)

  # 1. Transfer cx to Julia data structures
  copy!(cache.x_julia_vec, Vec(cx))

  # 2. Evaluate residual into Julia data structures
  residual!(cache.res_julia_vec, cache.op, cache.x_julia_vec)

  # 3. Transfer Julia residual to PETSc residual (cfx)
  copy!(Vec(cfx), cache.res_julia_vec)

  return PetscInt(0)
end

function snes_jacobian(csnes:: Ptr{Cvoid},
                       cx   :: Ptr{Cvoid},
                       cA   :: Ptr{Cvoid},
                       cP   :: Ptr{Cvoid},
                       ctx::Ptr{Cvoid})::PetscInt

  cache = unsafe_pointer_to_objref(ctx)

  # 1. Transfer cx to Julia data structures
  #    Extract pointer to array of values out of cx and put it in a PVector
  copy!(cache.x_julia_vec, Vec(cx))

  # 2.
  jacobian!(cache.jac_julia_mat_A,cache.op,cache.x_julia_vec)

  # 3. Transfer nls.jac_julia_mat_A/P to PETSc (cA/cP)
  copy!(Mat(cA), cache.jac_julia_mat_A)

  return PetscInt(0)
end

function Finalize(cache::PETScNonlinearSolverCache)
  if GridapPETSc.Initialized() && cache.initialized
     GridapPETSc.Finalize(cache.x_petsc_vec)
     GridapPETSc.Finalize(cache.res_petsc_vec)
     GridapPETSc.Finalize(cache.jac_petsc_mat_A)
     if !(cache.jac_petsc_mat_P === cache.jac_petsc_mat_A)
       GridapPETSc.Finalize(cache.jac_petsc_mat_P)
     end
     @check_error_code PETSC.SNESDestroy(cache.snes)
     cache.initialized=false
  end
end

snes_from_options(snes) = @check_error_code PETSC.SNESSetFromOptions(snes[])

function PETScNonlinearSolver(comm::MPI.Comm)
  PETScNonlinearSolver(snes_from_options,comm)
end

PETScNonlinearSolver(setup::Function) = PETScNonlinearSolver(setup,MPI.COMM_WORLD)
PETScNonlinearSolver() = PETScNonlinearSolver(MPI.COMM_WORLD)

function _set_petsc_residual_function!(nls::PETScNonlinearSolver, cache)
  ctx = pointer_from_objref(cache)
  fptr = @cfunction(snes_residual, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
  PETSC.SNESSetFunction(cache.snes[],cache.res_petsc_vec.vec[],fptr,ctx)
end

function _set_petsc_jacobian_function!(nls::PETScNonlinearSolver, cache)
  ctx = pointer_from_objref(cache)
  fptr = @cfunction(snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(cache.snes[],cache.jac_petsc_mat_A.mat[],cache.jac_petsc_mat_A.mat[],fptr,ctx)
end

function _setup_cache(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator)

  res_julia_vec, jac_julia_mat_A = residual_and_jacobian(op,x)
  res_petsc_vec   = convert(PETScVector,res_julia_vec)
  jac_petsc_mat_A = convert(PETScMatrix,jac_julia_mat_A)

  # In a parallel MPI context, x is a vector with a data layout typically different from
  # the one of res_julia_vec. On the one hand, x holds the free dof values of a FE
  # Function, and thus has the data layout of the FE space (i.e., local DOFs
  # include all DOFs touched by local cells, i.e., owned and ghost cells).
  # On the other hand, res_petsc_vec has the data layout of the rows of the
  # distributed linear system (e.g., local DoFs only include those touched from owned
  # cells/facets during assembly, assuming the SubAssembledRows strategy).
  # The following lines of code generate a version of x, namely, x_julia_vec, with the
  # same data layout as the columns of jac_julia_mat_A, but the contents of x
  # (for the owned dof values).
  x_julia_vec = similar(res_julia_vec,eltype(res_julia_vec),(axes(jac_julia_mat_A)[2],))
  copy!(x_julia_vec,x)
  x_petsc_vec = convert(PETScVector,x_julia_vec)

  snes_ref=Ref{SNES}()
  @check_error_code PETSC.SNESCreate(nls.comm,snes_ref)
  nls.setup(snes_ref)

  PETScNonlinearSolverCache(snes_ref, op, x_julia_vec,res_julia_vec,
                           jac_julia_mat_A,jac_julia_mat_A,
                           x_petsc_vec,res_petsc_vec,
                           jac_petsc_mat_A, jac_petsc_mat_A)
end

# Helper private functions to implement the solve! methods below.
# It allows to execute the solve! methods below in a serial context, i.e.,
# whenever
function _myexchange!(x::AbstractVector)
  x
end
function _myexchange!(x::PVector)
  exchange!(x)
end

function _copy_and_exchange!(a::AbstractVector,b::PETScVector)
  copy!(a,b.vec[])
  _myexchange!(a)
end


function Algebra.solve!(x::T,
                        nls::PETScNonlinearSolver,
                        op::NonlinearOperator,
                        cache::PETScNonlinearSolverCache{<:T}) where T <: AbstractVector

  @assert cache.op === op
  @check_error_code PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc_vec.vec[])
  _copy_and_exchange!(x,cache.x_petsc_vec)
  cache
end

function Algebra.solve!(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator)
  cache=_setup_cache(x,nls,op)
  _set_petsc_residual_function!(nls,cache)
  _set_petsc_jacobian_function!(nls,cache)
  @check_error_code PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc_vec.vec[])
  _copy_and_exchange!(x,cache.x_petsc_vec)
  cache
end


mutable struct PETScNonlinearSolver{F} <: NonlinearSolver
  setup::F
  fieldsplit::Union{PETScFieldSplit, Nothing}
end

mutable struct PETScNonlinearSolverCache{A,B,C,D,E}
  initialized::Bool
  comm::MPI.Comm
  snes::Ref{SNES}
  op::NonlinearOperator

  # The input vector to solve!
  x_fe_space_layout::A

  # Julia LA data structures
  x_sys_layout::B
  res_sys_layout::B
  jac_mat_A::C
  jac_mat_P::C

  # Counterpart PETSc data structures
  x_petsc::D
  res_petsc::D
  jac_petsc_mat_A::E
  jac_petsc_mat_P::E

  function PETScNonlinearSolverCache(comm::MPI.Comm, snes::Ref{SNES}, op::NonlinearOperator,
                                     x_fe_space_layout::A,
                                     x_sys_layout::B, res_sys_layout::B,
                                     jac_mat_A::C, jac_mat_P::C,
                                     x_petsc::D, res_petsc::D,
                                     jac_petsc_mat_A::E, jac_petsc_mat_P::E) where {A,B,C,D,E}
      cache=new{A,B,C,D,E}(true, comm,
                         snes, op,
                         x_fe_space_layout,
                         x_sys_layout, res_sys_layout,
                         jac_mat_A, jac_mat_P,
                         x_petsc, res_petsc,
                         jac_petsc_mat_A, jac_petsc_mat_P)

      @assert Threads.threadid() == 1
      _NREFS[] += 1
      finalizer(Finalize,cache)
   end
end


function snes_residual(csnes::Ptr{Cvoid},
                       cx::Ptr{Cvoid},
                       cfx::Ptr{Cvoid},
                       ctx::Ptr{Cvoid})::PetscInt
  cache  = unsafe_pointer_to_objref(ctx)

  # 1. Transfer cx to Julia data structures
  _copy!(cache.x_sys_layout, Vec(cx))
  copy!(cache.x_fe_space_layout,cache.x_sys_layout)

  # 2. Evaluate residual into Julia data structures
  residual!(cache.res_sys_layout, cache.op, cache.x_fe_space_layout)

  # 3. Transfer Julia residual to PETSc residual (cfx)
  _copy!(Vec(cfx), cache.res_sys_layout)

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
  _copy!(cache.x_sys_layout, Vec(cx))
  copy!(cache.x_fe_space_layout,cache.x_sys_layout)

  # 2. Evaluate Jacobian into Julia data structures
  jacobian!(cache.jac_mat_A,cache.op,cache.x_fe_space_layout)

  # 3. Transfer nls.jac_mat_A/P to PETSc (cA/cP)
  _copy!(Mat(cA), cache.jac_mat_A)

  return PetscInt(0)
end

function Finalize(cache::PETScNonlinearSolverCache)
  if GridapPETSc.Initialized() && cache.initialized
     GridapPETSc.Finalize(cache.x_petsc)
     GridapPETSc.Finalize(cache.res_petsc)
     GridapPETSc.Finalize(cache.jac_petsc_mat_A)
     if !(cache.jac_petsc_mat_P === cache.jac_petsc_mat_A)
       GridapPETSc.Finalize(cache.jac_petsc_mat_P)
     end
     if cache.comm == MPI.COMM_SELF
       @check_error_code PETSC.SNESDestroy(cache.snes)
     else
       @check_error_code PETSC.PetscObjectRegisterDestroy(cache.snes[].ptr)
     end
     @assert Threads.threadid() == 1
     cache.initialized=false
     _NREFS[] -= 1
  end
  nothing
end

snes_from_options(snes) = @check_error_code PETSC.SNESSetFromOptions(snes[])


function PETScNonlinearSolver(SplitField::PETScFieldSplit)
  PETScNonlinearSolver(snes_from_options,SplitField)
end

function PETScNonlinearSolver(snes_options)
  PETScNonlinearSolver(snes_options,nothing)
end


function PETScNonlinearSolver()
  PETScNonlinearSolver(snes_from_options,nothing)
end

function _set_petsc_residual_function!(nls::PETScNonlinearSolver, cache)
  ctx = pointer_from_objref(cache)
  fptr = @cfunction(snes_residual, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
  PETSC.SNESSetFunction(cache.snes[],cache.res_petsc.vec[],fptr,ctx)
end

function _set_petsc_jacobian_function!(nls::PETScNonlinearSolver, cache)
  ctx = pointer_from_objref(cache)
  fptr = @cfunction(snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(cache.snes[],cache.jac_petsc_mat_A.mat[],cache.jac_petsc_mat_A.mat[],fptr,ctx)
end

function _setup_cache(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator)

  res_sys_layout, jac_mat_A = residual_and_jacobian(op,x)
  res_petsc   = convert(PETScVector,res_sys_layout)
  jac_petsc_mat_A = convert(PETScMatrix,jac_mat_A)

  # In a parallel MPI context, x is a vector with a data layout typically different from
  # the one of res_sys_layout. On the one hand, x holds the free dof values of a FE
  # Function, and thus has the data layout of the FE space (i.e., local DOFs
  # include all DOFs touched by local cells, i.e., owned and ghost cells).
  # On the other hand, res_petsc has the data layout of the rows of the
  # distributed linear system (e.g., local DoFs only include those touched from owned
  # cells/facets during assembly, assuming the SubAssembledRows strategy).
  # The following lines of code generate a version of x, namely, x_sys_layout, with the
  # same data layout as the columns of jac_mat_A, but the contents of x
  # (for the owned dof values).
  x_sys_layout = similar(res_sys_layout,eltype(res_sys_layout),(axes(jac_mat_A)[2],))
  copy!(x_sys_layout,x)
  x_petsc = convert(PETScVector,x_sys_layout)

  snes_ref=Ref{SNES}()
  @check_error_code PETSC.SNESCreate(jac_petsc_mat_A.comm,snes_ref)

  PETScNonlinearSolverCache(jac_petsc_mat_A.comm, snes_ref, op, x, x_sys_layout, res_sys_layout,
                           jac_mat_A, jac_mat_A,
                           x_petsc, res_petsc,
                           jac_petsc_mat_A, jac_petsc_mat_A)
end

function Algebra.solve!(x::T,
                        nls::PETScNonlinearSolver,
                        op::NonlinearOperator,
                        cache::PETScNonlinearSolverCache{<:T}) where T <: AbstractVector

  #@assert cache.op === op
  cache.op = op
  @check_error_code PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end

function Algebra.solve!(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator,::Nothing)
  cache=_setup_cache(x,nls,op)

  if (cache.comm != MPI.COMM_SELF)
    gridap_petsc_gc() # Do garbage collection of PETSc objects
  end

  # set petsc residual function
  ctx  = pointer_from_objref(cache)
  fptr = @cfunction(snes_residual, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
  PETSC.SNESSetFunction(cache.snes[],cache.res_petsc.vec[],fptr,ctx)

  # set petsc jacobian function
  fptr = @cfunction(snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(cache.snes[],cache.jac_petsc_mat_A.mat[],cache.jac_petsc_mat_A.mat[],fptr,ctx)

  nls.setup(cache.snes)

  if typeof(nls.fieldsplit) == PETScFieldSplit
    set_fieldsplit(cache.snes, nls.fieldsplit)
  end
  
  @check_error_code PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end

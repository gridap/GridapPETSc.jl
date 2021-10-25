
mutable struct PETScNonlinearSolver <: NonlinearSolver
  comm::MPI.Comm
  snes::Ref{SNES}

  op

  # Julia LA data structures
  x_julia_vec
  res_julia_vec
  jac_julia_mat_A
  jac_julia_mat_P

  # Counterpart PETSc data structures
  x_petsc_vec
  res_petsc_vec
  jac_petsc_mat_A
  jac_petsc_mat_P
end

function snes_residual(csnes::Ptr{Cvoid},
                       cx::Ptr{Cvoid},
                       cfx::Ptr{Cvoid},
                       ctx::Ptr{Cvoid})::PetscInt
  nls  = unsafe_pointer_to_objref(ctx)
  x    = Vec(cx)
  fx   = Vec(cfx)

  # 1. Transfer cx to Julia data structures
  #    Extract pointer to array of values out of cx and put it in a PVector
  # xxx

  # 2. Evaluate residual into Julia data structures
  residual!(nls.res_julia_vec, nls.op, nls.x_julia_vec)

  # 3. Transfer Julia residual to PETSc residual (cfx)
  #
  # xxx

  return PetscInt(0)
end

function snes_jacobian(csnes:: Ptr{Cvoid},
                       cx   :: Ptr{Cvoid},
                       cA   :: Ptr{Cvoid},
                       cP   :: Ptr{Cvoid},
                       ctx::Ptr{Cvoid})::PetscInt

  nls = unsafe_pointer_to_objref(ctx)

  # 1. Transfer cx to Julia data structures
  #    Extract pointer to array of values out of cx and put it in a PVector
  # xxx

  # 2.
  jacobian!(nls.jac_julia_mat_A,nls.op,nls.x_julia_vec)

  # 3. Transfer nls.jac_julia_mat_A/P to PETSc (cA/cP)
  # xxx

  return PetscInt(0)
end

function PETScNonlinearSolver(setup,comm)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  snes_ref=Ref{SNES}()
  @check_error_code PETSC.SNESCreate(comm,snes_ref)
  setup(snes_ref)
  snes=PETScNonlinearSolver(comm, snes_ref,
                            nothing,
                            nothing,nothing,nothing,nothing,
                            nothing,nothing,nothing,nothing)
  finalizer(Finalize, snes)
  snes
end

function Finalize(nls::PETScNonlinearSolver)
  if GridapPETSc.Initialized()
    @check_error_code PETSC.SNESDestroy(nls.snes)
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end
snes_from_options(snes) = @check_error_code PETSC.SNESSetFromOptions(snes[])

function PETScNonlinearSolver(comm::MPI.Comm)
  PETScNonlinearSolver(snes_from_options,comm)
end

PETScNonlinearSolver(setup::Function) = PETScNonlinearSolver(setup,MPI.COMM_WORLD)
PETScNonlinearSolver() = PETScNonlinearSolver(MPI.COMM_WORLD)

function _set_petsc_residual_function!(nls::PETScNonlinearSolver)
  ctx = pointer_from_objref(nls)
  fptr = @cfunction(snes_residual, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
  PETSC.SNESSetFunction(nls.snes[],nls.res_petsc_vec.vec[],fptr,ctx)
end

function _set_petsc_jacobian_function!(nls::PETScNonlinearSolver)
  ctx = pointer_from_objref(nls)
  fptr = @cfunction(snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(nls.snes[],nls.jac_petsc_mat_A.mat[],nls.jac_petsc_mat_A.mat[],fptr,ctx)
end

function _setup_before_solve!(x,nls::PETScNonlinearSolver,op)
  # TO-DO: allocate_residual_and_jacobian
  res_julia_vec, jac_julia_mat_A = residual_and_jacobian(op,x)


   # Does this perform a copy?
   res_petsc_vec   = PETScVector(res_julia_vec)
   x_petsc_vec     = PETScVector(x)
   jac_petsc_mat_A = PETScMatrix(jac_julia_mat_A)
   println(jac_julia_mat_A)
   PETSC.@check_error_code PETSC.MatView(jac_petsc_mat_A.mat[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)

   nls.op = op

   nls.res_julia_vec   = res_julia_vec
   nls.x_julia_vec     = copy(res_julia_vec)
   nls.x_petsc_vec     = x_petsc_vec
   nls.jac_julia_mat_A = jac_julia_mat_A
   nls.jac_julia_mat_P = jac_julia_mat_A

   nls.res_petsc_vec   = res_petsc_vec
   nls.jac_petsc_mat_A = jac_petsc_mat_A
   nls.jac_petsc_mat_P = jac_petsc_mat_A

   _set_petsc_residual_function!(nls)
   _set_petsc_jacobian_function!(nls)
end

function Algebra.solve!(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator)
  _setup_before_solve!(x,nls,op)
  @check_error_code PETSC.SNESSolve(nls.snes[],C_NULL,nls.x_petsc_vec.vec[])
  nothing
end

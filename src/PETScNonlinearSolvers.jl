
mutable struct PETScNonlinearSolver <: NonlinearSolver
  comm::MPI.Comm
  snes::Ref{SNES}
  initialized::Bool
end

mutable struct PETScNonlinearSolverCache{A,B,C,D}
  initialized::Bool
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

  function PETScNonlinearSolverCache(op::NonlinearOperator,x_julia_vec::A,res_julia_vec::A,
                                     jac_julia_mat_A::B,jac_julia_mat_P::B,
                                     x_petsc_vec::C,res_petsc_vec::C,
                                     jac_petsc_mat_A::D,jac_petsc_mat_P::D) where {A,B,C,D}

      cache=new{A,B,C,D}(true,op,x_julia_vec,res_julia_vec,jac_julia_mat_A,jac_julia_mat_P,
                     x_petsc_vec,res_petsc_vec,jac_petsc_mat_A,jac_petsc_mat_P)
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

function PETScNonlinearSolver(setup,comm)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  snes_ref=Ref{SNES}()
  @check_error_code PETSC.SNESCreate(comm,snes_ref)
  setup(snes_ref)
  snes=PETScNonlinearSolver(comm,snes_ref,true)
  finalizer(Finalize, snes)
  snes
end

function Finalize(nls::PETScNonlinearSolver)
  if GridapPETSc.Initialized() && nls.initialized
    @check_error_code PETSC.SNESDestroy(nls.snes)
    @assert Threads.threadid() == 1
    nls.initialized=false
    _NREFS[] -= 1
  end
  nothing
end

function Finalize(cache::PETScNonlinearSolverCache)
  if GridapPETSc.Initialized() && cache.initialized
     GridapPETSc.Finalize(cache.x_petsc_vec)
     GridapPETSc.Finalize(cache.res_petsc_vec)
     GridapPETSc.Finalize(cache.jac_petsc_mat_A)
     if !(cache.jac_petsc_mat_P === cache.jac_petsc_mat_A)
       GridapPETSc.Finalize(cache.jac_petsc_mat_P)
     end
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
  PETSC.SNESSetFunction(nls.snes[],cache.res_petsc_vec.vec[],fptr,ctx)
end

function _set_petsc_jacobian_function!(nls::PETScNonlinearSolver, cache)
  ctx = pointer_from_objref(cache)
  fptr = @cfunction(snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(nls.snes[],cache.jac_petsc_mat_A.mat[],cache.jac_petsc_mat_A.mat[],fptr,ctx)
end


function _setup_cache(x,nls::PETScNonlinearSolver,op)
  res_julia_vec, jac_julia_mat_A = residual_and_jacobian(op,x)
  res_petsc_vec   = PETScVector(res_julia_vec)
  x_petsc_vec     = PETScVector(x)
  jac_petsc_mat_A = PETScMatrix(jac_julia_mat_A)
  x_julia_vec     = copy(res_julia_vec)

  PETScNonlinearSolverCache(op, x_julia_vec,res_julia_vec,
                           jac_julia_mat_A,jac_julia_mat_A,
                           x_petsc_vec,res_petsc_vec,
                           jac_petsc_mat_A, jac_petsc_mat_A)
end

function Algebra.solve!(x::T,
                        nls::PETScNonlinearSolver,
                        op::NonlinearOperator,
                        cache::PETScNonlinearSolverCache{<:T}) where T <: AbstractVector

  @assert cache.op === op
  @check_error_code PETSC.SNESSolve(nls.snes[],C_NULL,cache.x_petsc_vec.vec[])
  copy!(x,cache.x_petsc_vec.vec[])
  cache
end


function Algebra.solve!(x::AbstractVector,nls::PETScNonlinearSolver,op::NonlinearOperator)
  cache=_setup_cache(x,nls,op)
  _set_petsc_residual_function!(nls,cache)
  _set_petsc_jacobian_function!(nls,cache)
  @check_error_code PETSC.SNESSolve(nls.snes[],C_NULL,cache.x_petsc_vec.vec[])
  copy!(x,cache.x_petsc_vec.vec[])
  cache
end

# ===== TO DO: MOVE Base.copy! overloads below TO APPROPRIATE SOURCE FILES

function Base.copy!(a::AbstractVector,petsc_vec::Vec)
  aux=PETScVector()
  aux.vec[] = petsc_vec.ptr
  Base.copy!(a,aux)
end

function Base.copy!(petsc_vec::Vec,a::AbstractVector)
  aux=PETScVector()
  aux.vec[] = petsc_vec.ptr
  Base.copy!(aux,a)
end

function Base.copy!(pvec::PVector,petscvec::PETScVector)
  map_parts(pvec) do values
    lg=get_local_oh_vector(petscvec)
    if (isa(lg,PETScVector)) # petsc_vec is a ghosted vector
      @assert pvec.rows.ghost
      lx=get_local_vector(lg)
      @assert length(lx)==length(values)
      values .= lx
      restore_local_vector!(lx,lg)
      GridapPETSc.Finalize(lg)
    else                    # petsc_vec is NOT a ghosted vector
      @assert !pvec.rows.ghost
      @assert length(lg)==length(values)
      values .= lg
      restore_local_vector!(petscvec,lg)
    end
  end
  pvec
end

function Base.copy!(petscvec::PETScVector,pvec::PVector)
  map_parts(pvec) do values
    lg=get_local_oh_vector(petscvec)
    if (isa(lg,PETScVector)) # petscvec is a ghosted vector
      @assert pvec.rows.ghost
      lx=get_local_vector(lg)
      @assert length(lx)==length(values)
      lx .= values
      restore_local_vector!(lx,lg)
      GridapPETSc.Finalize(lg)
    else                     # petscvec is NOT a ghosted vector
      @assert !pvec.rows.ghost
      @assert length(lg)==length(values)
      lg .= values
      restore_local_vector!(lg,petscvec)
    end
  end
  pvec
end

function Base.copy!(vec::AbstractVector,petscvec::PETScVector)
  lg=get_local_oh_vector(petscvec)
  if isa(lg,PETScVector) # petscvec is a ghosted vector
    lx=get_local_vector(lg)
    @assert length(lx)==length(vec)
    vec .= lx
    restore_local_vector!(lx,lg)
    GridapPETSc.Finalize(lg)
  else                   # petscvec is NOT a ghosted vector
    @assert length(lg)==length(vec)
    vec .= lg
    restore_local_vector!(lg,petscvec)
  end
end

function Base.copy!(petscvec::PETScVector,vec::AbstractVector)
  lg=get_local_oh_vector(petscvec)
  if isa(lg,PETScVector) # petscvec is a ghosted vector
    lx=get_local_vector(lg)
    @assert length(lx)==length(vec)
    lx .= vec
    restore_local_vector!(lx,lg)
    GridapPETSc.Finalize(lg)
  else                   # petscvec is NOT a ghosted vector
    @assert length(lg)==length(vec)
    lg .= vec
    restore_local_vector!(lg,petscvec)
  end
end

function Base.copy!(petscmat::Mat,a::AbstractMatrix)
  aux=PETScMatrix()
  aux.mat[] = petscmat.ptr
  Base.copy!(aux,a)
end


function Base.copy!(petscmat::PETScMatrix,mat::AbstractMatrix)
   n    = size(mat)[2]
   cols = [PetscInt(j-1) for j=1:n]
   row  = Vector{PetscInt}(undef,1)
   vals = Vector{eltype(mat)}(undef,n)
   for i=1:size(mat)[1]
     row[1]=PetscInt(i-1)
     vals .= view(mat,i,:)
     PETSC.MatSetValues(petscmat.mat[],
                        PetscInt(1),
                        row,
                        n,
                        cols,
                        vals,
                        PETSC.INSERT_VALUES)
   end
   @check_error_code PETSC.MatAssemblyBegin(petscmat.mat[], PETSC.MAT_FINAL_ASSEMBLY)
   @check_error_code PETSC.MatAssemblyEnd(petscmat.mat[]  , PETSC.MAT_FINAL_ASSEMBLY)
end

function Base.copy!(petscmat::PETScMatrix,mat::PSparseMatrix)
  @notimplemented
end
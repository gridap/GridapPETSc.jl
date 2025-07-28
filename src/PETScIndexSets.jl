
"""
    struct PETScIndexSet

Julia interface for a PETSc `IS` object.
"""
mutable struct PETScIndexSet
  is :: Base.RefValue{IS}
  initialized::Bool
  length::Int
  comm::MPI.Comm
  PETScIndexSet(comm::MPI.Comm) = new(Ref{IS}(),false,-1,comm)
end

function Init(a::PETScIndexSet)
  @assert Threads.threadid() == 1
  n = Ref{PetscInt}()
  @check_error_code PETSC.ISGetSize(a.is[],n)
  a.length = Int(n[])
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize,a)
end

function Finalize(a::PETScIndexSet)
  if a.initialized && GridapPETSc.Initialized()
    if a.comm == MPI.COMM_SELF
      @check_error_code PETSC.ISDestroy(a.is)
    else
      @check_error_code PETSC.PetscObjectRegisterDestroy(a.is[])
    end
    a.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function destroy(a::PETScIndexSet)
  if a.initialized && GridapPETSc.Initialized()
    @check_error_code PETSC.ISDestroy(a.is)
    a.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function PETScIndexSet(indices::AbstractVector{<:Integer})
  petsc_indices = convert(Vector{PetscInt},indices)
  PETScIndexSet(petsc_indices)
end

function PETScIndexSet(indices::Vector{PetscInt})
  is = PETScIndexSet(MPI.COMM_SELF)
  n = length(indices)
  petsc_indices = indices .- one(PetscInt)
  @check_error_code PETSC.ISCreateGeneral(MPI.COMM_SELF,n,petsc_indices,PETSC.PETSC_COPY_VALUES,is.is)
  Init(is)
end

function PETScIndexSet(prange::PRange)
  _petsc_index_set(prange,partition(prange))
end

function _petsc_index_set(prange,::DebugArray)
  k = 1
  indices = zeros(PetscInt,length(prange))
  map(own_to_global(prange)) do o2g
    n = length(o2g)
    indices[k:k+n-1] .= o2g .- one(PetscInt)
    k += n
  end
  PETScIndexSet(indices)
end

function _petsc_index_set(prange,::MPIArray)
  n = length(prange)
  is = PETScIndexSet(partition(prange).comm)
  map(local_to_global(prange)) do l2g
    indices = collect(PetscInt,l2g) .- one(PetscInt)
    @check_error_code PETSC.ISCreateGeneral(MPI.COMM_SELF,n,indices,PETSC.PETSC_COPY_VALUES,is.is)
  end
  Init(is)
end

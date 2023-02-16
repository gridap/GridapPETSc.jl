# Index

mutable struct PETScIS <: AbstractVector{PetscInt}
  is::Base.RefValue{IS}
  initialized::Bool
  size::Tuple{Int}
  comm::MPI.Comm
  PETScIS(comm::MPI.Comm) = new(Ref{IS}(), false, (-1,), comm)
end


function Init(a::PETScIS)
  n = Ref{PetscInt}()
  @check_error_code PETSC.ISGetSize(a.is[], n)
  a.size = (Int(n[]),)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize, a)
end

function Finalize(a::PETScIS)
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

function Base.size(v::PETScIS)
  @assert v.initialized
  v.size
end

Base.@propagate_inbounds function Base.getindex(v::PETScIS, i1::Integer)
  @boundscheck checkbounds(v, i1)
  n = one(PetscInt)
  i0 = Ref(i1 - n)
  pi0 = reinterpret(Ptr{PetscInt}, pointer_from_objref(i0))
  @check_error_code PETSC.ISGetIndices(v.is[], pi0)
  return pi0
end

function PETScIS(idx::Vector{PetscInt}, bs=1)
  comm = MPI.COMM_SELF
  is = PETScIS(comm)
  n = length(idx)
  @check_error_code GridapPETSc.PETSC.ISCreateGeneral(comm, n, idx, GridapPETSc.PETSC.PETSC_COPY_VALUES, is.is)
  Init(is)
end

function PETScIS(idx::AbstractVector, bs=1)
  idx = PetscInt.(idx)
  PETScIS(idx)
end

# #Block Implementation
#   function PETScIS(array::Vector{PetscInt},n, bs=1)
#     comm = MPI.COMM_SELF
#     is = PETScIS(comm)
#     n = PetscInt(n)
#     bs = PetscInt(bs)
#     @check_error_code GridapPETSc.PETSC.ISCreateBlock(comm, n, bs, array, GridapPETSc.PETSC.PETSC_COPY_VALUES, is.is)
#     Init(is)
#   end

# Constructors

function PETScIS(n::Integer)
  println("PETScIS")
  v = Ref{Ptr{PetscInt}}()
  println("Construct\n")
  @check_error_code PETSC.PetscMalloc1(n, v)
  Init(v)
end
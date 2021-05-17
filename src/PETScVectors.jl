# Vector

mutable struct PETScVector <: AbstractVector{PetscScalar}
  vec::Ref{Vec}
  initialized::Bool
  PETScVector() = new(Ref{Vec}(),false)
end

function Init(a::PETScVector)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize,a)
end

function Finalize(a::PETScVector)
  if a.initialized && GridapPETSc.Initialized()
    @check_error_code PETSC.VecDestroy(a.vec)
    a.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function Base.size(v::PETScVector)
  n = Ref{PetscInt}()
  @check_error_code PETSC.VecGetSize(v.vec[],n)
  (Int(n[]),)
end

Base.@propagate_inbounds function Base.getindex(v::PETScVector,i1::Integer)
  @boundscheck checkbounds(v, i1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  vi = Ref{PetscScalar}()
  @check_error_code PETSC.VecGetValues(v.vec[],n,i0,vi)
  vi[]
end

Base.@propagate_inbounds function Base.setindex!(v::PETScVector,y,i1::Integer)
  @boundscheck checkbounds(v, i1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  vi = Ref(PetscScalar(y))
  @check_error_code PETSC.VecSetValues(v.vec[],n,i0,vi,PETSC.INSERT_VALUES)
  y
end

# Constructors

function PETScVector(n::Integer)
  v = PETScVector()
  @check_error_code PETSC.VecCreateSeq(MPI.COMM_SELF,n,v.vec)
  Init(v)
end

function Base.similar(::PETScVector,::Type{PetscScalar},ax::Tuple{Int})
  PETScVector(ax[1])
end

function Base.similar(a::PETScVector,::Type{PetscScalar},ax::Tuple{<:Base.OneTo})
  similar(a,PetscScalar,map(length,ax))
end

function Base.similar(::Type{PETScVector},ax::Tuple{Int})
  PETScVector(ax[1])
end

function Base.similar(::Type{PETScVector},ax::Tuple{<:Base.OneTo})
  PETScVector(map(length,ax))
end

# Matrix

mutable struct PETScMatrix <: AbstractMatrix{PetscScalar}
  mat::Ref{Mat}
  initialized::Bool
  PETScMatrix() = new(Ref{Mat}(),false)
end

function Init(a::PETScMatrix)
  @assert Threads.threadid() == 1
  _NREFS[] += 1
  a.initialized = true
  finalizer(Finalize,a)
end

function Finalize(a::PETScMatrix)
  if a.initialized && GridapPETSc.Initialized()
    @check_error_code PETSC.MatDestroy(a.mat)
    a.initialized = false
    @assert Threads.threadid() == 1
    _NREFS[] -= 1
  end
  nothing
end

function Base.size(v::PETScMatrix)
  m = Ref{PetscInt}()
  n = Ref{PetscInt}()
  @check_error_code PETSC.MatGetSize(v.mat[],m,n)
  (Int(m[]),Int(n[]))
end

Base.@propagate_inbounds function Base.getindex(v::PETScMatrix,i1::Integer,j1::Integer)
  @boundscheck checkbounds(v, i1, j1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  j0 = Ref(PetscInt(j1-n))
  vi = Ref{PetscScalar}()
  @check_error_code PETSC.MatGetValues(v.mat[],n,i0,n,j0,vi)
  vi[]
end

# Setting values in this way is VERY inefficient
Base.@propagate_inbounds function Base.setindex!(v::PETScMatrix,y,i1::Integer,j1::Integer)
  @boundscheck checkbounds(v, i1, j1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  j0 = Ref(PetscInt(j1-n))
  vi = Ref(PetscScalar(y))
  @check_error_code PETSC.MatSetValues(v.mat[],n,i0,n,j0,vi,PETSC.INSERT_VALUES)
  @check_error_code PETSC.MatAssemblyBegin(v.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(v.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  y
end

# Constructors

# Using a matrix in this way is VERY inefficient
function PETScMatrix(m::Integer,n::Integer)
  v = PETScMatrix()
  nz = PETSC.PETSC_DEFAULT
  nnz = C_NULL
  @check_error_code PETSC.MatCreateSeqAIJ(MPI.COMM_SELF,m,n,nz,nnz,v.mat)
  @check_error_code PETSC.MatAssemblyBegin(v.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(v.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  Init(v)
end

function Base.similar(::PETScMatrix,::Type{PetscScalar},ax::Tuple{Int,Int})
  PETScMatrix(ax[1],ax[2])
end

function Base.similar(a::PETScMatrix,::Type{PetscScalar},ax::Tuple{<:Base.OneTo,<:Base.OneTo})
  similar(a,PetscScalar,map(length,ax))
end

function Base.similar(::Type{PETScMatrix},ax::Tuple{Int,Int})
  PETScMatrix(ax[1],ax[2])
end

function Base.similar(::Type{PETScMatrix},ax::Tuple{<:Base.OneTo,<:Base.OneTo})
  PETScMatrix(map(length,ax))
end

# Vector

mutable struct PETScVector <: AbstractVector{PetscScalar}
  vec::Base.RefValue{Vec}
  initialized::Bool
  ownership::Any
  size::Tuple{Int}
  PETScVector() = new(Ref{Vec}(),false,nothing,(-1,))
end

function Init(a::PETScVector)
  n = Ref{PetscInt}()
  @check_error_code PETSC.VecGetSize(a.vec[],n)
  a.size = (Int(n[]),)
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
  @assert v.initialized
  v.size
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
  @check_error_code PETSC.VecSetOption(v.vec[],PETSC.VEC_IGNORE_NEGATIVE_INDICES,PETSC.PETSC_TRUE)
  Init(v)
end

function PETScVector(array::Vector{PetscScalar},bs=1)
  v = PETScVector()
  comm = MPI.COMM_SELF
  n = length(array)
  @check_error_code PETSC.VecCreateSeqWithArray(comm,bs,n,array,v.vec)
  @check_error_code PETSC.VecSetOption(v.vec[],PETSC.VEC_IGNORE_NEGATIVE_INDICES,PETSC.PETSC_TRUE)
  v.ownership = array
  Init(v)
end

function PETScVector(a::PetscScalar,ax::AbstractUnitRange)
  PETScVector(fill(a,length(ax)))
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

function Base.copy(a::PETScVector)
  v = PETScVector()
  @check_error_code PETSC.VecDuplicate(a.vec[],v.vec)
  @check_error_code PETSC.VecCopy(a.vec[],v.vec[])
  Init(v)
end

function Base.convert(::Type{PETScVector},a::PETScVector)
  a
end

function Base.convert(::Type{PETScVector},a::AbstractVector)
  array = convert(Vector{PetscScalar},a)
  PETScVector(array)
end


function Base.copy!(vec::AbstractVector,petscvec::Vec)
  ni = length(vec)
  ix = collect(PetscInt,0:(ni-1))
  v = convert(Vector{PetscScalar},vec)
  @check_error_code PETSC.VecGetValues(petscvec.ptr,ni,ix,v)
  if !(v === vec)
    vec .= v
  end
end

function Base.copy!(petscvec::Vec,vec::AbstractVector)
  ni = length(vec)
  ix = collect(PetscInt,0:(ni-1))
  v = convert(Vector{PetscScalar},vec)
  @check_error_code PETSC.VecSetValues(petscvec.ptr,ni,ix,v,PETSC.INSERT_VALUES)
end

function get_local_oh_vector(a::PETScVector)
  v=PETScVector()
  @check_error_code PETSC.VecGhostGetLocalForm(a.vec[],v.vec)
  if v.vec[] != C_NULL  # a is a ghosted vector
    v.ownership=a
    Init(v)
    return v
  else                  # a is NOT a ghosted vector
    return get_local_vector(a)
  end
end

function _local_size(a::PETScVector)
  r_sz = Ref{PetscInt}()
  @check_error_code PETSC.VecGetLocalSize(a.vec[], r_sz)
  r_sz[]
end

# This function works with either ghosted or non-ghosted MPI vectors.
# In the case of a ghosted vector it solely returns the locally owned
# entries.
function get_local_vector(a::PETScVector)
  r_pv = Ref{Ptr{PetscScalar}}()
  @check_error_code PETSC.VecGetArray(a.vec[], r_pv)
  v = unsafe_wrap(Array, r_pv[], _local_size(a); own = false)
  return v
end

function restore_local_vector!(v::Array,a::PETScVector)
  @check_error_code PETSC.VecRestoreArray(a.vec[], Ref(pointer(v)))
  nothing
end

# Matrix

mutable struct PETScMatrix <: AbstractMatrix{PetscScalar}
  mat::Base.RefValue{Mat}
  initialized::Bool
  ownership::Any
  size::Tuple{Int,Int}
  PETScMatrix() = new(Ref{Mat}(),false,nothing,(-1,-1))
end

function Init(a::PETScMatrix)
  m = Ref{PetscInt}()
  n = Ref{PetscInt}()
  @check_error_code PETSC.MatGetSize(a.mat[],m,n)
  a.size = (Int(m[]),Int(n[]))
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
  @check v.initialized
  v.size
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

function SparseArrays.nnz(a::PETScMatrix)
  info = Ref{PETSC.MatInfo}()
  @check_error_code PETSC.MatGetInfo(a.mat[],PETSC.MAT_GLOBAL_SUM,info)
  Int(info[].nz_used)
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

function PETScMatrix(csr::SparseMatrixCSR{0,PetscScalar,PetscInt})
  m, n = size(csr); i = csr.rowptr; j = csr.colval; v = csr.nzval
  A = PETScMatrix()
  A.ownership = csr
  @check_error_code PETSC.MatCreateSeqAIJWithArrays(MPI.COMM_SELF,m,n,i,j,v,A.mat)
  @check_error_code PETSC.MatAssemblyBegin(A.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(A.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  Init(A)
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

function Base.similar(::PETScMatrix,::Type{PetscScalar},ax::Tuple{Int})
  similar(PETScVector,ax)
end

function Base.similar(a::PETScMatrix,::Type{PetscScalar},ax::Tuple{<:Base.OneTo})
  similar(PETScVector,ax)
end

function Base.copy(a::PETScMatrix)
  v = PETScMatrix()
  @check_error_code PETSC.MatConvert(
    a.mat[],PETSC.MATSAME,PETSC.MAT_INITIAL_MATRIX,v.mat)
  Init(v)
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



function Base.convert(::Type{PETScMatrix},a::PETScMatrix)
  a
end

function Base.convert(::Type{PETScMatrix},a::AbstractSparseMatrix)
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  csr = convert(Tm,a)
  PETScMatrix(csr)
end

function Base.convert(::Type{PETScMatrix}, a::AbstractMatrix{PetscScalar})
  m, n = size(a)
  i = [PetscInt(n*(i-1)) for i=1:m+1]
  j = [PetscInt(j-1) for i=1:m for j=1:n]
  v = [ a[i,j] for i=1:m for j=1:n]
  A = PETScMatrix()
  A.ownership = a
  @check_error_code PETSC.MatCreateSeqAIJWithArrays(MPI.COMM_SELF,m,n,i,j,v,A.mat)
  @check_error_code PETSC.MatAssemblyBegin(A.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(A.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  Init(A)
end

function petsc_sparse(i,j,v,m,n)
  csr = sparsecsr(Val(0),i,j,v,m,n)
  convert(PETScMatrix,csr)
end

# Operations
# TODO implement more ops including broadcasting

function Base.fill!(x::PETScVector,a)
  @check_error_code PETSC.VecSet(x.vec[],a)
  x
end

function LinearAlgebra.mul!(c::PETScVector,a::PETScMatrix,b::PETScVector)
  @assert c !== b
  @check_error_code PETSC.MatMult(a.mat[],b.vec[],c.vec[])
  c
end

function Base.:*(A::PETScMatrix,b::AbstractVector)
  c = convert(PETScVector,b)
  A*c
end

function Base.:*(A::PETScMatrix,b::PETScVector)
  c = similar(b,size(A,1))
  mul!(c,A,b)
  c
end

function Base.:*(a::Number,b::PETScVector)
  c = copy(b)
  @check_error_code PETSC.VecScale(c.vec[],a)
  c
end

function Base.:*(b::PETScVector,a::Number)
  a*b
end

function Base.:*(a::Number,b::PETScMatrix)
  c = copy(b)
  @check_error_code PETSC.MatScale(c.mat[],a)
  c
end

function Base.:*(b::PETScMatrix,a::Number)
  a*b
end

function Base.:+(a::PETScVector,b::PETScVector)
  @assert length(a) == length(b)
  c = copy(a)
  α = one(PetscScalar)
  @check_error_code PETSC.VecAXPY(c.vec[],α,b.vec[])
  c
end

function Base.:-(a::PETScVector,b::PETScVector)
  @assert length(a) == length(b)
  c = copy(a)
  α = -one(PetscScalar)
  @check_error_code PETSC.VecAXPY(c.vec[],α,b.vec[])
  c
end

function LinearAlgebra.norm(a::PETScVector, p::Real=2)
  val = Ref{PETSC.PetscReal}()
  if p==1
    nt = PETSC.NORM_1
  elseif p==2
    nt = PETSC.NORM_2
  else
    @notimplemented
  end
  @check_error_code PETSC.VecNorm(a.vec[],nt,val)
  Float64(val[])
end

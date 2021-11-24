
# Some methods from the Gridap.Algebra interface
# not sure if all are needed....

function LinearAlgebra.fillstored!(a::PETScMatrix,v)
  @notimplementedif v!=zero(typeof(v))
  @check_error_code PETSC.MatZeroEntries(a.mat[])
  a
end

function Algebra.copy_entries!(a::PETScVector,b::PETScVector)
  if a!==b
    @check_error_code PETSC.VecCopy(b.vec[],a.vec[])
  end
  a
end

function Algebra.copy_entries!(a::PETScMatrix,b::PETScMatrix)
  if a!==b
    @check_error_code PETSC.MatCopy(b.mat[],a.mat[],PETSC.SAME_NONZERO_PATTERN)
  end
  a
end

function LinearAlgebra.rmul!(a::PETScVector,b::Number)
  @check_error_code PETSC.VecScale(a.vec[],PetscScalar(b))
  a
end

function LinearAlgebra.rmul!(a::PETScMatrix,b::Number)
  @check_error_code PETSC.MatScale(a.mat[],PetscScalar(b))
  a
end

function Algebra.muladd!(c::PETScVector,a::PETScMatrix,b::PETScVector)
  @check_error_code PETSC.MatMultAdd(a.mat[],b.vec[],c.vec[],d.vec[])
  c
end

# Vector assembly

@inline function Algebra.add_entry!(::typeof(+),A::PETScVector,v::Number,i1)
  @boundscheck checkbounds(A, i1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  vi = Ref(PetscScalar(v))
  PETSC.VecSetValues(A.vec[],n,i0,vi,PETSC.ADD_VALUES)
  nothing
end

@inline function Algebra.add_entries!(
  ::typeof(+),a::PETScVector,v::Vector{PetscScalar},i1)
  @check all(i -> i<0 || ( 0<i && i<=size(a,1)), i1)

  u = one(PetscInt)
  ni = length(i1)
  i0 = Vector{PetscInt}(undef,ni)
  @inbounds for k in 1:ni
    i0[k] = i1[k]-u
  end
  PETSC.VecSetValues(a.vec[],ni,i0,v,PETSC.ADD_VALUES)
  nothing
end

function Arrays.return_cache(
  k::AddEntriesMap,
  A::PETScVector,
  v::Vector{PetscScalar},
  i1::AbstractVector{<:Integer})

  ni = length(i1)
  i0 = Vector{PetscInt}(undef,ni)
  CachedArray(i0)
end

function Arrays.evaluate!(
  cache,
  k::AddEntriesMap,
  A::PETScVector,
  v::Vector{PetscScalar},
  i1::AbstractVector{<:Integer})

  @check k.combine == +
  @check all(i -> i<0 || ( 0<i && i<=size(A,1)), i1)
  @check length(v) == length(i1)
  ni = length(i1)
  setsize!(cache,(ni,))
  i0 = cache.array
  u = one(PetscInt)
  @inbounds for k in 1:ni
    i0[k] = i1[k]-u
  end
  PETSC.VecSetValues(A.vec[],PetscInt(ni),i0,v,PETSC.ADD_VALUES)
  nothing
end

function Arrays.return_cache(
  k::TouchEntriesMap,
  A::PETScVector,
  v::AbstractVector,
  i1::AbstractVector{<:Integer})

  ni = length(i1)
  z = zeros(PetscScalar,ni)
  c2 = return_cache(AddEntriesMap(+),A,z,i1)
  CachedArray(z), c2
end

function Arrays.evaluate!(
  cache,
  k::TouchEntriesMap,
  A::PETScVector,
  v::AbstractVector,
  i1::AbstractVector{<:Integer})
  c1,c2 = cache

  ni = length(i1)
  setsize!(c1,(ni,))
  z = c1.array
  fill!(z,zero(PetscScalar))
  evaluate!(c2,AddEntriesMap(+),A,z,i1)
end

function Algebra.create_from_nz(a::PETScVector)
  @check_error_code PETSC.VecAssemblyBegin(a.vec[])
  @check_error_code PETSC.VecAssemblyEnd(a.vec[])
  a
end

# Matrix assembly

struct MatCounter{L}
  nrows::Int
  ncols::Int
  rownnzmax::Vector{PetscInt}
  loop_style::L
end

Algebra.LoopStyle(::Type{MatCounter{L}}) where L = L()

@inline function Algebra.add_entry!(::typeof(+),a::MatCounter{Loop},v,i,j)
  a.rownnzmax[i] += PetscInt(1)
  nothing
end

@inline function Algebra.add_entry!(::typeof(+),a::MatCounter{DoNotLoop},v,i,j)
  nothing
end

function Algebra.nz_counter(
  builder::SparseMatrixBuilder{PETScMatrix,<:MinMemory},axes)

  nrows = length(axes[1])
  ncols = length(axes[2])
  maxnnz = builder.approach.maxnnz
  if isa(maxnnz,Nothing)
    rownnzmax = zeros(PetscInt,nrows)
    MatCounter(nrows,ncols,rownnzmax,Loop())
  elseif isa(maxnnz,Integer)
    rownnzmax = fill(PetscInt(maxnnz),nrows)
    MatCounter(nrows,ncols,rownnzmax,DoNotLoop())
  elseif isa(maxnnz,Vector{PetscInt})
    rownnzmax = maxnnz
    @assert length(rownnzmax) == nrows
    MatCounter(nrows,ncols,rownnzmax,DoNotLoop())
  else
    @notimplemented
  end
end

function Algebra.nz_allocation(a::MatCounter)
  comm = MPI.COMM_SELF
  m = a.nrows
  n = a.ncols
  nz = PETSC.PETSC_DEFAULT
  # nnz cannot be larger than the number of columns
  # Otherwise PETSc complains when compiled in DEBUG mode
  nnz = broadcast(min,a.rownnzmax,PetscInt(n))
  b = PETScMatrix(comm)
  @check_error_code PETSC.MatCreateSeqAIJ(comm,m,n,nz,nnz,b.mat)
  Init(b)
end

Algebra.LoopStyle(::Type{PETScMatrix}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::PETScMatrix,v::Nothing,i,j)
  add_entry(+,a,zero(PetscScalar),i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::PETScMatrix,v,i1,j1)
  @boundscheck checkbounds(a, i1, j1)
  n = one(PetscInt)
  i0 = Ref(PetscInt(i1-n))
  j0 = Ref(PetscInt(j1-n))
  vi = Ref(PetscScalar(v))
  #@check_error_code PETSC.MatSetValues(a.mat[],n,i0,n,j0,vi,PETSC.ADD_VALUES)
  PETSC.MatSetValues(a.mat[],n,i0,n,j0,vi,PETSC.ADD_VALUES)
  nothing
end

@inline function Algebra.add_entries!(
  ::typeof(+),a::PETScMatrix,v::Matrix{PetscScalar},i1,j1)
  @check all(i -> i<0 || ( 0<i && i<=size(a,1)), i1)
  @check all(j -> j<0 || ( 0<j && j<=size(a,2)), j1)

  u = one(PetscInt)
  ni = length(i1)
  nj = length(j1)
  i0 = Vector{PetscInt}(undef,ni)
  j0 = Vector{PetscInt}(undef,nj)
  @inbounds for k in 1:ni
    i0[k] = i1[k]-u
  end
  @inbounds for k in 1:nj
    j0[k] = j1[k]-u
  end
  PETSC.MatSetValues(a.mat[],ni,i0,nj,j0,v,PETSC.ADD_VALUES)
  nothing
end

function Arrays.return_cache(
  k::AddEntriesMap,
  A::PETScMatrix,
  v::Matrix{PetscScalar},
  i1::AbstractVector{<:Integer},
  j1::AbstractVector{<:Integer})

  ni = length(i1)
  nj = length(j1)
  i0 = Vector{PetscInt}(undef,ni)
  j0 = Vector{PetscInt}(undef,nj)
  CachedArray(i0), CachedArray(j0)
end

function Arrays.evaluate!(
  cache,
  k::AddEntriesMap,
  A::PETScMatrix,
  v::Matrix{PetscScalar},
  i1::AbstractVector{<:Integer},
  j1::AbstractVector{<:Integer})

  @check k.combine == +
  @check all(i -> i<0 || ( 0<i && i<=size(A,1)), i1)
  @check all(j -> j<0 || ( 0<j && j<=size(A,2)), j1)
  @check size(v,1) == length(i1)
  @check size(v,2) == length(j1)
  u = one(PetscInt)
  ci,cj = cache
  ni,nj = size(v)
  setsize!(ci,(ni,))
  setsize!(cj,(nj,))
  i0 = ci.array
  j0 = cj.array
  @inbounds for k in 1:ni
    i0[k] = i1[k]-u
  end
  @inbounds for k in 1:nj
    j0[k] = j1[k]-u
  end
  PETSC.MatSetValues(A.mat[],PetscInt(ni),i0,PetscInt(nj),j0,v,PETSC.ADD_VALUES)
  nothing
end

function Arrays.return_cache(
  k::TouchEntriesMap,
  A::PETScMatrix,
  v::AbstractVector,
  i1::AbstractVector{<:Integer},
  j1::AbstractVector{<:Integer})

  ni = length(i1)
  nj = length(j1)
  z = zeros(PetscScalar,ni,nj)
  c2 = return_cache(AddEntriesMap(+),A,z,i1,j1)
  CachedArray(z), c2
end

function Arrays.evaluate!(
  cache,
  k::TouchEntriesMap,
  A::PETScMatrix,
  v::AbstractVector,
  i1::AbstractVector{<:Integer},
  j1::AbstractVector{<:Integer})

  c1,c2 = cache
  ni = length(i1)
  nj = length(j1)
  setsize!(c1,(ni,nj))
  z = c1.array
  fill!(z,zero(PetscScalar))
  evaluate!(c2,AddEntriesMap(+),A,z,i1,j1)
end

function Algebra.create_from_nz(a::PETScMatrix)
  @check_error_code PETSC.MatAssemblyBegin(a.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(a.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  a
end

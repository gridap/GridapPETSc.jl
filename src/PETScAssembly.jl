
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
  nnz = a.rownnzmax
  b = PETScMatrix()
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

function Algebra.create_from_nz(a::PETScMatrix)
  @check_error_code PETSC.MatAssemblyBegin(a.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(a.mat[],PETSC.MAT_FINAL_ASSEMBLY)
  a
end


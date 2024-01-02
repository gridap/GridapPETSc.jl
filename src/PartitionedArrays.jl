
# Support for PartitionedArrays

Base.convert(::Type{PETScVector},v::PVector) = _petsc_vector(v)

function _petsc_vector(v::PVector)
  values = partition(v)
  _petsc_vector(v,values)
end

function _petsc_vector(v::PVector,::DebugArray)
  gid_to_value = zeros(eltype(v),length(v))

  rows = axes(v,1)
  values = partition(v)
  map(values,partition(rows)) do values,rows
    @check isa(rows,OwnAndGhostIndices) "Unsupported partition for PETSc vectors" # to be consistent with MPI
    oid_to_gid = own_to_global(rows)
    oid_to_value = view(values,own_to_local(rows))
    gid_to_value[oid_to_gid] = oid_to_value
  end
  PETScVector(gid_to_value)
end

function _petsc_vector(v::PVector,::MPIArray)
  rows = axes(v,1)
  values = partition(v)
  comm = values.comm

  w = PETScVector(comm)
  N = length(rows)

  map(values,partition(rows)) do lid_to_value, rows
    @check isa(rows,OwnAndGhostIndices) "Unsupported partition for PETSc vectors"
    n_owned = own_length(rows)
    n_ghost = ghost_length(rows)
    array   = convert(Vector{PetscScalar},lid_to_value)
    ghost   = convert(Vector{PetscInt},ghost_to_global(rows)) .- PetscInt(1)
    w.ownership = (array,ghost)
    @check_error_code PETSC.VecCreateGhostWithArray(comm,n_owned,N,n_ghost,ghost,array,w.vec)
    @check_error_code PETSC.VecSetOption(w.vec[],PETSC.VEC_IGNORE_NEGATIVE_INDICES,PETSC.PETSC_TRUE)
    Init(w)
  end
  return w
end

function PETScVector(a::PetscScalar,ax::PRange)
  rows = partition(ax)
  convert(PETScVector,pfill(a,rows))
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange)
  rows = partition(ids)
  PVector(v,ids,rows)
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange,::DebugArray)
  @assert length(v) == length(ids)
  ni = length(v)
  ix = collect(PetscInt,0:(ni-1))
  y  = zeros(PetscScalar,ni)
  @check_error_code PETSC.VecGetValues(v.vec[],ni,ix,y)
  values = map(partition(ids)) do ids
    oid_to_gid = own_to_global(ids)
    oid_to_lid = own_to_local(ids)
    z = zeros(PetscScalar,local_length(ids))
    z[oid_to_lid] .= y[oid_to_gid]
    z
  end
  return PVector(values,partition(ids))
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange,::MPIArray)
  # TODO a way to avoid duplicating memory?
  values = map(partition(ids)) do ids
    ni = local_length(ids)
    ix = fill(PetscInt(-1),ni)
    y  = zeros(PetscScalar,ni)
    u  = PetscInt(1)
    oid_to_lid = own_to_local(ids)
    lid_to_gid = local_to_global(ids)
    for oid in 1:own_length(ids)
      lid = oid_to_lid[oid]
      gid = lid_to_gid[lid]
      ix[lid] = gid - u
    end
    @check_error_code PETSC.VecGetValues(v.vec[],ni,ix,y)
    y
  end
  return PVector(values,partition(ids))
end

function _copy!(a::PVector{T,<:DebugArray},b::Vec) where T
  ni = length(a)
  ix = collect(PetscInt,0:(ni-1))
  y  = zeros(PetscScalar,ni)
  @check_error_code PETSC.VecGetValues(b,ni,ix,y)

  rows = axes(a,1)
  values = partition(a)
  map(partition(rows),values) do rows,values
    oid_to_gid = own_to_global(rows)
    oid_to_lid = own_to_local(rows)
    values[oid_to_lid] .= y[oid_to_gid]
  end
  return a
end

function _copy!(a::Vec,b::PVector{T,<:DebugArray}) where T
  ni = length(b)
  ix = collect(PetscInt,0:(ni-1))
  y  = zeros(PetscScalar,ni)

  rows = axes(b,1)
  values = partition(b)
  map(partition(rows),values) do rows,values
    oid_to_gid = own_to_global(rows)
    oid_to_lid = own_to_local(rows)
    y[oid_to_gid] .= values[oid_to_lid]
  end
  @check_error_code PETSC.VecSetValues(a,ni,ix,y,PETSC.INSERT_VALUES)
  return a
end

function _copy!(pvec::PVector{T,<:MPIArray},petscvec::Vec) where T
  rows = axes(pvec,1)
  map(own_values(pvec),partition(rows)) do values, indices
    lg = _get_local_oh_vector(petscvec)
    if (isa(lg,PETScVector)) # A) petsc_vec is a ghosted vector
      # Only copying owned DoFs. This should be followed by
      # an exchange if the ghost DoFs of pvec are to be consumed.
      # We are assuming here that the layout of pvec and petsvec
      # are compatible. We do not have any information about the
      # layout of petscvec to check this out. We decided NOT to
      # convert petscvec into a PVector to avoid extra memory allocation
      # and copies.
      lx = _get_local_vector_read(lg)
      values .= lx[1:own_length(indices)]
      _restore_local_vector!(lx,lg)
      GridapPETSc.Finalize(lg)
    else                    # B) petsc_vec is NOT a ghosted vector
      # @assert length(lg)==length(values)
      # values .= lg
      # _restore_local_vector!(petscvec,lg)

      # If am not wrong, the code should never enter here. At least
      # given how it is being leveraged at present from GridapPETsc.
      # If in the future we need this case, the commented lines of
      # code above could serve the purpose (not tested).
      @notimplemented
    end
  end
  return pvec
end

function _copy!(petscvec::Vec,pvec::PVector{T,<:MPIArray}) where T
  rows = axes(pvec,1)
  map(own_values(pvec),partition(rows)) do values, indices
    @check isa(indices,OwnAndGhostIndices) "Unsupported partition for PETSc vectors"
    lg = _get_local_oh_vector(petscvec)
    if (isa(lg,PETScVector)) # A) petscvec is a ghosted vector
      lx=_get_local_vector(lg)
      # Only copying owned DoFs. This should be followed by
      # an exchange if the ghost DoFs of petscvec are to be consumed.
      # We are assuming here that the layout of pvec and petsvec
      # are compatible. We do not have any information about the
      # layout of petscvec to check this out.
      lx[1:own_length(indices)] .= values
      _restore_local_vector!(lx,lg)
      GridapPETSc.Finalize(lg)
    else                     # B) petscvec is NOT a ghosted vector
    #  @assert length(lg)==length(values)
    #  lg .= values
    #  restore_local_vector!(lg,petscvec)
    # See comment in the function copy! right above
    @notimplemented
    end
  end
  return petscvec
end

Base.convert(::Type{PETScMatrix},a::PSparseMatrix) = _petsc_matrix(a)

function _petsc_matrix(a::PSparseMatrix)
  values = partition(a)
  _petsc_matrix(a,values)
end

function _petsc_matrix(a::PSparseMatrix,::DebugArray)
  rows, cols = axes(a)
  map(partition(rows),partition(cols)) do rows,cols
    @check isa(rows,OwnAndGhostIndices) "Not supported partition for PETSc matrices" # to be consistent with MPI
    @check isa(cols,OwnAndGhostIndices) "Not supported partition for PETSc matrices" # to be consistent with MPI
  end
  a_main = PartitionedArrays.to_trivial_partition(a) # Assemble global matrix in MAIN
  A = PartitionedArrays.getany(partition(a_main))
  convert(PETScMatrix,A)
end

function _petsc_matrix(a::PSparseMatrix,::MPIArray)
  rows, cols = axes(a)
  values = partition(a)
  comm = values.comm # Not sure about this
  b = PETScMatrix(comm)
  M = length(rows)
  N = length(cols)
  map(values,partition(rows),partition(cols)) do values,rows,cols
    @check isa(rows,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    @check isa(cols,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    Tm  = SparseMatrixCSR{0,PetscScalar,PetscInt}
    csr = convert(Tm,values)
    i = csr.rowptr; _j = csr.colval; v = csr.nzval
    if values === csr
      j = copy(_j)
    else
      j = _j
    end
    u = PetscInt(1)
    cols_lid_to_gid = local_to_global(cols)
    for k in 1:length(j)
      lid = j[k] + u
      gid = cols_lid_to_gid[lid]
      j[k] = gid - u
    end
    m = own_length(rows)
    n = own_length(cols)
    @check_error_code PETSC.MatCreateMPIAIJWithArrays(comm,m,n,M,N,i,j,v,b.mat)
    @check_error_code PETSC.MatAssemblyBegin(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code PETSC.MatAssemblyEnd(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    Init(b)
  end
  return b
end

function _copy!(petscmat::Mat,mat::PSparseMatrix{T,<:DebugArray}) where {T}
  rows, cols = axes(mat)
  values = partition(mat)
  map(values,partition(rows),partition(cols)) do values, rows, cols
    @check isa(rows,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    @check isa(cols,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    Tm  = SparseMatrixCSR{0,PetscScalar,PetscInt}
    csr = convert(Tm,values)
    ia  = csr.rowptr
    ja  = csr.colval
    a   = csr.nzval

    rows_own_to_local    = own_to_local(rows)
    rows_local_to_global = local_to_global(rows)
    cols_local_to_global = local_to_global(cols)
  
    maxnnz = maximum(map(i -> ia[i+1]-ia[i], 1:csr.m))
    petsc_row  = Vector{PetscInt}(undef,1)
    petsc_cols = Vector{PetscInt}(undef,maxnnz)
    for row_lid in rows_own_to_local
      petsc_row[1] = PetscInt(rows_local_to_global[row_lid]-1)
      for (col_counter,j) in enumerate(ia[row_lid]+1:ia[row_lid+1])
        col_lid = ja[j]+1
        petsc_cols[col_counter] = PetscInt(cols_local_to_global[col_lid]-1)
      end
      num_nz = ia[row_lid+1]-ia[row_lid]
      petsc_nzvals = view(a,ia[row_lid]+1:ia[row_lid+1])
      PETSC.MatSetValues(petscmat.ptr,
                         PetscInt(1),
                         petsc_row,
                         num_nz,
                         petsc_cols,
                         petsc_nzvals,
                         PETSC.INSERT_VALUES)
    end
  end
  @check_error_code PETSC.MatAssemblyBegin(petscmat.ptr, PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(petscmat.ptr, PETSC.MAT_FINAL_ASSEMBLY)
end

_copy!(::PSparseMatrix{T,<:DebugArray},::Mat) where T = @notimplemented
_copy!(::PSparseMatrix{T,<:MPIArray},::Mat) where T = @notimplemented

function _copy!(petscmat::Mat,mat::PSparseMatrix{T,<:MPIArray}) where T
  rows, cols = axes(mat)
  values = partition(mat)
  map(values,partition(rows),partition(cols)) do values, rows, cols
    @check isa(rows,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    @check isa(cols,OwnAndGhostIndices) "Not supported partition for PETSc matrices"
    Tm  = SparseMatrixCSR{0,PetscScalar,PetscInt}
    csr = convert(Tm,values)
    ia  = csr.rowptr
    ja  = csr.colval
    a   = csr.nzval

    rows_own_to_local    = own_to_local(rows)
    rows_local_to_global = local_to_global(rows)
    cols_local_to_global = local_to_global(cols)
  
    maxnnz = maximum(map(i -> ia[i+1]-ia[i], 1:csr.m))
    petsc_row  = Vector{PetscInt}(undef,1)
    petsc_cols = Vector{PetscInt}(undef,maxnnz)
    for row_lid in rows_own_to_local
      petsc_row[1] = PetscInt(rows_local_to_global[row_lid]-1)
      for (col_counter,j) in enumerate(ia[row_lid]+1:ia[row_lid+1])
        col_lid = ja[j]+1
        petsc_cols[col_counter] = PetscInt(cols_local_to_global[col_lid]-1)
      end
      num_nz = ia[row_lid+1]-ia[row_lid]
      petsc_nzvals = view(a,ia[row_lid]+1:ia[row_lid+1])
      PETSC.MatSetValues(petscmat.ptr,
                         PetscInt(1),
                         petsc_row,
                         num_nz,
                         petsc_cols,
                         petsc_nzvals,
                         PETSC.INSERT_VALUES)
    end
  end
  @check_error_code PETSC.MatAssemblyBegin(petscmat.ptr, PETSC.MAT_FINAL_ASSEMBLY)
  @check_error_code PETSC.MatAssemblyEnd(petscmat.ptr, PETSC.MAT_FINAL_ASSEMBLY)
end

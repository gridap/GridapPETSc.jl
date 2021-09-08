
# Support for PartitionedArrays

function PETScVector(v::PVector)
  backend = get_backend(v.values)
  PETScVector(v,backend)
end

function PETScVector(v::PVector,::SequentialBackend)
  gid_to_value = zeros(eltype(v),length(v))
  map_parts(v.values,v.rows.partition) do values,rows
    @check isa(rows,IndexRange) "Not supported partition for PETSc vectors" # to be consistent with MPI
    oid_to_gid = view(rows.lid_to_gid,rows.oid_to_lid)
    oid_to_value = view(values,rows.oid_to_lid)
    gid_to_value[oid_to_gid] =  oid_to_value
  end
  PETScVector(gid_to_value)
end

function PETScVector(v::PVector,::MPIBackend)
  w = PETScVector()
  N = num_gids(v.rows)
  comm = v.values.comm # Not sure about this
  map_parts(v.values,v.rows.partition) do lid_to_value, rows
    @check isa(rows,IndexRange) "Not supported partition for PETSc vectors"
    array = convert(Vector{PetscScalar},lid_to_value)
    n = num_oids(rows)
    nghost = num_hids(rows)
    hid_to_gid = view(rows.lid_to_gid,rows.hid_to_lid)
    ghost = convert(Vector{PetscInt},hid_to_gid)
    ghost .= ghost .- PetscInt(1)
    w.ownership = (array,ghost)
    @check_error_code PETSC.VecCreateGhostWithArray(comm,n,N,nghost,ghost,array,w.vec)
    @check_error_code PETSC.VecSetOption(w.vec[],PETSC.VEC_IGNORE_NEGATIVE_INDICES,PETSC.PETSC_TRUE)
    Init(w)
  end
  w
end

function PETScVector(a::PetscScalar,ax::PRange)
  PETScVector(PVector(a,ax))
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange)
  backend = get_backend(ids.partition)
  PVector(v,ids,backend)
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange,::SequentialBackend)
  @assert length(v) == length(ids)
  ni = length(v)
  ix = collect(PetscInt,0:(ni-1))
  y = zeros(PetscScalar,ni)
  @check_error_code PETSC.VecGetValues(v.vec[],ni,ix,y)
  values = map_parts(ids.partition) do ids
    oid_to_gid = view(ids.lid_to_gid,ids.oid_to_lid)
    z = zeros(PetscScalar,num_lids(ids))
    z[ids.oid_to_lid] = y[oid_to_gid]
    z
  end
  PVector(values,ids)
end

function PartitionedArrays.PVector(v::PETScVector,ids::PRange,::MPIBackend)
  # TODO a way to avoid duplicating memory?
  values = map_parts(ids.partition) do ids
    ni = num_lids(ids)
    ix = fill(PetscInt(-1),ni)
    y = zeros(PetscScalar,ni)
    u = PetscInt(1)
    for oid in 1:num_oids(ids)
      lid = ids.oid_to_lid[oid]
      gid = ids.lid_to_gid[lid]
      ix[lid] = gid - u
    end
    @check_error_code PETSC.VecGetValues(v.vec[],ni,ix,y)
    y
  end
  PVector(values,ids)
end

function PETScMatrix(a::PSparseMatrix)
  backend = get_backend(a.values)
  PETScMatrix(a,backend)
end

function PETScMatrix(a::PSparseMatrix,::SequentialBackend)
  map_main(a.rows.partition,a.cols.partition) do rows,cols
    @check isa(rows,IndexRange) "Not supported partition for PETSc matrices" # to be consistent with MPI
    @check isa(cols,IndexRange) "Not supported partition for PETSc matrices" # to be consistent with MPI
  end
  A = get_main_part(gather(a).values)
  convert(PETScMatrix,A)
end

function PETScMatrix(a::PSparseMatrix,::MPIBackend)
  b = PETScMatrix()
  M = num_gids(a.rows)
  N = num_gids(a.cols)
  comm = a.values.comm # Not sure about this
  map_parts(a.values,a.rows.partition,a.cols.partition) do values,rows,cols
    @check isa(rows,IndexRange) "Not supported partition for PETSc matrices"
    @check isa(cols,IndexRange) "Not supported partition for PETSc matrices"
    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    csr = convert(Tm,values)
    i = csr.rowptr; j = csr.colval; v = csr.nzval
    u = PetscInt(1)
    for k in 1:length(j)
      lid = j[k] + u
      gid = cols.lid_to_gid[lid]
      j[k] = gid - u
    end
    m = num_oids(rows)
    n = num_oids(cols)
    @check_error_code PETSC.MatCreateMPIAIJWithArrays(comm,m,n,M,N,i,j,v,b.mat)
    @check_error_code PETSC.MatAssemblyBegin(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code PETSC.MatAssemblyEnd(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    Init(b)
  end
  b
end




# Support for PartitionedArrays

Base.convert(::Type{PETScVector},v::PVector)=_petsc_vector(v)

function _petsc_vector(v::PVector)
  backend = get_backend(v.values)
  _petsc_vector(v,backend)
end

function _petsc_vector(v::PVector,::SequentialBackend)
  gid_to_value = zeros(eltype(v),length(v))
  map_parts(v.values,v.rows.partition) do values,rows
    @check isa(rows,IndexRange) "Unsupported partition for PETSc vectors" # to be consistent with MPI
    oid_to_gid = view(rows.lid_to_gid,rows.oid_to_lid)
    oid_to_value = view(values,rows.oid_to_lid)
    gid_to_value[oid_to_gid] =  oid_to_value
  end
  PETScVector(gid_to_value)
end

function _petsc_vector(v::PVector,::MPIBackend)
  w = PETScVector()
  N = num_gids(v.rows)
  comm = v.values.comm # Not sure about this

  map_parts(v.values,v.rows.partition) do lid_to_value, rows
    @check isa(rows,IndexRange) "Unsupported partition for PETSc vectors"
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
  convert(PETScVector,PVector(a,ax))
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


function Base.copy!(pvec::PVector,petscvec::PETScVector)
  if get_backend(pvec.values) == mpi
      map_parts(get_part_ids(pvec.values),pvec.values,pvec.rows.partition) do part, values, indices
        lg=get_local_oh_vector(petscvec)
        if (isa(lg,PETScVector)) # petsc_vec is a ghosted vector
          # Only copying owned DoFs. This should be followed by
          # an exchange if the ghost DoFs of pvec are to be consumed.
          # We are assuming here that the layout of pvec and petsvec
          # are compatible. We do not have any information about the
          # layout of petscvec to check this out. We decided NOT to
          # convert petscvec into a PVector to avoid extra memory allocation
          # and copies.
          @assert pvec.rows.ghost
          lx=get_local_vector(lg)
          vvalues=view(values,indices.oid_to_lid)
          vvalues .= lx[1:num_oids(indices)]
          restore_local_vector!(lx,lg)
          GridapPETSc.Finalize(lg)
        else                    # petsc_vec is NOT a ghosted vector
          # @assert !pvec.rows.ghost
          # @assert length(lg)==length(values)
          # values .= lg
          # restore_local_vector!(petscvec,lg)

          # If am not wrong, the code should never enter here. At least
          # given how it is being leveraged at present from GridapPETsc.
          # If in the future we need this case, the commented lines of
          # code above could serve the purpose (not tested).
          @notimplemented
        end
      end
  elseif get_backend(pvec.values) == sequential
    # I left this as an exercise to the interested.
    @notimplemented
  end
  pvec
end

function Base.copy!(petscvec::PETScVector,pvec::PVector)
  if get_backend(pvec.values) == mpi
     map_parts(pvec.values,pvec.rows.partition) do values, indices
       @check isa(indices,IndexRange) "Unsupported partition for PETSc vectors"
       lg=get_local_oh_vector(petscvec)
       if (isa(lg,PETScVector)) # petscvec is a ghosted vector
         lx=get_local_vector(lg)
         # Only copying owned DoFs. This should be followed by
         # an exchange if the ghost DoFs of petscvec are to be consumed.
         # We are assuming here that the layout of pvec and petsvec
         # are compatible. We do not have any information about the
         # layout of petscvec to check this out.
         lx[1:num_oids(indices)] .= values[1:num_oids(indices)]
         restore_local_vector!(lx,lg)
         GridapPETSc.Finalize(lg)
       else
        #  @assert !pvec.rows.ghost
        #  @assert length(lg)==length(values)
        #  lg .= values
        #  restore_local_vector!(lg,petscvec)
        # See comment in the function copy! right above
        @notimplemented
       end
     end
  elseif get_backend(pvec.values) == sequential
     # I leave this as an exercise to the interested. Essentially the
     # same approach as in Base.copy!(petscvec::PETScMatrix,pvec::PSparseMatrix)
     # has to be followed.
     @notimplemented
  end
  petscvec
end

Base.convert(::Type{PETScMatrix},a::PSparseMatrix) = _petsc_matrix(a)

function _petsc_matrix(a::PSparseMatrix)
  backend = get_backend(a.values)
  _petsc_matrix(a,backend)
end

function _petsc_matrix(a::PSparseMatrix,::SequentialBackend)
  map_main(a.rows.partition,a.cols.partition) do rows,cols
    @check isa(rows,IndexRange) "Not supported partition for PETSc matrices" # to be consistent with MPI
    @check isa(cols,IndexRange) "Not supported partition for PETSc matrices" # to be consistent with MPI
  end
  A = get_main_part(gather(a).values)
  convert(PETScMatrix,A)
end

function _petsc_matrix(a::PSparseMatrix,::MPIBackend)
  b = PETScMatrix()
  M = num_gids(a.rows)
  N = num_gids(a.cols)
  comm = a.values.comm # Not sure about this
  map_parts(a.values,a.rows.partition,a.cols.partition) do values,rows,cols
    @check isa(rows,IndexRange) "Not supported partition for PETSc matrices"
    @check isa(cols,IndexRange) "Not supported partition for PETSc matrices"
    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    csr = convert(Tm,values)
    if PartitionedArrays.get_part_id(a.values)==1
      println(values)
      println(csr)
    end

    i = csr.rowptr; j = csr.colval; v = csr.nzval
    u = PetscInt(1)
    for k in 1:length(j)
      lid = j[k] + u
      gid = cols.lid_to_gid[lid]
      j[k] = gid - u
    end

    m = num_oids(rows)
    n = num_oids(cols)

    ip=collect(i[1:m+1])
    jp=collect(j[1:ip[m+1]])
    vp=collect(v[1:ip[m+1]])

    @check_error_code PETSC.MatCreateMPIAIJWithArrays(comm,m,n,M,N,ip,jp,vp,b.mat)
    @check_error_code PETSC.MatAssemblyBegin(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code PETSC.MatAssemblyEnd(b.mat[],PETSC.MAT_FINAL_ASSEMBLY)
    Init(b)
  end
  b
end

function Base.copy!(petscmat::PETScMatrix,mat::PSparseMatrix)
   parts=get_part_ids(mat.values)
   map_parts(parts, mat.values,mat.rows.partition,mat.cols.partition) do part, lmat, rdofs, cdofs
      @check isa(rdofs,IndexRange) "Not supported partition for PETSc matrices"
      @check isa(cdofs,IndexRange) "Not supported partition for PETSc matrices"
      Tm  = SparseMatrixCSR{0,PetscScalar,PetscInt}
      csr = convert(Tm,lmat)
      ia  = csr.rowptr
      ja  = csr.colval
      a   = csr.nzval
      m   = csr.m
      n   = csr.n
      maxnnz = maximum( ia[i+1]-ia[i] for i=1:m )
      row    = Vector{PetscInt}(undef,1)
      cols   = Vector{PetscInt}(undef,maxnnz)
      for i=1:num_oids(rdofs)
        lid=rdofs.oid_to_lid[i]
        row[1]=PetscInt(rdofs.lid_to_gid[lid]-1)
        current=1
        for j=ia[lid]+1:ia[lid+1]
          col=ja[j]+1
          cols[current]=PetscInt(cdofs.lid_to_gid[col]-1)
          current=current+1
        end
        vals = view(a,ia[lid]+1:ia[lid+1])
        PETSC.MatSetValues(petscmat.mat[],
                           PetscInt(1),
                           row,
                           ia[lid+1]-ia[lid],
                           cols,
                           vals,
                           PETSC.INSERT_VALUES)
      end
   end
   @check_error_code PETSC.MatAssemblyBegin(petscmat.mat[], PETSC.MAT_FINAL_ASSEMBLY)
   @check_error_code PETSC.MatAssemblyEnd(petscmat.mat[], PETSC.MAT_FINAL_ASSEMBLY)
end

# Index

mutable struct PETScIS <: AbstractVector{PetscInt}
    is::Base.RefValue{IS}
    initialized::Bool
    size::Tuple{Int}
    comm::MPI.Comm
    PETScIS(comm::MPI.Comm) =  new(Ref{IS}(),false,(-1,),comm)
end

  
  function Init(a::PETScIS)
    n = Ref{PetscInt}()
    @check_error_code PETSC.ISGetSize(a.is[], n)
    a.size = (Int(n[]),)
    @assert Threads.threadid() == 1
    _NREFS[] += 1
    a.initialized = true
    finalizer(Finalize,a)
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
  
  Base.@propagate_inbounds function Base.getindex(v::PETScIS,i1::Integer)
    @boundscheck checkbounds(v, i1)
    n = one(PetscInt)
    i0 = Ref(i1-n)
    pi0 = reinterpret(Ptr{PetscInt},pointer_from_objref(i0))
    @check_error_code PETSC.ISGetIndices(v.is[], pi0)
    return pi0
  end
  
  # Base.@propagate_inbounds function Base.setindex!(v::PETScIS,y,i1::Integer)
  #   @boundscheck checkbounds(v, i1)
  #   n = one(PetscInt)
  #   i0 = Ref(PetscInt(i1-n))
  #   vi = Ref(PetscInt(y))
  #   @check_error_code PETSC.VecSetValues(v.is[],n,i0,vi,PETSC.INSERT_VALUES)
  #   v.is[i0] .= y
  #   y
  # end
  
  # Constructors
  
  function PETScIS(n::Integer)
    println("PETScIS")
    v = Ref{Ptr{PetscInt}}()
    println("Construct\n")
    @check_error_code PETSC.PetscMalloc1(n, v)
    Init(v)
  end
  



  function PETScIS(idx::Vector{PetscInt}, bs=1)
    comm = MPI.COMM_SELF
    is = PETScIS(comm)
    n = length(idx)
    @check_error_code GridapPETSc.PETSC.ISCreateGeneral(comm, n, idx, GridapPETSc.PETSC.PETSC_COPY_VALUES, is.is)
    Init(is)
  end

  function PETScIS(idx::Vector, bs=1)
    idx = PetscInt.(idx)
    PETScIS(idx)
  end

#Block Implementation
  function PETScIS(array::Vector{PetscInt},n, bs=1)
    comm = MPI.COMM_SELF
    is = PETScIS(comm)
    n = PetscInt(n)
    bs = PetscInt(bs)
    @check_error_code GridapPETSc.PETSC.ISCreateBlock(comm, n, bs, array, GridapPETSc.PETSC.PETSC_COPY_VALUES, is.is)
    Init(is)
  end

#   function PETScVector(a::PetscScalar,ax::AbstractUnitRange)
#     PETScVector(fill(a,length(ax)))
#   end
  
#   function Base.similar(::PETScVector,::Type{PetscScalar},ax::Tuple{Int})
#     PETScVector(ax[1])
#   end
  
#   function Base.similar(a::PETScVector,::Type{PetscScalar},ax::Tuple{<:Base.OneTo})
#     similar(a,PetscScalar,map(length,ax))
#   end
  
#   function Base.similar(::Type{PETScVector},ax::Tuple{Int})
#     PETScVector(ax[1])
#   end
  
#   function Base.similar(::Type{PETScVector},ax::Tuple{<:Base.OneTo})
#     PETScVector(map(length,ax))
#   end
  
#   function Base.copy(a::PETScVector)
#     v = PETScVector(a.comm)
#     @check_error_code PETSC.VecDuplicate(a.vec[],v.vec)
#     @check_error_code PETSC.VecCopy(a.vec[],v.vec[])
#     Init(v)
#   end
  
#   function Base.convert(::Type{PETScVector},a::PETScVector)
#     a
#   end
  
#   function Base.convert(::Type{PETScVector},a::AbstractVector)
#     array = convert(Vector{PetscScalar},a)
#     PETScVector(array)
#   end
  
#   function Base.copy!(a::AbstractVector,b::PETScVector)
#     @check length(a) == length(b)
#     _copy!(a,b.vec[])
#   end
  
#   function _copy!(a::Vector,b::Vec)
#     ni = length(a)
#     ix = collect(PetscInt,0:(ni-1))
#     v = convert(Vector{PetscScalar},a)
#     @check_error_code PETSC.VecGetValues(b,ni,ix,v)
#     if !(v === a)
#       a .= v
#     end
#   end
  
#   function Base.copy!(a::PETScVector,b::AbstractVector)
#     @check length(a) == length(b)
#     _copy!(a.vec[],b)
#   end
  
#   function _copy!(a::Vec,b::Vector)
#     ni = length(b)
#     ix = collect(PetscInt,0:(ni-1))
#     v = convert(Vector{PetscScalar},b)
#     @check_error_code PETSC.VecSetValues(a,ni,ix,v,PETSC.INSERT_VALUES)
#   end
  
#   function _get_local_oh_vector(a::Vec)
#     v=PETScVector(MPI.COMM_SELF)
#     @check_error_code PETSC.VecGhostGetLocalForm(a,v.vec)
#     if v.vec[] != C_NULL  # a is a ghosted vector
#       v.ownership=a
#       Init(v)
#       return v
#     else                  # a is NOT a ghosted vector
#       return _get_local_vector(a)
#     end
#   end
  
#   function _local_size(a::PETScVector)
#     r_sz = Ref{PetscInt}()
#     @check_error_code PETSC.VecGetLocalSize(a.vec[], r_sz)
#     r_sz[]
#   end
  
#   # This function works with either ghosted or non-ghosted MPI vectors.
#   # In the case of a ghosted vector it solely returns the locally owned
#   # entries.
#   function _get_local_vector(a::PETScVector)
#     r_pv = Ref{Ptr{PetscScalar}}()
#     @check_error_code PETSC.VecGetArray(a.vec[], r_pv)
#     v = unsafe_wrap(Array, r_pv[], _local_size(a); own = false)
#     return v
#   end
  
#   # This function works with either ghosted or non-ghosted MPI vectors.
#   # In the case of a ghosted vector it solely returns the locally owned
#   # entries.
#   function _get_local_vector_read(a::PETScVector)
#     r_pv = Ref{Ptr{PetscScalar}}()
#     @check_error_code PETSC.VecGetArrayRead(a.vec[], r_pv)
#     v = unsafe_wrap(Array, r_pv[], _local_size(a); own = false)
#     return v
#   end
  
#   function _restore_local_vector!(v::Array,a::PETScVector)
#     @check_error_code PETSC.VecRestoreArray(a.vec[], Ref(pointer(v)))
#     nothing
#   end
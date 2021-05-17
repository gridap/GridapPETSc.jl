
struct ZeroTo{T} <: AbstractUnitRange{T}
  n::T
end
Base.first(a::ZeroTo{T}) where T = zero(T)
Base.last(a::ZeroTo{T}) where T = a.n - one(T)
function Base.getindex(a::LinearIndices{1,Tuple{ZeroTo{T}}},i::Int) where T
  a.indices[1][i]
end

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

function Base.axes(v::PETScVector)
  n, = size(v)
  (ZeroTo(n),)
end

function Base.getindex(v::PETScVector,inds::Vector{PetscInt})
  n = length(inds)
  itms = Vector{PetscScalar}(undef,n)
  @check_error_code PETSC.VecGetValues(v.vec[],n,inds,itms)
  itms
end

function Base.setindex!(v::PETScVector,itms::Vector{PetscScalar},inds::Vector{PetscInt})
  @assert length(itms) == length(inds)
  n = length(inds)
  @check_error_code PETSC.VecSetValues(v.vec[],n,inds,itms,PETSC.INSERT_VALUES)
  itms
end

function Base.getindex(v::PETScVector,i::Integer)
  v[[PetscInt(i)]][1] # Very inefficient, do not use
end

function Base.setindex!(v::PETScVector,y,i::Integer)
  v[[PetscInt(i)]] = [PetscScalar(y)] # Very inefficient, do not use
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

function Base.similar(a::PETScVector,::Type{PetscScalar},ax::Tuple{<:ZeroTo})
  similar(a,PetscScalar,map(length,ax))
end

function Base.similar(::Type{PETScVector},ax::Tuple{Int})
  PETScVector(ax[1])
end

function Base.similar(::Type{PETScVector},ax::Tuple{<:ZeroTo})
  PETScVector(map(length,ax))
end


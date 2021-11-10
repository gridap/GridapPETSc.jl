
function Init(;args=String[],file="",help="",finalize_atexit=true)
  if !MPI.Initialized()
      MPI.Init()
  end
  if finalize_atexit
    atexit(Finalize)
  end
  Finalize()
  _args = ["GridapPETSc"]
  append!(_args,args)
  @check_error_code PETSC.PetscInitializeNoPointers(length(_args),_args,file,help)
  nothing
end

function Initialized()
  flag = Ref{PetscBool}()
  @check_error_code PETSC.PetscInitialized(flag)
  flag[] == PETSC.PETSC_TRUE
end

const _NREFS = Ref(0)

function Finalize()
  if Initialized()
    GC.gc() # Finalize all object out of scope at this point
    if _NREFS[] != 0
      @warn "$(_NREFS[]) objects still not finalized before calling GridapPETSc.Finalize()"
    end
    _NREFS[] = 0
    @check_error_code PETSC.PetscFinalize()
  end
  nothing
end

function with(f;kwargs...)
  Init(;kwargs...)
  f()
  Finalize()
end

const _INITIALIZED = Int8(0)
const _PENDING = Int8(1)
const _FINALIZED = Int8(2)
const _OID_TO_REF = Dict{UInt8,Int}()
const _STATES = Int8[]
const _REFS = Ref[] # This is type instable, but it should not be very problematic in this context

# Call this function each time a petsc wrapper object is initialized
function init_petsc_gc(a)
  oid = objectid(a)
  @check !haskey(_OID_TO_REF,oid)
  push!(_STATES,_INITIALIZED)
  push!(_REFS,get_ref(a))
  _OID_TO_REF[oid] = length(_REFS)
  nothing
end

# Install this function in a finalizer
# This function will be called by julia gc
function schedule_petsc_gc(a)
  # This if is needed because one can manually call Finalize()
  if a.initialized && GridapPETSc.Initialized()
    oid = objectid(a)
    @check haskey(_OID_TO_REF,oid)
    i = _OID_TO_REF[oid]
    @check _STATES[i] == _INITIALIZED
    _STATES[i] = _PENDING
    delete!(_OID_TO_REF,oid)
  end
  nothing
end

# This has to be called just before or after manually calling
# the C destroy function in PETSc (e.g. VecDestroy)
function finalize_petsc_gc(a)
  oid = objectid(a)
  @check haskey(_OID_TO_REF,oid)
  i = _OID_TO_REF[oid]
  @check _STATES[i] == _INITIALIZED
  _STATES[i] = _FINALIZED
  delete!(_OID_TO_REF,oid)
  nothing
end

# calls C destroy functions
# for orphan low level petsc handles
function petsc_gc()
  # Make sure that gc has processed all objects out of scope
  # This is needed to ensure that destroy is called collectively
  Base.gc()
  n = length(_REFS)
  # We destroy in LIFO order
  for i in reverse(1:n)
    if _STATES[i] == _PENDING
      destroy(_REFS[i])
      _STATES[i] = _FINALIZED
    end
  end
  # Now perform some cleanup (i.e, remove finalized entries)
  nj = 0
  for i in 1:n
    @assert _STATES[i] != _PENDING
    if _STATES[i] == _INITIALIZED
      nj += 1
    end
  end
  states = fill(_INITIALIZED,nj)
  refs = Vector{Ref}(undef,nj)
  j = zeros(Int,n)
  nj = 0
  for i in 1:n
    if _STATES[i] == _INITIALIZED
      nj += 1
      j[i] = j
      refs[nj] = _REFS[i]
    end
  end
  _STATES = states
  _REFS = refs
  for (k,i) in _OID_TO_REF
    _OID_TO_REF[k] = j[i]
  end
  nothing
end

destroy(r::Ref{Vec}) = @check_error_code PETSC.VecDestroy(r)
destroy(r::Ref{Mat}) = @check_error_code PETSC.MatDestroy(r)
destroy(r::Ref{KSP}) = @check_error_code PETSC.KSPDestroy(r)
destroy(r::Ref{SNES}) = @check_error_code PETSC.SNESDestroy(r)



function Init(;args=String[],file="",help="",finalize_atexit=true)
  if !MPI.Initialized()
      MPI.Init()
  end
  #To avoid multiple printing of the same line in parallel
  # if MPI.Comm_rank(MPI.COMM_WORLD) != 0
  #   redirect_stderr(devnull)
  #   redirect_stdout(devnull)
  # end 

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
  out = f()
  Finalize()
  out
end

# In an MPI environment context,
# this function has global collective semantics.
function gridap_petsc_gc()
  GC.gc()
  @check_error_code PETSC.PetscObjectRegisterDestroyAll()
end

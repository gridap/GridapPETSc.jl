
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


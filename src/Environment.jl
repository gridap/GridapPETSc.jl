
# This is needed since finalizers are called after atexit
const _REFS = Set{Ref}()

function _manage_mem(x)
  push!(_REFS,x)
  #finalizer(x) do y
  #  if y in _REFS
  #    _destroy(y)
  #    delete!(_REFS,y)
  #  end
  #  nothing
  #end
  x
end

_destroy(x::Ref{Vec}) = @check_error_code PETSC.VecDestroy(x)
_destroy(x::Ref{Mat}) = @check_error_code PETSC.MatDestroy(x)
_destroy(x::Ref{KSP}) = @check_error_code PETSC.KSPDestroy(x)

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

function Finalize()
  if Initialized()
    while length(_REFS) != 0
      _destroy(pop!(_REFS))
    end
    @check_error_code PETSC.PetscFinalize()
  end
  nothing
end




"""
    GridapPETSc.Init(;args=String[],file="",help="",finalize_atexit=true)

Initialize PETSc with optional command line arguments, a file name, and help text.
Wrapper for `PetscInitializeNoPointers`.
"""
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

"""
    GridapPETSc.Initialized()

Returns true if PETSc has been initialized. Wrapper for `PetscInitialized`.
"""
function Initialized()
  flag = Ref{PetscBool}()
  @check_error_code PETSC.PetscInitialized(flag)
  flag[] == PETSC.PETSC_TRUE
end

const _NREFS = Ref(0)

"""
    GridapPETSc.Finalize()

Finalize PETSc and clean up resources.
Wrapper for `PetscFinalize`.
"""
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

"""
    GridapPETSc.with(f; kwargs...)

Similar to the `Base.with` execution block, but for PETSc:

## Usage: 

```julia
  GridapPETSc.with() do
    # PETSc-dependent code here
  end
```
"""
function with(f;kwargs...)
  Init(;kwargs...)
  out = f()
  Finalize()
  out
end

"""
    GridapPETSc.gridap_petsc_gc()

Call `PetscObjectRegisterDestroyAll` to destroy all PETSc objects registered for destruction.
This is a collective operation and must be called on all processes at the same time.
"""
function gridap_petsc_gc()
  GC.gc()
  @check_error_code PETSC.PetscObjectRegisterDestroyAll()
end

"""
    GridapPETSc.destroy(a)

Call the appropriate PETSc destroy function for the object `a`.
This is a collective operation and must be called on all processes at the same time.
"""
function destroy(a)
  @notimplemented
end

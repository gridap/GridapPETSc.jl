deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")

if !isfile(deps_jl)
  error("Package GridapPETSc not installed properly.")
end

include(deps_jl)

if !PETSC_FOUND
  error("PETSc library not found.")
end

const PetscInitializeNoArguments_ptr  = Ref{Ptr}()
const PetscFinalize_ptr               = Ref{Ptr}()
const VecCreateSeqWithArray_ptr       = Ref{Ptr}()
const MatCreateSeqBAIJWithArrays_ptr  = Ref{Ptr}()
const MatCreateSeqSBAIJWithArrays_ptr = Ref{Ptr}()
const KSPCreate_ptr                   = Ref{Ptr}()
const KSPSetOperators_ptr             = Ref{Ptr}()
const KSPSolve_ptr                    = Ref{Ptr}()
const KSPSolveTranspose_ptr           = Ref{Ptr}()
const KSPDestroy_ptr                  = Ref{Ptr}()
const PETSC_LOADED                    = Ref(false)

function __init__()
    flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
    PETSC = Libdl.dlopen(PETSC_LIB, flags)

    GridapPETSc.PetscInitializeNoArguments_ptr[]  = Libdl.dlsym(PETSC,:PetscInitializeNoArguments)
    GridapPETSc.PetscFinalize_ptr[]               = Libdl.dlsym(PETSC,:PetscFinalize)
    GridapPETSc.VecCreateSeqWithArray_ptr[]       = Libdl.dlsym(PETSC,:VecCreateSeqWithArray)
    GridapPETSc.MatCreateSeqBAIJWithArrays_ptr[]   = Libdl.dlsym(PETSC,:MatCreateSeqBAIJWithArrays)
    GridapPETSc.MatCreateSeqSBAIJWithArrays_ptr[]   = Libdl.dlsym(PETSC,:MatCreateSeqSBAIJWithArrays)

    GridapPETSc.KSPCreate_ptr[]                   = Libdl.dlsym(PETSC,:KSPCreate)
    GridapPETSc.KSPSetOperators_ptr[]             = Libdl.dlsym(PETSC,:KSPSetOperators)
    GridapPETSc.KSPSolve_ptr[]                    = Libdl.dlsym(PETSC,:KSPSolve)
    GridapPETSc.KSPSolveTranspose_ptr[]           = Libdl.dlsym(PETSC,:KSPSolveTranspose)
    GridapPETSc.KSPDestroy_ptr[]                  = Libdl.dlsym(PETSC,:KSPDestroy)

    PETSC_LOADED[] = true
end


macro check_if_loaded()
  quote
    if ! PETSC_LOADED[]
      error("PETSc is not properly loaded")
    end
  end
end

const PetscScalar        = PETSC_SCALAR_DATATYPE
const PetscReal          = PETSC_REAL_DATATYPE
const PetscInt           = PETSC_INT_DATATYPE


deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")

if !isfile(deps_jl)
  error("Package GridapPETSc not installed properly.")
end

include(deps_jl)

if !PETSC_FOUND
  error("PETSc library not found.")
end

const PetscInitializeNoArguments_ptr  = Ref{Ptr}()
const PetscInitializeNoPointers_ptr   = Ref{Ptr}()
const PetscInitialized_ptr            = Ref{Ptr}()
const PetscFinalize_ptr               = Ref{Ptr}()
const PetscFinalized_ptr              = Ref{Ptr}()
const VecCreateSeqWithArray_ptr       = Ref{Ptr}()
const VecDestroy_ptr                  = Ref{Ptr}()
const VecView_ptr                     = Ref{Ptr}()
const MatCreateSeqBAIJWithArrays_ptr  = Ref{Ptr}()
const MatCreateSeqSBAIJWithArrays_ptr = Ref{Ptr}()
const MatGetSize_ptr                  = Ref{Ptr}()
const MatEqual_ptr                    = Ref{Ptr}()
const MatDestroy_ptr                  = Ref{Ptr}()
const MatView_ptr                     = Ref{Ptr}()
const KSPCreate_ptr                   = Ref{Ptr}()
const KSPSetOperators_ptr             = Ref{Ptr}()
const KSPSetFromOptions_ptr           = Ref{Ptr}()
const KSPSetUp_ptr                    = Ref{Ptr}()
const KSPSetOperators_ptr             = Ref{Ptr}()
const KSPSolve_ptr                    = Ref{Ptr}()
const KSPSolveTranspose_ptr           = Ref{Ptr}()
const KSPDestroy_ptr                  = Ref{Ptr}()
const PETSC_LOADED                    = Ref(false)

function __init__()
    flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
    PETSC = Libdl.dlopen(PETSC_LIB, flags)
    # Initialization / Finalization
    GridapPETSc.PetscInitializeNoArguments_ptr[]  = Libdl.dlsym(PETSC,:PetscInitializeNoArguments)
    GridapPETSc.PetscInitializeNoPointers_ptr[]   = Libdl.dlsym(PETSC,:PetscInitializeNoPointers)
    GridapPETSc.PetscInitialized_ptr[]            = Libdl.dlsym(PETSC,:PetscInitialized)
    GridapPETSc.PetscFinalize_ptr[]               = Libdl.dlsym(PETSC,:PetscFinalize)
    GridapPETSc.PetscFinalized_ptr[]              = Libdl.dlsym(PETSC,:PetscFinalized)
    # Vec
    GridapPETSc.VecCreateSeqWithArray_ptr[]       = Libdl.dlsym(PETSC,:VecCreateSeqWithArray)
    GridapPETSc.VecDestroy_ptr[]                  = Libdl.dlsym(PETSC,:VecDestroy)
    GridapPETSc.VecView_ptr[]                     = Libdl.dlsym(PETSC,:VecView)
    # Mat
    GridapPETSc.MatCreateSeqBAIJWithArrays_ptr[]  = Libdl.dlsym(PETSC,:MatCreateSeqBAIJWithArrays)
    GridapPETSc.MatCreateSeqSBAIJWithArrays_ptr[] = Libdl.dlsym(PETSC,:MatCreateSeqSBAIJWithArrays)
    GridapPETSc.MatGetSize_ptr[]                  = Libdl.dlsym(PETSC,:MatGetSize)
    GridapPETSc.MatEqual_ptr[]                    = Libdl.dlsym(PETSC,:MatEqual)
    GridapPETSc.MatDestroy_ptr[]                  = Libdl.dlsym(PETSC,:MatDestroy)
    GridapPETSc.MatView_ptr[]                     = Libdl.dlsym(PETSC,:MatView)
    # KSP
    GridapPETSc.KSPCreate_ptr[]                   = Libdl.dlsym(PETSC,:KSPCreate)
    GridapPETSc.KSPSetOperators_ptr[]             = Libdl.dlsym(PETSC,:KSPSetOperators)
    GridapPETSc.KSPSetFromOptions_ptr[]           = Libdl.dlsym(PETSC,:KSPSetFromOptions)
    GridapPETSc.KSPSetUp_ptr[]           = Libdl.dlsym(PETSC,:KSPSetUp)
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


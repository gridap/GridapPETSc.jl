const Petsc64bitInt       = Int64
const PetscBool           = UInt32
const PetscDataType       = Cint
const PetscLogDouble      = Cdouble
const PetscErrorCode      = Cint
const PetscViewer         = Ptr{Cvoid}

const MatFactorType       = UInt32

const PETSC_FALSE         = (UInt32)(0)
const PETSC_TRUE          = (UInt32)(1)

const MAT_FACTOR_NONE     = (UInt32)(0)
const MAT_FACTOR_LU       = (UInt32)(1)
const MAT_FACTOR_CHOLESKY = (UInt32)(2)
const MAT_FACTOR_ILU      = (UInt32)(3)
const MAT_FACTOR_ICC      = (UInt32)(4)
const MAT_FACTOR_ILUDT    = (UInt32)(5)

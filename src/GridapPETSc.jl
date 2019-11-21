module GridapPETSc

using Libdl
using MPI

# Initialization / Finalization
export PetscInitialize
export PetscFinalize

# Mat
export MatCreateSeqAIJWithArrays
export MatDestroy
export MatView

# Vec
export VecCreateSeqWithArray
export VecDestroy
export VecView

# KSP
export KSPCreate
export KSPSetOperators
export KSPSolve
export KSPDestroy

# GridapPETSc datatypes
export PetscMat
export PetscVec
export PetscKSP

include("load.jl")
include("const.jl")
include("init.jl")
include("vec.jl")
include("mat.jl")
include("ksp.jl")

end # module

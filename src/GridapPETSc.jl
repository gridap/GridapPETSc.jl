module GridapPETSc

using Libdl
using MPI

# Mat
export MatCreateSeqAIJWithArrays!
export MatCreateSeqAIJWithArrays
export MatDestroy!
export MatView

# Vec
export VecCreateSeqWithArray!
export VecCreateSeqWithArray
export VecDestroy!
export VecView

# KSP
export KSPCreate!
export KSPCreate
export KSPSetOperators!
export KSPSolve!
export KSPDestroy!

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

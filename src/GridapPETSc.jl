module GridapPETSc

using Libdl
using MPI
using SparseArrays
using Gridap.Algebra

# LinearSolver
import Gridap.Algebra: LinearSolver
import Gridap.Algebra: symbolic_setup, SymbolicSetup
import Gridap.Algebra: numerical_setup, numerical_setup!, NumericalSetup
import Gridap.Algebra: solve, solve!

# Supported PETSc data types 
export PetscInt
export PetscReal
export PetscScalar

# Mat
#export MatCreateSeqBAIJWithArrays!
#export MatCreateSeqBAIJWithArrays
#export MatCreateSeqSBAIJWithArrays!
#export MatCreateSeqSBAIJWithArrays
#export MatDestroy!
#export MatView

# Vec
#export VecCreateSeqWithArray!
#export VecCreateSeqWithArray
#export VecDestroy!
#export VecView

# KSP
#export KSPCreate!
#export KSPCreate
#export KSPSetOperators!
#export KSPSolve!
#export KSPSolveTranspose!
#export KSPDestroy!

# GridapPETSc datatypes
#export PetscMat
#export PetscVec
#export PetscKSP
export PETScSolver
export PETScSymbolicSetup
export PETScNumericalSetup

include("load.jl")
include("const.jl")
include("init.jl")
include("vec.jl")
include("mat.jl")
include("ksp.jl")
include("LinearSolver.jl")

end # module

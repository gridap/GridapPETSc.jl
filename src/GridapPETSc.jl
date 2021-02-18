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

# GridapPETSc datatypes
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

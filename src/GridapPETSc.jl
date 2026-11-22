module GridapPETSc

using MPI
using Libdl
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Arrays
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using PartitionedArrays

let deps_jl = joinpath(@__DIR__,"..","deps","deps.jl")

  if !isfile(deps_jl)
    msg = """
    GridapPETSc needs to be configured before use. Type

    pkg> build

    and try again.
    """
    error(msg)
  end

  include(deps_jl)
end

if !libpetsc_found
  msg = """
  GridapPETSc was not configured correcnly. See the errors in file:

  $(joinpath(@__DIR__,"..","deps","build.log"))

  If you are using the environment variable JULIA_PETSC_LIBRARY, make sure
  that it points to a correct PETSc dynamic library object.

  Solve the issue and

  pkg> build

  again.
  """
end

@static if libpetsc_provider != "JULIA_PETSC_LIBRARY"
  using PETSc_jll
end

const libpetsc_handle = Ref{Ptr{Cvoid}}()
include("Version.jl")

const _PRELOADS = Tuple{Ref{Ptr{Cvoid}},Symbol}[]

function __init__()
  if libpetsc_provider == "JULIA_PETSC_LIBRARY"
    path = libpetsc_path
  else
    path = PETSc_jll.libpetsc_path
  end
  # We always dlopen by path, since PETSc_jll >= 3.25 does not load the library itself
  flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
  libpetsc_handle[] = Libdl.dlopen(path, flags)

  # The PETSc ABI can change between minor versions, but not between patch releases
  version = get_petsc_version(libpetsc_handle[])
  if Base.thisminor(version) != Base.thisminor(libpetsc_version)
    error("""
    GridapPETSc was built for PETSc v$(libpetsc_version), but is loading PETSc v$(version) from

    $(path)

    Type

    pkg> build GridapPETSc

    and restart Julia.
    """)
  end

  for (handle,sym) in _PRELOADS
    _handle = Libdl.dlsym(libpetsc_handle[],sym;throw_error=false)
    if _handle !== nothing
      handle[] = _handle
    end
  end
end

include("PETSC.jl")

using GridapPETSc.PETSC: @check_error_code
using GridapPETSc.PETSC: PetscBool, PetscInt, PetscReal, PetscScalar, Vec, Mat, KSP, PC, SNES, IS

export @check_error_code
export PetscBool, PetscInt, PetscReal, PetscScalar, Vec, Mat, KSP, PC, SNES

include("Environment.jl")

export PETScVector
export PETScMatrix
export PETScIndexSet
export petsc_sparse
include("PETScArrays.jl")
include("PETScIndexSets.jl")
include("PartitionedArrays.jl")

export PETScLinearSolver
include("PETScLinearSolvers.jl")

export PETScNonlinearSolver
include("PETScNonlinearSolvers.jl")

include("PETScAssembly.jl")

end # module

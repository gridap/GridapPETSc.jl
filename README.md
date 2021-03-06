# GridapPETSc

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapPETSc.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapPETSc.jl/dev)
[![Build Status](https://github.com/gridap/GridapPETSc.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridap/GridapPETSc.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/gridap/GridapPETSc.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapPETSc.jl)

[Gridap](https://github.com/gridap/Gridap.jl) plugin to use PETSC ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)).

## Installation

`GridapPETSc` julia package requires the `PETSC` library ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)) and `MPI` to work correctly. You have two main options to install these dependencies. 

- **Do nothing [recommended in most cases].** Use the default precompiled `MPI` installation provided by [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) and the pre-compiled `PETSc` library provided by [`PETSc_jll`](https://github.com/JuliaBinaryWrappers/PETSc_jll.jl). This will happen under the hood when you install `GridapPETSc`. You can also force the installation of these default dependencies by setting the environment variables `JULIA_MPI_BINARY` and `JULIA_PETSC_LIBRARY` to empty values.

- **Choose a specific installation of `MPI` and `PETSc` available in the system [recommended in HPC clusters]**.
  - First, choose a `MPI` installation. See the documentation of  [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) for further details. An easy way to achieve this is to create the environment variable `JULIA_MPI_BINARY` containing the path to the  `MPI` binary.
  - Second, choose a `PETSc` installation. To this end, create an environment variable `JULIA_PETSC_LIBRARY` containing the path to the dynamic library object of the `PETSC` installation (i.e., the `.so` file in linux systems). **Very important: The chosen `PETSc` lirbary needs to be configured with the `MPI` installation considered in previous step**.


## Notes

`GridapPETSc` default sparse matrix format is 0-based compressed sparse row. This types of sparse matrix can be described by `SparseMatrixCSR{0,PetscReal,PetscInt}` and `SymSparseMatrixCSR{0,PetscReal,PetscInt}` implemented in the [SparseMatricesCSR](https://gridap.github.io/SparseMatricesCSR.jl/stable/) Julia package.

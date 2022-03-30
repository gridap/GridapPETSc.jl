# GridapPETSc

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapPETSc.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapPETSc.jl/dev)
[![Build Status](https://github.com/gridap/GridapPETSc.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridap/GridapPETSc.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/gridap/GridapPETSc.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapPETSc.jl)

`GridapPETSc` is a plugin of [`GridapDistributed.jl`](https://github.com/gridap/GridapDistributed.jl) that provides the  full set of scalable linear and nonlinear solvers in the [PETSc](https://petsc.org/release/) library. It also provides serial solvers to [`Gridap.jl`](https://github.com/gridap/Gridap.jl).

## Documentation

Take a look at this [tutorial](https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/#Tutorial-16:-Poisson-equation-on-parallel-distributed-memory-computers-1) for learning how to use `GridapPETSc` in distributed-memory simulations of PDEs. 

It can also be used in the serial case, as shown in this [test](https://github.com/gridap/GridapPETSc.jl/blob/master/test/sequential/PoissonDriver.jl).

## Installation

`GridapPETSc` julia package requires the `PETSC` library ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)) and `MPI` to work correctly. You have two main options to install these dependencies. 

- **Do nothing [recommended in most cases].** Use the default precompiled `MPI` installation provided by [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) and the pre-compiled `PETSc` library provided by [`PETSc_jll`](https://github.com/JuliaBinaryWrappers/PETSc_jll.jl). This will happen under the hood when you install `GridapPETSc`. You can also force the installation of these default dependencies by setting the environment variables `JULIA_MPI_BINARY` and `JULIA_PETSC_LIBRARY` to empty values.

- **Choose a specific installation of `MPI` and `PETSc` available in the system [recommended in HPC clusters]**.
  - First, choose a `MPI` installation. See the documentation of  [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) for further details. An easy way to achieve this is to create the environment variable `JULIA_MPI_BINARY` containing the path to the  `MPI` binary.
  - Second, choose a `PETSc` installation. To this end, create an environment variable `JULIA_PETSC_LIBRARY` containing the path to the dynamic library object of the `PETSC` installation (i.e., the `.so` file in linux systems). **Very important: The chosen `PETSc` library needs to be configured with the `MPI` installation considered in the previous step**.


## Notes

* `GridapPETSc` default sparse matrix format is 0-based compressed sparse row. This type of sparse matrix storage format can be described by the `SparseMatrixCSR{0,PetscReal,PetscInt}` and `SymSparseMatrixCSR{0,PetscReal,PetscInt}` Julia types as implemented in the [SparseMatricesCSR](https://gridap.github.io/SparseMatricesCSR.jl/stable/) Julia package.
* **When running in MPI parallel mode** (i.e., with a MPI communicator different from `MPI.COMM_SELF`), `GridapPETSc` implements a sort of limited garbage collector in order to automatically deallocate PETSc objects. This garbage collector can be manually triggered by a call to the function `GridapPETSc.gridap_petsc_gc()`. `GridapPETSc` automatically calls this function inside at different strategic points, and **this will be sufficient for most applications**. However, for some applications, with a very frequent allocation of PETSc objects, it might be needed to call this function from application code. This need will be signaled by PETSc via the following internal message error `PETSC ERROR: No more room in array, limit 256 
recompile src/sys/objects/destroy.c with larger value for MAXREGDESOBJS`

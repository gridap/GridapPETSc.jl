# GridapPETSc

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapPETSc.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapPETSc.jl/dev)
[![Build Status](https://travis-ci.com/gridap/GridapPETSc.jl.svg?branch=master)](https://travis-ci.com/gridap/GridapPETSc.jl)
[![Codecov](https://codecov.io/gh/gridap/GridapPETSc.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapPETSc.jl)

[Gridap](https://github.com/gridap/Gridap.jl) (Grid-based approximation of partial differential equations in Julia) plugin to use PETSC ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)).

## Basic Usage

```julia
using MPI
using Gridap
using GridapPETSC

MPI.Init()
GridapPETSc.Init()

A = sparse([1,2,3,4,5],[1,2,3,4,5],[1.0,2.0,3.0,4.0,5.0])
b = ones(A.n)
x = similar(b)
ps = PETScSolver()
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)

GridapPETSc.Finalize()
MPI.Finalize()
```

## Usage in a Finite Element computation

```julia
using Gridap
using GridapPETSc

MPI.Init()
GridapPETSc.Init()

# Define the FE problem
# -Δu = x*y in (0,1)^3, u = 0 on the boundary.

model = CartesianDiscreteModel((0,1,0,1,0,1), (10,10,10))

V = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

U = TrialFESpace(V)

trian = get_triangulation(model)
quad = CellQuadrature(trian,2)

t_Ω = AffineFETerm(
  (v,u) -> inner(∇(v),∇(u)),
  (v) -> inner(v, (x) -> x[1]*x[2] ),
  trian, quad)

op = AffineFEOperator(SparseMatrixCSR{0,PetscReal,PetscInt},V,U,t_Ω)

ls = PETScSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

GridapPETSc.Finalize()
MPI.Finalize()
```

## Installation

**GridPETSc** itself is installed when you add and use it into another project.

Please, ensure that your system fulfill the requirements.

To include into your project form Julia REPL, use the following commands:

```
pkg> add GridapPETSc
julia> using GridapPETSc
```

If, for any reason, you need to manually build the project, write down the following commands in Julia REPL:
```
pkg> add GridapPETSc
pkg> build GridPETSc
julia> using GridapPETSc
```

### Requirements

`GridapPETSc` julia package requires `PETSC` library ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)) and `OPENMPI` to work correctly. `PETSc` library can be manually installed in any path on your local machine. In order to succesfull describe your custom installation to be located by `GridapPETSc`, you must export `PETSC_DIR` and `PETSC_ARCH` environment variables. If this environment variables are not available, `GridapPETSc` will try to find the library in the usual linux user library directory (`/usr/lib`).

`PETSC_DIR` and `PETSC_ARCH` are a couple of variables that control the configuration and build process of PETSc: 

  - `PETSC_DIR`: this variable should point to the location of the PETSc installation that is used. Multiple PETSc versions can coexist on the same file-system. By changing `PETSC_DIR` value, one can switch between these installed versions of PETSc.
  - `PETSC_ARCH`: this variable gives a name to a configuration/build. Configure uses this value to stores the generated config makefiles in `${PETSC_DIR}/${PETSC_ARCH}`. Make uses this value to determine this location of these makefiles which intern help in locating the correct include and library files.

Thus one can install multiple variants of PETSc libraries - by providing different `PETSC_ARCH` values to each configure build. Then one can switch between using these variants of libraries from make by switching the `PETSC_ARCH` value used.

If configure doesn't find a `PETSC_ARCH` value (either in env variable or command line option), it automatically generates a default value and uses it. Also - if make doesn't find a `PETSC_ARCH` env variable - it defaults to the value used by last successful invocation of previous configure. `PETSC_ARCH` value can be an empty string too.

#### Basic PETSc installation on Debian-based systems

`PETSc` can be obtained from the default repositories of your Debian-based OS by means of `apt` tool.

Basic `PETSc` installation in order to use it from `GridapPETSc` julia package is as follows:

```
$ sudo apt-get update
$ sudo apt-get openmpi petsc-dev
```

## Continuous integration

In order to take advantage of `GridapPETSc` julia package during continuous integration, you must ensure that the requirements are fullfilled in the CI environment.

If your CI process is based on `Travis-CI` you can add the following block at the beginning of your `.travis.yml` file:

```
addons:
  apt:
    update: true
    packages:
    - openmpi-bin
    - petsc-dev
```

## Notes

`GridapPETSc` default sparse matrix format is 0-based compressed sparse row. This types of sparse matrix can be described by `SparseMatrixCSR{0,PetscReal,PetscInt}` and `SymSparseMatrixCSR{0,PetscReal,PetscInt}`.These types of matrix are implemented in the [SparseMatricesCSR](https://gridap.github.io/SparseMatricesCSR.jl/stable/)) julia package.

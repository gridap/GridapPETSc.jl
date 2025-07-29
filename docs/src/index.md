# GridapPETSc.jl

`GridapPETSc` is a plugin of [`GridapDistributed.jl`](https://github.com/gridap/GridapDistributed.jl) that provides the  full set of scalable linear and nonlinear solvers in the [PETSc](https://petsc.org/release/) library. It also provides serial solvers to [`Gridap.jl`](https://github.com/gridap/Gridap.jl).

## Installation

`GridapPETSc` requires the `PETSC` library ([Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)) and `MPI` to work correctly. You have two main options to install these dependencies:

- **Do nothing [recommended in most cases].** Use the default precompiled `MPI` installation provided by [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) and the pre-compiled `PETSc` library provided by [`PETSc_jll`](https://github.com/JuliaBinaryWrappers/PETSc_jll.jl). This will happen under the hood when you install `GridapPETSc`. In the case of `GridapPETSc`, you can also force the installation of these default dependencies by setting the environment variable `JULIA_PETSC_LIBRARY` to an empty value.

- **Choose a specific installation of `MPI` and `PETSc` available in the system [recommended in HPC clusters]**.
  - First, choose a `MPI` installation. See the documentation of  [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) for further details.
  - Second, choose a `PETSc` installation. To this end, create an environment variable `JULIA_PETSC_LIBRARY` containing the path to the dynamic library object of the `PETSC` installation (i.e., the `.so` file in linux systems). **Very important: The chosen `PETSc` library needs to be configured with the `MPI` installation considered in the previous step**.

## Memory management

In Julia, memory management is handled by the garbage collector (GC). When an object is unreachable, the GC will automatically reclaim its memory. When using distributed computing, the GC jobs are run independently on each task. This is not a problem for our own Julia objects, but it causes quite a lot of issues when combined with PETSc: The PETSc destroy routines have collective semantics, meaning they must be called on all processes at the same time. If this is not the case, the application will be softlocked.

### Mimicking GC using finalizers

In order to maintain a GC-like experience within `GridapPETSc`, we use [finalizers](https://docs.julialang.org/en/v1/base/base/#Base.finalizer) for all our PETSc-related objects (`PETScVector`, `PETScMatrix`, ...).

Within the finalizer, called by Julia's GC, we call [`PetscObjectRegisterDestroy`](https://petsc.org/release/manualpages/Sys/PetscObjectRegisterDestroy/). This is non-blocking and will therefore not cause a softlock. However it does not immediatley destroy the PETSc object, but rather delays its destruction until [`PetscObjectRegisterDestroyAll`](https://petsc.org/release/manualpages/Sys/PetscObjectRegisterDestroyAll/) is called. This is usually called by PETSc at the end of the program, but can also be called manually by the user.

The above, however, it heavily discouraged by PETSc. Moreover, there is a maximum number of objects that can be registered (defaults to 256, can be changed on compilation). The call to `PetscObjectRegisterDestroyAll` can be forced by calling [`GridapPETSc.gridap_petsc_gc`](@ref), which will in theory empty the queue. We have noticed, however, that it can lead PETSc to crash.

### Manual memory management

We therefore recommend the use of the [`GridapPETSc.destroy`](@ref), which will call the appropriate destroy routine for each PETSc object. Note that, as mentioned above, this is a collective operation and must be called on all processes at the same time.

An important note is that the [`GridapPETSc.destroy`](@ref) function will **NOT** destroy the Julia object itself, but only the underlying PETSc object. The Julia interface will be marked as un-initialized, but the memory will not be freed until Julia's GC runs on it.

In some cases, and whenever possible, the underlying memory is aliased (shared) between the Julia interface and it's PETSc object. In this case, the memory itself is owned by Julia (not PETSc!) and will be freed when the Julia object is destroyed (not when [`GridapPETSc.destroy`](@ref) is called!). This means one can safely destroy the PETSc object without worrying about consequences on Julia's GC.

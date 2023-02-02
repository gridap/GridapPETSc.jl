module PETScIndexesTests

using GridapPETSc
using Test
using SparseArrays
using SparseMatricesCSR
using GridapPETSc: PetscScalar, PetscInt
using LinearAlgebra




options = "-info"
GridapPETSc.with(args=split(options)) do

array = collect(1:1:100)
is = PETScIS(array)
@check_error_code GridapPETSc.PETSC.ISView(is.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD)
@check_error_code GridapPETSc.PETSC.ISView(is.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_SELF)
@test is.size[1] == length(array)
@test is.initialized == true
end


end # module

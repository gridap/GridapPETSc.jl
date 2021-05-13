module BindingsTests

using Test
using MPI
using GridapPETSc
using GridapPETSc: PetscBool, PETSC_TRUE, PETSC_FALSE
using GridapPETSc: PetscInt, PetscScalar
using GridapPETSc: Vec, INSERT_VALUES, ADD_VALUES
using GridapPETSc: @check_error_code
using GridapPETSc: @PETSC_VIEWER_STDOUT_SELF, @PETSC_VIEWER_STDOUT_WORLD

if !MPI.Initialized()
    MPI.Init()
end

flag = Ref{PetscBool}()
@check_error_code GridapPETSc.PetscInitialized(flag)
if flag[] == PETSC_TRUE
  @check_error_code GridapPETSc.PetscFinalize()
end
args = ["julia"]#,"-info"]
argc = length(args)
file = ""
help = ""
@check_error_code GridapPETSc.PetscInitializeNoPointers(argc,args,file,help)

comm = MPI.COMM_SELF
bs = PetscInt(1)
array = PetscScalar[1,2,4,1]
n = PetscInt(length(array))
vec = Ref{Vec}()
@check_error_code GridapPETSc.VecCreateSeqWithArray(comm,bs,n,array,vec)

@check_error_code GridapPETSc.VecView(vec[],@PETSC_VIEWER_STDOUT_SELF)
@check_error_code GridapPETSc.VecView(vec[],@PETSC_VIEWER_STDOUT_WORLD)

ids = PetscInt[0,1,3]
vals = PetscScalar[10,20,30]
@check_error_code GridapPETSc.VecSetValues(vec[],length(ids),ids,vals,INSERT_VALUES)
@check_error_code GridapPETSc.VecSetValues(vec[],length(ids),ids,vals,ADD_VALUES)
@check_error_code GridapPETSc.VecView(vec[],C_NULL)

vals2 = similar(vals)
@check_error_code GridapPETSc.VecGetValues(vec[],length(ids),ids,vals2)

@test array[ids.+1] == vals2

@check_error_code GridapPETSc.VecAssemblyBegin(vec[])
@check_error_code GridapPETSc.VecAssemblyEnd(vec[])

@check_error_code GridapPETSc.VecDestroy(vec)

@check_error_code GridapPETSc.PetscInitialized(flag)
if flag[] == PETSC_TRUE
  @check_error_code GridapPETSc.PetscFinalize()
end

end

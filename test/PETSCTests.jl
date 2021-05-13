module BindingsTests

using Test
using MPI
using GridapPETSc.PETSC

if !MPI.Initialized()
    MPI.Init()
end

flag = Ref{PETSC.PetscBool}()
PETSC.@check_error_code PETSC.PetscInitialized(flag)
if flag[] == PETSC.PETSC_TRUE
  PETSC.@check_error_code PETSC.PetscFinalize()
end
args = ["julia"]#,"-info"]
argc = length(args)
file = ""
help = ""
PETSC.@check_error_code PETSC.PetscInitializeNoPointers(argc,args,file,help)

comm = MPI.COMM_SELF
bs = PETSC.PetscInt(1)
array = PETSC.PetscScalar[1,2,4,1]
n = PETSC.PetscInt(length(array))
vec = Ref{PETSC.Vec}()
PETSC.@check_error_code PETSC.VecCreateSeqWithArray(comm,bs,n,array,vec)

PETSC.@check_error_code PETSC.VecView(vec[],PETSC.@PETSC_VIEWER_STDOUT_SELF)
PETSC.@check_error_code PETSC.VecView(vec[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)

ids = PETSC.PetscInt[0,1,3]
vals = PETSC.PetscScalar[10,20,30]
PETSC.@check_error_code PETSC.VecSetValues(vec[],length(ids),ids,vals,PETSC.INSERT_VALUES)
PETSC.@check_error_code PETSC.VecSetValues(vec[],length(ids),ids,vals,PETSC.ADD_VALUES)
PETSC.@check_error_code PETSC.VecView(vec[],C_NULL)

vals2 = similar(vals)
PETSC.@check_error_code PETSC.VecGetValues(vec[],length(ids),ids,vals2)

@test array[ids.+1] == vals2

PETSC.@check_error_code PETSC.VecAssemblyBegin(vec[])
PETSC.@check_error_code PETSC.VecAssemblyEnd(vec[])
PETSC.@check_error_code PETSC.VecDestroy(vec)

PETSC.@check_error_code PETSC.PetscInitialized(flag)
if flag[] == PETSC.PETSC_TRUE
 PETSC.@check_error_code PETSC.PetscFinalize()
end

end

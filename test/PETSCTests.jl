module BindingsTests

using Test
using MPI
using SparseMatricesCSR
using GridapPETSc.PETSC
using GridapPETSc.PETSC: @check_error_code
using GridapPETSc.PETSC: PetscBool, PetscInt, PetscScalar, Vec, Mat

if !MPI.Initialized()
    MPI.Init()
end

flag = Ref{PetscBool}()
@check_error_code PETSC.PetscInitialized(flag)
if flag[] == PETSC.PETSC_TRUE
  @check_error_code PETSC.PetscFinalize()
end
args = ["julia"]#,"-info"]
argc = length(args)
file = ""
help = ""
@check_error_code PETSC.PetscInitializeNoPointers(argc,args,file,help)

comm = MPI.COMM_SELF
bs = PetscInt(1)
array = PetscScalar[1,2,4,1]
n = PetscInt(length(array))
vec = Ref{Vec}()
@check_error_code PETSC.VecCreateSeqWithArray(comm,bs,n,array,vec)

@check_error_code PETSC.VecView(vec[],PETSC.@PETSC_VIEWER_STDOUT_SELF)
@check_error_code PETSC.VecView(vec[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)

ids = PETSC.PetscInt[0,1,3]
vals = PETSC.PetscScalar[10,20,30]
@check_error_code PETSC.VecSetValues(vec[],length(ids),ids,vals,PETSC.INSERT_VALUES)
@check_error_code PETSC.VecSetValues(vec[],length(ids),ids,vals,PETSC.ADD_VALUES)
@check_error_code PETSC.VecView(vec[],C_NULL)

vals2 = similar(vals)
@check_error_code PETSC.VecGetValues(vec[],length(ids),ids,vals2)

@test array[ids.+1] == vals2

@check_error_code PETSC.VecAssemblyBegin(vec[])
@check_error_code PETSC.VecAssemblyEnd(vec[])
@check_error_code PETSC.VecDestroy(vec)

mat = Ref{Mat}()
m = PetscInt(4)
n = PetscInt(5)
nz = PETSC.PETSC_DEFAULT
#nz = PetscInt(3)
nnz = C_NULL
@check_error_code PETSC.MatCreateSeqAIJ(comm,m,n,nz,nnz,mat)

idxm = PetscInt[0,2,1]
idxn = PetscInt[3,1]
v = PetscScalar[1,2,3,4,5,6]
@check_error_code PETSC.MatSetValues(mat[],length(idxm),idxm,length(idxn),idxn,v,PETSC.ADD_VALUES)

idxm = PetscInt[0,3,1]
idxn = PetscInt[4,3]
v = PetscScalar[1,2,3,4,5,6]
@check_error_code PETSC.MatSetValues(mat[],length(idxm),idxm,length(idxn),idxn,v,PETSC.ADD_VALUES)

@test Int(PETSC.MAT_FINAL_ASSEMBLY) == 0
@test Int(PETSC.MAT_FLUSH_ASSEMBLY) == 1
@check_error_code PETSC.MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
@check_error_code PETSC.MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)

v2 = similar(v)
idxm = PetscInt[0,2,1]
idxn = PetscInt[3,1]
@check_error_code PETSC.MatGetValues(mat[],length(idxm),idxm,length(idxn),idxn,v2)
@test v2 == PetscScalar[3.0, 2.0, 3.0, 4.0, 11.0, 6.0]

@check_error_code PETSC.MatView(mat[],C_NULL)

@check_error_code PETSC.MatDestroy(mat)

I = PetscInt[1,3,4,2]
J = PetscInt[3,1,1,5]
V = PetscScalar[1,2,3,4]
m = PetscInt(4)
n = PetscInt(5)
A = sparsecsr(Val(0),I,J,V,m,n)

i = A.rowptr
j = A.colval
a = A.nzval
@check_error_code PETSC.MatCreateSeqAIJWithArrays(comm,m,n,i,j,a,mat)

@check_error_code PETSC.MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
@check_error_code PETSC.MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)

@check_error_code PETSC.MatView(mat[],C_NULL)

@check_error_code PETSC.MatDestroy(mat)

@check_error_code PETSC.PetscInitialized(flag)
if flag[] == PETSC.PETSC_TRUE
 @check_error_code PETSC.PetscFinalize()
end

end

using GridapPETSc
using Test

using MPI
using SparseArrays
using SparseMatricesCSR

tol = 1.0e-13

# Define 0-based CSR sparse matrix data
m = 5
n = 5
I_ = [1,1,1,2,2,3,3,3,4,4,4,5,5] 
J_ = [1,2,4,1,2,3,4,5,1,3,4,2,5]
V_ = [1.0,-1,-3,-2,5,4,6,4,-4,2,7,8,-5]
I = Vector{GridapPETSc.PetscInt}()
J = Vector{GridapPETSc.PetscInt}()
V = Vector{GridapPETSc.PetscScalar}()

for (ik, jk, vk) in zip(I_, J_, V_)
    push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
end

A = sparsecsr(SparseMatrixCSR{0},I, J, V,m,n)

# Define vectors
bvector = ones(GridapPETSc.PetscScalar,m)
xvector = ones(GridapPETSc.PetscScalar,n)

# Define PETSc data types wrappers
Mat = PetscMat()
b   = PetscVec()
x   = PetscVec()
Ksp = PetscKSP()

#####################################
# PETSc basic workflow
#####################################
# Initialization
MPI.Init()
GridapPETSc.init!(["-info","-malloc_debug","-malloc_dump","-malloc_test","-mat_view", "::ascii_info_detail"]) 

# Create objects
error = VecCreateSeqWithArray!(MPI.COMM_SELF,1,m,bvector,b)
@test iszero(error)
error = VecCreateSeqWithArray!(MPI.COMM_SELF,1,n,xvector,x)
@test iszero(error)
error = MatCreateSeqAIJWithArrays!(MPI.COMM_SELF, m, n, getptr(A), getindices(A), nonzeros(A), Mat)
@test iszero(error)
error = KSPCreate!(MPI.COMM_SELF, Ksp)
@test iszero(error)

# Show data objects
error = VecView(x)
@test iszero(error)
error = VecView(b)
@test iszero(error)
error = MatView(Mat)
@test iszero(error)

# Solve
error = KSPSetOperators!(Ksp, Mat, Mat)
@test iszero(error)
error = KSPSolve!(Ksp, b, x)
@test iszero(error)

@test maximum(abs.(A*xvector-bvector)) < tol

# Destroy objects
error = KSPDestroy!(Ksp)
@test iszero(error)
error = MatDestroy!(Mat)
@test iszero(error)
error = VecDestroy!(b)
@test iszero(error)
error = VecDestroy!(x)
@test iszero(error)

# Finlization
GridapPETSc.finalize!()
MPI.Finalize()

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
PetscInitialize(["-info","-malloc_debug","-malloc_dump","-malloc_test","-mat_view", "::ascii_info_detail"]) 

# Create objects
VecCreateSeqWithArray(MPI.COMM_SELF,1,m,bvector,b)
VecCreateSeqWithArray(MPI.COMM_SELF,1,n,xvector,x)
MatCreateSeqAIJWithArrays(MPI.COMM_SELF, m, n, getptr(A), getindices(A), nonzeros(A), Mat)
KSPCreate(MPI.COMM_SELF, Ksp)

# Show data objects
VecView(x)
VecView(b)
MatView(Mat)

# Solve
KSPSetOperators(Ksp, Mat, Mat)
KSPSolve(Ksp, b, x)

@test maximum(abs.(A*xvector-bvector)) < tol

# Destroy objects
KSPDestroy(Ksp)
MatDestroy(Mat)
VecDestroy(b)
VecDestroy(x)

# Finlization
PetscFinalize()
MPI.Finalize()

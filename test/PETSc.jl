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
    push_coo!(SparseMatrixCSR, I, J, V, ik, jk, vk)
end

A = sparsecsr(SparseMatrixCSR{0}, I, J, V, m, n)

# Define native Julia vectors
B = ones(GridapPETSc.PetscScalar, m)
X = ones(GridapPETSc.PetscScalar, n)

#####################################
# PETSc basic workflow
#####################################
# Initialization
GridapPETSc.Init!(["-info","-malloc_debug","-malloc_dump","-malloc_test","-mat_view", "::ascii_info_detail"]) 

# Create objects
b = GridapPETSc.VecCreateSeqWithArray(MPI.COMM_SELF, 1, m, B)
x = GridapPETSc.VecCreateSeqWithArray(MPI.COMM_SELF, 1, n, X)
Mat = GridapPETSc.MatCreateSeqBAIJWithArrays(MPI.COMM_SELF, 1, m, n, getptr(A), getindices(A), nonzeros(A))
Ksp = GridapPETSc.KSPCreate(MPI.COMM_SELF)

# Show data objects
error = GridapPETSc.VecView(x)
@test iszero(error)
error = GridapPETSc.VecView(b)
@test iszero(error)
error = GridapPETSc.MatView(Mat)
@test iszero(error)

# Solve
error = GridapPETSc.KSPSetOperators!(Ksp, Mat, Mat)
@test iszero(error)
error = GridapPETSc.KSPSolve!(Ksp, b, x)
@test iszero(error)

@test maximum(abs.(A*X-B)) < tol

# Destroy objects
error = GridapPETSc.KSPDestroy!(Ksp)
@test iszero(error)
error = GridapPETSc.MatDestroy!(Mat)
@test iszero(error)
error = GridapPETSc.VecDestroy!(b)
@test iszero(error)
error = GridapPETSc.VecDestroy!(x)
@test iszero(error)

# Finalization
GridapPETSc.Finalize!()


module PETScSolversTests

using SparseMatricesCSR
using Gridap.Algebra
using GridapPETSc
using GridapPETSc.PETSC: PetscInt, PetscScalar, @check_error_code
using GridapPETSc.PETSC
using Test

options = "-ksp_type gmres -ksp_monitor -pc_type ilu"
GridapPETSc.Init(args=split(options))

I = PetscInt[1,1,2,2,2,3,3,3,4,4]
J = PetscInt[1,2,1,2,3,2,3,4,3,4]
V = PetscScalar[4,-2,-1,6,-2,-1,6,-2,-1,4]
m = PetscInt(4)
n = PetscInt(4)
A = sparsecsr(Val(0),I,J,V,m,n)

x = ones(PetscScalar,m)
b = A*x

# Setup solver via cml options
solver = PETScSolver()
@check_error_code PETSC.KSPSetFromOptions(solver.ksp[])

x2 = solve(solver,A,b)
@test x ≈ x2

ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)
x2 .= 0
solve!(x2,ns,b)
solve!(x2,ns,b)
@test x ≈ x2

test_linear_solver(solver,A,b,x)
test_linear_solver(solver,A,b,x)

# Setup solver via low level PETSC API calls
solver = PETScSolver()
pc = Ref{PETSC.PC}()
@check_error_code PETSC.KSPSetType(solver.ksp[],PETSC.KSPGMRES)
@check_error_code PETSC.KSPGetPC(solver.ksp[],pc)
@check_error_code PETSC.PCSetType(pc[],PETSC.PCJACOBI)
@check_error_code PETSC.KSPView(solver.ksp[],C_NULL)

x2 = solve(solver,A,b)
@test x ≈ x2

# Default solver
solver = PETScSolver()

x2 = solve(solver,A,b)
@test x ≈ x2

solver = nothing
GC.gc()

end # module

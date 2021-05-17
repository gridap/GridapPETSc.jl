module PETScSolversTests

using SparseArrays
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

B = sparsecsr(Val(0),I,J,PetscInt(2)*V,m,n)
c = B*x

numerical_setup!(ns,B)
solve!(x2,ns,B*x)
@test x ≈ x2

# Move ns out of scope before calling GridapPETSc.Finalize()
ns = nothing

C = sparsecsr(Val(0),1*I,1*J,2*V,m,n)
x2 = solve(solver,C,C*x)
@test x ≈ x2
test_linear_solver(solver,C,C*x,x)

C = sparse(I,J,V,m,n)
x2 = solve(solver,C,C*x)
@test x ≈ x2
test_linear_solver(solver,C,C*x,x)


# Setup solver via low level PETSC API calls
function mykspsetup(ksp)
  pc = Ref{PETSC.PC}()
  @check_error_code PETSC.KSPSetType(ksp[],PETSC.KSPGMRES)
  @check_error_code PETSC.KSPGetPC(ksp[],pc)
  @check_error_code PETSC.PCSetType(pc[],PETSC.PCJACOBI)
  @check_error_code PETSC.KSPView(ksp[],C_NULL)
end
solver = PETScSolver(mykspsetup)

x2 = solve(solver,A,b)
@test x ≈ x2

GridapPETSc.Finalize()

options = "-ksp_monitor"
GridapPETSc.with(args=split(options)) do

  I = PetscInt[1,1,2,2,2,3,3,3,4,4]
  J = PetscInt[1,2,1,2,3,2,3,4,3,4]
  V = PetscScalar[4,-2,-1,6,-2,-1,6,-2,-1,4]
  m = PetscInt(4)
  n = PetscInt(4)

  A = petsc_sparse(I,J,V,m,n)
  x = similar(A,size(A,2))
  @test typeof(x) == PETScVector
  fill!(x,1)
  b = A*x
  @test typeof(b) == PETScVector

  solver = PETScSolver()
  x2 = solve(solver,A,b)
  @test typeof(x2) == PETScVector
  @test x ≈ x2

  test_linear_solver(solver,A,b,x)
  test_linear_solver(solver,A,b,x)

end

end # module

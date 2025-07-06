
struct HPDDMLinearSolver <: LinearSolver
  setup
  mats
  gids
end

struct HPDDMLinearSolverSS <: SymbolicSetup
  solver::HPDDMLinearSolver
end

function Algebra.symbolic_setup(solver::HPDDMLinearSolver,mat::AbstractMatrix)
  HPDDMLinearSolverSS(solver)
end

function Algebra.numerical_setup(ss::HPDDMLinearSolverSS,A::AbstractMatrix)
  B = convert(PETScMatrix,A)
  ns = PETScLinearSolverNS(A,B)
  @check_error_code PETSC.KSPCreate(B.comm,ns.ksp)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  hpddm_setup(ss.solver,ns.ksp)
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  Init(ns)
end

function hpddm_setup(solver::HPDDMLinearSolver,ksp)
  solver.setup(ksp)

  pc = Ref{PETSC.PC}()
  @check_error_code PETSC.KSPGetPC(ksp[],pc)
  @check_error_code PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCHPDDM)

  is = Ref{PETSC.IS}()
  @check_error_code PETSC.HPDDMSetAuxiliaryMat(pc[],is[],mat[],C_NULL,C_NULL)
end

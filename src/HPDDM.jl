
struct HPDDMLinearSolver{A} <: LinearSolver
  ranks :: A
  mat   :: PETScMatrix
  is    :: PETScIndexSet
  setup :: Function
end

function HPDDMLinearSolver(indices::PRange,mats::AbstractArray,setup::Function)
  ranks = linear_indices(indices)
  is = PETScIndexSet(indices)
  
  HPDDMLinearSolver(ranks,is,PETScMatrix(mats,indices),setup)
end

function HPDDMLinearSolver(space::FESpace,biform::Function,setup::Function)
  assem = SparseMatrixAssembler(
    SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},space,space
  )
  indices, mats = subassemble_matrices(space,biform,assem)
  HPDDMLinearSolver(indices,mats,setup)
end

function subassemble_matrices(space,biform,assem)
  @assert PartitionedArrays.matching_local_indices(get_rows(assem),get_cols(assem))
  @assert PartitionedArrays.matching_local_indices(get_rows(assem),get_free_dof_ids(space))

  u, v = get_trial_fe_basis(space), get_fe_basis(space)
  data = collect_cell_matrix(space,space,biform(u,v))

  indices = get_cols(assem)
  mats = map(assemble_matrix, local_views(assem), data)

  return indices, mats
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

  mat, is = solver.mat.mat, solver.is.is
  @check_error_code PETSC.HPDDMSetAuxiliaryMat(pc[],is[],mat[],C_NULL,C_NULL)
end

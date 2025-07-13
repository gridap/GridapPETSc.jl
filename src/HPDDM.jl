
struct HPDDMLinearSolver <: LinearSolver
  ranks :: MPIArray
  mat   :: PETScMatrix
  is    :: PETScIndexSet
  setup :: Function
end

function HPDDMLinearSolver(indices::MPIArray,mats::MPIArray,setup::Function)
  ranks = linear_indices(mats)
  is  = PETScIndexSet(PartitionedArrays.getany(indices))
  mat = PETScMatrix(PartitionedArrays.getany(mats))
  display(indices)
  display(mats)
  HPDDMLinearSolver(ranks,mat,is,setup)
end

function HPDDMLinearSolver(indices::AbstractArray,mats::AbstractArray,setup::Function)
  @error """
    HPDDMLinearSolver only makes sense in a distributed context.
  """
end

function HPDDMLinearSolver(space::FESpace,biform::Function,setup::Function)
  assems = map(local_views(space)) do space 
    SparseMatrixAssembler(
      SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},space,space
    )
  end
  indices, mats = subassemble_matrices(space,biform,assems)
  HPDDMLinearSolver(indices,mats,setup)
end

function subassemble_matrices(space,biform,assems)

  u, v = get_trial_fe_basis(space), get_fe_basis(space)
  data = collect_cell_matrix(space,space,biform(u,v))

  indices = local_to_global(get_free_dof_ids(space))
  mats = map(assemble_matrix, assems, data)

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
  @check_error_code PETSC.PCHPDDMSetAuxiliaryMat(pc[],is[],mat[],C_NULL,C_NULL)
end

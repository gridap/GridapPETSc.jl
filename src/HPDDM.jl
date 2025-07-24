
struct HPDDMLinearSolver <: LinearSolver
  ranks     :: MPIArray
  mat       :: PETScMatrix
  is        :: PETScIndexSet
  ksp_setup :: Function
  pc_setup  :: Function
end

"""
    HPDDMLinearSolver(ranks::MPIArray,mat::PETScMatrix,is::PETScIndexSet[,ksp_setup[,pc_setup]])

Wrapper for a `PETScLinearSolver` preconditioned with the HPDDM library.

# Arguments
- `indices::MPIArray`: For each rank, the local-to-global index map for the matrix rows/cols.
- `mats::MPIArray`: For each rank, the matrix for the local overlapping Neumann problem.
- `ksp_setup::Function`: PETSc setup options for the KSP solver. By default, it sets options from the command line.
- `pc_setup::Function`: Extra setup options for the PCHPDDM preconditioner. By default, it sets options from the command line.
"""
function HPDDMLinearSolver(
  indices::MPIArray,mats::MPIArray,
  ksp_setup::Function = hpddm_default_setup_ksp,
  pc_setup::Function = hpddm_default_setup_pc
)
  ranks = linear_indices(mats)
  is  = PETScIndexSet(PartitionedArrays.getany(indices))
  mat = PETScMatrix(PartitionedArrays.getany(mats))
  HPDDMLinearSolver(ranks,mat,is,ksp_setup,pc_setup)
end

"""
    HPDDMLinearSolver(space::FESpace,biform::Function[,args...])

Creates a `HPDDMLinearSolver` for a finite element space and a bilinear form for the local overlapping
Neumann problems.

To have overlapping Neumann problems, the `Measure` has to be modified to include ghost cells.
For instance, for a Poisson problem we would have:

```julia
Ωg  = Triangulation(with_ghost,model)
dΩg = Measure(Ωg,qdegree)
a(u,v) = ∫(∇(u)⋅∇(v))dΩg
```
"""
function HPDDMLinearSolver(space::FESpace,biform::Function,args...)
  assems = map(local_views(space)) do space
    SparseMatrixAssembler(
      SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},space,space
    )
  end
  indices, mats = subassemble_matrices(space,biform,assems)
  HPDDMLinearSolver(indices,mats,args...)
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

function hpddm_default_setup_ksp(ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
end

function hpddm_default_setup_pc(pc)
  @check_error_code PETSC.PCSetFromOptions(pc[])
end

function hpddm_setup(solver::HPDDMLinearSolver,ksp)
  solver.ksp_setup(ksp)

  pc = Ref{PETSC.PC}()
  mat, is = solver.mat.mat, solver.is.is
  @check_error_code PETSC.KSPGetPC(ksp[],pc)
  @check_error_code PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCHPDDM)
  @check_error_code PETSC.PCHPDDMSetAuxiliaryMat(pc[],is[],mat[],C_NULL,C_NULL)
  @check_error_code PETSC.PCHPDDMHasNeumannMat(pc[],PETSC.PETSC_TRUE)
  @check_error_code PETSC.PCHPDDMSetSTShareSubKSP(pc[],PETSC.PETSC_TRUE)

  solver.pc_setup(pc)
end

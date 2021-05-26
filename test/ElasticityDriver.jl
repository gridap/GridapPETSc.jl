module ElasticityDriver

using Test
using Gridap
using Gridap.FESpaces
using SparseArrays
using SparseMatricesCSR
using MPI
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code

Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
Tv = Vector{PetscScalar}

k = 1
n = 10

domain = (0,1,0,1,0,1)
cells  = (n,n,n)
model  = CartesianDiscreteModel(domain,cells)

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},k)
V = TestFESpace(model,reffe,dirichlet_tags="boundary",vector_type=Tv)
U = TrialFESpace(V)
uh = zero(U)

Ω = Triangulation(model)
degree = max((k-1)+(k-1),k)
dΩ = Measure(Ω,degree)

function σ(ε)
  E = 70.0e9
  ν = 0.33
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  λ*tr(ε)*one(ε) + 2*μ*ε
end

f = VectorValue(1.,1.,1.)
a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )dΩ
l(v) = ∫(v⋅f)dΩ
assem = SparseMatrixAssembler(Tm,Tv,U,V)
op = AffineFEOperator(a,l,U,V,assem)

tol = 1e-8
maxits = 1000

options = "-ksp_monitor_short -ksp_rtol $tol -ksp_converged_reason -ksp_max_it $maxits -ksp_norm_type unpreconditioned -ksp_type cg -pc_type gamg -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"

GridapPETSc.with(args=split(options)) do

  bs = 3
  A = PETScMatrix(get_matrix(op))
  @check_error_code PETSC.MatSetBlockSize(A.mat[],bs)
  b = PETScVector(get_vector(op))
  x = PETScVector(get_free_dof_values(uh))

  ns_funs = [
    x->VectorValue(1.,0.,0.),
    x->VectorValue(0.,1.,0.),
    x->VectorValue(0.,0.,1.),
    x->VectorValue(-x[2],x[1],0.),
    x->VectorValue(-x[3],0.,x[1]),
    x->VectorValue(0.,-x[3],x[2])
  ]

  ns_vecs = PETScVector[]
  for ns_fun in ns_funs
    nsh = interpolate(ns_fun,U)
    array = get_free_dof_values(nsh)
    push!(ns_vecs,PETScVector(array))
  end

  nulls = Ref{PETSC.MatNullSpace}()
  comm = MPI.COMM_SELF
  has_constant = PETSC.PETSC_TRUE
  vecs = map(i->i.vec[],ns_vecs)
  nvecs = length(vecs)
  @check_error_code PETSC.MatNullSpaceCreate(comm,has_constant,nvecs,vecs,nulls)
  @check_error_code PETSC.MatSetNearNullSpace(A.mat[],nulls[])

  solver = PETScSolver()
  solve!(x,solver,A,b)

  @check_error_code PETSC.MatNullSpaceDestroy(nulls)

  r = A*x-b
  @test norm(r) < tol

end

end # module

using Gridap
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC
using SparseMatricesCSR
using PartitionedArrays

function main(distribute,nparts)
  parts = distribute(LinearIndices((prod(nparts),)))

  if PETSC.MatMumpsSetIcntl_handle[] == C_NULL
    @info "Skipping DarcyTests since petsc is not configured with mumps."
    return nothing
  end

  domain = (0,1,0,1)
  partition = (100,100)
  model = CartesianDiscreteModel(parts,nparts,domain,partition)

  k = 1
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)

  V = FESpace(model,reffe_u,dirichlet_tags=[5,6])
  Q = FESpace(model,reffe_p,conformity=:L2)

  uD = VectorValue(0.0,0.0)
  U = TrialFESpace(V,uD)
  P = TrialFESpace(Q)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  Ω = Interior(model)
  degree = 2
  dΩ = Measure(Ω,degree)

  neumanntags = [8,]
  Γ = Boundary(model,tags=neumanntags)
  dΓ = Measure(Γ,degree)

  kinv1 = TensorValue(1.0,0.0,0.0,1.0)

  σ(u) =kinv1⋅u

  a((u,p), (v,q)) = ∫(v⋅(σ∘u) - (∇⋅v)*p + q*(∇⋅u) + 0*p*q)dΩ
  n_Γ = get_normal_vector(Γ)
  h = -1.0

  b((v,q)) = ∫((v⋅n_Γ)*h)dΓ

  op = AffineFEOperator(a,b,X,Y)

  options = "-ksp_error_if_not_converged true -ksp_converged_reason -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
  xh = GridapPETSc.with(args=split(options)) do
    ls = PETScLinearSolver()
    xh = solve(ls,op)
  end
  uh, ph = xh

end


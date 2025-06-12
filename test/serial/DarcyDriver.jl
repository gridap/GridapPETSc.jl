module DarcyDriver

using Gridap
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC
using SparseMatricesCSR

options = "-ksp_converged_reason -ksp_monitor -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"

out = GridapPETSc.with(args=split(options)) do

  if PETSC.MatMumpsSetIcntl_handle[] == C_NULL
    @info "Skipping DarcyDriver since petsc is not configured with mumps."
    return nothing
  end

  domain = (0,1,0,1)
  partition = (100,100)
  model = CartesianDiscreteModel(domain,partition)

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
  kinv2 = TensorValue(100.0,90.0,90.0,100.0)

  function σ(x,u)
     if ((abs(x[1]-0.5) <= 0.1) && (abs(x[2]-0.5) <= 0.1))
        return kinv2⋅u
     else
        return kinv1⋅u
     end
  end

  px = get_physical_coordinate(Ω)
  a((u,p), (v,q)) = ∫(v⋅(σ∘(px,u)) - (∇⋅v)*p + q*(∇⋅u) + 0*p*q)dΩ
  n_Γ = get_normal_vector(Γ)
  h = -1.0

  b((v,q)) = ∫((v⋅n_Γ)*h)dΓ

  op = AffineFEOperator(a,b,X,Y)
  ls = PETScLinearSolver()
  xh = solve(ls,op)
  uh, ph = xh

end

end # module
